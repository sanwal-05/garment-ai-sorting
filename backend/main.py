"""FastAPI app for the Returns Vetting Platform."""

from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, JSON
from sqlmodel import Field as SQLField, Session, SQLModel, create_engine, select

from backend.decision import decide, scores_from_probs
from backend.model_service import load_model, mock_enabled, predict

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR.parent / ".env")
load_dotenv(BACKEND_DIR / ".env")

UPLOADS_DIR = BACKEND_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CUSTOMER_STATUSES = {"received", "approved", "needs_more_info", "under_review", "declined"}
CLAIM_CATEGORIES = {
    "arrived_stained",
    "arrived_damaged",
    "poor_quality",
    "wrong_size",
    "not_as_described",
    "changed_mind",
}

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'returns.db'}")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)


class ReturnRecord(SQLModel, table=True):
    __tablename__ = "returns"

    id: str = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc))
    claim_category: str
    claim_text: str
    front_path: str
    back_path: str
    scores: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    levels: dict = SQLField(default_factory=dict, sa_column=Column(JSON))
    decision: str
    disposition: str
    reasons: list = SQLField(default_factory=list, sa_column=Column(JSON))
    status: str = SQLField(index=True)
    reviewer_override: bool = False
    reviewer_note: Optional[str] = None
    model_decision: Optional[str] = None
    model_disposition: Optional[str] = None
    model_status: Optional[str] = None
    thresholds: dict = SQLField(default_factory=dict, sa_column=Column(JSON))


class OverrideBody(BaseModel):
    decision: str = Field(pattern="^(ACCEPT|REVIEW|REJECT)$")
    disposition: str = Field(
        pattern="^(RESELL_AS_NEW|RESELL_AS_USED|REFURBISH|RECYCLE|INSPECT_MANUALLY)$"
    )
    note: str = Field(min_length=3, max_length=1000)


app = FastAPI(title="Returns Vetting Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)
    load_model()


def get_session():
    with Session(engine) as session:
        yield session


def admin_token() -> str:
    token = os.getenv("ADMIN_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured.")
    return token


def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    expected = admin_token()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing admin bearer token.")
    provided = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token.")


def _sniff_image(data: bytes, content_type: Optional[str]) -> str:
    if data.startswith(JPEG_MAGIC):
        return ".jpg"
    if data.startswith(PNG_MAGIC):
        return ".png"
    if content_type in ALLOWED_TYPES:
        return ALLOWED_TYPES[content_type]
    raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")


async def _read_image(upload: UploadFile, field: str) -> tuple[bytes, str]:
    if not upload.filename:
        raise HTTPException(status_code=400, detail=f"{field} image is required.")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{field} image is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"{field} image must be smaller than 8MB.")
    suffix = _sniff_image(data, upload.content_type)
    return data, suffix


def _public_image_url(path: str) -> str:
    name = Path(path).name
    return f"/uploads/{name}"


def _customer_status(status: str) -> str:
    if status in CUSTOMER_STATUSES:
        return status
    return "under_review"


def _admin_payload(row: ReturnRecord) -> dict:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "claim_category": row.claim_category,
        "claim_text": row.claim_text,
        "front_url": _public_image_url(row.front_path),
        "back_url": _public_image_url(row.back_path),
        "scores": row.scores,
        "levels": row.levels,
        "decision": row.decision,
        "disposition": row.disposition,
        "reasons": row.reasons,
        "status": row.status,
        "reviewer_override": row.reviewer_override,
        "reviewer_note": row.reviewer_note,
        "model_decision": row.model_decision,
        "model_disposition": row.model_disposition,
        "model_status": row.model_status,
        "thresholds": row.thresholds,
        "mock_model": mock_enabled(),
    }


@app.post("/api/returns")
async def create_return(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    claim_category: str = Form(...),
    claim_text: str = Form(...),
    session: Session = Depends(get_session),
):
    if claim_category not in CLAIM_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid claim category.")
    text = claim_text.strip()
    if len(text) < 15 or len(text) > 500:
        raise HTTPException(status_code=400, detail="Claim text must be between 15 and 500 characters.")

    front_bytes, front_suffix = await _read_image(front, "front")
    back_bytes, back_suffix = await _read_image(back, "back")

    return_id = str(uuid.uuid4())
    front_name = f"{return_id}_front{front_suffix}"
    back_name = f"{return_id}_back{back_suffix}"
    front_path = UPLOADS_DIR / front_name
    back_path = UPLOADS_DIR / back_name
    front_path.write_bytes(front_bytes)
    back_path.write_bytes(back_bytes)

    try:
        probs = predict(front_bytes, back_bytes)
        scores = scores_from_probs(probs)
        outcome = decide(scores, claim_category, text)
    except HTTPException:
        front_path.unlink(missing_ok=True)
        back_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 — keep the upload and surface a safe error
        front_path.unlink(missing_ok=True)
        back_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not score this return: {exc}") from exc

    row = ReturnRecord(
        id=return_id,
        claim_category=claim_category,
        claim_text=text,
        front_path=str(front_path),
        back_path=str(back_path),
        scores={k: scores.get(k, 0.0) for k in ("stain_present", "hole_major", "heavily_worn")},
        levels=outcome["levels"],
        decision=outcome["decision"],
        disposition=outcome["disposition"],
        reasons=outcome["reasons"],
        status=outcome["status"],
        model_decision=outcome["decision"],
        model_disposition=outcome["disposition"],
        model_status=outcome["status"],
        thresholds=outcome["thresholds"],
    )
    session.add(row)
    session.commit()
    return {"id": return_id, "status": "received"}


@app.get("/api/returns/{return_id}/status")
def get_status(return_id: str, session: Session = Depends(get_session)):
    row = session.get(ReturnRecord, return_id)
    if not row:
        raise HTTPException(status_code=404, detail="Return not found.")
    return {"id": row.id, "status": _customer_status(row.status)}


@app.get("/api/admin/returns")
def list_returns(
    status: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
):
    statement = select(ReturnRecord).order_by(ReturnRecord.created_at.desc())
    if status:
        statement = statement.where(ReturnRecord.status == status)
    if decision:
        statement = statement.where(ReturnRecord.decision == decision)
    rows = session.exec(statement).all()
    return [_admin_payload(row) for row in rows]


@app.get("/api/admin/returns/{return_id}")
def get_return(
    return_id: str,
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
):
    row = session.get(ReturnRecord, return_id)
    if not row:
        raise HTTPException(status_code=404, detail="Return not found.")
    return _admin_payload(row)


@app.post("/api/admin/returns/{return_id}/override")
def override_return(
    return_id: str,
    body: OverrideBody,
    session: Session = Depends(get_session),
    _: None = Depends(require_admin),
):
    row = session.get(ReturnRecord, return_id)
    if not row:
        raise HTTPException(status_code=404, detail="Return not found.")
    status_map = {"ACCEPT": "approved", "REVIEW": "under_review", "REJECT": "declined"}
    row.decision = body.decision
    row.disposition = body.disposition
    row.reviewer_note = body.note.strip()
    row.reviewer_override = True
    row.status = status_map[body.decision]
    session.add(row)
    session.commit()
    session.refresh(row)
    return _admin_payload(row)


@app.get("/api/health")
def health():
    return {"ok": True, "mock_model": mock_enabled()}
