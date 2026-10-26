# Returns Vetting Platform

Customer returns are photographed (front and back), scored by a two-view garment model, then routed through a dual-threshold policy. Customers only ever see a friendly status. Reviewers see scores, reasons, and can override with an audit trail.

The original multi-task learning work (EfficientNetB0, four heads: condition, pilling, stains, holes) still lives in this repo: `DL_Proj_Improved.ipynb`, `run_experiments.py`, and `results/`.

## Layout

- `/frontend` — React + Vite + TypeScript + Tailwind
- `/backend` — FastAPI + SQLModel (SQLite)

## Quick start

### 1. Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
# keep MOCK_MODEL=1 until the Keras file is in place
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Run that from the **repository root** so `backend.main:app` resolves.

Optional real model: copy `best_improved_garment_model.keras` to `backend/model/` and set `MOCK_MODEL=0`. Install TensorFlow first (`pip install tensorflow`). Images are decoded, resized to 224×224, and passed as raw 0–255 float32 — no extra normalisation — with inputs `front_image` and `back_image`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173  
Admin desk: http://localhost:5173/admin (token from `ADMIN_TOKEN` in `.env`)

Vite proxies `/api` and `/uploads` to the API on port 8000.

### Docker (backend)

From the repository root:

```bash
docker build -t returns-api -f backend/Dockerfile backend
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=change-me -e MOCK_MODEL=1 returns-api
```

## API

| Method | Path | Who | Notes |
| --- | --- | --- | --- |
| POST | `/api/returns` | customer | multipart `front`, `back`, `claim_category`, `claim_text`. Returns `{id, status:"received"}` only. |
| GET | `/api/returns/{id}/status` | customer | `received` \| `approved` \| `needs_more_info` \| `under_review` \| `declined`. No scores, thresholds, or reasons. |
| GET | `/api/admin/returns?status=&decision=` | admin | Full internal rows. |
| GET | `/api/admin/returns/{id}` | admin | Detail, including image URLs. |
| POST | `/api/admin/returns/{id}/override` | admin | `{decision, disposition, note}`. Model output is kept. |

Admin routes require `Authorization: Bearer $ADMIN_TOKEN`. Photos must be JPEG/PNG and under 8MB.

## Decision policy

`backend/decision.py` maps softmax heads to `stain_present`, `hole_present`, `hole_major`, and `heavily_worn`, then `decide(scores, claim, text)` using hi/lo thresholds in `backend/thresholds.json`.

Replace `decision.py` with your own if you already have one — keep `scores_from_probs()` and `decide(scores, claim, text)`.

## Dataset (research)

Second-Hand Fashion Dataset v2: https://zenodo.org/records/12518734
