# Returns Vetting Platform (Atelier Returns)

> **AI-Assisted Garment Return Verification, Defect Assessment, and Circular Refurbishment Routing**

Customer returns are photographed (**front** and **back**), scored by a two-view multi-task garment vision model, and routed through a calibrated dual-threshold policy engine. Customers only ever see a friendly, non-gameable live status tracker. Internal warehouse reviewers see defect scores, threshold zones, and policy reasons, and can override decisions with a preserved audit trail.

The original multi-task learning research work (`EfficientNetB0`, four heads: `condition`, `pilling`, `stains`, `holes`) also lives in this repository (`DL_Proj_Improved.ipynb`, `run_experiments.py`, and `results/`), alongside the full-stack documentation in `docs/Project_Documentation.docx` and presentation slides in `docs/PPT_Content.md`.

---

## 1. Problem Statement

E-commerce and apparel retail companies face a high-cost dilemma when processing returned garments:
- **Blind auto-approvals** expose retailers to return fraud, "wardrobing" (returning worn or customer-damaged items under *"wrong size"* or *"changed mind"* claims), and false defect claims.
- **Manual warehouse inspection** of every returned item creates severe labor bottlenecks and delays customer refunds.
- **Standard defect detectors** only classify whether a flaw exists—they do not check whether the visual evidence matches the customer's claimed return reason or determine the garment's next circular-economy disposition (`RESELL_AS_NEW`, `RESELL_AS_USED`, `REFURBISH`, `RECYCLE`, or `INSPECT_MANUALLY`).

---

## 2. Proposed Solution

**Atelier Returns (Returns Vetting Platform)** automates first-line return verification and warehouse triage:
1. **Two-View Customer Capture (`/`):** A 3-step guided wizard (`Photos` $\rightarrow$ `Reason` $\rightarrow$ `Review`) collects front and back garment photos (`JPEG`/`PNG`, $\le 8\text{ MB}$, verified via binary magic bytes), a structured claim category, and a written explanation ($15\text{–}500$ characters).
2. **Multi-Task Visual Scoring (`backend/`):** Predicts multi-task probabilities across four garment attributes (**condition**, **pilling**, **stains**, and **holes**) from paired front and back views, and collapses them into four composite risk scores (`stain_present`, `hole_present`, `hole_major`, `heavily_worn`).
3. **Calibrated Dual-Threshold Policy (`backend/decision.py` + `backend/thresholds.json`):** Categorizes each score into `lo`, `mid`, or `hi` zones and cross-checks visual evidence against the customer's claim category (`CLAIM_EVIDENCE` vs. `NON_DEFECT_CLAIMS`) and description length.
4. **Reviewer Console & Audit Trail (`/admin`):** Bearer-token-protected dashboard with KPI cards, search/filter queues, zoomable inspection photos, dual-threshold `ScoreBar` gauges, automated policy reasons, and a human override workflow that preserves the original AI decision (`model_decision`, `model_disposition`, `model_status`) in SQLite.

---

## 3. Key Features

- **Cross-Modal Claim Verification:** Approves defect claims (`arrived_stained`, `arrived_damaged`, `poor_quality`) when corroborated by high defect scores (`hi`), rejects them when scores are `lo`, and flags non-defect claims (`wrong_size`, `not_as_described`, `changed_mind`) for human review if unexpected damage is detected.
- **Circular Disposition Routing:** Automatically assigns physical routing (`RESELL_AS_NEW`, `RESELL_AS_USED`, `REFURBISH`, `RECYCLE`, `INSPECT_MANUALLY`).
- **Short-Description Escalation:** Inconclusive (`mid`) defect claims with brief customer notes ($< 40$ characters) automatically transition to `needs_more_info`.
- **Zero-Leakage Customer API:** Public endpoints (`/api/returns`, `/api/returns/{id}/status`) never expose raw probabilities, thresholds, or internal reasons, preventing adversarial gaming.
- **Mock & Real Inference Modes:** Supports zero-setup UI testing via `MOCK_MODEL=1`, end-to-end Keras inference (`backend/model_service.py`), and a standalone multi-backbone Vision Transformer + LightGBM/LogisticRegression ensemble (`backend/predict.py`).

---

## 4. Tech Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | React + TypeScript + Vite | `^18.3.1` / `^5.6.3` / `^5.4.11` | SPA customer wizard (`/`) and reviewer console (`/admin`) |
| **Styling** | Tailwind CSS + PostCSS | `^3.4.15` | Custom editorial palette (`sage`, `charcoal`, `sand`, `terracotta`) |
| **State & Forms** | TanStack React Query + React Hook Form + Zod | `^5.59.20` / `^7.53.2` / `^3.23.8` | 5-second live status polling, cache invalidation, and schema validation |
| **Backend API** | FastAPI + Uvicorn | `>=0.115.0` / `>=0.32.0` | REST endpoints, magic-byte image validation, static file serving, Bearer auth |
| **Database / ORM** | SQLModel + SQLite | `>=0.0.22` | Persistent storage (`backend/returns.db`) for returns and override audit trails |
| **ML / Vision** | TensorFlow/Keras, PyTorch, HuggingFace `transformers`, `scikit-learn`, `lightgbm`, `joblib`, `Pillow` | `transformers>=4.45.0`, `scikit-learn>=1.3.0`, `lightgbm>=4.0.0` | Two-view inference (`model_service.py` and `predict.py`) |

---

## 5. Dataset Summary

- **Source:** [Zenodo Second-Hand Fashion Dataset v2](https://zenodo.org/records/12518734) (front and back garment photographs with expert annotations).
- **Fitted Bundle Scale:** The serialized feature scalers and PCA transformers in `backend/model/garment_bundle.joblib` were fitted on **7,658 training garment records** across **25 garment categories** (`Top`, `T-Shirt`, `Trousers`, `Sweater`, `Dress`, `Shirt`, `Jeans`, `Jacket`, etc., mapped in `type_map`). *(Exact validation/test split counts are not stored inside `backend/` or `frontend/`—see `docs/Project_Documentation.docx` for details).*
- **Tasks & Grading Scale:**
  - `condition`: Grades `1` (worst) to `5` (best) in `model_service.py` / `decision.py`; grouped into 3 ordinal buckets (`low[1-2]`, `mid[3]`, `high[4-5]`) in `predict.py`.
  - `pilling`: Grades `1` (worst) to `5` (best) in `model_service.py` / `decision.py`; grouped into 3 ordinal buckets (`1-3`, `4`, `5`) in `predict.py`.
  - `stains`: 3 classes (`None`, `Minor`, `Major`) in `model_service.py` / `decision.py`; binary (`p_stains`) in `predict.py`.
  - `holes`: 3 classes (`None`, `Minor`, `Major`) in `model_service.py` / `decision.py`; binary (`p_holes`) in `predict.py`.

---

## 6. Model & Decision Policy Overview

### A. Active FastAPI Model Service (`backend/model_service.py`)
- Loads `backend/model/best_improved_garment_model.keras` (two-view `EfficientNetB0` multi-task network) when `MOCK_MODEL=0`, or generates simulated probability vectors when `MOCK_MODEL=1`.
- Decodes front and back images to 3-channel RGB, resizes to `224 × 224`, and passes raw `[0, 255]` `float32` tensors (`front_image` and `back_image`) to predict 4 softmax heads (`condition` [5], `pilling` [5], `stains` [3], `holes` [3]).

### B. Standalone Multi-Backbone ViT + GBDT/Linear Ensemble (`backend/predict.py` + `backend/model/garment_bundle.joblib`)
- Extracts frozen embeddings with **horizontal-flip Test-Time Augmentation (TTA)** across three Vision Transformers:
  - `facebook/dinov2-base` (`252 × 252`, `1,536-d` per view = CLS + patch mean)
  - `facebook/dinov2-large` (`252 × 252`, `2,048-d` per view = CLS + patch mean)
  - `openai/clip-vit-base-patch16` (`224 × 224`, `512-d` per view)
- Concatenates front and back embeddings into 4 feature spaces (`clip-vit-base-patch16` [1024-d], `dinov2-base` [3072-d], `dinov2-large` [4096-d], and `all` [8192-d]), applies `StandardScaler` + `PCA` (`256` or `384` components), appends a `26-d` one-hot garment-type vector, and averages predictions across **48 trained estimators** (`LogisticRegression(C=0.01, class_weight='balanced')` and `LGBMClassifier(n_estimators=300, lr=0.03, num_leaves=15, reg_lambda=5, class_weight='balanced')`, including Frank & Hall `+ordinal` binary cascades).

### C. Score Aggregation & Dual-Threshold Policy (`backend/decision.py` & `backend/thresholds.json`)
- **Composite Vetting Scores:**
  - `stain_present = stain_minor + stain_major` (`lo: 0.312`, `hi: 0.663`)
  - `hole_present = hole_minor + hole_major` (`lo: 0.168`, `hi: 0.485`)
  - `hole_major = hole_major` (`lo: 0.200`, `hi: 0.600`)
  - `heavily_worn = 0.65 * (cond[0] + cond[1]) + 0.35 * (pill[0] + pill[1])` (`lo: 0.280`, `hi: 0.508`)

---

## 7. Results & Saved Artifacts Snapshot

- **Within `backend/` and `frontend/`:**
  - Calibrated dual thresholds are stored in `backend/thresholds.json`.
  - Fitted multi-backbone ensemble weights (48 model entries, fitted on $N = 7,658$ garments) are stored in `backend/model/garment_bundle.joblib`.
  - A sample end-to-end return transaction (`id = 6261a770-2493-4572-9619-60eb1ebaf632`, claim `arrived_damaged`, text *"The whole cloth is torn up"*, routed to `REVIEW` / `INSPECT_MANUALLY` / `needs_more_info`) is persisted in `backend/returns.db` with uploaded images in `backend/uploads/`.
  - *(Note: Offline classification accuracy, macro-F1, and confusion matrices are not stored inside `backend/` or `frontend/`; see `results/` and `DL_Proj_Improved.ipynb` for offline benchmark logs).*

---

## 8. Project Structure

```text
.
├── backend/
│   ├── Dockerfile                  # Python 3.11 container baking DINOv2 & CLIP weights
│   ├── requirements.txt            # Backend Python dependencies
│   ├── main.py                     # FastAPI app, SQLModel ReturnRecord, routes & auth
│   ├── model_service.py            # Two-view Keras loader (224x224) & MOCK_MODEL fallback
│   ├── predict.py                  # Standalone DINOv2 + CLIP + LightGBM/LR ensemble predictor
│   ├── decision.py                 # Score collapsing & dual-threshold decision/disposition engine
│   ├── thresholds.json             # Calibrated lo/hi cutoffs for the 4 vetting scores
│   ├── returns.db                  # SQLite database storing return records & audit trails
│   ├── model/
│   │   ├── README.md               # Instructions for placing best_improved_garment_model.keras
│   │   └── garment_bundle.joblib   # Fitted 48-model ensemble bundle (PCA + LR + LightGBM)
│   └── uploads/                    # Stored front/back customer return photos
├── frontend/
│   ├── package.json                # React 18, TypeScript, Vite, Tailwind, React Query, Zod
│   ├── vite.config.ts              # Dev server on :5173 proxying /api and /uploads to :8000
│   ├── tailwind.config.js          # Custom Atelier Returns color palette & typography
│   └── src/
│       ├── App.tsx                 # Header navigation & route definitions (/ and /admin)
│       ├── api.ts                  # Typed REST client & sessionStorage Bearer token helpers
│       ├── types.ts                # Shared TypeScript interfaces & claim options
│       ├── pages/
│       │   ├── CustomerPage.tsx    # 3-step return wizard + live 5s polling Confirmation
│       │   └── AdminPage.tsx       # Bearer login, KPI desk, filter table & DetailDrawer
│       └── components/
│           ├── PhotoDropzone.tsx   # Drag-and-drop & mobile camera upload with 8MB/MIME check
│           ├── ScoreBar.tsx        # Threshold-annotated progress bar (lo / mid / hi zones)
│           ├── StatusTracker.tsx   # 5-stage customer live status progress component
│           └── Badges.tsx          # DecisionBadge & DispositionBadge components
└── docs/
    ├── Project_Documentation.docx  # Comprehensive viva-ready technical documentation
    └── PPT_Content.md              # Slide-by-slide presentation content & speaker notes
```

---

## 9. Setup and Run Instructions

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

- **Optional Real Keras Model:** Copy `best_improved_garment_model.keras` to `backend/model/` and set `MOCK_MODEL=0` in `.env`. Install TensorFlow first (`pip install tensorflow`). Images are decoded, resized to `224 × 224`, and passed as raw `0–255` `float32` — no extra normalisation — with inputs `front_image` and `back_image`.
- **Standalone Multi-Backbone Predictor CLI (`backend/predict.py`):**
  ```bash
  python backend/predict.py path/to/front.jpg path/to/back.jpg --type "T-Shirt" --bundle backend/model/garment_bundle.joblib
  ```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

- **Customer Portal:** http://localhost:5173
- **Reviewer Desk:** http://localhost:5173/admin (sign in with `ADMIN_TOKEN` from `.env`, default `dev-admin-token`)
- Vite automatically proxies `/api` and `/uploads` to the FastAPI backend on port `8000`.

### 3. Docker (Backend)

From the repository root:

```bash
docker build -t returns-api -f backend/Dockerfile backend
docker run --rm -p 8000:8000 -e ADMIN_TOKEN=change-me -e MOCK_MODEL=1 returns-api
```

---

## 10. API Overview

| Method | Path | Who | Request | Response / Notes |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/returns` | Customer | `multipart/form-data`: `front`, `back`, `claim_category`, `claim_text` | Returns `{"id", "status": "received"}` only. |
| `GET` | `/api/returns/{id}/status` | Customer | Path `id` | Returns `{"id", "status"}` (`received` \| `approved` \| `needs_more_info` \| `under_review` \| `declined`). No scores, thresholds, or reasons. |
| `GET` | `/api/admin/returns` | Admin | Optional query `?status=&decision=` | Array of full internal `ReturnRecord` rows. |
| `GET` | `/api/admin/returns/{id}` | Admin | Path `id` | Full detail payload including image URLs, scores, thresholds, and audit fields. |
| `POST` | `/api/admin/returns/{id}/override` | Admin | JSON `{decision, disposition, note}` | Updates active decision/disposition/status while preserving `model_decision` & `model_disposition`. |
| `GET` | `/api/health` | Public | None | Returns `{"ok": true, "mock_model": bool}`. |

Admin routes require `Authorization: Bearer $ADMIN_TOKEN`. Photos must be JPEG or PNG and under 8 MB.

---

## 11. Target Users & Future Work

- **Target Users:**
  - **E-Commerce & Apparel Brands:** Automate pre-shipment return approvals and deter return abuse.
  - **Warehouse & Reverse-Logistics Teams:** Triage incoming parcels into resale, refurbishment, or textile recycling streams.
  - **Sustainability & Circular-Fashion Operators:** Maximize garment recovery value with auditable grading.
- **Future Work:**
  - Unify `backend/predict.py` (`garment_bundle.joblib` DINOv2+CLIP ensemble) directly into `backend/model_service.py` as a selectable production backend alongside the `.keras` model.
  - Add an optional 3rd high-resolution close-up defect crop upload for micro-defects (pinholes, seam tears).
  - Surface Grad-CAM / Vision Transformer attention heatmaps inside the Reviewer Desk drawer.
  - Incorporate multimodal text-image alignment to verify that the customer's described defect location matches the visual flaw.
