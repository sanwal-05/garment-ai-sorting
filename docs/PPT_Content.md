# Presentation Deck: Atelier Returns — AI-Assisted Garment Return Vetting & Refurbishment Routing

---

## Title Slide

### **Atelier Returns: AI-Assisted Garment Return Vetting & Refurbishment Routing**
**Subtitle:** Two-View Multi-Task Defect Assessment, Calibrated Dual-Threshold Policy, and Human-in-the-Loop Triage

- **Domain:** Retail E-Commerce Reverse Logistics & Circular Fashion AI
- **Core Stack:** FastAPI, SQLModel/SQLite, PyTorch (DINOv2 + CLIP) / Keras, LightGBM, React + TypeScript
- **Visual Tasks:** Condition, Pilling, Stains, and Holes (Front & Back Garment Views)
- **Operational Outputs:** Financial Verdict (`ACCEPT` / `REVIEW` / `REJECT`) + Warehouse Disposition (`RESELL_AS_NEW` / `RESELL_AS_USED` / `REFURBISH` / `RECYCLE` / `INSPECT_MANUALLY`)

**Suggested Visual:** Split hero graphic showing the Customer Return Portal (`CustomerPage.tsx`) on the left and the Internal Reviewer Desk (`AdminPage.tsx` with `ScoreBar` threshold gauges) on the right.

**Speaker Notes:**
> "Hello everyone. Today I am presenting Atelier Returns, an AI-assisted garment return vetting and circular-economy routing platform. Our system helps e-commerce retailers verify whether a returned garment's physical condition matches the customer's stated return reason, while simultaneously deciding whether the item should be resold as new, resold as used, refurbished, or recycled."

---

## Section 1: Problem Statement and Motivation

### Slide 1.1 — The Retail Returns Crisis & Verification Bottleneck
- **Massive Return Volume:** Apparel e-commerce suffers from high return rates, forcing warehouses to inspect millions of garments manually.
- **Blind Auto-Approvals Lose Money:** Refunding before inspection exposes brands to "wardrobing" (returning worn/stained items as "wrong size") and false defect claims.
- **Manual Inspection Does Not Scale:** Unpacking, flipping, grading, and routing every garment by hand creates costly warehouse backlogs.
- **Single-Task Defect Detectors Fall Short:** Standard vision models only detect flaws—they never check if the flaw matches what the customer claimed.
- **Missing Circular Routing:** Retailers need an immediate physical disposition (`RESELL_AS_NEW`, `RESELL_AS_USED`, `REFURBISH`, `RECYCLE`), not just a binary pass/fail.

**Suggested Visual:** A 2×2 problem contrast matrix comparing *Customer Claim (Defect vs. Non-Defect)* against *Actual Garment State (Intact vs. Damaged)*, highlighting where fraud and margin loss occur.

**Speaker Notes:**
> "Why does this problem matter? In apparel e-commerce, manual return inspection is slow and expensive, but blind auto-approval invites return abuse. Crucially, traditional defect detection isn't enough: if a customer claims 'Arrived stained', a stain validates their claim. But if they claim 'Changed my mind' and the garment is stained, that's customer damage. Retailers need a system that cross-checks visual evidence against the claim and routes the physical garment into the right refurbishment or resale stream."

---

### Slide 1.2 — What Makes This Problem Technically Unique
- **Two-View Requirement (Front + Back):** Stains and tears often appear on only one side; single-image models miss rear-side damage.
- **Multi-Task Visual Assessment:** A single garment can simultaneously exhibit surface pilling, localized stains, structural holes, and overall wear.
- **Claim-vs-Evidence Polarity Flip:** The exact same visual defect score means `ACCEPT` under a defect claim (`arrived_stained`) but `REVIEW` under a non-defect claim (`wrong_size`).
- **Calibrated Grey-Zone Uncertainty:** Borderline home-lighting photos should not trigger unfair customer rejections—they require dual thresholds (`lo` / `hi`).
- **Adversarial Gaming Risk:** Exposing raw AI confidence scores to customers would let bad actors retake photos until they bypass the threshold.

**Suggested Visual:** Architecture callout diagram illustrating how identical visual scores (`stain_present = hi`) branch into opposite financial decisions depending on `CLAIM_EVIDENCE` vs. `NON_DEFECT_CLAIMS`.

**Speaker Notes:**
> "From an ML engineering perspective, five factors make this unique: we must process paired front and back views, predict four distinct garment attributes simultaneously, invert decision polarity based on the customer's claim category, buffer uncertain predictions in a middle review zone, and hide internal scores from the customer API so the system cannot be gamed."

---

## Section 2: Proposed Solution and Why It Matters

### Slide 2.1 — End-to-End Platform Architecture (Atelier Returns)
- **3-Step Customer Wizard (`/`):** Guided upload of front & back photos (`<= 8 MB`, magic-byte verified JPEG/PNG), 6 claim categories, and 15–500 char note.
- **Two-View Multi-Task Inference (`backend/`):** Supports end-to-end 224×224 Keras inference (`model_service.py`), a 3-backbone DINOv2+CLIP ensemble (`predict.py`), and `MOCK_MODEL` mode.
- **4 Composite Vetting Scores (`decision.py`):** Collapses multi-class probabilities into `stain_present`, `hole_present`, `hole_major`, and `heavily_worn`.
- **Dual-Threshold Policy Engine (`thresholds.json`):** Maps each score to `lo`, `mid`, or `hi` and cross-checks against claim type and text length (`< 40` chars check).
- **Asymmetric Customer vs. Admin Views:** Customers poll a clean 5-stage `StatusTracker` every 5s; reviewers access full diagnostics via Bearer token auth.
- **Human-in-the-Loop Override & Audit Trail:** Reviewers can override decisions in SQLite (`returns.db`) while preserving `model_decision` and `model_disposition`.

**Suggested Visual:** Horizontal system dataflow diagram: `CustomerPage.tsx` $\rightarrow$ `POST /api/returns` $\rightarrow$ `model_service.py / predict.py` $\rightarrow$ `decision.py + thresholds.json` $\rightarrow$ `SQLite (returns.db)` $\rightarrow$ `AdminPage.tsx`.

**Speaker Notes:**
> "Our proposed solution is a complete full-stack platform. Customers complete a 3-step return wizard with front and back photos. The FastAPI backend validates the binary file headers, runs two-view multi-task inference, computes four composite defect scores, and runs them through our dual-threshold rule engine. Customers only see a friendly live status tracker, while warehouse specialists get a rich triage desk with full override and audit capabilities."

---

### Slide 2.2 — Business Value & Circular-Economy Impact
- **Automated Triage of Clear Cases:** Confident intact or defective claims are auto-resolved (`ACCEPT` or `REJECT`), slashing manual review volume.
- **Targeted Specialist Focus:** Human reviewers only inspect ambiguous `mid`-zone claims (`INSPECT_MANUALLY`) or damaged non-defect returns.
- **Automated Follow-Up on Brief Claims:** Inconclusive defect claims with `< 40` character descriptions automatically trigger `needs_more_info`.
- **4-Tier Circular Disposition Routing:** Maximizes recovery value across `RESELL_AS_NEW` $\rightarrow$ `RESELL_AS_USED` $\rightarrow$ `REFURBISH` $\rightarrow$ `RECYCLE`.
- **Compliance-Ready Auditability:** Every human override requires a written note (`3–1000` chars) and logs alongside the original AI verdict.

**Suggested Visual:** Screenshot of the Reviewer Desk (`AdminPage.tsx`) showing the 4 KPI summary cards (`Total Claims`, `Needs Review`, `Accepted`, `Rejected`) and the slide-over inspection drawer with `ScoreBar` threshold markers.

**Speaker Notes:**
> "For retail operations teams, the business value is immediate: routine claims are settled automatically, vague borderline claims automatically ask the customer for more information, and every accepted return is assigned a circular disposition route to recover maximum resale value."

---

## Section 3: Dataset Description and Methodology

### Slide 3.1 — Dataset Schema, Tasks & Imbalance Handling
- **Dataset & Scale:** Second-Hand Fashion Dataset v2 (front + back views); `garment_bundle.joblib` scalers/PCA fitted on **7,658 training garments** across **25 garment types** (`Top`, `T-Shirt`, `Trousers`, `Sweater`, `Dress`, etc.). *(Val/test split sizes: Not found in `backend/` or `frontend/` code).*
- **Four Multi-Task Prediction Heads:**
  - *Active Service Schema (`model_service.py` / `decision.py`):* `condition` (5 grades: 1=worst to 5=best), `pilling` (5 grades), `stains` (3: None/Minor/Major), `holes` (3: None/Minor/Major).
  - *Ensemble Bundle Schema (`predict.py` / `garment_bundle.joblib`):* `condition` (3 ordinal buckets: `1–2`, `3`, `4–5`), `pilling` (3 buckets: `1–3`, `4`, `5`), `stains` (binary), `holes` (binary).
- **3-Layer Imbalance Mitigation:**
  1. `class_weight='balanced'` across all 48 `LogisticRegression` and `LightGBM` estimators in `garment_bundle.joblib`.
  2. Cumulative probability aggregation (`stain_minor + stain_major`, `hole_minor + hole_major`).
  3. Lower calibrated thresholds in `thresholds.json` for rare defects (`hole_present` `lo=0.168, hi=0.485` vs. `stain_present` `lo=0.312, hi=0.663`).

**Suggested Visual:** Table comparing the 4 tasks (`condition`, `pilling`, `stains`, `holes`), their class schemas, and how they collapse into the 4 vetting scores in `backend/decision.py`.

**Speaker Notes:**
> "Looking at the data and task structure in our backend: our serialized ensemble bundle was fitted on 7,658 garments spanning 25 apparel categories. Because severe defects like holes are much rarer than clean garments, we handle class imbalance at three levels: balanced class weights in all 48 classifiers, cumulative probability pooling for minor+major defects, and task-calibrated low/high thresholds."

---

### Slide 3.2 — Two-View Preprocessing & Multi-Backbone Methodology
- **Pipeline A — End-to-End Keras Service (`backend/model_service.py`):**
  - Decodes RGB bytes, resizes front & back to `224 × 224` (`float32` in `[0, 255]`), and executes two-view inference (`best_improved_garment_model.keras`).
- **Pipeline B — Multi-Backbone ViT + Tabular Ensemble (`backend/predict.py`):**
  - Resizes to `256 × 256` (PIL bilinear), then antialias-interpolates to `252 × 252` for **DINOv2-base** & **DINOv2-large** and `224 × 224` for **CLIP-ViT-B/16**.
  - **Horizontal-Flip TTA:** Averages embeddings of original `x` and mirrored `x.flip(-1)` for both front and back views.
  - **Feature Fusion & PCA:** Concatenates front+back embeddings (`1,024-d` CLIP, `3,072-d` DINOv2-B, `4,096-d` DINOv2-L, `8,192-d` All) $\rightarrow$ `StandardScaler` $\rightarrow$ `PCA` (`256` / `384` dims) + `26-d` garment-type one-hot vector.
  - **48-Model Ensemble:** Averages `LogisticRegression(C=0.01)` and `LGBMClassifier(n_estimators=300, lr=0.03, num_leaves=15, reg_lambda=5)`, including Frank & Hall `+ordinal` binary cascades.
- **Composite Vetting Formulas (`backend/decision.py`):**
  - `stain_present = stains[1] + stains[2]` | `hole_present = holes[1] + holes[2]` | `hole_major = holes[2]`
  - `heavily_worn = 0.65 * (cond[0] + cond[1]) + 0.35 * (pill[0] + pill[1])`

**Suggested Visual:** Block diagram of the `predict.py` ensemble pipeline (Front/Back Images $\rightarrow$ DINOv2-B / DINOv2-L / CLIP-B/16 with TTA $\rightarrow$ StandardScaler + PCA + 26-d Type One-Hot $\rightarrow$ 48 LR/LightGBM + Ordinal Heads $\rightarrow$ `decision.py`).

**Speaker Notes:**
> "Our backend contains two inference pipelines: the FastAPI service layer for the 224x224 Keras multi-task network, and the standalone multi-backbone predictor in predict.py. In predict.py, we extract horizontal-flip TTA embeddings from DINOv2-base, DINOv2-large, and CLIP-ViT-B/16, compress the 8,192 concatenated front-and-back features via PCA, append a 26-dimensional garment-type one-hot vector, and ensemble 48 regularized Logistic Regression and LightGBM models before feeding the probabilities into decision.py."

---

## Section 4: Results and Insights

### Slide 4.1 — Calibrated Thresholds, Saved Case Study & Codebase Findings
- **Offline Test Metrics Status:** Offline accuracy, macro-F1, and confusion matrices are *Not found in code* within `backend/` and `frontend/` (stored in external research files outside the web app folders).
- **Calibrated Dual Thresholds (`backend/thresholds.json`):**
  - `stain_present`: `lo = 0.312`, `hi = 0.663` | `hole_present`: `lo = 0.168`, `hi = 0.485`
  - `hole_major`: `lo = 0.200`, `hi = 0.600` | `heavily_worn`: `lo = 0.280`, `hi = 0.508`
- **Real Persisted Case in `backend/returns.db` (`6261a770...`):**
  - Claim: `arrived_damaged` | Note: *"The whole cloth is torn up"* (`26` chars `< 40`).
  - Scores: `stain_present=0.630`, `hole_present=0.534`, `hole_major=0.482`, `heavily_worn=0.448`.
  - Outcome: Routed to `decision = REVIEW`, `disposition = INSPECT_MANUALLY`, and customer `status = needs_more_info` with automated explanations.

**Suggested Visual:** Visual recreation or screenshot of Record `6261a770` inside the Admin Detail Drawer showing the four `ScoreBar` gauges with `lo`/`hi` tick marks and the generated policy reasons.

**Speaker Notes:**
> "Within the backend and frontend codebase itself, we have the calibrated operating thresholds in thresholds.json and a persisted end-to-end return transaction in returns.db. In that test case, the customer claimed 'arrived_damaged' with a brief 26-character note. Because the scores fell in the inconclusive middle zone and the note was under 40 characters, the policy engine automatically routed the item to manual inspection and set the customer status to 'needs_more_info'."

---

### Slide 4.2 — Key Insights, Limitations & Future Roadmap
- **Key Operational Insights:**
  - Separating **claim verification** (`ACCEPT`/`REVIEW`/`REJECT`) from **physical disposition** (`RESELL`/`REFURBISH`/`RECYCLE`) allows brands to approve valid defect claims while automatically routing damaged stock to repair or recycling.
  - The `mid` threshold zone acts as a safety valve against poor home lighting and camera angles.
- **Current Technical Limitations Identified in Audit:**
  - `main.py` currently calls `model_service.py` (expecting `.keras` 5/5/3/3 outputs), whereas `predict.py` (`garment_bundle.joblib`, 3/3/2/2 outputs) is not yet wired into `model_service.py`.
  - Downsampling full garments to `224×224` / `252×252` can miss micro-defects (`< 2 mm` pinholes); no NLP semantic location matching on `claim_text` yet.
- **Future Enhancements:**
  - Unify `predict.py` and `model_service.py` behind a selectable model adapter; add a 3rd close-up defect crop upload and Grad-CAM / attention heatmaps in the Reviewer Desk.

**Suggested Visual:** A 3-column roadmap graphic: *Current Production Capabilities* $\rightarrow$ *Identified Integration Gaps* $\rightarrow$ *Next-Gen Multimodal & High-Res Upgrades*.

**Speaker Notes:**
> "To summarize our insights and honest engineering audit: decoupling financial approval from warehouse disposition makes the system immediately actionable for circular retail. Our audit also identified a clear integration step—unifying the standalone DINOv2+CLIP joblib ensemble with the live FastAPI model service—alongside future upgrades like close-up defect crop uploads and Grad-CAM heatmaps for reviewers."

---

## Closing Slide

### **Thank You — Questions & Viva Discussion**
**Atelier Returns — AI-Assisted Garment Return Vetting Platform**

- **Customer Portal:** `http://localhost:5173/` (3-Step Return Wizard + Live Polling Status Tracker)
- **Reviewer Console:** `http://localhost:5173/admin` (KPI Dashboard + Dual-Threshold ScoreBars + Override Audit Trail)
- **API & Policy Engine:** FastAPI (`/api/returns`, `/api/admin/returns`) + `backend/decision.py` + `backend/thresholds.json`
- **Documentation & Deep Dive:** Available in `docs/Project_Documentation.docx` and `README.md`

**Suggested Visual:** Clean summary card displaying the repository layout, API table, and QR/link placeholder for live demo.

**Speaker Notes:**
> "Thank you for your time. Both the customer portal and the reviewer console are ready for walkthrough, and I would be glad to answer any questions on our multi-task vision architectures, the dual-threshold policy engine, or the full-stack implementation."
