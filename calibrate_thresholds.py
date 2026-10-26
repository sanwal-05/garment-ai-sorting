"""Pick backend/thresholds.json from out-of-fold predictions."""
import json
from pathlib import Path
import numpy as np

d = np.load("results/oof_predictions.npz", allow_pickle=True)
TARGETS = {
    "stain_present": (d["p_stains"][:, 1], d["y_stains"]),
    "hole_present":  (d["p_holes"][:, 1], d["y_holes"]),
    "heavily_worn":  (0.65 * d["p_condition"][:, 0] + 0.35 * d["p_pilling"][:, 0], (d["y_condition"] == 0).astype(int)),
}
RECALL_LO, PREC_HI = 0.90, 0.60       # below lo: <=10% of true cases; above hi: >=60% are true cases

path = Path("backend/thresholds.json"); out = json.loads(path.read_text())
for key, (s, y) in TARGETS.items():
    lo = float(np.quantile(s[y == 1], 1 - RECALL_LO))
    hi, fallback = None, False
    for t in np.unique(np.quantile(s, np.linspace(0.5, 0.995, 120))):
        sel = s >= t
        if sel.sum() >= 50 and y[sel].mean() >= PREC_HI: hi = float(t); break
    if hi is None: hi, fallback = float(np.quantile(s, 0.90)), True
    hi = max(hi, lo + 0.02)
    out[key] = {"lo": round(lo, 3), "hi": round(hi, 3)}
    print(f"{key:14} lo={lo:.3f} hi={hi:.3f}{' (FALLBACK: target precision not reachable)' if fallback else ''} | "
          f"base rate {y.mean():.2f} | below lo {np.mean(s <= lo):.0%} (true cases missed {y[s <= lo].mean():.2f}) | "
          f"above hi {np.mean(s >= hi):.0%} (precision {y[s >= hi].mean():.2f}) | rest = REVIEW {np.mean((s > lo) & (s < hi)):.0%}")
path.write_text(json.dumps(out, indent=2)); print("wrote", path)