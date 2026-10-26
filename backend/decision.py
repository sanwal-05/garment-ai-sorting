"""Rule-based return decision from model scores, claim category, and customer text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

BACKEND_DIR = Path(__file__).resolve().parent
THRESHOLDS_PATH = BACKEND_DIR / "thresholds.json"

SCORE_KEYS = ("stain_present", "hole_major", "heavily_worn")

CLAIM_EVIDENCE = {
    "arrived_stained": "stain_present",
    "arrived_damaged": "hole_major",
    "poor_quality": "heavily_worn",
}

NON_DEFECT_CLAIMS = {"wrong_size", "not_as_described", "changed_mind"}

CUSTOMER_STATUS = {
    "ACCEPT": "approved",
    "REVIEW": "under_review",
    "REJECT": "declined",
}


def load_thresholds() -> dict[str, dict[str, float]]:
    with THRESHOLDS_PATH.open() as f:
        data = json.load(f)
    return {key: {"lo": float(data[key]["lo"]), "hi": float(data[key]["hi"])} for key in SCORE_KEYS}


def _as_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return [float(x) for x in value]
    raise TypeError(f"Expected probability vector, got {type(value)!r}")


def _level(score: float, lo: float, hi: float) -> str:
    if score >= hi:
        return "hi"
    if score <= lo:
        return "lo"
    return "mid"


def scores_from_probs(probs: Mapping[str, Any]) -> dict[str, float]:
    """Collapse model heads into the active vetting scores (hole_major replaces flawed hole_present)."""
    stains = _as_list(probs["stains"])
    holes = _as_list(probs["holes"])
    condition = _as_list(probs["condition"])
    pilling = _as_list(probs["pilling"])

    stain_none, stain_minor, stain_major = (stains + [0.0, 0.0, 0.0])[:3]
    stain_p = float(stain_minor + stain_major) if len(stains) >= 3 else float((stains + [0.0, 0.0])[1])

    # For holes: if 3-class (none, minor, major), use major class. If 2-class, use hole prob as major damage score.
    if len(holes) >= 3:
        hole_major = float(holes[2])
        hole_none = float(holes[0])
    else:
        h_arr = (holes + [0.0, 0.0])[:2]
        hole_none = float(h_arr[0])
        hole_major = float(h_arr[1])

    # Dataset grades: condition/pilling 1 = worst, 5 = best (or 3-class [low, mid, high]).
    if len(condition) >= 5:
        heavily_worn = float(0.65 * (condition[0] + condition[1]) + 0.35 * (pilling[0] + pilling[1]))
    else:
        heavily_worn = float(0.65 * condition[0] + 0.35 * pilling[0])

    return {
        "stain_present": stain_p,
        "hole_major": hole_major,
        "heavily_worn": heavily_worn,
        "stain_none": float(stain_none),
        "hole_none": hole_none,
    }


def _disposition(levels: Mapping[str, str]) -> str:
    if levels.get("hole_major") == "hi":
        return "RECYCLE"
    if levels.get("stain_present") == "hi":
        return "REFURBISH"
    if levels.get("heavily_worn") == "hi":
        return "RESELL_AS_USED"
    if any(levels[key] == "mid" for key in SCORE_KEYS):
        return "INSPECT_MANUALLY"
    return "RESELL_AS_NEW"


def decide(scores: Mapping[str, float], claim: str, text: str) -> dict[str, Any]:
    thresholds = load_thresholds()
    levels = {
        key: _level(float(scores[key]), thresholds[key]["lo"], thresholds[key]["hi"])
        for key in SCORE_KEYS
    }
    reasons: list[str] = []
    claim = (claim or "").strip()
    text = (text or "").strip()

    if claim in CLAIM_EVIDENCE:
        key = CLAIM_EVIDENCE[claim]
        level = levels[key]
        if level == "hi":
            decision = "ACCEPT"
            reasons.append(f"Photos support the '{claim}' claim ({key} above the high threshold).")
        elif level == "lo":
            decision = "REJECT"
            reasons.append(f"Photos do not support the '{claim}' claim ({key} below the low threshold).")
        else:
            decision = "REVIEW"
            reasons.append(f"Evidence for '{claim}' is inconclusive ({key} between thresholds).")
    elif claim in NON_DEFECT_CLAIMS:
        severe = levels.get("hole_major") == "hi" or levels.get("stain_present") == "hi"
        clean = all(levels[key] == "lo" for key in SCORE_KEYS)
        if clean:
            decision = "ACCEPT"
            reasons.append("Item appears intact; non-defect claim can be processed.")
        elif severe:
            decision = "REVIEW"
            reasons.append("Non-defect claim, but photos show significant damage/stains that need a human check.")
        else:
            decision = "REVIEW"
            reasons.append("Non-defect claim with mixed visual quality; send to review.")
    else:
        decision = "REVIEW"
        reasons.append(f"Unrecognised claim category '{claim}'.")

    if levels.get("hole_major") == "hi":
        reasons.append("Major-hole score is high (significant damage detected).")
    if levels.get("stain_present") == "hi":
        reasons.append("Stain-present score is high.")
    if levels.get("heavily_worn") == "hi":
        reasons.append("Garment appears heavily worn.")
    if len(text) < 40 and decision == "REVIEW":
        reasons.append("Customer description is brief, which adds uncertainty.")

    disposition = _disposition(levels)
    if decision == "REVIEW":
        disposition = "INSPECT_MANUALLY"

    status = CUSTOMER_STATUS[decision]
    if decision == "REVIEW" and claim in CLAIM_EVIDENCE and len(text) < 40:
        status = "needs_more_info"

    return {
        "decision": decision,
        "disposition": disposition,
        "reasons": reasons,
        "levels": levels,
        "status": status,
        "thresholds": thresholds,
    }
