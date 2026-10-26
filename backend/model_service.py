"""Load the garment multi-task model once and run two-view inference."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "model" / "best_improved_garment_model.keras"
TASKS = ("condition", "pilling", "stains", "holes")
NUM_CLASSES = {"condition": 5, "pilling": 5, "stains": 3, "holes": 3}

_model = None
_tf = None


def mock_enabled() -> bool:
    return os.getenv("MOCK_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _softmax_sample(n: int) -> list[float]:
    raw = [random.random() + 0.05 for _ in range(n)]
    total = sum(raw)
    return [x / total for x in raw]


def mock_predict(_front_bytes: bytes, _back_bytes: bytes) -> dict[str, list[float]]:
    return {task: _softmax_sample(NUM_CLASSES[task]) for task in TASKS}


def load_model() -> None:
    global _model, _tf
    if mock_enabled():
        _model = "mock"
        return
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Place the .keras file there or set MOCK_MODEL=1."
        )
    import tensorflow as tf  # lazy: not required in mock mode

    _tf = tf
    _model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)


def _decode_resize(image_bytes: bytes):
    tf = _tf
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [float(x) for x in value]


def predict(front_img: bytes, back_img: bytes) -> dict[str, list[float]]:
    if mock_enabled() or _model == "mock":
        return mock_predict(front_img, back_img)
    if _model is None or _tf is None:
        raise RuntimeError("Model is not loaded.")

    front = _decode_resize(front_img)
    back = _decode_resize(back_img)
    outputs = _model({"front_image": front, "back_image": back}, training=False)

    result: dict[str, list[float]] = {}
    if isinstance(outputs, dict):
        for task in TASKS:
            result[task] = _to_list(outputs[task])
        return result

    names = [getattr(layer, "name", "") for layer in getattr(_model, "output", [])] or list(TASKS)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    for name, tensor in zip(names, outputs):
        result[name if name in TASKS else TASKS[len(result)]] = _to_list(tensor)
    for task in TASKS:
        result.setdefault(task, _softmax_sample(NUM_CLASSES[task]))
    return result
