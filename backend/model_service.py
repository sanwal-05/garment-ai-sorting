"""Load the garment multi-task model once and run two-view inference.

Supports:
1. Production bundle: backend/model/garment_bundle.joblib (Vision Transformer embeddings + LightGBM/Logistic ensemble)
2. Keras model: backend/model/best_improved_garment_model.keras (Dual-view EfficientNet-B0)
3. Mock fallback when MOCK_MODEL=1 or model files are unavailable.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger("uvicorn.error")

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "model"
JOBLIB_PATH = MODEL_DIR / "garment_bundle.joblib"
KERAS_PATH = MODEL_DIR / "best_improved_garment_model.keras"

TASKS = ("condition", "pilling", "stains", "holes")
NUM_CLASSES = {"condition": 3, "pilling": 3, "stains": 2, "holes": 2}

_model: Any = None
_model_type: str = "mock"
_tf = None


def mock_enabled() -> bool:
    return os.getenv("MOCK_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _softmax_sample(n: int) -> list[float]:
    raw = [random.random() + 0.05 for _ in range(n)]
    total = sum(raw)
    return [round(x / total, 4) for x in raw]


def mock_predict(_front_bytes: bytes, _back_bytes: bytes) -> dict[str, list[float]]:
    return {task: _softmax_sample(NUM_CLASSES[task]) for task in TASKS}


def load_model() -> None:
    global _model, _model_type, _tf
    if mock_enabled():
        logger.info("MOCK_MODEL enabled. Using mock inference service.")
        _model = "mock"
        _model_type = "mock"
        return

    # 1. Try joblib production ensemble
    if JOBLIB_PATH.is_file():
        try:
            from backend.predict import Predictor

            logger.info("Loading production ensemble from %s...", JOBLIB_PATH)
            _model = Predictor(str(JOBLIB_PATH))
            _model_type = "joblib"
            logger.info("Successfully loaded garment_bundle.joblib.")
            return
        except Exception as exc:
            logger.warning("Could not load joblib bundle (%s). Checking alternatives.", exc)

    # 2. Try Keras model if available
    if KERAS_PATH.is_file():
        try:
            import tensorflow as tf

            _tf = tf
            logger.info("Loading Keras model from %s...", KERAS_PATH)
            _model = tf.keras.models.load_model(str(KERAS_PATH), safe_mode=False)
            _model_type = "keras"
            logger.info("Successfully loaded Keras model.")
            return
        except Exception as exc:
            logger.warning("Could not load Keras model (%s).", exc)

    # 3. Fallback to mock mode so service remains resilient
    logger.warning("No functional deep learning model loaded. Defaulting to mock inference mode.")
    _model = "mock"
    _model_type = "mock"


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


def predict(front_img: bytes, back_img: bytes, garment_type: str = "") -> dict[str, list[float]]:
    if mock_enabled() or _model == "mock" or _model_type == "mock":
        return mock_predict(front_img, back_img)

    if _model_type == "joblib" and _model is not None:
        try:
            out = _model.predict([front_img], [back_img], [garment_type])
            result: dict[str, list[float]] = {}
            for task in TASKS:
                if task in out:
                    result[task] = _to_list(out[task][0])
                else:
                    result[task] = _softmax_sample(NUM_CLASSES[task])
            return result
        except Exception as exc:
            logger.error("Joblib predict failed: %s. Falling back to mock.", exc)
            return mock_predict(front_img, back_img)

    if _model_type == "keras" and _model is not None and _tf is not None:
        try:
            front = _decode_resize(front_img)
            back = _decode_resize(back_img)
            outputs = _model({"front_image": front, "back_image": back}, training=False)

            result = {}
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
        except Exception as exc:
            logger.error("Keras predict failed: %s. Falling back to mock.", exc)
            return mock_predict(front_img, back_img)

    return mock_predict(front_img, back_img)
