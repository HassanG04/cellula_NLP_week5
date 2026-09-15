from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models/toxic_lstm.keras"
TOKENIZER_PATH = BASE_DIR / "models/tokenizer.json"
_model = None
_tokenizer = None
_version = "unloaded"


def _load_model_and_tokenizer():
    global _model, _tokenizer, _version
    if _model is not None:
        return _model, _tokenizer
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise RuntimeError("Classifier model or tokenizer is missing")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    _version = "lstm-" + hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    return _model, _tokenizer


def classify_with_metadata(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("text must contain 1 to 10000 characters")
    model, tokenizer = _load_model_and_tokenizer()
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    padded = pad_sequences(
        tokenizer.texts_to_sequences([text]), maxlen=100, padding="pre", truncating="pre"
    )
    started = time.perf_counter()
    probabilities = model(padded, training=False).numpy()[0]
    labels = json.loads((BASE_DIR / "models/labels.json").read_text(encoding="utf-8"))
    if (
        len(probabilities) != len(labels)
        or not np.isfinite(probabilities).all()
        or not np.isclose(probabilities.sum(), 1, atol=1e-5)
    ):
        raise RuntimeError("Invalid classifier probability schema")
    index = int(probabilities.argmax())
    return {
        "predicted_index": index,
        "predicted_label": labels[index],
        "probabilities": probabilities.tolist(),
        "model_version": _version,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def lstm_classify(text):
    return classify_with_metadata(text)["predicted_index"]
