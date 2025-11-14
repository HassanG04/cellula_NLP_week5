# assistant/services/lstm_service.py
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Project root (.. / .. / .. from this file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "lstm_text_classifier.h5"
TOKENIZER_PATH = BASE_DIR / "lstm_tokenizer.pkl"
MAX_LEN = 100

# Lazy globals (loaded only when needed)
_model = None
_tokenizer = None


def _load_model_and_tokenizer():
    """Load the LSTM model and tokenizer lazily, with clear error messages."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"LSTM model file not found: {MODEL_PATH}\n"
            "Put your trained 'lstm_text_classifier.h5' in the project root "
            "(same folder as manage.py) or update MODEL_PATH."
        )

    if not TOKENIZER_PATH.exists():
        raise RuntimeError(
            f"Tokenizer file not found: {TOKENIZER_PATH}\n"
            "Put your 'lstm_tokenizer.pkl' in the project root "
            "(same folder as manage.py) or update TOKENIZER_PATH."
        )

    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    return _model, _tokenizer


def lstm_classify(text: str) -> int:
    """
    Classify a text using the LSTM model.
    Returns the predicted class index as an int.
    """
    model, tokenizer = _load_model_and_tokenizer()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    preds = model.predict(padded)
    return int(preds.argmax(axis=1)[0])
