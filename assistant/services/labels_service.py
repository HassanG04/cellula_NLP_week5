# assistant/services/labels_service.py
import pickle
from pathlib import Path

# Project root ( .. / .. / .. from this file )
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LABEL_ENCODER_PATH = BASE_DIR / "lstm_label_encoder.pkl"

_label_encoder = None  # lazy-loaded


def _load_label_encoder():
    """Load the label encoder lazily, with a clear error message if missing."""
    global _label_encoder

    if _label_encoder is not None:
        return _label_encoder

    if not LABEL_ENCODER_PATH.exists():
        raise RuntimeError(
            f"Label encoder file not found: {LABEL_ENCODER_PATH}\n"
            "Make sure you saved your LabelEncoder as 'lstm_label_encoder.pkl' "
            "in the project root (same folder as manage.py), "
            "or update LABEL_ENCODER_PATH in labels_service.py."
        )

    with open(LABEL_ENCODER_PATH, "rb") as f:
        _label_encoder = pickle.load(f)

    return _label_encoder


def idx_to_label(idx: int) -> str:
    """Convert numeric class index to string label."""
    le = _load_label_encoder()
    return le.classes_[idx]
