import json
from functools import lru_cache
from pathlib import Path


@lru_cache
def _labels():
    path = Path(__file__).resolve().parents[2] / "models/labels.json"
    return json.loads(path.read_text(encoding="utf-8"))


def idx_to_label(index):
    labels = _labels()
    if not 0 <= index < len(labels):
        raise ValueError("class index is outside the label schema")
    return labels[index]
