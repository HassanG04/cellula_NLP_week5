import hashlib
import json
import pickle
from pathlib import Path

import tensorflow as tf

root = Path(__file__).resolve().parents[1]
output = root / "models"
output.mkdir(exist_ok=True)
model = tf.keras.models.load_model(root / "lstm_text_classifier.h5", compile=False)
with (root / "lstm_tokenizer.pkl").open("rb") as stream:
    tokenizer = pickle.load(stream)
with (root / "lstm_label_encoder.pkl").open("rb") as stream:
    labels = pickle.load(stream).classes_.tolist()
model.save(output / "toxic_lstm.keras")
(output / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
(output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
(output / "metadata.json").write_text(
    json.dumps(
        {
            "source": "Recovered from Quantization_Cellula_Week_2/code.rar; tokenizer and label encoder matched byte-for-byte",
            "max_length": 100,
            "padding": "pre",
            "truncating": "pre",
            "num_classes": len(labels),
            "legacy_model_sha256": hashlib.sha256(
                (root / "lstm_text_classifier.h5").read_bytes()
            ).hexdigest(),
            "converted_with_tensorflow": tf.__version__,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print({"input_shape": model.input_shape, "output_shape": model.output_shape, "labels": labels})
