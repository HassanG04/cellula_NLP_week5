import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase
from PIL import Image

from assistant.services.langgraph_code_service import classify_intent, validate_generated_python
from assistant.services.lstm_service import MODEL_PATH, TOKENIZER_PATH, classify_with_metadata
from assistant.services.rag_service import SimpleRAGService, simple_split_text


class FakeStore:
    def __init__(self):
        self.records = {}

    def add_texts(self, texts, ids, metadatas):
        self.records.update({identity: text for identity, text in zip(ids, texts)})

    def similarity_search_with_relevance_scores(self, query, k):
        return [(SimpleNamespace(page_content="Python source", metadata={"source": "guide"}), 0.9)]


class EngineeringTests(SimpleTestCase):
    def test_chunk_validation_and_idempotent_ingestion(self):
        with self.assertRaises(ValueError):
            simple_split_text("text", 0, 0)
        self.assertEqual(simple_split_text("abcdef", 4, 1), ["abcd", "def"])
        store = FakeStore()
        service = SimpleRAGService(
            store, SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="answer"))
        )
        service.add_documents(["same"])
        service.add_documents(["same"])
        self.assertEqual(len(store.records), 1)
        self.assertEqual(service.ask_with_metadata("Python")["sources"], ["guide"])

    def test_empty_retrieval_abstains(self):
        store = FakeStore()
        store.similarity_search_with_relevance_scores = lambda query, k: []
        self.assertEqual(
            SimpleRAGService(store, object()).ask("unknown"), "No relevant source was found."
        )

    def test_code_routing_and_syntax(self):
        self.assertEqual(classify_intent({"user_input": "explain this"})["intent"], "explain_code")
        self.assertTrue(validate_generated_python("```python\ndef add(a,b):\n return a+b\n```"))
        self.assertFalse(validate_generated_python("def broken("))

    def test_boots_without_provider_and_missing_classifier_is_safe(self):
        client = Client()
        self.assertEqual(client.get("/assistant/").status_code, 200)
        self.assertEqual(
            client.get("/assistant/health/").json()["lstm_artifacts_present"],
            MODEL_PATH.exists() and TOKENIZER_PATH.exists(),
        )
        with patch("assistant.views.classify_with_metadata", side_effect=RuntimeError("missing")):
            self.assertEqual(
                client.post("/assistant/classify/", {"text": "hello"}).status_code, 503
            )

    def test_caption_temporary_file_is_cleaned_on_success_and_failure(self):
        for fail in (False, True):
            visited = []

            def caption(path):
                visited.append(Path(path))
                self.assertTrue(Path(path).exists())
                if fail:
                    raise RuntimeError("model unavailable")
                return "a blue square"

            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
            upload = SimpleUploadedFile(
                "../../square.png", buffer.getvalue(), content_type="image/png"
            )
            with (
                patch("assistant.views.generate_caption", side_effect=caption),
                patch("assistant.views.lstm_classify") as classifier,
            ):
                response = Client().post("/assistant/image/", {"image": upload, "mode": "caption"})
                self.assertEqual(response.status_code, 503 if fail else 200)
                classifier.assert_not_called()
            self.assertEqual(len(visited), 1)
            self.assertFalse(visited[0].exists())

    @skipUnless(importlib.util.find_spec("tensorflow"), "optional TensorFlow runtime")
    def test_native_lstm_real_inference_contract(self):
        result = classify_with_metadata("How can I stay safe online?")
        self.assertEqual(len(result["probabilities"]), 9)
        self.assertAlmostEqual(sum(result["probabilities"]), 1, places=5)
        self.assertIn(
            result["predicted_label"],
            [
                "Child Sexual Exploitation",
                "Elections",
                "Non-Violent Crimes",
                "Safe",
                "Sex-Related Crimes",
                "Suicide & Self-Harm",
                "Unknown S-Type",
                "Violent Crimes",
                "unsafe",
            ],
        )
        self.assertTrue(result["model_version"].startswith("lstm-"))
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_csrf_is_enforced_and_invalid_upload_rejected(self):
        self.assertEqual(
            Client(enforce_csrf_checks=True)
            .post("/assistant/code/", {"user_input": "hello"})
            .status_code,
            403,
        )
        upload = SimpleUploadedFile("../../bad.txt", b"not image", content_type="image/png")
        self.assertEqual(Client().post("/assistant/image/", {"image": upload}).status_code, 400)

    @patch("assistant.views.run_code_assistant", return_value="def add(a,b): return a+b")
    def test_code_view_and_empty_input(self, _mock):
        self.assertEqual(
            Client().post("/assistant/code/", {"user_input": "generate add"}).status_code, 200
        )
        self.assertEqual(Client().post("/assistant/code/", {"user_input": ""}).status_code, 400)
