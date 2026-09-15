import logging
import os
import tempfile
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image, UnidentifiedImageError

from .services.image_caption_service import generate_caption
from .services.labels_service import idx_to_label
from .services.langgraph_code_service import run_code_assistant
from .services.lstm_service import MODEL_PATH, TOKENIZER_PATH, classify_with_metadata, lstm_classify
from .services.rag_service import rag_service

logger = logging.getLogger(__name__)


def home(request):
    return render(request, "assistant/home.html")


def health(_request):
    return JsonResponse(
        {
            "status": "ok",
            "provider_configured": bool(os.getenv("OPENAI_API_KEY")),
            "lstm_artifacts_present": MODEL_PATH.exists() and TOKENIZER_PATH.exists(),
        }
    )


def _text(request, field):
    value = request.POST.get(field, "").strip()
    if not value or len(value) > 10000:
        raise ValueError(f"{field} must contain 1 to 10000 characters")
    return value


def _feature_view(request, template, field, operation):
    context = {field: request.POST.get(field, "")}
    status = 200
    if request.method == "POST":
        try:
            context.update(operation(_text(request, field)))
        except ValueError as exc:
            context["error"], status = str(exc), 400
        except Exception:
            logger.exception("Assistant feature failed: %s", field)
            context["error"], status = (
                "This feature is unavailable. Check its configured dependencies and artifacts.",
                503,
            )
    return render(request, template, context, status=status)


def code_assistant_view(request):
    return _feature_view(
        request,
        "assistant/code_assistant.html",
        "user_input",
        lambda text: {"result": run_code_assistant(text)},
    )


def rag_view(request):
    return _feature_view(request, "assistant/rag.html", "question", rag_service.ask_with_metadata)


def classify_view(request):
    return _feature_view(request, "assistant/classify.html", "text", classify_with_metadata)


def image_view(request):
    context, status = {}, 200
    path = None
    if request.method == "POST":
        try:
            image = request.FILES.get("image")
            if image is None or image.size > 5 * 1024 * 1024:
                raise ValueError("Upload an image of at most 5 MB")
            try:
                Image.open(image).verify()
                image.seek(0)
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError("Upload a valid image") from exc
            with tempfile.NamedTemporaryFile(suffix=".image", delete=False) as temporary:
                path = Path(temporary.name)
                for chunk in image.chunks():
                    temporary.write(chunk)
            context["caption"] = generate_caption(str(path))
            if request.POST.get("mode", "caption") == "classify":
                context["label"] = idx_to_label(lstm_classify(context["caption"]))
        except ValueError as exc:
            context["error"], status = str(exc), 400
        except Exception:
            logger.exception("Image workflow failed")
            context["error"], status = "Image processing is unavailable.", 503
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
    return render(request, "assistant/image.html", context, status=status)
