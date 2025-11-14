from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpRequest

from .services.langgraph_code_service import run_code_assistant
from .services.rag_service import rag_service
from .services.lstm_service import lstm_classify
from .services.labels_service import idx_to_label
from .services.image_caption_service import generate_caption
from pathlib import Path
import os

@csrf_exempt
def home(request: HttpRequest):
    return render(request, "assistant/home.html")

@csrf_exempt
def code_assistant_view(request: HttpRequest):
    result = None
    user_input = ""
    if request.method == "POST":
        user_input = request.POST.get("user_input", "")
        if user_input:
            result = run_code_assistant(user_input)
    return render(request, "assistant/code_assistant.html", {
        "user_input": user_input,
        "result": result,
    })

@csrf_exempt
def rag_view(request: HttpRequest):
    answer = None
    question = ""
    if request.method == "POST":
        question = request.POST.get("question", "")
        if question:
            answer = rag_service.ask(question)
    return render(request, "assistant/rag.html", {
        "question": question,
        "answer": answer,
    })

@csrf_exempt
def classify_view(request: HttpRequest):
    text = ""
    label = None
    if request.method == "POST":
        text = request.POST.get("text", "")
        if text:
            idx = lstm_classify(text)
            label = idx_to_label(idx)
    return render(request, "assistant/classify.html", {
        "text": text,
        "label": label,
    })

@csrf_exempt
def image_view(request: HttpRequest):
    caption = None
    label = None
    if request.method == "POST" and request.FILES.get("image"):
        img = request.FILES["image"]
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        img_path = upload_dir / img.name
        with open(img_path, "wb") as f:
            for chunk in img.chunks():
                f.write(chunk)

        caption = generate_caption(str(img_path))
        idx = lstm_classify(caption)
        label = idx_to_label(idx)

        # optionally delete file
        os.remove(img_path)

    return render(request, "assistant/image.html", {
        "caption": caption,
        "label": label,
    })
