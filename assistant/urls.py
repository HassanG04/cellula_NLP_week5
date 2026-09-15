# assistant/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="assistant_home"),
    path("health/", views.health, name="health"),
    path("code/", views.code_assistant_view, name="code_assistant"),
    path("rag/", views.rag_view, name="rag"),
    path("classify/", views.classify_view, name="classify"),
    path("image/", views.image_view, name="image"),
]
