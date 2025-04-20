
from django.urls import path
from .views import GenerateOutfitAPIView

urlpatterns = [
    path('generate-outfit/', GenerateOutfitAPIView.as_view(), name='generate_outfit'),
]
