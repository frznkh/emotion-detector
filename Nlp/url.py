from django.urls import path
from .views import EmotionDetectionAPIView

urlpatterns = [
    path('detect-emotion/', EmotionDetectionAPIView.as_view(), name='detect-emotion'),
]
