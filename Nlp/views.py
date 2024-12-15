from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token="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")


class EmotionDetectionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        persian_text = request.data.get("persian_text", "")
        if not persian_text:
            return Response({"error": "متن ورودی خالی است."}, status=status.HTTP_400_BAD_REQUEST)

        inputs = tokenizer(persian_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        class_labels = ['منفی', 'خنثی', 'مثبت']
        result = class_labels[predicted_class]

        return Response({"result": result}, status=status.HTTP_200_OK)
