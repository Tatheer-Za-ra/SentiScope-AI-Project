import os
import time
from functools import lru_cache


MODEL_NAME = "DistilBERT Sentiment Transformer"
HF_MODEL_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# This is a text-only NLP project. These flags prevent optional vision helpers
# from being used by Transformers, so torchvision is not required.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("USE_TORCHVISION", "0")


def normalize_transformer_label(label):
    """Convert model labels into the project labels: Positive, Negative, Neutral."""
    label_text = str(label).strip().lower()

    if "positive" in label_text or label_text in {"pos", "label_2", "2"}:
        return "Positive"
    if "negative" in label_text or label_text in {"neg", "label_0", "0"}:
        return "Negative"
    if "neutral" in label_text or label_text in {"neu", "label_1", "1"}:
        return "Neutral"

    return "Neutral"


@lru_cache(maxsize=1)
def load_distilbert_model():
    """
    Load the DistilBERT tokenizer and transformer model once.

    Tokenizer:
    The tokenizer converts raw text into token IDs that the transformer can read.
    It also adds padding/truncation so the input has a model-friendly shape.

    Transformer model:
    DistilBERT is a smaller, faster version of BERT. It reads the full sentence
    context and outputs class scores for sentiment labels.
    """
    try:
        import torch
        import transformers.utils.import_utils as transformers_import_utils
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "DistilBERT dependencies are missing. Run: pip install -r requirements.txt"
        ) from error

    # Extra safety for environments where Transformers checks optional vision
    # packages while loading text models. Sentiment analysis does not need them.
    transformers_import_utils._torchvision_available = False

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()

    return tokenizer, model, torch


def predict_distilbert_sentiment(text):
    """
    Predict sentiment using DistilBERT.

    Prediction process:
    1. Load tokenizer and model.
    2. Tokenize the raw user text.
    3. Send token IDs through DistilBERT.
    4. Convert output scores into probabilities.
    5. Return the highest-confidence sentiment label.
    """
    start_time = time.perf_counter()

    try:
        tokenizer, model, torch = load_distilbert_model()

        encoded_text = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**encoded_text)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = round(float(probabilities[predicted_index].item()) * 100, 2)

        label_lookup = model.config.id2label
        raw_label = label_lookup.get(predicted_index, str(predicted_index))
        sentiment = normalize_transformer_label(raw_label)

        processing_time = time.perf_counter() - start_time

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "processing_time": processing_time,
            "model_name": MODEL_NAME,
            "error": None,
        }

    except Exception as error:
        processing_time = time.perf_counter() - start_time
        return {
            "sentiment": "Unavailable",
            "confidence": 0.0,
            "processing_time": processing_time,
            "model_name": MODEL_NAME,
            "error": str(error),
        }
