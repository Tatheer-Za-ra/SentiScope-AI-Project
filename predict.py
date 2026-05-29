import pickle
import time
from pathlib import Path

from preprocessing import preprocess_text


MODEL_NAME = "TF-IDF + Logistic Regression"
MODEL_PATH = Path("models/traditional_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")


def model_files_exist():
    """Check whether the trained model and vectorizer files are available."""
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


def load_model_and_vectorizer():
    """Load the saved Logistic Regression model and TF-IDF vectorizer."""
    if not model_files_exist():
        raise FileNotFoundError(
            "Traditional model not found. Please run python train_traditional.py first."
        )

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


def predict_sentiment(text):
    """
    Predict sentiment for new text using the trained traditional ML model.

    Returns:
    - predicted sentiment
    - confidence score
    - processing time
    - final preprocessed text
    """
    start_time = time.perf_counter()
    model, vectorizer = load_model_and_vectorizer()

    preprocessing_steps = preprocess_text(text)
    processed_text = preprocessing_steps["final_text"]
    text_features = vectorizer.transform([processed_text])

    predicted_sentiment = model.predict(text_features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_features)[0]
        confidence = round(float(max(probabilities)) * 100, 2)
    else:
        confidence = 0.0

    processing_time = time.perf_counter() - start_time

    return {
        "sentiment": predicted_sentiment,
        "confidence": confidence,
        "processing_time": processing_time,
        "processed_text": processed_text,
    }
