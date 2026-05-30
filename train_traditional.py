import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_text


# Place your Twitter sentiment CSV inside the data/ folder.
# Change this path if your file has a different name.
DATASET_PATH = Path("data/Twitter_Data.csv")

# If auto-detection chooses the wrong columns, set these manually.
# Example:
# TEXT_COLUMN = "tweet"
# LABEL_COLUMN = "sentiment"
TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"

MODEL_PATH = Path("models/traditional_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
RESULTS_PATH = Path("report_assets/traditional_model_results.txt")
TRADITIONAL_METRICS_PATH = Path("report_assets/traditional_metrics.txt")
DISTILBERT_METRICS_PATH = Path("report_assets/distilbert_metrics.txt")
TRADITIONAL_CM_PATH = Path("report_assets/traditional_confusion_matrix.png")
DISTILBERT_CM_PATH = Path("report_assets/distilbert_confusion_matrix.png")
MODEL_COMPARISON_PATH = Path("report_assets/model_comparison.csv")

# DistilBERT can be slow on CPU. This evaluates it on a sample from the same
# test split. Increase this number for a stronger final report.
DISTILBERT_EVAL_SAMPLE_SIZE = 500

TEXT_COLUMN_CANDIDATES = ["text", "tweet", "content", "review", "comment", "clean_text"]
LABEL_COLUMN_CANDIDATES = ["sentiment", "label", "class", "category", "target"]


def detect_column(columns, candidates):
    """Find a matching column name using common dataset column names."""
    normalized_columns = {column.lower().strip(): column for column in columns}

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


def get_text_and_label_columns(dataframe):
    """Auto-detect text and label columns, unless manually configured above."""
    text_column = TEXT_COLUMN or detect_column(dataframe.columns, TEXT_COLUMN_CANDIDATES)
    label_column = LABEL_COLUMN or detect_column(dataframe.columns, LABEL_COLUMN_CANDIDATES)

    if text_column is None or label_column is None:
        print("Available columns in your CSV:")
        for column in dataframe.columns:
            print(f"- {column}")

        raise ValueError(
            "Could not auto-detect the text or label column. "
            "Open train_traditional.py and set TEXT_COLUMN and LABEL_COLUMN manually."
        )

    return text_column, label_column


def map_label(label):
    """Clean dataset labels and map them to Positive, Negative, or Neutral."""
    value = str(label).strip().lower()

    # Many Twitter sentiment datasets store labels as numbers.
    # For your dataset:
    # -1 = Negative, 0 = Neutral, 1 = Positive
    # Pandas may read these as -1.0, 0.0, and 1.0, so we handle both forms.
    try:
        numeric_value = int(float(value))
        if numeric_value == -1:
            return "Negative"
        if numeric_value == 0:
            return "Neutral"
        if numeric_value == 1:
            return "Positive"
    except ValueError:
        pass

    positive_values = {"positive", "pos", "1", "4", "happy", "good"}
    negative_values = {"negative", "neg", "-1", "sad", "bad"}
    neutral_values = {"neutral", "neu", "0", "2", "3", "none", "mixed"}

    if value in positive_values:
        return "Positive"
    if value in negative_values:
        return "Negative"
    if value in neutral_values:
        return "Neutral"

    if "positive" in value:
        return "Positive"
    if "negative" in value:
        return "Negative"
    if "neutral" in value:
        return "Neutral"

    return None


def prepare_dataset(dataframe, text_column, label_column):
    """Clean text and labels before model training."""
    working_data = dataframe[[text_column, label_column]].copy()
    working_data = working_data.dropna()

    working_data["sentiment"] = working_data[label_column].apply(map_label)
    working_data = working_data.dropna(subset=["sentiment"])

    working_data["processed_text"] = working_data[text_column].astype(str).apply(
        lambda text: preprocess_text(text)["final_text"]
    )
    working_data = working_data[working_data["processed_text"].str.strip() != ""]

    return working_data


def print_label_summary(dataframe, label_column, prepared_data):
    """Print raw and cleaned label counts for easier debugging."""
    print("\nRaw label values:")
    print(dataframe[label_column].value_counts(dropna=False).head(10))

    print("\nMapped sentiment values:")
    print(prepared_data["sentiment"].value_counts())


def save_results(report_text):
    """Save model evaluation results for the project report."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(report_text)


def save_text_file(path, text):
    """Save a plain text report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def save_confusion_matrix_image(matrix, labels, title, output_path):
    """Save a confusion matrix image for report and presentation use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def calculate_metric_row(model_name, y_true, y_pred, average_confidence, average_time):
    """Create one row of comparison metrics."""
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Average Confidence": average_confidence,
        "Average Prediction Time": average_time,
    }


def evaluate_distilbert(test_texts, y_test):
    """
    Evaluate DistilBERT on a sample from the same test split.

    This keeps the workflow local and practical. For a larger final evaluation,
    increase DISTILBERT_EVAL_SAMPLE_SIZE near the top of this file.
    """
    from bert_predict import predict_distilbert_sentiment

    sample_size = min(DISTILBERT_EVAL_SAMPLE_SIZE, len(test_texts))
    sampled_texts = test_texts.iloc[:sample_size]
    sampled_labels = y_test.iloc[:sample_size]

    predictions = []
    confidences = []
    processing_times = []

    print(f"\nEvaluating DistilBERT on {sample_size} test rows...")

    for index, text in enumerate(sampled_texts, start=1):
        result = predict_distilbert_sentiment(str(text))

        if result["error"]:
            raise RuntimeError(f"DistilBERT evaluation failed: {result['error']}")

        predictions.append(result["sentiment"])
        confidences.append(result["confidence"])
        processing_times.append(result["processing_time"])

        if index % 50 == 0:
            print(f"DistilBERT evaluated {index}/{sample_size} rows")

    average_confidence = sum(confidences) / len(confidences)
    average_time = sum(processing_times) / len(processing_times)

    return sampled_labels, predictions, average_confidence, average_time


def main():
    """Train and save the traditional ML sentiment model."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Place your Twitter sentiment CSV in the data/ folder or update DATASET_PATH."
        )

    dataframe = pd.read_csv(DATASET_PATH, encoding_errors="ignore")
    text_column, label_column = get_text_and_label_columns(dataframe)

    print(f"Using text column: {text_column}")
    print(f"Using label column: {label_column}")

    prepared_data = prepare_dataset(dataframe, text_column, label_column)
    print_label_summary(dataframe, label_column, prepared_data)

    if prepared_data.empty:
        raise ValueError("No usable rows found after cleaning labels and text.")

    label_counts = prepared_data["sentiment"].value_counts()
    stratify_labels = prepared_data["sentiment"] if label_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        prepared_data["processed_text"],
        prepared_data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_vectors, y_train)

    prediction_start = time.perf_counter()
    predictions = model.predict(x_test_vectors)
    traditional_total_time = time.perf_counter() - prediction_start
    traditional_probabilities = model.predict_proba(x_test_vectors)
    traditional_confidences = traditional_probabilities.max(axis=1) * 100

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["Positive", "Negative", "Neutral"])
    traditional_average_confidence = float(traditional_confidences.mean())
    traditional_average_time = traditional_total_time / len(x_test)

    report_text = f"""
SentiScope AI - Traditional ML Model Results
Model: TF-IDF + Logistic Regression
Dataset path: {DATASET_PATH}
Text column: {text_column}
Label column: {label_column}
Total usable rows: {len(prepared_data)}

Accuracy: {accuracy:.4f}

Precision, Recall, F1-score:
{report}

Confusion Matrix
Labels order: Positive, Negative, Neutral
{matrix}
"""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    save_results(report_text)
    save_text_file(TRADITIONAL_METRICS_PATH, report_text)
    save_confusion_matrix_image(
        matrix,
        ["Positive", "Negative", "Neutral"],
        "Traditional Model Confusion Matrix",
        TRADITIONAL_CM_PATH,
    )

    comparison_rows = [
        calculate_metric_row(
            "TF-IDF + Logistic Regression",
            y_test,
            predictions,
            traditional_average_confidence,
            traditional_average_time,
        )
    ]

    try:
        distilbert_y_test, distilbert_predictions, distilbert_avg_confidence, distilbert_avg_time = evaluate_distilbert(
            x_test,
            y_test,
        )
        distilbert_report = classification_report(
            distilbert_y_test,
            distilbert_predictions,
            zero_division=0,
        )
        distilbert_matrix = confusion_matrix(
            distilbert_y_test,
            distilbert_predictions,
            labels=["Positive", "Negative", "Neutral"],
        )
        distilbert_metrics_text = f"""
SentiScope AI - DistilBERT Model Results
Model: DistilBERT Sentiment Transformer
Dataset path: {DATASET_PATH}
Evaluation rows: {len(distilbert_y_test)}

Accuracy: {accuracy_score(distilbert_y_test, distilbert_predictions):.4f}

Precision, Recall, F1-score:
{distilbert_report}

Confusion Matrix
Labels order: Positive, Negative, Neutral
{distilbert_matrix}
"""
        save_text_file(DISTILBERT_METRICS_PATH, distilbert_metrics_text)
        save_confusion_matrix_image(
            distilbert_matrix,
            ["Positive", "Negative", "Neutral"],
            "DistilBERT Confusion Matrix",
            DISTILBERT_CM_PATH,
        )
        comparison_rows.append(
            calculate_metric_row(
                "DistilBERT Sentiment Transformer",
                distilbert_y_test,
                distilbert_predictions,
                distilbert_avg_confidence,
                distilbert_avg_time,
            )
        )
        print(distilbert_metrics_text)
    except Exception as error:
        distilbert_metrics_text = (
            "DistilBERT evaluation was not completed.\n"
            f"Reason: {error}\n\n"
            "Install dependencies and ensure the Hugging Face model can be downloaded, "
            "then run python train_traditional.py again."
        )
        save_text_file(DISTILBERT_METRICS_PATH, distilbert_metrics_text)
        comparison_rows.append(
            {
                "Model": "DistilBERT Sentiment Transformer",
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1 Score": None,
                "Average Confidence": None,
                "Average Prediction Time": None,
            }
        )
        print(distilbert_metrics_text)

    comparison_dataframe = pd.DataFrame(comparison_rows)
    comparison_dataframe.to_csv(MODEL_COMPARISON_PATH, index=False)

    print(report_text)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Traditional metrics saved to: {TRADITIONAL_METRICS_PATH}")
    print(f"DistilBERT metrics saved to: {DISTILBERT_METRICS_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
