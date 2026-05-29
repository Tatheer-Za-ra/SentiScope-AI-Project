# SentiScope AI

SentiScope AI is a BSCS Natural Language Processing semester project. Version 3 is a professional sentiment analysis platform that compares a traditional NLP model with a modern transformer-based NLP model.

## Project Overview

The project analyzes text such as tweets, reviews, and comments. It shows how the same sentiment analysis problem can be solved using:

- Traditional NLP: TF-IDF + Logistic Regression
- Modern NLP: DistilBERT Transformer

The app compares both models using prediction output, confidence score, processing time, and evaluation metrics.

## Objective

The objective is to demonstrate a complete NLP comparison workflow:

1. Preprocess text
2. Train a traditional ML sentiment model
3. Run DistilBERT sentiment prediction
4. Compare both approaches
5. Generate report-ready metrics and confusion matrices

## Dataset Description

The project uses a Twitter sentiment CSV dataset stored inside the `data/` folder.

Current dataset path:

```text
data/Twitter_Data.csv
```

Current columns:

```text
clean_text = tweet text
category = sentiment label
```

The `category` labels are mapped as:

```text
-1 = Negative
 0 = Neutral
 1 = Positive
```

If your dataset file or columns are different, edit these values in `train_traditional.py`:

```python
DATASET_PATH = Path("data/Twitter_Data.csv")
TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"
```

## Traditional NLP Approach

The traditional model uses:

- TF-IDF Vectorizer
- Logistic Regression Classifier
- Train/test split
- Weighted evaluation metrics

This approach is fast, lightweight, and easy to explain.

## Modern NLP Approach

The modern model uses DistilBERT through Hugging Face Transformers.

DistilBERT is a smaller and faster version of BERT. It understands context better than frequency-based models, but it requires more computation and takes longer during inference.

## TF-IDF Explanation

TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numbers by measuring how important a word is in a sentence compared with the full dataset.

TF-IDF is useful because it gives more importance to meaningful words and less importance to very common words.

## DistilBERT Explanation

DistilBERT uses a transformer architecture. Instead of only counting words, it reads the sentence context and learns relationships between words.

Example:

```text
The movie was not bad.
```

A transformer can better understand that this sentence may not be negative because it considers context.

## System Architecture

```text
User Text
   |
   |-- Preprocessing Visualization
   |
   |-- Traditional NLP
   |      |-- TF-IDF Vectorizer
   |      |-- Logistic Regression
   |
   |-- Modern NLP
          |-- DistilBERT Tokenizer
          |-- DistilBERT Transformer Model
```

## Evaluation Metrics

The project compares:

- Accuracy
- Precision
- Recall
- F1 Score
- Average Confidence
- Average Prediction Time
- Confusion Matrix

## Results Comparison

The training workflow generates report assets inside `report_assets/`:

```text
traditional_metrics.txt
distilbert_metrics.txt
traditional_confusion_matrix.png
distilbert_confusion_matrix.png
model_comparison.csv
```

These files can be used in the final report, presentation slides, and screenshots.

## Installation Instructions

Install all required packages:

```bash
pip install -r requirements.txt
```

On Windows, if `pip` or `streamlit` is not recognized, use:

```bash
python -m pip install -r requirements.txt
```

## Training Instructions

Train the traditional model and generate evaluation assets:

```bash
python train_traditional.py
```

This saves:

```text
models/traditional_model.pkl
models/tfidf_vectorizer.pkl
report_assets/model_comparison.csv
```

DistilBERT evaluation is also attempted from the same test split. By default it uses a sample for practical local execution. You can increase this in `train_traditional.py`:

```python
DISTILBERT_EVAL_SAMPLE_SIZE = 500
```

## Running Instructions

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized:

```bash
python -m streamlit run app.py
```

## Project Screenshots

Add final screenshots inside the `screenshots/` folder.

Recommended screenshots:

- Home page
- Analyze page with both model results
- Model Comparison page
- Performance Metrics page
- Preprocessing visualization

## Project Structure

```text
SentiScope AI/
|-- app.py
|-- preprocessing.py
|-- predict.py
|-- bert_predict.py
|-- train_traditional.py
|-- requirements.txt
|-- README.md
|-- data/
|-- models/
|-- screenshots/
`-- report_assets/
```
