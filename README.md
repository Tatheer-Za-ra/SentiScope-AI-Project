# SentiScope AI

SentiScope AI is a BSCS Natural Language Processing semester project. Version 4 is a complete sentiment analytics platform with single text analysis, batch CSV analysis, model comparison, analytics dashboards, and report-ready exports.

## Project Overview

The project analyzes text such as tweets, reviews, and comments. It shows how the same sentiment analysis problem can be solved using:

- Traditional NLP: TF-IDF + Logistic Regression
- Modern NLP: DistilBERT Transformer

The app compares both models using prediction output, confidence score, processing time, evaluation metrics, and batch-level analytics.

## Objective

The objective is to demonstrate a complete NLP analytics workflow:

1. Preprocess text
2. Train a traditional ML sentiment model
3. Run DistilBERT sentiment prediction
4. Compare both approaches
5. Analyze uploaded CSV files in batch
6. Generate report-ready metrics, exports, and charts

## Version 4 Features

- Batch CSV Analysis
- Analytics Dashboard
- DistilBERT Final Analytics Model
- CSV Export
- Sentiment Statistics
- Downloadable prediction results
- Model Agreement Analysis
- Interactive Plotly visualizations
- Report export support inside `report_assets/`

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

The traditional model uses TF-IDF vectorization and Logistic Regression. It is fast, lightweight, easy to explain, and useful for showing classic NLP feature engineering.

## Modern NLP Approach

The modern model uses DistilBERT through Hugging Face Transformers. DistilBERT understands context and word relationships better than frequency-based models, but it uses more computation and takes longer during inference.

## TF-IDF Explanation

TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numbers by measuring how important a word is in a sentence compared with the full dataset.

## DistilBERT Explanation

DistilBERT uses a transformer architecture. Instead of only counting words, it reads sentence context and learns relationships between words.

## System Architecture

```text
User Text / Uploaded CSV
   |
   |-- Preprocessing Visualization
   |
   |-- Traditional NLP
   |      |-- TF-IDF Vectorizer
   |      |-- Logistic Regression
   |
   |-- Modern NLP
   |      |-- DistilBERT Tokenizer
   |      |-- DistilBERT Transformer Model
   |
   |-- Analytics Dashboard
          |-- Sentiment Statistics
          |-- Agreement Analysis
          |-- Exportable Results
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

## Batch CSV Analysis

The Batch CSV Analysis page lets users upload a CSV file, automatically detect text columns, preview the dataset, and run both models on each row.

Exported prediction results include:

- Original Text
- Traditional Sentiment
- Traditional Confidence
- DistilBERT Sentiment
- DistilBERT Confidence

## Analytics Dashboard

The Analytics Dashboard displays:

- Total records analyzed
- Positive, negative, and neutral counts
- Average confidence
- Most common sentiment
- Traditional model accuracy
- DistilBERT accuracy
- Best performing model
- Average processing time
- Total predictions generated

Final dashboard analytics are generated using DistilBERT predictions. TF-IDF + Logistic Regression is retained as the baseline comparison model.

## Agreement Analysis

The dashboard compares whether both models predicted the same sentiment for each record.

It shows:

- Number of agreements
- Number of disagreements
- Agreement percentage
- Agreement pie chart
- Agreement bar chart

## How to Run Version 4

1. Place the Twitter sentiment dataset inside the `data/` folder.
2. Train the traditional model and generate model comparison assets:

```bash
python train_traditional.py
```

3. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

If `streamlit` is not recognized on Windows:

```bash
python -m streamlit run app.py
```

4. Use the sidebar to open:

- Single Text Analysis
- Batch CSV Analysis
- Analytics Dashboard
- Model Comparison
- Performance Metrics

Dataset and trained model files are not included in GitHub due to file size and reproducibility. Users should place the dataset inside `data/` and run `python train_traditional.py`.

## Visualization Features

Version 4 includes interactive Plotly charts:

- Sentiment Distribution Pie Chart
- Sentiment Distribution Bar Chart
- Confidence Distribution Histogram
- Traditional vs DistilBERT Agreement Chart
- Class Frequency Chart

## Report Assets

Training and batch analysis generate report assets inside `report_assets/`:

```text
traditional_metrics.txt
distilbert_metrics.txt
traditional_confusion_matrix.png
distilbert_confusion_matrix.png
model_comparison.csv
batch_analysis_results.csv
sentiment_statistics.csv
dashboard_summary.txt
agreement_analysis.csv
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

DistilBERT evaluation is also attempted from the same test split. By default it uses a practical local sample. You can increase this in `train_traditional.py`:

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
- Single Text Analysis page
- Batch CSV Analysis page
- Analytics Dashboard page
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
|-- .gitignore
|-- data/
|-- models/
|-- screenshots/
`-- report_assets/
```

## GitHub Preparation

The `.gitignore` file keeps GitHub clean by ignoring:

- Dataset CSV files
- Trained model files
- Generated reports
- Cache files
- Virtual environments
- Large screenshot images

Folder structure is preserved with `.gitkeep` files.
