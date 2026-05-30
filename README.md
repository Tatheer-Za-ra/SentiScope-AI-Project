# SentiScope AI

## Project Title

**SentiScope AI: A Sentiment Analysis Platform Using Traditional NLP and Transformer-Based NLP**

## Project Objective

The objective of SentiScope AI is to build a complete BSCS Natural Language Processing project that analyzes sentiment from reviews, tweets, and comments. The system demonstrates preprocessing, traditional machine learning, transformer-based NLP, batch CSV analysis, explainable NLP, and analytics dashboards.

## Problem Statement

Large volumes of online comments, tweets, and reviews are difficult to analyze manually. SentiScope AI solves this by automatically classifying text into Positive, Negative, or Neutral sentiment and presenting the results in a clear dashboard suitable for reports, demos, and screenshots.

## Dataset Used

The project uses a Twitter sentiment dataset stored locally inside the `data/` folder.

Current expected dataset path:

```text
data/Twitter_Data.csv
```

Current expected columns:

```text
clean_text = tweet/comment text
category = sentiment label
```

Label mapping:

```text
-1 = Negative
 0 = Neutral
 1 = Positive
```

Dataset and trained model files are not pushed to GitHub due to size and reproducibility. Place the dataset in the `data/` folder and run `python train_traditional.py`.

## Tools and Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Hugging Face Transformers
- PyTorch
- DistilBERT
- Plotly
- Matplotlib

## System Features

- Dark Streamlit dashboard
- Single Text Analysis
- Preprocessing visualization
- TF-IDF + Logistic Regression prediction
- DistilBERT prediction
- Traditional vs modern NLP comparison
- Batch CSV Analysis
- Downloadable CSV exports
- Analytics Dashboard
- KPI cards
- Sentiment distribution charts
- Model agreement analysis
- Performance metrics
- Explainable NLP keyword highlighting
- Report-ready sections for screenshots

## Methodology

1. User enters text or uploads a CSV file.
2. Text is preprocessed for traditional NLP.
3. TF-IDF converts cleaned text into numerical features.
4. Logistic Regression predicts sentiment from TF-IDF features.
5. DistilBERT predicts sentiment using contextual transformer understanding.
6. Results are compared using confidence, time, and evaluation metrics.
7. Batch results are converted into dashboard analytics and export files.

## Preprocessing Steps

The preprocessing pipeline includes:

1. Original text display
2. Lowercasing
3. Punctuation removal
4. Tokenization
5. Stopword removal
6. Final processed text

These steps are shown visually in the Streamlit app.

## Traditional Model

### TF-IDF

TF-IDF stands for Term Frequency-Inverse Document Frequency. It converts text into numerical values based on word importance in a document and across the dataset.

### Logistic Regression

Logistic Regression is used as the traditional machine learning classifier. It is fast, lightweight, and useful as a clear baseline model for NLP classification.

## Modern Model

### DistilBERT

DistilBERT is a transformer-based NLP model. It understands context and word relationships better than simple frequency-based approaches. In SentiScope AI, DistilBERT is used as the final analytics model in the dashboard, while TF-IDF + Logistic Regression remains the baseline comparison model.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Average Confidence
- Average Prediction Time

Evaluation outputs are saved inside `report_assets/`.

## Batch CSV Analysis

The Batch CSV Analysis page allows users to:

- Upload a CSV file
- Automatically detect text columns
- Select the text column manually if needed
- Preview the uploaded dataset
- Run both TF-IDF and DistilBERT predictions
- Export prediction results as CSV

Exported columns include:

- Original Text
- Traditional Sentiment
- Traditional Confidence
- DistilBERT Sentiment
- DistilBERT Confidence

## Analytics Dashboard

The Analytics Dashboard displays batch-level insights using DistilBERT final predictions:

- Total records analyzed
- Positive predictions
- Negative predictions
- Neutral predictions
- Average DistilBERT confidence
- Most common sentiment
- Sentiment distribution pie chart
- Sentiment distribution bar chart
- Confidence distribution histogram
- Class frequency chart

## Explainable NLP

The Explainable NLP section highlights keyword-level sentiment clues:

- Positive words are highlighted in green
- Negative words are highlighted in red
- Neutral or unknown words are listed separately

This is a simple keyword-level explanation for learning and reporting. It is not full BERT explainability. The final prediction is generated using the ML/Transformer models.

## Model Agreement Analysis

The app compares whether both models predicted the same sentiment for each batch record.

It shows:

- Number of agreements
- Number of disagreements
- Agreement percentage
- Agreement pie chart
- Agreement bar chart

## How to Train

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the traditional model and generate evaluation report assets:

```bash
python train_traditional.py
```

This creates:

```text
models/traditional_model.pkl
models/tfidf_vectorizer.pkl
report_assets/model_comparison.csv
```

## How to Run

Run the app:

```bash
streamlit run app.py
```

If `streamlit` is not recognized on Windows:

```bash
python -m streamlit run app.py
```

Expected workflow:

```bash
pip install -r requirements.txt
python train_traditional.py
streamlit run app.py
```

## Demo Flow

Recommended demo order:

1. Open the Home page.
2. Go to Single Text Analysis.
3. Try Positive, Negative, Neutral, and Mixed Sentiment sample tweets.
4. Show Traditional vs DistilBERT results.
5. Show Explainable NLP keyword highlighting.
6. Show Preprocessing Pipeline.
7. Open Batch CSV Analysis.
8. Upload a CSV and run batch predictions.
9. Download prediction results.
10. Open Analytics Dashboard.
11. Show KPI cards, charts, and agreement analysis.
12. Open Model Comparison and Performance Metrics.

## Limitations

- DistilBERT can be slower on CPU.
- The explainability section is keyword-based, not deep transformer explainability.
- DistilBERT is not fine-tuned on the local Twitter dataset in this version.
- Model performance may vary depending on dataset quality and class balance.
- Large datasets may take longer during batch analysis.

## Future Work

- Fine-tune DistilBERT on the Twitter sentiment dataset.
- Add SHAP or LIME explanations.
- Add user authentication for saved reports.
- Add database storage for previous analyses.
- Add deployment with cached model loading.
- Add more robust multilingual sentiment support.

## Deployment Notes

The project is prepared for later Streamlit deployment, but full deployment is not included yet.

For future Streamlit Community Cloud deployment:

- Keep source code on GitHub.
- Do not push datasets or trained model files.
- Add the dataset through a supported storage method or use a small demo dataset.
- Ensure model files can be generated with `python train_traditional.py`.
- Be aware that DistilBERT downloads are large and may require extra setup.

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
|-- .streamlit/
|   `-- config.toml
|-- data/
|   `-- .gitkeep
|-- models/
|   `-- .gitkeep
|-- screenshots/
|   `-- .gitkeep
`-- report_assets/
    `-- .gitkeep
```

## GitHub Notes

The following are intentionally ignored:

- Dataset files
- Trained model files
- Generated report files
- Cache files
- Virtual environments
- Large screenshots

Folder structure is preserved using `.gitkeep` files.
