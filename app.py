import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bert_predict import MODEL_NAME as BERT_MODEL_NAME
from bert_predict import predict_distilbert_sentiment
from predict import MODEL_NAME as TRADITIONAL_MODEL_NAME
from predict import model_files_exist, predict_sentiment
from preprocessing import analyze_sentiment, preprocess_text


st.set_page_config(
    page_title="SentiScope AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


REPORT_ASSETS = Path("report_assets")
COMPARISON_CSV = REPORT_ASSETS / "model_comparison.csv"
TRADITIONAL_CM_IMAGE = REPORT_ASSETS / "traditional_confusion_matrix.png"
DISTILBERT_CM_IMAGE = REPORT_ASSETS / "distilbert_confusion_matrix.png"
BATCH_RESULTS_CSV = REPORT_ASSETS / "batch_analysis_results.csv"
SENTIMENT_STATS_CSV = REPORT_ASSETS / "sentiment_statistics.csv"
DASHBOARD_SUMMARY_TXT = REPORT_ASSETS / "dashboard_summary.txt"
AGREEMENT_ANALYSIS_CSV = REPORT_ASSETS / "agreement_analysis.csv"
MODEL_NAME_ALIASES = {
    TRADITIONAL_MODEL_NAME: [TRADITIONAL_MODEL_NAME],
    BERT_MODEL_NAME: [BERT_MODEL_NAME, "DistilBERT", "DistilBERT Transformer"],
}


def load_custom_css():
    """Keep the finalized Version 2 dark dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #070b14;
            --sidebar-bg: #0b1220;
            --panel: #111827;
            --panel-soft: #172033;
            --input-bg: #0f172a;
            --text: #f8fafc;
            --muted: #cbd5e1;
            --soft-muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.28);
            --primary: #38bdf8;
            --primary-strong: #0ea5e9;
            --green: #22c55e;
            --red: #f87171;
            --neutral: #60a5fa;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--app-bg) !important;
            color: var(--text) !important;
        }

        [data-testid="stHeader"] {
            background: var(--app-bg) !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        [data-testid="stToolbar"], [data-testid="stDecoration"] {
            background: transparent !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%) !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] .stRadio label {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radiogroup"] span {
            color: #f8fafc !important;
        }

        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: var(--text);
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.3rem;
            color: var(--text);
        }

        .subtitle {
            color: var(--muted);
            font-size: 1.12rem;
            margin-bottom: 1.5rem;
        }

        .section-heading {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 750;
            margin: 1.5rem 0 0.75rem;
        }

        .card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.15rem;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.35);
            min-height: 100%;
        }

        .feature-card {
            background: rgba(23, 32, 51, 0.98);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            min-height: 150px;
        }

        .positive-card {
            border-left: 7px solid var(--green);
            background: linear-gradient(180deg, rgba(20, 83, 45, 0.42), rgba(17, 24, 39, 0.96));
        }

        .negative-card {
            border-left: 7px solid var(--red);
            background: linear-gradient(180deg, rgba(127, 29, 29, 0.42), rgba(17, 24, 39, 0.96));
        }

        .neutral-card {
            border-left: 7px solid var(--neutral);
            background: linear-gradient(180deg, rgba(30, 58, 138, 0.42), rgba(17, 24, 39, 0.96));
        }

        .small-label {
            color: var(--soft-muted);
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .big-value {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .step-box {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
        }

        .flow {
            text-align: center;
            background: rgba(17, 24, 39, 0.96);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            font-weight: 750;
            color: #e0f2fe;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.42);
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.24), rgba(15, 23, 42, 0.92));
            color: #f8fafc !important;
            padding: 0.7rem 0.9rem;
            font-weight: 750;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border-color: var(--primary);
            background: rgba(14, 165, 233, 0.34);
            color: white !important;
            transform: translateY(-1px);
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.55);
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.30), rgba(15, 23, 42, 0.94));
            color: #ffffff !important;
            padding: 0.75rem 0.95rem;
            font-weight: 800;
        }

        div.stDownloadButton > button:hover {
            border-color: #7dd3fc;
            background: rgba(14, 165, 233, 0.40);
            color: #ffffff !important;
        }

        textarea, input, .stTextArea textarea {
            background: #0b1220 !important;
            color: #f8fafc !important;
            caret-color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.58) !important;
            border-radius: 14px !important;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.95), 0 0 0 1px rgba(56, 189, 248, 0.08) !important;
            line-height: 1.55 !important;
        }

        textarea::placeholder, input::placeholder {
            color: #b6c5d8 !important;
            opacity: 1 !important;
        }

        textarea:focus, input:focus, .stTextArea textarea:focus {
            border-color: #38bdf8 !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.20) !important;
        }

        textarea::selection, input::selection {
            background: rgba(56, 189, 248, 0.36) !important;
            color: #ffffff !important;
        }

        [data-testid="stTextArea"] label,
        [data-testid="stTextInput"] label {
            color: var(--muted) !important;
            font-weight: 650;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stFileUploader"] label {
            color: #e2e8f0 !important;
            font-weight: 700;
        }

        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {
            background: #0b1220 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(56, 189, 248, 0.48) !important;
            border-radius: 14px !important;
        }

        [data-baseweb="select"] span,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stUploadedFile"] * {
            color: #f8fafc !important;
        }

        [data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.96);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.26);
        }

        [data-testid="stMetric"] * {
            color: var(--text) !important;
        }

        [data-testid="stMetricLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 750 !important;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 850 !important;
        }

        [data-testid="stDataFrame"] {
            background: var(--panel) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] * {
            color: #f8fafc !important;
        }

        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="gridcell"] {
            color: #f8fafc !important;
            background-color: #111827 !important;
        }

        [role="tab"], [data-baseweb="tab"] {
            color: #e2e8f0 !important;
            font-weight: 750 !important;
        }

        [aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
            color: #7dd3fc !important;
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .stAlert * {
            color: #f8fafc !important;
        }

        code {
            color: #bae6fd !important;
            background: rgba(8, 47, 73, 0.65) !important;
            border-radius: 6px;
            padding: 0.1rem 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value):
    """Escape dynamic text before placing it inside custom HTML."""
    return html.escape(str(value))


def show_sidebar():
    """Display sidebar branding and navigation."""
    pages = [
        "Home",
        "Single Text Analysis",
        "Batch CSV Analysis",
        "Model Comparison",
        "Performance Metrics",
        "Analytics Dashboard",
        "How it Works",
        "About Project",
    ]

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Home"
    if "pending_page" in st.session_state:
        st.session_state["selected_page"] = st.session_state.pop("pending_page")
        st.session_state["selected_page_radio"] = st.session_state["selected_page"]
    if st.session_state["selected_page"] not in pages:
        st.session_state["selected_page"] = "Home"

    with st.sidebar:
        st.title("🧠 SentiScope AI")
        st.caption("Version 4 - Sentiment Analytics Platform")

        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state["selected_page"]),
            key="selected_page_radio",
        )
        st.session_state["selected_page"] = selected_page

        st.markdown("---")
        st.markdown(
            """
            **Project Description**

            SentiScope AI compares a traditional TF-IDF + Logistic Regression
            model with a modern DistilBERT transformer model.
            """
        )

        st.markdown("---")
        st.caption("Models: Traditional ML + DistilBERT")

    return selected_page


def feature_card(icon, title, text):
    """Render a dashboard-style feature card."""
    st.markdown(
        f"""
        <div class="feature-card">
            <h3>{icon} {title}</h3>
            <p style="color:#cbd5e1; margin-bottom:0;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_card(title, sentiment, confidence, processing_time):
    """Render a sentiment card with the existing visual style."""
    sentiment_class = {
        "Positive": "positive-card",
        "Negative": "negative-card",
        "Neutral": "neutral-card",
    }.get(sentiment, "neutral-card")

    sentiment_icon = {
        "Positive": "✅ Positive",
        "Negative": "⚠️ Negative",
        "Neutral": "ℹ️ Neutral",
    }.get(sentiment, "Unavailable")

    st.markdown(
        f"""
        <div class="card {sentiment_class}">
            <div class="small-label">{safe_text(title)}</div>
            <div class="big-value">{sentiment_icon}</div>
            <p style="color:#dbeafe; margin-top:0.6rem;">
                Confidence Score: <strong>{confidence:.2f}%</strong>
            </p>
            <p style="color:#cbd5e1; margin-bottom:0;">
                Processing Time: <strong>{processing_time:.4f} seconds</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_comparison_data():
    """Load saved model comparison metrics if they exist."""
    if not COMPARISON_CSV.exists():
        return None

    return pd.read_csv(COMPARISON_CSV)


def detect_text_columns(dataframe):
    """Find likely text columns in an uploaded CSV file."""
    common_names = ["text", "tweet", "content", "review", "comment", "clean_text", "message"]
    detected_columns = [
        column
        for column in dataframe.columns
        if str(column).strip().lower() in common_names
    ]

    if detected_columns:
        return detected_columns

    # Fallback: object/string columns are likely to contain text.
    return [
        column
        for column in dataframe.columns
        if dataframe[column].dtype == "object" or str(dataframe[column].dtype).startswith("string")
    ]


def detect_label_columns(dataframe):
    """Find possible sentiment/class columns for dataset summary display."""
    common_names = ["sentiment", "label", "class", "category", "target"]
    return [
        column
        for column in dataframe.columns
        if str(column).strip().lower() in common_names
    ]


def dataframe_to_csv_bytes(dataframe):
    """Convert a dataframe into downloadable CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def save_batch_report_assets(results_dataframe):
    """Save Version 4 batch analytics files into report_assets/."""
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    results_dataframe.to_csv(BATCH_RESULTS_CSV, index=False)

    stats_rows = []
    for model_column in ["Traditional Sentiment", "DistilBERT Sentiment"]:
        counts = results_dataframe[model_column].value_counts()
        for sentiment, count in counts.items():
            stats_rows.append(
                {
                    "Model": model_column.replace(" Sentiment", ""),
                    "Sentiment": sentiment,
                    "Count": int(count),
                }
            )

    pd.DataFrame(stats_rows).to_csv(SENTIMENT_STATS_CSV, index=False)

    agreement_dataframe = create_agreement_dataframe(results_dataframe)
    agreement_dataframe.to_csv(AGREEMENT_ANALYSIS_CSV, index=False)

    summary_text = build_dashboard_summary(results_dataframe)
    DASHBOARD_SUMMARY_TXT.write_text(summary_text, encoding="utf-8")


def create_agreement_dataframe(results_dataframe):
    """Create agreement/disagreement summary rows for both models."""
    agreement_series = results_dataframe["Traditional Sentiment"] == results_dataframe["DistilBERT Sentiment"]
    agree_count = int(agreement_series.sum())
    disagree_count = int((~agreement_series).sum())
    total_records = len(results_dataframe)
    agreement_percentage = round((agree_count / total_records) * 100, 2) if total_records else 0.0

    return pd.DataFrame(
        [
            {"Agreement Status": "Agree", "Count": agree_count, "Percentage": agreement_percentage},
            {"Agreement Status": "Disagree", "Count": disagree_count, "Percentage": round(100 - agreement_percentage, 2)},
        ]
    )


def build_dashboard_summary(results_dataframe):
    """Create a plain text summary for reports and presentations."""
    total_records = len(results_dataframe)
    sentiment_counts = results_dataframe["DistilBERT Sentiment"].value_counts()
    most_common = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"
    average_confidence = results_dataframe["DistilBERT Confidence"].mean()
    agreement_data = create_agreement_dataframe(results_dataframe)
    agreement_percentage = float(
        agreement_data.loc[agreement_data["Agreement Status"] == "Agree", "Percentage"].iloc[0]
    )

    return f"""SentiScope AI - Batch Analytics Summary
Final Analytics Model: DistilBERT Transformer
Total Records Analyzed: {total_records}
Positive Count: {int(sentiment_counts.get("Positive", 0))}
Negative Count: {int(sentiment_counts.get("Negative", 0))}
Neutral Count: {int(sentiment_counts.get("Neutral", 0))}
Average DistilBERT Confidence: {average_confidence:.2f}%
Most Common Sentiment: {most_common}
Model Agreement: {agreement_percentage:.2f}%
"""


def run_batch_predictions(dataframe, text_column):
    """Run both sentiment models for each row in the selected text column."""
    if not model_files_exist():
        raise FileNotFoundError("Traditional model not found. Please run python train_traditional.py first.")

    results = []
    total_rows = len(dataframe)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for index, row in dataframe.iterrows():
        original_text = str(row[text_column])
        progress_text.info(f"Analyzing record {len(results) + 1} of {total_rows}...")

        traditional_prediction = predict_sentiment(original_text)
        bert_prediction = predict_distilbert_sentiment(original_text)

        results.append(
            {
                "Original Text": original_text,
                "Traditional Sentiment": traditional_prediction["sentiment"],
                "Traditional Confidence": traditional_prediction["confidence"],
                "Traditional Processing Time": traditional_prediction["processing_time"],
                "DistilBERT Sentiment": bert_prediction["sentiment"],
                "DistilBERT Confidence": bert_prediction["confidence"],
                "DistilBERT Processing Time": bert_prediction["processing_time"],
                "Models Agree": traditional_prediction["sentiment"] == bert_prediction["sentiment"],
            }
        )

        progress_bar.progress((len(results)) / total_rows)

    progress_text.success("Batch analysis completed.")
    return pd.DataFrame(results)


def get_active_batch_results():
    """Load batch results from session state or saved report assets."""
    if "batch_results" in st.session_state:
        return st.session_state["batch_results"]

    if BATCH_RESULTS_CSV.exists():
        return pd.read_csv(BATCH_RESULTS_CSV)

    return None


def get_sentiment_counts(results_dataframe, model_column="Traditional Sentiment"):
    """Return counts for Positive, Negative, and Neutral labels."""
    counts = results_dataframe[model_column].value_counts()
    return {
        "Positive": int(counts.get("Positive", 0)),
        "Negative": int(counts.get("Negative", 0)),
        "Neutral": int(counts.get("Neutral", 0)),
    }


def style_plotly_chart(figure):
    """Apply readable chart styling for both dark and light Streamlit themes."""
    figure.update_layout(
        paper_bgcolor="#070b14",
        plot_bgcolor="#111827",
        font={"color": "#f8fafc", "size": 14},
        title={"font": {"color": "#f8fafc", "size": 18}},
        legend={"font": {"color": "#f8fafc"}},
        xaxis={
            "title_font": {"color": "#e2e8f0"},
            "tickfont": {"color": "#e2e8f0"},
            "gridcolor": "rgba(148, 163, 184, 0.22)",
        },
        yaxis={
            "title_font": {"color": "#e2e8f0"},
            "tickfont": {"color": "#e2e8f0"},
            "gridcolor": "rgba(148, 163, 184, 0.22)",
        },
    )
    return figure


def get_model_metric(comparison_data, model_name, metric_name):
    """Safely read one metric value from model_comparison.csv."""
    if comparison_data is None or metric_name not in comparison_data.columns:
        return None

    possible_names = MODEL_NAME_ALIASES.get(model_name, [model_name])
    row = comparison_data[comparison_data["Model"].isin(possible_names)]
    if row.empty or pd.isna(row[metric_name].iloc[0]):
        return None

    return float(row[metric_name].iloc[0])


def determine_best_model(comparison_data):
    """Select best model by F1 Score first, then Accuracy if F1 is unavailable."""
    if comparison_data is None or comparison_data.empty:
        return "N/A", "Model comparison metrics are not available."

    metric_name = "F1 Score" if comparison_data["F1 Score"].notna().any() else "Accuracy"
    valid_rows = comparison_data.dropna(subset=[metric_name])

    if valid_rows.empty:
        return "N/A", "F1 Score and Accuracy are not available."

    best_row = valid_rows.sort_values(metric_name, ascending=False).iloc[0]
    best_model = best_row["Model"]
    reason = f"{best_model} selected because it achieved the highest {metric_name}."
    return best_model, reason


def show_homepage():
    """Display the project landing dashboard."""
    st.markdown('<div class="main-title">SentiScope AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A Version 3 NLP platform comparing traditional ML with DistilBERT transformers.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Project Objective</div>
                <p style="font-size:1.05rem; color:#dbeafe; margin-top:0.65rem;">
                    SentiScope AI demonstrates how traditional NLP and modern
                    transformer-based NLP solve the same sentiment analysis task,
                    then compares their confidence, speed, and evaluation metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Version 3 Focus</div>
                <div class="big-value">NLP Model Comparison</div>
                <p style="color:#cbd5e1; margin-top:0.6rem;">
                    TF-IDF + Logistic Regression vs DistilBERT sentiment analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Core Features</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        feature_card("📝", "Text Analysis", "Run both models automatically on the same user text.")
    with col2:
        feature_card("🧹", "Preprocessing", "Keep Version 1 visual preprocessing steps.")
    with col3:
        feature_card("🤖", "Dual Models", "Compare TF-IDF + Logistic Regression with DistilBERT.")
    with col4:
        feature_card("📊", "Report Assets", "Generate metrics, confusion matrices, and comparison CSV.")


def set_sample_text(sample_text):
    """Store selected sample text in Streamlit session state."""
    st.session_state["input_text"] = sample_text


def show_sample_buttons():
    """Display quick sample inputs for testing the app."""
    st.markdown("**Try a sample input:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.button(
            "Positive Review",
            on_click=set_sample_text,
            args=("This product is amazing, helpful, and excellent. I really love it!",),
        )
    with col2:
        st.button(
            "Negative Review",
            on_click=set_sample_text,
            args=("The service was terrible, slow, and disappointing. I hate the experience.",),
        )
    with col3:
        st.button(
            "Neutral Review",
            on_click=set_sample_text,
            args=("The package arrived today and the item is on the table.",),
        )
    with col4:
        st.button(
            "Mixed Review",
            on_click=set_sample_text,
            args=("The design is beautiful and useful, but delivery was bad and very slow.",),
        )


def show_preprocessing_steps(steps):
    """Display preprocessing output in clear visual blocks."""
    st.markdown('<div class="section-heading">Preprocessing Steps</div>', unsafe_allow_html=True)

    labels = [
        ("Original Text", steps["original_text"]),
        ("Lowercased Text", steps["lowercased_text"]),
        ("Cleaned Text", steps["cleaned_text"]),
        ("Tokens", ", ".join(steps["raw_tokens"]) if steps["raw_tokens"] else "No tokens found"),
        (
            "Stopword Removal",
            ", ".join(steps["stopwords_removed"]) if steps["stopwords_removed"] else "No tokens left",
        ),
        ("Final Processed Text", steps["final_text"]),
    ]

    for label, value in labels:
        display_value = safe_text(value) if value else "N/A"
        st.markdown(
            f"""
            <div class="step-box">
                <div class="small-label">{label}</div>
                <div style="color:#f8fafc; margin-top:0.35rem;">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def rule_based_explanation(tokens):
    """Keep the V1 explanation as a supporting comparison."""
    result = analyze_sentiment(tokens)
    positive_words = safe_text(", ".join(result["positive_matches"]) or "None")
    negative_words = safe_text(", ".join(result["negative_matches"]) or "None")

    with st.expander("Optional V1 Rule-Based Explanation"):
        st.markdown(
            f"""
            <div class="card">
                <div class="small-label">Simple Word Matching Result</div>
                <p style="color:#dbeafe;"><strong>Rule-Based Sentiment:</strong> {safe_text(result["sentiment"])}</p>
                <p style="color:#bbf7d0;"><strong>Positive words:</strong> {positive_words}</p>
                <p style="color:#fecaca;"><strong>Negative words:</strong> {negative_words}</p>
                <p style="color:#cbd5e1; margin-bottom:0;">
                    This section is kept from Version 1 for explanation only.
                    Version 3 final comparison uses the trained traditional model
                    and DistilBERT.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_confidence_summary(traditional_prediction, bert_prediction):
    """Show which model is more confident for the current user text."""
    st.markdown('<div class="section-heading">Which model appears more confident?</div>', unsafe_allow_html=True)

    chart_data = pd.DataFrame(
        {
            "Model": [TRADITIONAL_MODEL_NAME, BERT_MODEL_NAME],
            "Confidence": [
                traditional_prediction["confidence"],
                bert_prediction["confidence"],
            ],
        }
    )
    figure = px.bar(
        chart_data,
        x="Model",
        y="Confidence",
        color="Model",
        text="Confidence",
        range_y=[0, 100],
        template="plotly_dark",
    )
    figure.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    figure.update_layout(showlegend=False, paper_bgcolor="#070b14", plot_bgcolor="#111827")
    st.plotly_chart(figure, use_container_width=True)

    if traditional_prediction["confidence"] > bert_prediction["confidence"]:
        st.info("TF-IDF + Logistic Regression appears more confident for this input.")
    elif bert_prediction["confidence"] > traditional_prediction["confidence"]:
        st.info("DistilBERT appears more confident for this input.")
    else:
        st.info("Both models show the same confidence for this input.")


def show_analyze_page():
    """Run both sentiment models on a single user input."""
    st.markdown('<div class="main-title">Analyze Sentiment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Enter text once and compare traditional NLP with DistilBERT automatically.</div>',
        unsafe_allow_html=True,
    )

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    show_sample_buttons()

    user_text = st.text_area(
        "Enter a review, tweet, or comment",
        key="input_text",
        height=160,
        placeholder="Example: The app is very useful and easy to use.",
    )

    analyze_clicked = st.button("Analyze Sentiment")

    if analyze_clicked:
        if not user_text.strip():
            st.error("Please enter some text before running sentiment analysis.")
            return

        with st.status("Analyzing text using TF-IDF and DistilBERT...", expanded=True) as status:
            st.write("Preparing preprocessing visualization...")
            preprocessing_steps = preprocess_text(user_text)

            if not model_files_exist():
                status.update(label="Traditional model files are missing.", state="error", expanded=True)
                st.warning("Traditional model not found. Please run python train_traditional.py first.")
                show_preprocessing_steps(preprocessing_steps)
                rule_based_explanation(preprocessing_steps["tokens"])
                return

            st.write("Running TF-IDF + Logistic Regression...")
            traditional_prediction = predict_sentiment(user_text)

            st.write("Transformer model is processing. Please wait...")
            bert_prediction = predict_distilbert_sentiment(user_text)

            status.update(label="Analysis complete.", state="complete", expanded=False)

        st.success("Analysis completed successfully.")

        left_col, right_col = st.columns([1, 1])
        with left_col:
            sentiment_card(
                TRADITIONAL_MODEL_NAME,
                traditional_prediction["sentiment"],
                traditional_prediction["confidence"],
                traditional_prediction["processing_time"],
            )
        with right_col:
            if bert_prediction["error"]:
                st.warning(f"DistilBERT prediction unavailable: {bert_prediction['error']}")
            sentiment_card(
                BERT_MODEL_NAME,
                bert_prediction["sentiment"],
                bert_prediction["confidence"],
                bert_prediction["processing_time"],
            )

        st.markdown(
            f"""
            <div class="card" style="margin-top:1rem;">
                <div class="small-label">Model Input Details</div>
                <p style="color:#dbeafe;"><strong>Traditional model input:</strong> <code>{safe_text(traditional_prediction["processed_text"]) or "N/A"}</code></p>
                <p style="color:#cbd5e1; margin-bottom:0;">
                    DistilBERT uses the original sentence because transformer
                    tokenizers are designed to preserve context and word order.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_confidence_summary(traditional_prediction, bert_prediction)
        show_preprocessing_steps(preprocessing_steps)
        rule_based_explanation(preprocessing_steps["tokens"])


def show_model_comparison_page():
    """Display table and charts comparing both models."""
    st.markdown('<div class="main-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Compare traditional NLP and DistilBERT using saved evaluation metrics.</div>',
        unsafe_allow_html=True,
    )

    comparison_data = load_comparison_data()
    if comparison_data is None:
        st.warning("Comparison results not found. Run python train_traditional.py to generate report_assets/model_comparison.csv.")
        return

    display_data = comparison_data.set_index("Model").T.reset_index()
    display_data = display_data.rename(columns={"index": "Metric"})
    st.dataframe(display_data, use_container_width=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "Average Prediction Time"]
    for metric in metric_names:
        st.markdown(f'<div class="section-heading">{metric} Comparison</div>', unsafe_allow_html=True)
        figure = px.bar(
            comparison_data,
            x="Model",
            y=metric,
            color="Model",
            text=metric,
            template="plotly_dark",
        )
        figure.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        figure.update_layout(showlegend=False, paper_bgcolor="#070b14", plot_bgcolor="#111827")
        st.plotly_chart(figure, use_container_width=True)


def show_batch_csv_analysis_page():
    """Upload a CSV, run both models, preview results, and export outputs."""
    st.markdown('<div class="main-title">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload a CSV file and generate sentiment predictions from both models.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin batch sentiment analysis.")
        return

    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Invalid CSV file. Please upload a readable CSV. Details: {error}")
        return

    if dataframe.empty:
        st.error("The uploaded CSV file is empty.")
        return

    text_columns = detect_text_columns(dataframe)
    if not text_columns:
        st.error("No text column found. Please upload a CSV with a text, tweet, review, comment, content, or string column.")
        return

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Text Column Selection</div>
            <p style="color:#dbeafe; margin-bottom:0;">
                Select the column that contains the review, tweet, or comment text to analyze.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    common_text_names = {"text", "tweet", "review", "comment", "content", "clean_text", "message"}
    detected_named_columns = [
        column for column in text_columns if str(column).strip().lower() in common_text_names
    ]
    if detected_named_columns:
        st.success(f"Detected text column: {detected_named_columns[0]}")
    else:
        st.info("Please choose the column that contains text for sentiment analysis.")

    selected_text_column = st.selectbox(
        "Select text column for sentiment analysis",
        text_columns,
        index=0,
        help="Choose the column containing reviews, tweets, comments, or other text values.",
    )

    st.markdown('<div class="section-heading">Selected Text Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        dataframe[[selected_text_column]].dropna().head(5),
        use_container_width=True,
    )

    st.markdown('<div class="section-heading">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(dataframe.head(10), use_container_width=True)

    missing_values = int(dataframe[selected_text_column].isna().sum())
    label_columns = detect_label_columns(dataframe)
    class_count = dataframe[label_columns[0]].dropna().nunique() if label_columns else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Records", len(dataframe))
    col2.metric("Missing Text Values", missing_values)
    col3.metric("Number of Classes", class_count if class_count is not None else "N/A")

    clean_dataframe = dataframe.dropna(subset=[selected_text_column]).copy()
    if clean_dataframe.empty:
        st.error("No usable text rows found after removing missing values.")
        return

    st.warning("Batch DistilBERT analysis can take time on CPU. Keep the browser open while it runs.")

    if st.button("Run Batch Analysis"):
        try:
            with st.status("Analyzing CSV using TF-IDF and DistilBERT...", expanded=True) as status:
                st.write("Preparing uploaded records...")
                results_dataframe = run_batch_predictions(clean_dataframe, selected_text_column)
                st.write("Saving report assets...")
                save_batch_report_assets(results_dataframe)
                st.session_state["batch_results"] = results_dataframe
                status.update(label="Batch analysis complete.", state="complete", expanded=False)
            st.success("Batch analysis completed and report assets were saved.")
        except Exception as error:
            st.error(f"Batch analysis failed: {error}")
            return

    results_dataframe = get_active_batch_results()
    if results_dataframe is None:
        return

    st.markdown('<div class="section-heading">Prediction Results</div>', unsafe_allow_html=True)
    st.dataframe(results_dataframe.head(50), use_container_width=True)

    st.download_button(
        "⬇ Download Results CSV",
        data=dataframe_to_csv_bytes(results_dataframe),
        file_name="sentiscope_batch_predictions.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="card" style="margin-top:1rem;">
            <div class="small-label">Next Step</div>
            <p style="color:#dbeafe; margin-bottom:0;">
                Open the analytics dashboard to view KPI cards, sentiment charts,
                model agreement analysis, and export-ready insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📊 View Analytics Dashboard"):
        st.session_state["pending_page"] = "Analytics Dashboard"
        st.rerun()


def show_project_statistics_panel(results_dataframe):
    """Display model and project statistics cards."""
    comparison_data = load_comparison_data()
    traditional_accuracy = get_model_metric(comparison_data, TRADITIONAL_MODEL_NAME, "Accuracy")
    traditional_f1 = get_model_metric(comparison_data, TRADITIONAL_MODEL_NAME, "F1 Score")
    distilbert_accuracy = get_model_metric(comparison_data, BERT_MODEL_NAME, "Accuracy")
    distilbert_f1 = get_model_metric(comparison_data, BERT_MODEL_NAME, "F1 Score")
    average_traditional_time = results_dataframe["Traditional Processing Time"].mean()
    average_distilbert_time = results_dataframe["DistilBERT Processing Time"].mean()
    best_model, best_reason = determine_best_model(comparison_data)

    st.markdown('<div class="section-heading">Project Statistics Panel</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Traditional Accuracy", f"{traditional_accuracy:.4f}" if traditional_accuracy is not None else "N/A")
    col2.metric("Traditional F1 Score", f"{traditional_f1:.4f}" if traditional_f1 is not None else "N/A")
    col3.metric("Traditional Avg Time", f"{average_traditional_time:.4f}s")

    col4, col5, col6 = st.columns(3)
    col4.metric("DistilBERT Accuracy", f"{distilbert_accuracy:.4f}" if distilbert_accuracy is not None else "N/A")
    col5.metric("DistilBERT F1 Score", f"{distilbert_f1:.4f}" if distilbert_f1 is not None else "N/A")
    col6.metric("DistilBERT Avg Time", f"{average_distilbert_time:.4f}s")

    st.markdown(
        f"""
        <div class="card" style="margin-top:1rem;">
            <div class="small-label">Best Performing Model</div>
            <div class="big-value">{safe_text(best_model)}</div>
            <p style="color:#dbeafe; margin-bottom:0;">{safe_text(best_reason)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_analytics_dashboard_page():
    """Display analytics, KPIs, agreement analysis, and export-ready insights."""
    st.markdown('<div class="main-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Explore batch sentiment statistics, model agreement, and export-ready insights.</div>',
        unsafe_allow_html=True,
    )

    results_dataframe = get_active_batch_results()
    if results_dataframe is None:
        st.warning("No batch analysis results found yet. Please run Batch CSV Analysis first, then return to this dashboard.")
        return

    st.markdown(
        """
        <div class="card">
            <div class="small-label">Final Analytics Model</div>
            <div class="big-value">DistilBERT Transformer</div>
            <p style="color:#dbeafe; margin-top:0.65rem;">
                DistilBERT is used as the final analytics engine because it
                provides higher contextual understanding and better sentiment
                classification performance. TF-IDF + Logistic Regression is
                retained as a baseline comparison model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sentiment_counts = get_sentiment_counts(results_dataframe, "DistilBERT Sentiment")
    total_records = len(results_dataframe)
    average_confidence = results_dataframe["DistilBERT Confidence"].mean()
    most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get) if total_records else "N/A"

    st.markdown('<div class="section-heading">KPI Cards</div>', unsafe_allow_html=True)
    st.caption("Generated using DistilBERT Final Predictions")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Records Analyzed", total_records)
    col2.metric("Positive Predictions", sentiment_counts["Positive"])
    col3.metric("Negative Predictions", sentiment_counts["Negative"])
    col4.metric("Neutral Predictions", sentiment_counts["Neutral"])
    col5.metric("Avg DistilBERT Confidence", f"{average_confidence:.2f}%")
    col6.metric("Most Common", most_common_sentiment)

    show_project_statistics_panel(results_dataframe)

    sentiment_distribution = pd.DataFrame(
        {"Sentiment": list(sentiment_counts.keys()), "Count": list(sentiment_counts.values())}
    )

    st.markdown('<div class="section-heading">Sentiment Distribution (DistilBERT Final Predictions)</div>', unsafe_allow_html=True)
    st.info("These charts are generated using the final predictions produced by DistilBERT.")
    left_col, right_col = st.columns(2)
    with left_col:
        pie_chart = px.pie(
            sentiment_distribution,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution Pie Chart",
            template="plotly_dark",
        )
        pie_chart.update_traces(textinfo="label+percent+value", textfont={"color": "#f8fafc", "size": 14})
        pie_chart = style_plotly_chart(pie_chart)
        st.plotly_chart(pie_chart, use_container_width=True)
    with right_col:
        bar_chart = px.bar(
            sentiment_distribution,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title="Sentiment Distribution Bar Chart",
            text="Count",
            template="plotly_dark",
        )
        bar_chart.update_traces(textposition="outside", textfont={"color": "#f8fafc", "size": 14})
        bar_chart = style_plotly_chart(bar_chart)
        st.plotly_chart(bar_chart, use_container_width=True)

    confidence_values = results_dataframe[
        ["Traditional Confidence", "DistilBERT Confidence"]
    ].melt(var_name="Model", value_name="Confidence")
    confidence_values["Model"] = confidence_values["Model"].str.replace(" Confidence", "", regex=False)
    histogram = px.histogram(
        confidence_values,
        x="Confidence",
        color="Model",
        barmode="group",
        opacity=0.95,
        nbins=20,
        title="Confidence Distribution Histogram: Traditional vs DistilBERT",
        color_discrete_map={
            "Traditional": "#60a5fa",
            "DistilBERT": "#f97316",
        },
        template="plotly_dark",
    )
    histogram = style_plotly_chart(histogram)
    st.plotly_chart(histogram, use_container_width=True)

    agreement_dataframe = create_agreement_dataframe(results_dataframe)
    st.markdown('<div class="section-heading">Model Agreement Analysis</div>', unsafe_allow_html=True)
    agree_count = int(agreement_dataframe.loc[agreement_dataframe["Agreement Status"] == "Agree", "Count"].iloc[0])
    disagree_count = int(agreement_dataframe.loc[agreement_dataframe["Agreement Status"] == "Disagree", "Count"].iloc[0])
    agreement_percentage = float(
        agreement_dataframe.loc[agreement_dataframe["Agreement Status"] == "Agree", "Percentage"].iloc[0]
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Both Models Agree", agree_count)
    col2.metric("Both Models Disagree", disagree_count)
    col3.metric("Agreement Percentage", f"{agreement_percentage:.2f}%")

    left_col, right_col = st.columns(2)
    with left_col:
        agreement_pie = px.pie(
            agreement_dataframe,
            names="Agreement Status",
            values="Count",
            title="Agreement Pie Chart",
            template="plotly_dark",
        )
        agreement_pie.update_traces(textinfo="label+percent+value", textfont={"color": "#f8fafc", "size": 14})
        agreement_pie = style_plotly_chart(agreement_pie)
        st.plotly_chart(agreement_pie, use_container_width=True)
    with right_col:
        agreement_bar = px.bar(
            agreement_dataframe,
            x="Agreement Status",
            y="Count",
            color="Agreement Status",
            title="Agreement Bar Chart",
            text="Count",
            template="plotly_dark",
        )
        agreement_bar.update_traces(textposition="outside", textfont={"color": "#f8fafc", "size": 14})
        agreement_bar = style_plotly_chart(agreement_bar)
        st.plotly_chart(agreement_bar, use_container_width=True)

    model_sentiment_data = results_dataframe.melt(
        value_vars=["Traditional Sentiment", "DistilBERT Sentiment"],
        var_name="Model",
        value_name="Sentiment",
    )
    class_frequency = px.histogram(
        model_sentiment_data,
        x="Sentiment",
        color="Model",
        barmode="group",
        title="Class Frequency Chart",
        template="plotly_dark",
    )
    class_frequency = style_plotly_chart(class_frequency)
    st.plotly_chart(class_frequency, use_container_width=True)

    st.markdown('<div class="section-heading">How Statistics Are Evaluated</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p style="color:#dbeafe;"><strong>Traditional Model Metrics:</strong>
            Generated from TF-IDF + Logistic Regression evaluation.</p>
            <p style="color:#dbeafe;"><strong>DistilBERT Metrics:</strong>
            Generated from DistilBERT evaluation.</p>
            <p style="color:#dbeafe;"><strong>Best Performing Model:</strong>
            Automatically selected using F1 Score comparison. If F1 Score is not
            available, Accuracy is used instead.</p>
            <p style="color:#dbeafe; margin-bottom:0;"><strong>Final Dashboard Analytics:</strong>
            Generated using DistilBERT predictions from the latest batch analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇ Download Results CSV",
        data=dataframe_to_csv_bytes(results_dataframe),
        file_name="sentiscope_batch_predictions.csv",
        mime="text/csv",
    )


def show_performance_metrics_page():
    """Show metric cards, confusion matrices, and comparison charts."""
    st.markdown('<div class="main-title">Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Evaluation outputs for project report, presentation, and screenshots.</div>',
        unsafe_allow_html=True,
    )

    comparison_data = load_comparison_data()
    if comparison_data is None:
        st.warning("Performance metrics not found. Run python train_traditional.py first.")
        return

    for _, row in comparison_data.iterrows():
        st.markdown(f'<div class="section-heading">{safe_text(row["Model"])}</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{row['Accuracy']:.4f}" if pd.notna(row["Accuracy"]) else "N/A")
        col2.metric("Precision", f"{row['Precision']:.4f}" if pd.notna(row["Precision"]) else "N/A")
        col3.metric("Recall", f"{row['Recall']:.4f}" if pd.notna(row["Recall"]) else "N/A")
        col4.metric("F1 Score", f"{row['F1 Score']:.4f}" if pd.notna(row["F1 Score"]) else "N/A")

    st.markdown('<div class="section-heading">Confusion Matrices</div>', unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Traditional Model**")
        if TRADITIONAL_CM_IMAGE.exists():
            st.image(str(TRADITIONAL_CM_IMAGE), use_container_width=True)
        else:
            st.info("Traditional confusion matrix image not found.")
    with right_col:
        st.markdown("**DistilBERT**")
        if DISTILBERT_CM_IMAGE.exists():
            st.image(str(DISTILBERT_CM_IMAGE), use_container_width=True)
        else:
            st.info("DistilBERT confusion matrix image not found.")

    st.markdown('<div class="section-heading">Metric Comparison</div>', unsafe_allow_html=True)
    melted_data = comparison_data.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )
    figure = px.bar(
        melted_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        template="plotly_dark",
    )
    figure.update_layout(paper_bgcolor="#070b14", plot_bgcolor="#111827")
    st.plotly_chart(figure, use_container_width=True)


def show_how_it_works():
    """Explain traditional NLP and DistilBERT in simple language."""
    st.markdown('<div class="main-title">How it Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A simple explanation of traditional NLP vs transformer-based NLP.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="flow">1. Input Text</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="flow">2. Preprocessing</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="flow">3. Two NLP Models</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="flow">4. Comparison Output</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">NLP Explanation</div>', unsafe_allow_html=True)
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Traditional NLP</div>
                <p style="color:#dbeafe;">
                    The traditional model uses TF-IDF features. TF-IDF converts
                    words into numerical values based on how important they are
                    in a sentence and across the dataset.
                </p>
                <p style="color:#cbd5e1;">
                    It relies heavily on word frequency, is fast during inference,
                    and has lower computational cost.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">DistilBERT</div>
                <p style="color:#dbeafe;">
                    DistilBERT is a transformer model. It reads text with context,
                    learns relationships between words, and understands meaning
                    better than simple frequency-based methods.
                </p>
                <p style="color:#cbd5e1;">
                    It usually has stronger language understanding, but needs
                    more memory and processing time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_about():
    """Display basic project information."""
    st.markdown('<div class="main-title">About SentiScope AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p style="color:#dbeafe;">
                SentiScope AI is a BSCS Natural Language Processing semester
                project. Version 3 turns the project into a comparison platform
                for traditional machine learning and modern transformer-based NLP.
            </p>
            <p style="color:#cbd5e1;">
                Version 1 preprocessing visualization and Version 2 traditional
                ML prediction remain available, with DistilBERT added as the
                modern NLP model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Run the Streamlit application."""
    load_custom_css()
    selected_page = show_sidebar()

    if selected_page == "Home":
        show_homepage()
    elif selected_page == "Single Text Analysis":
        show_analyze_page()
    elif selected_page == "Batch CSV Analysis":
        show_batch_csv_analysis_page()
    elif selected_page == "Model Comparison":
        show_model_comparison_page()
    elif selected_page == "Performance Metrics":
        show_performance_metrics_page()
    elif selected_page == "Analytics Dashboard":
        show_analytics_dashboard_page()
    elif selected_page == "How it Works":
        show_how_it_works()
    else:
        show_about()


if __name__ == "__main__":
    main()
