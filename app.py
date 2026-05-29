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

        [data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.96);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem;
        }

        [data-testid="stMetric"] * {
            color: var(--text) !important;
        }

        [data-testid="stDataFrame"] {
            background: var(--panel) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.96) !important;
            color: var(--text) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
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
    with st.sidebar:
        st.title("🧠 SentiScope AI")
        st.caption("Version 3 - Traditional NLP vs Modern NLP")

        selected_page = st.radio(
            "Navigation",
            [
                "Home",
                "Analyze",
                "Model Comparison",
                "Performance Metrics",
                "How it Works",
                "About",
            ],
            index=0,
        )

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
    elif selected_page == "Analyze":
        show_analyze_page()
    elif selected_page == "Model Comparison":
        show_model_comparison_page()
    elif selected_page == "Performance Metrics":
        show_performance_metrics_page()
    elif selected_page == "How it Works":
        show_how_it_works()
    else:
        show_about()


if __name__ == "__main__":
    main()
