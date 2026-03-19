# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Amazon Review Intelligence", page_icon="🛍️")
st.title("🛍️ Customer Review Intelligence System")
st.caption("Powered by fine-tuned DistilBERT + Zero-shot Classification")

# load models 
@st.cache_resource
def load_models():
    # Pipeline 1:  fine-tuned actual Hugging Face model paths
    sentiment_pipe = pipeline(
        "text-classification",
        model="YOUR_HF_USERNAME/YOUR_FINETUNED_MODEL"
    )
    # Pipeline 2: Zero-shot
    zeroshot_pipe = pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-small"
    )
    return sentiment_pipe, zeroshot_pipe

sentiment_pipe, zeroshot_pipe = load_models()

CANDIDATE_LABELS = [
    "fragrance issue", "packaging issue", "skin irritation",
    "texture problem", "effectiveness issue",
    "delivery or damaged item", "value for money"
]

# LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
# PRIORITY_MAP = {"Negative": "🔴 High", "Neutral": "🟡 Medium", "Positive": "🟢 Low"}
LABEL_MAP     = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
PRIORITY_MAP  = {"Negative": "🔴 High Priority", "Positive": "🟢 Low Priority"}
# ── 主界面 ───────────────────────────────────────────────────────
review_input = st.text_area("📝 Paste a customer review here:", height=150)

if st.button("Analyze Review") and review_input.strip():
    with st.spinner("Analyzing..."):

        # Pipeline 1
        s_result  = sentiment_pipe(review_input[:512])[0]
        sentiment = LABEL_MAP.get(s_result["label"], s_result["label"])
        priority  = PRIORITY_MAP[sentiment]

        col1, col2 = st.columns(2)
        col1.metric("Sentiment", sentiment)
        col2.metric("Service Priority", priority)

        # Pipeline 2
        if sentiment in ["Negative", "Neutral"]:
            st.subheader("🔍 Detected Issue")
            z_result  = zeroshot_pipe(review_input[:512], CANDIDATE_LABELS)
            top_issue = z_result["labels"][0]
            top_score = z_result["scores"][0]
            st.success(f"**{top_issue}** (confidence: {top_score:.1%})")

            # show all issue scores in an expander
            with st.expander("All issue scores"):
                for label, score in zip(z_result["labels"], z_result["scores"]):
                    st.write(f"{label}: {score:.1%}")
        else:
            st.info("✅ Positive review — no issue detected.")
