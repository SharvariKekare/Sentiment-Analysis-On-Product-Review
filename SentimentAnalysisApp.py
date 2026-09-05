import streamlit as st
import joblib

MODEL_PATH = "sentiment_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered",
)

# ---------------------------------------------------------
# Load model + vectorizer (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate 'sentiment_model.joblib' and 'tfidf_vectorizer.joblib'."
    )
    st.stop()

# ---------------------------------------------------------
# Sentiment -> emoji / color mapping
# ---------------------------------------------------------
SENTIMENT_STYLE = {
    "Positive": {"emoji": "😊", "color": "#28a745", "label": "Positive"},
    "Negative": {"emoji": "😞", "color": "#dc3545", "label": "Negative"},
    "Neutral":  {"emoji": "😐", "color": "#ffc107", "label": "Neutral"},
}

EXAMPLE_SENTENCES = [
    "Fantastic product with good build quality and smooth performance.",
    "The product arrived damaged and did not work properly.",
    "The product matches the description and is fairly average.",
    "Very happy with my purchase, I would buy it again!",
    "Very bad quality, definitely not worth the money.",
]

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🛍️ Product Review Sentiment Analyzer")
st.markdown(
    "Enter a product review below and this app will predict whether the "
    "sentiment is **Positive**, **Negative**, or **Neutral** using a "
    "**TF-IDF + LinearSVC** machine learning model."
)
st.divider()

# ---------------------------------------------------------
# Example sentences (buttons)
# ---------------------------------------------------------
st.subheader("💡 Try an example")
cols = st.columns(len(EXAMPLE_SENTENCES))
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

for i, example in enumerate(EXAMPLE_SENTENCES):
    if cols[i].button(f"Example {i + 1}", use_container_width=True):
        st.session_state.review_text = example

# ---------------------------------------------------------
# Text input area
# ---------------------------------------------------------
st.subheader("✍️ Enter your review")
review_text = st.text_area(
    "Review text",
    value=st.session_state.review_text,
    height=150,
    placeholder="Type or paste a product review here...",
    label_visibility="collapsed",
)

predict_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_clicked:
    if not review_text or not review_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Fast TF-IDF transform + prediction
        text_vector = vectorizer.transform([review_text])
        prediction = model.predict(text_vector)[0]

        style = SENTIMENT_STYLE.get(
            prediction, {"emoji": "🤔", "color": "#6c757d", "label": prediction}
        )

        st.divider()
        st.markdown(
            f"""
            <div style="
                background-color:{style['color']}22;
                border: 2px solid {style['color']};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <h2 style="color:{style['color']}; margin:0;">
                    {style['emoji']} {style['label']}
                </h2>
                <p style="margin-top:8px; font-size:16px;">
                    The predicted sentiment for your review is
                    <b>{style['label']}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📄 Review text analyzed"):
            st.write(review_text)

st.divider()
st.caption(
    "Model: LinearSVC | Features: TF-IDF | Built with Streamlit 🚀"
)