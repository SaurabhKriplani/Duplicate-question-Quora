import nltk
import streamlit as st
import helper
import pickle
import numpy as np

# ── NLTK downloads (silent after first run) ──────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Background & font ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Header ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ── Card ── */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem auto;
        max-width: 720px;
        backdrop-filter: blur(10px);
    }

    /* ── Text areas ── */
    textarea {
        background: rgba(255,255,255,0.08) !important;
        color: #e8e8f0 !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    textarea:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167,139,250,0.3) !important;
    }
    label {
        color: #c4b5fd !important;
        font-weight: 600 !important;
    }

    /* ── Button ── */
    div.stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.1s;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ── Result banners ── */
    .result-duplicate {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.25));
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-not-duplicate {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.25));
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-emoji { font-size: 3rem; display: block; margin-bottom: 0.5rem; }
    .result-title { font-size: 1.8rem; font-weight: 800; margin: 0; }
    .result-subtitle { color: #9ca3af; margin-top: 0.3rem; font-size: 0.95rem; }

    /* ── Feature chips ── */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
        justify-content: center;
    }
    .chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #c4b5fd;
    }

    /* ── Footer ── */
    footer {visibility: hidden;}
    .custom-footer {
        text-align: center;
        color: #4b5563;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.9);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 About")
    st.markdown(
        """
        This app detects whether two questions are **semantically duplicate**
        using a machine-learning model trained on the
        [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset.

        **Features used:**
        - Basic length & word-count features
        - Token overlap (stopwords / non-stopwords)
        - Fuzzy string similarity (FuzzyWuzzy)
        - Longest common substring ratio
        - Bag-of-Words (BoW) representation

        **Model:** Random Forest classifier
        """
    )
    st.markdown("---")
    st.markdown("### 💡 Example pairs")
    examples = [
        ("What is the best way to learn Python?", "How can I learn Python effectively?"),
        ("How do I make pizza?", "What is the capital of France?"),
        ("What is machine learning?", "Can you explain machine learning?"),
    ]
    for i, (e1, e2) in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state["q1"] = e1
            st.session_state["q2"] = e2

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🔍 Duplicate Question Detector</h1>
        <p>Enter two questions to find out if they ask the same thing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

q1 = st.text_area(
    "✏️ Question 1",
    value=st.session_state.get("q1", ""),
    placeholder="e.g. What is the best way to learn Python?",
    height=100,
    key="q1_input",
)
q2 = st.text_area(
    "✏️ Question 2",
    value=st.session_state.get("q2", ""),
    placeholder="e.g. How can I learn Python effectively?",
    height=100,
    key="q2_input",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🚀 Check for Duplicates")

st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter **both** questions before checking.", icon="⚠️")
    else:
        with st.spinner("Analysing questions…"):
            try:
                query = helper.query_point_creator(q1, q2)
                result = model.predict(query)[0]

                # Compute a confidence-like score (predict_proba if available)
                confidence_str = ""
                try:
                    proba = model.predict_proba(query)[0]
                    conf = max(proba) * 100
                    confidence_str = f"<p class='result-subtitle'>Model confidence: <b>{conf:.1f}%</b></p>"
                except AttributeError:
                    pass

                if result:
                    st.markdown(
                        f"""
                        <div class="result-duplicate">
                            <span class="result-emoji">🔁</span>
                            <p class="result-title" style="color:#f87171;">Duplicate Questions</p>
                            <p class="result-subtitle">These questions appear to be asking the same thing.</p>
                            {confidence_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-not-duplicate">
                            <span class="result-emoji">✅</span>
                            <p class="result-title" style="color:#34d399;">Not Duplicate</p>
                            <p class="result-subtitle">These questions are asking about different things.</p>
                            {confidence_str}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Feature breakdown ─────────────────────────────────────────
                with st.expander("📊 Feature breakdown", expanded=False):
                    import helper as _h
                    p1 = _h.preprocess(q1)
                    p2 = _h.preprocess(q2)
                    fuzzy = _h.test_fetch_fuzzy_features(p1, p2)
                    tokens = _h.test_fetch_token_features(p1, p2)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Q-Ratio", f"{fuzzy[0]:.0f}")
                        st.metric("Partial Ratio", f"{fuzzy[1]:.0f}")
                    with c2:
                        st.metric("Token Sort", f"{fuzzy[2]:.0f}")
                        st.metric("Token Set", f"{fuzzy[3]:.0f}")
                    with c3:
                        common = _h.test_common_words(p1, p2)
                        total = _h.test_total_words(p1, p2)
                        st.metric("Common Words", common)
                        st.metric("Total Words", total)

                    st.markdown(
                        f"""
                        <div class="chip-row">
                            <span class="chip">Q1 length: {len(p1)}</span>
                            <span class="chip">Q2 length: {len(p2)}</span>
                            <span class="chip">Same first word: {'✓' if tokens[7] else '✗'}</span>
                            <span class="chip">Same last word: {'✓' if tokens[6] else '✗'}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="custom-footer">Built with Streamlit · Quora Question Pairs dataset · Random Forest</div>',
    unsafe_allow_html=True,
)
