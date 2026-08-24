"""
Medical Corrective RAG Assistant — Streamlit GUI
Integrates with the existing CorrectiveRAG pipeline (source.rag.CorrectiveRAG).
"""

import streamlit as st
from source.rag import CorrectiveRAG


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Corrective RAG Assistant",
    page_icon="🩺",
    layout="centered",
)


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        /* Clinical palette — teal/mist base, not the default SaaS blue */
        --ink: #0F2A2C;
        --muted: #5B7876;
        --surface: #FFFFFF;
        --mist: #EFF6F5;
        --line: #DCE6E4;
        --teal: #0E7C7B;
        --teal-deep: #095F5E;
        --teal-tint: #E4F3F1;
        --sage: #2F8F5B;
        --sage-tint: #E7F5EC;
        --amber: #B7660B;
        --amber-tint: #FCEFDC;
        --coral: #C23B3B;
        --coral-tint: #FBEAEA;

        /* legacy aliases kept so existing markup still resolves */
        --primary-blue: var(--teal);
        --light-blue: var(--teal-tint);
        --soft-gray: var(--mist);
        --border-gray: var(--line);
        --text-dark: var(--ink);
        --text-muted: var(--muted);
        --green: var(--sage);
        --orange: var(--amber);
        --red: var(--coral);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 50% -120px, #FBFEFD 0%, var(--mist) 62%);
    }

    .block-container {
        padding-top: 2.2rem;
        max-width: 780px;
    }

    /* ---------- Header ---------- */
    .app-header {
        text-align: center;
        padding: 6px 20px 2px 20px;
    }
    .app-header .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--teal);
        margin-bottom: 10px;
    }
    .app-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.15rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .app-header p {
        font-size: 1rem;
        color: var(--muted);
        margin: 0 auto;
        max-width: 520px;
    }

    /* Signature element: a pulse / vital-sign line beneath the header */
    .pulse-line {
        width: 100%;
        max-width: 460px;
        height: 34px;
        margin: 18px auto 22px auto;
        display: block;
    }
    .pulse-line path {
        fill: none;
        stroke: var(--teal);
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .pulse-line circle {
        fill: var(--teal);
    }

    /* ---------- Generic card ---------- */
    .card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 3px solid var(--teal);
        border-radius: 12px;
        padding: 22px 26px;
        box-shadow: 0 1px 2px rgba(15,42,44,0.04), 0 8px 20px rgba(15,42,44,0.03);
        margin-bottom: 18px;
    }
    .card h3 {
        margin-top: 0;
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
        font-weight: 600;
        font-size: 1.08rem;
    }

    .info-note {
        background: var(--teal-tint);
        border-left: 3px solid var(--teal);
        border-radius: 8px;
        padding: 10px 16px;
        color: var(--teal-deep);
        font-size: 0.9rem;
        margin-top: 10px;
    }

    .topic-pill {
        display: inline-block;
        background: var(--surface);
        color: var(--teal-deep);
        border-radius: 8px;
        padding: 6px 13px;
        margin: 4px 6px 4px 0;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid var(--line);
    }

    /* ---------- Pipeline ---------- */
    .pipeline-wrap {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin: 10px 0 14px 0;
    }
    .pipeline-step {
        background: var(--mist);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 9px 15px;
        text-align: center;
        min-width: 132px;
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--muted);
    }
    .pipeline-step.done {
        border-color: var(--teal);
        background: var(--teal-tint);
        color: var(--teal-deep);
    }
    .pipeline-step.skip {
        border-color: var(--line);
        background: var(--surface);
        color: #A9BAB8;
        text-decoration: line-through;
    }
    .pipeline-arrow {
        align-self: center;
        color: #A9BAB8;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* ---------- Badges (device-readout style) ---------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 15px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.02em;
    }
    .badge::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .badge-strong { background: var(--sage-tint); color: var(--sage); }
    .badge-strong::before { background: var(--sage); }
    .badge-moderate { background: var(--amber-tint); color: var(--amber); }
    .badge-moderate::before { background: var(--amber); }
    .badge-weak { background: var(--amber-tint); color: var(--amber); }
    .badge-weak::before { background: var(--amber); }
    .badge-poor { background: var(--coral-tint); color: var(--coral); }
    .badge-poor::before { background: var(--coral); }

    /* ---------- Source card ---------- */
    .source-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .source-card .doc-name {
        font-weight: 600;
        color: var(--ink);
        font-size: 0.94rem;
    }
    .source-card .doc-meta {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ---------- Answer box ---------- */
    .answer-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 3px solid var(--teal);
        border-radius: 12px;
        padding: 22px 26px;
        font-size: 1rem;
        line-height: 1.75;
        color: var(--ink);
        box-shadow: 0 1px 2px rgba(15,42,44,0.04), 0 8px 20px rgba(15,42,44,0.03);
    }

    /* ---------- Status card ---------- */
    .status-line {
        font-size: 0.92rem;
        color: var(--ink);
        padding: 4px 0;
    }

    /* ---------- Top status strip (replaces sidebar) ---------- */
    .status-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin: 0 0 22px 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 7px;
        padding: 5px 12px;
        font-size: 0.74rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        border: 1px solid var(--line);
        background: var(--surface);
        color: var(--muted);
    }
    .status-pill::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-pill.status-good { color: var(--teal-deep); }
    .status-pill.status-good::before { background: var(--sage); }
    .status-pill.status-bad { color: var(--coral); }
    .status-pill.status-bad::before { background: var(--coral); }
    .status-pill.status-pending { color: var(--amber); }
    .status-pill.status-pending::before { background: var(--amber); }

    /* ---------- Section title ---------- */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
        margin: 4px 0 10px 0;
    }

    /* ---------- Text input / textarea ---------- */
    .stTextArea textarea {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border: 1.5px solid var(--line) !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
        padding: 14px 16px !important;
    }
    .stTextArea textarea::placeholder {
        color: #93A8A6 !important;
        opacity: 1 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(14,124,123,0.14) !important;
    }

    .stButton>button {
        border-radius: 9px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: 1px solid var(--line);
        background: var(--surface);
        color: var(--ink);
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    .stButton>button:hover {
        border-color: var(--teal);
        color: var(--teal-deep);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--teal);
        color: #ffffff;
        border: none;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: var(--teal-deep);
        color: #ffffff;
    }

    footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.85rem;
    }

    hr {
        border-color: var(--line) !important;
    }

    /* ---------- Query Improvement ---------- */
    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    div[data-testid="stExpander"] summary {
        color: var(--ink) !important;
        font-weight: 600 !important;
    }
    .query-improve {
        background: var(--mist);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 20px;
    }
    .query-improve-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--teal);
        margin-bottom: 4px;
    }
    .query-improve-text {
        color: var(--ink) !important;
        font-size: 0.95rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .query-improve-final {
        border-color: var(--teal);
        font-weight: 500;
    }
    .query-improve-arrow {
        text-align: center;
        color: var(--teal);
        font-size: 1.2rem;
        margin: 8px 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "rag" not in st.session_state:
    st.session_state.rag = None

if "rag_error" not in st.session_state:
    st.session_state.rag_error = None


@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """Load the CorrectiveRAG pipeline once and cache it across reruns."""
    return CorrectiveRAG()


def get_rag():
    if st.session_state.rag is None and st.session_state.rag_error is None:
        try:
            st.session_state.rag = load_rag_pipeline()
        except Exception as exc:  # noqa: BLE001
            st.session_state.rag_error = str(exc)
    return st.session_state.rag


def clear_all():
    st.session_state.result = None
    st.session_state.question_input = ""


# ============================================================
# SYSTEM STATUS (inline, top-right strip — sidebar removed)
# ============================================================

if st.session_state.rag_error:
    status_html = '<span class="status-pill status-bad">Pipeline failed to load</span>'
elif st.session_state.rag is not None:
    status_html = (
        '<span class="status-pill status-good">Embedding Model</span>'
        '<span class="status-pill status-good">Vector Store</span>'
        '<span class="status-pill status-good">Groq Connected</span>'
        '<span class="status-pill status-good">Documents Loaded</span>'
    )
else:
    status_html = '<span class="status-pill status-pending">Pipeline not loaded yet</span>'

st.markdown(f'<div class="status-strip">{status_html}</div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="app-header">
    <div class="eyebrow">Corrective RAG · Clinical Q&A</div>
    <h1>🩺 Medical Corrective RAG Assistant</h1>
    <p>AI-powered medical question answering using Corrective Retrieval-Augmented Generation</p>
</div>
<svg class="pulse-line" viewBox="0 0 460 34" xmlns="http://www.w3.org/2000/svg">
    <path d="M0,17 L150,17 L168,17 L178,3 L190,31 L200,17 L215,17 L225,10 L235,17 L460,17" />
    <circle cx="235" cy="17" r="3.2" />
</svg>
    """,
    unsafe_allow_html=True,
)

topics = [
    "🦠 Tuberculosis",
    "🩸 Diabetes",
    "❤️ Hypertension",
    "🦟 Malaria",
    "🥗 Nutrition & Malnutrition",
    "🧠 Mental Health",
]

topic_pills = "".join(f'<span class="topic-pill">{t}</span>' for t in topics)

st.markdown(
    f"""
<div class="card">
    <h3>Supported Medical Topics</h3>
    {topic_pills}
    <div class="info-note">Ask questions only about the provided medical documents.</div>
</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown('<div class="section-title">💬 Ask a Question</div>', unsafe_allow_html=True)

question = st.text_area(
    label="Question",
    key="question_input",
    placeholder='Example:\n"What are the symptoms of tuberculosis?"',
    height=100,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑 Clear", use_container_width=True, on_click=clear_all)


# ============================================================
# PIPELINE RENDERING HELPER
# ============================================================

def render_pipeline(quality, query_rewritten):
    corrective_needed = quality in ("Weak", "Poor")

    steps = [
        ("Question", True),
        ("Query Rewriter", query_rewritten),
        ("Retriever (FAISS)", True),
        ("Context Evaluation", True),
        ("Corrective Retrieval", corrective_needed),
        ("LLM Answer", True),
    ]

    html_parts = ['<div class="pipeline-wrap">']

    for i, (label, done) in enumerate(steps):

        if i > 0:
            html_parts.append(
                '<div class="pipeline-arrow">➜</div>'
            )

        # Query Rewriter was skipped
        if label == "Query Rewriter" and not query_rewritten:
            html_parts.append(
                f'<div class="pipeline-step skip">{label}</div>'
            )

        # Corrective Retrieval was skipped
        elif label == "Corrective Retrieval" and not corrective_needed:
            html_parts.append(
                f'<div class="pipeline-step skip">{label}</div>'
            )

        # Step was executed
        else:
            css_class = "pipeline-step done" if done else "pipeline-step"
            check = " ✔" if done else ""

            html_parts.append(
                f'<div class="{css_class}">{label}{check}</div>'
            )

    html_parts.append("</div>")

    st.markdown(
        "".join(html_parts),
        unsafe_allow_html=True
    )

    # Status message below the pipeline
    if not query_rewritten and not corrective_needed:

        st.markdown(
            "✔ **Query Rewriter Skipped**  \n"
            "✔ **Corrective Retrieval Skipped**"
        )

    elif not query_rewritten and corrective_needed:

        st.markdown(
            "✔ **Query Rewriter Skipped**  \n"
            "✔ **Corrective Retrieval Triggered**"
        )

    elif query_rewritten and not corrective_needed:

        st.markdown(
            "✔ **Query Rewritten**  \n"
            "✔ **Corrective Retrieval Skipped**"
        )

    else:

        st.markdown(
            "✔ **Query Rewritten**  \n"
            "✔ **Corrective Retrieval Triggered**"
        )

# ============================================================
# ASK QUESTION LOGIC
# ============================================================

if ask_clicked:
    if not question or not question.strip():
        st.warning("Please enter a medical question first.")
    else:
        rag = get_rag()

        if rag is None:
            st.error(f"⚠️ Could not initialize the RAG pipeline: {st.session_state.rag_error}")
        else:
            with st.spinner("🩺 Analyzing your medical question..."):
                try:
                    result = rag.ask(question.strip())
                    st.session_state.result = result
                except Exception as exc:  # noqa: BLE001
                    st.session_state.result = None
                    st.error(f"⚠️ An error occurred while processing your question: {exc}")


# ============================================================
# RESULTS DISPLAY
# ============================================================

result = st.session_state.result

if result:

    st.markdown("---")

    # ---------- Pipeline visualization ----------
    st.markdown('<div class="section-title">⚙️ Pipeline Overview</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_pipeline(
    result["quality"],
    result["query_rewritten"]
)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Query rewriting ----------
    original_q = result["original_question"]
    rewritten_q = result["rewritten_question"]

    if original_q.strip().lower() != rewritten_q.strip().lower():
        with st.expander("🔄 Query Improvement", expanded=False):
            st.markdown(
                f"""
<div class="query-improve">
    <div class="query-improve-label">Original Question</div>
    <div class="query-improve-text">{original_q}</div>
    <div class="query-improve-arrow">↓</div>
    <div class="query-improve-label">Rewritten Question</div>
    <div class="query-improve-text query-improve-final">{rewritten_q}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    # ---------- Retrieval evaluation ----------
    quality = result["quality"]
    avg_score = result["average_score"]

    badge_map = {
        "Strong": ("badge-strong", "Strong", "The retrieved documents are highly relevant."),
        "Moderate": ("badge-moderate", "Moderate", "The retrieved documents are reasonably relevant."),
        "Weak": ("badge-weak", "Weak", "The system performed corrective retrieval."),
        "Poor": ("badge-poor", "Poor", "Relevant information could not be confidently retrieved."),
    }
    badge_class, badge_label, explanation = badge_map.get(
        quality, ("badge-moderate", quality, "")
    )

    avg_score_display = f"{avg_score:.4f}" if avg_score is not None else "N/A"

    st.markdown(
        f"""
<div class="card">
    <h3>📊 Retrieval Evaluation</h3>
    <span class="badge {badge_class}">{badge_label}</span>
    <p style="margin-top:14px; color:#334155;">
        <b>Context Quality:</b> {quality} &nbsp; | &nbsp; <b>Average Similarity Score:</b> {avg_score_display}
    </p>
    <div class="info-note">{explanation}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Answer ----------
    st.markdown('<div class="section-title">📝 Medical Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Sources ----------
    documents = result.get("documents", [])

    answer_text = (result.get("answer") or "").strip().lower()
    no_info_found = (
        "couldn't find enough information" in answer_text
        or "could not find enough information" in answer_text
        or "don't have enough information" in answer_text
        or "do not have enough information" in answer_text
    )

    if documents and not no_info_found:
        st.markdown('<div class="section-title">📚 Retrieved Sources</div>', unsafe_allow_html=True)

        cols = st.columns(2)

        for i, doc in enumerate(documents):
            source_name = doc.metadata.get("source", "Unknown Source")
            page_number = doc.metadata.get("page_number", "Unknown")

            with cols[i % 2]:
                st.markdown(
                    f"""
<div class="source-card">
    <div class="doc-name">📄 {source_name}</div>
    <div class="doc-meta">📑 Page {page_number}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

else:
    st.markdown(
        """
<div style="text-align:center; color:#94a3b8; padding: 30px 0;">
    Ask a medical question above to see the Corrective RAG pipeline in action.
</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding-bottom: 20px;">
    Developed using <b>Streamlit</b> · <b>FAISS</b> · <b>Sentence Transformers</b> · <b>Groq</b><br>
    Corrective Retrieval-Augmented Generation (Corrective RAG)
</div>
    """,
    unsafe_allow_html=True,
)