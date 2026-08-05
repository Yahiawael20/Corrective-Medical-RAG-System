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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --primary-blue: #2563eb;
        --light-blue: #eff6ff;
        --soft-gray: #f8fafc;
        --border-gray: #e2e8f0;
        --text-dark: #1e293b;
        --text-muted: #64748b;
        --green: #16a34a;
        --orange: #ea580c;
        --red: #dc2626;
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background-color: var(--soft-gray);
    }

    /* ---------- Header ---------- */
    .app-header {
        text-align: center;
        padding: 28px 20px 8px 20px;
    }
    .app-header h1 {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--text-dark);
        margin-bottom: 4px;
    }
    .app-header p {
        font-size: 1.05rem;
        color: var(--text-muted);
        margin-top: 0;
    }

    /* ---------- Generic card ---------- */
    .card {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }
    .card h3 {
        margin-top: 0;
        color: var(--text-dark);
        font-weight: 700;
    }

    .info-note {
        background: var(--light-blue);
        border-left: 4px solid var(--primary-blue);
        border-radius: 10px;
        padding: 10px 16px;
        color: var(--text-dark);
        font-size: 0.92rem;
        margin-top: 10px;
    }

    .topic-pill {
        display: inline-block;
        background: var(--light-blue);
        color: var(--primary-blue);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #bfdbfe;
    }

    /* ---------- Pipeline ---------- */
    .pipeline-wrap {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin: 10px 0 6px 0;
    }
    .pipeline-step {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        padding: 10px 16px;
        text-align: center;
        min-width: 140px;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-muted);
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .pipeline-step.done {
        border-color: #86efac;
        background: #f0fdf4;
        color: var(--green);
    }
    .pipeline-step.skip {
        border-color: var(--border-gray);
        background: #f8fafc;
        color: #94a3b8;
        text-decoration: line-through;
    }
    .pipeline-arrow {
        align-self: center;
        color: #94a3b8;
        font-weight: 700;
        font-size: 1rem;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-strong { background: #dcfce7; color: var(--green); }
    .badge-moderate { background: #ffedd5; color: var(--orange); }
    .badge-weak { background: #ffedd5; color: var(--orange); }
    .badge-poor { background: #fee2e2; color: var(--red); }

    /* ---------- Source card ---------- */
    .source-card {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    }
    .source-card .doc-name {
        font-weight: 700;
        color: var(--text-dark);
        font-size: 0.98rem;
    }
    .source-card .doc-meta {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* ---------- Answer box ---------- */
    .answer-box {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-left: 5px solid var(--primary-blue);
        border-radius: 14px;
        padding: 22px 26px;
        font-size: 1.02rem;
        line-height: 1.7;
        color: var(--text-dark);
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }

    /* ---------- Status card ---------- */
    .status-line {
        font-size: 0.92rem;
        color: var(--text-dark);
        padding: 4px 0;
    }

    /* ---------- Top status strip (replaces sidebar) ---------- */
    .status-strip {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin: 0 0 6px 0;
    }
    .status-pill {
        display: inline-block;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid var(--border-gray);
    }
    .status-pill.status-good {
        background: #f0fdf4;
        color: var(--green);
        border-color: #bbf7d0;
    }
    .status-pill.status-bad {
        background: #fef2f2;
        color: var(--red);
        border-color: #fecaca;
    }
    .status-pill.status-pending {
        background: #fffbeb;
        color: #b45309;
        border-color: #fde68a;
    }

    /* ---------- Section title ---------- */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-dark);
        margin: 6px 0 10px 0;
    }

    /* ---------- Text input / textarea ---------- */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: var(--text-dark) !important;
        border: 1.5px solid var(--border-gray) !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
        padding: 14px 16px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.03) !important;
    }
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: 1px solid var(--border-gray);
        background: #ffffff;
        color: var(--text-dark);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--primary-blue);
        color: #ffffff;
        border: none;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #1d4ed8;
        color: #ffffff;
    }

    footer-note {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    /* ---------- Query Improvement ---------- */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--border-gray);
        border-radius: 14px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }
    .query-improve {
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .query-improve-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #7c3aed;
        margin-bottom: 4px;
    }
    .query-improve-text {
        color: var(--text-dark) !important;
        font-size: 0.98rem;
        background: #ffffff;
        border: 1px solid #e9e5fc;
        border-radius: 8px;
        padding: 10px 14px;
    }
    .query-improve-final {
        border-color: #c4b5fd;
        font-weight: 600;
    }
    .query-improve-arrow {
        text-align: center;
        color: #a78bfa;
        font-size: 1.3rem;
        margin: 10px 0;
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
    status_html = '<span class="status-pill status-bad">🔴 Pipeline failed to load</span>'
elif st.session_state.rag is not None:
    status_html = (
        '<span class="status-pill status-good">🟢 Embedding Model</span>'
        '<span class="status-pill status-good">🟢 Vector Store</span>'
        '<span class="status-pill status-good">🟢 Groq Connected</span>'
        '<span class="status-pill status-good">🟢 Documents Loaded</span>'
    )
else:
    status_html = '<span class="status-pill status-pending">🟡 Pipeline not loaded yet</span>'

st.markdown(f'<div class="status-strip">{status_html}</div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="app-header">
    <h1>🩺 Medical Corrective RAG Assistant</h1>
    <p>AI-powered Medical Question Answering using Corrective Retrieval-Augmented Generation (Corrective RAG)</p>
</div>
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

def render_pipeline(quality):
    corrective_needed = quality in ("Weak", "Poor")

    steps = [
        ("Question", True),
        ("Query Rewriter", True),
        ("Retriever (FAISS)", True),
        ("Context Evaluation", True),
        ("Corrective Retrieval", corrective_needed),
        ("LLM Answer", True),
    ]

    html_parts = ['<div class="pipeline-wrap">']

    for i, (label, done) in enumerate(steps):
        if i > 0:
            html_parts.append('<div class="pipeline-arrow">➜</div>')

        if label == "Corrective Retrieval" and not corrective_needed:
            html_parts.append(f'<div class="pipeline-step skip">{label}</div>')
        else:
            css_class = "pipeline-step done" if done else "pipeline-step"
            check = " ✔" if done else ""
            html_parts.append(f'<div class="{css_class}">{label}{check}</div>')

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    if not corrective_needed:
        st.markdown("✔ **Corrective Retrieval Skipped**")
    else:
        st.markdown("✔ **Query Corrected**  \n✔ **Retrieval Repeated**")


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
        render_pipeline(result["quality"])
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
        "Strong": ("badge-strong", "🟢 Strong", "The retrieved documents are highly relevant."),
        "Moderate": ("badge-moderate", "🟠 Moderate", "The retrieved documents are reasonably relevant."),
        "Weak": ("badge-weak", "🟠 Weak", "The system performed corrective retrieval."),
        "Poor": ("badge-poor", "🔴 Poor", "Relevant information could not be confidently retrieved."),
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

    if documents:
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