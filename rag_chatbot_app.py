"""
RAG Chatbot — PDF-Grounded Question Answering
A production-ready Streamlit application powered by LangChain, FAISS, and HuggingFace models.

Features:
  • PDF ingestion & recursive text chunking
  • Zero-shot technical document validation (facebook/bart-large-mnli)
  • Semantic vector search via FAISS + all-MiniLM-L6-v2 embeddings
  • Answer generation with google/flan-t5-base
  • Persistent chat history with source citations
"""

import streamlit as st
import time
import textwrap
from pathlib import Path

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DocMind · RAG Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2230;
    --border-hi: #2e3450;
    --cyan:      #00e5c8;
    --cyan-dim:  #00b89c;
    --amber:     #f5a623;
    --red:       #ff4d6a;
    --muted:     #4a5070;
    --text:      #cdd6f4;
    --text-dim:  #8891b0;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Logo / Title Block ── */
.logo-block {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: -0.02em;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.logo-block span { color: var(--text-dim); font-weight: 400; font-size: 0.75rem; display: block; margin-top: 4px; letter-spacing: 0.05em; }

/* ── Status Pills ── */
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.06em;
    padding: 4px 10px; border-radius: 20px;
    background: var(--border); color: var(--text-dim);
}
.pill.active  { background: rgba(0,229,200,0.12); color: var(--cyan);  border: 1px solid var(--cyan-dim); }
.pill.warn    { background: rgba(245,166,35,0.12); color: var(--amber); border: 1px solid var(--amber); }
.pill.error   { background: rgba(255,77,106,0.12); color: var(--red);   border: 1px solid var(--red); }
.pill-dot { width:6px; height:6px; border-radius:50%; background:currentColor; }

/* ── Section headers ── */
.sec-label {
    font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.12em;
    color: var(--muted); text-transform: uppercase; margin: 1.2rem 0 0.5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border-hi) !important;
    border-radius: 10px !important;
    background: rgba(0,229,200,0.03) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan-dim) !important; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 10px; margin: 0.8rem 0; }
.metric-card {
    flex: 1; padding: 12px 14px; border-radius: 8px;
    background: var(--border); border: 1px solid var(--border-hi);
}
.metric-card .val { font-family: var(--mono); font-size: 1.3rem; color: var(--cyan); font-weight: 700; }
.metric-card .lbl { font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* ── Chat messages ── */
.chat-wrap {
    display: flex; flex-direction: column; gap: 14px;
    padding: 1rem 0;
}
.msg {
    display: flex; gap: 12px; align-items: flex-start;
    animation: fadeUp 0.3s ease;
}
.msg.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.7rem; font-weight: 700;
}
.avatar.ai   { background: rgba(0,229,200,0.15); color: var(--cyan);  border: 1px solid var(--cyan-dim); }
.avatar.user { background: rgba(245,166,35,0.15); color: var(--amber); border: 1px solid var(--amber); }
.bubble {
    max-width: 76%; padding: 12px 16px; border-radius: 12px;
    font-size: 0.92rem; line-height: 1.65; color: var(--text);
}
.bubble.ai   { background: var(--surface); border: 1px solid var(--border-hi); border-top-left-radius: 2px; }
.bubble.user { background: rgba(245,166,35,0.1); border: 1px solid rgba(245,166,35,0.25); border-top-right-radius: 2px; }

/* ── Source accordion ── */
.src-header {
    font-family: var(--mono); font-size: 0.7rem; color: var(--muted);
    margin-top: 10px; letter-spacing: 0.05em; cursor: pointer;
}
.src-block {
    margin-top: 6px; padding: 10px 12px; border-radius: 6px;
    background: rgba(0,0,0,0.3); border-left: 2px solid var(--cyan-dim);
    font-size: 0.8rem; color: var(--text-dim); line-height: 1.6;
    font-family: var(--mono);
}

/* ── Input row ── */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: var(--sans) !important; font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--cyan-dim) !important;
    box-shadow: 0 0 0 2px rgba(0,229,200,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--mono) !important; font-size: 0.78rem !important;
    letter-spacing: 0.05em !important; text-transform: uppercase !important;
    background: rgba(0,229,200,0.08) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan-dim) !important;
    border-radius: 6px !important; padding: 0.45rem 1.1rem !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: rgba(0,229,200,0.18) !important;
    box-shadow: 0 0 14px rgba(0,229,200,0.2) !important;
}

/* ── Progress / Spinner ── */
.stProgress > div > div > div { background: var(--cyan) !important; }
[data-testid="stSpinner"] { color: var(--cyan) !important; }

/* ── Selectbox / Slider ── */
[data-baseweb="select"] { background: var(--surface) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.4; }
}
.pulsing { animation: pulse 1.8s infinite; }

/* ── Welcome card ── */
.welcome-card {
    border: 1px solid var(--border-hi); border-radius: 12px;
    padding: 2.5rem; text-align: center;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,229,200,0.06) 0%, transparent 70%);
}
.welcome-card h2 { font-family: var(--mono); font-size: 1.4rem; color: var(--cyan); margin-bottom: 0.5rem; }
.welcome-card p  { color: var(--text-dim); font-size: 0.9rem; line-height: 1.7; max-width: 460px; margin: 0 auto; }
.step-list {
    display: flex; justify-content: center; gap: 2rem; margin-top: 1.8rem;
    flex-wrap: wrap;
}
.step { text-align: center; }
.step-num {
    width: 32px; height: 32px; border-radius: 50%; margin: 0 auto 8px;
    background: rgba(0,229,200,0.12); border: 1px solid var(--cyan-dim);
    color: var(--cyan); font-family: var(--mono); font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center;
}
.step-txt { font-size: 0.78rem; color: var(--text-dim); }

/* ── Pipeline tracker ── */
.pipeline {
    display: flex; align-items: center; gap: 0; margin: 0.8rem 0;
}
.p-step {
    display: flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.04em;
}
.p-step.done  { color: var(--cyan); }
.p-step.active{ color: var(--amber); }
.p-step.idle  { color: var(--muted); }
.p-icon { font-size: 0.8rem; }
.p-arrow { margin: 0 6px; color: var(--border-hi); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }

/* ── Suggestion chips ── */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 1rem 0 0.5rem; }
.chip {
    padding: 5px 12px; border-radius: 20px; cursor: pointer;
    border: 1px solid var(--border-hi); background: var(--surface);
    font-size: 0.78rem; color: var(--text-dim);
    transition: all 0.15s;
}
</style>
""", unsafe_allow_html=True)


# ─── Lazy imports (cached so they load once) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource(show_spinner=False)
def load_generator():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")


# ─── Core Pipeline Functions ──────────────────────────────────────────────────
def process_pdf(uploaded_file) -> list:
    """Load PDF and split into overlapping chunks."""
    import tempfile, os
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    except Exception as e:
        os.unlink(tmp_path)
        err = str(e)
        if "cryptography" in err.lower() or "aes" in err.lower() or "encrypted" in err.lower():
            st.error(
                "🔒 **Encrypted PDF detected.**\n\n"
                "This PDF uses AES encryption. Please either:\n"
                "- Remove the password using Adobe Acrobat, Preview, or an online tool (e.g. smallpdf.com)\n"
                "- Or run: `pip install cryptography` and retry\n\n"
                f"Technical detail: `{err}`"
            )
        else:
            st.error(f"❌ **Failed to read PDF:** {err}")
        st.stop()

    os.unlink(tmp_path)

    if not documents:
        st.error("❌ No text could be extracted from this PDF. It may be scanned/image-only.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.get("chunk_size", 200),
        chunk_overlap=st.session_state.get("chunk_overlap", 100),
    )
    return splitter.split_documents(documents)


def validate_document(chunks: list) -> tuple[bool, float]:
    """Zero-shot classify a sample of the document as technical/non-technical."""
    classifier = load_classifier()
    sample = (
        chunks[0].page_content
        + " "
        + chunks[len(chunks) // 2].page_content
        + " "
        + chunks[-1].page_content
    )
    labels = ["technical document", "non-technical document"]
    result = classifier(sample[:800], candidate_labels=labels)  # truncate for speed
    is_tech = result["labels"][0] == "technical document"
    confidence = result["scores"][0]
    return is_tech, confidence


def build_vectorstore(chunks: list):
    """Build FAISS vectorstore from document chunks."""
    from langchain_community.vectorstores import FAISS
    embeddings = load_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


def answer_query(vectorstore, query: str, k: int = 3) -> tuple[str, list]:
    """Retrieve relevant chunks and generate a grounded answer."""
    generator = load_generator()
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Answer the question using ONLY the context below.
If the answer is a specific value, return only the value.
If explanation is needed, give a short clear answer.
If the answer is not present, say "Not found in document."

Context:
{context}

Question:
{query}

Answer:"""

    response = generator(prompt, max_length=150, min_length=5)
    answer_text = response[0]["generated_text"].strip()
    return answer_text, results

# ─── Session State Defaults ───────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vectorstore": None,
    "doc_valid": None,
    "doc_confidence": 0.0,
    "doc_name": None,
    "chunk_count": 0,
    "processing": False,
    "pipeline_stage": 0,  # 0=idle,1=chunking,2=validating,3=indexing,4=ready
    "chunk_size": 200,
    "chunk_overlap": 100,
    "top_k": 3,
    "show_sources": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='logo-block'>
        ⚡ DocMind
        <span>RAG · FAISS · HuggingFace</span>
    </div>
    """, unsafe_allow_html=True)

    # Status
    stage = st.session_state.pipeline_stage
    if stage == 0:
        st.markdown("<div class='pill'>● Awaiting document</div>", unsafe_allow_html=True)
    elif stage == 4:
        valid = st.session_state.doc_valid
        if valid:
            st.markdown("<div class='pill active'><div class='pill-dot'></div> Index ready</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pill warn'><div class='pill-dot'></div> Non-technical doc</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='pill warn pulsing'><div class='pill-dot'></div> Processing…</div>", unsafe_allow_html=True)

    # Document upload
    st.markdown("<div class='sec-label'>📄 Document</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    # Chunk metrics
    if st.session_state.chunk_count:
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='val'>{st.session_state.chunk_count}</div>
                <div class='lbl'>Chunks</div>
            </div>
            <div class='metric-card'>
                <div class='val'>{int(st.session_state.doc_confidence*100)}%</div>
                <div class='lbl'>Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Settings
    st.markdown("<div class='sec-label'>⚙ Settings</div>", unsafe_allow_html=True)
    with st.expander("Chunking & Retrieval", expanded=False):
        st.session_state.chunk_size = st.slider("Chunk size", 100, 500, st.session_state.chunk_size, 50)
        st.session_state.chunk_overlap = st.slider("Chunk overlap", 0, 200, st.session_state.chunk_overlap, 25)
        st.session_state.top_k = st.slider("Retrieved chunks (k)", 1, 6, st.session_state.top_k)
        st.session_state.show_sources = st.checkbox("Show source chunks", value=st.session_state.show_sources)

    # Model info
    st.markdown("<div class='sec-label'>🤖 Models</div>", unsafe_allow_html=True)
    for label, model in [
        ("Embeddings", "MiniLM-L6-v2"),
        ("Classifier", "bart-large-mnli"),
        ("Generator", "flan-t5-base"),
        ("Vector DB", "FAISS (in-memory)"),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:4px 0;"
            f"font-size:0.75rem'><span style='color:#4a5070'>{label}</span>"
            f"<span style='font-family:Space Mono,monospace;font-size:0.68rem;"
            f"color:#8891b0'>{model}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🗑  Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄  Reset All"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ─── PDF Processing Pipeline ──────────────────────────────────────────────────
if uploaded and uploaded.name != st.session_state.doc_name:
    st.session_state.doc_name = uploaded.name
    st.session_state.pipeline_stage = 1
    st.session_state.messages = []

    prog_placeholder = st.empty()

    # ── Stage 1: Chunking ──
    with prog_placeholder.container():
        st.markdown("""
        <div style='padding:1.5rem;border:1px solid #1e2230;border-radius:10px;background:#111318'>
            <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#00e5c8;margin-bottom:1rem'>
                ⚙ Processing Pipeline
            </div>
            <div style='font-size:0.8rem;color:#8891b0'>Stage 1/3 — Splitting PDF into chunks…</div>
        </div>
        """, unsafe_allow_html=True)
        bar = st.progress(10, text="")

    chunks = process_pdf(uploaded)
    st.session_state.pipeline_stage = 2
    bar.progress(35, text="")

    # ── Stage 2: Validation ──
    with prog_placeholder.container():
        st.markdown("""
        <div style='padding:1.5rem;border:1px solid #1e2230;border-radius:10px;background:#111318'>
            <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#00e5c8;margin-bottom:1rem'>
                ⚙ Processing Pipeline
            </div>
            <div style='font-size:0.8rem;color:#8891b0'>Stage 2/3 — Validating document type…</div>
        </div>
        """, unsafe_allow_html=True)
        bar = st.progress(50, text="")

    is_valid, confidence = validate_document(chunks)
    st.session_state.doc_valid = is_valid
    st.session_state.doc_confidence = confidence
    st.session_state.chunk_count = len(chunks)
    st.session_state.pipeline_stage = 3
    bar.progress(70, text="")

    # ── Stage 3: Indexing ──
    if is_valid:
        with prog_placeholder.container():
            st.markdown("""
            <div style='padding:1.5rem;border:1px solid #1e2230;border-radius:10px;background:#111318'>
                <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#00e5c8;margin-bottom:1rem'>
                    ⚙ Processing Pipeline
                </div>
                <div style='font-size:0.8rem;color:#8891b0'>Stage 3/3 — Building FAISS vector index…</div>
            </div>
            """, unsafe_allow_html=True)
            bar = st.progress(85, text="")

        st.session_state.vectorstore = build_vectorstore(chunks)
        bar.progress(100, text="")

    st.session_state.pipeline_stage = 4
    time.sleep(0.4)
    prog_placeholder.empty()
    st.rerun()


# ─── Main Chat Area ───────────────────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1], gap="large")

with col_chat:
    # ── Page Header ──
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <h1 style='font-family:Space Mono,monospace;font-size:1.6rem;
                   color:#cdd6f4;margin:0;font-weight:700;letter-spacing:-0.02em'>
            DocMind
            <span style='color:#00e5c8'>·</span>
            <span style='font-size:1rem;font-weight:400;color:#4a5070'>
                PDF Question Answering
            </span>
        </h1>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome screen (no doc yet) ──
    if st.session_state.pipeline_stage == 0:
        st.markdown("""
        <div class='welcome-card'>
            <h2>Upload a technical PDF to begin</h2>
            <p>
                DocMind ingests your document, validates it as a technical source,
                builds a semantic vector index, and answers your questions with
                full source attribution — all running locally.
            </p>
            <div class='step-list'>
                <div class='step'>
                    <div class='step-num'>1</div>
                    <div class='step-txt'>Upload PDF<br>in sidebar</div>
                </div>
                <div class='step'>
                    <div class='step-num'>2</div>
                    <div class='step-txt'>Auto-validation<br>+ indexing</div>
                </div>
                <div class='step'>
                    <div class='step-num'>3</div>
                    <div class='step-txt'>Ask anything<br>about the doc</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Invalid doc warning ──
    elif st.session_state.pipeline_stage == 4 and not st.session_state.doc_valid:
        conf = int(st.session_state.doc_confidence * 100)
        st.markdown(f"""
        <div style='padding:1.5rem;border:1px solid rgba(255,77,106,0.3);
                    border-radius:10px;background:rgba(255,77,106,0.05);margin-bottom:1.5rem'>
            <div style='font-family:Space Mono,monospace;font-size:0.9rem;
                        color:#ff4d6a;margin-bottom:0.5rem'>
                ✗ Document not classified as technical ({conf}% confidence)
            </div>
            <div style='font-size:0.85rem;color:#8891b0'>
                This RAG system is optimized for technical documentation such as datasheets,
                engineering manuals, research papers, or specifications. Please upload a
                relevant technical document to proceed.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Active chat ──
    elif st.session_state.pipeline_stage == 4 and st.session_state.doc_valid:

        # Document pill
        st.markdown(
            f"<div style='margin-bottom:1rem'>"
            f"<span class='pill active'><div class='pill-dot'></div>"
            f"📄 {st.session_state.doc_name} &nbsp;·&nbsp; "
            f"{st.session_state.chunk_count} chunks</span></div>",
            unsafe_allow_html=True,
        )

        # Render chat history
        if st.session_state.messages:
            chat_html = "<div class='chat-wrap'>"
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                avatar = "AI" if role == "assistant" else "YOU"
                bubble_cls = "ai" if role == "assistant" else "user"
                chat_html += f"""
                <div class='msg {role}'>
                    <div class='avatar {bubble_cls}'>{avatar}</div>
                    <div class='bubble {bubble_cls}'>{content}</div>
                </div>"""
                if role == "assistant" and msg.get("sources") and st.session_state.show_sources:
                    chat_html += "<div style='margin-left:46px'>"
                    chat_html += "<div class='src-header'>▸ SOURCE CHUNKS</div>"
                    for i, src in enumerate(msg["sources"]):
                        snippet = textwrap.shorten(src, width=200, placeholder="…")
                        chat_html += f"<div class='src-block'>[{i+1}] {snippet}</div>"
                    chat_html += "</div>"
            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Key counter — incrementing it forces Streamlit to treat the
        # text_input as a brand-new widget, which clears the value.
        if "input_key" not in st.session_state:
            st.session_state.input_key = 0

        # Consume any pending suggestion-chip query
        pending = st.session_state.pop("_pending_query", "")

        # Input row
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            query = st.text_input(
                label="query",
                placeholder="Ask a question about your document…",
                label_visibility="collapsed",
                key=f"query_input_{st.session_state.input_key}",
                value=pending,
            )
        with btn_col:
            send = st.button("Send ➤", use_container_width=True)

        # ── CRITICAL FIX ──────────────────────────────────────────────────────
        # Only fire when the user explicitly pressed Send OR a suggestion chip
        # was clicked.  The old `(send or query)` condition re-triggered on
        # every Streamlit rerun because the input still held the typed text,
        # causing the infinite answer loop.
        # ─────────────────────────────────────────────────────────────────────
        should_submit = (send and bool(query.strip())) or bool(pending.strip())

        if should_submit:
            user_query = (pending or query).strip()

            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("Retrieving and generating…"):
                answer, source_docs = answer_query(
                    st.session_state.vectorstore,
                    user_query,
                    k=st.session_state.top_k,
                )

            src_texts = [doc.page_content for doc in source_docs]
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": src_texts,
            })
            # Bump the key → clears the input box on next render
            st.session_state.input_key += 1
            st.rerun()


# ─── Right Info Panel ─────────────────────────────────────────────────────────
with col_info:
    st.markdown("<div style='height:3.2rem'></div>", unsafe_allow_html=True)

    # Pipeline status
    stages = [
        ("Chunk", "Split PDF"),
        ("Validate", "Classify doc"),
        ("Index", "Build FAISS"),
        ("Ready", "Serve queries"),
    ]
    stage = st.session_state.pipeline_stage
    st.markdown("<div class='sec-label'>Pipeline</div>", unsafe_allow_html=True)

    for i, (name, desc) in enumerate(stages, 1):
        if i < stage:
            icon, cls = "✓", "done"
        elif i == stage:
            icon, cls = "◉", "active"
        else:
            icon, cls = "○", "idle"
        st.markdown(
            f"<div class='p-step {cls}' style='margin:6px 0;gap:8px'>"
            f"<span class='p-icon'>{icon}</span>"
            f"<span><b>{name}</b><br>"
            f"<span style='font-size:0.62rem;opacity:0.6'>{desc}</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if i < 4:
            st.markdown("<div style='width:1px;height:10px;background:#1e2230;margin-left:6px'></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Stats
    st.markdown("<div class='sec-label'>Session Stats</div>", unsafe_allow_html=True)
    n_qa = sum(1 for m in st.session_state.messages if m["role"] == "user")
    for label, val in [
        ("Questions asked", n_qa),
        ("Messages total", len(st.session_state.messages)),
        ("Chunks indexed", st.session_state.chunk_count or "—"),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"margin:5px 0;font-size:0.75rem'>"
            f"<span style='color:#4a5070'>{label}</span>"
            f"<span style='font-family:Space Mono,monospace;"
            f"color:#cdd6f4'>{val}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # About
    st.markdown("""
    <div style='font-size:0.72rem;color:#4a5070;line-height:1.7'>
        <b style='color:#8891b0'>Stack</b><br>
        LangChain · FAISS<br>
        HuggingFace Transformers<br>
        Sentence Transformers<br>
        Streamlit
    </div>
    """, unsafe_allow_html=True)