"""
app.py - Streamlit chat UI for the RAG-Based University Academic Assistant.
Run:  streamlit run app.py
"""
import streamlit as st

from rag_engine import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM,
    DOCS_DIR,
    SUPPORTED_EXTENSIONS,
    UNIVERSITY_NAME,
    RAGEngine,
    list_ollama_models,
)

st.set_page_config(page_title="Academic Assistant", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_engine(embed_model: str) -> RAGEngine:
    engine = RAGEngine(embed_model=embed_model)
    if not engine.load_index():
        try:
            engine.build_index()
        except FileNotFoundError:
            pass  # user will upload documents from the sidebar
    return engine


# ------------------------------ Sidebar ------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    installed, ollama_error = list_ollama_models()
    if ollama_error:
        st.error("Ollama is not reachable. Start the Ollama app and refresh.")
        installed = []
    llm_options = installed or [DEFAULT_LLM]
    default_idx = llm_options.index(DEFAULT_LLM) if DEFAULT_LLM in llm_options else 0
    llm_model = st.selectbox("Local LLM", llm_options, index=default_idx)

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
    min_score = st.slider("Minimum relevance", 0.0, 0.6, 0.20, 0.05,
                          help="Chunks scoring below this are ignored. Raise it for stricter answers.")

    st.divider()
    st.subheader("📚 Knowledge base")
    engine = get_engine(DEFAULT_EMBED_MODEL)
    engine.llm_model = llm_model

    uploads = st.file_uploader(
        "Add documents (PDF / TXT / DOCX)",
        type=[e.strip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if st.button("➕ Add & rebuild index", use_container_width=True):
        DOCS_DIR.mkdir(exist_ok=True)
        for up in uploads or []:
            (DOCS_DIR / up.name).write_bytes(up.getbuffer())
        with st.spinner("Indexing documents..."):
            try:
                stats = engine.build_index()
                st.success(f"Indexed {stats['files']} files, {stats['chunks']} chunks.")
            except Exception as e:
                st.error(str(e))

    if engine.ready:
        st.caption(f"{len(engine.files)} document(s) · {len(engine.chunks)} chunks")
        for name in engine.files:
            st.markdown(f"- {name}")
    else:
        st.warning("No documents indexed yet. Upload a PDF to begin.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------ Main chat ----------------------------------
st.title("🎓 University Academic Assistant")
st.caption(f"{UNIVERSITY_NAME} · Private & offline · Powered by local LLM + RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']}** — page {s['page']} · match {s['score']:.0%}")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


if not st.session_state.messages:
    st.info("Ask things like: *What is the attendance policy?* · *Late fee kitni lagti hai?* · "
            "*Can I take medical leave?*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

question = st.chat_input("Ask about courses, attendance, exams, fees...")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not engine.ready:
            reply, sources = "Please upload and index at least one document first (see sidebar).", []
            st.markdown(reply)
        else:
            stream, sources = engine.stream_answer(question, history, k=top_k, min_score=min_score)
            reply = st.write_stream(stream)
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
