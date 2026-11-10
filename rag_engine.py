"""
rag_engine.py
Core of the RAG-Based University Academic Assistant (fully local).

Pipeline:  documents -> pages -> chunks -> embeddings -> FAISS index
           question  -> embedding -> top-k chunks -> prompt -> local LLM (Ollama)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Avoid the hf_xet DLL (blocked by some Windows Device Guard / App Control policies)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
STORE_DIR = BASE_DIR / "vectorstore"

UNIVERSITY_NAME = "ITM University"
# For better Hindi/Hinglish retrieval you can switch to:
#   "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.1:8b"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

NOT_FOUND_MSG = (
    "I couldn't find this in the official university documents I have. "
    "Please contact the concerned department / Registrar's office, "
    "or ask the admin to add the relevant document to the knowledge base."
)

SYSTEM_PROMPT = f"""You are the Academic Assistant of {UNIVERSITY_NAME}.
You help students with questions about courses, attendance, exams, fees, grading and academic policies.

STRICT RULES:
1. Answer ONLY using the numbered CONTEXT passages given to you. Never use outside knowledge.
2. If the answer is not clearly present in the CONTEXT, say you could not find it in the official documents and suggest contacting the concerned department. Do not guess.
3. Never invent dates, fees, percentages or rules.
4. Reply in the same language/style the student used (English, Hindi, or Hinglish in Roman script).
5. Be concise and clear. Use short bullet points when listing rules or steps.
6. When you use a passage, cite it like [1], [2] using the passage numbers."""


# ----------------------------------------------------------------------------
# Document loading
# ----------------------------------------------------------------------------
def _load_pdf(path: Path) -> List[Dict]:
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": path.name, "page": i})
    return pages


def _load_docx(path: Path) -> List[Dict]:
    try:
        import docx  # python-docx
    except ImportError:
        return []
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [{"text": text, "source": path.name, "page": 1}] if text.strip() else []


def load_document(path: Path) -> List[Dict]:
    """Return a list of {text, source, page} for one file."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "source": path.name, "page": 1}] if text.strip() else []
    if ext == ".docx":
        return _load_docx(path)
    return []


# ----------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------
def chunk_pages(
    pages: List[Dict], chunk_words: int = 200, overlap: int = 40, min_words: int = 25
) -> List[Dict]:
    """Section-aware chunking.

    Paragraphs (blank-line separated) are the basic unit, so a policy section stays together.
    Tiny paragraphs (e.g. headings) are merged with the next one until ~min_words is reached;
    paragraphs longer than chunk_words are split into overlapping word windows.
    Every chunk keeps source + page metadata.
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    chunks: List[Dict] = []
    step = chunk_words - overlap

    for page in pages:
        text = re.sub(r"[ \t]+", " ", page["text"]).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer: List[str] = []
        count = 0

        def flush():
            nonlocal buffer, count
            if count >= 5:
                chunks.append(
                    {"text": "\n".join(buffer), "source": page["source"], "page": page["page"]}
                )
            buffer, count = [], 0

        for para in paragraphs:
            words = para.split()
            if len(words) > chunk_words:  # very long paragraph -> sliding windows
                flush()
                for start in range(0, len(words), step):
                    window = words[start:start + chunk_words]
                    if len(window) >= 5:
                        chunks.append(
                            {"text": " ".join(window), "source": page["source"], "page": page["page"]}
                        )
                    if start + chunk_words >= len(words):
                        break
                continue
            if count + len(words) > chunk_words:
                flush()
            buffer.append(para)
            count += len(words)
            if count >= min_words:
                flush()
        flush()
    return chunks


# ----------------------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------------------
def list_ollama_models() -> Tuple[List[str], Optional[str]]:
    """Return (installed_model_names, error_message_or_None)."""
    try:
        import ollama

        resp = ollama.list()
        models = resp["models"] if isinstance(resp, dict) else resp.models
        names = []
        for m in models:
            if isinstance(m, dict):
                names.append(m.get("model") or m.get("name"))
            else:
                names.append(getattr(m, "model", None) or getattr(m, "name", None))
        return [n for n in names if n], None
    except Exception as e:  # server not running, etc.
        return [], str(e)


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL, llm_model: str = DEFAULT_LLM):
        self.embed_model_name = embed_model
        self.llm_model = llm_model
        self._embedder = None
        self.index = None
        self.chunks: List[Dict] = []
        self.files: List[str] = []

    # -- embeddings ---------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _embed(self, texts: List[str]):
        import faiss
        import numpy as np

        vecs = self.embedder.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        vecs = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(vecs)  # cosine similarity via inner product
        return vecs

    # -- index build / load -------------------------------------------------
    def build_index(self, docs_dir: Path = DOCS_DIR) -> Dict:
        import faiss

        docs_dir = Path(docs_dir)
        docs_dir.mkdir(exist_ok=True)
        files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No documents found in '{docs_dir}'. Add PDF/TXT/DOCX files first.")

        pages: List[Dict] = []
        for f in files:
            pages.extend(load_document(f))
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No readable text found (scanned/image PDFs need OCR).")

        vecs = self._embed([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        STORE_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(STORE_DIR / "index.faiss"))
        meta = {
            "embed_model": self.embed_model_name,
            "files": [f.name for f in files],
            "chunks": chunks,
        }
        (STORE_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        self.index, self.chunks, self.files = index, chunks, meta["files"]
        return {"files": len(files), "pages": len(pages), "chunks": len(chunks)}

    def load_index(self) -> bool:
        import faiss

        idx_path, meta_path = STORE_DIR / "index.faiss", STORE_DIR / "meta.json"
        if not (idx_path.exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embed_model") != self.embed_model_name:
            return False  # index built with a different embedding model
        self.index = faiss.read_index(str(idx_path))
        self.chunks = meta["chunks"]
        self.files = meta.get("files", [])
        return True

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if not self.ready:
            raise RuntimeError("Knowledge base is empty. Build the index first.")
        q = self._embed([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            hit = dict(self.chunks[i])
            hit["score"] = float(score)
            hits.append(hit)
        return hits

    @staticmethod
    def _retrieval_query(question: str, history: List[Dict]) -> str:
        """Short follow-ups ("and late fee?") borrow the previous question for better retrieval."""
        prev_users = [m["content"] for m in history if m["role"] == "user"]
        if prev_users and len(question.split()) <= 6:
            return f"{prev_users[-1]} {question}"
        return question

    # -- generation ---------------------------------------------------------
    def _build_messages(self, question: str, hits: List[Dict], history: List[Dict]) -> List[Dict]:
        context = "\n\n".join(
            f"[{i}] (Source: {h['source']}, page {h['page']})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in history[-6:])
        messages.append(
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nSTUDENT QUESTION: {question}"}
        )
        return messages

    def _stream(self, messages: List[Dict]) -> Iterator[str]:
        try:
            import ollama

            for part in ollama.chat(
                model=self.llm_model,
                messages=messages,
                stream=True,
                options={"temperature": 0.2},
            ):
                yield part["message"]["content"]
        except Exception as e:
            yield (
                f"\n\n⚠️ Could not get a response from the local LLM ({e}). "
                f"Make sure Ollama is running and the model '{self.llm_model}' is installed "
                f"(`ollama pull {self.llm_model}`)."
            )

    def stream_answer(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        k: int = 4,
        min_score: float = 0.20,
    ) -> Tuple[Iterator[str], List[Dict]]:
        """Return (token_stream, source_chunks). Skips the LLM when nothing relevant is found."""
        history = history or []
        hits = self.retrieve(self._retrieval_query(question, history), k=k)
        relevant = [h for h in hits if h["score"] >= min_score]
        if not relevant:
            return iter([NOT_FOUND_MSG]), []
        return self._stream(self._build_messages(question, relevant, history)), relevant

    def answer(self, question: str, history=None, k: int = 4, min_score: float = 0.20):
        stream, sources = self.stream_answer(question, history, k, min_score)
        return "".join(stream), sources
