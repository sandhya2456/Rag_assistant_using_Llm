"""
ingest.py - (Re)build the vector index from everything inside ./docs
Run:  python ingest.py
"""
from rag_engine import DEFAULT_EMBED_MODEL, RAGEngine

if __name__ == "__main__":
    stats = RAGEngine(embed_model=DEFAULT_EMBED_MODEL).build_index()
    print(f"Done: {stats['files']} files, {stats['pages']} pages, {stats['chunks']} chunks indexed.")
