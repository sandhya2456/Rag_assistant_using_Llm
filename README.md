# RAG-Based University Academic Assistant using Local LLMs

A fully offline chatbot that answers student questions from official university documents
(PDF / DOCX / TXT) using Retrieval-Augmented Generation. No cloud API, no data leaves your PC.

**Stack:** Python 3.11 · Ollama (LLaMA 3.1 8B) · Sentence-Transformers · FAISS · Streamlit

## How it works
1. `docs/` files are read page by page and split into overlapping chunks (with source + page metadata)
2. Each chunk is embedded (all-MiniLM-L6-v2) and stored in a FAISS cosine-similarity index
3. A student question is embedded, the top-k chunks are retrieved (weak matches are filtered out)
4. Retrieved context + chat history go to the local LLM with a strict "answer only from context" prompt
5. The answer streams back with clickable source citations (file + page)

## Setup (Windows)
```bash
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Run
```bash
python ingest.py          # build index from ./docs (the UI also does this automatically)
streamlit run app.py      # web UI  ->  http://localhost:8501
python cli.py             # terminal chat (fallback if Streamlit is blocked)
python evaluate.py        # Hit@k + latency report for your project results
```

## Folder structure
```
rag_project/
├── app.py               Streamlit chat UI (upload docs, settings, sources)
├── cli.py               Terminal chat
├── rag_engine.py        Loading, chunking, FAISS, retrieval, prompting, streaming
├── ingest.py            Rebuild the index
├── evaluate.py          Retrieval accuracy + latency evaluation
├── eval_questions.json  Test questions
├── docs/                Put your university PDFs here
└── vectorstore/         Auto-generated index
```

## Tips
- Better Hindi/Hinglish retrieval: set `DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"` in `rag_engine.py`, then re-run `python ingest.py`
- Low RAM? Use a smaller model: `ollama pull llama3.2:3b` and pick it in the sidebar
- Scanned (image-only) PDFs have no text layer and need OCR first
- If Windows blocks a DLL (Device Guard), use `python -m pip ...` and `python cli.py`
