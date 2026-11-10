"""
cli.py - Terminal chat (use this if Streamlit is blocked on your machine).
Run:  python cli.py
"""
from rag_engine import DEFAULT_EMBED_MODEL, RAGEngine


def main():
    print("Loading embedding model and knowledge base...")
    engine = RAGEngine(embed_model=DEFAULT_EMBED_MODEL)
    if not engine.load_index():
        print("Building index from ./docs ...")
        stats = engine.build_index()
        print(f"Indexed {stats['files']} files, {stats['chunks']} chunks.")

    print("\nUniversity Academic Assistant ready. Type 'exit' to quit.\n")
    history = []
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        stream, sources = engine.stream_answer(q, history)
        print("\nAssistant: ", end="", flush=True)
        reply = ""
        for token in stream:
            print(token, end="", flush=True)
            reply += token
        print()
        for i, s in enumerate(sources, start=1):
            print(f"  [{i}] {s['source']} (page {s['page']}, match {s['score']:.0%})")
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
