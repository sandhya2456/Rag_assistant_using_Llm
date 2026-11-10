"""
evaluate.py - Measures retrieval accuracy (Hit@k) and latency on eval_questions.json.
Run:  python evaluate.py            (retrieval only, fast)
      python evaluate.py --llm      (also times full answer generation)
Use these numbers in the 'Results / Evaluation' part of your report.
"""
import json
import sys
import time

from rag_engine import BASE_DIR, DEFAULT_EMBED_MODEL, RAGEngine

USE_LLM = "--llm" in sys.argv


def main():
    engine = RAGEngine(embed_model=DEFAULT_EMBED_MODEL)
    if not engine.load_index():
        engine.build_index()

    items = json.loads((BASE_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    ks = [1, 3, 5]
    hits_at = {k: 0 for k in ks}
    retrieval_times, answer_times = [], []

    for item in items:
        t0 = time.time()
        hits = engine.retrieve(item["question"], k=max(ks))
        retrieval_times.append(time.time() - t0)
        kw = item["keyword"].lower()
        for k in ks:
            if any(kw in h["text"].lower() for h in hits[:k]):
                hits_at[k] += 1
        status = "OK  " if any(kw in h["text"].lower() for h in hits[:3]) else "MISS"
        print(f"[{status}] {item['question']}")
        if USE_LLM:
            t0 = time.time()
            ans, _ = engine.answer(item["question"], k=3)
            answer_times.append(time.time() - t0)
            print(f"        -> {ans[:150].strip()}...")

    n = len(items)
    print("\n=== RESULTS ===")
    for k in ks:
        print(f"Hit@{k}: {hits_at[k]}/{n} = {hits_at[k] / n:.0%}")
    print(f"Avg retrieval latency: {sum(retrieval_times) / n * 1000:.0f} ms")
    if answer_times:
        print(f"Avg end-to-end answer time: {sum(answer_times) / n:.1f} s")


if __name__ == "__main__":
    main()
