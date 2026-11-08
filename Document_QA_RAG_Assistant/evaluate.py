"""Run tests/questions.json against the index and report how well the pipeline performs.

  python evaluate.py                      use the default chat model
  python evaluate.py --model llama3.2:3b  compare another model
  python evaluate.py --limit 5            only the first 5 questions (quick check)

For every question it checks:
  - retrieval: did the retrieved chunks include the expected file and page?
  - answer: does the answer contain the expected facts?
  - citation: are the pages cited under the answer exactly the right ones?
  - refusal: for questions NOT covered by the documents, did the model refuse instead of guessing?
The 'best dist' column shows the distance of the closest chunk. Compare it for answerable and
unanswerable questions to choose a sensible MAX_DISTANCE in rag.py.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import rag


def norm(text: str) -> str:
    return text.lower().replace(",", "")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=rag.CHAT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions")
    args = parser.parse_args()

    questions = json.loads(Path("tests/questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    if rag.index_size() == 0:
        print("No index found. Run first:  python ask.py index")
        sys.exit(1)
    store, llm = rag.open_store(), rag.get_llm(args.model)

    rows, answerable_ok, refusal_ok = [], [], []
    for q in questions:
        start = time.time()
        raw = rag.retrieve(store, q["question"], max_distance=float("inf"))   # unfiltered, for the distance column
        best = raw[0][1] if raw else float("nan")
        hits = [(d, s) for d, s in raw if s <= rag.MAX_DISTANCE]
        answer = "".join(rag.stream_answer(llm, q["question"], hits)).strip()
        refused = (not hits) or rag.is_refusal(answer)
        cited_txt = ""

        if q["doc"]:   # answerable question
            retrieved = any(d.metadata["source"] == q["doc"] and d.metadata["page"] in q["pages"] for d, _ in hits)
            correct = (not refused) and all(norm(k) in norm(answer) for k in q["expect"])
            cited = [] if refused else rag.cite(store, answer, hits, question=q["question"])
            cited_ok = bool(cited) and all(
                d.metadata["source"] == q["doc"] and d.metadata["page"] in q["pages"] for d, _ in cited)
            answerable_ok.append((retrieved, correct, cited_ok))
            cited_txt = "; ".join(f"{d.metadata['source'][:8]} p{d.metadata['page']} ({s:.2f})" for d, s in cited)
            verdict = (f"retr {'OK' if retrieved else 'MISS'}, ans {'OK' if correct else 'WRONG'}, "
                       f"cite {'OK' if cited_ok else 'WRONG'}")
        else:          # must be refused
            refusal_ok.append(refused)
            verdict = "refused OK" if refused else "HALLUCINATED"

        one_line = re.sub(r"\s+", " ", answer)[:90]
        rows.append((q["id"], q["question"], best, verdict, one_line, time.time() - start))
        print(f"{q['id']:>2}. {verdict:40} best dist {best:.3f}  {time.time() - start:4.0f}s | {q['question']}")
        print(f"      -> {one_line}")
        if cited_txt:
            print(f"      cited: {cited_txt}   expected: p{q['pages']}")

    n_ans, n_ref = len(answerable_ok), len(refusal_ok)
    summary = [f"Model: {args.model} | embeddings: {rag.EMBED_MODEL} | chunk {rag.CHUNK_SIZE}/{rag.CHUNK_OVERLAP} | top-k {rag.TOP_K}"]
    if n_ans:
        summary.append(f"Retrieval hit rate:  {sum(r for r, _, _ in answerable_ok)}/{n_ans}")
        summary.append(f"Correct answers:     {sum(c for _, c, _ in answerable_ok)}/{n_ans}")
        summary.append(f"Correct citations:   {sum(k for _, _, k in answerable_ok)}/{n_ans}")
    if n_ref:
        summary.append(f"Correct refusals:    {sum(refusal_ok)}/{n_ref}")
    print("\n" + "\n".join(summary))

    out = Path("tests") / f"results_{re.sub(r'[^A-Za-z0-9.-]', '_', args.model)}.md"
    table = ["| # | Question | Best dist | Result | Time |", "|---|---|---|---|---|"]
    table += [f"| {i} | {qt} | {b:.3f} | {v} | {t:.0f}s |" for i, qt, b, v, _, t in rows]
    out.write_text("\n".join(summary) + "\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
