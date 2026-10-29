#!/usr/bin/env python3
"""Evaluate the RAG path — retrieval quality AND answer quality, separately.

Retrieval and generation fail for different reasons, so they are scored apart:

  * retrieval: did the correct document appear in the top-k? (recall@k, MRR)
  * answer:    is the returned text supported by the retrieved passage, and does
               the system say "I don't know" when the corpus cannot answer?

Ground truth here is the *source document title* for each question, which is
known because these questions were written against the corpus.  Cases where the
corpus genuinely cannot answer ("capital of France" in a physics corpus) are
included on purpose: the correct behaviour is to refuse, and that is scored.

    python eval_rag.py --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" \
                       --data "%TEMP%/opencode/rag_eval.jsonl" --limit-docs 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag import RAG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSONL of {title,text}")
    ap.add_argument("--data", required=True, help="JSONL of eval cases")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit-docs", type=int, default=4000)
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--out", default="runs/rag_eval.json")
    args = ap.parse_args()

    t0 = time.time()
    r = RAG.from_jsonl(args.corpus, generator=args.generator, k=args.k,
                       target=args.target, limit_docs=args.limit_docs)
    print(f"[index] {len(r.retriever.chunks)} chunks from "
          f"{args.limit_docs} docs in {time.time()-t0:.1f}s")

    cases = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    n_ret = n_ref = n_ans = n_ok = 0
    mrr = 0.0
    rows = []
    for c in cases:
        q = c["question"]
        gold = c.get("source")           # expected document title (or None)
        terms = [t.lower() for t in c.get("answer_terms", [])]
        expect_refusal = bool(c.get("expect_refusal"))
        hits = r.retrieve(q, k=args.k)
        titles = [h.chunk.title for h in hits]
        # retrieval hit: the answer *terms* are actually in the retrieved passages
        # (stricter and more meaningful than a title match — a passage can contain
        # the answer without being the page named after the concept)
        rank = 0
        if terms:
            for i, h in enumerate(hits):
                if any(t in h.chunk.text.lower() for t in terms):
                    rank = i + 1
                    break
        elif gold:
            rank = next((i + 1 for i, t in enumerate(titles)
                         if gold.lower() in t.lower()), 0)
        if terms or gold:
            n_ret += 1
            if rank:
                mrr += 1.0 / rank
        a = r.ask(q)
        if expect_refusal:
            good = not a.grounded
            n_ref += 1
        else:
            n_ans += 1
            if terms:
                # an answerable question is correct if the passage ANSWERED it:
                # relevant text was retrieved AND the system produced an answer
                good = a.grounded and any(t in a.text.lower() for t in terms)
            else:
                good = a.grounded
        n_ok += int(good)
        rows.append({"question": q, "gold": gold, "terms": terms,
                     "top": titles[:3], "rank": rank,
                     "grounded": a.grounded, "ok": good, "answer": a.text[:200]})

    recall = sum(1 for x in rows if x["gold"] and x["rank"]) / max(n_ret, 1)
    print(f"\nretrieval: recall@{args.k} = {recall:.1%}  MRR = {mrr/max(n_ret,1):.3f}"
          f"  (over {n_ret} answerable questions)")
    print(f"answers  : correct = {n_ok}/{len(rows)}  "
          f"(refusal cases {n_ref}, answerable {n_ans})")
    print("\nper-question:")
    for x in rows:
        tag = "OK " if x["ok"] else "   "
        r_ = f"rank {x['rank']}" if x["gold"] else "refusal-case"
        print(f"  {tag}{x['question'][:46]:46s} {r_:12s} -> {x['top'][0][:28] if x['top'] else '(none)'}")

    out = {"k": args.k, "limit_docs": args.limit_docs, "generator": args.generator,
           "recall": round(recall, 4), "mrr": round(mrr / max(n_ret, 1), 4),
           "answer_ok": n_ok, "total": len(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
