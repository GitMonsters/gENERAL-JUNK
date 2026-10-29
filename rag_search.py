"""Print the top retrieved passages for a query — retrieval-only debugging.

Useful when an answer looks wrong: it shows whether the fault is retrieval
(nothing relevant found) or generation (relevant passage was found but the
answer did not use it).  That split is the whole point of RAG evaluation.

    python rag_search.py "what is the SI unit of force" --corpus docs.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from rag import BM25Retriever
from rag.retriever import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit-docs", type=int, default=4000)
    ap.add_argument("--target", type=int, default=900)
    args = ap.parse_args()
    q = " ".join(args.query)

    r = BM25Retriever.from_jsonl(args.corpus, target=args.target,
                                 limit_docs=args.limit_docs)
    qs = set(tokenize(q))
    print(f"query: {q}\ntokens: {sorted(qs)}\n")
    for i, h in enumerate(r.search(q, k=args.k)):
        cov = len(qs & set(tokenize(h.chunk.text))) / max(len(qs), 1)
        print(f"[{i+1}] score={h.score:.2f} coverage={cov:.2f}  {h.chunk.title}")
        print(f"    {h.chunk.text[:260]}\n")


if __name__ == "__main__":
    main()
