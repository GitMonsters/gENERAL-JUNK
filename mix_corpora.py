#!/usr/bin/env python3
"""Merge domain corpora into one continued-pretrain stream.

Why one file instead of sequential passes: training physics then world (or vice
versa) makes the model forget whichever came first.  A single shuffled stream
with each domain upsampled to a target share trains on both at once.

    python mix_corpora.py --out pretrain_mix.jsonl \
        --in "physics=%TEMP%/opencode/physics_corpus_enwiki.jsonl:2.0" \
        --in "world=%TEMP%/opencode/world_corpus_enwiki.jsonl:1.0"

Each --in is  NAME=PATH[:WEIGHT]  (weight = integer repeat count, default 1).
Reports the realised share of each domain by characters so the mix is auditable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="NAME=PATH[:WEIGHT]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_rows = []
    stats = []
    for spec in args.inputs:
        name, rest = spec.split("=", 1)
        if ":" in rest:
            path, weight = rest.rsplit(":", 1)
            weight = int(weight)
        else:
            path, weight = rest, 1
        rows = read(path)
        chars = sum(len(r.get("text", "")) for r in rows)
        for _ in range(weight):
            out_rows.extend(rows)
        stats.append((name, len(rows), chars, weight))

    random.Random(args.seed).shuffle(out_rows)
    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows),
                   encoding="utf-8")
    total_chars = sum(c * w for _, _, c, w in stats)
    print(f"{len(out_rows)} rows -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    for name, n, c, w in stats:
        share = 100 * c * w / total_chars
        print(f"  {name:10s} {n:6d} pages x{w} = {c*w/1e6:7.1f} M chars  "
              f"({share:5.1f}% of corpus)")


if __name__ == "__main__":
    main()
