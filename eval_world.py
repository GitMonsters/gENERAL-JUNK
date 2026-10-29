#!/usr/bin/env python3
"""World-knowledge probes + held-out evaluation.

Two independent checks, because a language model can be good at prose and bad at
recall (see eval_physics_facts.py, which found 0/40 at 3.3 M params):

  1. **Held-out world PPL** — perplexity on encyclopedia articles that were NOT
     in the pretrain mix.  This is the quantity continued-pretraining moves.
  2. **Fact-completion probes** — "Abraham Lincoln was the ___ President" ->
     "16th".  A hit-rate, not a perplexity, so a 0% score is visible.

    python eval_world.py --checkpoints "base=runs/small/best.pt,mix=runs/worldmix/best.pt"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from eval_physics import load, ppl
from gmllm.data import chunk_stream, read_jsonl_texts, texts_to_token_stream

T = Path.home() / "AppData/Local/Temp/opencode"

# (prompt, [acceptable answer substrings, case-insensitive])
FACTS = [
    ("Abraham Lincoln was the", ["16th", "sixteenth", "president"]),
    ("The capital of France is", ["paris"]),
    ("The capital of Japan is", ["tokyo"]),
    ("William Shakespeare wrote", ["hamlet", "romeo", "macbeth", "play"]),
    ("The largest planet in the solar system is", ["jupiter"]),
    ("The first president of the United States was", ["washington"]),
    ("World War II ended in", ["1945", "forty-five"]),
    ("The United Nations was founded in", ["1945", "forty-five"]),
    ("The author of \"Pride and Prejudice\" is", ["austen"]),
    ("The chemical symbol for gold is", ["gold", "au"]),
    ("The longest river in Africa is the", ["nile"]),
    ("The Mona Lisa was painted by", ["da vinci", "vinci", "leonardo"]),
    ("The theory of relativity was developed by", ["einstein"]),
    ("The language spoken in Brazil is", ["portuguese"]),
    ("Mount Everest is located in", ["nepal", "himalaya", "china"]),
    ("The human body has", ["bones", "organs", "cells", "systems"]),
    ("A group of lions is called a", ["pride"]),
    ("The currency of Japan is the", ["yen"]),
    ("Charles Darwin is known for the theory of", ["evolution", "natural selection"]),
    ("The Roman Empire fell in the year", ["476", "395", "400", "fifth"]),
]


@torch.no_grad()
def complete(model, tok, prompt, max_new=10, greedy=True, temperature=0.7):
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, max_new_tokens=max_new,
                         temperature=0.0 if greedy else temperature,
                         top_k=40, rep_penalty=1.0)
    gen = out[0].tolist()[len(ids[0]):]
    return tok.decode(gen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--data", default=str(T / "world_corpus_enwiki.jsonl"))
    ap.add_argument("--n", type=int, default=200, help="held-out articles (tail)")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/world_eval.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    texts = read_jsonl_texts([args.data])
    held = texts[-args.n:]
    report = {"held_docs": len(held), "facts": [p for p, _ in FACTS],
              "checkpoints": {}}

    for spec in args.checkpoints.split(","):
        spec = spec.strip()
        if not spec:
            continue
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).parent.name, spec)
        if not Path(path).exists():
            print(f"[{label}] MISSING {path}")
            continue
        model, tok, ck = load(path)
        chunks = chunk_stream(texts_to_token_stream(held, tok), args.ctx)
        p = ppl(model, chunks)
        hits, rows = 0, []
        for prompt, accept in FACTS:
            for mode in ("greedy", "sampled"):
                text = complete(model, tok, prompt, greedy=(mode == "greedy"))
                ok = any(a.lower() in text.lower() for a in accept)
                hits += int(ok)
                rows.append({"prompt": prompt, "mode": mode, "ok": ok,
                             "completion": text[:60]})
        total = len(FACTS) * 2
        report["checkpoints"][label] = {
            "path": path, "step": ck.get("step"), "world_ppl": round(p, 2),
            "hits": hits, "total": total, "accuracy": round(hits / total, 3),
            "rows": rows,
        }
        print(f"[{label}] step {ck.get('step')} | world-PPL(held) {p:.1f} | "
              f"fact hit-rate {hits}/{total} = {hits/total:.1%}")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[done] wrote {args.out}")
    for label, e in report["checkpoints"].items():
        print(f"\n=== {label} (world-PPL {e['world_ppl']}, {e['accuracy']:.1%}) ===")
        for r in e["rows"]:
            if r["mode"] != "greedy":
                continue
            comp = r["completion"].encode("ascii", "replace").decode("ascii")
            print(f"  {'OK ' if r['ok'] else '   '}{r['prompt'][:48]:48s} -> {comp[:38]!r}")


if __name__ == "__main__":
    main()
