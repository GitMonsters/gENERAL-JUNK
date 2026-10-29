#!/usr/bin/env python3
"""Did the physics knowledge survive SFT?  Compare physics LM-PPL across the chain.

    base (no physics) -> physics2 (continued pretrain) -> chat_* (SFT variants)

Answers the question "does SFT wash out the pretrain gain, and does the
auxiliary LM loss (sft.py --lm-data) stop it?" with numbers.

    python physics_chain.py
    python physics_chain.py --checkpoints "a=runs/small/best.pt,b=runs/physics2/best.pt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from eval_physics import load, ppl
from gmllm.data import chunk_stream, read_jsonl_texts, texts_to_token_stream

T = Path.home() / "AppData/Local/Temp/opencode"

DEFAULT_CKPTS = (
    "base=runs/small/best.pt,"
    "physics2=runs/physics2/best.pt,"
    "chat_atomic=runs/chat_atomic/best.pt,"
    "chat_physics=runs/chat_physics/best.pt,"
    "chat_physics_lm=runs/chat_physics_lm/best.pt"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=DEFAULT_CKPTS)
    ap.add_argument("--data", default=str(T / "physics_corpus.jsonl"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    texts = read_jsonl_texts([args.data])
    held = texts[-args.n:]
    print(f"[data] {len(texts)} pages; evaluating on last {len(held)}\n")

    print(f"{'stage':32s} {'step':>5s} {'vocab':>6s}  physics-LM-PPL")
    print("-" * 62)
    for spec in args.checkpoints.split(","):
        spec = spec.strip()
        if not spec:
            continue
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).parent.name, spec)
        if not Path(path).exists():
            print(f"{label:32s} {'':>5s} {'':>6s}  MISSING {path}")
            continue
        model, tok, ck = load(str(path))
        chunks = chunk_stream(texts_to_token_stream(held, tok), args.ctx)
        v = ppl(model, chunks)
        print(f"{label:32s} {ck.get('step'):>5} {tok.vocab_size:>6}  {v:.1f}")


if __name__ == "__main__":
    main()
