#!/usr/bin/env python3
"""Measure how well a checkpoint models *physics* text.

The question this answers: did continued-pretraining on the physics corpus
actually give the model physics world knowledge?  Perplexity on held-out
physics prose is the measurable proxy -- lower is better, and a checkpoint
that has never seen physics should be clearly worse.

    python eval_physics.py --checkpoints "base=runs/small/best.pt,physics=runs/physics/best.pt" \
        --data "%TEMP%/opencode/physics_corpus.jsonl" --n 2000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.data import chunk_stream, read_jsonl_texts, texts_to_token_stream
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt


@torch.no_grad()
def ppl(model, chunks, batch=16, max_batches=64):
    total, count = 0.0, 0
    n = min(len(chunks), max_batches * batch)
    for i in range(0, n, batch):
        xb = chunks[i:i + batch]
        _, _, loss, _ = model(xb, targets=xb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    return math.exp(min(total / max(count, 1), 20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="base=runs/small/best.pt")
    p.add_argument("--data", default=str(Path.home() /
                   "AppData/Local/Temp/opencode/physics_corpus.jsonl"))
    p.add_argument("--n", type=int, default=2000, help="held-out pages")
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--out", default="runs/physics_eval.json")
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    texts = read_jsonl_texts([args.data])
    held = texts[-args.n:]                       # tail = deterministic held-out
    print(f"[data] {len(texts)} physics pages; evaluating on last {len(held)}")

    out = {"n_heldout": len(held), "checkpoints": {}}
    for spec in args.checkpoints.split(","):
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[skip] {label}: {path} missing")
            continue
        model, tok, ckpt = load(path)
        stream = texts_to_token_stream(held, tok)
        chunks = chunk_stream(stream, args.ctx)
        v = ppl(model, chunks)
        out["checkpoints"][label] = {
            "path": path, "step": ckpt.get("step"), "vocab": tok.vocab_size,
            "physics_ppl": round(v, 2), "chunks": len(chunks),
        }
        print(f"  {label:10s} {path:28s} step={ckpt.get('step')} "
              f"physics-PPL={v:.2f}")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
