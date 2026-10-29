#!/usr/bin/env python3
"""Evaluate a gmllm checkpoint: held-out perplexity + generated samples.

    python evaluate.py --checkpoint runs/small/best.pt \
        --text-data "%TEMP%/opencode/gitmonsters/octotetrahedral-agi/data/combined_train.jsonl" \
        --ood "%TEMP%/opencode/gitmonsters/octotetrahedral-agi/data/eval_heldout.jsonl" \
        --prompt "the "
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.data import chunk_stream, read_jsonl_texts, texts_to_token_stream, train_val_split
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


@torch.no_grad()
def ppl_of(model, chunks, batch_size=16, max_batches=128) -> float:
    model.eval()
    total, count = 0.0, 0
    n = min(len(chunks), max_batches * batch_size)
    for i in range(0, n, batch_size):
        xb = chunks[i : i + batch_size]
        _, _, loss, _ = model(xb, targets=xb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    return math.exp(min(total / max(count, 1), 20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text-data", default=None, help="JSONL corpus for in-domain PPL")
    p.add_argument("--ood", default=None, help="JSONL/text for out-of-domain PPL")
    p.add_argument("--prompt", default="the ")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint: {args.checkpoint} (step {ckpt.get('step')}, "
          f"val_ppl at save: {ckpt.get('val_ppl')})")
    print(f"model: {model.num_params()/1e6:.2f}M params, causal={cfg.causal}, "
          f"limbs={cfg.use_limbs}, tetra_bias={cfg.use_tetra_bias}, torus_rope={cfg.use_torus_rope}")

    if args.text_data:
        texts = read_jsonl_texts([s.strip() for s in args.text_data.split(",")])
        stream = texts_to_token_stream(texts, tok)
        chunks = chunk_stream(stream, cfg.ctx)
        _, val = train_val_split(chunks, 0.02, seed=1337)  # same split rule as training
        print(f"in-domain held-out causal PPL: {ppl_of(model, val):.3f}  (n={len(val)} chunks)")

    if args.ood:
        ood_texts = read_jsonl_texts([args.ood])
        stream = texts_to_token_stream(ood_texts, tok)
        chunks = chunk_stream(stream, cfg.ctx)
        if len(chunks):
            print(f"out-of-domain causal PPL:    {ppl_of(model, chunks):.3f}  (n={len(chunks)} chunks)")

    ids = torch.tensor([tok.encode(args.prompt) or [ord("a")]], dtype=torch.long)
    for i in range(args.samples):
        out = model.generate(ids.clone(), max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k)
        print(f"sample {i+1}: {tok.decode(out[0].tolist())!r}")


if __name__ == "__main__":
    main()
