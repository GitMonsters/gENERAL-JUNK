#!/usr/bin/env python3
"""Leakage ablation — reproduces the upstream octotetrahedral-agi defect and fixes it.

Upstream (octotetrahedral-agi/train_transformer.py:292 + core/tetrahedral_transformer_layer.py)
trained an LM whose attention had NO causal mask. Position i could attend to the input
embedding of position i+1, which contains token i+1 itself: teacher-forced perplexity
becomes an artifact (they measured an "impossibly good" 1.32 while generation collapsed).

This script trains two identical small models on the same corpus slice:
  leaky : causal=False  (upstream behavior)
  causal: causal=True   (the fix)
and reports, for each:
  * teacher-forced PPL measured in its native mode (what upstream reported)
  * honest causal PPL of the same weights (what upstream never measured)
  * autoregressive samples from the same prompt

Expected signature (this is the point of the experiment): the leaky model wins on its
own metric and collapses on generation; the causal model is worse on the leaky metric,
honest on its own, and generates real text.

    python leak_ablation.py --data ".../data/combined_train.jsonl" --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.data import (
    ChunkBatcher, chunk_stream, read_jsonl_texts, texts_to_token_stream, train_val_split,
)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/ablation")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--max-bytes", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


@torch.no_grad()
def native_ppl(model, chunks, batch_size=16) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, min(len(chunks), 32 * batch_size), batch_size):
        xb = chunks[i : i + batch_size]
        _, _, loss, _ = model(xb, targets=xb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    model.train()
    return math.exp(min(total / max(count, 1), 20))


def train_one(causal: bool, name: str, args, train_chunks, val_chunks, tok) -> dict:
    torch.manual_seed(args.seed)
    cfg = GMConfig(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_head=args.n_head,
        n_layer=args.n_layer, ctx=args.ctx, dropout=0.1, causal=causal,
    )
    model = GMGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    batcher = ChunkBatcher(train_chunks, args.batch, seed=args.seed)
    model.train()
    t0 = time.time()
    last_lm = 0.0
    for step in range(args.steps):
        xb = batcher.sample()
        _, hidden, loss, _ = model(xb, targets=xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_lm = loss.item()
        if step % 100 == 0:
            print(f"  [{name}] step {step:4d} lm={last_lm:.4f} ppl={math.exp(min(last_lm, 12)):.2f}")

    native = native_ppl(model, val_chunks)
    # Honest (causal) PPL of the same weights: force the causal mask at eval time.
    was_causal = model.blocks[0].attn.causal
    for b in model.blocks:
        b.attn.causal = True
        b.attn._bias_cache = None
    honest = native_ppl(model, val_chunks)
    for b in model.blocks:
        b.attn.causal = was_causal

    prompt = torch.tensor([tok.encode("the ") or [ord("t")]], dtype=torch.long)
    sample_ids = model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=40)
    result = {
        "name": name, "causal": causal, "params_m": round(model.num_params() / 1e6, 2),
        "train_time_s": round(time.time() - t0, 1),
        "native_mode_ppl": round(native, 3),
        "honest_causal_ppl": round(honest, 3),
        "sample": tok.decode(sample_ids[0].tolist()),
    }
    print(f"  [{name}] native-mode PPL={native:.3f} | honest causal PPL={honest:.3f}")
    print(f"  [{name}] sample: {result['sample']!r}")
    return result


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = read_jsonl_texts([s.strip() for s in args.data.split(",")], max_bytes=args.max_bytes)
    print(f"[data] {len(texts)} docs, {sum(len(t) for t in texts)/1e6:.1f} MB")

    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists():
        tok = ByteBPETokenizer.load(tok_path)
    else:
        tok = ByteBPETokenizer.train(texts, vocab_size=512)
        tok.save(tok_path)

    stream = texts_to_token_stream(texts, tok)
    chunks = chunk_stream(stream, args.ctx)
    train_chunks, val_chunks = train_val_split(chunks, 0.05, seed=args.seed)
    print(f"[data] {len(chunks)} chunks (train {len(train_chunks)}, val {len(val_chunks)})")

    print("\n=== LEAKY (upstream: no causal mask) ===")
    leaky = train_one(False, "leaky", args, train_chunks, val_chunks, tok)
    print("\n=== CAUSAL (the fix) ===")
    fixed = train_one(True, "causal", args, train_chunks, val_chunks, tok)

    summary = {
        "config": {k: getattr(args, k) for k in
                   ("steps", "batch", "ctx", "d_model", "n_layer", "lr", "max_bytes")},
        "leaky": leaky, "causal": fixed,
        "interpretation": (
            "The leaky model scores better ONLY under its own non-causal measurement: "
            "position i reads the answer from position i+1's input embedding. Under the "
            "honest causal measurement (and in generation) the advantage disappears."
        ),
    }
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] summary -> {out_dir/'ablation.json'}")


if __name__ == "__main__":
    main()
