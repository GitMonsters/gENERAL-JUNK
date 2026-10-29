#!/usr/bin/env python3
"""Train gmllm on a GitMonsters JSONL corpus.

Example (the run used for RESULTS.md):
    python train.py --data "%TEMP%/opencode/gitmonsters/octotetrahedral-agi/data/combined_train.jsonl" \
        --out runs/small --steps 1000 --batch 16 --ctx 256 --d-model 256 --n-layer 4

Everything is measured and logged: per-term losses, held-out perplexity, tokens/s,
and periodic text samples. Best checkpoint (by held-out causal PPL) is saved with its
config and tokenizer.
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
    ChunkBatcher, chunk_stream, encode_free_text, read_jsonl_texts,
    texts_to_token_stream, train_val_split,
)
from gmllm.losses import CompositeLoss
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train gmllm")
    p.add_argument("--data", required=True, help="Comma-separated JSONL paths ({'text': ...})")
    p.add_argument("--init", default=None, help="Checkpoint to continue training from "
                   "(weights + config + tokenizer); used for the physics continued-pretrain")
    p.add_argument("--eval-text", default=None, help="Optional extra text file for OOD PPL (e.g. eval_heldout.jsonl converted)")
    p.add_argument("--out", default="runs/small")
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--max-bytes", type=int, default=0, help="Subsample corpus (0 = all)")
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-limbs", type=int, default=8)
    p.add_argument("--limb-taps", type=int, default=32)
    p.add_argument("--no-limbs", action="store_true")
    p.add_argument("--no-tetra-bias", action="store_true")
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--no-phi", action="store_true", help="Use plain (rational) torus winding")
    p.add_argument("--leak", action="store_true", help="NON-CAUSAL attention (upstream bug; ablation only)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--lambda-wm", type=float, default=0.05)
    p.add_argument("--lambda-stability", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate_ppl(model, chunks, batch_size=16, max_batches=64, device="cpu") -> float:
    model.eval()
    total, count = 0.0, 0
    n = min(len(chunks), max_batches * batch_size)
    for i in range(0, n, batch_size):
        xb = chunks[i : i + batch_size].to(device)
        _, _, loss, _ = model(xb, targets=xb)
        tok = xb.numel()
        total += loss.item() * tok
        count += tok
    model.train()
    return math.exp(min(total / max(count, 1), 20))


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_paths = [p.strip() for p in args.data.split(",") if p.strip()]
    print(f"[data] reading {data_paths}")
    texts = read_jsonl_texts(data_paths, max_bytes=args.max_bytes or None)
    n_chars = sum(len(t) for t in texts)
    print(f"[data] {len(texts)} documents, {n_chars/1e6:.1f} MB of text")

    tok_path = out_dir / "tokenizer.json"
    init_ckpt = None
    if args.init:
        init_ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        src_tok = ByteBPETokenizer.load(init_ckpt["tokenizer"])
        src_tok.save(tok_path)
        tok = src_tok
        print(f"[init] continuing from {args.init} "
              f"(step {init_ckpt.get('step')}, PPL {init_ckpt.get('val_ppl')}) "
              f"| tokenizer {tok.vocab_size}-vocab carried over")
    elif tok_path.exists():
        tok = ByteBPETokenizer.load(tok_path)
        print(f"[tokenizer] loaded existing {tok.vocab_size}-vocab from {tok_path}")
    else:
        print(f"[tokenizer] training byte-BPE (vocab<={args.vocab}) ...")
        t0 = time.time()
        tok = ByteBPETokenizer.train(texts, vocab_size=args.vocab)
        tok.save(tok_path)
        print(f"[tokenizer] {tok.vocab_size} merges done in {time.time()-t0:.1f}s -> {tok_path}")

    stream = texts_to_token_stream(texts, tok)
    chunks = chunk_stream(stream, args.ctx)
    train_chunks, val_chunks = train_val_split(chunks, args.val_fraction, seed=args.seed)
    print(f"[data] {len(stream):,} tokens -> {len(chunks):,} chunks of {args.ctx} "
          f"(train {len(train_chunks):,}, val {len(val_chunks):,})")

    if init_ckpt is not None:
        cfg = GMConfig.from_dict(init_ckpt["config"])
        print(f"[init] architecture from checkpoint: d_model={cfg.d_model} "
              f"n_layer={cfg.n_layer} ctx={cfg.ctx} vocab={cfg.vocab_size}")
    else:
        cfg = GMConfig(
            vocab_size=tok.vocab_size, d_model=args.d_model, n_head=args.n_head,
            n_layer=args.n_layer, ctx=args.ctx, dropout=args.dropout,
            n_limbs=args.n_limbs, limb_taps=args.limb_taps,
            use_tetra_bias=not args.no_tetra_bias, use_torus_rope=not args.no_rope,
            use_phi_winding=not args.no_phi, use_limbs=not args.no_limbs,
            causal=not args.leak,
        )
    model = GMGPT(cfg).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model"])
        print("[init] weights loaded (continued pretrain)")
    criterion = CompositeLoss(
        cfg.d_model, lambda_wm=args.lambda_wm, lambda_stability=args.lambda_stability,
        label_smoothing=args.label_smoothing,
    ).to(device)
    n_params = model.num_params() + sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    print(f"[model] {n_params/1e6:.2f}M params | causal={cfg.causal} limbs={cfg.use_limbs} "
          f"tetra_bias={cfg.use_tetra_bias} torus_rope={cfg.use_torus_rope} phi={cfg.use_phi_winding}")

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    batcher = ChunkBatcher(train_chunks, args.batch, seed=args.seed)

    history = {"steps": [], "val": []}
    best_val = float("inf")
    t_start = time.time()
    tokens_per_step = args.batch * args.ctx
    model.train()

    for step in range(args.steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        xb = batcher.sample().to(device)
        logits, hidden, _, _ = model(xb)
        terms = criterion(logits[:, :-1].contiguous(), hidden[:, :-1].contiguous(), xb[:, 1:].contiguous())
        opt.zero_grad(set_to_none=True)
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0
        )
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            dt = time.time() - t_start
            tps = tokens_per_step * (10 if step else 1) / max(dt, 1e-9)
            row = {
                "step": step,
                "lr": round(lr, 6),
                "lm": round(terms["lm"].item(), 4),
                "total": round(terms["total"].item(), 4),
                "ppl": round(math.exp(min(terms["lm"].item(), 12)), 2),
                "tok_s": int(tps),
            }
            if "world_model" in terms:
                row["wm"] = round(terms["world_model"].item(), 5)
            history["steps"].append(row)
            print(f"  step {step:5d}/{args.steps} lm={row['lm']:.4f} ppl={row['ppl']:.2f} "
                  f"lr={lr:.1e} {row['tok_s']} tok/s" + (f" wm={row.get('wm', 0):.5f}" if "wm" in row else ""),
                  flush=True)
            t_start = time.time()

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            val_ppl = evaluate_ppl(model, val_chunks, device=device)
            history["val"].append({"step": step, "val_ppl": round(val_ppl, 3)})
            print(f"  [eval] step {step} held-out causal PPL = {val_ppl:.3f}")
            if val_ppl < best_val:
                best_val = val_ppl
                torch.save({
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "config": cfg.to_dict(),
                    "step": step, "val_ppl": val_ppl,
                    "tokenizer": str(tok_path),
                }, out_dir / "best.pt")
                print(f"  [eval] new best -> {out_dir/'best.pt'}")

        if (step + 1) % args.sample_every == 0 or step == args.steps - 1:
            prompt = "the "
            ids = torch.tensor([tok.encode(prompt) or [ord("t")]], dtype=torch.long)
            sample = model.generate(ids, max_new_tokens=96, temperature=0.8, top_k=40)
            print(f"  [sample] {tok.decode(sample[0].tolist())!r}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({
        "model": model.state_dict(), "criterion": criterion.state_dict(),
        "config": cfg.to_dict(), "step": args.steps, "val_ppl": best_val,
        "tokenizer": str(tok_path),
    }, out_dir / "final.pt")
    print(f"\n[done] best held-out causal PPL = {best_val:.3f} | "
          f"checkpoints in {out_dir}/ (best.pt, final.pt, history.json)")


if __name__ == "__main__":
    main()
