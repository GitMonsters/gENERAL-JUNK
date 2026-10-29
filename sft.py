#!/usr/bin/env python3
"""Instruction-tune (SFT) GM-GPT into a chat model.

Starts from a base checkpoint, trains on Dolly-15k rendered with the chat
template from ``gmllm/chat.py``, and supervises **only assistant tokens**
(prompt tokens are masked out of the loss).  Saves a chat checkpoint plus a
console transcript at several points, so quality is visible, not assumed.

    python sft.py --base runs/small/best.pt --data "%TEMP%/opencode/dolly.jsonl" \
        --out runs/chat --steps 450 --batch 8 --ctx 320
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import (DEFAULT_SYSTEM, E, MARKERS, render_prompt,
                        render_training_example, split_prompt_and_answer,
                        strip_stop)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="runs/small/best.pt")
    p.add_argument("--data", required=True, help="Dolly-15k JSONL")
    p.add_argument("--out", default="runs/chat")
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ctx", type=int, default=320)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-size", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=150)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--atomic-markers", action="store_true",
                   help="add <|system|>/<|user|>/<|assistant|>/<|end|> as single "
                        "tokens (fixes the unreachable-stop-marker ceiling)")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="do not train tok_emb/lm_head (only sensible with "
                        "--atomic-markers; keeps the base LM's byte knowledge)")
    p.add_argument("--lm-data", default=None,
                   help="JSONL of {'text': ...} prose mixed into the objective as a "
                        "plain next-token LM loss, so SFT cannot discard the base "
                        "LM's knowledge (the physics-preservation fix; see RESULTS.md §3d)")
    p.add_argument("--lm-weight", type=float, default=0.5,
                   help="weight of the auxiliary LM loss when --lm-data is given")
    p.add_argument("--lm-batch", type=int, default=4,
                   help="number of prose chunks per step for the auxiliary LM loss")
    p.add_argument("--lm-ctx", type=int, default=256,
                   help="context length of the auxiliary LM chunks")
    return p.parse_args()


def build_tokenizer(base_tok_path: str, atomic: bool) -> tuple[ByteBPETokenizer, int]:
    """Load the base tokenizer, optionally appending atomic control markers."""
    tok = ByteBPETokenizer.load(base_tok_path)
    if atomic:
        tok.add_special_tokens(list(MARKERS))
    return tok, tok.vocab_size


def load_examples(path: str, tok: ByteBPETokenizer, ctx: int):
    """Tokenize Dolly rows into (ids, labels) with assistant-only labels."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ins = obj.get("instruction") or ""
            resp = obj.get("response") or ""
            ctx_text = obj.get("context") or ""
            if not ins.strip() or not resp.strip():
                continue
            full = render_training_example(ins, resp, ctx_text)
            prompt, answer = split_prompt_and_answer(full)
            p_ids = tok.encode(prompt)
            a_ids = tok.encode(answer)
            if len(a_ids) < 3 or len(p_ids) + len(a_ids) > ctx:
                continue
            ids = p_ids + a_ids
            labels = [IGNORE] * len(p_ids) + a_ids
            rows.append((ids, labels))
    return rows


def collate(batch, pad_id: int = 0):
    T = max(len(ids) for ids, _ in batch)
    B = len(batch)
    idx = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE, dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        idx[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return idx, labels


def load_lm_chunks(path: str, tok, ctx: int, max_docs: int = 100000):
    """Pack a prose JSONL into fixed-length next-token chunks (labels = inputs).

    Used for the auxiliary LM loss.  Pads the final partial chunk and ignores
    no positions -- this is plain language modelling, exactly what the base LM
    was trained on, so the gradient points the same way it always did.
    """
    from gmllm.data import texts_to_token_stream

    stream = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text") or obj.get("content") or ""
            if t:
                stream.extend(tok.encode(t))
                stream.append(tok.encode("\n")[0])
            if len(stream) > max_docs * ctx:
                break
    chunks = [stream[i:i + ctx] for i in range(0, len(stream) - ctx, ctx)]
    chunks = [c for c in chunks if len(c) == ctx]
    return torch.tensor(chunks, dtype=torch.long) if chunks else torch.empty(0, ctx, dtype=torch.long)


def masked_loss(model, idx, labels):
    logits, _, _, _ = model(idx)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
    )


def lm_loss(model, chunks):
    """Plain causal next-token loss on a batch of prose chunks."""
    xb = chunks
    logits, _, loss, _ = model(xb, targets=xb)
    return loss


@torch.no_grad()
def evaluate(model, rows, batch=8, max_batches=32):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, min(len(rows), batch * max_batches), batch):
        idx, labels = collate(rows[i:i + batch])
        logits, _, _, _ = model(idx)
        n = (labels[:, 1:] != IGNORE).sum().item()
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1), ignore_index=IGNORE,
        )
        total += loss.item() * n
        count += n
    model.train()
    return math.exp(min(total / max(count, 1), 20))


@torch.no_grad()
def chat_sample(model, tok, user: str, max_new: int = 80, temperature: float = 0.7):
    model.eval()
    prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": user}])
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    eos = tok.id_for_special(E)
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                         top_k=40, rep_penalty=1.05,
                         eos_id=(eos if eos >= 0 else None))
    gen = out[0].tolist()[len(ids[0]):]
    text = tok.decode(gen)
    return strip_stop(text).strip()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])

    if args.atomic_markers:
        tok.add_special_tokens(list(MARKERS))
        old_vocab = cfg.vocab_size
        model.resize_vocab(tok.vocab_size)
        print(f"[atomic] control markers are now single tokens: vocab "
              f"{old_vocab} -> {tok.vocab_size}, ids "
              f"{[tok.id_for_special(m) for m in MARKERS]}")

    if args.freeze_embeddings:
        model.tok_emb.weight.requires_grad_(False)
        print("[freeze] tok_emb/lm_head are frozen")

    # persist the (possibly enlarged) tokenizer next to the checkpoints, so the
    # saved model is self-contained and load() reconstructs the same ids
    tok_path_saved = out / "tokenizer.json"
    tok.save(tok_path_saved)
    print(f"[tokenizer] vocab {tok.vocab_size} -> {tok_path_saved}")

    print(f"[base] {args.base} step={ckpt.get('step')} ppl={ckpt.get('val_ppl')} "
          f"params={model.num_params()/1e6:.2f}M")

    rows = load_examples(args.data, tok, args.ctx)
    random.shuffle(rows)
    val = rows[: args.val_size]
    train = rows[args.val_size:]
    print(f"[data] {len(rows)} usable conversations (<= {args.ctx} tokens, answer>=3) "
          f"| train {len(train)} val {len(val)}")

    lm_chunks = None
    if args.lm_data:
        lm_chunks = load_lm_chunks(args.lm_data, tok, args.lm_ctx)
        print(f"[lm] auxiliary LM loss on {len(lm_chunks)} chunks of {args.lm_ctx} "
              f"from {args.lm_data} (weight {args.lm_weight}) -- physics-preservation fix")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    history = {"steps": [], "val": [], "samples": []}
    best = float("inf")
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = random.sample(train, min(args.batch, len(train)))
        idx, labels = collate(batch)
        loss = masked_loss(model, idx, labels)
        lm_val = None
        if lm_chunks is not None and len(lm_chunks):
            sel = torch.randint(0, len(lm_chunks), (min(args.lm_batch, len(lm_chunks)),))
            lm_val = lm_loss(model, lm_chunks[sel])
            loss = loss + args.lm_weight * lm_val
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            history["steps"].append({"step": step, "lr": round(lr, 6),
                                     "loss": round(loss.item(), 4),
                                     "lm": (round(lm_val.item(), 4)
                                            if lm_val is not None else None)})
            print(f"  step {step:4d}/{args.steps} loss={loss.item():.4f} "
                  f"ppl={math.exp(min(loss.item(), 12)):.2f} lr={lr:.1e} "
                  f"{('lm=' + format(lm_val.item(), '.4f') + ' ') if lm_val is not None else ''}"
                  f"({time.time()-t0:.0f}s)")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vp = evaluate(model, val)
            history["val"].append({"step": step, "val_ppl": round(vp, 3)})
            print(f"  [eval] step {step} answer-token PPL = {vp:.2f}")
            if vp < best:
                best = vp
                torch.save({"model": model.state_dict(), "criterion": None,
                            "config": cfg.to_dict(), "step": step, "val_ppl": vp,
                            "tokenizer": tok_path_saved}, out / "best.pt")
                print(f"  [eval] new best -> {out/'best.pt'}")

        if (step + 1) % args.sample_every == 0 or step == args.steps - 1:
            for q in ["What is the capital of France?",
                      "Give me one tip for learning a new language."]:
                a = chat_sample(model, tok, q)
                history["samples"].append({"step": step, "q": q, "a": a})
                print(f"  [sample] Q: {q}\n           A: {a[:160]!r}")

    torch.save({"model": model.state_dict(), "criterion": None,
                "config": cfg.to_dict(), "step": args.steps, "val_ppl": best,
                "tokenizer": tok_path_saved}, out / "final.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n[done] best answer-token PPL = {best:.2f} | checkpoints in {out}/")


if __name__ == "__main__":
    main()
