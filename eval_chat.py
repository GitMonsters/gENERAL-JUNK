#!/usr/bin/env python3
"""Honest chat evaluation: GitMonsters' own QA pairs vs Dolly-15k, measured and sampled.

Two things are reported:

1. **Measured** -- assistant-token perplexity on a held-out split of
   GitMonsters' own instruction corpus (``data/instructions*.jsonl``) and on
   held-out Dolly-15k rows, for any number of checkpoints.  Lower is better.
   Each corpus is also given a *frequency baseline*: predicting the globally
   most common token at every answer position.

2. **Qualitative** -- greedy + sampled replies to (a) questions drawn from the
   upstream corpus and (b) generic questions nobody trained on, so the reader
   can see for themselves whether the model answers or free-associates.

    python eval_chat.py --checkpoints runs/small/best.pt,runs/chat/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import (DEFAULT_SYSTEM, load_dolly_pairs, load_qa_pairs,
                        render_prompt, render_training_example, split_prompt_and_answer,
                        strip_stop)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

IGNORE = -100
UPSTREAM = Path.home() / "AppData/Local/Temp/opencode/gitmonsters/octotetrahedral-agi"

PROBES = [
    ("generic", "What is the capital of France?"),
    ("generic", "Give me one tip for learning a new language."),
    ("generic", "Hello, how are you today?"),
    ("generic", "Why is the sky blue?"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True,
                   help="comma-separated label=path pairs or bare paths")
    p.add_argument("--dolly", default=str(Path.home() /
                   "AppData/Local/Temp/opencode/dolly.jsonl"))
    p.add_argument("--ctx", type=int, default=320)
    p.add_argument("--n", type=int, default=500, help="examples per eval corpus")
    p.add_argument("--samples", type=int, default=3, help="replies per probe")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="runs/chat_eval.json")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def to_rows(pairs, tok, ctx, with_context: bool = False):
    """[(q, a)] or [(q, a, c)] -> [(ids, labels)] with answer-only labels."""
    rows = []
    for item in pairs:
        q, a = item[0], item[1]
        c = item[2] if with_context and len(item) > 2 else None
        full = render_training_example(q, a, c)
        prompt, answer = split_prompt_and_answer(full)
        p_ids, a_ids = tok.encode(prompt), tok.encode(answer)
        if len(a_ids) < 3 or len(p_ids) + len(a_ids) > ctx:
            continue
        rows.append((p_ids + a_ids, [IGNORE] * len(p_ids) + a_ids))
    return rows


def collate(batch, pad_id=0):
    T = max(len(ids) for ids, _ in batch)
    idx = torch.full((len(batch), T), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        idx[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
    return idx, labels


@torch.no_grad()
def answer_ppl(model, rows, batch=8, max_batches=64):
    total, count = 0.0, 0
    for i in range(0, min(len(rows), batch * max_batches), batch):
        idx, labels = collate(rows[i:i + batch])
        logits, _, _, _ = model(idx)
        n = int((labels[:, 1:] != IGNORE).sum())
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1), ignore_index=IGNORE)
        total += loss.item() * n
        count += n
    return math.exp(min(total / max(count, 1), 20)), count


@torch.no_grad()
def unigram_baseline(train_rows, eval_rows):
    """PPL of always predicting the most frequent answer token."""
    counts = Counter()
    for _, lab in train_rows:
        counts.update(t for t in lab if t != IGNORE)
    top = counts.most_common(1)[0][0]
    total, count = 0.0, 0
    for _, lab in eval_rows:
        for t in lab[1:]:
            if t == IGNORE:
                continue
            total += 0.0 if t == top else 20.0     # clamp: never matches
            count += 1
    return math.exp(min(total / max(count, 1), 20))


@torch.no_grad()
def reply(model, tok, question, max_new=60, temperature=0.0, top_k=40):
    msg = [{"role": "system", "content": DEFAULT_SYSTEM},
           {"role": "user", "content": question}]
    prompt = render_prompt(msg)
    ids = torch.tensor([tok.encode(prompt)])
    gen_ids = ids
    past = None
    out_ids = ids[0].tolist()
    for _ in range(max_new):
        step_in = gen_ids if past is None else gen_ids[:, -1:]
        logits, _, _, past = model(step_in, use_cache=True, past=past)
        logits = logits[:, -1, :].float()
        if temperature <= 1e-6:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        gen_ids = torch.cat([gen_ids, nxt], dim=1)
        out_ids.append(int(nxt.item()))
        text = tok.decode(out_ids[len(ids[0]):])
        if strip_stop(text) != text:
            break
    text = strip_stop(tok.decode(out_ids[len(ids[0]):]))
    return text.strip()


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)

    specs = []
    for part in args.checkpoints.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            label, path = part.split("=", 1)
        else:
            label, path = Path(part).parent.name, part
        specs.append((label, path))

    dolly = load_dolly_pairs(args.dolly)
    random.shuffle(dolly)
    dolly_train, dolly_eval = dolly[:200], dolly[200:200 + args.n]
    gm_pairs = load_qa_pairs(sorted(UPSTREAM.glob("data/instructions*.jsonl")))
    random.shuffle(gm_pairs)
    gm_train, gm_eval = gm_pairs[:200], gm_pairs[200:200 + args.n]
    print(f"[data] dolly {len(dolly)} rows | upstream QA {len(gm_pairs)} rows")

    report = {"checkpoints": {}, "probes": PROBES, "config": vars(args)}
    for label, path in specs:
        model, tok, ckpt = load_model(path)
        d_rows = to_rows(dolly_eval, tok, args.ctx, with_context=True)
        g_rows = to_rows(gm_eval, tok, args.ctx)
        d_tr = to_rows(dolly_train, tok, args.ctx, with_context=True)
        g_tr = to_rows(gm_train, tok, args.ctx)
        d_ppl, d_tok = answer_ppl(model, d_rows)
        g_ppl, g_tok = answer_ppl(model, g_rows)
        entry = {
            "path": path,
            "step": ckpt.get("step"),
            "params": sum(p.numel() for p in model.parameters()),
            "dolly_answer_ppl": round(d_ppl, 2),
            "dolly_answer_tokens": d_tok,
            "upstream_answer_ppl": round(g_ppl, 2),
            "upstream_answer_tokens": g_tok,
        }
        print(f"[{label}] step {entry['step']} | dolly PPL {d_ppl:.1f} "
              f"({d_tok} ans tokens) | upstream QA PPL {g_ppl:.1f} ({g_tok} ans tokens)")

        if label == specs[-1][0]:
            entry["unigram_baseline"] = {
                "dolly": round(unigram_baseline(d_tr, d_rows), 2),
                "upstream": round(unigram_baseline(g_tr, g_rows), 2),
            }

        samples = []
        for kind, q in PROBES:
            row = {"kind": kind, "q": q, "replies": []}
            for temp in (0.0, 0.7, 0.9):
                for _ in range(args.samples if temp else 1):
                    row["replies"].append({"temp": temp,
                                           "text": reply(model, tok, q, temperature=temp)})
            samples.append(row)
        upstream_qs = random.sample(gm_eval, 3)
        for q, a in upstream_qs:
            row = {"kind": "upstream", "q": q, "reference": a, "replies": []}
            row["replies"].append({"temp": 0.0, "text": reply(model, tok, q, temperature=0.0)})
            row["replies"].append({"temp": 0.8, "text": reply(model, tok, q, temperature=0.8)})
            samples.append(row)
        entry["samples"] = samples
        report["checkpoints"][label] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out}")

    for label, entry in report["checkpoints"].items():
        print(f"\n=== {label} ({entry['path']}) ===")
        for row in entry["samples"]:
            print(f"\n[{row['kind']}] Q: {row['q']}")
            if row.get("reference"):
                print(f"    upstream reference: {row['reference'][:140]}")
            for r in row["replies"]:
                print(f"    T={r['temp']}: {r['text'][:180]!r}")


if __name__ == "__main__":
    main()
