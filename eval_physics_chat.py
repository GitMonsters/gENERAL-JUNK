#!/usr/bin/env python3
"""Does the model answer *physics* questions?  Measured, not assumed.

Companion to eval_chat.py, but the held-out corpus is physics: the extractive
QA pairs built from the Simple-English-Wikipedia physics slice, rendered with
the same chat template the SFT used.  Reports assistant-token perplexity plus
verbatim replies to a fixed set of physics questions, so the reader can judge
whether continued-pretraining on physics moved anything.

    python eval_physics_chat.py --checkpoints "base=runs/chat/best.pt,p2=.../best.pt"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import (DEFAULT_SYSTEM, render_prompt, render_training_example,
                        split_prompt_and_answer, strip_stop)
from eval_chat import answer_ppl, collate, load_model, to_rows
from gmllm.tokenizer import ByteBPETokenizer

T = Path.home() / "AppData/Local/Temp/opencode"

PHYS_PROBES = [
    "What is force?",
    "What is energy?",
    "Why is the sky blue?",
    "What is gravity?",
    "What is an atom?",
    "What is acceleration?",
    "What is temperature?",
    "How does a magnet work?",
]


def reply(model, tok, question, max_new=64, temperature=0.0, top_k=40):
    """Greedy/sampled reply, independent of eval_chat's copy (kept in sync)."""
    import torch.nn.functional as F

    msg = [{"role": "system", "content": DEFAULT_SYSTEM},
           {"role": "user", "content": question}]
    ids = torch.tensor([tok.encode(render_prompt(msg))])
    out_ids = ids[0].tolist()
    gen = ids
    past = None
    for _ in range(max_new):
        step_in = gen if past is None else gen[:, -1:]
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
        gen = torch.cat([gen, nxt], dim=1)
        out_ids.append(int(nxt.item()))
        text = tok.decode(out_ids[len(ids[0]):])
        if strip_stop(text) != text:
            break
    return strip_stop(tok.decode(out_ids[len(ids[0]):])).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--qa", default=str(T / "physics_qa.jsonl"))
    ap.add_argument("--ctx", type=int, default=320)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/physics_chat_eval.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)

    pairs = []
    for line in Path(args.qa).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            pairs.append((o["instruction"], o["response"],
                          o.get("context") or None))
    random.shuffle(pairs)
    train_p, eval_p = pairs[:100], pairs[100:100 + args.n]
    print(f"[data] physics QA {len(pairs)} rows | eval on {len(eval_p)}")

    report = {"checkpoints": {}, "probes": PHYS_PROBES}
    for spec in args.checkpoints.split(","):
        spec = spec.strip()
        if not spec:
            continue
        label, path = spec.split("=", 1) if "=" in spec else (
            Path(spec).parent.name, spec)
        model, tok, ckpt = load_model(path)
        rows = to_rows(eval_p, tok, args.ctx, with_context=True)
        tr = to_rows(train_p, tok, args.ctx, with_context=True)
        ppl, ntok = answer_ppl(model, rows)
        entry = {"path": path, "step": ckpt.get("step"),
                 "physics_answer_ppl": round(ppl, 2), "answer_tokens": ntok}
        print(f"[{label}] step {entry['step']} | physics answer PPL {ppl:.1f} "
              f"({ntok} tokens)")

        samples = []
        for q in PHYS_PROBES:
            reps = [{"temp": 0.0, "text": reply(model, tok, q, temperature=0.0)}]
            for _ in range(args.samples):
                reps.append({"temp": 0.7,
                             "text": reply(model, tok, q, temperature=0.7)})
            samples.append({"q": q, "replies": reps})
        entry["samples"] = samples
        report["checkpoints"][label] = entry

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[done] wrote {args.out}")
    for label, entry in report["checkpoints"].items():
        print(f"\n=== {label} ({entry['path']}) ===")
        for row in entry["samples"]:
            print(f"\nQ: {row['q']}")
            for r in row["replies"]:
                print(f"  T={r['temp']}: {r['text'][:180]!r}")


if __name__ == "__main__":
    main()
