#!/usr/bin/env python3
"""Head-to-head: subword-marker SFT vs atomic-marker SFT.

Answers the question the atomic-marker retrain was built to answer:

    Is the end-of-reply marker now reachable, and can the model end a reply
    on its own (rather than being cut off by a token budget)?

For each checkpoint it reports, on the same fixed prompts:

  * ``E rank``    -- where ``<|end|>`` sits in the next-token ranking
                     (subword checkpoint: rank of its first *piece*, '<')
  * ``E logit``   -- its raw logit (competitive vs deeply negative)
  * ``stopped``   -- did greedy decoding reach the stop token on its own?
  * ``tokens``    -- tokens generated before stopping or hitting the budget
  * the reply text, verbatim

Also repeats the measurement across several prompts and reports how often the
stop marker wins the argmax anywhere in the reply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import (DEFAULT_SYSTEM, E, MARKERS, STOP_MARKERS, render_prompt,
                        strip_stop)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

PROMPTS = [
    "What is the capital of France?",
    "Give me one tip for learning a new language.",
    "Hello, how are you today?",
    "Why is the sky blue?",
    "Write one sentence about the ocean.",
    "What is a black hole?",
]


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt


@torch.no_grad()
def probe(model, tok, q, budget=96):
    """Marker reachability + greedy stop behaviour for one prompt."""
    prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": q}])
    ids = tok.encode(prompt)
    logits = model(torch.tensor([ids]))[0][:, -1, :]

    atomic = bool(tok.specials)
    if atomic:
        eid = tok.id_for_special(E)
    else:
        eid = tok.encode(E)[0]              # first piece of the subword marker
    rank = int((logits[0] > logits[0, eid]).sum())
    e_logit = float(logits[0, eid])

    # greedy decode
    idx = torch.tensor([ids])
    past = None
    gen = []
    stopped_at = None
    for step in range(budget):
        si = idx if past is None else idx[:, -1:]
        lg, _, _, past = model(si, use_cache=True, past=past)
        nxt = int(lg[0, -1].argmax())
        gen.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
        if atomic and nxt == tok.id_for_special(E):
            stopped_at = step
            break

    reply = strip_stop(tok.decode(gen)).strip()
    return {"q": q, "e_rank": rank, "e_logit": round(e_logit, 3),
            "argmax_is_e": int(logits[0].argmax()) == eid,
            "stopped": stopped_at is not None, "stop_step": stopped_at,
            "tokens": len(gen), "reply": reply}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="subword=runs/chat/best.pt,"
                                            "atomic=runs/chat_atomic/best.pt")
    p.add_argument("--out", default="runs/atomic_markers.json")
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    report = {"checkpoints": {}, "prompts": PROMPTS}
    for spec in args.checkpoints.split(","):
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"[skip] {label}: {path} not found")
            continue
        model, tok, ckpt = load(path)
        rows = [probe(model, tok, q) for q in PROMPTS]
        stopped = sum(r["stopped"] for r in rows)
        ranks = [r["e_rank"] for r in rows]
        entry = {
            "path": path,
            "step": ckpt.get("step"),
            "vocab": tok.vocab_size,
            "atomic_markers": bool(tok.specials),
            "params": sum(x.numel() for x in model.parameters()),
            "stopped_on_own": stopped,
            "n_prompts": len(rows),
            "e_rank_min": min(ranks), "e_rank_max": max(ranks),
            "rows": rows,
        }
        report["checkpoints"][label] = entry
        print(f"\n=== {label} ({path}) vocab={tok.vocab_size} "
              f"atomic={entry['atomic_markers']} ===")
        print(f"    stopped on its own: {stopped}/{len(rows)} prompts")
        print(f"    <|end|> rank range: {min(ranks)}..{max(ranks)}")
        for r in rows:
            print(f"\n  Q: {r['q']}")
            print(f"     reply: {r['reply'][:150]!r}")
            print(f"     [E rank {r['e_rank']}, E logit {r['e_logit']}, "
                  f"stopped={r['stopped']}, tokens={r['tokens']}]")

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
