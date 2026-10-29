#!/usr/bin/env python3
"""How often does the *sampled* model emit a stop marker on its own?

Greedy decoding never reaches <|end|> (probe_atomic.py).  But the shipped server
samples (temp 0.7, top-k 40, top-p 0.95, repetition penalty), and there the
atomic marker is a single reachable id with logit ~ -1.2.  This measures the
practical consequence: under real serving conditions, does the model ever end
its own reply -- and if so, after how many tokens, and does the text look like
anything?

Reported per checkpoint, over N seeds per prompt, using the server's own
stream_reply so the measurement is exactly what a user sees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import chat_server as cs
from gmllm.chat import DEFAULT_SYSTEM, render_prompt

PROMPTS = [
    "What is the capital of France?",
    "Give me one tip for learning a new language.",
    "Hello, how are you today?",
    "Why is the sky blue?",
    "Write one sentence about the ocean.",
    "What is a black hole?",
]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def run(ckpt_path: str) -> dict:
    cs.load_checkpoint(ckpt_path)
    torch.manual_seed(0)
    rows = []
    for q in PROMPTS:
        prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                                {"role": "user", "content": q}])
        for seed in SEEDS:
            torch.manual_seed(seed)
            text, n, finished = "", 0, False
            for delta, n, finished in cs.stream_reply(prompt, temperature=0.7,
                                                      top_k=40, top_p=0.95,
                                                      max_tokens=64):
                text += delta
            rows.append({"q": q, "seed": seed, "finished": bool(finished),
                         "tokens": n, "reply": text.strip()})
    stops = sum(r["finished"] for r in rows)
    return {"checkpoint": ckpt_path, "atomic": cs.STATE["atomic"],
            "trials": len(rows), "stopped": stops,
            "stop_rate": round(stops / len(rows), 3), "rows": rows}


def main():
    ckpts = sys.argv[1:] or ["runs/chat/best.pt", "runs/chat_atomic/best.pt"]
    out = {}
    for c in ckpts:
        if not Path(c).exists():
            print(f"[skip] {c}")
            continue
        r = run(c)
        out[Path(c).parent.name] = r
        print(f"\n=== {c} (atomic={r['atomic']}) ===")
        print(f"    stopped on its own: {r['stopped']}/{r['trials']} "
              f"({r['stop_rate']*100:.1f}%) sampled replies")
        for x in r["rows"][:8]:
            print(f"    seed {x['seed']} finish={'stop' if x['finished'] else 'len '} "
                  f"n={x['tokens']:>2}  {x['reply'][:70]!r}")
    Path("runs/stop_rate.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[done] wrote runs/stop_rate.json")


if __name__ == "__main__":
    main()
