#!/usr/bin/env python3
"""Interactive generation from a gmllm checkpoint.

    python generate.py --checkpoint runs/small/best.pt --prompt "the " --tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="the ")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--rep-penalty", type=float, default=1.1)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(GMConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = torch.tensor([tok.encode(args.prompt) or [ord("a")]], dtype=torch.long)
    out = model.generate(
        ids, max_new_tokens=args.tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep_penalty,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
