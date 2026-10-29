"""Merge Dolly-15k with physics QA into one SFT file.

The physics rows are few (378) next to Dolly's 15011, so they'd be ~2.5% of the
mix and barely move behaviour.  We upsample them (default 5x) to ~11% so the
re-SFT actually gets told what a physics answer looks like -- and we keep the
raw copies so the ablation (Dolly-only vs Dolly+physics) stays reproducible.
"""
import argparse
import json
import random
from pathlib import Path

T = Path.home() / "AppData/Local/Temp/opencode"


def rows(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolly", default=str(T / "dolly.jsonl"))
    ap.add_argument("--physics", default=str(T / "physics_qa.jsonl"))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=str(T / "sft_physics_mix.jsonl"))
    args = ap.parse_args()

    d = rows(Path(args.dolly))
    p = rows(Path(args.physics))
    mixed = d + p * args.repeat
    random.Random(0).shuffle(mixed)
    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mixed),
                   encoding="utf-8")
    print(f"dolly {len(d)} + physics {len(p)}x{args.repeat}={len(p)*args.repeat}"
          f" -> {len(mixed)} rows ({out.name}) "
          f"physics share {100*len(p)*args.repeat/len(mixed):.1f}%")


if __name__ == "__main__":
    main()
