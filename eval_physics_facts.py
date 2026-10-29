#!/usr/bin/env python3
"""Physics *fact* probes — beyond perplexity.

Perplexity says "how unsurprised is the model by physics text"; it does not say
"does the model know F = ma".  This script asks both things explicitly:

  1. **Term recall** — for a set of physics relations/units, does greedy
     continuation contain the expected answer term?  (a crude but honest
     accuracy: it is exactly the kind of factual recall a world model needs.)
  2. **Unit completion** — "The SI unit of force is the ___" -> newton, etc.

Reported as a hit-rate per checkpoint, plus the verbatim completions, so a low
score is as visible as a high one.  No answer is hidden.

    python eval_physics_facts.py --checkpoints "base=runs/physics2/best.pt,sft=runs/chat_physics_lm/best.pt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

# (prompt, [acceptable answer terms, case-insensitive substring])
FACTS = [
    ("Newton's second law says force equals mass times ", ["acceleration"]),
    ("The SI unit of force is the ", ["newton"]),
    ("The SI unit of energy is the ", ["joule"]),
    ("The SI unit of power is the ", ["watt"]),
    ("The SI unit of electric current is the ", ["ampere", "amp"]),
    ("The speed of light in a vacuum is about ", ["300", "299", "3.0"]),
    ("The force that pulls objects toward the Earth is called ", ["gravity", "gravitation"]),
    ("Energy cannot be created or destroyed, only ", ["transferred", "converted", "changed", "transformed"]),
    ("The acceleration of a falling object due to gravity is about ", ["9.8", "10", "0.98"]),
    ("Water boils at ", ["100", "one hundred"]),
    ("Water freezes at ", ["0", "zero"]),
    ("The temperature at which water freezes in the Celsius scale is ", ["0", "zero"]),
    ("Light travels faster than ", ["sound"]),
    ("The Earth goes around the ", ["sun", "solar system", "galaxy"]),
    ("In physics, the product of mass and velocity is ", ["momentum"]),
    ("A push or a pull on an object is called a ", ["force"]),
    ("The resistance of a moving object to a change in motion is ", ["inertia"]),
    ("The energy of motion is called ", ["kinetic"]),
    ("The energy stored in an object due to its position is ", ["potential"]),
    ("The number of protons in an atom is its ", ["atomic number", "atomic"]),
]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ck["config"])
    tok = ByteBPETokenizer.load(ck["tokenizer"])
    model = GMGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, ck


@torch.no_grad()
def complete(model, tok, prompt, max_new=12, greedy=True, temperature=0.7):
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, max_new_tokens=max_new,
                         temperature=0.0 if greedy else temperature,
                         top_k=40, rep_penalty=1.0)
    gen = out[0].tolist()[len(ids[0]):]
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/physics_facts.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import json
    report = {"facts": [{"prompt": p, "accept": a} for p, a in FACTS], "checkpoints": {}}
    for spec in args.checkpoints.split(","):
        spec = spec.strip()
        if not spec:
            continue
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).parent.name, spec)
        model, tok, ck = load(path)
        hits, rows = 0, []
        for prompt, accept in FACTS:
            for mode in ("greedy", "sampled"):
                text = complete(model, tok, prompt, greedy=(mode == "greedy"))
                ok = any(a.lower() in text.lower() for a in accept)
                hits += int(ok)
                rows.append({"prompt": prompt, "mode": mode, "ok": ok,
                             "completion": text[:60]})
        total = len(FACTS) * 2
        report["checkpoints"][label] = {
            "path": path, "step": ck.get("step"),
            "hits": hits, "total": total, "accuracy": round(hits / total, 3),
            "rows": rows,
        }
        print(f"[{label}] step {ck.get('step')} | physics fact hit-rate "
              f"{hits}/{total} = {hits/total:.1%}")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"[done] wrote {args.out}\n")
    for label, e in report["checkpoints"].items():
        print(f"=== {label} ({e['accuracy']:.1%}) ===")
        for r in e["rows"]:
            if r["mode"] != "greedy":
                continue
            mark = "OK " if r["ok"] else "   "
            comp = r["completion"].encode("ascii", "replace").decode("ascii")
            print(f"  {mark}{r['prompt'][:52]:52s} -> {comp[:40]!r}")


if __name__ == "__main__":
    main()
