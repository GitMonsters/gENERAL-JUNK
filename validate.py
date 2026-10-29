#!/usr/bin/env python3
"""Validate the trained checkpoint end-to-end.

Checks (all against runs/small/best.pt, the actual trained weights):
  1. checkpoint integrity      -- config/state-dict load cleanly, param count matches
  2. tokenizer losslessness    -- encode/decode round-trip on adversarial Unicode
  3. causality on real weights -- perturbing the future must not change past logits
  4. KV-cache exactness        -- incremental decoding == full forward (real model)
  5. leak mode still leaks     -- causal=False on the same weights shows the defect
  6. held-out PPL reproduces   -- number matches what training reported
Exit code 0 = all validations pass.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

# Windows console defaults to cp1252; force UTF-8 so failures print readably
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.data import chunk_stream, read_jsonl_texts, texts_to_token_stream, train_val_split
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

CKPT = Path("runs/small/best.pt")
CORPUS = os.path.join(os.environ["TEMP"], "opencode", "gitmonsters",
                      "octotetrahedral-agi", "data", "combined_train.jsonl")

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    print("== 1. checkpoint integrity ==")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = GMConfig.from_dict(ckpt["config"])
    tok = ByteBPETokenizer.load(ckpt["tokenizer"])
    model = GMGPT(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    check("state dict loads", not missing and not unexpected,
          f"missing={missing} unexpected={unexpected}")
    check("config flags sane", cfg.causal and cfg.use_tetra_bias and cfg.use_torus_rope,
          f"causal={cfg.causal} tetra={cfg.use_tetra_bias} rope={cfg.use_torus_rope} "
          f"limbs={cfg.use_limbs} phi={cfg.use_phi_winding}")
    n = model.num_params()
    check("param count", 3_000_000 < n < 4_000_000, f"{n:,} params (expect ~3.3M)")
    check("checkpoint provenance", ckpt.get("step") == 99,
          f"step={ckpt.get('step')} val_ppl={ckpt.get('val_ppl'):.3f}")
    model.eval()

    print("== 2. tokenizer losslessness ==")
    for s in ["plain ascii text", "ünïcödé — em-dash …", "中文 tabs\tand\nnewlines",
              "emoji 🚀 and rare bytes ÿþ", " " * 10 + " leading/trailing "]:
        ids = tok.encode(s)
        got = tok.decode(ids)
        ok = got == s and all(0 <= i < tok.vocab_size for i in ids)
        if not ok:
            check(f"round-trip {s[:24]!r}", False,
                  f"decoded={got[:60]!r} ids_ok={all(0 <= i < tok.vocab_size for i in ids)}")
            break
    else:
        check("round-trip on 5 adversarial strings", True,
              f"vocab={tok.vocab_size}, zero <unk> possible by construction")

    print("== 3. causality on trained weights ==")
    torch.manual_seed(0)
    x = torch.randint(0, tok.vocab_size, (1, 64))
    with torch.no_grad():
        la, _, _, _ = model(x)
        x2 = x.clone()
        x2[:, 40:] = torch.randint(0, tok.vocab_size, (1, 24))
        lb, _, _, _ = model(x2)
    past_same = torch.allclose(la[:, :40], lb[:, :40], atol=1e-5)
    future_diff = not torch.allclose(la[:, 40:], lb[:, 40:], atol=1e-5)
    check("past logits invariant to future tokens", past_same)
    check("perturbation actually mattered", future_diff)

    print("== 4. KV-cache exactness on trained weights ==")
    with torch.no_grad():
        full, _, _, _ = model(x)
        past, steps = None, []
        for t in range(x.size(1)):
            inp = x[:, :1] if past is None else x[:, t:t + 1]
            lg, _, _, past = model(inp, use_cache=True, past=past)
            steps.append(lg[:, -1])
        inc = torch.stack(steps, dim=1)
    max_err = (full - inc).abs().max().item()
    check("incremental == full forward", max_err < 1e-4, f"max abs err = {max_err:.2e}")

    print("== 5. leak mode (upstream defect) still reproduces on these weights ==")
    for b in model.blocks:
        b.attn.causal = False
    with torch.no_grad():
        ll, _, _, _ = model(x)
    for b in model.blocks:
        b.attn.causal = True
    leak = (ll[:, :40] - la[:, :40]).abs().max().item()
    check("non-causal mode changes past logits (the defect)", leak > 1e-3,
          f"max shift = {leak:.4f} nats -- upstream training had this at every step")

    print("== 6. held-out PPL reproduces ==")
    texts = read_jsonl_texts([CORPUS])
    stream = texts_to_token_stream(texts, tok)
    chunks = chunk_stream(stream, cfg.ctx)
    _, val = train_val_split(chunks, 0.02, seed=1337)
    total = count = 0.0
    with torch.no_grad():
        for i in range(0, len(val), 16):
            xb = val[i:i + 16]
            _, _, loss, _ = model(xb, targets=xb)
            total += loss.item() * xb.numel()
            count += xb.numel()
    ppl = math.exp(min(total / count, 20))
    check("PPL matches training report", abs(ppl - ckpt["val_ppl"]) < 1.0,
          f"recomputed {ppl:.3f} vs reported {ckpt['val_ppl']:.3f}")

    print("== 7. generation smoke ==")
    ids = torch.tensor([tok.encode("the ") or [ord("t")]], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=64, temperature=0.8, top_k=40)
    ok = out.shape == (1, ids.size(1) + 64) and out.min() >= 0 and out.max() < tok.vocab_size
    check("generates 64 tokens in-range", ok, f"shape={tuple(out.shape)}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} validations passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
