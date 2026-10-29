"""Tests for gmllm.

The two most important tests here are:
  * test_causality_invariance — logits at position i must not depend on future tokens
    (the defect in upstream octotetrahedral-agi that this build fixes), and
  * test_leak_mode_sees_the_future — with causal=False the leak DOES appear, proving
    the ablation reproduces the upstream bug rather than strawmanning it.
  * test_kv_cache_matches_full_forward — incremental decoding is exact.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmllm.data import chunk_stream, texts_to_token_stream
from gmllm.losses import CompositeLoss
from gmllm.model import GMConfig, GMGPT
from gmllm.modules import FractionalLimbMixer, TorusRoPE
from gmllm.tokenizer import ByteBPETokenizer


def tiny_cfg(**kw) -> GMConfig:
    base = dict(vocab_size=280, d_model=64, n_head=4, n_layer=2, ctx=64, dropout=0.0,
                n_limbs=4, limb_taps=16)
    base.update(kw)
    return GMConfig(**base)


# --------------------------------------------------------------------- tokenizer
def test_tokenizer_roundtrip():
    text = "the quick brown fox jumps over the lazy dog — twice!"
    tok = ByteBPETokenizer.train([text] * 50, vocab_size=384)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size <= 384 and tok.vocab_size >= 256


def test_tokenizer_never_unks():
    tok = ByteBPETokenizer.train(["common words here"], vocab_size=300)
    weird = "ÿ ⊗ 中文 emoji 🚀"
    ids = tok.encode(weird)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == weird


def test_tokenizer_save_load(tmp_path):
    tok = ByteBPETokenizer.train(["alpha beta gamma alpha beta"] * 20, vocab_size=300)
    tok.save(tmp_path / "t.json")
    tok2 = ByteBPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode("alpha beta") == tok.encode("alpha beta")


# ------------------------------------------------------------------ causality
def test_causality_invariance():
    torch.manual_seed(0)
    model = GMGPT(tiny_cfg())
    model.eval()
    x = torch.randint(0, 280, (1, 32))
    with torch.no_grad():
        logits_a, _, _, _ = model(x)
        x2 = x.clone()
        x2[:, 20:] = torch.randint(0, 280, (1, 12))  # perturb the future
        logits_b, _, _, _ = model(x2)
    # positions <= 20 must be identical
    assert torch.allclose(logits_a[:, :20], logits_b[:, :20], atol=1e-5), \
        "future tokens changed past logits — attention is not causal"
    # and the present/future must differ (sanity that the perturbation mattered)
    assert not torch.allclose(logits_a[:, 20:], logits_b[:, 20:], atol=1e-5)


def test_leak_mode_sees_the_future():
    """causal=False must reproduce the upstream defect (this is the ablation's premise)."""
    torch.manual_seed(0)
    model = GMGPT(tiny_cfg(causal=False))
    model.eval()
    x = torch.randint(0, 280, (1, 32))
    with torch.no_grad():
        logits_a, _, _, _ = model(x)
        x2 = x.clone()
        x2[:, 20:] = torch.randint(0, 280, (1, 12))
        logits_b, _, _, _ = model(x2)
    assert not torch.allclose(logits_a[:, :20], logits_b[:, :20], atol=1e-5), \
        "leak mode should let future tokens affect past logits"


# -------------------------------------------------------------------- kv cache
def test_kv_cache_matches_full_forward():
    torch.manual_seed(1)
    model = GMGPT(tiny_cfg())
    model.eval()
    x = torch.randint(0, 280, (2, 16))
    with torch.no_grad():
        full_logits, _, _, _ = model(x)
        past = None
        step_logits = []
        for t in range(x.size(1)):
            inp = x[:, :1] if past is None else x[:, t:t + 1]
            lg, _, _, past = model(inp, use_cache=True, past=past)
            step_logits.append(lg[:, -1])
    step_logits = torch.stack(step_logits, dim=1)
    assert torch.allclose(full_logits, step_logits, atol=1e-4), \
        "incremental (KV-cached) decoding differs from full forward pass"


# ------------------------------------------------------------------ components
def test_frac_kernel_matches_grunwald_letnikov():
    m = FractionalLimbMixer(d_model=8, n_limbs=1, taps=8)
    with torch.no_grad():
        m.raw_alpha.copy_(torch.log(torch.expm1((torch.tensor([0.5]) - 0.05) / 1.9)))
    w = m.kernels()[0]
    # raw GL weights for alpha=0.5: w_i = w_{i-1} * (1 - 1.5/i)
    raw = torch.tensor([1.0, -0.5, -0.125, -0.0625, -0.0390625,
                        -0.02734375, -0.0205078125, -0.01611328125])
    raw = raw / raw.abs().sum()
    assert torch.allclose(w, raw, atol=1e-5)


def test_alpha_zero_kernel_is_identity():
    m = FractionalLimbMixer(d_model=4, n_limbs=1, taps=8)
    with torch.no_grad():
        m.raw_alpha.copy_(torch.log(torch.expm1((torch.tensor([0.05]) - 0.05) / 1.9)))
    w = m.kernels()[0]
    # alpha -> 0 approaches a delta kernel; after L1 normalisation w0 ~ 0.89, rest tiny
    assert w[0].item() > 0.85
    assert w[1:].abs().sum().item() < 0.15


def test_frac_mixer_is_causal():
    torch.manual_seed(2)
    m = FractionalLimbMixer(d_model=16, n_limbs=4, taps=16)
    m.eval()
    x = torch.randn(1, 32, 16)
    with torch.no_grad():
        y = m(x)
        x2 = x.clone()
        x2[:, 20:] = torch.randn(1, 12, 16)
        y2 = m(x2)
    assert torch.allclose(y[:, :20], y2[:, :20], atol=1e-5)


def test_frac_mixer_gradients_flow():
    m = FractionalLimbMixer(d_model=8, n_limbs=4, taps=8)
    x = torch.randn(2, 16, 8, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.raw_alpha.grad is not None, "fractional orders must be learnable"
    assert m.gain.grad is not None


def test_torus_rope_no_periodic_aliasing():
    """The upstream torus encoding repeats exactly every M*m positions; ours must not."""
    rope = TorusRoPE(head_dim=16, use_phi=True)
    device = rope.inv_freq_a.device
    all_pos = torch.arange(0, 2048, device=device)
    ang_a = torch.outer(all_pos, rope.inv_freq_a)
    ang_b = torch.outer(all_pos * rope.phi, rope.inv_freq_b)
    joint = torch.cat([ang_a, ang_b], dim=-1)  # [P, head_dim/2]

    # No two positions within 2048 may share the same joint angle vector (mod 2pi).
    # Golden-ratio winding is irrational, so exact repeats are impossible; check that
    # even near-repeats are rare for the v band.
    v_angle = (ang_b[:, 0] % (2 * math.pi)).sort().values
    gaps = v_angle[1:] - v_angle[:-1]
    assert gaps.min().item() > 1e-6, "golden winding produced a near-alias"

    # Contrast: the UPSTREAM mapping u = 2*pi*(p mod M)/M is exactly periodic.
    M = 16.0
    u = lambda p: 2 * math.pi * (p % M) / M
    assert u(3) == u(3 + M), "upstream-style mod-M encoding aliases after M positions (the defect)"
    # ...and with both radii, after M*m:
    m_r = 8.0
    alias = lambda p: (2 * math.pi * (p % M) / M, 2 * math.pi * ((p // M) % m_r) / m_r)
    assert alias(5) == alias(5 + M * m_r)


def test_tetra_bias_prefers_near_positions():
    cfg = tiny_cfg()
    model = GMGPT(cfg)
    attn = model.blocks[0].attn
    bias = attn._tetra_bias(8, 8, torch.device("cpu"), torch.float32)  # [H, T, S]
    # for each head: bias[i, j] must become more negative as |i-j| grows
    for h in range(attn.n_head):
        diag = bias[h].diag()
        far = torch.diagonal(bias[h], offset=7)
        assert (far < diag).all()


# ----------------------------------------------------------------- integration
def test_forward_shapes_and_params():
    model = GMGPT(tiny_cfg())
    x = torch.randint(0, 280, (3, 32))
    logits, hidden, loss, _ = model(x, targets=x)
    assert logits.shape == (3, 32, 280)
    assert hidden.shape == (3, 32, 64)
    assert loss is not None and torch.isfinite(loss)
    assert model.num_params() > 100_000


def test_generate_shape_and_range():
    torch.manual_seed(3)
    model = GMGPT(tiny_cfg())
    prompt = torch.randint(0, 280, (1, 4))
    out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=30)
    assert out.shape == (1, 24)
    assert out.min() >= 0 and out.max() < 280


def test_composite_loss_terms():
    crit = CompositeLoss(d_model=64, lambda_wm=0.1, lambda_stability=0.1)
    logits = torch.randn(2, 16, 280)
    hidden = torch.randn(2, 16, 64)
    targets = torch.randint(0, 280, (2, 16))
    terms = crit(logits, hidden, targets)
    assert {"lm", "world_model", "stability", "total"} <= set(terms)
    for v in terms.values():
        assert torch.isfinite(v)
    terms["total"].backward()
    assert crit.wm_head[0].weight.grad is not None


def test_training_smoke_reduces_loss():
    torch.manual_seed(4)
    # synthetic corpus with real structure: repeating pattern text
    text = ("the cat sat on the mat . " * 40 + "the dog sat on the log . " * 40)
    tok = ByteBPETokenizer.train([text] * 5, vocab_size=320)
    stream = texts_to_token_stream([text] * 30, tok)
    chunks = chunk_stream(stream, 32)
    model = GMGPT(tiny_cfg(vocab_size=tok.vocab_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(30):
        xb = chunks[:8]
        _, _, loss, _ = model(xb, targets=xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9, f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
