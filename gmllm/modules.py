"""Neural modules for GM-GPT.

Salvaged and re-implemented from GitMonsters' octotetrahedral-agi:

- ``TorusRoPE``  -- their "tetrahedral torus positional encoding", rebuilt with
  golden-ratio winding so no two positions share a code.  Their original used
  ``phase = (t * k) mod M`` which aliases position ``t`` and ``t + M`` exactly.
- ``CausalSelfAttention`` -- their Gaussian per-head relative ("tetrahedral")
  bias, but with a *strict causal mask* (their code had none: that is the
  leakage bug documented in ANALYSIS.md).  Near-lag heads look close, far-lag
  heads look deep into the past.
- ``FractionalLimbMixer`` -- their fractional-order Gruenwald-Letnikov
  "octolimb" convolution, re-implemented so it is (a) causal, (b) trainable end
  to end, and (c) streamable via a rolling history buffer so KV-cached decoding
  matches a full forward pass exactly.  Their original applied a centered
  (non-causal) kernel, was frozen under ``@torch.no_grad()``, and was never
  registered in the model graph.

Everything else (RMSNorm, SwiGLU) is standard stable infrastructure.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Norm / MLP
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=bias)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=bias)   # up
        self.w2 = nn.Linear(hidden, dim, bias=bias)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------- #
# Positional encoding: torus / RoPE
# --------------------------------------------------------------------------- #
class TorusRoPE(nn.Module):
    """Rotary encoding with two winding bands: standard + golden-ratio scaled.

    This is GitMonsters' torus idea (two circular phase bands, one wound faster
    than the other) with two fixes:

    1. Their phase map was ``u = 2*pi*(p mod M)/M`` -- exactly periodic with
       period M (and the second band period M*m), so positions p and p + M*m
       received *identical* encodings.  We wind continuously: band B rotates at
       ``phi * freq`` per position (phi = golden ratio, irrational), so exact
       repeats are impossible and near-repeats stay rare.
    2. Applied as rotary (relative) rotations rather than absolute additive
       vectors, so attention scores depend only on position *differences*.

    ``head_dim`` must be divisible by 4: the first half of the head dims use
    band A (``inv_freq_a``), the second half use band B (``inv_freq_b``).
    """

    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_seq: int = 8192, use_phi: bool = True):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 (two bands x pairs)"
        quarter = head_dim // 4
        inv_freq_a = base ** (-2.0 * torch.arange(quarter, dtype=torch.float32) / (head_dim // 2))
        inv_freq_b = base ** (-2.0 * (torch.arange(quarter, dtype=torch.float32)
                                      + quarter) / (head_dim // 2))
        phi = (1.0 + 5.0 ** 0.5) / 2.0 if use_phi else 1.0
        self.register_buffer("inv_freq_a", inv_freq_a, persistent=False)
        self.register_buffer("inv_freq_b", inv_freq_b, persistent=False)
        self.phi = phi
        self.head_dim = head_dim
        self.max_seq = max_seq

    def _angles(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ang_a = torch.outer(pos.float(), self.inv_freq_a)          # (T, q)
        ang_b = torch.outer(pos.float() * self.phi, self.inv_freq_b)  # (T, q)
        return ang_a, ang_b

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, H, T, head_dim).  ``pos`` = absolute positions (T,) for
        incremental decoding; defaults to 0..T-1."""
        B, H, T, d = x.shape
        if pos is None:
            pos = torch.arange(T, device=x.device)
        ang_a, ang_b = self._angles(pos.to(x.device))
        ca, sa = ang_a.cos(), ang_a.sin()      # (T, q)
        cb, sb = ang_b.cos(), ang_b.sin()
        q = d // 4
        xa1, xa2 = x[..., :q], x[..., q:2 * q]
        xb1, xb2 = x[..., 2 * q:3 * q], x[..., 3 * q:]
        ca = ca.view(1, 1, T, q); sa = sa.view(1, 1, T, q)
        cb = cb.view(1, 1, T, q); sb = sb.view(1, 1, T, q)
        oa1 = xa1 * ca - xa2 * sa
        oa2 = xa1 * sa + xa2 * ca
        ob1 = xb1 * cb - xb2 * sb
        ob2 = xb1 * sb + xb2 * cb
        return torch.cat([oa1, oa2, ob1, ob2], dim=-1)


# --------------------------------------------------------------------------- #
# Attention (causal, tetrahedral Gaussian bias)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head attention with per-head Gaussian relative bias.

    Upstream octotetrahedral-agi applied ``tetra_bias_lookup`` with *no causal
    mask* -- token t could attend to token t+1, which is target leakage for an
    autoregressive LM (see ANALYSIS.md).  Here the mask is strict (or absent
    only in the deliberate ``causal=False`` leak mode used by the ablation),
    and the per-head Gaussian lags are kept: head h is centred at lag c_h
    (0 = attends to itself/recent, large |c_h| = reads deep history).
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0,
                 tetra_bias: bool = True, causal: bool = True,
                 max_seq: int = 8192, bias: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.tetra_bias = tetra_bias
        self.causal = causal
        self.dropout = dropout
        self.max_seq = max_seq
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

    # -- salvaged idea: per-head Gaussian lag bias (log space, additive) ----- #
    def _tetra_bias(self, T: int, S: int, device, dtype,
                    offset: int = 0) -> torch.Tensor:
        """Return [H, T, S] additive log-bias.

        ``offset`` = number of cached keys before the current query window, so
        query i sits at absolute position ``offset + i``.  Head h is a Gaussian
        falloff over |lag| with per-head scale sigma_h growing geometrically
        from near (head 0: sharp falloff, specialises in recent context) to far
        (last head: flat, specialises in long-range structure) -- the
        "tetrahedral" near/far head asymmetry.  Depends only on the lag
        (offset + i) - j, so it is identical for full and cached decoding.
        """
        H = self.n_head
        sigmas = (self.max_seq / 4.0) ** (
            torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1))
        sigmas = sigmas.clamp(min=1.0)                             # (H,)
        i = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1) + offset
        j = torch.arange(S, device=device, dtype=torch.float32).unsqueeze(0)
        d = (i - j).abs()                                          # (T, S)
        bias = -0.5 * (d.unsqueeze(0) / sigmas.view(H, 1, 1)) ** 2  # (H, T, S)
        return bias.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[TorusRoPE] = None,
        pos: Optional[torch.Tensor] = None,
        layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        S_old = 0
        if layer_cache is not None:
            k = torch.cat([layer_cache[0], k], dim=2)
            v = torch.cat([layer_cache[1], v], dim=2)
            S_old = layer_cache[0].size(2)
        S = k.size(2)

        if rope is not None:
            q = rope(q, pos if pos is not None
                     else torch.arange(S_old, S_old + T, device=x.device))
            if S_old == 0:
                k = rope(k)
            else:
                # cached keys were already rotated at their own positions
                k_new = rope(k[:, :, S_old:], torch.arange(S_old, S, device=x.device))
                k = torch.cat([k[:, :, :S_old], k_new], dim=2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)         # (B,H,T,S)

        if self.tetra_bias:
            att = att + self._tetra_bias(T, S, x.device, att.dtype, offset=S_old)

        if self.causal:
            # query i (abs pos S_old + i) may see keys j <= S_old + i
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S_old)
            att = att.masked_fill(~mask.view(1, 1, T, S), float("-inf"))

        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        new_cache = (k, v) if use_cache else None
        return y, new_cache


# --------------------------------------------------------------------------- #
# Fractional (Gruenwald-Letnikov) limb mixer -- salvaged, made causal + trainable
# --------------------------------------------------------------------------- #
class FractionalLimbMixer(nn.Module):
    """Depthwise causal fractional-order convolution, one order per limb.

    Upstream (parallplexity "fractional kernels" idea, shipped dead): a bank of
    Gruenwald-Letnikov difference kernels applied to the token sequence.  Their
    implementation used a centered window (future leakage), was wrapped in
    ``@torch.no_grad()`` with a fixed ``alpha`` schedule, and -- because it
    never returned grad-enabled outputs -- contributed nothing to the model.

    Here the fractional orders ``alpha_l`` are learned parameters (through a
    softplus reparametrisation keeping them in (0.05, ~2)), the kernel is
    strictly causal, and a rolling history buffer makes the op exactly
    streamable for KV-cached decoding.

    Kernel recurrence (matching ``kernels()``): w_0 = 1,
    w_i = w_{i-1} * (i - 1 - alpha) / i, then L1 normalised by absolute sum.
    """

    def __init__(self, d_model: int, n_limbs: int = 8, taps: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_limbs = n_limbs
        self.taps = taps
        # alpha = 0.05 + 1.9 * softplus(raw_alpha)  ->  alpha in (0.05, inf), init ~0.5
        init_raw = torch.log(torch.expm1((torch.full((n_limbs,), 0.5) - 0.05) / 1.9))
        self.raw_alpha = nn.Parameter(init_raw)
        self.gain = nn.Parameter(torch.ones(n_limbs, d_model))

    @property
    def alpha(self) -> torch.Tensor:
        return 0.05 + 1.9 * F.softplus(self.raw_alpha)

    def kernels(self) -> torch.Tensor:
        """[n_limbs, taps] L1(abs)-normalised GL kernel per limb. Differentiable."""
        a = self.alpha                                             # (L,)
        ws = [torch.ones_like(a)]
        for i in range(1, self.taps):
            ws.append(ws[-1] * ((i - 1) - a) / i)
        w = torch.stack(ws, dim=1)                                 # (L, W)
        return w / w.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(
        self,
        x: torch.Tensor,
        hist: Optional[torch.Tensor] = None,
        return_hist: bool = False,
    ):
        """x: (B, T, d_model).  ``hist``: (B, taps-1, d_model) past inputs.

        Returns y (B, T, d_model), or (y, new_hist) if ``return_hist``.
        y[t] depends only on x[<= t] (and the supplied history) -- causal.
        """
        B, T, C = x.shape
        W = self.taps
        w = self.kernels()                                   # (L, W)
        pad = x.new_zeros(B, W - 1, C) if hist is None else hist
        seq = torch.cat([pad, x], dim=1)                     # (B, W-1+T, C)
        new_hist = seq[:, -(W - 1):, :].detach() if W > 1 else x.new_zeros(B, 0, C)
        # windows[:, t, c, j] = seq[t + j]  (j = 0..W-1) -> only past of t
        windows = seq.unfold(1, W, 1)                        # (B, T, C, W)
        conv = torch.einsum("btcw,lw->bltc", windows, w)     # (B, L, T, C)
        y = (conv * self.gain.view(1, self.n_limbs, 1, C)).sum(dim=1)
        return (y, new_hist) if return_hist else y

    # alias used by the streaming path in Block
    def step(self, x: torch.Tensor, hist: Optional[torch.Tensor] = None):
        return self.forward(x, hist=hist, return_hist=True)


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-norm block: causal attention (+tetra bias, +RoPE) -> frac limbs -> SwiGLU.

    Returns ``(x, cache)`` where ``cache = (k, v, frac_hist)`` (or None when
    ``use_cache=False``); pass it back in as ``layer_cache`` for incremental
    decoding.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0,
                 tetra_bias: bool = True, use_rope: bool = True,
                 rope_phi: bool = True, causal: bool = True,
                 use_limbs: bool = True, n_limbs: int = 8, limb_taps: int = 32,
                 max_seq: int = 8192, bias: bool = False):
        super().__init__()
        self.use_limbs = use_limbs
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_head, dropout=dropout, tetra_bias=tetra_bias,
            causal=causal, max_seq=max_seq, bias=bias,
        )
        self.rope = TorusRoPE(d_model // n_head, max_seq=max_seq,
                              use_phi=rope_phi) if use_rope else None
        self.ln2 = RMSNorm(d_model)
        if use_limbs:
            self.frac = FractionalLimbMixer(d_model, n_limbs=n_limbs, taps=limb_taps)
        hidden = (int(d_model * 8 / 3) // 2) * 2
        self.mlp = SwiGLU(d_model, hidden, bias=bias)
        self.ln3 = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                layer_cache: Optional[tuple] = None,
                use_cache: bool = False):
        attn_cache = None if layer_cache is None else layer_cache[:2]
        a, new_attn_cache = self.attn(self.ln1(x), rope=self.rope,
                                      layer_cache=attn_cache, use_cache=use_cache)
        x = x + self.drop(a)
        frac_hist = None if layer_cache is None else (
            layer_cache[2] if len(layer_cache) > 2 else None)
        new_frac_hist = None
        if self.use_limbs:
            f, new_frac_hist = self.frac(self.ln2(x), hist=frac_hist, return_hist=True)
            x = x + self.drop(f)
        x = x + self.drop(self.mlp(self.ln3(x)))
        cache = None
        if use_cache:
            cache = (new_attn_cache[0], new_attn_cache[1], new_frac_hist)
        return x, cache
