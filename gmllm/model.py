"""GMGPT — the gmllm language model.

A compact pre-norm decoder-only transformer:

    token emb -> [ Block x N ] -> RMSNorm -> LM head

where each Block contains:
    * CausalSelfAttention with TorusRoPE + tetrahedral Gaussian bias
      (octotetrahedral-agi attention, made causal; torus-attention geometry,
      de-aliased)
    * SwiGLU MLP
    * FractionalLimbMixer — 8 learnable power-law temporal filters
      (parallplexity fractional kernels, made trainable and causal)

`causal=False` reproduces the upstream octotetrahedral-agi defect (bidirectional
attention in an autoregressive LM) and exists only for the leakage ablation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Block, RMSNorm


@dataclass
class GMConfig:
    vocab_size: int = 512
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    ctx: int = 256
    dropout: float = 0.1
    n_limbs: int = 8
    limb_taps: int = 32
    use_tetra_bias: bool = True
    use_torus_rope: bool = True
    use_phi_winding: bool = True
    use_limbs: bool = True
    causal: bool = True          # False => reproduce upstream leakage (ablation only)
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class GMGPT(nn.Module):
    def __init__(self, cfg: GMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(
                d_model=cfg.d_model, n_head=cfg.n_head, dropout=cfg.dropout,
                tetra_bias=cfg.use_tetra_bias, use_rope=cfg.use_torus_rope,
                rope_phi=cfg.use_phi_winding, causal=cfg.causal,
                use_limbs=cfg.use_limbs, n_limbs=cfg.n_limbs, limb_taps=cfg.limb_taps,
            )
            for _ in range(cfg.n_layer)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual-path projections
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    @torch.no_grad()
    def resize_vocab(self, new_size: int) -> "GMGPT":
        """Grow (or shrink) the embedding + LM head to ``new_size`` rows.

        Existing rows are copied verbatim, so a checkpoint trained at 512 tokens
        keeps every learned id; new rows get the standard scaled init.  Used to
        add atomic control-marker tokens without retraining the base model.
        """
        old = self.cfg.vocab_size
        if new_size == old:
            return self
        emb = nn.Embedding(new_size, self.cfg.d_model)
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        emb.weight[: min(old, new_size)] = self.tok_emb.weight[: min(old, new_size)]
        self.tok_emb = emb

        head = nn.Linear(self.cfg.d_model, new_size, bias=False)
        nn.init.normal_(head.weight, mean=0.0, std=0.02)
        head.weight[: min(old, new_size)] = self.lm_head.weight[: min(old, new_size)]
        self.lm_head = head
        if self.cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.cfg.vocab_size = new_size
        return self

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past: Optional[List] = None,
    ):
        """idx: [B, T]. Returns (logits, hidden, loss, new_past)."""
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        new_past = [] if use_cache else None
        offset = past[0][0].size(2) if past else 0
        for i, block in enumerate(self.blocks):
            pc = past[i] if past else None
            # RoPE offset differs from KV length only never here (both grow together)
            x, cache = block(x, layer_cache=pc, use_cache=use_cache)
            if use_cache:
                new_past.append(cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, x, loss, new_past

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        rep_penalty: float = 1.1,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        past = None
        window: List[int] = []
        for _ in range(max_new_tokens):
            step_in = idx if past is None else idx[:, -1:]
            logits, _, _, past = self.forward(step_in, use_cache=True, past=past)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if rep_penalty != 1.0 and window:
                seen = list(set(window[-64:]))
                logits[:, seen] = torch.where(
                    logits[:, seen] > 0, logits[:, seen] / rep_penalty, logits[:, seen] * rep_penalty
                )
            logits[:, 1] = float("-inf")  # byte 0x01 unused as a starter (parity w/ upstream UNK mask)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            window.append(int(nxt.item()))
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
        return idx

    def param_groups(self, weight_decay: float):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "raw_alpha" in name or "gain" in name or "log_slope" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
