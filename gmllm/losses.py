"""Composite objective — the salvageable part of the upstream six-term loss.

Upstream: octotetrahedral-agi/core/recursive_engine_objective.py defines
    L_total = L_task + 0.10*L_WM + 0.05*L_meta + 0.02*L_resource
              + 0.05*L_ground + 0.15*L_stability
but several terms were fed by monitors running under torch.no_grad(), so they could
not affect training at all (see ANALYSIS.md §3). What survives inspection:

  * L_WM — a world-model auxiliary: predict the next position's hidden state from the
    current one. With a causal model this is a real, non-trivial self-supervised task
    (upstream it was trivially satisfied by the leak).
  * L_stability — upstream "cohesion": consecutive hidden states should not jitter.
    Kept as an optional cosine-smoothness penalty (default weight 0: measured, not
    assumed, to help).

Everything is logged term-by-term so nothing hides.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    def __init__(
        self,
        d_model: int,
        lambda_wm: float = 0.05,
        lambda_stability: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_wm = lambda_wm
        self.lambda_stability = lambda_stability
        self.label_smoothing = label_smoothing
        if lambda_wm > 0:
            self.wm_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
        else:
            self.wm_head = None

    def forward(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """logits/hidden: [B, T, ...]; targets: [B, T] (next-token ids, already shifted by caller)."""
        B, T, V = logits.shape
        lm = F.cross_entropy(
            logits.reshape(-1, V).float(), targets.reshape(-1),
            ignore_index=-1, label_smoothing=self.label_smoothing,
        )
        terms: Dict[str, torch.Tensor] = {"lm": lm}
        total = lm

        if self.wm_head is not None and T > 1:
            pred = self.wm_head(hidden[:, :-1])
            true = hidden[:, 1:].detach()
            wm = F.mse_loss(pred, true)
            terms["world_model"] = wm
            total = total + self.lambda_wm * wm

        if self.lambda_stability > 0 and T > 1:
            cos = F.cosine_similarity(hidden[:, 1:], hidden[:, :-1].detach(), dim=-1)
            stab = (1.0 - cos).mean()
            terms["stability"] = stab
            total = total + self.lambda_stability * stab

        terms["total"] = total
        return terms
