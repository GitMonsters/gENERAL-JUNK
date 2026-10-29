"""Data pipeline: JSONL text corpora -> BPE token stream -> fixed-size chunks.

Upstream corpora (downloaded to %TEMP%\\opencode\\gitmonsters\\octotetrahedral-agi\\):
  data/combined_train.jsonl  102,358 sentences (wiki + CLARIN news + transcripts)
  data/eval_heldout.jsonl    500 sentences (out-of-domain SEO text; used as an OOD set)
Each line is {"text": "..."}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .tokenizer import ByteBPETokenizer


def read_jsonl_texts(paths: Sequence[str | Path], field: str = "text", max_bytes: int | None = None) -> List[str]:
    texts: List[str] = []
    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get(field)
                if isinstance(t, str) and t.strip():
                    texts.append(t)
                    total += len(t)
                    if max_bytes is not None and total >= max_bytes:
                        return texts
    return texts


def texts_to_token_stream(texts: Iterable[str], tok: ByteBPETokenizer) -> np.ndarray:
    """Encode texts and concatenate with a newline byte (10) as document separator."""
    parts: List[List[int]] = []
    ntok = 0
    for n, t in enumerate(texts):
        ids = tok.encode(t)
        if ids:
            parts.append(ids + [10])
            ntok += len(ids) + 1
        if n and n % 2000 == 0:
            print(f"    [encode] {n} docs, {ntok:,} tokens", flush=True)
    flat = np.empty(sum(len(p) for p in parts), dtype=np.int32)
    i = 0
    for p in parts:
        flat[i : i + len(p)] = p
        i += len(p)
    return flat


def chunk_stream(stream: np.ndarray, ctx: int) -> torch.Tensor:
    n = (len(stream) // ctx) * ctx
    return torch.from_numpy(stream[:n].copy().reshape(-1, ctx).astype(np.int64))


def train_val_split(chunks: torch.Tensor, val_fraction: float, seed: int = 0):
    n = chunks.size(0)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_fraction))
    return chunks[perm[n_val:]], chunks[perm[:n_val]]


def encode_free_text(path: str | Path, tok: ByteBPETokenizer, ctx: int) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return chunk_stream(np.array(tok.encode(text), dtype=np.int32), ctx)


class ChunkBatcher:
    """Random fixed-size batches from a chunk tensor (simple, deterministic per seed)."""

    def __init__(self, chunks: torch.Tensor, batch_size: int, seed: int = 0):
        self.chunks = chunks
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def sample(self) -> torch.Tensor:
        idx = [self.rng.randrange(len(self.chunks)) for _ in range(self.batch_size)]
        return self.chunks[idx]
