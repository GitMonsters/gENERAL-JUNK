"""Retrieval for the RAG assistant — no external services required.

Why lexical (BM25) rather than dense embeddings *first*: RESULTS.md §3e shows this
project's value is in honest, measurable behaviour, and retrieval quality is
measurable.  A lexical retriever needs no model download, is deterministic, and
for the dominant RAG case here -- looking up a fact stated in a document -- it is
competitive with or better than dense embeddings.  A dense backend is left as a
plug-in point (`Retriever` protocol) rather than pretended.

Everything is pure Python + numpy so it runs in the audit environment.

    from rag.retriever import BM25Retriever
    r = BM25Retriever.from_jsonl("corpus.jsonl")
    hits = r.search("what is the SI unit of force", k=3)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Sequence, Tuple

import numpy as np

#: A conservative tokenizer: word chars plus intra-word digits/dots (9.8, f=ma).
_WORD = re.compile(r"[a-z0-9][a-z0-9._\-']*")

STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "and", "or", "by", "with", "as", "that", "this", "it", "be", "from",
    "what", "which", "who", "how", "why", "when", "where", "does", "do",
}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords and trailing punctuation removed.

    The trailing-strip matters: sentence-final words are indexed as ``joule.``
    otherwise, so ``search("joule")`` would miss them -- which silently breaks
    retrieval for most answer words (caught by tests/test_rag.py).
    """
    out = []
    for t in _WORD.findall(text.lower()):
        t = t.strip("._-'")          # drop trailing/leading punctuation, keep 9.8
        if t and t not in STOP:
            out.append(t)
    return out


@dataclass
class Chunk:
    """A retrievable passage.  `doc_id`/`title` support citation in answers."""

    text: str
    doc_id: str = ""
    title: str = ""
    start: int = 0


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Any backend that can rank chunks for a query."""

    def search(self, query: str, k: int = 3) -> List[Hit]: ...


def chunk_text(text: str, target: int = 900, overlap: int = 150) -> List[str]:
    """Split into ~`target`-char passages on sentence boundaries, with overlap.

    Sentence-aligned so a fact and its value are not cut apart, and overlapping so
    a fact straddling a boundary is still fully contained in some chunk.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > target:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + " " + s).strip()
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


class BM25Retriever:
    """Okapi BM25 over sentence-aligned chunks. Deterministic, no downloads."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks: List[Chunk] = list(chunks)
        self.k1, self.b = k1, b
        self.tokens: List[List[str]] = [tokenize(c.text) for c in self.chunks]
        self.tf: List[Counter] = [Counter(t) for t in self.tokens]
        self.dl = np.array([max(len(t), 1) for t in self.tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0
        df = Counter()
        for t in self.tokens:
            for w in set(t):
                df[w] += 1
        n = max(len(self.chunks), 1)
        # BM25 idf with the +0.5 smoothing; negative idf clamped to a small floor
        self.idf: Dict[str, float] = {
            w: max(math.log((n - c + 0.5) / (c + 0.5) + 1.0), 1e-6)
            for w, c in df.items()
        }

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_texts(cls, texts: Iterable[str], **kw) -> "BM25Retriever":
        chunks = [Chunk(text=t, doc_id=str(i), title=f"doc {i}")
                  for i, t in enumerate(texts)]
        return cls(chunks, **kw)

    @classmethod
    def from_jsonl(cls, path: str | Path, text_field: str = "text",
                   title_field: str = "title", target: int = 900,
                   limit_docs: int | None = None, **kw) -> "BM25Retriever":
        """Build from a {'title':..., 'text':...} JSONL (the corpus format here)."""
        chunks: List[Chunk] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit_docs is not None and i >= limit_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                body = obj.get(text_field) or ""
                title = obj.get(title_field) or f"doc {i}"
                doc_id = obj.get("doc_id", str(i))
                for j, ch in enumerate(chunk_text(body, target=target)):
                    chunks.append(Chunk(text=ch, doc_id=str(doc_id), title=title,
                                        start=j))
        return cls(chunks, **kw)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 3) -> List[Hit]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        # deterministic ordering: score desc, then index asc
        order = np.lexsort((np.arange(len(scores)), -scores))
        out: List[Hit] = []
        for i in order[:k]:
            if scores[i] <= 0:
                break
            out.append(Hit(chunk=self.chunks[int(i)], score=float(scores[i])))
        return out
