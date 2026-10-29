"""Answer generators for RAG — pluggable, with a no-LLM fallback.

Three backends, in order of honesty about what they can do:

  * ``ExtractiveGenerator`` — no model at all.  Picks the best sentence from the
    retrieved passages.  Always grounded (it is a verbatim quote) and never
    hallucinates, but it cannot paraphrase.  This is what makes the assistant
    *usable today* in an environment with no LLM server.
  * ``OllamaGenerator`` — a local LLM (e.g. llama3.1:8b).  Reads the passages and
    writes an answer.  Recommended when available.
  * ``OpenAIGenerator`` — any OpenAI-compatible endpoint.

Every backend returns sources, and every backend can answer "I don't know" when
retrieval finds nothing relevant — the behaviour our own evaluation checks for.

    from rag.generator import get_generator
    g = get_generator("extractive")
    ans = g.answer("what is the SI unit of force", hits)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .retriever import Hit, tokenize

PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the passages "
    "below. If the passages do not contain the answer, reply exactly: "
    "I don't know based on the provided documents. Cite the passage numbers you "
    "used, like [1].\n\nPassages:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)


@dataclass
class Answer:
    text: str
    sources: List[dict]
    backend: str
    grounded: bool = False  # is the answer a quote / cited to the passages?


def _context(hits: Sequence[Hit]) -> str:
    return "\n".join(f"[{i+1}] {h.chunk.title}: {h.chunk.text}"
                     for i, h in enumerate(hits))


def _sources(hits: Sequence[Hit]) -> List[dict]:
    return [{"n": i + 1, "title": h.chunk.title, "doc_id": h.chunk.doc_id,
             "score": round(h.score, 3), "text": h.chunk.text[:300]}
            for i, h in enumerate(hits)]


class ExtractiveGenerator:
    """No-LLM baseline: quote the best-matching sentence. Cannot hallucinate.

    Relevance gate: an answer is only produced when the best hit (a) shares
    enough of the query's content words (``min_coverage``), and (b) scores at
    least ``min_score_frac`` of the maximum achievable BM25 score for that query
    in this index.  The second condition is deliberately a *fraction*: BM25
    magnitude depends on corpus size and IDF, so an absolute threshold tuned on
    one corpus silently mis-gates another (caught by tests/test_rag.py).
    """

    backend = "extractive"
    min_coverage = 0.5
    min_score_frac = 0.35   # of the best possible score for this query
    retriever = None        # set by RAG so the gate can normalise scores

    def _max_score(self, query: str) -> float:
        """Upper bound on BM25 for `query` against a perfect (whole-index) match."""
        r = self.retriever
        if r is None:
            return 0.0
        total = 0.0
        for w in set(tokenize(query)):
            idf = r.idf.get(w)
            if idf is None:
                continue
            # tf -> inf gives the BM25 saturation limit (k1+1) * idf
            total += idf * (r.k1 + 1)
        return total

    def answer(self, query: str, hits: Sequence[Hit]) -> Answer:
        qs = set(tokenize(query))
        if not hits:
            return Answer("I don't know based on the provided documents.", [],
                          self.backend, grounded=False)
        top = hits[0]
        coverage = len(qs & set(tokenize(top.chunk.text))) / max(len(qs), 1)
        ceiling = self._max_score(query)
        frac = top.score / ceiling if ceiling > 0 else 0.0
        if coverage < self.min_coverage or frac < self.min_score_frac:
            return Answer("I don't know based on the provided documents.",
                          _sources(hits), self.backend, grounded=False)
        # Score sentences by query-term overlap; prefer SHORT, dense sentences
        # (they are definition-shaped: "the SI unit of force is the newton")
        # rather than long ones that merely mention the terms.
        best, best_score = None, -1.0
        for h in hits:
            for s in re.split(r"(?<=[.!?])\s+", h.chunk.text):
                n = len(s.split())
                if n < 4 or n > 60:
                    continue
                ov = len(qs & set(tokenize(s))) / max(len(qs), 1)
                dens = ov / (1.0 + 0.02 * n)          # penalise long sentences
                sc = dens + 0.01 * min(h.score, 10.0)  # nudge by chunk rank
                if sc > best_score:
                    best, best_score = s.strip(), sc
        if not best:
            return Answer("I don't know based on the provided documents.",
                          _sources(hits), self.backend, grounded=False)
        cite = 1
        for i, h in enumerate(hits):
            if best in h.chunk.text:
                cite = i + 1
                break
        return Answer(f"{best} [{cite}]", _sources(hits), self.backend, grounded=True)


class OllamaGenerator:
    """Local LLM via Ollama's /api/chat."""

    backend = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 host: str = "http://127.0.0.1:11434"):
        self.model, self.host = model, host.rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def answer(self, query: str, hits: Sequence[Hit]) -> Answer:
        if not hits:
            return Answer("I don't know based on the provided documents.", [],
                          self.backend, grounded=False)
        body = json.dumps({
            "model": self.model, "stream": False,
            "messages": [{"role": "user",
                          "content": PROMPT.format(ctx=_context(hits), q=query)}],
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            obj = json.loads(r.read().decode())
        text = (obj.get("message") or {}).get("content", "").strip()
        grounded = "don't know" not in text.lower()
        return Answer(text, _sources(hits), self.backend, grounded=grounded)


class OpenAIGenerator:
    """Any OpenAI-compatible chat endpoint (OpenAI, vLLM, LM Studio, llama.cpp)."""

    backend = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def answer(self, query: str, hits: Sequence[Hit]) -> Answer:
        if not hits:
            return Answer("I don't know based on the provided documents.", [],
                          self.backend, grounded=False)
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "user",
                          "content": PROMPT.format(ctx=_context(hits), q=query)}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            obj = json.loads(r.read().decode())
        text = obj["choices"][0]["message"]["content"].strip()
        return Answer(text, _sources(hits), self.backend,
                      grounded="don't know" not in text.lower())


def get_generator(name: str, **kw):
    name = (name or "extractive").lower()
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "ollama":
        return OllamaGenerator(**kw)
    if name in ("openai", "compatible"):
        return OpenAIGenerator(**kw)
    raise ValueError(f"unknown generator {name!r}")
