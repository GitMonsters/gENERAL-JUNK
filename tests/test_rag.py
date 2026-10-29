"""RAG: chunking, ranking, relevance gating, and citation behaviour.

These pin the properties that make the RAG path trustworthy: retrieval is
deterministic, the right passage is found, and the system refuses instead of
guessing when the corpus cannot answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import BM25Retriever, Chunk, RAG
from rag.generator import ExtractiveGenerator
from rag.retriever import chunk_text, tokenize

DOCS = [
    "Force is an interaction that changes the motion of an object. "
    "The SI unit of force is the newton, symbol N. "
    "Forces can be measured with a spring scale.",
    "Energy is the capacity to do work. "
    "The SI unit of energy is the joule. "
    "Kinetic energy is the energy of motion.",
    "The capital of France is Paris, its largest city. "
    "France is a country in western Europe.",
]


def _rag():
    chunks = []
    for i, t in enumerate(DOCS):
        for j, ch in enumerate(chunk_text(t, target=200)):
            chunks.append(Chunk(text=ch, doc_id=str(i), title=f"doc{i}"))
    return RAG(BM25Retriever(chunks), ExtractiveGenerator(), k=3)


# ----------------------------------------------------------------- chunking
def test_chunk_text_sentence_aligned_with_overlap():
    text = ("First sentence here. Second sentence here. Third sentence here. "
            "Fourth sentence here. Fifth sentence here.")
    ch = chunk_text(text, target=50, overlap=10)
    assert len(ch) >= 2
    for c in ch:
        # chunk boundaries land on sentence ends, not mid-word
        assert c.rstrip().endswith(".")
        assert not c.startswith(" ") and not c.endswith(" ")


def test_tokenize_drops_stopwords_keeps_content_and_numbers():
    t = tokenize("What is the SI unit of force at 9.8 m/s?")
    assert "force" in t and "unit" in t and "si" in t
    assert "the" not in t and "what" not in t
    assert "9.8" in t  # numbers survive (units matter in physics)


# ---------------------------------------------------------------- retrieval
def test_search_ranks_the_right_document_first():
    r = _rag()
    hits = r.retriever.search("what is the SI unit of force", k=3)
    assert hits, "expected at least one hit"
    assert "newton" in hits[0].chunk.text.lower()
    assert hits[0].score > 0


def test_search_is_deterministic():
    r = _rag()
    a = [(h.chunk.text, h.score) for h in r.retriever.search("unit of energy", k=3)]
    b = [(h.chunk.text, h.score) for h in r.retriever.search("unit of energy", k=3)]
    assert a == b


def test_search_empty_query_returns_nothing():
    r = _rag()
    assert r.retriever.search("", k=3) == []
    assert r.retriever.search("the of and", k=3) == []  # all stopwords


def test_retriever_builds_from_jsonl(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(
        '{"title": "T%d", "text": "%s"}' % (i, t.replace('"', "'"))
        for i, t in enumerate(DOCS)), encoding="utf-8")
    r = BM25Retriever.from_jsonl(p, target=200)
    assert len(r.chunks) >= 3
    hits = r.search("joule", k=2)
    assert hits and "joule" in hits[0].chunk.text

# ------------------------------------------------------------ answer + gate
def test_extractive_answer_quotes_and_cites():
    r = _rag()
    a = r.ask("what is the SI unit of force")
    assert a.grounded
    assert "newton" in a.text.lower()
    assert "[1]" in a.text or "[2]" in a.text
    assert a.sources and a.sources[0]["title"].startswith("doc")


def test_extractive_refuses_when_corpus_cannot_answer():
    """An off-domain question must NOT get a confident quote."""
    r = _rag()
    a = r.ask("how do I bake sourdough bread")
    assert not a.grounded
    assert "don't know" in a.text.lower()


def test_answer_includes_source_text_for_checking():
    r = _rag()
    a = r.ask("what is the SI unit of energy")
    assert a.sources
    assert all({"n", "title", "score", "text"} <= set(s) for s in a.sources)


def test_rag_retrieve_matches_ask_sources():
    r = _rag()
    hits = r.retrieve("force", k=2)
    a = r.ask("force", k=2)
    assert [s["title"] for s in a.sources] == [h.chunk.title for h in hits]
