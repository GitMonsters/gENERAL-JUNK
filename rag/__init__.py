"""rag — retrieval-augmented answering over your own documents.

Built because this repo's own measurements (RESULTS.md §3e) show a 3.3 M-param
CPU LM learns distributions but never facts: fact recall was 0/40 for every
checkpoint and every corpus.  Retrieval is the correct fix for fact questions —
the information is looked up at question time and the model's weights are
unchanged.

    from rag import RAG
    r = RAG.from_jsonl("docs.jsonl", generator="extractive")
    print(r.ask("what is the SI unit of force").text)
"""

from .retriever import BM25Retriever, Chunk, Hit, chunk_text
from .generator import Answer, get_generator

__all__ = ["RAG", "BM25Retriever", "Chunk", "Hit", "Answer",
           "chunk_text", "get_generator"]


class RAG:
    """Retrieve-then-read.  `ask()` returns an Answer with citations."""

    def __init__(self, retriever, generator, k: int = 4):
        self.retriever = retriever
        self.generator = generator
        self.k = k
        # let score-normalising generators (the extractive gate) see the index
        if getattr(generator, "retriever", None) is None:
            try:
                generator.retriever = retriever
            except Exception:
                pass

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_jsonl(cls, path, generator: str = "extractive", k: int = 4,
                   target: int = 900, limit_docs=None, retriever=None, **gen_kw):
        r = retriever or BM25Retriever.from_jsonl(path, target=target,
                                                  limit_docs=limit_docs)
        return cls(r, get_generator(generator, **gen_kw), k=k)

    @classmethod
    def from_texts(cls, texts, generator: str = "extractive", k: int = 4,
                   target: int = 900, **gen_kw):
        chunks = []
        for i, t in enumerate(texts):
            for j, ch in enumerate(chunk_text(t, target=target)):
                chunks.append(Chunk(text=ch, doc_id=str(i), title=f"doc {i}",
                                    start=j))
        return cls(BM25Retriever(chunks), get_generator(generator, **gen_kw), k=k)

    # --------------------------------------------------------------------- ask
    def retrieve(self, query: str, k: int | None = None):
        return self.retriever.search(query, k=k or self.k)

    def ask(self, query: str, k: int | None = None) -> Answer:
        hits = self.retrieve(query, k=k)
        return self.generator.answer(query, hits)
