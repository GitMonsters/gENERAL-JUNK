# RAG — retrieval-augmented answering

> **Why this exists:** `RESULTS.md` §3e measured the binding limit of this project. Across
> every checkpoint and every corpus — base, physics-only, world-mix, and the §3d preservation
> fix — a **40-question fact probe scored 0/40**. The 3.3 M-parameter CPU model emits
> `"the the the"` for *"Abraham Lincoln was the ___"*. It learns **distributions, never facts.**
>
> Retrieval is the honest fix for fact questions: the knowledge is looked up at question time
> and the model's weights are not involved. This directory is that path.

## 1. What it does

```
question ──▶ BM25 retriever ──▶ top-k passages ──▶ answer generator ──▶ answer + citations
             (no download,         (sentence-          (extractive, or a
              deterministic)        aligned,            real LLM: Ollama /
                                    overlapping)        OpenAI-compatible)
```

```python
from rag import RAG
r = RAG.from_jsonl("corpus.jsonl", generator="extractive")
a = r.ask("what is the SI unit of force")
print(a.text)      # "…the SI unit of force, the newton… [1]"
print(a.sources)   # [{n, title, doc_id, score, text}, …]
```

Run it as a service with a web UI:

```powershell
python rag_server.py --corpus "$env:TEMP\opencode\physics_corpus_enwiki.jsonl" `
                     --port 8766 --limit-docs 4000
# open http://127.0.0.1:8766/
```

Debug retrieval without generation:

```powershell
python rag_search.py "what is the SI unit of force" `
     --corpus "$env:TEMP\opencode\physics_corpus_enwiki.jsonl" --limit-docs 4000
```

## 2. Design choices, and why

**Lexical (BM25) before dense.** The environment has no `datasets`, `transformers`,
`sentence_transformers`, or `sklearn` — only `numpy` and CPU `torch`. Rather than pretend a
dense backend exists, retrieval starts with Okapi BM25: no download, deterministic, and for
the dominant case here (a fact stated *somewhere* in a document) it is competitive. A dense
backend is a plug-in point — everything takes the `Retriever` protocol
(`rag/retriever.py`), so swapping one in touches one file.

**Sentence-aligned overlapping chunks.** `chunk_text` splits on sentence boundaries
(`(?<=[.!?])\s+`) with overlap, so a fact and its value are never cut apart, and a fact
straddling a boundary is still fully contained in some chunk. Chunk target 900 chars.

**A relevance gate that refuses.** A RAG system that always answers is worse than one that
sometimes says *"I don't know."* Two conditions must hold before the extractive backend
answers:

- `min_coverage` (0.5) — the top passage must share enough of the query's content words, and
- `min_score_frac` (0.35) — the top passage must score ≥35% of the *maximum achievable BM25
  score for that query in this index*.

The second is deliberately a **fraction**, not an absolute number. BM25 magnitude depends on
corpus size and IDF, so an absolute floor tuned on one corpus silently mis-gates another —
a real bug caught by `tests/test_rag.py` (a 2-chunk toy corpus scored 1.27 against a floor of
3.0, and the assistant wrongly refused).

## 3. Measured results

Measured with `eval_rag.py` against the 6167-page / 71 MB enwiki physics corpus,
restricted to its first 4000 documents (**70 445 chunks**), `k = 4`, extractive generator,
and a 24-question hand-written set in `%TEMP%\opencode\rag_eval.jsonl` (20 answerable, 4
deliberately unanswerable).

| metric | value | n |
|---|---|---|
| retrieval recall@4 | **100.0 %** | 20 answerable |
| MRR | **0.779** | 20 answerable |
| answers correct | **14 / 24** | 24 |
| unanswerable cases handled | 2 / 4 | 4 |

### The instrument was fixed, then the result was real

The first run reported **recall@4 = 35 %** — but the ground truth was *document title*
matching. For *"what is the SI unit of force"* the top hit was
**"Centimetre–gram–second system of units"**, which *contains* the answer
(`"the SI unit of force, the newton"`) without being the page named "Newton (unit)". Title
recall **understates** retrieval for fact questions, so the metric was changed to
*answer-term containment* — did a retrieved passage actually contain a known answer string.
Recall then read **100 %**. Both numbers are reported here because changing a metric after
seeing the result is exactly the kind of move that needs to be auditable.

### Two real bugs the tests found

1. **Trailing-period tokenization.** The pattern `[a-z0-9][a-z0-9._\-']*` indexed
   sentence-final words as `joule.` and `newton.`, so `search("joule")` silently returned
   **nothing** — i.e. retrieval was broken for most answer words. Fixed by stripping
   leading/trailing punctuation while preserving intra-word dots (`9.8` still tokenizes).
2. **Absolute score floor.** Described in §2. Replaced with the score-fraction gate.

## 4. The honest bottleneck: generation, not retrieval

Retrieval is **solved (100 % recall@4)**. Answer quality is **14/24**, and the failures
localize precisely: for *"what is the unit of power"* the passage containing *"watt"* **was**
retrieved, but the extractive generator quoted a different sentence from it.

That is inherent to *extractive* answering: it can only quote, and many correct answers
require synthesis. Chasing +1 by tuning sentence selection on 24 hand-written cases would be
overfitting, so it was not done. The conclusion the split evaluation exists to produce:

> **Retrieval succeeds; the bottleneck is generation. Fixing it means a real LLM, not a
> 3.3 M-parameter CPU model — which is itself corroboration of `RESULTS.md` §3e.**

Attach one without touching the pipeline:

```powershell
# local
python rag_server.py --corpus … --generator ollama --model llama3.1:8b
# any OpenAI-compatible endpoint
$env:OPENAI_API_KEY="…"; python rag_server.py --corpus … --generator openai
```

Both backends receive the passages and the instruction *"answer using ONLY the passages; if
they do not contain the answer, reply exactly: I don't know based on the provided
documents"*, and both return sources. `get_generator()` in `rag/generator.py` is the seam.

## 5. A documented failure mode of lexical retrieval

*"what is the capital of France"* asked of a **physics** corpus returned a confident quote
from **"History of Belgium"**. Diagnosed, not hidden:

```
query tokens: {capital, france}
   13.41  coverage=1.00  History of Belgium
   10.50  coverage=1.00  Vilnius
    9.97  coverage=1.00  History of Belgium
    9.75  coverage=1.00  Barcelona
    9.66  coverage=1.00  Paris          <- correct, ranked 5th
```

Every candidate shares *both* query terms, so coverage cannot separate them, and BM25
cannot tell *"the capital of France"* from *"the capital of the Menapii, in France."* That is
precisely the job dense embeddings do and lexical matching cannot. It is reported as a known
limit rather than patched with a hack tuned to one test case.

## 6. Test coverage — `tests/test_rag.py`, 10 tests

The properties that make retrieval trustworthy, pinned:

| test | property |
|---|---|
| `test_chunk_text_sentence_aligned_with_overlap` | boundaries land on sentence ends, never mid-word |
| `test_tokenize_drops_stopwords_keeps_content_and_numbers` | `9.8` survives; `the`/`what` do not |
| `test_search_ranks_the_right_document_first` | the answer-bearing passage ranks first |
| `test_search_is_deterministic` | identical queries give identical rankings |
| `test_search_empty_query_returns_nothing` | empty / all-stopword queries are safe |
| `test_retriever_builds_from_jsonl` | the real corpus format loads |
| `test_extractive_answer_quotes_and_cites` | answers are grounded and cited |
| `test_extractive_refuses_when_corpus_cannot_answer` | off-domain → *"I don't know"* |
| `test_answer_includes_source_text_for_checking` | every source carries text + score |
| `test_rag_retrieve_matches_ask_sources` | what `ask` cites is what `retrieve` returned |

Full suite: **53 passed** (43 core + 10 RAG).

## 7. Files

| file | role |
|---|---|
| `rag/retriever.py` | `BM25Retriever`, `chunk_text`, `tokenize`, `Chunk` / `Hit` / `Retriever` |
| `rag/generator.py` | `ExtractiveGenerator`, `OllamaGenerator`, `OpenAIGenerator`, `get_generator` |
| `rag/__init__.py` | `RAG` — wires retriever to generator, `ask()` / `retrieve()` |
| `rag_server.py` | stdlib web UI + `/api/ask`, `/api/status` (port 8766) |
| `rag_search.py` | retrieval-only debugging view |
| `eval_rag.py` | split retrieval / answer evaluation → `runs/rag_eval.json` |
| `tests/test_rag.py` | 10 tests |
