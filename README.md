# GM-GPT â€” a small, honest LLM rebuilt from the GitMonsters repos

This repository contains an **independent analysis and clean-room re-implementation**
of the language-modeling ideas found in the [GitMonsters](https://github.com/GitMonsters)
GitHub repositories (41 repos audited; see `ANALYSIS.md` for the full report).

The flagship repo, `octotetrahedral-agi`, ships a 52.9M-parameter "AGI" language model
that reports an implausibly good perplexity of **1.32** and states that it **cannot
generate text**. The analysis in `ANALYSIS.md` shows both facts have the same cause:

> **The upstream attention layer has no causal mask.** Position *i* attends to the
> input embedding of position *i+1*, which contains the very token the loss asks the
> model to predict. The low perplexity is target leakage, not learning â€” and because
> no causal mask exists anywhere, autoregressive generation has no valid procedure.
> Their "generative wall" is a missing mask, not a structural limit.

This build fixes that, keeps their salvageable ideas, drops the rest, and trains a
small model on their own corpus with honest, causal evaluation.

## What was salvaged (and what changed)

| Upstream idea | Where it lives now | Change made |
|---|---|---|
| Gaussian per-head relative ("tetrahedral") attention bias | `gmllm/modules.py::CausalSelfAttention` | **Made causal.** Bias is a log-Gaussian falloff over lag with a per-head scale from near (head 0, sharp) to far (last head, flat); applied with a strict causal mask. |
| "Torus" positional encoding on two circular bands | `gmllm/modules.py::TorusRoPE` | Upstream phase was `2Ï€(p mod M)/M` â€” exactly periodic, so positions `p` and `p+M` got *identical* codes. Rewound band B at the golden ratio (irrational) and applied as rotary (relative) rotations: no aliases. |
| Fractional-order GrÃ¼nwaldâ€“Letnikov convolution bank ("octolimb" mixer, from parallplexity) | `gmllm/modules.py::FractionalLimbMixer` | Upstream applied a centered (non-causal) kernel, frozen under `@torch.no_grad()`, never registered in the graph â€” it contributed nothing. Now strictly causal, end-to-end trainable fractional orders, and streamable via a rolling history so KV-cached decoding is exact. |
| World-model auxiliary loss | `gmllm/losses.py::CompositeLoss` | Kept at low weight (default Î»_wm = 0.05), off the critical path. |
| Word-level tokenizer that emits `<unk>` on any unseen word (eval corpus was 23% OOV) | replaced by `gmllm/tokenizer.py::ByteBPETokenizer` | Byte-level BPE: **lossless** (decode(encode(x)) == x for any Unicode, whitespace runs included), no `<unk>` ever. |

Dropped as not salvageable: the "11D spacetime-hull" tensor machinery (dimensionally
incoherent, never exercised), the SEO-spam evaluation set, the self-contradictory
`README-GEMINI.md` claims, and 39 repos of duplicated/placeholder code. Details in
`ANALYSIS.md`.

## Layout

```
ANALYSIS.md          repo-by-repo audit of all 41 GitMonsters repos + claims table
REPO_PICK.md         which repo is the best LLM base, measured (answers the
                     "transcendplexity?" question: octotetrahedral-agi, and why
                     TRANSCENDPLEXITY is not a candidate)
CHAT.md              chat model: template, SFT run, streaming server + window,
                     measured results, and the honest limit
DEPLOY.md            audit of a Gemini-generated dashboard (fabricated hardware/
                     RSI layer) and the port that put the real model behind it
RAG.md               retrieval-augmented answering: BM25 retriever, extractive /
                     Ollama / OpenAI generators, measured recall, and the
                     retrieval-vs-generation split that localises the bottleneck
RESULTS.md           measured training/ablation numbers from this build
gmllm/
  tokenizer.py       byte-level BPE (lossless round-trip)
  modules.py         RMSNorm, TorusRoPE, CausalSelfAttention (+tetra bias),
                     FractionalLimbMixer, SwiGLU, Block
  model.py           GMGPT (config, forward, KV-cached generate)
  losses.py          CompositeLoss (LM + world-model + stability)
  data.py            JSONL reading, token streams, batching
  chat.py            chat template, prompt budgeting, stop markers, QA/Dolly readers
                     (incl. load_qa_pairs_silent: no console output, skip ledger)
train.py             training loop with honest (causal) held-out PPL eval
evaluate.py          checkpoint evaluation: in-domain + OOD PPL, samples
leak_ablation.py     trains leaky (upstream) vs causal (fixed) models, compares
sft.py               instruction tuning (assistant-only loss) on Dolly-15k
eval_chat.py         chat evaluation: answer PPL on 2 corpora + verbatim replies
probe_formats.py     chat template vs upstream prompt format, greedy
probe_markers.py     how reachable the end-of-reply marker actually is
probe_atomic.py      same, for the atomic-marker checkpoint
compare_markers.py   subword vs atomic markers head-to-head
measure_stop.py      how often the model ends its own reply (48 sampled trials)
chat_server.py       stdlib SSE chat server (KV-cached streaming generation)
                     + `"stream": false` -> single JSON {"reply": ...}
chat/index.html      the chat window it serves
deploy_node.py       dashboard port: streams the real model behind the
                     {"messages":[...]} -> {"reply":...} contract
deploy_node.html     the page it serves (no fabricated hardware/RSI controls)
build_physics_corpus.py / build_physics_corpus_fast.py   physics slice from a wiki dump
build_physics_corpus_large.py   full-dump physics slice (scaling; thousands of pages)
build_world_corpus.py  general encyclopedia slice (world knowledge, not just physics)
build_physics_qa.py  extractive physics QA from the physics corpus
build_sft_mix.py     Dolly-15k + upsampled physics QA (the re-SFT mix)
mix_corpora.py       merge domains into one shuffled pretrain stream (NAME=PATH[:WEIGHT])
eval_physics.py      held-out physics perplexity, base vs continued-pretrained
eval_physics_chat.py physics-QA answer PPL + verbatim replies (does it answer?)
eval_physics_facts.py physics fact-completion hit-rate (F=ma, units) â€” recall, not PPL
physics_chain.py     physics-LM-PPL at every stage (base -> pretrain -> SFT)
rag/                 RAG core: retriever (BM25 + chunking), generators, pipeline
rag_server.py        stdlib RAG web UI + /api/ask, /api/status (port 8766)
rag_search.py        retrieval-only debugging view (rank + coverage per hit)
eval_rag.py          split retrieval (recall@k, MRR) / answer evaluation
tests/               71 tests incl. causality invariance, exact KV-cache check, RAG, silent pairs, dashboard
runs/                checkpoints + history + ablation + chat + physics results
validate.py          end-to-end validation of a trained checkpoint (11 checks)
demo/index.html      self-contained visual demo (open in any browser)
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate     # pip install torch numpy (CPU is fine)
pip install -r requirements.txt

python -m pytest tests -q          # 71 passed

# train on the upstream GitMonsters corpus (downloaded to %TEMP%\opencode\gitmonsters)
python train.py --data "%TEMP%/opencode/gitmonsters/octotetrahedral-agi/data/combined_train.jsonl" \
    --out runs/small --steps 600 --batch 8 --ctx 256 --d-model 256 --n-layer 4

python evaluate.py --checkpoint runs/small/best.pt \
    --text-data ".../combined_train.jsonl" --ood ".../eval_heldout.jsonl"

# reproduce the leakage diagnosis
python leak_ablation.py --data ".../combined_train.jsonl" --steps 300

# turn it into a chat model and talk to it (see CHAT.md)
#   --atomic-markers makes the control markers single tokens: the model goes
#   from never ending its own reply to ~65% of sampled replies (CHAT.md Â§8)
python sft.py --base runs/small/best.pt --data ".../dolly.jsonl" \
    --out runs/chat_atomic --atomic-markers --steps 600 --batch 8 --ctx 320 --threads 4
python chat_server.py --checkpoint runs/chat_atomic/best.pt --port 8765
#   -> http://127.0.0.1:8765/

# factual Q&A that actually works: retrieval over documents (see RAG.md)
python rag_server.py --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" \
    --port 8766 --limit-docs 4000          # add --generator ollama for LLM answers
#   -> http://127.0.0.1:8766/
```

## The two tests that matter

`tests/test_model.py` encodes the central claims:

- **`test_causality_invariance`** â€” logits at position *i* must not change when the
  future tokens are perturbed. This is the property upstream violated.
- **`test_leak_mode_sees_the_future`** â€” with `causal=False` the leak *does* appear,
  proving the ablation reproduces the upstream bug faithfully rather than strawmanning it.
- **`test_kv_cache_matches_full_forward`** â€” incremental decoding (attention KV cache
  + fractional-mixer history) is bit-for-bit consistent with the full forward pass.

## Honest evaluation

Upstream reported teacher-forced perplexity under non-causal attention (â‰ˆ1.32). That
number is meaningless for an autoregressive LM. Everything measured here is **strictly
causal**: the mask is on, position *i* predicts token *i+1* from tokens â‰¤ *i* only,
and the same code path is used for training, evaluation, and generation.
See `RESULTS.md` for measured numbers from this build.

## The chat layer

`sft.py` instruction-tunes the base checkpoint on Dolly-15k with an
assistant-only loss, `chat_server.py` streams replies token-by-token, and
`chat/index.html` is the window you talk to.  Measured on held-out data, replies
got **3.8Ã— less surprising** (answer-token PPL 189 â†’ 50.0) and the model learned
the conversation format â€” but at 3.29 M parameters it is **not** a model that
can hold a conversation, and `CHAT.md` Â§6 shows the measured reason (the
control markers are not reachable tokens) rather than hiding it.  Read
`CHAT.md` before assuming the demo is smarter than it is.

## The retrieval layer (facts)

The training experiments end at a measured wall.  `RESULTS.md` Â§3e: a 40-question fact
probe scored **0/40** for *every* checkpoint and *every* corpus (base, physics-only,
world-mix, and the Â§3d preservation fix).  Perplexity moves; facts do not.  At 3.3 M
parameters the model learns **distributions, never recall**:

```
"Abraham Lincoln was the ___"   ->  "the the the"
"The capital of France is ___"  ->  "the the"
```

So the repo does not pretend the LM can answer fact questions, and instead adds the path
that genuinely works â€” **retrieval**, in `rag/`:

| | this project's LM | RAG over documents |
|---|---|---|
| "what is the SI unit of force" | `0/40` fact probe | **correct** (`â€¦the newtonâ€¦`), with a citation |

Design: Okapi **BM25** over sentence-aligned, overlapping chunks (no downloads,
deterministic), an **extractive** answerer that can only quote â€” so it cannot hallucinate â€”
and a relevance gate that **refuses** ("I don't know") when the corpus cannot answer.
`OllamaGenerator` / `OpenAIGenerator` plug in a real LLM for paraphrased answers behind the
same `Retriever`/generator seam.

Measured (`eval_rag.py`, 6167-page enwiki physics corpus, first 4000 docs = 70 445 chunks,
k=4): **retrieval recall@4 = 100 %** (20 answerable questions), **MRR 0.779**, answers
correct **14/24** (4 deliberately unanswerable cases).  The split is the point: **retrieval
is solved; generation is the bottleneck** â€” which is itself corroboration that facts need a
real generator, not a 3.3 M CPU model.  `RAG.md` also documents a genuine lexical failure
(*"capital of France"* â†’ *"History of Belgium"*, Paris ranked 5th) rather than hiding it.

