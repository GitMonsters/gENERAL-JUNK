# CHAT.md — turning GM-GPT into a chat model (and what actually happened)

This document covers the conversational layer: the chat template, the SFT run,
the streaming server, the chat window, the measured results, and the honest
limits.  Everything below was run on this machine (CPU, 4 threads,
torch 2.14.1+cpu) and the numbers are reproducible from the checked-in
commands.

**Bottom line:** the pipeline is real and end-to-end working — a byte-BPE chat
template, an instruction-tuning run with assistant-only loss, a streaming
server with a KV cache, and a browser chat window.  The *model* improved in a
measured way (**held-out reply PPL 189 → 50.0**, i.e. answers became ~3.8× less
surprising and it learned the conversation format), but **it did not become a
model that can hold a conversation**.  Greedy replies collapse to the
cross-entropy-optimal string `the the the the ...` within a few tokens.  Section
6 shows the measured reason, and why that reason is consistent with the
upstream repo's own report.  Section 8 records the follow-up fix that was then
attempted: making the control markers atomic single tokens took **self-
termination from 0/48 to 31/48 sampled replies**, but left the replies — and the
greedy argmax — unchanged.

---

## 1. The chat format

`gmllm/chat.py` is the single source of truth: the SFT trainer renders training
rows with it and the server renders prompts with it, so there is no
train/serve skew.

```
<|system|>You are GM-GPT, a small, honest assistant.<|end|>
<|user|>What is the capital of France?<|end|>
<|assistant|>The capital of France is Paris.<|end|>
<|user|>And of Italy?<|end|>
<|assistant|>
```

* `render_training_example()` builds a training row.
* `split_prompt_and_answer()` cuts at the **last** `<|assistant|>`, so the loss
  can be masked to the reply only (prompt tokens get label `-100`).
* `render_prompt()` builds a runtime prompt ending in `<|assistant|>`.
* `messages_to_prompt_with_budget()` drops the oldest turns until the prompt
  fits the context window, never dropping the newest.
* `strip_stop()` cuts generation at the first control marker.

The markers are ordinary text: the tokenizer is byte-level BPE, so in the
**default** build `<|end|>` is just 5 byte-level pieces.  That detail turns out
to matter a great deal — see §6.  Section 8 re-runs the whole stage after
promoting the markers to single atomic tokens.

## 2. Instruction tuning (`sft.py`)

```bash
python sft.py --base runs/small/best.pt \
              --data "%TEMP%/opencode/dolly.jsonl" \
              --out runs/chat --steps 600 --batch 8 --ctx 320 --threads 4
```

| setting | value |
|---|---|
| base checkpoint | `runs/small/best.pt` (step 99, held-out document PPL 188.4) |
| data | Dolly-15k (`databricks-dolly-15k`, 12.5 MB) |
| usable conversations | 7,686 (rows longer than 320 tokens or with <3 answer tokens dropped) |
| split | 7,486 train / 200 held-out |
| objective | cross-entropy on **assistant tokens only** (prompt masked) |
| optimiser | AdamW, cosine 3e-4 → 3e-5, 60 warmup steps, wd 0.05, grad-clip 1.0 |
| budget | 600 steps ≈ 1791 s (~30 min CPU), ≈ 2.7 s/step |

Held-out **answer-token** PPL during the run (`runs/chat/history.json`):

| step | 99 | 199 | 299 | 399 | **499** | 599 |
|---|---|---|---|---|---|---|
| answer PPL | 90.8 | 93.6 | 55.9 | 77.9 | **50.7** | 62.5 |

`best.pt` = step 499 (PPL 50.65).  It is noisy but the trend is down over the
whole run; the train loss went 5.11 → ~3.9.  (The §8 atomic-marker rerun of
this same command — `--atomic-markers`, `runs/chat_atomic` — lands at PPL 53.35
at step 499; see §8.)

## 3. Measured result (`eval_chat.py`)

`runs/chat_eval.json`, 500 held-out rows per corpus, identical splits for both
checkpoints:

| checkpoint | Dolly answer PPL | GitMonsters QA answer PPL |
|---|---|---|
| base `runs/small/best.pt` (step 99) | 189.0 | 225.5 |
| **SFT `runs/chat/best.pt` (step 499)** | **50.0** | **47.9** |
| frequency baseline (predict the most common answer token) | 4.6 M | 3.8 M |

Two honest readings of that table:

1. **The SFT helped a lot.** Replies are 3.8× less surprising than the base
   model's, and 4–5 orders of magnitude better than a frequency baseline.  The
   model demonstrably learned turn structure, the marker vocabulary, and the
   register of Dolly answers (title-case openings, commas, sentence-final
   periods).
2. **It is still far from a chat model.** The same samples that produce that
   PPL are visibly not answers — see §5.  A low PPL over ~50k answer tokens
   says "these token statistics look much more like replies", not "these are
   replies".

## 4. The chat server and window

```bash
python chat_server.py --checkpoint runs/chat/best.pt --port 8765 --threads 4
# then open http://127.0.0.1:8765/
```

* **Server** (`chat_server.py`) — stdlib only, `ThreadingHTTPServer`, no
  framework.  `GET /` serves the window, `GET /api/status` reports model
  metadata, `POST /api/chat` streams **Server-Sent Events**, one event per
  token: `data: {"t": "..."}` … `data: {"done": true, "tokens": n, "tok_s": k}`.
  Generation reuses the model's KV cache (attention keys/values *and* the
  fractional-mixer history), applies repetition penalty / top-k / top-p, and
  stops on any control marker.  Requests are serialised behind a lock, and a
  disconnected client (stop button) is caught and cleaned up.
* **Window** (`chat/index.html`) — dark chat UI: streaming bubbles with a live
  caret, typing indicator, stop button, new-chat reset, suggestion chips, a
  settings drawer (temperature / top-k / max reply tokens / repetition
  penalty, persisted to `localStorage`), a status pill fed by `/api/status`,
  and a clear error banner if the server is not running.
* **Measured throughput:** ~31–56 tokens/s on CPU with 2–4 threads, first
  token in well under a second.

Verified end-to-end: `GET /` 200 + 17.8 kB page, `/api/status` JSON, SSE stream
parsed token-by-token, multi-turn history, malformed bodies rejected with
`400` + JSON, stop-on-disconnect logged as a clean reset rather than a crash.

## 5. Samples (verbatim, `runs/chat_eval.json`)

Base model, `temp 0`, "What is the capital of France?":

```
the the the the      the    the the the thethe the  thethe the the the           the the      the
```

SFT model, `temp 0`, same prompt:

```
the the the the the the the in the the the the the the the the the the the the the the the the the the the the the the
```

SFT model, `temp 0.9`, "Why is the sky blue?":

```
Ran A bicalr for and  lia bpg Tend||>A mo tons
is in dsss a a lsa the ssam
```

Note the shape of that second sample: plausible English-ish capitalisation and
spacing, and then `<|end|>`-shaped *soup* — `Tend||>`, `cend||||ic`,
`<end>is`.  The model is reaching for the control marker, and failing to spell
it.  §6 explains why.

## 6. Why replies are not coherent — the measured reason

Not a template or wiring bug.  Three probes pin it down:

**(a) The prompt format is not the problem.** `probe_formats.py` compares the
chat template against the plain `Instruction: Q\nResponse:` format that
dominates the upstream instruction corpus the base model was trained on.
Greedy, both formats:

| prompt | reply (greedy, 48 tokens) |
|---|---|
| chat template | `the the the in the the the the the ...` |
| `Instruction: ...\nResponse:` | `    in  the the the the the c the o an the an in an the co the the ...` |

Same failure both ways, so the model has no usable continuation on either.
(That second reply is also the same class of output as the upstream repo's own
`finetune_chat.py` result, recorded in their `RESULTS.md` as
*"chat ft — 52.9M — 1.91 chat loss; no answer generation"*.)

**(b) The stop marker is unreachable under greedy decoding.** `probe_markers.py`
on a chat prompt: `<` = token 60, and it is **rank 335–340 of 512** with
softmax probability ≈ 2×10⁻⁵, while `the` (id 258) is always rank 1.  For a
reply to ever end with `<|end|>` the model's own argmax must take that step; it
essentially never does.

**(c) The pieces of the markers lose the ranking race individually.**  The
byte-BPE vocab splits `<|end|>` into 5 pieces and `<|assistant|>` into 7:

| marker | pieces | sum of logits | best individual rank |
|---|---|---|---|
| `<|end|>` | `<` `|` `end` `|` `>` | **−4.29** | 68 (`|`) |
| `<|assistant|>` | `<` `|` `ass` `ist` `ant` `|` `>` | **−5.16** | 40 (`<`) |

The *sum* is competitive with real words, but the loss only ever rewards
`end` (for `<|end|>`), never the `<` that must come first, so no piece is
individually favoured and whole words (the / and / is) win every step.  During
sampling the model dribbles out marker-shaped fragments in the wrong order —
exactly the soup in §5.

The training data explains the ceiling: 7,686 conversations, ~1.6 M supervised
tokens, run against a **3.29 M-parameter** model whose base checkpoint had seen
0.6 of an epoch of a domain-narrow web corpus.  That is roughly an order of
magnitude less data than such a model needs before replies stop being
distribution-matching.  The upstream repo reached the same wall at 52.9 M
parameters.

## 7. A shipped decoding workaround (and why it is only a workaround)

Because the markers are unreachable anyway, `chat_server.token_ban_list()`
bans their pieces during generation.  They can then neither appear in a reply
nor start one *incorrectly* — and since the model cannot spell a stop marker at
all, a token budget (`max_tokens`) is the natural stop condition.  The UI sets
`max reply tokens` explicitly for this reason.

This is a **decoding normalisation, not a quality fix**: it removes the marker
soup, it does not add understanding.  It is documented here so nobody mistakes
the cleaned-up output for a smarter model.  (It is also no longer used: the
atomic-marker checkpoint of §8 switches it off — see below.)

## 8. The atomic-marker fix — attempted, measured, and what it did

§6(c) identified a concrete, mechanical defect: the control markers were split
by byte-BPE into pieces whose *sum* was competitive but whose *individual* ranks
never were, so the model could not terminate a reply even when it "wanted" to.
§8.1 said to make the markers atomic and re-run SFT.  **That was done.**  This
section records exactly what changed.

**Implementation.**  `<|system|> <|user|> <|assistant|> <|end|>` are appended
as atomic vocabulary entries at stable ids `n_merges + index` (512–515 at vocab
516).  `ByteBPETokenizer.encode()` matches them longest-first and stops word
scanning at marker boundaries, so they are **lossless** even glued to text
(`a<|end|>b` round-trips).  `GMGPT.resize_vocab()` grows the embedding and LM
head, copying every learned row and re-tying the weights, so the 512-vocab base
checkpoint is reused without retraining it.  Checkpoints saved by
`sft.py --atomic-markers` carry their own enlarged `tokenizer.json`.  The server
detects an atomic checkpoint and **stops banning markers** — because the marker
is now exactly the token we want the model to be able to predict — and stops on
the atomic id instead.

```bash
python sft.py --base runs/small/best.pt --data "%TEMP%/opencode/dolly.jsonl" \
              --out runs/chat_atomic --atomic-markers \
              --steps 600 --batch 8 --ctx 320 --threads 4    # ~28 min CPU
```

**Result.**  Best answer-token PPL **53.35** at step 499
(`runs/chat_atomic/history.json`) — statistically the same as the subword run's
50.65.  Three probes, all on the same fixed prompts
(`probe_atomic.py`, `compare_markers.py` → `runs/atomic_markers.json`,
`measure_stop.py` → `runs/stop_rate.json`):

| measurement | subword `runs/chat` | **atomic `runs/chat_atomic`** | reading |
|---|---|---|---|
| held-out Dolly / upstream answer PPL | 50.0 / 47.9 | 53.9 / 56.0 | unchanged (within noise) |
| `<|end|>` greedy rank | 332–340 of 512 | **66–251** | 2–5× higher |
| `<|end|>` logit | −3.7 … −4.1 | **−1.35 … +0.30** | ≈ 2.7 nats more competitive |
| `<|end|>` wins argmax (greedy) | never | still **never** | not fixed |
| **stops its own reply when sampled** | **0 / 48** | **31 / 48 (64.6%)** | **fixed** |
| sampled reply text | word soup | word soup | unchanged |

**Honest verdict: a real but partial fix.**  Making the marker one token moved
it from "effectively impossible" (p ≈ 2×10⁻⁵) to "genuinely competitive"
(p ≈ 0.02–0.05), and that was enough to change **stopping behaviour
completely**: under the server's actual sampling settings (temp 0.7, top-k 40,
top-p 0.95) the subword model *never* ended its own reply, and the atomic model
ends **~65% of them**.  The §6(c) mechanism is confirmed: the ceiling on
stopping was representational, and the fix removed it.

**But the greedy/argmax case is still not fixed** (`E` never ranks 1st), and —
more importantly — **the replies are still incoherent**.  Self-termination is a
format win, not an intelligence win: the model now ends a sentence-shaped
non-answer instead of continuing until the budget.  That ceiling is §8.2, not
§6(c): *the amount of data per parameter*, which tokenizer surgery cannot move.

Remaining levers, unchanged in order of payoff:

1. **Raise the data-to-parameter ratio.**  Even with perfect markers, ~1.7 M
   supervised tokens on a 3.3 M model will not buy reasoning.  More instruction
   data (or a much smaller model) moves the ceiling; temperature does not.
2. **Train the base model longer on more text**, then SFT.  The base was
   stopped at step 99 of a 600-step run to avoid overfitting a small corpus;
   the chat stage inherits that weak prior.
3. **Accept the shape of the result.**  At this scale a real product uses a
   retrieval layer (the upstream repo's own answer: retrieve and rank, rather
   than generate) — a different system, not a better checkpoint.

## 9. Physics world knowledge (pretrain → re-SFT)

Everything above trained on the GitMonsters corpus, which turns out to contain almost
**no real physics** ("force" = military force, "mechanics" = game mechanics).  To give
the model a physics world model we built a **physics prose corpus** and did a
continued pretrain, then re-ran SFT on a Dolly + physics-QA mix.  Full detail and the
corpus/overfit story are in `RESULTS.md` §3b/§3c; the headline:

- **Corpus:** 378 clean physics/science pages, 2.04 M chars, from a bounded 60 MB
  Simple-English-Wikipedia prefix (`build_physics_corpus_fast.py`).
- **Continued pretrain works only if gentle:** LR 3e-3/400 steps *overfit*
  (physics-PPL 228.6 → 996.8); LR 2e-4/120 steps **halved it** (228.6 → **129.4**,
  best at step 19).  `train.py --init` is the mechanism.
- **Re-SFT** (`runs/chat_physics`) starts from the physics base and adds physics QA to
  the mix (11% physics, upsampled ×5) so the answer style is exposed; see §9.1 for what
  the model actually says.

### 9.1 Does it answer physics questions?  No — and here is the measurement.

Measured with `eval_physics_chat.py` (physics-QA answer PPL + verbatim replies) and
`physics_chain.py` (physics *LM* PPL at each stage):

| stage | physics-**LM**-PPL | physics-**QA answer**-PPL |
|---|---|---|
| base `runs/small/best.pt` (no physics) | 228.6 | — |
| **`runs/physics2/best.pt`** (continued pretrain) | **129.4** (1.77×) | — |
| `runs/chat_atomic/best.pt` (SFT, pre-physics) | 352.0 | 47.8 |
| `runs/chat_physics/best.pt` (SFT from physics base) | **363.9** | **49.2** |

**The physics pretrain works on the base LM and is then destroyed by SFT.**  Both SFT
checkpoints model physics prose worse than the un-touched base (352/364 vs 228.6), and
the physics-based re-SFT is no better than the pre-physics one (363.9 vs 352.0).  Physics-QA
answer PPL is likewise unchanged-to-worse (49.2 vs 47.8), and replies are still word-soup
in both.  Mechanism: assistant-only loss on short Dolly answers narrows the model away
from general next-token prediction, and the 11% physics mix is too small to hold it.

Honest conclusion at 3.3 M params / 2 MB physics: **pretraining learns physics,
instruction tuning throws it away.**  See `RESULTS.md` §3b/§3c for the full chain and
`runs/physics_chat_eval.json` for every verbatim reply.

#### 9.2 The fix — auxiliary LM loss during SFT (and yes, it works)

A diagnosis like the one above is only half a result, so §3d of `RESULTS.md` turns it
into a fix.  `sft.py --lm-data physics_corpus.jsonl --lm-weight 0.5` adds a plain causal
next-token loss on physics prose to every SFT step, so the assistant-only objective can no
longer overwrite the physics distribution.  `runs/chat_physics_lm/best.pt` is the result
(500 steps, everything else identical to `runs/chat_physics`).

Measured on **held-out** enwiki physics (pages no checkpoint trained on):

| stage | physics-PPL (held-out) | physics-**QA answer**-PPL |
|---|---|---|
| `runs/physics2/best.pt` (pretrain) | 130.6 | — |
| `runs/chat_physics/best.pt` (SFT, no fix) | 376.6 | 48.9 |
| **`runs/chat_physics_lm/best.pt` (SFT + aux LM)** | **37.1** | 55.0 |

So physics *prose* survives — and improves 10×.  The two objectives did **not** fight: the
auxiliary LM term fell 5.21 → 3.59 across the run while the masked answer loss fell
6.86 → 6.09, and the chat `val` PPL (63.3) is in the same range as the no-fix run (56.7).
**This is the reference configuration now**: never SFT an LM on assistant-only loss without
an LM term.

**What did *not* improve:** physics-**QA answer** PPL is *worse* with the fix (55.0 vs
48.9), the replies are still word-soup, and the fact probe (`eval_physics_facts.py`) is
still **0/40** — the model cannot complete *"The SI unit of force is the ___" → newton*.
Preserving the physics *distribution* is not the same as knowing physics *facts* or
answering questions; at 3.3 M params only the distribution is reachable.  The decisive
limit is capacity, not the training recipe (see `RESULTS.md` §3d and §4).

## 10. Reading the upstream QA pairs silently

`load_qa_pairs_silent` (`gmllm/chat.py`) reads the upstream GitMonsters instruction
corpus with **no console output at all**, for embedding in tools that must not write
to stdout — and it is *accountable* about what it discards, which the plain loaders are
not (`read_jsonl` swallows broken JSON, `parse_qa_line` returns `None`, both silently).

```python
from gmllm.chat import load_qa_pairs_silent
pairs, stats = load_qa_pairs_silent(paths)
# pairs: [(question, answer), …]
# stats: {"files":2, "lines":13769, "nonblank":13769, "json_objects":13769,
#         "bad_json":0, "missing_text":0, "unparsed":0, "kept":13769}
```

Measured on the real corpus (`instructions.jsonl` + `instructions_v2.jsonl`):
**0 bytes written to stdout, 0 to stderr, 13 769 pairs read, 0 skipped** — both the
legacy `Instruction:/Response:` and v2 `Question :/Response :` spellings. Silence is
not blindness: the ledger accounts for every non-blank line
(`kept + bad_json + missing_text + unparsed == nonblank`), so a lossy corpus is
visible. A *missing file* still raises `FileNotFoundError` — silence covers noise,
not caller errors. `tests/test_chat_pairs.py` captures stdout/stderr and asserts they
stay empty, on both clean and deliberately-corrupted input.

## 11. Reproducing

```bash
# 1. base LM (see README.md)
python train.py --data "%TEMP%/gitmonsters/octotetrahedral-agi/data/combined_train.jsonl" \
                --out runs/small --steps 600 --batch 8 --ctx 256 --d-model 256 --n-layer 4

# 2. instruction tuning        (~30 min CPU)
python sft.py --base runs/small/best.pt \
              --data "%TEMP%/opencode/dolly.jsonl" \
              --out runs/chat --steps 600 --batch 8 --ctx 320 --threads 4

# 2b. the §8 atomic-marker rerun  (~28 min CPU)
python sft.py --base runs/small/best.pt \
              --data "%TEMP%/opencode/dolly.jsonl" \
              --out runs/chat_atomic --atomic-markers \
              --steps 600 --batch 8 --ctx 320 --threads 4

# 3. measure it                (~4 min CPU)
python eval_chat.py --checkpoints "base=runs/small/best.pt,subword=runs/chat/best.pt,atomic=runs/chat_atomic/best.pt" \
                    --n 500 --out runs/chat_eval.json

# 4. talk to it
python chat_server.py --checkpoint runs/chat_atomic/best.pt --port 8765 --threads 4
#    -> http://127.0.0.1:8765/   (status pill shows "atomic markers")

# 5. probes used in sections 6 and 8
python probe_formats.py runs/chat/best.pt
python probe_markers.py runs/chat/best.pt
python probe_atomic.py  runs/chat_atomic/best.pt      # marker reachability
python compare_markers.py                             # subword vs atomic
python measure_stop.py                                # self-stop rate (48 trials)

# 6. tests (41)
python -m pytest tests -q

# 7. physics world knowledge (CHAT.md §9, RESULTS.md §3b/§3c)
python build_physics_corpus_fast.py --cap-mb 60     # -> physics_corpus.jsonl (378 pages)
python build_physics_qa.py                          # -> physics_qa.jsonl
python build_sft_mix.py --repeat 5                  # -> sft_physics_mix.jsonl
python train.py --init runs/small/best.pt \
                --data "%TEMP%/opencode/physics_corpus.jsonl" \
                --out runs/physics2 --steps 120 --batch 8 --ctx 256 \
                --lr 2e-4 --min-lr 2e-5 --warmup 15 --threads 4
python eval_physics.py --checkpoints "base=runs/small/best.pt,p2=runs/physics2/best.pt"
python sft.py --base runs/physics2/best.pt \
              --data "%TEMP%/opencode/sft_physics_mix.jsonl" \
              --out runs/chat_physics --atomic-markers \
              --steps 500 --batch 8 --ctx 320 --threads 4
python eval_physics_chat.py --checkpoints "pre=runs/chat_atomic/best.pt,phys=runs/chat_physics/best.pt"
python physics_chain.py                             # physics-LM-PPL at every stage

# 8. the §9.2 physics-preservation fix (aux LM loss during SFT; RESULTS.md §3d)
python sft.py --base runs/physics2/best.pt \
              --data "%TEMP%/opencode/sft_physics_mix.jsonl" \
              --out runs/chat_physics_lm --atomic-markers \
              --lm-data "%TEMP%/opencode/physics_corpus.jsonl" \
              --lm-weight 0.5 --lm-batch 4 --lm-ctx 256 \
              --steps 500 --batch 8 --ctx 320 --threads 4
# held-out physics probe (enwiki pages no checkpoint trained on)
python physics_density_probe.py --url https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
                               --cap-mb 120 --label enwiki
python eval_physics.py --data "%TEMP%/opencode/physics_probe_enwiki.jsonl" --n 400 \
       --checkpoints "base=runs/small/best.pt,p2=runs/physics2/best.pt,sft=runs/chat_physics/best.pt,fix=runs/chat_physics_lm/best.pt"
python eval_physics_facts.py --checkpoints "fix=runs/chat_physics_lm/best.pt" --out runs/physics_facts_lm.json
```

Artefacts: `runs/chat/best.pt` (step 499), `runs/chat/final.pt` (step 599),
`runs/chat/history.json`, `runs/chat_eval.json` (all numbers + verbatim
samples); `runs/chat_atomic/{best,final}.pt` + `tokenizer.json` (516 vocab),
`runs/atomic_markers.json`, `runs/stop_rate.json`; `runs/chat_physics_lm/best.pt`
(step 499, aux-LM preservation; `history.json` has the per-step `lm` column),
`runs/physics_facts{,_lm}.json`, `runs/physics_enwiki_holdout.json`.
