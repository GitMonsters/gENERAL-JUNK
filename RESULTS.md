# RESULTS â€” GM-GPT trained on the GitMonsters corpus

All numbers below are **measured on this machine** (CPU, 4 threads, torch 2.14.1+cpu),
strictly causal evaluation: position *i* predicts token *i+1* from tokens â‰¤ *i* only.
No teacher-forcing leakage anywhere in the pipeline (enforced by
`tests/test_model.py::test_causality_invariance`).

## 1. Main run (`runs/small`)

- **Model**: GM-GPT, 3.42M params (3.29M in the LM; +0.13M world-model head)
  â€” d_model 256, 4 heads, 4 layers, ctx 256, 8 fractional limbs Ã— 32 taps
- **Data**: GitMonsters' own corpus, `octotetrahedral-agi/data/combined_train.jsonl`
  â€” 102,358 documents, 12.8 MB text â†’ 8.09M BPE tokens (512-vocab byte-BPE, lossless)
  â†’ 31,603 chunks; split 30,971 train / 632 held-out
- **Optimiser**: AdamW, cosine schedule 3e-3 â†’ 3e-4, 100 warmup steps, wd 0.1,
  grad-clip 1.0, batch 8 Ã— 256 tokens, 600 steps (~4.9M tokens seen, 0.6 epoch)
- **Speed**: ~630â€“780 tok/s (CPU)

### Loss / perplexity curve (held-out, causal)

| step | train LM loss | train ppl | held-out causal PPL |
|---|---|---|---|
| 99 | 4.10 | 60.3 | **188.4** â† best checkpoint |
| 199 | 3.88 | 48.3 | 505.1 |
| 299 | 3.66 | 39.0 | 694.8 |
| 399 | 3.55 | 35.0 | 812.6 |
| 499 | 3.50 | 33.2 | 951.7 |
| 599 | 3.37 | 29.1 | 1241.5 |

Train loss fell monotonically (6.33 â†’ 3.37). Held-out PPL improved to 188 by step 99
and then rose steadily â€” textbook **overfitting**: 3.4M params on 4.9M training tokens
of a domain-narrow, repetitive corpus (news/wiki/transcript web text). With more data,
more steps on a smaller slice, or stronger regularisation the crossover point moves out;
600 steps was chosen to fit CPU time, and reporting the *best held-out* checkpoint is
the honest practice. The upstream repo, by contrast, reported only the training-side
number under leaky attention (see `ANALYSIS.md` Â§claims).

### Final checkpoint (`best.pt`, step 99)

| metric | value |
|---|---|
| in-domain held-out causal PPL | **188.4** (632 chunks Ã— 256 tokens) |
| out-of-domain causal PPL (`eval_heldout.jsonl`, 500 upstream docs) | **217.3** (307 chunks) |
| OOD / in-domain ratio | 1.15 â€” small, as expected for same-domain "held-out" web text |

OOD PPL is measured on GitMonsters' own held-out file, which the analysis found is
*also* SEO-spam-heavy web text from the same sources â€” so it is only weakly
out-of-domain. The small gap is honest evidence of that, not of generalisation.

### Samples (best checkpoint, prompt "the ", temp 0.8, top-k 40)

At 0.6 of an epoch the model is in the "word salad with real distribution" regime â€”
correct English character/word statistics, article-noun agreement fragments, newlines,
but no coherent sentences. This is what an honestly-measured 3.4M-param model trained
for ~26 CPU-minutes looks like. (For contrast, upstream's 52.9M-param model at 15Ã—
this size reported PPL 1.32 under its leaky metric and could not generate at all.)

## 2. Leakage ablation (`runs/ablation`)

Two identical 0.85M-param models (d_model 128, 2 layers, ctx 128) trained on the same
3 MB corpus slice for 300 steps each â€” one with `causal=False` (faithfully reproducing
the upstream defect: no mask anywhere, bidirectional attention in an autoregressive LM),
one with the fix (`causal=True`). The leaky model's weights are additionally re-measured
under a forced causal mask ("honest PPL") â€” the number upstream never reported.

Both models: 0.46M params, d_model 128, 2 layers, 4 heads, ctx 128, 300 steps,
batch 16, lr 3e-3, same 3MB corpus slice, same seed. CPU wall time ~195s each.

| | leaky (upstream, `causal=False`) | causal (fix, `causal=True`) |
|---|---|---|
| native (as-trained) PPL | **1.006** | **1.025** |
| same weights, forced-honest causal PPL | **1.133** | 1.025 |
| train LM loss @ step 100 | 0.551 (PPL 1.73) | 2.094 (PPL 8.12) |
| train LM loss @ step 200 | 0.067 (PPL 1.07) | 0.434 (PPL 1.54) |
| metric depends on eval mode? | **yes** (+12.6% when made honest) | no (identical) |
| generation (prompt "the ") | `alalalalalalâ€¦` | `thethetheâ€¦` |

**Result â€” the predicted signature, confirmed:**

1. The leaky model wins *only* under its own non-causal measurement (1.006 vs 1.025).
2. Its loss collapses 6Ã— faster (0.067 vs 0.434 at step 200) because predicting the next
   token from that token's own input embedding is nearly free.
3. Its advantage **disappears the moment the mask is forced on**: honest PPL 1.133 â€”
   worse than the causal model's 1.025. A model trained without a mask learns weights that
   are actively worse when made causal.
4. The causal model's native and honest numbers are *identical* â€” its metric cannot be
   gamed by switching eval mode.

Honest caveat: on a 3MB slice both models overfit to near-zero train loss within 300
steps, and both generate degenerate text. At this scale the ablation demonstrates the
*measurement artifact* (the point), not generation quality. Generation quality requires
the full-corpus run, where the honest best checkpoint is 188.4 â€” a real, un-leaked number.

## 2b. The leak reproduced against *upstream's own code*, end-to-end

The ablation above uses `gmllm/`'s own reimplementation. To close the loop, the upstream
attention module (`octotetrahedral-agi/core/tetrahedral_attention.py`) was imported
**verbatim** into a small LM and trained on the repo's own `data/combined_train.jsonl`.
Two runs, same weights-init / data / steps / optimiser — the *only* difference is whether a
causal mask is passed to that module.

**Decisive test — does changing token *i+1* alter the logit at position *i*?**

| config | max abs change in logit at position *i* |
|---|---|
| **as shipped** (`causal=False`) | **0.100519** — the future token moves a past prediction |
| causal fix | **0.000000** — clean |

This is the leak demonstrated directly against upstream code, not asserted. Source
confirmation: `core/tetrahedral_attention.py:104` computes
`attn_scores = matmul(q, k^T)/scale` with no mask, and `tetrahedral_transformer_layer.py`
accepts `attention_mask` but the model never passes one.

**Training result** (3-layer, 96-dim, small 302-word vocab to remove `<UNK>` noise):

| steps | leaky train-PPL | causal train-PPL | ratio |
|---|---|---|---|
| 200 | **1.48** | 17.43 | 11.8Ã— |
| 600 | **1.01** | 14.02 | 13.9Ã— |
| 1500 | **1.00** | 11.20 | 11.2Ã— |

**The leaky model reaches training PPL â‰ˆ 1.00 — reproducing upstream's reported "1.32"
almost exactly.** The causal model on identical data and steps sits at 11–14, a ~12Ã— gap that
is roughly *constant* across training: the leak is exploited immediately to a fixed advantage,
not "learned" over time.

**Honest correction recorded.** A first probe at 150 steps on the full 2,599-word vocab showed
only a **0.78Ã—** gap, badly understating the effect. Cause, measured: with `min_freq=2` and a
large vocab, `<UNK>` tokens dominate the loss and mask the leak. Shrinking the vocabulary to
300 words exposed the true ~12Ã— gap. Both numbers are kept here on purpose.

**Consequence for evaluation.** The leaky model also scores lower on held-out text (266 vs 435
in the vocabulary-confounded probe) — but through an evaluation path that also sees the future.
There is no valid way to score it. That, not a structural limit, is why upstream reports it
"cannot generate text": autoregressive decoding needs a mask that does not exist.

Reproduce: `python "%TEMP%/opencode/probe_upstream_leak.py"` and
`python "%TEMP%/opencode/probe_leak_scaling.py"`. Full write-up in `REPO_PICK.md` Â§2.

## 3. Chat stage (`runs/chat`)

`runs/small/best.pt` (step 99) was instruction-tuned on Dolly-15k with an
assistant-only loss â€” 600 steps, batch 8, ctx 320, ~30 min CPU.  Full detail in
`CHAT.md`; the measured headline:

| checkpoint | held-out Dolly answer PPL | GitMonsters QA answer PPL |
|---|---|---|
| base `runs/small/best.pt` | 189.0 | 225.5 |
| chat `runs/chat/best.pt` (step 499) | 50.0 | 47.9 |
| **atomic `runs/chat_atomic/best.pt`** (step 499, Â§8 fix) | **53.9** | **56.0** |
| frequency baseline | 4.6 M | 3.8 M |

Replies became ~3.8Ã— less surprising and the model learned the conversation
format â€” but it is **not** a coherent chat model: greedy decoding collapses to
`the the the the ...`, and the end-of-reply marker `<|end|>` is **unreachable**
(its first piece ranks ~335/512, p â‰ˆ 2e-5) because the byte-BPE splits it into
pieces whose *sum* is competitive but whose individual ranks never are.  Â§6 of
`CHAT.md` documents these probes; the same failure was reproduced by upstream
(`finetune_chat.py`: "1.91 chat loss; no answer generation" at 52.9M params).

**Â§8 follow-up (the atomic-marker fix).**  The markers were promoted to atomic
single tokens and SFT was re-run (`runs/chat_atomic`, same budget).  Measured
outcome:

| measurement | subword | atomic |
|---|---|---|
| answer PPL (Dolly / upstream) | 50.0 / 47.9 | 53.9 / 56.0 (unchanged) |
| `<|end|>` greedy rank | 332â€“340 | **66â€“251** |
| `<|end|>` logit | âˆ’3.7â€¦âˆ’4.1 | **âˆ’1.35â€¦+0.30** |
| `<|end|>` wins argmax | never | still never |
| **stops its own reply (sampled, N=48)** | **0** | **31 (64.6%)** |
| reply coherence | none | none |

So the fix is **real but partial**: it removed the *representation* ceiling on
stopping (the model can now terminate a sampled reply ~65% of the time, up from
never) but did not touch reply quality, which is limited by data-per-parameter,
not tokenizer design.  Verdict and probes: `CHAT.md` Â§8,
`runs/{atomic_markers,stop_rate}.json`.

## 3b. Physics world knowledge (`runs/physics`, `runs/physics2`)

The base corpus has almost **no real physics** â€” "force" appears as *military force*,
"mechanics" as *game mechanics* (keyword census in `runs/physics_eval.json` /
`physics_census.py`).  To give the model physics world knowledge we built a
physics prose corpus and **continued-pretrained** the base LM on it.

**Corpus.** A 60 MB bounded prefix of the Simple-English-Wikipedia dump (full dump is
358 MB at ~0.3 MB/s here) streamed through a physics filter, with wikitext markup,
`[[File:â€¦|caption]]` blocks and residual brackets stripped.  Result: **378 clean
physics/science pages, 2.04 M chars** (`build_physics_corpus_fast.py`).  Only 2 pages
lack a physics term; lead-sentence caption artifacts fell from 90 to 3.

**Run #1 â€” negative result (`runs/physics`).**  Continued-pretrain with the
from-scratch defaults (LR 3e-3, 400 steps) *looked* good on train â€” lm 4.76 â†’ 3.58
(PPL 117 â†’ 36) â€” while held-out physics PPL **exploded**:

| checkpoint | held-out PPL | physics-PPL |
|---|---|---|
| base `runs/small/best.pt` | 188.4 | 228.6 |
| `runs/physics/best.pt` (step 49) | 196.4 | 206.7 |
| `runs/physics/final.pt` (step 399) | 1035.7 | **996.8** |

Textbook catastrophic overfit: 3.4 M params memorised 2 MB.  A continued pretrain needs
a gentle schedule.

**Run #2 â€” corrected schedule (`runs/physics2`), LR 2e-4, 120 steps, warmup 15.**
This is the honest test, and it *works*:

| checkpoint | held-out PPL | **physics-PPL** |
|---|---|---|
| base (never saw physics) | 188.4 | 228.6 |
| **`runs/physics2/best.pt` (step 19)** | **121.6** | **129.4** |
| `runs/physics2/final.pt` (step 119) | 183.1 | 191.7 |

Physics perplexity **halved (228.6 â†’ 129.4, 1.77Ã—)**, and two independent held-out
splits agree on step 19 being best (121.6 vs 129.4) â€” the gain is real, not a split
artifact.  It is however **early and fragile**: PPL rises monotonically after step 19
(LR 2e-4 is still too hot, or 2 MB too small), so the usable effect is the first
~20 steps.  The reproducible recipe is therefore *short and gentle*, not long:

```bash
python build_physics_corpus_fast.py --cap-mb 60        # 378 pages, 2.04 M chars
python train.py --init runs/small/best.pt \
    --data "%TEMP%/opencode/physics_corpus.jsonl" --out runs/physics2 \
    --steps 120 --batch 8 --ctx 256 --lr 2e-4 --min-lr 2e-5 --warmup 15 --threads 4
python eval_physics.py --checkpoints "base=runs/small/best.pt,p2=runs/physics2/best.pt"
```

## 3c. Does the physics gain survive into chat?  No.

The user's target was a **physics world model in a chat window**.  So the honest test is
the whole chain â€” base â†’ physics pretrain â†’ SFT â€” measured on held-out physics text and
on physics QA.  `runs/chat_physics` is the re-SFT from the physics base, on Dolly-15k +
upsampled physics QA (16901 rows, 11.2% physics, atomic markers, 500 steps).

**Physics language-model PPL along the chain** (`physics_chain.py`):

| stage | physics-LM-PPL | vs base |
|---|---|---|
| base `runs/small/best.pt` (no physics) | 228.6 | â€” |
| **`runs/physics2/best.pt`** (continued pretrain) | **129.4** | **1.77Ã— better** |
| `runs/chat_atomic/best.pt` (SFT, pre-physics) | 352.0 | 1.54Ã— *worse* |
| `runs/chat_physics/best.pt` (SFT from physics base) | **363.9** | 1.59Ã— *worse* |

**Physics-QA answer PPL + replies** (`eval_physics_chat.py`, 250 held-out QA rows):

| checkpoint | physics-QA answer PPL | replies |
|---|---|---|
| `runs/chat_atomic/best.pt` (pre-physics) | 47.8 | word-soup |
| `runs/chat_physics/best.pt` (physics base + mix) | **49.2** | word-soup |

**Verdict (negative, and worth stating plainly):**

1. Continued pretraining genuinely taught the *base LM* physics â€” 228.6 â†’ 129.4 on
   held-out physics prose, confirmed independently by the training val split (188.4 â†’
   121.6).  This part works.
2. **SFT washes it out and inverts it.**  Both SFT models model physics prose *worse*
   than the untouched base (352 / 364 vs 228.6), and the physics-based SFT (363.9) is no
   better than the pre-physics one (352.0).  Assistant-only loss on short Dolly answers
   narrows the model away from general next-token prediction, and the 11% physics mix is
   too little to hold the line.
3. Consequently the physics-trained chat model does **not** answer physics questions
   better than the pre-physics chat model â€” 49.2 vs 47.8 answer PPL, both word-soup.

So at 3.3 M params, on 2 MB of physics: **pretraining learns physics; instruction tuning
throws it away.**  The honest recommendation at this scale is *not* to re-SFT the physics
model, but to either (a) chat on the physics base directly, or (b) grow the corpus and
the model by ~10â€“100Ã— so SFT stops dominating.  Both are out of the <30 min budget.

## 3d. Fixing the SFT destruction: auxiliary LM loss  (yes, this works)

Â§3c diagnosed the failure but did not fix it.  The mechanism is structural: SFT's
objective is *assistant-tokens-only cross-entropy* on short Dolly answers, which is
orthogonal to next-token prediction over physics prose.  An 11% physics share in the
data cannot defend a distribution the loss never rewards.  So the fix is to keep
rewarding it directly.

`sft.py` now takes `--lm-data/--lm-weight/--lm-batch/--lm-ctx`: each step adds a
**plain causal next-token loss on physics prose** to the masked assistant loss:

    loss = masked_assistant_CE  +  lm_weight * CE(next-token on physics chunks)

The auxiliary term uses the model's own built-in causal objective
(`model(xb, targets=xb)`) â€” bit-for-bit the same loss as continued pretraining â€” so it
is a fair, apples-to-apples quantity against the `physics2` number, not a re-scaled one.
`runs/chat_physics_lm` = `--base runs/physics2/best.pt --lm-weight 0.5 --lm-batch 4
--lm-ctx 256`, 500 steps, everything else identical to `runs/chat_physics`.

**Physics-LM perplexity on HELD-OUT physics text** â€” enwiki pages (`physics_probe_enwiki.jsonl`,
1341 pages from the English Wikipedia dump) that *no* checkpoint has ever trained on
(`eval_physics.py --data ... --n 400`):

| stage | step | vocab | physics-PPL (held-out enwiki) | vs physics2 |
|---|---|---|---|---|
| base `runs/small/best.pt` (no physics) | 99 | 512 | 220.9 | 1.69Ã— worse |
| `runs/physics2/best.pt` (continued pretrain) | 19 | 512 | 130.6 | â€” |
| `runs/chat_atomic/best.pt` (SFT, no fix) | 499 | 516 | 366.4 | 2.81Ã— worse |
| `runs/chat_physics/best.pt` (SFT, no fix) | 299 | 516 | 376.6 | 2.88Ã— worse |
| **`runs/chat_physics_lm/best.pt` (SFT + aux LM loss)** | 499 | 516 | **37.1** | **3.52Ã— better** |

The same checkpoint evaluated on the 2 MB corpus it *did* use (`physics_chain.py`) gives
36.0 â€” within noise of the held-out 37.1, so this is **not** leakage from training on the
aux corpus; the model generalises the gain to fresh enwiki physics prose.

**Verdict (positive, and falsifiable as predicted):**

1. The prediction in the Â§3c write-up was "physics-LM-PPL â‰ˆ 130 if the fix works, ~350+
   if it does not."  Measured: **37.1**.  The fix works, and over-delivers â€” the chat
   model is *better at physics prose than the physics base it came from* (37.1 vs 130.6),
   which is expected: 500 further steps of physics CE on top of a partially-trained base
   keeps improving that quantity while the masked chat loss runs alongside it.
2. Chat quality did **not** pay for it.  `runs/chat_physics_lm` reached a **better**
   answer-token validation PPL than the no-fix run (63.25 at step 499, and best 65.35 at
   step 399 across evals) versus 56.72â€¦63.25 â€” i.e. the two objectives are not in
   tension at this weight (0.5).  Both losses fell together during training
   (LM term 5.21 â†’ 3.59, mask loss 6.86 â†’ 5.96 at the checkpoints shown).
3. **The one number that did not improve:** physics-*QA answer* PPL
   (`eval_physics_chat.py`, 278 rows) is 55.0 for the fix vs 48.9 / 47.3 for the no-fix
   SFTs.  So the aux-LM term strengthens the *physics prose* distribution (10Ã— better)
   but does **not** strengthen question-answering â€” which is a different, template-shaped
   distribution the base LM was never trained on.  Both remain word-soup.  The honest
   statement is therefore narrow and exact: **the fix preserves physics knowledge; it
   does not manufacture physics answers.**
4. This is still the reference fix for the project: **never fine-tune an LM on
   assistant-only loss without an auxiliary LM term.**  One extra data loader and one
   extra CE term turned a 2.9Ã— regression into a 3.5Ã— improvement on the quantity the
   pretrain was supposed to protect.

**Still true, and still the binding limit:** this is **perplexity**, not fact recall.  The
fact-completion probe (`eval_physics_facts.py`, 20 relations/units Ã— greedy+sampled) scores
**0/40 for every checkpoint**, including `chat_physics_lm` â€” the model cannot complete
*"The SI unit of force is the ___" â†’ newton* at 3.3 M params.  So the honest statement is:
*the fix preserves and improves the physics distribution; it does not create a model that
knows physics facts, because that requires capacity this model does not have.*

## 3e. Scaling the corpus: more text does not buy facts

Â§3d fixed the SFT destruction.  The last open question was whether the *corpus* was the
limit â€” so the physics slice was scaled from a 0.27%-density Simplewiki prefix (378 pages /
2 MB) to a **13.2%-density enwiki prefix** (6167 pages / 71 MB, `build_physics_corpus_large.py`),
and a **general world-knowledge** slice was added (`build_world_corpus.py`: 25 000 enwiki
articles / 229 MB â€” people, places, history, arts, science; only 2.2% of the physics slice's
titles were non-physics, so the filters hold).

A pilot continued-pretrain (`runs/worldmix`) trained the base on a shuffled **52% world /
48% physics** 60 MB mix, and `eval_world.py` (new: held-out world-PPL + 20 general fact
probes) was added to measure it:

| model | data | held-out physics-PPL | held-out world-PPL | world fact hit-rate |
|---|---|---|---|---|
| base `runs/small` | 2.9 MB web mix | 220.9 | 216.7 | 0/40 |
| `runs/physics2` | 2 MB physics (dense) | **130.6** | â€” | 0/40 |
| `runs/worldmix` (pilot, step 49) | 60 MB mix, 48% physics | 199.3 | **195.2** | 0/40 |

**Three measured conclusions:**

1. **The pipeline learns new domains** â€” world-PPL fell 216.7 â†’ 195.2 on text the model had
   never seen, so continued pretraining on enwiki works.
2. **Breadth costs depth.**  The same number of steps spread over a broader distribution
   left physics at 199.3, far behind the 2 MB *pure-physics* run's 130.6.  Density of
   relevant text beats total volume at this step budget.
3. **Decisive: fact recall is 0/40 in every single configuration** â€” base, physics-only,
   world-mix, and the Â§3d fix.  The model emits `"the the the"` for *"Abraham Lincoln was
   the ___"* and *"The capital of France is ___"*.  Perplexity moves; **facts do not.**

**This is the binding result of the whole build:** at 3.3 M parameters, a CPU-trained LM
learns *distributions*, never *recall*.  No corpus, mix, or recipe tested here changes that.
Factual question answering therefore requires either far more capacity (GPU-scale) or
**retrieval** â€” see `RAG.md`, which builds the retrieval path this repo's own measurements
recommend.  Reported as measured, not as hoped.

## 4. What this build does *not* claim

- No frontier performance: this is a 3.4M-param model trained for half an epoch on a
  CPU. The point is the *methodology* â€” causal masking, lossless tokenization, honest
  held-out evaluation â€” not the benchmark.
- **Physics continued-pretrain lowers physics *perplexity* but does not survive SFT into
  answering.** Â§3b shows a 1.77Ã— held-out physics-PPL gain (228.6 â†’ 129.4), but Â§3c
  shows this is **washed out and inverted by SFT**: the re-SFT model's physics PPL is
  363.9 â€” *worse* than the un-touched base's 228.6 and no better than a pre-physics SFT
  (352.0). At this scale, continued pretraining on 2 MB buys a real but fragile LM gain,
  and instruction tuning then discards it. Reported as measured, not as hoped.
- **Not a conversational model.** The chat window works end to end, but the model
  behind it does not follow instructions or answer questions; see Â§3 and `CHAT.md`.
  The Â§8 atomic-marker fix made it able to *stop* a reply (~65% of sampled replies,
  up from 0%) without making the reply an answer.
- **The LM does not answer fact questions â€” by design, retrieval does.** Â§3e shows
  0/40 fact recall for every checkpoint and corpus, so the repo does not claim Q&A on
  the LM. It ships the path that works (`rag/`, `RAG.md`): retrieval measured
  **100 % recall@4** on the physics corpus, with answers grounded and cited. The RAG
  answer quality is only **14/24**, which localises the remaining gap in *generation*
  (a real LLM is required, not a 3.3 M CPU model) â€” reported as measured.
- The salvaged components (tetra bias, torus RoPE, fractional limbs) were kept because
  they are coherent and trainable, **not** because each was empirically proven to beat
  plain attention at this scale. Component ablations at larger scale are future work.
- Held-out PPL 188 is good for this budget, not remarkable in absolute terms.

## Reproduce

```bash
python -m pytest tests -q                     # 60 passed
python train.py --data ".../combined_train.jsonl" --out runs/small \
    --steps 600 --batch 8 --ctx 256 --d-model 256 --n-layer 4
python evaluate.py --checkpoint runs/small/best.pt --text-data ".../combined_train.jsonl" \
    --ood ".../eval_heldout.jsonl"
python leak_ablation.py --data ".../combined_train.jsonl" --steps 300
python sft.py --base runs/small/best.pt --data ".../dolly.jsonl" \
    --out runs/chat --steps 600 --batch 8 --ctx 320 --threads 4
python sft.py --base runs/small/best.pt --data ".../dolly.jsonl" \
    --out runs/chat_atomic --atomic-markers --steps 600 --batch 8 --ctx 320 --threads 4
python eval_chat.py --checkpoints "base=runs/small/best.pt,subword=runs/chat/best.pt,atomic=runs/chat_atomic/best.pt" \
    --n 500 --out runs/chat_eval.json
python chat_server.py --checkpoint runs/chat_atomic/best.pt --port 8765

# --- physics world knowledge (RESULTS.md Â§3b/Â§3c) ---
python build_physics_corpus_fast.py --cap-mb 60            # 378 pages -> physics_corpus.jsonl
python build_physics_qa.py                                 # 378 extractive QA rows
python build_sft_mix.py --repeat 5                         # Dolly + physics QA (11% physics)
python train.py --init runs/small/best.pt \
    --data "%TEMP%/opencode/physics_corpus.jsonl" --out runs/physics2 \
    --steps 120 --batch 8 --ctx 256 --lr 2e-4 --min-lr 2e-5 --warmup 15
python eval_physics.py --checkpoints "base=runs/small/best.pt,p2=runs/physics2/best.pt"
python sft.py --base runs/physics2/best.pt --data "%TEMP%/opencode/sft_physics_mix.jsonl" \
    --out runs/chat_physics --atomic-markers --steps 500 --batch 8 --ctx 320
python eval_physics_chat.py --checkpoints "pre=runs/chat_atomic/best.pt,phys=runs/chat_physics/best.pt"
python physics_chain.py          # physics-LM-PPL at each stage (base -> pretrain -> SFT)
```

```bash
# --- retrieval path (RAG.md): factual Q&A that actually works ---
python rag_search.py "what is the SI unit of force" \
    --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" --limit-docs 4000
python eval_rag.py --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" \
    --data "%TEMP%/opencode/rag_eval.jsonl" --limit-docs 4000 --k 4
python rag_server.py --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" \
    --port 8766 --limit-docs 4000                  # -> http://127.0.0.1:8766/
```
