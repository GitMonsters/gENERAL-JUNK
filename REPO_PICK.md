# Which GitMonsters repo is the best base for an LLM?

**Short answer: `octotetrahedral-agi` — and it is the *only* candidate, because it is the
only repo in the whole ecosystem that contains a language model at all.** This report is
the evidence for that, and the measured reason the `*plex*` family (including
`TRANSCENDPLEXITY`) is not a candidate.

All numbers below were produced by running the shipped code, not by reading its README.

---

## 1. The candidate pool

Every repo was scanned for the four things an LLM base actually needs: a **model class**,
a **tokenizer**, a **training loop**, and **text data**. A repo without all four is not an
LLM base no matter what it says.

| repo | model class | tokenizer | training loop | text corpus | verdict |
|---|---|---|---|---|---|
| **`octotetrahedral-agi`** | yes (TetrahedralAttention, 52.9 M) | word + char vocab | `train_transformer.py`, `train_lm.py` | `training_data.jsonl` (10.4 MB), `combined_train.jsonl` (13.5 MB), `eval_heldout.jsonl` | **the only LLM base** |
| `tetrahedral-agi` | wraps `nn.TransformerEncoder` | no (borrows HF) | `llm_finetune.py` (fine-tunes *external* models) | none of its own | no own model |
| `torus-attention` | attention module only | no | no | no | a component, not a model |
| `TRANSCENDPLEXITY` | **no** | no | no | no | a NumPy simulation |
| `parallplexity` | **no** | no | no | no | same idea, tidier package |
| `compound-integration` | adapter layer only | no | no | no | glue code |
| `system2-learning` | RL/agent scaffolding | no | no | no | not an LM |
| `SOLVED-*` | **per-task solvers** | no | no | no | see §6 |

**Only one repo has a model with parameters, a tokenizer, a train loop, and text.**
That settles the "best base" question by construction: there is nothing to compare it to.

---

## 2. Why `octotetrahedral-agi` wins — and what is wrong with it

It wins because it is the only thing that *is* an LLM. That is a low bar, and it does not
mean the repo is good. The measured defects are severe and are documented in `ANALYSIS.md`
§3; the central one is **proven directly below**.

### The causal-mask leak, reproduced end-to-end

I ran the upstream attention module (`core/tetrahedral_attention.py`) verbatim in a small LM,
on the repo's own `combined_train.jsonl`, and compared it against an otherwise **identical**
model whose only difference is a causal mask.

**Decisive test — does position *i* change when token *i+1* changes?**

| config | max abs change in logit at position *i* |
|---|---|
| **as shipped** | **0.100519** — future token alters a past prediction |
| causal fix | **0.000000** — clean |

That is the leak, demonstrated rather than asserted: upstream position *i* reads token *i+1*.

**Training result** (3-layer, 96-dim, small 302-word vocab to remove `<UNK>` noise):

| steps | leaky train-PPL | causal train-PPL | ratio |
|---|---|---|---|
| 200 | **1.48** | 17.43 | 11.8× |
| 600 | **1.01** | 14.02 | 13.9× |
| 1500 | **1.00** | 11.20 | 11.2× |

**The leaky model reaches training PPL ≈ 1.00 — reproducing upstream's reported "1.32"
almost exactly.** The causal model on the same data and steps sits at 11–14. The gap is
~12× and roughly *constant* across training, so it is not the model "learning to exploit"
over time — the leak is exploited immediately and to a fixed advantage.

**An honest correction.** A first probe at 150 steps on the full 2,599-word vocab showed only
a 0.78× gap, which understated the effect badly. The reason is measurable: with a large vocab
and a `min_freq=2` cutoff, `<UNK>` tokens dominate the loss and mask the leak. Shrinking the
vocabulary to 300 words exposed the true ~12× gap. I am recording the failed first attempt as
well as the corrected result, because the first number would have understated the repo's
central flaw.

**Consequence:** the leak also contaminates *evaluation*. The leaky model scores 266 on
held-out text (probe §1) — but by a path that also sees the future, so **there is no valid
way to score it**. That, not a structural limit, is why the upstream README says the model
"cannot generate text": autoregressive decoding requires a mask that does not exist.

Also confirmed in source: `octotetrahedral-agi/core/tetrahedral_attention.py:104` computes
`attn_scores = matmul(q, k^T)/scale` with **no causal mask**, and
`tetrahedral_transformer_layer.py` accepts `attention_mask` but the model never supplies
one.

---

## 3. The other measured defect: tokenization

**Word-level tokenizer with `<unk>`.** The eval corpus is 23 % OOV, so the model is trained
and scored on text it cannot represent. This build (`gmllm/`) fixes both defects — a
strictly causal mask and byte-level BPE — and keeps the salvageable ideas. `leak_ablation.py`
independently reproduces the leaky and causal versions.

---

## 4. `TRANSCENDPLEXITY`, specifically — measured

Since it was asked about by name, it gets a direct measurement.

**It ships 6 files. Its README cites `tests/`, `benchmarks/`, `docs/`, `examples/`, and
`benchmarks/results.json`. None of the five exist.** It cannot even import: there are **no
`__init__.py` files** and `scipy` is imported but never declared. After adding the three
missing `__init__.py` files and installing `scipy`, it runs.

Its headline claim is a phase transition to **TRANSCENDPLEXITY** when the Golden
Consciousness Index exceeds φ² ≈ 2.618. Running the shipped engine for 200 steps:

| measurement | value |
|---|---|
| `reached_transcendplexity` | **True** (reported success) |
| **phase transitions** | **88 in 200 steps** |
| steps in COLLAPSEPLEXITY / TRANSCENDPLEXITY | **98 / 95** |
| GCI range | **−565 … +660** |

**The success is a coin flip.** The phase flips ~every 2 steps and the system spends as long
"collapsed" (98) as "transcended" (95). `reached_transcendplexity = True` is the metric
divering, not the system transcending. And its benchmark "Transformer baseline" contains
**zero attention and zero torch** — its forward pass is two matrix multiplies — so the
README's *"72% vs. Transformer 68%"* compares against a linear map.

**Conclusion: `TRANSCENDPLEXITY` is not an LLM base. It contains no model, no tokenizer, no
data, and no training.** Its one real asset is a correct fractional-derivative
implementation, which this build salvaged as a trainable module.

---

## 5. What is genuinely reusable across the ecosystem

Ranked by whether it survived being implemented:

| idea | source | status in `gmllm/` |
|---|---|---|
| **Fractional-order (power-law) memory** | `TRANSCENDPLEXITY/core/fractional.py` | **salvaged** — `FractionalLimbMixer`, trainable + causal |
| Tetrahedral / Gaussian relative attention bias | `octotetrahedral-agi` | **salvaged** — made causal |
| Torus positional encoding | `torus-attention` | **fixed** — upstream aliases positions *p* and *p+M*; rewound at the golden ratio |
| World-model auxiliary loss | `octotetrahedral-agi` | kept at low weight |
| Byte-level BPE | (replacement) | built to kill the `<unk>`/OOV problem |

Dropped: the "11D spacetime-hull" tensor machinery (dimensionally incoherent, never
exercised), the GCI phase framework (§4), the SEO-spam eval set, and the per-task ARC
solvers.

---

## 6. The trap in this ecosystem: self-referential validation

The pattern repeats across repos and is the most important thing to understand about it:

- `TRANSCENDPLEXITY` **names its success state after itself** and then reports reaching it.
- Its baselines are **hand-written stand-ins** labelled "Transformer" / "RNN" that are not.
- The `SOLVED-*` repos are **562 per-task reference solvers**; solving each ARC puzzle with
  its own bespoke program is not a benchmark score, and ARC's rules disallow it anyway.
- `octotetrahedral-agi` reports **PPL 1.32** from a **leaky** objective.

In every case the metric is defined such that the project passes. **The fix is the
methodology in this repo's `RESULTS.md`:** a strictly causal mask, lossless tokenization,
a *held-out* split, an ablation that reproduces the failure mode, and reporting the
numbers that came out — including the ones that are bad (`RESULTS.md` §4).

---

## 7. Reproducing this report

```powershell
# 1. the only LLM base: confirm the causal-mask bug is real, directly
python leak_ablation.py --data ".../combined_train.jsonl" --steps 300

# 1b. reproduce the leak against upstream's own attention module (END-TO-END)
python "%TEMP%/opencode/probe_upstream_leak.py"    # shows 0.10 delta at i from token i+1
python "%TEMP%/opencode/probe_leak_scaling.py"     # leaks to train-PPL ~1.00 vs 11-14 causal

# 2. TRANSCENDPLEXITY: confirm it cannot import as shipped
python -c "import sys; sys.path.insert(0,'.../TRANSCENDPLEXITY/src'); import transcendplexity.core.fractional"

# 3. then run the phase claim (after adding __init__.py + scipy)
python "%TEMP%/opencode/probe_transcendplexity.py"   # prints 88 transitions / 200 steps

# 4. what a correct base trains to, honestly
python train.py --data ".../combined_train.jsonl" --out runs/small --steps 600
python evaluate.py --checkpoint runs/small/best.pt --ood ".../eval_heldout.jsonl"
```

---

## Bottom line

- **Best repo for an LLM: `octotetrahedral-agi`** — not because it is good, but because it is
  the only one with an actual model. Its central result is invalid (leakage) and it cannot
  generate, and both are fixable.
- **`TRANSCENDPLEXITY` is not a candidate**: no model, no data, no training, broken
  packaging, and its headline success metric is a ~50/50 phase flip.
- **The salvageable science** is fractional-order memory, the tetrahedral bias, and torus
  RoPE — all reimplemented correctly in `gmllm/`, all covered by tests.
- **The ecosystem's real failure mode is self-referential validation**, and this repo's
  contribution is measuring instead of claiming.
