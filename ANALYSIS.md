# Analysis of the GitMonsters GitHub Repositories

**Date:** 2026-10-20
**Account:** [github.com/GitMonsters](https://github.com/GitMonsters) (41 public repositories)
**Method:** All 41 repos were inventoried via the GitHub API. The ~12 repos relevant to
language modelling were downloaded in full (61 source/data files, 37.5 MB) and read
line-by-line; the rest were characterized from metadata and file trees. Local sources:
`%TEMP%\opencode\gitmonsters\`.

---

## 1. Executive summary

The GitMonsters account contains **one genuinely interesting, honestly-reported negative
result** (`octotetrahedral-agi`), a set of **salvageable architectural ideas** scattered
across several repos, a large amount of **aspirational pseudo-code that cannot run**, and
a layer of **marketing claims ("quantum", "consciousness", "100% ARC-AGI-3") that the code
does not support**.

The most valuable finding is a **bug diagnosis the original authors missed**: their
flagship 52.9M-parameter language model reports an apparently miraculous eval perplexity
of 1.32 but "cannot generate fluent text at any size" (their words). Reading the code
shows this is **target leakage from a missing causal mask** — the model was trained as a
bidirectional encoder and evaluated as if it were an autoregressive LM. The low perplexity
is an artifact; the "generative wall" is not "structural", it is a one-line bug. The build
in this repository (`gmllm/`) fixes it and demonstrates the fix by training.

**Verdict per repo class:**

| Class | Repos | Verdict |
|---|---|---|
| Honest, useful negative result | `octotetrahedral-agi` | Real PyTorch, real corpus, real measurements. One critical bug (non-causal attention). Best starting point. |
| Salvageable ideas, broken code | `parallplexity`, `TRANSCENDPLEXITY`, `torus-attention` | Ideas worth keeping (fractional-order kernels, multi-timescale limbs, periodic positional geometry). Code ranges from toy simulations to unrunnable API-mixing. |
| Marketing / claim layer | `SOLVED-*`, `13-Impossible-ARC-Tasks-*`, `NGVT`, `simpleCOMPLEXITY`, `enhanced-multi-pinnacle-torus-*` | Per-task solvers presented as general intelligence; "quantum coupling" and "consciousness" are random matrices and buzzwords. Not a foundation for anything. |
| Unrelated / copies | `tinygrad`, `openpilot`, `opendbc`, `instasak`, `KELDO`, … | Mirrors of other projects or empty placeholder repos. |

---

## 2. Ecosystem map (how the repos relate)

```
TRANSCENDPLEXITY ──┬── parallplexity (v0.3.0 rewrite: fractional-order "parallplexity")
(brand name)       ├── octotetrahedral-agi (the actual LM: tetrahedral attention + core/*)
                   ├── tetrahedral-agi (GAIA benchmark framing, geometric grids)
                   ├── torus-attention / torus-collider (Rust/tinygrad side quests)
                   ├── twistedWorm-Template / Prime-directive (Rust "mimicry engine" + ARC)
                   └── SOLVED-* (ARC-AGI per-task solver collections)
compound-integration, system2-learning, octocomputer  (agent/integration layer around the above)
```

Nearly everything is 0-star, single-author work. Several "repos" (`Prime-directive`,
`twistedWorm-Template`) contain **duplicated directories of each other** (both contain a
`Prime-directive/` subtree and the same `src/mimicry/` Rust tree). `octotetrahedral-agi`
is the only repo with a self-critical, measurement-driven README/RESULTS pair.

---

## 3. Deep dive: `octotetrahedral-agi` (the flagship)

**What it claims (README/RESULTS):** a from-scratch 52.9M-param LM (d=512, 8 heads,
8 layers, ff=2048, word vocab 26,603) with "tetrahedral geometric-bias attention", a
6-term cognitive loss, working memory and reservoir dynamics. Eval perplexity **1.32**
(teacher-forced, 500 held-out sentences), but **autoregressive generation collapses into
repetition ("token soup") at every size tried (13.7M–52.9M)**. Their conclusion: "that
gap is structural at this parameter/data scale, not a tuning miss."

**What the code actually does:**

1. **`core/tetrahedral_attention.py`** — standard multi-head attention plus an additive
   geometric bias. The bias ("`_make_geometric_bias`" in
   `core/tetrahedral_transformer_layer.py:113`) is a **Gaussian decay over |i−j|**,
   normalized to [−1, 1], scaled by a learnable global and per-head factor. This is
   essentially **ALiBi with a Gaussian instead of linear falloff** — a reasonable
   positional prior, though not "tetrahedral" in any geometric sense (no tetrahedron
   appears anywhere in the computation).

2. **The critical bug — no causal mask.** `TetrahedralTransformerEncoder.forward()`
   calls the encoder on the full sequence with **no causal mask** anywhere
   (`train_transformer.py:292`, `core/tetrahedral_transformer_layer.py:149`). The LM loss
   then shifts: logits at position *i* are trained to predict token *i+1*
   (`train_transformer.py:304`). But the hidden state at position *i* has already
   attended to position *i+1*, whose **input embedding contains token *i+1***. The model
   can reach forward, copy the answer, and never learn language. The Gaussian proximity
   bias actively makes this easier by up-weighting nearby positions.

   **The evidence is in their own numbers:**
   - Eval PPL **1.32** is impossibly good for 52.9M params on 102K sentences of noisy
     text (frontier labs need ~1000× more data for low-single-digit byte-level PPL).
   - **Eval PPL (1.32) is lower than train PPL (6.23)** — the classic signature of a
     train/eval asymmetry, not of generalization (train runs with dropout on, eval
     without, which widens the gap further).
   - The failure signature matches exactly: at generation time position *i+1* does not
     exist yet, the leak delivers only padding/zeros, and the model degenerates to
     high-frequency tokens — "the cat sat sat united offer d d d d". Their "robustness"
     score of 0.721 (single-word swaps raise PPL) is also explained: a swap near position
     *i* corrupts the leaked answer.
   - Their proposed structural explanation ("short-horizon washout": per-token survival
     0.75 compounds to 2e-4 over 30 tokens) is a symptom, not a cause — a model that
     actually learned p(next | past) at PPL 1.32 would generate beautifully.

   **Fix:** one line — build and apply a causal mask. `gmllm/` does this and proves it
   with a training ablation (`leak_ablation.py`).

3. **Auxiliary "cognitive" modules are mostly dead weight.** In
   `train_transformer.py:311-324`, the reservoir, TranscendPlexity controller and
   cohesion tracker run under `torch.no_grad()` — they are **monitors that cannot affect
   the output**. The working memory (`core/working_memory.py`, a learned-slot DNC-style
   memory) is instantiated but **never called in the forward pass at all**. Only
   `CognitiveGeometryEngine` and the 6-term `RecursiveEngineObjective` loss touch
   gradients, and the world-model term (`L_WM`) predicts hidden→next-hidden — which,
   given the leak, is trivially satisfied by copying. Their own measured finding that
   "char embeddings are a dead feature" (predictions identical with real/random/zeroed
   char IDs) is consistent with a model that routes everything through the leak.

4. **Corpus (downloaded and inspected):** `combined_train.jsonl` — 102,358 sentences
   (~13.5 MB): WikiText-2-style encyclopedic text (`training_data.jsonl`, 16K long
   lines), CLARIN news (12K), 8.5K YouTube-science transcripts, plus instruction pairs
   auto-extracted from transcripts (`instructions_v2.jsonl` pairs a question with
   "A few months later, they may not." — retrieval noise, not supervision). The
   "clean" held-out eval set (`eval_heldout.jsonl`, 500 sentences) is **SEO spam** about
   celebrity-inspired engagement rings in Ormond Beach, FL. It is out-of-domain, not clean.

5. **What is genuinely good here:** the honest RESULTS.md, the warm-start/rollback
   checkpoint tooling, label smoothing as a one-hot-collapse fix, the retrieval+ranking
   chatbot pattern (using the LM as a *scorer*, not a generator — the right call given
   their measurements), and the multi-draft self-consistency reasoner. The POS tagger
   (BiLSTM, 99.65%) is the repo's best real result.

---

## 4. `torus-attention` — unrunnable, but two ideas worth keeping

`src/torus_attention_mechanism.py` **cannot execute**: it imports tinygrad's `Tensor`
and `nn`, then freely mixes in `torch.sigmoid`, `torch.roll`, `F.softmax` (never
imported), `nn.Parameter` from both frameworks, and `register_buffer` on a plain Python
object. `apply_vortex_dynamics` also `torch.roll`s attention weights, which would mix
future attention into current positions (destroying causality) even if it ran.

Two ideas inside are worth salvaging:
- **Torus positional encoding:** map position *p* to angles on a torus
  (u = 2π(p mod M)/M, v = 2π(⌊p/M⌋ mod m)/m) instead of a linear index. The original
  mapping **repeats exactly every M·m positions** (aliasing). `gmllm/` keeps the torus
  idea but uses a **golden-ratio winding** (v rotates φ× faster than u), which is dense
  on the torus — the classic non-repeating torus-knot curve — and preserves causality
  because it is a pure function of position, applied rotarily.
- **Block attention residuals ("Kimi AttnRes"):** pseudo-query softmax over completed
  block summaries. Reasonable in principle; the cited arXiv id (2603.15031) does not
  resolve to any known paper — treat citations in these repos with suspicion.

The rest (vortex strength, poloidal/toroidal flow, "Laderman 23-multiplication 3×3
matmul" for d_head=3) is decorative: no benchmark anywhere compares it to plain
attention.

---

## 5. `TRANSCENDPLEXITY` / `parallplexity` — measured: it runs, and the claim is noise

The user asked specifically about **`TRANSCENDPLEXITY`**. It is the "framing" repo of the
`*plex*` family: a NumPy-only simulation that names a **phase** after itself. Its README
presents a validated framework with a results table and cites `tests/`, `benchmarks/`,
`docs/`, `examples/`, and `benchmarks/results.json`.

**What is actually in the repo: 6 files. None of the five cited paths exist.** Measured:

```
MISSING: tests            <-- claimed in README
MISSING: benchmarks       <-- claimed in README
MISSING: docs             <-- claimed in README
MISSING: examples         <-- claimed in README
MISSING: benchmarks\results.json  <-- claimed in README ("Full results in ...")
actual file count: 6
```

### It cannot even import as shipped

There are **no `__init__.py` files** anywhere, and it imports `scipy` without declaring it:

```
FAIL transcendplexity.core.fractional: ModuleNotFoundError: No module named 'scipy'
FAIL transcendplexity.engines.compound: ImportError: cannot import name 'ParallplexityTensor'
```

After adding the three missing `__init__.py` files (`core/`, `engines/`, package root) and
installing `scipy`, the engine **does** run. So the code is real, merely unpackaged and
untested — consistent with a `tests/` directory that was never committed.

### The central claim, tested

The framework's headline is a phase transition to **TRANSCENDPLEXITY** when the Golden
Consciousness Index exceeds φ² ≈ 2.618. Running the shipped 8-layer WormsEngine with its
own `CompoundingTracker` for 200 steps (`seed=42`):

| measurement | value |
|---|---|
| runtime | 3.04 s (15.2 ms/step) |
| final phase | TRANSCENDPLEXITY |
| **`reached_transcendplexity`** | **True** |
| **phase transitions in 200 steps** | **88** |
| GCI range | **−565.1 … +659.6** |
| steps in COLLAPSEPLEXITY / TRANSCENDPLEXITY | **98 / 95** |

**The reported success is a coin flip.** The phase flips 88 times in 200 steps; the system
spends almost exactly as long "collapsed" (98 steps) as "transcended" (95). A state that
alternates every ~2 steps between COLLAPSE and TRANSCEND is measuring a numerical
instability, not a physical phase transition — the GCI is the time-derivative of a
`1/λ_min` spectral quantity, so near-zero eigenvalues make it diverge. `reached_
transcendplexity = True` is therefore **not evidence of compounding cognition**; it is
evidence that the metric is unbounded and oscillates. This is reported as measured, and it
is the reason the repo is *not* the basis for an LLM.

### The benchmark comparison is against a non-transformer

`TRANSCENDPLEXITY/baselines.py` is NumPy-only and its `TransformerBaseline` contains
**zero attention** — measured: `'attention' occurrences: 0`, `'MultiheadAttention': 0`,
`'torch': 0`. Its entire forward pass is two matrix multiplies:

```python
def forward(self, X):
    h = X @ self.embed_W          # (hidden,)
    logits = h @ self.output_W    # (out,)
    return logits
```

So the README's *"Complex compositions: 72% (vs. Transformer 68%)"* compares
TranscendPlexity against a **two-layer linear map** labelled "Transformer". No claimed
transformer result is meaningful. (The `SimpleRNN` baseline likewise has no recurrence.)

### The one real idea, still

`core/fractional.py` implements a correct **Caputo / Grünwald–Letnikov fractional
derivative** — power-law memory with weights `w_k = Π(1 − (α+1)/j)`, the Caputo kernel
`(t−τ)^(n−1−α)`, and a Mittag-Leffler series. `CANTOR_GOLDEN_DIM = log2/logφ ≈ 1.4404` is
decorative, but the fractional machinery is legitimate and published. It is what the
ecosystem actually contributes, and `gmllm/modules.py::FractionalLimbMixer` implements it
as a **trainable, causal, vectorized** module instead of a simulation nothing learns from.

### Where TranscendPlexity *is* used

`octotetrahedral-agi/core/transcendplexity_integration.py` (421 lines) ports three concepts
into PyTorch as `nn.Module`s — `AlphaOrderDynamics`, `CompoundingLossTracker`,
`PhaseDetector`. Unlike the NumPy original, this one is genuinely differentiable. But it is
**not on the LM's critical path**: it modulates loop/gate/entropy settings of the
`CompoundLoop` scaffolding around the model. The language modelling itself happens in
`TetrahedralAttention`, which is where the causal-mask bug lives (§3).

**Verdict:** `TRANSCENDPLEXITY` is a toy simulation with broken packaging and a
self-referential success metric. It is interesting as an idea (`fractional order` =
multi-timescale memory) and as evidence of how the ecosystem's claims are generated, but it
is **not** a base for an LLM and contains no model.

---

## 6. The remaining repos, briefly

| Repo | What it actually contains |
|---|---|
| `tetrahedral-agi` | Tetrahedral grid math + GAIA benchmark plumbing around an `nn.TransformerEncoder`; fine-tune scripts for external models; no evidence of a trained model of its own (results JSONs are near-empty). |
| `TRANSCENDPLEXITY` | Earlier version of parallplexity + "worms engine" state machine + many `*_COMPLETE.md` / `DEPLOYMENT_READY.md` status documents. |
| `compound-integration` | Block/Rule engine ("UMG") with a PyTorch adapter; an orchestration DSL, not a model. |
| `system2-learning` | System-1/System-2 routing framework (complexity estimator → LLM call vs. step-by-step chain). Sensible agent pattern; no model. |
| `octocomputer` | Computer-use agent wiring the above to GUI automation; largest coherent test suite in the account. |
| `Prime-directive`, `twistedWorm-Template` | ~6 MB each, duplicated: Rust "mimicry engine" (claims universal AI mimicry) + Python physics-simulation scripts (Ising "empathy module", "formal proof verifier") that duplicate each other inside the same repo. |
| `SOLVED-562-verified`, `SOLVED---abc82100`, `13-Impossible-ARC-Tasks-SOLVED` | Hundreds of small per-task ARC solvers + result JSONs. Claims (540/540 on ARC-AGI-3, "13/13 where GPT-4/Claude/Gemini got 0") are not independently verifiable from these repos and use task-specific code, which ARC-AGI disallows as "solving". |
| `torus-collider` (Rust, 145 MB), `tinygrad`/`openpilot`/`opendbc` | Tensor-anomaly-detection experiment; mirrors of other people's projects. |
| `NGVT`, `simpleCOMPLEXITY`, `enhanced-multi-pinnacle-torus-*`, `EMERGANT`, `KELDO`, `instasak`, `BEAT-IT`, … | Theory documents, HTML pages, empty or near-empty repos. |

---

## 7. Claims audit

| Claim (from READMEs) | Code reality |
|---|---|
| "quantum coupling matrix" | `np.ones((K,K))*0.1`, randomly updated scalars |
| "RNA-inspired adaptation layers" | Sigmoid gates named after biology; no sequence or biological structure |
| "8-limb cognitive architecture" | 8 parallel NumPy complex vectors, not connected to the trained model |
| "attention on torus topology / vortex dynamics" | Unrunnable tinygrad/torch hybrid; would break causality via `torch.roll` |
| "100% ARC-AGI-1/2/3 (540/540)" | Per-task hand-written solvers; ARC rules disallow this as a general-intelligence claim |
| "eval perplexity 1.32" | True but meaningless: measured with a **non-causal** model (target leakage); the same model cannot generate |
| "generative wall is structural" | **False** — caused by the missing causal mask (this build demonstrates the fix) |
| "deschooled consciousness", "AI consciousness detection" | No operational definition; CLARIN bridge is a doc converter |

---

## 8. What was salvaged into the build (`gmllm/`)

| Idea | Source repo | What we changed |
|---|---|---|
| Geometric (Gaussian) per-head attention bias | `octotetrahedral-agi/core/tetrahedral_attention.py` | **Added the missing causal mask** (the fix); bias is now a learnable ALiBi-Gaussian slope; verified causality by test |
| Torus positional geometry | `torus-attention/src/torus_attention_mechanism.py` | Reimplemented as rotary embeddings with **golden-ratio torus winding** (no periodic aliasing, causality preserved); original mod-M mapping repeats every M·m positions |
| Fractional-order power-law kernels + 8-limb multi-timescale processing | `parallplexity/core/fractional.py`, `emergent/eight_limb.py`, `octotetrahedral-agi/core/reservoir_dynamics.py` | Grünwald–Letnikov FIR applied causally over the sequence with **learnable orders** (init at their per-limb α table); replaces NumPy simulation with a trainable torch module |
| Composite objective (world-model aux loss) | `octotetrahedral-agi/core/recursive_engine_objective.py` | Kept the sensible part (predict next hidden state) at low weight; dropped EWC/grounding metaphors; every term logged |
| Word-level whitespace tokenization | `octotetrahedral-agi/train_transformer.py` | Replaced with **byte-level BPE** (their own RESULTS.md documents UNK/format-token failures this causes) |
| Honest evaluation culture | `octotetrahedral-agi/RESULTS.md` | Kept: measured numbers only, failure modes stated, held-out data, ablations |

**Not salvaged (with reason):** working-memory slots and cognitive-geometry monitors
(not wired into the forward pass / no gradient path — they cannot have helped);
reservoir pacemaker (decorative oscillation, no task); vortex rolling of attention
weights (breaks causality); "quantum"/"consciousness" layers (no operational content);
per-task ARC solvers (not a model).

---

## 9. Reproducing this analysis

```powershell
# sources downloaded to:
$env:TEMP\opencode\gitmonsters\
# key files
octotetrahedral-agi\train_transformer.py      # line 292: transformer(x) — no causal mask
octotetrahedral-agi\core\tetrahedral_transformer_layer.py   # line 113: Gaussian |i−j| bias
torus-attention\src\torus_attention_mechanism.py            # tinygrad/torch API mixing
parallplexity\parallplexity\core\fractional.py             # real GL/Caputo kernels
```

The build that fixes and supersedes this architecture is documented in `README.md`;
measured results are in `RESULTS.md`.
