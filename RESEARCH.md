# Research notes: compact iterative (latent) reasoning

## How sources were checked

Every arXiv paper below was resolved through the arXiv API on 2026-10-30 to
confirm its identifier, title, first author and year; the summaries of what each
paper *claims* are taken from its abstract, not from memory. The ARC Prize HRM
analysis was fetched from arcprize.org on the same day. I did **not** re-read the
full text of each paper, so finer details (exact hyperparameters, the precise
form of a loss) are from my background knowledge and are marked *(unverified
detail)* where they matter. Nothing in this document is a result of this project
except the section "Hypotheses", which is tested in `results/`.

## 1. Prior work

### Adaptive and recurrent depth

| Work | What it does | What it showed (per abstract) |
|---|---|---|
| **Adaptive Computation Time (ACT)** — Graves 2016, [arXiv:1603.08983](https://arxiv.org/abs/1603.08983) | RNN emits a halting probability each step; a ponder cost penalises steps. | Learned variable compute per input. |
| **Universal Transformer** — Dehghani et al. 2018, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819) | One Transformer block applied recurrently over depth, optionally with per-position ACT. | Weight-shared depth recurrence; better on algorithmic tasks than a vanilla Transformer. |
| **PonderNet** — Banino et al. 2021, [arXiv:2107.05407](https://arxiv.org/abs/2107.05407) | Probabilistic halting with a KL prior on the number of steps. | Better than earlier adaptive-compute methods on a synthetic task and extrapolation tests. |
| **Deep Thinking networks** — Schwarzschild et al. 2021, [arXiv:2106.04537](https://arxiv.org/abs/2106.04537); Bansal et al. 2022, [arXiv:2202.05826](https://arxiv.org/abs/2202.05826) | Weight-tied recurrent (conv) nets trained on small problems, run longer on big ones. *Recall* (re-inject the input each iteration) and *progressive training* (stop gradients through a random prefix of iterations). | Easy-to-hard extrapolation (prefix sums, mazes, chess) by adding test-time iterations; "overthinking" (degradation when iterated too long) mitigated by recall + progressive loss. |
| **Looped Transformers as programmable computers** — Giannou et al. 2023, [arXiv:2301.13196](https://arxiv.org/abs/2301.13196) | Theory: a looped shallow Transformer can emulate general-purpose programs. | Expressivity, not learnability. |
| **Looped Transformers learn learning algorithms** — Yang et al. 2023, [arXiv:2311.12424](https://arxiv.org/abs/2311.12424) | Looped Transformer with input injection for in-context regression. | Matches a standard Transformer with <10% of the parameters. |
| **Reasoning with Latent Thoughts** — Saunshi et al. 2025, [arXiv:2502.17416](https://arxiv.org/abs/2502.17416) | k-layer block looped L times vs. kL-layer model. | On addition, p-hop induction and math, looped ≈ deep non-looped at equal *effective depth*, and ≫ the k-layer model; looped models can simulate T CoT steps with T loops. **Note: their comparison is at matched FLOPs, where looping saves parameters, not compute.** |
| **Recurrent-depth LM (Huginn)** — Geiping et al. 2025, [arXiv:2502.05171](https://arxiv.org/abs/2502.05171) | 3.5B LM with a recurrent core unrolled to arbitrary depth; random depth in training. | Reasoning benchmarks improve with test-time recurrence up to compute equivalent to a 50B model. |
| **Mixture-of-Depths** — Raposo et al. 2024, [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) | Routes only some tokens through each block. | Token-level adaptive compute in LMs. |

### Recursive reasoners with deep supervision

| Work | What it does | What it showed |
|---|---|---|
| **Hierarchical Reasoning Model (HRM)** — Wang et al. 2025, [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) | Two coupled recurrent modules at two timescales, one-step gradient approximation, deep supervision across "segments", ACT-style Q-learning halting. 27M params, ~1000 training examples. | Near-perfect hard Sudoku / large mazes; strong ARC-AGI results without pretraining or CoT. |
| **ARC Prize analysis of HRM** — ARC Prize team, 15 Aug 2025, [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis) | Ablations on ARC-AGI. | A same-size plain Transformer comes within ~5 pp; the **outer refinement loop** (deep supervision) drives most of the gain; training with refinement matters more than refining at inference. |
| **Tiny Recursive Model (TRM)** — Jolicoeur-Martineau 2025, [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) | Single 2-layer network recursively updating a latent z and an answer y; deep supervision; simplified halting. 7M params. | 45% ARC-AGI-1 / 8% ARC-AGI-2, higher than HRM. |

### Latent vs. explicit (token) reasoning

| Work | What it does | What it showed |
|---|---|---|
| **Scratchpads** — Nye et al. 2021, [arXiv:2112.00114](https://arxiv.org/abs/2112.00114); **Chain-of-Thought** — Wei et al. 2022, [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) | Emit intermediate steps as tokens. | Large gains on multi-step tasks; cost grows with the number of reasoning tokens. |
| **Coconut** — Hao et al. 2024, [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) | Feed the LM's last hidden state back as the next input embedding ("continuous thought") instead of decoding a token. | Beats CoT on search-heavy logical tasks with a better accuracy/efficiency trade-off; continuous thoughts can hold several candidate next steps (BFS-like). |
| **Mamba** — Gu & Dao 2023, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Selective state-space model: linear-time sequence mixing with a fixed-size recurrent state. | Transformer-quality LM at linear cost in sequence length. Relevant as a compact *sequence* state, not as depth recurrence. |

### Arithmetic and length generalisation (affects task design)

| Work | Relevance |
|---|---|
| Zhou et al. 2023, [arXiv:2310.16028](https://arxiv.org/abs/2310.16028) | Transformers length-generalise on tasks expressible as short RASP-L programs; data format matters. |
| Zhou et al. 2024, [arXiv:2402.09371](https://arxiv.org/abs/2402.09371) | Addition length generalisation (~2.5x) is achievable with the right format + position encoding, but is **fragile across seeds**. |
| McLeish et al. 2024, [arXiv:2405.17399](https://arxiv.org/abs/2405.17399) | Digit-position ("Abacus") embeddings plus input injection and recurrent layers reach high accuracy on 100-digit addition after training on 20 digits. |

## 2. What is demonstrated vs. uncertain

**Reasonably well demonstrated**
- Weight-tied recurrence can reach the accuracy of a deeper non-tied model at *equal depth/FLOPs* with far fewer parameters (Yang 2023, Saunshi 2025).
- Recurrent models can extrapolate to harder instances by running more iterations, if trained with recall and progressive/random-depth training (Bansal 2022; Geiping 2025; McLeish 2024).
- Deep supervision / outer refinement loops are a strong ingredient for small recursive reasoners on puzzle tasks (HRM, TRM, ARC Prize ablation).
- Latent "thoughts" can replace some CoT tokens (Coconut).

**Uncertain / not established**
- Whether latent iteration is more *compute*-efficient than explicit CoT, as opposed to parameter-efficient. Most looped-model results match FLOPs (so the savings are parameters/memory), and HRM/TRM report accuracy with many recursion steps, not accuracy per FLOP against a CoT model with equal compute.
- How much of HRM/TRM's advantage is the architecture versus deep supervision, data augmentation and test-time voting (the ARC Prize analysis suggests mostly the latter).
- Whether learned halting saves meaningful compute in practice, and whether it remains calibrated out of distribution.
- Whether any of this transfers from puzzles to open-ended language reasoning at scale (Geiping 2025 is the main large-scale evidence; it is a single model family).

## 3. Where this project sits (and what it does *not* claim)

Model D is **not novel**: it is a TRM-style two-state recursion (from TRM/HRM) with
deep supervision and detached states (HRM/TRM), input re-injection (recall;
Bansal 2022) and a halting head (TRM's Q-head, a simplification of ACT/PonderNet),
applied per-position to sequence tasks. Model C is the looped/recurrent-depth
Transformer of Yang 2023 / Saunshi 2025 / Geiping 2025 with random training depth.
The contribution here is only the *controlled comparison*: four model families on
the same tokenisation, the same training-FLOP budget, and accuracy plotted against
**inference FLOPs per answer** — including every scratchpad token — on in-distribution
and harder unseen problems.

## 4. Hypotheses (tested in `results/`)

- **H1 (recurrence vs. depth).** At matched inference FLOPs, a looped model (C) is
  at least as accurate as non-recurrent Transformers (A, A-deep) in distribution
  and better on harder OOD instances. *Expected from Saunshi 2025 / Bansal 2022.*
- **H2 (latent vs. explicit reasoning).** A latent recursive model (D) reaches the
  scratchpad model's (B) accuracy with fewer inference FLOPs per answer.
- **H3 (adaptive computation).** Learned halting in D reduces average FLOPs at
  equal accuracy versus a fixed number of steps, and spends more steps on harder
  inputs.
- **H4 (mechanisms).** Deep supervision with detached states (D) beats end-only
  supervision with full back-propagation (D_nodeep) at equal training FLOPs.

Falsification criteria: H2 fails if, at every difficulty level, some baseline
reaches D's accuracy at equal or lower FLOPs/answer, or D cannot reach B's
accuracy at any budget. H3 fails if adaptive D is not on or above the fixed-S
accuracy-vs-FLOPs curve.
