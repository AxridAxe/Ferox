# Results and analysis

All numbers below are copied from [summary.md](summary.md), which `analysis/plots.py`
generates from the JSON files in `results/runs/` and `results/benchmark.json`.
Hardware: 4-core CPU, no GPU; PyTorch 2.15 (CPU build). Main models use 3 seeds and
ablations 1 seed. Every main and ablation run has the same budget of 8e12 training
FLOPs (about 4–7 minutes on one thread). Accuracy is exact match on the whole answer.

## 1. What ran

| | runs |
|---|---|
| Main comparison | 2 tasks × {A, A-deep, B, C, D} × 3 seeds = 30 |
| Ablations (seed 0) | 2 tasks × {C_fixed, D_nodeep, D_inner1, D_S12} = 8 |
| Post-hoc diagnostic (chain, 3× budget) | A_3x s0, C_3x s0, D_3x s0, D_3x s1 = 4 |
| Test-time compute sweeps | C at T ∈ {1…48}; D at S ∈ {1…32} and adaptive halting (cap 32) |
| Benchmarks | latency (batch 1 and 256), peak-memory increase, FLOP-counter cross-check: 32 configurations |

The 3× diagnostic was added **after** seeing that D never left chance on the chain
task. It is not budget-matched to the main table and is reported separately.

## 2. Main findings

### Addition (train 1–8 digits)

| model | acc@8 (ID) | acc@12 | acc@16 | acc@24 | FLOPs/answer @8 |
|---|---|---|---|---|---|
| A (4L) | 99.9 ± 0.1 | 98.4 ± 1.4 | **94.7 ± 2.8** | 3.3 | 3.8e6 |
| A-deep (8L) | 100 | 93.5 ± 7.5 | 84.7 ± 18.1 | 2.3 | 7.4e6 |
| B scratchpad | 100 | **98.9 ± 1.2** | 61.8 ± 8.6 | 0.1 | 9.7e6 |
| C looped, T=8 | 100 | 96.1 ± 4.6 | 78.5 ± 6.9 | 1.5 | 7.4e6 |
| D latent, S=6 | 99.8 | 82.3 ± 14.3 | 58.1 ± 11.7 | 0.4 | 1.7e7 |
| D latent, adaptive | 99.7 ± 0.1 | 85.4 ± 10.5 | 63.9 ± 6.9 | 0.5 | **2.9e6** |

- Every model solves the training range. Beyond it, the plain 4-layer Transformer
  extrapolates best. The recurrent models (C, D) are *worse* at 16 digits. No model
  handles 24 or 32 digits, a known limit of RoPE length generalisation (Zhou 2024).
- Extra iterations help C only up to T≈3–4 (accuracy saturates there). Beyond
  T≈24, accuracy drops ("overthinking"; see `addition_acc_vs_flops.png`).
- D reaches its accuracy after **one** outer step (3 block applications) and never
  improves with more steps. Its halting head learned to stop after step 1 on every
  example, in distribution *and* out of distribution. This gives the cheapest
  in-distribution point of any model (2.9e6 FLOPs, 99.7%). But the model is effectively
  a 3-layer weight-tied network, not an iterative reasoner, and its halting is
  over-confident on 12–16-digit inputs it gets wrong.
- The scratchpad model costs 2.6× A's FLOPs per answer in distribution with no
  accuracy gain. At batch 1 it is 20–30× slower in wall-clock time, because decoding is sequential.

### Chain (multi-hop, train 1–6 hops, chance 10%)

| model | acc@4 | acc@6 (ID) | acc@8 | acc@12 | seeds that learned the task |
|---|---|---|---|---|---|
| A (4L) | 93.3 ± 2.7 | 11.2 | 10 | 10 | 0/3 at 6 hops |
| A-deep (8L) | 58.6 ± 34 | 10.9 | 10 | 8 | 0/3 |
| B scratchpad | 43.3 ± 36 | 26.9 ± 17 | 12 | 11 | 1/3 partly (51% @6) |
| C looped, T=8 | 69.9 ± 42 | 70.7 ± 41 | 11 | 10 | **2/3** (99–100% @6) |
| D latent | 10.6 | 9.8 | 11 | 10 | **0/3 (chance)** |

The mean ± std hides a bimodal outcome: each seed either "gets" the task or stays
at chance. The per-seed table in summary.md is the honest view.

**Test-time iteration on the looped model works here.** The seeds of C that learned
the task, evaluated with more iterations than they were trained with, reach
96–99% at 8 hops (T=12) and 62–89% at 12 hops (T=16–24), on chain lengths never seen
in training. The 3× budget run C_3x reaches 99% at 8 hops, 97% at 12, 96% at 16 and
67% at 24. The number of iterations needed grows about linearly with hop count (≈k+2;
`chain_3x_diagnostic.png`). The model appears to have learned a one-hop-per-iteration
algorithm, the easy-to-hard extrapolation reported for Deep Thinking nets
(Bansal 2022) and recurrent-depth Transformers.
**Caveat:** these OOD numbers use the best T for each level (oracle selection). C has no
stopping rule, and at the default T=8 its OOD accuracy is at chance.

Neither A model extrapolates to more hops at any budget (A_3x: 89% at 6 hops, chance
at 8 hops).

### Matched inference compute (chain, 6 hops, mean over seeds)

| reference | its FLOPs | its acc | C at ≤ same FLOPs | D at ≤ same FLOPs |
|---|---|---|---|---|
| A (4L) | 6.4e6 | 11.2 | T=4: 10.9 | S=1: 10.1 |
| A-deep (8L) | 1.25e7 | 10.9 | **T=8: 70.7** | S=2: 10.6 |
| B scratchpad | 1.16e7 | 26.9 | T=6: 11.1 | S=2: 10.6 |

At identical FLOPs (C at T=8 vs. 8 distinct layers), weight-tied recurrence beat added
depth on chain. It did not on addition (C T=8 96.1% vs. A-deep 93.5% at 12 digits, but
78.5% vs. 84.7% at 16 digits, both within seed noise).

## 3. Verdict on the hypotheses

| | Verdict | Evidence |
|---|---|---|
| **H1** recurrence ≥ depth at matched FLOPs, better OOD | **Supported on chain, not on addition** | Chain: C beats A-deep at equal FLOPs (2/3 seeds) and extrapolates with more iterations; addition: no advantage, slightly worse OOD. |
| **H2** latent model D matches the scratchpad's accuracy with fewer FLOPs | **Not supported as a latent-reasoning effect** | On addition D adaptive is 3.4× cheaper than B at equal ID accuracy, but A and C at T=4 are just as cheap; the saving comes from *not generating tokens*, not from D's mechanism. On chain, D failed at the matched budget. |
| **H3** adaptive halting saves compute at equal accuracy | **Partly** | Addition: 6× fewer FLOPs than fixed S=6 at equal accuracy, but only because D does nothing after step 1, and it halts over-confidently OOD. Chain (3×): halting spends more steps on harder inputs (≈2 steps at 4 hops, ≈22 at 8 hops), but stays below the best fixed S and runs to the cap on OOD inputs it cannot solve (most expensive *and* wrong). |
| **H4** deep supervision beats end-only training | **Not supported** (1 seed) | Addition: D_nodeep ≥ D OOD (94.6 vs. 82.3 at 12 digits). Chain: both at chance. D_S12 (12 supervised steps) was much worse; it saw half as many examples. |

## 4. Which mechanisms helped and which did not

- **Helped: weight-tied recurrence + input injection + random training depth (C).**
  Best parameter efficiency (83k params vs. 232k–431k) and the only model that
  extrapolated to harder chain problems. C_fixed (always T=8) also extrapolated to 8 hops,
  but reached only 16% at 12 hops versus 62–89% for random-depth C. This is one seed, but it
  matches Geiping 2025 and Bansal 2022.
- **Hurt at a fixed budget: deep supervision with an optimiser step per outer step (D).**
  Each training example costs S×(n_inner+1) = 18 block applications, so D saw 2.3× fewer
  examples than C and 4.6× fewer than A. Chain learning has a long plateau and then a
  sudden drop. D did not reach the drop at 8e12 FLOPs, but did at 2.4e13 (D_3x s0: 81% at 6
  hops). It was still less sample-efficient than C, and showed **no** extrapolation (≤32% at 8 hops).
- **Did not help: the two-state (y, z) TRM-style recursion** over a single looped
  state. Nothing here isolates a benefit. D's best results come at 2–4 outer steps,
  and it degrades with more steps on both tasks.
- **Mixed: learned halting.** It is cheap when confident, but not calibrated out of distribution.
- **Explicit scratchpad (B)** was the costliest per answer and the slowest. It was unstable
  to train on chain (1/3 seeds) and did not extrapolate in length on either task.
  Its scratchpad format and positional scheme may be sub-optimal (see limitations).

## 5. Cost measurements (summary.md, "Latency / memory")

- Analytic FLOPs equal the `torch.utils.flop_counter` count exactly in all 32 configurations.
- Batch-1 latency (1 thread) roughly follows FLOPs for the encoder models (≈1–6 ms). B is
  8–64 ms because each reasoning token is a separate sequential forward pass. At batch 256
  B is 3–7.5× slower per answer than A.
- Peak-RSS increase during a 256-example batch: 9–47 MB for all models. B is highest at
  every setting (19–47 MB, from the KV cache plus longer sequences). C and D are within a
  few MB of each other and of A at equal length. These are RSS measurements that include allocator behaviour, not exact
  activation sizes.
- Training: every run used 8e12 FLOPs, taking 237–406 s with 4 runs in parallel.
- **Energy was not measured.** No power interface is available, and nothing here supports an energy claim.

## 6. Limitations and confounders

1. **Toy scale.** Under 0.5M parameters, d=64, two synthetic tasks, minutes of CPU training.
   Nothing here transfers to language models without new evidence.
2. **Seed variance dominates the chain task.** With 3 seeds and bimodal outcomes, mean
   differences such as C 70.7 vs. B 26.9 are not statistically robust. Ablations and
   diagnostics use 1–2 seeds.
3. **Oracle iteration counts.** The best OOD results for C and D pick T/S on the test level.
   A deployable system needs a halting rule, and D's learned halting did not deliver one OOD.
4. **Training budget.** One budget per task, with no hyper-parameter tuning for any model
   (same LR 1e-3, AdamW, batch 64). Recurrent models may prefer other settings. D is clearly
   budget-limited on chain, so a larger budget could change the D vs. C ordering in distribution.
5. **Format choices favour some models.** Addition uses an aligned, LSB-first encoding that
   makes direct prediction easy and the scratchpad unnecessary. B's scratchpad and RoPE
   positions were not tuned; index hints or Abacus embeddings (McLeish 2024) would likely
   help B and every other model at 24+ digits. Chain uses no positional encoding.
6. **Dense supervision.** A, C and D are supervised at every position (all variable values),
   and B on its scratchpad tokens. This is extra signal that a pure answer-only setup would not have.
7. **D is one particular adaptation** of TRM/HRM (1 grad cycle per step, no no-grad cycles,
   no EMA, no data augmentation, sequence tasks rather than grid puzzles). A negative result
   for this D is not a negative result for TRM.
8. **FLOPs exclude** LayerNorm, softmax, GELU and embeddings (≤ a few %), and count full N×N
   attention for teacher-forced causal training.

## 7. Conclusion

At this scale, **compact iterative computation was useful mainly in its simplest form**:
a weight-tied looped block that can be run longer on harder inputs. On the multi-hop task,
it matched or beat a non-recurrent Transformer at equal FLOPs with ~5× fewer parameters, and
extrapolated to 2–4× more hops than it was trained on. It did so with compute that grows with
problem difficulty and without writing any reasoning tokens. **The more elaborate latent
reasoner (D: two-state recursion, deep supervision, learned halting) showed no advantage**: it
was less sample-efficient under a fixed training budget, did not extrapolate, and its halting
was miscalibrated out of distribution. The measured efficiency gain over the scratchpad
model comes from avoiding token-by-token decoding, which all direct-prediction models share.

So: compact internal iteration is a *credible* route to compute that scales with difficulty
without reasoning tokens, but this experiment provides no evidence that it preserves
capability more efficiently than simply looping a small Transformer. It also provides
no evidence about frontier-scale reasoning.

## 8. Next most valuable experiment

**Give the looped model a reliable stopping rule and test it at matched compute against
the oracle.** C's OOD success depends on choosing T, and without that it is not deployable.
Concretely: train C with a per-example halting signal (PonderNet-style or a fixed-point
criterion ‖h_{t+1}−h_t‖<ε, which needs no learned head). Use 5+ seeds and a 2–3× budget.
Report accuracy vs. FLOPs on chain up to 32 hops against (i) oracle T, (ii) fixed T, and
(iii) B with a tuned scratchpad (index hints). If halting recovers most of the oracle curve,
the iterative approach is genuinely adaptive. If not, the extrapolation result is mostly
an oracle artifact. A second priority is
repeating D with its budget matched by *examples seen* as well as by FLOPs, to separate
"D is sample-inefficient" from "D is computationally wasteful".
