# Efficient Intelligence — compact latent reasoning vs. Transformer baselines

An exploratory, CPU-scale experiment asking:

> Can a compact architecture that reasons by iterating an internal state (no reasoning
> tokens) solve harder problems more *compute-efficiently* than a standard Transformer,
> a scratchpad (chain-of-thought) Transformer and a looped Transformer?

Efficiency is judged as **accuracy per inference FLOP**, under a **matched training-FLOP
budget**, never by accuracy or parameter count alone. These are toy algorithmic tasks
with models under 0.5M parameters; nothing here says anything direct about frontier-scale
language models.

- Literature review, what is and is not novel, and hypotheses: [RESEARCH.md](RESEARCH.md)
- Results, analysis and limitations: [results/RESULTS.md](results/RESULTS.md)
- Generated tables (all numbers): [results/summary.md](results/summary.md); plots in `results/plots/`

## Models (`models/`)

All four use the same pre-LN Transformer block (d=64, 4 heads, MLP 4d), the same input
encoding and the same output vocabulary. "One block applied to one token" is the unit
of compute.

| | File | Description | Params |
|---|---|---|---|
| A | `transformer.py` | 4 distinct layers, predicts every answer position directly. **A-deep**: 8 layers. | 232k / 431k |
| B | `scratchpad.py` | 4-layer causal decoder; writes intermediate reasoning tokens, then the answer; greedy KV-cached decoding. All processed tokens are counted. | 232k |
| C | `looped.py` | One shared block looped T times with input injection; random T∈[4,12] in training (mean 8); any T at test. | 83k |
| D | `latent.py` | TRM-style recursion of an answer state y and latent state z with one shared block; deep supervision with detached states; recall; learned halting head for adaptive steps. One outer step = 3 block applications. Trained with S=6 outer steps. | 83k |

Ablations: `C_fixed` (T=8 always), `D_nodeep` (loss only after the last step, full
back-propagation), `D_inner1` (1 latent update per step), `D_S12` (12 supervised steps).

## Tasks (`tasks/`)

- **addition** — n-digit addition, digits aligned and least-significant first, with
  forced long carry chains. Train n=1–8; test n=2,4,6,8 (held-out problems) and n=12,16,24,32 (unseen lengths).
  Scratchpad for B: an explicit carry token after every output digit.
- **chain** — multi-hop variable resolution with modular arithmetic: shuffled
  statements `v_j = v_{j-1} + c (mod 10)` for a queried chain and an equal-length
  distractor chain. Train k=1–6 hops; test k=2,4,6 (held-out) and 8,12,16,24 (unseen).
  Scratchpad for B: backtrack the chain names, then forward values. Chance = 10%.

Evaluation sets are generated from fixed seeds; any training example identical to an
in-distribution test problem is discarded (checked by hash).

## Setup

```bash
pip install -r requirements.txt          # CPU PyTorch is enough
# (optional) pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproduce

```bash
bash run_all.sh                    # everything: ~1-1.5 h on 4 CPU cores
# or individually:
python train.py    --task chain --model D --seed 0     # trains to the 8e12-FLOP budget
python evaluate.py --task chain --model D --seed 0     # accuracy x compute on all levels
python benchmark.py                                    # latency, memory, FLOP cross-check
python analysis/plots.py                               # tables + plots
```

Seeds 0, 1, 2 for main models; ablations use seed 0. Training uses one thread per run
with four runs in parallel; benchmarks run sequentially with one thread.

## How compute is measured

- **FLOPs** are counted analytically for every matmul: projections, MLP, attention
  scores and values (full N×N for encoders and for B's teacher-forced training; growing
  KV length for B's decoding), the output head, and the halting head. LayerNorm, GELU,
  softmax and embedding lookups are excluded (small at this size). The analytic count
  is checked against `torch.utils.flop_counter` in `benchmark.py` (they agree).
- **Training compute** = 3 × forward FLOPs of all differentiated computation; every run
  stops at the same budget (8e12 FLOPs), so models that cost more per example see
  fewer examples.
- **Latency** is wall-clock CPU time (1 thread, median). **Memory** is the peak-RSS
  increase during a batch of 256. **Energy was not measured** — no power interface
  is available in this container, and FLOPs/time are not energy measurements.
