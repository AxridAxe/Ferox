#!/usr/bin/env bash
# Reproduces every experiment in results/. ~1-1.5 h on 4 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-4}
{
  for task in addition chain; do
    for seed in 0 1 2; do for m in A Adeep B C D; do echo "$task $m $seed"; done; done
    for m in C_fixed D_nodeep D_inner1 D_S12; do echo "$task $m 0"; done
  done
} > results/jobs.txt
# 1. training (one thread per run, JOBS runs in parallel)
xargs -P "$JOBS" -L 1 sh -c 'python3 train.py --task $0 --model $1 --seed $2 > /dev/null' < results/jobs.txt
# 2. evaluation on fixed ID / OOD test sets
xargs -P "$JOBS" -L 1 sh -c 'python3 evaluate.py --task $0 --model $1 --seed $2' < results/jobs.txt
# 3. latency / memory / FLOP-counter cross-check (sequential, so timings do not interfere)
python3 benchmark.py
# 4. tables and plots
python3 analysis/plots.py
# 5. post-hoc diagnostic: is chain-task failure budget-limited? (3x training FLOPs)
for j in "A_3x 0" "C_3x 0" "D_3x 0" "D_3x 1"; do set -- $j
  python3 train.py --task chain --model $1 --seed $2 --budget 2.4e13 > /dev/null &
done; wait
for j in "A_3x 0" "C_3x 0" "D_3x 0" "D_3x 1"; do set -- $j; python3 evaluate.py --task chain --model $1 --seed $2; done
python3 analysis/plots.py
