#!/usr/bin/env bash
# Reproduces Experiment 2 (stopping rules for the looped model). Needs C_3x_s0 from run_all.sh
# (or trains it here). ~1 h on 4 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
LEVELS="2 4 6 8 12 16 24 30"
{ for s in 0 1 2 3 4; do echo "C_3x $s"; echo "A_3x $s"; echo "B_3x $s"; done; } > results/jobs_halting_all.txt
xargs -P 4 -L 1 sh -c '[ -f results/runs/chain/$0_s$1/model.pt ] || python3 train.py --task chain --model $0 --seed $1 --budget 2.4e13 > /dev/null' < results/jobs_halting_all.txt
xargs -P 4 -L 1 sh -c "python3 evaluate.py --task chain --model \$0 --seed \$1 --levels $LEVELS" < results/jobs_halting_all.txt
for s in 0 1 2 3 4; do python3 halting.py --seed $s & done; wait
python3 halting.py --latency
python3 analysis/halting_plots.py
python3 analysis/plots.py
