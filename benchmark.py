"""Wall-clock latency, peak memory and FLOP-counter cross-check for trained models.

Every configuration runs in a fresh subprocess so peak-RSS measurements do not leak
between configurations. All timings use ONE CPU thread (torch.set_num_threads(1)),
fixed seeds, warm-up runs, and report the median over repetitions.

  latency_b1    median seconds per answer at batch size 1 (true per-example early exit
                for adaptive D; real KV-cached greedy decoding for B)
  latency_b256  median seconds per batch of 256 / 256 (throughput setting; adaptive D
                removes halted examples from the batch as it goes)
  peak_mem_mb   increase of process peak RSS while running one batch of 256
                (inference activations + allocator overhead; weights are excluded
                because they are loaded before the baseline is taken)
  flops_counted FLOPs per answer measured by torch.utils.flop_counter for one batch,
                compared against the analytic count used everywhere else.

Energy is NOT measured: no RAPL/powercap interface is exposed in this container
(/sys/class/powercap is absent), and CPU time is not a valid proxy for energy.

Usage: python benchmark.py            # all configs, writes results/benchmark.json
       python benchmark.py --smoke    # one quick config
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from tasks import eval_sets
from train import build, run_dir, MODELS

LEVELS = {"addition": [8, 24], "chain": [6, 12]}
CONFIGS = [("A", None), ("Adeep", None), ("B", None), ("C", 8), ("C", 16),
           ("D", 2), ("D", 6), ("D", "adaptive")]
S_MAX = 32


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@torch.no_grad()
def d_adaptive(model, x):
    """Adaptive-halting inference with early exit: halted examples leave the batch."""
    e = model.embed(x)
    rope = model.rope(x.shape[1], x.device)
    y, z = model.init_state(x)
    B = x.shape[0]
    preds = torch.zeros(B, x.shape[1], dtype=torch.long)
    active = torch.arange(B)
    steps = torch.zeros(B, dtype=torch.long)
    for s in range(1, S_MAX + 1):
        y, z, logits, q = model.step(e, y, z, rope)
        stop = (q > 0) | (s == S_MAX)
        idx = active[stop]
        preds[idx] = logits[stop].argmax(-1)
        steps[idx] = s
        keep = ~stop
        if not keep.any():
            break
        active, e, y, z = active[keep], e[keep], y[keep], z[keep]
    return preds, steps


def make_runner(model, name, iters, b):
    arch = MODELS[name]["arch"]
    if arch == "B":
        dx = torch.from_numpy(b["dec_x"])
        return lambda sl: model.generate(dx[sl], b["dec_forced"])
    x = torch.from_numpy(b["enc_x"])
    if arch == "A":
        return lambda sl: model(x[sl])
    if arch == "C":
        return lambda sl: model(x[sl], iters)
    if iters == "adaptive":
        return lambda sl: d_adaptive(model, x[sl])
    return lambda sl: model(x[sl], iters, checkpoints={iters})


def analytic(model, name, iters, b, steps=None):
    arch = MODELS[name]["arch"]
    n = b["enc_x"].shape[1]
    if arch == "B":
        return model.flops_generate(b["dec_forced"])
    if arch == "A":
        return model.flops(n)
    if arch == "C":
        return model.flops(n, iters)
    if iters == "adaptive":
        return float(steps.float().mean()) * model.flops_per_step(n)
    return model.flops(n, iters)


def bench_one(task, name, iters, level, seed=0, reps=30):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = build(task, name)
    model.load_state_dict(torch.load(os.path.join(run_dir(task, name, seed), "model.pt")))
    model.eval()
    b = eval_sets(task, [level], 256)[level]
    run = make_runner(model, name, iters, b)
    with torch.no_grad():
        # memory: peak RSS increase during one full batch (after a batch-1 warm-up so
        # one-off runtime/kernel initialisation is not attributed to the model)
        run(slice(0, 1))
        base = peak_rss_mb()
        out = run(slice(0, 256))
        mem = peak_rss_mb() - base
        with FlopCounterMode(display=False) as fc:
            run(slice(0, 256))
        counted = fc.get_total_flops() / 256
        steps = out[1] if iters == "adaptive" else None
        flops = analytic(model, name, iters, b, steps)
        # throughput at batch 256
        t = []
        for _ in range(5):
            t0 = time.perf_counter(); run(slice(0, 256)); t.append(time.perf_counter() - t0)
        lat256 = float(np.median(t)) / 256
        # latency at batch 1 (different examples, after warm-up)
        for i in range(5):
            run(slice(i, i + 1))
        t = []
        for i in range(reps):
            t0 = time.perf_counter(); run(slice(i, i + 1)); t.append(time.perf_counter() - t0)
        lat1 = float(np.median(t))
    return dict(task=task, model=name, iters=iters, level=level, flops_analytic=flops,
                flops_counted=counted, latency_b1_ms=lat1 * 1e3, latency_b256_ms=lat256 * 1e3,
                peak_mem_mb=mem, params=sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=4, metavar=("TASK", "MODEL", "ITERS", "LEVEL"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.one:
        task, name, iters, level = a.one
        iters = None if iters == "none" else (iters if iters == "adaptive" else int(iters))
        print(json.dumps(bench_one(task, name, iters, int(level))))
        sys.exit()
    jobs = [(t, m, i, l) for t in LEVELS for l in LEVELS[t] for m, i in CONFIGS]
    if a.smoke:
        jobs = jobs[:1]
    results = []
    for t, m, i, l in jobs:
        p = subprocess.run([sys.executable, __file__, "--one", t, m, str(i).lower() if i else "none", str(l)],
                           capture_output=True, text=True, check=True)
        r = json.loads(p.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"{t:8s} L={l:<3d} {m:6s} iters={str(i):8s} flops={r['flops_analytic']:.3g} "
              f"(counted {r['flops_counted']:.3g})  b1={r['latency_b1_ms']:.2f}ms  "
              f"b256={r['latency_b256_ms']:.3f}ms/ex  mem={r['peak_mem_mb']:.1f}MB")
    if not a.smoke:
        meta = dict(threads=1, torch=torch.__version__, cpu_count=os.cpu_count(),
                    energy="not measured (no RAPL/powercap interface available)")
        with open("results/benchmark.json", "w") as f:
            json.dump(dict(meta=meta, results=results), f, indent=1)
