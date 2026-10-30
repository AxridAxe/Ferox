"""Controlled evaluation on fixed held-out (in-distribution) and harder OOD test sets.

For every trained run and every difficulty level, records accuracy together with the
analytic inference FLOPs per answer:
  A / Adeep : one point (fixed depth)
  B         : one point; FLOPs include every given and generated token (KV cache)
  C         : one point per iteration budget T (test-time compute sweep)
  D         : one point per fixed outer-step budget S, plus adaptive halting
              (stop at the first step whose halting head says "correct", cap S_MAX);
              adaptive FLOPs are averaged over the per-example steps actually used.
Accuracy = exact match of the complete answer (all digits / the queried value).

Usage:  python evaluate.py --task addition --model D --seed 0
        python evaluate.py --all            # every run under results/runs
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from tasks import eval_sets, TASKS
from train import build, run_dir, MODELS

C_ITERS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
D_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
S_MAX = 32


def exact(pred, y, ans):
    return ((pred == y) | ~ans).all(-1).float()


@torch.no_grad()
def eval_run(task, name, seed, levels=None, size=None):
    d = run_dir(task, name, seed)
    model = build(task, name)
    model.load_state_dict(torch.load(os.path.join(d, "model.pt")))
    model.eval()
    arch = MODELS[name]["arch"]
    sets = eval_sets(task, levels) if size is None else eval_sets(task, levels, size)
    res = {}
    for lvl, b in sets.items():
        r = dict(level=lvl, ood=lvl >= TASKS[task].OOD_FROM)
        if arch == "B":
            dx = torch.from_numpy(b["dec_x"])
            seq, f = model.generate(dx, b["dec_forced"])
            ans = b["dec_ans"]
            acc = (seq[:, ans] == dx[:, ans, 2]).all(-1).float().mean().item()
            r["points"] = [dict(iters=None, flops=f / len(dx), acc=acc,
                                gen_tokens=int((~b["dec_forced"]).sum()))]
        else:
            x = torch.from_numpy(b["enc_x"])
            y = torch.from_numpy(b["enc_y"])
            ans = torch.from_numpy(b["enc_ans"])
            n = x.shape[1]
            if arch == "A":
                acc = exact(model(x).argmax(-1), y, ans).mean().item()
                r["points"] = [dict(iters=len(model.blocks), flops=model.flops(n), acc=acc)]
            elif arch == "C":
                outs = model(x, max(C_ITERS), checkpoints=set(C_ITERS))
                r["points"] = [dict(iters=t, flops=model.flops(n, t),
                                    acc=exact(outs[t].argmax(-1), y, ans).mean().item())
                               for t in C_ITERS]
            else:
                outs = model(x, S_MAX)
                r["points"] = [dict(iters=s, flops=model.flops(n, s),
                                    acc=exact(outs[s][0].argmax(-1), y, ans).mean().item())
                               for s in D_STEPS]
                # adaptive halting: first step with sigmoid(q) > 0.5, else S_MAX
                B = x.shape[0]
                halted = torch.zeros(B, dtype=torch.bool)
                used = torch.full((B,), S_MAX)
                correct = torch.zeros(B)
                for s in range(1, S_MAX + 1):
                    logits, q = outs[s]
                    stop = (~halted) & ((q > 0) | (s == S_MAX))
                    used[stop] = s
                    correct[stop] = exact(logits.argmax(-1), y, ans)[stop]
                    halted |= stop
                fps = model.flops_per_step(n)
                r["adaptive"] = dict(acc=correct.mean().item(), mean_steps=used.float().mean().item(),
                                     flops=(used.float() * fps).mean().item(),
                                     max_steps=int(used.max()), s_max=S_MAX)
        res[lvl] = r
    out = dict(task=task, model=name, seed=seed, levels=res)
    with open(os.path.join(d, "eval.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.all:
        jobs = []
        for p in sorted(glob.glob("results/runs/*/*/model.pt")):
            task, rn = p.split(os.sep)[-3:-1]
            name, s = rn.rsplit("_s", 1)
            jobs.append((task, name, int(s)))
    else:
        jobs = [(a.task, a.model, a.seed)]
    for task, name, seed in jobs:
        out = eval_run(task, name, seed)
        summary = {l: max(p["acc"] for p in r["points"]) for l, r in out["levels"].items()}
        print(task, name, seed, "best acc per level:", {k: round(v, 3) for k, v in summary.items()})
