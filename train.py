"""Training pipeline with a common training-FLOP budget for every model.

Each run trains until it has spent `--budget` training FLOPs (analytic count, see
models/common.py), not for a fixed number of steps, so recurrent / scratchpad models
that cost more per example see proportionally fewer examples.

Usage:
  python train.py --task addition --model A  --seed 0
  python train.py --task chain    --model D  --seed 1 --budget 2e13
Model specs (see MODELS below): A, Adeep, B, C, C_fixed, D, D_nodeep, D_inner1, ...
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tasks import TASKS, TrainStream, USE_ROPE, IGNORE, dec_targets
from models import (StandardTransformer, ScratchpadTransformer, LoopedTransformer,
                    LatentReasoner, n_params)

D_MODEL = 64
BATCH = 64
BUDGET = 8e12   # training FLOPs per run (same for every model)

# name -> config. T/S are training-time iteration counts.
MODELS = {
    "A":        dict(arch="A", layers=4),
    "Adeep":    dict(arch="A", layers=8),
    "B":        dict(arch="B", layers=4),
    "C":        dict(arch="C", T_lo=4, T_hi=12),           # random depth, mean 8
    "C_fixed":  dict(arch="C", T_lo=8, T_hi=8),            # ablation: fixed depth
    "D":        dict(arch="D", n_inner=2, S=6, deep=True),  # 6 x 3 = 18 block apps
    "D_nodeep": dict(arch="D", n_inner=2, S=6, deep=False),  # ablation: loss only at end, full BPTT
    "D_inner1": dict(arch="D", n_inner=1, S=6, deep=True),   # ablation: fewer latent updates
    "D_S12":    dict(arch="D", n_inner=2, S=12, deep=True),  # ablation: longer supervision horizon
    # post-hoc diagnostic (chain only): identical configs trained with 3x the budget
    "A_3x":     dict(arch="A", layers=4),
    "C_3x":     dict(arch="C", T_lo=4, T_hi=12),
    "D_3x":     dict(arch="D", n_inner=2, S=6, deep=True),
}


def build(task, name, d=D_MODEL):
    cfg = MODELS[name]
    rope = USE_ROPE[task]
    if cfg["arch"] == "A":
        return StandardTransformer(d, cfg["layers"], rope)
    if cfg["arch"] == "B":
        return ScratchpadTransformer(d, cfg["layers"], rope)
    if cfg["arch"] == "C":
        return LoopedTransformer(d, rope)
    return LatentReasoner(d, rope, cfg["n_inner"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=IGNORE)


def lr_at(frac, base, warm=0.03):
    if frac < warm:
        return base * frac / warm
    p = (frac - warm) / (1 - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def train(task, name, seed, budget, lr, out_dir, log_every=50):
    set_seed(seed)
    cfg = MODELS[name]
    model = build(task, name)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    stream = TrainStream(task, seed, BATCH)
    used, steps, opt_steps, t0 = 0.0, 0, 0, time.time()
    log = []

    def opt_step(loss):
        nonlocal opt_steps
        for g in opt.param_groups:
            g["lr"] = lr_at(used / budget, lr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt_steps += 1

    while used < budget:
        b = stream.next()
        model.train()
        if cfg["arch"] == "B":
            x = torch.from_numpy(b["dec_x"])
            y = torch.from_numpy(dec_targets(b["dec_x"], b["dec_forced"]))
            loss = masked_ce(model(x[:, :-1]), y)
            used += 3 * BATCH * model.flops_train(x.shape[1])
            opt_step(loss)
        else:
            x = torch.from_numpy(b["enc_x"])
            y = torch.from_numpy(b["enc_y"])
            n = x.shape[1]
            if cfg["arch"] == "A":
                loss = masked_ce(model(x), y)
                used += 3 * BATCH * model.flops(n)
                opt_step(loss)
            elif cfg["arch"] == "C":
                T = random.randint(cfg["T_lo"], cfg["T_hi"])
                loss = masked_ce(model(x, T), y)
                used += 3 * BATCH * model.flops(n, T)
                opt_step(loss)
            else:
                ans = torch.from_numpy(b["enc_ans"])
                e_fn = lambda: model.embed(x)
                rope = model.rope(n, x.device)
                yS, zS = model.init_state(x)
                if cfg["deep"]:
                    # deep supervision: one optimiser step per outer step, states detached
                    for s in range(cfg["S"]):
                        yS, zS, logits, q = model.step(e_fn(), yS.detach(), zS.detach(), rope)
                        correct = ((logits.argmax(-1) == y) | ~ans).all(-1).float()
                        loss = masked_ce(logits, y) + F.binary_cross_entropy_with_logits(q, correct)
                        used += 3 * BATCH * model.flops_per_step(n)
                        opt_step(loss)
                else:
                    e = e_fn()
                    for s in range(cfg["S"]):
                        yS, zS, logits, q = model.step(e, yS, zS, rope)
                    correct = ((logits.argmax(-1) == y) | ~ans).all(-1).float()
                    loss = masked_ce(logits, y) + F.binary_cross_entropy_with_logits(q, correct)
                    used += 3 * BATCH * model.flops(n, cfg["S"])
                    opt_step(loss)
        steps += 1
        if steps % log_every == 0:
            log.append(dict(step=steps, flops=used, loss=loss.item(), time=time.time() - t0))

    info = dict(task=task, model=name, seed=seed, cfg=cfg, d=D_MODEL, batch=BATCH,
                params=n_params(model), train_flops=used, budget=budget, batches=steps,
                examples=steps * BATCH, opt_steps=opt_steps, train_time_s=time.time() - t0,
                dropped_overlapping_examples=stream.dropped, threads=torch.get_num_threads(),
                final_loss=loss.item(), log=log)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(info, f, indent=1)
    return model, info


def run_dir(task, name, seed, root="results/runs"):
    return os.path.join(root, task, f"{name}_s{seed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=None, help="training FLOPs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    budget = a.budget or BUDGET
    _, info = train(a.task, a.model, a.seed, budget, a.lr, run_dir(a.task, a.model, a.seed))
    print(json.dumps({k: v for k, v in info.items() if k != "log"}))
