"""Experiment 2: non-oracle stopping rules for the looped Transformer (model C).

Experiment 1 showed C extrapolates to longer chains only when the iteration count T is
picked per test level (oracle). Here each example chooses its own T with a rule whose
parameters are calibrated on an in-distribution VALIDATION set only (hops 1-6, a seed
disjoint from the test sets) and then frozen for every test level.

Rules (per example, cap T_MAX):
  fixedpoint  stop at the first t >= 2 with ||h_t - h_{t-1}|| / ||h_t|| < eps
  stable      stop once the answer prediction has been unchanged for m iterations
  probe       stop when a linear probe on the query-position state says
              P(current answer correct) > tau. The probe is trained post hoc on the frozen
              trunk with a separate in-distribution set; its cost is reported.
Calibration (two pre-declared variants, both using validation data only):
  default  the cheapest setting (mean FLOPs on validation) whose validation accuracy is
           within 1 pp of the best fixed T
  _safe    the setting with the highest validation accuracy (ties within 0.1 pp -> cheaper)

FLOPs per answer for a rule = sum over the iterations actually run of one block
application, plus the output head wherever the rule needs a prediction (every
iteration for `stable`, once at the end otherwise) and the probe (2d per iteration).

Usage: python halting.py --seed 0        # writes results/halting/C_s0.json
       python halting.py --latency       # early-exit wall-clock check (seed 0)
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from tasks import TASKS, eval_sets
from train import build, run_dir
from models.common import block_flops, head_flops

TASK = "chain"
MODEL = "C_3x"
T_MAX = 64
TEST_LEVELS = [2, 4, 6, 8, 12, 16, 24, 30]  # 30 = most the 64-name pool allows
VAL_SEED, PROBE_SEED = 2024, 4048            # distinct from tasks.EVAL_SEED and train streams
VAL_SIZE, PROBE_SIZE = 300, 600
FIXED_T = [8, 16, 32, 64]
EPS_GRID = [float(e) for e in np.logspace(-4, 0, 41)]
M_GRID = list(range(1, 16))
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def load(seed):
    m = build(TASK, MODEL)
    m.load_state_dict(torch.load(os.path.join(run_dir(TASK, MODEL, seed), "model.pt")))
    return m.eval()


def id_sets(seed, size):
    mod = TASKS[TASK]
    return {l: mod.make_batch(np.random.default_rng([seed, l]), l, size) for l in mod.TRAIN_LEVELS}


@torch.no_grad()
def trajectory(model, b, chunk=250):
    """Run T_MAX iterations; record per-iteration answer prediction, correctness, relative
    state change and the query-position state (probe features)."""
    out = {k: [] for k in ("pred", "correct", "delta", "feat")}
    for i in range(0, len(b["enc_x"]), chunk):
        x = torch.from_numpy(b["enc_x"][i:i + chunk])
        y = torch.from_numpy(b["enc_y"][i:i + chunk])[:, -1]  # chain answer = query position
        e = model.embed(x)
        h = e
        P, C, D, Fe = [], [], [], []
        for t in range(1, T_MAX + 1):
            hn = model.state_norm(model.block(h + e, None))
            D.append(((hn - h).flatten(1).norm(dim=1) / hn.flatten(1).norm(dim=1)))
            h = hn
            q = h[:, -1]
            p = model.head(q).argmax(-1)
            P.append(p); C.append(p == y); Fe.append(q)
        for k, v in zip(("pred", "correct", "delta", "feat"), (P, C, D, Fe)):
            out[k].append(torch.stack(v, 1))  # [B, T, ...]
    return {k: torch.cat(v, 0) for k, v in out.items()}


def costs(model, n):
    blk, hd, pr = block_flops(model.d, n), head_flops(model.d, n), 2 * model.d
    return dict(blk=blk, head=hd, probe=pr)


def rule_cost(rule, T, c):
    """Mean FLOPs per answer for per-example stopping times T (tensor)."""
    T = T.double()
    if rule == "stable":
        return float((T * (c["blk"] + c["head"])).mean())
    if rule == "probe":
        return float((T * (c["blk"] + c["probe"]) + c["head"]).mean())
    return float((T * c["blk"] + c["head"]).mean())


def stop_times(rule, tr, param, probe=None):
    B = tr["pred"].shape[0]
    if rule == "fixedpoint":
        hit = tr["delta"] < param
        hit[:, 0] = False  # delta at t=1 compares with the raw embedding
    elif rule == "stable":
        P = tr["pred"]
        same = torch.zeros_like(P, dtype=torch.long)
        for t in range(1, T_MAX):
            same[:, t] = torch.where(P[:, t] == P[:, t - 1], same[:, t - 1] + 1, 0)
        hit = same >= param
    else:
        hit = torch.sigmoid(probe(tr["feat"]).squeeze(-1)) > param
    hit[:, -1] = True
    return hit.float().argmax(1) + 1  # first hit, 1-based


def acc_at(tr, T):
    return float(tr["correct"].gather(1, (T - 1).unsqueeze(1)).float().mean())


def train_probe(model, seed):
    torch.manual_seed(seed)
    sets = id_sets(PROBE_SEED, PROBE_SIZE)
    X, Y, flops = [], [], 0
    for l, b in sets.items():
        tr = trajectory(model, b)
        X.append(tr["feat"].reshape(-1, model.d)); Y.append(tr["correct"].reshape(-1).float())
        flops += len(b["enc_x"]) * T_MAX * (block_flops(model.d, b["enc_x"].shape[1]) + 2 * model.d)
    X, Y = torch.cat(X), torch.cat(Y)
    probe = torch.nn.Linear(model.d, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):  # full-batch logistic regression
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(X).squeeze(-1), Y)
        loss.backward(); opt.step()
        flops += 3 * 2 * X.shape[0] * model.d
    return probe.requires_grad_(False), dict(train_flops=flops, final_bce=loss.item(),
                                             positive_rate=float(Y.mean()), n=int(X.shape[0]))


def calibrate(model, probe, seed):
    """Choose each rule's parameter on the in-distribution validation set only."""
    trs = {l: trajectory(model, b) for l, b in id_sets(VAL_SEED, VAL_SIZE).items()}
    cs = {l: costs(model, b["enc_x"].shape[1]) for l, b in id_sets(VAL_SEED, 1).items()}

    def pooled(T_fn, rule):
        a = np.mean([acc_at(trs[l], T_fn(l)) for l in trs])
        f = np.mean([rule_cost(rule, T_fn(l), cs[l]) for l in trs])
        return a, f

    fixed = {T: pooled(lambda l: torch.full((trs[l]["pred"].shape[0],), T), "fixed")[0]
             for T in range(1, T_MAX + 1)}
    best_fixed = max(fixed.values())
    chosen, grids = {}, {}
    for rule, grid in (("fixedpoint", EPS_GRID), ("stable", M_GRID), ("probe", TAU_GRID)):
        res = [(p, *pooled(lambda l: stop_times(rule, trs[l], p, probe), rule)) for p in grid]
        grids[rule] = [dict(param=p, acc=a, flops=f) for p, a, f in res]
        ok = [r for r in res if r[1] >= best_fixed - 0.01]
        p, a, f = min(ok, key=lambda r: r[2]) if ok else max(res, key=lambda r: r[1])
        chosen[rule] = dict(param=p, val_acc=a, val_flops=f, within_1pp=bool(ok))
        # "safe" calibration: highest validation accuracy (within 0.1 pp), then cheapest
        top = max(r[1] for r in res)
        p, a, f = min((r for r in res if r[1] >= top - 0.001), key=lambda r: r[2])
        chosen[rule + "_safe"] = dict(param=p, val_acc=a, val_flops=f)
    best_T = min(T for T, a in fixed.items() if a >= best_fixed - 1e-9)
    return dict(best_fixed_val_acc=best_fixed, best_fixed_val_T=best_T,
                fixed_val_acc={T: fixed[T] for T in FIXED_T}, chosen=chosen, grids=grids)


def run(seed):
    torch.set_num_threads(1)
    model = load(seed)
    probe, pinfo = train_probe(model, seed)
    cal = calibrate(model, probe, seed)          # parameters frozen from here on
    test = eval_sets(TASK, TEST_LEVELS)          # first time the test sets are touched
    levels = {}
    for l, b in test.items():
        tr = trajectory(model, b)
        c = costs(model, b["enc_x"].shape[1])
        B = tr["pred"].shape[0]
        per_t = tr["correct"].float().mean(0)
        r = dict(level=l, ood=l >= TASKS[TASK].OOD_FROM,
                 oracle=dict(acc=float(per_t.max()), T=int(per_t.argmax()) + 1,
                             flops=(int(per_t.argmax()) + 1) * c["blk"] + c["head"]),
                 fixed={T: dict(acc=float(per_t[T - 1]), flops=T * c["blk"] + c["head"]) for T in FIXED_T},
                 acc_per_T=[float(a) for a in per_t])
        for name in ("fixedpoint", "stable", "probe", "fixedpoint_safe", "stable_safe", "probe_safe"):
            rule = name.replace("_safe", "")
            T = stop_times(rule, tr, cal["chosen"][name]["param"], probe)
            r[name] = dict(acc=acc_at(tr, T), flops=rule_cost(rule, T, c),
                           mean_T=float(T.float().mean()),
                           T_quartiles=[float(q) for q in torch.quantile(T.float(), torch.tensor([.25, .5, .75]))],
                           capped=float((T == T_MAX).float().mean()))
        # post-hoc sensitivity over the whole parameter grid (uses test data: diagnostic only,
        # never used to pick a reported rule setting)
        r["sensitivity"] = {
            rule: [dict(param=p, acc=acc_at(tr, T), mean_T=float(T.float().mean()))
                   for p in grid for T in [stop_times(rule, tr, p, probe)]]
            for rule, grid in (("fixedpoint", EPS_GRID), ("stable", M_GRID), ("probe", TAU_GRID))}
        levels[l] = r
    pinfo["weight"] = probe.weight.flatten().tolist()
    pinfo["bias"] = float(probe.bias)
    out = dict(seed=seed, model=MODEL, t_max=T_MAX, probe=pinfo, calibration=cal, levels=levels,
               train=json.load(open(os.path.join(run_dir(TASK, MODEL, seed), "train.json")))["examples"])
    os.makedirs("results/halting", exist_ok=True)
    with open(f"results/halting/C_s{seed}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


@torch.no_grad()
def latency(seed=0, levels=(6, 12, 24), reps=40):
    """Wall-clock check that early exit with the calibrated probe (safe) really saves time
    versus fixed T (batch 1, one thread). Accuracy is not re-measured here."""
    torch.set_num_threads(1)
    model = load(seed)
    saved = json.load(open(f"results/halting/C_s{seed}.json"))
    tau = saved["calibration"]["chosen"]["probe_safe"]["param"]
    w = torch.tensor(saved["probe"]["weight"]); bias = saved["probe"]["bias"]
    res = []
    for l, b in eval_sets(TASK, list(levels), reps).items():
        def run_one(i, mode):
            x = torch.from_numpy(b["enc_x"][i:i + 1])
            e = model.embed(x); h = e
            for t in range(1, T_MAX + 1):
                hn = model.state_norm(model.block(h + e, None))
                h = hn
                if mode == "probe" and float(torch.sigmoid(h[0, -1] @ w + bias)) > tau:
                    break
                if mode == "fixed16" and t == 16:
                    break
            return model.head(h[:, -1]).argmax(-1), t
        for mode in ("probe", "fixed16", "fixed64"):
            for i in range(3):
                run_one(i, mode)
            ts, Ts = [], []
            for i in range(reps):
                t0 = time.perf_counter(); _, T = run_one(i, mode); ts.append(time.perf_counter() - t0); Ts.append(T)
            res.append(dict(level=l, mode=mode, median_ms=1e3 * float(np.median(ts)), mean_T=float(np.mean(Ts))))
            print(res[-1])
    with open("results/halting/latency.json", "w") as f:
        json.dump(dict(threads=1, seed=seed, rule="probe_safe", tau=tau, results=res), f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency", action="store_true")
    a = ap.parse_args()
    if a.latency:
        latency(a.seed)
    else:
        o = run(a.seed)
        print("calibrated:", {k: v["param"] for k, v in o["calibration"]["chosen"].items()})
        for l, r in o["levels"].items():
            print(l, "oracle", round(r["oracle"]["acc"], 3), "@", r["oracle"]["T"],
                  "| T8", round(r["fixed"][8]["acc"], 3),
                  "|", " ".join(f"{k}={r[k]['acc']:.3f}(T~{r[k]['mean_T']:.1f})" for k in ("fixedpoint", "stable", "probe", "fixedpoint_safe", "stable_safe", "probe_safe")))
