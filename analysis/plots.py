"""Aggregate results/runs/*/*/{train,eval}.json and results/benchmark.json into
results/summary.md, results/summary.json and the plots in results/plots/.

Nothing here is hand-edited: every number in summary.md comes from the JSON files.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks import TASKS  # noqa: E402

OUT = "results/plots"
# Categorical slots in fixed order (validated reference palette), plus a marker per model
# so identity never relies on colour alone.
STYLE = {
    "A":     dict(color="#2a78d6", marker="o", label="A standard (4L)"),
    "Adeep": dict(color="#eb6834", marker="s", label="A-deep (8L)"),
    "B":     dict(color="#1baf7a", marker="^", label="B scratchpad"),
    "C":     dict(color="#eda100", marker="D", label="C looped"),
    "D":     dict(color="#e87ba4", marker="v", label="D latent (fixed S)"),
    "Dad":   dict(color="#6250d6", marker="*", label="D latent (adaptive)"),
}
ABL = ["C", "C_fixed", "D", "D_nodeep", "D_inner1", "D_S12"]
C_DEFAULT, D_DEFAULT = 8, 6
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "lines.linewidth": 2})


def load():
    runs = defaultdict(dict)  # (task, model) -> seed -> {train, eval}
    for p in glob.glob("results/runs/*/*/eval.json"):
        d = os.path.dirname(p)
        task, rn = d.split(os.sep)[-2:]
        name, s = rn.rsplit("_s", 1)
        runs[(task, name)][int(s)] = dict(train=json.load(open(os.path.join(d, "train.json"))),
                                          eval=json.load(open(p)))
    return runs


def point(ev, lvl, iters=None, adaptive=False):
    r = ev["levels"][str(lvl)]
    if adaptive:
        return r["adaptive"]["acc"], r["adaptive"]["flops"]
    pts = r["points"]
    p = pts[0] if iters is None else next(q for q in pts if q["iters"] == iters)
    return p["acc"], p["flops"]


def series(runs, task, name, lvl, iters=None, adaptive=False):
    vals = [point(r["eval"], lvl, iters, adaptive) for r in runs[(task, name)].values()]
    a = np.array([v[0] for v in vals])
    f = np.array([v[1] for v in vals])
    return a.mean(), a.std(), f.mean(), len(a)


def fmt(m, s, n):
    return f"{100*m:.1f} ± {100*s:.1f}" if n > 1 else f"{100*m:.1f}"


def configs(task):
    return [("A", None, False, "A"), ("Adeep", None, False, "Adeep"), ("B", None, False, "B"),
            ("C", C_DEFAULT, False, "C"), ("D", D_DEFAULT, False, "D"), ("D", None, True, "Dad")]


def plot_difficulty(runs, task):
    levels = TASKS[task].EVAL_LEVELS
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, it, ad, key in configs(task):
        if (task, name) not in runs:
            continue
        m = [series(runs, task, name, l, it, ad) for l in levels]
        mu = np.array([x[0] for x in m]); sd = np.array([x[1] for x in m])
        st = STYLE[key]
        lab = st["label"] + (f" T={it}" if key == "C" else f" S={it}" if key == "D" else "")
        ax.plot(levels, mu, color=st["color"], marker=st["marker"], markersize=6, label=lab)
        ax.fill_between(levels, mu - sd, mu + sd, color=st["color"], alpha=0.12, linewidth=0)
    ax.axvline(TASKS[task].OOD_FROM - 0.5, color="#8a8984", linestyle="--", linewidth=1)
    ax.text(TASKS[task].OOD_FROM - 0.3, 0.97, "unseen difficulty →", fontsize=8, color="#52514e", va="top")
    if task == "chain":
        ax.axhline(0.1, color="#8a8984", linestyle=":", linewidth=1)
        ax.text(levels[-1], 0.11, "chance", fontsize=8, color="#52514e", ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(levels); ax.set_xticklabels(levels)
    ax.set_xlabel("digits" if task == "addition" else "hops"); ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{task}: accuracy vs difficulty (mean ± std over seeds)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/{task}_acc_vs_difficulty.png", dpi=150); plt.close(fig)


def plot_acc_flops(runs, task):
    levels = TASKS[task].EVAL_LEVELS
    fig, axes = plt.subplots(2, (len(levels) + 1) // 2, figsize=(13, 6), sharey=True)
    for ax, lvl in zip(axes.flat, levels):
        for name, key in [("C", "C"), ("D", "D")]:
            if (task, name) not in runs:
                continue
            ev0 = next(iter(runs[(task, name)].values()))["eval"]
            its = [p["iters"] for p in ev0["levels"][str(lvl)]["points"]]
            m = [series(runs, task, name, lvl, i) for i in its]
            ax.plot([x[2] for x in m], [x[0] for x in m], color=STYLE[key]["color"],
                    marker=STYLE[key]["marker"], markersize=4, label=STYLE[key]["label"].replace(" (fixed S)", " (S sweep)") + (" (T sweep)" if key == "C" else ""))
        for name, key, ad in [("A", "A", False), ("Adeep", "Adeep", False), ("B", "B", False), ("D", "Dad", True)]:
            if (task, name) not in runs:
                continue
            mu, sd, f, n = series(runs, task, name, lvl, None, ad)
            ax.errorbar([f], [mu], yerr=[sd], color=STYLE[key]["color"], marker=STYLE[key]["marker"],
                        markersize=9 if key == "Dad" else 7, linestyle="none", label=STYLE[key]["label"],
                        markeredgecolor="white", markeredgewidth=1)
        ax.set_xscale("log")
        ood = lvl >= TASKS[task].OOD_FROM
        ax.set_title(f"{lvl} {'digits' if task == 'addition' else 'hops'}{' (OOD)' if ood else ''}")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes.flat[len(levels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("inference FLOPs per answer")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(0.99, 0.05))
    fig.suptitle(f"{task}: accuracy vs inference compute (mean over seeds; error bar = std)")
    fig.tight_layout(); fig.savefig(f"{OUT}/{task}_acc_vs_flops.png", dpi=150); plt.close(fig)


def plot_ablations(runs, task):
    levels = TASKS[task].EVAL_LEVELS
    names = [n for n in ABL if (task, n) in runs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, family in zip(axes, ["C", "D"]):
        fam = [n for n in names if n.startswith(family)]
        for j, lvl in enumerate([l for l in levels if l in (TASKS[task].OOD_FROM - 1, levels[-3])]):
            pass
        for k, n in enumerate(fam):
            ev = runs[(task, n)][0]["eval"]  # ablations are single-seed: compare to seed 0
            lvl_hi = TASKS[task].OOD_FROM - 1 if TASKS[task].OOD_FROM - 1 in levels else levels[2]
            for lvl, ls in [(lvl_hi, "-"), (next(l for l in levels if l >= TASKS[task].OOD_FROM), "--")]:
                pts = ev["levels"][str(lvl)]["points"]
                ax.plot([p["iters"] for p in pts], [p["acc"] for p in pts], linestyle=ls,
                        color=["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"][k], marker="os^v"[k], markersize=4,
                        label=f"{n} @ {lvl}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("test-time iterations (T block apps for C; S outer steps for D)")
        ax.set_ylabel("accuracy"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{task}: {family} ablations, seed 0 (solid = hardest ID level, dashed = first OOD level)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/{task}_ablations.png", dpi=150); plt.close(fig)


def summary(runs):
    lines, js = ["# Generated summary (do not edit; produced by analysis/plots.py)\n"], {}
    for task in TASKS:
        mod = TASKS[task]
        levels = mod.EVAL_LEVELS
        id_l = [l for l in levels if l < mod.OOD_FROM]
        ood_l = [l for l in levels if l >= mod.OOD_FROM]
        lines.append(f"\n## {task}\n")
        lines.append(f"In-distribution (held-out) levels: {id_l}; OOD levels: {ood_l}. "
                     f"Accuracy in %, mean ± std over seeds (n shown). FLOPs = inference FLOPs per answer.\n")
        hdr = "| model | n | params | train FLOPs | train time (s) | examples seen | " + \
              " | ".join(f"acc@{l}" for l in levels) + f" | FLOPs@{id_l[-1]} | FLOPs@{ood_l[1]} |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        rows = configs(task) + [(n, D_DEFAULT if n.startswith("D") else C_DEFAULT if n.startswith("C") else None,
                                 False, n) for n in ABL if n not in ("C", "D")]
        rows += [(n, None, True, n + "-adaptive") for n in ABL if n.startswith("D") and n != "D"]
        for name, it, ad, key in rows:
            if (task, name) not in runs:
                continue
            rr = runs[(task, name)]
            tr = [r["train"] for r in rr.values()]
            if name.startswith("D") and not ad:
                it = rr[0]["train"]["cfg"]["S"]
            accs = [series(runs, task, name, l, it, ad) for l in levels]
            label = {"Dad": "D adaptive"}.get(key, key) + (f" (T={it})" if name.startswith("C") else f" (S={it})" if name.startswith("D") and not ad else "")
            fl_id = series(runs, task, name, id_l[-1], it, ad)[2]
            fl_ood = series(runs, task, name, ood_l[1], it, ad)[2]
            lines.append(f"| {label} | {len(tr)} | {tr[0]['params']:,} | {np.mean([t['train_flops'] for t in tr]):.2e} | "
                         f"{np.mean([t['train_time_s'] for t in tr]):.0f} | {np.mean([t['examples'] for t in tr]):,.0f} | " +
                         " | ".join(fmt(a[0], a[1], a[3]) for a in accs) + f" | {fl_id:.2e} | {fl_ood:.2e} |")
            js[f"{task}/{label}"] = dict(acc={l: a[0] for l, a in zip(levels, accs)},
                                         std={l: a[1] for l, a in zip(levels, accs)},
                                         flops={l: a[2] for l, a in zip(levels, accs)})
        # iso-compute comparisons: C/D at the iteration budget closest to A / Adeep / B FLOPs
        lines.append(f"\n**Matched-compute comparison on {task}** (C/D evaluated at the fixed iteration budget whose "
                     "FLOPs are closest to, without exceeding, the reference model's FLOPs; mean over seeds):\n")
        lines.append("| level | reference | ref FLOPs | ref acc | C iters | C FLOPs | C acc | D steps | D FLOPs | D acc |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for lvl in [id_l[-1]] + ood_l[:2]:
            for ref in ["A", "Adeep", "B"]:
                if (task, ref) not in runs:
                    continue
                ra, _, rf, _ = series(runs, task, ref, lvl)
                cells = []
                for fam in ["C", "D"]:
                    ev0 = runs[(task, fam)][0]["eval"]["levels"][str(lvl)]["points"]
                    ok = [p["iters"] for p in ev0 if p["flops"] <= rf * 1.001]
                    if not ok:
                        cells += ["-", "-", "-"]; continue
                    it = max(ok)
                    a, _, f, _ = series(runs, task, fam, lvl, it)
                    cells += [str(it), f"{f:.2e}", f"{100*a:.1f}"]
                lines.append(f"| {lvl} | {ref} | {rf:.2e} | {100*ra:.1f} | " + " | ".join(cells) + " |")
    if os.path.exists("results/benchmark.json"):
        bj = json.load(open("results/benchmark.json"))
        lines.append("\n## Latency / memory (1 CPU thread, seed-0 checkpoints; see benchmark.py)\n")
        lines.append(f"Energy: {bj['meta']['energy']}.\n")
        lines.append("| task | level | model | iters | params | FLOPs/answer (analytic) | FLOPs (torch counter) | latency b=1 (ms) | latency b=256 (ms/answer) | peak mem Δ (MB, b=256) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in bj["results"]:
            lines.append(f"| {r['task']} | {r['level']} | {r['model']} | {r['iters']} | {r['params']:,} | {r['flops_analytic']:.3g} | "
                         f"{r['flops_counted']:.3g} | {r['latency_b1_ms']:.2f} | {r['latency_b256_ms']:.3f} | {r['peak_mem_mb']:.1f} |")
    open("results/summary.md", "w").write("\n".join(lines) + "\n")
    json.dump(js, open("results/summary.json", "w"), indent=1)


def plot_latency():
    if not os.path.exists("results/benchmark.json"):
        return
    bj = json.load(open("results/benchmark.json"))["results"]
    tasks = sorted({r["task"] for r in bj})
    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 3.8))
    for ax, task in zip(np.atleast_1d(axes), tasks):
        rs = [r for r in bj if r["task"] == task]
        lvls = sorted({r["level"] for r in rs})
        labels = [f"{r['model']}{'' if r['iters'] is None else ' ' + str(r['iters'])}" for r in rs if r["level"] == lvls[0]]
        x = np.arange(len(labels)); w = 0.38
        for j, (lvl, c) in enumerate(zip(lvls, ["#2a78d6", "#eb6834"])):
            v = [r["latency_b1_ms"] for r in rs if r["level"] == lvl]
            ax.bar(x + (j - 0.5) * w, v, w - 0.02, color=c, label=f"level {lvl}")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("ms per answer (batch 1, 1 thread)"); ax.set_title(f"{task}: measured CPU latency")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/latency.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    for task in TASKS:
        if not any(k[0] == task for k in runs):
            continue
        plot_difficulty(runs, task)
        plot_acc_flops(runs, task)
        plot_ablations(runs, task)
    plot_latency()
    summary(runs)
    print(open("results/summary.md").read())
