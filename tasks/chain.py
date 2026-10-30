"""Variable-chain resolution: multi-hop pointer following with modular arithmetic.

A problem with k hops contains two chains of k+1 statements each (one queried, one
distractor of equal length), written in random order:
    v_0 = r            (root: a literal digit)
    v_j = v_{j-1} + c_j (mod 10)
plus a query "Q = v_k + 0". The answer is the value of v_k. Variable names are drawn
from a pool of 64 symbols, freshly permuted per problem, so values cannot be memorised
per name. Statements are presented in shuffled order, so the model must actually
follow the dependency chain: resolving v_k needs k sequential lookups (or a log-depth
pointer-doubling scheme). Chance accuracy is 10%.

Each statement is ONE sequence element with fields (dst name, src name or root digit,
addend). The models see the statements as a set (no positional encoding on this task).
Dense targets for A/C/D: every statement position is labelled with the value of the
variable it defines; the query position is labelled with the answer (the metric).

Scratchpad for model B (generated after the prompt):
    backtrack: names v_{k-1} ... v_0, then the root digit r,
    forward:   values of v_1 ... v_k (the last one is the answer).
Output vocabulary: name j -> 20+j, value/digit d -> 90+d.
"""
import numpy as np

NAMES = 64
ROOT_BASE = 65   # f1 value for a root literal digit d: 65+d
QUERY = 75       # f0 value of the query element


def make_batch(rng, k, batch):
    S = 2 * (k + 1)
    N = S + 1
    enc_x = np.zeros((batch, N, 3), dtype=np.int64)
    enc_y = np.zeros((batch, N), dtype=np.int64)
    enc_ans = np.zeros((batch, N), dtype=bool)
    enc_ans[:, -1] = True
    G = 2 * k + 1
    dec_x = np.zeros((batch, N + G, 3), dtype=np.int64)
    dec_forced = np.zeros(N + G, dtype=bool)
    dec_forced[:N] = True
    dec_ans = np.zeros(N + G, dtype=bool)
    dec_ans[-1] = True

    for r in range(batch):
        names = rng.permutation(NAMES)[:S]
        rows = []
        main_vals = None
        main_names = names[: k + 1]
        for ci in range(2):
            ch = names[ci * (k + 1):(ci + 1) * (k + 1)]
            root = rng.integers(0, 10)
            adds = rng.integers(0, 10, size=k)
            vals = [root]
            rows.append((ch[0] + 1, ROOT_BASE + root, 0, root))
            for j in range(1, k + 1):
                vals.append((vals[-1] + adds[j - 1]) % 10)
                rows.append((ch[j] + 1, ch[j - 1] + 1, adds[j - 1] + 1, vals[-1]))
            if ci == 0:
                main_vals = vals
        order = rng.permutation(S)
        for p, o in enumerate(order):
            d, s, c, v = rows[o]
            enc_x[r, p] = (d, s, c)
            enc_y[r, p] = 90 + v
        enc_x[r, S] = (QUERY, main_names[k] + 1, 1)
        enc_y[r, S] = 90 + main_vals[k]

        dec_x[r, :N] = enc_x[r]
        gen = [20 + main_names[j] for j in range(k - 1, -1, -1)]
        gen.append(90 + main_vals[0])
        gen += [90 + main_vals[j] for j in range(1, k + 1)]
        dec_x[r, N:, 2] = gen
    return dict(enc_x=enc_x, enc_y=enc_y, enc_ans=enc_ans, dec_x=dec_x,
                dec_forced=dec_forced, dec_ans=dec_ans)


TRAIN_LEVELS = list(range(1, 7))
EVAL_LEVELS = [2, 4, 6, 8, 12, 16, 24]
OOD_FROM = 7
