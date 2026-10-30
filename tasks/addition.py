"""Multi-digit addition with aligned, least-significant-digit-first encoding.

Difficulty = number of digits n. Inputs: n positions holding the digit pair (a_i, b_i)
plus one END position. Targets: sum digit s_i at position i and the final carry at END.
Exact match requires the whole carry chain to be right, which is an inherently
sequential dependency (position i depends on all positions < i).

To stop models from scoring well by ignoring carries, half of all problems are built
with long carry-propagation runs (digit pairs summing to exactly 9, seeded by a carry).

Output vocabulary: digit s -> 1+s, carry c -> 11+c.
Scratchpad (model B) sequence, streamed:  p_0 s_0 c_1 p_1 s_1 c_2 ... p_{n-1} s_{n-1} c_n
where p_i (digit pair) is given and s_i, c_{i+1} are generated. The carries are explicit
intermediate reasoning tokens; the answer is (s_0..s_{n-1}, c_n).
"""
import numpy as np
from .common import IGNORE

END = 11


def sample_digits(rng, n, batch):
    a = rng.integers(0, 10, size=(batch, n))
    b = rng.integers(0, 10, size=(batch, n))
    # Half the batch: force long carry chains (pairs summing to 9 propagate a carry).
    hard = rng.random(batch) < 0.5
    for r in np.nonzero(hard)[0]:
        if n < 2:
            continue
        start = rng.integers(0, n - 1)
        length = rng.integers(1, n - start)
        a[r, start] = rng.integers(5, 10)
        b[r, start] = rng.integers(10 - a[r, start], 10)  # generates a carry
        for i in range(start + 1, min(n, start + 1 + length)):
            a[r, i] = rng.integers(0, 10)
            b[r, i] = 9 - a[r, i]                       # propagates it
    return a, b


def make_batch(rng, n, batch):
    a, b = sample_digits(rng, n, batch)
    s = np.zeros_like(a)
    carries = np.zeros((batch, n + 1), dtype=np.int64)
    for i in range(n):
        t = a[:, i] + b[:, i] + carries[:, i]
        s[:, i] = t % 10
        carries[:, i + 1] = t // 10

    enc_x = np.zeros((batch, n + 1, 3), dtype=np.int64)
    enc_x[:, :n, 0] = a + 1
    enc_x[:, :n, 1] = b + 1
    enc_x[:, n, 0] = END
    enc_x[:, n, 1] = END
    enc_y = np.zeros((batch, n + 1), dtype=np.int64)
    enc_y[:, :n] = 1 + s
    enc_y[:, n] = 11 + carries[:, n]
    enc_ans = np.ones((batch, n + 1), dtype=bool)

    M = 3 * n
    dec_x = np.zeros((batch, M, 3), dtype=np.int64)
    dec_forced = np.zeros(M, dtype=bool)
    dec_ans = np.zeros(M, dtype=bool)
    for i in range(n):
        dec_x[:, 3 * i, 0] = a[:, i] + 1
        dec_x[:, 3 * i, 1] = b[:, i] + 1
        dec_forced[3 * i] = True
        dec_x[:, 3 * i + 1, 2] = 1 + s[:, i]
        dec_ans[3 * i + 1] = True
        dec_x[:, 3 * i + 2, 2] = 11 + carries[:, i + 1]
    dec_ans[M - 1] = True
    return dict(enc_x=enc_x, enc_y=enc_y, enc_ans=enc_ans, dec_x=dec_x,
                dec_forced=dec_forced, dec_ans=dec_ans)


TRAIN_LEVELS = list(range(1, 9))
EVAL_LEVELS = [2, 4, 6, 8, 12, 16, 24, 32]
OOD_FROM = 9  # levels >= this were never seen in training
