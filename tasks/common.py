"""Shared encoding conventions for all tasks.

Every sequence element is a triple of integer fields (f0, f1, f2); a model embeds
each field with its own table (index 0 = "empty", zero vector) and sums them.
All models (A-D) use exactly the same input encoding and the same output
vocabulary, so differences come from architecture/training, not tokenisation.

Each generated batch is a dict of numpy arrays:
  enc_x  [B, N, 3]  input for direct-prediction models (A, C, D)
  enc_y  [B, N]     dense per-position targets (IGNORE where unlabeled)
  enc_ans[B, N]     bool mask of positions that form the final answer
  dec_x  [B, M, 3]  full scratchpad sequence for model B (prompt + reasoning + answer)
  dec_forced [M]    True where the token is given (input), False where B must generate it
  dec_ans[M]        True at positions whose (generated) token is part of the answer
"""
import hashlib
import numpy as np

IGNORE = -100
VOCAB = 128          # size of every embedding table and of the output head


def problem_hashes(enc_x):
    """Stable hash per problem (used to keep train and eval sets disjoint)."""
    return {hashlib.sha1(row.tobytes()).hexdigest() for row in np.ascontiguousarray(enc_x)}


def dec_targets(dec_x, dec_forced):
    """Next-token training targets for B: position t predicts token t+1 if that token is generated."""
    tgt = dec_x[:, 1:, 2].copy()
    tgt[:, dec_forced[1:]] = IGNORE
    return tgt
