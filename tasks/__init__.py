"""Task registry: deterministic generators, fixed eval sets, train/eval disjointness."""
import numpy as np
from . import addition, chain
from .common import IGNORE, VOCAB, problem_hashes, dec_targets

TASKS = {"addition": addition, "chain": chain}
USE_ROPE = {"addition": True, "chain": False}
EVAL_SEED = 12345
EVAL_SIZE = 500


def eval_sets(task, levels=None, size=EVAL_SIZE):
    """Fixed evaluation sets: one per difficulty level, generated from a fixed seed."""
    mod = TASKS[task]
    out = {}
    for lvl in levels or mod.EVAL_LEVELS:
        rng = np.random.default_rng([EVAL_SEED, lvl])
        out[lvl] = mod.make_batch(rng, lvl, size)
    return out


class TrainStream:
    """Infinite seeded stream of training batches; one difficulty level per batch
    (no padding needed). Any problem that also appears in an eval set is dropped
    and resampled, so in-distribution eval problems are guaranteed unseen."""

    def __init__(self, task, seed, batch):
        self.mod = TASKS[task]
        self.rng = np.random.default_rng([seed, 777])
        self.batch = batch
        ev = eval_sets(task, [l for l in self.mod.EVAL_LEVELS if l < self.mod.OOD_FROM])
        self.banned = set().union(*(problem_hashes(b["enc_x"]) for b in ev.values()))
        self.dropped = 0

    def next(self):
        lvl = int(self.rng.choice(self.mod.TRAIN_LEVELS))
        b = self.mod.make_batch(self.rng, lvl, self.batch)
        hs = [problem_hashes(r[None]).pop() for r in b["enc_x"]]
        keep = np.array([h not in self.banned for h in hs])
        if not keep.all():
            self.dropped += int((~keep).sum())
            idx = np.nonzero(keep)[0]
            idx = np.concatenate([idx, idx[: self.batch - len(idx)]])  # refill by duplication
            for key in ("enc_x", "enc_y", "enc_ans", "dec_x"):
                b[key] = b[key][idx]
        b["level"] = lvl
        return b
