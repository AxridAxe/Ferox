"""Model C - looped Transformer: one shared block applied T times.

Follows the looped / recurrent-depth recipe of Giannou et al. 2023, Yang et al. 2023,
Saunshi et al. 2025 and Geiping et al. 2025: weight-tied block, input injection (the
embedded problem is re-added every iteration, cf. "recall" in Bansal et al. 2022),
a normalised state, and a randomised number of iterations during training so that
the model can be run for a different number of iterations at test time.
Loss is applied only after the final iteration, with full back-propagation through
all iterations.
"""
import torch.nn as nn
from .common import Block, EncoderBase, block_flops, head_flops


class LoopedTransformer(EncoderBase):
    kind = "encoder"

    def __init__(self, d, use_rope):
        super().__init__(d, use_rope)
        self.block = Block(d)
        self.state_norm = nn.LayerNorm(d)

    def forward(self, x, T, checkpoints=None):
        """Returns logits after T iterations, or {t: logits} for every t in checkpoints."""
        e = self.embed(x)
        rope = self.rope(x.shape[1], x.device)
        h = e
        outs = {}
        for t in range(1, T + 1):
            h = self.state_norm(self.block(h + e, rope))
            if checkpoints is not None and t in checkpoints:
                outs[t] = self.head(h)
        return outs if checkpoints is not None else self.head(h)

    def flops(self, n, T):
        return T * block_flops(self.d, n) + head_flops(self.d, n)
