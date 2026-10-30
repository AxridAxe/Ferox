"""Model D - latent recursive reasoner with deep supervision and learned halting.

This is an ADAPTATION of existing ideas, not a novel architecture:
  * two-state recursion (answer state y, latent reasoning state z) refined by ONE tiny
    shared network: Tiny Recursive Models (Jolicoeur-Martineau 2025), itself a
    simplification of the Hierarchical Reasoning Model (Wang et al. 2025);
  * deep supervision: the answer is decoded and supervised after every outer step,
    states are detached between steps (truncated / one-step gradient, HRM/TRM), so
    the model must make incremental progress that can be continued indefinitely
    (cf. progressive training in Bansal et al. 2022);
  * input injection into the latent update ("recall", Bansal et al. 2022);
  * a halting head predicting "my current answer is correct" (TRM's Q-head, a
    simplification of ACT, Graves 2016 / PonderNet, Banino et al. 2021), used at
    inference for per-example adaptive computation.
The state is fixed-size per input position: reasoning happens by iterating this
compact state, never by emitting tokens.

One outer step ("cycle") = n_inner latent updates z <- f(z + y + x), then one answer
update y <- f(y + z): (n_inner + 1) block applications.
"""
import torch
import torch.nn as nn
from .common import Block, EncoderBase, block_flops, head_flops


class LatentReasoner(EncoderBase):
    kind = "latent"

    def __init__(self, d, use_rope, n_inner=2):
        super().__init__(d, use_rope)
        self.n_inner = n_inner
        self.block = Block(d)
        self.state_norm = nn.LayerNorm(d)
        self.y0 = nn.Parameter(torch.zeros(d))
        self.z0 = nn.Parameter(torch.zeros(d))
        self.halt = nn.Linear(d, 1)

    def init_state(self, x):
        B, N, _ = x.shape
        return self.y0.expand(B, N, -1), self.z0.expand(B, N, -1)

    def cycle(self, e, y, z, rope):
        for _ in range(self.n_inner):
            z = self.state_norm(self.block(z + y + e, rope))
        y = self.state_norm(self.block(y + z, rope))
        return y, z

    def step(self, e, y, z, rope):
        """One supervised outer step: returns new states, logits, halting logit."""
        y, z = self.cycle(e, y, z, rope)
        return y, z, self.head(y), self.halt(y.mean(1)).squeeze(-1)

    def forward(self, x, S, checkpoints=None):
        """Run S outer steps without halting. Returns {s: (logits, halt_logit)}."""
        e = self.embed(x)
        rope = self.rope(x.shape[1], x.device)
        y, z = self.init_state(x)
        outs = {}
        for s in range(1, S + 1):
            y, z, logits, q = self.step(e, y, z, rope)
            if checkpoints is None or s in checkpoints:
                outs[s] = (logits, q)
        return outs

    def flops_per_step(self, n):
        return (self.n_inner + 1) * block_flops(self.d, n) + head_flops(self.d, n) + 2 * self.d

    def flops(self, n, S):
        return S * self.flops_per_step(n)
