"""Model A - standard (non-recurrent) Transformer encoder that predicts answers directly.

L distinct blocks, one pass, no intermediate reasoning. A-deep (L=8) is the
non-recurrent control at the same FLOPs as the looped model at T=8.
"""
import torch.nn as nn
from .common import Block, EncoderBase, block_flops, head_flops


class StandardTransformer(EncoderBase):
    kind = "encoder"

    def __init__(self, d, layers, use_rope):
        super().__init__(d, use_rope)
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))

    def forward(self, x):
        h = self.embed(x)
        rope = self.rope(x.shape[1], x.device)
        for b in self.blocks:
            h = b(h, rope)
        return self.head(h)

    def flops(self, n):
        return len(self.blocks) * block_flops(self.d, n) + head_flops(self.d, n)
