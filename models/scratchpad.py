"""Model B - causal Transformer that writes explicit intermediate reasoning tokens.

Trained with teacher forcing on (prompt, scratchpad, answer) sequences, decoded greedily
with a key/value cache. All tokens processed at inference - given input tokens and
every generated reasoning token - are counted in its FLOPs and latency.
"""
import torch
import torch.nn as nn
from .common import Block, Embed, Head, rope_cache, block_flops, head_flops


class ScratchpadTransformer(nn.Module):
    kind = "decoder"

    def __init__(self, d, layers, use_rope):
        super().__init__()
        self.d, self.use_rope = d, use_rope
        self.embed = Embed(d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))
        self.head = Head(d)

    def forward(self, x, cache=None, offset=0):
        h = self.embed(x)
        rope = None
        if self.use_rope:
            cos, sin = rope_cache(offset + x.shape[1], self.d // 4, x.device)
            rope = (cos[offset:], sin[offset:])
        for i, b in enumerate(self.blocks):
            h = b(h, rope, causal=True, kv=None if cache is None else cache[i])
        return self.head(h)

    @torch.no_grad()
    def generate(self, dec_x, forced):
        """Greedy decoding. Forced positions take the given token; others take the model's
        prediction from the previous position. Consecutive forced tokens are prefilled
        together. Returns predicted tokens [B, M] (forced positions copied) and FLOPs."""
        B, M, _ = dec_x.shape
        cache = [dict() for _ in self.blocks]
        seq = dec_x.clone()
        seq[:, ~forced, :] = 0
        flops, t, logits = 0, 0, None
        while t < M:
            if forced[t]:
                e = t
                while e < M and forced[e]:
                    e += 1
                chunk = seq[:, t:e]
            else:
                tok = logits[:, -1].argmax(-1)
                seq[:, t, 2] = tok
                e = t + 1
                chunk = seq[:, t:e]
            if e >= M:  # last token needs no further prediction
                break
            logits = self.forward(chunk, cache, offset=t)
            flops += self.flops_chunk(t, e - t)
            t = e
        return seq[:, :, 2], flops

    def flops_chunk(self, past, n):
        return len(self.blocks) * block_flops(self.d, n, past + n) + head_flops(self.d, n)

    def flops_generate(self, forced):
        M, t, total = len(forced), 0, 0
        while t < M:
            e = t + 1
            if forced[t]:
                while e < M and forced[e]:
                    e += 1
            if e >= M:
                break
            total += self.flops_chunk(t, e - t)
            t = e
        return total

    def flops_train(self, M):
        """Forward FLOPs of one teacher-forced sequence of length M-1 (full causal matrix)."""
        return len(self.blocks) * block_flops(self.d, M - 1) + head_flops(self.d, M - 1)
