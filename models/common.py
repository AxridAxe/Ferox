"""Shared building blocks and the analytic FLOP model used by every experiment.

Every model is built from the same pre-LayerNorm Transformer block (width d, 4 heads,
MLP 4d, GELU, optional rotary position embedding). "One block application to one
token" is the common unit of computation; recurrence and depth are compared in it.

FLOP accounting (forward, multiply-add = 2 FLOPs), counted for every matmul we run:
  per block, per token:     2 * (4 d^2 + 2 d f)                 (QKV/out proj + MLP)
  per block, attention:     2 * 2 * Nq * Nk * d                  (QK^T and AV)
  output head per token:    2 * d * V
LayerNorm, softmax, GELU, RoPE, embedding lookups and adds are NOT counted (they are
O(N d) and < ~3% at these sizes); torch's FlopCounterMode is used in benchmark.py
to cross-check that the analytic numbers match the matmuls actually executed.
Training FLOPs = 3 x forward FLOPs of every activation that receives gradients
(+1 x forward for no-grad computation).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.common import VOCAB


def rope_cache(n, dh, device=None):
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    t = torch.arange(n, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x [B, H, N, dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, rope=None, causal=False, kv=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv is not None:  # incremental decoding cache
            if kv.get("k") is not None:
                k = torch.cat([kv["k"], k], dim=2)
                v = torch.cat([kv["v"], v], dim=2)
            kv["k"], kv["v"] = k, v
        Nk = k.shape[2]
        mask = None
        if causal and N > 1:
            mask = torch.ones(N, Nk, dtype=torch.bool, device=x.device).tril(Nk - N)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, heads=4, ff_mult=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, x, rope=None, causal=False, kv=None):
        x = x + self.attn(self.n1(x), rope, causal, kv)
        return x + self.mlp(self.n2(x))


class Embed(nn.Module):
    """Sum of three field embeddings (index 0 = empty)."""

    def __init__(self, d):
        super().__init__()
        self.tabs = nn.ModuleList(nn.Embedding(VOCAB, d, padding_idx=0) for _ in range(3))
        for t in self.tabs:
            nn.init.normal_(t.weight, std=0.02)
            with torch.no_grad():
                t.weight[0].zero_()

    def forward(self, x):
        return sum(t(x[..., i]) for i, t in enumerate(self.tabs))


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, VOCAB)

    def forward(self, h):
        return self.proj(self.norm(h))


# ----------------------------------------------------------------------------- FLOPs
def block_flops(d, nq, nk=None, ff_mult=4):
    nk = nq if nk is None else nk
    return 2 * nq * (4 * d * d + 2 * d * ff_mult * d) + 4 * nq * nk * d


def head_flops(d, n):
    return 2 * n * d * VOCAB


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EncoderBase(nn.Module):
    """Common parent for direct-prediction models (A, C, D): input -> per-position logits."""

    def __init__(self, d, use_rope):
        super().__init__()
        self.d, self.use_rope = d, use_rope
        self.embed = Embed(d)
        self.head = Head(d)

    def rope(self, n, device):
        if not self.use_rope:
            return None
        cos, sin = rope_cache(n, self.d // 4, device)  # 4 heads
        return cos, sin
