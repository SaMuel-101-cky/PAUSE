"""RoPE + causal self-attention (PAUSE, Scheme A).

RoPE is the only positional encoding in this project (no separate position
encoding addition). The causal mask is provided by
F.scaled_dot_product_attention(is_causal=True), equivalent to KV-cache for
future inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None):
    """Return cos, sin, both of shape [seq_len, head_dim//2]."""
    assert head_dim % 2 == 0, "head_dim must be even"
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))  # [half]
    t = torch.arange(seq_len, device=device).float()  # [T]
    angles = torch.outer(t, freqs)                      # [T, half]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE to x[B,H,T,head_dim] (interleaved pairs, RoFormer original convention).

    cos,sin: [T, head_dim//2]. out_even = x1*cos - x2*sin; out_odd = x1*sin + x2*cos.
    """
    x1 = x[..., 0::2]  # [B,H,T,half]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)  # Re-interleave back to [B,H,T,head_dim]


class RoPECausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, base=10000.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even to support RoPE"
        self.base = base
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,T,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = build_rope_cache(T, self.head_dim, self.base, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # [B,H,T,hd]
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))

    def forward_step(self, x_t, kv_cache):
        """Placeholder: reserved for S2 streaming early stopping (not implemented in this phase)."""
        raise NotImplementedError("forward_step belongs to S2 (streaming early stopping), not implemented in this phase")


class RoPEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, base=10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPECausalSelfAttention(dim, num_heads, dropout, base=base)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RoPECausalEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4, dropout=0.1, base=10000.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            RoPEBlock(dim, num_heads, mlp_ratio, dropout, base=base) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
