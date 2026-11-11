"""PAUSE cross-epoch packed RoPE attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rope_attention import apply_rope, build_rope_cache


def build_epoch_memory_allowed(
    num_epochs,
    num_patches,
    device=None,
    cross_epoch_memory=True,
):
    local = torch.tril(torch.ones(
        num_patches,
        num_patches,
        dtype=torch.bool,
        device=device,
    ))
    target = torch.arange(num_epochs, device=device)[:, None]
    source = torch.arange(num_epochs, device=device)[None, :]
    memory = source < target
    if not cross_epoch_memory:
        memory = torch.zeros_like(memory)
    return torch.cat([
        local[None].expand(num_epochs, num_patches, num_patches),
        memory[:, None].expand(num_epochs, num_patches, num_epochs),
    ], dim=-1)


def build_epoch_memory_bias(allowed, dtype, device):
    epochs, patches, width = allowed.shape
    return torch.zeros(
        1,
        epochs,
        1,
        patches,
        width,
        dtype=dtype,
        device=device,
    ).masked_fill(
        ~allowed[None, :, None].to(device=device),
        float("-inf"),
    )


def _validate_packed_inputs(
    q,
    k,
    v,
    end_idx,
    allowed,
    *,
    validate_end_idx,
):
    if q.ndim != 5 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must share [B,E,H,P,Dh]")
    batch, epochs, _, patches, _ = q.shape
    if end_idx.shape != (batch, epochs):
        raise ValueError("end_idx must be [B,E]")
    if end_idx.dtype != torch.long:
        raise ValueError("end_idx must use torch.long")
    if validate_end_idx and torch.any((end_idx < 0) | (end_idx >= patches)):
        raise ValueError("end_idx must be in [0,P-1]")
    if allowed.shape != (epochs, patches, patches + epochs):
        raise ValueError("allowed must be [E,P,P+E]")


def packed_epoch_memory_sdpa(
    q,
    k,
    v,
    end_idx,
    allowed,
    dropout_p=0.0,
    *,
    attn_bias=None,
    validate_end_idx=True,
):
    _validate_packed_inputs(
        q,
        k,
        v,
        end_idx,
        allowed,
        validate_end_idx=validate_end_idx,
    )
    batch, epochs, heads, patches, head_dim = q.shape
    gather_idx = end_idx[:, :, None, None, None].expand(
        batch, epochs, heads, 1, head_dim
    )
    mem_k = k.gather(dim=3, index=gather_idx).squeeze(3)
    mem_v = v.gather(dim=3, index=gather_idx).squeeze(3)
    mem_k = mem_k[:, None].expand(
        batch, epochs, epochs, heads, head_dim
    ).permute(0, 1, 3, 2, 4)
    mem_v = mem_v[:, None].expand(
        batch, epochs, epochs, heads, head_dim
    ).permute(0, 1, 3, 2, 4)
    k_cat = torch.cat([k, mem_k], dim=3)
    v_cat = torch.cat([v, mem_v], dim=3)
    q_flat = q.reshape(batch * epochs, heads, patches, head_dim)
    k_flat = k_cat.reshape(
        batch * epochs, heads, patches + epochs, head_dim
    )
    v_flat = v_cat.reshape(
        batch * epochs, heads, patches + epochs, head_dim
    )
    if attn_bias is None:
        attn_bias = build_epoch_memory_bias(allowed, q.dtype, q.device)
    expected_bias_shape = (1, epochs, 1, patches, patches + epochs)
    if attn_bias.shape != expected_bias_shape:
        raise ValueError(f"attn_bias must be {expected_bias_shape}")
    if attn_bias.device != q.device or attn_bias.dtype != q.dtype:
        raise ValueError("attn_bias must match q device and dtype")
    bias = attn_bias
    bias = bias.expand(
        batch, epochs, 1, patches, patches + epochs
    ).reshape(batch * epochs, 1, patches, patches + epochs)
    out = F.scaled_dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        attn_mask=bias,
        dropout_p=dropout_p,
        is_causal=False,
    )
    return out.reshape(
        batch, epochs, heads, patches, head_dim
    )


def apply_epoch_window_rope(q, k, base, cache=None):
    if q.shape != k.shape or q.ndim != 5:
        raise ValueError("q/k must share [B,E,H,P,Dh]")
    batch, epochs, heads, patches, head_dim = q.shape
    q_flat = q.permute(0, 2, 1, 3, 4).reshape(
        batch, heads, epochs * patches, head_dim
    )
    k_flat = k.permute(0, 2, 1, 3, 4).reshape(
        batch, heads, epochs * patches, head_dim
    )
    if cache is None:
        cos, sin = build_rope_cache(
            epochs * patches,
            head_dim,
            base,
            q.device,
        )
    else:
        cos, sin = cache
    q_flat = apply_rope(q_flat, cos, sin)
    k_flat = apply_rope(k_flat, cos, sin)
    q = q_flat.reshape(
        batch, heads, epochs, patches, head_dim
    ).permute(0, 2, 1, 3, 4)
    k = k_flat.reshape(
        batch, heads, epochs, patches, head_dim
    ).permute(0, 2, 1, 3, 4)
    return q, k


def build_axis_rope_cache(num_positions, rope_dim, base=10000.0, device=None):
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    half = rope_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(num_positions, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope_half(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


def apply_epoch_window_rope_2d(q, k, base=10000.0, cache=None):
    if q.shape != k.shape or q.ndim != 5:
        raise ValueError("q/k must share [B,E,H,P,Dh]")
    _, epochs, _, patches, head_dim = q.shape
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
    axis_dim = head_dim // 2
    if cache is None:
        cos_p, sin_p = build_axis_rope_cache(
            patches, axis_dim, base, q.device
        )
        cos_e, sin_e = build_axis_rope_cache(
            epochs, axis_dim, base, q.device
        )
    else:
        cos_p, sin_p, cos_e, sin_e = cache
    cos_p = cos_p[None, None, None]      # [1,1,1,P,axis_dim//2]
    sin_p = sin_p[None, None, None]
    cos_e = cos_e[None, :, None, None]   # [1,E,1,1,axis_dim//2]
    sin_e = sin_e[None, :, None, None]

    def rot(x):
        xp = apply_rope_half(x[..., :axis_dim], cos_p, sin_p)
        xe = apply_rope_half(x[..., axis_dim:], cos_e, sin_e)
        return torch.cat([xp, xe], dim=-1)

    return rot(q), rot(k)


class EpochMemoryRoPESelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_epochs,
        num_patches,
        dropout,
        base=10000.0,
        rope_mode="1d",
        cross_epoch_memory=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if rope_mode not in ("1d", "2d"):
            raise ValueError("rope_mode must be '1d' or '2d'")
        if rope_mode == "2d" and self.head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        self.num_epochs = num_epochs
        self.num_patches = num_patches
        self.base = base
        self.rope_mode = rope_mode
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer(
            "allowed",
            build_epoch_memory_allowed(
                num_epochs,
                num_patches,
                cross_epoch_memory=cross_epoch_memory,
            ),
            persistent=False,
        )
        for name in (
            "attn_bias",
            "rope_cos",
            "rope_sin",
            "rope_cos_patch",
            "rope_sin_patch",
            "rope_cos_epoch",
            "rope_sin_epoch",
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._attn_bias_key = None
        self._rope_cache_device = None

    def _get_attention_bias(self, q):
        key = (str(q.device), q.dtype)
        if self._attn_bias_key != key:
            self.attn_bias = build_epoch_memory_bias(
                self.allowed,
                q.dtype,
                q.device,
            )
            self._attn_bias_key = key
        return self.attn_bias

    def _get_rope_cache(self, q):
        device_key = str(q.device)
        if self._rope_cache_device != device_key:
            if self.rope_mode == "1d":
                self.rope_cos, self.rope_sin = build_rope_cache(
                    self.num_epochs * self.num_patches,
                    self.head_dim,
                    self.base,
                    q.device,
                )
            else:
                axis_dim = self.head_dim // 2
                (
                    self.rope_cos_patch,
                    self.rope_sin_patch,
                ) = build_axis_rope_cache(
                    self.num_patches,
                    axis_dim,
                    self.base,
                    q.device,
                )
                (
                    self.rope_cos_epoch,
                    self.rope_sin_epoch,
                ) = build_axis_rope_cache(
                    self.num_epochs,
                    axis_dim,
                    self.base,
                    q.device,
                )
            self._rope_cache_device = device_key
        if self.rope_mode == "1d":
            return self.rope_cos, self.rope_sin
        return (
            self.rope_cos_patch,
            self.rope_sin_patch,
            self.rope_cos_epoch,
            self.rope_sin_epoch,
        )

    def forward(self, x, end_idx):
        if x.ndim != 4:
            raise ValueError("x must be [B,E,P,D]")
        batch, epochs, patches, dim = x.shape
        if epochs != self.num_epochs or patches != self.num_patches:
            raise ValueError("x E/P does not match configured E/P")
        qkv = self.qkv(x).reshape(
            batch,
            epochs,
            patches,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        rope_cache = self._get_rope_cache(q)
        if self.rope_mode == "2d":
            q, k = apply_epoch_window_rope_2d(
                q,
                k,
                self.base,
                cache=rope_cache,
            )
        else:
            q, k = apply_epoch_window_rope(
                q,
                k,
                self.base,
                cache=rope_cache,
            )
        out = packed_epoch_memory_sdpa(
            q,
            k,
            v,
            end_idx,
            self.allowed,
            dropout_p=self.attn_dropout if self.training else 0.0,
            attn_bias=self._get_attention_bias(q),
            validate_end_idx=False,
        )
        out = out.permute(0, 1, 3, 2, 4).reshape(
            batch, epochs, patches, dim
        )
        return self.proj_drop(self.proj(out))


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        self.hidden_dim = max(1, int(round(dim * float(mlp_ratio))))
        self.gate = nn.Linear(dim, self.hidden_dim)
        self.up = nn.Linear(dim, self.hidden_dim)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.down = nn.Linear(self.hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.drop1(x)
        x = self.down(x)
        return self.drop2(x)


class ClassicMLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        self.hidden_dim = max(1, int(round(dim * float(mlp_ratio))))
        self.up = nn.Linear(dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.down = nn.Linear(self.hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.up(x))
        x = self.drop1(x)
        x = self.down(x)
        return self.drop2(x)


class EpochMemoryRoPEBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_epochs,
        num_patches,
        mlp_ratio,
        dropout,
        base=10000.0,
        rope_mode="1d",
        activation="swiglu",
        cross_epoch_memory=True,
    ):
        super().__init__()
        if activation not in ("swiglu", "classic"):
            raise ValueError("activation must be 'swiglu' or 'classic'")
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = EpochMemoryRoPESelfAttention(
            dim,
            num_heads,
            num_epochs,
            num_patches,
            dropout,
            base=base,
            rope_mode=rope_mode,
            cross_epoch_memory=cross_epoch_memory,
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        mlp_cls = SwiGLUMLP if activation == "swiglu" else ClassicMLP
        self.mlp = mlp_cls(dim, mlp_ratio, dropout)

    def forward(self, x, end_idx):
        x = x + self.attn(self.norm1(x), end_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class EpochMemoryRoPEEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        num_epochs,
        num_patches,
        mlp_ratio=8 / 3,
        dropout=0.1,
        base=10000.0,
        rope_mode="1d",
        activation="swiglu",
        cross_epoch_memory=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            EpochMemoryRoPEBlock(
                dim,
                num_heads,
                num_epochs,
                num_patches,
                mlp_ratio,
                dropout,
                base=base,
                rope_mode=rope_mode,
                activation=activation,
                cross_epoch_memory=cross_epoch_memory,
            )
            for _ in range(depth)
        ])

    def forward(self, x, end_idx):
        for block in self.blocks:
            x = block(x, end_idx)
        return x
