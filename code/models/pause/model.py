"""PAUSE cross-epoch packed-attention classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import EpochMemoryRoPEEncoder


def resolve_rope_base(rope_mode, rope_base):
    """Resolve rope_base default from rope_mode: 1d->1000, 2d->30; explicit value passes through."""
    if rope_mode not in ("1d", "2d"):
        raise ValueError("rope_mode must be '1d' or '2d'")
    if rope_base is None:
        return 1000.0 if rope_mode == "1d" else 30.0
    return rope_base


class PatchEmbeddingSwiGLU(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.hidden_dim = embed_dim
        self.gate = nn.Linear(in_dim, self.hidden_dim)
        self.up = nn.Linear(in_dim, self.hidden_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, embed_dim)

    def forward(self, x):
        hidden = self.act(self.gate(x)) * self.up(x)
        return self.down(self.norm(hidden))


class ClassicPatchEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.hidden_dim = embed_dim
        self.up = nn.Linear(in_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, embed_dim)

    def forward(self, x):
        hidden = self.act(self.up(x))
        return self.down(self.norm(hidden))


class CausalPatchStem(nn.Module):
    def __init__(self, dim, left_context=4, dropout=0.0, activation="swiglu"):
        super().__init__()
        if left_context < 0:
            raise ValueError("left_context must be non-negative")
        if activation not in ("swiglu", "classic"):
            raise ValueError("activation must be 'swiglu' or 'classic'")
        self.dim = dim
        self.activation = activation
        self.left_context = int(left_context)
        kernel_size = self.left_context + 1
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=0,
        )
        self.expanded_dim = 2 * dim
        up_out = 2 * self.expanded_dim if activation == "swiglu" else self.expanded_dim
        self.pointwise_up = nn.Conv1d(
            dim,
            up_out,
            kernel_size=1,
        )
        self.act = nn.SiLU()
        self.pointwise_down = nn.Conv1d(
            self.expanded_dim,
            dim,
            kernel_size=1,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("x must be [B,E,P,D]")
        batch, epochs, patches, dim = x.shape
        if dim != self.dim:
            raise ValueError("x last dimension does not match stem dim")
        residual = x
        hidden = self.norm(x)
        hidden = hidden.reshape(batch * epochs, patches, dim).transpose(1, 2)
        hidden = F.pad(hidden, (self.left_context, 0))
        hidden = self.depthwise(hidden)
        if self.activation == "swiglu":
            gate, value = self.pointwise_up(hidden).chunk(2, dim=1)
            hidden = self.act(gate) * value
        else:
            hidden = self.act(self.pointwise_up(hidden))
        hidden = self.pointwise_down(hidden)
        hidden = self.drop(hidden)
        hidden = hidden.transpose(1, 2).reshape(batch, epochs, patches, dim)
        return residual + hidden


class PAUSE(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_ratio=8 / 3,
        dropout=0.1,
        num_epochs=32,
        num_patches=30,
        num_classes=5,
        in_dim=128,
        rope_base=None,
        rope_mode="2d",
        causal_patch_stem=True,
        patch_stem_left_context=4,
        activation="swiglu",
        cross_epoch_memory=True,
    ):
        super().__init__()
        if activation not in ("swiglu", "classic"):
            raise ValueError("activation must be 'swiglu' or 'classic'")
        rope_base = resolve_rope_base(rope_mode, rope_base)
        self.rope_base = rope_base
        self.activation = activation
        self.cross_epoch_memory = bool(cross_epoch_memory)
        self.num_epochs = num_epochs
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.causal_patch_stem_enabled = bool(causal_patch_stem)
        self.patch_stem_left_context = int(patch_stem_left_context)
        self.patch_embedding = (
            PatchEmbeddingSwiGLU(in_dim, embed_dim)
            if activation == "swiglu"
            else ClassicPatchEmbedding(in_dim, embed_dim)
        )
        self.causal_patch_stem = (
            CausalPatchStem(
                embed_dim,
                left_context=self.patch_stem_left_context,
                dropout=dropout,
                activation=activation,
            )
            if self.causal_patch_stem_enabled
            else nn.Identity()
        )
        self.encoder = EpochMemoryRoPEEncoder(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_epochs=num_epochs,
            num_patches=num_patches,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            base=rope_base,
            rope_mode=rope_mode,
            activation=activation,
            cross_epoch_memory=cross_epoch_memory,
        )
        self.norm = nn.RMSNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, end_idx, return_summary=False):
        if x.ndim != 4:
            raise ValueError("x must be [B,E,P,F]")
        batch, epochs, patches, _ = x.shape
        if epochs != self.num_epochs or patches != self.num_patches:
            raise ValueError("x shape does not match configured E/P")
        if end_idx.shape != (batch, epochs):
            raise ValueError("end_idx must be [B,E]")
        hidden = self.patch_embedding(x)
        hidden = self.causal_patch_stem(hidden)
        hidden = self.encoder(hidden, end_idx)
        hidden = self.norm(hidden)
        logits = self.head(hidden)
        if not return_summary:
            return logits
        gather_idx = end_idx[:, :, None, None].expand(
            batch, epochs, 1, self.embed_dim
        )
        summary = hidden.gather(2, gather_idx).squeeze(2)
        return logits, summary
