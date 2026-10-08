"""1D patch embedding and a small pre-norm transformer encoder."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Non-overlapping temporal patches. Leads are the conv channels."""

    def __init__(self, n_leads: int, patch_size: int, dim: int):
        super().__init__()
        self.proj = nn.Conv1d(
            n_leads,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, leads, time) -> (B, n_patches, dim)
        return self.proj(x).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class LeadPatchEmbed(nn.Module):
    """Patch each lead with the same convolution. Output is (B, leads, time, dim)."""

    def __init__(self, patch_size: int, lead_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(1, lead_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, leads, _time = x.shape
        flat = x.reshape(batch * leads, 1, _time)
        tokens = self.proj(flat).transpose(1, 2)
        n_patches = tokens.shape[1]
        return tokens.reshape(batch, leads, n_patches, -1)


class LeadMixer(nn.Module):
    """Mix the eight leads at one time, then project back to the model width.

    Masked times are removed before this mix, so a hidden time cannot be read
    from any lead.
    """

    def __init__(self, n_leads: int, lead_dim: int, dim: int, heads: int):
        super().__init__()
        if lead_dim % heads != 0:
            raise ValueError(f"lead_dim {lead_dim} must be divisible by heads {heads}")
        self.lead_pos = nn.Parameter(torch.zeros(1, n_leads, lead_dim))
        nn.init.trunc_normal_(self.lead_pos, std=0.02)
        self.block = TransformerBlock(lead_dim, heads, mlp_ratio=2.0)
        self.proj = nn.Linear(n_leads * lead_dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, leads, n_times, dim = tokens.shape
        mixed = tokens.permute(0, 2, 1, 3).reshape(batch * n_times, leads, dim)
        mixed = self.block(mixed + self.lead_pos)
        return self.proj(mixed.reshape(batch, n_times, leads * dim))


class Encoder(nn.Module):
    """Patch embed, learned 1D positions, then transformer blocks.

    `keep_idx` selects visible time patches. None means the full recording,
    which is what the target encoder uses.

    `token_mode="temporal"` mixes leads inside the patch convolution.
    `token_mode="per_lead"` patches each lead, drops masked times, then mixes
    the leads that remain at each visible time.
    """

    def __init__(
        self,
        n_leads: int,
        n_patches: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        token_mode: str = "temporal",
        lead_dim: int = 32,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        if token_mode not in ("temporal", "per_lead"):
            raise ValueError(f"unknown token_mode {token_mode}")
        self.token_mode = token_mode
        if token_mode == "temporal":
            self.patch_embed = PatchEmbed(n_leads, patch_size, dim)
        else:
            self.lead_embed = LeadPatchEmbed(patch_size, lead_dim)
            self.lead_mixer = LeadMixer(n_leads, lead_dim, dim, heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def embed(self, x: torch.Tensor, keep_idx: torch.Tensor | None) -> torch.Tensor:
        if self.token_mode == "temporal":
            tokens = self.patch_embed(x)
            if keep_idx is not None:
                tokens = tokens.index_select(1, keep_idx)
            return tokens
        tokens = self.lead_embed(x)
        if keep_idx is not None:
            tokens = tokens.index_select(2, keep_idx)
        return self.lead_mixer(tokens)

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None) -> torch.Tensor:
        positions = self.pos_embed if keep_idx is None else self.pos_embed[:, keep_idx]
        tokens = self.embed(x, keep_idx) + positions
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
