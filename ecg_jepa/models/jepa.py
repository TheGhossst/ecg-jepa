"""JEPA objective for a multi-lead ECG.

Maps onto Kim et al. Section 3 (arXiv:2410.08559), at a much smaller scale:

- context encoder: the student. It sees only visible time patches (Sec 3.2).
- target encoder: the teacher. It sees the full ECG. Its weights are an EMA of
  the student and it gets no gradient (Appendix B).
- predictor: a narrower transformer. Mask tokens carry the hidden times (Sec 3.2).
- loss: smooth L1 between predicted and teacher embeddings at masked times.

The mask is a set of time indices shared by the batch. In temporal mode one
token already contains every lead. In per_lead mode each lead is patched on
its own and masked times are dropped before the leads are mixed, so a hidden
time cannot be copied from another lead. CroPA is not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ecg_jepa.config import Config
from ecg_jepa.models.transformer import Encoder, TransformerBlock


@dataclass
class JEPAOutput:
    loss: torch.Tensor
    embed_std: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    keep_idx: torch.Tensor
    mask_idx: torch.Tensor


class Predictor(nn.Module):
    """Predict target-encoder embeddings at the masked times.

    Context embeddings are projected into the predictor width. A single learned
    mask token, plus a 1D position embedding, fills each masked index. Predictor
    positions are also added to the context tokens so both streams share one
    time axis after the projection.
    """

    def __init__(
        self,
        n_patches: int,
        enc_dim: int,
        pred_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        if pred_dim % heads != 0:
            raise ValueError(f"pred_dim {pred_dim} must be divisible by heads {heads}")
        self.proj = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, pred_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(pred_dim, heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.head = nn.Linear(pred_dim, enc_dim)

    def forward(
        self,
        context: torch.Tensor,
        keep_idx: torch.Tensor,
        mask_idx: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        batch = context.shape[0]
        ctx = self.proj(context) + self.pos_embed[:, keep_idx]
        mask = self.mask_token.expand(batch, mask_idx.numel(), -1)
        mask = mask + self.pos_embed[:, mask_idx]
        dim = ctx.shape[-1]
        seq = ctx.new_zeros(batch, n_patches, dim)
        seq = seq.scatter(1, keep_idx.view(1, -1, 1).expand(batch, -1, dim), ctx)
        seq = seq.scatter(1, mask_idx.view(1, -1, 1).expand(batch, -1, dim), mask)
        for block in self.blocks:
            seq = block(seq)
        pred = self.head(self.norm(seq))
        return pred.index_select(1, mask_idx)


def sample_time_mask(
    n_patches: int,
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One mask for the whole batch, so every sample has the same shape."""
    ratio = torch.empty((), device=device).uniform_(ratio_min, ratio_max).item()
    n_mask = int(round(ratio * n_patches))
    n_mask = min(max(n_mask, 1), n_patches - 1)
    perm = torch.randperm(n_patches, device=device)
    mask_idx = perm[:n_mask].sort().values
    keep_idx = perm[n_mask:].sort().values
    return keep_idx, mask_idx


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """target <- momentum * target + (1 - momentum) * online."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)


def ema_momentum(step: int, total_steps: int, ema_start: float, ema_end: float) -> float:
    """Linear ramp used by Kim et al. Appendix B. `step` is the iteration index."""
    if total_steps <= 0:
        return ema_end
    return ema_start + step * (ema_end - ema_start) / total_steps


class JEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_patches = cfg.n_patches
        self.mask_ratio_min = cfg.mask_ratio_min
        self.mask_ratio_max = cfg.mask_ratio_max
        encoder_kwargs = dict(
            n_leads=cfg.n_leads,
            n_patches=cfg.n_patches,
            patch_size=cfg.patch_size,
            dim=cfg.enc_dim,
            depth=cfg.enc_depth,
            heads=cfg.enc_heads,
            mlp_ratio=cfg.mlp_ratio,
            token_mode=cfg.token_mode,
            lead_dim=cfg.lead_dim,
        )
        self.context_encoder = Encoder(**encoder_kwargs)
        self.target_encoder = Encoder(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = Predictor(
            n_patches=cfg.n_patches,
            enc_dim=cfg.enc_dim,
            pred_dim=cfg.pred_dim,
            depth=cfg.pred_depth,
            heads=cfg.pred_heads,
            mlp_ratio=cfg.mlp_ratio,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The teacher is an EMA copy and should not use dropout if it is added later.
        self.target_encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> JEPAOutput:
        keep_idx, mask_idx = sample_time_mask(
            self.n_patches,
            self.mask_ratio_min,
            self.mask_ratio_max,
            x.device,
        )
        context = self.context_encoder(x, keep_idx)
        with torch.no_grad():
            target_tokens = self.target_encoder(x)
        pred = self.predictor(context, keep_idx, mask_idx, self.n_patches)
        target = target_tokens.index_select(1, mask_idx)
        # Guess, not used: several I-JEPA codebases LayerNorm the target
        # embeddings before the loss. Turn this on only if embed_std collapses.
        # target = F.layer_norm(target, (target.shape[-1],))
        loss = F.smooth_l1_loss(pred, target)
        embed_std = target_tokens.std(dim=0, unbiased=False).mean()
        return JEPAOutput(
            loss=loss,
            embed_std=embed_std,
            pred=pred,
            target=target,
            keep_idx=keep_idx,
            mask_idx=mask_idx,
        )
