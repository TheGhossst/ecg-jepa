"""Tiny defaults for the first PTB-XL JEPA run.

Paper-scale settings (ViT-B, 2.5e-5, 100 epochs, drop-path 0.1) are not used
here. See README for the deviations.
"""

from dataclasses import dataclass


@dataclass
class Config:
    n_leads: int = 8
    signal_length: int = 1000
    patch_size: int = 25
    enc_dim: int = 128
    enc_depth: int = 4
    enc_heads: int = 4
    pred_dim: int = 64
    pred_depth: int = 2
    pred_heads: int = 4
    mlp_ratio: float = 4.0
    # temporal: one token mixes all leads. per_lead: each lead is patched, then mixed.
    token_mode: str = "temporal"
    lead_dim: int = 32
    mask_ratio_min: float = 0.6
    mask_ratio_max: float = 0.7
    # random: independent time indices. multiblock: 4 overlapping spans.
    mask_mode: str = "random"
    multiblock_count: int = 4
    multiblock_ratio_min: float = 0.175
    multiblock_ratio_max: float = 0.225
    ema_start: float = 0.996
    ema_end: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 16
    epochs: int = 10
    warmup_epochs: float = 0.5
    seed: int = 0
    log_every: int = 10
    num_workers: int = 0
    ckpt_dir: str = "checkpoints"

    @property
    def n_patches(self) -> int:
        if self.signal_length % self.patch_size != 0:
            raise ValueError("signal_length must be divisible by patch_size")
        return self.signal_length // self.patch_size
