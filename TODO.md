# TODO

Tracked work for this repo. Paper-scale reproduction is explicitly out of scope; see [README](README.md#deviations-from-the-papers).

## Pretraining & evaluation

- [x] Run a full PTB-XL pretrain (folds 1–8) and confirm `embed_std` stays healthy without target LayerNorm. Run 1 (`checkpoints/run1`, ~24 min on RTX 5070): val loss 0.214 → 0.149 by epoch 9, flat after epoch 5; `embed_std` 0.821 → 0.678, no collapse. Target LayerNorm left off.
- [ ] If `embed_std` collapses toward 0, enable target embedding LayerNorm in `ecg_jepa/models/jepa.py` (variant noted in code)
- [x] Linear probe on fold 10: freeze target encoder, mean-pool tokens, train a linear head on PTB-XL diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`). `checkpoints/run1`: fold-10 macro AUC 0.814 (NORM 0.891, MI 0.778, STTC 0.865, CD 0.786, HYP 0.750).
- [x] Baseline probe on a randomly initialized encoder (same head and protocol): fold-10 macro AUC 0.762. Pretrained minus random is +0.052.

## Masking & architecture experiments

- [x] Per-lead tokens (each lead patched, then mixed). Five seeds, fold-10 macro AUC 0.825 ± 0.006 versus temporal 0.820 ± 0.005. The gap is inside the seed spread.
- [x] Same masked times on every lead: masked times are dropped before the lead mixer, so no lead at a hidden time is visible. This is the per-lead run above, not a separate mask schedule.
- [ ] Multi-block masks
- [ ] Compare masking strategies before adding CroPA

## Training & ops

- [ ] Document expected PTB-XL disk layout and fold counts in run logs
- [ ] Optional: resume training from `checkpoints/last.pt`
- [ ] Optional: configurable `val_batches`, `log_every`, and model dims via CLI (today only epochs, batch size, lr, ckpt dir, seed)

## Out of scope (for now)

- Waveform augmentations in the JEPA objective
- ViT-B scale, `2.5e-5` LR, 100 epochs, drop-path, extra pretraining corpora
- Direct port of Kim et al. (arXiv:2410.08559) or Weimann and Conrad (arXiv:2410.13867)
