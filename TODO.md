# TODO

Tracked work for this repo. Paper-scale reproduction is explicitly out of scope; see [README](README.md#deviations-from-the-papers).

## Four checks after the locked baselines

- [x] Lock the 5-seed per-class table. Frozen JEPA 0.820 ± 0.005, frozen random 0.767 ± 0.006, fine-tuned JEPA 0.872 ± 0.003, from scratch 0.869 ± 0.003. Written to `logs/baselines_per_class.json`.
- [x] Multilabel probe on the 44 diagnostic statements. Frozen JEPA 0.680 ± 0.011, random encoder 0.658 ± 0.016.
- [x] Overlap analysis. Pure MI 0.697 ± 0.008 versus mixed MI 0.848 ± 0.003. Pure HYP 0.568 ± 0.010 versus mixed HYP 0.804 ± 0.007. Overlap makes MI and HYP look easier. Isolated HYP is the hard slice.
- [x] Representation probe. Subclass AUC stays at 0.702 ± 0.026 after the five superclass directions are removed, against 0.762 ± 0.012 on the full embedding. An MLP on the five superclasses gains about the same amount for JEPA and for a random encoder.
- [x] Decision: a new pretraining method is not justified. From-scratch supervised training matches JEPA, and the finer labels the five-class head misses are already in the embedding.

## Fine-tuning

- [x] Finish 5-seed pretrained fine-tuning. Fold-10 macro AUC 0.872 ± 0.003.
- [x] Finish 5-seed random-initialization fine-tuning. Fold-10 macro AUC 0.869 ± 0.003.
- [x] Compare against frozen JEPA (0.820 ± 0.005). Paired fine-tune minus frozen, by seed: +0.060, +0.049, +0.050, +0.060, +0.043.
- [x] Report mean ± sample std and paired seed differences. Pretrained minus scratch: +0.006, +0.001, −0.002, +0.007, +0.003. Mean +0.003 ± 0.004.
- [x] Decide whether JEPA pretraining provides a fine-tuning advantage. It does not, when folds 1–8 are fully labeled. Unfreezing the encoder does. The gain over a frozen head is about +0.05 and is positive on every seed.

## Label budgets

- [x] Freeze the temporal checkpoints. Train the same linear head and the same fine-tune on 1% (174/17418) and 10% (1741/17418) of folds 1–8. Fold 9 chooses the epoch. Fold 10 is scored once. Hyperparameters are the full-label ones. Written to `logs/low_label.json`.
- [x] At 1%, fine-tuned JEPA is 0.756 ± 0.019 and from scratch is 0.750 ± 0.011. Paired differences −0.030, +0.009, +0.022, +0.014, +0.010. Mean +0.005 ± 0.020, inside the seed spread. The frozen head is 0.606 ± 0.028, which is −0.145 ± 0.027 versus training from scratch.
- [x] At 10%, fine-tuned JEPA is 0.831 ± 0.002 and from scratch is 0.817 ± 0.003. Paired differences +0.016, +0.011, +0.016, +0.018, +0.011. Mean +0.014 ± 0.003, above the seed spread, positive on every seed. The frozen head is 0.779 ± 0.017 and still loses to scratch (−0.038 ± 0.015).
- [x] Decision: JEPA does not help at 1%. The fine-tune helps at 10%. The fully labeled conclusion stays. CroPA, target LayerNorm, a new mask, and ViT-B training stay off.

## Pure HYP

- [x] Break out the 56 fold-10 pure-HYP recordings. 50 LVH, 2 RVH, 4 atrial enlargement (3 RAO/RAE, 1 LAO/LAE). No recording is both LVH and RVH. Written to `logs/hyp_voltage.json`.
- [x] Peak R in millivolts, tallest positive sample on V1–V6, scores pure HYP at 0.740 and mixed HYP at 0.724. The frozen embedding is 0.568 ± 0.010 on pure HYP. The millivolt peak is above every seed.
- [x] The same peak after the loader's per-lead z-score scores pure HYP at 0.535, below every seed. The separating feature is absolute voltage, removed before the patch embedding. A larger encoder is not justified. `embed_std` stayed near 0.68, so the LayerNorm variant in `jepa.py` stays off.

## Amplitude features

- [x] A. Linear head on millivolt features, folds 1–8, fold 9 chooses the epoch, fold 10 scored once. Untrained peak R remains pure HYP 0.740. The one-feature head matches that on three seeds and reverses it on two. Per-lead peak, standard deviation, and peak-to-peak (24 features): macro 0.722 ± 0.002, HYP 0.796 ± 0.002, pure HYP 0.813 ± 0.010. Written to `logs/amplitude.json`.
- [x] B. Same head on the frozen embedding concatenated with those 24 features. Macro 0.844 ± 0.004 versus embedding 0.820 ± 0.005. Paired +0.024, +0.025, +0.023, +0.026, +0.022. Mean +0.024 ± 0.002. Pure HYP +0.193 ± 0.016. The checkpoint was not updated.
- [x] C. Ran because B cleared the seed spread. Width-32 amplitude branch plus the frozen embedding: macro 0.858 ± 0.004, which is +0.014 ± 0.002 over the linear concatenation and +0.038 ± 0.003 over the embedding. Pure HYP 0.817 ± 0.009. Encoder weights stayed put.
- [x] Decision: attach millivolt amplitude beside the frozen encoder. Do not change the pretraining objective. LayerNorm stays off. A larger encoder stays off.

## Classification

- [x] Save the width-32 two-branch head beside each temporal encoder as `two_branch.pt`. The file holds the head weights and the folds 1–8 amplitude mean and scale. The encoder file stays `last.pt`.
- [x] `python -m ecg_jepa.predict` reads one millivolt ECG and prints five sigmoid scores (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
- [x] Threshold 0.5 per class, fixed before fold 10. Fold 9 still chooses the epoch. Fold-10 exact-match accuracy, all five calls correct, is 0.489 ± 0.011. Per-label accuracy averages about 0.845 because one wrong call does not fail the other four. HYP precision is 0.714 ± 0.015 and HYP recall is 0.335 ± 0.008. On the 56 pure-HYP recordings the full five-label call is right for 0.096 ± 0.037. Written to `logs/amplitude.json`.

## Next experiments

- [x] Test CroPA only if fine-tuning results justify further JEPA changes. They do not. CroPA stays off.
- [x] Analyze MI/HYP performance. On fold 10, MI is 550/2198 records and HYP is 262/2198, with only 56 hypertrophy-only recordings. 59% of HYP also has STTC, and 32% of MI also has a conduction defect. Frozen AUCs are MI 0.778 and HYP 0.754. Fine-tuning raises MI to 0.859 and HYP only to 0.786. Scratch matches both.
- [x] Add a small supervised-from-scratch baseline. This is the random-initialization fine-tune above, same encoder and schedule.
- [x] Analyze learned embedding structure. On fold 10, seed 0, class means with the shared offset removed: normal is opposite myocardial infarction (−0.94) and conduction defect (−0.90). Infarction and conduction defect point the same way (0.90). Hypertrophy and ST-T change point the same way (0.85). The frozen embedding mostly separates normal from abnormal, and that is why MI and HYP stay hard.

## Pretraining & evaluation

- [x] Run a full PTB-XL pretrain (folds 1–8) and confirm `embed_std` stays healthy without target LayerNorm. Run 1 (`checkpoints/run1`, ~24 min on RTX 5070): val loss 0.214 → 0.149 by epoch 9, flat after epoch 5; `embed_std` 0.821 → 0.678, no collapse. Target LayerNorm left off.
- [ ] If `embed_std` collapses toward 0, enable target embedding LayerNorm in `ecg_jepa/models/jepa.py` (variant noted in code)
- [x] Linear probe on fold 10: freeze target encoder, mean-pool tokens, train a linear head on PTB-XL diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`). `checkpoints/run1`: fold-10 macro AUC 0.814 (NORM 0.891, MI 0.778, STTC 0.865, CD 0.786, HYP 0.750).
- [x] Baseline probe on a randomly initialized encoder (same head and protocol): fold-10 macro AUC 0.762. Pretrained minus random is +0.052.
- [x] Fine-tune temporal encoder versus the same fine-tune from random initialization. Five seeds, fold 9 selects the epoch, fold 10 is scored once. Fine-tuned pretrained 0.872 ± 0.003, from scratch 0.869 ± 0.003, frozen probe 0.820 ± 0.005. The frozen head was the limit. Pretraining does not beat supervised training from scratch when folds 1–8 are fully labeled.

## Masking & architecture experiments

- [x] Per-lead tokens (each lead patched, then mixed). Five seeds, fold-10 macro AUC 0.825 ± 0.006 versus temporal 0.820 ± 0.005. The gap is inside the seed spread.
- [x] Same masked times on every lead: masked times are dropped before the lead mixer, so no lead at a hidden time is visible. This is the per-lead run above, not a separate mask schedule.
- [x] Multi-block masks on the temporal encoder: 4 overlapping spans at ratio (0.175, 0.225). Five seeds, fold-10 macro AUC 0.828 ± 0.005 versus random time masks at 0.820 ± 0.005. Mean gap +0.008.
- [x] Compare masking strategies before adding CroPA. Random time masks remain the default. Multi-block is a small consistent lift, not a 0.015 gap. CroPA is not justified by this comparison.

## Training & ops

- [ ] Document expected PTB-XL disk layout and fold counts in run logs
- [ ] Optional: resume training from `checkpoints/last.pt`
- [ ] Optional: configurable `val_batches`, `log_every`, and model dims via CLI (today only epochs, batch size, lr, ckpt dir, seed)

## Out of scope (for now)

- Waveform augmentations in the JEPA objective
- ViT-B scale, `2.5e-5` LR, 100 epochs, drop-path, extra pretraining corpora
- Direct port of Kim et al. (arXiv:2410.08559) or Weimann and Conrad (arXiv:2410.13867)
