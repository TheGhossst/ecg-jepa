# ecg-jepa

Learn useful ECG representations without labels. A context encoder sees the visible part of a recording and predicts the embedding of the masked time segments. The targets come from a second encoder, an exponential moving average of the first, which sees the full recording and receives no gradient. The loss is in embedding space, so the model is not asked to reconstruct waveform noise. That objective is the joint-embedding predictive architecture of Assran et al. (2023).

This repository is a small, testable core on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) version 1.0.3 (Wagner et al., 2020, 2022). It is not a port of ECG-JEPA (Kim, 2024) or of the JEPA pretraining study of Weimann and Conrad (2025).

Planned work is tracked in [TODO.md](TODO.md).

## Model

Each recording is 10 seconds from `records100` (100 Hz, 1000 samples). The eight independent leads `I, II, V1–V6` are kept. `III, aVR, aVL, aVF` are linear combinations of `I` and `II` and are dropped, as in Kim (2024).

A stride-25 convolution turns the signal into 40 tokens. Each token is 250 ms across all eight leads. One mask ratio in `(0.6, 0.7)` is drawn per step and shared by the batch. The context encoder (depth 4, dim 128) sees only the visible tokens. The predictor (depth 2, dim 64) reads those embeddings plus a learned mask token at each hidden time, and smooth-L1 matches the target embeddings at the masked positions.

Pretraining uses the official folds 1–8 and ignores diagnostic labels (Wagner et al., 2022). Fold 9 is for loss and embedding-std monitoring. Fold 10 is not loaded.

## Project layout

```
ecg_jepa/
  config.py          # defaults (patch size, dims, LR, folds via data module)
  train.py           # pretrain loop and CLI
  data/ptbxl.py      # PTB-XL loader and synthetic fallback
  models/
    jepa.py          # context/target encoders, predictor, EMA, loss
    transformer.py   # patch embed + transformer blocks
tests/
  test_jepa.py       # unit tests (synthetic data)
```

## Setup

Python 3.10+ recommended (type hints and `dataclass` usage).

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `wfdb`, `scipy`, `tqdm`.

## Smoke run without PTB-XL

```bash
python -m ecg_jepa.train --synthetic --max-steps 1
```

## Pretrain

Point `--data-dir` at the folder that contains `ptbxl_database.csv` and `records100/`, or at a parent that has that folder one level down. The Kaggle extract uses the second layout:

```text
dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
  ptbxl_database.csv
  records100/
```

```bash
python -m ecg_jepa.train --data-dir dataset
```

Useful CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | (required unless `--synthetic`) | PTB-XL root directory |
| `--synthetic` | off | Random sine waves; no download |
| `--epochs` | 10 | Full passes over fold 1–8 |
| `--batch-size` | 16 | Training batch size |
| `--lr` | `1e-3` | Peak learning rate (cosine after warmup) |
| `--max-steps` | none | Stop early after N optimizer steps |
| `--ckpt-dir` | `checkpoints` | Where to write checkpoints |
| `--seed` | 0 | RNG seed |
| `--device` | auto | `cuda` if available, else `cpu` |

Checkpoints land in `checkpoints/` (`epoch_*.pt` each epoch, `last.pt` at the end). Logs include smooth-L1 loss and the standard deviation of target embeddings across the batch. A falling std toward 0 means the representation is collapsing.

## Linear probe

Freeze the target encoder, mean-pool its tokens, and train a linear head on the five diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`) from the PTB-XL benchmark (Wagner et al., 2020; Strodthoff et al., 2021). The head trains on folds 1–8, the checkpoint is chosen by macro AUC on fold 9, and the reported number is fold 10. The same head is trained on a randomly initialized encoder.

```bash
python -m ecg_jepa.probe --data-dir dataset --ckpt checkpoints/run1/last.pt
```

On `checkpoints/run1`, fold-10 macro AUC was 0.814 for the pretrained encoder and 0.762 for a randomly initialized encoder.

Five seeds, with the head chosen on fold 9 and fold 10 scored once:

```bash
python -m ecg_jepa.robustness --variant temporal --seeds 0,1,2,3,4 --data-dir dataset
python -m ecg_jepa.robustness --variant per_lead --seeds 0,1,2,3,4 --data-dir dataset
```

| Variant | Pretrained | Random encoder | Difference |
| --- | --- | --- | --- |
| Temporal tokens | 0.820 ± 0.005 | 0.767 ± 0.006 | +0.053 ± 0.004 |
| Per-lead tokens | 0.825 ± 0.006 | 0.766 ± 0.010 | +0.058 ± 0.012 |
| Multi-block mask | 0.828 ± 0.005 | — | — |

Per-lead tokens patch each lead, drop a masked time before any lead can see it, then mix the eight leads. The mean gain over temporal tokens is about 0.005, inside the seed-to-seed spread.

Multi-block masking keeps the temporal encoder and hides four overlapping spans, each 17.5–22.5% of the recording. Across the same five seeds the fold-10 mean is 0.008 above random time masking. That is larger than the seed spread of either run, and smaller than a 0.015 gap.

Fine-tuning trains the encoder and the linear layer together for 20 epochs at learning rate `1e-4`. The epoch is chosen on fold 9. Fold 10 is scored once. The from-scratch run uses the same schedule and the same initialization seed as pretraining, without loading the checkpoint.

| Training | Fold-10 macro AUC |
| --- | --- |
| Frozen pretrained encoder | 0.820 ± 0.005 |
| Fine-tuned from the pretrained encoder | 0.872 ± 0.003 |
| Fine-tuned from random initialization | 0.869 ± 0.003 |

Unfreezing the encoder adds about 0.05 on every seed. Starting from the pretrained weights adds 0.003 ± 0.004 over training from scratch, and one of the five seeds goes the other way. The same comparison on 1% and 10% of folds 1–8 is in [Label budgets](#label-budgets).

### Locked per-class baselines

Fold-10 AUC, five seeds, sample standard deviation. The fine-tune minus frozen gaps by seed are +0.060, +0.049, +0.050, +0.060, +0.043. The fine-tune minus scratch gaps are +0.006, +0.001, −0.002, +0.007, +0.003.

| | NORM | MI | STTC | CD | HYP | Macro |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen JEPA | 0.896 ± 0.005 | 0.778 ± 0.002 | 0.882 ± 0.012 | 0.789 ± 0.009 | 0.754 ± 0.006 | 0.820 ± 0.005 |
| Frozen random | 0.831 ± 0.008 | 0.732 ± 0.008 | 0.822 ± 0.008 | 0.746 ± 0.005 | 0.703 ± 0.018 | 0.767 ± 0.006 |
| Fine-tuned JEPA | 0.923 ± 0.004 | 0.859 ± 0.006 | 0.916 ± 0.003 | 0.878 ± 0.004 | 0.786 ± 0.004 | 0.872 ± 0.003 |
| Fine-tuned from scratch | 0.919 ± 0.002 | 0.855 ± 0.004 | 0.916 ± 0.004 | 0.868 ± 0.005 | 0.787 ± 0.007 | 0.869 ± 0.003 |

### Finer labels, overlap, and unused directions

The same frozen encoders, with each new head chosen on fold 9:

| Probe | JEPA | Random encoder |
| --- | --- | --- |
| 44 diagnostic statements | 0.680 ± 0.011 | 0.658 ± 0.016 |
| 23 diagnostic subclasses | 0.762 ± 0.012 | 0.713 ± 0.010 |
| Subclasses after removing the 5 superclass directions | 0.702 ± 0.026 | 0.661 ± 0.024 |
| MLP on the 5 superclasses | 0.848 ± 0.003 | 0.798 ± 0.003 |

On fold 10 the frozen JEPA head scores pure myocardial infarction at 0.697 ± 0.008 and infarction that shares another superclass at 0.848 ± 0.003. Pure hypertrophy, 56 recordings, scores 0.568 ± 0.010. Hypertrophy that shares another label scores 0.804 ± 0.007. Overlap makes those two classes look easier, because the shared label is already visible. The isolated labels are the hard ones.

A new pretraining method is not justified by these four checks. Supervised training from scratch matches JEPA fine-tuning when folds 1–8 are fully labeled. Subclass information is already in the frozen embedding, including in directions the five-class head does not use. Isolated hypertrophy is barely above chance, and training from scratch does not solve it either.

### Label budgets

The checkpoints stay frozen. The linear head and the fine-tune use the same hyperparameters as the full-label runs. Each seed draws its own subset of folds 1–8 and shares it across the linear head, the fine-tune, and the from-scratch run. Fold 9 stays the full monitoring fold (2183 recordings) and chooses the epoch. Fold 10 (2198) is scored once. A gap counts when the mean paired fold-10 difference is larger than its sample standard deviation.

```bash
python -m ecg_jepa.low_label --data-dir dataset --seeds 0,1,2,3,4
```

Folds 1–8 have 17418 recordings. One percent is 174 recordings. Ten percent is 1741.

| Budget | Frozen JEPA | Frozen random | Fine-tuned JEPA | From scratch |
| --- | --- | --- | --- | --- |
| 1% | 0.606 ± 0.028 | 0.568 ± 0.023 | 0.756 ± 0.019 | 0.750 ± 0.011 |
| 10% | 0.779 ± 0.017 | 0.675 ± 0.010 | 0.831 ± 0.002 | 0.817 ± 0.003 |

At 1%, fine-tune minus scratch is −0.030, +0.009, +0.022, +0.014, +0.010. The mean is +0.005 ± 0.020. The frozen head minus scratch is −0.145 ± 0.027. JEPA does not win at the budget where a pretrained representation is supposed to matter.

At 10%, fine-tune minus scratch is +0.016, +0.011, +0.016, +0.018, +0.011. The mean is +0.014 ± 0.003, and every seed is positive. The frozen head still loses to training from scratch, by −0.038 ± 0.015. The useful regime is a fine-tune on 10% of folds 1–8. The fully labeled result stays as measured: +0.003 ± 0.004.

### Pure hypertrophy

```bash
python -m ecg_jepa.hyp_voltage --data-dir dataset
```

Of the 56 fold-10 recordings that are hypertrophy and nothing else, 50 are LVH (48 alone, one with septal hypertrophy, one with left atrial enlargement), 2 are RVH alone, and 4 are atrial enlargement with no ventricular hypertrophy (3 RAO/RAE, 1 LAO/LAE). None is both LVH and RVH.

Peak R is the tallest positive sample on V1–V6, in millivolts, with no fitted threshold. On the same pure-versus-mixed split as the frozen head, it scores pure HYP at 0.740 and mixed HYP at 0.724. The frozen embedding scores pure HYP at 0.568 ± 0.010, and the highest seed is 0.582. The millivolt peak is above every seed. Mixed HYP is the other way around: the embedding scores 0.804 ± 0.007 there, because the shared label is already visible.

The loader z-scores each lead before the encoder. The same peak after that z-score scores pure HYP at 0.535, below every frozen seed. The amplitude that separates isolated hypertrophy is the millivolt scale. The patch embedding is applied after that scale has been removed.

### Amplitude features

The features below are computed from the millivolt waveform before that z-score. Per-lead standard deviation is the scale the loader divides by. Peak-to-peak is max minus min. The linear head, the epoch rule, and the five temporal checkpoints are unchanged. A gap counts when the mean paired fold-10 difference is larger than its sample standard deviation.

```bash
python -m ecg_jepa.amplitude --data-dir dataset --seeds 0,1,2,3,4
```

| Model | Macro | HYP | Pure HYP |
| --- | --- | --- | --- |
| Amplitude only (24 features) | 0.722 ± 0.002 | 0.796 ± 0.002 | 0.813 ± 0.010 |
| Frozen embedding | 0.820 ± 0.005 | 0.754 ± 0.006 | 0.568 ± 0.010 |
| Embedding + amplitude | 0.844 ± 0.004 | 0.834 ± 0.004 | 0.761 ± 0.011 |
| Two-branch head | 0.858 ± 0.004 | 0.865 ± 0.005 | 0.817 ± 0.009 |

The untrained peak-R ranking is still pure HYP 0.740. A linear head on that single feature matches it on three seeds and lands on the reversed ranking (0.260) on the other two, so the one-feature head is not stable under the probe schedule. The 24-feature head is stable. It trails the frozen embedding on macro AUC and leads it on pure HYP.

Concatenating the 24 features with the frozen embedding raises macro AUC by +0.024, +0.025, +0.023, +0.026, +0.022. The mean is +0.024 ± 0.002. HYP rises by +0.080 ± 0.004, pure HYP by +0.193 ± 0.016, and mixed HYP by about +0.050 on every seed. MI rises by about +0.023. NORM does not move. That cleared the seed spread, so the two-branch head was trained: a width-32 network on the amplitude features, concatenated with the frozen embedding, then a linear layer. The encoder weights were unchanged.

The two-branch head is +0.038 ± 0.003 above the embedding and +0.014 ± 0.002 above the linear concatenation, both on every seed. Pure HYP is 0.817 ± 0.009, in line with the amplitude-only head. `embed_std` on the existing pretrain stayed near 0.68, so target-embedding LayerNorm stays off. Cross-Pattern Attention (CroPA; Kim, 2024), a new mask schedule, and a larger encoder stay off as well.

That head is the classifier. The amplitude run writes it beside the encoder as `two_branch.pt`: `checkpoints/run1/two_branch.pt` for seed 0, and `checkpoints/temporal/seed_<n>/two_branch.pt` for the others. The file holds the width-32 amplitude branch, the linear layer, and the mean and scale of the 24 millivolt features fit on folds 1–8. The encoder file is not rewritten.

```bash
python -m ecg_jepa.predict --data-dir dataset --record records100/00000/00001_lr
```

The input is one recording in millivolts, leads I, II, V1–V6. It is cropped the same way as training. The encoder still sees each lead z-scored. The amplitude branch sees peak, standard deviation, and peak-to-peak, scaled with the statistics stored in the head. The printout is five sigmoid scores: `NORM`, `MI`, `STTC`, `CD`, `HYP`.

A positive call is a score of at least 0.5 on that class. The threshold is fixed. Fold 9 still chooses the epoch. Fold-10 per-label accuracy and F1, five seeds:

| | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- |
| Accuracy | 0.823 ± 0.009 | 0.803 ± 0.003 | 0.853 ± 0.013 | 0.842 ± 0.002 | 0.905 ± 0.002 |
| F1 | 0.809 ± 0.010 | 0.496 ± 0.008 | 0.643 ± 0.041 | 0.559 ± 0.010 | 0.456 ± 0.009 |

Those are separate scores for each label. A recording counts as correct only when all five calls match. That exact-match accuracy on fold 10 is 0.489 ± 0.011. The five per-label accuracies average to about 0.845, which is the share of individual calls that are correct. One wrong call still leaves the other four correct, so that average stays high.

Precision is the share of positive calls that are truly that condition. Recall is the share of true cases the head calls:

| | NORM | MI | STTC | CD | HYP |
| --- | --- | --- | --- | --- | --- |
| Precision | 0.765 ± 0.008 | 0.692 ± 0.011 | 0.755 ± 0.022 | 0.756 ± 0.004 | 0.714 ± 0.015 |
| Recall | 0.859 ± 0.013 | 0.386 ± 0.008 | 0.562 ± 0.052 | 0.444 ± 0.014 | 0.335 ± 0.008 |

Hypertrophy is the weak identification. Fold 10 has 262 HYP recordings out of 2198, so never calling HYP is already 0.881 accurate, and the head's HYP accuracy is 0.905 ± 0.002. It calls about one third of those 262 recordings (recall 0.335 ± 0.008). When it does call HYP, 0.714 ± 0.015 of those calls are hypertrophy. On the 56 hypertrophy-only recordings, the full five-label call is right for 0.096 ± 0.037. Epoch selection stays macro AUC.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Deviations from the papers

- The default tokens are temporal: all eight leads sit inside one patch. Per-lead tokens and multi-block masks were each compared across five seeds. Neither cleared a 0.015 fold-10 gap. CroPA (Kim, 2024) stays off.
- No waveform augmentations. The JEPA objective does not use them.
- Tiny encoder, 100 Hz, PTB-XL folds 1–8 only. Learning rate is `1e-3` and drop-path is off. The ViT-B figures in Kim (2024) and Weimann and Conrad (2025) (`2.5e-5`, 100 epochs, extra pretraining corpora, AUC around 0.89–0.94) are not a target for this run.
- A fine-tune of the temporal encoder is reported above. It uses learning rate `1e-4` for 20 epochs, which is a small-model guess, not the ViT-B rate in those papers.
- Target embeddings are not LayerNormed before the loss. That I-JEPA variant (Assran et al., 2023) is commented in `ecg_jepa/models/jepa.py` and should be turned on only if `embed_std` collapses.

## References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 15619–15629. [doi:10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499).

Kim, S. (2024). Learning general representation of 12-lead ECG with a joint-embedding predictive architecture. arXiv:2410.08559. [doi:10.48550/arXiv.2410.08559](https://doi.org/10.48550/arXiv.2410.08559).

Strodthoff, N., Wagner, P., Schaeffter, T., and Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*, 25(5):1519–1528. [doi:10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989).

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., and Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7:154. [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).

Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., and Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. [doi:10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).

Weimann, K. and Conrad, T. O. F. (2025). Self-supervised pre-training with joint-embedding predictive architecture boosts ECG classification performance. *Computers in Biology and Medicine*, 196:110809. [doi:10.1016/j.compbiomed.2025.110809](https://doi.org/10.1016/j.compbiomed.2025.110809).

```bibtex
@inproceedings{assran2023ijepa,
  author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {15619--15629},
  doi       = {10.1109/CVPR52729.2023.01499}
}

@article{kim2024ecgjepa,
  author  = {Kim, Sehun},
  title   = {Learning General Representation of 12-Lead {ECG} with a Joint-Embedding Predictive Architecture},
  journal = {arXiv preprint arXiv:2410.08559},
  year    = {2024},
  doi     = {10.48550/arXiv.2410.08559}
}

@article{strodthoff2021ptbxl,
  author  = {Strodthoff, Nils and Wagner, Patrick and Schaeffter, Tobias and Samek, Wojciech},
  title   = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB-XL}},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2021},
  volume  = {25},
  number  = {5},
  pages   = {1519--1528},
  doi     = {10.1109/JBHI.2020.3022989}
}

@article{wagner2020ptbxl,
  author  = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I. and Samek, Wojciech and Schaeffter, Tobias},
  title   = {{PTB-XL}, a Large Publicly Available Electrocardiography Dataset},
  journal = {Scientific Data},
  year    = {2020},
  volume  = {7},
  pages   = {154},
  doi     = {10.1038/s41597-020-0495-6}
}

@misc{wagner2022ptbxl,
  author = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Samek, Wojciech and Schaeffter, Tobias},
  title  = {{PTB-XL}, a Large Publicly Available Electrocardiography Dataset},
  year   = {2022},
  note   = {PhysioNet, version 1.0.3},
  doi    = {10.13026/kfzx-aw45}
}

@article{weimann2025jepa,
  author  = {Weimann, Kuba and Conrad, Tim O. F.},
  title   = {Self-Supervised Pre-Training with Joint-Embedding Predictive Architecture Boosts {ECG} Classification Performance},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  volume  = {196},
  pages   = {110809},
  doi     = {10.1016/j.compbiomed.2025.110809}
}
```
