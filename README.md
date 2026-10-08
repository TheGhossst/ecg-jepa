# ecg-jepa

Learn useful ECG representations without labels. A context encoder sees the visible part of a recording and predicts the embedding of the masked time segments. The targets come from a second encoder, an exponential moving average of the first, which sees the full recording and receives no gradient. The loss is in embedding space, so the model is not asked to reconstruct waveform noise.

This repository is a small, testable core on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/). It is not a port of Kim et al. (arXiv:2410.08559) or Weimann and Conrad (arXiv:2410.13867).

Planned work is tracked in [TODO.md](TODO.md).

## Model

Each recording is 10 seconds from `records100` (100 Hz, 1000 samples). The eight independent leads `I, II, V1–V6` are kept. `III, aVR, aVL, aVF` are linear combinations of `I` and `II` and are dropped.

A stride-25 convolution turns the signal into 40 tokens. Each token is 250 ms across all eight leads. One mask ratio in `(0.6, 0.7)` is drawn per step and shared by the batch. The context encoder (depth 4, dim 128) sees only the visible tokens. The predictor (depth 2, dim 64) reads those embeddings plus a learned mask token at each hidden time, and smooth-L1 matches the target embeddings at the masked positions.

Pretraining uses official folds 1–8 and ignores diagnostic labels. Fold 9 is for loss and embedding-std monitoring. Fold 10 is not loaded.

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

Freeze the target encoder, mean-pool its tokens, and train a linear head on the five diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`). The head trains on folds 1–8, the checkpoint is chosen by macro AUC on fold 9, and the reported number is fold 10. The same head is trained on a randomly initialized encoder.

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

Unfreezing the encoder adds about 0.05 on every seed. Starting from the pretrained weights adds 0.003 ± 0.004 over training from scratch, and one of the five seeds goes the other way.

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

A new pretraining method is not justified by these four checks. Supervised training from scratch matches JEPA fine-tuning. Subclass information is already in the frozen embedding, including in directions the five-class head does not use. Isolated hypertrophy is barely above chance, and training from scratch does not solve it either.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Deviations from the papers

- The default tokens are temporal: all eight leads sit inside one patch. Per-lead tokens and multi-block masks were each compared across five seeds. Neither cleared a 0.015 fold-10 gap. CroPA is still open (see [TODO.md](TODO.md)).
- No waveform augmentations. The JEPA objective does not use them.
- Tiny encoder, 100 Hz, PTB-XL folds 1–8 only. Learning rate is `1e-3` and drop-path is off. The paper figures (ViT-B, `2.5e-5`, 100 epochs, extra pretraining corpora, AUC around 0.89–0.94) are not a target for this run.
- A fine-tune of the temporal encoder is reported above. It uses learning rate `1e-4` for 20 epochs, which is a small-model guess, not the paper's ViT-B rate.
- Target embeddings are not LayerNormed before the loss. That I-JEPA variant is commented in `ecg_jepa/models/jepa.py` and should be turned on only if `embed_std` collapses.
