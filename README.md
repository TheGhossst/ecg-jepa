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

Per-lead tokens patch each lead, drop a masked time before any lead can see it, then mix the eight leads. The mean gain over temporal tokens is about 0.005, inside the seed-to-seed spread.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Deviations from the papers

- The default tokens are temporal: all eight leads sit inside one patch. Per-lead tokens were compared across the same five seeds and did not clear the seed noise. Multi-block masks and CroPA are still open (see [TODO.md](TODO.md)).
- No waveform augmentations. The JEPA objective does not use them.
- Tiny encoder, 100 Hz, PTB-XL folds 1–8 only. Learning rate is `1e-3` and drop-path is off. The paper figures (ViT-B, `2.5e-5`, 100 epochs, extra pretraining corpora, AUC around 0.89–0.94) are not a target for this run.
- The linear probe is a frozen mean-pool plus one linear layer. There is no fine-tune yet.
- Target embeddings are not LayerNormed before the loss. That I-JEPA variant is commented in `ecg_jepa/models/jepa.py` and should be turned on only if `embed_std` collapses.
