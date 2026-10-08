"""Fine-tune the temporal encoder, and train the same encoder from scratch.

Both runs use folds 1–8, pick the epoch by macro AUC on fold 9, and score
fold 10 once. The frozen linear probe is the third number, already measured.
Hyperparameters below were fixed before any fold-10 score from this script.

Learning rate 1e-4 is a guess for this small encoder. The paper's fine-tune
rate is scaled for a ViT-B and is not used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_jepa.config import Config
from ecg_jepa.data.ptbxl import PTBXLLabeledDataset, SPLIT_FOLDS
from ecg_jepa.models.transformer import Encoder
from ecg_jepa.probe import fresh_target_encoder, load_target_encoder, macro_auc
from ecg_jepa.robustness import format_mean_std
from ecg_jepa.train import learning_rate, set_seed

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1.0


class EncoderClassifier(nn.Module):
    """Mean-pool every time token, then a linear layer. The encoder is trained."""

    def __init__(self, encoder: Encoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        width = encoder.norm.normalized_shape[0]
        self.head = nn.Linear(width, n_classes)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(signals)
        return self.head(tokens.mean(dim=1))


def unfreeze(encoder: Encoder) -> Encoder:
    for parameter in encoder.parameters():
        parameter.requires_grad = True
    encoder.train()
    return encoder


def cache_split(data_dir: str, split: str, signal_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = PTBXLLabeledDataset(data_dir, SPLIT_FOLDS[split], signal_length)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    signals: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_signals, batch_labels in loader:
        signals.append(batch_signals)
        labels.append(batch_labels)
    return torch.cat(signals), torch.cat(labels)


def temporal_checkpoint(seed: int) -> Path:
    if seed == 0:
        return Path("checkpoints/run1/last.pt")
    return Path("checkpoints/temporal") / f"seed_{seed}" / "last.pt"


@torch.no_grad()
def score_split(
    model: EncoderClassifier,
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, signals.shape[0], BATCH_SIZE):
        batch = signals[start : start + BATCH_SIZE].to(device, non_blocking=True)
        chunks.append(torch.sigmoid(model(batch)).cpu())
    scores = torch.cat(chunks).numpy()
    return macro_auc(labels.numpy(), scores)


def finetune_split(
    encoder: Encoder,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int = EPOCHS,
) -> dict:
    """Train encoder and head. Fold 10 is scored only after the fold-9 choice."""
    unfreeze(encoder)
    set_seed(10_000 + seed)
    model = EncoderClassifier(encoder, train_y.shape[1]).to(device)
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)
    best_val = -1.0
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    val_curve: list[float] = []
    step = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for signals, target in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, warmup_steps, LEARNING_RATE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals.to(device, non_blocking=True))
            loss = loss_fn(logits, target.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            running += loss.item() * signals.shape[0]
            seen += signals.shape[0]
            step += 1
        val_auc, _ = score_split(model, val_x, val_y, device)
        val_curve.append(val_auc)
        print(
            f"finetune epoch {epoch} loss {running / max(seen, 1):.4f} val_auc {val_auc:.4f}",
            flush=True,
        )
        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_auc, per_class = score_split(model, test_x, test_y, device)
    if best_epoch != int(max(range(len(val_curve)), key=val_curve.__getitem__)):
        raise RuntimeError("selected epoch is not the best fold-9 score")
    return {
        "test_auc": float(test_auc),
        "val_auc": float(best_val),
        "best_epoch": best_epoch,
        "per_class": {key: float(value) for key, value in per_class.items()},
        "val_curve": val_curve,
    }


def results_path() -> Path:
    return Path("logs/finetune_temporal.json")


def load_rows() -> list[dict]:
    path = results_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["seeds"]


def save_rows(rows: list[dict]) -> None:
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": len(rows),
        "pretrained": format_mean_std([row["pretrained_auc"] for row in rows]) if rows else "",
        "scratch": format_mean_std([row["scratch_auc"] for row in rows]) if rows else "",
        "delta": format_mean_std([row["delta"] for row in rows]) if rows else "",
    }
    path.write_text(
        json.dumps({"seeds": rows, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    if rows:
        print(
            f"finetune n={summary['n']} pretrained {summary['pretrained']} "
            f"scratch {summary['scratch']} delta {summary['delta']}",
            flush=True,
        )


def run_finetune(data_dir: str, seeds: list[int], device: str | None = None) -> None:
    missing = [seed for seed in seeds if not temporal_checkpoint(seed).is_file()]
    if missing:
        raise FileNotFoundError(f"missing temporal checkpoints for seeds {missing}")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, cfg = load_target_encoder(temporal_checkpoint(seeds[0]), torch.device("cpu"))
    signal_length = cfg.signal_length
    print("caching labeled recordings", flush=True)
    cached = {
        split: cache_split(data_dir, split, signal_length) for split in ("train", "val", "test")
    }
    done = {int(row["seed"]): row for row in load_rows()}
    ordered: list[dict] = []
    for seed in seeds:
        if seed in done:
            print(f"reusing finetune seed {seed}", flush=True)
            ordered.append(done[seed])
            continue
        ckpt = temporal_checkpoint(seed)
        print(f"finetune pretrained seed {seed} from {ckpt}", flush=True)
        encoder, cfg = load_target_encoder(ckpt, torch_device)
        pretrained = finetune_split(encoder, *cached["train"], *cached["val"], *cached["test"], seed, torch_device)
        print(
            f"pretrained seed {seed} fold9 {pretrained['val_auc']:.4f} "
            f"fold10 {pretrained['test_auc']:.4f} epoch {pretrained['best_epoch']}",
            flush=True,
        )
        print(f"finetune scratch seed {seed}", flush=True)
        scratch_encoder = fresh_target_encoder(cfg, torch_device, seed)
        scratch = finetune_split(
            scratch_encoder, *cached["train"], *cached["val"], *cached["test"], seed, torch_device
        )
        print(
            f"scratch seed {seed} fold9 {scratch['val_auc']:.4f} "
            f"fold10 {scratch['test_auc']:.4f} epoch {scratch['best_epoch']}",
            flush=True,
        )
        row = {
            "seed": seed,
            "ckpt": str(ckpt),
            "pretrained_auc": pretrained["test_auc"],
            "scratch_auc": scratch["test_auc"],
            "delta": pretrained["test_auc"] - scratch["test_auc"],
            "pretrained_val_auc": pretrained["val_auc"],
            "scratch_val_auc": scratch["val_auc"],
            "pretrained_best_epoch": pretrained["best_epoch"],
            "scratch_best_epoch": scratch["best_epoch"],
            "pretrained_per_class": pretrained["per_class"],
            "scratch_per_class": scratch["per_class"],
        }
        done[seed] = row
        ordered = [done[item] for item in seeds if item in done]
        save_rows(ordered)
    save_rows(ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune temporal ECG-JEPA versus training from scratch")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run_finetune(args.data_dir, seeds, args.device)


if __name__ == "__main__":
    main()
