"""Frozen linear probe on the five PTB-XL diagnostic superclasses.

The target encoder is frozen and mean-pooled. A linear head is trained on
folds 1–8, chosen by macro AUC on fold 9, and scored on fold 10. The same
protocol is run on a randomly initialized encoder. The comparison is that
pair, not a published AUC.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_jepa.config import Config
from ecg_jepa.data.ptbxl import PTBXLLabeledDataset, SPLIT_FOLDS, SUPERCLASSES
from ecg_jepa.models.jepa import JEPA
from ecg_jepa.models.transformer import Encoder
from ecg_jepa.train import learning_rate, set_seed


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with average ranks for tied scores. Returns nan if one class is missing."""
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and scores[stop + 1] == scores[start]:
            stop += 1
        ranks[start : stop + 1] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class = {
        name: roc_auc(labels[:, i], scores[:, i]) for i, name in enumerate(SUPERCLASSES)
    }
    values = [value for value in per_class.values() if value == value]
    if not values:
        return float("nan"), per_class
    return float(sum(values) / len(values)), per_class


def config_from_checkpoint(blob: dict) -> Config:
    known = {item.name for item in fields(Config)}
    return Config(**{key: value for key, value in blob["config"].items() if key in known})


def load_target_encoder(ckpt: Path, device: torch.device) -> tuple[Encoder, Config]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(blob)
    model = JEPA(cfg)
    model.target_encoder.load_state_dict(blob["target_encoder"])
    encoder = model.target_encoder.to(device)
    encoder.eval()
    return encoder, cfg


def fresh_target_encoder(cfg: Config, device: torch.device, seed: int) -> Encoder:
    set_seed(seed)
    encoder = JEPA(cfg).target_encoder.to(device)
    encoder.eval()
    return encoder


def labeled_loader(
    data_dir: str,
    split: str,
    cfg: Config,
    batch_size: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = PTBXLLabeledDataset(data_dir, SPLIT_FOLDS[split], cfg.signal_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def extract_features(
    encoder: Encoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for signals, target in loader:
        tokens = encoder(signals.to(device, non_blocking=True))
        features.append(tokens.mean(dim=1).cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


def train_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, float | dict[str, float]]:
    set_seed(seed)
    head = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    train_x = train_x.to(device)
    val_x = val_x.to(device)
    test_x = test_x.to(device)
    best_val = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    step = 0
    for epoch in range(epochs):
        head.train()
        for features, target in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, warmup_steps, lr)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            step += 1
        head.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(head(val_x)).cpu().numpy()
        val_auc, _ = macro_auc(val_y.numpy(), val_scores)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        print(f"probe epoch {epoch} val_auc {val_auc:.4f}", flush=True)
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(head(test_x)).cpu().numpy()
    test_auc, per_class = macro_auc(test_y.numpy(), test_scores)
    return {"val_auc": best_val, "test_auc": test_auc, "per_class": per_class}


def format_result(name: str, result: dict) -> str:
    per_class = " ".join(
        f"{label} {result['per_class'][label]:.3f}" for label in SUPERCLASSES
    )
    return (
        f"{name} fold9_auc {result['val_auc']:.4f} "
        f"fold10_auc {result['test_auc']:.4f} {per_class}"
    )


def run_probe(
    data_dir: str,
    ckpt: str,
    epochs: int = 20,
    batch_size: int = 64,
    head_batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    seed: int = 0,
    device: str | None = None,
) -> str:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = torch_device.type == "cuda"
    encoder, cfg = load_target_encoder(Path(ckpt), torch_device)
    loaders = {
        split: labeled_loader(data_dir, split, cfg, batch_size, pin_memory)
        for split in ("train", "val", "test")
    }
    print("extracting pretrained features", flush=True)
    cached = {
        split: extract_features(encoder, loaders[split], torch_device) for split in loaders
    }
    pretrained = train_head(
        *cached["train"],
        *cached["val"],
        *cached["test"],
        epochs=epochs,
        batch_size=head_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=torch_device,
    )
    print(format_result("pretrained", pretrained), flush=True)

    print("extracting random-encoder features", flush=True)
    random_encoder = fresh_target_encoder(cfg, torch_device, seed + 1)
    random_cached = {
        split: extract_features(random_encoder, loaders[split], torch_device)
        for split in loaders
    }
    baseline = train_head(
        *random_cached["train"],
        *random_cached["val"],
        *random_cached["test"],
        epochs=epochs,
        batch_size=head_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=torch_device,
    )
    print(format_result("random", baseline), flush=True)
    summary = (
        f"{format_result('pretrained', pretrained)}\n"
        f"{format_result('random', baseline)}\n"
        f"delta_fold10 {pretrained['test_auc'] - baseline['test_auc']:+.4f}"
    )
    print(summary, flush=True)
    return {
        "pretrained_auc": float(pretrained["test_auc"]),
        "random_auc": float(baseline["test_auc"]),
        "delta": float(pretrained["test_auc"] - baseline["test_auc"]),
        "pretrained_per_class": {key: float(value) for key, value in pretrained["per_class"].items()},
        "random_per_class": {key: float(value) for key, value in baseline["per_class"].items()},
        "pretrained_val_auc": float(pretrained["val_auc"]),
        "random_val_auc": float(baseline["val_auc"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe on frozen ECG-JEPA features")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--ckpt", default="checkpoints/run1/last.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_probe(
        data_dir=args.data_dir,
        ckpt=args.ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
