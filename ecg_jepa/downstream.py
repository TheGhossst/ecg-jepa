"""Four checks on the locked temporal JEPA encoder.

1. Per-class table from the existing five-seed probe and fine-tune.
2. Frozen linear probe on the 44 diagnostic statements.
3. Whether MI and HYP errors track superclass overlap.
4. Whether embeddings still predict diagnostic subclasses after the five
   superclass directions are removed.

Fold 9 chooses each head. Fold 10 is scored once. Statement and subclass
learning rates match the frozen superclass probe and were not tuned on fold 10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_jepa.data.ptbxl import (
    PTBXLLabeledDataset,
    SPLIT_FOLDS,
    SUPERCLASSES,
    diagnostic_label_names,
    encode_named_labels,
    resolve_ptbxl_root,
)
from ecg_jepa.finetune import cache_split, temporal_checkpoint
from ecg_jepa.probe import fresh_target_encoder, load_target_encoder, roc_auc
from ecg_jepa.robustness import format_mean_std
from ecg_jepa.train import learning_rate, set_seed

HEAD_EPOCHS = 20
HEAD_BATCH = 256
HEAD_LR = 1e-3
HEAD_WD = 0.05


def column_macro_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, int]:
    values: list[float] = []
    for index in range(labels.shape[1]):
        value = roc_auc(labels[:, index], scores[:, index])
        if value == value:
            values.append(value)
    if not values:
        return float("nan"), 0
    return float(sum(values) / len(values)), len(values)


def trainable_columns(labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum(dim=0)
    keep = (positive > 0) & (positive < labels.shape[0])
    return keep.nonzero(as_tuple=False).flatten()


def remove_head_directions(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Drop the row space of a linear head. `weight` is (classes, dim)."""
    basis, _ = torch.linalg.qr(weight.detach().T.cpu().float())
    projected = features.float() @ basis @ basis.T
    return features.float() - projected


def positive_subset_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    positive_mask: np.ndarray,
) -> tuple[float, int]:
    """AUC of one subset of positives against all negatives of that class."""
    keep = (labels == 0) | positive_mask
    return roc_auc(labels[keep], scores[keep]), int(positive_mask.sum())


class MLPHead(nn.Module):
    def __init__(self, width: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def fit_linear(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device: torch.device,
    kind: str = "linear",
) -> dict:
    columns = trainable_columns(train_y)
    train_y = train_y[:, columns]
    val_y = val_y[:, columns]
    test_y = test_y[:, columns]
    set_seed(seed)
    if kind == "linear":
        module: nn.Module = nn.Linear(train_x.shape[1], train_y.shape[1])
    elif kind == "mlp":
        module = MLPHead(train_x.shape[1], train_y.shape[1])
    else:
        raise ValueError(kind)
    module = module.to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=HEAD_BATCH, shuffle=True)
    steps_per_epoch = max(1, len(loader))
    total_steps = HEAD_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch
    train_x = train_x.to(device)
    val_x = val_x.to(device)
    test_x = test_x.to(device)
    best_val = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    step = 0
    for _epoch in range(HEAD_EPOCHS):
        module.train()
        for features, target in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, warmup_steps, HEAD_LR)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(module(features.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            step += 1
        module.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(module(val_x)).cpu().numpy()
        val_auc, _ = column_macro_auc(val_y.numpy(), val_scores)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(module(test_x)).cpu().numpy()
    test_auc, n_labels = column_macro_auc(test_y.numpy(), test_scores)
    weight = None
    if isinstance(module, nn.Linear):
        weight = module.weight.detach().cpu().clone()
    return {
        "val_auc": float(best_val),
        "test_auc": float(test_auc),
        "n_labels": n_labels,
        "scores": test_scores,
        "weight": weight,
        "columns": columns.cpu(),
    }


def cache_targets(data_dir: str, signal_length: int) -> dict[str, dict[str, torch.Tensor]]:
    root = resolve_ptbxl_root(data_dir)
    statements, subclasses, code_to_subclass = diagnostic_label_names(root)
    cached: dict[str, dict[str, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        dataset = PTBXLLabeledDataset(data_dir, SPLIT_FOLDS[split], signal_length)
        raw_codes = dataset.frame["scp_codes"].tolist()
        statement_y = np.stack([encode_named_labels(raw, statements) for raw in raw_codes])
        subclass_y = np.stack(
            [encode_named_labels(raw, subclasses, code_to_subclass) for raw in raw_codes]
        )
        signals, super_y = cache_split(data_dir, split, signal_length)
        if len(signals) != len(statement_y):
            raise RuntimeError(f"{split} signal count does not match label count")
        cached[split] = {
            "signals": signals,
            "super": super_y,
            "statement": torch.from_numpy(statement_y),
            "subclass": torch.from_numpy(subclass_y),
        }
    cached["names"] = {
        "statements": statements,
        "subclasses": subclasses,
    }
    return cached


@torch.no_grad()
def embed_signals(encoder: nn.Module, signals: torch.Tensor, device: torch.device) -> torch.Tensor:
    encoder.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, signals.shape[0], 64):
        batch = signals[start : start + 64].to(device, non_blocking=True)
        chunks.append(encoder(batch).mean(dim=1).cpu())
    return torch.cat(chunks)


def overlap_report(labels: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, name in enumerate(SUPERCLASSES):
        column = labels[:, index]
        others = np.delete(labels, index, axis=1).sum(axis=1) > 0
        pure = (column == 1) & ~others
        mixed = (column == 1) & others
        pure_auc, n_pure = positive_subset_auc(column, scores[:, index], pure)
        mixed_auc, n_mixed = positive_subset_auc(column, scores[:, index], mixed)
        report[name] = {
            "pure_auc": pure_auc,
            "mixed_auc": mixed_auc,
            "n_pure": n_pure,
            "n_mixed": n_mixed,
        }
    return report


def cooccurrence(labels: np.ndarray) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for index, name in enumerate(SUPERCLASSES):
        group = labels[labels[:, index] == 1]
        table[name] = {
            other: float(group[:, other_index].mean()) if len(group) else float("nan")
            for other_index, other in enumerate(SUPERCLASSES)
        }
        table[name]["n"] = float(len(group))
    return table


def per_class_stats(rows: list[dict], field: str) -> dict[str, str]:
    return {
        name: format_mean_std([row[field][name] for row in rows])
        for name in SUPERCLASSES
    }


def lock_baselines() -> dict:
    temporal = json.loads(Path("logs/robustness_temporal.json").read_text(encoding="utf-8"))["seeds"]
    finetune = json.loads(Path("logs/finetune_temporal.json").read_text(encoding="utf-8"))["seeds"]
    frozen = [row["pretrained_auc"] for row in temporal]
    random_encoder = [row["random_auc"] for row in temporal]
    tuned = [row["pretrained_auc"] for row in finetune]
    scratch = [row["scratch_auc"] for row in finetune]
    paired_frozen = [tuned[index] - frozen[index] for index in range(len(frozen))]
    paired_scratch = [row["delta"] for row in finetune]
    payload = {
        "macro": {
            "frozen_jepa": format_mean_std(frozen),
            "frozen_random": format_mean_std(random_encoder),
            "finetune_jepa": format_mean_std(tuned),
            "finetune_scratch": format_mean_std(scratch),
        },
        "per_class": {
            "frozen_jepa": per_class_stats(temporal, "pretrained_per_class"),
            "frozen_random": per_class_stats(temporal, "random_per_class"),
            "finetune_jepa": per_class_stats(finetune, "pretrained_per_class"),
            "finetune_scratch": per_class_stats(finetune, "scratch_per_class"),
        },
        "paired_finetune_minus_frozen": [round(value, 4) for value in paired_frozen],
        "paired_finetune_minus_scratch": [round(value, 4) for value in paired_scratch],
        "paired_finetune_minus_frozen_mean": format_mean_std(paired_frozen),
        "paired_finetune_minus_scratch_mean": format_mean_std(paired_scratch),
    }
    path = Path("logs/baselines_per_class.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("locked baselines", json.dumps(payload["macro"]), flush=True)
    for name in SUPERCLASSES:
        print(
            f"{name} frozen {payload['per_class']['frozen_jepa'][name]} "
            f"random {payload['per_class']['frozen_random'][name]} "
            f"finetune {payload['per_class']['finetune_jepa'][name]} "
            f"scratch {payload['per_class']['finetune_scratch'][name]}",
            flush=True,
        )
    return payload


def evaluate_encoder(
    features: dict[str, torch.Tensor],
    targets: dict[str, dict[str, torch.Tensor]],
    seed: int,
    device: torch.device,
) -> dict:
    linear = fit_linear(
        features["train"], targets["train"]["super"],
        features["val"], targets["val"]["super"],
        features["test"], targets["test"]["super"],
        seed, device, "linear",
    )
    mlp = fit_linear(
        features["train"], targets["train"]["super"],
        features["val"], targets["val"]["super"],
        features["test"], targets["test"]["super"],
        seed, device, "mlp",
    )
    statements = fit_linear(
        features["train"], targets["train"]["statement"],
        features["val"], targets["val"]["statement"],
        features["test"], targets["test"]["statement"],
        seed, device, "linear",
    )
    subclasses = fit_linear(
        features["train"], targets["train"]["subclass"],
        features["val"], targets["val"]["subclass"],
        features["test"], targets["test"]["subclass"],
        seed, device, "linear",
    )
    residual = {
        split: remove_head_directions(features[split], linear["weight"])
        for split in ("train", "val", "test")
    }
    residual_sub = fit_linear(
        residual["train"], targets["train"]["subclass"],
        residual["val"], targets["val"]["subclass"],
        residual["test"], targets["test"]["subclass"],
        seed, device, "linear",
    )
    overlap = overlap_report(targets["test"]["super"].numpy(), linear["scores"])
    return {
        "linear_auc": linear["test_auc"],
        "mlp_auc": mlp["test_auc"],
        "statement_auc": statements["test_auc"],
        "statement_n": statements["n_labels"],
        "subclass_auc": subclasses["test_auc"],
        "subclass_n": subclasses["n_labels"],
        "residual_subclass_auc": residual_sub["test_auc"],
        "residual_subclass_n": residual_sub["n_labels"],
        "overlap": overlap,
    }


def results_path() -> Path:
    return Path("logs/downstream.json")


def load_rows() -> list[dict]:
    path = results_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["seeds"]


def save_rows(rows: list[dict], extra: dict) -> None:
    path = results_path()
    path.write_text(json.dumps({"seeds": rows, **extra}, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict]) -> dict[str, str]:
    summary = {}
    for source in ("jepa", "random"):
        for key in ("linear_auc", "mlp_auc", "statement_auc", "subclass_auc", "residual_subclass_auc"):
            summary[f"{source}_{key}"] = format_mean_std([row[source][key] for row in rows])
    for name in ("MI", "HYP"):
        for kind in ("pure_auc", "mixed_auc"):
            summary[f"jepa_{name}_{kind}"] = format_mean_std(
                [row["jepa"]["overlap"][name][kind] for row in rows]
            )
    return summary


def run_downstream(data_dir: str, seeds: list[int], device: str | None = None) -> None:
    baselines = lock_baselines()
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, cfg = load_target_encoder(temporal_checkpoint(seeds[0]), torch.device("cpu"))
    print("caching recordings and multilabel targets", flush=True)
    targets = cache_targets(data_dir, cfg.signal_length)
    names = targets.pop("names")
    done = {int(row["seed"]): row for row in load_rows()}
    ordered: list[dict] = []
    for seed in seeds:
        if seed in done:
            print(f"reusing downstream seed {seed}", flush=True)
            ordered.append(done[seed])
            continue
        print(f"downstream seed {seed}", flush=True)
        encoder, _ = load_target_encoder(temporal_checkpoint(seed), torch_device)
        pretrained = {
            split: embed_signals(encoder, targets[split]["signals"], torch_device)
            for split in ("train", "val", "test")
        }
        random_encoder = fresh_target_encoder(cfg, torch_device, seed + 1)
        random_features = {
            split: embed_signals(random_encoder, targets[split]["signals"], torch_device)
            for split in ("train", "val", "test")
        }
        row = {
            "seed": seed,
            "jepa": evaluate_encoder(pretrained, targets, seed, torch_device),
            "random": evaluate_encoder(random_features, targets, seed, torch_device),
        }
        print(
            f"seed {seed} statements {row['jepa']['statement_auc']:.4f} "
            f"subclass {row['jepa']['subclass_auc']:.4f} "
            f"residual {row['jepa']['residual_subclass_auc']:.4f} "
            f"mlp {row['jepa']['mlp_auc']:.4f} linear {row['jepa']['linear_auc']:.4f}",
            flush=True,
        )
        done[seed] = row
        ordered = [done[item] for item in seeds if item in done]
        save_rows(
            ordered,
            {
                "baselines": baselines,
                "cooccurrence_fold10": cooccurrence(targets["test"]["super"].numpy()),
                "label_names": names,
                "summary": summarize_rows(ordered),
            },
        )
    print(json.dumps(summarize_rows(ordered), indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock baselines and run the three frozen-encoder checks")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run_downstream(args.data_dir, seeds, args.device)


if __name__ == "__main__":
    main()
