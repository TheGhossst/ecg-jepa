"""1% and 10% of folds 1–8, on the frozen temporal checkpoints.

The full-label result trains on every recording in folds 1–8. This script
does not pretrain again. It reuses those checkpoints and repeats two
procedures that were already fixed: the frozen linear head from `probe.py`,
and the fine-tune from `finetune.py`. Fold 9 stays whole and still picks the
epoch. Fold 10 is scored once.

Each percent is drawn once per seed and then shared. The from-scratch run
sees those same recordings. A comparison wins only when the mean paired
fold-10 delta is larger than the sample standard deviation of that delta.
That is the same bar as the full-label fine-tune, whose paired deltas were
+0.006, +0.001, −0.002, +0.007, +0.003 and did not clear the spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecg_jepa.data.ptbxl import SUPERCLASSES
from ecg_jepa.downstream import embed_signals
from ecg_jepa.finetune import EPOCHS, cache_split, finetune_split, temporal_checkpoint
from ecg_jepa.probe import fresh_target_encoder, load_target_encoder, train_head
from ecg_jepa.robustness import format_mean_std, sample_mean_std

PERCENTS = (1, 10)
HEAD_EPOCHS = 20
HEAD_BATCH = 256
HEAD_LR = 1e-3
HEAD_WD = 0.05
CACHE_PATH = Path("checkpoints/labeled_splits.pt")


def budget_count(n: int, percent: int) -> int:
    """Largest whole share of `n` that does not exceed `percent` percent."""
    if percent < 1 or percent > 100:
        raise ValueError(f"percent must be from 1 to 100, got {percent}")
    return max(1, (n * percent) // 100)


def subset_seed(seed: int, percent: int) -> int:
    """Distinct stream per seed and percent. Independent of the global RNG."""
    return seed + 1_000_003 * percent


def budget_indices(n: int, percent: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(subset_seed(seed, percent))
    picked = rng.choice(n, size=budget_count(n, percent), replace=False)
    return np.sort(picked.astype(np.int64))


def clears_seed_spread(deltas: list[float]) -> bool:
    """True when the mean paired gap is larger than the seed-to-seed spread."""
    if len(deltas) < 2:
        return False
    mean, std = sample_mean_std(deltas)
    return mean > std


def results_path() -> Path:
    return Path("logs/low_label.json")


def load_rows() -> list[dict]:
    path = results_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def row_key(row: dict) -> tuple[int, int]:
    return int(row["seed"]), int(row["percent"])


def ordered_rows(done: dict[tuple[int, int], dict], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        for percent in PERCENTS:
            key = (seed, percent)
            if key in done:
                rows.append(done[key])
    return rows


def _mean_line(rows: list[dict], field: str) -> dict[str, object]:
    values = [float(row[field]) for row in rows]
    return {
        "mean_std": format_mean_std(values),
        "paired": [round(value, 4) for value in values],
        "clears_seed_spread": clears_seed_spread(values),
    }


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for percent in PERCENTS:
        group = [row for row in rows if int(row["percent"]) == percent]
        if len(group) < 2:
            continue
        finetune = _mean_line(group, "finetune_delta")
        linear_vs_scratch = _mean_line(group, "linear_minus_scratch")
        summary[str(percent)] = {
            "n": len(group),
            "n_train": group[0]["n_train"],
            "linear_jepa": format_mean_std([row["linear_pretrained_auc"] for row in group]),
            "linear_random": format_mean_std([row["linear_random_auc"] for row in group]),
            "linear_delta": _mean_line(group, "linear_delta"),
            "finetune_jepa": format_mean_std([row["finetune_pretrained_auc"] for row in group]),
            "finetune_scratch": format_mean_std([row["finetune_scratch_auc"] for row in group]),
            "finetune_delta": finetune,
            "linear_minus_scratch": linear_vs_scratch,
            "jepa_wins": bool(finetune["clears_seed_spread"] or linear_vs_scratch["clears_seed_spread"]),
        }
    return summary


def save_rows(rows: list[dict]) -> None:
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "percents": list(PERCENTS),
            "folds_1_8": "subsampled",
            "fold_9": "full, chooses the epoch",
            "fold_10": "scored once",
            "checkpoints": "frozen temporal",
            "linear_head": "probe.py hyperparameters, not retuned",
            "finetune": "finetune.py hyperparameters, not retuned",
            "subset": "floor(n * percent / 100), Generator(seed + 1000003 * percent), shared",
            "win": "mean paired fold-10 delta greater than its sample standard deviation",
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for percent, block in payload["summary"].items():
        print(
            f"label {percent}% n={block['n']} "
            f"linear jepa {block['linear_jepa']} random {block['linear_random']} "
            f"finetune jepa {block['finetune_jepa']} scratch {block['finetune_scratch']} "
            f"finetune_wins {block['finetune_delta']['clears_seed_spread']} "
            f"linear_vs_scratch_wins {block['linear_minus_scratch']['clears_seed_spread']}",
            flush=True,
        )


def load_splits(data_dir: str, signal_length: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if CACHE_PATH.is_file():
        blob = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        if blob.get("signal_length") == signal_length and blob.get("version") == 1:
            print(f"reusing {CACHE_PATH}", flush=True)
            return blob["splits"]
    print("caching labeled recordings", flush=True)
    splits = {
        split: cache_split(data_dir, split, signal_length) for split in ("train", "val", "test")
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    torch.save({"version": 1, "signal_length": signal_length, "splits": splits}, temporary)
    temporary.replace(CACHE_PATH)
    return splits


def class_counts(labels: torch.Tensor) -> dict[str, int]:
    return {name: int(labels[:, index].sum().item()) for index, name in enumerate(SUPERCLASSES)}


def run_linear(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict:
    result = train_head(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        epochs=HEAD_EPOCHS,
        batch_size=HEAD_BATCH,
        lr=HEAD_LR,
        weight_decay=HEAD_WD,
        seed=seed,
        device=device,
    )
    return {
        "test_auc": float(result["test_auc"]),
        "val_auc": float(result["val_auc"]),
        "per_class": {key: float(value) for key, value in result["per_class"].items()},
    }


def run_low_label(data_dir: str, seeds: list[int], device: str | None = None) -> None:
    missing = [seed for seed in seeds if not temporal_checkpoint(seed).is_file()]
    if missing:
        raise FileNotFoundError(f"missing temporal checkpoints for seeds {missing}")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, cfg = load_target_encoder(temporal_checkpoint(seeds[0]), torch.device("cpu"))
    splits = load_splits(data_dir, cfg.signal_length)
    done = {row_key(row): row for row in load_rows()}
    for seed in seeds:
        pending = [percent for percent in PERCENTS if (seed, percent) not in done]
        if not pending:
            print(f"reusing label budgets for seed {seed}", flush=True)
            continue
        ckpt = temporal_checkpoint(seed)
        print(f"extracting frozen features seed {seed} from {ckpt}", flush=True)
        encoder, cfg = load_target_encoder(ckpt, torch_device)
        features = {
            split: embed_signals(encoder, splits[split][0], torch_device)
            for split in ("train", "val", "test")
        }
        del encoder
        # Same offset as probe.py: the random encoder is not the fine-tune seed.
        random_encoder = fresh_target_encoder(cfg, torch_device, seed + 1)
        random_features = {
            split: embed_signals(random_encoder, splits[split][0], torch_device)
            for split in ("train", "val", "test")
        }
        del random_encoder
        train_x, train_y = splits["train"]
        val_x, val_y = splits["val"]
        test_x, test_y = splits["test"]
        for percent in pending:
            index = torch.from_numpy(budget_indices(train_x.shape[0], percent, seed))
            counts = class_counts(train_y[index])
            print(
                f"seed {seed} {percent}% n={int(index.numel())}/{train_x.shape[0]} {counts}",
                flush=True,
            )
            linear = run_linear(
                features["train"][index],
                train_y[index],
                features["val"],
                val_y,
                features["test"],
                test_y,
                seed,
                torch_device,
            )
            linear_random = run_linear(
                random_features["train"][index],
                train_y[index],
                random_features["val"],
                val_y,
                random_features["test"],
                test_y,
                seed,
                torch_device,
            )
            print(f"finetune pretrained seed {seed} {percent}%", flush=True)
            encoder, cfg = load_target_encoder(ckpt, torch_device)
            tuned = finetune_split(
                encoder,
                train_x[index],
                train_y[index],
                val_x,
                val_y,
                test_x,
                test_y,
                seed,
                torch_device,
                epochs=EPOCHS,
            )
            print(f"finetune scratch seed {seed} {percent}%", flush=True)
            scratch = finetune_split(
                fresh_target_encoder(cfg, torch_device, seed),
                train_x[index],
                train_y[index],
                val_x,
                val_y,
                test_x,
                test_y,
                seed,
                torch_device,
                epochs=EPOCHS,
            )
            row = {
                "seed": seed,
                "percent": percent,
                "ckpt": str(ckpt),
                "n_train_full": int(train_x.shape[0]),
                "n_train": int(index.numel()),
                "class_counts": counts,
                "indices": index.tolist(),
                "linear_pretrained_auc": linear["test_auc"],
                "linear_random_auc": linear_random["test_auc"],
                "linear_delta": linear["test_auc"] - linear_random["test_auc"],
                "linear_pretrained_val_auc": linear["val_auc"],
                "linear_random_val_auc": linear_random["val_auc"],
                "linear_pretrained_per_class": linear["per_class"],
                "linear_random_per_class": linear_random["per_class"],
                "finetune_pretrained_auc": tuned["test_auc"],
                "finetune_scratch_auc": scratch["test_auc"],
                "finetune_delta": tuned["test_auc"] - scratch["test_auc"],
                "linear_minus_scratch": linear["test_auc"] - scratch["test_auc"],
                "finetune_pretrained_val_auc": tuned["val_auc"],
                "finetune_scratch_val_auc": scratch["val_auc"],
                "finetune_pretrained_best_epoch": tuned["best_epoch"],
                "finetune_scratch_best_epoch": scratch["best_epoch"],
                "finetune_pretrained_per_class": tuned["per_class"],
                "finetune_scratch_per_class": scratch["per_class"],
            }
            print(
                f"seed {seed} {percent}% linear {linear['test_auc']:.4f} "
                f"random {linear_random['test_auc']:.4f} "
                f"finetune {tuned['test_auc']:.4f} scratch {scratch['test_auc']:.4f}",
                flush=True,
            )
            done[row_key(row)] = row
            save_rows(ordered_rows(done, seeds))
    save_rows(ordered_rows(done, seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear head and fine-tune on 1% and 10% of folds 1–8"
    )
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run_low_label(args.data_dir, seeds, args.device)


if __name__ == "__main__":
    main()
