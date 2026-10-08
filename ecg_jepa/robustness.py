"""Five-seed pretrain and probe.

Each seed trains on folds 1–8. The linear head is chosen by macro AUC on fold
9. Fold 10 is scored once, after that choice, and is not used to pick a seed,
an epoch, or a variant. The random-encoder probe uses the same head seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_jepa.config import Config
from ecg_jepa.probe import run_probe
from ecg_jepa.train import run


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("need at least one score")
    if scores.size == 1:
        return float(scores[0]), 0.0
    return float(scores.mean()), float(scores.std(ddof=1))


def format_mean_std(values: list[float]) -> str:
    mean, std = sample_mean_std(values)
    return f"{mean:.4f} ± {std:.4f}"


def checkpoint_path(variant: str, seed: int) -> Path:
    return Path("checkpoints") / variant / f"seed_{seed}" / "last.pt"


def train_seed(variant: str, seed: int, data_dir: str, epochs: int) -> Path:
    path = checkpoint_path(variant, seed)
    if path.is_file():
        print(f"reusing {path}", flush=True)
        return path
    reused = Path("checkpoints/run1/last.pt")
    if variant == "temporal" and seed == 0 and reused.is_file():
        print(f"reusing seed 0 from {reused}", flush=True)
        return reused
    cfg = Config(
        seed=seed,
        epochs=epochs,
        token_mode=variant,
        ckpt_dir=str(path.parent),
        log_every=100,
    )
    print(f"training {variant} seed {seed}", flush=True)
    return run(cfg, data_dir=data_dir, synthetic=False)


def results_path(variant: str) -> Path:
    return Path("logs") / f"robustness_{variant}.json"


def load_results(variant: str) -> list[dict]:
    path = results_path(variant)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["seeds"]


def save_results(variant: str, seeds: list[dict]) -> None:
    path = results_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(seeds)
    path.write_text(
        json.dumps({"variant": variant, "seeds": seeds, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(
        f"{variant} n={summary['n']} "
        f"pretrained {summary['pretrained']} random {summary['random']} "
        f"delta {summary['delta']}",
        flush=True,
    )


def summarize(seeds: list[dict]) -> dict[str, str | int]:
    if not seeds:
        return {"n": 0, "pretrained": "", "random": "", "delta": ""}
    return {
        "n": len(seeds),
        "pretrained": format_mean_std([row["pretrained_auc"] for row in seeds]),
        "random": format_mean_std([row["random_auc"] for row in seeds]),
        "delta": format_mean_std([row["delta"] for row in seeds]),
    }


def run_variant(
    variant: str,
    seeds: list[int],
    data_dir: str,
    epochs: int,
    probe_epochs: int,
) -> dict[str, str | int]:
    done = {int(row["seed"]): row for row in load_results(variant)}
    ordered: list[dict] = []
    for seed in seeds:
        if seed in done:
            print(f"reusing probe for {variant} seed {seed}", flush=True)
            ordered.append(done[seed])
            continue
        ckpt = train_seed(variant, seed, data_dir, epochs)
        probe = run_probe(
            data_dir=data_dir,
            ckpt=str(ckpt),
            epochs=probe_epochs,
            seed=seed,
        )
        row = {"seed": seed, "ckpt": str(ckpt), **probe}
        done[seed] = row
        ordered = [done[item] for item in seeds if item in done]
        save_results(variant, ordered)
    save_results(variant, ordered)
    return summarize(ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed pretrain and fold-10 probe")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--variant", choices=("temporal", "per_lead"), required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--probe-epochs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run_variant(args.variant, seeds, args.data_dir, args.epochs, args.probe_epochs)


if __name__ == "__main__":
    main()
