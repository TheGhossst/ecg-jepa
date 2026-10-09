"""Amplitude features from the millivolt waveform, before per-lead z-scoring.

The encoder never sees absolute voltage. These experiments ask whether that
scale is useful on its own, and whether the frozen temporal embedding still
needs it.

A. A linear head on amplitude features alone. The first set is peak R. The
   second adds the per-lead scale the loader divides out, and peak-to-peak.
B. The same head on the frozen embedding concatenated with the second set.
   The encoder checkpoint is not updated.
C. A small amplitude branch in front of that concatenation. It runs only
   when B's paired fold-10 macro AUC clears the seed spread against the
   embedding-only head. That is the same bar as the full-label fine-tune.
   The trained head is written beside the encoder as two_branch.pt. That
   file is the classifier: one millivolt ECG in, five sigmoid scores out.

Fold 9 chooses the epoch. Fold 10 is scored once. Feature means and scales
are fit on folds 1–8 only. Head hyperparameters match the frozen probe and
are not retuned.
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
    fit_length,
    resolve_ptbxl_root,
    zscore_leads,
)
from ecg_jepa.downstream import (
    HEAD_BATCH,
    HEAD_EPOCHS,
    HEAD_LR,
    HEAD_WD,
    embed_signals,
    fit_linear,
    overlap_report,
)
from ecg_jepa.finetune import temporal_checkpoint
from ecg_jepa.hyp_voltage import peak_r_amplitude, read_millivolts
from ecg_jepa.low_label import clears_seed_spread, load_splits
from ecg_jepa.probe import load_target_encoder, macro_auc
from ecg_jepa.robustness import format_mean_std
from ecg_jepa.train import learning_rate, set_seed

FEATURE_CACHE = Path("checkpoints/amplitude_features.pt")
BRANCH_WIDTH = 32
N_LEADS = 8
HEAD_FILE = "two_branch.pt"
# Fixed before fold 10. Fold 9 still chooses the epoch; it does not tune this cut.
DECISION_THRESHOLD = 0.5


def amplitude_features(signal: np.ndarray) -> np.ndarray:
    """Per-lead peak, standard deviation, and peak-to-peak, in that order.

    `signal` is (8, time) in millivolts, already cropped, not z-scored.
    Standard deviation is the scale `zscore_leads` divides by. RMS is the
    other scale summary and is not a separate feature.
    """
    if signal.ndim != 2 or signal.shape[0] != N_LEADS:
        raise ValueError(f"expected shape (8, time), got {signal.shape}")
    peak = np.max(signal, axis=-1)
    scale = np.std(signal, axis=-1)
    span = peak - np.min(signal, axis=-1)
    return np.concatenate([peak, scale, span]).astype(np.float32)


def feature_mean_std(train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Column mean and scale from the training fold only."""
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def standardize(train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Z-score columns using the training fold only."""
    mean, std = feature_mean_std(train)
    return tuple((tensor - mean) / std for tensor in (train, *others))


def threshold_report(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Per-label accuracy and F1. A score at the threshold counts as positive."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 2 or labels.shape[1] != len(SUPERCLASSES):
        raise ValueError(f"expected labels and scores shaped (n, {len(SUPERCLASSES)})")
    predicted = scores >= threshold
    report: dict[str, dict[str, float]] = {}
    for index, name in enumerate(SUPERCLASSES):
        y = labels[:, index].astype(bool)
        call = predicted[:, index]
        tp = int(np.count_nonzero(call & y))
        tn = int(np.count_nonzero(~call & ~y))
        fp = int(np.count_nonzero(call & ~y))
        fn = int(np.count_nonzero(~call & y))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        report[name] = {"accuracy": float(accuracy), "f1": float(f1)}
    return report


def amplitude_helps(deltas: list[float]) -> bool:
    """B works when the paired macro gap clears the seed spread."""
    return clears_seed_spread(deltas)


class TwoBranch(nn.Module):
    """Frozen embedding plus a small amplitude network, then a linear head.

    The waveform branch is the frozen temporal encoder, applied before this
    module. Its parameters are not in this module and are not updated.
    """

    def __init__(self, embed_dim: int, n_amplitude: int, n_classes: int, width: int = BRANCH_WIDTH):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_amplitude = n_amplitude
        self.n_classes = n_classes
        self.width = width
        self.amplitude = nn.Sequential(
            nn.Linear(n_amplitude, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.head = nn.Linear(embed_dim + width, n_classes)

    def forward(self, embedding: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([embedding, self.amplitude(amplitude)], dim=-1))


def score_split(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | dict[str, float]]:
    auc, per_class = macro_auc(labels, scores)
    overlap = overlap_report(labels, scores)["HYP"]
    if int(overlap["n_pure"]) != 56 or int(overlap["n_mixed"]) != 206:
        raise RuntimeError(
            f"expected 56 pure and 206 mixed HYP, got {overlap['n_pure']} and {overlap['n_mixed']}"
        )
    return {
        "test_auc": float(auc),
        "per_class": {name: float(per_class[name]) for name in SUPERCLASSES},
        "hyp_pure_auc": float(overlap["pure_auc"]),
        "hyp_mixed_auc": float(overlap["mixed_auc"]),
    }


def train_linear_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict:
    result = fit_linear(train_x, train_y, val_x, val_y, test_x, test_y, seed, device, "linear")
    if result["columns"].tolist() != list(range(len(SUPERCLASSES))):
        raise RuntimeError("the head dropped a superclass")
    packed = score_split(test_y.numpy(), result["scores"])
    packed["val_auc"] = float(result["val_auc"])
    return packed


def train_two_branch(
    train_emb: torch.Tensor,
    train_amp: torch.Tensor,
    train_y: torch.Tensor,
    val_emb: torch.Tensor,
    val_amp: torch.Tensor,
    val_y: torch.Tensor,
    test_emb: torch.Tensor,
    test_amp: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[dict, TwoBranch]:
    """Same schedule as the linear head. Fold 10 is scored after the fold-9 choice."""
    set_seed(seed)
    model = TwoBranch(train_emb.shape[1], train_amp.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_emb, train_amp, train_y),
        batch_size=HEAD_BATCH,
        shuffle=True,
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = HEAD_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch
    batches = {
        "val": (val_emb.to(device), val_amp.to(device)),
        "test": (test_emb.to(device), test_amp.to(device)),
    }
    best_val = -1.0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    step = 0
    for _epoch in range(HEAD_EPOCHS):
        model.train()
        for embedding, amplitude, target in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total_steps, warmup_steps, HEAD_LR)
            optimizer.zero_grad(set_to_none=True)
            logits = model(embedding.to(device), amplitude.to(device))
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            optimizer.step()
            step += 1
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(*batches["val"])).cpu().numpy()
        val_auc, _ = macro_auc(val_y.numpy(), val_scores)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(*batches["test"])).cpu().numpy()
    packed = score_split(test_y.numpy(), test_scores)
    packed["val_auc"] = float(best_val)
    report = threshold_report(test_y.numpy(), test_scores)
    packed["threshold"] = DECISION_THRESHOLD
    packed["fold10_accuracy"] = {name: report[name]["accuracy"] for name in SUPERCLASSES}
    packed["fold10_f1"] = {name: report[name]["f1"] for name in SUPERCLASSES}
    return packed, model


def head_checkpoint_path(encoder_ckpt: Path) -> Path:
    """`two_branch.pt` in the same directory as the encoder checkpoint."""
    return Path(encoder_ckpt).parent / HEAD_FILE


def save_two_branch(
    encoder_ckpt: Path,
    model: TwoBranch,
    amplitude_mean: torch.Tensor,
    amplitude_std: torch.Tensor,
    seed: int,
    signal_length: int,
) -> Path:
    """Write the head beside the encoder. The encoder file is not opened for writing."""
    encoder_ckpt = Path(encoder_ckpt)
    if not encoder_ckpt.is_file():
        raise FileNotFoundError(encoder_ckpt)
    if model.n_classes != len(SUPERCLASSES):
        raise ValueError(f"expected {len(SUPERCLASSES)} classes, got {model.n_classes}")
    path = head_checkpoint_path(encoder_ckpt)
    payload = {
        "version": 1,
        "kind": "two_branch",
        "classes": list(SUPERCLASSES),
        "embed_dim": model.embed_dim,
        "n_amplitude": model.n_amplitude,
        "n_classes": model.n_classes,
        "width": model.width,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "amplitude_mean": amplitude_mean.detach().cpu().float().clone(),
        "amplitude_std": amplitude_std.detach().cpu().float().clone(),
        "encoder_file": encoder_ckpt.name,
        "seed": seed,
        "signal_length": signal_length,
        "threshold": DECISION_THRESHOLD,
        "features": "per-lead peak, standard deviation, peak-to-peak",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def prepare_recording(
    signal_mv: np.ndarray,
    signal_length: int,
    amplitude_mean: torch.Tensor,
    amplitude_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Millivolt `(8, time)` array to a z-scored waveform and scaled amplitude features.

    The waveform is what the encoder sees. The features keep absolute voltage.
    """
    signal = np.asarray(signal_mv, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != N_LEADS:
        raise ValueError(f"expected millivolts shaped (8, time), got {tuple(signal.shape)}")
    raw = fit_length(signal, signal_length)
    features = torch.from_numpy(amplitude_features(raw))
    mean = amplitude_mean.detach().cpu().float()
    std = amplitude_std.detach().cpu().float()
    scaled = (features - mean) / std
    waveform = torch.from_numpy(np.ascontiguousarray(zscore_leads(raw), dtype=np.float32))
    return waveform, scaled


@torch.no_grad()
def predict_scores(
    signal_mv: np.ndarray,
    head_path: str | Path,
    device: str | None = None,
) -> dict[str, float]:
    """One millivolt ECG in, five sigmoid scores out.

    `signal_mv` is shaped (8, time), leads I, II, V1–V6, not z-scored.
    The encoder is loaded from the same directory as the head and is not updated.
    """
    head_path = Path(head_path)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    if blob.get("version") != 1 or blob.get("kind") != "two_branch":
        raise ValueError(f"{head_path} is not a two-branch head checkpoint")
    if tuple(blob["classes"]) != SUPERCLASSES:
        raise ValueError(f"checkpoint classes {blob['classes']} are not {SUPERCLASSES}")
    model = TwoBranch(
        int(blob["embed_dim"]),
        int(blob["n_amplitude"]),
        int(blob["n_classes"]),
        int(blob["width"]),
    ).to(torch_device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    encoder_path = head_path.parent / str(blob["encoder_file"])
    encoder, cfg = load_target_encoder(encoder_path, torch_device)
    if int(blob["signal_length"]) != cfg.signal_length:
        raise RuntimeError(
            f"head signal_length {blob['signal_length']} does not match the encoder ({cfg.signal_length})"
        )
    embed_dim = int(encoder.norm.normalized_shape[0])
    if embed_dim != int(blob["embed_dim"]):
        raise RuntimeError(f"head embed_dim {blob['embed_dim']} does not match the encoder ({embed_dim})")
    waveform, amplitude = prepare_recording(
        signal_mv,
        cfg.signal_length,
        blob["amplitude_mean"],
        blob["amplitude_std"],
    )
    embedding = encoder(waveform.unsqueeze(0).to(torch_device)).mean(dim=1)
    logits = model(embedding, amplitude.unsqueeze(0).to(torch_device))
    probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu()
    return {name: float(probabilities[index]) for index, name in enumerate(SUPERCLASSES)}


def results_path() -> Path:
    return Path("logs/amplitude.json")


def _mean_block(rows: list[dict], field: str) -> dict[str, object]:
    values = [float(row[field]) for row in rows]
    block: dict[str, object] = {
        "mean_std": format_mean_std(values),
        "values": [round(value, 4) for value in values],
    }
    if "delta" in field:
        block["clears_seed_spread"] = clears_seed_spread(values)
    return block


def summarize_rows(rows: list[dict], fields: tuple[str, ...]) -> dict[str, dict]:
    if len(rows) < 2:
        return {}
    return {field: _mean_block(rows, field) for field in fields}


def summarize_label_values(rows: list[dict], field: str) -> dict[str, dict[str, object]]:
    """Mean ± sample std of a per-label map stored on each seed row."""
    if len(rows) < 2:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for name in SUPERCLASSES:
        values = [float(row[field][name]) for row in rows]
        summary[name] = {
            "mean_std": format_mean_std(values),
            "values": [round(value, 4) for value in values],
        }
    return summary


def save_payload(payload: dict) -> None:
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_amplitude(data_dir: str, signal_length: int) -> dict[str, dict[str, torch.Tensor]]:
    if FEATURE_CACHE.is_file():
        blob = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
        if blob.get("version") == 1 and blob.get("signal_length") == signal_length:
            print(f"reusing {FEATURE_CACHE}", flush=True)
            return blob["splits"]
    root = resolve_ptbxl_root(data_dir)
    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        dataset = PTBXLLabeledDataset(data_dir, SPLIT_FOLDS[split], signal_length)
        peaks = np.empty(len(dataset), dtype=np.float32)
        features = np.empty((len(dataset), 3 * N_LEADS), dtype=np.float32)
        for index in range(len(dataset)):
            if index % 2000 == 0:
                print(f"amplitude {split} {index}/{len(dataset)}", flush=True)
            relative = str(dataset.frame.loc[index, "filename_lr"])
            raw = read_millivolts(root, relative, signal_length)
            peaks[index] = peak_r_amplitude(raw)
            features[index] = amplitude_features(raw)
        splits[split] = {
            "peak_r": torch.from_numpy(peaks).unsqueeze(1),
            "amplitude": torch.from_numpy(features),
            "labels": torch.from_numpy(dataset.labels.copy()),
        }
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEATURE_CACHE.with_suffix(".tmp")
    torch.save({"version": 1, "signal_length": signal_length, "splits": splits}, temporary)
    temporary.replace(FEATURE_CACHE)
    return splits


def check_peak_r_ranking(labels: np.ndarray, peak_r: np.ndarray) -> float:
    """Untrained ranking, so this can be compared with the earlier pure-HYP number."""
    scores = np.zeros((labels.shape[0], len(SUPERCLASSES)), dtype=np.float64)
    scores[:, SUPERCLASSES.index("HYP")] = peak_r
    pure = float(overlap_report(labels, scores)["HYP"]["pure_auc"])
    logged = Path("logs/hyp_voltage.json")
    if logged.is_file():
        expected = float(json.loads(logged.read_text(encoding="utf-8"))["peak_r_mv"]["pure_auc"])
        if abs(pure - expected) > 1e-4:
            raise RuntimeError(f"peak R pure-HYP AUC {pure:.4f} does not match {expected:.4f}")
    return pure


def run_amplitude(data_dir: str, seeds: list[int], device: str | None = None) -> dict:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, cfg = load_target_encoder(temporal_checkpoint(seeds[0]), torch.device("cpu"))
    waveforms = load_splits(data_dir, cfg.signal_length)
    amplitude = collect_amplitude(data_dir, cfg.signal_length)
    for split in ("train", "val", "test"):
        cached = waveforms[split][1]
        fresh = amplitude[split]["labels"]
        if cached.shape != fresh.shape or not torch.equal(cached, fresh):
            raise RuntimeError(f"{split} amplitude rows do not match the labeled cache")
    labels = {split: amplitude[split]["labels"] for split in amplitude}
    ranking = check_peak_r_ranking(labels["test"].numpy(), amplitude["test"]["peak_r"].numpy().reshape(-1))
    print(f"untrained peak R pure-HYP {ranking:.4f}", flush=True)

    peak_scaled = standardize(
        amplitude["train"]["peak_r"], amplitude["val"]["peak_r"], amplitude["test"]["peak_r"]
    )
    amp_mean, amp_std = feature_mean_std(amplitude["train"]["amplitude"])
    amp_scaled = tuple(
        (tensor - amp_mean) / amp_std
        for tensor in (
            amplitude["train"]["amplitude"],
            amplitude["val"]["amplitude"],
            amplitude["test"]["amplitude"],
        )
    )
    protocol = {
        "A": "linear head on peak R, then on per-lead peak, std, and peak-to-peak",
        "B": "frozen temporal embedding concatenated with the 24 amplitude features",
        "C": "amplitude MLP of width 32, only if B clears the seed spread on macro AUC",
        "fold_9": "chooses the epoch",
        "fold_10": "scored once",
        "scaler": "training fold only",
        "head": "probe hyperparameters, not retuned",
        "checkpoint": "frozen temporal",
        "head_checkpoint": "two_branch.pt beside the encoder; encoder file is not rewritten",
        "threshold": "0.5 per class, fixed before fold 10",
        "fold10_decision": "per-label accuracy and F1 at that threshold",
    }
    payload: dict = {
        "protocol": protocol,
        "untrained_peak_r_pure_auc": ranking,
        "A": {},
        "B": {},
        "C": None,
    }

    feature_sets = {
        "peak_r": peak_scaled,
        "amplitude": amp_scaled,
    }
    for name, tensors in feature_sets.items():
        rows = []
        for seed in seeds:
            print(f"A {name} seed {seed}", flush=True)
            row = train_linear_head(
                tensors[0],
                labels["train"],
                tensors[1],
                labels["val"],
                tensors[2],
                labels["test"],
                seed,
                torch_device,
            )
            row["seed"] = seed
            row["hyp_auc"] = row["per_class"]["HYP"]
            rows.append(row)
            print(
                f"A {name} seed {seed} macro {row['test_auc']:.4f} "
                f"HYP {row['per_class']['HYP']:.4f} pure {row['hyp_pure_auc']:.4f}",
                flush=True,
            )
        payload["A"][name] = {
            "seeds": rows,
            "summary": summarize_rows(
                rows, ("test_auc", "hyp_auc", "hyp_pure_auc", "hyp_mixed_auc")
            ),
        }
        save_payload(payload)

    b_rows = []
    for seed in seeds:
        print(f"extracting frozen embedding seed {seed}", flush=True)
        encoder, _ = load_target_encoder(temporal_checkpoint(seed), torch_device)
        embedding = {
            split: embed_signals(encoder, waveforms[split][0], torch_device)
            for split in ("train", "val", "test")
        }
        del encoder
        print(f"B embedding seed {seed}", flush=True)
        embedding_only = train_linear_head(
            embedding["train"],
            labels["train"],
            embedding["val"],
            labels["val"],
            embedding["test"],
            labels["test"],
            seed,
            torch_device,
        )
        scaled = {"train": amp_scaled[0], "val": amp_scaled[1], "test": amp_scaled[2]}
        concat = {
            split: torch.cat([embedding[split], scaled[split]], dim=1)
            for split in ("train", "val", "test")
        }
        print(f"B concat seed {seed}", flush=True)
        both = train_linear_head(
            concat["train"],
            labels["train"],
            concat["val"],
            labels["val"],
            concat["test"],
            labels["test"],
            seed,
            torch_device,
        )
        row = {
            "seed": seed,
            "embedding_auc": embedding_only["test_auc"],
            "concat_auc": both["test_auc"],
            "delta": both["test_auc"] - embedding_only["test_auc"],
            "embedding_hyp": embedding_only["per_class"]["HYP"],
            "concat_hyp": both["per_class"]["HYP"],
            "hyp_delta": both["per_class"]["HYP"] - embedding_only["per_class"]["HYP"],
            "embedding_pure": embedding_only["hyp_pure_auc"],
            "concat_pure": both["hyp_pure_auc"],
            "pure_delta": both["hyp_pure_auc"] - embedding_only["hyp_pure_auc"],
            "embedding_mixed": embedding_only["hyp_mixed_auc"],
            "concat_mixed": both["hyp_mixed_auc"],
            "embedding_per_class": embedding_only["per_class"],
            "concat_per_class": both["per_class"],
            "embedding_val_auc": embedding_only["val_auc"],
            "concat_val_auc": both["val_auc"],
        }
        b_rows.append(row)
        print(
            f"B seed {seed} embedding {row['embedding_auc']:.4f} "
            f"concat {row['concat_auc']:.4f} delta {row['delta']:+.4f} "
            f"pure {row['embedding_pure']:.4f} -> {row['concat_pure']:.4f}",
            flush=True,
        )
        works = amplitude_helps([item["delta"] for item in b_rows]) if len(b_rows) >= 2 else False
        payload["B"] = {
            "seeds": b_rows,
            "summary": summarize_rows(
                b_rows,
                ("embedding_auc", "concat_auc", "delta", "hyp_delta", "pure_delta"),
            ),
            "works": works,
        }
        save_payload(payload)

    works = bool(payload["B"]["works"])
    print(f"B works {works}", flush=True)
    if not works:
        payload["C"] = {"skipped": True, "reason": "B did not clear the seed spread on macro AUC"}
        save_payload(payload)
        return payload

    c_rows = []
    for seed in seeds:
        print(f"C two-branch seed {seed}", flush=True)
        encoder, _ = load_target_encoder(temporal_checkpoint(seed), torch_device)
        before = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
        embedding = {
            split: embed_signals(encoder, waveforms[split][0], torch_device)
            for split in ("train", "val", "test")
        }
        row, model = train_two_branch(
            embedding["train"],
            amp_scaled[0],
            labels["train"],
            embedding["val"],
            amp_scaled[1],
            labels["val"],
            embedding["test"],
            amp_scaled[2],
            labels["test"],
            seed,
            torch_device,
        )
        after = encoder.state_dict()
        if any(not torch.equal(before[key], value.detach().cpu()) for key, value in after.items()):
            raise RuntimeError("two-branch training updated the frozen encoder")
        head_path = save_two_branch(
            temporal_checkpoint(seed),
            model,
            amp_mean,
            amp_std,
            seed,
            cfg.signal_length,
        )
        del encoder
        paired = next(item for item in b_rows if item["seed"] == seed)
        row["seed"] = seed
        row["checkpoint"] = head_path.as_posix()
        row["delta_vs_embedding"] = row["test_auc"] - paired["embedding_auc"]
        row["delta_vs_concat"] = row["test_auc"] - paired["concat_auc"]
        row["pure_delta_vs_embedding"] = row["hyp_pure_auc"] - paired["embedding_pure"]
        c_rows.append(row)
        macro_f1 = sum(row["fold10_f1"].values()) / len(SUPERCLASSES)
        print(
            f"C seed {seed} macro {row['test_auc']:.4f} "
            f"vs embedding {row['delta_vs_embedding']:+.4f} "
            f"vs concat {row['delta_vs_concat']:+.4f} "
            f"pure {row['hyp_pure_auc']:.4f} "
            f"macro_f1 {macro_f1:.4f} wrote {head_path}",
            flush=True,
        )
        summary = summarize_rows(
            c_rows,
            ("test_auc", "delta_vs_embedding", "delta_vs_concat", "hyp_pure_auc", "pure_delta_vs_embedding"),
        )
        if len(c_rows) >= 2:
            summary["fold10_accuracy"] = summarize_label_values(c_rows, "fold10_accuracy")
            summary["fold10_f1"] = summarize_label_values(c_rows, "fold10_f1")
        payload["C"] = {
            "skipped": False,
            "seeds": c_rows,
            "summary": summary,
        }
        save_payload(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amplitude features with and without the frozen embedding")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run_amplitude(args.data_dir, seeds, args.device)


if __name__ == "__main__":
    main()
