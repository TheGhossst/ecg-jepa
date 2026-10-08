"""PTB-XL loading for the unlabeled JEPA pretrain and the linear probe.

Uses the 100 Hz recordings and official stratification folds. Folds 1–8 are
pretrain, fold 9 is monitoring, and fold 10 is held out of `make_dataloader`.
The probe reads fold 10 through `PTBXLLabeledDataset`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ecg_jepa.config import Config

# III, aVR, aVL, and aVF are linear combinations of I and II.
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

# Fold 10 is the official test fold. Pretraining does not load it.
SPLIT_FOLDS = {
    "train": (1, 2, 3, 4, 5, 6, 7, 8),
    "val": (9,),
    "test": (10,),
}

# Five diagnostic superclasses used by the PTB-XL benchmark.
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def canonical_lead(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def zscore_leads(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score along time. `signal` is (leads, time)."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / np.maximum(std, 1e-6)


def fit_length(signal: np.ndarray, length: int) -> np.ndarray:
    """Center-crop or zero-pad time so every record is `length` samples."""
    current = signal.shape[-1]
    if current == length:
        return signal
    if current > length:
        start = (current - length) // 2
        return signal[:, start : start + length]
    pad = length - current
    left = pad // 2
    return np.pad(signal, ((0, 0), (left, pad - left)))


def diagnostic_superclass_map(root: str | Path) -> dict[str, str]:
    """Map an SCP statement code to NORM, MI, STTC, CD, or HYP."""
    table = pd.read_csv(Path(root) / "scp_statements.csv", index_col=0)
    diagnostic = table[table["diagnostic"] == 1.0]
    return diagnostic["diagnostic_class"].astype(str).to_dict()


def encode_named_labels(
    raw_codes: str,
    names: list[str],
    code_to_name: dict[str, str] | None = None,
) -> np.ndarray:
    """Multi-hot vector over `names`.

    Same rule as the superclass encoding: a key present in `scp_codes` counts,
    including likelihood 0. `code_to_name` maps a statement code onto a coarser
    label such as a diagnostic subclass.
    """
    present = set(ast.literal_eval(raw_codes))
    hits: set[str] = set()
    allowed = set(names)
    for code in present:
        label = code if code_to_name is None else code_to_name.get(code)
        if label in allowed:
            hits.add(label)
    return np.array([1.0 if name in hits else 0.0 for name in names], dtype=np.float32)


def diagnostic_label_names(root: str | Path) -> tuple[list[str], list[str], dict[str, str]]:
    """Diagnostic statement names, subclass names, and statement-to-subclass map."""
    table = pd.read_csv(Path(root) / "scp_statements.csv", index_col=0)
    diagnostic = table[table["diagnostic"] == 1.0]
    statements = [str(name) for name in diagnostic.index]
    code_to_subclass = diagnostic["diagnostic_subclass"].astype(str).to_dict()
    subclasses = sorted(set(code_to_subclass.values()))
    return statements, subclasses, code_to_subclass


def encode_superclasses(raw_codes: str, mapping: dict[str, str]) -> np.ndarray:
    """Multi-hot superclass vector.

    Follows the PhysioNet PTB-XL aggregation: every diagnostic key present in
    `scp_codes` counts, including likelihood 0. That choice matches the
    published superclass counts (NORM 9514, MI 5469, STTC 5235, CD 4898, HYP 2649).
    """
    codes = ast.literal_eval(raw_codes)
    labels = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    index = {name: i for i, name in enumerate(SUPERCLASSES)}
    for code in codes:
        superclass = mapping.get(code)
        slot = index.get(superclass)
        if slot is not None:
            labels[slot] = 1.0
    return labels


def select_independent_leads(signal: np.ndarray, lead_names: list[str]) -> np.ndarray:
    """Pick I, II, V1–V6 from a WFDB array of shape (time, leads)."""
    index = {canonical_lead(name): i for i, name in enumerate(lead_names)}
    missing = [lead for lead in INDEPENDENT_LEADS if lead not in index]
    if missing:
        raise ValueError(f"recording is missing leads {missing}; got {lead_names}")
    chosen = signal[:, [index[lead] for lead in INDEPENDENT_LEADS]]
    return np.ascontiguousarray(chosen.T, dtype=np.float32)


def _synthetic_signals(
    n_records: int,
    n_leads: int,
    length: int,
    seed: int,
) -> np.ndarray:
    """Eight-lead sine-plus-pulse recordings, z-scored per lead."""
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float32) / 100.0
    signals = np.zeros((n_records, n_leads, length), dtype=np.float32)
    for i in range(n_records):
        rate = float(rng.uniform(0.8, 2.0))
        for lead in range(n_leads):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            amplitude = float(rng.uniform(0.4, 1.5))
            baseline = 0.15 * np.sin(2.0 * np.pi * 0.15 * time + phase)
            cycle = np.mod(time * rate, 1.0) - 0.25
            pulse = np.exp(-0.5 * (cycle / 0.03) ** 2)
            signals[i, lead] = amplitude * pulse + baseline
        signals[i] = zscore_leads(signals[i])
    return signals


class SyntheticECG(Dataset):
    """Fixed synthetic batch source so tests and smoke runs need no PTB-XL."""

    def __init__(self, n_records: int, n_leads: int, signal_length: int, seed: int):
        self.signals = _synthetic_signals(n_records, n_leads, signal_length, seed)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.signals[index])


def resolve_ptbxl_root(root: str | Path) -> Path:
    """Find the folder that contains ptbxl_database.csv.

    PhysioNet checkouts put the csv at the root. The Kaggle archive nests it
    one level down, under a long ptb-xl-...-1.0.3 directory.
    """
    root = Path(root)
    direct = root / "ptbxl_database.csv"
    if direct.is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} is not a PTB-XL folder (missing ptbxl_database.csv)."
        )
    matches = sorted(path.parent for path in root.glob("*/ptbxl_database.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"expected {direct}, or one subfolder that contains it. "
            "The Kaggle extract keeps the csv one level down."
        )
    listed = ", ".join(str(path) for path in matches)
    raise FileNotFoundError(
        f"found multiple PTB-XL roots under {root}: {listed}. Pass one as --data-dir."
    )


def read_record(root: Path, relative: str, signal_length: int) -> torch.Tensor:
    """Load one 100 Hz record as (8, signal_length), z-scored per lead."""
    import wfdb

    record = str(root / relative)
    signal, fields = wfdb.rdsamp(record)
    sample_rate = float(fields["fs"])
    if sample_rate != 100.0:
        raise ValueError(
            f"{record} is {sample_rate} Hz; this model uses records100 (100 Hz)"
        )
    leads = select_independent_leads(signal, list(fields["sig_name"]))
    leads = zscore_leads(fit_length(leads, signal_length))
    return torch.from_numpy(np.ascontiguousarray(leads, dtype=np.float32))


def _fold_frame(root: Path, folds: tuple[int, ...]) -> pd.DataFrame:
    csv_path = root / "ptbxl_database.csv"
    frame = pd.read_csv(csv_path)
    frame = frame[frame["strat_fold"].isin(folds)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no PTB-XL rows for folds {folds} in {csv_path}")
    print(f"PTB-XL {root} folds {folds} records {len(frame)}", flush=True)
    return frame


class PTBXLDataset(Dataset):
    """100 Hz PTB-XL records from a local 1.0.3 tree. Labels are not read."""

    def __init__(self, root: str | Path, folds: tuple[int, ...], signal_length: int):
        self.root = resolve_ptbxl_root(root)
        self.signal_length = signal_length
        self.frame = _fold_frame(self.root, folds)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> torch.Tensor:
        relative = str(self.frame.loc[index, "filename_lr"])
        return read_record(self.root, relative, self.signal_length)


class PTBXLLabeledDataset(Dataset):
    """Same recordings as `PTBXLDataset`, plus a 5-way multi-hot superclass label."""

    def __init__(self, root: str | Path, folds: tuple[int, ...], signal_length: int):
        self.root = resolve_ptbxl_root(root)
        self.signal_length = signal_length
        self.frame = _fold_frame(self.root, folds)
        mapping = diagnostic_superclass_map(self.root)
        self.labels = np.stack(
            [encode_superclasses(raw, mapping) for raw in self.frame["scp_codes"]]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        relative = str(self.frame.loc[index, "filename_lr"])
        signal = read_record(self.root, relative, self.signal_length)
        label = torch.from_numpy(self.labels[index])
        return signal, label


def make_dataloader(
    cfg: Config,
    split: str,
    synthetic: bool,
    data_dir: str | None,
    pin_memory: bool = False,
) -> DataLoader:
    if split not in SPLIT_FOLDS:
        raise ValueError(
            "split must be 'train', 'val', or 'test'."
        )
    if synthetic:
        n_records = 64 if split == "train" else 16
        seed = cfg.seed if split == "train" else cfg.seed + 1
        dataset: Dataset = SyntheticECG(n_records, cfg.n_leads, cfg.signal_length, seed)
    else:
        if not data_dir:
            raise ValueError("pass --data-dir pointing at PTB-XL, or use --synthetic")
        dataset = PTBXLDataset(data_dir, SPLIT_FOLDS[split], cfg.signal_length)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
