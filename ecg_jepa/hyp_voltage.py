"""What the fold-10 pure-HYP recordings are, and whether peak R beats the embedding.

Pure HYP means the HYP superclass is present and NORM, MI, STTC, and CD are
not. That is the same split as the frozen linear head. Subclasses come from
diagnostic statements whose diagnostic class is HYP. LVH and RVH are named;
every other HYP subclass is the rest.

Peak R-wave amplitude is the tallest positive sample on V1–V6. This repo has
no QRS detector, so the sample is not a delineated R peak. The score is in
millivolts, with no z-score and no fitted threshold. Higher amplitude ranks
as more HYP. A second score applies the same peak after the per-lead z-score
the encoder actually receives.

The feature beats the embedding when its pure-HYP AUC is greater than every
frozen temporal seed already logged. Nothing here retrains the encoder.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from ecg_jepa.data.ptbxl import (
    INDEPENDENT_LEADS,
    PTBXLLabeledDataset,
    SPLIT_FOLDS,
    SUPERCLASSES,
    canonical_lead,
    diagnostic_label_names,
    diagnostic_superclass_map,
    fit_length,
    resolve_ptbxl_root,
    select_independent_leads,
    zscore_leads,
)
from ecg_jepa.downstream import overlap_report
from ecg_jepa.finetune import temporal_checkpoint
from ecg_jepa.probe import config_from_checkpoint
from ecg_jepa.robustness import format_mean_std

PRECORDIAL = slice(2, 8)
PURE_HYP_N = 56
MIXED_HYP_N = 206


def peak_r_amplitude(signal: np.ndarray) -> float:
    """Tallest positive sample on V1–V6. `signal` is (8, time), I, II, then V1–V6."""
    if signal.shape[0] != 8:
        raise ValueError(f"expected 8 independent leads, got {signal.shape[0]}")
    return float(np.max(signal[PRECORDIAL]))


def hyp_subclasses(
    raw_codes: str,
    code_to_class: dict[str, str],
    code_to_subclass: dict[str, str],
) -> set[str]:
    """HYP diagnostic subclasses present in one `scp_codes` string.

    A key counts even when its likelihood is 0, matching superclass encoding.
    """
    hits: set[str] = set()
    for code in ast.literal_eval(raw_codes):
        if code_to_class.get(code) != "HYP":
            continue
        subclass = code_to_subclass.get(code)
        if subclass is None:
            raise KeyError(f"HYP statement {code} has no diagnostic subclass")
        hits.add(str(subclass))
    return hits


def hyp_bucket(hits: set[str]) -> str:
    """Exclusive partition: LVH, RVH, both, or the rest."""
    has_lvh = "LVH" in hits
    has_rvh = "RVH" in hits
    if has_lvh and has_rvh:
        return "LVH+RVH"
    if has_lvh:
        return "LVH"
    if has_rvh:
        return "RVH"
    return "rest"


def beats_embedding(feature_auc: float, embedding_aucs: list[float]) -> bool:
    """True only when the feature is above every frozen-encoder seed."""
    if not embedding_aucs:
        raise ValueError("need the frozen pure-HYP scores")
    return feature_auc > max(embedding_aucs)


def hyp_gap_reason(raw_beats: bool, zscore_beats: bool) -> str:
    """Why pure HYP is hard, once peak R has been scored both ways."""
    if raw_beats and zscore_beats:
        return "temporal"
    if raw_beats:
        return "scale"
    if zscore_beats:
        return "temporal_zscore"
    return "sample_size"


def diagnosis(reason: str) -> str:
    text = {
        "temporal": (
            "Peak R beats the frozen embedding on pure HYP, in millivolts and after "
            "the per-lead z-score the encoder sees. The gap is temporal resolution."
        ),
        "scale": (
            "Peak R in millivolts beats the frozen embedding on pure HYP. The same "
            "peak after the per-lead z-score does not. Absolute voltage is removed "
            "before the encoder, so a shorter patch on z-scored inputs does not restore it."
        ),
        "temporal_zscore": (
            "Peak R after the per-lead z-score beats the frozen embedding on pure HYP. "
            "The raw millivolt peak does not. A cue that survives preprocessing is still "
            "missing from the embedding, which points at temporal resolution."
        ),
        "sample_size": (
            "Peak R does not beat the frozen embedding on pure HYP. Isolated HYP is a "
            "sample-size problem, and a bigger encoder will not move it."
        ),
    }
    if reason not in text:
        raise ValueError(reason)
    return text[reason]


def embedding_pure_hyp(path: Path | None = None) -> list[dict[str, float | int]]:
    log_path = path or Path("logs/downstream.json")
    blob = json.loads(log_path.read_text(encoding="utf-8"))
    return [
        {
            "seed": int(row["seed"]),
            "pure_auc": float(row["jepa"]["overlap"]["HYP"]["pure_auc"]),
        }
        for row in blob["seeds"]
    ]


def _independent_units(lead_names: list[str], units: list[str]) -> list[str]:
    index = {canonical_lead(name): i for i, name in enumerate(lead_names)}
    missing = [lead for lead in INDEPENDENT_LEADS if lead not in index]
    if missing:
        raise ValueError(f"recording is missing leads {missing}")
    return [units[index[lead]] for lead in INDEPENDENT_LEADS]


def read_millivolts(root: Path, relative: str, signal_length: int) -> np.ndarray:
    """Raw independent leads as (8, time), cropped like the model, not z-scored."""
    import wfdb

    signal, fields = wfdb.rdsamp(str(root / relative))
    if float(fields["fs"]) != 100.0:
        raise ValueError(f"{relative} is {fields['fs']} Hz; peak R uses records100")
    units = _independent_units(list(fields["sig_name"]), list(fields["units"]))
    if any(unit.casefold() != "mv" for unit in units):
        raise ValueError(f"{relative} units are {units}, not mV")
    leads = select_independent_leads(signal, list(fields["sig_name"]))
    return fit_length(leads, signal_length)


def pure_hyp_mask(labels: np.ndarray) -> np.ndarray:
    index = SUPERCLASSES.index("HYP")
    column = labels[:, index]
    others = np.delete(labels, index, axis=1).sum(axis=1) > 0
    return (column == 1) & ~others


def subclass_breakdown(
    raw_codes: list[str],
    pure: np.ndarray,
    code_to_class: dict[str, str],
    code_to_subclass: dict[str, str],
) -> dict:
    buckets = {"LVH": 0, "RVH": 0, "LVH+RVH": 0, "rest": 0}
    inclusive = {"LVH": 0, "RVH": 0, "rest": 0}
    histogram: dict[str, int] = {}
    rest_detail: dict[str, int] = {}
    for row in np.flatnonzero(pure):
        hits = hyp_subclasses(raw_codes[int(row)], code_to_class, code_to_subclass)
        if not hits:
            raise RuntimeError(f"pure HYP row {int(row)} has no HYP subclass")
        key = "+".join(sorted(hits))
        histogram[key] = histogram.get(key, 0) + 1
        buckets[hyp_bucket(hits)] += 1
        if "LVH" in hits:
            inclusive["LVH"] += 1
        if "RVH" in hits:
            inclusive["RVH"] += 1
        if "LVH" not in hits and "RVH" not in hits:
            inclusive["rest"] += 1
            for name in hits:
                rest_detail[name] = rest_detail.get(name, 0) + 1
    if sum(buckets.values()) != int(pure.sum()):
        raise RuntimeError("pure-HYP buckets do not add up")
    return {
        "buckets": buckets,
        "inclusive": inclusive,
        "histogram": dict(sorted(histogram.items())),
        "rest_detail": dict(sorted(rest_detail.items())),
    }


def score_feature(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    scores = np.zeros((labels.shape[0], len(SUPERCLASSES)), dtype=np.float64)
    scores[:, SUPERCLASSES.index("HYP")] = values
    report = overlap_report(labels, scores)["HYP"]
    if int(report["n_pure"]) != int(pure_hyp_mask(labels).sum()):
        raise RuntimeError("pure-HYP mask does not match the overlap split")
    return {
        "pure_auc": float(report["pure_auc"]),
        "mixed_auc": float(report["mixed_auc"]),
        "n_pure": int(report["n_pure"]),
        "n_mixed": int(report["n_mixed"]),
    }


def results_path() -> Path:
    return Path("logs/hyp_voltage.json")


def run_hyp_voltage(data_dir: str, downstream_log: str | None = None) -> dict:
    blob = torch.load(temporal_checkpoint(0), map_location="cpu", weights_only=False)
    signal_length = config_from_checkpoint(blob).signal_length
    root = resolve_ptbxl_root(data_dir)
    dataset = PTBXLLabeledDataset(data_dir, SPLIT_FOLDS["test"], signal_length)
    labels = dataset.labels
    raw_peaks = np.empty(len(dataset), dtype=np.float64)
    zscore_peaks = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        if index % 400 == 0:
            print(f"peak R {index}/{len(dataset)}", flush=True)
        relative = str(dataset.frame.loc[index, "filename_lr"])
        raw = read_millivolts(root, relative, signal_length)
        raw_peaks[index] = peak_r_amplitude(raw)
        zscore_peaks[index] = peak_r_amplitude(zscore_leads(raw))
    pure = pure_hyp_mask(labels)
    code_to_class = diagnostic_superclass_map(root)
    _statements, _subclasses, code_to_subclass = diagnostic_label_names(root)
    breakdown = subclass_breakdown(
        dataset.frame["scp_codes"].tolist(),
        pure,
        code_to_class,
        code_to_subclass,
    )
    raw_score = score_feature(labels, raw_peaks)
    zscore_score = score_feature(labels, zscore_peaks)
    if raw_score["n_pure"] != PURE_HYP_N or raw_score["n_mixed"] != MIXED_HYP_N:
        raise RuntimeError(
            f"expected {PURE_HYP_N} pure and {MIXED_HYP_N} mixed HYP, "
            f"got {raw_score['n_pure']} and {raw_score['n_mixed']}"
        )
    logged = embedding_pure_hyp(Path(downstream_log) if downstream_log else None)
    embedding_aucs = [float(row["pure_auc"]) for row in logged]
    raw_beats = beats_embedding(raw_score["pure_auc"], embedding_aucs)
    zscore_beats = beats_embedding(zscore_score["pure_auc"], embedding_aucs)
    reason = hyp_gap_reason(raw_beats, zscore_beats)
    payload = {
        "n_fold10": int(len(dataset)),
        "n_pure": raw_score["n_pure"],
        "n_mixed": raw_score["n_mixed"],
        "subclasses": breakdown,
        "peak_r_mv": raw_score,
        "peak_r_zscore": zscore_score,
        "embedding_pure_auc": format_mean_std(embedding_aucs),
        "embedding_pure_by_seed": logged,
        "raw_beats_embedding": raw_beats,
        "zscore_beats_embedding": zscore_beats,
        "reason": reason,
        "diagnosis": diagnosis(reason),
    }
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"pure HYP {payload['n_pure']} buckets {breakdown['buckets']} "
        f"rest {breakdown['rest_detail']}",
        flush=True,
    )
    print(
        f"peak R mV pure {raw_score['pure_auc']:.4f} mixed {raw_score['mixed_auc']:.4f} "
        f"zscore pure {zscore_score['pure_auc']:.4f} mixed {zscore_score['mixed_auc']:.4f} "
        f"embedding {payload['embedding_pure_auc']}",
        flush=True,
    )
    print(payload["diagnosis"], flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down pure HYP and score peak R amplitude")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--downstream-log", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hyp_voltage(args.data_dir, args.downstream_log)


if __name__ == "__main__":
    main()
