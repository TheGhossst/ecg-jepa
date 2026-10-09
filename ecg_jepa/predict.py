"""Score one ECG with the two-branch head saved beside the encoder.

The recording is millivolts on leads I, II, V1–V6. The encoder still sees each
lead z-scored. The amplitude branch sees peak, standard deviation, and
peak-to-peak, scaled with the training-fold statistics stored in the head.
The encoder file is loaded and not updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecg_jepa.amplitude import predict_scores
from ecg_jepa.data.ptbxl import resolve_ptbxl_root
from ecg_jepa.hyp_voltage import read_millivolts

DEFAULT_HEAD = Path("checkpoints/run1/two_branch.pt")


def load_recording(data_dir: str, record: str, head_path: Path) -> np.ndarray:
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    root = resolve_ptbxl_root(data_dir)
    return read_millivolts(root, record, int(blob["signal_length"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five superclass scores for one ECG")
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--record", default=None, help="filename_lr relative to the PTB-XL root")
    parser.add_argument("--npy", type=Path, default=None, help="millivolt array shaped (8, time)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.record is None) == (args.npy is None):
        raise SystemExit("pass exactly one of --record or --npy")
    if not args.head.is_file():
        raise SystemExit(
            f"missing {args.head}. Train it with "
            "python -m ecg_jepa.amplitude --data-dir dataset --seeds 0,1,2,3,4"
        )
    if args.npy is not None:
        signal = np.load(args.npy)
    else:
        signal = load_recording(args.data_dir, args.record, args.head)
    scores = predict_scores(signal, args.head, args.device)
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
