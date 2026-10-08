"""Tiny JEPA pretrain loop.

AdamW, cosine decay, and a warmup scaled down from the paper's 5 epochs.
Logs smooth-L1 and the cross-batch standard deviation of target embeddings.
Fold 9 (or a synthetic holdout) is monitored. Fold 10 is never loaded.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from ecg_jepa.config import Config
from ecg_jepa.data.ptbxl import make_dataloader
from ecg_jepa.models.jepa import JEPA, ema_momentum, ema_update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    span = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / span, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    model: JEPA,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg.__dict__,
            "loss": loss,
        },
        path,
    )


@torch.no_grad()
def evaluate(model: JEPA, loader: torch.utils.data.DataLoader, device: torch.device, max_batches: int) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    stds: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model(batch.to(device))
        losses.append(out.loss.item())
        stds.append(out.embed_std.item())
    model.train()
    if not losses:
        return float("nan"), float("nan")
    return sum(losses) / len(losses), sum(stds) / len(stds)


def run(
    cfg: Config,
    data_dir: str | None = None,
    synthetic: bool = False,
    max_steps: int | None = None,
    device: str | None = None,
    val_batches: int = 8,
) -> Path:
    set_seed(cfg.seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = torch_device.type == "cuda"
    train_loader = make_dataloader(cfg, "train", synthetic, data_dir, pin_memory=pin_memory)
    val_loader = make_dataloader(cfg, "val", synthetic, data_dir, pin_memory=pin_memory)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("training loader is empty; lower batch_size or add records")
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    model = JEPA(cfg).to(torch_device)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    ckpt_dir = Path(cfg.ckpt_dir)
    last_path = ckpt_dir / "last.pt"
    step = 0
    last_loss = float("nan")
    stopped_early = False
    model.train()

    for epoch in range(cfg.epochs):
        for batch in train_loader:
            if max_steps is not None and step >= max_steps:
                stopped_early = True
                break
            lr = learning_rate(step, total_steps, warmup_steps, cfg.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            out = model(batch.to(torch_device))
            out.loss.backward()
            optimizer.step()
            momentum = ema_momentum(step, total_steps, cfg.ema_start, cfg.ema_end)
            ema_update(model.target_encoder, model.context_encoder, momentum)
            last_loss = out.loss.item()
            if step % cfg.log_every == 0:
                print(
                    f"step {step} epoch {epoch} loss {last_loss:.4f} "
                    f"embed_std {out.embed_std.item():.4f} lr {lr:.6e} ema {momentum:.6f}",
                    flush=True,
                )
            step += 1
        if stopped_early:
            break
        val_loss, val_std = evaluate(model, val_loader, torch_device, val_batches)
        print(
            f"val epoch {epoch} loss {val_loss:.4f} embed_std {val_std:.4f}",
            flush=True,
        )
        save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, step, cfg, last_loss)

    if stopped_early:
        val_loss, val_std = evaluate(model, val_loader, torch_device, val_batches)
        print(
            f"val end loss {val_loss:.4f} embed_std {val_std:.4f}",
            flush=True,
        )
    save_checkpoint(last_path, model, optimizer, step, cfg, last_loss)
    return last_path


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Pretrain a tiny ECG JEPA encoder")
    parser.add_argument("--data-dir", default=None, help="PTB-XL root with ptbxl_database.csv")
    parser.add_argument("--synthetic", action="store_true", help="train on synthetic sines, no PTB-XL")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ckpt-dir", default=cfg.ckpt_dir)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
    )
    if not args.synthetic and not args.data_dir:
        raise SystemExit("pass --data-dir PATH_TO_PTBXL or --synthetic")
    path = run(
        cfg,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
