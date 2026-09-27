"""Full fine-tuning run: DA V2 (vitb) on the full GAMUS train/val split.

Standalone script (not a notebook) because this runs for hours -- meant to
be launched in the background and checked on via the log file / per-epoch
checkpoints, not watched live. See training/finetune_da_v2_gamus.ipynb for
the interactive smoke-test version this was scaled up from, and its "Next
steps" section for the reasoning behind these settings.

Config chosen from a throughput/VRAM benchmark on the RTX 5060 (8GB):
  - input_size=518 (DA V2's native resolution) at batch_size=2 peaks at
    ~5.95GB VRAM, ~0.29s/step (~12min/epoch on the full 1251-step train set).
  - batch_size=4 at this resolution peaks at >10GB and spills into shared
    system memory under WDDM, collapsing throughput to ~19s/step -- so we
    use batch_size=2 with 2-step gradient accumulation for an effective
    batch of 4 instead of raising batch_size directly.

Usage:
    backend/.venv/Scripts/python.exe training/finetune_da_v2_full.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))

from gamus_dataset import GamusDataset  # noqa: E402
from ssi_loss import scale_shift_invariant_l1, aligned_metrics  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# --- config ---
GAMUS_ROOT = Path(r"D:\Datasets\GAMUS")
BASE_CHECKPOINT = REPO_ROOT / "model" / "depth_anything_v2_vitb.pth"
ENCODER = "vitb"
MODEL_CONFIG = {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]}

INPUT_SIZE = 518
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2  # effective batch size 4
EPOCHS = 12
LR_BACKBONE = 1e-6
LR_HEAD = 1e-5
NUM_WORKERS = 2
GRAD_CLIP_NORM = 1.0

RUN_DIR = REPO_ROOT / "training" / "runs" / "da_v2_gamus_full"
LOG_FILE = RUN_DIR / "train_log.jsonl"
BEST_CHECKPOINT = REPO_ROOT / "model" / "depth_anything_v2_vitb_gamus_best.pth"
LAST_CHECKPOINT = REPO_ROOT / "model" / "depth_anything_v2_vitb_gamus_last.pth"


def log(record: dict) -> None:
    record["ts"] = time.time()
    print(record, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def report_vram(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    log({
        "event": "vram",
        "label": label,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
    })


def run_validation(model, val_loader, device) -> dict[str, float]:
    model.eval()
    agg = {"rmse_m": [], "mae_m": [], "correlation": []}
    with torch.no_grad():
        for img, height, mask in val_loader:
            img, height, mask = img.to(device), height.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                pred = model(img)
            m = aligned_metrics(pred.float(), height, mask)
            for k in agg:
                agg[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "start", "device": str(device), "config": {
        "input_size": INPUT_SIZE, "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": EPOCHS, "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
    }})

    train_ds = GamusDataset(GAMUS_ROOT, "train", input_size=INPUT_SIZE)
    val_ds = GamusDataset(GAMUS_ROOT, "val", input_size=INPUT_SIZE)
    log({"event": "data", "train_pairs": len(train_ds), "val_pairs": len(val_ds)})

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location="cpu"))
    model = model.to(device)
    report_vram(device, "after_load")

    optimizer = torch.optim.AdamW([
        {"params": model.pretrained.parameters(), "lr": LR_BACKBONE},
        {"params": model.depth_head.parameters(), "lr": LR_HEAD},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    baseline = run_validation(model, val_loader, device)
    log({"event": "baseline", **baseline})

    best_rmse = baseline["rmse_m"]
    torch.save(model.state_dict(), BEST_CHECKPOINT)  # baseline is the best we've seen until an epoch beats it

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        losses = []
        optimizer.zero_grad(set_to_none=True)

        for step, (img, height, mask) in enumerate(train_loader):
            img, height, mask = img.to(device), height.to(device), mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                pred = model(img)
                loss = scale_shift_invariant_l1(pred.float(), height, mask) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            losses.append(loss.item() * GRAD_ACCUM_STEPS)

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if epoch == 1 and step == 0:
                report_vram(device, "first_step")
            if step % 100 == 0:
                log({"event": "step", "epoch": epoch, "step": step, "loss": float(np.mean(losses[-20:]))})

        val_metrics = run_validation(model, val_loader, device)
        epoch_time_min = (time.time() - epoch_start) / 60
        log({
            "event": "epoch_end", "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **val_metrics,
            "epoch_time_min": epoch_time_min,
        })

        torch.save(model.state_dict(), LAST_CHECKPOINT)
        if val_metrics["rmse_m"] < best_rmse:
            best_rmse = val_metrics["rmse_m"]
            torch.save(model.state_dict(), BEST_CHECKPOINT)
            log({"event": "new_best", "epoch": epoch, "rmse_m": best_rmse})

    log({"event": "done", "best_rmse_m": best_rmse, "baseline_rmse_m": baseline["rmse_m"]})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    report_vram(device, "after_cleanup")


if __name__ == "__main__":
    main()
