"""Scale-and-shift-invariant loss for fine-tuning a relative depth model
against metric ground truth (closed-form affine alignment per Ranftl et al.,
"Towards Robust Monocular Depth Estimation", used to train MiDaS/DA V2).

DA V2 predicts an unscaled "closeness" field (larger = nearer the camera).
For nadir aerial imagery, taller objects are physically nearer the airborne
sensor, so AGL height is monotonically consistent with that convention -- but
the model's own scale/offset for a given scene is still arbitrary. Per-image,
we solve the least-squares (a, b) that best maps prediction -> target, then
penalize the aligned residual. This keeps DA V2's output a *relative* field
(matching how app/depth/da_v2_adapter.py and the downstream SRTM/GCP
calibration step already treat it) while still supervising it with real
metric heights.
"""

from __future__ import annotations

import torch


def compute_scale_and_shift(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Closed-form per-sample least-squares fit of target ~= a*pred + b.

    pred, target, mask: (B, H, W). Returns (a, b) each shaped (B,).
    Differentiable w.r.t. pred (all ops below are elementwise/sums).
    """
    mask = mask.float()
    n = mask.sum(dim=(1, 2)).clamp_min(1.0)

    sum_p = (pred * mask).sum(dim=(1, 2))
    sum_t = (target * mask).sum(dim=(1, 2))
    sum_pp = (pred * pred * mask).sum(dim=(1, 2))
    sum_pt = (pred * target * mask).sum(dim=(1, 2))

    denom = (n * sum_pp - sum_p * sum_p).clamp_min(1e-6)
    a = (n * sum_pt - sum_p * sum_t) / denom
    b = (sum_pp * sum_t - sum_p * sum_pt) / denom
    return a, b


def scale_shift_invariant_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 residual after per-sample affine alignment of pred to target."""
    a, b = compute_scale_and_shift(pred, target, mask)
    aligned = a.view(-1, 1, 1) * pred + b.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask.float()
    return diff.sum() / mask.float().sum().clamp_min(1.0)


@torch.no_grad()
def aligned_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """RMSE / MAE / Pearson correlation in meters, after the same alignment
    used by the loss -- consistent with the PRD's validation metrics."""
    a, b = compute_scale_and_shift(pred, target, mask)
    aligned = a.view(-1, 1, 1) * pred + b.view(-1, 1, 1)

    m = mask.bool()
    diff = (aligned - target)[m]
    rmse = torch.sqrt((diff**2).mean()).item()
    mae = diff.abs().mean().item()

    p = aligned[m]
    t = target[m]
    p_c, t_c = p - p.mean(), t - t.mean()
    denom = (p_c.norm() * t_c.norm()).clamp_min(1e-6)
    corr = (p_c * t_c).sum().item() / denom.item()

    return {"rmse_m": rmse, "mae_m": mae, "correlation": corr}
