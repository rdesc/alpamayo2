"""The same displacement + corner-distance metrics used for TruckDrive on
Alpamayo 1.5, ported so Alpamayo 2 Super eval numbers are directly comparable.

Ported from alpamayo-recipes (branch ``feature/lora-and-traj-viz``):
``recipes/alpamayo1_5_sft/truckdrive/horizon_metrics.py`` (displacement
metrics -- verbatim, "single source of truth" for both its offline scorer and
online eval) and ``alpamayo_r1.metrics.metric_api.compute_grouped_corner_distance``
+ ``alpamayo_r1.geometry.coordinates.xyzrot_to_corners`` (corner distance,
inlined here to avoid an ``alpamayo_r1`` dependency).

Metrics are BEV (XY-plane) L2 for displacement, full 3D for corner distance.
For a set of K sampled trajectories per window:

  min_ade/by_t=T  best-of-K ADE, **convention B**: the single trajectory with
                  the lowest error over the FULL predicted horizon is selected
                  once, then its mean error over 0..T is reported (NOT a
                  per-horizon re-selection).
  ade/by_t=T      ADE of sample 0 over 0..T.
  min_fde/by_t=T  best-of-K final displacement: min over K of the L2 at step T.
  fde/by_t=T      final displacement of sample 0 at step T.
  corner_distance mean over timesteps of the min-over-K L2 distance between
                  the 8 corners of the predicted vs. ground-truth ego bounding
                  box (``EGO_VEHICLE_LWH``), averaged over corners.

T is in seconds; the step index is ``round(T / time_step)``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from alpamayo2_super.common.constants import EGO_VEHICLE_LWH

# Horizons (seconds) reported by default. 6.4 s is the full trajectory (16
# history + 64 future steps @ 0.1s); 6.0 s is the literal "6 s" truncation;
# 3.0 s mirrors the Alpamayo 1.5 TruckDrive eval's streamed by_t.
DEFAULT_HORIZONS_S: tuple[float, ...] = (3.0, 6.0, 6.4)

_BOX_CORNER_SIGNS = torch.tensor(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]
)


def xyzrot_to_corners(xyz: torch.Tensor, rot: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """3D bounding-box corners. ``xyz``: ...x3, ``rot``: ...x3x3, ``dims``: ...x3 -> ...x8x3."""
    corns = _BOX_CORNER_SIGNS.to(device=xyz.device, dtype=xyz.dtype)
    corns = dims.unsqueeze(-2) * corns  # scale
    corns = (rot.unsqueeze(-3) @ corns.unsqueeze(-1)).squeeze(-1)  # rotate
    return corns + xyz.unsqueeze(-2)  # translate


def displacement_metrics_from_tensors(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
    time_step: float = 0.1,
) -> dict[str, float]:
    """Compute horizon metrics from batched tensors.

    Args:
        pred_xyz: [B, N, K, T, 3] predicted trajectories (N traj-sets, K samples).
        gt_xyz:   [B, T, 3] ground-truth future.
        horizons_s: horizons in seconds to report.
        time_step: seconds per trajectory step.

    Returns:
        Flat ``{name: float}`` dict, e.g. ``{"min_ade/by_t=3.0": ..., ...}``.
    """
    pred_xyz = pred_xyz.float()
    gt_xyz = gt_xyz.float()
    l2 = torch.linalg.norm((pred_xyz - gt_xyz[:, None, None])[..., :2], dim=-1)  # [B, N, K, T]
    _, _, _, total_steps = l2.shape

    k_best = l2.mean(dim=-1).argmin(dim=2)  # [B, N]
    sel = torch.take_along_dim(l2, k_best[:, :, None, None], dim=2).squeeze(2)  # [B, N, T]

    out: dict[str, float] = {}
    for sec in horizons_s:
        t = int(round(sec / time_step))
        if t < 1 or t > total_steps:
            continue
        idx = t - 1
        out[f"min_ade/by_t={sec:.1f}"] = sel[..., :t].mean(dim=-1).mean().item()
        out[f"ade/by_t={sec:.1f}"] = l2[:, :, 0, :t].mean(dim=-1).mean().item()
        out[f"min_fde/by_t={sec:.1f}"] = l2[..., idx].min(dim=2).values.mean().item()
        out[f"fde/by_t={sec:.1f}"] = l2[:, :, 0, idx].mean().item()
    return out


def corner_distance(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_rot: torch.Tensor,
    dims: Iterable[float] = EGO_VEHICLE_LWH,
) -> float:
    """Mean best-of-K corner distance (meters) between predicted and GT ego boxes.

    Args:
        pred_xyz: [B, N, K, T, 3]
        pred_rot: [B, N, K, T, 3, 3]
        gt_xyz: [B, T, 3]
        gt_rot: [B, T, 3, 3]
        dims: (length, width, height), defaults to Alpamayo's rough ego dims.
    """
    pred_xyz = pred_xyz.float()
    pred_rot = pred_rot.float()
    dims_t = torch.tensor(list(dims), dtype=pred_xyz.dtype, device=pred_xyz.device)

    corner_pred = xyzrot_to_corners(pred_xyz, pred_rot, dims_t.view(1, 1, 1, 1, 3))
    corner_gt = xyzrot_to_corners(gt_xyz.float(), gt_rot.float(), dims_t.view(1, 1, 3))
    distance = (corner_pred - corner_gt[:, None, None]).norm(dim=-1)  # [B, N, K, T, 8]
    distance = distance.min(dim=2)[0].mean(dim=(2, 3))  # [B, N] (min over K, mean over T,corners)
    return distance.mean(dim=1).mean().item()  # mean over N groups (N=1 here), then batch
