# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera-overlay + BEV visualization for TruckDrive Alpamayo 2 Super inference.

TruckDrive ships pinhole intrinsics (`calibrations/calib_camera_*.json`) and an
extrinsic TF tree (`calibrations/calib_tf_tree_full.json`,
`vehicle -> cab -> camera`) -- a different calibration representation than
PhysicalAI-AV's ray-based ``camera_model``/``sensor_pose`` objects consumed by
``alpamayo2_super.viz_utils``. Notably, the TF tree's own "vehicle" frame is
y-right (FRU) while the pose file (and the model's ego trajectory) is y-left
(FLU) -- a mirror flip, which cannot be represented as the single proper
rotation ``viz_utils`` expects from a ``sensor_pose``. So TruckDrive gets its
own small projector rather than being adapted into that pipeline.

The projection math (``project_ego_to_image``, the FLU/FRU handling) is ported
verbatim from the already-validated implementation in alpamayo-recipes
(``recipes/alpamayo1_5_sft/truckdrive/projection.py``): verified there that the
projected future tracks lanes and painted turn arrows in the correct direction.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_DISPLAY_NAMES
from alpamayo2_super.viz_utils import (
    DRIVING_CAMERA_GRID_LAYOUT,
    TRAJECTORY_EGO_COLOR,
    TRAJECTORY_GROUND_TRUTH_COLOR,
    TRAJECTORY_HISTORY_COLOR,
    TRAJECTORY_PREDICTION_COLOR,
)

# Loader/model poses are FLU (y-left); the TF vehicle frame is FRU (y-right).
_FLU_TO_FRU = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def parse_intrinsics(cam_json: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """Pinhole ``K`` (3x3) and ``(width, height)`` from a parsed calib_camera json."""
    K = np.asarray(cam_json["K"], dtype=np.float64).reshape(3, 3)
    return K, (cam_json["width"], cam_json["height"])


def _tf_to_matrix(entry: dict) -> np.ndarray:
    tr = entry["transform"]["translation"]
    q = entry["transform"]["rotation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [tr["x"], tr["y"], tr["z"]]
    return T


def parse_extrinsics(tf_json: dict, view: str) -> np.ndarray:
    """``T_vehicle_camera`` (4x4) from a parsed ``calib_tf_tree_full.json``."""
    T_v_cab = _tf_to_matrix(tf_json["vehicle_cab"])
    T_cab_cam = _tf_to_matrix(tf_json[f"cab_camera_leopard_{view}"])
    return T_v_cab @ T_cab_cam


def load_view_calibration(backend: Any, scene_id: str, view: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """``(K, T_vehicle_cam, (width, height))`` for one view, via a byte-reader backend.

    ``backend`` is any object exposing ``read_bytes(rel_path) -> bytes``
    (``load_truckdrive._Backend``), so calibration can be pulled from S3
    without staging the scene locally.
    """
    cam = json.loads(
        backend.read_bytes(f"{scene_id}/calibrations/calib_camera_leopard_{view}.json")
    )
    tf = json.loads(backend.read_bytes(f"{scene_id}/calibrations/calib_tf_tree_full.json"))
    K, size = parse_intrinsics(cam)
    return K, parse_extrinsics(tf, view), size


def project_ego_to_image(
    traj_xyz: np.ndarray,
    K: np.ndarray,
    T_vehicle_cam: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame (FLU) trajectory points onto the camera image plane.

    Args:
        traj_xyz: (N, 3) points in the ego frame at t0 (FLU, y-left).
        K: (3, 3) pinhole intrinsics.
        T_vehicle_cam: (4, 4) camera pose in the vehicle frame.
        image_size: optional ``(width, height)``; if given, points outside the
            image are marked invalid.

    Returns:
        uv: (N, 2) pixel coordinates.
        valid: (N,) bool -- in front of the camera (and on-image if size given).
    """
    pts = np.asarray(traj_xyz, dtype=np.float64).reshape(-1, 3) @ _FLU_TO_FRU
    T_cam_v = np.linalg.inv(T_vehicle_cam)
    Xc = (T_cam_v[:3, :3] @ pts.T + T_cam_v[:3, 3:4]).T
    valid = Xc[:, 2] > 0.1
    z = np.where(valid, Xc[:, 2], 1.0)
    u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    if image_size is not None:
        w, h = image_size
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv, valid


def _plot_traj_on_axis(
    ax: plt.Axes,
    traj_xyz: np.ndarray,
    calib: dict[str, Any] | None,
    color: str,
    label: str | None = None,
    linewidth: float = 2.0,
) -> bool:
    """Project + draw one trajectory on a camera-tile axis. Returns whether any point is visible."""
    if calib is None:
        return False
    uv, valid = project_ego_to_image(traj_xyz, calib["K"], calib["T_vehicle_cam"], calib["image_size"])
    if not valid.any():
        return False
    # Only draw contiguous visible runs so a segment that leaves frame doesn't
    # get an across-image chord.
    idx = np.where(valid)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, breaks + 1)
    for i, run in enumerate(runs):
        if len(run) < 2:
            ax.scatter(uv[run, 0], uv[run, 1], color=color, s=8, zorder=5)
            continue
        ax.plot(
            uv[run, 0], uv[run, 1], color=color, linewidth=linewidth, zorder=5,
            label=label if i == 0 else None,
        )
    return True


def plot_truckdrive_inference_result(
    data: dict[str, Any],
    pred_xyz: torch.Tensor,
    extra: dict[str, Any] | None = None,
    output_path: str | None = None,
    json_path: str | None = None,
    model_id: str | None = None,
    seed: int | None = None,
    overlay_sample: int = 0,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Render TruckDrive camera tiles + BEV with GT/predicted trajectory overlays.

    Mirrors ``alpamayo2_super.visualization.plot_inference_result`` in spirit
    (camera grid + BEV, same trajectory color palette) but projects through
    TruckDrive's own pinhole calibration (see module docstring) instead of
    ``viz_utils``'s PhysicalAI-AV camera-model path.
    """
    image_frames = data["image_frames"]  # (N_cam, num_frames, 3, H, W)
    camera_indices = data["camera_indices"].tolist()
    calibration = data.get("truckdrive_calibration", {})
    hist_xyz = data["ego_history_xyz"][0, 0].cpu().numpy()  # (H, 3)
    gt_future_xyz = data["ego_future_xyz"][0, 0].cpu().numpy() if "ego_future_xyz" in data else None
    pred_xy_all = pred_xyz.detach().cpu().numpy()[0, 0]  # (K, T, 2 or 3)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2])
    projection_available = False

    for cam_idx, (row, col) in DRIVING_CAMERA_GRID_LAYOUT.items():
        ax = fig.add_subplot(grid[row, col])
        if cam_idx not in camera_indices:
            ax.set_title(CAMERA_INDICES_TO_DISPLAY_NAMES.get(cam_idx, str(cam_idx)), fontsize=9, color="#888")
            ax.text(
                0.5, 0.5, "no TruckDrive source view", color="#888", fontsize=9, ha="center",
                va="center", transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        slot = camera_indices.index(cam_idx)
        img = image_frames[slot, -1].permute(1, 2, 0).cpu().numpy()  # last (t0) frame
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        ax.imshow(img)
        ax.set_title(CAMERA_INDICES_TO_DISPLAY_NAMES.get(cam_idx, str(cam_idx)), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        calib = calibration.get(cam_idx)

        drew = _plot_traj_on_axis(ax, hist_xyz, calib, TRAJECTORY_HISTORY_COLOR, "history")
        if gt_future_xyz is not None:
            drew = _plot_traj_on_axis(ax, gt_future_xyz, calib, TRAJECTORY_GROUND_TRUTH_COLOR, "ground truth") or drew
        for k in range(pred_xy_all.shape[0]):
            traj = pred_xy_all[k]
            xyz = np.concatenate([traj, np.zeros((traj.shape[0], 1))], axis=1) if traj.shape[-1] == 2 else traj
            lw = 2.5 if k == overlay_sample else 1.0
            drew = (
                _plot_traj_on_axis(
                    ax, xyz, calib, TRAJECTORY_PREDICTION_COLOR,
                    "prediction" if k == 0 else None, linewidth=lw,
                )
                or drew
            )
        projection_available = projection_available or drew
        if calib is None:
            ax.text(
                0.5, 0.06, "no calibration", color="white", fontsize=8, ha="center",
                transform=ax.transAxes, bbox={"facecolor": "black", "alpha": 0.5},
            )

    ax_bev = fig.add_subplot(grid[:, 3])
    ax_bev.plot(-hist_xyz[:, 1], hist_xyz[:, 0], color=TRAJECTORY_HISTORY_COLOR, label="history")
    if gt_future_xyz is not None:
        ax_bev.plot(-gt_future_xyz[:, 1], gt_future_xyz[:, 0], color=TRAJECTORY_GROUND_TRUTH_COLOR, label="ground truth")
    min_ade = None
    if gt_future_xyz is not None:
        gt_xy = gt_future_xyz[:, :2]
        diffs = np.linalg.norm(pred_xy_all[..., :2] - gt_xy[None, ...], axis=-1).mean(-1)
        min_ade = float(diffs.min())
    for k in range(pred_xy_all.shape[0]):
        traj = pred_xy_all[k]
        lw = 2.5 if k == overlay_sample else 1.0
        ax_bev.plot(-traj[:, 1], traj[:, 0], color=TRAJECTORY_PREDICTION_COLOR, linewidth=lw,
                    label="prediction" if k == 0 else None)
    ax_bev.scatter([0], [0], color=TRAJECTORY_EGO_COLOR, s=24, zorder=6, label="ego")
    ax_bev.set_aspect("equal")
    # Straight-driving scenes have a tiny lateral (-y) extent next to a large
    # longitudinal (x) one; under equal aspect that renders as a razor-thin
    # vertical sliver. Pad -y out to a minimum half-width so the panel keeps a
    # sane, legible shape regardless of how straight the road is.
    min_half_width_m = 15.0
    x_min, x_max = ax_bev.get_xlim()
    center = (x_min + x_max) / 2
    half_width = max((x_max - x_min) / 2, min_half_width_m)
    ax_bev.set_xlim(center - half_width, center + half_width)
    ax_bev.set_xlabel("-y (m)")
    ax_bev.set_ylabel("x (m)")
    ax_bev.set_title(f"BEV{f' (minADE={min_ade:.2f}m)' if min_ade is not None else ''}", fontsize=10)
    ax_bev.legend(fontsize=8, loc="upper left")
    ax_bev.grid(True, alpha=0.3)

    cot = None
    if extra is not None and "cot" in extra:
        cot = extra["cot"][0] if isinstance(extra["cot"], (list, tuple)) else extra["cot"]
    title = f"TruckDrive {data.get('scene_id', '?')}  t0={data.get('t0_us', '?')}us"
    fig.suptitle(title, fontsize=12)
    if cot:
        fig.text(0.01, 0.01, f"CoT: {cot}", fontsize=7, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    metadata: dict[str, Any] = {
        "figure_style": "truckdrive",
        "scene_id": data.get("scene_id"),
        "t0_us": data.get("t0_us"),
        "model_id": model_id,
        "seed": seed,
        "min_ade_m": min_ade,
        "projection_available": projection_available,
        "cot": cot,
        "cameras": [CAMERA_INDICES_TO_DISPLAY_NAMES.get(i, str(i)) for i in camera_indices],
    }

    if output_path:
        fig.savefig(output_path, dpi=150)
    if json_path:
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    return fig, metadata
