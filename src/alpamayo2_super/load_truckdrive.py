"""Load a TruckDrive (Torc Robotics) scene sample for Alpamayo 2 Super inference.

TruckDrive has no first-party Alpamayo integration; this module builds the same
sample-dict contract ``load_physical_aiavdataset`` produces for PhysicalAI-AV
(``image_frames``, ``camera_indices``, ``ego_history_xyz/rot``,
``ego_future_xyz/rot``, ...), sourced instead from a TruckDrive scene's
per-view camera JPEGs and its single ``poses/gt_trajectory.txt``.

Ported from the TruckDrive fine-tuning work in ``alpamayo-recipes``
(``src/alpamayo/data/truckdrive.py``, branch ``feature/lora-and-traj-viz``),
which validated this pose math and camera-slot mapping for Alpamayo 1.5.
Alpamayo 2 Super's camera-slot names/indices
(``alpamayo2_super.common.constants.CAMERA_NAMES_TO_INDICES``) are identical,
so the mapping carries over unchanged.

Key findings carried over from that work (see alpamayo-recipes'
``docs/truckdrive_finetuning.md`` for the full derivation):

- TruckDrive's ``gt_trajectory`` poses are already FLU (x-forward, y-left,
  z-up), same as Alpamayo -- no lateral flip needed for the ego trajectory.
- Ego heading must be derived from the position track (the tangent), not the
  pose quaternions -- the quaternions encode body orientation, which only
  equals travel heading during forward driving (~cm error vs. 0.5-9 m error).
- TruckDrive cameras run at ~5 Hz, slower than the 10 Hz trajectory grid, so
  the requested per-frame image timestamps are snapped to the nearest
  available frame rather than assumed to exist exactly. Frame spacing
  (``image_time_step``) defaults to 0.2 s -- matching the ~5 Hz camera rate
  and the validated ``sft_truckdrive.yaml`` config -- NOT the 0.1 s trajectory
  grid; at 0.1 s spacing the "4 frames" mostly snap to the same 1-2 real
  frames, which looked superficially fine (4 distinct timestamps) without
  actually giving the model well-separated temporal context.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from alpamayo2_super.common.constants import (
    CAMERA_NAMES_TO_INDICES,
    CROSS_LEFT_CAMERA_NAME,
    CROSS_RIGHT_CAMERA_NAME,
    FRONT_WIDE_CAMERA_NAME,
    REAR_LEFT_CAMERA_NAME,
    REAR_RIGHT_CAMERA_NAME,
)

DEFAULT_DATA_ROOT = "s3://torc-data/datasets/TruckDrivePublic"

# View -> Alpamayo camera-slot mapping. This is the exact 5-camera set
# validated for TruckDrive on Alpamayo 1.5 (alpamayo-recipes'
# docs/rod_inference_truckdrive.md / recipes/alpamayo1_5_sft/configs/
# sft_truckdrive.yaml, `camera_views`) -- deliberately NOT alpamayo-recipes'
# full 15-view->slot heuristic table (`DEFAULT_VIEW_TO_ALPAMAYO` in
# src/alpamayo/data/truckdrive.py), which lists every view that *could* fill a
# slot rather than the one actually used in practice. Note this only fills 5
# of the 6 camera slots in Alpamayo 2 Super's "trajectory" task profile
# (ids 0,1,2,3,5,6) -- FRONT_TELE (id 6) has no validated TruckDrive source
# view and is left unfilled.
DEFAULT_VIEW_TO_ALPAMAYO: dict[str, str] = {
    "forward_center_medium": FRONT_WIDE_CAMERA_NAME,
    "sideward_left_front_wide": CROSS_LEFT_CAMERA_NAME,
    "sideward_right_front_wide": CROSS_RIGHT_CAMERA_NAME,
    "rearward_left_bottom_medium": REAR_LEFT_CAMERA_NAME,
    "rearward_right_bottom_medium": REAR_RIGHT_CAMERA_NAME,
}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Official TruckDrive devkit split keys (metainfo.json). The three training
# lists differ only in 3-D *box* annotation style, which trajectory training
# doesn't use -- all three are trainable. Val/test hold out whole collection
# batches (scene groups), the devkit's leak protection. Mirrors
# alpamayo-recipes' ``alpamayo.data.truckdrive._METAINFO_TRAIN_KEYS`` /
# ``_scenes_from_metainfo`` exactly, so ``split="train"`` here resolves to the
# identical scene set used to train Alpamayo 1.5 on TruckDrive.
_METAINFO_TRAIN_KEYS = (
    "sequentially_labelled_training_scenes",
    "non_sequentially_labelled_training_scenes",
    "unlabelled_training_scenes",
)


def scenes_from_metainfo(path: str, split: str) -> list[str]:
    """Resolve scene IDs for ``split`` from the devkit's official metainfo.json.

    ``split="train"`` is the union of all three training-scene lists (dedup'd,
    sorted); ``"val"``/``"test"`` are ``validation_scenes``/``test_scenes``.
    """
    import json

    with open(path) as f:
        meta = json.load(f)
    if split == "train":
        ids = [s for k in _METAINFO_TRAIN_KEYS for s in meta[k]]
    elif split == "val":
        ids = list(meta["validation_scenes"])
    elif split == "test":
        ids = list(meta["test_scenes"])
    else:
        raise ValueError(f"split={split!r} invalid with metainfo (use 'train'/'val'/'test')")
    return sorted(set(ids))


# --------------------------------------------------------------------------- #
# Read backends
# --------------------------------------------------------------------------- #
class _Backend:
    def list_files(self, rel_dir: str) -> list[str]:
        raise NotImplementedError

    def read_bytes(self, rel_path: str) -> bytes:
        raise NotImplementedError


class _LocalBackend(_Backend):
    """Reads from a local directory root (staged files or a mount)."""

    def __init__(self, root: str) -> None:
        self._root = root.rstrip("/")

    def _abs(self, rel: str) -> str:
        return os.path.join(self._root, rel.strip("/"))

    def list_files(self, rel_dir: str) -> list[str]:
        try:
            return sorted(e.name for e in os.scandir(self._abs(rel_dir)) if e.is_file())
        except OSError:
            return []

    def read_bytes(self, rel_path: str) -> bytes:
        with open(self._abs(rel_path), "rb") as f:
            return f.read()


class _S3Backend(_Backend):
    """Reads objects from S3 via boto3."""

    def __init__(self, bucket: str, prefix: str, region: str = "us-east-1") -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def list_files(self, rel_dir: str) -> list[str]:
        rel_dir = rel_dir.strip("/")
        prefix = f"{self._prefix}/{rel_dir}/" if rel_dir else f"{self._prefix}/"
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        names: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter="/"):
            for obj in page.get("Contents", []):
                base = obj["Key"][len(prefix) :]
                if base and "/" not in base:
                    names.append(base)
        return sorted(names)

    def read_bytes(self, rel_path: str) -> bytes:
        key = f"{self._prefix}/{rel_path.strip('/')}"
        resp = self._get_client().get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()


def _make_backend(backend: str, data_root: str, region: str) -> _Backend:
    if backend == "local":
        return _LocalBackend(data_root)
    if backend == "s3":
        uri = data_root[len("s3://") :] if data_root.startswith("s3://") else data_root
        bucket, _, prefix = uri.partition("/")
        return _S3Backend(bucket=bucket, prefix=prefix, region=region)
    raise ValueError(f"unknown backend {backend!r} (use 'local' or 's3')")


# --------------------------------------------------------------------------- #
# Pose file parsing + interpolation
# --------------------------------------------------------------------------- #
def _sync_key(basename: str) -> str:
    """``0034_200027227.jpg`` -> ``0034`` (the temporal alignment key)."""
    return basename.split("_", 1)[0]


class _ScenePoses:
    """Parsed ego trajectory for one scene, with time interpolation."""

    def __init__(self, text: str) -> None:
        sync_keys: list[str] = []
        t: list[float] = []
        xyz: list[list[float]] = []
        quat: list[list[float]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("refined"):
                continue
            f = line.split()
            if len(f) < 9:
                continue
            try:
                row_t = float(f[1])
                row_xyz = [float(f[2]), float(f[3]), float(f[4])]
                row_quat = [float(f[5]), float(f[6]), float(f[7]), float(f[8])]  # x,y,z,w
            except ValueError:
                continue  # header/junk line
            sync_keys.append(f[0])
            t.append(row_t)
            xyz.append(row_xyz)
            quat.append(row_quat)

        if len(t) < 2:
            raise ValueError("pose file has fewer than 2 valid rows")

        self.sync_keys = sync_keys
        self.t = np.asarray(t, dtype=np.float64)
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self._rot = Rotation.from_quat(np.asarray(quat, dtype=np.float64))
        self._slerp = Slerp(self.t, self._rot)
        self.key_to_t = {k: self.t[i] for i, k in enumerate(sync_keys)}

        # Per-pose reverse signal: speed, and cos(angle) between travel direction
        # (position tangent) and body-forward (quaternion x-axis). cos < 0 means
        # the vehicle is moving backward relative to where it points (reversing).
        # Ported from alpamayo-recipes' _ScenePoses (same math, used to match its
        # val-window enumeration exactly).
        vel = np.gradient(self.xyz[:, :2], self.t, axis=0)  # (N, 2) m/s
        self.speed = np.linalg.norm(vel, axis=1)
        vel_dir = vel / (self.speed[:, None] + 1e-9)
        body_fwd = self._rot.as_matrix()[:, :2, 0]  # body x-axis in xy
        body_fwd /= np.linalg.norm(body_fwd, axis=1, keepdims=True) + 1e-9
        self.cos_vel_body = (vel_dir * body_fwd).sum(axis=1)

    def is_forward_window(
        self,
        t0: float,
        horizon: float,
        min_speed: float,
        reverse_angle_deg: float,
        max_reverse_fraction: float,
    ) -> bool:
        """True if the future span [t0, t0+horizon] is forward driving (see
        alpamayo-recipes' ``_ScenePoses.is_forward_window`` for the derivation)."""
        sel = (self.t >= t0) & (self.t <= t0 + horizon) & (self.speed >= min_speed)
        moving = self.cos_vel_body[sel]
        if moving.size == 0:
            return True
        reverse = moving < np.cos(np.radians(reverse_angle_deg))
        return reverse.mean() <= max_reverse_fraction

    def max_speed(self, t_start: float, t_end: float) -> float:
        """Peak pose speed (m/s) over ``[t_start, t_end]`` (``inf`` if no pose falls inside)."""
        sel = (self.t >= t_start) & (self.t <= t_end)
        return float(self.speed[sel].max()) if sel.any() else float("inf")

    @property
    def t_min(self) -> float:
        return float(self.t[0])

    @property
    def t_max(self) -> float:
        return float(self.t[-1])

    def sample(self, timestamps: np.ndarray) -> np.ndarray:
        """Interpolate ego position at ``timestamps`` -> xyz (N, 3)."""
        ts = np.clip(timestamps, self.t_min, self.t_max)
        return np.stack([np.interp(ts, self.t, self.xyz[:, d]) for d in range(3)], axis=-1)


def _heading_from_positions(
    xyz: np.ndarray, time_step: float, min_speed_mps: float
) -> np.ndarray:
    """Derive travel heading (yaw) from the position track (see module docstring)."""
    d = np.gradient(xyz[:, :2], axis=0)
    speed = np.linalg.norm(d, axis=1) / time_step
    yaw = np.arctan2(d[:, 1], d[:, 0])

    good = speed >= min_speed_mps
    if good.any():
        last = -1
        for i in range(len(good)):
            if good[i]:
                last = i
            elif last >= 0:
                yaw[i] = yaw[last]
        first = int(np.argmax(good))
        yaw[:first] = yaw[first]
    else:
        yaw[:] = 0.0
    return np.unwrap(yaw)


def _rot_z(yaw: np.ndarray) -> np.ndarray:
    """Yaw angles (N,) -> planar rotation matrices (N, 3, 3)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _standstill_snap(
    poses: _ScenePoses,
    t0: float,
    hist_start_offset: float,
    fut_end_offset: float,
    threshold: float,
    hist_xyz: np.ndarray,
    hist_rot: np.ndarray,
    fut_xyz: np.ndarray,
    fut_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replace parked windows with exact zeros/identity (ported from
    alpamayo-recipes' ``TruckDriveDataset._standstill_snap`` -- see there for
    the full derivation of why this matters for the accel-token targets).
    All-or-nothing over the FULL window (history AND future both parked)."""
    if poses.max_speed(t0 + hist_start_offset, t0 + fut_end_offset) >= threshold:
        return hist_xyz, hist_rot, fut_xyz, fut_rot
    return (
        np.zeros_like(hist_xyz),
        np.broadcast_to(np.eye(3), hist_rot.shape).copy(),
        np.zeros_like(fut_xyz),
        np.broadcast_to(np.eye(3), fut_rot.shape).copy(),
    )


# --------------------------------------------------------------------------- #
# Camera frame loading
# --------------------------------------------------------------------------- #
def _load_view_frames(
    backend: _Backend,
    scene_id: str,
    view: str,
    poses: _ScenePoses,
    img_times: np.ndarray,
) -> tuple[list[Image.Image], np.ndarray]:
    """Load the nearest available frame to each of ``img_times`` for one view."""
    files = [f for f in backend.list_files(f"{scene_id}/camera/leopard/{view}/images") if f.lower().endswith(_IMAGE_EXTS)]
    if not files:
        raise RuntimeError(f"no images found for {scene_id}/{view}")
    timed = sorted(
        (poses.key_to_t[k], name)
        for name in files
        if (k := _sync_key(name)) in poses.key_to_t
    )
    if not timed:
        raise RuntimeError(f"no {scene_id}/{view} frames join the pose timeline (SYNC_KEY mismatch)")
    times = np.array([t for t, _ in timed], dtype=np.float64)
    names = [n for _, n in timed]

    sel = np.searchsorted(times, img_times)
    sel = np.clip(sel, 0, len(times) - 1)
    for j, s in enumerate(sel):
        if s > 0 and abs(times[s - 1] - img_times[j]) < abs(times[s] - img_times[j]):
            sel[j] = s - 1

    frames: list[Image.Image] = []
    frame_ts: list[float] = []
    for s in sel:
        rel = f"{scene_id}/camera/leopard/{view}/images/{names[s]}"
        frames.append(Image.open(BytesIO(backend.read_bytes(rel))).convert("RGB"))
        frame_ts.append(float(times[s]))
    return frames, np.asarray(frame_ts, dtype=np.float64)


def _ego_traj_from_poses(
    poses: _ScenePoses,
    scene_id: str,
    t0_s: float,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    min_speed_mps: float,
    standstill_snap_mps: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pose track -> ego-frame ``(hist_xyz, hist_rot, fut_xyz, fut_rot)``.

    Pose-only (no image loads) -- exactly the trajectory ``load_truckdrive_sample``
    feeds the model, factored out so ``prefilter_tok_recon`` (index-build time,
    no images) can reuse it. Mirrors alpamayo-recipes'
    ``TruckDriveDataset._ego_traj``. Shapes: xyz ``(T, 3)``, rot ``(T, 3, 3)``.
    """
    hist_offsets = np.array(
        [-(num_history_steps - 1 - i) * time_step for i in range(num_history_steps)]
    )
    fut_offsets = np.array([(i + 1) * time_step for i in range(num_future_steps)])
    all_offsets = np.concatenate([hist_offsets, fut_offsets])
    t_window = t0_s + all_offsets

    if t_window[0] < poses.t_min or t_window[-1] > poses.t_max:
        raise ValueError(
            f"t0_s={t0_s} leaves the pose timeline [{poses.t_min:.2f}, {poses.t_max:.2f}]s "
            f"for scene {scene_id} once history/future margins are applied"
        )

    xyz_w = poses.sample(t_window)
    yaw_w = _heading_from_positions(xyz_w, time_step, min_speed_mps)
    rot_w = _rot_z(yaw_w)

    H = num_history_steps
    t0_xyz = xyz_w[H - 1]
    t0_rot_inv = rot_w[H - 1].T  # orthonormal -> inverse is transpose
    xyz_local = (xyz_w - t0_xyz) @ t0_rot_inv.T
    rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rot_w)
    hist_xyz, fut_xyz = xyz_local[:H], xyz_local[H:]
    hist_rot, fut_rot = rot_local[:H], rot_local[H:]

    if standstill_snap_mps is not None:
        hist_xyz, hist_rot, fut_xyz, fut_rot = _standstill_snap(
            poses, t0_s, hist_offsets[0], fut_offsets[-1], standstill_snap_mps,
            hist_xyz, hist_rot, fut_xyz, fut_rot,
        )
    return hist_xyz, hist_rot, fut_xyz, fut_rot


def prefilter_tok_recon(
    windows: Sequence[tuple[str, float]],
    tokenizer: Any,
    data_root: str = DEFAULT_DATA_ROOT,
    backend: str = "s3",
    region: str = "us-east-1",
    pose_file: str = "poses/gt_trajectory.txt",
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    min_speed_mps: float = 0.5,
    standstill_snap_mps: float | None = 0.5,
    max_xy_m: float = 1.0,
    reduce: str = "mean",
    chunk_size: int = 512,
    verbose: bool = True,
) -> list[tuple[str, float]]:
    """Drop windows whose future round-trips badly through the frozen future
    trajectory tokenizer (encode -> decode XY error above ``max_xy_m``).

    Pose-only (no image loads); the encode/decode runs vectorised in chunks on
    CPU. Mirrors alpamayo-recipes' ``TruckDriveDataset._prefilter_tok_recon``
    exactly (same ``tokenizer.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)`` ->
    ``tokenizer.decode(hist_xyz, hist_rot, tokens)`` contract, which Alpamayo 2
    Super's ``DeltaTrajectoryTokenizer`` also implements), so passing this
    ``max_xy_m``/``reduce`` matches the same filter ``sft_truckdrive.yaml``
    applies to Alpamayo 1.5's TRAIN split (never applied to val -- dropping val
    windows would renumber the split and invalidate any hardcoded indices).

    Args:
        windows: ``(scene_id, t0_s)`` pairs, e.g. from ``enumerate_val_windows``.
        tokenizer: A future-trajectory tokenizer instance (e.g.
            ``hydra.utils.instantiate(model_config.future_traj_tokenizer_cfg)``) --
            deterministic bin-based quantization, no checkpoint weights needed.
    """
    if reduce not in ("mean", "max", "median", "p95"):
        raise ValueError(f"reduce must be mean/max/median/p95, got {reduce!r}")

    be = _make_backend(backend, data_root, region)
    pose_cache: dict[str, _ScenePoses | None] = {}

    def _poses(scene_id: str) -> _ScenePoses | None:
        if scene_id not in pose_cache:
            try:
                text = be.read_bytes(f"{scene_id}/{pose_file}").decode("utf-8")
                pose_cache[scene_id] = _ScenePoses(text)
            except Exception:  # noqa: BLE001 - skip unreadable scenes
                pose_cache[scene_id] = None
        return pose_cache[scene_id]

    kept: list[tuple[str, float]] = []
    errs: list[float] = []
    n_fail = 0
    buf: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def _flush() -> None:
        nonlocal n_fail
        if not buf:
            return
        hx = torch.from_numpy(np.stack([b[2] for b in buf])).float()
        hr = torch.from_numpy(np.stack([b[3] for b in buf])).float()
        fx = torch.from_numpy(np.stack([b[4] for b in buf])).float()
        fr = torch.from_numpy(np.stack([b[5] for b in buf])).float()
        try:
            with torch.no_grad():
                tokens = tokenizer.encode(hist_xyz=hx, hist_rot=hr, fut_xyz=fx, fut_rot=fr)
                rec, _, _ = tokenizer.decode(hist_xyz=hx, hist_rot=hr, tokens=tokens)
        except Exception:  # noqa: BLE001 - fail-open: a tokenizer failure is a
            # sign of a pathological window, but don't silently nuke a whole
            # chunk over it -- keep them and count so it's visible in logs.
            n_fail += len(buf)
            kept.extend((b[0], b[1]) for b in buf)
            return
        e = torch.linalg.norm(rec[..., :2] - fx[..., :2], dim=-1)  # (B, F)
        if reduce == "mean":
            red = e.mean(dim=-1)
        elif reduce == "max":
            red = e.amax(dim=-1)
        elif reduce == "median":
            red = e.median(dim=-1).values
        else:
            red = torch.quantile(e, 0.95, dim=-1)
        for b, r in zip(buf, red.tolist()):
            errs.append(r)
            if r <= max_xy_m:
                kept.append((b[0], b[1]))

    for scene_id, t0_s in windows:
        poses = _poses(scene_id)
        if poses is None:
            continue
        try:
            hx, hr, fx, fr = _ego_traj_from_poses(
                poses, scene_id, t0_s, num_history_steps, num_future_steps,
                time_step, min_speed_mps, standstill_snap_mps,
            )
        except ValueError:
            continue
        buf.append((scene_id, t0_s, hx, hr, fx, fr))
        if len(buf) >= chunk_size:
            _flush()
            buf.clear()
    _flush()

    if verbose:
        n_drop = len(windows) - len(kept)
        e_arr = np.asarray(errs, dtype=np.float64)
        stats = ""
        if e_arr.size:
            stats = (
                f" [{reduce} recon err: median={np.median(e_arr):.3f}m "
                f"p95={np.quantile(e_arr, 0.95):.3f}m max={e_arr.max():.3f}m]"
            )
        print(
            f"prefilter_tok_recon: dropped {n_drop}/{len(windows)} windows "
            f"(threshold {reduce}={max_xy_m:.3f}m){stats}"
        )
        if n_fail:
            print(
                f"prefilter_tok_recon: {n_fail} windows errored during "
                "encode/decode and were kept (fail-open)"
            )
    return kept


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_truckdrive_sample(
    scene_id: str,
    t0_s: float,
    data_root: str = DEFAULT_DATA_ROOT,
    backend: str = "s3",
    region: str = "us-east-1",
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    num_frames: int = 4,
    time_step: float = 0.1,
    image_time_step: float | None = 0.2,
    view_to_alpamayo: dict[str, str] | None = None,
    pose_file: str = "poses/gt_trajectory.txt",
    min_speed_mps: float = 0.5,
    image_size: tuple[int, int] | None = (1920, 1080),
    include_calibration: bool = True,
    standstill_snap_mps: float | None = None,
) -> dict[str, Any]:
    """Load one TruckDrive ``(scene_id, t0_s)`` window for Alpamayo 2 Super inference.

    Mirrors ``load_physical_aiavdataset``'s return contract. Args:
        scene_id: TruckDrive scene, e.g. ``"scene_28_1"``.
        t0_s: Scene-relative timestamp (seconds) to sample the trajectory at.
        data_root: Local dir (``backend="local"``) or ``s3://bucket/prefix``
            (``backend="s3"``) containing ``scene_*`` directories.
        view_to_alpamayo: TruckDrive view -> Alpamayo camera-slot name.
            Defaults to ``DEFAULT_VIEW_TO_ALPAMAYO``, the 5 views validated for
            TruckDrive on Alpamayo 1.5 (camera ids 0,1,2,3,5 -- FRONT_TELE/id 6
            of Alpamayo 2's 6-camera "trajectory" task profile is unfilled).
        image_time_step: Seconds between the ``num_frames`` camera frames fed
            per view (they end at ``t0``). Defaults to 0.2 s (matching
            TruckDrive's ~5 Hz camera rate and ``sft_truckdrive.yaml``), so with
            the default ``num_frames=4`` the frames are at
            ``t0-0.6, t0-0.4, t0-0.2, t0`` -- NOT the 0.1 s ``time_step`` used
            for the trajectory grid.
        image_size: ``(width, height)`` all camera views are resized to before
            stacking into one tensor (views have different native lens
            resolutions, e.g. 3848x2168 for the wide views). Defaults to
            1920x1080, matching the PhysicalAI-AV camera resolution Alpamayo 2
            Super's release checkpoint was validated against. Pass ``None`` to
            use the first-loaded view's native size instead.
        include_calibration: Best-effort load of pinhole intrinsics + vehicle
            extrinsics per view (rescaled to ``image_size``), for camera-overlay
            visualization (see ``truckdrive_viz.py``). Stored under
            ``data["truckdrive_calibration"]``.
        standstill_snap_mps: If set, zero the history+future ego trajectory
            (the ground truth used for scoring) when the window is parked the
            whole time (peak pose speed under this threshold across BOTH
            history and future) -- see ``_standstill_snap``. ``None`` (default)
            is off. ``sft_truckdrive.yaml`` uses 0.5; pass that to reproduce its
            val-set ground truth exactly.

    Returns a dict with the same core keys as ``load_physical_aiavdataset``:
    ``image_frames``, ``camera_indices``, ``camera_names``, ``ego_history_xyz``,
    ``ego_history_rot``, ``ego_future_xyz``, ``ego_future_rot``,
    ``relative_timestamps``, ``absolute_timestamps``, ``t0_us``, ``clip_id``.
    """
    view_to_alpamayo = dict(view_to_alpamayo or DEFAULT_VIEW_TO_ALPAMAYO)
    for view, slot in view_to_alpamayo.items():
        if slot not in CAMERA_NAMES_TO_INDICES:
            raise ValueError(f"unknown Alpamayo camera slot {slot!r} for view {view!r}")

    be = _make_backend(backend, data_root, region)
    pose_text = be.read_bytes(f"{scene_id}/{pose_file}").decode("utf-8")
    poses = _ScenePoses(pose_text)

    hist_xyz, hist_rot, fut_xyz, fut_rot = _ego_traj_from_poses(
        poses, scene_id, t0_s, num_history_steps, num_future_steps, time_step,
        min_speed_mps, standstill_snap_mps,
    )

    image_time_step = image_time_step if image_time_step is not None else time_step
    img_offsets = np.array([-(num_frames - 1 - i) * image_time_step for i in range(num_frames)])
    img_times = t0_s + img_offsets

    frames_per_view: dict[str, list[Image.Image]] = {}
    ts_per_view: dict[str, np.ndarray] = {}
    for view in view_to_alpamayo:
        frames, ts = _load_view_frames(be, scene_id, view, poses, img_times)
        frames_per_view[view] = frames
        ts_per_view[view] = ts

    target_size = image_size or frames_per_view[next(iter(frames_per_view))][0].size  # (W, H)

    image_frames_list: list[torch.Tensor] = []
    camera_indices_list: list[int] = []
    abs_ts_list: list[torch.Tensor] = []
    calibration: dict[int, dict[str, Any]] = {}
    for view, slot in view_to_alpamayo.items():
        native_size = frames_per_view[view][0].size  # (W, H)
        resized = [
            im if im.size == target_size else im.resize(target_size, Image.BICUBIC)
            for im in frames_per_view[view]
        ]
        arrs = [torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in resized]  # (3,H,W)
        image_frames_list.append(torch.stack(arrs, dim=0))  # (num_frames, 3, H, W)
        cam_idx = CAMERA_NAMES_TO_INDICES[slot]
        camera_indices_list.append(cam_idx)
        abs_ts_list.append(torch.from_numpy(ts_per_view[view]))

        if include_calibration:
            try:
                from alpamayo2_super.truckdrive_viz import load_view_calibration

                K, T_vehicle_cam, (w0, h0) = load_view_calibration(be, scene_id, view)
                sx, sy = target_size[0] / w0, target_size[1] / h0
                K_scaled = K.copy()
                K_scaled[0, 0] *= sx
                K_scaled[0, 2] *= sx
                K_scaled[1, 1] *= sy
                K_scaled[1, 2] *= sy
                calibration[cam_idx] = {
                    "K": K_scaled,
                    "T_vehicle_cam": T_vehicle_cam,
                    "image_size": target_size,
                    "view": view,
                }
            except Exception:  # noqa: BLE001 - calibration is best-effort
                pass

    image_frames = torch.stack(image_frames_list, dim=0)  # (N_cam, num_frames, 3, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    abs_ts = torch.stack(abs_ts_list, dim=0)  # (N_cam, num_frames), seconds

    order = torch.argsort(camera_indices)
    image_frames = image_frames[order]
    camera_indices = camera_indices[order]
    abs_ts = abs_ts[order]
    camera_names = [
        name for name, idx in sorted(CAMERA_NAMES_TO_INDICES.items(), key=lambda kv: kv[1])
        if idx in camera_indices.tolist()
    ]

    camera_tmin = float(abs_ts.min())
    relative_timestamps = (abs_ts - camera_tmin).float()
    absolute_timestamps = (abs_ts * 1e6).long()  # seconds -> microseconds, matches PAI convention

    t0_us = int(round(t0_s * 1e6))
    ego_history_tvals = torch.arange(-num_history_steps + 1, 1, dtype=torch.float32) * time_step
    ego_future_tvals = torch.arange(1, num_future_steps + 1, dtype=torch.float32) * time_step

    data: dict[str, Any] = {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "camera_names": camera_names,
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([t0_us], dtype=torch.int64),
        "ego_t0_relative": torch.tensor(
            [(t0_us - camera_tmin * 1e6) * 1e-6], dtype=torch.float32
        ),
        "ego_t0_frame_idx": torch.tensor([num_frames - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": ego_history_tvals,
        "ego_history_xyz": torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0),
        "ego_future_tvals": ego_future_tvals,
        "ego_future_xyz": torch.from_numpy(fut_xyz).float().unsqueeze(0).unsqueeze(0),
        "ego_future_rot": torch.from_numpy(fut_rot).float().unsqueeze(0).unsqueeze(0),
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "camera_tmin": camera_tmin,
        "t0_us": t0_us,
        "clip_id": scene_id,
        "scene_id": scene_id,
    }
    if include_calibration and calibration:
        data["truckdrive_calibration"] = calibration
    return data


# --------------------------------------------------------------------------- #
# Val-window enumeration (matches TruckDriveDataset._build_index)
# --------------------------------------------------------------------------- #
def enumerate_val_windows(
    scene_ids: Sequence[str],
    data_root: str = DEFAULT_DATA_ROOT,
    backend: str = "s3",
    region: str = "us-east-1",
    view_to_alpamayo: dict[str, str] | None = None,
    pose_file: str = "poses/gt_trajectory.txt",
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    num_frames: int = 4,
    time_step: float = 0.1,
    image_time_step: float = 0.2,
    t0_stride: int = 10,
    filter_reverse: bool = True,
    reverse_angle_deg: float = 90.0,
    max_reverse_fraction: float = 0.2,
    min_speed_mps: float = 0.5,
    verbose: bool = True,
) -> list[tuple[str, float]]:
    """Enumerate ``(scene_id, t0_s)`` windows exactly as ``TruckDriveDataset._build_index``
    does (metadata only -- pose file + camera directory listings, no image bytes).

    This is what turns ``sft_truckdrive.yaml``'s val split into its reported
    "2,474 windows / 139 scenes": every ``t0_stride``-th pose step across each
    scene's full valid timeline (not just one window per scene), scenes missing
    any requested view dropped entirely, and reversing/heavy-sideslip windows
    dropped via ``filter_reverse`` -- same defaults as the trainer.
    """
    view_to_alpamayo = dict(view_to_alpamayo or DEFAULT_VIEW_TO_ALPAMAYO)
    be = _make_backend(backend, data_root, region)

    hist_margin = (num_history_steps - 1) * time_step
    img_margin = (num_frames - 1) * image_time_step
    back_margin = max(hist_margin, img_margin)
    fut_margin = num_future_steps * time_step
    fut_horizon = num_future_steps * time_step

    windows: list[tuple[str, float]] = []
    n_missing_views = 0
    n_reverse = 0
    n_unusable = 0
    for scene_id in scene_ids:
        try:
            pose_text = be.read_bytes(f"{scene_id}/{pose_file}").decode("utf-8")
            poses = _ScenePoses(pose_text)
        except Exception:  # noqa: BLE001 - skip unreadable/too-short scenes
            n_unusable += 1
            continue

        missing = False
        for view in view_to_alpamayo:
            try:
                files = [
                    f for f in be.list_files(f"{scene_id}/camera/leopard/{view}/images")
                    if f.lower().endswith(_IMAGE_EXTS)
                ]
            except Exception:  # noqa: BLE001
                files = []
            if not any(_sync_key(f) in poses.key_to_t for f in files):
                missing = True
                break
        if missing:
            n_missing_views += 1
            continue

        lo = poses.t_min + back_margin
        hi = poses.t_max - fut_margin
        if hi <= lo:
            n_unusable += 1
            continue

        for i in range(0, len(poses.t), t0_stride):
            t0 = float(poses.t[i])
            if not (lo <= t0 <= hi):
                continue
            if filter_reverse and not poses.is_forward_window(
                t0, fut_horizon, min_speed_mps, reverse_angle_deg, max_reverse_fraction
            ):
                n_reverse += 1
                continue
            windows.append((scene_id, t0))

    if verbose:
        print(
            f"enumerate_val_windows: {len(windows)} windows across "
            f"{len({s for s, _ in windows})} scenes (skipped: {n_missing_views} "
            f"missing-view scenes, {n_unusable} unusable scenes, {n_reverse} reverse windows)"
        )
    return windows
