"""N-way qualitative comparison: Alpamayo 2 configs vs Alpamayo 1.5, on TruckDrive.

Per scene, renders every window (~1s apart) as one video frame:
  - Camera tiles (5 views), tightly tiled with no gaps (forward row: cross-left/
    front/cross-right; rearward row: rear-left/rear-right -- matches
    alpamayo-recipes' render_scene_video.py layout), camera name drawn directly
    on each tile (top-left), with GT + the single best-scoring model's
    prediction + the single worst-scoring model's prediction projected on (not
    all models -- that many overlapping lines on a camera tile is unreadable;
    the full fan is in the BEV panel instead).
  - BEV panel: grey history, red GT future, every model's best-of-K prediction
    in its own colour, full legend. Axis limits are fixed per scene (computed
    from every window's trajectories) so the view does not jitter frame to frame.
  - CoC panel below: each CoT-capable model's best-of-K sample's generated
    reasoning, colour-matched to its BEV curve.

Reads directly from each model's saved ``predictions.pt`` (no GPU, no model
loading -- this only re-loads camera images per window). Scene selection comes
from TruckDrive/truckdrive_val_ade_selection.json and truckdrive_val_interesting.json,
keyed by the devkit's own per-window ``idx``.

Usage:

    python examples/truckdrive_model_comparison.py --out-dir outputs/truckdrive_comparison
"""

from __future__ import annotations

import argparse
import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_DISPLAY_NAMES
from alpamayo2_super.load_truckdrive import load_truckdrive_sample
from alpamayo2_super.truckdrive_viz import project_ego_to_image

HISTORY_COLOR = "#777777"
GT_COLOR = "#D62728"
EGO_COLOR = "#D62728"

# Forward row (cross-left, front-wide, cross-right), rearward row (rear-left,
# rear-right) -- same grouping as alpamayo-recipes' render_scene_video.py
# _VIEW_ROWS, restricted to the 5 validated TruckDrive views (no front-tele).
CAMERA_ROWS = [[0, 1, 2], [3, 5]]
PANEL_W = 1400

# (label, predictions.pt path, colour, has_cot)
MODELS = [
    (
        "Alpamayo 1.5 zero-shot + CoC",
        "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-A1-format/eval_20260721_225240/predictions.pt",
        "#0072B2",
        True,
    ),
    (
        "Alpamayo 1.5 finetuned + CoC",
        "/mnt/efs/users/rod/ckpts/alpamayo_truckdrive_finetuning/"
        "alpamayo1.5_truckdrive_STAGE1_full_train_with_camids_2_epochs_checkpoint-8238/"
        "eval_20260722_213958/predictions.pt",
        "#009E73",
        True,
    ),
    (
        "Alpamayo 2 (4-frame, CoC-on)",
        "outputs/truckdrive_preds_baseline_preds_merged.pt",
        "#E69F00",
        True,
    ),
    (
        "Alpamayo 2 (4-frame, CoC-off)",
        "outputs/truckdrive_preds_nocot_baseline_preds_merged.pt",
        "#CC79A7",
        False,
    ),
]

# 10 distinct scenes spanning easy/standstill, typical, worst-ADE, and big-turn
# maneuvers -- picked from the devkit's curated ADE- and turn-angle-ranked
# selections (see module docstring). A video covers every window in the scene,
# so only scene_id matters here (not a specific t0). ``tag`` is used only for
# the output filename, not drawn in the figure.
SCENES = [
    ("scene_42_12", "easy_standstill"),
    ("scene_28_24", "hardest_turn_210deg"),
    ("scene_41_1", "typical"),
    ("scene_35_42", "worst_ade_nonturn"),
    ("scene_35_43", "worst_ade_nonturn_2"),
    ("scene_36_33", "moderate"),
    ("scene_35_11", "moderate_high"),
    ("scene_41_7", "turn_105deg_hard"),
    ("scene_28_23", "turn_108deg"),
    ("scene_42_4", "turn_103deg"),
    ("scene_36_46", "additional"),
    ("scene_35_27", "additional"),
    ("scene_35_22", "bin0_easy"),
    ("scene_36_26", "bin1"),
    ("scene_35_24", "bin2_typical"),
    ("scene_36_39", "bin3_moderate_high"),
    ("scene_41_8", "bin4_worst_ade"),
    ("scene_42_6", "bin4_worst_ade_2"),
]


def _pil_font(bold: bool = False, size: int = 16) -> ImageFont.FreeTypeFont:
    fname = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", fname)
    return ImageFont.truetype(path, size)


def load_model_index(path: str) -> dict[tuple[str, int], dict]:
    recs = torch.load(path, map_location="cpu", weights_only=False)
    return {(r["scene_id"], int(r["t0_us"])): r for r in recs}


def best_worst_sample(rec: dict) -> tuple[int, float]:
    """(best sample index, its full-horizon ADE). Uses the saved sample_ade when
    present; deterministic baselines don't save it, so fall back to computing
    BEV ADE directly from pred_xyz vs. ego_future_xyz."""
    if "sample_ade" in rec:
        ade = np.asarray(rec["sample_ade"]).reshape(-1)
    else:
        pred = np.asarray(rec["pred_xyz"])[0]  # [K, T, 3]
        gt = np.asarray(rec["ego_future_xyz"])[0]  # [T, 3]
        ade = np.linalg.norm((pred - gt[None])[..., :2], axis=-1).mean(axis=-1)  # [K]
    k = int(ade.argmin())
    return k, float(ade[k])


def xy(t) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(t).reshape(-1, 3)
    return a[:, 0], a[:, 1]


def scene_windows(scene_id: str, model_indices: dict) -> list[int]:
    """Sorted t0_us for every window of ``scene_id`` (union across models, since
    a few windows can be missing from one model's saved file)."""
    t0s: set[int] = set()
    for index in model_indices.values():
        t0s.update(t for (s, t) in index if s == scene_id)
    return sorted(t0s)


def gather_window(scene_id: str, t0_us: int, model_indices: dict) -> dict | None:
    """Per-model best-of-K info for one window. None if no model has this window."""
    per_model = []
    gt_xyz = hist_xyz = None
    for label, path, color, has_cot in MODELS:
        rec = model_indices[path].get((scene_id, t0_us))
        if rec is None:
            continue
        if gt_xyz is None:
            gt_xyz = rec["ego_future_xyz"][0].numpy()
            hist_xyz = rec["ego_history_xyz"][0].numpy()
        k, ade = best_worst_sample(rec)
        per_model.append({"label": label, "color": color, "has_cot": has_cot,
                           "rec": rec, "k": k, "ade": ade})
    if gt_xyz is None:
        return None
    return {"t0_us": t0_us, "gt_xyz": gt_xyz, "hist_xyz": hist_xyz, "per_model": per_model}


def scene_bev_limits(windows: list[dict], pad: float = 8.0) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for w in windows:
        for key in ("hist_xyz", "gt_xyz"):
            x, y = xy(w[key])
            xs.append(x)
            ys.append(y)
        for m in w["per_model"]:
            pred = np.asarray(m["rec"]["pred_xyz"])[0]  # [K, T, 3]
            x, y = xy(pred)
            xs.append(x)
            ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    half = max(0.5 * (x.max() - x.min()), 0.5 * (y.max() - y.min()), 8.0) + pad
    return cx - half, cx + half, cy - half, cy + half


def _draw_traj_pil(draw: ImageDraw.ImageDraw, xyz, calib, color: str, tile_w: int, tile_h: int, lw: int) -> None:
    ow, oh = calib["image_size"]
    uv, valid = project_ego_to_image(xyz, calib["K"], calib["T_vehicle_cam"], calib["image_size"])
    uv = uv * np.array([tile_w / ow, tile_h / oh])
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return
    breaks = np.where(np.diff(idx) > 1)[0]
    for run in np.split(idx, breaks + 1):
        if len(run) < 2:
            continue
        draw.line([tuple(p) for p in uv[run]], fill=color, width=lw, joint="curve")


def _camera_panel(camera_data, calibration, hist_xyz, gt_xyz, overall_best, overall_worst,
                   width: int = PANEL_W) -> np.ndarray:
    """Tightly-tiled camera mosaic (no subplot gaps), labels drawn on-image."""
    camera_indices = camera_data["camera_indices"].tolist()
    label_font = _pil_font(bold=True, size=15)
    row_imgs = []
    for row in CAMERA_ROWS:
        row = [c for c in row if c in camera_indices]
        if not row:
            continue
        tile_w = width // len(row)
        tiles = []
        for cam_idx in row:
            slot = camera_indices.index(cam_idx)
            arr = camera_data["image_frames"][slot, -1].permute(1, 2, 0).numpy()
            oh0, ow0 = arr.shape[:2]
            tile_h = max(1, round(oh0 * tile_w / ow0))
            im = Image.fromarray(arr).resize((tile_w, tile_h), Image.BILINEAR)
            draw = ImageDraw.Draw(im)
            calib = calibration.get(cam_idx)
            if calib is not None:
                _draw_traj_pil(draw, hist_xyz, calib, HISTORY_COLOR, tile_w, tile_h, 3)
                _draw_traj_pil(draw, gt_xyz, calib, GT_COLOR, tile_w, tile_h, 4)
                _draw_traj_pil(draw, np.asarray(overall_best["rec"]["pred_xyz"])[0, overall_best["k"]],
                                calib, overall_best["color"], tile_w, tile_h, 4)
                _draw_traj_pil(draw, np.asarray(overall_worst["rec"]["pred_xyz"])[0, overall_worst["k"]],
                                calib, overall_worst["color"], tile_w, tile_h, 3)
            label = CAMERA_INDICES_TO_DISPLAY_NAMES.get(cam_idx, str(cam_idx))
            draw.text((6, 4), label, font=label_font, fill=(255, 255, 255),
                       stroke_width=2, stroke_fill=(0, 0, 0))
            tiles.append(np.asarray(im))
        h_max = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, h_max - t.shape[0]), (0, 0), (0, 0))) if t.shape[0] < h_max else t
                 for t in tiles]
        row_img = np.hstack(tiles)
        if row_img.shape[1] < width:
            row_img = np.pad(row_img, ((0, 0), (0, width - row_img.shape[1]), (0, 0)))
        row_imgs.append(row_img[:, :width])
    return np.vstack(row_imgs)


def _bev_panel(scene_id: str, t0_s: float, hist_xyz, gt_xyz, per_model, bev_limits,
               height_px: int, dpi: int = 110) -> np.ndarray:
    width_px = int(height_px * 1.05)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    hx, hy = xy(hist_xyz)
    ax.plot(-hy, hx, color=HISTORY_COLOR, lw=2.0, label="history")
    gx, gy = xy(gt_xyz)
    ax.plot(-gy, gx, color=GT_COLOR, lw=2.5, label="GT future")
    for m in per_model:
        pred = np.asarray(m["rec"]["pred_xyz"])[0]  # [K, T, 3]
        for s in range(pred.shape[0]):
            px, py = xy(pred[s])
            is_best_k = s == m["k"]
            ax.plot(-py, px, color=m["color"], lw=2.4 if is_best_k else 0.8,
                     alpha=0.95 if is_best_k else 0.35,
                     label=f"{m['label']} (ADE {m['ade']:.2f}m)" if is_best_k else None)
    ax.scatter([0], [0], color=EGO_COLOR, s=30, zorder=6, marker="s")
    ax.set_aspect("equal")
    x_lo, x_hi, y_lo, y_hi = bev_limits
    ax.set_xlim(-y_hi, -y_lo)
    ax.set_ylim(x_lo, x_hi)
    ax.set_xlabel("-y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"{scene_id}  t0={t0_s:.2f}s", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _cot_panel(per_model, width: int, row_h: int = 68, header_h: int = 28) -> np.ndarray:
    active = [m for m in per_model if m["has_cot"]]
    height = header_h + row_h * max(1, len(active)) + 6
    img = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(img)
    header_font = _pil_font(bold=True, size=15)
    label_font = _pil_font(bold=True, size=13)
    text_font = _pil_font(bold=False, size=13)
    draw.text((10, 6), "Chain-of-Causation (best-of-K sample per model):", font=header_font, fill=(20, 20, 20))
    y = header_h
    wrap_chars = max(20, int((width - 44) / 7.2))
    for m in active:
        cot = m["rec"].get("gen_text/cot")
        text = cot[0][m["k"]] if cot else ""
        draw.rectangle([10, y + 6, 26, y + 22], fill=m["color"])
        draw.text((34, y + 2), f"{m['label']}:", font=label_font, fill=m["color"])
        wrapped = textwrap.fill(text or "(empty)", width=wrap_chars)
        draw.multiline_text((34, y + 22), wrapped, font=text_font, fill=(25, 25, 25), spacing=4)
        y += row_h
    return np.asarray(img)


def render_frame(scene_id: str, t0_s: float, window: dict, camera_data, bev_limits) -> np.ndarray:
    hist_xyz, gt_xyz, per_model = window["hist_xyz"], window["gt_xyz"], window["per_model"]
    calibration = camera_data.get("truckdrive_calibration", {})

    overall_best = min(per_model, key=lambda m: m["ade"])
    overall_worst = max(per_model, key=lambda m: m["ade"])

    cam_panel = _camera_panel(camera_data, calibration, hist_xyz, gt_xyz, overall_best, overall_worst)
    bev_panel = _bev_panel(scene_id, t0_s, hist_xyz, gt_xyz, per_model, bev_limits, height_px=cam_panel.shape[0])
    top = np.hstack([cam_panel, bev_panel[: cam_panel.shape[0]]])
    cot_panel = _cot_panel(per_model, width=top.shape[1])
    return np.vstack([top, cot_panel])


def write_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
    import av

    h, w = frames[0].shape[:2]
    w -= w % 2
    h -= h % 2
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    for f in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def render_scene(scene_id: str, tag: str, model_indices: dict, out_dir: str, fps: int) -> None:
    print(f"[{scene_id}] ({tag})")
    t0_list = scene_windows(scene_id, model_indices)
    windows = [w for t0 in t0_list if (w := gather_window(scene_id, t0, model_indices)) is not None]
    if not windows:
        print("  no windows found; skipping")
        return
    print(f"  {len(windows)} windows")
    limits = scene_bev_limits(windows)

    frames = []
    for i, w in enumerate(windows):
        t0_s = w["t0_us"] / 1e6
        data = load_truckdrive_sample(
            scene_id=scene_id, t0_s=t0_s, num_frames=1, include_calibration=True,
        )
        frame = render_frame(scene_id, t0_s, w, data, limits)
        frames.append(frame)
        print(f"  {i + 1}/{len(windows)} t0={t0_s:.2f}s", flush=True)

    out_path = os.path.join(out_dir, f"{scene_id}_{tag}.mp4")
    write_mp4(frames, out_path, fps)
    print(f"  wrote {out_path} ({len(frames)} frames @ {fps}fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/truckdrive_comparison")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--scenes", default=None, help="Comma-separated scene_ids; default = curated 10.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model prediction indices...")
    model_indices = {path: load_model_index(path) for _, path, _, _ in MODELS}

    scenes = SCENES
    if args.scenes:
        wanted = set(args.scenes.split(","))
        scenes = [(s, t) for s, t in SCENES if s in wanted]

    for scene_id, tag in scenes:
        render_scene(scene_id, tag, model_indices, args.out_dir, args.fps)


if __name__ == "__main__":
    main()
