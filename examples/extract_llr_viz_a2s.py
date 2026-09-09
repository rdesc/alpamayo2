# SPDX-License-Identifier: Apache-2.0
"""Camera + BEV + velocity viz assets for Alpamayo-2-Super's 9 min/mean/max llr_act examples.

Mirrors the Alpamayo-1.5 side's alpamayo1_5_v2 asset set (see
recipes/alpamayo1_x_rl/scripts_fork/extract_llr_viz_predictions_v2.py in the
alpamayo-recipes repo for the reference schema/plot style) so both can sit in the same
published artifact. A2S has a clean, already-validated no-reasoning ablation (drop "cot"
from components_prompt -- see examples/eval_pai_av_val_a2.py's --no_coc docstring), unlike
Alpamayo-1.5's RLWrapperReasoningVLA which needed a manual coc_text="" splice.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physical_ai_av  # noqa: E402

from alpamayo2_super import helper  # noqa: E402
from alpamayo2_super.chat_template.conversation import build_conversation  # noqa: E402
from alpamayo2_super.common.constants import PUBLIC_MODEL_ID  # noqa: E402
from alpamayo2_super.input_profiles import select_task_input  # noqa: E402
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402

OUT_ROOT = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo2_super_v2"
)
MIN_LATERAL_SPAN_M = 16.0
DT = 0.1
CAMERA_IDS = (0, 1, 2, 3, 5, 6)

EVENTS = [
    ("min", 1, "5c0b2864-ca4b-4cde-8dd8-e772381fe695", 0, 1933258.0, -0.324634,
     "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "train"),
    ("min", 2, "d05dcfff-5aee-4402-85fa-2cc1cdbd2545", 0, 10267575.0, -0.213386,
     "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "train"),
    ("min", 3, "5766b082-42e9-4e5d-b25b-ff81b419fdcf", 0, 5719639.0, -0.188335,
     "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "train"),
    ("mean", 1, "d6bd9a24-aa65-44c4-b31d-38e0bda599ae", 1, 14082916.0, 0.037031,
     "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "train"),
    ("mean", 2, "6b84c3ec-d91f-4d2b-9391-63dbf66e8d8f", 1, 14934510.0, 0.036891,
     "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "train"),
    ("mean", 3, "d7bc2b2e-3938-42e5-8583-eda119c8a001", 1, 14045801.0, 0.037137,
     "WORK_ZONES_TEMP_TRAFFIC_CONTROL", "train"),
    ("max", 1, "40a11a64-aed4-49c1-b747-b3befdcd708d", 0, 5512155.0, 0.532319,
     "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "val"),
    ("max", 2, "85766484-e0fc-489e-aad9-04ec31843c01", 0, 5687574.0, 0.468126,
     "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "train"),
    ("max", 3, "594d2c54-ca9a-4bda-be8e-532b2fc359fb", 0, 6659114.0, 0.463373,
     "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY", "val"),
]


def speed_from_xy(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    n = len(xy)
    v = np.zeros(n)
    if n >= 2:
        v[0] = np.linalg.norm(xy[1] - xy[0]) / dt
        v[-1] = np.linalg.norm(xy[-1] - xy[-2]) / dt
        for i in range(1, n - 1):
            v[i] = np.linalg.norm(xy[i + 1] - xy[i - 1]) / (2 * dt)
    return v


def render_bev(history_xy, future_xy, pred_with_xy, pred_without_xy, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")
    ax.plot(history_xy[:, 0], history_xy[:, 1], color="#4f8fd6", lw=2.2, marker="o", ms=3, label="history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color="#4fc7c7", lw=2.2, marker="o", ms=3, label="future (GT)")
    ax.plot(pred_with_xy[:, 0], pred_with_xy[:, 1], color="#e8a23a", lw=2.0, ls="--",
            marker="^", ms=3, label="pred (with reasoning)", alpha=0.95)
    ax.plot(pred_without_xy[:, 0], pred_without_xy[:, 1], color="#c86ee2", lw=2.0, ls="--",
            marker="v", ms=3, label="pred (no reasoning)", alpha=0.95)
    ax.scatter([0], [0], color="#e7ecf1", marker="*", s=140, zorder=5, label="ego @ t0")

    all_xy = np.concatenate([history_xy, future_xy, pred_with_xy, pred_without_xy], axis=0)
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()
    y_center = (y_min + y_max) / 2.0
    y_span = max(y_max - y_min, MIN_LATERAL_SPAN_M)
    ax.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)
    x_pad = max((x_max - x_min) * 0.08, 2.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forward (m)", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("lateral (m)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.legend(loc="upper left", fontsize=6.5, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_velocity(t_hist, v_hist, t_fut, v_fut, t_pred_w, v_pred_w, t_pred_wo, v_pred_wo, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")
    ax.plot(t_hist, v_hist, color="#4f8fd6", lw=2.0, marker="o", ms=2.5, label="history")
    ax.plot(t_fut, v_fut, color="#4fc7c7", lw=2.0, marker="o", ms=2.5, label="future (GT)")
    ax.plot(t_pred_w, v_pred_w, color="#e8a23a", lw=1.8, ls="--", marker="^", ms=2.5,
            label="pred (with reasoning)", alpha=0.95)
    ax.plot(t_pred_wo, v_pred_wo, color="#c86ee2", lw=1.8, ls="--", marker="v", ms=2.5,
            label="pred (no reasoning)", alpha=0.95)
    ax.axvline(0, color="#e7ecf1", lw=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("time (s)  [0 = ego @ t0]", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("speed (m/s)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=6, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _prepare_model_inputs(data, model_config, tokenizer, enable_cot: bool):
    components_prompt = ["cot", "traj_future"] if enable_cot else ["traj_future"]
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=components_prompt,
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=model_config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    processor = helper.get_processor(tokenizer, model_config)
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=not has_assistant_content,
        add_vision_id=False, continue_final_message=has_assistant_content,
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(text=text, images=images, videos=None, padding=False,
                  return_tensors="pt", do_rescale=False)
    )
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }


def main() -> None:
    torch.manual_seed(42)
    os.makedirs(OUT_ROOT, exist_ok=True)
    model_id = os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    print(f"[a2s-viz] loading model {model_id} ...", flush=True)
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def predict(data, enable_cot: bool):
        model_inputs = _prepare_model_inputs(data, model.config, model.tokenizer, enable_cot)
        model_inputs = helper.to_device(model_inputs, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _, extra = model.sample_trajectories_from_data(
                data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
                diffusion_kwargs={"inference_step": 10}, return_extra=True,
            )
        xy = pred_xyz.detach().float().cpu().numpy()[0, 0, 0, :, :2]
        cot_text = str(np.asarray(extra["cot"])[0, 0, 0]) if enable_cot else ""
        return xy, cot_text

    manifest = []
    for label, rank, clip_id, event_idx, t0_us, llr_act, cluster, split in EVENTS:
        clip_short = clip_id.split("-")[0]
        name = f"{label}{rank}_{clip_short}"
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        source = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
        data = select_task_input(source, "trajectory")

        gold_coc = source.get("coc") or source.get("gold_coc") or ""

        # camera images: most recent history frame (last of the 4 loaded frames) per camera.
        camera_files = []
        frames = data["image_frames"]  # [n_cams, n_frames, 3, H, W] uint8 (per prior convention)
        for i, cam_id in enumerate(CAMERA_IDS):
            img = frames[i, -1].permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(img).convert("RGB")
            im.thumbnail((960, 540))
            fname = f"cam_{cam_id}.jpg"
            im.save(os.path.join(out_dir, fname), quality=85)
            camera_files.append(fname)

        pred_with_xy, coc_with = predict(data, enable_cot=True)
        pred_without_xy, coc_without = predict(data, enable_cot=False)

        hist_xyz = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()
        fut_xyz = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()

        n = min(len(pred_with_xy), len(pred_without_xy))
        divergence = float(np.linalg.norm(pred_with_xy[:n] - pred_without_xy[:n], axis=-1).mean())

        render_bev(hist_xyz[:, :2], fut_xyz[:, :2], pred_with_xy, pred_without_xy,
                   os.path.join(out_dir, "bev.png"))

        v_hist = speed_from_xy(hist_xyz[:, :2])
        v_fut = speed_from_xy(fut_xyz[:, :2])
        v_pred_w = speed_from_xy(pred_with_xy)
        v_pred_wo = speed_from_xy(pred_without_xy)
        n_hist = len(hist_xyz)
        t_hist = (np.arange(n_hist) - (n_hist - 1)) * DT
        t_fut = (np.arange(len(fut_xyz)) + 1) * DT
        t_pred_w = (np.arange(len(pred_with_xy)) + 1) * DT
        t_pred_wo = (np.arange(len(pred_without_xy)) + 1) * DT
        render_velocity(t_hist, v_hist, t_fut, v_fut, t_pred_w, v_pred_w, t_pred_wo, v_pred_wo,
                         os.path.join(out_dir, "velocity.png"))

        manifest.append({
            "label": label, "rank": rank, "clip_id": clip_id, "event_idx": event_idx,
            "t0_us": t0_us, "llr_act": llr_act, "event_cluster": cluster, "split": split,
            "gold_coc": gold_coc, "out_dir": out_dir, "camera_files": camera_files,
            "camera_indices": list(CAMERA_IDS), "bev_file": "bev.png", "velocity_file": "velocity.png",
            "pred_coc_with_reasoning": coc_with, "pred_traj_with_reasoning": pred_with_xy.tolist(),
            "pred_coc_without_reasoning": coc_without, "pred_traj_without_reasoning": pred_without_xy.tolist(),
            "pred_with_without_divergence_m": divergence,
        })
        print(f"[a2s-viz] {name} llr_act={llr_act:.3f} divergence={divergence:.2f}m "
              f"coc_with={coc_with[:60]!r}", flush=True)
        torch.cuda.empty_cache()

    manifest_path = os.path.join(OUT_ROOT, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[a2s-viz] wrote {manifest_path}")


if __name__ == "__main__":
    main()
