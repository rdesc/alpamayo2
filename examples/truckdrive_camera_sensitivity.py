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

"""Camera/temporal-config sensitivity study on a handful of TruckDrive scenes.

Fixes (scene, t0, seed) and varies only the camera-input config, to see how
much Alpamayo 2 Super's trajectory prediction depends on:
  - camera count (5 validated views vs. front-only vs. front+cross hemisphere)
  - temporal context (4 history frames vs. just the t0 frame)
  - frame spacing (0.2s, matching TruckDrive's ~5Hz rate, vs. the 0.1s PAI-style
    spacing that turned out to be a bug -- included as an ablation, not just a
    fix, to quantify how much it actually mattered)

Not a rigorous eval (5 scenes, 1 window each) -- a quick sensitivity read to
guide where to spend more eval budget.

Example:

    python examples/truckdrive_camera_sensitivity.py \\
      --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \\
      --out outputs/truckdrive_camera_sensitivity.jsonl
"""

import argparse
import json
import os
import time

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
from alpamayo2_super.load_truckdrive import DEFAULT_DATA_ROOT

# (scene_id, t0_s) picked to span the minADE/6.4s distribution of the earlier
# 1-window-per-scene val pass (outputs/truckdrive_val_eval.jsonl): min, p25,
# median, p75, max.
SCENES = [
    ("scene_35_4", 10.48, "p0 (easiest, 0.18m)"),
    ("scene_36_49", 10.53, "p25 (1.04m)"),
    ("scene_36_35", 10.50, "median (1.74m)"),
    ("scene_28_4", 10.48, "p75 (3.19m)"),
    ("scene_28_14", 10.55, "p100 (worst, 14.46m)"),
]

FRONT_VIEW = {"forward_center_medium": "camera_front_wide_120fov"}
THREE_VIEWS = {
    "forward_center_medium": "camera_front_wide_120fov",
    "sideward_left_front_wide": "camera_cross_left_120fov",
    "sideward_right_front_wide": "camera_cross_right_120fov",
}

# (config name, view_to_alpamayo override or None=default 5 views, num_frames, image_time_step)
CONFIGS = [
    ("5view_4frame_0.2s_baseline", None, 4, 0.2),
    ("5view_1frame_currentonly", None, 1, 0.2),
    ("3view_4frame_0.2s", THREE_VIEWS, 4, 0.2),
    ("front_only_4frame_0.2s", FRONT_VIEW, 4, 0.2),
    ("front_only_1frame_currentonly", FRONT_VIEW, 1, 0.2),
    ("5view_4frame_0.1s_buggy_spacing", None, 4, 0.1),
]


def run(model_id: str, data_root: str, backend: str, out_path: str, num_traj_samples: int,
        diffusion_steps: int, seed: int, scenes=None) -> None:
    scenes = scenes if scenes is not None else SCENES
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from alpamayo2_super import helper, truckdrive_metrics
    from alpamayo2_super.load_truckdrive import load_truckdrive_sample
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    if not torch.cuda.is_available():
        raise RuntimeError("requires a CUDA GPU.")

    print(f"Loading {model_id}...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records = []
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for scene_id, t0, tag in scenes:
            for cfg_name, views, num_frames, img_dt in CONFIGS:
                t_w = time.time()
                try:
                    data = load_truckdrive_sample(
                        scene_id=scene_id, t0_s=t0, data_root=data_root, backend=backend,
                        view_to_alpamayo=views, num_frames=num_frames, image_time_step=img_dt,
                        standstill_snap_mps=0.5, include_calibration=False,
                    )
                    model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
                    model_inputs = helper.to_device(model_inputs, "cuda")

                    torch.cuda.manual_seed_all(seed)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred_xyz, pred_rot, _logprob, extra = model.sample_trajectories_from_data(
                            data=model_inputs, top_p=0.98, temperature=0.6,
                            num_traj_samples=num_traj_samples,
                            diffusion_kwargs={"inference_step": diffusion_steps},
                            return_extra=True,
                        )
                    pred_xyz_cpu = pred_xyz.float().cpu()
                    pred_rot_cpu = pred_rot.float().cpu()
                    gt_xyz_b = data["ego_future_xyz"][:, -1].cpu()
                    gt_rot_b = data["ego_future_rot"][:, -1].cpu()
                    metrics = truckdrive_metrics.displacement_metrics_from_tensors(pred_xyz_cpu, gt_xyz_b)
                    metrics["corner_distance"] = truckdrive_metrics.corner_distance(
                        pred_xyz_cpu, pred_rot_cpu, gt_xyz_b, gt_rot_b
                    )
                    cot = extra["cot"][0] if isinstance(extra["cot"], (list, tuple)) else extra["cot"]
                    record = {
                        "scene_id": scene_id, "t0_s": t0, "scene_tag": tag, "config": cfg_name,
                        "n_cameras": int(data["camera_indices"].numel()), "num_frames": num_frames,
                        "image_time_step": img_dt, **metrics, "cot": cot, "status": "ok",
                    }
                except Exception as exc:  # noqa: BLE001
                    record = {
                        "scene_id": scene_id, "t0_s": t0, "config": cfg_name,
                        "status": "error", "error": repr(exc),
                    }
                record["elapsed_s"] = round(time.time() - t_w, 2)
                records.append(record)
                out_f.write(json.dumps(record, default=str) + "\n")
                out_f.flush()
                tag_str = (
                    f"minADE/6.4s={record['min_ade/by_t=6.4']:.2f}m"
                    if record["status"] == "ok" else record["status"]
                )
                print(f"{scene_id} [{cfg_name}] {tag_str} ({record['elapsed_s']:.1f}s)", flush=True)

    print(f"\nDone in {time.time() - t_start:.0f}s. Wrote {out_path}")

    # Pivot: mean minADE/6.4s per config, across the 5 scenes.
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n=== Mean minADE/6.4s by config ({len(scenes)} scenes) ===")
    for cfg_name, *_ in CONFIGS:
        vals = [r["min_ade/by_t=6.4"] for r in ok if r["config"] == cfg_name]
        if vals:
            print(f"  {cfg_name:35s} mean={sum(vals)/len(vals):6.3f}m  n={len(vals)}")
        else:
            print(f"  {cfg_name:35s} (no successful runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID))
    parser.add_argument("--data-root", default=os.environ.get("TRUCKDRIVE_DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--backend", choices=["s3", "local"], default="s3")
    parser.add_argument("--out", default="outputs/truckdrive_camera_sensitivity.jsonl")
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenes-file", default=None,
        help="JSON file of [scene_id, t0_s, tag] triples. Defaults to the built-in 5-scene set.",
    )
    args = parser.parse_args()
    scenes = None
    if args.scenes_file:
        with open(args.scenes_file) as f:
            scenes = [tuple(row) for row in json.load(f)]
    run(args.model_id, args.data_root, args.backend, args.out, args.num_traj_samples,
        args.diffusion_steps, args.seed, scenes=scenes)


if __name__ == "__main__":
    main()
