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

"""Run Alpamayo 2 Super inference on a TruckDrive scene instead of PhysicalAI-AV.

Same shape as ``alpamayo2_super.inference_smoke``: loads one sample, samples one
trajectory from the diffusion expert, prints the generated CoT + minADE against
TruckDrive's ground-truth future, and (optionally) saves a camera+BEV figure.

Example:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python examples/truckdrive_inference.py \\
      --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \\
      --scene-id scene_28_1 \\
      --t0-s 10.0 \\
      --save-viz outputs/truckdrive_scene_28_1.png \\
      --save-json outputs/truckdrive_scene_28_1.json
"""

import argparse
import os

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
from alpamayo2_super.load_truckdrive import DEFAULT_DATA_ROOT


def run(
    model_id: str,
    scene_id: str,
    t0_s: float,
    data_root: str,
    backend: str,
    save_viz: str | None,
    save_json: str | None,
    diffusion_steps: int,
    seed: int,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.load_truckdrive import load_truckdrive_sample
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.truckdrive_viz import plot_truckdrive_inference_result

    if not torch.cuda.is_available():
        raise RuntimeError("TruckDrive inference requires a CUDA GPU.")

    print(f"Loading TruckDrive scene {scene_id} @ t0={t0_s}s from {data_root} (backend={backend})...")
    data = load_truckdrive_sample(scene_id=scene_id, t0_s=t0_s, data_root=data_root, backend=backend)
    print(f"Loaded {data['image_frames'].shape[0]} cameras: {data['camera_names']}")

    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, logprob, extra = model.sample_trajectories_from_data(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            diffusion_kwargs={"inference_step": diffusion_steps},
            return_extra=True,
        )
    del pred_rot, logprob
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    print("minADE:", diff.min(), "meters")

    if save_viz or save_json:
        _, metadata = plot_truckdrive_inference_result(
            data=data,
            pred_xyz=pred_xyz,
            extra=extra,
            output_path=save_viz,
            json_path=save_json,
            model_id=model_id,
            seed=seed,
        )
        print("projection_available:", metadata["projection_available"])
        if save_viz:
            print("Saved visualization:", save_viz)
        if save_json:
            print("Saved metadata:", save_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    )
    parser.add_argument("--scene-id", default="scene_28_1", help="TruckDrive scene id.")
    parser.add_argument("--t0-s", type=float, default=10.0, help="Scene-relative timestamp (s).")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TRUCKDRIVE_DATA_ROOT", DEFAULT_DATA_ROOT),
        help="Local dir or s3://bucket/prefix containing scene_* directories.",
    )
    parser.add_argument("--backend", choices=["s3", "local"], default="s3")
    parser.add_argument("--save-viz", default=None)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        model_id=args.model_id,
        scene_id=args.scene_id,
        t0_s=args.t0_s,
        data_root=args.data_root,
        backend=args.backend,
        save_viz=args.save_viz,
        save_json=args.save_json,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
