# SPDX-License-Identifier: Apache-2.0
"""Export front-wide-camera PNG stills for the motion-confidence experiment writeup.

No model load needed -- just the dataset loader + task profile selection. Saves the
most recent (t0) front-wide-camera frame for each requested (clip_id, t0_us), plus,
for --time_sweep, one frame per offset in ``motion_confidence_time_sensitivity.py``'s
default sweep so the writeup can show how little the image changes across the offsets
that produced such a large confidence swing.
"""

import argparse
import json
import os

from PIL import Image

from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES, FRONT_WIDE_CAMERA_NAME
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

FRONT_WIDE_ID = CAMERA_NAMES_TO_INDICES[FRONT_WIDE_CAMERA_NAME]
MIN_T0_US = 1_700_000


def save_front_wide(clip_id: str, t0_us: int, out_path: str) -> None:
    source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    vqa_data = select_task_input(source_data, "vqa")
    camera_position = vqa_data["camera_indices"].tolist().index(FRONT_WIDE_ID)
    frame = vqa_data["image_frames"][camera_position, -1]  # [C, H, W], most recent frame
    arr = frame.permute(1, 2, 0).numpy()  # [H, W, C], uint8
    Image.fromarray(arr).save(out_path)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="outputs/motion_confidence_images")
    parser.add_argument(
        "--events", nargs="+", default=[],
        help="clip_id:t0_us:slug triples, e.g. abcd1234:14100000:example1",
    )
    parser.add_argument(
        "--time_sweep_clip_id", default=None,
        help="If set, also export one frame per offset in --time_sweep_offsets around this clip's t0.",
    )
    parser.add_argument("--time_sweep_t0_us", type=int, default=None)
    parser.add_argument(
        "--time_sweep_offsets", type=float, nargs="+",
        default=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []

    for spec in args.events:
        clip_id, t0_us, slug = spec.split(":")
        out_path = os.path.join(args.out_dir, f"{slug}.png")
        save_front_wide(clip_id, int(t0_us), out_path)
        manifest.append({"clip_id": clip_id, "t0_us": int(t0_us), "slug": slug, "path": out_path})

    if args.time_sweep_clip_id:
        for offset_s in args.time_sweep_offsets:
            t0_us = max(args.time_sweep_t0_us + round(offset_s * 1_000_000), MIN_T0_US)
            slug = f"timesweep_{offset_s:+.1f}s".replace("+", "p").replace("-", "m").replace(".", "_")
            out_path = os.path.join(args.out_dir, f"{slug}.png")
            save_front_wide(args.time_sweep_clip_id, t0_us, out_path)
            manifest.append({
                "clip_id": args.time_sweep_clip_id, "t0_us": t0_us,
                "offset_s": offset_s, "slug": slug, "path": out_path,
            })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
