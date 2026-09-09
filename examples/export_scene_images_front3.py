# SPDX-License-Identifier: Apache-2.0
"""Export current-frame (t0) stills from the 3 front-facing cameras (cross_left,
front_wide, cross_right -- camera ids 0/1/2) for a list of events, for use as a reduced
visual-QA input when hand-labeling self-generated CoC correctness. Deliberately narrower
than the model's own "vqa" profile (6 cameras x 4 frames) -- this is meant for a cheaper,
targeted human/Claude-vision spot-check, not to reproduce the model's own input contract.

Usage
-----
    python examples/export_scene_images_front3.py \\
        --inputs outputs/motion_confidence_multi_coc_val_n289.json \\
        --limit 10 --out_dir outputs/motion_confidence_images/front3_qa
"""

import argparse
import json
import os
import time

from PIL import Image

from alpamayo2_super.input_profiles import InputProfile, select_input_profile
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

FRONT3_CURRENT_FRAME = InputProfile(camera_ids=(0, 1, 2), frame_indices=(-1,))
CAMERA_SLUGS = {0: "front_left", 1: "front_wide", 2: "front_right"}


def _load_with_retry(clip_id, t0_us, max_attempts=8, base_delay=20.0):
    """See eval_common.load_scene_with_retry -- a 429 from the shared HF Hub quota can
    surface here as zipfile.BadZipFile rather than an HTTP error, so retry any exception
    with backoff instead of pattern-matching error strings."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            return select_input_profile(source_data, FRONT3_CURRENT_FRAME)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt
            print(f"  [retry {attempt}/{max_attempts}] load failed (possibly transient HF rate limit), retrying in {delay:.0f}s: {e!r}")
            time.sleep(delay)
    raise last_err  # pragma: no cover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="motion_confidence_multi_coc*.json path")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/motion_confidence_images/front3_qa")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip any event whose 3 images already exist in --out_dir (by global index).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    events = json.load(open(args.inputs))["results"][args.skip:args.skip + args.limit]

    manifest = []
    for i, ev in enumerate(events):
        global_index = args.skip + i
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        if args.resume and all(
            os.path.exists(os.path.join(args.out_dir, f"event{global_index:03d}_{slug}.png"))
            for slug in CAMERA_SLUGS.values()
        ):
            paths = {
                slug: os.path.join(args.out_dir, f"event{global_index:03d}_{slug}.png")
                for slug in CAMERA_SLUGS.values()
            }
            print(f"[{global_index}] {clip_id} t0={t0_us} -> already exported, skipping")
            manifest.append({
                "index": global_index, "clip_id": clip_id, "t0_us": t0_us,
                "self_cocs": ev["self_cocs"], "image_paths": paths,
            })
            continue
        selected = _load_with_retry(clip_id, t0_us)
        camera_ids = selected["camera_indices"].tolist()
        image_frames = selected["image_frames"]  # [3 cams, 1 frame, C, H, W]
        paths = {}
        for pos, cam_id in enumerate(camera_ids):
            frame = image_frames[pos, 0]  # [C, H, W]
            arr = frame.permute(1, 2, 0).numpy()
            slug = CAMERA_SLUGS[cam_id]
            out_path = os.path.join(args.out_dir, f"event{global_index:03d}_{slug}.png")
            Image.fromarray(arr).save(out_path)
            paths[slug] = out_path
        print(f"[{global_index}] {clip_id} t0={t0_us} -> {list(paths.values())}")
        manifest.append({
            "index": global_index, "clip_id": clip_id, "t0_us": t0_us,
            "self_cocs": ev["self_cocs"], "image_paths": paths,
        })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
