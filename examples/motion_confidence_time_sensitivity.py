# SPDX-License-Identifier: Apache-2.0
"""How sensitive is the yes/no first-token confidence to a small shift in t0?

``load_physical_aiavdataset`` samples camera context frames on a fixed 0.1s-spaced grid
ending at t0 (10 Hz, ``time_step=0.1`` default) -- so shifting t0 by +-0.1s moves the
model's most recent context frame to the adjacent one on that grid, a small, physically
real change (roughly one frame of camera motion) rather than a a resampling artifact.
This probes whether the baseline single-shot yes/no confidence (``motion_confidence_smoke.
BASELINE_TEMPLATE``) is a stable read of the scene or swings noisily under a shift this
small -- a confidence signal that flips sign under a 0.1s nudge is not trustworthy.

Runs BOTH scoring methods from ``motion_confidence_smoke.py`` at every offset, for both
the positive and negative gold-CoC actions of one event:
  - baseline: single-shot, no scene description, no sampling.
  - scene_conditioned: sample ``--num_samples`` scene descriptions (candidate action not
    in context), judge each, average. Averaging here is over language-sampling noise at a
    FIXED frame -- it says nothing about frame-to-frame sensitivity, so comparing its
    range across offsets to the baseline's range answers a different question than the
    smoke-test calibration comparison did: does resampling at one frame also happen to
    smooth out the model's sensitivity to *which* frame it was given?

Usage
-----
    python examples/motion_confidence_time_sensitivity.py \\
        --clip_id 428709c9-4a9c-4e90-bdbd-e56d430bd195 --t0_us 14100000 \\
        --positive_action "Steer left following temporary traffic delineators." \\
        --negative_action "Go straight following temporary traffic delineators." \\
        --out outputs/motion_confidence_time_sensitivity.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_confidence_smoke as mc  # noqa: E402

MIN_T0_US = 1_700_000  # history window (1.6s) + margin, see eval_common.py


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip_id", required=True)
    parser.add_argument("--t0_us", type=int, required=True)
    parser.add_argument("--positive_action", required=True)
    parser.add_argument("--negative_action", required=True)
    parser.add_argument(
        "--offsets_s", type=float, nargs="+", default=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        help="Time offsets from t0_us, in seconds, on the loader's 0.1s frame grid.",
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Scene-description samples per offset.")
    parser.add_argument("--scene_max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_time_sensitivity.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from alpamayo2_super.chat_template.conversation import construct_image, construct_system_prompt
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.helper import get_processor
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs
    from alpamayo2_super import helper

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_time_sensitivity requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    results = []
    for oi, offset_s in enumerate(args.offsets_s):
        t0_us = max(args.t0_us + round(offset_s * 1_000_000), MIN_T0_US)
        print(f"\noffset={offset_s:+.1f}s  t0_us={t0_us}")
        try:
            source_data = load_physical_aiavdataset(args.clip_id, t0_us=t0_us)
            vqa_data = select_task_input(source_data, "vqa")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load: {e}")
            results.append({"offset_s": offset_s, "t0_us": t0_us, "error": str(e)})
            continue

        image_content = mc.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
        image_frames = vqa_data["image_frames"]

        entry = {"offset_s": offset_s, "t0_us": t0_us, "baseline": {}, "scene_conditioned": {}}

        # --- baseline: single-shot, no scene description, no sampling ---
        for label, action in (("positive", args.positive_action), ("negative", args.negative_action)):
            messages = [
                {"role": "system", "content": construct_system_prompt()},
                {"role": "user", "content": image_content + [
                    {"type": "text", "text": mc.BASELINE_TEMPLATE.format(action=action)}
                ]},
            ]
            conf, yes_p, no_p = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
            entry["baseline"][label] = {"confidence": conf, "yes_p": yes_p, "no_p": no_p}
            print(f"  baseline/{label}: conf={conf:.3f} (yes={yes_p:.4f}, no={no_p:.4f})")

        # --- scene-conditioned: sample N scene descriptions at this frame, judge each, average ---
        torch.cuda.manual_seed_all(args.seed + oi)
        vqa_inputs = prepare_vqa_inputs(
            data=vqa_data, model_config=model.config, tokenizer=tokenizer, question=mc.SCENE_QUESTION,
        )
        vqa_inputs = helper.to_device(vqa_inputs, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scene_result = generate_text(
                model, vqa_inputs, top_p=args.top_p, temperature=args.temperature,
                num_samples=args.num_samples, max_new_tokens=args.scene_max_new_tokens,
            )
        descriptions = scene_result["answer"]
        entry["scene_descriptions"] = descriptions

        for label, action in (("positive", args.positive_action), ("negative", args.negative_action)):
            confs, yes_ps, no_ps = [], [], []
            for desc in descriptions:
                messages = [
                    {"role": "system", "content": construct_system_prompt()},
                    {"role": "user", "content": image_content + [{"type": "text", "text": mc.SCENE_QUESTION}]},
                    {"role": "assistant", "content": [{"type": "text", "text": desc}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": mc.JUDGE_TEMPLATE.format(action=action)}
                    ]},
                ]
                conf, yes_p, no_p = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
                confs.append(conf)
                yes_ps.append(yes_p)
                no_ps.append(no_p)
            mean_conf = sum(confs) / len(confs)
            entry["scene_conditioned"][label] = {
                "mean_confidence": mean_conf, "per_sample_confidence": confs,
            }
            print(f"  scene/{label}: mean_conf={mean_conf:.3f} per_sample={['%.2f' % c for c in confs]}")

        results.append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")

    ok = [r for r in results if "error" not in r]
    if len(ok) >= 2:
        for method, get_conf in (
            ("baseline", lambda r, label: r["baseline"][label]["confidence"]),
            ("scene_conditioned", lambda r, label: r["scene_conditioned"][label]["mean_confidence"]),
        ):
            print(f"\n{method}:")
            for label in ("positive", "negative"):
                confs = [get_conf(r, label) for r in ok]
                print(f"  {label}: min={min(confs):.3f} max={max(confs):.3f} "
                      f"range={max(confs) - min(confs):.3f} mean={sum(confs) / len(confs):.3f}")


if __name__ == "__main__":
    main()
