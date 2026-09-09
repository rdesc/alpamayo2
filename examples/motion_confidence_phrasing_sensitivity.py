# SPDX-License-Identifier: Apache-2.0
"""How sensitive is the yes/no first-token confidence to paraphrasing the SAME action?

Companion to ``motion_confidence_time_sensitivity.py``: that one holds the action text
fixed and shifts t0 by small amounts; this one holds t0 fixed and rewords the candidate
action in ways that preserve its meaning (synonym swaps, clause reordering, voice
changes -- never changing which maneuver is being asserted). If the yes/no confidence
swings meaningfully across paraphrases of a claim the model should treat identically,
that is evidence the signal is picking up surface phrasing, not scene understanding.

Runs the baseline (single-shot, no scene description, no sampling) judging path from
``motion_confidence_smoke.py`` for each paraphrase of the positive and negative gold-CoC
actions of one event, at that event's own t0.

Usage
-----
    python examples/motion_confidence_phrasing_sensitivity.py \\
        --clip_id 428709c9-4a9c-4e90-bdbd-e56d430bd195 --t0_us 14100000 \\
        --out outputs/motion_confidence_phrasing_sensitivity.json
    # default positive/negative paraphrase sets are for that clip's gold CoC pair;
    # pass --positive_actions / --negative_actions to override for a different event.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_confidence_smoke as mc  # noqa: E402

DEFAULT_POSITIVE_PARAPHRASES = [
    "Steer left following temporary traffic delineators.",
    "Move to the left, guided by the temporary traffic delineators.",
    "Turn left in accordance with the temporary traffic delineators.",
    "Follow the temporary traffic delineators by steering left.",
    "Bear left as indicated by the temporary traffic delineators.",
]
DEFAULT_NEGATIVE_PARAPHRASES = [
    "Go straight following temporary traffic delineators.",
    "Continue straight, guided by the temporary traffic delineators.",
    "Proceed straight in accordance with the temporary traffic delineators.",
    "Keep going straight along the path marked by the temporary traffic delineators.",
    "Maintain a straight course as indicated by the temporary traffic delineators.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip_id", required=True)
    parser.add_argument("--t0_us", type=int, required=True)
    parser.add_argument("--positive_actions", nargs="+", default=DEFAULT_POSITIVE_PARAPHRASES)
    parser.add_argument("--negative_actions", nargs="+", default=DEFAULT_NEGATIVE_PARAPHRASES)
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_phrasing_sensitivity.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from alpamayo2_super.chat_template.conversation import construct_image, construct_system_prompt
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.helper import get_processor
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super import helper

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_phrasing_sensitivity requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    source_data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    vqa_data = select_task_input(source_data, "vqa")
    image_content = mc.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
    image_frames = vqa_data["image_frames"]

    def judge_all(actions, label):
        entries = []
        for action in actions:
            messages = [
                {"role": "system", "content": construct_system_prompt()},
                {"role": "user", "content": image_content + [
                    {"type": "text", "text": mc.BASELINE_TEMPLATE.format(action=action)}
                ]},
            ]
            conf, yes_p, no_p = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
            entries.append({"action": action, "confidence": conf, "yes_p": yes_p, "no_p": no_p})
            print(f"  [{label}] conf={conf:.3f} (yes={yes_p:.4f}, no={no_p:.4f})  '{action}'")
        return entries

    print(f"\n{args.clip_id} t0={args.t0_us}")
    print("positive paraphrases:")
    positive_results = judge_all(args.positive_actions, "positive")
    print("negative paraphrases:")
    negative_results = judge_all(args.negative_actions, "negative")

    out = {
        "clip_id": args.clip_id, "t0_us": args.t0_us,
        "positive": positive_results, "negative": negative_results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")

    for label, entries in (("positive", positive_results), ("negative", negative_results)):
        confs = [e["confidence"] for e in entries]
        print(f"\n{label}: min={min(confs):.3f} max={max(confs):.3f} "
              f"range={max(confs) - min(confs):.3f} mean={sum(confs) / len(confs):.3f}")


if __name__ == "__main__":
    main()
