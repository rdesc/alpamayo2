# SPDX-License-Identifier: Apache-2.0
"""Independent verification of the "shifted-CoC" negatives used by
``motion_confidence_smoke.py`` (and friends): those negatives are built by cyclically
pairing event i with event (i+k)'s gold CoC text, and are *assumed* inapplicable to
event i's scene purely by construction. This script uses a second, independent VLM
(Qwen3-VL-32B-Instruct, NOT Alpamayo2Super) as a judge to check that assumption: for
each event, for each of its K negative action strings, show the judge the actual scene
images (current-frame 6-camera ring, same "vqa" input profile Alpamayo uses) and ask,
NEUTRALLY (the gold/positive CoC is never mentioned), whether the candidate action is a
reasonable and correct driving decision for the ego vehicle right now.

Judge reliability (v2): same CoT-then-answer prompting + 3-sample majority vote as
``motion_confidence_self_coc_qa.py`` (see ``docs/motion_confidence_experiment.md``
Experiment 7 and ``motion_confidence_judge_common.py``) -- applied here for
consistency. NOTE: the full-scale 867-call run with the OLD verdict-first/single-sample
judge is already saved at ``outputs/motion_confidence_negative_qa_qwen3vl32b.json``;
this script is updated and ready to re-run with the new judge but has NOT been re-run
at full scale yet.

"No" = confirmed good negative (genuinely inapplicable, as intended by construction).
"Yes" / "Uncertain" / "no_majority" / unparseable = FLAG for human review -- the
shifted CoC might coincidentally still fit this scene.

Usage
-----
    python examples/motion_confidence_negative_qa.py \\
        --inputs outputs/motion_confidence_k3_n289_part1.json \\
                 outputs/motion_confidence_k3_n289_part2.json \\
        --out outputs/motion_confidence_negative_qa_qwen3vl32b_v2.json
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_confidence_judge_common import (  # noqa: E402
    JUDGE_PROMPT_COT,
    build_pil_images,
    generate_judge_samples,
)


def load_events_from_outputs(paths):
    """Read motion_confidence_smoke.py output JSON(s) and return a flat list of
    {clip_id, t0_us, negative_actions: [str, ...]} for every event that loaded
    successfully (has "negative_actions") -- skips events recorded with an "error"."""
    events = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for r in payload["results"]:
            if "error" in r or "negative_actions" not in r:
                continue
            events.append({
                "clip_id": r["clip_id"],
                "t0_us": r["t0_us"],
                "negative_actions": r["negative_actions"],
            })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/motion_confidence_negative_qa_qwen3vl32b.json")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index to pin the judge model to.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Increased from the verdict-first default (96) to leave room for CoT reasoning before the verdict. 384 was tried first and observed truncating a non-trivial fraction (~18%) of samples mid-reasoning before they reached a final verdict line; 512 was chosen to make that rare.")
    parser.add_argument("--num_judge_samples", type=int, default=3, help="Number of independent judge samples per candidate; final verdict is the 2-of-3 majority.")
    parser.add_argument("--judge_temperature", type=float, default=0.7)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--limit_events", type=int, default=None, help="For smoke-testing: only process the first N events.")
    parser.add_argument("--limit_negatives", type=int, default=None, help="For smoke-testing: only process the first N negatives per event.")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    device = f"cuda:{args.gpu}"
    events = load_events_from_outputs(args.inputs)
    if args.limit_events:
        events = events[: args.limit_events]
    print(f"Loaded {len(events)} events from {args.inputs}")

    print(f"Loading judge model {args.model_id} on {device} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    records = []
    counts = {"Yes": 0, "No": 0, "Uncertain": 0, "no_majority": 0, "unparseable": 0}

    for ei, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        neg_actions = ev["negative_actions"]
        if args.limit_negatives:
            neg_actions = neg_actions[: args.limit_negatives]
        print(f"\n[{ei + 1}/{len(events)}] {clip_id} t0={t0_us} ({len(neg_actions)} negatives)")

        try:
            source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            vqa_data = select_task_input(source_data, "vqa")
            pil_images = build_pil_images(source_data, vqa_data)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load scene: {e}")
            traceback.print_exc()
            for k, action in enumerate(neg_actions, start=1):
                records.append({
                    "clip_id": clip_id, "t0_us": t0_us, "negative_index": k,
                    "negative_action": action, "majority_verdict": "unparseable",
                    "raw_samples": [{"verdict": "unparseable", "explanation": f"SCENE LOAD ERROR: {e}"}],
                })
                counts["unparseable"] += 1
            continue

        for k, action in enumerate(neg_actions, start=1):
            samples, majority_verdict = generate_judge_samples(
                model, processor, process_vision_info, device, pil_images, action,
                prompt_template=JUDGE_PROMPT_COT,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_judge_samples,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
            )

            counts[majority_verdict] = counts.get(majority_verdict, 0) + 1
            record = {
                "clip_id": clip_id, "t0_us": t0_us, "negative_index": k,
                "negative_action": action, "majority_verdict": majority_verdict,
                "raw_samples": samples,
            }
            records.append(record)
            raw_verdicts = [s["verdict"] for s in samples]
            print(f"  negative_{k}: majority={majority_verdict}  raw_verdicts={raw_verdicts}")
            print(f"    [0] {samples[0]['explanation'][:200]!r}")

        if (ei + 1) % 10 == 0 or ei == len(events) - 1:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({
                    "args": vars(args), "judge_model": args.model_id,
                    "counts": counts, "records": records,
                }, f, indent=2)
            print(f"  [checkpoint] wrote {len(records)} records to {args.out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args), "judge_model": args.model_id,
            "counts": counts, "records": records,
        }, f, indent=2)

    print(f"\nWrote {len(records)} records to {args.out}")
    print(f"Verdict counts: {counts}")


if __name__ == "__main__":
    main()
