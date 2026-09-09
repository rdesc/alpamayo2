# SPDX-License-Identifier: Apache-2.0
"""Score ALL K self-generated CoC samples per event (from motion_confidence_multi_coc.py's
output) through every motion_confidence_smoke.py method, WITHOUT picking a positive/
negative pairing -- that decision is deferred to a later, separate analysis step.

Motivation: we have 8 i.i.d. self-generated CoC candidates per val event
(``outputs/motion_confidence_multi_coc_val_n289.json``). Which one (if any) should count
as "the" self-generated action, and which should serve as within/cross-scene negatives,
is still an open question (see docs/motion_confidence_experiment.md open questions). Since
every method here (baseline/digit/likelihood/scene_conditioned) scores ONE candidate
against a scene independently of any other candidate, we can compute all 8 candidates'
raw scores now and defer positive/negative selection to pure post-processing later, with
no need to re-run the model.

Per event: loads the "vqa" scene images ONCE, and (for scene_conditioned) samples
``--num_samples`` scene descriptions ONCE (candidate action never in that context) -- both
reused across all 8 self-CoC candidate judgments for that event, exactly mirroring how
motion_confidence_smoke.py reuses one image load per event across its positive+K
negatives.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python examples/motion_confidence_multi_coc_scores.py \\
        --inputs outputs/motion_confidence_multi_coc_val_n289.json \\
        --out outputs/motion_confidence_multi_coc_scores_val_n289.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common as ec  # noqa: E402
import motion_confidence_smoke as mc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more motion_confidence_multi_coc.py outputs; events are concatenated.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "digit", "likelihood", "scene_conditioned"],
        choices=["baseline", "digit", "likelihood", "scene_conditioned"],
    )
    parser.add_argument("--num_samples", type=int, default=5, help="Scene-description samples per event.")
    parser.add_argument("--scene_max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--out", default="outputs/motion_confidence_multi_coc_scores.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # GPU selection: set CUDA_VISIBLE_DEVICES=N in the environment before launching this
    # script (matches motion_confidence_smoke.py's convention) -- no --gpu flag here, so
    # one script can't silently override another's placement.

    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.chat_template.conversation import construct_image, construct_system_prompt
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.helper import get_processor
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_multi_coc_scores requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    system_prompt = construct_system_prompt()

    events = []
    for path in args.inputs:
        d = json.load(open(path))
        events.extend(d["results"])
    events = events[args.skip:args.skip + args.limit] if args.limit else events[args.skip:]

    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    results = []
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        self_cocs = ev["self_cocs"]
        print(f"\n[{i + 1}/{len(events)}] {clip_id} t0={t0_us}  {len(self_cocs)} self-CoC candidates")

        try:
            source_data, vqa_data = ec.load_scene_with_retry(
                load_physical_aiavdataset, select_task_input, clip_id, t0_us
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load: {e}")
            results.append({"clip_id": clip_id, "t0_us": t0_us, "error": str(e)})
            continue

        image_content = mc.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
        image_frames = vqa_data["image_frames"]

        event_result = {"clip_id": clip_id, "t0_us": t0_us, "self_cocs": self_cocs, "scores": []}

        # --- scene-conditioned: sample N scene descriptions ONCE per event, reused below ---
        descriptions = None
        if "scene_conditioned" in args.methods:
            torch.cuda.manual_seed_all(args.seed + i)
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
            event_result["scene_descriptions"] = descriptions

        for sample_idx, action in enumerate(self_cocs):
            sample_scores = {}
            print(f"  sample {sample_idx}: '{action}'")

            if "baseline" in args.methods:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": image_content + [
                        {"type": "text", "text": mc.BASELINE_TEMPLATE.format(action=action)}
                    ]},
                ]
                conf, yes_p, no_p = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
                sample_scores["baseline"] = {"confidence": conf, "yes_p": yes_p, "no_p": no_p}
                print(f"    baseline: conf={conf:.3f} (yes={yes_p:.3f}, no={no_p:.3f})")

            if "digit" in args.methods:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": image_content + [
                        {"type": "text", "text": mc.DIGIT_TEMPLATE.format(action=action)}
                    ]},
                ]
                conf, digit_probs = mc.judge_digit(model, helper, tokenizer, processor, image_frames, messages)
                sample_scores["digit"] = {"confidence": conf, "digit_probs": digit_probs}
                print(f"    digit: conf={conf:.3f}")

            if "likelihood" in args.methods:
                prefix_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": image_content + [{"type": "text", "text": mc.LIKELIHOOD_QUESTION}]},
                ]
                score = mc.score_likelihood(
                    model, helper, tokenizer, processor, image_frames, prefix_messages, action
                )
                sample_scores["likelihood"] = score
                if score is None:
                    print("    likelihood: SKIPPED (tokenization-boundary mismatch)")
                else:
                    print(f"    likelihood: mean_logprob={score['mean_logprob']:.3f}")

            if "scene_conditioned" in args.methods:
                confs = []
                for desc in descriptions:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": image_content + [{"type": "text", "text": mc.SCENE_QUESTION}]},
                        {"role": "assistant", "content": [{"type": "text", "text": desc}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": mc.JUDGE_TEMPLATE.format(action=action)}
                        ]},
                    ]
                    conf, _, _ = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
                    confs.append(conf)
                mean_conf = sum(confs) / len(confs)
                sample_scores["scene_conditioned"] = {"mean_confidence": mean_conf, "per_sample_confidence": confs}
                print(f"    scene_conditioned: mean_conf={mean_conf:.3f}")

            event_result["scores"].append(sample_scores)

        results.append(event_result)

        if (i + 1) % args.checkpoint_every == 0 or i == len(events) - 1:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)
            print(f"  [checkpoint] wrote {len(results)}/{len(events)} events to {args.out}")

    print(f"\nDone. Wrote {args.out}")


if __name__ == "__main__":
    main()
