# SPDX-License-Identifier: Apache-2.0
"""One worked example of "faithful" (self-consistency) calibration, as distinct from the
"factual" (ground-truth) calibration in motion_confidence_smoke.py.

Two separate acts, same image, same candidate action:
  1. Ask the model to answer Yes/No AND verbalize a numeric confidence (0-100%) in one
     greedy generation -- a self-report.
  2. Separately, resample the PLAIN yes/no question (no confidence request) N times at
     a fixed temperature, decoding the actual sampled token each time (not reading the
     softmax) -- the model's empirical behavioral frequency.

Faithful calibration asks: does (1)'s stated number match (2)'s empirical frequency?
This needs no ground truth at all -- it's a question about self-consistency, not
correctness.

Usage
-----
    python examples/motion_confidence_faithfulness_example.py \\
        --clip_id 428709c9-4a9c-4e90-bdbd-e56d430bd195 --t0_us 14100000 \\
        --action "Steer left following temporary traffic delineators." \\
        --out outputs/motion_confidence_faithfulness_example.json
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_confidence_smoke as mc  # noqa: E402

VERBALIZED_TEMPLATE = (
    "Is '{action}' the right driving decision for the ego vehicle right now? "
    "First answer Yes or No, then on a new line state your confidence as a percentage "
    "(0-100%) that your answer is correct."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip_id", required=True)
    parser.add_argument("--t0_us", type=int, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--num_resamples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_faithfulness_example.json")
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
        raise RuntimeError("motion_confidence_faithfulness_example requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"
    gen_model = getattr(model, "vlm", model)

    source_data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    vqa_data = select_task_input(source_data, "vqa")
    image_content = mc.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
    image_frames = vqa_data["image_frames"]

    # --- act 1: verbalized confidence, one greedy generation ---
    messages = [
        {"role": "system", "content": construct_system_prompt()},
        {"role": "user", "content": image_content + [
            {"type": "text", "text": VERBALIZED_TEMPLATE.format(action=args.action)}
        ]},
    ]
    tokenized = mc.tokenize_messages(messages, image_frames, processor)
    tokenized = helper.to_device(tokenized, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = gen_model.generate(
            input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"],
            pixel_values=tokenized["pixel_values"], image_grid_thw=tokenized["image_grid_thw"],
            max_new_tokens=60, do_sample=False, return_dict_in_generate=True,
        )
    prompt_len = tokenized["input_ids"].shape[1]
    verbalized_text = tokenizer.decode(gen_out.sequences[0][prompt_len:], skip_special_tokens=True).strip()
    match = re.search(r"(\d{1,3})\s*%", verbalized_text)
    p_stated = float(match.group(1)) / 100.0 if match else None

    print("\n=== Act 1: verbalized confidence (greedy) ===")
    print(f"prompt: {VERBALIZED_TEMPLATE.format(action=args.action)}")
    print(f"raw output: {verbalized_text!r}")
    print(f"parsed p_stated: {p_stated}")

    # --- act 2: resample the plain yes/no question N times, decode actual sampled tokens ---
    yes_ids = set()
    for v in mc.YES_VARIANTS:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    no_ids = set()
    for v in mc.NO_VARIANTS:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])

    messages_plain = [
        {"role": "system", "content": construct_system_prompt()},
        {"role": "user", "content": image_content + [
            {"type": "text", "text": mc.BASELINE_TEMPLATE.format(action=args.action)}
        ]},
    ]
    tokenized_plain = mc.tokenize_messages(messages_plain, image_frames, processor)
    tokenized_plain = helper.to_device(tokenized_plain, "cuda")
    torch.cuda.manual_seed_all(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = gen_model.generate(
            input_ids=tokenized_plain["input_ids"], attention_mask=tokenized_plain["attention_mask"],
            pixel_values=tokenized_plain["pixel_values"], image_grid_thw=tokenized_plain["image_grid_thw"],
            max_new_tokens=1, do_sample=True, temperature=args.temperature, top_p=args.top_p,
            num_return_sequences=args.num_resamples, return_dict_in_generate=True,
        )
    prompt_len_plain = tokenized_plain["input_ids"].shape[1]
    sampled_token_ids = gen_out.sequences[:, prompt_len_plain].tolist()
    sampled_tokens = [tokenizer.decode([tid]) for tid in sampled_token_ids]
    classified = [
        "yes" if tid in yes_ids else ("no" if tid in no_ids else "other")
        for tid in sampled_token_ids
    ]
    n_yes = classified.count("yes")
    n_no = classified.count("no")
    n_other = classified.count("other")
    p_empirical = n_yes / (n_yes + n_no) if (n_yes + n_no) > 0 else None

    print(f"\n=== Act 2: {args.num_resamples} plain resamples (temperature={args.temperature}, top_p={args.top_p}) ===")
    print(f"prompt: {mc.BASELINE_TEMPLATE.format(action=args.action)}")
    for tok, cls in zip(sampled_tokens, classified):
        print(f"  sampled token: {tok!r:12s} -> {cls}")
    print(f"n_yes={n_yes} n_no={n_no} n_other={n_other}  p_empirical={p_empirical}")

    print("\n=== Comparison ===")
    print(f"p_stated (verbalized)  = {p_stated}")
    print(f"p_empirical (resample) = {p_empirical}")
    if p_stated is not None and p_empirical is not None:
        print(f"gap = {abs(p_stated - p_empirical):.3f}")

    out = {
        "clip_id": args.clip_id, "t0_us": args.t0_us, "action": args.action,
        "verbalized_text": verbalized_text, "p_stated": p_stated,
        "sampled_tokens": sampled_tokens, "classified": classified,
        "n_yes": n_yes, "n_no": n_no, "n_other": n_other, "p_empirical": p_empirical,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
