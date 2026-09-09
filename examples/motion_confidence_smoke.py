# SPDX-License-Identifier: Apache-2.0
"""Smoke test: does describing the scene first, then sampling the yes/no judgment
multiple times, give a better-calibrated motion-confidence signal than a single-shot
first-token yes/no probability?

Ground truth comes from PAI-AV OOD reasoning gold CoC (``reasoning/ood_reasoning.parquet``,
loaded via ``eval_common.load_events``): each event's own gold CoC is the positive
("right") driving action, and another random event's gold CoC (paired by a cyclic shift
over the sampled set) is the negative ("wrong") action -- a real, human-written action
that is very likely inapplicable to this scene.

Four scoring methods:
  - baseline: single-turn "Is '<action>' the right driving decision?" prompt, one forward
    pass, no sampling (mirrors the notebooks/vqa.ipynb probe this is meant to improve on).
    Confidence = P(yes) / (P(yes) + P(no)), read off the first generated token's softmax.
  - digit: single-turn "On a scale from 0 to 9, how confident are you..." prompt, one
    forward pass. Confidence = probability-weighted expected digit / 9, i.e.
    sum_i(i * P(digit=i)) / (9 * sum_i P(digit=i)), read off the first generated token's
    softmax.
  - likelihood: no yes/no or digit framing at all -- teacher-force the candidate action
    text itself as the model's answer to an open "what is the right driving decision?"
    question, and score it by its own mean per-token log-likelihood under the model
    (length-normalized, since candidate actions vary in length). Doesn't need the model to
    emit any particular token; it reads out what the model already assigns probability to.
  - scene-conditioned: a neutral scene-description question (candidate action NOT in
    context, to avoid biasing the description toward confirming it) is sampled
    ``--num_samples`` times; each sampled description is then used as prior turn context
    for a separate "given the scene above, is '<action>' right?" judgment; the yes-prob is
    averaged across samples.

Usage
-----
    python examples/motion_confidence_smoke.py --num_events 10 --num_samples 5 \\
        --out outputs/motion_confidence_smoke.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common as ec  # noqa: E402

MIN_T0_US = 1_700_000  # history window (1.6s) + margin, see eval_common.py

SCENE_QUESTION = (
    "Describe the current driving scene relevant to the ego vehicle's next driving "
    "decision: nearby road users, road and lane conditions, traffic control, and any "
    "hazards. Do not mention or evaluate any specific driving decision."
)
JUDGE_TEMPLATE = (
    "Given the scene described above, is '{action}' the right driving decision for the "
    "ego vehicle right now? Answer Yes or No."
)
BASELINE_TEMPLATE = (
    "Is '{action}' the right driving decision for the ego vehicle right now? "
    "Answer Yes or No."
)
DIGIT_TEMPLATE = (
    "On a scale from 0 to 9, how confident are you that '{action}' is the right driving "
    "decision for the ego vehicle right now? 9 means certain it is correct, 0 means "
    "certain it is incorrect. Answer with a single digit only, no words."
)
LIKELIHOOD_QUESTION = "What is the right driving decision for the ego vehicle right now?"

YES_VARIANTS = ["Yes", "yes", " Yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", "no", " No", " no", "NO", " NO"]


def yes_no_confidence(tokenizer, probs):
    """probs: [vocab] softmax over the first generated token. Returns (conf, yes_p, no_p)
    where conf = yes_p / (yes_p + no_p), nan if neither appears with nonzero mass."""

    def text_prob(variants):
        total = 0.0
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                total += float(probs[ids].sum().item())
        return total

    yes_p = text_prob(YES_VARIANTS)
    no_p = text_prob(NO_VARIANTS)
    denom = yes_p + no_p
    conf = yes_p / denom if denom > 0 else float("nan")
    return conf, yes_p, no_p


def digit_confidence(tokenizer, probs):
    """probs: [vocab] softmax over the first generated token. Returns (conf, digit_probs)
    where digit_probs[i] is the probability mass on digit i, and conf is the
    probability-weighted expected digit rescaled to [0, 1] (E[digit] / 9), nan if no
    digit token appears with nonzero mass.

    Only single-token variants count: e.g. " 9" tokenizes as [space, "9"] (two tokens),
    not a single-token alternative spelling of "9" -- summing its ids would double-count
    the (digit-independent) leading-space token's probability into every digit's bucket,
    inflating and cross-contaminating all ten (verified against this tokenizer: bare
    digits "0".."9" are each one token; " 0".." 9" are each [space_id, digit_id]).
    """

    def text_prob(variants):
        total = 0.0
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                total += float(probs[ids].sum().item())
        return total

    digit_probs = [text_prob([str(d), f" {d}"]) for d in range(10)]
    denom = sum(digit_probs)
    conf = sum(i * p for i, p in enumerate(digit_probs)) / (9.0 * denom) if denom > 0 else float("nan")
    return conf, digit_probs


def build_image_content(construct_image, vqa_data, model_config, include_frame_nums):
    return construct_image(
        data=vqa_data,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=vqa_data["camera_indices"],
        include_frame_nums=include_frame_nums,
    )


def tokenize_messages(
    messages, image_frames, processor, add_generation_prompt=True, continue_final_message=False
):
    import torch

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message, add_vision_id=False,
    )
    images = image_frames.flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    return dict(
        processor(
            text=text, images=images, videos=None, padding=False,
            return_tensors="pt", do_rescale=False,
        )
    )


def _first_token_probs(model, helper, tokenizer, processor, image_frames, messages):
    tokenized = tokenize_messages(messages, image_frames, processor)
    tokenized = helper.to_device(tokenized, "cuda")
    gen_model = getattr(model, "vlm", model)
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen_out = gen_model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            pixel_values=tokenized["pixel_values"],
            image_grid_thw=tokenized["image_grid_thw"],
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    return torch.softmax(gen_out.scores[0][0].float(), dim=-1)


def judge_yes_no(model, helper, tokenizer, processor, image_frames, messages):
    probs = _first_token_probs(model, helper, tokenizer, processor, image_frames, messages)
    return yes_no_confidence(tokenizer, probs)


def judge_digit(model, helper, tokenizer, processor, image_frames, messages):
    probs = _first_token_probs(model, helper, tokenizer, processor, image_frames, messages)
    return digit_confidence(tokenizer, probs)


def score_likelihood(model, helper, tokenizer, processor, image_frames, prefix_messages, action):
    """Teacher-force ``action`` as the model's answer to prefix_messages (which must NOT
    already have an assistant turn) and return its own mean/sum per-token log-likelihood
    under the model, length-normalized (mean) to be comparable across actions of
    different lengths. Returns None if the tokenization-boundary check fails (i.e.
    appending the assistant turn changed how the prefix itself tokenizes) -- rare, but
    real, since BPE merges are not guaranteed to respect the prefix/suffix split."""
    import torch

    full_messages = prefix_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": action}]}
    ]
    prefix_tok = tokenize_messages(prefix_messages, image_frames, processor, add_generation_prompt=True)
    full_tok = tokenize_messages(
        full_messages, image_frames, processor,
        add_generation_prompt=False, continue_final_message=True,
    )
    prefix_ids = prefix_tok["input_ids"][0]
    full_ids = full_tok["input_ids"][0]
    prefix_len = prefix_ids.shape[0]
    if full_ids.shape[0] <= prefix_len or not torch.equal(full_ids[:prefix_len], prefix_ids):
        return None
    action_ids = full_ids[prefix_len:]

    full_tok = helper.to_device(full_tok, "cuda")
    gen_model = getattr(model, "vlm", model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = gen_model(
            input_ids=full_tok["input_ids"],
            attention_mask=full_tok["attention_mask"],
            pixel_values=full_tok["pixel_values"],
            image_grid_thw=full_tok["image_grid_thw"],
            use_cache=False,
        )
    logits = outputs.logits[0].float()  # [seq_len, vocab]
    logprobs = torch.log_softmax(logits, dim=-1)
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + action_ids.shape[0])
    token_logprobs = logprobs[positions, action_ids.to(logprobs.device)]
    return {
        "mean_logprob": float(token_logprobs.mean().item()),
        "sum_logprob": float(token_logprobs.sum().item()),
        "num_tokens": int(action_ids.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", default=ec.DEFAULT_PARQUET)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_events", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5, help="Scene-description samples per event.")
    parser.add_argument("--scene_max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip this many events from the front of the seed-shuffled event list before "
             "taking --num_events -- lets separate runs at the same --seed cover disjoint "
             "events (e.g. --skip 10 --num_events 40 to extend a prior --num_events 10 run).",
    )
    parser.add_argument(
        "--t0_offset_s", type=float, default=0.0,
        help="Shift t0_us by this many seconds (loader's 0.1s frame grid) before loading "
             "images for EVERY event -- same positive/negative action text, but the scene "
             "images (and thus every method's score) come from a nearby frame instead of "
             "the exact recorded t0. Clamped to MIN_T0_US per event, same as "
             "motion_confidence_time_sensitivity.py. Default 0.0 reproduces prior behavior "
             "exactly.",
    )
    parser.add_argument(
        "--num_negatives", type=int, default=1,
        help="K mismatched-gold-CoC negatives per event instead of just 1, via cyclic "
             "shifts by 1..K -- reduces the luck-of-the-draw variance of comparing "
             "against a single arbitrary negative. Schema always stores "
             "event_result[method]['negatives'] as a K-length list (even for K=1).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "digit", "likelihood", "scene_conditioned"],
        choices=["baseline", "digit", "likelihood", "scene_conditioned"],
        help="Which scoring methods to run per event -- scene_conditioned is the "
             "expensive one (1 + num_samples generations/event); drop it to re-run "
             "baseline/digit/likelihood cheaply over the same event selection.",
    )
    parser.add_argument(
        "--system_prompt", default=None,
        help="Override the default system prompt (construct_system_prompt()) for all "
             "turns/methods this run. Useful for testing whether prompt wording affects "
             "how reliably the model commits to the requested answer format.",
    )
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_smoke.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        raise RuntimeError("motion_confidence_smoke requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    system_prompt = (
        [{"type": "text", "text": args.system_prompt}] if args.system_prompt
        else construct_system_prompt()
    )

    rows = ec.load_events(args.parquet, all_events=False, split=args.split)
    rows = [r for r in rows if r["coc"]]
    rng = random.Random(args.seed)
    shuffled_rows = rows[:]
    rng.shuffle(shuffled_rows)
    sample_rows = shuffled_rows[args.skip:args.skip + args.num_events]
    # Cyclic-shift pairing: event i's K negatives are events (i+1..i+K mod N)'s gold CoC.
    neg_actions = [
        [sample_rows[(i + k) % len(sample_rows)]["coc"] for k in range(1, args.num_negatives + 1)]
        for i in range(len(sample_rows))
    ]

    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    results = []
    for i, row in enumerate(sample_rows):
        clip_id, t0_us = row["clip_id"], row["t0_us"]
        pos_action, neg_action_list = row["coc"], neg_actions[i]
        actions = [("positive", pos_action)] + [
            (f"negative_{k}", a) for k, a in enumerate(neg_action_list)
        ]
        print(f"\n[{i + 1}/{len(sample_rows)}] {clip_id} t0={t0_us}")
        print(f"  positive: {pos_action}")
        for k, a in enumerate(neg_action_list):
            print(f"  negative_{k}: {a}")

        load_t0_us = max(t0_us + round(args.t0_offset_s * 1_000_000), MIN_T0_US)
        try:
            source_data, vqa_data = ec.load_scene_with_retry(
                load_physical_aiavdataset, select_task_input, clip_id, load_t0_us
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load: {e}")
            results.append({"clip_id": clip_id, "t0_us": t0_us, "error": str(e)})
            continue

        image_content = build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
        image_frames = vqa_data["image_frames"]

        event_result = {
            "clip_id": clip_id, "t0_us": t0_us, "load_t0_us": load_t0_us,
            "positive_action": pos_action, "negative_actions": neg_action_list,
        }

        def _store(method_dict, label, value):
            if label == "positive":
                method_dict["positive"] = value
            else:
                method_dict.setdefault("negatives", []).append(value)

        # --- baseline: single-shot yes/no, no scene description, no sampling ---
        if "baseline" in args.methods:
            event_result["baseline"] = {}
            for label, action in actions:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": image_content + [
                        {"type": "text", "text": BASELINE_TEMPLATE.format(action=action)}
                    ]},
                ]
                conf, yes_p, no_p = judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
                _store(event_result["baseline"], label, {"confidence": conf, "yes_p": yes_p, "no_p": no_p})
                print(f"  baseline/{label}: conf={conf:.3f} (yes={yes_p:.3f}, no={no_p:.3f})")

        # --- digit: single-shot 0-9 confidence scale, no scene description, no sampling ---
        if "digit" in args.methods:
            event_result["digit"] = {}
            for label, action in actions:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": image_content + [
                        {"type": "text", "text": DIGIT_TEMPLATE.format(action=action)}
                    ]},
                ]
                conf, digit_probs = judge_digit(model, helper, tokenizer, processor, image_frames, messages)
                _store(event_result["digit"], label, {"confidence": conf, "digit_probs": digit_probs})
                print(f"  digit/{label}: conf={conf:.3f} digit_probs={['%.3f' % p for p in digit_probs]}")

        # --- likelihood: teacher-forced mean per-token log-likelihood of the action text ---
        if "likelihood" in args.methods:
            event_result["likelihood"] = {}
            prefix_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": image_content + [{"type": "text", "text": LIKELIHOOD_QUESTION}]},
            ]
            for label, action in actions:
                score = score_likelihood(
                    model, helper, tokenizer, processor, image_frames, prefix_messages, action
                )
                _store(event_result["likelihood"], label, score)
                if score is None:
                    print(f"  likelihood/{label}: SKIPPED (tokenization-boundary mismatch)")
                else:
                    print(f"  likelihood/{label}: mean_logprob={score['mean_logprob']:.3f} "
                          f"sum_logprob={score['sum_logprob']:.3f} num_tokens={score['num_tokens']}")

        # --- scene-conditioned: sample N scene descriptions, judge each, average ---
        if "scene_conditioned" in args.methods:
            event_result["scene_conditioned"] = {}
            torch.cuda.manual_seed_all(args.seed + i)
            vqa_inputs = prepare_vqa_inputs(
                data=vqa_data, model_config=model.config, tokenizer=tokenizer, question=SCENE_QUESTION,
            )
            vqa_inputs = helper.to_device(vqa_inputs, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scene_result = generate_text(
                    model, vqa_inputs, top_p=args.top_p, temperature=args.temperature,
                    num_samples=args.num_samples, max_new_tokens=args.scene_max_new_tokens,
                )
            descriptions = scene_result["answer"]
            event_result["scene_descriptions"] = descriptions

            for label, action in actions:
                confs, yes_ps, no_ps = [], [], []
                for desc in descriptions:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": image_content + [{"type": "text", "text": SCENE_QUESTION}]},
                        {"role": "assistant", "content": [{"type": "text", "text": desc}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": JUDGE_TEMPLATE.format(action=action)}
                        ]},
                    ]
                    conf, yes_p, no_p = judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
                    confs.append(conf)
                    yes_ps.append(yes_p)
                    no_ps.append(no_p)
                mean_conf = sum(confs) / len(confs)
                _store(event_result["scene_conditioned"], label, {
                    "mean_confidence": mean_conf, "per_sample_confidence": confs,
                    "per_sample_yes_p": yes_ps, "per_sample_no_p": no_ps,
                })
                print(f"  scene/{label}: mean_conf={mean_conf:.3f} per_sample={['%.2f' % c for c in confs]}")

        results.append(event_result)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")

    # --- summary ---
    def summarize(get_pos, get_negs, name):
        """get_pos(r) -> float|None. get_negs(r) -> list[float|None], length K.
        Reports 3 metrics: mean_conf(pos)/mean_conf(neg) (pooled over all K negatives),
        pairwise_acc = fraction of the N*K (positive, one negative) comparisons the
        positive wins (reduces to the old single-negative metric at K=1), and
        top1_acc = fraction of events where the positive beats ALL K of its negatives
        (the practically-relevant "would a verifier pick the right one out of K+1
        candidates" question -- only meaningfully different from pairwise_acc at K>1)."""
        pos_list, neg_lists = [], []
        for r in results:
            if "error" in r:
                continue
            p = get_pos(r)
            ns = get_negs(r)
            if p is None or any(x is None for x in ns):
                continue
            pos_list.append(p)
            neg_lists.append(ns)
        n = len(pos_list)
        neg_flat = [x for ns in neg_lists for x in ns]
        n_pairs = len(neg_flat)
        pairwise_wins = sum(1 for p, ns in zip(pos_list, neg_lists) for x in ns if p > x)
        pairwise_acc = pairwise_wins / n_pairs if n_pairs else float("nan")
        top1_acc = (
            sum(1 for p, ns in zip(pos_list, neg_lists) if p > max(ns)) / n if n else float("nan")
        )
        mean_pos = sum(pos_list) / n if n else float("nan")
        mean_neg = sum(neg_flat) / n_pairs if n_pairs else float("nan")
        k = n_pairs // n if n else 0
        print(f"\n{name}: n={n} K={k}  mean_conf(pos)={mean_pos:.3f}  mean_conf(neg)={mean_neg:.3f}  "
              f"pairwise_acc={pairwise_acc:.3f}  top1_acc(pos beats all K)={top1_acc:.3f}")
        if n_pairs:
            y_true = [1] * n + [0] * n_pairs
            y_score = pos_list + neg_flat
            try:
                from sklearn.metrics import roc_auc_score

                auroc = roc_auc_score(y_true, y_score)
            except ImportError:
                # Mann-Whitney U / rank-sum AUROC, no sklearn dependency.
                scored = sorted(zip(y_score, y_true), key=lambda t: t[0])
                ranks, i = {}, 0
                while i < len(scored):
                    j = i
                    while j < len(scored) and scored[j][0] == scored[i][0]:
                        j += 1
                    avg_rank = (i + 1 + j) / 2.0  # average rank over ties, 1-indexed
                    for kk in range(i, j):
                        ranks[kk] = avg_rank
                    i = j
                sum_ranks_pos = sum(ranks[kk] for kk, (_, label) in enumerate(scored) if label == 1)
                n_neg = len(neg_flat)
                auroc = (sum_ranks_pos - n * (n + 1) / 2.0) / (n * n_neg)
            print(f"  AUROC={auroc:.3f}")

    def negs(method, field):
        return lambda r: [x[field] if x else None for x in r[method].get("negatives", [])]

    if "baseline" in args.methods:
        summarize(lambda r: r["baseline"]["positive"]["confidence"], negs("baseline", "confidence"), "baseline")
    if "digit" in args.methods:
        summarize(lambda r: r["digit"]["positive"]["confidence"], negs("digit", "confidence"), "digit")
    if "likelihood" in args.methods:
        summarize(
            lambda r: r["likelihood"]["positive"]["mean_logprob"] if r["likelihood"]["positive"] else None,
            negs("likelihood", "mean_logprob"),
            "likelihood (mean logprob)",
        )
    if "scene_conditioned" in args.methods:
        summarize(
            lambda r: r["scene_conditioned"]["positive"]["mean_confidence"],
            negs("scene_conditioned", "mean_confidence"),
            "scene_conditioned",
        )


if __name__ == "__main__":
    main()
