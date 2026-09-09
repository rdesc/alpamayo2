# SPDX-License-Identifier: Apache-2.0
"""Does scoring the candidate action through the model's REAL trained CoT-generation
conditioning (image + trajectory-history tokens + the fixed "output the chain-of-thought
reasoning..." instruction, teacher-forced with real ``cot_start``/``cot_end`` special
tokens) give a better likelihood-based motion-confidence signal than the existing ad-hoc
``likelihood`` method in ``motion_confidence_smoke.py`` (which asks an open natural-language
question with no trajectory-history conditioning and teacher-forces the raw action text with
no special tokens)?

Background
----------
``motion_confidence_smoke.py``'s baseline/digit/scene_conditioned methods build messages via
the same skeleton as ``text_tasks.build_text_task_messages(task="vqa")`` (image + plain-text
question, no special tokens, no trajectory-history) -- that IS the model's real, documented,
public "no-special VQA generation" inference contract, so those three methods are correct as
they stand and are NOT touched here.

``likelihood`` is different: teacher-forcing raw action text as the answer to an open VQA
question is not how the model was ever trained to produce CoT text. The model's real
CoT-generation pathway (used by ``eval_pai_av_val_a2.py`` / ``truckdrive_eval_val.py`` for
real inference) conditions on discretized trajectory-history tokens fused into a
``traj_history`` placeholder span, and wraps CoT text in ``<|cot_start|>``/``<|cot_end|>``
special tokens. ``native_likelihood`` (this script) reproduces that exact pathway:

    build_conversation(components_order=["image","traj_history","prompt"],
                        components_prompt=["cot"], generation_mode=True, ...)
    -> drop the empty trailing assistant message
    -> tokenize the prefix (add_generation_prompt=True)
    -> fuse_traj_tokens(...) to replace the traj_history placeholder span with real
       discretized ego_history_xyz/rot tokens
    -> teacher-force the candidate action wrapped via construct_cot({"cot": action}) as the
       assistant turn, tokenized with continue_final_message=True
    -> same prefix-boundary check + per-token log-likelihood scoring as score_likelihood()

This script runs BOTH methods (``likelihood`` via direct reuse of ``score_likelihood`` /
``LIKELIHOOD_QUESTION`` from motion_confidence_smoke.py, and ``native_likelihood``) on the
SAME sampled events for a clean side-by-side comparison. Same event-sampling / cyclic-shift
K-negative pairing convention as motion_confidence_smoke.py.

Usage
-----
    python examples/motion_confidence_native_cot.py --num_events 3 --debug \\
        --out outputs/motion_confidence_native_cot_sanity.json
    python examples/motion_confidence_native_cot.py --num_events 20 --num_negatives 3 \\
        --seed 0 --split val --out outputs/motion_confidence_native_cot_n20.json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common as ec  # noqa: E402
import motion_confidence_smoke as ms  # noqa: E402

MIN_T0_US = ms.MIN_T0_US


def score_native_likelihood(
    model, helper, tokenizer, processor, vqa_data, action, cot_start_id, cot_end_id,
    debug: bool = False,
):
    """Teacher-force ``action`` as the CoT-generation assistant turn, using the model's
    real trained conditioning (image + fused traj-history tokens + fixed CoT-generation
    instruction), and return its mean/sum per-token log-likelihood under the model.

    Returns None if the tokenization-boundary check fails (same rare BPE-merge caveat as
    ``score_likelihood``).
    """
    import torch

    from alpamayo2_super.chat_template.conversation import build_conversation, construct_cot
    from alpamayo2_super.models.utils import fuse_traj_tokens

    model_config = model.config
    include_frame_nums = getattr(model_config, "frame_label", "frame_num") == "frame_num"

    base_messages = build_conversation(
        data=vqa_data,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=["cot"],
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=vqa_data["camera_indices"],
        include_frame_nums=include_frame_nums,
    )
    if base_messages[-1]["role"] == "assistant" and not base_messages[-1]["content"]:
        prefix_messages = base_messages[:-1]
    else:
        prefix_messages = base_messages
    full_messages = prefix_messages + [
        {"role": "assistant", "content": construct_cot({"cot": action}, ask_for_component=False)}
    ]

    image_frames = vqa_data["image_frames"]

    def _tokenize(messages, add_generation_prompt, continue_final_message):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message, add_vision_id=False,
        )
        images = image_frames.flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tok = dict(processor(
            text=text, images=images, videos=None, padding=False,
            return_tensors="pt", do_rescale=False,
        ))
        return tok, text

    prefix_tok, prefix_text = _tokenize(prefix_messages, add_generation_prompt=True, continue_final_message=False)
    full_tok, full_text = _tokenize(full_messages, add_generation_prompt=False, continue_final_message=True)

    if debug:
        print(f"    [debug] prefix text tail: ...{prefix_text[-200:]!r}")
        print(f"    [debug] full text tail:   ...{full_text[-200:]!r}")
        print(f"    [debug] ego_history_xyz.shape={tuple(vqa_data['ego_history_xyz'].shape)} "
              f"ego_history_rot.shape={tuple(vqa_data['ego_history_rot'].shape)}")

    prefix_ids = prefix_tok["input_ids"][0]
    full_ids = full_tok["input_ids"][0]
    prefix_len = prefix_ids.shape[0]
    if full_ids.shape[0] <= prefix_len or not torch.equal(full_ids[:prefix_len], prefix_ids):
        if debug:
            print("    [debug] BOUNDARY CHECK FAILED")
        return None
    appended_ids = full_ids[prefix_len:]  # outside the traj_history placeholder span, unaffected by fuse

    if debug:
        print(f"    [debug] boundary check OK, prefix_len={prefix_len}, appended_ids={appended_ids.tolist()}")
        print(f"    [debug] appended decoded: {tokenizer.decode(appended_ids)!r}")

    traj_data = {
        "ego_history_xyz": vqa_data["ego_history_xyz"],
        "ego_history_rot": vqa_data["ego_history_rot"],
    }
    # Guard the placeholder-count assertion in replace_pad_token: fuse both prefix and full
    # so the traj_history span is real in both (they share the identical placeholder count
    # since they share the identical prefix text).
    prefix_tok["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer, model.future_traj_tokenizer,
        prefix_tok["input_ids"], traj_data, model_config.traj_ids,
    )
    full_tok["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer, model.future_traj_tokenizer,
        full_tok["input_ids"], traj_data, model_config.traj_ids,
    )

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
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + appended_ids.shape[0])
    token_logprobs = logprobs[positions, appended_ids.to(logprobs.device)]

    result = {
        "mean_logprob": float(token_logprobs.mean().item()),
        "sum_logprob": float(token_logprobs.sum().item()),
        "num_tokens": int(appended_ids.shape[0]),
    }
    # cot_start/cot_end are dedicated special tokens (tokenize atomically regardless of
    # neighboring text), so the appended span is [cot_start, <action tokens...>, cot_end]
    # whenever construct_cot wrapped a non-empty action -- split it out when that holds.
    if (
        appended_ids.shape[0] >= 3
        and int(appended_ids[0].item()) == cot_start_id
        and int(appended_ids[-1].item()) == cot_end_id
    ):
        action_logprobs = token_logprobs[1:-1]
        result["action_only_mean_logprob"] = float(action_logprobs.mean().item())
        result["action_only_sum_logprob"] = float(action_logprobs.sum().item())
        result["action_only_num_tokens"] = int(action_logprobs.shape[0])
    return result


def summarize(results, get_pos, get_negs, name):
    """Same metric definitions as motion_confidence_smoke.py's summarize(): pairwise_acc
    (all N*K positive-vs-one-negative comparisons), top1_acc (positive beats all K
    negatives), AUROC."""
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
    top1_acc = sum(1 for p, ns in zip(pos_list, neg_lists) if p > max(ns)) / n if n else float("nan")
    mean_pos = sum(pos_list) / n if n else float("nan")
    mean_neg = sum(neg_flat) / n_pairs if n_pairs else float("nan")
    k = n_pairs // n if n else 0
    print(f"\n{name}: n={n} K={k}  mean(pos)={mean_pos:.3f}  mean(neg)={mean_neg:.3f}  "
          f"pairwise_acc={pairwise_acc:.3f}  top1_acc(pos beats all K)={top1_acc:.3f}")
    auroc = float("nan")
    if n_pairs:
        y_true = [1] * n + [0] * n_pairs
        y_score = pos_list + neg_flat
        try:
            from sklearn.metrics import roc_auc_score

            auroc = roc_auc_score(y_true, y_score)
        except ImportError:
            scored = sorted(zip(y_score, y_true), key=lambda t: t[0])
            ranks, i = {}, 0
            while i < len(scored):
                j = i
                while j < len(scored) and scored[j][0] == scored[i][0]:
                    j += 1
                avg_rank = (i + 1 + j) / 2.0
                for kk in range(i, j):
                    ranks[kk] = avg_rank
                i = j
            sum_ranks_pos = sum(ranks[kk] for kk, (_, label) in enumerate(scored) if label == 1)
            n_neg = len(neg_flat)
            auroc = (sum_ranks_pos - n * (n + 1) / 2.0) / (n * n_neg)
        print(f"  AUROC={auroc:.3f}")
    return {"n": n, "k": k, "mean_pos": mean_pos, "mean_neg": mean_neg,
            "pairwise_acc": pairwise_acc, "top1_acc": top1_acc, "auroc": auroc}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", default=ec.DEFAULT_PARQUET)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_events", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_native_cot.json")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index to run on.")
    parser.add_argument("--debug", action="store_true",
                         help="Print per-event tokenization diagnostics (prefix/full text "
                              "tails, trajectory shapes, boundary-check outcome) -- intended "
                              "for the small-n sanity check, noisy at scale.")
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
    from alpamayo2_super.models.utils import SPECIAL_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_native_cot requires a CUDA GPU.")

    device = f"cuda:{args.gpu}"
    model_id = args.model_id or PUBLIC_MODEL_ID
    system_prompt = construct_system_prompt()

    rows = ec.load_events(args.parquet, all_events=False, split=args.split)
    rows = [r for r in rows if r["coc"]]
    rng = random.Random(args.seed)
    shuffled_rows = rows[:]
    rng.shuffle(shuffled_rows)
    sample_rows = shuffled_rows[args.skip:args.skip + args.num_events]
    neg_actions = [
        [sample_rows[(i + k) % len(sample_rows)]["coc"] for k in range(1, args.num_negatives + 1)]
        for i in range(len(sample_rows))
    ]

    print(f"Loading model {model_id} on {device} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    cot_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_start"])
    cot_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_end"])
    print(f"cot_start_id={cot_start_id} cot_end_id={cot_end_id}")

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

        try:
            source_data, vqa_data = ec.load_scene_with_retry(
                load_physical_aiavdataset, select_task_input, clip_id, t0_us
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load: {e}")
            results.append({"clip_id": clip_id, "t0_us": t0_us, "error": str(e)})
            continue

        image_content = ms.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
        image_frames = vqa_data["image_frames"]

        event_result = {
            "clip_id": clip_id, "t0_us": t0_us,
            "positive_action": pos_action, "negative_actions": neg_action_list,
            "likelihood": {}, "native_likelihood": {},
        }

        def _store(method_dict, label, value):
            if label == "positive":
                method_dict["positive"] = value
            else:
                method_dict.setdefault("negatives", []).append(value)

        # --- old ad-hoc likelihood (unmodified, reused directly from motion_confidence_smoke) ---
        prefix_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": ms.LIKELIHOOD_QUESTION}]},
        ]
        for label, action in actions:
            score = ms.score_likelihood(
                model, helper, tokenizer, processor, image_frames, prefix_messages, action
            )
            _store(event_result["likelihood"], label, score)
            if score is None:
                print(f"  likelihood/{label}: SKIPPED (tokenization-boundary mismatch)")
            else:
                print(f"  likelihood/{label}: mean_logprob={score['mean_logprob']:.3f} "
                      f"num_tokens={score['num_tokens']}")

        # --- new native CoT-format likelihood ---
        for label, action in actions:
            debug = args.debug and i == 0 and label == "positive"
            score = score_native_likelihood(
                model, helper, tokenizer, processor, vqa_data, action,
                cot_start_id, cot_end_id, debug=debug,
            )
            _store(event_result["native_likelihood"], label, score)
            if score is None:
                print(f"  native_likelihood/{label}: SKIPPED (tokenization-boundary mismatch)")
            else:
                extra = ""
                if "action_only_mean_logprob" in score:
                    extra = f" action_only_mean_logprob={score['action_only_mean_logprob']:.3f}"
                print(f"  native_likelihood/{label}: mean_logprob={score['mean_logprob']:.3f} "
                      f"num_tokens={score['num_tokens']}{extra}")

        results.append(event_result)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")

    def negs(method, field):
        return lambda r: [x[field] if x else None for x in r[method].get("negatives", [])]

    print("\n" + "=" * 70)
    summary = {}
    summary["likelihood"] = summarize(
        results,
        lambda r: r["likelihood"]["positive"]["mean_logprob"] if r["likelihood"].get("positive") else None,
        negs("likelihood", "mean_logprob"),
        "likelihood (ad-hoc, mean logprob)",
    )
    summary["native_likelihood"] = summarize(
        results,
        lambda r: r["native_likelihood"]["positive"]["mean_logprob"] if r["native_likelihood"].get("positive") else None,
        negs("native_likelihood", "mean_logprob"),
        "native_likelihood (real CoT-generation contract, mean logprob)",
    )
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
