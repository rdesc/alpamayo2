# SPDX-License-Identifier: Apache-2.0
"""Same calibration benchmark as motion_confidence_smoke.py's baseline method, but with
the model's OWN generated CoT (chain-of-thought reasoning -- Alpamayo 2 Super's
equivalent of a predicted CoC) as the candidate action, instead of PAI-AV's gold
human-written CoC.

Motivation: everything in motion_confidence_smoke.py asks "is this HUMAN-written action
right?" -- this asks whether the model is equally (more? less?) well-calibrated about
its OWN utterances. If self-referential calibration is much better than gold-referential
calibration, that is evidence of a self-confirmation/sycophancy bias (the model trusts
what it would say more than it trusts an equally-plausible human statement) rather than
genuine scene-grounded judgment.

Two data loads per event, matching each mechanism's trained input contract:
  - "trajectory" task profile (6 cams [0,1,2,3,5,6], the model's trained CoT/trajectory
    conditioning) to generate the model's own CoT via
    ``model.sample_trajectories_from_data(..., return_extra=True)["cot"]`` (num_traj_samples=1,
    minimal diffusion steps since the trajectory itself is discarded -- only the CoT text
    is used).
  - "vqa" task profile (6 cams [0,1,2,3,4,5], same as motion_confidence_smoke.py) for the
    actual yes/no confidence judging step, so the judging step stays on the same input
    contract as every other experiment in this series.

Ground truth pairing mirrors motion_confidence_smoke.py exactly: positive = event i's own
self-generated CoT, negative = event (i+1 mod N)'s self-generated CoT (cyclic shift).

Usage
-----
    python examples/motion_confidence_self_coc.py --events_from outputs/motion_confidence_smoke.json \\
        --out outputs/motion_confidence_self_coc_smoke.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common as ec  # noqa: E402
import motion_confidence_smoke as mc  # noqa: E402


def load_event_list(args):
    if args.events_from:
        rows = []
        for path in args.events_from:
            d = json.load(open(path))
            rows.extend(
                {"clip_id": r["clip_id"], "t0_us": r["t0_us"]}
                for r in d["results"] if "error" not in r
            )
        return rows
    import random

    rows = ec.load_events(args.parquet, all_events=False, split=args.split)
    rows = [r for r in rows if r["coc"]]
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[args.skip:args.skip + args.num_events]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--events_from", nargs="+", default=None,
        help="Reuse the exact (clip_id, t0_us) list from one or more prior "
             "motion_confidence_smoke.py JSON outputs, for apples-to-apples comparison. "
             "If omitted, falls back to --parquet/--split/--seed/--skip/--num_events "
             "(same sampling logic as motion_confidence_smoke.py).",
    )
    parser.add_argument("--parquet", default=ec.DEFAULT_PARQUET)
    parser.add_argument("--split", default="val", choices=["val", "train", "both"])
    parser.add_argument("--num_events", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj_temperature", type=float, default=0.6)
    parser.add_argument("--traj_top_p", type=float, default=0.98)
    parser.add_argument("--diffusion_steps", type=int, default=2,
                         help="Trajectory diffusion steps -- kept minimal since the "
                              "trajectory itself is discarded, only extra['cot'] is used.")
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--out", default="outputs/motion_confidence_self_coc.json")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.chat_template.conversation import construct_image, construct_system_prompt
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.helper import get_processor
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_pai_av_val_a2 import _prepare_model_inputs  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_self_coc requires a CUDA GPU.")

    model_id = args.model_id or PUBLIC_MODEL_ID
    events = load_event_list(args)
    print(f"Loading model {model_id} ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer
    processor = get_processor(tokenizer, model.config)
    include_frame_nums = getattr(model.config, "frame_label", "frame_num") == "frame_num"

    # --- stage 1: generate the model's own CoT/CoC for every event ---
    self_cocs = []
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            source = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            traj_data = select_task_input(source, "trajectory")
            model_inputs = _prepare_model_inputs(
                traj_data, model.config, tokenizer, helper, enable_cot=True
            )
            model_inputs = helper.to_device(model_inputs, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, _, extra = model.sample_trajectories_from_data(
                    data=model_inputs, top_p=args.traj_top_p, temperature=args.traj_temperature,
                    num_traj_samples=1, diffusion_kwargs={"inference_step": args.diffusion_steps},
                    return_extra=True,
                )
            self_coc = str(np.asarray(extra["cot"]).reshape(-1)[0])
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(events)}] {clip_id} t0={t0_us}  FAILED to generate CoT: {e}")
            self_coc = None
        print(f"[{i + 1}/{len(events)}] {clip_id} t0={t0_us}  self_coc={self_coc!r}")
        self_cocs.append(self_coc)

    # --- stage 2: cyclic-shift pairing + baseline yes/no judging on "vqa" profile images ---
    n = len(events)
    neg_cocs = [self_cocs[(i + 1) % n] for i in range(n)]

    results = []
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        pos_action, neg_action = self_cocs[i], neg_cocs[i]
        entry = {"clip_id": clip_id, "t0_us": t0_us, "positive_action": pos_action, "negative_action": neg_action}
        if pos_action is None or neg_action is None:
            entry["error"] = "missing self-generated CoT"
            results.append(entry)
            continue
        try:
            source = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            vqa_data = select_task_input(source, "vqa")
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)
            results.append(entry)
            continue
        image_content = mc.build_image_content(construct_image, vqa_data, model.config, include_frame_nums)
        image_frames = vqa_data["image_frames"]

        entry["baseline"] = {}
        for label, action in (("positive", pos_action), ("negative", neg_action)):
            messages = [
                {"role": "system", "content": construct_system_prompt()},
                {"role": "user", "content": image_content + [
                    {"type": "text", "text": mc.BASELINE_TEMPLATE.format(action=action)}
                ]},
            ]
            conf, yes_p, no_p = mc.judge_yes_no(model, helper, tokenizer, processor, image_frames, messages)
            entry["baseline"][label] = {"confidence": conf, "yes_p": yes_p, "no_p": no_p}
            print(f"  baseline/{label}: conf={conf:.3f} (yes={yes_p:.3f}, no={no_p:.3f})  '{action}'")
        results.append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")

    ok = [r for r in results if "error" not in r]
    pos = [r["baseline"]["positive"]["confidence"] for r in ok]
    neg = [r["baseline"]["negative"]["confidence"] for r in ok]
    n_ok = len(pos)
    if n_ok:
        pairwise_acc = sum(1 for p, ng in zip(pos, neg) if p > ng) / n_ok
        print(f"\nself-coc baseline: n={n_ok}  mean_conf(pos)={sum(pos)/n_ok:.3f}  "
              f"mean_conf(neg)={sum(neg)/n_ok:.3f}  pairwise_acc(pos>neg)={pairwise_acc:.3f}")
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score([1] * n_ok + [0] * n_ok, pos + neg)
            print(f"  AUROC={auroc:.3f}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
