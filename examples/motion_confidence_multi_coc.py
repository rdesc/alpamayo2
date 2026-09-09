# SPDX-License-Identifier: Apache-2.0
"""Generate K independent self-generated chain-of-causation (CoC) samples per event.

Pure data-generation script -- no judging, no scoring. For each event, draws
``--num_samples`` (default 8) i.i.d. self-generated CoT/CoC strings from A2S's own
trajectory-sampling path (temperature-based sampling), via the same mechanism as stage 1
of ``motion_confidence_self_coc.py``:
    select_task_input(source, "trajectory") -> _prepare_model_inputs(..., enable_cot=True)
    -> model.sample_trajectories_from_data(..., temperature=..., num_traj_samples=N,
       diffusion_kwargs={"inference_step": 2}, return_extra=True)["cot"]

Only the CoT text matters (the trajectory itself is discarded), so diffusion steps are
kept minimal, matching the reference script.

Memory note (see eval_pai_av_val_a2.py's --sample_chunk_size): on an 80GB H100, K>=4
trajectory samples in a single ``sample_trajectories_from_data`` call OOMs; K=3 fits. This
script draws K in sequential chunks of ``--sample_chunk_size`` and concatenates
``extra["cot"]`` across chunks via ``np.concatenate(..., axis=2)`` -- statistically
equivalent to one K-sample call since chunks are i.i.d. given fixed conditioning.

Usage
-----
    python examples/motion_confidence_multi_coc.py \\
        --events_from outputs/motion_confidence_n289_part1.json outputs/motion_confidence_n289_part2.json \\
        --out outputs/motion_confidence_multi_coc_val_n289.json --gpu 0

    python examples/motion_confidence_multi_coc.py \\
        --events_from outputs/motion_confidence_train_n300.json \\
        --out outputs/motion_confidence_multi_coc_train_n300.json --gpu 1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_event_list(paths):
    rows = []
    for path in paths:
        d = json.load(open(path))
        rows.extend(
            {"clip_id": r["clip_id"], "t0_us": r["t0_us"]}
            for r in d["results"] if "error" not in r
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--events_from", nargs="+", required=True,
        help="One or more prior motion_confidence_*.json outputs; (clip_id, t0_us) pairs "
             "from all non-error result entries are concatenated (exact same events, no "
             "fresh sampling).",
    )
    parser.add_argument("--num_samples", type=int, default=8,
                         help="Number of independent self-generated CoCs per event.")
    parser.add_argument(
        "--sample_chunk_size", type=int, default=8,
        help="Max trajectory samples per forward pass; num_samples is drawn in sequential "
             "chunks of this size and concatenated (see module docstring). Default assumes "
             "a large-memory GPU; drop to 3 on an 80GB card if you OOM.",
    )
    parser.add_argument("--traj_temperature", type=float, default=0.6)
    parser.add_argument("--traj_top_p", type=float, default=0.98)
    parser.add_argument("--diffusion_steps", type=int, default=2,
                         help="Trajectory diffusion steps -- kept minimal since the "
                              "trajectory itself is discarded, only extra['cot'] is used.")
    parser.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID"))
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to use.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N events (for smoke testing).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint_every", type=int, default=10,
                         help="Write partial results to --out every N events.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    from eval_pai_av_val_a2 import _prepare_model_inputs  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("motion_confidence_multi_coc requires a CUDA GPU.")
    torch.cuda.manual_seed_all(args.seed)

    model_id = args.model_id or PUBLIC_MODEL_ID
    events = load_event_list(args.events_from)
    if args.limit:
        events = events[:args.limit]

    print(f"Loading model {model_id} on physical GPU {args.gpu} (cuda:0 after masking) ...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = model.tokenizer

    chunk = max(1, min(args.sample_chunk_size, args.num_samples))
    chunks = [
        min(chunk, args.num_samples - i)
        for i in range(0, args.num_samples, chunk)
    ]
    print(f"n_events={len(events)}  num_samples={args.num_samples}  chunks={chunks}")

    def write_out(results):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        os.replace(tmp, args.out)

    results = []
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            source = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            traj_data = select_task_input(source, "trajectory")
            model_inputs = _prepare_model_inputs(
                traj_data, model.config, tokenizer, helper, enable_cot=True
            )
            model_inputs = helper.to_device(model_inputs, "cuda")
            cot_parts = []
            for n in chunks:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, _, extra = model.sample_trajectories_from_data(
                        data=model_inputs, top_p=args.traj_top_p, temperature=args.traj_temperature,
                        num_traj_samples=n, diffusion_kwargs={"inference_step": args.diffusion_steps},
                        return_extra=True,
                    )
                cot_parts.append(np.asarray(extra["cot"]))
                del extra
                torch.cuda.empty_cache()
            merged = np.concatenate(cot_parts, axis=2)
            self_cocs = [str(x) for x in merged.reshape(-1).tolist()]
            if len(self_cocs) != args.num_samples:
                raise RuntimeError(
                    f"expected {args.num_samples} CoCs, got {len(self_cocs)} "
                    f"(merged shape {merged.shape})"
                )
            entry = {"clip_id": clip_id, "t0_us": t0_us, "self_cocs": self_cocs}
            print(f"[{i + 1}/{len(events)}] {clip_id} t0={t0_us}  OK  "
                  f"example='{self_cocs[0][:80]}...'")
        except Exception as e:  # noqa: BLE001
            entry = {"clip_id": clip_id, "t0_us": t0_us, "error": str(e)}
            print(f"[{i + 1}/{len(events)}] {clip_id} t0={t0_us}  FAILED: {e}")
        results.append(entry)

        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(events):
            write_out(results)

    write_out(results)
    print(f"\nWrote {args.out}")
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"done: {n_ok}/{len(results)} ok, {len(results) - n_ok} failed")


if __name__ == "__main__":
    main()
