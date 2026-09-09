# SPDX-License-Identifier: Apache-2.0
"""Phase-0 LLR diagnostic (langforce_readme.md Sec 4.2, action-direction) for Alpamayo 2 Super.

Ported from the Alpamayo-1.5 version at
``/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_x_rl/scripts_fork/
phase0_llr_action_direction.py`` -- read that file's docstring for the full rationale
(especially: why this uses NATURAL training order throughout rather than the reordered
``[v, a*, cot]`` sequence Sec 4.1 describes, which an earlier control run showed to be ~95%
positional-novelty artifact, not content signal).

Measures, with ZERO training:

    llr_act = logp_num - logp_den
    logp_num = mean_a[ log p(a* | v, ell) ]        -- Pass C: cot span = real gold CoC text
    logp_den = mean_a[ log p(a* | v, ell_blanked) ] -- Pass D: cot span content-blanked (same
                                                         position/length), rest of sequence
                                                         byte-identical to Pass C

Alpamayo 2 Super confirmed this session to have the SAME dual-loss structure as Alpamayo-1.5's
Stage-1 ``TrainableReasoningVLA``: ``Alpamayo2Super.forward(tokenized_data, traj_data,
labels_mask=None)`` (``src/alpamayo2_super/models/alpamayo2_super.py``) fuses ground-truth
trajectory into the ``traj_future`` placeholder span via ``fuse_traj_tokens``, and returns
``loss_future_traj`` -- next-token CE over exactly the trajectory-token span (identified
internally via ``config.traj_ids``), regardless of ``labels_mask``. So exactly as in the 1.5
script, no custom span-gather code is needed: ``logp = -out.loss_future_traj``.

Key differences from the 1.5 script (this repo's own conventions/APIs):
  - Training-mode conversation built via ``build_conversation`` (not
    ``get_preprocess_data_fn_from_model_config``), ``components_order=["image","traj_history",
    "prompt","cot","traj_future"]``, ``components_prompt=["cot","traj_future"]``,
    ``generation_mode=False`` (so real cot content, not an ask-for-component placeholder).
  - No ``get_label_mask`` utility exists in this repo. The cot span is located directly via the
    ``<|cot_start|>``/``<|cot_end|>`` special-token ids (``SPECIAL_TOKENS`` in
    ``models/utils.py``) already present in the tokenized ``input_ids`` -- the strictly-between
    span is blanked with the tokenizer's pad token; the start/end tags themselves are left
    intact (preserves sequence structure exactly, only the content between them changes).
  - Uses the model's own 6-camera "trajectory" task profile (``select_task_input(..,
    "trajectory")``) per this repo's trained input contract (see ``docs/pai_av_ood_eval.md``),
    not the 4-camera profile 1.5/R1 use. This means llr_act magnitude is not expected to be
    directly comparable to the 1.5 run's numbers in absolute terms -- report both, note the
    camera-count caveat, don't over-read small differences.
  - ``traj_data`` shapes from this repo's own ``load_physical_aiavdataset`` are already
    ``[1, 1, T, 3]`` / ``[1, 1, T, 3, 3]`` (batch, n_traj, ...) -- exactly what
    ``fuse_traj_tokens`` wants, no reshaping needed.

Usage::

    CUDA_VISIBLE_DEVICES=4 .venv/bin/python examples/phase0_llr_action_direction_a2s.py \\
      --limit 8 --out /tmp/phase0_llr_act_a2s_smoke.parquet

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) .venv/bin/python examples/phase0_llr_action_direction_a2s.py \\
        --num_shards 4 --shard_idx $i \\
        --out outputs/phase0_llr_act_a2s/llr.parquet &
    done; wait
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID  # noqa: E402

DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID))
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--first_event_only", action="store_true")
    p.add_argument("--task_profile", default="trajectory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event. Same shape as
    eval_common.load_events / the 1.5 repo's phase0 scripts."""
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        events = row["events"]
        idxs = range(len(events)) if all_events else [0]
        for ei in idxs:
            ev = events[ei]
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ei,
                    "t0_us": max(int(ev["event_start_timestamp"]), MIN_T0_US),
                    "gold_coc": ev["coc"],
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    import physical_ai_av

    from alpamayo2_super import helper
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.models.utils import SPECIAL_TOKENS

    print(f"[phase0-act-a2s] loading model {args.model_id} ...", flush=True)
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0").eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer, model.config)

    blank_token_id = tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = tokenizer.eos_token_id
    cot_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_start"])
    cot_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_end"])
    print(
        f"[phase0-act-a2s] blank_token_id={blank_token_id} cot_start_id={cot_start_id} "
        f"cot_end_id={cot_end_id}",
        flush=True,
    )

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[phase0-act-a2s] shard {args.shard_idx}/{args.num_shards}: {len(events)} events "
        f"(split={args.split}, task_profile={args.task_profile})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _build_tokenized(data: dict, cot_text: str) -> dict:
        data = dict(data)
        data["cot"] = cot_text
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=model.config.tokens_per_future_traj,
            components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
            components_prompt=["cot", "traj_future"],
            generation_mode=False,
            include_camera_ids=model.config.include_camera_ids,
            camera_ids=data["camera_indices"],
            include_frame_nums=model.config.frame_label == "frame_num",
        )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, add_vision_id=False,
        )
        images = data["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            processor(text=text, images=images, videos=None, padding=False, return_tensors="pt", do_rescale=False)
        )
        return tokenized_data

    def _run_pass(tokenized_data: dict, traj_data: dict) -> float:
        model_inputs = helper.to_device({"tokenized_data": tokenized_data, **traj_data}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                traj_data={
                    "ego_history_xyz": model_inputs["ego_history_xyz"],
                    "ego_history_rot": model_inputs["ego_history_rot"],
                    "ego_future_xyz": model_inputs["ego_future_xyz"],
                    "ego_future_rot": model_inputs["ego_future_rot"],
                },
                labels_mask=None,
            )
        return -out.loss_future_traj.item()

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            source = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            data = select_task_input(source, args.task_profile)
            traj_data = {
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
            }

            # --- Pass C: real gold CoC content ---
            td_real = _build_tokenized(data, ev["gold_coc"])
            input_ids_real = td_real["input_ids"]

            pos = (input_ids_real[0] == cot_start_id).nonzero(as_tuple=True)[0]
            pos_end = (input_ids_real[0] == cot_end_id).nonzero(as_tuple=True)[0]
            if len(pos) == 0 or len(pos_end) == 0:
                raise RuntimeError("cot_start/cot_end token not found in tokenized sequence")
            cot_lo, cot_hi = int(pos[0].item()) + 1, int(pos_end[0].item())  # strictly between tags
            n_cot_tokens = max(0, cot_hi - cot_lo)
            if n_cot_tokens == 0:
                raise RuntimeError("empty cot span (cot_start immediately followed by cot_end)")

            logp_num = _run_pass(td_real, traj_data)

            # --- Pass D: cot span content-blanked, same position/length, tags kept intact ---
            input_ids_blanked = input_ids_real.clone()
            input_ids_blanked[0, cot_lo:cot_hi] = blank_token_id
            td_blanked = dict(td_real)
            td_blanked["input_ids"] = input_ids_blanked
            logp_den = _run_pass(td_blanked, traj_data)

            llr_act = logp_num - logp_den
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "t0_us": t0_us,
                    "gold_coc": ev["gold_coc"],
                    "n_cot_tokens": n_cot_tokens,
                    "seq_len": int(input_ids_real.shape[1]),
                    "logp_num": logp_num,
                    "logp_den": logp_den,
                    "llr_act": llr_act,
                    "ok": True,
                }
            )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                recent = [r["llr_act"] for r in rows[-20:] if r.get("ok")]
                print(
                    f"[phase0-act-a2s] [{i + 1}/{len(events)}] {clip_id} split={ev['split']} "
                    f"llr_act_mean(last20)={np.mean(recent):.4f} nats ({elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )
        except Exception as e:
            rows.append(
                {"clip_id": clip_id, "event_idx": ev["event_idx"], "split": ev["split"], "ok": False, "error": str(e)}
            )
            print(f"[phase0-act-a2s] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out_path, index=False)

    ok = df[df.get("ok", False) == True]  # noqa: E712
    print(
        f"\n[phase0-act-a2s] shard {args.shard_idx}/{args.num_shards} done in "
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} events ok. Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        print(
            f"[phase0-act-a2s] shard summary: llr_act mean={ok['llr_act'].mean():.4f} "
            f"median={ok['llr_act'].median():.4f} logp_num mean={ok['logp_num'].mean():.4f} "
            f"logp_den mean={ok['logp_den'].mean():.4f} n_cot_tokens mean={ok['n_cot_tokens'].mean():.1f}",
            flush=True,
        )
        print(ok.groupby("event_cluster")["llr_act"].agg(["mean", "median", "count"]), flush=True)


if __name__ == "__main__":
    main()
