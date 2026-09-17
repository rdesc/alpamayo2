# SPDX-License-Identifier: Apache-2.0
"""Corrected Phase-0 LLR measurement for Alpamayo 2 Super -- per trajectory token, with a
genuine ``p(a*|v)`` denominator. Supersedes ``phase0_llr_action_direction_a2s.py`` (+0.037, on a
basis shown to be contaminated).

    llr_act = log p(a* | v, ell_gold) - log p(a* | v)

scored over the 128 FUTURE trajectory tokens only. ``--blank-mode splice`` (default) builds the
denominator by re-tokenizing with an empty CoC -- markers kept, prose gone, which is the
in-distribution empty-reasoning sequence the validated no-CoC eval path produces.

WHY THE OLD BASIS WAS WRONG HERE (measured, see ``phase0_llr_verify_setup_a2s.py``):

A2S avoids BOTH of the defects that invalidated the Alpamayo-1.5 measurement -- its history and
future trajectory tokens occupy DISJOINT id blocks ([151669,152669) vs [152669,155669)), so the
future mask catches 0 history tokens (1.5 wrongly scored 48 of them, pinned at exactly 0 by
causality and purely diluting); and the old script blanked only the span strictly between
``<|cot_start|>``/``<|cot_end|>``, so the markers survived (1.5's blanking destroyed them and
injected ~+23 nats on the delimiters).

But two contaminations remain, both specific to A2S and both in the DELIMITERS:

  - under interior pad-blanking they move by up to 3.95 nats (mean -0.79), even with the markers
    intact;
  - under splicing they move ~-1.1 nats, enough to drag the 130-token mask-mean to ~+0.000 while
    the 128 real trajectory tokens sit at +0.018.

Either way the 2 delimiters must be excluded. Measured side by side over the same 24 events
(``phase0_llr_verify_setup_a2s.py --limit 24``):

    old basis (mask-mean, interior pad-blank)   +0.0395   <- reproduces the published +0.037
    future-only, interior pad-blank             +0.0277
    future-only, cot_text="" splice             +0.0176   <- what this script reports
    future-only, wrong reasoning prose          +0.0110

So this script (a) scores future tokens only, never the delimiters, and (b) defaults to the
splice denominator so no pad token is introduced anywhere. Delimiter and (empty) history
components are still reported per event, so the decomposition stays auditable rather than folded
invisibly into the headline.

FULL-SPLIT RESULT: +0.0179 nats (median +0.0152, std 0.0528, 69.2% of events positive,
15.4 sigma, n=2,071). Unlike Alpamayo 1.5 -- which came out at +0.0010, 0.9 sigma, i.e. nothing --
A2S's reasoning IS measurably load-bearing. Two signatures support that beyond the sigma: the
acceleration channel (+0.0265) carries ~3x the curvature channel (+0.0093), matching the
predominantly longitudinal content of the CoC annotations, and the wrong-prose arm attributes
~63% of the effect (+0.0110 of +0.0176) to the reasoning being CORRECT for the scene rather than
merely present. A2S's own controls put reasoning at ~6% of what its cameras are worth
(wrong vision +0.2785, wrong trajectory +0.9173).

Caveat: A2S uses a 6-camera profile and a different training recipe than 1.5's 4-camera setup, so
this is each model measured against its own baseline, not a controlled comparison.

Token layout: ``tokens_per_future_traj = 128``, interleaved (accel, curvature) per waypoint at
dt = 0.1 s, so token k -> t = (k // 2 + 1) * 0.1 s, with even k = accel and odd k = curvature.
Verified against the config, not assumed.

Usage::

    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i .venv/bin/python examples/llr/phase0_llr_per_token_a2s.py \\
        --num_shards 8 --shard_idx $i --out outputs/llr_a2s_splice/pt.parquet &
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID  # noqa: E402

DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000
DT_S = 0.1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID))
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--task_profile", default="trajectory")
    p.add_argument("--blank-mode", dest="blank_mode", default="splice", choices=["splice", "interior"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path, split):
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        for ei, ev in enumerate(row["events"]):
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


def main():
    args = parse_args()
    torch.manual_seed(42)

    import physical_ai_av

    from alpamayo2_super import helper
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.models.utils import SPECIAL_TOKENS, fuse_traj_tokens

    print(f"[a2s-pt] loading {args.model_id} ...", flush=True)
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0").eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer, model.config)
    cfg = model.config
    tids = cfg.traj_ids
    fut_lo, fut_hi = tids["future_id0"], tids["future_id0"] + cfg.future_vocab_size
    cot_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_start"])
    cot_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_end"])
    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    events = load_events(args.parquet, args.split)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[a2s-pt] shard {args.shard_idx}/{args.num_shards}: {len(events)} events "
        f"(split={args.split}, mode={args.blank_mode}, profile={args.task_profile})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def build_tokenized(data, cot_text):
        data = dict(data)
        data["cot"] = cot_text
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=cfg.tokens_per_history_traj,
            num_tokens_per_future_traj=cfg.tokens_per_future_traj,
            components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
            components_prompt=["cot", "traj_future"],
            generation_mode=False,
            include_camera_ids=cfg.include_camera_ids,
            camera_ids=data["camera_indices"],
            include_frame_nums=cfg.frame_label == "frame_num",
        )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, add_vision_id=False
        )
        images = data["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        return dict(
            processor(
                text=text, images=images, videos=None, padding=False,
                return_tensors="pt", do_rescale=False,
            )
        )

    def score(td, traj_data):
        mi = helper.to_device({"tokenized_data": dict(td), **traj_data}, "cuda")
        td_dev = mi["tokenized_data"]
        tdata = {k: mi[k] for k in
                 ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}
        fused = fuse_traj_tokens(
            model.history_traj_tokenizer, model.future_traj_tokenizer,
            td_dev["input_ids"].clone(), tdata, tids,
        )
        traj_mask = (
            ((fused >= fut_lo) & (fused < fut_hi))
            | (fused == tids["future_start"])
            | (fused == tids["future_end"])
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(tokenized_data=td_dev, traj_data=tdata, labels_mask=None)
        m = traj_mask[:, 1:]
        sl = fused[:, 1:][m].contiguous()
        lg = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(lg, sl, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        fsp = int((fused[0] == tids["future_start"]).nonzero().flatten()[-1].item())
        is_val = (sl >= fut_lo) & (sl < fut_hi)
        is_fut = is_val & (seq_pos > fsp)
        return {
            "fut": logp[is_fut].cpu().numpy(),
            "delim": float(logp[~is_val].mean().item()) if bool((~is_val).any()) else float("nan"),
            "maskmean": float(logp.mean().item()),
            "scalar": -out.loss_future_traj.item(),
        }

    rows = []
    t0 = time.time()
    for i, ev in enumerate(events):
        try:
            src = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
            data = select_task_input(src, args.task_profile)
            traj_data = {k: data[k] for k in
                         ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}

            td_gold = build_tokenized(data, ev["gold_coc"])
            r_num = score(td_gold, traj_data)
            recon = abs(r_num["maskmean"] - r_num["scalar"])
            if recon > 1e-3:
                raise RuntimeError(f"reconciliation failed: {recon:.2e}")

            ids = td_gold["input_ids"]
            p_s = (ids[0] == cot_start_id).nonzero(as_tuple=True)[0]
            p_e = (ids[0] == cot_end_id).nonzero(as_tuple=True)[0]
            if p_s.numel() == 0 or p_e.numel() == 0:
                raise RuntimeError("cot markers not found")
            cot_lo, cot_hi = int(p_s[0]) + 1, int(p_e[0])
            n_cot = cot_hi - cot_lo

            if args.blank_mode == "splice":
                r_den = score(build_tokenized(data, ""), traj_data)
            else:
                ids_b = ids.clone()
                ids_b[0, cot_lo:cot_hi] = blank_id
                td_b = dict(td_gold)
                td_b["input_ids"] = ids_b
                r_den = score(td_b, traj_data)

            if len(r_num["fut"]) != len(r_den["fut"]):
                raise RuntimeError(
                    f"future-token count differs ({len(r_num['fut'])} vs {len(r_den['fut'])})"
                )

            lp_n, lp_d = r_num["fut"], r_den["fut"]
            for k in range(len(lp_n)):
                rows.append(
                    {
                        "clip_id": ev["clip_id"], "event_idx": ev["event_idx"],
                        "event_cluster": ev["event_cluster"], "split": ev["split"],
                        "gold_coc": ev["gold_coc"], "n_cot_tokens": n_cot,
                        "blank_mode": args.blank_mode,
                        "traj_idx": k,
                        "channel": "accel" if k % 2 == 0 else "curvature",
                        "t_s": (k // 2 + 1) * DT_S,
                        "logp_gold": float(lp_n[k]),
                        "logp_den": float(lp_d[k]),
                        "llr_act": float(lp_n[k] - lp_d[k]),
                        # kept visible, never folded into llr_act
                        "delim_gold": r_num["delim"], "delim_den": r_den["delim"],
                        # mask-mean under THIS run's denominator -- NOT the originally published basis
                        # (that used a pad-blank denominator). Kept only to show why the
                        # delimiters must be excluded: they move ~-1.1 nats under splicing,
                        # which drags the 130-token mean to ~0 while the 128 real tokens sit
                        # at +0.018.
                        "llr_maskmean_same_denom": r_num["maskmean"] - r_den["maskmean"],
                    }
                )
            if (i + 1) % 10 == 0 or (i + 1) == len(events):
                print(
                    f"[a2s-pt] [{i+1}/{len(events)}] {ev['clip_id'][:8]} "
                    f"llr={float(np.mean(lp_n - lp_d)):+.4f} ({(time.time()-t0)/60:.1f} min)",
                    flush=True,
                )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[a2s-pt] [{i+1}/{len(events)}] {ev['clip_id'][:8]} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out, index=False)
    n_ev = len(df[["clip_id", "event_idx"]].drop_duplicates()) if len(df) else 0
    print(f"\n[a2s-pt] done in {(time.time()-t0)/60:.1f} min. {n_ev} events, {len(df)} rows -> {out}", flush=True)
    if len(df):
        print(f"[a2s-pt] llr_act (future tokens only) mean={df.llr_act.mean():+.4f}", flush=True)
        print(df.groupby("channel")["llr_act"].agg(["mean", "std", "count"]).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
