# SPDX-License-Identifier: Apache-2.0
"""Verify the A2S LLR measurement setup BEFORE trusting (or re-running) any number.

The Alpamayo-1.5 LLR measurement reported +0.227 nats and it was an artifact -- see
``~/repos/alpamayo-recipes/scripts_fork/llr/results/phase0_llr_action_direction.md``. Two
structural facts about that model, neither visible from the scoring API, caused it:

  (a) history- and future-trajectory tokens were drawn from ONE token-id block, so the mask
      behind ``loss_future_traj`` silently also scored 48 history tokens. Those precede the
      reasoning span, so causal attention pins their LLR to exactly 0 -- they diluted the mean.
  (b) the "cot span" being blanked was INCLUSIVE of the ``<|cot_start|>``/``<|cot_end|>`` markers,
      so the ablation deleted the end-of-reasoning marker and the 2 traj delimiters went from
      log p of exactly 0.0 to -18/-28 nats -- a near-constant +0.27 offset on the mean.

A2S looks different on BOTH counts: it has separate ``traj_ids["history_id0"]`` and
``traj_ids["future_id0"]``, and ``phase0_llr_action_direction_a2s.py`` blanks only
``cot_lo:cot_hi`` (strictly between the tags). If that holds, its +0.037 is roughly valid rather
than an artifact -- which would be a real difference between the two models, so it needs to be
established by measurement, not inferred from reading the source.

This script establishes it. It asserts nothing and concludes nothing on its own; it prints the
structural facts and the control ladder so the setup can be signed off (or not) before a full run.

WHAT IT CHECKS

Structure (printed, no model rollout needed beyond a load):
  1. traj_ids / vocab sizes, and whether the history and future id ranges OVERLAP. This is
     defect (a) -- the single most important number on the page.
  2. The exact role composition of the span ``loss_future_traj`` averages over: how many history
     tokens, delimiters and future tokens. For 1.5 this was 48 / 2 / 128 = 178.
  3. That the cot span's first/last tokens really are the markers, and that the existing script's
     ``cot_lo:cot_hi`` slice excludes them -- defect (b).

Numerical invariants (must hold, else the scoring is misaligned):
  4. per-token mean over the full mask == ``-loss_future_traj`` (reconciliation).
  5. null_identical / null_retokenize == exactly 0.
  6. history-token LLR == exactly 0 (structural: they precede the cot span).
  7. delimiter LLR under interior blanking ~= 0 (on 1.5 this was +4e-6 with markers intact, and
     +23.63 when they were destroyed -- this is the direct test for defect (b)).

Sensitivity (the null is uninterpretable without these):
  8. wrong_vision / wrong_traj must be LARGE, through the identical scoring path.

Denominators compared side by side:
  9. interior pad-blank (what the retracted-basis script did, markers intact) vs. cot_text=""
     splice (in-distribution). On 1.5 these differed by ~0.012 nats and flipped sign, which is
     the same order as the effect -- so this comparison decides whether a re-run is required.

Usage::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/llr/phase0_llr_verify_setup_a2s.py --limit 6
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID))
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--task_profile", default="trajectory")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_events(parquet_path):
    df = pd.read_parquet(parquet_path)
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
    torch.manual_seed(args.seed)

    import physical_ai_av  # noqa: F841

    from alpamayo2_super import helper
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.models.utils import SPECIAL_TOKENS, fuse_traj_tokens

    print(f"[verify] loading {args.model_id} ...", flush=True)
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0").eval()
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer, model.config)
    cfg = model.config

    # ---------------- 1. structural: id blocks ----------------
    tids = dict(cfg.traj_ids)
    fut_lo = tids["future_id0"]
    fut_hi = fut_lo + cfg.future_vocab_size
    hist_lo = tids["history_id0"]
    hist_vocab = getattr(cfg, "history_vocab_size", None)
    hist_hi = hist_lo + hist_vocab if hist_vocab is not None else None

    print("\n" + "=" * 78)
    print("1. TOKEN ID BLOCKS  -- does the future-token mask also catch history tokens?")
    print("=" * 78)
    print(f"  traj_ids                 {tids}")
    print(f"  future block             [{fut_lo}, {fut_hi})   future_vocab_size={cfg.future_vocab_size}")
    print(f"  history block            [{hist_lo}, {hist_hi})   history_vocab_size={hist_vocab}")
    print(f"  tokens_per_future_traj   {cfg.tokens_per_future_traj}")
    print(f"  tokens_per_history_traj  {cfg.tokens_per_history_traj}")
    if hist_hi is None:
        overlap = None
        print("  OVERLAP                  UNKNOWN (no history_vocab_size on config)")
    else:
        overlap = not (hist_hi <= fut_lo or fut_hi <= hist_lo)
        print(f"  OVERLAP                  {overlap}")
        if overlap:
            print("    -> SAME defect as Alpamayo 1.5: history tokens WILL be scored and dilute the mean.")
        else:
            print("    -> history and future ids are DISJOINT: the future mask cannot catch history")
            print("       tokens, so Alpamayo 1.5's dilution defect does NOT apply to A2S.")

    # ---------------- 2/3. structural: spans on a real event ----------------
    events = load_events(args.parquet)
    donor_tok = [
        np.asarray(tokenizer(e["gold_coc"], add_special_tokens=False)["input_ids"], dtype=np.int64)
        for e in events
    ]
    use = events[: args.limit]
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    cot_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_start"])
    cot_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_end"])
    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def build_tokenized(data, cot_text, vision_from=None):
        data = dict(data)
        data["cot"] = cot_text
        if vision_from is not None:
            data["image_frames"] = vision_from["image_frames"]
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
        """Per-token log p over the model's own traj_mask, split by role."""
        mi = helper.to_device({"tokenized_data": dict(td), **traj_data}, "cuda")
        td_dev = mi["tokenized_data"]
        input_ids = td_dev["input_ids"]
        tdata = {
            "ego_history_xyz": mi["ego_history_xyz"],
            "ego_history_rot": mi["ego_history_rot"],
            "ego_future_xyz": mi["ego_future_xyz"],
            "ego_future_rot": mi["ego_future_rot"],
        }
        # fuse_traj_tokens is a module-level helper here, not a method (unlike Alpamayo 1.5).
        fused = fuse_traj_tokens(
            model.history_traj_tokenizer,
            model.future_traj_tokenizer,
            input_ids.clone(),
            tdata,
            cfg.traj_ids,
        )
        # Replicate alpamayo2_super.py:228-235 exactly.
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
        is_hist = is_val & ~is_fut
        is_delim = ~is_val
        f = lambda sel: float(logp[sel].mean().item()) if bool(sel.any()) else float("nan")  # noqa: E731
        return {
            "future": f(is_fut), "hist": f(is_hist), "delim": f(is_delim),
            "maskmean": float(logp.mean().item()),
            "n_future": int(is_fut.sum().item()),
            "n_hist": int(is_hist.sum().item()),
            "n_delim": int(is_delim.sum().item()),
            "n_mask": int(logp.numel()),
            "scalar": -out.loss_future_traj.item(),
            "per_token_future": logp[is_fut].cpu().numpy(),
        }

    rows = []
    t0 = time.time()
    printed_struct = False
    for i, ev in enumerate(use):
        try:
            src = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
            data = select_task_input(src, args.task_profile)
            traj_data = {k: data[k] for k in
                         ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}
            gold = ev["gold_coc"]

            td_gold = build_tokenized(data, gold)
            ref = score(td_gold, traj_data)

            ids = td_gold["input_ids"]
            p_s = (ids[0] == cot_start_id).nonzero(as_tuple=True)[0]
            p_e = (ids[0] == cot_end_id).nonzero(as_tuple=True)[0]
            cot_lo, cot_hi = int(p_s[0]) + 1, int(p_e[0])

            if not printed_struct:
                print("\n" + "=" * 78)
                print("2. WHAT loss_future_traj ACTUALLY AVERAGES OVER")
                print("=" * 78)
                print(f"  total tokens in mask     {ref['n_mask']}")
                print(f"    history-traj tokens    {ref['n_hist']}   (1.5 had 48 -- pure dilution)")
                print(f"    delimiters             {ref['n_delim']}")
                print(f"    future-traj tokens     {ref['n_future']}   (the ones we mean to score)")
                dil = ref["n_mask"] / max(ref["n_future"], 1)
                print(f"  dilution factor          {dil:.4f}  (1.5 was 178/128 = 1.3906)")
                print(f"  => a published mask-mean of +0.037 corresponds to a future-only value of")
                print(f"     roughly {0.037 * dil:+.4f} nats")

                print("\n" + "=" * 78)
                print("3. IS THE ABLATED SPAN INCLUSIVE OF THE MARKERS?")
                print("=" * 78)
                print(f"  cot_start at index {int(p_s[0])}, cot_end at index {int(p_e[0])}")
                print(f"  script blanks [{cot_lo}:{cot_hi}] -> {cot_hi - cot_lo} interior tokens")
                tok_lo = tokenizer.convert_ids_to_tokens(int(ids[0, cot_lo]))
                tok_before = tokenizer.convert_ids_to_tokens(int(ids[0, cot_lo - 1]))
                tok_after = tokenizer.convert_ids_to_tokens(int(ids[0, cot_hi]))
                print(f"  token before span        {tok_before!r}  <- must be the cot_start marker")
                print(f"  first blanked token      {tok_lo!r}")
                print(f"  token after span         {tok_after!r}   <- must be the cot_end marker")
                ok_markers = (int(ids[0, cot_lo - 1]) == cot_start_id) and (int(ids[0, cot_hi]) == cot_end_id)
                print(f"  MARKERS PRESERVED        {ok_markers}")
                print("    -> True means A2S never had Alpamayo 1.5's marker-deletion defect.")
                printed_struct = True

            # --- denominators ---
            ids_b = ids.clone()
            ids_b[0, cot_lo:cot_hi] = blank_id
            td_blank = dict(td_gold)
            td_blank["input_ids"] = ids_b
            r_blank = score(td_blank, traj_data)

            td_splice = build_tokenized(data, "")
            r_splice = score(td_splice, traj_data)

            # --- nulls ---
            r_id = score(build_tokenized(data, gold), traj_data)

            # --- positive controls ---
            rng = np.random.default_rng(args.seed + i)
            j = int(rng.choice([k for k in range(len(events)) if events[k]["clip_id"] != ev["clip_id"]]))
            oth_src = load_physical_aiavdataset(events[j]["clip_id"], t0_us=events[j]["t0_us"], avdi=avdi)
            oth = select_task_input(oth_src, args.task_profile)
            r_wv = score(build_tokenized(data, gold, vision_from=oth), traj_data)
            traj_swap = dict(traj_data)
            traj_swap["ego_future_xyz"] = oth["ego_future_xyz"]
            traj_swap["ego_future_rot"] = oth["ego_future_rot"]
            r_wt = score(td_gold, traj_swap)

            # --- wrong reasoning prose, same interior length ---
            n_int = cot_hi - cot_lo
            dt = donor_tok[j]
            if len(dt) >= n_int:
                ids_w = ids.clone()
                ids_w[0, cot_lo:cot_hi] = torch.from_numpy(dt[:n_int]).to(ids_w.dtype)
                td_w = dict(td_gold)
                td_w["input_ids"] = ids_w
                wrong_coc = ref["future"] - score(td_w, traj_data)["future"]
            else:
                wrong_coc = float("nan")

            rows.append(
                {
                    "clip_id": ev["clip_id"], "event_idx": ev["event_idx"],
                    "recon_err": abs(ref["maskmean"] - ref["scalar"]),
                    "null_identical": r_id["future"] - ref["future"],
                    "history_llr": (r_splice["hist"] - ref["hist"]) if ref["n_hist"] else 0.0,
                    "delim_llr_blank": r_blank["delim"] - ref["delim"],
                    "llr_blank_future": ref["future"] - r_blank["future"],
                    "llr_splice_future": ref["future"] - r_splice["future"],
                    "llr_blank_maskmean_OLD": ref["maskmean"] - r_blank["maskmean"],
                    "wrong_coc_llr": wrong_coc,
                    "wrong_vision_llr": ref["future"] - r_wv["future"],
                    "wrong_traj_llr": ref["future"] - r_wt["future"],
                    "ref_future": ref["future"],
                }
            )
            print(
                f"[verify] [{i+1}/{len(use)}] {ev['clip_id'][:8]} "
                f"blank={rows[-1]['llr_blank_future']:+.4f} splice={rows[-1]['llr_splice_future']:+.4f} "
                f"OLD_maskmean={rows[-1]['llr_blank_maskmean_OLD']:+.4f} "
                f"wv={rows[-1]['wrong_vision_llr']:+.4f} wt={rows[-1]['wrong_traj_llr']:+.4f} "
                f"({(time.time()-t0)/60:.1f} min)",
                flush=True,
            )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[verify] [{i+1}/{len(use)}] {ev['clip_id'][:8]} FAILED: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("\n[verify] no events succeeded")
        return
    df = pd.DataFrame(rows)
    if args.out:
        df.to_parquet(args.out, index=False)

    print("\n" + "=" * 78)
    print(f"4-8. INVARIANTS AND SENSITIVITY  (n={len(df)} events)")
    print("=" * 78)
    print(f"{'check':24s} {'mean':>11s} {'max|dev|':>11s}  expected     verdict")
    for col, exp in [
        ("recon_err", "== 0"),
        ("null_identical", "== 0"),
        ("history_llr", "== 0"),
        ("delim_llr_blank", "~= 0"),
    ]:
        v = df[col]
        tol = 1e-6 if exp == "== 0" else 1e-3
        ok = "PASS" if v.abs().max() < tol else f"FAIL (max {v.abs().max():.2e})"
        print(f"{col:24s} {v.mean():+11.6f} {v.abs().max():11.2e}  {exp:11s}  {ok}")
    for col, exp in [("wrong_vision_llr", "> 0"), ("wrong_traj_llr", ">> 0")]:
        v = df[col]
        ok = "PASS" if v.mean() > 0.05 else f"SUSPECT ({v.mean():+.4f})"
        print(f"{col:24s} {v.mean():+11.4f} {'':>11s}  {exp:11s}  {ok}")

    print("\n" + "=" * 78)
    print("9. DENOMINATOR COMPARISON  -- does A2S's number need re-running?")
    print("=" * 78)
    for col, lbl in [
        ("llr_blank_maskmean_OLD", "old basis (mask-mean, interior blank)"),
        ("llr_blank_future", "future-only, interior pad-blank"),
        ("llr_splice_future", "future-only, cot_text='' splice  <- in-distribution"),
        ("wrong_coc_llr", "future-only, wrong reasoning prose"),
    ]:
        v = df[col]
        print(f"  {lbl:42s} {v.mean():+.4f}  (median {v.median():+.4f}, std {v.std():.4f})")
    d = abs(df["llr_splice_future"].mean() - df["llr_blank_future"].mean())
    print(f"\n  blank vs splice differ by {d:.4f} nats.")
    print(f"  reference log p(a*) per future token = {df['ref_future'].mean():.4f}")


if __name__ == "__main__":
    main()
