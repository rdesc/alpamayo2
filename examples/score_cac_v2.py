# SPDX-License-Identifier: Apache-2.0
"""Recompute CoC-action consistency on existing PAI-AV OOD eval parquets using the
updated scorer (E3(P3 regex fallback) x M5-abstain, segmented trajectory classifier)
ported into ``alpamayo-recipes/recipes/alpamayo1_x_rl/rewards/coc_action_consistency_*.py``
from ``alpamayo-coc-autolabeler``.

Scores the model's OWN medoid ("most-likely") predicted trajectory against its OWN
medoid CoC (``pred_coc_ml``) -- the same pairing Sec 5.3.2 grades, and the same one the
RL reward's ``compute_component`` consumes.

Caveat (see printed banner): the eval parquets only stored ``pred_xy`` (K,T,2), not
``pred_rot``, so ``coc_action_consistency_trajectory.py``'s heading input is
reconstructed from the predicted path's own tangent direction (central-difference
atan2(vy, vx)) rather than a true predicted orientation channel (never sampled -- the
eval script discarded it, see ``eval_pai_av_val_a2.py``). This is exact for turn-rate
based lateral classification (heading-change-rate does not care where the heading
number came from) but makes the longitudinal `reverse` bucket structurally unreachable
(speed-along-heading is tautologically the speed magnitude when heading is derived from
velocity direction). Reverse is a rare bucket in this corpus (~7/2077 events in the
validated gold set), so the effect on the aggregate is expected to be small, but it is
a real, uncorrected deviation from the validated design -- not a rounding issue.

Usage
-----
    python examples/score_cac_v2.py outputs/pai_av_val/a2_coc.parquet \\
        outputs/pai_av_val/a2_nococ.parquet \\
        outputs/pai_av_train/train_coc.parquet \\
        outputs/pai_av_train/train_nococ.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

RECIPES_DIR = os.path.expanduser("~/repos/alpamayo-recipes/recipes")
sys.path.insert(0, RECIPES_DIR)

from alpamayo1_x_rl.rewards.coc_action_consistency_extract import QwenClaimExtractor  # noqa: E402
from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component  # noqa: E402

DEFAULT_QWEN_EXTRACTOR = (
    "/mnt/efs/users/rod/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def _rot_from_path_tangent(xy: np.ndarray) -> torch.Tensor:
    """[T,2] -> [T,3,3] rotation matrices whose heading is the path's own tangent angle
    (central-difference atan2(vy, vx)). See module docstring for why this is an
    approximation of true predicted orientation, which was never captured."""
    T = xy.shape[0]
    vx, vy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    h = np.arctan2(vy, vx)
    cos_h, sin_h = np.cos(h), np.sin(h)
    rot = torch.zeros(T, 3, 3, dtype=torch.float32)
    rot[:, 0, 0], rot[:, 0, 1] = torch.from_numpy(cos_h), torch.from_numpy(-sin_h)
    rot[:, 1, 0], rot[:, 1, 1] = torch.from_numpy(sin_h), torch.from_numpy(cos_h)
    rot[:, 2, 2] = 1.0
    return rot


def score_parquet(path: str, extractor: QwenClaimExtractor | None = None,
                   batch_size: int = 16) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index(drop=True)
    ok_mask = df["ok"] if "ok" in df else pd.Series(True, index=df.index)

    claims_by_row: dict[int, list] = {}
    if extractor is not None:
        # Batch the GPU extraction pass separately from the (cheap, CPU) trajectory
        # classification below -- extract_batch() is 2.2-2.7x faster than one-at-a-time
        # at batch>=8 (measured in coc_action_consistency_extract.py's docstring).
        idxs = [i for i in df.index if ok_mask[i]]
        texts = [(df.at[i, "pred_coc_ml"] or "") for i in idxs]
        for start in range(0, len(idxs), batch_size):
            chunk_idxs = idxs[start:start + batch_size]
            chunk_texts = texts[start:start + batch_size]
            chunk_claims = extractor.extract_batch(chunk_texts)
            for i, c in zip(chunk_idxs, chunk_claims):
                claims_by_row[i] = c
            print(f"  extracted {min(start + batch_size, len(idxs))}/{len(idxs)}", flush=True)

    out_rows = []
    for i, row in df.iterrows():
        if not ok_mask[i]:
            out_rows.append({"cac2_binary": np.nan, "cac2_graded": np.nan,
                              "cac2_reward": np.nan, "cac2_abstained": np.nan,
                              "cac2_parsed": np.nan, "cac2_n_scored": np.nan})
            continue
        n_traj, n_future = int(row["n_traj"]), int(row["n_future"])
        pred_xy = np.asarray(row["pred_xy"], dtype=np.float32).reshape(n_traj, n_future, 2)
        medoid = int(row["medoid_idx"])
        xy = pred_xy[medoid]  # (T, 2)
        xyz = torch.zeros(n_future, 3, dtype=torch.float32)
        xyz[:, :2] = torch.from_numpy(xy)
        rot = _rot_from_path_tangent(xy)
        claims = claims_by_row.get(i) if extractor is not None else None
        comp = compute_component(row.get("pred_coc_ml"), xyz, rot, claims=claims)
        out_rows.append({
            "cac2_binary": comp["binary"],
            "cac2_graded": comp["graded"],
            "cac2_reward": comp["reward_contribution"],
            "cac2_abstained": comp["abstained"],
            "cac2_parsed": comp["parsed"],
            "cac2_n_scored": comp["n_scored"],
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)


def summarize(df: pd.DataFrame, label: str) -> None:
    ok = df[df["ok"] & df["cac2_parsed"].notna()] if "ok" in df else df
    parsed = ok[ok["cac2_parsed"] == True]  # noqa: E712
    scored = parsed[parsed["cac2_abstained"] == False]  # noqa: E712
    n_total = len(ok)
    n_unparsed = int((parsed["cac2_parsed"] == False).sum()) if len(parsed) else 0  # placeholder, see below
    n_unparsed = int((ok["cac2_parsed"] == False).sum())  # noqa: E712
    n_abstained = int((ok["cac2_abstained"] == True).sum())  # noqa: E712
    print(f"\n{label}: n={n_total}  unparsed={n_unparsed}  abstained={n_abstained}  "
          f"scored={len(scored)}")
    if len(scored):
        print(f"  mean binary (scored only)  = {scored['cac2_binary'].mean():.3f}")
        print(f"  mean graded (scored only)  = {scored['cac2_graded'].mean():.3f}")
    if n_total:
        # Corpus-accuracy convention: unparsed counts as 0 in the denominator that
        # includes them; abstained events are excluded entirely (see cac_scorer_design_results.md).
        denom = ok[ok["cac2_abstained"] != True]  # noqa: E712
        if len(denom):
            print(f"  mean binary (unparsed=0, abstained excluded) = {denom['cac2_binary'].fillna(0).mean():.3f}  n={len(denom)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--suffix", default=None,
                     help="Default: .cac_v2.parquet (regex) or .cac_v2_qwen.parquet (--qwen).")
    ap.add_argument("--qwen", action="store_true",
                     help="Use the in-loop QwenClaimExtractor (P3 prompt) instead of the regex "
                          "fallback for claim extraction.")
    ap.add_argument("--qwen-extractor-path", default=DEFAULT_QWEN_EXTRACTOR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    print(__doc__)
    extractor = None
    if args.qwen:
        print(f"Loading QwenClaimExtractor from {args.qwen_extractor_path} on {args.device} ...")
        extractor = QwenClaimExtractor(args.qwen_extractor_path, device=args.device)
    suffix = args.suffix or (".cac_v2_qwen.parquet" if args.qwen else ".cac_v2.parquet")

    for path in args.parquets:
        scored = score_parquet(path, extractor=extractor, batch_size=args.batch_size)
        out_path = path.rsplit(".parquet", 1)[0] + suffix
        scored.to_parquet(out_path, index=False)
        print(f"\nWrote {out_path}")
        summarize(scored, os.path.basename(path))


if __name__ == "__main__":
    main()
