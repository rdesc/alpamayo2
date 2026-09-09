# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic helpers shared by the Alpamayo 1.5, R1 and 2 Super PAI-AV evals.

Verbatim copy of ``eval_common.py`` from /mnt/efs/users/rod/repos/alpamayo1.5 (only this
docstring differs), so Alpamayo 2 results land in exactly the same parquet schema as the
1.5/R1 runs and stay parseable by that repo's ``parse_results.py`` / ``compute_metrics.py``
/ ``score_reasoning_multigpu.py``. Keep the two copies in sync.

This module deliberately has NO torch / model imports so it can be imported under
either model's venv (it only needs numpy + pandas). It holds the dataset event
loader, the metric definitions, argument parsing, sharding, the per-clip run loop,
and result I/O so eval_pai_av_val.py (1.5) and eval_pai_av_val_r1.py (R1) stay thin.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# t0 must exceed the history window (16 steps * 0.1s = 1.6s); leave a small margin.
MIN_T0_US = 1_700_000
TIME_STEP = 0.1  # seconds between trajectory waypoints (10 Hz)
DEFAULT_PARQUET = "/home/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
ADE_HORIZONS = (1, 2, 3, 4, 5, 6)  # seconds
METRIC_KEYS = tuple(f"min_ade_{h}s" for h in ADE_HORIZONS) + tuple(f"ade_ml_{h}s" for h in ADE_HORIZONS)


# --------------------------------------------------------------------------- data
def load_events(parquet_path: str, all_events: bool, split: str = "val") -> list[dict]:
    """Return a flat list of {clip_id, event_idx, t0_us, coc, event_cluster} rows.

    split: "val", "train", or "both" (no split filter).
    """
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    else:
        df = df.copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        events = row["events"]
        idxs = range(len(events)) if all_events else [0]
        for ei in idxs:
            ev = events[ei]
            rows.append({
                "clip_id": clip_id,
                "event_idx": ei,
                "t0_us": max(int(ev["event_start_timestamp"]), MIN_T0_US),
                "coc": ev["coc"],
                "event_cluster": row["event_cluster"],
            })
    return rows


# ------------------------------------------------------------------------ metrics
def clip_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray) -> tuple[dict, int]:
    """Per-clip metrics from raw trajectories.

    Args:
        pred_xy: (K, T, 2) the K sampled trajectories.
        gt_xy:   (T, 2) the ground-truth future.

    For each horizon h in ADE_HORIZONS returns:
      - ``min_ade_{h}s``: best-of-K cumulative ADE over [0, h]  (multimodal metric)
      - ``ade_ml_{h}s`` : cumulative ADE over [0, h] of the *medoid* trajectory
        (sample closest to the other K-1 = the "most likely"/consensus mode).
    Also returns the medoid index.
    """
    err = np.linalg.norm(pred_xy - gt_xy[None], axis=2)  # (K, T) meters
    t_len = err.shape[1]
    pair = np.linalg.norm(pred_xy[:, None] - pred_xy[None, :], axis=3).sum(2)  # (K, K)
    medoid = int(pair.sum(1).argmin())
    out = {}
    for h in ADE_HORIZONS:
        idx = min(max(int(round(h / TIME_STEP)), 1), t_len)
        out[f"min_ade_{h}s"] = float(err[:, :idx].mean(1).min())   # best-of-K
        out[f"ade_ml_{h}s"] = float(err[medoid, :idx].mean())      # medoid / most-likely
    return out, medoid


def load_scene_with_retry(load_physical_aiavdataset, select_task_input, clip_id, t0_us, max_attempts=8, base_delay=20.0):
    """Scene loading hits the HF Hub API (list_repo_refs) on every call, and camera
    chunk zips are read via HfFileSystem byte-range streaming with no local caching --
    both are shared org-wide, so a large unrelated job on the same cluster can
    transiently exhaust the shared quota (2500 API req/5min, 12000 resolver req/5min)
    and return 429. That 429 doesn't always surface as an HTTP error at the call site
    we can see: a 429 during a zip byte-range read shows up here as
    ``zipfile.BadZipFile: File is not a zip file`` (the underlying HfHubHTTPError is
    only visible in ``__cause__``), so rather than pattern-match error strings, retry
    ANY exception from this call with backoff -- a genuinely permanent bug still
    surfaces (with full traceback) once max_attempts is exhausted."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            vqa_data = select_task_input(source_data, "vqa")
            return source_data, vqa_data
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt
            print(f"  [retry {attempt}/{max_attempts}] scene load failed (possibly transient HF rate limit), retrying in {delay:.0f}s: {e!r}")
            time.sleep(delay)
    raise last_err  # pragma: no cover


def coc_to_str(coc) -> str:
    """Serialize a CoC (possibly a list/array of per-trajectory strings) to one string."""
    if isinstance(coc, np.ndarray):
        coc = coc.tolist()
    if isinstance(coc, str):
        return coc
    try:
        return json.dumps(coc, default=str)
    except (TypeError, ValueError):
        return str(coc)


def build_wrong_coc_map(events: list[dict]) -> dict:
    """(clip_id, event_idx) -> another event's gold CoC text -- the "wrong scene" ablation.

    Walks forward through the sorted event list to the first event whose CoC text actually
    differs from this one's (a plain rotation can silently pair an event with a duplicate of
    its own text when the same sentence recurs across events). Pass the FULL (unsharded,
    un-limited) event list for the split so the pairing is deterministic and identical across
    every shard process.

    Ported from the alpamayo1.5 repo's eval_common.py (not otherwise kept in lockstep with
    that copy -- see this repo's own --retry_failed, which diverges deliberately).
    """
    keyed = {(str(e["clip_id"]), int(e["event_idx"])): coc_to_str(e["coc"]) for e in events}
    keys = sorted(keyed)
    out = {}
    for i, k in enumerate(keys):
        for step in range(1, len(keys)):
            cand_key = keys[(i + step) % len(keys)]
            if keyed[cand_key] != keyed[k]:
                out[k] = keyed[cand_key]
                break
        else:
            out[k] = keyed[k]  # degenerate: every event shares identical text
    return out


def build_pred_record(pred_xyz, ego_future_xyz, extra: dict | None) -> dict:
    """Turn raw model outputs into a metrics + stored-trajectory record.

    Works for both models: pred_xyz is (B, n_sets, K, T, >=2); ego_future_xyz is
    (B, n_sets, T, >=2); extra (optional) carries the per-trajectory "cot" text.
    """
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2].astype(np.float32)   # (K, T, 2)
    gt_xy = ego_future_xyz.detach().cpu().numpy()[0, 0, :, :2].astype(np.float32)  # (T, 2)
    metrics, medoid = clip_metrics(pred_xy, gt_xy)
    # Per-trajectory CoC traces (aligned with the K trajectory samples). The medoid
    # is our "most-likely" rollout, so pred_coc_ml is the trace to grade for reasoning
    # quality (mirrors the paper's "most-likely of 6" reasoning grading).
    pred_cocs = []
    if extra and "cot" in extra:
        pred_cocs = [coc_to_str(c) for c in np.asarray(extra["cot"]).reshape(-1).tolist()]
    pred_coc_ml = pred_cocs[medoid] if medoid < len(pred_cocs) else (pred_cocs[0] if pred_cocs else "")
    rec = {
        **metrics,
        "medoid_idx": int(medoid),
        "pred_coc_ml": pred_coc_ml,             # CoC of the medoid ("most-likely") trajectory
        "pred_cocs": json.dumps(pred_cocs),     # all K CoC traces (JSON list)
        # Raw trajectories (flattened) so any other horizon/metric is computable
        # offline from the parquet without re-running inference (compute_metrics.py).
        "n_traj": int(pred_xy.shape[0]),
        "n_future": int(pred_xy.shape[1]),
        "pred_xy": pred_xy.reshape(-1).tolist(),  # K*T*2, row-major [k, t, xy]
        "gt_xy": gt_xy.reshape(-1).tolist(),      # T*2
    }
    # Native action-space controls (acceleration, curvature) per waypoint, when the sampler
    # exposed them (the model adds extra["action"], shape [B, ns, nj, T, 2]). This is the
    # model's *own* trajectory representation (UnicycleAccelCurvatureActionSpace) -- exact,
    # not re-derived from xy -- so a meta-action / consistency check can read the longitudinal
    # control (a) and lateral control (kappa) directly. Stored with the same [k, t, c] layout
    # as pred_xy; the medoid sequence is also broken out for convenience.
    if extra is not None and extra.get("action") is not None:
        act = np.asarray(extra["action"])[0, 0].astype(np.float32)   # (K, T, 2) = (accel, curvature)
        rec["n_action_dim"] = int(act.shape[-1])
        rec["pred_action"] = act.reshape(-1).tolist()                # K*T*2, row-major [k, t, (a, kappa)]
        rec["pred_accel_ml"] = act[medoid, :, 0].tolist()            # medoid longitudinal control (m/s^2)
        rec["pred_curvature_ml"] = act[medoid, :, 1].tolist()        # medoid lateral control (1/m)
    return rec


# ---------------------------------------------------------------------- reporting
def summarize(ok: "pd.DataFrame", k: int | None = None) -> None:
    """Print a horizon table (best-of-K vs most-likely) and per-cluster means."""
    if len(ok) == 0:
        print("No successful clips to summarize.")
        return
    klbl = f"_{k}" if k else "_K"
    print(f"\nHorizon |  minADE{klbl} (best-of-{k or 'K'})  |  ADE (most-likely)   [mean | median, meters]")
    print("--------+----------------------------+--------------------------")
    for h in ADE_HORIZONS:
        a, b = ok[f"min_ade_{h}s"], ok[f"ade_ml_{h}s"]
        print(f"  {h}s    |   {a.mean():.3f} | {a.median():.3f}            |   {b.mean():.3f} | {b.median():.3f}")
    print("\nPer event cluster (mean minADE@6s / ADE_ml@6s, count):")
    per = ok.groupby("event_cluster").agg(
        min_ade_6s=("min_ade_6s", "mean"),
        ade_ml_6s=("ade_ml_6s", "mean"),
        count=("min_ade_6s", "count"),
    )
    print(per.sort_values("min_ade_6s").to_string(float_format=lambda x: f"{x:.3f}"))


# ------------------------------------------------------------------------ harness
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by both model eval scripts."""
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Path to ood_reasoning.parquet")
    parser.add_argument("--split", default="val", choices=["val", "train", "both"],
                        help="Dataset split to evaluate (train has ~1450 clips vs val's ~290).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N events")
    parser.add_argument("--first_event_only", action="store_true",
                        help="Evaluate only the first event per clip (default: ALL annotated events -> "
                             "one eval row per scenario/decision moment).")
    parser.add_argument("--num_traj_samples", type=int, default=6,
                        help="K trajectories per clip; paper reports minADE_6 (K=6). VRAM scales with K.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_generation_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        choices=["sdpa", "flash_attention_2"], help="Attention backend.")
    parser.add_argument("--no_coc", action="store_true",
                        help="Skip chain-of-causation reasoning (inject empty CoC -> straight to trajectory).")
    parser.add_argument("--out", default=None, help="Path to write per-clip results (.parquet or .csv)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards for multi-GPU runs")
    parser.add_argument("--shard_idx", type=int, default=0, help="This process's shard index in [0, num_shards)")
    parser.add_argument(
        "--retry_failed", default=None,
        help="Path to a previous run's output (.parquet/.csv). Restricts this run to just the "
             "(clip_id, event_idx) pairs that had ok=False AND a CUDA out-of-memory error there "
             "-- i.e. transient GPU-contention failures, not deterministic ones (e.g. pose "
             "interpolation-range errors, which will just fail again). Output can then be merged "
             "back into the original with merge_retry().",
    )


def select_rows(args) -> list[dict]:
    """Load events for the chosen split, apply --limit, then take this process's shard."""
    all_events = not args.first_event_only
    rows = load_events(args.parquet, all_events=all_events, split=args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.retry_failed:
        prev = pd.read_parquet(args.retry_failed) if args.retry_failed.endswith(".parquet") \
            else pd.read_csv(args.retry_failed)
        oom = prev[(prev["ok"] == False) & prev["error"].str.contains("CUDA out of memory", na=False)]
        keys = set(zip(oom["clip_id"], oom["event_idx"]))
        rows = [r for r in rows if (r["clip_id"], r["event_idx"]) in keys]
        print(f"--retry_failed {args.retry_failed}: retrying {len(rows)}/{len(keys)} OOM-failed events "
              f"(rest were excluded by --split/--limit)")
    mode = "first-event-only" if args.first_event_only else "all-events"
    if args.num_shards > 1:
        rows = rows[args.shard_idx :: args.num_shards]
        print(f"[shard {args.shard_idx}/{args.num_shards}] split={args.split} ({mode}) {len(rows)} events "
              f"(K={args.num_traj_samples}, no_coc={args.no_coc})")
    else:
        print(f"Evaluating split={args.split} ({mode}): {len(rows)} events "
              f"(K={args.num_traj_samples}, no_coc={args.no_coc})")
    return rows


def run_eval(rows: list[dict], loader, predict, avdi, args) -> "pd.DataFrame":
    """Shared per-clip loop. `loader(clip_id, t0_us, avdi)` -> data dict;
    `predict(data)` -> per-clip record dict (metrics + trajectories)."""
    results = []
    t_start = time.time()
    n = len(rows)
    for i, r in enumerate(rows):
        try:
            data = loader(r["clip_id"], t0_us=r["t0_us"], avdi=avdi)
            rec = predict(data)
            results.append({**r, **rec, "ok": True})
            print(f"[{i + 1}/{n}] {r['clip_id']} ({r['event_cluster']}) "
                  f"minADE@3s={rec['min_ade_3s']:.3f}m  minADE@6s={rec['min_ade_6s']:.3f}m")
        except Exception as e:  # keep going; report failures at the end
            results.append({**r, **{k: np.nan for k in METRIC_KEYS}, "ok": False, "error": str(e)})
            print(f"[{i + 1}/{n}] {r['clip_id']} FAILED: {e}")

    res_df = pd.DataFrame(results)
    ok = res_df[res_df["ok"]] if len(res_df) else res_df
    print("\n" + "=" * 70)
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(res_df)} succeeded.")
    if len(ok):
        summarize(ok, k=args.num_traj_samples)
    write_results(res_df, args)
    return res_df


def write_results(res_df: "pd.DataFrame", args) -> None:
    if not args.out:
        return
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if out_path.endswith(".csv"):
        res_df.to_csv(out_path, index=False)
    else:
        res_df.to_parquet(out_path, index=False)
    print(f"\nWrote per-clip results to {out_path}")
