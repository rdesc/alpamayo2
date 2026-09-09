# SPDX-License-Identifier: Apache-2.0
"""Aggregate the K=3 temporal-jitter sweep (Experiment 9): for each t0 offset, merge
part1+part2 (145+144 = 289 events) and recompute, per method, the same 3 metrics
motion_confidence_smoke.py's own summarize() reports per-shard, but over the combined
set. Also reports MIN_T0_US clamping counts using load_t0_us vs t0_us.

Usage:
    python examples/motion_confidence_aggregate_toff.py
"""

import glob
import json
import math
import os

MIN_T0_US = 1_700_000

OFFSETS = ["-0.2", "-0.1", "0.0", "0.1", "0.2"]
METHODS = ["baseline", "digit", "likelihood", "scene_conditioned"]

FIELD = {
    "baseline": "confidence",
    "digit": "confidence",
    "likelihood": "mean_logprob",
    "scene_conditioned": "mean_confidence",
}

OUT_DIR = "outputs"


def load_parts(offset):
    p1 = os.path.join(OUT_DIR, f"motion_confidence_k3_toff{offset}_n289_part1.json")
    p2 = os.path.join(OUT_DIR, f"motion_confidence_k3_toff{offset}_n289_part2.json")
    d1 = json.load(open(p1))
    d2 = json.load(open(p2))
    return d1["results"] + d2["results"], (p1, p2)


def auroc(pos_list, neg_flat):
    n, n_neg = len(pos_list), len(neg_flat)
    if n == 0 or n_neg == 0:
        return float("nan")
    y_score = pos_list + neg_flat
    y_true = [1] * n + [0] * n_neg
    try:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y_true, y_score)
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
        return (sum_ranks_pos - n * (n + 1) / 2.0) / (n * n_neg)


def summarize(results, method):
    field = FIELD[method]
    pos_list, neg_lists = [], []
    n_error = 0
    for r in results:
        if "error" in r:
            n_error += 1
            continue
        if method not in r:
            continue
        pos_entry = r[method].get("positive")
        neg_entries = r[method].get("negatives", [])
        if pos_entry is None or any(x is None for x in neg_entries):
            continue
        p = pos_entry[field]
        ns = [x[field] for x in neg_entries]
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
    a = auroc(pos_list, neg_flat)
    return {
        "n": n, "n_error": n_error, "mean_pos": mean_pos, "mean_neg": mean_neg,
        "pairwise_acc": pairwise_acc, "top1_acc": top1_acc, "auroc": a,
    }


def clamp_count(results, offset_f):
    """Count events where load_t0_us was clamped to MIN_T0_US (no real shift happened),
    i.e. t0_us + round(offset*1e6) < MIN_T0_US but load_t0_us == MIN_T0_US."""
    n_clamped = 0
    n_total = 0
    for r in results:
        if "error" in r or "load_t0_us" not in r:
            continue
        n_total += 1
        intended = r["t0_us"] + round(offset_f * 1_000_000)
        if intended < MIN_T0_US and r["load_t0_us"] == MIN_T0_US:
            n_clamped += 1
    return n_clamped, n_total


def main():
    all_summaries = {}
    all_errors = {}
    all_clamp = {}
    for offset in OFFSETS:
        results, (p1, p2) = load_parts(offset)
        n_events = len(results)
        n_err = sum(1 for r in results if "error" in r)
        all_errors[offset] = (n_err, n_events)
        offset_f = float(offset)
        n_clamped, n_total = clamp_count(results, offset_f)
        all_clamp[offset] = (n_clamped, n_total)
        all_summaries[offset] = {}
        for method in METHODS:
            all_summaries[offset][method] = summarize(results, method)

    print(f"{'offset':>7} | {'n_events':>8} | {'n_error':>7} | {'n_clamped':>9}")
    for offset in OFFSETS:
        n_err, n_ev = all_errors[offset]
        n_cl, n_tot = all_clamp[offset]
        print(f"{offset:>7} | {n_ev:>8} | {n_err:>7} | {n_cl:>9} (/{n_tot})")

    print()
    for method in METHODS:
        print(f"\n=== {method} ===")
        print(f"{'offset':>7} | {'n':>4} | {'mean_pos':>9} | {'mean_neg':>9} | {'pairwise_acc':>12} | {'top1_acc':>8} | {'AUROC':>6}")
        for offset in OFFSETS:
            s = all_summaries[offset][method]
            print(
                f"{offset:>7} | {s['n']:>4} | {s['mean_pos']:>9.3f} | {s['mean_neg']:>9.3f} | "
                f"{s['pairwise_acc']:>12.3f} | {s['top1_acc']:>8.3f} | {s['auroc']:>6.3f}"
            )

    out = {"summaries": all_summaries, "errors": all_errors, "clamp_counts": all_clamp}
    out_path = os.path.join(OUT_DIR, "motion_confidence_toff_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
