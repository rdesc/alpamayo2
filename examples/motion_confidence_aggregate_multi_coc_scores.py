# SPDX-License-Identifier: Apache-2.0
"""Merge the two-part full self-CoC candidate scoring run (Experiment 10) into one file,
and report a purely descriptive within-event spread statistic per method (max-min and
std of that method's confidence/score across the 8 self-CoC candidates per event, then
averaged across events). This is NOT a correctness/accuracy metric -- positive/negative
pairing for this data is explicitly deferred -- it only describes how much within-scene
disagreement each method assigns to 8 candidates that are mostly paraphrases of the same
underlying claim.

Usage:
    python examples/motion_confidence_aggregate_multi_coc_scores.py
"""

import json
import math
import os

OUT_DIR = "outputs"
PART1 = os.path.join(OUT_DIR, "motion_confidence_multi_coc_scores_val_n289_part1.json")
PART2 = os.path.join(OUT_DIR, "motion_confidence_multi_coc_scores_val_n289_part2.json")
MERGED_OUT = os.path.join(OUT_DIR, "motion_confidence_multi_coc_scores_val_n289_merged.json")

METHODS = ["baseline", "digit", "likelihood", "scene_conditioned"]
FIELD = {
    "baseline": "confidence",
    "digit": "confidence",
    "likelihood": "mean_logprob",
    "scene_conditioned": "mean_confidence",
}


def std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def main():
    d1 = json.load(open(PART1))
    d2 = json.load(open(PART2))
    results = d1["results"] + d2["results"]

    with open(MERGED_OUT, "w") as f:
        json.dump({"part1_args": d1["args"], "part2_args": d2["args"], "results": results}, f, indent=2)
    print(f"Wrote merged raw scores: {MERGED_OUT} ({len(results)} events)")

    n_error = sum(1 for r in results if "error" in r)
    print(f"n_events={len(results)}  n_error={n_error} ({100*n_error/len(results):.1f}%)")
    if n_error:
        print("Sample errors:")
        shown = 0
        for r in results:
            if "error" in r and shown < 3:
                print(f"  {r['clip_id']} t0={r.get('t0_us')}: {r['error']}")
                shown += 1

    print(f"\n{'method':>18} | {'n_events':>8} | {'mean spread (max-min)':>22} | {'mean std':>9}")
    per_method_summary = {}
    for method in METHODS:
        field = FIELD[method]
        spreads, stds = [], []
        n_skipped = 0
        for r in results:
            if "error" in r:
                continue
            scores = r.get("scores", [])
            vals = []
            for s in scores:
                entry = s.get(method)
                if entry is None:
                    continue
                v = entry.get(field)
                if v is None:
                    continue
                vals.append(v)
            if len(vals) < 2:
                n_skipped += 1
                continue
            spreads.append(max(vals) - min(vals))
            stds.append(std(vals))
        n = len(spreads)
        mean_spread = sum(spreads) / n if n else float("nan")
        mean_std = sum(stds) / n if n else float("nan")
        per_method_summary[method] = {
            "n_events": n, "n_skipped": n_skipped,
            "mean_spread": mean_spread, "mean_std": mean_std,
        }
        print(f"{method:>18} | {n:>8} | {mean_spread:>22.3f} | {mean_std:>9.3f}")

    out = {"n_events": len(results), "n_error": n_error, "per_method": per_method_summary}
    summary_path = os.path.join(OUT_DIR, "motion_confidence_multi_coc_scores_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
