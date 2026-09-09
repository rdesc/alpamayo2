# SPDX-License-Identifier: Apache-2.0
"""Calibration + selective-prediction analysis for the motion-confidence verifier scores.

Measures whether a raw confidence score means what it claims, per method:
- Brier score and Expected Calibration Error (ECE), plus a reliability-diagram table
  (quantile-binned mean predicted score vs. empirical correctness rate per bin).
- A risk-coverage (selective-prediction) curve: sort decisions by how far the score sits
  from 0.5 and check whether abstaining on the least-confident fraction actually raises
  accuracy on what's left -- the more decision-relevant question if the score is meant to
  flag "the model is unsure here."

Two ground-truth sources are supported (--source):
- `k3` (default): the K=3 mismatched-gold-CoC corpus (Experiment 7 / Experiment 9's
  offset=0.0 row, `outputs/motion_confidence_k3_toff0.0_n289_part{1,2}.json`) --
  `positive_action` is correct-by-construction (label 1), each of the 3 `negative_actions`
  is mismatched-by-construction (label 0), n=289 events. Per the independent negative-QA
  judge (`outputs/motion_confidence_negative_qa_qwen3vl32b.json`), ~19% of these negatives
  are actually still plausible for their scene -- known label noise that would otherwise
  deflate every method's calibration score. By default those flagged negatives are
  EXCLUDED (--include_flagged_negatives to keep them and see the effect). These negatives
  are "easy" (Experiment 13: usually a different, unrelated scene's action).
- `selfcoc_labeled`: Experiment 13's real same-scene correct/incorrect corpus
  (`outputs/motion_confidence_selfcoc_labeled_scoring_n100.json`) -- an independent Claude-
  vision judge's Yes/No verdict on each of an event's own 8 self-generated CoC candidates,
  joined against each method's raw score on that same candidate. Only the 42/100 "mixed"
  events (>=1 Yes and >=1 No) contribute, since unanimous events have no within-event
  negative. This is the harder, more decision-relevant ground truth -- Experiment 13 found
  every method's ranking ability collapses toward chance on it relative to `k3`, so
  calibration should be expected to look different (likely worse) here too.

`baseline`/`digit`/`scene_conditioned` scores are already read as P(correct) in [0,1] and
calibrated directly. `likelihood`/`native_likelihood` are raw mean log-probabilities, not
probabilities -- they are first mapped to [0,1] via Platt scaling (a 1-D logistic
regression fit on this same labeled corpus) before Brier/ECE are computed. This is an
in-sample fit, so the likelihood rows measure how calibratable the raw score *can be made*
by an optimal monotonic remapping, not how calibrated the raw score already is -- noted
per-method in the output.

Usage
-----
    python examples/motion_confidence_calibration.py \\
        --out outputs/motion_confidence_calibration.json
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

BOUNDED_METHODS = ["baseline", "digit", "scene_conditioned"]
LIKELIHOOD_METHODS = ["likelihood", "native_likelihood"]
ALL_METHODS = BOUNDED_METHODS + LIKELIHOOD_METHODS

DEFAULT_INPUTS = [
    "outputs/motion_confidence_k3_toff0.0_n289_part1.json",
    "outputs/motion_confidence_k3_toff0.0_n289_part2.json",
]
DEFAULT_SELFCOC_LABELED = "outputs/motion_confidence_selfcoc_labeled_scoring_n100.json"


def load_events(paths):
    events = []
    for p in paths:
        events.extend(json.load(open(p, encoding="utf-8"))["results"])
    return events


def build_examples_selfcoc_labeled(path, method):
    """Experiment 13's format: results[method] = {"pos": [[candidate_idx, event_idx, score], ...],
    "neg": [...]} -- already real same-scene Yes/No labels, no filtering needed."""
    data = json.load(open(path, encoding="utf-8"))
    if method not in data["results"]:
        return []
    block = data["results"][method]
    examples = []
    for ci, ei, score in block.get("pos", []):
        examples.append({"score": score, "label": 1, "event_index": ei, "candidate_index": ci, "kind": "positive"})
    for ci, ei, score in block.get("neg", []):
        examples.append({"score": score, "label": 0, "event_index": ei, "candidate_index": ci, "kind": "negative"})
    return examples


def load_negative_qa(path):
    if not path:
        return {}
    records = json.load(open(path, encoding="utf-8"))["records"]
    return {(r["clip_id"], r["t0_us"], r["negative_index"] - 1): r["qa_verdict"] for r in records}


def raw_score(block, method):
    if method in LIKELIHOOD_METHODS:
        return block["mean_logprob"]
    return block["confidence"] if "confidence" in block else block["mean_confidence"]


def build_examples(events, method, neg_qa_lut, include_flagged):
    examples = []
    for ev in events:
        if method not in ev:
            continue
        block = ev[method]
        examples.append({
            "score": raw_score(block["positive"], method), "label": 1,
            "clip_id": ev["clip_id"], "t0_us": ev["t0_us"], "kind": "positive",
        })
        for i, neg_block in enumerate(block["negatives"]):
            verdict = neg_qa_lut.get((ev["clip_id"], ev["t0_us"], i))
            if verdict == "Yes" and not include_flagged:
                continue  # confirmed-plausible negative -- known label noise, excluded by default
            examples.append({
                "score": raw_score(neg_block, method), "label": 0,
                "clip_id": ev["clip_id"], "t0_us": ev["t0_us"], "kind": "negative", "qa_verdict": verdict,
            })
    return examples


def platt_scale(examples):
    """In-sample 1-D logistic-regression mapping from raw score to P(correct) -- see the
    module docstring's caveat on what this can and can't tell you about the raw score."""
    x = np.array([[e["score"]] for e in examples])
    y = np.array([e["label"] for e in examples])
    clf = LogisticRegression()
    clf.fit(x, y)
    probs = clf.predict_proba(x)[:, 1]
    for e, p in zip(examples, probs):
        e["prob"] = float(p)
    return examples


def brier_score(examples):
    return float(np.mean([(e["prob"] - e["label"]) ** 2 for e in examples]))


def reliability_table(examples, n_bins):
    """Quantile-binned (equal-count) reliability diagram, plus the ECE it implies."""
    sorted_ex = sorted(examples, key=lambda e: e["prob"])
    n = len(sorted_ex)
    edges = [round(i * n / n_bins) for i in range(n_bins + 1)]
    rows = []
    ece = 0.0
    for b in range(n_bins):
        chunk = sorted_ex[edges[b]:edges[b + 1]]
        if not chunk:
            continue
        mean_pred = float(np.mean([e["prob"] for e in chunk]))
        empirical = float(np.mean([e["label"] for e in chunk]))
        gap = abs(mean_pred - empirical)
        rows.append({
            "bin": b, "n": len(chunk),
            "mean_predicted": round(mean_pred, 4),
            "empirical_correct_rate": round(empirical, 4),
            "gap": round(gap, 4),
        })
        ece += len(chunk) / n * gap
    return rows, float(ece)


def risk_coverage(examples, n_points=20):
    """Sort by decision confidence |prob-0.5| descending (most confident first); at each
    coverage level, accuracy of thresholding prob>0.5 among that most-confident subset.
    A useful confidence signal should show accuracy rising as coverage shrinks."""
    scored = sorted(examples, key=lambda e: abs(e["prob"] - 0.5), reverse=True)
    n = len(scored)
    points = []
    for k in range(1, n_points + 1):
        cov = k / n_points
        m = max(1, round(cov * n))
        subset = scored[:m]
        correct = sum(1 for e in subset if (e["prob"] > 0.5) == (e["label"] == 1))
        points.append({"coverage": round(cov, 3), "n": m, "accuracy": round(correct / m, 4)})
    return points


def summarize_method(examples, method, n_bins):
    if not examples:
        return None
    if method in LIKELIHOOD_METHODS:
        examples = platt_scale(examples)
        calibration_note = "Platt-scaled (in-sample logistic fit) from raw mean_logprob -- see module docstring caveat"
    else:
        for e in examples:
            e["prob"] = e["score"]
        calibration_note = "raw confidence used directly as P(correct)"
    brier = brier_score(examples)
    rel_table, ece = reliability_table(examples, n_bins)
    rc = risk_coverage(examples)
    n_pos = sum(1 for e in examples if e["label"] == 1)
    n_neg = sum(1 for e in examples if e["label"] == 0)
    acc = sum(1 for e in examples if (e["prob"] > 0.5) == (e["label"] == 1)) / len(examples)
    return {
        "method": method, "n": len(examples), "n_positive": n_pos, "n_negative": n_neg,
        "calibration_note": calibration_note,
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "accuracy_at_0.5": round(acc, 4),
        "reliability_table": rel_table,
        "risk_coverage": rc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["k3", "selfcoc_labeled"], default="k3")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="--source k3 only")
    parser.add_argument("--negative_qa", default="outputs/motion_confidence_negative_qa_qwen3vl32b.json",
                         help="--source k3 only")
    parser.add_argument(
        "--include_flagged_negatives", action="store_true",
        help="--source k3 only: keep negatives the negative-QA judge flagged as still-plausible "
             "(known label noise). Default: exclude.",
    )
    parser.add_argument("--selfcoc_labeled_input", default=DEFAULT_SELFCOC_LABELED, help="--source selfcoc_labeled only")
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--out", default="outputs/motion_confidence_calibration.json")
    args = parser.parse_args()

    if args.source == "k3":
        events = load_events(args.inputs)
        neg_qa_lut = load_negative_qa(args.negative_qa)
        n_events_desc = len(events)
        example_sets = {
            m: build_examples(events, m, neg_qa_lut, args.include_flagged_negatives) for m in ALL_METHODS
        }
    else:
        example_sets = {m: build_examples_selfcoc_labeled(args.selfcoc_labeled_input, m) for m in ALL_METHODS}
        n_events_desc = json.load(open(args.selfcoc_labeled_input, encoding="utf-8"))["mixed_events"]

    results = {}
    for method in ALL_METHODS:
        summary = summarize_method(example_sets[method], method, args.n_bins)
        if summary is not None:
            results[method] = summary

    out = {
        "args": vars(args),
        "source": args.source,
        "n_events": n_events_desc,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"source={args.source}  n_events={n_events_desc}\n")
    header = f"{'method':<18}{'n':>6}{'brier':>10}{'ece':>10}{'acc@0.5':>10}"
    print(header)
    print("-" * len(header))
    for method, s in results.items():
        print(f"{method:<18}{s['n']:>6}{s['brier_score']:>10.4f}{s['ece']:>10.4f}{s['accuracy_at_0.5']:>10.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
