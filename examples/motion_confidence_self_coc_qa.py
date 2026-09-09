# SPDX-License-Identifier: Apache-2.0
"""Independent verification of A2S's own self-generated chain-of-causation (CoC)
action strings used by ``motion_confidence_smoke.py --multi_coc``-style outputs
(e.g. ``outputs/motion_confidence_multi_coc_val_n289.json``): for each event, A2S
was independently sampled (temperature=0.6, top_p=0.98) 8 times to produce 8
candidate "self CoC" action/reasoning strings. Nobody has ever checked whether
these self-generated CoCs are actually correct for their scene -- they have been
assumed "correct by construction" (i.e. because A2S itself produced them), but
that assumption is never verified. This script uses a second, independent VLM
(Qwen3-VL-32B-Instruct, NOT Alpamayo2Super) as a judge to check that assumption:
for each event, for each of its 8 self-generated CoC strings, show the judge the
actual scene images (current-frame 6-camera ring, same "vqa" input profile
Alpamayo uses) and ask, NEUTRALLY (the judge is never told the string came from
the model itself, nor is any gold/reference CoC ever mentioned), whether the
candidate action is a reasonable and correct driving decision for the ego
vehicle right now.

Judge reliability (v2): the smoke test surfaced within-event verdict splits on
near-identical phrasings that looked like judge noise rather than a real
distinction (see ``docs/motion_confidence_experiment.md`` Experiment 7). To
address that, this script now (1) prompts the judge to reason step-by-step
BEFORE committing to a verdict (CoT-then-answer, not verdict-first), and (2)
samples the judge 3x per candidate (do_sample=True, moderate temperature) and
takes a 2-of-3 majority vote, recording all 3 raw verdicts+explanations so
disagreement can still be inspected. See ``motion_confidence_judge_common.py``.

"Yes" = self-generated CoC judged correct for its scene (supports the "correct
by construction" assumption). "No" / "Uncertain" / "no_majority" / unparseable =
FLAG for human review -- the self-generated CoC may not actually fit the scene.

Usage
-----
    python examples/motion_confidence_self_coc_qa.py \\
        --inputs outputs/motion_confidence_multi_coc_val_n289.json \\
        --out outputs/motion_confidence_self_coc_qa_smoke_v2.json \\
        --gpu 4 --limit_events 8
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_confidence_judge_common import (  # noqa: E402
    JUDGE_PROMPT_COT,
    build_pil_images,
    generate_judge_samples,
)


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
    import time

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


def load_events_from_outputs(paths):
    """Read motion_confidence multi-CoC output JSON(s) and return a flat list of
    {clip_id, t0_us, self_cocs: [str, ...]} for every event that loaded
    successfully (has "self_cocs") -- skips events recorded with an "error"."""
    events = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for r in payload["results"]:
            if "error" in r or "self_cocs" not in r:
                continue
            events.append({
                "clip_id": r["clip_id"],
                "t0_us": r["t0_us"],
                "self_cocs": r["self_cocs"],
            })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/motion_confidence_self_coc_qa_qwen3vl32b.json")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index to pin the judge model to.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Increased from the verdict-first default (96) to leave room for CoT reasoning before the verdict. 384 was tried first and observed truncating a non-trivial fraction (~18%) of samples mid-reasoning before they reached a final verdict line; 512 was chosen to make that rare.")
    parser.add_argument("--num_judge_samples", type=int, default=3, help="Number of independent judge samples per candidate; final verdict is the 2-of-3 majority.")
    parser.add_argument("--judge_temperature", type=float, default=0.7)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--limit_events", type=int, default=None, help="For smoke-testing: only process the first N events.")
    parser.add_argument("--limit_samples", type=int, default=None, help="For smoke-testing: only process the first N self_cocs per event.")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset

    device = f"cuda:{args.gpu}"
    events = load_events_from_outputs(args.inputs)
    if args.limit_events:
        events = events[: args.limit_events]
    print(f"Loaded {len(events)} events from {args.inputs}")

    print(f"Loading judge model {args.model_id} on {device} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()

    records = []
    counts = {"Yes": 0, "No": 0, "Uncertain": 0, "no_majority": 0, "unparseable": 0}

    for ei, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        self_cocs = ev["self_cocs"]
        if args.limit_samples:
            self_cocs = self_cocs[: args.limit_samples]
        print(f"\n[{ei + 1}/{len(events)}] {clip_id} t0={t0_us} ({len(self_cocs)} self_cocs)")

        try:
            source_data, vqa_data = load_scene_with_retry(
                load_physical_aiavdataset, select_task_input, clip_id, t0_us
            )
            pil_images = build_pil_images(source_data, vqa_data)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to load scene: {e}")
            traceback.print_exc()
            for k, action in enumerate(self_cocs):
                records.append({
                    "clip_id": clip_id, "t0_us": t0_us, "sample_index": k,
                    "self_coc": action, "majority_verdict": "unparseable",
                    "raw_samples": [{"verdict": "unparseable", "explanation": f"SCENE LOAD ERROR: {e}"}],
                })
                counts["unparseable"] += 1
            continue

        for k, action in enumerate(self_cocs):
            samples, majority_verdict = generate_judge_samples(
                model, processor, process_vision_info, device, pil_images, action,
                prompt_template=JUDGE_PROMPT_COT,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_judge_samples,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
            )

            counts[majority_verdict] = counts.get(majority_verdict, 0) + 1
            record = {
                "clip_id": clip_id, "t0_us": t0_us, "sample_index": k,
                "self_coc": action, "majority_verdict": majority_verdict,
                "raw_samples": samples,
            }
            records.append(record)
            raw_verdicts = [s["verdict"] for s in samples]
            print(f"  sample_{k}: majority={majority_verdict}  raw_verdicts={raw_verdicts}")
            print(f"    [0] {samples[0]['explanation'][:200]!r}")

        if (ei + 1) % 10 == 0 or ei == len(events) - 1:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({
                    "args": vars(args), "judge_model": args.model_id,
                    "counts": counts, "records": records,
                }, f, indent=2)
            print(f"  [checkpoint] wrote {len(records)} records to {args.out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args), "judge_model": args.model_id,
            "counts": counts, "records": records,
        }, f, indent=2)

    print(f"\nWrote {len(records)} records to {args.out}")
    print(f"Verdict counts: {counts}")


if __name__ == "__main__":
    main()
