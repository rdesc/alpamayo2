# SPDX-License-Identifier: Apache-2.0
"""Backfill `gold_coc` into the Alpamayo-2-Super LLR example manifest.

`extract_llr_viz_a2s.py` populated gold_coc via
``source.get("coc") or source.get("gold_coc") or ""`` -- but the gold Chain-of-Causation text
lives in the event metadata parquet (``reasoning/ood_reasoning.parquet``, carried through to the
merged llr_act results), not in what ``load_physical_aiavdataset`` returns. So every entry came
out as "". The Alpamayo-1.5 side was unaffected because it read gold_coc straight off its merged
parquet.

This matters because the published artifact's panels show gold CoC as the caption for each
example -- it is half of what the llr_act measurement conditions on.

Reads the merged A2S llr_act parquet, matches on (clip_id, event_idx), and writes gold_coc back
into the manifest in place. Everything else is left untouched.
"""

from __future__ import annotations

import json

import pandas as pd

MANIFEST_PATH = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo2_super_v2/manifest.json"
)
MERGED_PARQUET = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/a2s_merged.parquet"
)


def main() -> None:
    df = pd.read_parquet(MERGED_PARQUET)
    lookup = {
        (str(r["clip_id"]), int(r["event_idx"])): r["gold_coc"]
        for _, r in df.iterrows()
    }

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    filled = 0
    for entry in manifest:
        key = (str(entry["clip_id"]), int(entry["event_idx"]))
        gold = lookup.get(key)
        if gold:
            entry["gold_coc"] = str(gold)
            filled += 1
            print(f"[backfill] {entry['label']}{entry['rank']}: {str(gold)[:70]}")
        else:
            print(f"[backfill] {entry['label']}{entry['rank']}: NOT FOUND for {key}")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[backfill] filled {filled}/{len(manifest)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
