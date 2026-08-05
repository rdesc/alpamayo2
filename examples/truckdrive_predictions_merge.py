"""Merge sharded truckdrive_eval_val.py --save-predictions .pt outputs into one file."""

import argparse
import glob

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_glob", help='e.g. "outputs/truckdrive_preds_baseline.shard*.pt"')
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        raise SystemExit(f"no files matched {args.shard_glob!r}")
    print(f"Merging {len(paths)} shards: {paths}")

    records = []
    for p in paths:
        records.extend(torch.load(p, map_location="cpu", weights_only=False))

    scenes = {r["scene_id"] for r in records}
    print(f"{len(records)} records across {len(scenes)} scenes")
    torch.save(records, args.out)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
