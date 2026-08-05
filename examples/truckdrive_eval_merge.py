"""Merge sharded truckdrive_eval_val.py outputs into one summary."""

import argparse
import glob
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_glob", help='e.g. "outputs/truckdrive_val_eval_windowed.shard*.jsonl"')
    parser.add_argument("--out", default=None, help="Merged .jsonl path (default: alongside shards).")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        raise SystemExit(f"no files matched {args.shard_glob!r}")
    print(f"Merging {len(paths)} shards: {paths}")

    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    out_path = args.out or args.shard_glob.replace("shard*", "merged").replace("*", "merged")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in records if r["status"] == "ok"]
    errored = [r for r in records if r["status"] != "ok"]
    summary = {
        "n_windows": len(records),
        "n_scenes": len({r["scene_id"] for r in records}),
        "n_ok": len(ok),
        "n_error": len(errored),
        "n_shards": len(paths),
    }
    if ok:
        metric_keys = [
            k for k in ok[0]
            if k not in ("scene_id", "t0_s", "n_cameras", "cot", "status", "elapsed_s")
        ]
        for key in metric_keys:
            values = np.array([r[key] for r in ok])
            summary[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
    print(json.dumps(summary, indent=2))
    summary_path = out_path.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote merged records to", out_path)
    print("Wrote merged summary to", summary_path)


if __name__ == "__main__":
    main()
