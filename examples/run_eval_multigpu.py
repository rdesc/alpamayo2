# SPDX-License-Identifier: Apache-2.0
"""Data-parallel launcher for the PAI-AV OOD eval across multiple GPUs.

Copied from /mnt/efs/users/rod/repos/alpamayo1.5 with the default --eval_script pointed
at this repo's Alpamayo 2 script; keep the two in sync. Shards run under sys.executable,
so launch it with this repo's .venv/bin/python.

Each GPU runs one shard of the clip list (``rows[shard_idx::num_shards]``) in a
separate process pinned via ``CUDA_VISIBLE_DEVICES``. When all shards finish, the
per-shard result parquets are concatenated and the combined minADE (overall +
per event cluster) is printed.

Usage
-----
    # Use all 8 GPUs, K=16, FA2:
    python run_eval_multigpu.py --gpus 0,1,2,3,4,5,6,7 \
        --num_traj_samples 16 --attn_implementation flash_attention_2 \
        --out pai_av_val_results.parquet

    # Use only the idle GPUs (leave GPU 0 for another job):
    python run_eval_multigpu.py --gpus 1,2,3,4,5,6,7 --out pai_av_val_results.parquet

Any unrecognized args are forwarded to eval_pai_av_val.py (e.g. --all_events,
--limit, --temperature). The merged output goes to --out; per-shard files are
written next to it as <stem>.shard<i>-of-<N>.<ext>.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from eval_common import summarize

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated physical GPU ids")
    parser.add_argument("--out", default="pai_av_val_results.parquet", help="Merged output path (.parquet/.csv)")
    parser.add_argument("--logdir", default="/tmp/pai_av_val_shards", help="Where per-shard logs are written")
    parser.add_argument("--eval_script", default="eval_pai_av_val_a2.py",
                        help="Per-shard eval script (eval_pai_av_val_a2.py for Alpamayo 2 Super)")
    # Everything else (e.g. --num_traj_samples, --attn_implementation, --no_coc) is forwarded.
    args, passthrough = parser.parse_known_args()
    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = HERE / eval_script

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    n = len(gpus)
    if n == 0:
        sys.exit("No GPUs specified.")
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    print(f"Launching {n} shards across GPUs {gpus} using {eval_script.name}")
    print(f"Forwarded args: {passthrough}")

    procs = []
    t0 = time.time()
    for shard_idx, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [
            sys.executable, str(eval_script),
            "--num_shards", str(n),
            "--shard_idx", str(shard_idx),
            "--out", args.out,
            *passthrough,
        ]
        log_path = Path(args.logdir) / f"shard{shard_idx}-of-{n}.log"
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((shard_idx, gpu, p, log_f, log_path))
        print(f"  shard {shard_idx} -> GPU {gpu}  (pid {p.pid}, log {log_path})")

    # Wait for all shards.
    failed = []
    for shard_idx, gpu, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        status = "ok" if rc == 0 else f"EXIT {rc}"
        print(f"  shard {shard_idx} (GPU {gpu}) finished: {status}  [{log_path}]")
        if rc != 0:
            failed.append((shard_idx, log_path))

    if failed:
        print(f"\nWARNING: {len(failed)} shard(s) failed; merged results will be incomplete.")
        for shard_idx, log_path in failed:
            print(f"  shard {shard_idx}: see {log_path}")

    # Merge per-shard parquets.
    root, _, ext = args.out.rpartition(".")
    shard_paths = [Path(f"{root}.shard{i}-of-{n}.{ext}") for i in range(n)]
    existing = [p for p in shard_paths if p.exists()]
    if not existing:
        sys.exit("No shard outputs found to merge.")

    if ext == "csv":
        merged = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        merged.to_csv(args.out, index=False)
    else:
        merged = pd.concat([pd.read_parquet(p) for p in existing], ignore_index=True)
        merged.to_parquet(args.out, index=False)

    ok = merged[merged["ok"]] if "ok" in merged.columns else merged
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"All shards done in {elapsed / 60:.1f} min. Merged {len(merged)} rows "
          f"({len(ok)} succeeded) -> {args.out}")
    # Use WOD-E2E summarizer if min_ade_6s is absent (capped at 5s)
    if "min_ade_6s" not in ok.columns:
        from eval_wod_e2e import summarize_wod
        summarize_wod(ok)
    else:
        summarize(ok)


if __name__ == "__main__":
    main()
