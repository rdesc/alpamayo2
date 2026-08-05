"""Alpamayo 2 Super pass over the full TruckDrive validation split.

Reproduces the val protocol ``recipes/alpamayo1_5_sft/configs/sft_truckdrive.yaml``
used for Alpamayo 1.5 (reported in alpamayo-recipes' docs as "2,474 windows /
139 scenes" for the 5-view config): every ``t0_stride``-th pose step across
each scene's full valid timeline (not one window per scene), scenes missing any
of the 5 validated camera views dropped entirely, reversing/heavy-sideslip
windows dropped (``filter_reverse``), and near-stationary windows' ground truth
snapped to exact standstill (``standstill_snap_mps``) -- same defaults as that
config, so the (windows, ground truth) pairs match theirs.

Loads the devkit's official ``validation_scenes`` list from
``TruckDrive/metainfo.json``. Results are written incrementally (one line per
window) so a partial run is never lost, plus a final aggregate summary using
the same metric set as their eval (see ``truckdrive_metrics.py``).

Example:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python examples/truckdrive_eval_val.py \\
      --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \\
      --metainfo /mnt/efs/users/rod/repos/TruckDrive/metainfo.json \\
      --out outputs/truckdrive_val_eval_windowed.jsonl
"""

import argparse
import json
import os
import time

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID
from alpamayo2_super.load_truckdrive import DEFAULT_DATA_ROOT


def _prepare_model_inputs(data, model_config, tokenizer, helper_mod, enable_cot: bool):
    """Same as ``helper.prepare_model_inputs`` but lets ``components_prompt`` drop
    "cot" to suppress Chain-of-Causation generation (verified: with "cot" absent,
    the instruction text drops "output the chain-of-thought reasoning..." and
    ``extra["cot"]`` comes back empty rather than a real reasoning string --
    ``helper.create_messages`` hardcodes ``["cot", "traj_future"]`` and has no
    such switch, hence this local copy)."""
    from alpamayo2_super.chat_template.conversation import build_conversation

    components_prompt = ["cot", "traj_future"] if enable_cot else ["traj_future"]
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=components_prompt,
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=model_config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    processor = helper_mod.get_processor(tokenizer, model_config)
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=not has_assistant_content,
        add_vision_id=False, continue_final_message=has_assistant_content,
    )
    import torch

    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(text=text, images=images, videos=None, padding=False,
                  return_tensors="pt", do_rescale=False)
    )
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }


def run(
    model_id: str,
    metainfo_path: str,
    data_root: str,
    backend: str,
    out_path: str,
    max_scenes: int,
    max_windows: int,
    t0_stride: int,
    filter_reverse: bool,
    standstill_snap_mps: float | None,
    num_traj_samples: int,
    diffusion_steps: int,
    seed: int,
    num_shards: int = 1,
    shard_index: int = 0,
    num_frames: int = 4,
    enable_cot: bool = True,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    from alpamayo2_super import helper, truckdrive_metrics
    from alpamayo2_super.load_truckdrive import enumerate_val_windows, load_truckdrive_sample
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    if not torch.cuda.is_available():
        raise RuntimeError("TruckDrive eval requires a CUDA GPU.")

    with open(metainfo_path) as f:
        meta = json.load(f)
    scene_ids = sorted(set(meta["validation_scenes"]))
    if max_scenes > 0:
        scene_ids = scene_ids[:max_scenes]
    print(f"{len(scene_ids)} validation scenes from {metainfo_path}")

    print("Enumerating val windows (pose + camera-listing metadata only)...")
    windows = enumerate_val_windows(
        scene_ids,
        data_root=data_root,
        backend=backend,
        t0_stride=t0_stride,
        filter_reverse=filter_reverse,
        num_frames=num_frames,
    )
    if num_shards > 1:
        full_count = len(windows)
        windows = windows[shard_index::num_shards]
        print(f"Shard {shard_index}/{num_shards}: {len(windows)}/{full_count} windows.")
    if max_windows > 0:
        windows = windows[:max_windows]
    print(f"Evaluating {len(windows)} windows.")

    print(f"Loading {model_id}...")
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records = []
    t_start = time.time()
    with open(out_path, "w") as out_f:
        for i, (scene_id, t0) in enumerate(windows, 1):
            t_win = time.time()
            try:
                data = load_truckdrive_sample(
                    scene_id=scene_id, t0_s=t0, data_root=data_root, backend=backend,
                    include_calibration=False, standstill_snap_mps=standstill_snap_mps,
                    num_frames=num_frames,
                )
                model_inputs = _prepare_model_inputs(
                    data, model.config, model.tokenizer, helper, enable_cot
                )
                model_inputs = helper.to_device(model_inputs, "cuda")

                torch.cuda.manual_seed_all(seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, _logprob, extra = model.sample_trajectories_from_data(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=num_traj_samples,
                        diffusion_kwargs={"inference_step": diffusion_steps},
                        return_extra=True,
                    )
                pred_xyz_cpu = pred_xyz.float().cpu()
                pred_rot_cpu = pred_rot.float().cpu()
                gt_xyz_b = data["ego_future_xyz"][:, -1].cpu()  # [B, T, 3]
                gt_rot_b = data["ego_future_rot"][:, -1].cpu()  # [B, T, 3, 3]

                metrics = truckdrive_metrics.displacement_metrics_from_tensors(pred_xyz_cpu, gt_xyz_b)
                metrics["corner_distance"] = truckdrive_metrics.corner_distance(
                    pred_xyz_cpu, pred_rot_cpu, gt_xyz_b, gt_rot_b
                )
                cot = extra["cot"][0] if isinstance(extra["cot"], (list, tuple)) else extra["cot"]

                record = {
                    "scene_id": scene_id,
                    "t0_s": t0,
                    "n_cameras": int(data["camera_indices"].numel()),
                    **metrics,
                    "cot": cot,
                    "status": "ok",
                }
            except Exception as exc:  # noqa: BLE001 - one bad window must not kill the run
                record = {"scene_id": scene_id, "t0_s": t0, "status": "error", "error": repr(exc)}

            record["elapsed_s"] = round(time.time() - t_win, 2)
            records.append(record)
            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()

            tag = (
                f"minADE/6.4s={record['min_ade/by_t=6.4']:.2f}m corner={record['corner_distance']:.2f}m"
                if record["status"] == "ok"
                else record["status"]
            )
            if i % 10 == 0 or i == len(windows):
                elapsed = time.time() - t_start
                rate = elapsed / i
                eta_min = rate * (len(windows) - i) / 60
                print(
                    f"[{i}/{len(windows)}] {scene_id}@{t0:.1f}s {tag} "
                    f"({record['elapsed_s']:.1f}s, ETA {eta_min:.0f}min)", flush=True,
                )

    ok = [r for r in records if r["status"] == "ok"]
    errored = [r for r in records if r["status"] != "ok"]
    summary = {
        "n_windows": len(records),
        "n_scenes": len({r["scene_id"] for r in records}),
        "n_ok": len(ok),
        "n_error": len(errored),
        "total_elapsed_s": round(time.time() - t_start, 1),
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
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    summary_path = out_path.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote per-window records to", out_path)
    print("Wrote summary to", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID)
    )
    parser.add_argument(
        "--metainfo",
        default="/mnt/efs/users/rod/repos/TruckDrive/metainfo.json",
        help="TruckDrive devkit metainfo.json (validation_scenes list).",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TRUCKDRIVE_DATA_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--backend", choices=["s3", "local"], default="s3")
    parser.add_argument("--out", default="outputs/truckdrive_val_eval_windowed.jsonl")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 = all validation scenes.")
    parser.add_argument("--max-windows", type=int, default=0, help="0 = all enumerated windows.")
    parser.add_argument("--t0-stride", type=int, default=10, help="Matches sft_truckdrive.yaml.")
    parser.add_argument("--no-filter-reverse", action="store_true")
    parser.add_argument(
        "--standstill-snap-mps", type=float, default=0.5,
        help="Matches sft_truckdrive.yaml. Pass a negative number to disable.",
    )
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Split the full window list into this many shards (one per GPU process).",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Which shard this process evaluates (0-indexed, < --num-shards).",
    )
    parser.add_argument(
        "--num-frames", type=int, default=4,
        help="Camera frames per view ending at t0. 1 = current-frame-only (drop history).",
    )
    parser.add_argument(
        "--no-cot", action="store_true",
        help="Drop 'cot' from the generation prompt, suppressing Chain-of-Causation text "
             "(verified to actually empty extra['cot'], not just skip reporting it). "
             "Untested relative to the release checkpoint's trained conditioning.",
    )
    args = parser.parse_args()

    run(
        model_id=args.model_id,
        metainfo_path=args.metainfo,
        data_root=args.data_root,
        backend=args.backend,
        out_path=args.out,
        max_scenes=args.max_scenes,
        max_windows=args.max_windows,
        t0_stride=args.t0_stride,
        filter_reverse=not args.no_filter_reverse,
        standstill_snap_mps=None if args.standstill_snap_mps < 0 else args.standstill_snap_mps,
        num_traj_samples=args.num_traj_samples,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        num_frames=args.num_frames,
        enable_cot=not args.no_cot,
    )


if __name__ == "__main__":
    main()
