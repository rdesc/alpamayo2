"""One-off: confirm no-CoT scores on the same 25-scene subset used for the
camera-sensitivity study, for the two configs that matter (baseline 5view/4frame
and 5view/1frame). Reuses the CoT-ON records already saved in
outputs/truckdrive_camera_sensitivity(.{,20}).jsonl for direct comparison --
only computes the CoT-OFF side here.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/efs/users/rod/repos/alpamayo2/src")
sys.path.insert(0, "/mnt/efs/users/rod/repos/alpamayo2/examples")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from alpamayo2_super import helper, truckdrive_metrics  # noqa: E402
from alpamayo2_super.load_truckdrive import load_truckdrive_sample  # noqa: E402
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from truckdrive_eval_val import _prepare_model_inputs  # noqa: E402
from truckdrive_camera_sensitivity import SCENES  # noqa: E402

with open("/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo2/6f0cebdf-3a32-440f-9cba-7988dee49630/scratchpad/sensitivity_scenes_20.json") as f:
    scenes_20 = [tuple(row) for row in json.load(f)]
all_scenes = list(SCENES) + scenes_20
print(f"{len(all_scenes)} scenes")

CONFIGS = [
    ("5view_4frame_0.2s_baseline", None, 4, 0.2),
    ("5view_1frame_currentonly", None, 1, 0.2),
]

model = Alpamayo2Super.from_pretrained("nvidia/Alpamayo2-Super", dtype=torch.bfloat16, device_map="cuda:0")

out_path = "outputs/truckdrive_cot_off_25scenes.jsonl"
records = []
t_start = time.time()
with open(out_path, "w") as out_f:
    for scene_id, t0, tag in all_scenes:
        for cfg_name, views, num_frames, img_dt in CONFIGS:
            t_w = time.time()
            try:
                data = load_truckdrive_sample(
                    scene_id=scene_id, t0_s=float(t0), view_to_alpamayo=views,
                    num_frames=num_frames, image_time_step=img_dt,
                    standstill_snap_mps=0.5, include_calibration=False,
                )
                model_inputs = _prepare_model_inputs(data, model.config, model.tokenizer, helper, enable_cot=False)
                model_inputs = helper.to_device(model_inputs, "cuda")
                torch.cuda.manual_seed_all(42)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, _lp, extra = model.sample_trajectories_from_data(
                        data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=6,
                        diffusion_kwargs={"inference_step": 10}, return_extra=True,
                    )
                pred_xyz_cpu, pred_rot_cpu = pred_xyz.float().cpu(), pred_rot.float().cpu()
                gt_xyz_b = data["ego_future_xyz"][:, -1].cpu()
                gt_rot_b = data["ego_future_rot"][:, -1].cpu()
                metrics = truckdrive_metrics.displacement_metrics_from_tensors(pred_xyz_cpu, gt_xyz_b)
                metrics["corner_distance"] = truckdrive_metrics.corner_distance(pred_xyz_cpu, pred_rot_cpu, gt_xyz_b, gt_rot_b)
                cot = extra["cot"][0] if isinstance(extra["cot"], (list, tuple)) else extra["cot"]
                record = {"scene_id": scene_id, "t0_s": t0, "config": cfg_name, **metrics, "cot": cot, "status": "ok"}
            except Exception as exc:  # noqa: BLE001
                record = {"scene_id": scene_id, "t0_s": t0, "config": cfg_name, "status": "error", "error": repr(exc)}
            record["elapsed_s"] = round(time.time() - t_w, 2)
            records.append(record)
            out_f.write(json.dumps(record, default=str) + "\n")
            out_f.flush()
            tag_s = f"minADE/6.4s={record.get('min_ade/by_t=6.4', float('nan')):.2f}m cot={record.get('cot')!r}" if record["status"] == "ok" else record["status"]
            print(f"{scene_id} [{cfg_name}] {tag_s} ({record['elapsed_s']:.1f}s)", flush=True)

print(f"\nDone in {time.time()-t_start:.0f}s")
ok = [r for r in records if r["status"] == "ok"]
for cfg_name, *_ in CONFIGS:
    vals = [r["min_ade/by_t=6.4"] for r in ok if r["config"] == cfg_name]
    if vals:
        print(f"NO-COT {cfg_name:35s} mean={sum(vals)/len(vals):.3f}m n={len(vals)}")
non_empty_cot = sum(1 for r in ok if r.get("cot"))
print(f"non-empty cot strings among {len(ok)} ok records: {non_empty_cot} (should be 0)")
