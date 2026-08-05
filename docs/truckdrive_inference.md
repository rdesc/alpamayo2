# Running Alpamayo 2 Super on TruckDrive — working notes & results

Alpamayo 2 Super ships with no TruckDrive integration (its release path only
knows PhysicalAI-AV). This doc tracks what was built to run inference and
eval on TruckDrive, the decisions behind it, and the full results of the
ablations run so far. Companion to `alpamayo-recipes`' TruckDrive docs
(`docs/rod_inference_truckdrive.md`, `docs/truckdrive_finetuning.md`), which
cover the equivalent Alpamayo 1.5 work this was ported from.

## What's in this repo

| File | Purpose |
|---|---|
| `src/alpamayo2_super/load_truckdrive.py` | TruckDrive scene → Alpamayo 2 sample-dict loader (S3/local), pose parsing, reverse-window filtering, standstill snap, `enumerate_val_windows` (matches `TruckDriveDataset._build_index`) |
| `src/alpamayo2_super/truckdrive_viz.py` | Single-run camera-projection + BEV figure (pinhole calibration, TruckDrive's own format) |
| `src/alpamayo2_super/truckdrive_metrics.py` | ADE/FDE-by-horizon + corner distance, matching alpamayo-recipes' TruckDrive eval exactly |
| `examples/truckdrive_inference.py` | Single-scene CLI, mirrors `inference_smoke.py` |
| `examples/truckdrive_eval_val.py` | Full windowed val-split eval — sharding, `--num-frames`, `--no-cot`, `--save-predictions` |
| `examples/truckdrive_eval_merge.py` / `truckdrive_predictions_merge.py` | Merge sharded metrics / predictions.pt outputs |
| `examples/truckdrive_camera_sensitivity.py` | Camera/frame-config ablation on a fixed scene set |
| `examples/truckdrive_cot_check.py` | One-off CoT-on vs CoT-off check on a fixed scene set |
| `examples/truckdrive_model_comparison.py` | N-way qualitative comparison video renderer (Alpamayo 2 configs × Alpamayo 1.5 × baselines) |

## Data pipeline decisions

**Camera views.** Alpamayo 2's "trajectory" task profile uses 6 of its 7
camera slots (ids 0,1,2,3,5,6). TruckDrive inference here uses exactly the 5
views validated for TruckDrive on Alpamayo 1.5 (`sft_truckdrive.yaml`), not
the fuller candidate list in alpamayo-recipes' 15-view→slot heuristic table.
FRONT_TELE (id 6) has no validated TruckDrive source view and is left
unfilled — the comparison figures label it "no TruckDrive source view"
rather than substituting an unvalidated one.

| Alpamayo slot (id) | TruckDrive view |
|---|---|
| cross_left (0) | `sideward_left_front_wide` |
| front_wide (1) | `forward_center_medium` |
| cross_right (2) | `sideward_right_front_wide` |
| rear_left (3) | `rearward_left_bottom_medium` |
| rear_right (5) | `rearward_right_bottom_medium` |

**Frame spacing.** 4 history frames per camera at **0.2s spacing**
(`t0-0.6, t0-0.4, t0-0.2, t0`), matching TruckDrive's ~5Hz camera rate and
`sft_truckdrive.yaml` — *not* the 0.1s PAI-style spacing Alpamayo 2's own
convention would suggest. This was originally a bug (see ablation below):
at 0.1s spacing the 4 "frames" mostly snap to the same 1-2 real camera
frames, which looks superficially fine (4 distinct timestamps returned) but
gives the model far less real temporal separation than intended.

**Windowing** (`enumerate_val_windows`, used by the full eval) reproduces
`TruckDriveDataset._build_index()`: every `t0_stride=10` pose step (poses
~10Hz → ~1 window/second) across each scene's full valid timeline, scenes
missing any of the 5 views dropped, reversing/heavy-sideslip windows dropped
(`filter_reverse`, `reverse_angle_deg=90`, `max_reverse_fraction=0.2`,
`min_speed_mps=0.5`), and near-stationary windows' ground truth snapped to
exact standstill (`standstill_snap_mps=0.5`) — matching `sft_truckdrive.yaml`
exactly so the (window, ground-truth) pairs are comparable to the Alpamayo
1.5 numbers. This produces **2,474 windows across 138 scenes**, matching the
"2,474/139" figure reported in alpamayo-recipes' docs almost exactly (one
scene difference, likely a minor filter edge case).

**CoT can be disabled**, even though Alpamayo 2 Super's standard inference
path (`helper.create_messages`) hardcodes `components_prompt=["cot",
"traj_future"]` with no exposed switch. Building the prompt directly via
`build_conversation(..., components_prompt=["traj_future"])` genuinely
suppresses generation — verified `extra["cot"]` comes back empty, not just
unreported (`truckdrive_eval_val.py`'s `--no-cot` flag, `_prepare_model_inputs`).
Caveat: the release expert was trained conditioning on a KV-cache that always
had CoT tokens present before the trajectory, so CoT-off is outside its
validated conditioning distribution — the results below say it still works
well in practice, but it's not an officially supported mode.

## Running inference

Single scene, with a figure:

```bash
python examples/truckdrive_inference.py \
  --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
  --scene-id scene_28_1 --t0-s 10.0 \
  --save-viz outputs/truckdrive_scene_28_1.png --save-json outputs/truckdrive_scene_28_1.json
```

Full windowed val eval, sharded across 8 GPUs, with predictions saved for
the comparison tooling:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python examples/truckdrive_eval_val.py \
    --model-id "$ALPAMAYO2_SUPER_MODEL_ID" \
    --num-shards 8 --shard-index $i \
    --out outputs/truckdrive_val_eval.shard${i}.jsonl \
    --save-predictions outputs/truckdrive_preds.shard${i}.pt &
done; wait
python examples/truckdrive_eval_merge.py "outputs/truckdrive_val_eval.shard*.jsonl" --out outputs/truckdrive_val_eval_merged.jsonl
python examples/truckdrive_predictions_merge.py "outputs/truckdrive_preds.shard*.pt" --out outputs/truckdrive_preds_merged.pt
```

Key flags: `--num-frames 1` drops history to just the t0 frame; `--no-cot`
disables CoT generation; `--standstill-snap-mps -1` disables the snap;
`--t0-stride` / `--no-filter-reverse` override the windowing to diverge from
the 1.5-matching defaults if needed.

## Results: full windowed val eval (2,474 windows, 138 scenes, 0 errors each)

Four Alpamayo 2 configs, varying only history length and CoT:

### baseline (CoT-on, 4-frame @0.2s) — the release-default config

| Horizon | min_ADE | ADE (sample 0) | min_FDE | FDE (sample 0) |
|---|---|---|---|---|
| 3.0s | 0.891m (median 0.632m) | 1.460m | 2.018m | 3.921m |
| 6.0s | 2.565m (median 1.829m) | 4.710m | 6.355m | 12.231m |
| 6.4s (full) | 2.848m (median 2.039m) | 5.233m | 7.022m | 13.610m |

corner_distance: mean 6.996m, median 6.962m, p95 10.971m, max 23.004m

### 1frame (CoT-on, history dropped)

| Horizon | min_ADE | ADE | min_FDE | FDE |
|---|---|---|---|---|
| 3.0s | 0.628m (median 0.432m) | 1.222m | 1.357m | 3.349m |
| 6.0s | 1.819m (median 1.251m) | 4.311m | 4.546m | 11.826m |
| 6.4s | 2.029m (median 1.390m) | 4.836m | 5.034m | 13.255m |

corner_distance: mean 6.635m, median 6.655m, p95 10.263m, max 26.893m

### nocot_baseline (CoT-off, 4-frame)

| Horizon | min_ADE | ADE | min_FDE | FDE |
|---|---|---|---|---|
| 3.0s | 0.706m (median 0.488m) | 1.036m | 1.598m | 2.791m |
| 6.0s | 2.050m (median 1.482m) | 3.740m | 5.327m | 10.948m |
| 6.4s | 2.291m (median 1.659m) | 4.255m | 5.968m | 12.611m |

corner_distance: mean 6.713m, median 6.785m, p95 10.408m, max 17.366m

### nocot_1frame (CoT-off, history dropped) — best of the four

| Horizon | min_ADE | ADE | min_FDE | FDE |
|---|---|---|---|---|
| 3.0s | 0.474m (median 0.307m) | 0.765m | 0.967m | 2.184m |
| 6.0s | 1.304m (median 0.944m) | 3.283m | 3.388m | 10.613m |
| 6.4s | 1.465m (median 1.059m) | 3.813m | 3.844m | 12.465m |

corner_distance: mean 6.350m, median 6.477m, p95 9.796m, max 16.532m

### Summary (min_ADE @ 6.4s)

| Config | mean | median | max |
|---|---|---|---|
| baseline (CoT-on, 4-frame) | 2.848m | 2.039m | 23.798m |
| nocot_baseline (CoT-off, 4-frame) | 2.291m | 1.659m | 16.247m |
| 1frame (CoT-on) | 2.029m | 1.390m | 26.105m |
| **nocot_1frame (CoT-off)** | **1.465m** | **1.059m** | 16.468m |

**Finding: less context wins.** Dropping both history and CoT beats every
other combination on mean, median, *and* has one of the best worst-case
tails — full-scale confirmation across all 2,474 windows, not just the
smaller ablation samples below. Plausible explanation: TruckDrive has zero
CoT/language training data and looks visually different from the model's
training distribution (highway trucking rig vs. the passenger-vehicle
camera rig it was validated on), so both the generated CoT reasoning and the
extra history frames may be introducing noise rather than useful signal on
this out-of-distribution domain — removing them removes the noise.

## Ablation: camera/frame-config sensitivity

Same (scene, t0, seed), varying only camera count and temporal context,
`examples/truckdrive_camera_sensitivity.py`. First pass on 5 scenes spanning
the score distribution, mean minADE/6.4s:

| Config | 5-scene mean |
|---|---|
| 5view_4frame_0.2s (baseline) | 1.563m |
| 3view_4frame_0.2s (drop rear cameras) | 1.606m |
| front_only_4frame_0.2s | 1.917m |
| front_only_1frame | 1.909m |
| 5view_1frame (drop history) | 3.285m |
| 5view_4frame_**0.1s** (the frame-spacing bug) | 4.202m |

This 5-scene sample suggested the frame-spacing bug was catastrophic and
history was essential — driven almost entirely by one outlier scene
(`scene_28_14`, a 210°-turn scene: 1.98m at 0.2s spacing vs. **14.89m** at
0.1s). Rerun on 25 scenes (5 + 20 more spanning the ADE distribution) to
check:

| Config | 25-scene mean | median | max |
|---|---|---|---|
| 5view_4frame_0.2s (baseline) | 2.246m | 1.911m | 7.490m |
| 5view_1frame (drop history) | 2.153m | **1.287m** | 12.093m |
| 3view_4frame_0.2s | 2.284m | 1.901m | **6.629m** |
| front_only_4frame_0.2s | **2.701m** (worst mean+median) | 2.790m | 6.293m |
| front_only_1frame | 2.216m | 1.745m | 10.618m |
| 5view_4frame_0.1s (buggy spacing) | 2.669m | 1.744m | **14.889m** (worst) |

At 25 scenes, all six configs sit within a fairly narrow 2.15–2.70m mean
band (noise-level differences), and baseline actually loses to 5view_1frame
on a per-scene paired basis (10/25 vs 15/25). What *does* hold up: baseline
has the best worst-case tail (lowest max), and `front_only_4frame` is
consistently the worst performer on mean/median with no compensating
strength. The frame-spacing bug's damage is less dominant at this sample
size but still produced the single worst result of the whole study. The
*full-scale* 2,474-window results above are the ones to trust over either of
these smaller samples.

**Rear cameras contribute almost nothing** to trajectory prediction — 3-view
(front + both cross views) is statistically indistinguishable from the
5-view baseline. Makes sense for a forward-driving task.

## Ablation: CoT-on vs CoT-off, same 25 scenes

Before committing to the full 2,474-window CoT-off runs, checked on the same
25-scene set (`examples/truckdrive_cot_check.py`):

| Config | CoT-ON mean | CoT-OFF mean | CoT-ON median | CoT-OFF median | CoT-off wins |
|---|---|---|---|---|---|
| baseline (4-frame) | 2.246m | **1.885m** | 1.911m | 1.810m | 16/25 scenes |
| 1frame | 2.153m | **1.311m** | 1.287m | 0.895m | 20/25 scenes |

Consistent direction, confirmed at full scale above (see summary table).

## Cross-model qualitative comparison

`examples/truckdrive_model_comparison.py` renders one MP4 per scene (every
window in the scene, ~2fps): a tightly-tiled camera mosaic (no subplot
gaps, camera names drawn on-image — matches alpamayo-recipes'
`render_scene_video.py` layout) with GT + best-scoring model + worst-scoring
model projected on, a fixed-axis BEV panel with every model's full best-of-K
fan, and a Chain-of-Causation panel comparing each CoT-capable model's
reasoning for its own best sample.

No GPU/model loading needed at render time — it reads directly from each
model's saved `predictions.pt`. Currently compares 4 models:

| Model | Source |
|---|---|
| Alpamayo 1.5 zero-shot + CoC | `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-A1-format/eval_20260721_225240/predictions.pt` |
| Alpamayo 1.5 finetuned + CoC | `/mnt/efs/users/rod/ckpts/alpamayo_truckdrive_finetuning/alpamayo1.5_truckdrive_STAGE1_full_train_with_camids_2_epochs_checkpoint-8238/eval_20260722_213958/predictions.pt` |
| Alpamayo 2 (4-frame, CoC-on) | `outputs/truckdrive_zero_shot_eval_results/truckdrive_preds_baseline_preds_merged.pt` |
| Alpamayo 2 (4-frame, CoC-off) | `outputs/truckdrive_zero_shot_eval_results/truckdrive_preds_nocot_baseline_preds_merged.pt` |

18 scenes rendered so far in `outputs/truckdrive_comparison/`, picked from
`TruckDrive/truckdrive_val_ade_selection.json` (ADE-difficulty bins) and
`truckdrive_val_interesting.json` (turn-angle ranking) — spanning the
easy/standstill case, the median-difficulty case, all 5 ADE-difficulty bins,
and several large-turn maneuvers (up to a ~210° roundabout).

Notable qualitative findings from these videos:
- On the hardest-turn scene (`scene_28_24`, 210° roundabout), Alpamayo 2
  (1.99m) clearly beats both Alpamayo 1.5 variants (5.91m, 7.03m) — the
  advantage seen in the aggregate numbers is concentrated in genuinely hard
  geometry, not uniform.
- On a standstill window, Alpamayo 2 (CoC-on) generated "Stop due to red
  traffic light ahead" — no traffic light is visible in any camera tile.
  The trajectory itself was correct (0.00m ADE); the stated reasoning looks
  hallucinated. Worth a closer look if auditing CoC reliability specifically,
  not just trajectory accuracy.
- One CoT string observed with Chinese characters mixed into otherwise-English
  reasoning (harmless — just triggers a missing-glyph warning when rendering
  with a Latin-only font) — a minor curiosity about the release checkpoint's
  text generation, not investigated further.

## Other saved predictions.pt (for reference / comparison)

All confirmed to cover the exact same 2,474-window / 138-scene set as the
Alpamayo 2 runs above (verified via `(scene_id, t0_us)` key overlap):

| Model | Path | Notes |
|---|---|---|
| Alpamayo 1.5 zero-shot, CoC-off | `.../Alpamayo-1.5-10B-A1-format/eval_20260721_201032/predictions.pt` | |
| Alpamayo 1.5 zero-shot, CoC-on | `.../Alpamayo-1.5-10B-A1-format/eval_20260721_225240/predictions.pt` | used above |
| Alpamayo 1.5 Stage-1 finetuned, no CoT field | `.../alpamayo1.5_truckdrive_STAGE1_full_train_with_camids_2_epochs_checkpoint-8238/eval/predictions.pt` | older eval, predates CoT saving |
| Alpamayo 1.5 Stage-1 finetuned, CoC-on | `.../alpamayo1.5_truckdrive_STAGE1_full_train_with_camids_2_epochs_checkpoint-8238/eval_20260722_213958/predictions.pt` | used above |
| Alpamayo 1.5 Stage-2 (diffusion expert) | `/mnt/efs/users/rod/truckdrive_cache/stage2_expert2_predictions_jul27.pt` | architecturally has no CoT at all (conditions on VLM KV-cache truncated before any reasoning text) |
| MLP baseline (ego-history only, no camera) | `alpamayo-recipes/recipes/alpamayo1_5_sft/output_mlp_baseline/predictions.pt` | K=1 deterministic, ADE@6.4s=1.42m — genuinely competitive despite no vision input |
| Constant-velocity baseline | `alpamayo-recipes/recipes/alpamayo1_5_sft/output_cv_baseline/predictions.pt` | K=1, ADE@6.4s=2.27m, the "is the model doing anything" floor |

Other checkpoint directories exist under
`/mnt/efs/users/rod/ckpts/alpamayo_truckdrive_finetuning/` (Stage-2
variants, an intermediate 1-epoch/t0_stride-30 run, a `checkpoint-4119` with
full weights still present) — not all have been used in the comparisons
above; check for an `eval*/predictions.pt` subdirectory before assuming a
fresh eval run is needed.

## Open items

- Only 4 of the 5 known models/configs are in the qualitative comparison
  tool today (Alpamayo 1.5 Stage-2 and the MLP/CV baselines were excluded
  from the video renderer at the user's request, but their predictions.pt
  files are ready to add back — `MODELS` list in
  `truckdrive_model_comparison.py`, `best_worst_sample` already handles the
  K=1 deterministic-baseline case).
- No CoT-off equivalent exists for `1frame` in the comparison videos (only
  `nocot_1frame`'s *metrics* were computed at full scale, not projected in
  this tool) — trivial to add given the predictions.pt already exists.
- The frame-spacing ablation's full-scale run (`5view_4frame_0.1s`) was
  never re-verified at the full 2,474-window level, only at 5- and 25-scene
  samples — the smaller samples disagree with each other on how bad it is.
