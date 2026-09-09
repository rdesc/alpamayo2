# PAI-AV OOD eval for Alpamayo 2 Super — how it was set up

Open-loop trajectory metrics + reasoning scores for Alpamayo 2 Super on the
PhysicalAI-AV (PAI-AV) long-tail **OOD reasoning** val split, reproducing the
protocol previously run for Alpamayo 1.5 and R1 so all three models are directly
comparable.

The 1.5/R1 side lives in `/mnt/efs/users/rod/repos/alpamayo1.5` and is documented
there in `docs/pai-av-ood-eval.md` (dataset semantics, split counts, leakage
caveat, metric definitions). **Read that doc for the "what/why" of the benchmark**;
this one covers only what was built here for Alpamayo 2 and the decisions specific
to it.

## What was added to this repo

| File | Purpose |
|---|---|
| `examples/eval_common.py` | Verbatim copy of the 1.5 repo's `eval_common.py` — event loader, minADE/medoid metrics, arg parsing, sharding, parquet writer. Deliberately torch-free. Keep in sync with the original. |
| `examples/eval_pai_av_val_a2.py` | The Alpamayo 2 Super eval. Thin: dataset → task profile → prompt → `sample_trajectories_from_data` → `ec.build_pred_record`. |
| `examples/run_eval_multigpu.py` | Copy of the 1.5 launcher, default `--eval_script` repointed at the A2 script. Shards run under `sys.executable`, so launch it with this repo's `.venv/bin/python`. |

Copying `eval_common.py` rather than importing across repos is deliberate: the two
repos have incompatible venvs, and an identical output schema is what lets the 1.5
repo's downstream tooling read A2 parquets unchanged.

## Protocol (identical to 1.5/R1 unless noted)

- **Events**: `reasoning/ood_reasoning.parquet`, `split=val` → 289 clips / **349
  events**. One prediction per annotated event (not per clip); `t0` floored to
  `MIN_T0_US = 1.7s`.
- **Window**: 1.6 s history (16 steps) + 6.4 s future (64 steps) @ 10 Hz.
- **Sampling**: K=6, `temperature=0.6`, `top_p=0.98`, expert diffusion
  `inference_step=10`.
- **Metrics**: minADE_K at 1–6 s + ADE of the **medoid** ("most-likely") sample.
  Raw `pred_xy`/`gt_xy` stored so horizons are re-derivable offline.
- **No route/nav conditioning** — the dataset has no such feature.

### ⚠️ The one real difference from the 1.5/R1 numbers: camera count

Alpamayo 2's canonical source ring is **7 cameras**, but no task consumes all 7.
The released per-task input profiles are **6 cameras × 4 frames**
(`src/alpamayo2_super/input_profiles.py`), which the README describes as the
model's training contract:

- trajectory / meta_action / auto_labeling / grounding → camera ids `[0,1,2,3,5,6]`
  (the ring minus **rear_tele**, id 4)
- vqa → `[0,1,2,3,4,5]`

So the eval loads the 7-cam ring via `load_physical_aiavdataset` and then calls
`select_task_input(source, "trajectory")` → **24 images**. Alpamayo 1.5/R1 saw
**4 cameras × 4 frames = 16 images**. Running A2 on the 1.5 camera set was
considered and rejected: it would put the model outside its validated profile.
**Flag this when comparing minADE across the three models** — it is not a pure
model-vs-model comparison.

Frame spacing stays at the loader default 0.1 s (`t0-0.3 … t0`), matching 1.5/R1.
(The 0.2 s spacing used in this repo's TruckDrive work is TruckDrive-specific and
does not apply here.)

### CoC ablation

`--no_coc` runs the no-reasoning arm. Alpamayo 2 supports this cleanly, unlike the
older models: `helper.create_messages` hardcodes
`components_prompt=["cot", "traj_future"]`, so `eval_pai_av_val_a2.py` has a local
`_prepare_model_inputs` that builds the conversation with `["traj_future"]` only.
Verified: `pred_coc_ml` comes back empty, i.e. generation is genuinely suppressed,
not merely unreported.

This is a cleaner ablation than either predecessor — 1.5 injected `coc_text=""`,
and R1 needed a subclass that also skipped its `<meta_action>` segment. **Caveat**:
the released expert was trained conditioning on a KV-cache that always had CoT
tokens before the trajectory, so CoT-off is outside its validated conditioning
distribution.

## Running it

```bash
cd /mnt/efs/users/rod/repos/alpamayo2 && source env.sh
mkdir -p outputs/pai_av_val/logs

# CoC arm
.venv/bin/python examples/run_eval_multigpu.py \
  --gpus 0,1,2,3,4,5,6,7 --num_traj_samples 6 --split val \
  --logdir outputs/pai_av_val/logs/a2_coc_shards \
  --out outputs/pai_av_val/a2_coc.parquet

# no-CoC arm: same, plus --no_coc, --out .../a2_nococ.parquet
```

Useful flags (from `eval_common.add_common_args`): `--limit N` (smoke test),
`--first_event_only`, `--split train|val|both`, `--num_traj_samples`.
A2-specific: `--model_id` (defaults to `$ALPAMAYO2_SUPER_MODEL_ID`),
`--diffusion_steps`, `--task_profile`. `--attn_implementation` is accepted for
signature compatibility with the 1.5 scripts but is unused by A2.

Note: per-shard logs are block-buffered, so they stay empty for the first several
minutes of a run. Check `nvidia-smi` (each shard holds ~70 GB) rather than the logs
to confirm progress early on.

## Downstream scoring — all 1.5-repo tools work unchanged

Verified against A2 parquets:

```bash
cd /mnt/efs/users/rod/repos/alpamayo1.5

# Reasoning grades: lingo [0,1] + rubric 0-5 (VL judge Cosmos-Reason2-8B).
# MUST use the R1 venv -- a1_5_venv has no vLLM and silently drops the rubric grader.
/mnt/efs/users/rod/repos/alpamayo/ar1_venv/bin/python score_reasoning_multigpu.py \
  /mnt/efs/users/rod/repos/alpamayo2/outputs/pai_av_val/a2_coc.parquet \
  --gpus 0,1,2,3,4,5,6,7 --graders lingo,rubric

# CoC-action consistency (Alpamayo-R1 Sec 5.3.2), offline/CPU
./a1_5_venv/bin/python coc_action_consistency.py <parquet> --gt --out <parquet>.cac.parquet

# Cross-model comparison table / offline horizon re-derivation
./a1_5_venv/bin/python parse_results.py a2_coc=<parquet> a15_coc=... r1_coc=...
./a1_5_venv/bin/python compute_metrics.py <parquet>
```

Two gotchas worth knowing:

- **CAC needs no A2 changes.** A2's `extra` carries text only, so the parquet has
  no `pred_action`/`pred_accel_ml`/`pred_curvature_ml` columns that the 1.5 runs
  had. This turns out not to matter: `coc_action_consistency.py` derives
  meta-actions from `pred_xy` + `medoid_idx` by finite differences
  (`_reshape_pred`, `trajectory_meta_action`), never from `pred_action`. If exact
  native controls are ever wanted, the expert's `sampled_action` (accel, curvature;
  `models/alpamayo2_super.py`, `UnicycleAccelCurvatureActionSpace`) would need to
  be surfaced in `extra`.
- **Single-GPU scoring prints a false alarm.** With `--gpus <one>`,
  `score_reasoning_multigpu.py` ends with "No shard outputs found to merge" because
  the lone shard writes straight to `--out`. The output file is correct. The
  multi-GPU path merges normally.

## Validation performed

- CoT-on and `--no_coc` single-GPU smoke runs (`--limit 2 --num_traj_samples 2`):
  both succeeded, CoC text populated / empty respectively.
- 2-GPU shard + merge via `run_eval_multigpu.py` (`--limit 4`): 4/4 rows merged.
- `coc_action_consistency.py --gt` on an A2 parquet: parsed 2/2, GT ceiling 1.000.
- `score_reasoning_multigpu.py --graders lingo` on an A2 parquet: 2/2 gradable,
  `lingo_score` written.
- The **rubric** grader was not smoke-tested (needs a vLLM Cosmos-Reason2-8B load);
  it reads the same columns as lingo, so it is expected to work, but that is
  inference from the schema rather than a verified run.

## ⚠️ Memory: K=6 does not fit in one forward pass

The first full run (2026-08-06 17:15) **failed on all 349 events with CUDA OOM**.
Measured on an 80 GB H100 (single event, ~6 GB co-tenant job on the card):

| K per call | result |
|---|---|
| 3 | fits |
| 4 | OOM (~73 GB peak) |
| 6 | OOM (~73 GB peak) |

Smoke tests had all used K=2, which is why this was not caught earlier — **validate
memory at the K the real run uses, not just the plumbing.**

Fix: `--sample_chunk_size` (default **3**). K is drawn in sequential chunks of that
size and concatenated along the sample axis before any metric is computed, so peak
memory is set by the largest single call while the output is a genuine K-sample set
(`n_traj=6`, 6 CoC traces, medoid over all 6). Statistically equivalent to one
K-sample call — the samples are i.i.d. given fixed conditioning — but **not
bit-identical**, since the RNG stream advances differently. Cost: the prompt is
prefilled once per chunk; in practice the rate is unchanged (~33 s/event).

Headroom is thin: this fits alongside a ~6 GB co-tenant, and would not fit at K=4.

## Results — val split, 2026-08-06

349/349 events succeeded in both arms, K=6, 8 GPUs, ~26 min per arm.

### Open-loop trajectory error (mean over 349 events)

| Horizon | minADE_6 (CoC) | ADE medoid (CoC) | minADE_6 (no-CoC) | ADE medoid (no-CoC) |
|---|---|---|---|---|
| 1s | 0.025 | 0.048 | 0.024 | 0.048 |
| 3s | 0.232 | 0.454 | 0.244 | 0.424 |
| 6s | 0.929 | 1.797 | 0.915 | 1.607 |

Medians are far lower (0.595 m for minADE_6 @6s), so the means are driven by a
right tail of hard clips — prefer the median or a per-cluster view for model
comparisons.

### CoC vs no-CoC, paired by event (n=349, CoC minus no-CoC, negative = CoC better)

| metric | delta | 95% CI | t |
|---|---|---|---|
| min_ade_3s | −0.012 | ±0.019 | −1.22 |
| min_ade_6s | +0.014 | ±0.081 | +0.34 |
| ade_ml_3s | +0.030 | ±0.028 | +2.07 |
| ade_ml_6s | **+0.190** | ±0.105 | **+3.54** |

**Generating CoC does not improve trajectory accuracy on this set, and measurably
hurts the medoid trajectory** (+0.19 m at 6 s, significant). minADE_6 is unaffected,
i.e. reasoning does not change the quality of the best-of-6 but does shift where the
consensus mode lands. Caveat: CoT-off is outside the released expert's validated
conditioning distribution, so this is not a clean "value of reasoning" measurement.

### Cross-model (val, CoC arm, all n=349)

| model | minADE_6 | ADE medoid @6s | lingo | rubric (0–5) | CAC |
|---|---|---|---|---|---|
| Alpamayo 1.5 | **0.853** | **1.633** | **0.706** | 3.166 | 0.344 |
| Alpamayo R1 | 0.902 | 1.695 | 0.528 | 3.095 | 0.381 |
| Alpamayo 2 Super | 0.929 | 1.797 | 0.524 | 3.132 | **0.438** |

**Alpamayo 2 is not better than 1.5 on this benchmark** — it is slightly worse on
both trajectory metrics, despite seeing 6 cameras to 1.5's 4. It leads on CoC-action
consistency (0.438) and is level with R1 on reasoning grades; 1.5's much higher
lingo score (0.706) is worth treating with suspicion given lingo is a text-similarity
proxy and the three models have different CoC styles.

Reference points: CAC is far below the R1 paper's 0.62 (SFT) / 0.85 (+RL), but read
it against the **GT-vs-gold-CoC ceiling of 0.825** measured here, not against 1.0.
The rubric grades cluster near 3/5 for all three models.

**⚠️ CAC column above uses the superseded B0 scorer.** See "CAC v2 (Qwen extractor) —
cross-model" further down for the updated, cross-model-comparable numbers (val: 1.5 0.587,
R1 0.585, A2 **0.607**) — the ranking direction is the same (A2 leads) but the absolute
values are not comparable across the two scorer generations.

All the leakage caveats from the 1.5 repo's doc apply — none of these numbers are
verified clean of the released checkpoints' training data.

## Results — train split, 2026-08-06/07

1,728 events attempted in both arms (K=6, 8 GPUs). Initial run: **CoC 1,717/1,728
succeeded (167.6 min), no-CoC 1,657/1,728 (141.4 min)** — the no-CoC arm lost far
more events (71 vs. 11) to CUDA OOM from a co-tenant process sharing the GPUs plus
allocator fragmentation eating into the thin K=6 headroom documented above, not
anything about the no-CoC config itself. Of each arm's failures, 6 were a separate,
permanent pose-interpolation-range error (`Interpolation times must be within the
range [...]`, a data issue independent of GPU memory or the CoC flag) — the rest
were pure OOM.

**Retried the OOM failures only** (`eval_common.py` gained a `--retry_failed <prior
output>` flag for this: reruns just the `ok=False` rows whose error was CUDA OOM,
skipping the deterministic interpolation failures) on an idle window of the same
8-GPU box: **5/5 CoC and 65/65 no-CoC recovered**, merged back into the original
parquets (originals backed up as `train_{coc,nococ}.pre_retry_backup.parquet`).
Both arms now stand at a matched **1,722/1,728** — the remaining 6/arm are the
permanent interpolation failures, identical clip/event pairs in both arms.

### Open-loop trajectory error (mean / median over succeeded events, meters), n=1722 both arms

| Horizon | minADE_6 (CoC) | ADE medoid (CoC) | minADE_6 (no-CoC) | ADE medoid (no-CoC) |
|---|---|---|---|---|
| 1s | 0.024 / 0.014 | 0.046 / 0.032 | 0.025 / 0.015 | 0.047 / 0.031 |
| 2s | 0.094 / 0.057 | 0.182 / 0.133 | 0.098 / 0.059 | 0.186 / 0.130 |
| 3s | 0.211 / 0.132 | 0.404 / 0.291 | 0.218 / 0.136 | 0.414 / 0.289 |
| 4s | 0.368 / 0.234 | 0.704 / 0.504 | 0.380 / 0.233 | 0.727 / 0.500 |
| 5s | 0.571 / 0.363 | 1.079 / 0.771 | 0.581 / 0.357 | 1.119 / 0.754 |
| 6s | 0.819 / 0.521 | 1.531 / 1.090 | 0.821 / 0.504 | 1.588 / 1.043 |

Compare to val (349 events): minADE_6@6s 0.929 (CoC) / 0.915 (no-CoC), ADE medoid@6s
1.797 / 1.607. **Train numbers are meaningfully better than val on every horizon**
(e.g. minADE_6@6s 0.819 vs. 0.929 CoC) — expected, since train is not held out from
the released checkpoint's training data (see the leakage caveat inherited from the
1.5 repo's doc), unlike val's long-tail OOD framing.

### CoC vs no-CoC, paired by event (n=1722, CoC minus no-CoC, negative = CoC better)

| metric | delta | 95% CI | t |
|---|---|---|---|
| min_ade_3s | −0.007 | ±0.008 | −1.64 |
| min_ade_6s | −0.002 | ±0.035 | −0.12 |
| ade_ml_3s | −0.010 | ±0.013 | −1.58 |
| ade_ml_6s | −0.057 | ±0.052 | −2.12 |

Unlike val (where CoC measurably **hurt** the medoid trajectory at 6s, +0.190m,
t=+3.54), on train the sign flips: CoC now **significantly helps** the medoid at 6s
(−0.057m, t=−2.12, |t|>1.96). The other three metrics are not significant. Same
caveat as val: CoT-off is outside the released expert's validated conditioning
distribution, so this isn't a clean "value of reasoning" measurement.

### Per-event-cluster (mean minADE@6s / ADE_ml@6s, meters), CoC arm, n=1722

| cluster | min_ade_6s | ade_ml_6s | count |
|---|---|---|---|
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 0.745 | 1.439 | 371 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 0.765 | 1.471 | 233 |
| OTHER_LONGTAIL | 0.775 | 1.717 | 29 |
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 0.792 | 1.438 | 887 |
| ANIMALS_BIRDS_ROADKILL | 0.864 | 1.640 | 28 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 1.067 | 1.895 | 88 |
| EMERGENCY_INCIDENT_SCENE | 1.169 | 2.148 | 28 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 1.172 | 2.338 | 17 |
| COMPLEX_INTERSECTION_INTERACTION | 1.462 | 2.979 | 41 |

Cluster ranking is consistent with val (pedestrian-density/work-zone easiest,
complex-intersection hardest), and `WORK_ZONES_TEMP_TRAFFIC_CONTROL` dominates train
by count (887/1722, ~52%) in a way it doesn't on val — worth keeping in mind before
reading the train overall mean as cluster-balanced.

### Reasoning scores (CoC arm, n=1722), 2026-08-07

| metric | train | val (349) |
|---|---|---|
| lingo | 0.525 (median 0.486) | 0.524 |
| rubric (0–5) | **3.371 (median 3.000)** | 3.132 |
| CAC | **0.520** | 0.438 |
| GT-vs-gold-CoC ceiling | 0.794 | 0.825 |
| reasoning parsed | 0.998 (1719/1722) | — |

Rubric per-cluster (mean, n=1722): ROAD_DEBRIS_OR_SAFETY_TRACES 2.647 (17),
COMPLEX_INTERSECTION_INTERACTION 2.390 (41), CYCLISTS_AND_MICROMOBILITY_COMPLEX 2.955
(88), OTHER_LONGTAIL 3.103 (29), EMERGENCY_INCIDENT_SCENE 3.250 (28),
WORK_ZONES_TEMP_TRAFFIC_CONTROL 3.281 (887), SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR 3.433
(233), PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY 3.774 (371), ANIMALS_BIRDS_ROADKILL 3.964
(28) — roughly tracks the trajectory-error ranking (complex-intersection/road-debris
worst, pedestrian-density/animals best), unlike CAC's ranking above, which diverges
from it. Train rubric (3.371) edges out val (3.132), same direction as CAC and
consistent with the same data-leakage read.

Lingo is essentially flat train-vs-val despite train's much better trajectory
metrics — consistent with the val doc's caveat that lingo is a text-similarity proxy,
not a trajectory-quality measure. **CAC is notably higher on train (0.520 vs. 0.438)**
— plausible given train clips aren't held out from the checkpoint's training data
(same leakage caveat as the trajectory numbers above), so the model's stated reasoning
and its actual trajectory cohere better on data it was likely trained on.

Per-cluster CAC (mean consistency, count): PEDESTRIAN_DENSITY 0.668 (371), OTHER_LONGTAIL
0.621 (29), EMERGENCY_INCIDENT_SCENE 0.571 (28), SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR
0.528 (233), ANIMALS_BIRDS_ROADKILL 0.500 (28), COMPLEX_INTERSECTION_INTERACTION 0.488
(41), CYCLISTS_AND_MICROMOBILITY_COMPLEX 0.477 (88), WORK_ZONES_TEMP_TRAFFIC_CONTROL
0.460 (887), ROAD_DEBRIS_OR_SAFETY_TRACES 0.353 (17) — a different ranking from the
trajectory-error cluster ranking above (e.g. work-zones is easiest for trajectory but
mid-pack for CAC), i.e. "predicts the right motion" and "reasoning matches the motion"
aren't the same axis.

**Rubric grader initially failed outright on this box, then was fixed and re-run
successfully here** (see the "sm_103a workaround" subsection below for the full
story and the reusable fix).

**Gotcha hit retrying on this box:** the eval's dataset-event parquet
(`/home/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet`, the small
metadata file listing clips/events/splits — not the imagery) is staged per-instance
via `stage-dataset pai_av_dataset` and isn't guaranteed present on whichever node the
Bash tool happens to be attached to. The original train run and this retry ran on
different physical boxes (confirmed by GPU memory capacity in error messages: 80GB
H100 vs. this box's ~275GB cards) — the retry crashed immediately with
`FileNotFoundError` until staged locally. Separately, `eval_common.run_eval` crashed
on `res_df["ok"]` when a shard was assigned zero retry rows (only 5 total CoC events
across 8 shards) — fixed with a length guard.

### sm_103a (B300) workaround for the rubric grader — reusable infra fix, 2026-08-07

The rubric grader (`score_reasoning_multigpu.py --graders rubric`, Cosmos-Reason2-8B
via vLLM) initially crashed outright on this box's GPUs (**B300, compute capability
sm_103a**) using the shared `ar1_venv` (torch 2.8.0+cu128, triton 3.4.0, vllm 0.11.0).
Root-caused **two independent, unrelated bugs** in that exact package build — neither
about the train data, the retry, or GPU memory:

1. **Triton's PTX-version lookup has no branch for CUDA major 13.** Triton resolves a
   PTX ISA version from the *installed* `ptxas`'s reported CUDA release
   (`triton/backends/nvidia/compiler.py::ptx_get_version`), via a hardcoded table that
   only covers major versions 10/11/12 — CUDA 13.x (needed for sm_103a codegen) hits
   its `else: raise RuntimeError(...)` branch. This box happens to also have a CUDA
   13.0 toolkit at `/mnt/efs/users/rod/cuda-13.0` whose `ptxas` *does* support
   `--gpu-name=sm_103a` (verified with a standalone `nvcc` compile) — the bug is
   purely in Triton's version-string parsing, not a missing capability.
2. **vLLM's flash-attn availability check doesn't verify SM-arch coverage.**
   vLLM already has a Blackwell-safe fallback
   (`CudaPlatform.get_vit_attn_backend`: `if has_device_capability(100): return
   TORCH_SDPA`), and it correctly returns `TORCH_SDPA` for this GPU in isolation.
   But `Qwen3-VL`'s vision tower then unconditionally re-checks
   `check_upstream_fa_availability()` (`vllm/attention/layer.py`), which only tests
   whether the `flash_attn` **package imports** — not whether its *compiled* kernels
   cover this SM arch — and silently overrides the correct choice back to
   `FLASH_ATTN`, which then dies with `CUDA error: no kernel image is available for
   execution on the device` (that `flash_attn` wheel was built without sm_103a in its
   target arch list).

**Fix, applied without touching the shared `ar1_venv`:** a `sitecustomize.py`
(auto-imported by every Python process on that interpreter, injected via
`PYTHONPATH`) monkeypatches both functions —
`triton.backends.nvidia.compiler.ptx_get_version` clamps any CUDA major ≥13 down to
`"12.9"` before the real lookup (only affects the PTX-ISA-version selection, not which
`ptxas` binary runs); `vllm.attention.layer.check_upstream_fa_availability` and the
name imported into `vllm.model_executor.models.qwen3_vl` are both forced to return
`False`, so vLLM's own Blackwell fallback sticks. Combined with `TRITON_PTXAS_PATH`
pointed at the CUDA 13.0 `ptxas`, this is fully reversible and touches no installed
package files:

```bash
export TRITON_PTXAS_PATH=/mnt/efs/users/rod/cuda-13.0/bin/ptxas
export PYTHONPATH=/mnt/efs/users/rod/repos/alpamayo2/outputs/pai_av_train/sm103a_patch
```

(patch file: `outputs/pai_av_train/sm103a_patch/sitecustomize.py`; launcher used for
the real run: `outputs/pai_av_train/score_rubric_b300.sh`)

Verified with a `--limit 2` smoke test (rubric scored 3.5/3.5) before the full
1,722-event run, which completed cleanly in 67.3 min across 8 shards with no scoring
errors (a benign vLLM usage-telemetry-thread `JSONDecodeError` appears 3x/shard in the
logs — unrelated to grading, harmless, present even in the smoke test).

**Caveat:** the `TRITON_MOCK_PTX_VERSION` env var looks like it should offer the same
fix without monkeypatching, but doesn't — it only affects a different, unused code
path (`get_ptxas_version()`), not the one actually called
(`get_ptx_version_from_options` → `get_ptxas().version`, populated at import time via
`NvidiaTool.from_path`). Confirmed by testing it directly; it still raised.

This should keep working for any future eval/scoring run on this box class (B300)
using `ar1_venv` as-is — reuse the two env vars above rather than re-deriving the fix.

## Status

- val {CoC, no-CoC} + reasoning + CAC: **done** (2026-08-06), outputs in
  `outputs/pai_av_val/`.
- train split (1,728 events) {CoC, no-CoC} open-loop trajectory metrics: **done**
  (2026-08-06/07, OOM stragglers retried and merged same day), outputs in
  `outputs/pai_av_train/`, both arms at a matched 1,722/1,728 (6/arm permanent
  interpolation failures).
- train split reasoning scoring (CoC arm): **lingo + rubric + CAC all done**
  (2026-08-07) — rubric initially blocked by a Triton/vLLM sm_103a incompatibility on
  this box's B300 GPUs, fixed with a non-invasive runtime patch (see the "sm_103a
  workaround" subsection above) and re-run successfully, n=1722 for all three.

## CAC v2 — rescored with the updated consistency scorer, 2026-08-18

The CAC numbers above (0.438 val / 0.520 train) used `alpamayo1.5/coc_action_consistency.py`
("B0" in the `alpamayo-coc-autolabeler` design docs) — a from-scratch reconstruction of the
Alpamayo-R1 §5.3.2 metric with a fixed-3s-window kinematic classifier. That design was
superseded in `alpamayo-coc-autolabeler` (segmented meta-action trajectory classifier +
boundary-tolerant matching, E3×M5-abstain, validated to 0.933-0.945 gold agreement on the
2077-event PAI-AV OOD gold corpus — see that repo's `docs/cac_scorer_design_results.md`) and
ported into `alpamayo-recipes` as the actual CoC-action-consistency **RL reward**
(`recipes/alpamayo1_x_rl/rewards/coc_action_consistency_*.py`,
`compute_component`/`compute_reward`, gated behind `coc_consistency_weight` in
`aggregated_reward_with_reasoning.py`).

This section rescores the same four eval parquets against that updated scorer, so the number
reported here is directly comparable to what the RL reward would assign these same rollouts.

**Method:** `examples/score_cac_v2.py` (this repo) calls
`alpamayo1_x_rl.rewards.coc_action_consistency_reward.compute_component` on each event's
medoid ("most-likely") predicted trajectory and medoid CoC (`pred_coc_ml`) — the same pairing
Sec 5.3.2 grades. Uses the regex fallback extractor (`extract_claims_regex`), which is also
`compute_component`'s own default when no Qwen extractor is configured — the same fallback the
`50chunks` RL TOML runs with (see below). Output: `<name>.cac_v2.parquet` next to each input
parquet, with `cac2_binary`/`cac2_graded`/`cac2_reward`/`cac2_abstained`/`cac2_parsed` columns.

**Caveat — heading is approximated.** These parquets only ever stored `pred_xy` (K,T,2); the
eval script never captured `pred_rot`, which the new trajectory-side classifier
(`coc_action_consistency_trajectory.py`) needs to project velocity onto a heading axis. Lacking
true predicted orientation, heading was reconstructed from the predicted path's own tangent
(central-difference `atan2(vy, vx)`). This is exact for lateral (turn-rate) classification —
heading-change-rate doesn't care where the heading number came from — but makes the
longitudinal `reverse` bucket structurally unreachable (speed-along-heading becomes
tautologically the speed magnitude, never negative, when heading is derived from velocity
direction itself). Reverse is rare in the validated gold corpus (~7/2077 events), so the
aggregate effect should be small, but this is a real, uncorrected deviation from the validated
design, not a rounding difference. Re-running inference to capture true `pred_rot` would remove
it.

**Finding — the abstain mechanism never fired on real model CoC.** M5-abstain declines to
score a lateral claim whose own text hedges the direction ("steer *slightly* left"),
empirically calibrated on gold CoC (0.512 vs 0.952 gold agreement, hedged vs unhedged). Across
all 2071 scored events here, `cac2_abstained` is `False` throughout. Checked by hand: Alpamayo
2's CoC phrases hedges on the *deceleration* clause ("decelerate **slightly** to create a gap
for a left lane change"), not the lateral verb itself, and the `_CUT` clause-truncation rule
(decision clause vs. justification) separates the hedge word from "left" before magnitude gets
attached to the lateral claim — so the trigger condition essentially never matches this
phrasing style. This is the "model-generated CoC is more discursive; re-measure before trusting
gold-corpus abstention rates" caveat the autolabeler docs already flagged, now confirmed on
real Alpamayo 2 output rather than assumed.

### Results

| split | n | unparsed | abstained | binary (unparsed=0) | graded/reward (unparsed=0) | binary (parsed only) |
|---|---|---|---|---|---|---|
| val CoC | 349 | 13 | 0 | **0.599** | 0.600 | 0.622 (n=336) |
| val no-CoC | 349 | 349 | 0 | 0.000 | 0.000 | — |
| train CoC | 1722 | 65 | 0 | **0.657** | 0.661 | 0.683 (n=1657) |
| train no-CoC | 1722 | 1722 | 0 | 0.000 | 0.000 | — |

No-CoC arms score exactly 0 as designed: `pred_coc_ml` is empty there, so every event is
"unparseable" claims-wise per Sec 5.3.2's rule — included for completeness, not a new finding.

**Vs. the old B0 scorer:** val 0.438 → **0.599**, train 0.520 → **0.657**. A large jump,
consistent with the autolabeler repo's own earlier finding that the trajectory-side classifier
(not the CoC parser) was the dominant source of scorer error, not model quality — most of this
delta is the scorer getting better at reading the same trajectories/CoC, not the model actually
reasoning more consistently.

Reference ceiling (not measured on these rollouts): the autolabeler repo's own gold-vs-gold
agreement for this scorer family is 0.933-0.945 — a different measurement (real trajectory vs.
auto-labeled gold CoC, not a model's own trajectory vs. its own generated CoC), included here
only as an orientation point, not a target these numbers should be expected to approach.

## CAC v2 — Qwen in-loop extractor vs. regex fallback, 2026-08-18

Rescored the two CoC-arm parquets (val, train) with `QwenClaimExtractor`
(`coc_action_consistency_extract.py`'s P3 prompt, `Qwen/Qwen3-VL-8B-Instruct`,
greedy decode) instead of the regex fallback the section above used — the same
extractor the offline validation actually recommends (0.941 gold agreement vs. the
regex path's 0.932) and the one `alpamayo_rvla_rl_coc_consistency_qwen_mini.toml` /
`alpamayo_rvla_rl_local_test_coc_consistency_qwen.toml` wire up for real RL runs.
No-CoC arms were not rerun — they score exactly 0 regardless of extractor, since
`pred_coc_ml` is empty.

**Method:** `examples/score_cac_v2.py --qwen` (this repo) — extended to batch
`extract_batch()` calls (batch size 16) ahead of the existing per-row trajectory
classification, then feeds precomputed `claims` into the same `compute_component`
call the regex path used. Same heading-approximation caveat as above (no `pred_rot`
in these parquets). Model checkpoint:
`/mnt/efs/users/rod/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`,
single B300 GPU per arm (val on GPU 4, train on GPU 5, ~4 min and ~13 min
respectively). Smoke-tested on 4 val rows before the full run.

### Results

| split | n | unparsed | abstained | scored | binary (unparsed=0, abstained excluded) | graded (same) |
|---|---|---|---|---|---|---|
| val CoC | 349 | 0 | 0 | 349 | **0.607** | 0.615 |
| train CoC | 1722 | 0 | 3 | 1719 | **0.638** | 0.641 |

### Regex vs. Qwen extractor, side by side (both: unparsed=0, abstained excluded)

| split | regex binary | Qwen binary | regex unparsed | Qwen unparsed |
|---|---|---|---|---|
| val | 0.599 | 0.607 | 13/349 | 0/349 |
| train | 0.657 | 0.638 | 65/1722 | 0/1722 |

Two findings:

- **Qwen essentially eliminates unparsed events** (0 vs. 13 on val, 0 vs. 65 on
  train) — it reads phrasings the regex extractor's fixed patterns miss (this is
  the same gap the regex module's own `_ADAPT_SPEED`/`_LEAD_DECEL_JUSTIFICATION`
  patches were added to close piecemeal, from real rollout text; Qwen doesn't need
  those patches to begin with).
- **The score itself does not consistently favor Qwen.** It's slightly higher on
  val (0.607 vs. 0.599) but *lower* on train (0.638 vs. 0.657) — a reversal on
  train that the offline gold-corpus validation's small, consistent Qwen edge
  (0.918 vs. 0.913) did not predict. Worth treating as a real discrepancy on
  model-generated CoC rather than assuming Qwen's offline edge transfers directly:
  the two extractors are reading the same text differently, not just failing on
  different lines, and it's not yet established which one is right where they
  disagree. Also new here: train's Qwen run produced 3 abstained events (M5-abstain
  fired) vs. 0 under regex on either split — plausible given Qwen's claims/wording
  differ from what the regex hedge-detection rule was tuned against, but not
  investigated further.

**Decision: use the Qwen-extractor number as the reported CAC v2 figure going forward** — it's
the extractor the offline validation actually recommends, it eliminates the unparsed-event gap
entirely, and the regex path was only ever a no-GPU fallback (see `score_cac_v2.py --qwen`
above).

## CAC v2 (Qwen extractor) — cross-model, 2026-08-18

Ran the same `score_cac_v2.py --qwen` scorer against Alpamayo 1.5's and R1's own existing
inference parquets in `~/repos/alpamayo1.5` (`results_ablation_{val,train}/{a15,r1}_coc.parquet`
— schema-compatible with the A2 parquets, no script changes needed). Full method/numbers/notes
live in that repo's `docs/first_round_eval_results.md` under "CAC v2 — rescored with the updated
consistency scorer + Qwen extractor"; summarized here for the cross-model view.

| Model | split | n | binary (unparsed=0, abstained excluded) | graded |
|---|---|---|---|---|
| Alpamayo 1.5 | val | 349 | 0.587 | 0.607 |
| Alpamayo R1 | val | 349 | 0.585 | 0.595 |
| Alpamayo 2 Super | val | 349 | **0.607** | 0.615 |
| Alpamayo 1.5 | train | 1722 | 0.618 (n=1719) | 0.623 |
| Alpamayo R1 | train | 1722 | 0.611 (n=1718) | 0.614 |
| Alpamayo 2 Super | train | 1722 | **0.638** (n=1719) | 0.641 |

**Alpamayo 2 Super leads on both splits under this scorer** (val: 0.607 vs 0.587/0.585; train:
0.638 vs 0.618/0.611), same direction as the old B0 scorer's cross-model result (0.438 val /
0.520 train) — this ranking is robust to the scorer choice, unlike the 1.5-vs-R1 ordering below.

The 1.5-vs-R1 ordering on val flips relative to B0 (B0 had R1 ahead: 0.381 vs. 0.344) but the two
are 0.002 apart under Qwen v2 vs. 0.037 apart under B0 — inside noise, not a real reordering.
Train keeps the same 1.5-ahead-of-R1 direction under both scorers (B0: 0.447 vs. 0.438; Qwen v2:
0.618 vs. 0.611).

Every model moved up substantially from B0 to Qwen v2 (val: 1.5 0.344→0.587, R1 0.381→0.585, A2
0.438→0.607; train: 1.5 0.447→0.618, R1 0.438→0.611, A2 0.520→0.638) — consistent with the
scorer-quality read above: most of this delta is B0's trajectory-side classifier reading the same
rollouts less accurately, not the models actually reasoning more consistently under Qwen v2.

**1.5/R1 temperature ablation (0.01 / 0.6 / 1.0), also Qwen v2-scored 2026-08-18** — not repeated
here (Alpamayo 2 wasn't run at other temperatures, so it's not a 3-model comparison); see the
1.5 repo's `docs/first_round_eval_results.md`, "CAC v2 (Qwen extractor) — temperature ablation".
Headline: consistency drops as temperature rises for both models on both splits (e.g. 1.5 train
0.640 @0.01 → 0.618 @0.6 → 0.607 @1.0), the expected direction, confirmed here for the first time
with a validated matching scorer.
