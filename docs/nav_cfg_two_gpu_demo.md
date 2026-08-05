# Navigation CFG two-GPU demo (`examples/two_gpu_nav_cfg_demo.py`)

Notes on how classifier-free guidance (CFG) is used to steer Alpamayo2-Super's
predicted trajectory with a natural-language navigation instruction (e.g.
`"Turn right in 30m"`), and why the demo manually places the VLM and the
diffusion expert on separate GPUs.

## What the demo does

1. Loads one PhysicalAI-AV sample (same manifest/`--sample-index` mechanism as
   `inference_smoke.py`).
2. Loads the model with **manual device placement** instead of
   `device_map="auto"`:
   - VLM (vision-language backbone) → `--vlm-device` (default `cuda:0`)
   - Diffusion "expert" denoiser + its KV caches → `--expert-device` (default
     `cuda:1`)

   Generic HF auto-sharding isn't used because the guided and unguided caches
   (below) must move to the expert device together before denoising — see
   `load_model_on_two_gpus` in `examples/two_gpu_nav_cfg_demo.py:232`.
3. Builds two prompts for the same sample: a **guided** one that includes the
   nav instruction, and an **unguided** one without it.
4. Runs the VLM once (with sampling) on the guided prompt to get the
   Chain-of-Causation (CoT) text, then replays those exact same CoT tokens
   through the unguided prompt to get a second, aligned KV cache. Both caches
   move to the expert GPU.
5. The diffusion expert runs twice per denoising step (once per cache) and
   combines the two outputs with a guidance weight — this is what makes the
   sampled trajectory actually respond to the nav instruction.
6. Saves a visualization PNG + JSON, same as `inference_smoke.py`, plus extra
   metadata (`nav_text`, `nav_guidance_weight`, `device_placement`, etc.).

Requires ≥2 visible CUDA devices.

## Guided vs. unguided KV cache: how they're built

Both caches end up covering the **same tokens** (images + ego history + the
same CoT text) — the only difference is whether the nav instruction was
present when the K/V values were computed.

**Guided cache** — built the normal way:

```python
vlm_outputs = model.vlm.generate(**tokenized_data, ...)   # tokenized_data includes the nav instruction
guided_prompt_cache = vlm_outputs.past_key_values
```

The model actually sees the instruction and *samples* the CoT with it
present. One `.generate()` call does prefill + autoregressive decode and
gives you both the CoT tokens and their cache.

**Unguided cache** — built in two forced (non-sampling) passes so it ends up
holding the *same* CoT tokens, computed as if the instruction was never
there:

1. **Prefill** the prompt with the instruction removed:

   ```python
   unguided_prefill_outputs = model.vlm(**unguided_prefill_inputs, use_cache=True, logits_to_keep=1)
   unguided_prompt_cache = unguided_prefill_outputs.past_key_values
   ```

   A plain forward pass (not `.generate()` — nothing is sampled here) over
   the unguided prompt (same images/history, no nav text — `pixel_values`/
   `image_grid_thw` are reused from the guided tokenization so both branches
   see bit-identical image tokens). `use_cache=True` populates the K/V cache
   for this prefix; `logits_to_keep=1` skips computing LM-head logits for
   positions we don't need. Cache only covers the unguided prefix so far.

2. **Replay** (teacher-force) the guided branch's already-sampled CoT tokens
   through that cache:

   ```python
   unguided_vlm_outputs = model.vlm(
       input_ids=guided_generated_tokens,        # CoT tokens sampled by the guided branch
       attention_mask=unguided_continuation_inputs["attention_mask"],
       past_key_values=unguided_prompt_cache,
       cache_position=unguided_continuation_inputs["cache_position"],
       use_cache=True,
       logits_to_keep=1,
   )
   unguided_prompt_cache = unguided_vlm_outputs.past_key_values
   ```

   `input_ids` is fixed to the guided branch's CoT tokens — there's no
   prediction/sampling, the model is just told "compute your internal state
   as if you'd generated this yourself." `cache_position` is remapped
   (`build_unguided_continuation_inputs` in
   `examples/two_gpu_nav_cfg_demo.py:195`) because the guided prompt is
   longer than the unguided one (extra nav-instruction tokens); without the
   remap, the CoT tokens would land at the wrong positional offset relative
   to the (shorter) unguided cache and corrupt the rotary position
   embeddings. The remap makes these tokens look like a normal continuation
   of the unguided prompt.

**Net result:** two KV caches, token-for-token aligned in their continuation
region (same CoT), differing only in what conditioned the attention
underneath (instruction present vs. absent). That alignment is what the CFG
step needs.

## The CFG combination itself

`diffusion/flow_matching.py:104` (`ReleaseFlowMatching._guided_v`):

```python
guided_v = step_fn(x=x, t=t)              # expert output attending into guided_prompt_cache
unguided_v = unguided_step_fn(x=x, t=t)   # expert output attending into unguided_prompt_cache
v = (1 - w) * unguided_v + w * guided_v   # w = inference_guidance_weight
```

- At `w = 1`, the unguided term's coefficient is `(1-1) = 0`, so this
  collapses to `v = guided_v` — i.e. no CFG at all, equivalent to just
  feeding the guided prompt straight to the diffusion head.
- The shipped `nvidia/Alpamayo2-Super` checkpoint sets
  `inference_guidance_weight = 3.0` (see its `config.json`). Plugging that
  in:

  ```
  v = (1 - 3)*unguided_v + 3*guided_v = guided_v + 2*(guided_v - unguided_v)
  ```

  `(guided_v - unguided_v)` estimates the part of the prediction
  attributable specifically to the nav instruction (everything else —
  images, ego history, CoT text — is identical between the two branches).
  The demo extrapolates 2x further along that direction rather than just
  using the plain conditional (`guided_v`) output.

**Why this is needed at all**: the nav instruction is a small piece of text
next to a much larger context (images, ego history). A model trained with
ordinary conditional likelihood tends to under-weight a conditioning signal
like that — plain conditional sampling often produces a trajectory that's
plausible but only weakly nudged by the instruction. CFG corrects for this at
inference time without retraining, by computing what the model predicts
*without* the instruction and amplifying the delta. Same trick used for
guidance scales in text-to-image diffusion models, applied here to
trajectory generation.

## Relevant files

- `examples/two_gpu_nav_cfg_demo.py` — the demo script itself
  (`prepare_nav_model_inputs`, `build_unguided_prefill_inputs`,
  `build_unguided_continuation_inputs`, `sample_with_nav_cfg`,
  `run_two_gpu_nav_cfg`).
- `src/alpamayo2_super/diffusion/flow_matching.py` — CFG combination
  (`_guided_v`) and the Euler integration loop (`_euler`).
- `src/alpamayo2_super/models/expert.py` — where
  `use_classifier_free_guidance` is read off the diffusion module.
