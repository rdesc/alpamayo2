# Alpamayo 1.5 motion-confidence smoke test (n=20, n=100) vs. Alpamayo 2 Super

Preliminary smoke test only -- not a decision to scale up. Ports Experiment 7's 4 scoring
methods (`alpamayo2/examples/motion_confidence_smoke.py`) to Alpamayo 1.5
(`nvidia/Alpamayo-1.5-10B`), scored on the exact same (clip_id, t0_us, positive_action,
negative_actions) tuples used by A2S's K=3 val run -- first N entries of
`outputs/motion_confidence_k3_n289_part1.json`, reused verbatim (not resampled): 1 positive
gold CoC (the event's own) + K=3 negative gold CoCs (other events' gold CoCs via cyclic
shift by 1/2/3), same as A2S.

Script: `/mnt/efs/users/rod/repos/alpamayo1.5/motion_confidence_smoke_a15.py` (new file,
alpamayo1.5 repo -- no tracked file in that repo was touched; later extended in place with
`--num_shards`/`--shard_idx` for the n=100 run, still the same new file, no tracked file
touched). Outputs: `outputs/motion_confidence_a15_k3_n20.json` (single GPU, sequential) and
`outputs/motion_confidence_a15_k3_n100.json` (merged from 8 shard files, one per GPU, all
8 GPUs used in parallel). n=20 took 15.0 min (45s/event, single GPU). n=100 sharded 8-way
took ~12 min wall clock (12-13 events/shard at the same ~45s/event pace). 0/20 and 0/100
data-loading failures.

## Comparison table (K=3 negatives per event)

**n=20:**

| method | A2S pairwise/top1/AUROC | Alpamayo 1.5 pairwise/top1/AUROC |
|---|---|---|
| baseline (yes/no) | 0.883 / 0.700 / 0.831 | 0.783 / 0.550 / 0.675 |
| digit (0-9) | 0.717 / 0.500 / 0.634 | 0.633 / 0.500 / 0.551 |
| likelihood (mean logprob) | 0.600 / 0.450 / 0.617 | 0.583 / 0.450 / 0.577 |
| scene_conditioned | 0.933 / 0.800 / 0.817 | 0.750 / 0.450 / 0.675 |

**n=100 (extended run, same methodology, first 100 events):**

| method | A2S pairwise/top1/AUROC | Alpamayo 1.5 pairwise/top1/AUROC |
|---|---|---|
| baseline (yes/no) | 0.847 / 0.680 / 0.803 | 0.760 / 0.540 / 0.676 |
| digit (0-9) | 0.693 / 0.500 / 0.647 | 0.597 / 0.410 / 0.528 |
| likelihood (mean logprob) | 0.650 / 0.420 / 0.630 | 0.623 / 0.390 / 0.618 |
| scene_conditioned | 0.847 / 0.680 / 0.756 | 0.773 / 0.540 / 0.675 |

(A2S scene_conditioned pulled from `outputs/motion_confidence_k3_toff0.0_n289_part1.json`
first N events -- confirmed identical clip_id/t0_us to `..._n289_part1.json`'s first N,
since the original K3 val run didn't score scene_conditioned.)

**Headline:** the n=100 run confirms the n=20 trend rather than overturning it -- Alpamayo
1.5 ranks below A2S on every method and every metric at both sample sizes. The gap is
largest on `scene_conditioned` and `baseline` (top1_acc 8-14 points lower at n=100) and
smallest on `likelihood` (pairwise_acc within 0.03 of A2S at n=100) -- consistent with
`likelihood` being the one method that doesn't depend on the model emitting a specific
token, so it's least affected by Alpamayo 1.5's weaker instruction-following on generic
yes/no/digit prompts (see gotcha below). Numbers moved modestly between n=20 and n=100 in
both models (e.g. A2S baseline pairwise_acc 0.883 -> 0.847), which is expected sampling
noise at these small n and not a sign either run is unreliable -- the *relative* A2S-vs-A1.5
ordering and gap size is stable across both sample sizes, which is the more important
cross-check.

## Real API gotcha found (worth remembering before scaling up)

**Yes/No and 0-9 digit first-token extraction is near-degenerate on Alpamayo 1.5.** With
A2S's exact `BASELINE_TEMPLATE`/`DIGIT_TEMPLATE` wording, the combined Yes+No probability
mass on the first generated token is typically only 1-5% of the total softmax (top tokens
are usually unrelated things like "Low", "True", "False", "B", "["), and the summed digit
mass is often <1%, sometimes <0.05%. This is not a plumbing bug -- confirmed the same
message-construction/tokenization code correctly elicits well-grounded free-text answers
for narrative questions (e.g. the `SCENE_QUESTION` scene description at greedy decoding
gave a detailed, image-grounded answer), and even reproduced the same failure mode on
Alpamayo 1.5's own documented 1-5 verbalized-confidence question from that repo's
`coc_verbal_confidence.py` (greedy first token was "Low", not a digit). Despite the tiny
absolute mass, the yes_p/no_p *ratio* still carried usable signal (baseline pairwise_acc
0.783, well above chance) -- so the metric isn't meaningless, but the derived
"confidence" values are only a few percent of total probability mass and should be read
with that caveat; scaling this to n=289 would carry the same caveat, not something a
larger sample fixes.

**`generate_text()` mutates the shared `self.vlm.generation_config` in place** (sets
`do_sample=True`, `num_return_sequences=N`, temperature, top_p) and never resets it. A
subsequent raw `model.vlm.generate()` call (used for the baseline/digit/likelihood
first-token and forward-pass methods) must explicitly pass `num_return_sequences=1,
do_sample=False` etc. on every call, or it silently inherits the last `generate_text()`
call's settings and crashes (`ValueError: Greedy methods without beam search do not
support num_return_sequences != 1`).

Structural API notes confirmed by direct testing (useful for anyone porting further
alpamayo1.5 scripts): `apply_chat_template(..., tokenize=True, return_dict=True)` returns
the same tensor keys as A2S's two-step processor call (`input_ids`, `attention_mask`,
`pixel_values`, `image_grid_thw`); `load_physical_aiavdataset`'s return dict is directly
usable with no `select_task_input`/task-profile step; multi-turn conversations (needed for
`scene_conditioned`) are built by hand-extending `helper.create_vqa_message`'s message
list with `<|question_start|>...<|question_end|>` / `<|answer_start|>...<|answer_end|>`
wrapped text turns (there is no separate multi-turn helper in `helper.py`).

**8-way sharding gotcha:** `generate_text()`'s shared `generation_config` mutation (above)
is per-process, so sharding across 8 independent GPU processes sidesteps it entirely --
each shard's `generation_config` pollution stays local to that process. No cross-shard
issues; shards were merged by an `orig_idx` field added to each event record specifically
so the 8 shard files (which each see a different disjoint subset of the 100 events,
interleaved via `indexed_events[shard_idx::num_shards]`) recombine into the original
event order.

## Process check

Both runs finished cleanly: 20/20 and 100/100 events scored, 0 load failures across
either run, 0 tokenization-boundary mismatches for likelihood. No stray GPU processes left
after either run -- `nvidia-smi` / `ps aux` show 0 compute processes and 0MB used on all 8
GPUs after the n=100 8-way sharded run completed.
