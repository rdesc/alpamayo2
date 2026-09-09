# Motion confidence experiment — handoff (2026-09-03)

## What this project is

Investigating whether Alpamayo 2 Super's VQA yes/no token-probability output is a
trustworthy "motion confidence" signal for its predicted driving actions, using PAI-AV OOD
gold chain-of-causation (CoC) text as ground truth. Full writeup with all methodology,
results, and reasoning: `docs/motion_confidence_experiment.md` (Experiments 1-13,
Takeaways, Open Questions, Files). This file is just a lightweight pointer/handoff, not a
substitute for that doc.

**Train split is tabled for now (explicit user decision, 2026-09-03)** — the 81/300 (27%)
permanently-lost train events from the data-loading concurrency failure are NOT being
retried or backfilled at this time. All new work below is val-only (n=289). Revisit train
later if/when needed; the 81 lost events are still recoverable in principle (see caveat
below), nothing about them has changed.

Background work that was in flight has now completed — see "Latest additions" section
below (Experiments 9-10).

## Latest result (K=3 negatives per event, val vs. train — Experiment 7)

Scaled from 1 to K=3 mismatched-gold-CoC negatives per event to fix a single-negative
coin-flip-variance problem. Reports both `pairwise_acc` (all N×K comparisons) and the
stricter `top1_acc` (positive must beat all K negatives):

| method | val (n=289) pairwise/top1/AUROC | train (n=219 usable) pairwise/top1/AUROC |
|---|---|---|
| baseline (yes/no) | 0.839 / 0.661 / 0.796 | 0.831 / 0.676 / 0.792 |
| digit (0-9) | 0.682 / 0.464 / 0.640 | 0.680 / 0.434 / 0.660 |
| likelihood | 0.616 / 0.374 / 0.616 | 0.653 / 0.416 / 0.637 |

Val and train agree closely on every method/metric — good cross-split consistency. The
key finding is the pairwise_acc-vs-top1_acc gap: baseline's strong pairwise_acc (~0.84)
drops to top1_acc ~0.66-0.68 (beats all 3 negatives), and digit/likelihood's top1_acc
(0.37-0.46) are only modestly above the K=3 random-chance rate of 25%. **Single-negative
pairwise comparisons substantially overstate how reliable these methods are as a
verifier/reranker.**

### Caveats on the table above

- **Train n is 219, not 300.** 111/300 train events failed to load with
  `zipfile.BadZipFile: File is not a zip file` — `physical_ai_av`'s streaming loader reads
  uncached chunks over HTTP without local caching, and this investigation had several GPU
  jobs running concurrently at the time, which triggered truncated reads. A retry pass
  recovered 30/111; 81/300 (27%) were permanently unrecoverable after 3 attempts each. This
  did **not** happen on val (0/289 failures) or the original sequential train n=300
  baseline run (0/300 failures) — it's a concurrency artifact of heavy parallel load
  against this dataset, not a data-quality issue with these specific events. Worth
  avoiding (or adding retry-with-backoff-on-any-exception, not just HTTP 429 matching) if
  running this many parallel jobs against PAI-AV again.
- **~19% of the K=3 negatives are mislabeled on both splits.** An independent Qwen3-VL-32B
  judge (neutral prompt, never shown the gold/positive action) found 18.3% of val negatives
  (159/867) and 19.1% of judged train negatives (51/267, out of 657 attempted — 390 lost to
  the same loading issue above) are actually still plausible actions for their scene,
  dominated by generic transferable phrasings ("steer right following temporary traffic
  delineators," "decelerate to maintain a safe distance from the lead vehicle ahead"). Read
  the table above knowing ~1 in 5 negatives carries this label noise in both directions.
- **Coordination near-miss, not a data problem but worth remembering**: the train
  negative-QA run's script (`examples/motion_confidence_negative_qa.py`) was concurrently
  rewritten mid-run by a separate task building a CoT+majority-vote reliability
  improvement (Experiment 8's follow-up). The train run reconstructed the original v1
  script from an earlier read to keep the val/train comparison apples-to-apples, but
  concurrent background agents that share files under `examples/` can clobber each other's
  in-progress runs if their file sets overlap. No harm done this time, but budget for it
  when dispatching multiple concurrent agents that touch the same scripts.

## Latest additions (completed 2026-09-03, val split only — Experiments 9-10)

All 12 background GPU jobs from the previous handoff (5 t0-offsets x 2 shards for the K=3
+ `scene_conditioned` sweep, + 2 shards for full self-CoC candidate scoring) finished
cleanly — 0/289 data-loading failures in every one of the 12 output files, driver exited
with "All 12 jobs completed", no stray processes left running. Full writeup: Experiment 9
and Experiment 10 in `docs/motion_confidence_experiment.md`.

**Experiment 9 (scene_conditioned + temporal jitter, K=3, n=289 per offset):**
`scene_conditioned` is below baseline at every one of the 5 offsets tested
(-0.2/-0.1/0.0/+0.1/+0.2s) on pairwise_acc/top1_acc/AUROC — Experiment 1/7's "no clear win"
finding replicates across the whole jitter sweep. All 4 methods are much more stable at
this aggregate (n=289) scale than Experiments 2/4's single-clip check suggested (aggregate
metrics move only 0.02-0.05 across the whole +-0.2s sweep vs. up to 0.90 range for one
clip's raw confidence) — per-clip noise and aggregate verifier accuracy are different
questions, one being robust doesn't imply the other is. Offset=0.0 reproduces Experiment
7's exact numbers (clean pass, not a red flag). Clamping check confirmed exactly against
actual output: 36/289 events clamped to `MIN_T0_US` at -0.1s, 38/289 at -0.2s, 0/289 at
positive offsets.

**Experiment 10 (full self-CoC candidate scoring, n=289, positive/negative pairing
deliberately NOT chosen):** baseline shows more within-event score spread across its 8
self-CoC candidates/event than digit or scene_conditioned (mean max-min spread 0.238 vs.
0.172/0.175) — but a worked example shows some of that spread reflects genuinely different
underlying claims among an event's 8 samples, not paraphrase noise, so the number alone
isn't a clean "noisier method" signal. The actual deliverable is
`outputs/motion_confidence_multi_coc_scores_val_n289_merged.json` (289 events x 8
candidates x 4 methods' raw scores) — ready for whichever positive/negative pairing scheme
gets chosen next; **that pairing decision is still open and was deliberately not made
here** (see Open decision points below).

## Latest addition (completed 2026-09-03, val split, n=25 — Experiment 11)

Checked whether the existing ad-hoc `likelihood` method (open VQA-question framing, raw
teacher-forced action text, no trajectory-history conditioning) was even asking the model in
the format it was trained to answer in — it wasn't. Separately confirmed
`baseline`/`digit`/`scene_conditioned` already match the model's real public "vqa" task
contract (`text_tasks.prepare_vqa_inputs`/`build_text_task_messages(task="vqa")`, "no-special
VQA generation" per its own docstring) and needed **no** changes. Built `native_likelihood`
(`examples/motion_confidence_native_cot.py`) to score the candidate action under the model's
real trained CoT-generation conditioning instead: trajectory-history tokens fused into the
prompt via `fuse_traj_tokens`, action teacher-forced inside `<|cot_start|>`/`<|cot_end|>`
special tokens, exactly as `eval_pai_av_val_a2.py` conditions for real inference. Sanity
check (n=3) confirmed all the plumbing risk points named going in: `cot_start`/`cot_end`
land in the right place in the decoded text, `fuse_traj_tokens` needed **no shape
workarounds** (`ego_history_xyz`/`ego_history_rot` arrive already batched `(1,1,16,3)` /
`(1,1,16,3,3)`), and the teacher-forcing prefix-boundary check succeeded every time (never
silently returned `None`).

Scaled to n=25 (K=3 negatives, seed 0, sharded across all 8 GPUs) and compared directly
against the existing ad-hoc `likelihood` on the identical 25 events (0/25 boundary-check
failures either way, 100/100 comparisons scored):

| method | pairwise_acc | top1_acc | AUROC |
|---|---|---|---|
| likelihood (ad-hoc, existing) | 0.427 | 0.040 | 0.581 |
| native_likelihood (full appended span) | 0.400 | 0.000 | 0.553 |
| native_likelihood (action-only span) | 0.413 | 0.080 | 0.559 |

**native_likelihood does not beat the existing ad-hoc likelihood** — both land within a few
points of each other and of chance. Putting the scoring into the model's real trained format
did not surface a cleaner likelihood signal, reinforcing the earlier explanation (Takeaway 5
in the main doc) that the problem is surface-form competition between fluent, plausible
candidate actions, not the prompt/token framing used to elicit the score. Full writeup,
worked example, and caveats (n=25 is small, action-only vs. full-span distinction, reduction
not varied): Experiment 11 in `docs/motion_confidence_experiment.md`.

## Other settled state (also in the main doc, listed here for quick recall)

- Experiment 8 (self-CoC correctness via independent VLM judge, n=8 smoke test): 8/64
  (12.5%) of the model's own self-generated CoCs judged not actually correct for their
  scene, including one event where the 8 self-CoCs contradicted each other and the judge
  correctly caught it.
- A CoT-before-verdict + 3-sample-majority-vote fix to that same judge eliminated two
  flagged noisy-verdict cases, but **also flipped the one genuinely-useful catch to a
  unanimous "Yes"** — looks like a lenience trade-off (judge biased toward Yes more
  broadly), not a clean reliability win. Not yet disentangled whether the CoT reasoning
  itself, the temperature-0.7 sampling, or both, caused this.

## Standing caveat (all experiments through Experiment 11)

Every scoring method loads images via the model's `"vqa"` task profile, which differs from
the model's real trajectory-generation profile in two ways: (1) one camera slot is swapped
(`rear_tele` instead of `front_tele`), and (2) no ego trajectory-history conditioning is
included for any method except `native_likelihood` (Experiment 11), which still used the
`"vqa"` camera set. Camera *image* history (4 frames/camera, ~1.6s) is present in every
experiment -- only the ego's own motion-history embedding is missing. See the "Standing
caveat" section near the top of `docs/motion_confidence_experiment.md` for full detail. Net:
no result in this investigation should be read as "confidence under the model's real driving
context" without this caveat. Building a trajectory-profile-matched variant to check whether
this changes anything is a new, not-yet-actioned open item.

## Latest addition (Experiment 13, same-scene labeled scoring, n=42 mixed events)

Used Experiment 12's Claude-vision Yes/No labels as ground truth (instead of gold-CoC vs.
mismatched-gold-CoC) to re-score `baseline`/`digit`/`likelihood`/`scene_conditioned` against
real, same-scene, plausible-sounding negatives -- no new model calls, joined against
Experiment 10's already-computed raw scores. Only the 42/100 events with mixed Yes/No labels
contribute a comparison (58 are unanimous-Yes, excluded, not zero-filled).

| method | pairwise_acc (same-scene labeled) | pairwise_acc (gold cross-event, Exp. 7) |
|---|---|---|
| baseline | 0.606 | 0.839 |
| digit | 0.578 | 0.682 |
| likelihood | 0.615 | 0.616 |
| scene_conditioned | 0.571 | -- |

**Every method drops sharply, and `baseline`'s big lead over `likelihood` disappears**
(0.839->0.606 vs. 0.616->0.615, now statistically tied) -- `baseline`'s strength against gold
negatives looks substantially inflated by those negatives being easy (random other-scene
actions), not by genuine same-scene discrimination. This is the "harder negatives" the Open
Questions section had been asking for since Experiment 1. See Experiment 13 in the main doc
for full caveats (n=42 is small; ground truth is itself an imperfect judge).

## Latest addition (Experiment 14, calibration + risk-coverage, both ground truths)

Checked whether raw confidence scores are calibrated probabilities, not just good rankers --
Brier score, ECE (10 equal-count bins), reliability-diagram table, and a risk-coverage
(selective-prediction) curve, on both `k3` (Experiment 7/9's gold-vs-mismatched-gold corpus,
n=289) and `selfcoc_labeled` (Experiment 13's real same-scene corpus, n=42 mixed events).
`likelihood`/`native_likelihood` are Platt-scaled (in-sample) to [0,1] first.

| method | k3: Brier / ECE / acc@0.5 | selfcoc_labeled: Brier / ECE / acc@0.5 |
|---|---|---|
| baseline | 0.186 / 0.181 / 0.760 | 0.419 / 0.410 / **0.444** |
| digit | 0.211 / 0.153 / 0.689 | 0.259 / 0.183 / 0.521 |
| scene_conditioned | 0.221 / 0.222 / 0.734 | 0.481 / 0.488 / **0.435** |
| likelihood (Platt-scaled) | 0.196 / 0.044 / 0.718 | 0.226 / 0.094 / 0.643 |

On `k3`, every bounded method is systematically *underconfident* but monotonic (a `baseline`
score of 0.61 is actually correct ~85% of the time) -- miscalibrated but directionally safe.
On the harder `selfcoc_labeled` corpus, `baseline`/`scene_conditioned` become non-monotonic
through most of their range and their accuracy at the naive 0.5 threshold falls *below
chance*; risk-coverage shows their usable signal lives almost entirely in the top ~10-15%
most-confident readings. `likelihood` -- the weakest method by every ranking metric in this
doc -- has the best-behaved calibration on the harder corpus, the first result here where it
doesn't simply underperform. Direct answer to "can a judge score of 0.5 mean the model is
uncertain": it depends on which ground truth and which method -- not readable literally
without a check like this. See Experiment 14 in `docs/motion_confidence_experiment.md` for
full detail, reliability tables, and caveats (Platt-scaling is in-sample; `selfcoc_labeled`
n=42 is still a small-n read). Script: `examples/motion_confidence_calibration.py`
(`--source {k3,selfcoc_labeled}`). Outputs: `outputs/motion_confidence_calibration_k3.json`
/ `_selfcoc_labeled.json`.

Also added `examples/motion_confidence_label_tool.py` -- a local stdlib-only web app for
manually labeling self-CoC correctness (Correct/Incorrect/Unsure, keyboard Y/N/U,
resumable), reading the 100-event `front3_qa` manifest and saving to
`outputs/motion_confidence_human_labels.json` in the same schema as the Claude-vision
labels, for a fourth independent judge once labeled.

## Open decision points, not yet actioned

0. **The Qwen3-VL-32B judge itself has now had one small independent visual spot-check
   (n=8, hand-picked flagged cases, not random)** -- see the new addendum at the end of
   Experiment 8 in `docs/motion_confidence_experiment.md`. Result: 6/8 agree, 1/8 clear
   disagreement (two visually near-identical scenes got opposite verdicts on the same claim
   type), 1/8 ambiguous. This is a real but small first check; it should be extended to a
   larger, *randomly* sampled set before treating Qwen verdicts as ground-truth correctness
   labels at scale. **A second, independent labeling pass now exists**: Claude directly
   vision-labeled all 8 self-CoCs for 10 events (n=80, reduced 3-camera/1-frame input) --
   see "Addendum 2" right after that spot-check in `docs/motion_confidence_experiment.md`
   and `outputs/motion_confidence_claude_vision_labels_n10.json`. It found 84% full 3-way
   agreement with Qwen v1/v2 on the 8 overlapping events, was noticeably closer to lenient
   Qwen v2 (95%) than stricter Qwen v1 (86%), and also corrected a factual error in the
   original spot-check's write-up (the two "near-identical scenes" were not actually
   near-identical). **Scaled to n=100 (800 candidates) -- Experiment 12**: 672 Yes / 128 No
   (84.0%), 58/100 events unanimous 8/8 Yes, 0/100 unanimous No, 42/100 mixed. 3 events
   (indices 11, 17, 64) scored only 1/8 Yes with concrete, checkable hallucinations (an
   oncoming bus called "stopped and blocking the lane," cones placed on the wrong side, an
   invented lane closure) -- the sharpest evidence yet that a meaningful minority of
   self-generated CoCs are factually wrong about the scene, not just imprecisely worded. No
   Qwen comparison at this scale (Qwen has only judged the first 8 events). See Experiment 12
   in `docs/motion_confidence_experiment.md` and `outputs/motion_confidence_claude_vision_labels_n100.json`.
1. Before scaling the self-CoC QA judge to the full 289+300 event sets: investigate the
   CoT+majority-vote judge's lenience shift (does it still catch genuinely-wrong self-CoCs
   on a larger, verdict-diverse sample?), or proceed with the original single-greedy-call
   v1 judge instead, accepting its noisier-but-less-lenient verdicts.
2. Re-run this whole calibration comparison on Alpamayo 1.5 and Qwen (requested, not
   started).
3. Find a simpler/cheaper correctness proxy for model-sampled CoC than a second full VLM
   judge pass (self-consistency across the 8 samples per event is one candidate, flagged
   in the doc as untested at scale).
4. Incorporate LLM-as-verifier literature ideas into the judging step (requested, not
   started).
5. **Self-generated CoC positive/negative pairing (Experiment 10) is still open and
   deliberately not resolved** — which of an event's 8 self-sampled CoCs should count as
   "the" positive, and what a matched negative should look like, given at least some
   events' 8 samples are genuinely different claims rather than paraphrases of one claim
   (see Experiment 10's worked example). The raw per-candidate scores for all 4 methods are
   ready in `outputs/motion_confidence_multi_coc_scores_val_n289_merged.json` for whoever
   picks this decision up.
6. Experiment 11's `native_likelihood` (real CoT-generation-format teacher-forcing) did not
   beat the existing ad-hoc `likelihood`, only checked at n=25 against the ad-hoc method.
   Not yet tried: a different score reduction (e.g. the PMI/context-free-baseline
   correction already flagged elsewhere in the doc) computed under the native conditioning
   specifically, before concluding likelihood-based scoring is unsalvageable regardless of
   prompt format.

Full detail, all numbers, and full case-by-case examples for everything above are in
`docs/motion_confidence_experiment.md`.
