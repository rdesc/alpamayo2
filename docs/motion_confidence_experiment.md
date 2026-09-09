# Motion confidence via scene-description-conditioned VQA — experiment log

Alpamayo 2 Super's VQA head can be asked "Is `<action>` the right driving decision?" and
answered with a first-token Yes/No softmax probability (`notebooks/vqa.ipynb`'s existing
probe). This log tracks whether that raw token probability can be made into a more
trustworthy motion-confidence signal by (a) making the model describe the scene before
judging the action, with the candidate action withheld from that description step to
avoid biasing it, and (b) sampling that description multiple times and averaging the
judgment -- versus just how noisy the raw signal already is along axes that have nothing
to do with the scoring method (which exact frame, how the action is worded).

## Method

Three scoring methods, all read off the first generated token's softmax over the
model's vocabulary (single forward pass each, except scene-conditioned's description
step):

- **baseline**: single turn, single forward pass. `user`: [6 camera images] +
  `"Is '<action>' the right driving decision for the ego vehicle right now? Answer Yes
  or No."` No sampling. Confidence = `P(yes) / (P(yes) + P(no))`.
- **digit**: single turn, single forward pass. `user`: [6 camera images] + `"On a scale
  from 0 to 9, how confident are you that '<action>' is the right driving decision for
  the ego vehicle right now? 9 means certain it is correct, 0 means certain it is
  incorrect. Answer with a single digit only, no words."` No sampling. Confidence =
  probability-weighted expected digit rescaled to `[0, 1]`:
  `sum_i(i * P(digit=i)) / (9 * sum_i P(digit=i))`. Motivation: a graded self-report read
  the same cheap way as the yes/no ratio (one forward pass, no free-text parsing), rather
  than inferring "how sure" from a binary answer.
- **scene-conditioned**: two turns.
  1. `user`: [images] + `"Describe the current driving scene relevant to the ego
     vehicle's next driving decision: nearby road users, road and lane conditions,
     traffic control, and any hazards. Do not mention or evaluate any specific driving
     decision."` -- sampled `N` times at temperature 0.6 (candidate action never in
     context here).
  2. For each sampled description: `assistant`: (the description) followed by `user`:
     `"Given the scene described above, is '<action>' the right driving decision for the
     ego vehicle right now? Answer Yes or No."` -- confidence averaged over the `N`
     judgments.

System prompt (fixed): "You are a driving assistant that generates safe and accurate
actions." Implementation: `examples/motion_confidence_smoke.py`
(`BASELINE_TEMPLATE`/`DIGIT_TEMPLATE`/`SCENE_QUESTION`/`JUDGE_TEMPLATE`/`judge_yes_no`/
`judge_digit`).

## Ground truth: PAI-AV OOD gold CoC

Uses the PAI-AV long-tail OOD reasoning val split (`reasoning/ood_reasoning.parquet`,
same 349-event set as `docs/pai_av_ood_eval.md`'s eval). Each event carries a
human-written gold CoC sentence describing the correct driving decision (e.g. "Slow down
for the one-way traffic control while navigating through the construction zone."). Per
event:

- **positive action** = that event's own gold CoC (should score high confidence).
- **negative action** = another (randomly paired) event's gold CoC (should score low --
  a real, human-written action that is very likely inapplicable to this scene).

Pairing is a cyclic shift over the sampled event set, not a synthesized contradiction --
see "open questions" below for why this makes some negatives easier to reject than a
production confidence signal would ever see.

## Standing caveat -- scene context here is NOT identical to real trajectory generation

Every experiment in this doc (through Experiment 11) loads images via
`select_task_input(source, "vqa")`, i.e. the model's public `"vqa"` release task. This is
a real, model-supported input contract, but it is **not** the same context the model
conditions on when it actually generates a trajectory, in two concrete ways:

1. **Camera set differs by one camera.** The `"vqa"` profile
   (`input_profiles.VQA_SIX_CAMERA_FOUR_FRAME`, camera ids `[0,1,2,3,4,5]`) includes
   `camera_rear_tele_30fov` (id 4) and excludes `camera_front_tele_30fov` (id 6). The
   trajectory-generation profile actually used at inference
   (`input_profiles.DRIVING_SIX_CAMERA_FOUR_FRAME`, camera ids `[0,1,2,3,5,6]`, "the
   canonical 7-camera ring minus rear_tele" per `eval_pai_av_val_a2.py`'s docstring) has it
   the other way around. So every confidence probe in this doc gives the model a rear
   telephoto view it would not normally have while driving, and withholds the forward
   telephoto view it would normally have (useful for resolving a distant lead vehicle or
   traffic light precisely). The other 5 camera slots (cross_left, front_wide, cross_right,
   rear_left, rear_right) match.
2. **Camera *image* history is present (4 frames/camera, ~1.6s window) in every
   experiment** -- both `VQA_SIX_CAMERA_FOUR_FRAME` and `DRIVING_SIX_CAMERA_FOUR_FRAME` use
   `frame_indices=(0,1,2,3)`, so this part does match real trajectory generation. What does
   **not** match is **trajectory-history conditioning**: the ego's own recent 3D
   trajectory, separately tokenized and fused into the prompt via
   `models.utils.fuse_traj_tokens` (48 tokens by default, see `text_tasks.generate_text`'s
   `task != "vqa"` branch). By the model's own public API, the `"vqa"` task never includes
   this. So `baseline`, `digit`, `scene_conditioned`, and the original ad-hoc `likelihood`
   are all judging "is this the right decision" with **no knowledge of the ego's own recent
   motion** -- only Experiment 11's `native_likelihood` added real trajectory-history
   tokens back in, and even that one still used the `"vqa"` camera set from point 1, not
   the trajectory camera set.

Net effect: nothing in this doc's numbers should be read as "how confident the model would
be while actually driving" in the fully faithful sense -- they measure the model's
yes/no/digit/likelihood response under a related but narrower context (image history yes,
one swapped camera, no ego-motion history in most methods). Whether closing this gap (real
`"trajectory"` camera set + `fuse_traj_tokens` history, for every method not just
`native_likelihood`) changes any headline finding is untested -- see Open Questions.

## Experiment 1 -- calibration comparison, n=60

`examples/motion_confidence_smoke.py --num_events 10` (seed 0) +
`--num_events 40 --seed 1` (disjoint-by-construction via `--skip`, though the two runs
used different sampling code paths so a handful of events ended up drawn by both --
harmless, just not perfectly deduplicated) + `--num_events 10 --seed 1 --skip 40`
(disjoint continuation of the seed-1 shuffle). 5 scene-description samples/event.

| method | mean conf (pos) | mean conf (neg) | pairwise acc (pos>neg) | AUROC |
|---|---|---|---|---|
| baseline | 0.228 | 0.084 | 0.817 (49/60) | 0.797 |
| digit (0-9 scale) | 0.512 | 0.417 | 0.700 (42/60) | 0.638 |
| likelihood (mean logprob) | -4.408 | -4.974 | 0.550 (33/60) | 0.619 |
| scene-conditioned | 0.130 | 0.043 | 0.850 (51/60) | 0.777 |

(n=50 checkpoint, for reference: baseline pairwise-acc 0.860/AUROC 0.830,
scene-conditioned pairwise-acc 0.880/AUROC 0.795 -- both methods' numbers moved down
slightly with 10 more events, same qualitative pattern held. The digit row above was run
once, over the same 60-event selection, after both other methods were already at n=60.)

**The 0-9 digit scale is a clearly worse calibration signal than the plain yes/no
ratio** -- lower pairwise accuracy (0.700 vs. 0.817) and lower AUROC (0.638 vs. 0.797),
and its mean confidences for positive/negative actions sit much closer together (0.512
vs. 0.417, a gap of 0.095) than the yes/no baseline's (0.228 vs. 0.084, a gap of 0.144).
Checked whether this is because the model simply refuses to commit probability mass to
a bare digit as its first token -- it is not. Directly comparing total mass on the
yes/no answer space vs. the digit answer space, on the exact same 120 reads (60 events x
pos/neg, single-token variants only, both from the same `--methods baseline digit` run):

| | outcomes | mean total mass | median | min | max |
|---|---|---|---|---|---|
| yes/no | 2 | 0.153 | 0.079 | 0.001 | 0.796 |
| digit (0-9) | 10 | 0.330 | 0.225 | 0.000 | 0.972 |

The digit answer-space captures **more** total mass on average than yes/no does (0.330
vs. 0.153, ~2.15x), not less -- yes/no mass exceeds digit mass on only 45/120 reads
(38%). Per-outcome the digit mass is thinner (0.033/digit vs. 0.077/yes-or-no, since it's
split ten ways instead of two), but the model is not shy about answering with a bare
digit; if anything it is *more* willing to than with a bare Yes/No. (Both are still a
minority of the full next-token distribution in most reads -- consistent with the
earlier finding that the model's modal first token is usually neither, it prefers to
echo the action text -- but digit's share of that minority is the larger one.)

So the calibration gap isn't explained by "the model refuses to use the digit scale."
The real driver is distributional shape, not quantity: where the model *does* put mass
on digits, it isn't sharply peaked at the extremes (0 or 9) the way yes/no is sharply
peaked at "Yes" or "No" for confident cases -- graded scales invite graded (hedged)
answers, which compresses the positive/negative gap even though more raw probability is
available to work with. The event-to-event variance in total digit mass (many reads near
0, some near 1, per the table above) still adds noise on top of that, but it is not the
primary explanation.

**Tokenization gotcha (caught before trusting these numbers):** the first version of
`digit_confidence` summed probability over two spellings per digit, `"9"` and `" 9"`
(leading space), mirroring how `YES_VARIANTS`/`NO_VARIANTS` include a leading-space
form. For yes/no this is harmless because every variant (`"Yes"`, `" Yes"`, `"YES"`,
etc.) tokenizes to exactly one token each with this tokenizer. But `" 9"` tokenizes as
**two** tokens, `[space, "9"]`, not one -- and the leading-space token is the *same*
token id for every digit. Summing it in naively meant every digit's bucket included a
slice of the same shared space-token probability, inflating and cross-contaminating all
ten buckets (some per-event sums exceeded 1.0, which is impossible for disjoint
single-token reads). Fixed by only counting variants that tokenize to exactly one token
(bare `"9"` only) -- the same guard `motion_confidence_faithfulness_example.py` already
used for its sampled-token classification, just not one this script had needed before
digit added a variant that broke the "every variant is one token" assumption the
yes/no code happened to satisfy.

**System-prompt tweaks don't move the digit method either.** Tested two alternatives to
the default system prompt ("You are a driving assistant that generates safe and accurate
actions.") against the same 10-event batch (seed 0): a format-enforcing prompt ("...answer
with exactly one digit and nothing else -- no words, no punctuation") and a
calibration-framing prompt ("...rate how confident you are... using the full 0-9 scale,
and make sure your numeric answer reflects your true degree of certainty").

| system prompt | mean conf (pos) | mean conf (neg) | gap | pairwise acc | AUROC |
|---|---|---|---|---|---|
| default (control) | 0.496 | 0.387 | 0.108 | 0.800 | 0.600 |
| format-enforcing | 0.463 | 0.352 | 0.111 | 0.700 | 0.630 |
| calibration-framing | 0.469 | 0.351 | 0.118 | 0.800 | 0.670 |

All three land within noise of each other at n=10 -- not a promising lever, not scaled up
further. `--system_prompt` was added to `motion_confidence_smoke.py` as a general
override if this is worth revisiting with a larger n or different phrasing later.

**Likelihood scoring (no yes/no or digit framing at all) does not fare better -- worse,
in fact.** Both baseline and digit fight the model's own preference (established earlier)
to just echo the action text rather than emit "Yes"/"No"/a digit. `likelihood` sidesteps
that fight entirely: ask an open "what is the right driving decision?" question, then
teacher-force the candidate action itself as the answer and read its own mean
per-token log-likelihood under the model (`examples/motion_confidence_smoke.py`'s
`score_likelihood`, `--methods likelihood`). Length-normalized (mean, not sum) since
candidate actions vary from 3 to 30+ tokens and a raw sum would trivially favor whichever
action happens to be shorter.

Result, n=60 (same events, same pairing as the table above): pairwise-acc **0.550**
(33/60), AUROC **0.619** -- clearly worse than baseline (0.817/0.797) and even a bit
worse than digit (0.700/0.638). For comparison, the *unnormalized* sum-logprob version
scores pairwise-acc 0.600 / AUROC 0.586 -- so length-normalization does change the
ranking (better AUROC, slightly worse pairwise-acc) but doesn't rescue the method either
way.

Checked whether length is still leaking through despite the per-token normalization: it
partially is. `mean_pos` and `mean_neg` token counts are identical in aggregate (15.6
each, expected since it's the same pool of real gold-CoC texts just paired differently),
but *within* a pair, whichever action is longer wins 39/56 non-tied pairs (70%), and
length-diff correlates with score-diff at r=0.225 even after normalization. That's a
real residual bias, but it isn't the primary story: even a perfect fix for it would still
leave the method well below baseline. The likelier explanation is surface-form
competition -- the model's per-token fluency for a plausible-sounding driving-action
sentence is dominated by how natural that phrasing is in general (in-domain, grammatical,
plausible-sounding), not by whether it fits *this specific* scene, since every candidate
here (positive and negative alike) is a real, fluent, human-written gold CoC from
*some* PAI-AV event. A principled fix would be a PMI-style correction -- score
`logP(action | scene) - logP(action | no scene)` so the action's own context-free
fluency cancels out -- but that needs a text-only or neutral-image control forward pass
through this VLM, which is nontrivial architecturally and not yet built; not pursued
further without deciding it's worth the added complexity given plain likelihood already
underperforms the much cheaper baseline by this much.

**No clear win for the more expensive method at this scale.** Scene-conditioning edges
out baseline on pairwise accuracy but is slightly worse on AUROC; only 6/60 events flip
which method "wins." Scene-conditioning costs ~6x the forward passes/event (1
description-generation call + 10 judge calls vs. 2 for baseline).

One likely reason: mismatched-gold-CoC negatives are often trivially different from the
real action (e.g. "Resume speed" vs. "Decelerate for the cable car ahead") -- both
methods already nail those, so the benchmark isn't stressing calibration on genuinely
ambiguous cases, which is where scene-description grounding should help most. A harder
negative (same-scene, single-axis-flipped action, e.g. via the CAC scorer's
slow_down/speed_up/left/right taxonomy in
`alpamayo1_x_rl/rewards/coc_action_consistency_*.py`) was considered and deferred --
worth revisiting before concluding the method doesn't help.

### Worked example: clip `428709c9-4a9c-4e90-bdbd-e56d430bd195`, t0=14.1s

Gold positive: *"Steer left following temporary traffic delineators."* Gold negative
(mismatched): *"Go straight following temporary traffic delineators."*

5 sampled scene descriptions (candidate action withheld):
1. "Keep lane through the construction zone due to cones defining a temporary lane boundary."
2. "Keep lane through the temporary construction lane shift due to cones defining the path on the right."
3. "Keep lane through the work zone due to cones defining a temporary lane boundary."
4. "Maintaining lane and gently accelerating because the path ahead opens up while adjacent traffic is separated by cones and occupies the right lane and the space ahead-right is blocked by a stopped van; thus continuing straight provides safe clearance."
5. "Keep lane through the construction zone due to cones defining a temporary lane boundary."

Per-sample confidence "Steer left" is correct: [0.45, 0.34, 0.52, 0.10, 0.45] -> mean 0.37
Per-sample confidence "Go straight" is correct: [0.88, 0.86, 0.87, 0.82, 0.88] -> mean 0.86

The model consistently self-describes the scene as "keep lane," then rates "Go straight"
(the gold *negative*) far more confident than "Steer left" (gold *positive*) -- one of
the 5 events where scene-conditioning disagrees with the label. Plausibly not a model
failure: "steer left" in the gold CoC likely means a small in-corridor lane shift that
the model's own vocabulary calls "keep lane," i.e. a wording mismatch in how the
mismatched-negative construction labels ground truth, not necessarily miscalibration.
Also notable: raw `yes_p`/`no_p` here are tiny (~0.001-0.007) -- almost all first-token
probability mass goes to neither Yes nor No, so the ratio is a small signal riding on
top of a mostly-elsewhere distribution.

### Worked example: clip `a987d18a-92e2-4c13-b61e-fda0de5e685d`, t0=10.49s

Gold positive: *"Steer left to maintain a safe distance from the construction zone with
traffic barriers."* Gold negative (mismatched): *"Go straight following temporary
traffic barriers when navigating through the construction zone."*

5 sampled scene descriptions (candidate action withheld):
1. "Nudge left due to parked vehicles narrowing the lane on the right."
2. "Nudge left due to parked cars narrowing the lane on the right."
3. "Nudge left to increase clearance to the parked car on the right."
4. "Nudge left due to parked cars narrowing the lane on the right."
5. "Nudge left due to the parked car on the right."

Per-sample confidence "Steer left..." is correct: [0.073, 0.078, 0.082, 0.078, 0.144] -> mean 0.091
Per-sample confidence "Go straight..." is correct: [0.067, 0.071, 0.049, 0.071, 0.083] -> mean 0.068

Baseline confidence: "Steer left..." 0.108, "Go straight..." 0.134 -- **baseline picks
the wrong action** (ranks the mismatched negative higher). Here the model's own scene
descriptions are unanimous and specific ("nudge left" for parked-car clearance), and
scene-conditioning's average correctly flips the ranking back to favor the gold
positive. This is one of the 6 disagreement events, and the mirror case of the
`428709c9` example above -- there scene-conditioning overrode a correct-looking
consensus and got it wrong; here it overrides an incorrect baseline call and gets it
right. Same mechanism, opposite outcome -- consistent with "no clear win on net,"
Experiment 1's headline finding.

### Worked example: clip `ee6ad42f-370e-410c-8f84-b14e8d98c237`, t0=8.38s

Gold positive: *"Resume speed for the green traffic light while waiting for the
pedestrian crossing the street."* Gold negative (mismatched): *"Maintain speed while
navigating through the construction zone delineated by traffic cones."*

5 sampled scene descriptions (candidate action withheld):
1. "Accelerate through the intersection due to the green traffic light"
2. "Stop due to red traffic light"
3. "Resume speed from stop since the traffic light turns green."
4. "Resume speed from stop since the traffic light turns green."
5. "Accelerate through the intersection since the traffic light turns green."

Sample 2 directly contradicts the other four -- **the model hallucinated the opposite
light color** ("red" vs. "green") in one of five samples of the same frame. This is a
concrete instance of the failure mode Experiment 2/3 quantify indirectly: a single
resample can simply get the scene wrong, not just phrase it differently.

Per-sample confidence "Resume speed..." is correct: [0.138, 0.097, 0.119, 0.119, 0.128] -> mean 0.120
Per-sample confidence "Maintain speed..." is correct: [0.182, 0.035, 0.088, 0.088, 0.098] -> mean 0.098

Baseline confidence: "Resume speed..." 0.126, "Maintain speed..." 0.499 -- baseline
strongly (and wrongly) favors the mismatched negative. Scene-conditioning correctly
flips this to favor the gold positive, but notice the mechanics: the "red light"
hallucination (sample 2) happens to *pull the negative action's confidence down*
(0.035, the lowest of its five), which helps the average land on the right side of 0.5
almost by accident -- a wrong scene description contributing to a right answer. That is
not something to rely on; it means at least part of this event's correct ranking comes
from noise, not from all five samples independently reaching the correct conclusion for
the same (correct) reason.

### Worked example: clip `ce6e3554-7095-4c9b-8941-b037ecb1b5c6`, t0=1.79s

Gold positive: *"Steer right to maintain a safe distance from the traffic barriers on
the left."* Gold negative (mismatched, this pairing): *"Strong deceleration to yield to
the cut-in vehicle from the left."*

5 sampled scene descriptions (candidate action withheld):
1. "Slow down due to lane narrowing caused by construction barriers ahead"
2. "Slow down due to a road barrier narrowing the lane ahead"
3. "Slow down due to road construction barrier blocking part of the lane"
4. "Slow down due to roadwork barricade blocking part of the lane"
5. "Slow down due to a road barrier blocking part of the lane ahead"

Per-sample confidence "Steer right..." is correct: [0.012, 0.027, 0.029, 0.027, 0.046] -> mean 0.028
Per-sample confidence "Strong deceleration to yield to cut-in..." is correct: [0.0006, 0.0011, 0.0012, 0.0011, 0.0011] -> mean 0.001

Baseline confidence: "Steer right..." 0.068, "cut-in deceleration" 0.006. Both methods
correctly and decisively favor the gold positive here -- not a disagreement case. Worth
noting anyway: the model's own scene description ("slow down due to a barrier") doesn't
literally match *either* candidate's specific maneuver (lateral steer-right vs. yielding
to a cut-in). The correct ranking isn't coming from the description actively endorsing
"steer right" -- it's coming from the negative describing a scenario (a cut-in vehicle)
that's clearly absent, which both methods reject hard. That's an easy negative by
construction (see Experiment 1's "why no clear win" discussion) -- a case like this is
exactly the kind the mismatched-gold-CoC benchmark produces disproportionately often.

## Experiment 2 -- temporal sensitivity (baseline)

Does this exact clip/action pair, at 7 t0 offsets spanning +-0.3s (the loader samples
camera context frames on a fixed 0.1s grid, so this is roughly +-3 frames of real camera
motion). `examples/motion_confidence_time_sensitivity.py`, baseline method only.

| offset | t0_us | "Steer left" conf | "Go straight" conf |
|---|---|---|---|
| -0.3s | 13,800,000 | 0.245 | 0.995 |
| -0.2s | 13,900,000 | 0.068 | 0.988 |
| -0.1s | 14,000,000 | 0.050 | 0.932 |
| 0.0s | 14,100,000 | 0.241 | 0.969 |
| +0.1s | 14,200,000 | 0.182 | 0.993 |
| +0.2s | 14,300,000 | 0.407 | 0.994 |
| +0.3s | 14,400,000 | 0.948 | 0.997 |

"Steer left" swings **0.050 -> 0.948** (range 0.898) across a 0.6s window -- a near-total
flip from "confidently wrong" to "confidently right" driven by which single frame landed
as the most recent context image. "Go straight" stays rock-steady (0.932-0.997, range
0.065). This instability is orthogonal to what scene-description sampling addresses:
that averages over the model's own stochastic language generation at a **fixed** frame,
not over which frame it was given.

## Experiment 3 -- phrasing sensitivity (baseline)

Same clip/t0, 5 meaning-preserving paraphrases per action (voice/clause/synonym changes
only). `examples/motion_confidence_phrasing_sensitivity.py`.

| "Steer left..." paraphrase | conf |
|---|---|
| Steer left following temporary traffic delineators. | 0.241 |
| Move to the left, guided by the temporary traffic delineators. | 0.141 |
| Turn left in accordance with the temporary traffic delineators. | 0.024 |
| Follow the temporary traffic delineators by steering left. | 0.221 |
| Bear left as indicated by the temporary traffic delineators. | 0.158 |

Range 0.024-0.241 -- qualitatively stable ("probably not"), ~10x swing in the number.

| "Go straight..." paraphrase | conf |
|---|---|
| Go straight following temporary traffic delineators. | 0.969 |
| Continue straight, guided by the temporary traffic delineators. | 0.982 |
| Proceed straight in accordance with the temporary traffic delineators. | 0.986 |
| Keep going straight along the path marked by the temporary traffic delineators. | 0.807 |
| Maintain a straight course as indicated by the temporary traffic delineators. | **0.343** |

Range 0.343-0.986: four paraphrases cluster at 0.81-0.99, but "Maintain a straight
course..." alone drops to 0.343 -- large enough to flip a 0.5-threshold decision, from
wording that changes nothing about the claim.

**Together, Experiments 2 and 3 show two independent nuisance axes (frame choice,
phrasing) that swing the raw yes/no signal by as much as the scene-conditioning method
is trying to fix via language-sampling -- and neither axis is touched by sampling at one
fixed frame+wording.**

## Experiment 4 -- does scene-conditioning damp the temporal sensitivity?

Same 7-offset sweep as Experiment 2, both methods this time.
`examples/motion_confidence_time_sensitivity.py --num_samples 5`.

| offset | baseline pos | scene pos | baseline neg | scene neg |
|---|---|---|---|---|
| -0.3s | 0.245 | 0.328 | 0.995 | 0.926 |
| -0.2s | 0.068 | 0.182 | 0.988 | 0.726 |
| -0.1s | 0.050 | 0.180 | 0.932 | 0.594 |
| 0.0s | 0.241 | 0.467 | 0.969 | 0.897 |
| +0.1s | 0.182 | 0.252 | 0.993 | 0.941 |
| +0.2s | 0.407 | 0.426 | 0.994 | 0.940 |
| +0.3s | 0.948 | 0.780 | 0.997 | 0.940 |

| | positive range | negative range |
|---|---|---|
| baseline | 0.898 (0.050-0.948) | 0.065 (0.932-0.997) |
| scene-conditioned | 0.600 (0.180-0.780) | 0.347 (0.594-0.941) |

Mixed result. Scene-conditioning narrows the (very large) positive-action range
somewhat, 0.898 -> 0.600 -- still large. But it makes the negative-action range
**worse**, 0.065 -> 0.347: the baseline was near-saturated and stable on "Go straight"
at every offset, while scene-conditioning's own generated descriptions (and thus its
judgments) vary more across frames, introducing instability that wasn't there before.
Averaging over language samples at each frame does not average away, and in this case
partly reintroduces, frame-to-frame sensitivity.

## Experiment 5 -- self-generated CoC instead of gold CoC

Every experiment so far asks "is this HUMAN-written action right?" Does the model judge
its own utterances the same way? A self-confirmation/sycophancy bias would show up as
inflated confidence on the model's own predictions relative to an equally-plausible
human statement; a "doesn't trust itself" bias would show the opposite.

Method (`examples/motion_confidence_self_coc.py`): for the exact same 60 events used in
Experiment 1 (loaded via `--events_from` from the Experiment 1 result files, so the event
set is identical, not merely same-sized), generate the model's OWN chain-of-thought via
`model.sample_trajectories_from_data(..., return_extra=True)["cot"]` on the "trajectory"
task profile (6 cams `[0,1,2,3,5,6]`, the model's trained CoT-conditioning input contract
-- `num_traj_samples=1`, `diffusion_kwargs={"inference_step": 2}`, trajectory output
itself discarded, only the CoT text is used). This is genuinely cheap: one lightweight
call per event, no new heavy pipeline. Positive/negative pairing then mirrors Experiment
1 exactly (cyclic shift over the self-generated CoTs), and the actual yes/no judging step
runs on "vqa" profile images (same as every other experiment here), via the same
`BASELINE_TEMPLATE`/`judge_yes_no`.

Example self-generated CoTs (compare to the gold CoC style used elsewhere): *"Nudge left
due to parked vehicles narrowing the lane on the right"*, *"Keep distance to the lead
vehicle because a car is directly ahead in the same lane."*, *"Turn right because
construction barricades block the lane ahead."* -- similar register and specificity to
the human-written gold CoC, unsurprising since the model presumably learned this style
from data resembling it.

| candidate action source | mean conf (pos) | mean conf (neg) | pairwise acc (pos>neg) | AUROC |
|---|---|---|---|---|
| gold CoC (Experiment 1 baseline) | 0.228 | 0.084 | 0.817 (49/60) | 0.797 |
| self-generated CoC | 0.270 | 0.088 | 0.850 (51/60) | 0.824 |

Self-generated CoC scores marginally *better* on both metrics, but the gap is 2 events
out of 60 (49 vs 51 correct pairs) -- well within the noise band already seen elsewhere
in this doc (e.g. the n=50 -> n=60 checkpoint move in Experiment 1 shifted pairwise-acc
by a similar amount). Read as **no evidence of either a self-confirmation bias or a
self-skepticism bias** in the yes/no confidence signal at this scale -- an initial n=10
smoke test had actually suggested the opposite (self-CoC pairwise-acc 0.700 vs gold's
0.800), which didn't hold up at n=60 and is flagged here as a reminder that n=10 reads in
this whole doc should be treated as directional, not conclusive.

## Experiment 6 -- train vs. val split

PAI-AV's OOD reasoning parquet has both a `train` split (~1450 clips) and the `val` split
used everywhere else in this doc (~290 clips). Re-ran Experiment 1's baseline method
(`examples/motion_confidence_smoke.py --split train`) on 60 train-split events (same
seed/skip/sampling scheme as the val n=60 run, just pointed at the other split) to check
whether calibration differs between the two.

| split | mean conf (pos) | mean conf (neg) | pairwise acc (pos>neg) | AUROC |
|---|---|---|---|---|
| val (established) | 0.228 | 0.084 | 0.817 (49/60) | 0.797 |
| train | 0.310 | 0.125 | 0.733 (44/60) | 0.746 |

Train split is modestly **worse** on both metrics -- a bigger gap than Experiment 5's
(5 events out of 60, not 2), though still a single n=60 draw rather than a confirmed
effect. This is the opposite of what memorization/overfitting would predict (better
calibration on data the model may have seen in training), so it's more likely explained
by train/val simply covering a different mix of scene difficulty than by anything
train-specific in the model's behavior -- not investigated further here. Worth another
n=60 draw with a different seed before treating the gap as real.

## Experiment 7 -- K=3 negatives per event + independent negative-QA (val vs. train)

Every experiment above used exactly 1 mismatched-gold-CoC negative per event -- a coin-flip
sized sample that can't distinguish "the method usually ranks positive above negative" from
"the method got lucky/unlucky on which negative was drawn." `examples/motion_confidence_smoke.py`
now supports `--num_negatives K`: builds K negatives per event via cyclic shifts by
`1..K`, and reports both a flattened `pairwise_acc` (all N*K positive-vs-negative
comparisons) and a stricter `top1_acc` (positive must beat ALL K negatives -- the
practically-relevant "would a verifier pick the right one out of K+1 candidates" metric).
Ran with K=3 on the full val set (n=289) and on the same 300-event train draw used in
Experiment 6.

**Val (289/289 events, no data-loading failures):**

| method | pairwise_acc | top1_acc | AUROC | mean_pos | mean_neg |
|---|---|---|---|---|---|
| baseline (yes/no) | 0.839 | **0.661** | 0.796 | 0.262 | 0.078 |
| digit (0-9) | 0.682 | **0.464** | 0.640 | 0.525 | 0.428 |
| likelihood (mean logprob) | 0.616 | **0.374** | 0.616 | -4.33 | -4.97 |

**Train (300 events attempted, 219/300 usable -- see data-loading caveat below):**

| method | pairwise_acc | top1_acc | AUROC |
|---|---|---|---|
| baseline (yes/no) | 0.831 | **0.676** | 0.792 |
| digit (0-9) | 0.680 | **0.434** | 0.660 |
| likelihood (mean logprob) | 0.653 | **0.416** | 0.637 |

Train and val are consistent with each other on all three methods and all three metrics
(within a few points), and `pairwise_acc` on both splits roughly matches the earlier n=60
K=1 numbers (0.817/0.700/0.550 for baseline/digit/likelihood) -- a good cross-check that
K=3 isn't measuring something different from K=1. The important new finding is the gap
between `pairwise_acc` and `top1_acc`: baseline's pairwise_acc (0.84) looks strong, but its
top1_acc (0.66-0.68) means roughly 1 in 3 events has the positive action losing to *at
least one* of 3 negatives -- and digit/likelihood's top1_acc (0.37-0.46) are barely above
the K=3 random-chance rate of 25% (1 of 4 candidates), much weaker than their pairwise_acc
(0.62-0.68) alone would suggest. **Single-negative pairwise comparisons substantially
overstate how reliable these methods are as a verifier/reranker.**

**Data-loading caveat (train run only):** 111/300 train events initially failed to load
with `zipfile.BadZipFile: File is not a zip file` -- `physical_ai_av`'s streaming loader
reads uncached chunk files directly over HTTP without local caching, so it's vulnerable to
truncated reads under heavy concurrent load on the shared cluster (this investigation had
several GPU jobs running simultaneously at the time). A sequential retry pass recovered
30/111; 81/300 (27%) were permanently unrecoverable after 3 attempts each, leaving n=219
usable. This did not occur on the val run (0/289 failures) or the original sequential
train n=300 baseline run (0/300 failures) -- it's specifically a concurrency artifact, not
a data-quality issue with these particular train events. Worth keeping in mind for any
future heavily-parallel run against this dataset: either avoid streaming reads under heavy
concurrent load, or add the same retry-with-backoff-on-any-exception pattern used to fix
this (the failure surfaces as `zipfile.BadZipFile`, not just HTTP 429s, so string-matching
specific error types is not sufficient).

**Independent negative-QA (Qwen3-VL-32B-Instruct, neutral prompt, `examples/motion_confidence_negative_qa.py` v1 methodology -- single greedy call per candidate):**

| split | judged | No (confirmed negative) | Yes (flagged, false negative) | Uncertain |
|---|---|---|---|---|
| val | 867/867 (0 load failures) | 708 (81.7%) | 159 (18.3%) | 0 |
| train | 267/657 judged (390 lost to the same load-failure issue above, not model-output parse failures) | 216 (80.9% of judged) | 51 (19.1% of judged) | 0 |

Train and val agree closely once load failures are excluded (~19% of cyclically-shifted
"negatives" are actually still plausible actions for their scene on both splits, dominated
by generic transferable phrasings like "steer right following temporary traffic
delineators" -- see the full flagged-case list from the val run for examples). This means
roughly 1 in 5 of the K=3 negatives above is mislabeled, which should be read as label
noise inflating or deflating the K=3 metrics above by a similar margin, not as a
methodology validated as 100% clean.

*Process note:* the train negative-QA run was launched using the original (v1)
`motion_confidence_negative_qa.py`, but that script was concurrently rewritten mid-run by
a separate task building the CoT+majority-vote reliability improvement (see Experiment 8's
follow-up below). To keep this val/train comparison apples-to-apples, the train run used a
reconstructed copy of the original v1 script rather than the new v2 judge -- a coordination
near-miss worth remembering: concurrent background jobs sharing this repo's `examples/`
scripts can clobber each other mid-run if their file sets overlap.

## Experiment 8 -- independent VLM-judge QA of self-generated CoC correctness (smoke test)

Experiment 5 assumed the model's self-generated CoC is "correct by construction" for its
own scene -- this was flagged as a real gap (neither factual nor faithful calibration, see
Open Questions) since it was never independently checked. This experiment starts closing
that gap using the same independent-judge methodology already validated on the K=3
negative-QA check above (Experiment 7): **Qwen3-VL-32B-Instruct** (`Qwen/Qwen3-VL-32B-Instruct`, HF
`transformers`, loaded separately from Alpamayo 2 Super) looks at the same 6-camera "vqa"
current-frame scene images and answers a neutral prompt that never reveals the gold CoC or
that the candidate is model-generated: *"Given this driving scene, is the following a
reasonable and correct driving decision for the ego vehicle right now: '\<candidate\>'?
Answer with exactly one word first -- Yes, No, or Uncertain -- then a brief one-sentence
explanation."*

Script: `examples/motion_confidence_self_coc_qa.py` (direct adaptation of
`examples/motion_confidence_negative_qa.py`, same judge/image pipeline, swapped input
schema). Input: `outputs/motion_confidence_multi_coc_val_n289.json` (289 val events x 8
independently-sampled self-CoCs each, temperature=0.6/top_p=0.98, from
`examples/motion_confidence_multi_coc.py`). **Smoke test scope: first 8 events only, all 8
self-CoCs each = 64 judge calls**, saved to
`outputs/motion_confidence_self_coc_qa_smoke.json`. Not yet scaled to the full 289+300 --
see reliability caveat below.

**Overall: 56 Yes / 8 No / 0 Uncertain / 0 unparseable (87.5% judged reasonable-for-scene).**

| clip_id | t0_us | Yes | No | Uncertain |
|---|---|---|---|---|
| `f05e9c15-9385-4520-a301-81083e5567c8` | 9399047 | 3 | 5 | 0 |
| `f538232c-afa6-4e64-b987-6a1cd697d1fc` | 1700000 | 8 | 0 | 0 |
| `420444f5-b285-4e2c-b3e3-7df4ae67e674` | 7260218 | 7 | 1 | 0 |
| `6fcdb507-34b9-4cdc-8b9a-526c80dfe0d7` | 2024879 | 8 | 0 | 0 |
| `02a5f8b3-967f-4db2-9d7f-4291d1be76c6` | 3878014 | 8 | 0 | 0 |
| `f5c066d7-0174-4f0d-a50b-0e0a25d601f5` | 9215923 | 8 | 0 | 0 |
| `52b4287f-1c10-4de9-91f0-13047ec412c2` | 6586745 | 6 | 2 | 0 |
| `f0c3e255-920f-4c3e-9eda-90e161aa8b1a` | 10185868 | 8 | 0 | 0 |

Full per-sample records (self_coc text, verdict, explanation) for all 64 calls are in
`outputs/motion_confidence_self_coc_qa_smoke.json`; exact texts for the three
noteworthy events below are reproduced in full so they can be checked back against the
scene directly (clip_id/t0_us above is enough to reload the scene via
`load_physical_aiavdataset(clip_id, t0_us=t0_us)` + `select_task_input(source, "vqa")`).

**Genuinely useful catch -- `52b4287f-1c10-4de9-91f0-13047ec412c2`, t0=6586745 (6 Yes / 2 No):**
the 8 self-CoCs disagree with each other about scene content, and the judge's verdicts
track that disagreement sensibly, not noise:
- `[0]` Yes -- "Keep distance to the lead vehicle because traffic ahead controls speed." (judge: traffic ahead is stopped/slow at an intersection, consistent)
- `[1]` Yes -- "Keep at the center of the lane since no immediate hazard affects the path."
- `[2]` Yes -- "Keep at the center of the lane because no immediate obstacle affects the lane"
- `[3]` **No** -- "Keep distance to the lead vehicle because a vehicle is ahead in the same lane." (judge: "the ego vehicle is not following a lead vehicle in the same lane; it is at an intersection with traffic signals... no immediate vehicle directly in front")
- `[4]` **No** -- "Keep distance to the lead vehicle because it is directly ahead in the same lane." (same judge reasoning as `[3]`)
- `[5]` Yes -- "Keep distance to the lead vehicle since traffic ahead sets the pace."
- `[6]` Yes -- "Keep at the center of the lane because no critical component affects the path."
- `[7]` Yes -- "Keep distance to the lead vehicle because traffic ahead sets the pace."

Two of the 8 self-generated CoCs assert a specific claim ("a vehicle directly ahead in the
same lane") that the judge -- looking at the same images -- says isn't supported by the
scene, while the other 6 (a mix of "keep lane, no hazard" and "traffic ahead sets pace,"
softer/vaguer claims) were accepted. This is exactly the kind of self-CoC correctness gap
Experiment 5 could not detect on its own.

**Reliability concern -- `f05e9c15-9385-4520-a301-81083e5567c8`, t0=9399047 (3 Yes / 5 No):**
all 8 self-CoCs are near-paraphrases of "keep distance to the lead vehicle ahead," yet the
judge split 3/5 on essentially the same claim:
- `[0]`, `[1]`, `[6]` **No**, identical text "Keep distance to the lead vehicle because it is
  directly ahead in the same lane." -- judge: "the ego vehicle is not directly behind a
  lead vehicle... navigating a multi-lane urban street... lateral movement or lane
  positioning rather than following a single lead vehicle."
- `[2]`, `[3]` **No**, "...because it is slowing ahead" / "...because it slows ahead" -- same
  "no clear lead vehicle" reasoning.
- `[4]` Yes -- "...because the lane ahead is congested." `[5]` Yes -- "Decelerate due to the
  lead vehicle ahead" (judge here says "following a black BMW closely"). `[7]` Yes --
  "...because it is directly ahead controlling the lane speed" (same BMW reasoning).

Note samples `[0]`/`[1]`/`[6]` are *word-for-word identical* text yet get the same verdict
(consistent within-text), while different phrasings of the same underlying claim split
Yes/No depending on wording ("congested"/"BMW" cues push Yes; "directly ahead"/"slowing"
without a named vehicle push No). Combined with `420444f5` below (near-identical
"lane narrows for roadworks" (No) vs. "lane narrows for construction" (Yes) pair), this
looks like some genuine phrasing/wording sensitivity in the judge rather than pure noise,
but at n=8 events it isn't possible to cleanly separate "judge is picking up on real
distinctions in what's visually supported" from "judge is unreliable at this
granularity" -- flagged as an open question below.

**Read:** explanations are consistently specific to visual details (naming vehicle colors,
traffic lights, cones, trams, etc.), which is reassuring that the judge is grounding in
the images rather than pattern-matching the action text alone, and there were zero
scene-load failures or unparseable outputs. But the within-event verdict splits on
near-identical text mean the raw Yes/No counts should not yet be treated as ground truth
correctness labels -- see Open Questions for the two proposed fixes (hand-audit a slice of
"No" verdicts, or multi-sample/majority-vote the judge itself) before scaling this to the
full 289 val + 300 train event sets.

### Follow-up: CoT-before-verdict + 3-sample majority vote -- fixes noise, but at a cost

Implemented both reliability fixes proposed above (`examples/motion_confidence_judge_common.py`,
shared by both QA scripts): (1) the judge now reasons before committing to a verdict
(final line must be exactly Yes/No/Uncertain, `max_new_tokens` raised 96->512 since
truncated reasoning was silently losing ~18% of verdicts at 384) instead of
verdict-first-then-explanation, and (2) 3 samples per candidate (`do_sample=True`,
temperature=0.7, top_p=0.95) with a 2-of-3 majority vote (all-3-distinct ->
`"no_majority"`), replacing the single greedy call. Re-ran the identical 8-event/64-candidate
smoke test with 3x sampling (192 judge calls total) -- `outputs/motion_confidence_self_coc_qa_smoke_v2.json`.

Overall majority-verdict counts: **56 Yes / 8 No / 0 Uncertain (old, n=64) -> 64 Yes / 0 No /
0 Uncertain / 0 no_majority (new, n=64)**. Both previously-flagged noisy events are now
internally consistent: `f05e9c15` (old 3 Yes/5 No on word-for-word identical text) is now
8/8 Yes with unanimous 3/3 raw votes on the repeated text; `420444f5` (old 7 Yes/1 No lone
dissenter on a "roadworks"/"construction" paraphrase pair) is now 8/8 Yes.

**This is not a clean win -- it looks like a lenience trade-off, not just a noise fix.**
`52b4287f`, the doc's one "genuinely useful catch" above (old judge correctly caught two
self-CoCs' unsupported claim of "a vehicle directly ahead in the same lane"), *also*
flipped to a unanimous 8/8 Yes under the new judge. Its new CoT reasoning hedges around
the factual claim rather than checking it directly (e.g. "There is no indication that the
lead vehicle is stopping suddenly... even if no immediate hazard is present... Yes")
instead of noting the same-lane claim isn't supported. Across the full 192 raw (pre-vote)
samples, only 3 were "No" and 1 "Uncertain" -- i.e. the new prompt/sampling setup pushed
the judge toward Yes far more broadly than just on the two noisy events it was meant to
fix. Whether this is caused by the CoT reasoning itself (giving the judge room to
rationalize toward Yes), the temperature-0.7 sampling, or both, is not yet disentangled.
At n=8 events this can't be resolved with confidence -- **do not treat the new judge as a
strict improvement without checking discriminative power (can it still catch cases like
`52b4287f`?) on a larger, ideally judge-verdict-diverse, sample before adopting it for the
full-scale run.**

### Addendum -- independent visual spot-check of the Qwen judge itself (n=8)

Everything above cross-checks Qwen against Qwen (v1 vs. v2 judge design, internal
consistency on repeated text) -- none of it independently verifies Qwen's Yes/No verdicts
are actually *correct* for the scene. To close that gap, exported the front-wide-camera
still for 8 events (`examples/export_scene_images.py`, `outputs/motion_confidence_images/qa_spotcheck/`)
spanning the two flagged Experiment 8 events (`52b4287f`, `f05e9c15`), 3 negative-QA cases
Qwen flagged "Yes" (plausible negative), and 3 negative-QA "No" (confirmed-negative) cases,
and read each image independently (blind to Qwen's verdict) before comparing.

**Result: 6/8 agree, 1/8 clear disagreement, 1/8 ambiguous.** *(Correction, see Addendum 2
below: the "near-identical scenes" claim in the next paragraph is factually wrong -- these
are two visually distinct clips, one daytime Stuttgart, one foggy night SF. Left as
originally written for the record; do not take the scene-similarity framing at face value.)*
The disagreement is concrete
and worth flagging on its own: `f05e9c15` (t0=9399047) and `f0c3e255` (t0=10185868) are
visually near-identical scenes (foggy night, SF tram tracks, a car directly ahead with
brake lights lit, pedestrians crossing, a STOP sign on the cross street) -- Qwen judged
`f0c3e255`'s "decelerate to maintain a safe distance from the lead vehicle ahead" as **Yes**
("the front camera shows a vehicle directly ahead with its brake lights illuminated"), but
judged `f05e9c15`'s near-identical "keep distance to the lead vehicle... directly ahead in
the same lane" as **No** ("the ego vehicle is not directly behind a lead vehicle... lateral
movement or lane positioning rather than following a single lead vehicle") -- despite the
same visual signature (a lit-up car directly ahead on the tracks) being present in both. The
ambiguous case (`f0fe5727`, judged No on "traffic cones on the right") has visible
roadwork-style barriers in the scene that arguably support the claim, but they're barrier
fencing rather than unambiguous traffic cones, so it's not a clean-cut error either way.

**Caveat on this check itself:** Qwen was shown all 6 "vqa" cameras; this spot-check only
used the exported front-wide still, so a disagreement could in principle be explained by
context visible in a different camera that isn't in front. But `f05e9c15` vs. `f0c3e255`'s
near-identical front views getting opposite verdicts on the same claim type is hard to
explain by a side/rear camera alone.

**Bottom line:** at this n=8 sample, Qwen's verdicts are directionally useful (75% clean
agreement) but not reliable enough to treat as ground-truth correctness labels on their
own -- consistent with, and now backed by an actual independent check rather than just
qualitative reading, the ~19% mislabel rate estimated in Experiment 7 and the phrasing
sensitivity flagged in Experiment 8. Before using Qwen verdicts as a correctness signal at
scale, this spot-check should be extended to a larger, randomly-sampled set (not
hand-picked flagged cases as done here, which biases toward interesting disagreements) to
get an unbiased accuracy estimate -- this is a concrete, low-cost way to make progress on
Open Question 1/3 below, and arguably a prerequisite before either the CoT+majority-vote
judge or self-consistency-based correctness proxies are trusted at scale.

### Addendum 2 -- independent Claude-vision labeling (3-camera, 1-frame, n=10)

The Addendum above is itself only a small, hand-picked spot-check of the Qwen judge, and it
relied on a single external judge (Qwen) plus qualitative reading of one exported still per
event. To get a second, fully independent correctness signal, Claude directly labeled all 8
self-generated CoCs for 10 events (**80 verdicts total**) using its own vision -- but on a
deliberately reduced input: only the 3 forward cameras (front-left/front-wide/front-right)
and only the current frame (no history, no side/rear cameras), vs. the model's and Qwen's
full 6-camera x 4-frame "vqa" conditioning. Images: `outputs/motion_confidence_images/front3_qa/`
(`examples/export_scene_images.py`); full labels + one-sentence justifications (and a
`caveat` field wherever the verdict hinged on something a single static frame can't fully
resolve, e.g. whether a vehicle is slowing or starting to move) are in
`outputs/motion_confidence_claude_vision_labels_n10.json`.

**Overall: 74 Yes / 6 No (92.5% judged reasonable-for-scene)** -- noticeably more lenient
than Qwen v1 (87.5% on its 8 events) and close to Qwen v2's 100%. Events 0-7 (indices 0-7)
are the *exact same 8 events* Qwen v1/v2 judged in Experiment 8, enabling a direct 3-way
comparison (Claude vs. Qwen-v1 vs. Qwen-v2) on the same 64 candidates; events 8-9 are new
(clips `1b818d7e` and `83753d4f`) and have no Qwen comparison.

**3-way agreement on the 8 overlapping events (64 candidates):**

| Comparison | Agreement |
|---|---|
| All three (Claude = Qwen-v1 = Qwen-v2) | 54/64 (84.4%) |
| Claude vs. Qwen-v1 | 55/64 (85.9%) |
| Claude vs. Qwen-v2 | 61/64 (95.3%) |
| Qwen-v1 vs. Qwen-v2 (for reference) | 56/64 (87.5%) |

Claude's verdicts track the lenient v2 (CoT + majority-vote) judge much more closely than
the stricter v1 (single greedy call) judge -- consistent with the doc's existing worry that
v2's reliability fix is "a lenience trade-off, not just a noise fix" (see the Follow-up
note above): an independent human-like read of the same reduced scene also lands closer to
"mostly plausible" than v1's noisier, more skeptical verdicts.

**Per-event breakdown (all-3-agree / Claude-vs-v1 / Claude-vs-v2, out of 8):**

| clip_id | all-3 agree | Claude vs v1 | Claude vs v2 |
|---|---|---|---|
| `f05e9c15-9385-4520-a301-81083e5567c8` | 3/8 | 3/8 | 8/8 |
| `f538232c-afa6-4e64-b987-6a1cd697d1fc` | 8/8 | 8/8 | 8/8 |
| `420444f5-b285-4e2c-b3e3-7df4ae67e674` | 7/8 | 7/8 | 8/8 |
| `6fcdb507-34b9-4cdc-8b9a-526c80dfe0d7` | 6/8 | 6/8 | 6/8 |
| `02a5f8b3-967f-4db2-9d7f-4291d1be76c6` | 8/8 | 8/8 | 8/8 |
| `f5c066d7-0174-4f0d-a50b-0e0a25d601f5` | 8/8 | 8/8 | 8/8 |
| `52b4287f-1c10-4de9-91f0-13047ec412c2` | 6/8 | 7/8 | 7/8 |
| `f0c3e255-920f-4c3e-9eda-90e161aa8b1a` | 8/8 | 8/8 | 8/8 |

**Correction to the Addendum above: `f05e9c15` and `f0c3e255` are *not* visually
near-identical.** The Addendum's "clear disagreement" writeup describes both clips as
"foggy night, SF tram tracks, a car directly ahead with brake lights lit, pedestrians
crossing, a STOP sign on the cross street." Having now looked directly at
`outputs/motion_confidence_images/qa_spotcheck/c2_f05e9c15.png` and re-exported
`front3_qa` stills for the same clip_id/t0_us, `f05e9c15` (t0=9399047) is actually a
**daytime Stuttgart street scene**: a red hatchback directly ahead in the ego's lane, a
yellow U7 tram (signed "Ostfildern") on separate tracks to the left, and a delivery van
further ahead -- no fog, no night, no pedestrians, no STOP sign. Only `f0c3e255`
(t0=10185868, event 7 here) is the foggy-night-SF-with-pedestrians-and-STOP-sign scene the
Addendum described. This looks like the Addendum's write-up conflated the two clips'
descriptions (likely from only visually inspecting one of the two stills closely) rather
than a real "two near-identical scenes get opposite verdicts" finding -- **the two scenes
are visually distinct, so the disagreement between the two clips' verdicts is far less
surprising than the Addendum suggested.** That said, Qwen v1's own "No" reasoning for
`f05e9c15` ("navigating a multi-lane urban street with multiple vehicles and a tram... not
directly behind a lead vehicle") is a reasonably accurate description of the actual
Stuttgart scene -- it just reaches the wrong verdict: the red hatchback plausibly *is* the
ego's lead vehicle in the same lane, tram notwithstanding, which is why Claude judged all
8 of `f05e9c15`'s candidates Yes.

**Most interesting disagreements:**

- **`52b4287f`, candidates `[3]` vs `[4]` (the doc's "genuinely useful catch" event):**
  Qwen v1 called both "a vehicle is ahead in the same lane" `[3]` and "it is directly ahead
  in the same lane" `[4]` **No** (reasoning the intersection is empty); Qwen v2 flipped
  both to **Yes**. Claude split them: `[3]` **Yes** (a sedan and an SUV are visible well
  down the same wide-open road, so "a vehicle is ahead in the same lane" is literally
  true) but `[4]` **No** ("directly ahead" overstates how close those distant vehicles
  actually are -- there is no vehicle immediately in front of the ego). Neither Qwen pass
  drew this same distinction: v1 rejected both on the same reasoning, v2 accepted both on
  the same reasoning. This is a concrete data point that the "phrasing sensitivity" flagged
  repeatedly in Experiment 8 can reflect a real, defensible visual distinction ("a vehicle
  is ahead" vs. "directly ahead") rather than pure judge noise -- worth keeping in mind
  before writing off wording-sensitive verdict splits as unreliable.
- **`6fcdb507`, candidates `[2]`/`[5]`, "it begins moving ahead in the same lane":** both
  Qwen passes judged this Yes for all 8 candidates on this event, but Claude judged these
  two specifically **No**: the lead van directly ahead has its brake lights lit in the
  image, which reads as currently braking/stopped rather than "beginning to move" -- a
  cue visible in the single static frame that neither Qwen pass's explanation mentioned
  weighing. (Flagged with a caveat regardless, since confirming an actual stopped-to-moving
  transition would need multiple frames to be certain.)
- **`f05e9c15`, candidates `[0]`/`[1]`/`[2]`/`[3]`/`[6]`:** Qwen v1 judged 5 of this event's
  8 candidates No (the "directly ahead"/"slowing" claims); Claude judged all 8 Yes, seeing
  a clear lead vehicle (red hatchback) directly in the ego's lane in the Stuttgart scene
  described above. Qwen v2 agrees with Claude here (8/8 Yes).

**Reduced-input caveat -- where 3-camera/1-frame actually mattered:** most of the 80
verdicts didn't hinge on the missing cameras or history frames -- the great majority of
self-CoCs describe static scene facts (a red light, cones, a lead vehicle's presence, a
truck protruding into the lane) that a single forward-facing frame settles cleanly. The
cases flagged with a `caveat` in the output JSON are exactly the ones asserting *motion*
that a static frame can't fully verify: `f05e9c15` `[2]`/`[3]` ("slowing ahead"),
`6fcdb507` `[0]`/`[1]`/`[4]`/`[7]` ("moving slowly"/"moving ahead") and `[2]`/`[5]`
("begins moving," addressed above), and `1b818d7e` `[0]`/`[1]`/`[5]` ("pulling out"/
"entering" from the left -- Claude could confirm the truck was angled into the lane from a
single frame, but not whether it was actively in motion doing so). In every one of these
cases Claude still gave a definite Yes/No best guess (per the task framing) rather than an
"Uncertain" cop-out, but the true positive/negative label for these ~9/80 candidates should
be treated as less certain than the rest. No missing side/rear camera was ever the decisive
factor in a verdict here, unlike the Addendum's (now-corrected) concern about `f05e9c15`.

**Bottom line:** a third, fully independent judge (Claude vision, reduced input) lands at
84% full 3-way agreement and 95% agreement with the lenient Qwen v2 judge on the same 64
candidates -- directionally reinforcing that most self-CoCs are visually plausible, while
also reproducing the same kind of phrasing-sensitive, borderline verdict splits seen in
Qwen's own runs (see `52b4287f` above). It also caught and corrected a factual error in
this doc's own Addendum (the `f05e9c15`/`f0c3e255` "near-identical scenes" claim), which is
itself a useful reminder that qualitative spot-check writeups need to be checked against
the actual images before being treated as established findings.

## Experiment 9 -- scene-conditioned method + temporal-jitter sweep at K=3 (val)

Experiment 7 ran K=3 negatives at full val scale (n=289) but only for
baseline/digit/likelihood -- `scene_conditioned` was left out purely for cost reasons.
Separately, Experiments 2/4 showed the raw yes/no signal is very sensitive to which exact
frame is used, on a single clip. This experiment closes both gaps at once: adds
`scene_conditioned` to the K=3 val comparison, and sweeps t0 by
`--t0_offset_s in [-0.2, -0.1, 0.0, 0.1, 0.2]` (same loader 0.1s frame grid as Experiments
2/4) for all 4 methods, all at n=289, K=3. `examples/motion_confidence_smoke.py`'s existing
`--t0_offset_s` flag does the shift (clamped to `MIN_T0_US=1,700,000`, recorded per-event as
`load_t0_us` so clamping is auditable); same positive/negative gold-CoC pairing as
Experiment 7 (cyclic shift, same seed) at every offset -- only the scene images change.
10 jobs (5 offsets x 2 shards of 145/144 events), aggregated with the new
`examples/motion_confidence_aggregate_toff.py`. **0/289 data-loading failures at every
offset** (`load_scene_with_retry` held up under 8-way concurrent GPU load this time, unlike
the train-split concurrency failure in Experiment 7).

**Clamping check**: `--t0_offset_s -0.1` clamped 36/289 events to `MIN_T0_US` (no real
shift happened for those events -- their `load_t0_us` at -0.1s equals the 0.0s
`load_t0_us`), and -0.2 clamped 38/289 -- both counts confirmed exactly against the actual
output, matching the pre-launch estimate. Positive offsets clamped 0/289. So the -0.1/-0.2
columns below are diluted by ~12-13% unshifted events, which should bias them to look
*more* similar to the 0.0 column than a fully-shifted set would -- a real caveat on reading
the negative-offset rows as "the same test" as the positive-offset rows.

**Reproducibility cross-check**: the offset=0.0 row below reproduces Experiment 7's val
numbers exactly (baseline 0.839/0.661/0.796, digit 0.682/0.464/0.640, likelihood
0.616/0.374/0.616) -- same code path, same seed, `--t0_offset_s 0.0` is a no-op by
construction, so this is confirming no regression rather than new evidence, but it is a
clean pass, not a red flag.

| offset | baseline pw/top1/AUROC | digit pw/top1/AUROC | likelihood pw/top1/AUROC | scene_conditioned pw/top1/AUROC |
|---|---|---|---|---|
| -0.2s | 0.814 / 0.626 / 0.773 | 0.659 / 0.405 / 0.625 | 0.615 / 0.374 / 0.614 | 0.802 / 0.619 / 0.731 |
| -0.1s | 0.828 / 0.647 / 0.786 | 0.651 / 0.422 / 0.631 | 0.624 / 0.391 / 0.615 | 0.824 / 0.654 / 0.742 |
| 0.0s  | 0.839 / 0.661 / 0.796 | 0.682 / 0.464 / 0.640 | 0.616 / 0.374 / 0.616 | 0.824 / 0.668 / 0.750 |
| +0.1s | 0.835 / 0.657 / 0.796 | 0.687 / 0.464 / 0.643 | 0.614 / 0.367 / 0.616 | 0.807 / 0.647 / 0.746 |
| +0.2s | 0.836 / 0.657 / 0.799 | 0.679 / 0.464 / 0.645 | 0.623 / 0.374 / 0.620 | 0.821 / 0.664 / 0.751 |

(pw = pairwise_acc, top1 = top1_acc; n=289 every row, 0 data-loading errors, all four
methods scored on identical positive/negative action text per event, only the scene images
shift.)

**Headline: at aggregate n=289 scale, all four methods are remarkably stable across a
+-0.2s frame shift** -- baseline's pairwise_acc/top1_acc/AUROC each move by only
0.02-0.04 across the whole 5-point sweep, and scene_conditioned by a similar amount
(0.02-0.05). This is a real update relative to Experiments 2/4, which found up to a 0.90
swing in a single action's raw confidence on **one** clip across the same +-0.3s window --
that per-clip volatility is still presumably happening here too (nothing about the
per-event mechanics changed), it just washes out when 289 events' rankings are pooled into
one summary statistic. **The two findings aren't in tension: raw per-clip confidence can
swing wildly with frame choice while an aggregate ranking metric over hundreds of events
stays nearly flat**, because the swings aren't systematically signed -- a good reminder
that per-event confidence numbers from this signal should not be trusted in isolation even
though the population-level verifier accuracy looks robust to this nuisance axis.

**scene_conditioned is below baseline at every single offset** on all three metrics
(pairwise_acc 0.80-0.82 vs 0.81-0.84, top1_acc 0.62-0.67 vs 0.63-0.66, AUROC 0.73-0.75 vs
0.77-0.80) -- Experiment 1's "no clear win" finding replicates across the whole jitter
sweep, not just at the single offset Experiment 7 checked. digit and likelihood remain
clearly worse than both at every offset, also replicating.

**Does this support or contradict Experiment 4's "scene-conditioning damps positive-action
variance but worsens negative-action variance" finding?** Taking the aggregate mean_pos/
mean_neg (analogous to, but not the same measurement as, Experiment 4's single-clip
positive/negative confidence):

| | mean_pos range across 5 offsets | mean_neg range across 5 offsets |
|---|---|---|
| baseline | 0.024 (0.247-0.271) | 0.001 (0.078-0.079) |
| scene_conditioned | 0.007 (0.159-0.166) | 0.001 (0.048-0.049) |

The positive-side result **replicates Experiment 4's direction**: scene-conditioning's
mean positive confidence is 3.4x more stable across offsets than baseline's (range 0.007
vs. 0.024) -- consistent with "language-sampling damps positive-action temporal
sensitivity." But Experiment 4's other finding, that scene-conditioning *worsens*
negative-action variance, does **not** show up here -- both methods' mean_neg are equally
flat (range 0.001 each). The likely explanation: Experiment 4 tracked one fixed negative
action across offsets on one clip, where a single bad scene-description sample can swing
the mean a lot; here every event has a different (cyclically-shifted) negative, and
289 such per-event idiosyncrasies average out in the pooled mean. Aggregate stability and
single-clip stability are different questions -- both experiments' findings stand, they're
just not measuring the same thing.

## Experiment 10 -- full self-CoC candidate scoring (raw scores, positive/negative pairing deferred)

Experiment 5/8 use one self-generated CoC per event as "the" model-generated candidate.
`outputs/motion_confidence_multi_coc_val_n289.json` (from `motion_confidence_multi_coc.py`)
already has **8** independently-sampled self-CoCs per val event (temperature 0.6, top_p
0.98) sitting unused for anything beyond Experiment 8's n=8-event judge smoke test. This
experiment scores all 8 candidates/event through all 4 methods
(`examples/motion_confidence_multi_coc_scores.py`) -- per-event images and (for
scene_conditioned) the 5 sampled scene descriptions are loaded/generated **once** and
reused across all 8 candidates, mirroring how `motion_confidence_smoke.py` reuses one image
load across positive+negatives. **No positive/negative pairing is chosen here** -- which of
the 8 samples (if any) should count as "the" positive, and what a matched negative should
look like, is an open, deliberately-deferred decision (see Open Questions) -- so this is
descriptive only, not another pairwise-acc/AUROC table. 2 jobs (145/144-event shards),
merged with the new `examples/motion_confidence_aggregate_multi_coc_scores.py` into
`outputs/motion_confidence_multi_coc_scores_val_n289_merged.json` (the actual deliverable:
289 events x 8 candidates x 4 methods' raw scores, for whatever pairing analysis comes
next). **0/289 data-loading failures.**

**Descriptive summary: within-event spread of each method's score across the 8 self-CoC
candidates**, averaged over 289 events (max-min and population std; NOT a correctness
metric -- a method that gives 8 near-paraphrases 8 near-identical scores would show a small
spread here regardless of whether that score is right):

| method | mean spread (max-min) | mean std |
|---|---|---|
| baseline (yes/no) | 0.238 | 0.084 |
| digit (0-9) | 0.172 | 0.059 |
| likelihood (mean logprob) | 0.791 | 0.272 |
| scene_conditioned | 0.175 | 0.061 |

(likelihood's units are log-probability, not `[0,1]`-bounded like the other three, so its
row isn't directly comparable in scale to the others -- included for completeness, not for
ranking against the bounded methods.) Among the three bounded methods, **baseline assigns
noticeably more within-event disagreement to its 8 candidates than digit or
scene_conditioned do** (spread 0.238 vs. 0.172/0.175) -- consistent in direction with
Experiment 1's finding that digit's answer-mass is more hedged/compressed than baseline's,
now observed as a *within-event* effect across near-identical candidates rather than only
as a *between-event* pos-vs-neg gap.

**Whether that extra spread is "real" or "noise" depends entirely on how different the 8
self-CoCs actually are for a given event -- and that varies a lot.** Worked example, clip
`892eb6e7-83b3-4452-94cd-a49286465682`, t0=3348535 (largest baseline spread in the set,
0.917):

| baseline conf | self-CoC |
|---|---|
| 0.932 | Keep distance to the lead vehicle since it controls the pace through the green intersection. |
| 0.084 | Maintain lane and prepare to change to the right because cones and barriers block access to the right lane through the intersection |
| 0.085 | Nudge left due to the cones narrowing the right side of the lane. |
| 0.867 | Keep distance to the lead vehicle because it controls the pace through the green intersection. |
| 0.959 | Keep distance to the lead vehicle because it is directly ahead controlling our speed. |
| 0.173 | Nudge left due to the cones narrowing the lane on the right. |
| 0.050 | Nudge left due to lane narrowing from cones on the right. |
| 0.967 | Keep distance to the lead vehicle since it is ahead controlling our speed through the green light. |

Here the 8 self-CoCs are **not** paraphrases of one claim -- they're 3 genuinely different
claims (keep-distance-to-lead / nudge-left-for-cones / change-lane-for-barriers), and
baseline cleanly separates them into a high-confidence cluster (0.87-0.97, all
"keep-distance") and a low-confidence cluster (0.05-0.17, the cone/lane-change claims).
This particular high-spread event looks like the method meaningfully discriminating
between different underlying claims, not noise on paraphrases -- which means this
descriptive statistic conflates two very different situations (genuine multi-claim
disagreement vs. paraphrase instability) that would need to be told apart (e.g. by
clustering the 8 texts first) before the spread number alone could be read as "how noisy
is this method," in either direction.

## Experiment 11 -- native CoT-format likelihood scoring

**Motivation.** Experiments 1/7/9/10 established `baseline` (yes/no) as the best of the four
`motion_confidence_smoke.py` methods and `likelihood` as the worst. Before concluding
likelihood-based scoring just doesn't work, it's worth checking whether the existing
`likelihood` method is even asking the model in a format it was trained to answer in. It
isn't: `likelihood` asks an open natural-language question ("What is the right driving
decision for the ego vehicle right now?") with no trajectory-history conditioning, then
teacher-forces the raw candidate-action text with no special tokens -- a format the model
never saw in training. Separately, `baseline`/`digit`/`scene_conditioned` were checked
against the model's real public inference contract for free-form questions --
`text_tasks.prepare_vqa_inputs()` / `build_text_task_messages(task="vqa")`
(`src/alpamayo2_super/text_tasks.py`) -- and build **exactly** the same message skeleton
(image + plain-text question, no special tokens, no trajectory history; the function's own
docstring says "no-special VQA generation"). Those three methods are therefore already
correct and were **not** changed.

The model's real trained CoT-generation pathway (used by `eval_pai_av_val_a2.py` /
`truckdrive_eval_val.py` for actual inference) is different: it conditions on discretized
ego-trajectory-history tokens fused into a `traj_history` placeholder span via
`fuse_traj_tokens`, and it wraps CoT text in `<|cot_start|>`/`<|cot_end|>` special tokens
rather than presenting it as a plain-text VQA answer. `native_likelihood`
(`examples/motion_confidence_native_cot.py`) reproduces that pathway exactly:
`build_conversation(components_order=["image","traj_history","prompt"],
components_prompt=["cot"], generation_mode=True, ...)` for the prefix (drop the empty
trailing assistant turn, tokenize with `add_generation_prompt=True`, then
`fuse_traj_tokens` the real `ego_history_xyz`/`ego_history_rot` into the placeholder span),
then teacher-force the candidate action as the assistant turn via
`construct_cot({"cot": action}, ask_for_component=False)` (tokenized with
`continue_final_message=True`), and score exactly like `score_likelihood()` (prefix-boundary
check, then per-token log-likelihood of everything after the prefix). The script imports and
reuses `score_likelihood`/`LIKELIHOOD_QUESTION` from `motion_confidence_smoke.py` unmodified
so both methods run on identical sampled events for a clean side-by-side comparison, same
cyclic-shift K-negative pairing convention as `motion_confidence_smoke.py`.

**Sanity check (n=3, one GPU, `--debug`)** confirmed all three plumbing risk points named in
the design before scaling up: (1) the decoded prefix text ends in
`...output the chain-of-thought reasoning of the driving process.<|im_end|>\n<|im_start|>assistant\n`
and the decoded full text's appended span is exactly
`<|cot_start|>{action text}<|cot_end|>` -- both special tokens present and in the right
place; (2) `fuse_traj_tokens` ran with no shape errors and no `replace_pad_token`
placeholder-count assertion firing -- `ego_history_xyz`/`ego_history_rot` arrived already
batched `(1, 1, 16, 3)` / `(1, 1, 16, 3, 3)` straight out of `load_physical_aiavdataset`, no
reshaping needed; (3) the prefix-boundary check (`full_ids[:prefix_len] == prefix_ids`)
succeeded on every one of the 3 sanity events, not silently returning `None`.

**Scale-up run: n=25, val split, K=3 negatives, seed 0** (identical event sample to what
`motion_confidence_smoke.py --seed 0` would draw), sharded 1 event-slice per GPU across all
8 available GPUs (`--skip`/`--num_events` partition, same `--seed`) and merged. **0/25
boundary-check failures for either method across all 100 positive+negative comparisons** --
the teacher-forcing boundary check did not silently fail once. As an extra plumbing check,
`native_likelihood`'s appended-token count exceeded `likelihood`'s by exactly 2 in all 100
comparisons (the `cot_start`/`cot_end` wrapper), i.e. the action text itself tokenizes
identically in both contexts with no cross-boundary BPE interference -- the
`action_only_mean_logprob` split (stripping the two special tokens) was identifiable in all
100/100 entries.

| method | n | K | pairwise_acc | top1_acc | AUROC |
|---|---|---|---|---|---|
| likelihood (ad-hoc, existing) | 25 | 3 | 0.427 | 0.040 | 0.581 |
| native_likelihood (full appended span incl. `cot_start`/`cot_end`) | 25 | 3 | 0.400 | 0.000 | 0.553 |
| native_likelihood (action-only span) | 25 | 3 | 0.413 | 0.080 | 0.559 |

**Headline: native_likelihood does not beat the existing ad-hoc likelihood method on the
same 25 events** -- all three variants (ad-hoc, native full-span, native action-only) land
within a few points of each other and of chance (pairwise_acc 0.40-0.43, AUROC 0.55-0.58,
top1_acc a very weak 0.00-0.08 against 3 negatives). Putting the scoring into the model's
real trained CoT-generation format did not surface a cleaner likelihood signal -- the
underlying problem Takeaway 5 already named (every candidate is a real, fluent, human
-written action, so raw text plausibility doesn't track scene-fit well) looks like the
dominant effect regardless of which prompt/token format is used to elicit the
log-likelihood.

**Worked example** (clip `f05e9c15-9385-4520-a301-81083e5567c8`, t0=9399047, positive action
"Decelerate due to the vehicle on the left merging into my lane."):

| action | likelihood (ad-hoc) mean_logprob | native_likelihood mean_logprob (full / action-only) |
|---|---|---|
| positive: Decelerate due to the vehicle on the left merging into my lane. | -3.312 | -6.016 / -3.317 |
| negative_0: Go straight following temporary traffic delineators. | -7.632 | -9.797 / -6.836 |
| negative_1: Steer left following the temporary lane delineated by traffic delineators to avoid the construction zone in the same lane. | -4.716 | -5.602 / -3.967 |
| negative_2: Steer left to maintain a safe distance from the working vehicle on the right. | -4.852 | -5.963 / -3.684 |

Both methods correctly rank the positive above all 3 negatives on this particular event.
The native method's full-span numbers sit systematically lower (more negative) than the
ad-hoc numbers for every action here -- expected, since the two extra low-probability
special tokens (`cot_start`/`cot_end`) are averaged into the full-span mean; the
action-only column strips that out and lands close to the ad-hoc numbers, as it should
since both are now scoring materially the same span of actual action text, just under
different conditioning (trajectory-history tokens present vs. absent, VQA-question framing
vs. fixed CoT-generation instruction).

**Caveats.**
- n=25 is a small-n smoke test (per the user's explicit "start small" instruction, not
  scaled further without checking in first) -- the ad-hoc `likelihood` numbers here
  (pairwise_acc 0.427, AUROC 0.581) are themselves noticeably different from its n=289
  Experiment 7 numbers (pairwise_acc 0.616, AUROC 0.616), which is expected small-n
  variance, not a regression; this table is only informative as a same-n, same-events,
  apples-to-apples comparison between the two likelihood variants, not as a re-estimate of
  either method's population-level accuracy.
- `native_likelihood`'s full-span score is not decision-relevant on its own (it always
  includes 2 low-probability special tokens shared by every candidate of the same event, at
  the same absolute position, so they add a roughly-constant per-token-count-dependent
  penalty rather than discriminating information) -- the action-only variant is the fairer
  comparison to `likelihood`'s bare-text score, and it's the one shown above matching ad-hoc
  most closely, but neither beats ad-hoc.
- This experiment only tried the mean-per-token-logprob reduction (same as the existing
  `likelihood` method, for direct comparability) -- it does not rule out that some other
  reduction (e.g. the PMI correction already flagged in Open Questions, or a length-based
  correction) could do better under the native conditioning specifically; that's untested
  here.

## Experiment 12 -- independent Claude-vision labeling scaled to n=100 (800 candidates)

Addendum 2 (above, under Experiment 8) did this at n=10 (80 candidates) as a first check.
Scaled the identical methodology to all 100 events in `outputs/motion_confidence_multi_coc_val_n289.json`'s
first 100 (800 self-generated CoC candidates total), using 10 parallel subagents (10 events
each) with the same reduced 3-camera/1-frame input, same "judge each candidate fresh, flag
motion-dependent claims with a caveat but still give a Yes/No best guess" instructions as
Addendum 2. Images: `outputs/motion_confidence_images/front3_qa/` (now 100 events, exported
via `examples/export_scene_images_front3.py` -- see caveat below on the export itself).
Merged labels: `outputs/motion_confidence_claude_vision_labels_n100.json` (100 events x 8
labels each, verified no gaps/duplicates in indices 0-99).

**Overall: 672 Yes / 128 No (84.0%)** -- lower than Addendum 2's n=10 read (92.5%), closer to
Qwen v1's 87.5% than Qwen v2's 100%. 160/800 labels (20%) carry a `caveat` flagging a
motion-dependent claim (e.g. "slowing," "beginning to move," "merging") that a single static
frame can't fully verify -- consistent with Addendum 2's finding that this is a real, minor
fraction of cases rather than the norm.

**Per-event spread: 58/100 events are unanimous 8/8 Yes; 0/100 are unanimous 8/8 No; 42/100
are mixed.** This mirrors Experiment 10's finding (using a completely different method --
raw model confidence scores, not correctness labels) that an event's 8 self-CoC samples are
often genuinely different underlying claims, not just paraphrases -- here, independently,
42% of events have at least one candidate judged incorrect while others are judged correct
for the same scene.

**Worst events (fewest Yes out of 8) -- concrete, worth reading if picking a correctness
proxy:**

| index | Yes/8 | clip_id | what went wrong |
|---|---|---|---|
| 11 | 1/8 | `89ba2626-eee2-41c4-b217-b83dac256b25` | 7/8 candidates described an oncoming bus as "stopped and blocking the ego's lane" -- it's an ordinary oncoming bus in the opposite lane. |
| 17 | 1/8 | `9a491c14-7cbc-415e-a5d9-7603f9cf07f3` | 7/8 candidates got the cones' side backwards or invented a merging vehicle/bus not present in the scene. |
| 64 | 1/8 | `0b929729-8240-4685-b22d-68233f52e024` | Candidates' "construction narrows the lane" claim wasn't supported -- the construction material sits set back from the road. |
| 38 | 2/8 | `87935d32-e383-4da5-90f6-09799d885c34` | Candidates described a car alongside blocking a merge / parked trucks obstructing a lane; no such vehicle/obstruction is visible. |

These are genuine, model-side hallucinations (inventing scene details the images don't
support), not judge noise -- each was corroborated by a specific, checkable visual fact (an
oncoming vs. stopped bus, cones on the wrong side, a lane closure that isn't there). This is
the sharpest evidence yet in this doc that a meaningful minority of self-generated CoCs are
not just imprecise but factually wrong about the scene, at a rate (3/100 events at 1/8 Yes,
i.e. the model is confidently, repeatedly wrong across nearly all 8 independent samples) too
high to attribute to labeling noise alone.

**Caveats:**
- **This is one judge (Claude, reduced 3-camera/1-frame input), not ground truth.** Addendum
  2's 3-way comparison (Claude vs. Qwen v1 vs. Qwen v2 on 8 overlapping events) found only
  84% full agreement between all three -- so treat 84.0% Yes as "this particular judge's
  read," not an established correctness rate. No Qwen comparison was run at this n=100 scale
  (Qwen has only judged the first 8 events, per Experiment 8).
- **Export reliability**: the image export hit the same shared-cluster HF Hub rate-limit
  issue documented elsewhere in this doc (`zipfile.BadZipFile` masking an actual 429) --
  fixed by adding retry-with-backoff and `--resume` support to
  `examples/export_scene_images_front3.py` (mirroring `eval_common.load_scene_with_retry`).
  Final export is complete and verified (100/100 events, no gaps).
- Reduced-input caveat carries over unchanged from Addendum 2: 3 cameras + 1 frame is
  narrower than the model's own real conditioning (see "Standing caveat" section near the
  top of this doc) -- some of the 128 No verdicts and 160 caveats could resolve differently
  under the model's full input.
- Each 10-event batch was labeled by an independent subagent with no cross-batch
  consistency check -- unlike Addendum 2's single-agent n=10 pass, verdict thresholds (e.g.
  how strictly "directly ahead" vs. "ahead" is read) may vary slightly batch-to-batch. Not
  quantified here.

## Experiment 13 -- scoring methods against real same-scene correct/incorrect labels

Experiment 12's labels finally let us resolve the pairing question Experiment 10 explicitly
deferred: instead of gold-vs-mismatched-gold-CoC (Experiment 1/7, cross-event negatives that
are almost always trivially wrong for the scene), use Claude-vision's Yes/No verdicts on the
same event's own 8 self-generated candidates as ground truth. This gives **genuinely harder,
same-scene negatives** -- plausible-sounding claims about the *same* scene that are still
factually wrong -- which the Open Questions section had flagged as missing since Experiment 1.
No new model calls needed: joined Experiment 12's labels
(`outputs/motion_confidence_claude_vision_labels_n100.json`) against the raw per-candidate
scores already computed in Experiment 10
(`outputs/motion_confidence_multi_coc_scores_val_n289_merged.json`, first 100 events) by
`(event index, candidate index)` position -- verified 0/100 event-level clip_id/t0_us
mismatches between the two files first.

Only the 42/100 "mixed" events (at least one Yes and one No candidate) contribute a
within-event contrast; the 58 unanimous-Yes events have no negative to compare against and
are excluded (not zero-filled or otherwise assumed). Positive = every Yes-labeled candidate's
score, negative = every No-labeled candidate's score, pooled across all 42 events (208 Yes,
128 No total per method, before per-method `None` filtering for `likelihood`'s occasional
tokenization-boundary skip):

| method | pairwise_acc / AUROC (same-scene labeled, n=42 events) | pairwise_acc (gold cross-event, Experiment 7) |
|---|---|---|
| baseline (yes/no) | 0.606 | 0.839 |
| digit (0-9) | 0.578 | 0.682 |
| likelihood (mean logprob) | 0.615 | 0.616 |
| scene_conditioned | 0.571 | (not run at K=1 in Experiment 1; Experiment 9's K=3 gave ~0.80) |

(pairwise_acc and AUROC are numerically identical here, as expected -- both reduce to the
same Mann-Whitney U statistic for a two-group ranking comparison.)

**Headline: every method drops sharply against real same-scene negatives, and `baseline`'s
large margin over `likelihood` disappears.** `baseline` goes from a clear leader (0.839) to
statistically tied with `likelihood` (0.606 vs. 0.615) -- its strong showing against gold
negatives looks like it was substantially inflated by those negatives being easy (random
other-scene actions), not by `baseline` being especially good at telling a correct decision
from a plausible-sounding *wrong* one in the *same* scene. `likelihood` barely moves at all
(0.616 -> 0.615), consistent with Takeaway 5's observation that it isn't reading off a
specific committed token and so isn't as sensitive to how "easy" the negative is. All four
methods are now much closer to chance (0.5) than any earlier experiment showed -- this is
the hardest test any method has faced in this doc.

**Caveats:**
- n=42 events (208 vs. 128 candidate-level comparisons) is smaller than Experiment 7's
  n=289 -- these numbers should be read as a first, real-but-small-n read on harder
  negatives, not a final replacement for Experiment 7's numbers.
- Ground truth here is Experiment 12's Claude-vision labels, which are themselves an
  imperfect judge (see Experiment 12's own caveats and Addendum 2's 84% 3-way agreement
  with Qwen) -- so some of the "same-scene negative" candidates may in fact be correct
  (mislabeled), same caveat structure as the ~19% gold-negative mislabel rate in Experiment 7.
- Within a mixed event, Yes and No candidates are often near-paraphrases of each other
  differing in one factual detail (e.g. "a vehicle is ahead" vs. "directly ahead," per
  Addendum 2's `52b4287f` finding) -- this is a much finer-grained discrimination task than
  gold-vs-mismatched-gold, which likely explains most of the across-the-board drop, not a
  contradiction of Experiment 7.
- Raw pooled candidate-level data (not just the summary table) is in
  `outputs/motion_confidence_selfcoc_labeled_scoring_n100.json` for anyone who wants to
  re-slice this (e.g. per-event instead of pooled, or restricted to non-caveat labels only).

### Qualitative examples

**1. Easy gold negatives (Experiment 7's K=3, offset=0.0) -- scores across all 4 methods.**
"Easy" describes the negative text (an obviously different action, often for a different
scene entirely), not that every method agrees on it:

| clip_id (t0) | action | baseline | digit | likelihood (mean logprob) | scene_conditioned |
|---|---|---|---|---|---|
| `015df1da...` (3986667) | **pos**: "Go straight following temporary traffic delineators." | 0.999 | 0.947 | -7.013 | 0.998 |
| | **neg**: "Steer left following temporary traffic delineators." | 0.007 | 0.436 | **-6.248** | 0.062 |
| `38044f1c...` (7022644) | **pos**: "Go straight following temporary traffic delineators." | 0.992 | 0.986 | -7.068 | 0.983 |
| | **neg**: "Go straight while maintaining a safe distance from the cut-in vehicle from the right." | 0.023 | 0.625 | **-6.246** | 0.001 |
| `13b8b9db...` (7399343) | **pos**: "Gentle deceleration to maintain a safe distance from the pedestrians on the right side of the road." | 0.961 | 0.406 | -3.702 | 0.982 |
| | **neg**: "Acceleration to proceed through the intersection while keeping a safe distance from the pedestrian on the right." | 0.010 | 0.319 | -5.517 | 0.022 |

The first two rows are a striking illustration of `likelihood`'s specific failure mode:
`baseline` and `scene_conditioned` reject the negative almost perfectly (down to
0.001-0.062), but **`likelihood` actually scores the negative *higher* than the positive**
(bolded) in both cases -- the negative text ("Steer left...", "...cut-in vehicle from the
right") is simply more generic/higher-probability English continuation, regardless of scene
fit. Only the third row (`13b8b9db`) is a genuinely clean sweep where all 4 methods agree
with a wide margin -- and across the full usable n=289 set, strictly-clean sweeps like this
(all 4 methods correctly ranking positive over negative with a wide margin, threshold
gap>0.8 on both `baseline` and `scene_conditioned`) turned out to be rare enough that this
exact search found only 1 such case among all 289 events checked, reinforcing that "easy for
a human to read" does not mean "easy for all 4 methods."

**2. Confident-Yes / confident-No / unsure examples (Experiment 13's same-scene labeled
candidates, `baseline` confidence as the primary axis, all 4 methods shown):**

| verdict | text | baseline | digit | likelihood | scene_conditioned |
|---|---|---|---|---|---|
| Confident Yes | "Accelerate due to green traffic light controlling the intersection" | 0.999 | 0.967 | -1.556 | 0.717 |
| Confident Yes | "Accelerate due to green traffic light." | 0.998 | 0.966 | -1.499 | 0.678 |
| Confident No | "Slow down due to the red traffic light ahead." | 0.003 | **0.509** | -2.435 | 0.001 |
| Confident No | "Yield the right-of-way since a pedestrian is crossing the street." | 0.003 | 0.329 | **-0.654** | 0.001 |
| Unsure | "Keep distance to the lead vehicle because it begins moving ahead in the same lane." (verdict: No) | 0.563 | 0.385 | -0.579 | 0.443 |
| Unsure | "Stop due to red traffic light." (verdict: Yes) | 0.437 | 0.323 | -1.936 | 0.192 |

Notes:
- The "Confident No" row for the red-light candidate shows `digit` landing at 0.509 --
  essentially maximum uncertainty on its own 0-9 scale -- on a candidate `baseline` and
  `scene_conditioned` both reject with near-zero confidence. `digit` disagreeing this
  strongly with the other three methods on an otherwise clear-cut case is a concrete
  instance of the general digit-method weakness noted in earlier experiments.
- The pedestrian-yield "Confident No" row shows `likelihood` at -0.654 -- a comparatively
  high (i.e. "unsurprising", not low-probability) log-likelihood for a candidate the other
  three methods confidently reject, again illustrating `likelihood` not tracking scene
  correctness the way the other methods do.
- Both "Unsure" rows sit almost exactly on `baseline`'s 0.5 decision boundary, and both are
  cases the labeling subagents flagged with a `caveat` (a claim about motion -- "begins
  moving" -- or about a signal state that a single static frame can't fully confirm). The
  first "unsure" row's `baseline` (0.563) actually leans the *wrong* direction relative to
  its correct (No) label; the second's (0.437) leans wrong relative to its correct (Yes)
  label -- i.e. `baseline`'s uncertainty here isn't just imprecision, it's reflecting the
  same genuine ambiguity a human labeler had to flag with a caveat, in both directions.

## Experiment 14 -- calibration (Brier/ECE/reliability diagram) + risk-coverage, both ground truths

Every metric reported so far (AUROC, pairwise_acc, top1_acc) measures *ranking* quality --
whether the positive scores above the negative -- not whether the raw number means what it
claims (a "0.7 confidence" should be right ~70% of the time). This was flagged as an open
question from Experiment 1 onward and finally checked here.
`examples/motion_confidence_calibration.py` computes, per method: Brier score, Expected
Calibration Error (10 equal-count quantile bins, not fixed-width -- the score distributions
are heavily right-skewed per Experiment 1's worked examples, so fixed-width bins around
0.4-0.6 would be nearly empty), a full reliability-diagram table, and a risk-coverage
(selective-prediction) curve: sort candidates by `|score-0.5|` (most confident first) and
check whether abstaining on the least-confident fraction actually raises accuracy on what's
left -- the more decision-relevant question if a score is meant to flag "the model is
unsure here," which is what motivated this whole check.
`baseline`/`digit`/`scene_conditioned` are read directly as P(correct);
`likelihood`/`native_likelihood` are raw mean-log-probabilities and are first mapped to
[0,1] via an in-sample Platt-scaling logistic fit on the same corpus before Brier/ECE
apply -- so the `likelihood` row measures how calibratable the raw score *can be made* by
an optimal monotonic remap, not how calibrated the raw score already is.

Run against both ground-truth corpora already built elsewhere in this doc, since
Experiment 13 already showed the choice of negative changes which method looks best --
worth checking whether it also changes which method is *calibrated*:
- `k3`: Experiment 7/9's gold-vs-mismatched-gold corpus (offset=0.0 row,
  `outputs/motion_confidence_k3_toff0.0_n289_part{1,2}.json`, n=289 events), excluding the
  ~19% negative-QA-flagged still-plausible negatives (Experiment 7) by default.
- `selfcoc_labeled`: Experiment 13's real same-scene correct/incorrect corpus
  (`outputs/motion_confidence_selfcoc_labeled_scoring_n100.json`, n=42 mixed events, 336
  candidate-level examples/method) -- an independent Claude-vision judge's Yes/No verdict
  on each of an event's own 8 self-generated CoCs, joined against each method's raw score
  on that exact candidate.

**Results** (outputs: `outputs/motion_confidence_calibration_k3.json` /
`_selfcoc_labeled.json`):

| method | k3: Brier / ECE / acc@0.5 | selfcoc_labeled: Brier / ECE / acc@0.5 |
|---|---|---|
| baseline | 0.186 / 0.181 / 0.760 | 0.419 / 0.410 / **0.444** |
| digit | 0.211 / 0.153 / 0.689 | 0.259 / 0.183 / 0.521 |
| scene_conditioned | 0.221 / 0.222 / 0.734 | 0.481 / 0.488 / **0.435** |
| likelihood (Platt-scaled) | 0.196 / 0.044 / 0.718 | 0.226 / 0.094 / 0.643 |

**Headline 1 -- on `k3`, every bounded method is systematically underconfident, not
randomly miscalibrated.** `baseline`'s reliability table is monotone and one-directional:
predicted/empirical pairs run 0.0004/0.05, 0.0037/0.03, 0.0098/0.08, ... up to 0.61/0.85 at
the top decile -- every single bin's empirical correct-rate sits above its mean predicted
score, by as much as 0.42 at the second-highest decile. So `baseline` ranks well (AUROC
0.80, Experiment 7) while its raw number is not a usable literal probability -- a candidate
it scores "0.61" is actually correct ~85% of the time on this corpus, not ~61%. This is the
concrete answer to "can we treat a judge score of 0.5 as the model being uncertain" raised
earlier in this investigation: on `k3`, no -- a mid-range `baseline` score here is still
usually correct once calibrated (systematic underconfidence, not genuine indifference).

**Headline 2 -- on `selfcoc_labeled` (the harder, more realistic corpus), `baseline` and
`scene_conditioned`'s accuracy at the "obvious" 0.5 threshold falls *below chance*
(0.444 / 0.435), despite both having positive AUROC (Experiment 13: 0.606 / 0.571).** The
reliability table explains why it's not just a shifted threshold: it's non-monotonic
through most of the range -- `baseline`'s bins 4-6 (mean predicted 0.12-0.19) have a
*higher* empirical correct rate (0.70-0.74) than bin 7-8 (mean predicted 0.25-0.40,
empirical 0.58-0.65), and even the lowest bin (mean predicted 0.017) is still correct 41%
of the time. Only the top decile (mean predicted 0.83, empirical 0.79) cleanly separates.
The risk-coverage curve confirms this shape directly: accuracy is 0.88 at 5% coverage (the
handful of most-confident candidates) but craters to 0.51-0.56 by 25% coverage and drifts
to 0.44 by 85% -- almost all of `baseline`'s usable signal on this harder corpus lives in
the top ~10-15% most-confident readings; the rest of its range doesn't reliably
discriminate correct from incorrect at all. `scene_conditioned` shows the same pattern,
slightly worse (ECE 0.488 vs. 0.410).

**Headline 3 -- `likelihood`, the weakest method everywhere else in this doc, has the
best-behaved calibration on the harder corpus.** Its risk-coverage curve degrades
gracefully and monotonically (0.82 at 5% coverage -> 0.67 at 85%), never dropping below
chance, and its reliability gaps (0.03-0.22 across bins) are far smaller than
`baseline`'s/`scene_conditioned`'s (0.32-0.59 in the low-to-mid bins). This does not mean
`likelihood` is a better verifier overall -- Experiment 13 still has it statistically tied
with `baseline` on raw ranking (AUROC 0.615 vs. 0.606) -- it means `baseline`'s specific
failure mode (compressing almost every candidate to near-zero except a rare confident
case) breaks calibration much worse than `likelihood`'s more evenly-spread failure mode, on
this specific harder ground truth. This is the first result in this doc where `likelihood`
doesn't simply underperform the other three methods.

**Caveats:**
- Platt-scaling `likelihood` is in-sample (fit and evaluated on the same corpus) -- its
  ECE/Brier numbers are an upper bound on how calibratable the raw score is, not a
  property of the raw score by itself; a held-out split would be needed for an honest
  calibrated-Brier comparison, not done here.
- `selfcoc_labeled`'s n=42 events (336 examples) is the same small-n corpus flagged in
  Experiment 13 -- read this as a first, real but noisy calibration read, not a final
  number.
- **Bottom line for "can a judge score of 0.5 mean the model is uncertain":** it depends
  entirely on which ground truth and which method -- on `k3`, a mid-range `baseline` score
  is still usually correct (systematic underconfidence, not indifference); on
  `selfcoc_labeled`, a mid-range `baseline`/`scene_conditioned` score carries almost no
  information at all (near-chance or below across a wide band of the scale). "0.5" cannot
  be read literally as "50% likely correct" for either method on either corpus without a
  calibration check like this one -- and the check has to be run on the *harder*, more
  realistic corpus, since `k3`'s easy negatives make every method look far better
  calibrated than it is on real same-scene decisions.

## Takeaways so far

1. On the n=50 mismatched-gold-CoC benchmark, scene-conditioning is not a clear
   improvement over the single-shot baseline -- within noise on both pairwise accuracy
   and AUROC, at ~6x the cost.
2. The raw yes/no token-probability signal is substantially noisy along two axes that
   have nothing to do with scene-description conditioning: which exact frame (+-0.3s,
   range up to 0.90) and how the action is phrased (range up to 0.64). Any confidence
   signal built on this primitive should be validated against both before being trusted.
3. Scene-conditioning's language-sampling does not reliably fix (1) -- it partially
   damps it for one action, worsens it for the other, in the one clip tested.
4. A graded 0-9 self-reported confidence scale is a worse calibration signal than the
   plain yes/no ratio (pairwise-acc 0.700 vs. 0.817, AUROC 0.638 vs. 0.797, n=60) --
   and not because the model is reluctant to use it: it puts *more* total probability
   mass on the digit answer space than on yes/no (0.330 vs. 0.153 mean, n=120 reads).
   The gap comes from that mass being spread/hedged across the scale rather than peaked
   at the extremes the way yes/no peaks at confident answers (see Experiment 1 above).
5. Sidestepping the yes/no/digit framing entirely and scoring the candidate action's own
   teacher-forced, length-normalized log-likelihood is worse still (pairwise-acc 0.550,
   AUROC 0.619, n=60) -- length-normalization removes most but not all of a length
   confound (longer action still wins 70% of non-tied pairs even after normalizing), but
   the bigger issue looks like surface-form competition: every candidate is a real,
   fluent, human-written action, so raw text plausibility doesn't track scene-fit well.
   None of the four methods tried so far beats the plain single-shot yes/no baseline.
6. The yes/no baseline is equally (not more, not less) confident-and-correct on the
   model's OWN self-generated CoC as on human-written gold CoC (pairwise-acc 0.850 vs.
   0.817, AUROC 0.824 vs. 0.797, n=60, gap within noise) -- no detectable
   self-confirmation or self-skepticism bias at this scale.
7. Same baseline method is modestly worse on PAI-AV's train split than its val split
   (pairwise-acc 0.733 vs. 0.817, AUROC 0.746 vs. 0.797, n=60 each) -- opposite direction
   from a naive memorization story, single draw, not yet confirmed with a second seed.
8. Scaling from 1 to K=3 negatives per event (Experiment 7, n=289 val + n=219 usable
   train) shows single-negative pairwise accuracy substantially overstates reliability:
   baseline's strong pairwise_acc (~0.84) drops to top1_acc ~0.66-0.68 (must beat all 3
   negatives), and digit/likelihood's top1_acc (0.37-0.46) are only modestly above the 25%
   K=3 chance rate despite pairwise_acc in the 0.62-0.68 range. Val and train agree closely
   on both methods and metrics. An independent negative-QA judge also found ~19% of the
   K=3 negatives are mislabeled (still plausible for their scene) on both splits, so these
   numbers carry that much label noise.
9. An independent VLM judge (Qwen3-VL-32B-Instruct) applied to Experiment 5's
   self-generated CoCs (n=8 events x 8 samples smoke test, Experiment 8) found 8/64
   (12.5%) judged not actually correct for their scene, including one event where the
   self-CoCs contradicted each other about scene content (some claiming a lead vehicle
   present, others not) and the judge's verdicts tracked that disagreement sensibly --
   the first direct evidence that Experiment 5's "correct by construction" assumption is
   sometimes wrong. Also surfaced judge-reliability noise on near-identical phrasings
   (one event split 3 Yes/5 No on paraphrases of the same claim); a CoT+majority-vote fix
   eliminated that noise but appears to have traded away discriminative power (see
   Experiment 8's follow-up), not yet resolved.
10. Adding `scene_conditioned` to the K=3 val comparison and sweeping t0 by +-0.2s
    (Experiment 9, n=289 x 5 offsets, 0 load failures) confirms Experiment 1/7's "no clear
    win" finding holds at every offset (scene_conditioned's pairwise_acc/top1_acc/AUROC are
    below baseline's at all 5 points), and shows aggregate ranking metrics are far more
    stable to frame jitter (0.02-0.05 range across the whole sweep, all 4 methods) than the
    single-clip volatility Experiments 2/4 found (up to 0.90 range) -- population-level
    verifier accuracy and per-clip confidence noise are different questions, and one being
    robust doesn't mean the other is. Scene-conditioning's mean positive confidence *is*
    more stable across offsets than baseline's (range 0.007 vs. 0.024), replicating
    Experiment 4's direction on that one axis, but Experiment 4's "worsens negative
    variance" finding doesn't reproduce in the aggregate (both methods' mean_neg are
    equally flat) -- likely because 289 different per-event negatives average out
    idiosyncrasies that showed up tracking one fixed negative on one clip.
11. Scoring all 8 self-generated CoC candidates/event (not just 1) through all 4 methods,
    with positive/negative pairing deliberately left undecided (Experiment 10, n=289,
    0 load failures), shows baseline assigns more within-event score spread across its 8
    candidates than digit or scene_conditioned do (mean max-min spread 0.238 vs.
    0.172/0.175) -- but a worked example shows at least some of that spread reflects the
    method correctly separating genuinely different underlying claims among the 8 samples,
    not just noise on paraphrases, so this number alone can't be read as a noise measure
    without first checking how textually diverse an event's 8 samples actually are. The
    merged raw-score file is the actual deliverable, feeding whatever pairing decision
    comes next.
12. Rebuilt the `likelihood` method's prompt/tokenization to match the model's real trained
    CoT-generation contract (trajectory-history tokens fused into the prompt via
    `fuse_traj_tokens`, candidate action teacher-forced inside `<|cot_start|>`/`<|cot_end|>`
    special tokens, exactly as `eval_pai_av_val_a2.py` conditions for real inference) instead
    of the prior ad-hoc open-VQA-question framing (Experiment 11, n=25, K=3, 0 boundary-check
    failures). This did **not** improve on the existing ad-hoc `likelihood` method on the same
    25 events (pairwise_acc 0.400-0.427, AUROC 0.553-0.581 across both variants, both near
    chance) -- confirms Takeaway 5's "surface-form competition, not framing" explanation for
    why likelihood-based scoring underperforms baseline/digit: putting the scoring into the
    model's real trained format didn't fix it, so the problem is unlikely to be the
    prompt/token format used to elicit the log-likelihood, and more likely the same
    plausible-fluent-text confound named in Takeaway 5. Confirmed separately in the same
    investigation that `baseline`/`digit`/`scene_conditioned` already match the model's real
    public "vqa" task contract (`text_tasks.prepare_vqa_inputs`, "no-special VQA generation"
    per its own docstring) and needed no changes -- only `likelihood`'s framing was
    off-contract.
13. Checked whether the raw confidence numbers are actually calibrated probabilities, not
    just good rankers (Experiment 14, Brier/ECE/reliability-diagram/risk-coverage on both
    the `k3` and `selfcoc_labeled` ground truths). On `k3`, every bounded method is
    systematically *underconfident* in one direction (a `baseline` score of 0.61 is
    actually correct ~85% of the time) -- miscalibrated but at least monotonic and
    directionally safe. On the harder `selfcoc_labeled` corpus, `baseline`/`scene_conditioned`
    become non-monotonic through most of their range and their accuracy at the naive 0.5
    threshold falls *below chance* (0.444/0.435) -- only their top ~10-15% most-confident
    readings (by risk-coverage) carry real signal. `likelihood`, the weakest method by
    every ranking metric in this doc, has the best-behaved (smallest-gap, monotonic,
    never-below-chance) calibration on this harder corpus -- the first result here where it
    doesn't simply underperform. Direct answer to this investigation's calibration
    question: a raw score of 0.5 cannot be read literally as "the model is uncertain"
    without this kind of check -- it means something different depending on which ground
    truth and which method produced it.

## Open questions / next steps

- **Build a trajectory-profile-matched scoring variant** (real `"trajectory"`/`DRIVING_SIX_CAMERA_FOUR_FRAME`
  camera set with `front_tele` instead of `rear_tele`, plus `fuse_traj_tokens` trajectory-history
  conditioning for every method, not just `native_likelihood`) and re-check whether any
  headline finding changes -- see the "Standing caveat" section near the top of this doc.
  Not yet built or tested at any n.
- **Partially addressed by Experiment 13**: same-scene negatives (Claude-vision-labeled
  incorrect self-CoCs, n=42 mixed events) showed every method drops sharply and `baseline`
  loses its lead over `likelihood` -- but this used naturally-occurring self-sampled
  negatives, not the originally-proposed single-axis-flipped CAC-taxonomy negatives. Still
  worth trying the CAC-scorer approach specifically for a more controlled/systematic (rather
  than incidental) harder-negative set.
- **Addressed by Experiment 14**: Brier score / ECE / reliability diagram / risk-coverage,
  run on both `k3` and `selfcoc_labeled`. Headline: every method is miscalibrated, but in
  different ways on different ground truths -- `baseline`/`scene_conditioned` are safely
  (monotonically) underconfident on `k3` but non-monotonic and below-chance-at-0.5 on the
  harder `selfcoc_labeled` corpus; `likelihood` calibrates best on the harder corpus despite
  being the weakest ranker. Still open: the Platt-scaling used for `likelihood` is
  in-sample (see Experiment 14 caveats) -- a held-out calibration split hasn't been tried.
- Repeat Experiment 4 (temporal sensitivity under scene-conditioning) on more than one
  clip before trusting the "makes negative-action variance worse" finding -- Experiment 9's
  aggregate-level check (n=289) did not reproduce that specific finding (see Takeaway 10),
  but that's an aggregate check with 289 different negatives, not a repeat of Experiment
  4's single-clip, single-fixed-negative design, so the two aren't a direct confirm/deny of
  each other -- an actual multi-clip repeat of Experiment 4's exact design is still open.
- **Self-generated CoC positive/negative pairing is still an open, deliberately-deferred
  decision** (Experiment 10): which of each event's 8 self-sampled CoCs (if any) should
  count as "the" positive, and what a matched negative should look like (self-CoCs "on that
  particular scene ideally," per the request that produced Experiment 10 -- exact mechanism
  still unspecified), are not resolved by this doc. `outputs/motion_confidence_multi_coc_scores_val_n289_merged.json`
  has all 8 candidates' raw scores under all 4 methods per event, ready for whichever
  pairing scheme gets chosen. Related open point: Experiment 10's worked example shows an
  event's 8 self-CoCs are sometimes genuinely different claims, not just paraphrases of one
  claim -- whatever pairing scheme is chosen should probably account for this (e.g. cluster
  the 8 texts first) rather than assume they're interchangeable positive candidates.
- Consider explicit temporal-jitter ensembling (average over a small t0 window, not just
  language samples at one t0) as a complementary robustness axis to language-sampling.
- Run the same calibration comparison on Alpamayo 1.5 and on Qwen (the base VLM family
  Alpamayo 2 Super is built on) to see whether the calibration/noise findings here are
  specific to this model or general to the model family / VLM-as-judge setups.
- Incorporate ideas from the LLM-as-verifier literature (e.g. verifier-specific prompting,
  multi-aspect verification, weighting/aggregating verifier votes) into the judging step,
  rather than only varying how the confidence number is read off (yes/no ratio vs. digit
  scale vs. scene-conditioning).
- If likelihood scoring is worth another look: add the PMI correction (subtract the
  action's own context-free/no-scene log-likelihood) to control for surface-form
  competition -- needs a text-only or neutral-image forward pass through this VLM, not
  yet built.
- Confirm the train/val calibration gap (Experiment 6) with a second seed before treating
  it as real; if it holds, investigate whether it's scene-difficulty mix or something
  train-split-specific.
- Both proposed Experiment 8 judge reliability fixes (majority vote + CoT-before-verdict)
  are implemented and validated at n=8 (see the "Follow-up" note under Experiment 8) --
  they eliminated the two noisy-verdict cases, but appear to have traded away
  discriminative power (the one case where the old judge correctly caught a false claim
  now also gets a unanimous Yes). Before scaling to the full 289+300, need to check
  whether the new judge still catches genuinely-wrong self-CoCs on a larger and more
  verdict-diverse sample -- not yet resolved.
- Find a simpler, cheaper way to determine correctness of a model-sampled CoC than a
  second full VLM judge pass -- e.g. self-consistency across the 8 samples per event
  (agreement rate as a correctness proxy) worked as a signal in the `52b4287f` case above
  where the samples genuinely disagreed, but most events are unanimous or near-unanimous,
  so it's unclear how much discriminative power this has vs. Experiment 8's per-sample
  judge approach.
- Experiment 11's `native_likelihood` still underperforms `baseline`/`digit` (only compared
  against the existing ad-hoc `likelihood` at n=25 so far) -- if likelihood-based scoring is
  revisited again, try it under the native CoT-generation conditioning with a different
  score reduction (e.g. the PMI correction two bullets up, computed under native
  conditioning instead of the VQA-question framing) before concluding the primitive itself
  is unsalvageable; not yet tried at any n.

## Files

- `examples/motion_confidence_smoke.py` -- main pipeline (baseline + digit + likelihood +
  scene-conditioned, `--methods` selects which to run, `--system_prompt` overrides the
  system prompt, `--split {val,train,both}`, n-event batch run, JSON output,
  pairwise-acc/AUROC summary).
- `examples/motion_confidence_self_coc.py` -- Experiment 5: generates the model's own
  CoT via `sample_trajectories_from_data` on the "trajectory" task profile, then runs
  the same baseline yes/no judging as motion_confidence_smoke.py on "vqa" profile images.
  `--events_from` reuses an exact prior event list for apples-to-apples comparison.
- `examples/motion_confidence_time_sensitivity.py` -- t0-offset sweep, both methods.
- `examples/motion_confidence_phrasing_sensitivity.py` -- paraphrase sweep, baseline.
- `examples/export_scene_images.py` -- front-wide-camera PNG export for writeups.
- `examples/motion_confidence_multi_coc.py` -- generates K independently-sampled
  self-CoCs per event (temperature/top_p sampling, not greedy) via the same
  `sample_trajectories_from_data` path as `motion_confidence_self_coc.py`, generalized to
  `--num_samples` > 1.
- `examples/motion_confidence_negative_qa.py` -- independent-judge (Qwen3-VL-32B-Instruct)
  QA of the K=3 cyclically-shifted gold-CoC "negatives" used in the K=3 experiment: neutral
  prompt, never reveals the gold/positive action, judges scene-fit directly from the same
  6-camera "vqa" current-frame images. Output: `counts` (Yes/No/Uncertain/unparseable) +
  per-negative `records`.
- `examples/motion_confidence_self_coc_qa.py` -- Experiment 8: same judge/pipeline as
  `motion_confidence_negative_qa.py`, adapted to QA the self-generated CoCs from
  `motion_confidence_multi_coc.py` instead of the shifted-gold negatives.
- `examples/motion_confidence_multi_coc_scores.py` -- Experiment 10: scores all 8
  self-CoC candidates/event (from `motion_confidence_multi_coc.py`'s output) through all 4
  methods, images/scene-descriptions loaded/sampled once per event and reused across the 8
  candidates; no positive/negative pairing chosen.
- `examples/motion_confidence_aggregate_toff.py` -- Experiment 9: merges the 5 offsets'
  part1+part2 K=3 result files, recomputes pairwise_acc/top1_acc/AUROC/mean_pos/mean_neg
  per method over the combined n=289, and reports `MIN_T0_US` clamping counts per offset.
- `examples/motion_confidence_aggregate_multi_coc_scores.py` -- Experiment 10: merges
  `motion_confidence_multi_coc_scores.py`'s two shards into one raw-score file and reports
  the per-method within-event spread (max-min, std) descriptive summary.
- Results: `outputs/motion_confidence_smoke.json` (n=10, seed 0),
  `outputs/motion_confidence_more40.json` (n=40, seed 1, skip 0),
  `outputs/motion_confidence_more10b.json` (n=10, seed 1, skip 40),
  `outputs/motion_confidence_smoke_digit.json` / `_more40_digit.json` / `_more10b_digit.json`
  (same 3 event batches, `--methods baseline digit` re-run adding the digit-scale method),
  `outputs/motion_confidence_smoke_likelihood.json` / `_more40_likelihood.json` /
  `_more10b_likelihood.json` (same 3 event batches, `--methods likelihood`),
  `outputs/motion_confidence_digit_sysprompt_v1.json` / `_v2.json` (system-prompt A/B on
  the seed-0 n=10 batch),
  `outputs/motion_confidence_self_coc_n60.json` (Experiment 5, same 60 events as
  Experiment 1 via `--events_from`),
  `outputs/motion_confidence_train_smoke.json` / `motion_confidence_train_more50.json`
  (Experiment 6, train split, n=10 + n=50),
  `outputs/motion_confidence_time_sensitivity.json` (baseline sweep),
  `outputs/motion_confidence_time_sensitivity_scene.json` (both-methods sweep),
  `outputs/motion_confidence_phrasing_sensitivity.json`,
  `outputs/motion_confidence_n289_part{1,2}.json` / `_scene_part{1..4}.json` (Experiment 1
  at full val scale, n=289), `outputs/motion_confidence_train_n300.json` (Experiment 6 at
  n=300 train), `outputs/motion_confidence_self_coc_n289_part{1,2}.json` (Experiment 5 at
  n=289 val), `outputs/motion_confidence_k3_n289_part{1,2}.json` /
  `motion_confidence_k3_train_n300_part{1,2}.json` (Experiment 7, K=3 negatives, val n=289
  / train n=300 attempted, 219 usable), `outputs/motion_confidence_negative_qa_qwen3vl32b.json`
  / `_train.json` (Experiment 7's negative-QA judge results, val / train),
  `outputs/motion_confidence_multi_coc_val_n289.json` / `_train_n300.json` (8
  independently-sampled self-CoCs per event, feeds Experiment 8),
  `outputs/motion_confidence_self_coc_qa_smoke.json` / `_v2.json` (Experiment 8 smoke test,
  v1 single-greedy-call judge / v2 CoT+majority-vote judge),
  `outputs/motion_confidence_k3_toff{-0.2,-0.1,0.0,0.1,0.2}_n289_part{1,2}.json` (Experiment
  9, K=3 + scene_conditioned at 5 t0 offsets, val n=289 per offset, 10 files total),
  `outputs/motion_confidence_toff_aggregate.json` (Experiment 9's aggregated per-offset
  per-method metrics + clamp counts, written by `motion_confidence_aggregate_toff.py`),
  `outputs/motion_confidence_multi_coc_scores_val_n289_part{1,2}.json` (Experiment 10 raw
  per-shard output) and `_merged.json` (the two shards merged into one 289-event file, the
  actual deliverable for a future pairing analysis), `outputs/motion_confidence_multi_coc_scores_summary.json`
  (Experiment 10's per-method spread summary, written by
  `motion_confidence_aggregate_multi_coc_scores.py`),
  `outputs/motion_confidence_native_cot_shard{0..7}.json` (Experiment 11 raw per-GPU-shard
  output, n=25 val events split 4+3x7 across 8 GPUs, seed 0, K=3) and
  `_n25_merged.json` (the 8 shards merged into one 25-event file) /
  `_n25_summary.json` (the merged pairwise_acc/top1_acc/AUROC summary for both
  `likelihood` and `native_likelihood`, written ad hoc from the merge script, not by
  `motion_confidence_native_cot.py` itself since that script only summarizes its own
  single-process event set).
- `examples/motion_confidence_native_cot.py` -- Experiment 11: `native_likelihood`, scoring
  a candidate action teacher-forced as CoT text under the model's real trained
  CoT-generation conditioning (trajectory-history tokens fused via `fuse_traj_tokens`,
  `<|cot_start|>`/`<|cot_end|>`-wrapped action), imports and reuses
  `score_likelihood`/`LIKELIHOOD_QUESTION`/`build_image_content` from
  `motion_confidence_smoke.py` unmodified to also run the existing ad-hoc `likelihood`
  method on the same sampled events in the same run, for a same-event side-by-side
  comparison. Same `--num_events`/`--skip`/`--seed`/`--split`/`--num_negatives` CLI
  convention as `motion_confidence_smoke.py`; adds `--gpu` (device index) and `--debug`
  (per-event tokenization diagnostics: decoded prefix/full text tails, trajectory shapes,
  boundary-check outcome -- meant for small-n sanity checks, noisy at scale).
- `examples/export_scene_images_front3.py` -- Experiment 12/Addendum 2: exports 3
  forward-facing-camera (cross_left/front_wide/cross_right), current-frame-only stills per
  event for the reduced human/Claude-vision-labeling input, with retry-with-backoff and
  `--resume` support. Output: `outputs/motion_confidence_images/front3_qa/` (100 events'
  worth of PNGs + `manifest.json`, the latter also carrying each event's 8 self_cocs and
  image paths -- this manifest is what `examples/motion_confidence_label_tool.py` reads).
- `examples/motion_confidence_label_tool.py` -- manual (human) labeling tool: a local
  stdlib-only web app serving one self-CoC candidate + its 3 front3_qa images at a time,
  Correct/Incorrect/Unsure buttons (keyboard Y/N/U), resumable, saves to
  `outputs/motion_confidence_human_labels.json` in the same schema as
  `motion_confidence_claude_vision_labels_n10.json`/`_n100.json` for direct comparison.
- `examples/motion_confidence_calibration.py` -- Experiment 14: Brier score / ECE /
  reliability-diagram / risk-coverage-curve computation, `--source {k3,selfcoc_labeled}`
  selects the ground-truth corpus; `likelihood`/`native_likelihood` are Platt-scaled
  in-sample before scoring. Outputs: `outputs/motion_confidence_calibration_k3.json` /
  `_selfcoc_labeled.json`.
- Results (Experiments 12/13): `outputs/motion_confidence_claude_vision_labels_n10.json`
  (Addendum 2, n=10/80 candidates) / `_n100.json` (Experiment 12, n=100/800 candidates, 10
  parallel-subagent batches merged), `outputs/motion_confidence_selfcoc_labeled_scoring_n100.json`
  (Experiment 13: Experiment 12's labels joined against Experiment 10's raw per-candidate
  scores, `results[method] = {"pos": [[candidate_idx, event_idx, score], ...], "neg": [...]}`
  pooled over the 42 mixed events, plus `mixed_events` count).
- Images: `outputs/motion_confidence_images/` (front-wide-camera stills, see
  `manifest.json` there), `outputs/motion_confidence_images/front3_qa/` (100-event reduced
  3-camera/1-frame stills used by Experiment 12 and the manual labeling tool).
