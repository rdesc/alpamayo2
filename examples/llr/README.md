# Reasoning–action consistency (LLR) diagnostics — Alpamayo 2 Super

Alpamayo-2-Super side of the Phase 0 measurement specified in
`~/repos/alpamayo-recipes/scripts_fork/llr/langforce_readme.md`: does the model's
Chain-of-Causation reasoning actually carry information its trajectory
prediction uses? Measurement only — no training.

The Alpamayo-1.5 counterpart (and the fuller write-up of method, failure modes,
and the two-ablation trap) lives in
`~/repos/alpamayo-recipes/scripts_fork/llr/`.

| script | what it does |
|---|---|
| `phase0_llr_action_direction_a2s.py` | **Superseded — produced the retracted number (see Result below).** Scores `llr_act = log p(a*\|v,ℓ) − log p(a*\|v,ℓ_blanked)` over the PAI-AV OOD reasoning split via `Alpamayo2Super.forward()`'s own `loss_future_traj`, blanking the CoC span with pad tokens at identical length/position. |
| `extract_llr_viz_a2s.py` | Pulls camera frames (A2S's native 6-camera trajectory profile, `[0,1,2,3,5,6]`) + GT trajectory for selected events, renders BEV and speed-vs-time plots including the model's own predicted trajectory with reasoning present vs. suppressed. |

## Two things specific to A2S

**It does have a trained discrete-token trajectory head.** Easy to miss, since
the released inference path generates CoC text and then samples through the
diffusion expert. But `Alpamayo2Super.forward()` fuses ground-truth trajectory
tokens and computes a real next-token `loss_future_traj` alongside
`loss_others` — which is what makes an exact `log p(a*)` available here at all,
the same way it is on Alpamayo-1.5's Stage-1 pathway.

**Its no-reasoning ablation is clean, unlike 1.5's.** `--no_coc` in
`../eval_pai_av_val_a2.py` builds the conversation with
`components_prompt=["traj_future"]` only, and `pred_coc_ml` comes back genuinely
empty. Alpamayo-1.5's `RLWrapperReasoningVLA` has no equivalent and needed a
manual `coc_text=""` splice; **do not port that workaround here** — A2S supports
this first-class. See `docs/pai_av_ood_eval.md`, "CoC ablation".

Caveat inherited from that doc: the released expert was trained conditioning on a
KV-cache that always had CoT tokens before the trajectory, so CoT-off sits
outside its validated conditioning distribution.

## Result: `+0.0179 nats`, 15.4σ — reasoning **is** load-bearing on A2S

Re-measured with `phase0_llr_per_token_a2s.py --blank-mode splice` over the full OOD reasoning
split (train+val, 2,071 events, 265,088 scored trajectory tokens):

| | Alpamayo 1.5 | **Alpamayo 2 Super** |
|---|---|---|
| corrected `llr_act` | +0.0010 | **+0.0179** |
| median | +0.0021 | +0.0152 |
| std | 0.0509 | 0.0528 |
| fraction > 0 | 54.4% | **69.2%** |
| distance from 0 | 0.9σ | **15.4σ** |
| by split | train +0.0006 / val +0.0030 | train +0.0176 / val +0.0193 |

**This is a real difference between the two models, not the same artifact twice.** Three things
beyond the sigma support it:

- **Channel asymmetry.** Acceleration **+0.0265** vs. curvature **+0.0093** — reasoning informs
  longitudinal control ~3× more than lateral, which matches the predominantly longitudinal content
  of the CoC annotations ("slow for…", "stop for…"). On 1.5 both channels were flat
  (−0.0001 / +0.0021) with no asymmetry at all.
- **Horizon structure.** +0.032 in the first second decaying to +0.014–0.017 past 5 s, versus
  1.5's flat ±0.007 noise.
- **Content, not just presence.** The wrong-prose arm splits the effect: `llr_content` (gold vs.
  another event's real CoC) = **+0.0110**, `llr_presence` (wrong prose vs. none) = +0.0066. So
  ~63% comes from the reasoning being *correct for this scene*. On 1.5 the same arm gave −0.008
  against a +0.001 total — no content signal whatsoever.

Sanity controls (`phase0_llr_verify_setup_a2s.py --limit 24`): reconciliation 2e-7,
`null_identical` and `history_llr` exactly 0, wrong vision **+0.2785**, wrong trajectory
**+0.9173**. So reasoning is worth ~6% of what the cameras are worth here, against ~0.4% on 1.5.

**Caveat:** A2S uses a 6-camera input profile and a different training recipe than 1.5's
4-camera setup. Each model is measured against its own baseline; the ratio between them is not a
controlled comparison.

### The previously published `+0.037` — what it was

`llr_act` mean +0.037 (median +0.036, range −0.325 → +0.532, n = 2,031 / 2,077), described as
"roughly 6× weaker than Alpamayo 1.5's +0.227". The direction of that claim was right but the
basis was inflated ~2×, and its comparison point (1.5's +0.227) was itself an artifact.

**This number is not a measurement of reasoning as it stands, but it is wrong in a different — and
milder — way than Alpamayo 1.5's was.** The 1.5 side was diagnosed first and in detail (see
`~/repos/alpamayo-recipes/scripts_fork/llr/results/phase0_llr_action_direction.md`), where two
defects compounded. **A2S has only the first of them.**

1. **Shared: `loss_future_traj` does not score only the future trajectory.** Its span also
   contains the history-trajectory tokens (same token-id block, so the mask's id-range test
   catches them) and the `traj_future_start`/`traj_future_end` delimiters. History tokens precede
   the cot span, so causal attention pins their LLR to **exactly 0** — they purely dilute the mean.

2. **NOT shared: marker deletion.** On 1.5 the ablation blanked
   `get_label_mask(..., ["cot"])`, whose span is *inclusive* of `<|cot_start|>`/`<|cot_end|>`
   (`get_label_mask.py:45`), leaving `<|traj_future_start|>` to follow an `<|endoftext|>`. That
   drove the delimiters from `log p` of exactly 0.0 to −18/−28 nats — a near-constant **+0.27
   offset**, larger than the entire effect it reported. **This script does not do that:** it
   locates `cot_start`/`cot_end` by token id and blanks `cot_lo:cot_hi`, the interior strictly
   between them, leaving both tags intact (lines 239–252). On 1.5, the equivalent interior-only
   blanking left the delimiters at +4e-6 — i.e. no offset at all.

**Measured, not inferred** (`phase0_llr_verify_setup_a2s.py`): A2S's id blocks really are disjoint
— history `[151669, 152669)`, future `[152669, 155669)` — so its mask contains **0 history
tokens**, and is 130 = 128 future + 2 delimiters. Dilution factor 1.016, against 1.5's 1.391. The
marker check passes on real token strings: the blanked span `[4581:4596]` is immediately preceded
by `<|cot_start|>` and followed by `<|cot_end|>`.

**But a third contamination showed up, unique to A2S, and it lives in the delimiters.** Even with
the markers intact, pad-blanking the interior moves the 2 delimiters by up to **3.95 nats**
(mean −0.79) — where 1.5's markers-intact blanking left them at 4e-6. And under *splicing* they
move ~−1.1 nats, enough to drag the 130-token mask-mean to ~+0.000 while the 128 real tokens sit
at +0.018. So the delimiters have to be excluded either way; the fix is scoring future tokens only,
not choosing a different denominator.

Measured side by side over the same 24 events:

| basis | mean |
|---|---|
| old basis (mask-mean, interior pad-blank) | **+0.0395** ← reproduces the published +0.037 |
| future-only, interior pad-blank | +0.0277 |
| **future-only, `cot_text=""` splice** | **+0.0176** |
| future-only, wrong reasoning prose | +0.0110 |

The old basis reproducing +0.0395 confirms the re-measurement is scoring the same thing the
original did, and the splice value at n=24 matches the full-split +0.0179.

**A2S has a cleaner fix available than 1.5 did.** `--no_coc` here is already first-class
(`components_prompt=["traj_future"]`, `pred_coc_ml` genuinely empty — see
`docs/pai_av_ood_eval.md`, "CoC ablation"), so the correct in-distribution denominator does not
need the manual splice that Alpamayo 1.5 required. The re-measurement should:

- score the **future trajectory tokens only**, split out by role (history / delimiter / future),
  reconciling the full-mask mean against `-loss_future_traj` as an alignment check;
- use `--no_coc` as the denominator rather than pad-blanking anything;
- run `phase0_llr_sanity_controls.py`'s equivalents first — three nulls that must be exactly 0
  (including the history tokens, where 0 is a structural invariant) and the wrong-vision /
  wrong-trajectory positive controls, without which a null is uninterpretable.

Caveat inherited from that doc and still standing: the released expert was trained conditioning
on a KV-cache that always had CoT tokens before the trajectory, so CoT-off sits outside its
validated conditioning distribution.
