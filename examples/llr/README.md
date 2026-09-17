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

## Result — ⚠️ RETRACTED, pending re-measurement

Previously reported: `llr_act` mean **+0.037 nats** (median +0.036, range −0.325 → +0.532,
n = 2,031 / 2,077), positive on every event cluster, "roughly 6× weaker than Alpamayo 1.5's
+0.227".

**Both of those numbers were artifacts of the ablation, not measurements of reasoning.** The
Alpamayo-1.5 side was diagnosed first and in detail — see
`~/repos/alpamayo-recipes/scripts_fork/llr/results/phase0_llr_action_direction.md`. Two causes,
and `phase0_llr_action_direction_a2s.py` shares both because it scores through the same
`loss_future_traj` mean and blanks the same `get_label_mask(..., ["cot"])` span:

1. **`loss_future_traj` does not score only the future trajectory.** Its span also contains the
   history-trajectory tokens (same token-id block, so the mask's id-range test catches them) and
   the `traj_future_start`/`traj_future_end` delimiters. History tokens precede the cot span, so
   causal attention pins their LLR to exactly 0 — pure dilution.
2. **Blanking that span deletes `<|cot_start|>`/`<|cot_end|>.`** `get_label_mask` is inclusive of
   both markers (`get_label_mask.py:45`). On Alpamayo 1.5 this left `<|traj_future_start|>`
   following an `<|endoftext|>` and drove the delimiters from `log p` of exactly 0.0 to −18/−28
   nats — a near-constant **+0.27 offset** on the mean, larger than the entire reported effect.

On Alpamayo 1.5, replacing the denominator with a genuine `p(a*|v)` collapsed `+0.227` to
**about −0.006 nats**, against null controls at exactly 0.0 and positive controls (wrong vision,
wrong trajectory) at 0.27 and 1.07. Given A2S's `+0.037` is an order of magnitude smaller than
1.5's retracted figure and was produced by the identical mechanism, it is very likely to be the
same offset diluted across a longer span — but that is a prediction, not a result, until it is
re-run.

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
