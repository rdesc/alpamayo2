# Reasoning–action consistency (LLR) diagnostics — Alpamayo 2 Super

Alpamayo-2-Super side of the Phase 0 measurement specified in
`~/repos/alpamayo-recipes/langforce_readme.md`: does the model's
Chain-of-Causation reasoning actually carry information its trajectory
prediction uses? Measurement only — no training.

The Alpamayo-1.5 counterpart (and the fuller write-up of method, failure modes,
and the two-ablation trap) lives in
`~/repos/alpamayo-recipes/recipes/alpamayo1_x_rl/scripts_fork/llr/`.

| script | what it does |
|---|---|
| `phase0_llr_action_direction_a2s.py` | The measurement. Scores `llr_act = log p(a*\|v,ℓ) − log p(a*\|v,ℓ_blanked)` over the PAI-AV OOD reasoning split via `Alpamayo2Super.forward()`'s own `loss_future_traj`, blanking the CoC span with pad tokens at identical length/position. |
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

## Result

`llr_act` mean **+0.037 nats** (median +0.036, range −0.325 → +0.532,
n = 2,031 / 2,077) — positive on every event cluster, so reasoning is measurably
load-bearing, but roughly **6× weaker than Alpamayo 1.5's +0.227**.
