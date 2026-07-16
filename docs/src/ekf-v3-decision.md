# EKF v3 decision record — adaptive-EKF-on-SINDy is retired (not promotable)

## Question
Can the operational Dst forecaster's shadow EKF — which online-adapts coefficients of the discovered SINDy ODE
`dDst*/dt = Θ(x)·ξ` — be upgraded to a "v3" forecast method that beats V2 and persistence?

## Experiments (fully causal: filter bootstrapped on OMNI years < storm year; 7 G4/G5 storms 2023–2025;
leads 1/2/3/6 h; pooled + main-phase (dDst/dt < −15 nT/h) regimes; paired-bootstrap 95% CI vs the **stronger**
of {v2, persistence}; fairness oracle `ekf_fixed` = rollout with fixed discovered coefficients, which must
reproduce the locked v2 engine → max|ekf_fixed − v2| = 0.00 confirms the comparison is exact).

| variant | file | verdict |
|---|---|---|
| decay-only EKF (adapt `Dst_star`) | `ekf_storm_replay.jl` | NOT PROMOTABLE — no lead/regime beats the stronger baseline with CI>0 |
| injection-adaptive EKF (adapt `[Dst_star, Bs]`) | `ekf_inject_replay.jl` | NOT PROMOTABLE — strictly worse than decay-only; 6 h-main CI below 0; robust across a 100× q_Bs sweep |

## Mechanism (why it fails — structural, not a tuning accident)
The discovered ODE's only large fast lever is the southward-**injection** term (ξ_Bs = −0.693); **decay**
(ξ_Dst* = −0.0479) is ~14× weaker and acts only in recovery, where persistence is already near-perfect — so the
decay-only EKF has no lever where skill is needed. Adding the injection lever gives a downward push, but EKF
coefficient adaptation is inherently **reactive**: it can only raise |Bs| *after* the assimilated Dst residual
reveals the storm is dropping, so it lags onset and then over-injects on the descent (+53 nT main-phase bias at
6 h). A filter that adapts coefficients is architecturally incapable of adding onset/main-phase skill to a
discovered ODE whose only fast lever is southward injection.

## Independent reflection (3 adversarial reflectors + synthesis)
- **Correctness: trustworthy.** Every headline number reproduced from the scored CSVs; the fairness oracle is
  exactly 0.0000 in both files (the v2 residual correction is ruled out as a confound by construction); the EKF
  is a genuine overlay (mean |ekf − v2| = 8.35 nT), not a no-op tie. No bug, sign/index error, leakage, or unfair
  comparison was found that inverts the loss. The "stronger-of-{v2,persistence}" bar is not stacked — relaxing to
  beat-v2-alone or beat-persistence-alone still yields no CI>0; the −15 nT/h threshold is generous (stricter makes
  the EKF strictly worse).
- **Two caveats, both non-decision-changing.** (1) The rollout omits the engine's per-step dDst clamp to
  [−200, 200], but |decay·Dst*| peaks ~63 nT so it never triggers (dormant). (2) The per-row IID paired bootstrap
  overstates effective n on autocorrelated within-storm hours; a storm-block bootstrap would only *widen* the
  already-failing CIs → it strengthens NOT-PROMOTABLE. Worth reporting block CIs only for a written record.
- **Direction: confirmed dead end, no worth-testing variant in this formulation.** The `sin(θ_c/2)` coupling-angle
  variant is the same reactive southward-driver lever in different clothing and will fail identically — a trap.

## Decision
Retire adaptive-EKF-on-SINDy. The EKF shadow line/endpoint/daemon-computation were removed from the live product.

## Recommended next direction (a different method, not an EKF tweak)
Pivot from "better filter" to "earlier/better driver." The O'Brien contrast (a physics injection+decay ODE beats
persistence +6.04 nT [2.69, 9.27] at 6 h) points the skill source at injection physics driven by a **forecast**
driver: condition the discovered injection term on a propagated L1 solar-wind driver (~30–60 min free lead) or seed
the injection coefficient from the existing DONKI/DBM CME-arrival prior in the repo, so injection rises *before* the
residual reveals the drop. Framing note for any writeup: "v2 loses to persistence at every lead" is true only
**pooled**; in the **main phase** v2/SINDy-v1 beat persistence at every lead — both regimes must be stated.
