# V2.4 Integration Report

Scope: integrate the Operational V2.4e super-learner as the **served** point center of the
live engine, keeping the static V2.2 regime stack and the V2.1 operator as the disclosed
fallback chain, the V2.3 analog candidate as a shadow forecast, and the published severity
depth-safe against both stages V2.4e replaces.

Governing evidence: the rolling-origin study under `V2_4_RESEARCH_PLAN.md` Amendment A3
(12 yearly folds 2014–2025, every model refitted per fold, every out-of-fold pool fit
embargoed by target with 168 h). Two decisions live in that study's artifacts and they say
different things, so both are carried here:

- `v2_4_decision.csv` records the plan's own §6 rule as `SHADOW`, because era E1
  (2014–2019, the data-poor early folds) fails G1 and G2 while ALL and E2 pass;
- `v2_4_serve_rule.csv` records the Amendment A3 **operational serve rule**, evaluated
  against the product actually being replaced, as `SERVE_ELIGIBLE_PENDING_G4` on ALL and
  E2: a pooled gain over the static stack with a positive one-sided 95 % lower bound at
  every model step, no bootstrap-supported storm-cell loss in any of the five storm cells,
  and G3 intervals passing. E1 is disclosed rather than decisive there, because the static
  stack was fitted on 2010–2017 and is partly in sample in that era.

The pending gate in that verdict is G4 — availability, latency and live-versus-offline
identity — which is what this integration supplies.

Selection: `v2_4_decision.csv` names `v2_4e` (E1 mean pooled RMSE over steps 2, 3, 6:
`v2_4e = 7.2786`, `v2_4a_floor = 7.2787`, `v2_4d = v2_4f = 7.2830`).

## What changed from the previous candidate, and why

The earlier integration served `v2_4d`: a nine-expert stack whose center was additionally
taken as `min(stack center, static V2.2 stack)` in a deepening cell. Independent verification
of that run found the shape wrong rather than the arithmetic wrong. NNLS weights are dominated
by moderate rows and the deep cells are data-poor in the early folds, so the physics
composition the static product already encodes was never recovered by the combination; the
guard imposed it afterwards, and bootstrap-supported losses in the deep cells survived anyway.

Amendment A3 puts the static V2.2 stack **into** the combination as expert ten and counts it
inside the 0.60 mass floor, because that product is itself a composition of the deployed SINDy
operators, so the floor keeps its meaning. The optimiser then uses that composition where it is
best and adapts elsewhere, and the selected variant needs no guard on the point forecast.

Depth safety did not leave the product with the guard. The published severity is taken through
`v24_serving_depth_safe_center` as the deepest of the V2.4e center, the static-stack center and
the V2.1 center, so a combination that blends toward a shallower state cannot lower a warning
either predecessor would have raised. The published watch edge is taken the same way, over the
edges themselves: the deepest of the served edge and the lower edge each predecessor would have
published for the same issue, both of which the engine now logs per row
(`v2_2_stack_ci05_dst_nt`, `v2_1_served_ci05_dst_nt`).

The earlier integration shifted the served edge down by the amount the point was lowered and
called that band-safe. It is not. A V2.4e row carries the study's depth-stratified conformal
half-width, and that is not the width the predecessor's band had, so
`served_ci05 + min(0, depth_safe − served)` lands at `deepest_partner_center − hw_V2.4` rather than
at the predecessor's own edge `deepest_partner_center − hw_partner`. The scratch live cycle below
shows the two widths differing at every issued lead. Wherever `hw_V2.4` is the smaller of the two
the shifted edge is shallower than the edge the predecessor would have published, and the audit
that raised this reported that case for the shallow bins at steps 1–2 against the archived adaptive
bands; that comparison against the production band history is not re-derived here, and the case is
instead constructed and pinned in the app suite, where the tier drops from 3 to 2.

The correction is stated as a minimum over the logged partner edges, which is idempotent, leaves a
deeper served edge untouched, drops a non-finite partner rather
than propagating it, and discloses in the payload which stage the published edge came from. A
row written before those columns existed has no deeper edge on record, so it keeps the shift rule
and the payload says so.

The correction is two-sided, and that is worth stating plainly. The shift rule lands at
`deepest_partner_center − hw_V2.4`, so it under-warns wherever the conformal band is narrower than
the band the predecessor served under and over-warns wherever it is wider. The minimum over edges
publishes exactly the deepest edge any stage would have published — never shallower than the
served edge, never shallower than a partner's own edge, and never deeper than all of them. The
scratch live cycle recorded under *Verification run* is a case of the second kind: at the 6 h step
the conformal half-width was 13.09 nT against the predecessors' 10.89 nT, so the published watch
edge moved from −77.3 nT under the shift rule to −75.1 nT, which is the V2.1 operator's own edge
and the deepest edge on record for that issue. Both readings sit in the same storm tier there. The
first kind — the shallow-bin steps where the conformal band is the narrower one — is the case the
app suite pins, and there the tier itself moves.

The guard arithmetic itself is retained and is driven by the bundle's own `guard.json`, so a
later bundle whose selection returns a guarded variant is servable without a source change.

## Served product

| Item | Value |
|---|---|
| Served identity | `v2.4+sindy20x11+superlearner10floor+conformal` |
| First fallback identity | `v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)` |
| Second fallback identity | `v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia` |
| Center | NNLS combination of ten experts per (model step, issue-time regime, depth bin), 0.60 mass floor on the SINDy family; no residual layer, no point-forecast guard |
| Experts (weight-column order) | served V2.1, frozen V2.1, T1r analog, persistence, Burton, Burton-full, O'Brien–McPherron, direct increment-GBM, climatology-relaxed persistence, static V2.2 stack |
| Floor group | served V2.1, frozen V2.1, T1r analog, static V2.2 stack — mass ≥ 0.60 |
| No residual layer | `selected.json` records `residual_applied = false`; the loader refuses a bundle that claims one |
| Guard | `guard.json` records `guard_applied = false`, `guard_reference = none`; the loader refuses a bundle whose switch and reference disagree, and applies the guard only when the artifact asks for it |
| Interval | Split-conformal half-widths per (model step, depth bin), `interval_source = v24_conformal_depth` |
| Static-stack precondition | The row's static-stack center (`v2_2_stack_pred_dst_nt`) is expert ten; without it the stage cannot act and the row serves the V2.1 operator |
| Depth bins | `shallow > -30`, `moderate ∈ (-30, -70]`, `deep ≤ -70` nT on the causal issue-time Dst |
| Deepening cell | one-hour fall steeper than −15 nT/h, or active gated coupling with Dst ≤ −50 nT; labelled on every row, acted on by none |
| Alerting center | `min(served, v2_2_stack, v2_1_served)` per row, then the existing minimum over the cycle |
| Alerting interval edge | `min(served_ci05, v2_2_stack_ci05, v2_1_served_ci05)` per row, then the minimum over the cycle; the watch tier is taken on this edge, and the payload names the stage that set it |
| Physical projection | `clamp(·, −2000, +50)` nT on the reported center, logged as `v24_projection_applied` |
| Bundle | `deploy/v2_4/`, manifest digest `057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede` |

## Deployable bundle

`validation/operational/v2_4_build_deploy.jl` produces a "fold-2026" refit: the same code
paths the rolling engine and the learning stage used, with the training window extended to
the end of the base table and the out-of-fold pool extended to every persisted fold.

| Item | Value |
|---|---|
| Fold year / label | 2026 / `fold2026` |
| Training window | max target ≤ `2025-12-24T16:00`, issues `2010-01-01T01:00 … 2025-12-24T09:00` |
| Training anchors | 138,721 (138,698 usable) |
| Embargo | 168 h on the training window; the same bound is applied to the out-of-fold pool by target (`pool_target_cutoff_utc = 2025-12-24T16:00`), which is Amendment A3's embargo fix |
| Out-of-fold pool | fold tables 2013–2025 (T1r correction, super-learner) |
| Conformal pool | learn-stage `v2_4e` centers 2014–2025 |
| Super-learner | 60 cells, variant `L1e` (the ten-expert 0.60-floor fit) |
| Conformal | 24 strata (6 steps × pooled + 3 depth bins), variant `v2_4e` |
| T1r correction | `operational_v2_4_t1r_year2026_memory_expert_lead_ridge100.0_fit675744` on 675,744 out-of-fold rows |
| Analog archive | 138,715 origins, `2010-01-01T07:00 … 2025-12-24T09:00`; frame `2010-01-01T00:00 … 2025-12-31T23:00` (140,256 rows) |
| Climatology τ | 7.576947282167329 h |
| Direct GBM | steps 1/2 `d6_r400_nb255`; steps 3/4/6/7 `d4_r400_nb255`, each fitted on 138,698 rows |
| Environment | Julia 1.12.6, 8 threads, EvoTrees 0.18.7, seed 22022026 |
| Base table / frame | SHA-256 `9dcbe8f2…0d3b700b` / `1bff81e2…c7bc9ac3` |
| Files | 15 artifacts + `manifest.csv`, 16 MB, every file digest-listed |

Boosted fits are thread-count sensitive: EvoTrees accumulates histogram sums in parallel, so
a fit is bit-identical only at a fixed thread count. The builder refuses to run at a thread
count other than the study's 8 unless `--allow-thread-drift` is passed, and records the
deviation in the manifest and `selected.json` when it is.

### The builder is checked against the study before a bundle is written

The same invocation also builds a **fold-2025 replica** whose training window and
out-of-fold pool are the rolling fold-2025 ones. Because that fold was scored by the study,
the builder compares its own fits against the study's persisted artifacts and refuses to
write on disagreement:

| Check | Result |
|---|---|
| Super-learner weights vs `v2_4_l1_weights.csv` (fold 2025, variant `L1e`) | 60 cells, max abs Δ = 0.0 |
| Conformal half-widths vs `v2_4_conformal.csv` (fold 2025, variant `v2_4e`) | 24 strata, max abs Δ = 0.0 nT |
| Climatology τ vs the fold manifest | 7.6990734741988325 h, exact |
| Direct-GBM configuration per step vs the fold manifest | identical at all six steps |
| T1r correction label vs the fold manifest | `operational_v2_4_t1r_year2025_memory_expert_lead_ridge100.0_fit623340`, exact |

Both bundles are then loaded through `load_v24_serving_artifacts`, the path the live engine
uses, before the build reports success.

### Two stacks share one weight schema, so the served cells name their own expert set

The study ships three fitted stacks per fold in one table — `L1`, `L1a` and `L1e` — whose
weight columns cover the widest set, so a nine-expert row carries a hard zero in
`w_static_v2_2` and is otherwise indistinguishable from a ten-expert row whose optimiser put
no mass there. Serving such a row would silently drop expert ten while the non-negativity,
unit-mass and three-member floor checks all still passed. The bundle's rows therefore carry
`expert_set` and `n_experts`, and the loader requires the served variant's cells to name
exactly the ten served experts in the served order; `guard.json` additionally records the
floor group, which the loader checks against the package's own.

## Offline identity oracle

`validation/operational/v2_4_serving_identity.jl` drives the *serving* functions —
`v24_serving_analog_features`, `v24_serving_analog_members`, `v24_serving_t1r_center`,
`v24_serving_direct_features`, `v24_serving_direct_center`,
`v24_serving_climatology_center`, `v24_serving_center` — with hourly archive inputs at zero
anchor lag against the fold-2025 replica, and compares every stage with
`learn_year_2025.csv` / `oof_year_2025.csv`.

| Column | n | max abs Δ (nT) |
|---|---:|---:|
| served_v2_1, frozen_v2_1, persistence, burton, burton_full, obrien, static_v2_2 | 4,518 each | 0.0 |
| t1_analog_raw, t1r_analog | 4,518 each | 0.0 |
| direct_gbm, climatology | 4,518 each | 0.0 |
| l1_center | 4,518 | 0.0 |
| v2_4e (served center) | 4,518 | 0.0 |
| v2_4e_lo_nt, v2_4e_hi_nt (interval endpoints) | 4,518 each | 0.0 |

753 anchors of 2025 (uniform stride ∪ the 150 deepest), all six model steps, 0 skipped and 0
served-fallback anchors; 0 guarded rows, with the deepening state and all three depth bins and
all three regimes present, so the deep cells are exercised rather than sampled away. Target
1e-9 nT; observed 0.0 nT, so the tolerance is not load-bearing.

The oracle also asserts, per row, that the serving path and the study agree on the regime,
the depth bin, the cell actually used, the deepening flag and the gated coupling proxy; that
the physical projection is inert on the compared centers (the study's `v2_4e` column carries
no projection, so a projection that ever acted would make the served center a different number
from the scored one); and that, because the bundle records no guard, the published center
equals the stack center on every row — including the deepening rows where a guard would have
acted.

That the projection is inert *on these anchors* is what the oracle can establish, and it is not the
same statement as the projection being inert. Adding `projection_applied` to the serving return and
`v24_projection_applied` to the log changes no arithmetic — the oracle was rerun and still reports
max abs Δ = 0.0 nT on every column — and closes the gap between the two statements: the live log
now records per row whether the clamp acted, so a real anchor state that reaches the ceiling is
visible instead of arriving as a center that quietly disagrees with the combination.

## Bundle-defect fixtures assert the check they were written for

The twenty-seven injected bundle defects were asserted with `@test_throws Exception`, which passes
whenever anything throws. Two of them were being caught by a different check than the fixture
documented, and in both cases the documented check was therefore untested:

- an unlisted direct-GBM model is refused by the manifest's required-artifact rule, not by the
  reader's own digest gate. That gate is unreachable through a full load, because manifest
  verification runs first; it is now reached directly through `_v24_serving_read_direct` with a
  hashed-name set that omits one model, so both refusals are exercised rather than one being
  mistaken for the other;
- a renamed served variant is refused by the selection record, not by the conformal keying the
  fixture comment described. `:conformal_variant_mismatch` was added — the conformal rows alone
  carry another variant's name — so the keying is exercised on its own. Twenty-eight defects now.

Every defect names the check it must trip, as a fragment of that check's message, and the suite
asserts that every mutation has such an entry, so a new defect cannot be added with a
"throws something" assertion.

Three further test-debt items in the same suite: the pooled conformal stratum's fixture half-width
was equal to the shallow bin's, which made the interval fallback untestable (a row that resolved to
the pooled stratum and one that resolved to the shallow stratum returned the same number) and is now
distinct at every step; `v24_serving_deepening`'s coupling branch is a strict `> 0` and is now
pinned against a `!= 0` reading with a negative and a `-1e-9` coupling on a deep, slowly recovering
ring current; and the physical projection has its own test on a synthetic panel at both the ceiling
and the floor.

## Live engine

`examples/live_forecast_verify.jl` computes the served stage after the static-stack stage
and before the interval selection. `_v2_4_served_center` is fail-closed: every failure is a
status, never an exception.

| `v24_status` | Meaning |
|---|---|
| `ok` | the row's served center is the V2.4e center |
| `fallback:deployment_absent` / `:deployment_invalid` | the bundle is missing or fails verification (remembered for a bounded 3600 s cool-down so a redeploy heals without a daemon restart) |
| `fallback:static_expert_unavailable` | the static-stack center is unavailable, so expert ten of the combination is missing |
| `fallback:static_expert_unpinned` | the static-stack center came from weights carrying no digest pin, so expert ten is not the fitted expert the tenth weight was estimated against; the row falls back to that stack center under the separate unpinned label rather than publishing unpinned weights under the V2.4 identity |
| `fallback:missing_driver_lag<N>` | a mandatory driver hour of the analog key or the direct design is not measured |
| `fallback:missing_dst_lag<N>` / `:missing_anchor_dst` / `:missing_previous_dst` | the direct-GBM design's Dst ladder or the analog key's anchor / one-hour Dst difference is incomplete |
| `fallback:unsupported_model_step`, `:calibration_absent`, `:serving_error`, `:model_not_v2_1` | the remaining fail-closed paths |
| `fallback:incomplete_analog_key`, `:incomplete_direct_design`, `:non_finite_center` | defence-in-depth branches, unreachable under the deployed loader (see below) |

Each status with a reachable code path is now produced by a test rather than assumed reachable:
the early refusals are driven directly against the state a served row logged, the two short-feed
cases end to end, and a v1 issuance for the row-level default. Three are defence-in-depth
branches and are reported as such rather than faked. `:non_finite_center` cannot fire because a
non-finite expert or combination is refused earlier and a conformal half-width must load positive
and finite. `:incomplete_analog_key` cannot fire because the key's own defect finder tests the
same five driver channels, with the same positivity condition, that the feature block's usability
predicate tests, and the block's two remaining rejections are the anchor and previous Dst — the
unit suite enumerates every single-input defect and shows each landing on a named reason, so a new
rejection condition in the feature block surfaces there. `:incomplete_direct_design` cannot fire
because the analog key is formed first and refuses the same driver defects at the lags the
design's own block reads.

Log columns: `v24_model_version`, `v24_status`, `v24_manifest_sha256`,
`v24_l1_center_dst_nt`, `v24_guard_applied`, `v24_projection_applied`, `v24_pred_dst_nt`,
`v24_ci05_nt`, `v24_ci95_nt`, `v24_regime_cell`, `v24_deepening_cell`, `v24_pooled_fallback`,
`v24_history_hours`, `v24_t1r_pred_dst_nt`, `direct_gbm_pred_dst_nt`,
`climatology_pred_dst_nt`, `v2_2_stack_pred_dst_nt`, `v2_2_stack_ci05_dst_nt`,
`v2_1_served_ci05_dst_nt`. `v24_guard_applied` is `false` on every row of the deployed bundle and
stays logged so a later guarded bundle is visible. `v24_projection_applied` is the separate
physical projection: the study's anchors sit far from the `+50` nT ceiling and the oracle asserts
the projection inert on the compared centers, but a real anchor Dst above roughly `+52` nT with a
panel to match would reach it, and an unlogged projection makes a clamped center
indistinguishable from an unclamped one. The identity `v24_pred_dst_nt == v24_l1_center_dst_nt`
therefore holds for an unguarded bundle *where the projection did not act*, and the tests assert
it under that condition rather than unconditionally. The two `*_ci05_dst_nt` columns are the
predecessor watch edges the alerting minimum is taken over; the stack one is `missing` on a row
whose stack stage could not act, because a stage that published no center published no edge. The
V2.1 continuity column, the V2.2 stage columns and the V2.3 shadow columns are unchanged. The
append path streams a schema upgrade, so an existing hot log gains the columns and its earlier
rows read back `missing`.

Two live-path details are load-bearing and are asserted by tests rather than assumed:

- the **frozen V2.1 expert is recomputed** through `v23_serving_frozen_center`, not taken
  from the logged `v2_pred_dst_nt`. The study's `frozen_v2_1` column is a rollout that holds
  the issue driver for every step, while the logged column admits L1-measured hours into the
  core rollout; substituting it moves the stack center.
- the **direct-GBM design is built from the same package function the study fitted on**
  (`v23_direct_features` over a 25-hour mini frame of Earth-arrival hourly means and Kyoto
  Dst lags), and its first 18 columns are checked against the analog key exactly. A
  disagreement means the two blocks saw different hourly means and the expert is refused.

## Dashboard and readiness

- `app/src/forecast_api.jl`: the served label chain is `(V2.4e, static stack, V2.1)`, ordered
  strongest to weakest, and a cycle is published under the weakest label any of its rows
  carries. The published severity is `v24_serving_depth_safe_center`, the shared
  dependency-free definition the package serves under, extended to any number of continuity
  partners; the watch edge is the same minimum taken over the logged predecessor edges. Each
  horizon of the payload discloses `v24_status`, `v24_pred_dst_nt`, `v24_guard_applied`,
  `v24_projection_applied`, `v24_regime_cell`, `v2_2_stack_pred_dst_nt`, both predecessor edges
  and `severity_ci05_source`; `build_status` additionally reports
  `threat.interval_lower_edge_source`, the stage whose own edge the cycle's watch was assessed
  on. `build_served_health` counts the fallback window on cycles that carry `v24_status` and
  reports which stage the window landed on.
- `app/public/app.js` names the served pipeline stages (`superlearner10floor`, `conformal`). An
  unrecognised stage token now falls back to the raw label for the whole pipeline instead of
  being dropped from the capability list, and the lookup goes through `hasOwnProperty`, so a
  token such as `toString` cannot resolve through `Object.prototype` and be presented as a
  capability. Both are behavioural, so the app suite extracts the capability block verbatim
  between its own sentinels and executes it. The alerting numbers the watch text quotes are now
  rendered: a labelled line under the served forecast reports the depth-safe severity centre and
  watch edge across the issued horizons and names the stage the edge came from. That block is
  likewise extracted and executed against a synthetic payload with a DOM stub, so the rendered text
  is checked rather than the source that produces it. Before this the alert text called those
  numbers "displayed" while nothing on the page showed them; rendering them was preferred over
  softening the wording, because a number an operator is alerted on that appears nowhere is not
  auditable.
- `validation/operational/v2_readiness_audit.jl`: `audit_v2_4_bundle!` loads the bundle the
  way the engine does and **fails** when it cannot; the fallback-rate window is keyed on
  `v24_status`; both stages of the chain disclose their per-row statuses; the newest-cycle
  label check accepts the three-label chain; the dashboard payload accepts the V2.4 sentence
  and warns (not fails) on either disclosed fallback sentence; the issue-identity contract
  requires the served label, the stack label, the bundle manifest digest **and the bundle's fold
  year and training-window bound** to agree with the loaded artifacts — the last two were written
  by `v2_1_issue_identity.jl` and read by nothing, so a record carrying another refit's provenance
  under the deployed digest passed. The bundle's identity is compared against what the bundle
  records on disk — its selection file and its manifest's build row — rather than against the
  loaded struct's `identity` field, which the type fills from the same source constant the check
  would compare it with; `_v2_4_artifacts()` in the engine reads the selection record for the same
  reason, and the loader's own refusal is unchanged. `audit_served_bundle_identity!` then ties that
  recorded identity to the label the newest cycle's served rows were published under, which is the
  only comparison of two independent statements about which product is being served. The
  newest-cycle failure rule is evaluated on the newest cycle rather than the newest *staged* one;
  when the newest cycle predates the served stage it carries no verdict and that is disclosed as
  its own check, matching the docstring and the dashboard's `newest_cycle_is_fallback`. The 1%
  fallback-rate target is reported but documented as inoperative against a ninety-six-cycle window
  — two fallbacks is already 2.1% — so the operative gates are the newest-cycle rule and the
  two-cycle threshold. The target is documented rather than raised to a binding value: the smallest
  operative setting is `2/96`, which moves the failure threshold to three fallbacks and contradicts
  the two-cycle rule the audit states and tests, so making the rate binding would be a behaviour
  change to the gate rather than a documentation fix. `audit_served_bundle_currency!` additionally compares the `v24_manifest_sha256`
  the newest cycle was **served under** with the digest of the bundle now on disk. The engine loads
  the bundle once per process and the monitor is one long-running process, so replacing the artifacts
  under a running daemon does not change what is published; every other V2.4 check reads the
  directory and would report the new bundle while every issued row still came from the old one. Only
  the newest cycle is compared, because older rows legitimately carry an earlier digest after a
  redeploy and restart.
- `validation/operational/v2_1_issue_identity.jl` records `served_model_version` = the V2.4
  identity plus `served_stack_model_version`, `served_bundle_manifest_sha256`,
  `served_bundle_fold_year` and `served_bundle_training_max_target_utc`. The last two are now read
  by the readiness identity contract; they were written and read by nothing, so a record carrying
  another refit's provenance under the deployed digest passed.

## Verification run

| Check | Result |
|---|---|
| `test/test_operational_v24_serving.jl` | 425 assertions pass; 28 injected bundle defects each refused by the check it was written for |
| `test/test_live_forecast_verify.jl` | 852 assertions pass (includes the served-V2.4e testsets and the status matrix) |
| `app/test/runtests.jl` | 1,035 assertions pass |
| `test/test_serving_identity_oracles.jl` | 199 assertions pass |
| `v2_readiness_audit.jl --self-test` | PASS, 37 independent checks |
| `validation/operational/v2_4_serving_identity.jl` | PASS, max abs Δ = 0.0 nT over 753 anchors / 4,518 rows |
| `examples/experiments.jl` | PASS (V2.1 deterministic smoke, including the V2.4 identity, depth-bin, deepening and depth-safe assertions) |
| Scratch live cycle (`live_monitor.jl --once`) | 4 rows, all `v24_status=ok`, served identity `v2.4+sindy20x11+superlearner10floor+conformal`, `interval_source=v24_conformal_depth`, cell `active_deepening/shallow`, `v24_guard_applied=false`, `v24_projection_applied=false` and `v24_pred_dst_nt = v24_l1_center_dst_nt` on every row, `v24_history_hours=12`, manifest digest `057aec0d…`, `v2_2_status=ok` with both severity partners and both predecessor edges finite; four horizons issued in ≈ 14 s wall clock after the first-load archive rebuild |
| New columns in that cycle | `v2_1_served_ci05_dst_nt` = −35.6 / −45.9 / −53.8 / −75.1 nT and `v2_2_stack_ci05_dst_nt` = −32.4 / −37.4 / −39.3 / −40.4 nT at steps 1/2/3/6, each its stage's own center less the 4.78 / 7.11 / 8.69 / 10.89 nT half-width the pre-V2.4 machinery produced for that lead; the served conformal edges were −32.9 / −35.9 / −42.3 / −44.6 nT, so the published watch edge is the V2.1 operator's −75.1 nT |
| `Pkg.test()` | 282,144 assertions pass, 0 fail, 14m22.7s |
| `dev-harness-audit.sh` on the worktree | 338 PASS / 3 WARN / 0 FAIL; the warnings are pre-existing (loose tolerances in the V2.2 tests, two absent paper directories) |

## Disclosures

- The study's own preregistered §6 rule is `SHADOW` (era E1 fails G1 and G2). What licenses
  serving is the Amendment A3 operational serve rule against the product being replaced:
  `SERVE_ELIGIBLE_PENDING_G4` on ALL and E2, with E1 disclosed only because the static stack
  is partly in sample there. E1 is not silently dropped — it is reported in
  `v2_4_serve_rule.csv` with its two failing storm-cell rows (`latest ≤ −50` at 7 h and
  recovery at 4/6/7 h) — and prospective accrual is the final arbiter.
- The served band changed source relative to the pre-V2.4 product: a V2.4e row carries the
  study's depth-stratified conformal interval instead of the shifted frozen-tail or adaptive
  band. That is the interval whose coverage the study's G3 gate scored on this center; the
  adaptive band was never calibrated on it. The switch is disclosed per row in
  `interval_source`, and the previous machinery still bands a fallback row.
- The point-forecast guard is retained as code and disabled by the artifact, not deleted.
  This is deliberate: the guard is the mitigation a future selected variant may need, and a
  disabled-but-tested path is auditable, while a deleted one would have to be rewritten from
  the decision record. The unit suite exercises it through a second fixture bundle whose own
  record enables it.
- Serving the V2.4e center costs one additional analog retrieval and rollout per issued
  horizon on top of the V2.3 shadow path, and the static-stack stage remains a precondition
  rather than a fallback-only column.

## Remaining risks

These are properties of the deployment that the identity oracle cannot test, because they are
differences between the live and offline information sets rather than differences in arithmetic. They
are stated here rather than treated as covered.

- **Provisional versus final Dst.** The live engine anchors on the Kyoto provisional/quicklook Dst
  available at issue; the study fitted and scored on final OMNI Dst. The same code path applied to the
  two series does not produce the same number, and the difference propagates into the anchor, the
  one-hour rate that selects the regime, the depth bin that selects the cell and the conformal
  stratum, the Dst ladder of the direct expert, and the analog key. The oracle is driven with archive
  inputs precisely so it isolates the arithmetic, so it is silent on this by construction: no offline
  check can bound it. Prospective accrual against the same feed the daemon reads is what measures it,
  which is what makes the accrued live record — not the oracle — the arbiter of the served product's
  skill. The revision history of the anchor also means an early cycle can be scored against a Dst
  value that was later revised; the log records the anchor it used per row, so a revision is
  reconstructable rather than silently absorbed.
- **Truncated southward-run feature under an unmeasured L1 hour.** The analog key's mandatory driver
  history is seven hours, but its consecutive-southward run-length feature reads up to twelve and
  stops at the first hour with no record. An L1 gap at lags 8–12 therefore does not refuse the key: it
  truncates the run length, so a long southward interval can present as a shorter one and retrieve
  different archive analogs. This is inherited from the V2.3 analog path and is unchanged by this
  integration. It is observable per row — `v24_history_hours` records how many of the twelve hourly
  means the key could draw on, so `v24_status = ok` with `v24_history_hours < 12` marks a row whose
  run-length feature may be truncated — and it is not currently quantified against the archive.
