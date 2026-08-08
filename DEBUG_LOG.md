# V2.1 migration debug log

## Scope

This log records defects found while migrating the operational package from the
historical V2.0 21-candidate/10-active-term core to the revised V2.1
20-candidate/11-active-term core. V2.0 remains available only through its
explicit historical artifact boundary.

## Confirmed defects and fixes

1. **Bare V2 paths loaded the historical discovery core.** The monitor,
   dashboard, replay code, and examples could resolve unqualified V2 artifacts
   to the 21-term file. A versioned artifact resolver now maps `v2` to V2.1 and
   requires an explicit `v2.0` request for the archived core. It validates term
   order, candidate count, active support, pressure terms, joint-draw shape, and
   removal or presence of `n*V^2`, as appropriate.
2. **Current and historical calibration could be mixed.** Calibration paths are
   now resolved through the same version boundary as the core. V2.0 point and
   conformal files are retained under `deploy/historical/v2_0/`; unqualified
   deployment files are V2.1.
3. **Replay row slicing used a Dst mask on a shorter driver table.** Storm
   windows with unequal driver and Dst support could throw a bounds error or
   select the wrong rows. `_slice_replay_window` now applies independent masks
   and all affected replay paths use it. A regression fixture deliberately uses
   unequal support.
4. **EKF replays omitted a current calibration feature.** The revised
   26-feature schema requires `baseline_spread_nt`; the EKF development paths
   used an older feature tuple. `_v2_calibration_features` now constructs the
   shared feature record, and the EKF scripts use the current core and schema.
5. **Two operational diagnostics called a stale `_run_v2` signature.** The
   research scorecard and sustained-southward-Bz stress replay now pass the
   current core and calibration explicitly. Both scripts execute successfully.
6. **Standalone live verification defaulted to V1.** The CLI default now issues
   V2.1, matching the monitor, API, and dashboard. Explicit `--model=v1` remains
   supported. A regression assertion pins the default issue model to `:v2`.
7. **Experimental tail reports used an ambiguous comparator label.** Look-ahead,
   envelope, fixed-composition, sub-hourly, ballistic, and EKF reports now state
   that they share the revised 20/11 core but replace or omit served-tail
   components. Their comparator is the V2.1 frozen-tail ablation, not the full
   served product, and they are labeled development lineage rather than
   promotion evidence.
8. **The one-minute OMNI HRO acquisition path was not reproducible.** A
   deterministic NASA CDAWeb monthly fetcher now validates month identity and
   file structure and writes SHA-256 provenance. Its initial temporary-file call
   used an incompatible `mktemp` form under Julia 1.12.6; the call was corrected
   and all eight required months were fetched and verified.
9. **Unqualified coefficient and research-result snapshots remained on the
   retired library.** `real_sindy_coefficients.csv` and the operational-paper
   mirrors still contained 21 rows, while phase, coupled, and legacy synthetic
   snapshots still contained `n*V^2`. The real-data snapshots were synchronized
   byte-for-byte from the verified final canonical revision run; synthetic
   snapshots and PlotlySupply figures were regenerated with the current
   identifiable library. A package test now rejects `n*V^2` in every
   unqualified CSV below `data/`.
10. **Paper staging did not validate or mirror the complete core artifact set.**
    The V2.1 staging program now checks the 20/11 point fit, stability table,
    inclusion summary, and 500 joint draws; it checks active pressure terms and
    cross-file equality before atomically staging both source records and
    canonical paper mirrors.
11. **The broad operational replay consumed the superseded storm-catalog
    schema.** Refreshing the package catalog from the revised canonical run
    changed `min_dst`/`min_dst_time` to the scientifically explicit
    `min_dst_star`/`min_dst_star_time`, reduced duplicate/invalid event entries,
    and renumbered event identifiers. The replay now requires the revised
    714-event schema and independently pins its 193/30/8 events below
    -100/-200/-300 nT. A regression fixture verifies that a legacy-schema file
    fails closed instead of silently changing the cohort.
12. **The state-inertia constant still reflected the superseded broad cohort.**
    Re-running the fail-closed selector on the corrected 193-storm archive chose
    a two-hour near-quiet model weight of 0.625 rather than the deployed 0.75.
    The live constant, identity audit, replay self-test, and independent operator
    regression now pin 0.625. All affected operational replays and paper
    products are regenerated from that selected center.
13. **Two final checks encoded environment-dependent or ambiguous V2 labels.**
    The operational-evidence path test inferred precedence from directory
    existence even when the generated directory contained the complete replay
    evidence, and the live comparison report labeled historical rows as V2.1
    whenever the newer served columns were absent. The path regression now
    applies the production artifact-completeness rule. The report boundary now
    labels legacy-only rows as Historical V2.0 and uses V2.1 only when served
    V2.1 columns are present; focused regressions pass for both schemas.
14. **The forecast-log retention fixture still emitted the historical served
    label.** The production validator correctly rejected those rows after the
    V2.1 boundary was tightened, but three retention assertions still built
    their synthetic row with `model_version="v2"` and the older served-model
    label. The fixture now uses the exported current-version constants. Its
    focused 300 assertions and the complete package suite pass without
    weakening the validator. Generic script entry points were also given
    script-specific names so package-owned includes no longer overwrite a
    shared `main()` method.

## Verification record

- Operational artifact-boundary regression: 68/68 assertions passed after the
  snapshot cleanup.
- Development-lineage replays for hourly look-ahead, fixed envelope, and fixed
  A+B composition completed successfully with zero current/frozen fairness gap.
- Measured-timeshift and ballistic sub-hourly component replays completed from
  SHA-verified NASA HRO files and improved over the V2.1 frozen-tail ablation at
  every tested lead; these remain component results, not served-product claims.
- The served V2.1 severe, broad, and exact Kp/G3+ replays use matched rows and the
  explicit historical V2.0 comparator. The complete package suite passes
  4,368/4,368 assertions, the deterministic experiment entry point passes, the
  strict documentation build completes without warnings or errors, the
  development harness records 136 PASS, one justified tolerance advisory, and
  zero failures, and the readiness audit records 132 PASS, four disclosed
  evidence-boundary warnings, and zero failures.
- The operational path regression passed 28/28 assertions, and the complete
  live verification workflow passed 550/550 assertions after the final
  V2.0/V2.1 report-label correction.
