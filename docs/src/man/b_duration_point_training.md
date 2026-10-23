# Duration point training and artifacts

```@docs
SolarSINDy.DurationPointTraining
SolarSINDy.DurationPointArtifacts
```

`DurationPointTraining` provides the source-bound B point design and gain primitives.
The initial executor is `tools/actual_b_chain/ActualBChain.jl`. It accepts an
independently admitted input prefix and a verified reference, then fits two duration
laws and all 36 initial L0 gain settings. Every setting must have at least 100
informative joint h4/h6 training origins before any gain solve starts.

The in-sample two-stage transform uses a private training context that binds the
fold, complete mature mask, source, duration law and role. Public duration forecasts
retain their issue-at-or-after-model-boundary requirement. The two hazard laws use
beta 5 and 20; nine historical A schedules share their two forcing calculations at
each origin. Nine cooperative A tasks deliver detached issue states and wait until
the coordinator has constructed all 18 continuations before reading that origin's
targets. Training uses the joint h4/h6 cohort. Validation keeps each lead's complete
cohort and their union, including terminal h4-only origins.

Compute the thresholded forcing `q` first, then its exact duration-law expectation
at each forecast hour. The point recurrence uses these expectations with the fixed
A state. Hazard estimation, gain estimation and subsequent model evaluation have
separate roles; the training transform alone establishes no forecasting result.

`DurationPointArtifacts` owns the B schemas, typed readers and independent
raw-source basis replay. Each point bundle contains six files:

- `manifest.json`, published last;
- `model.json`, `checkpoints.json` and `provenance.json`;
- `hazard/model.json` and `hazard/manifest.json`.

The hazard pair embeds the unchanged duration model data schema. A standalone
duration artifact has its own completion contract; the public duration loader
rejects this embedded directory. B bundle readers reconstruct and validate the
duration law and verify its identity directly.

Saved provenance has exactly 22 fields. Its expected values come from independently
admitted input, reference and execution authority, the model, and the SHA-bound
replayed basis. Both bundle save and load compare every value. An export context
without the separate expected-provenance scope supports fit-input export only.
Initial publication records exactly two hazard fits, 36 gain fits, zero final or
calibration fits, and false development, scientific-verdict and serving flags.

Matched-reference admission requires the exact 22-member prediction, eight-setting
tuning, grid, independent-audit and qualification tree. Each lead uses its full raw
cohort. The reader verifies every candidate column, recomputes maintained RMSE,
MAE and bias, checks both stored and recomputed first strict minima, and compares
the selected columns bitwise. The original-reference diagnostic uses a separate
four-member adapter and remains L0 only. A matched GBM implementation and its
independent qualification are separate prerequisites.

Reference-component and point-execution software qualification use distinct,
versioned private wrappers. Each wrapper binds its own externally supplied head,
complete source map and whole-file SHA, plus raw gate, log, harness and source-proof
identity links. Root-issued software wrappers require independent full-gate review;
synthetic parser fixtures grant no actual execution authority. Qualification
counters retain canonical unsigned integer tokens. Existing release qualification
receipts require their declared format rather than implicit reinterpretation.

All input and artifact readers enforce bounded ordinary paths, closed field sets,
fixed binary records, source ownership and entry/exit checks. The initial stage
leaves primary joint-delay selection, later refits and distribution calibration to
their separately admitted stages. Actual execution additionally requires a new
qualified point-stage permit.

Run `examples/b_point_training_synthetic.jl` for a generated 1900-start,
source-to-36-fit-to-bundle-to-prediction example with independent readback. It uses
temporary files and supplies no archive observations or forecasting verdict.
