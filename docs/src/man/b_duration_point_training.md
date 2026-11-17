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

Saved provenance binds independently admitted input and reference identities,
the model, and the SHA-bound replayed basis. Both bundle save and load compare
every scientific value. An export context without the separate expected-provenance
scope supports fit-input export only. Initial publication records exactly two
hazard fits, 36 gain fits, zero final or calibration fits, and false development,
scientific-verdict and serving flags.

Matched-reference admission checks the complete prediction, eight-setting tuning,
grid and independent-audit member tree. Each lead uses its full raw cohort. The
reader verifies every candidate column, recomputes maintained RMSE, MAE and bias,
checks both stored and recomputed first strict minima, and compares the selected
columns bitwise. The original-reference diagnostic has its own data-only adapter
and remains L0 only.

Ordinary actual fitting uses admitted source files, external artifact pins and
the current numerical source identity. Historical qualification fields and files
remain readable as hashed provenance; readers do not load grants or execution
permits. Runtime, allocation, process-lifetime peak RSS and thread observations
are reported during execution. The matched342 actual entry fits all four
delay/lead cells and eight recipes per cell, independently audits the saved
models and predictions, then publishes and reads back the reference artifacts.
See `tools/actual_b_chain/README.md` for the request format and command.

All input and artifact readers enforce bounded ordinary paths, closed field sets,
fixed binary records, source ownership and entry/exit checks. The initial stage
leaves primary joint-delay selection, later refits and distribution calibration
to their respective scientific stages. Package tests and numerical examples
provide software verification; predictive claims require separate evaluation.

Run `examples/b_point_training_synthetic.jl` for a generated 1900-start,
source-to-36-fit-to-bundle-to-prediction example with independent readback. It uses
temporary files and supplies no archive observations or forecasting verdict.
