# Interval-mean forecaster

`SolarSINDy.IntervalMeanForecaster` is an optional research namespace for hourly
Dst interval observations. It implements continuous-time endpoint/interval
Gaussian moments, receipt-aware replay, an affine stream basis, nonnegative gain
fitting, predictions at four- and six-hour horizons, and source-bound model
bundles. Importing the library makes the API available; default monitoring and
serving paths do not call it.

The hourly input convention uses tags at interval starts. An `h`-hour forecast
issued at `t` targets the hourly mean over `[t + Hour(h-1), t + Hour(h))`,
for `h = 4` or `6`. An issue at hour `t` admits the driver interval tagged
`t - Hour(1)`. `delay_hours=0` assumes that the
completed Dst interval ending at `t` is available at the issue; `delay_hours=1`
uses the preceding interval. These are explicit availability assumptions, not
measurements of operational publication latency. Fitting checks both target
completion and receipt against the embargoed boundary. Uncertainty is
conditional on the forcing supplied to the model.

The shipped example generates the 57-column hourly input format, checks its
SHA256, then streams, fits, predicts, saves and reloads separate L0/L1 bundles.
It uses an explicit fixed recipe and synthetic dates. The bounded synthetic
experiment interface also exercises the initial-fit, validation, final-fit and
publication control flow. Those checks demonstrate implemented numerical and
library behavior. They do not establish empirical forecast skill, qualify an
actual archive fit or authorize deployment.

The input reader requires the caller's expected raw-file digest. Streams can
resume only from a checkpoint with a matching recipe and consumed input prefix.
Bundle loading requires the expected manifest digest and implementation source
map. `source_identity()` verifies all eight loaded implementation files against
their current disk bytes. Bundle directories must be new.

For longer research panels, `export_fit_input` accepts a producer callback whose
sink receives nonempty affine chunks of at most 4096 rows, with at most 500000
source rows. It writes the eligible joint horizon design into a new directory
and publishes its pinned manifest after successful completion.
`fit_gains_from_input` requires that manifest digest and produces the same
canonical model as `fit_gains` for identical rows and source bytes. Predictions
and bundle save/load use the existing interfaces. Export buffering is bounded;
the solver still materializes the full matrix. The caller owns the directory
and can record the fit-input manifest digest in bundle provenance. This path
provides fitting mechanics without qualifying an actual archive experiment.
Canonical model fitting through either fit API requires an exact-hour training
boundary, matching the model bundle contract. The affine exporter also supports
fractional boundaries for embargo diagnostics; those inputs cannot be converted
to a canonical model until exported with a valid exact-hour boundary.

```jldoctest
julia> using SolarSINDy

julia> F = SolarSINDy.IntervalMeanForecaster;

julia> length(F.settings())
18

julia> F.interval_kernel(0.0, 2.0, 3.0).phi
1.0
```

## Public namespace

```@docs
SolarSINDy.IntervalMeanForecaster
SolarSINDy.IntervalMeanForecaster.HourlyInputs
SolarSINDy.IntervalMeanForecaster.StreamCheckpoint
SolarSINDy.IntervalMeanForecaster.read_author_hourly
SolarSINDy.IntervalMeanForecaster.stream_basis
SolarSINDy.IntervalMeanForecaster.forecast_clock
SolarSINDy.IntervalMeanForecaster.fit_eligible
SolarSINDy.IntervalMeanForecaster.joint_fit_mask
SolarSINDy.IntervalMeanForecaster.settings
SolarSINDy.IntervalMeanForecaster.nnls2
SolarSINDy.IntervalMeanForecaster.interval_kernel
SolarSINDy.IntervalMeanForecaster.interval_moments
SolarSINDy.IntervalMeanForecaster.condition_interval
SolarSINDy.IntervalMeanForecaster.fit_gains
SolarSINDy.IntervalMeanForecaster.export_fit_input
SolarSINDy.IntervalMeanForecaster.fit_gains_from_input
SolarSINDy.IntervalMeanForecaster.predict_rows
SolarSINDy.IntervalMeanForecaster.select_setting
SolarSINDy.IntervalMeanForecaster.save_bundle
SolarSINDy.IntervalMeanForecaster.load_bundle
SolarSINDy.IntervalMeanForecaster.run_synthetic_experiment
SolarSINDy.IntervalMeanForecaster.source_identity
```

The fixed independent Gaussian/replay oracles retain their original tolerances.
Uncached and optimized source fixtures are also compared on each tested runtime
and platform. That checks before/after bit preservation within the platform;
it does not promise universal bit identity across BLAS implementations,
architectures or operating systems. Native Windows, Linux and macOS
qualification remains a recorded release gate.

## Implementation module references

The following references describe the numerical, replay, feature, streaming,
training and bundle interfaces used by the public namespace above. The same
research scope and explicit availability assumptions apply to these interfaces.

```@autodocs
Modules = [SolarSINDy.IntervalMeanForecaster.DstIntervalAdapter.Kernel,
           SolarSINDy.IntervalMeanForecaster.DstIntervalAdapter.Replay,
           SolarSINDy.IntervalMeanForecaster.DstIntervalAdapter.Features,
           SolarSINDy.IntervalMeanForecaster.DstIntervalAdapter,
           SolarSINDy.IntervalMeanForecaster.DstIntervalStream,
           SolarSINDy.IntervalMeanForecaster.DstIntervalTraining,
           SolarSINDy.IntervalMeanForecaster.DstIntervalArtifacts,
           SolarSINDy.IntervalMeanForecaster.DstIntervalExperiment]
Private = false
```
