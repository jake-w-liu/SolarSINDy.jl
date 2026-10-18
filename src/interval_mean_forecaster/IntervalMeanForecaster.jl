"""
    IntervalMeanForecaster

Optional interval-mean Dst research implementation. The namespace provides
receipt-aware hourly streaming, constrained gain fitting, prediction, and
source-bound model bundles. Its bounded experiment driver accepts explicitly
declared synthetic inputs. Availability assumptions are explicit (`delay_hours`
is 0 or 1); forecast uncertainty is conditional on the supplied forcing scenario.

These library interfaces do not enable serving or establish real-data forecast
skill. See the interval-mean forecaster documentation for the evidence boundary.
"""
module IntervalMeanForecaster

include("DstIntervalAdapter.jl")
include("DstIntervalStream.jl")
include("DstIntervalTraining.jl")
include("DstIntervalArtifacts.jl")
include("DstIntervalExperiment.jl")

export HourlyInputs, StreamCheckpoint, read_author_hourly, stream_basis,
       forecast_clock, fit_eligible, joint_fit_mask, settings, nnls2,
       interval_kernel, interval_moments, condition_interval,
       fit_gains, predict_rows, select_setting, save_bundle, load_bundle,
       run_synthetic_experiment, source_identity

"""Hourly channel values and hourly tags with explicit source digest and availability kind."""
const HourlyInputs = DstIntervalStream.HourlyInputs
"""A resumable stream state bound to the recipe, source, and consumed input prefix."""
const StreamCheckpoint = DstIntervalStream.StreamCheckpoint
"""Read the 57-column hourly author format, requiring its exact SHA256 digest."""
const read_author_hourly = DstIntervalAdapter.Features.read_author_hourly
"""Stream basis rows with explicit latency, decay times, noise, chunk size and optional checkpoint."""
const stream_basis = DstIntervalStream.stream_basis
"""Construct issue, driver, anchor and target clocks for a 4- or 6-hour horizon and latency 0 or 1."""
const forecast_clock = DstIntervalAdapter.forecast_clock
"""Check issue, target completion and label receipt against the embargoed fitting boundary."""
const fit_eligible = DstIntervalAdapter.fit_eligible
"""Return the joint complete-and-received fitting mask for both target horizons."""
const joint_fit_mask = DstIntervalAdapter.joint_fit_mask
"""Return the fixed 18 decay/noise/regularization recipes; this does not select a recipe."""
const settings = DstIntervalAdapter.settings
"""Solve the two-column nonnegative weighted least-squares problem with candidate KKT checks."""
const nnls2 = DstIntervalAdapter.nnls2
"""Compute one scalar continuous-time endpoint/interval kernel with validated finite inputs."""
const interval_kernel = DstIntervalAdapter.Kernel.interval_kernel
"""Compute the joint endpoint and interval-mean Gaussian moments for the supplied forcing."""
const interval_moments = DstIntervalAdapter.Kernel.interval_moments
"""Condition joint endpoint/interval moments on one noisy interval-mean observation."""
const condition_interval = DstIntervalAdapter.Kernel.condition_interval
"""Fit nonnegative gains from eligible rows at an explicit embargoed boundary and recipe."""
const fit_gains = DstIntervalTraining.fit_gains
"""Predict both horizons from a source-bound fitted model and compatible streamed rows."""
const predict_rows = DstIntervalTraining.predict_rows
"""Apply the fixed validation rule to complete panels and an explicitly admitted reference."""
const select_setting = DstIntervalTraining.select_setting
"""Save a source-bound model and checkpoint in a new bundle directory with explicit provenance."""
const save_bundle = DstIntervalArtifacts.save_bundle
"""Load a bundle only with its expected manifest digest and expected implementation sources."""
const load_bundle = DstIntervalArtifacts.load_bundle
"""Run the bounded synthetic initial-fit/validation/refit/bundle control flow for both latencies."""
const run_synthetic_experiment = DstIntervalExperiment.run_synthetic_experiment

"""
    source_identity()

Verify loaded bytes against disk and return the exact eight implementation
filename/SHA256 pairs. Bundle identities include these implementation digests.
"""
source_identity() = DstIntervalExperiment.verify_source_identity()

end
