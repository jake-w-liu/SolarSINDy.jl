# Live Forecast Verification

Live forecast verification must lock the prediction before the target
observation is available. A forecast is not verified merely because the code
produces a number.

## Evidence Standard

The accepted sequence is:

```text
issue forecast at t0
log target time, prediction, interval, drivers, and model state anchor
wait until observed Dst exists for the target time
compare the locked prediction to that observation
record residual and interval coverage
```

The logged row is the source of truth. Do not recompute or edit the prediction
after the target observation arrives.

## Command

From the repository root:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --wait
```

The script:

- fetches live NOAA SWPC plasma and magnetic-field data,
- fetches the latest SWPC Kyoto Dst observation,
- anchors the rolling Dst* state to the latest observed Dst,
- forecasts a strictly future hourly Dst target,
- appends the locked prediction to `live_forecasts/live_forecast_log.csv`,
- logs persistence, Burton, BurtonFull, O'Brien--McPherron, and SINDy
  predictions for the same target,
- polls the Dst feed until the target observation appears,
- updates the same log row with the observed Dst, residual, and 90% interval
  coverage flag.

Available modes:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --issue
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --verify-pending
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --backfill-baselines
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --replay-recent
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --fit-v2-calibration
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --campaign
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --summary
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --comparison-report
```

Use `--poll-seconds=N`, `--timeout-hours=N`, `--horizon-hours=N`, and
`--log=PATH` to adjust the run.

Use `--campaign --campaign-horizons=1,2,3,6 --model=v2` to issue multiple
future operational-v2 forecasts, poll the observation feed, verify the locked
rows, and regenerate the comparison report from one command. `--campaign`
defaults to `--model=v2` if no model is specified.

Use `--backfill-baselines` after schema changes to fill baseline forecasts and
residuals for older rows that already contain locked SINDy predictions and live
driver values.

Use `--replay-recent --replay-hours=N` to build a longer predicted-versus-
observed comparison table from the recent live feeds. The replay is causal: each
row anchors on the observed Dst at the issue hour, persists the previous
complete solar-wind hour into the one-hour-ahead target, and compares the
prediction with the already-published target Dst. This provides more rows for
diagnostics, but it is not a substitute for locked forecasts that were issued
before observations arrived.

Use `--comparison-report` after `--verify-pending` to create the standard
locked-live report:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --comparison-report \
  --log=live_forecasts/live_forecast_log.csv \
  --report=live_forecasts/live_comparison_report.md
```

The headline comparison in this report is a same-row table:
`Operational v2` versus `SINDy v1`, persistence, Burton, BurtonFull, and
O'Brien--McPherron. `Operational v2` is the upgraded method. Internal v2
component/selector metadata is reported only in the audit section and is not a
separate headline model.

## Operational v2 Calibration

The default live model is `--model=v1`, which preserves the paper model path.
`--model=v2` enables an experimental operational wrapper that applies a causal
ridge residual correction to the v1 forecast and inflates the prediction
interval. The correction uses only issue-time fields: latest Dst, solar-wind
speed, IMF components, density, dynamic pressure, and derived causal coupling
features such as southward IMF, `V Bs`, transverse IMF magnitude, IMF clock-angle
coupling, and square-root dynamic pressure.

Current v2 calibration is guarded and baseline-aware. When replay or live-log
baseline columns are available, the calibration stores skill estimates for
corrected SINDy-v2, persistence, Burton, BurtonFull, and O'Brien--McPherron. The
issued operational v2 output is the upgraded forecast. The internal
`v2_selected_component` field records whether v2 used corrected SINDy directly
or an allowed guarded component for auditability; it should not be read as a
second deployed model.

Fit the calibration from a prior replay or locked live log:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --fit-v2-calibration \
  --table=live_forecasts/live_replay_144h.csv \
  --v2-calibration=live_forecasts/operational_v2_calibration.csv \
  --v2-selector-margin=0.5
```

Then run a calibrated replay or live issue:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --replay-recent \
  --model=v2 \
  --v2-calibration=live_forecasts/operational_v2_calibration.csv \
  --replay-hours=144 \
  --table=live_forecasts/live_replay_v2_144h.csv
```

The v2 calibration is not evidence of industrial readiness by itself. It must be
scored chronologically against held-out rows and then accumulated through locked
live forecasts exactly like v1.

## Interpretation

One verified live row proves that the live ingestion, forecast, and observation
comparison loop worked for that case. General reliability requires many locked
forecast rows scored the same way, without discarding misses or changing the
model after seeing the observation.
