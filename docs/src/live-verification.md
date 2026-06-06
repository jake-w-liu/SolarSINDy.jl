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
- forecasts the next unavailable hourly Dst target,
- appends the locked prediction to `live_forecasts/live_forecast_log.csv`,
- polls the Dst feed until the target observation appears,
- updates the same log row with the observed Dst, residual, and 90% interval
  coverage flag.

Use `--no-wait` to create only the locked forecast row. Use `--poll-seconds=N`,
`--timeout-hours=N`, `--horizon-hours=N`, and `--log=PATH` to adjust the run.

## Interpretation

One verified live row proves that the live ingestion, forecast, and observation
comparison loop worked for that case. General reliability requires many locked
forecast rows scored the same way, without discarding misses or changing the
model after seeing the observation.
