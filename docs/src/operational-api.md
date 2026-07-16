# Operational API

Discovery, baseline, data-processing, metric, and utility functions are
documented in the [core API](api.md).

## Forecasting And Alarms

```@docs
ForecastState
ForecastResult
OperationalV2Calibration
default_operational_v2_calibration
operational_v2_feature_tuple
init_forecast
step_forecast!
forecast_ahead
fit_operational_v2_calibration
operational_v2_predict
score_operational_v2
write_operational_v2_calibration
read_operational_v2_calibration
StormSeverity
QUIET
MODERATE
INTENSE
SUPERINTENSE
Alarm
AlarmConfig
default_alarm_config
classify_severity
check_alarm
maybe_fire_horizon_alarm!
alarm_print
alarm_log
fetch_swpc_plasma
fetch_swpc_mag
fetch_swpc_dst
fetch_realtime_solar_wind
run_monitor
recover_shadow_state
DEFAULT_FEED_DEADMAN_THRESHOLD
feed_deadman_tripped
storm_lambda_grid
select_storm_lambda
write_storm_lambda_selection
read_storm_lambda_selection
```

## Conformal Calibration

```@docs
ConformalCalibration
ConformalStratum
fit_conformal
conformal_stratum
conformal_halfwidth
conformal_interval
conformal_coverage
write_conformal_calibration
read_conformal_calibration
AdaptiveConformal
init_adaptive_conformal
adaptive_conformal_step!
run_adaptive_conformal
```

## Online Assimilation

```@docs
AssimilationFilter
init_assimilation
current_dst
current_coeffs
dst_variance
assimilation_predict!
assimilation_update!
run_assimilation
```
