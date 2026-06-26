# API

## Module

```@docs
SolarSINDy
```

## Discovery

```@docs
CandidateLibrary
build_solar_wind_library
build_minimal_library
evaluate_library
get_term_names
stlsq
sindy_discover
ensemble_sindy
sindy_predict
simulate_sindy
sweep_lambda
```

## Baselines

```@docs
burton_model
burton_model_full
newell_coupling
obrien_mcpherron_model
simulate_burton
simulate_burton_full
simulate_obrien
```

## Data And Storm Processing

```@docs
SolarWindData
StormEvent
generate_synthetic_storm
generate_multistorm_dataset
prepare_sindy_data
identify_storm_phases
download_omni2
prepare_omni_data
extract_omni2_columns
parse_omni2
load_omni2_csv
clean_omni_data!
StormCatalogEntry
build_storm_catalog
extract_storm_data
extract_all_storms
save_storm_catalog
load_storm_catalog
```

## Metrics

```@docs
rmse
correlation
skill_score
prediction_efficiency
metrics_summary
wilcoxon_signed_rank_p
```

## Forecasting And Alarms

```@docs
ForecastState
ForecastResult
OperationalV2Calibration
operational_v2_feature_tuple
init_forecast
step_forecast!
forecast_ahead
fit_operational_v2_calibration
operational_v2_predict
StormSeverity
Alarm
AlarmConfig
default_alarm_config
classify_severity
check_alarm
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
assimilation_predict!
assimilation_update!
run_assimilation
```

## Utilities

```@docs
numerical_derivative
smooth_moving_average
pressure_correct_dst
halfwave_rectify
imf_clock_angle
get_data_dir
```
