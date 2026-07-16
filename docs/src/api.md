# Core API

Forecast issuance, alarms, calibration, and assimilation are documented in the
[operational API](operational-api.md).

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
collinearity_diagnostics
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
add_original_observation_flags!
original_sindy_mask
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
mae
correlation
skill_score
prediction_efficiency
metrics_summary
wilcoxon_signed_rank_p
paired_storm_statistics
write_paired_storm_statistics
holm_adjust
write_holm_adjustment
```

## Utilities

```@docs
numerical_derivative
smooth_moving_average
pressure_correct_dst
dynamic_pressure
dst_to_dst_star
dst_star_to_dst
resolve_pdyn
halfwave_rectify
imf_clock_angle
get_data_dir
```
