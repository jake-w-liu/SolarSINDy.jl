"""
    SolarSINDy

Julia tools for sparse equation discovery and forecast-style evaluation of
solar wind-magnetosphere coupling models.
"""
module SolarSINDy

using LinearAlgebra
using Statistics
using Random
using CSV
using DataFrames
using Dates
using FileWatching: Pidfile

include("utils.jl")
include("library.jl")
include("baselines.jl")
include("data.jl")
include("sindy.jl")
include("storm_selection.jl")
include("data_pipeline.jl")
include("data_cleaning.jl")
include("metrics.jl")
include("performance_statistics.jl")
include("forecast.jl")
include("operational_artifacts.jl")
include("operational_v22_history.jl")
include("operational_v22.jl")
include("operational_v22_residual.jl")
include("operational_v22_boost.jl")
include("operational_v22_driver.jl")
include("operational_v22_arrival.jl")
include("operational_v22_core_path.jl")
include("operational_v22_error_state.jl")
include("operational_v22_error_exogenous.jl")
include("conformal.jl")
include("operational_v22_shadow_chain.jl")
include("assimilation.jl")
include("alarm.jl")
include("realtime.jl")
include("monitor.jl")
include("operational_v23_features.jl")
include("operational_v23_analog.jl")
include("operational_v23_gbm.jl")
include("operational_v22_serving.jl")
include("operational_v23_stats.jl")
include("operational_v23_serving.jl")
include("operational_v24_serving.jl")

export # Utils
       numerical_derivative, smooth_moving_average, pressure_correct_dst,
       halfwave_rectify, imf_clock_angle, get_data_dir,
       # Canonical Dst*/pressure helpers
       dynamic_pressure, dst_to_dst_star, dst_star_to_dst, resolve_pdyn,
       # Library
       CandidateLibrary, build_solar_wind_library, build_minimal_library,
       evaluate_library, get_term_names,
       # SINDy
       stlsq, sindy_discover, ensemble_sindy, sindy_predict,
       simulate_sindy, sweep_lambda, collinearity_diagnostics,
       storm_lambda_grid, select_storm_lambda, write_storm_lambda_selection,
       read_storm_lambda_selection,
       # Baselines
       burton_model, burton_model_full, newell_coupling, obrien_mcpherron_model,
       simulate_burton, simulate_burton_full, simulate_obrien,
       # Synthetic data
       SolarWindData, StormEvent, generate_synthetic_storm,
       generate_multistorm_dataset, identify_storm_phases,
       prepare_sindy_data,
       # Real data pipeline
       download_omni2, prepare_omni_data, extract_omni2_columns, parse_omni2, load_omni2_csv,
       # Data cleaning & storm catalog
       clean_omni_data!, StormCatalogEntry,
       add_original_observation_flags!, original_sindy_mask,
       build_storm_catalog, extract_storm_data, extract_all_storms,
       save_storm_catalog, load_storm_catalog,
       # Metrics
       rmse, mae, correlation, skill_score, prediction_efficiency,
       metrics_summary, wilcoxon_signed_rank_p,
       paired_storm_statistics, holm_adjust,
       write_paired_storm_statistics, write_holm_adjustment,
       # Forecast
       ForecastState, ForecastResult, init_forecast,
       step_forecast!, forecast_ahead,
       OperationalV2Calibration, default_operational_v2_calibration,
       operational_v2_feature_tuple,
       fit_operational_v2_calibration, operational_v2_predict,
       score_operational_v2, write_operational_v2_calibration,
       read_operational_v2_calibration,
       # Versioned operational core
       OPERATIONAL_V2_1_MODEL_VERSION, OPERATIONAL_V2_0_MODEL_VERSION,
       OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS,
       OperationalCoreArtifacts, OperationalCalibrationArtifacts,
       OperationalCore, canonical_operational_version,
       operational_core_artifacts, operational_calibration_artifacts,
       validate_operational_core_artifacts, load_operational_core,
       init_operational_forecast, operational_core_forecast,
       # Operational V2.2-M1 causal sparse-history core
       OPERATIONAL_V22_HISTORY_TERMS,
       OPERATIONAL_V22_HISTORY_SUPPORTED_ANCHOR_LAGS,
       OPERATIONAL_V22_HISTORY_SCHEMA_VERSION,
       OPERATIONAL_V22_HISTORY_DEFAULT_COUPLING_BOUND_MVM,
       OperationalV22HistoryDriver, OperationalV22HistoryState,
       OperationalV22HistoryArtifact,
       operational_v22_history_rho, operational_v22_history_coupling,
       operational_v22_history_memory, operational_v22_history_features,
       operational_v22_history_derivative, operational_v22_history_step,
       init_operational_v22_history_state, operational_v22_history_rollout,
       fit_operational_v22_history, operational_v22_history_sha256,
       write_operational_v22_history, read_operational_v22_history,
       # Operational V2.2 constrained stack
       OPERATIONAL_V22_COMPONENTS, DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS,
       OperationalV22Cell, OperationalV22Stack,
       operational_v22_regime, fit_operational_v22_stack,
       operational_v22_predict, score_operational_v22,
       write_operational_v22_stack, read_operational_v22_stack,
       # Operational V2.2 secondary residual
       OPERATIONAL_V22_RESIDUAL_FEATURES,
       OPERATIONAL_V22_RESIDUAL_RIDGE_GRID,
       OPERATIONAL_V22_RESIDUAL_TOP_K_GRID,
       OperationalV22ResidualCell, OperationalV22ResidualCore,
       fit_operational_v22_residual, operational_v22_residual_predict,
       score_operational_v22_residual,
       write_operational_v22_residual, read_operational_v22_residual,
       # Operational V2.2 portable boosted residual
       OPERATIONAL_V22_BOOST_FEATURES,
       OPERATIONAL_V22_BOOST_SUPPORTED_MODEL_STEPS,
       OPERATIONAL_V22_BOOST_SCHEMA_VERSION,
       OperationalV22BoostArtifact,
       extract_operational_v22_boost, fit_operational_v22_boost,
       operational_v22_boost_raw_predict, operational_v22_boost_predict,
       score_operational_v22_boost,
       write_operational_v22_boost, read_operational_v22_boost,
       # Operational V2.2-M2 sparse driver continuation
       OPERATIONAL_V22_DRIVER_STATES, OPERATIONAL_V22_DRIVER_LAGS,
       OPERATIONAL_V22_DRIVER_CADENCE_MINUTES,
       OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS,
       OPERATIONAL_V22_DRIVER_STABILITY_TOLERANCE,
       OPERATIONAL_V22_DRIVER_RIDGE_GRID,
       OPERATIONAL_V22_DRIVER_THRESHOLD_GRID,
       OperationalV22DriverArtifact,
       fit_operational_v22_driver,
       operational_v22_driver_coefficients,
       operational_v22_driver_support,
       operational_v22_driver_companion,
       operational_v22_driver_spectral_radius,
       operational_v22_driver_rollout,
       operational_v22_driver_sha256,
       write_operational_v22_driver, read_operational_v22_driver,
       # Operational V2.2-M2 receipt transport and arrival queue
       OPERATIONAL_V22_ARRIVAL_PAIR_SCHEMA_VERSION,
       OPERATIONAL_V22_ARRIVAL_SCHEMA_VERSION,
       OPERATIONAL_V22_ARRIVAL_PATH_SCHEMA_VERSION,
       OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS,
       OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES,
       OPERATIONAL_V22_ARRIVAL_TRAILING_MINUTES,
       OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS,
       OPERATIONAL_V22_ARRIVAL_PATH_STEPS,
       OPERATIONAL_V22_ARRIVAL_MIN_DELAY_MINUTES,
       OPERATIONAL_V22_ARRIVAL_MAX_DELAY_MINUTES,
       OPERATIONAL_V22_ARRIVAL_MAX_FRESHNESS_MINUTES,
       OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM,
       OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM,
       OperationalV22ArrivalBin, OperationalV22ArrivalQueue,
       OperationalV22ArrivalPathStep, OperationalV22ArrivalPath,
       build_operational_v22_arrival_queue,
       operational_v22_arrival_history,
       operational_v22_arrival_sha256,
       verify_operational_v22_arrival_queue,
       build_operational_v22_arrival_path,
       operational_v22_arrival_path_matrix,
       operational_v22_arrival_path_sha256,
       verify_operational_v22_arrival_path,
       # Operational V2.2 M2-to-frozen-core path
       OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION,
       OPERATIONAL_V22_CORE_PATH_HOURS,
       OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR,
       OPERATIONAL_V22_CORE_PATH_SUPPORTED_MODEL_STEPS,
       operational_v22_hourly_drivers,
       operational_v22_core_path_forecast,
       # Operational V2.2-M3 causal error-state control
       OPERATIONAL_V22_ERROR_LAGS_H,
       OPERATIONAL_V22_ERROR_SUPPORTED_MODEL_STEPS,
       OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS,
       OperationalV22H1Innovation, operational_v22_h1_innovation,
       OperationalV22ErrorHistory, operational_v22_matured_h1_history,
       OperationalV22ErrorStateArtifact,
       fit_operational_v22_error_state,
       operational_v22_error_state_predict,
       operational_v22_error_state_sha256,
       write_operational_v22_error_state, read_operational_v22_error_state,
       # Operational V2.2-M3 full exogenous error model
       OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H,
       OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES,
       OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES,
       OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
       OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS,
       OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS,
       OPERATIONAL_V22_ERROR_EXOGENOUS_SUPPORTED_MODEL_STEPS,
       OPERATIONAL_V22_ERROR_EXOGENOUS_RIDGE_GRID,
       OPERATIONAL_V22_ERROR_EXOGENOUS_THRESHOLD_GRID,
       OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_SPECTRAL_RADIUS,
       OperationalV22ErrorExogenousIssue,
       OperationalV22ErrorExogenousFeatures,
       OperationalV22ErrorExogenousFitRow,
       OperationalV22ErrorExogenousArtifact,
       operational_v22_error_exogenous_features,
       fit_operational_v22_error_exogenous,
       operational_v22_error_exogenous_predict,
       operational_v22_error_exogenous_sha256,
       write_operational_v22_error_exogenous,
       read_operational_v22_error_exogenous,
       # Operational V2.2 checksum-bound shadow chain
       OPERATIONAL_V22_SHADOW_SCHEMA_VERSION,
       OPERATIONAL_V22_SHADOW_PRODUCT_VERSION,
       OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H,
       OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA,
       OperationalV22ShadowBindings,
       OperationalV22BaseCenterForecast,
       OperationalV22ShadowChainArtifact,
       operational_v22_regular_file_sha256,
       operational_v22_core_sha256,
       operational_v22_conformal_sha256,
       operational_v22_base_center_sha256,
       validate_operational_v22_shadow_chain,
       operational_v22_shadow_research_predict,
       operational_v22_shadow_predict,
       operational_v22_shadow_chain_sha256,
       write_operational_v22_shadow_chain,
       read_operational_v22_shadow_chain,
       # Conformal UQ
       ConformalCalibration, ConformalStratum, fit_conformal,
       conformal_stratum, conformal_halfwidth, conformal_interval,
       conformal_coverage, write_conformal_calibration, read_conformal_calibration,
       AdaptiveConformal, init_adaptive_conformal, adaptive_conformal_step!,
       run_adaptive_conformal,
       # Online assimilation
       AssimilationFilter, init_assimilation, assimilation_predict!,
       assimilation_update!, run_assimilation, current_dst, current_coeffs,
       dst_variance,
       # Alarm
       StormSeverity, QUIET, MODERATE, INTENSE, SUPERINTENSE,
       Alarm, AlarmConfig, default_alarm_config,
       check_alarm, maybe_fire_horizon_alarm!, classify_severity, alarm_print, alarm_log,
       # Real-time
       fetch_swpc_plasma, fetch_swpc_mag, fetch_swpc_dst, fetch_realtime_solar_wind,
       recover_shadow_state, feed_deadman_tripped, DEFAULT_FEED_DEADMAN_THRESHOLD,
       # Monitor
       run_monitor,
       # Operational V2.3 analog-driver-continuation features
       V23_FEATURE_NAMES, V23_FEATURE_COUNT, V23_HISTORY_LAGS_H,
       V23_SOUTH_RUN_CAP_H, V23_MAGNETIC_WEIGHT_FEATURES, V23_WEIGHTS,
       v23_weights, v23_feature_index, v23_feature_matrix,
       v23_feature_stats, v23_standardize,
       # Operational V2.3 analog retrieval and driver continuation
       V23_ANALOG_EXCLUSION_HOURS, V23_ANALOG_MAX_STEP, V23_KNN_BLOCK,
       V23_MEMBER_MIN_V_KMS, V23_MEMBER_MIN_N_CM3, V23_MEMBER_MAX_N_CM3,
       v23_knn, v23_analog_origin_ok, v23_member_driver, v23_analog_member,
       # Operational V2.3 boosted models (GDC, direct-GBM comparator, E2)
       V23_GBM_EVOTREES_VERSION, V23_GBM_DEFAULT_SEED,
       V23_DIRECT_DST_LAG_HOURS, V23_DIRECT_VBS_LAG_STEPS,
       V23_DIRECT_EXTRA_FEATURE_NAMES, V23_DIRECT_FEATURE_COUNT,
       v23_fit_gbm, v23_predict, v23_save, v23_load,
       v23_direct_extra_features, v23_direct_feature_names,
       v23_direct_features, v23_gdc_targets,
       # Operational V2.3 confirmatory inference primitives
       V23_BOOTSTRAP_BLOCK_HOURS, V23_BOOTSTRAP_REPLICATES, V23_BOOTSTRAP_SEED,
       V23_BOOTSTRAP_ALPHA, V23_BOOTSTRAP_EPOCH, V23_CELL_LABELS,
       V23_CELL_DEEP_DST_NT, V23_CELL_INTENSE_DST_NT, V23_CELL_INTENSE_RATE_NT_PER_H,
       v23_block_bootstrap, v23_holm, v23_regime_cells,
       # Operational V2.2 static-stack served product
       V22_SERVED_STACK_LABEL, V22_SERVED_STACK_SHA256, V22_SERVED_STACK_FILE,
       V22_SERVED_STACK_MANIFEST, V22_SERVED_IDENTITY, V22_SERVED_DRIVER_ASSUMPTION,
       v22_serving_coupling_active, v22_serving_stack_sha256, v22_serving_stack_manifest_rows,
       load_v22_serving_stack, v22_serving_center, v22_serving_depth_safe_center,
       V22_SERVING_DEPTH_SAFE_FILE,
       # Operational V2.3 shadow product
       V23_SERVING_MODEL_STEPS, V23_SERVING_STEP_SLOT, V23_SERVING_IDENTITY,
       V23_SERVING_SHADOW_IDENTITY, v23_serving_identity,
       V23_SERVING_DRIVER_ASSUMPTION, V23_SERVING_E_INNOVATION_LAGS,
       V23_SERVING_E_FEATURE_NAMES, V23_SERVING_STATS_ATOL, V23_SERVING_REQUIRED_FILES,
       V23ServingArtifacts, V23ServingELayer, V23ServingRidge,
       v23_serving_e_cap, v23_serving_transit_hours, v23_serving_file_sha256,
       v23_serving_verify_manifest, v23_serving_frame_lookup, load_v23_serving_artifacts,
       v23_serving_features, v23_serving_step_driver_from_frame, v23_serving_members,
       v23_serving_calibration_features, v23_serving_t1r_features,
       v23_serving_frozen_center, v23_serving_innovation_lags, v23_serving_center,
       v23_serving_manifest_hashed_names, v23_serving_innovations_from_step1_centers,
       v24_serving_depth_safe_center,
       # Operational V2.4e served product
       V24_SERVING_MODEL_STEPS, V24_SERVING_STEP_SLOT, V24_SERVED_IDENTITY,
       V24_SERVED_VARIANT, V24_SERVING_STATIC_EXPERT, V24_SERVING_GUARD_REFERENCE,
       V24_SERVING_GUARD_REFERENCE_NONE,
       V24_SERVED_GUARD_IDENTITY_TOKEN, V24_SERVED_GUARDED_IDENTITY,
       V24_SERVED_GUARDED_VARIANT, v24_served_identity, v24_served_conformal_variant,
       V24_SERVED_DRIVER_ASSUMPTION, V24_SERVING_EXPERTS, V24_SERVING_EXPERT_COUNT,
       V24_SERVING_SINDY_FAMILY, V24_SERVING_SINDY_FLOOR, V24_SERVING_REGIMES,
       V24_SERVING_POOLED, V24_SERVING_DEPTH_BINS, V24_SERVING_DEPTH_MODERATE_NT,
       V24_SERVING_DEPTH_DEEP_NT, V24_SERVING_GUARD_RATE_NT_PER_H,
       V24_SERVING_GUARD_DEPTH_NT, V24_SERVING_COVERAGE, V24_SERVING_DST_FLOOR_NT,
       V24_SERVING_DST_CEIL_NT, V24_SERVING_STATS_ATOL, V24_SERVING_REQUIRED_FILES,
       V24_SERVING_STACK_VARIANT, V24_SERVING_ANALOG_K, V24_SERVING_ANALOG_WEIGHT_SET,
       V24_SERVING_ANALOG_IDENTITY, V24_SERVING_DRIVER_LAGS_H, V24_SERVING_DST_LAG_MAX_H,
       V24ServingArtifacts, V24ServingCell, V24ServingStratum,
       v24_serving_direct_file, v24_serving_depth_bin, v24_serving_cell_chain,
       v24_serving_cell_grid, v24_serving_deepening, v24_serving_guard,
       v24_serving_verify_manifest, v24_serving_manifest_hashed_names,
       load_v24_serving_artifacts, v24_serving_analog_features, v24_serving_analog_members,
       v24_serving_t1r_center, v24_serving_direct_frame, v24_serving_direct_features,
       v24_serving_direct_center, v24_serving_climatology_center, v24_serving_cell,
       v24_serving_interval, v24_serving_center

end # module
