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
include("lund_dst_2002.jl")
export LundDst2002
include("completed_minute_replay.jl")
export CompletedMinuteReplay
include("completed_source_dbdt/CompletedSourceDBDT.jl")
export CompletedSourceDBDT
include("data.jl")
include("sindy.jl")
include("storm_selection.jl")
include("data_pipeline.jl")
include("archive_dst_timing.jl")
include("data_cleaning.jl")
include("metrics.jl")
include("performance_statistics.jl")
include("forecast.jl")
include("operational_artifacts.jl")
include("static_stack_history.jl")
include("static_stack.jl")
include("static_stack_residual.jl")
include("static_stack_boost.jl")
include("static_stack_driver.jl")
include("static_stack_arrival.jl")
include("static_stack_core_path.jl")
include("static_stack_error_state.jl")
include("static_stack_error_exogenous.jl")
include("conformal.jl")
include("static_stack_shadow_chain.jl")
include("assimilation.jl")
include("alarm.jl")
include("realtime.jl")
include("monitor.jl")
include("analog_features.jl")
include("analog.jl")
include("analog_gbm.jl")
include("static_stack_serving.jl")
include("analog_stats.jl")
include("analog_serving.jl")
include("superlearner_serving.jl")

include("interval_mean_forecaster/IntervalMeanForecaster.jl")
export IntervalMeanForecaster
include("duration_conditioned_forcing.jl")
export DurationConditionedForcing
include("duration_current_state_stream.jl")
export DurationCurrentStateStream
include("duration_point_training.jl")
export DurationPointTraining
include("duration_point_artifacts.jl")
export DurationPointArtifacts

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
       archive_dst_forecast_timing, archive_dst_provenance,
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
       InertialSindyCalibration, default_inertial_sindy_calibration,
       inertial_sindy_feature_tuple,
       fit_inertial_sindy_calibration, operational_inertial_sindy_predict,
       score_inertial_sindy, write_inertial_sindy_calibration,
       read_inertial_sindy_calibration,
       # Versioned operational core
       OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, OPERATIONAL_SINDY_CORE_MODEL_VERSION,
       OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS,
       OperationalCoreArtifacts, OperationalCalibrationArtifacts,
       OperationalCore, canonical_operational_version, canonical_operational_model_version,
       operational_core_artifacts, operational_calibration_artifacts,
       validate_operational_core_artifacts, load_operational_core,
       init_operational_forecast, operational_core_forecast,
       # Operational static stack-M1 causal sparse-history core
       STATIC_STACK_HISTORY_TERMS,
       STATIC_STACK_HISTORY_SUPPORTED_ANCHOR_LAGS,
       STATIC_STACK_HISTORY_SCHEMA_VERSION,
       STATIC_STACK_HISTORY_DEFAULT_COUPLING_BOUND_MVM,
       StaticStackHistoryDriver, StaticStackHistoryState,
       StaticStackHistoryArtifact,
       static_stack_history_rho, static_stack_history_coupling,
       static_stack_history_memory, static_stack_history_features,
       static_stack_history_derivative, static_stack_history_step,
       init_static_stack_history_state, static_stack_history_rollout,
       fit_static_stack_history, static_stack_history_sha256,
       write_static_stack_history, read_static_stack_history,
       # Operational static stack constrained stack
       STATIC_STACK_COMPONENTS, DEFAULT_STATIC_STACK_COMPONENT_COLUMNS,
       StaticStackCell, StaticStack,
       static_stack_regime, fit_static_stack,
       static_stack_predict, score_static_stack,
       write_static_stack, read_static_stack,
       # Operational static stack secondary residual
       STATIC_STACK_RESIDUAL_FEATURES,
       STATIC_STACK_RESIDUAL_RIDGE_GRID,
       STATIC_STACK_RESIDUAL_TOP_K_GRID,
       StaticStackResidualCell, StaticStackResidualCore,
       fit_static_stack_residual, static_stack_residual_predict,
       score_static_stack_residual,
       write_static_stack_residual, read_static_stack_residual,
       # Operational static stack portable boosted residual
       STATIC_STACK_BOOST_FEATURES,
       STATIC_STACK_BOOST_SUPPORTED_MODEL_STEPS,
       STATIC_STACK_BOOST_SCHEMA_VERSION,
       StaticStackBoostArtifact,
       extract_static_stack_boost, fit_static_stack_boost,
       static_stack_boost_raw_predict, static_stack_boost_predict,
       score_static_stack_boost,
       write_static_stack_boost, read_static_stack_boost,
       # Operational static stack-M2 sparse driver continuation
       STATIC_STACK_DRIVER_STATES, STATIC_STACK_DRIVER_LAGS,
       STATIC_STACK_DRIVER_CADENCE_MINUTES,
       STATIC_STACK_DRIVER_ROLLOUT_STEPS,
       STATIC_STACK_DRIVER_STABILITY_TOLERANCE,
       STATIC_STACK_DRIVER_RIDGE_GRID,
       STATIC_STACK_DRIVER_THRESHOLD_GRID,
       StaticStackDriverArtifact,
       fit_static_stack_driver,
       static_stack_driver_coefficients,
       static_stack_driver_support,
       static_stack_driver_companion,
       static_stack_driver_spectral_radius,
       static_stack_driver_rollout,
       static_stack_driver_sha256,
       write_static_stack_driver, read_static_stack_driver,
       # Operational static stack-M2 receipt transport and arrival queue
       STATIC_STACK_ARRIVAL_PAIR_SCHEMA_VERSION,
       STATIC_STACK_ARRIVAL_SCHEMA_VERSION,
       STATIC_STACK_ARRIVAL_PATH_SCHEMA_VERSION,
       STATIC_STACK_ARRIVAL_PATH_GATE_STATUS,
       STATIC_STACK_ARRIVAL_CADENCE_MINUTES,
       STATIC_STACK_ARRIVAL_TRAILING_MINUTES,
       STATIC_STACK_ARRIVAL_HISTORY_ROWS,
       STATIC_STACK_ARRIVAL_PATH_STEPS,
       STATIC_STACK_ARRIVAL_MIN_DELAY_MINUTES,
       STATIC_STACK_ARRIVAL_MAX_DELAY_MINUTES,
       STATIC_STACK_ARRIVAL_MAX_FRESHNESS_MINUTES,
       STATIC_STACK_ARRIVAL_X_REF_GSE_KM,
       STATIC_STACK_ARRIVAL_INERTIAL_SINDY_COMPATIBILITY_DISTANCE_KM,
       StaticStackArrivalBin, StaticStackArrivalQueue,
       StaticStackArrivalPathStep, StaticStackArrivalPath,
       build_static_stack_arrival_queue,
       static_stack_arrival_history,
       static_stack_arrival_sha256,
       verify_static_stack_arrival_queue,
       build_static_stack_arrival_path,
       static_stack_arrival_path_matrix,
       static_stack_arrival_path_sha256,
       verify_static_stack_arrival_path,
       # Operational static stack M2-to-frozen-core path
       STATIC_STACK_CORE_PATH_SCHEMA_VERSION,
       STATIC_STACK_CORE_PATH_HOURS,
       STATIC_STACK_CORE_PATH_SUBSTEPS_PER_HOUR,
       STATIC_STACK_CORE_PATH_SUPPORTED_MODEL_STEPS,
       static_stack_hourly_drivers,
       static_stack_core_path_forecast,
       # Operational static stack-M3 causal error-state control
       STATIC_STACK_ERROR_LAGS_H,
       STATIC_STACK_ERROR_SUPPORTED_MODEL_STEPS,
       STATIC_STACK_ERROR_MAX_SPECTRAL_RADIUS,
       StaticStackH1Innovation, static_stack_h1_innovation,
       StaticStackErrorHistory, static_stack_matured_h1_history,
       StaticStackErrorStateArtifact,
       fit_static_stack_error_state,
       static_stack_error_state_predict,
       static_stack_error_state_sha256,
       write_static_stack_error_state, read_static_stack_error_state,
       # Operational static stack-M3 full exogenous error model
       STATIC_STACK_ERROR_EXOGENOUS_LAGS_H,
       STATIC_STACK_ERROR_EXOGENOUS_TEMPORAL_VARIABLES,
       STATIC_STACK_ERROR_EXOGENOUS_DRIVER_FEATURES,
       STATIC_STACK_ERROR_EXOGENOUS_FEATURES,
       STATIC_STACK_ERROR_EXOGENOUS_GROUPS,
       STATIC_STACK_ERROR_EXOGENOUS_FEATURE_GROUPS,
       STATIC_STACK_ERROR_EXOGENOUS_SUPPORTED_MODEL_STEPS,
       STATIC_STACK_ERROR_EXOGENOUS_RIDGE_GRID,
       STATIC_STACK_ERROR_EXOGENOUS_THRESHOLD_GRID,
       STATIC_STACK_ERROR_EXOGENOUS_MAX_SPECTRAL_RADIUS,
       StaticStackErrorExogenousIssue,
       StaticStackErrorExogenousFeatures,
       StaticStackErrorExogenousFitRow,
       StaticStackErrorExogenousArtifact,
       static_stack_error_exogenous_features,
       fit_static_stack_error_exogenous,
       static_stack_error_exogenous_predict,
       static_stack_error_exogenous_sha256,
       write_static_stack_error_exogenous,
       read_static_stack_error_exogenous,
       # Operational static stack checksum-bound shadow chain
       STATIC_STACK_SHADOW_SCHEMA_VERSION,
       STATIC_STACK_SHADOW_PRODUCT_VERSION,
       STATIC_STACK_SHADOW_SUPPORTED_HORIZONS_H,
       STATIC_STACK_SHADOW_DEFAULT_FEATURE_SCHEMA,
       StaticStackShadowBindings,
       StaticStackBaseCenterForecast,
       StaticStackShadowChainArtifact,
       static_stack_regular_file_sha256,
       static_stack_core_sha256,
       static_stack_conformal_sha256,
       static_stack_base_center_sha256,
       validate_static_stack_shadow_chain,
       static_stack_shadow_research_predict,
       static_stack_shadow_predict,
       static_stack_shadow_chain_sha256,
       write_static_stack_shadow_chain,
       read_static_stack_shadow_chain,
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
       # Operational analog ensemble-driver-continuation features
       ANALOG_FEATURE_NAMES, ANALOG_FEATURE_COUNT, ANALOG_HISTORY_LAGS_H,
       ANALOG_SOUTH_RUN_CAP_H, ANALOG_MAGNETIC_WEIGHT_FEATURES, ANALOG_WEIGHTS,
       analog_weights, analog_feature_index, analog_feature_matrix,
       analog_feature_stats, analog_standardize,
       # Operational analog ensemble retrieval and driver continuation
       ANALOG_EXCLUSION_HOURS, ANALOG_MAX_STEP, ANALOG_KNN_BLOCK,
       ANALOG_MEMBER_MIN_V_KMS, ANALOG_MEMBER_MIN_N_CM3, ANALOG_MEMBER_MAX_N_CM3,
       analog_knn, analog_origin_ok, analog_member_driver, analog_member,
       # Operational analog ensemble boosted models (GDC, direct-GBM comparator, E2)
       GBM_EVOTREES_VERSION, GBM_DEFAULT_SEED,
       DIRECT_GBM_DST_LAG_HOURS, DIRECT_GBM_VBS_LAG_STEPS,
       DIRECT_GBM_EXTRA_FEATURE_NAMES, DIRECT_GBM_FEATURE_COUNT,
       gbm_fit, gbm_predict, gbm_save, gbm_load,
       direct_gbm_extra_features, direct_gbm_feature_names,
       direct_gbm_features, gbm_driver_targets,
       # Operational analog ensemble confirmatory inference primitives
       ANALOG_BOOTSTRAP_BLOCK_HOURS, ANALOG_BOOTSTRAP_REPLICATES, ANALOG_BOOTSTRAP_SEED,
       ANALOG_BOOTSTRAP_ALPHA, ANALOG_BOOTSTRAP_EPOCH, ANALOG_CELL_LABELS,
       ANALOG_CELL_DEEP_DST_NT, ANALOG_CELL_INTENSE_DST_NT, ANALOG_CELL_INTENSE_RATE_NT_PER_H,
       analog_block_bootstrap, analog_holm, analog_regime_cells,
       # Operational static-stack served product
       STATIC_STACK_SERVED_LABEL, STATIC_STACK_SERVED_SHA256, STATIC_STACK_SERVED_FILE,
       STATIC_STACK_SERVED_MANIFEST, STATIC_STACK_SERVED_IDENTITY, STATIC_STACK_SERVED_DRIVER_ASSUMPTION,
       static_stack_serving_coupling_active, static_stack_sha256, static_stack_manifest_rows,
       load_static_stack, static_stack_serving_center, static_stack_serving_depth_safe_center,
       STATIC_STACK_SERVING_DEPTH_SAFE_FILE,
       # Analog-continuation ensemble machinery
       ANALOG_SERVING_MODEL_STEPS, ANALOG_SERVING_STEP_SLOT, ANALOG_SERVING_IDENTITY,
       analog_serving_identity, ANALOG_SERVING_E_INNOVATION_LAGS,
       ANALOG_SERVING_E_FEATURE_NAMES, ANALOG_SERVING_STATS_ATOL,
       AnalogServingArtifacts, AnalogServingELayer, AnalogServingRidge,
       analog_serving_e_cap, analog_serving_transit_hours, analog_serving_file_sha256,
       analog_serving_frame_lookup,
       analog_serving_features, analog_serving_step_driver_from_frame, analog_serving_members,
       analog_serving_calibration_features, analog_serving_t1r_features,
       analog_serving_frozen_center, analog_serving_innovation_lags, analog_serving_center,
       analog_serving_innovations_from_step1_centers,
       superlearner_serving_depth_safe_center,
       # Operational super-learner served product
       SUPERLEARNER_SERVING_MODEL_STEPS, SUPERLEARNER_SERVING_STEP_SLOT, SUPERLEARNER_SERVED_IDENTITY,
       SUPERLEARNER_SERVED_VARIANT, SUPERLEARNER_SERVING_STATIC_EXPERT, SUPERLEARNER_SERVING_GUARD_REFERENCE,
       SUPERLEARNER_SERVING_GUARD_REFERENCE_NONE,
       SUPERLEARNER_SERVED_GUARD_IDENTITY_TOKEN, SUPERLEARNER_SERVED_GUARDED_IDENTITY,
       SUPERLEARNER_SERVED_GUARDED_VARIANT, superlearner_served_identity, superlearner_served_conformal_variant,
       SUPERLEARNER_SERVED_DRIVER_ASSUMPTION, SUPERLEARNER_SERVING_EXPERTS, SUPERLEARNER_SERVING_EXPERT_COUNT,
       SUPERLEARNER_SERVING_SINDY_FAMILY, SUPERLEARNER_SERVING_SINDY_FLOOR, SUPERLEARNER_SERVING_REGIMES,
       SUPERLEARNER_SERVING_POOLED, SUPERLEARNER_SERVING_DEPTH_BINS, SUPERLEARNER_SERVING_DEPTH_MODERATE_NT,
       SUPERLEARNER_SERVING_DEPTH_DEEP_NT, SUPERLEARNER_SERVING_GUARD_RATE_NT_PER_H,
       SUPERLEARNER_SERVING_GUARD_DEPTH_NT, SUPERLEARNER_SERVING_COVERAGE, SUPERLEARNER_SERVING_DST_FLOOR_NT,
       SUPERLEARNER_SERVING_DST_CEIL_NT, SUPERLEARNER_SERVING_STATS_ATOL, SUPERLEARNER_SERVING_REQUIRED_FILES,
       SUPERLEARNER_SERVING_STACK_VARIANT, SUPERLEARNER_SERVING_ANALOG_K, SUPERLEARNER_SERVING_ANALOG_WEIGHT_SET,
       SUPERLEARNER_SERVING_ANALOG_IDENTITY, SUPERLEARNER_SERVING_DRIVER_LAGS_H, SUPERLEARNER_SERVING_DST_LAG_MAX_H,
       SuperlearnerServingArtifacts, SuperlearnerServingCell, SuperlearnerServingStratum,
       superlearner_serving_direct_file, superlearner_serving_depth_bin, superlearner_serving_cell_chain,
       superlearner_serving_cell_grid, superlearner_serving_deepening, superlearner_serving_guard,
       superlearner_serving_verify_manifest, superlearner_serving_manifest_hashed_names,
       load_superlearner_serving_artifacts, superlearner_serving_analog_features, superlearner_serving_analog_members,
       superlearner_serving_t1r_center, superlearner_serving_direct_frame, superlearner_serving_direct_features,
       superlearner_serving_direct_center, superlearner_serving_climatology_center, superlearner_serving_cell,
       superlearner_serving_interval, superlearner_serving_center

end # module
