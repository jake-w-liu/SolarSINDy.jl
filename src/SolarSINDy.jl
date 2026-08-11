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
include("operational_v22_error_state.jl")
include("conformal.jl")
include("assimilation.jl")
include("alarm.jl")
include("realtime.jl")
include("monitor.jl")

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
       run_monitor

end # module
