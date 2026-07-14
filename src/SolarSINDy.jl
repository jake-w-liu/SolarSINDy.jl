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
