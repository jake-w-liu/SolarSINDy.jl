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
using Artifacts

include("utils.jl")
include("library.jl")
include("baselines.jl")
include("data.jl")
include("sindy.jl")
include("data_pipeline.jl")
include("data_cleaning.jl")
include("metrics.jl")
include("forecast.jl")
include("alarm.jl")
include("realtime.jl")
include("monitor.jl")

export # Utils
       numerical_derivative, smooth_moving_average, pressure_correct_dst,
       halfwave_rectify, imf_clock_angle, get_data_dir,
       # Library
       CandidateLibrary, build_solar_wind_library, build_minimal_library,
       evaluate_library, get_term_names,
       # SINDy
       stlsq, sindy_discover, ensemble_sindy, sindy_predict,
       simulate_sindy, sweep_lambda,
       # Baselines
       burton_model, burton_model_full, newell_coupling, obrien_mcpherron_model,
       simulate_burton, simulate_burton_full, simulate_obrien,
       # Synthetic data
       SolarWindData, StormEvent, generate_synthetic_storm,
       generate_multistorm_dataset, identify_storm_phases,
       prepare_sindy_data,
       # Real data pipeline
       download_omni2, extract_omni2_columns, parse_omni2, load_omni2_csv,
       # Data cleaning & storm catalog
       clean_omni_data!, StormCatalogEntry,
       build_storm_catalog, extract_storm_data, extract_all_storms,
       save_storm_catalog, load_storm_catalog,
       # Metrics
       rmse, correlation, skill_score, prediction_efficiency,
       metrics_summary,
       # Forecast
       ForecastState, ForecastResult, init_forecast,
       step_forecast!, forecast_ahead,
       # Alarm
       StormSeverity, QUIET, MODERATE, INTENSE, SUPERINTENSE,
       Alarm, AlarmConfig, default_alarm_config,
       check_alarm, classify_severity, alarm_print, alarm_log,
       # Real-time
       fetch_swpc_plasma, fetch_swpc_mag, fetch_realtime_solar_wind,
       # Monitor
       run_monitor

end # module
