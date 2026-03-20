#!/usr/bin/env julia
# storm_monitor.jl — Real-time geomagnetic storm monitor
#
# Usage:
#   julia --project=SolarSINDy.jl examples/storm_monitor.jl
#
# Fetches live solar wind data from NOAA SWPC, runs the SINDy-discovered
# 10-term equation with 500 ensemble coefficient sets, and displays a
# rolling forecast with configurable storm alarms.
#
# Press Ctrl-C to stop.

using SolarSINDy

const DATA_DIR = get_data_dir()

# Paths to discovered coefficients
const COEF_CSV = joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv")
const ENS_CSV = joinpath(DATA_DIR, "real_ensemble_inclusion.csv")

# Verify files exist
for f in [COEF_CSV, ENS_CSV]
    isfile(f) || error("Missing: $f\nRun the discovery pipeline first.")
end

println("=" ^ 60)
println("  SINDy Real-Time Storm Monitor")
println("  Equation: 10-term discovered ODE")
println("  Ensemble: 500 coefficient sets for UQ")
println("  Data: NOAA SWPC (DSCOVR L1)")
println("=" ^ 60)

# Configure alarms
alarm_config = AlarmConfig(
    Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
    true,                    # alarm on worst-case (5th percentile)
    alarm_print,             # print to terminal
    6,                       # 6-hour cooldown between alarms
)

# Start monitoring
run_monitor(;
    poll_interval_min = 5,
    forecast_horizon_hr = 6,
    alarm_config = alarm_config,
    coefficients_csv = COEF_CSV,
    ensemble_csv = ENS_CSV,
    log_file = "storm_monitor.log",
    display = true,
)
