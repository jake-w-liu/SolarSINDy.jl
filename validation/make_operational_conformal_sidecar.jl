# Generate the V2 conformal sidecar (cold-start/fallback interval; ACI remains the live primary).
# Source = the live verified log, which spans all issued horizons (1/2/3/6 h), so the fallback is horizon-
# stratified and matches the live v2 residual distribution. (The operational point-calibration set is h1-only,
# which would under-cover multi-hour leads as a fallback.)
#
# Paths default to the package runtime and deployment locations and are
# overridable so the sidecar can be regenerated against a scratch directory.
#
# Env:
#   SOLARSINDY_LIVE_LOG              verified live forecast log (default <package>/var/monitor/live_forecast_log.csv)
#   SOLARSINDY_V2_CONFORMAL_SIDECAR  output sidecar CSV (default <package>/deploy/operational_v2_calibration_conformal.csv)
using SolarSINDy, CSV, DataFrames, Dates, SHA

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const LOG = get(
    ENV, "SOLARSINDY_LIVE_LOG",
    joinpath(PACKAGE_ROOT, "var", "monitor", "live_forecast_log.csv"),
)
const SIDECAR = get(ENV, "SOLARSINDY_V2_CONFORMAL_SIDECAR",
                    joinpath(PACKAGE_ROOT, "deploy", "operational_v2_calibration_conformal.csv"))
const POINT_CALIBRATION = get(
    ENV, "SOLARSINDY_V2_CALIBRATION",
    joinpath(PACKAGE_ROOT, "deploy", "operational_v2_calibration.csv"),
)

df = CSV.read(LOG, DataFrame; missingstring=["", "NaN", "missing"])
_f(x) = (x === missing ? NaN : Float64(x))
_todt(x) = x isa DateTime ? x : DateTime(String(x))

pts = _f.(df.v2_pred_dst_nt)
obs = _f.(df.observation_dst_nt)
ld  = _f.(df.latest_dst_nt)
hz  = Float64.([ (_todt(df.target_time_utc[i]) - _todt(df.latest_dst_time_utc[i])) / Hour(1) for i in 1:nrow(df) ])

# Verified forecast rows only: real lead (target strictly after the anchor) + finite forecast/observation.
keep = isfinite.(pts) .& isfinite.(obs) .& isfinite.(hz) .& (hz .> 0)
println("log rows=", nrow(df), "  verified usable=", count(keep),
        "  horizons=", sort(unique(round.(Int, hz[keep]))))

isfile(POINT_CALIBRATION) || error("point calibration not found: $POINT_CALIBRATION")
# Bind the sidecar to the point calibration's model-step support exactly as the live
# fitter does (examples/live_forecast_verify.jl `_fit_deployed_conformal` /
# `_supported_horizon_edges`); `_load_conformal_for_model` rejects a sidecar whose
# supported_model_steps differs from the point calibration's.
point_cal = SolarSINDy.read_operational_v2_calibration(String(POINT_CALIBRATION))
isempty(point_cal.supported_model_steps) &&
    error("point calibration declares no supported_model_steps: $POINT_CALIBRATION")
steps = sort!(unique(Int.(point_cal.supported_model_steps)))
edges = vcat(0.0, [(steps[i] + steps[i + 1]) / 2 for i in 1:(length(steps) - 1)], Inf)
cal = SolarSINDy.fit_conformal(pts[keep], obs[keep], hz[keep], ld[keep];
                               horizon_edges=edges)
SolarSINDy.write_conformal_calibration(
    SIDECAR, cal;
    point_calibration_sha256=bytes2hex(SHA.sha256(read(POINT_CALIBRATION))),
    supported_model_steps=join(point_cal.supported_model_steps, ";"),
)
SolarSINDy.read_conformal_calibration(SIDECAR)   # re-asserts invariants on reload
println("SIDECAR_OK: ", SIDECAR)
run(`cat $SIDECAR`)
