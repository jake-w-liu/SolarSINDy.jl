#!/usr/bin/env julia
# Long-run offline verification: does conformal calibration deliver ~90% interval
# coverage where the legacy interval-scale band undercovered (0.667 locked-live)?
#
# Fully offline: replays archived NASA OMNI data, fits the leakage-free v2 +
# conformal calibration, and measures HOLDOUT coverage (interval-scale vs
# conformal) overall and stratified by horizon and activity regime. No network.
#
# Usage:
#   julia --project=. validation/operational/run_conformal_coverage_test.jl

using SolarSINDy
using CSV, DataFrames, Statistics, Dates

include(joinpath(@__DIR__, "paths.jl"))
isdefined(@__MODULE__, :LiveVerifyConfig) || include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const OMNI = OPERATIONAL_OMNI
const OUTDIR = OPERATIONAL_OUTPUT_DIR
const TABLE = joinpath(OUTDIR, "omni_conformal_replay.csv")
const CALIB = joinpath(OUTDIR, "omni_conformal_v2_calibration.csv")
const REPORT = joinpath(OUTDIR, "conformal_coverage_report.md")
const REPLAY_HOURS = 2000
const HORIZONS = [1, 2, 3, 6]
const YEAR_START = 2023
const YEAR_END = 2025
const COVERAGE = 0.90

cover(lo, hi, y) = min(lo, hi) <= y <= max(lo, hi)

function main()
    t_start = now()
    println("[1/3] Building multi-horizon OMNI replay table ($YEAR_START-$YEAR_END, ",
            "$REPLAY_HOURS anchor hours, horizons=$(HORIZONS))...")
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, YEAR_START, YEAR_END)
    df = replay_recent_table(plasma, mag, dst_times, dst_vals;
                             replay_hours=REPLAY_HOURS, horizons=HORIZONS, model=:v1)
    CSV.write(TABLE, df)
    println("    replay rows: $(nrow(df)); wrote $TABLE")

    println("[2/3] Fitting leakage-free v2 + conformal calibration...")
    cfg = LiveVerifyConfig(; mode=:fit_v2_calibration, table_path=TABLE,
                           v2_calibration_path=CALIB,
                           v2_train_fraction=0.6, v2_validation_fraction=0.2,
                           v2_ridge_grid=Float64[0.0, 1.0, 10.0, 100.0, 1000.0],
                           v2_interval_coverage=COVERAGE, v2_coverage_floor=0.85)
    fit_v2_calibration!(cfg)

    println("[3/3] Measuring holdout coverage (interval-scale vs static conformal vs ACI)...")
    scored = CSV.read(replace(CALIB, r"\.csv$" => "_scored.csv"), DataFrame)
    cc = read_conformal_calibration(_conformal_path(CALIB))

    # Online ACI is run over the full chronologically-ordered v2-prediction stream
    # (per horizon) with MATURITY-DELAYED residual feedback: the interval served for
    # the forecast issued at t is scored against its eventual observation, but that
    # residual updates the ACI state only once its target time (t + h) has passed,
    # exactly as the deployed verified-log path advances ACI only over rows whose
    # observation is published (verify-then-update). Feeding each residual back at the
    # next hourly issue would grant the 2/3/6 h streams up to h-1 hours of look-ahead
    # relative to any deployable path. The served interval uses the identical deployed
    # gap-step call adaptive_conformal_step!(ac, center, NaN), which forms the band
    # from the current state without mutating it. This is the fair online comparison
    # against the static split-conformal holdout.
    sort!(scored, :issue_time_utc)
    hor_all = _scored_horizons(scored)
    issue_dt = _parse_dt.(scored.issue_time_utc)
    aci_covered = falses(nrow(scored))
    for h in unique(hor_all)
        idx = findall(==(h), hor_all)          # already time-ordered (scored is sorted)
        ac = init_adaptive_conformal(; target_coverage=COVERAGE, gamma=0.03, warmup=30)
        pend = Tuple{DateTime,Float64,Float64}[]   # (target_time, point, observed)
        qi = 1
        for i in idx
            t = issue_dt[i]
            while qi <= length(pend) && pend[qi][1] <= t
                _, pp, po = pend[qi]; qi += 1
                adaptive_conformal_step!(ac, pp, po)
            end
            point = Float64(scored.v2_pred_dst_nt[i])
            observed = Float64(scored.observation_dst_nt[i])
            s = adaptive_conformal_step!(ac, point, NaN)   # gap step: serve, no mutation
            aci_covered[i] = min(s.lo, s.hi) <= observed <= max(s.lo, s.hi)
            push!(pend, (t + Hour(Int(round(h))), point, observed))
        end
    end

    hold_mask = scored.v2_split .== "holdout"
    holdout = scored[hold_mask, :]
    hor = _scored_horizons(holdout)
    obs = Float64.(holdout.observation_dst_nt)
    pred = Float64.(holdout.v2_pred_dst_nt)
    latest = Float64.(holdout.latest_dst_nt)
    is_lo = Float64.(holdout.v2_pred_dst_ci05_nt)
    is_hi = Float64.(holdout.v2_pred_dst_ci95_nt)
    aci_hit = aci_covered[hold_mask]

    conf = [conformal_interval(cc, pred[i], hor[i], latest[i]) for i in eachindex(pred)]
    is_hit = [cover(is_lo[i], is_hi[i], obs[i]) for i in eachindex(obs)]
    cf_hit = [cover(conf[i][1], conf[i][2], obs[i]) for i in eachindex(obs)]
    cf_width = [conf[i][2] - conf[i][1] for i in eachindex(obs)]
    is_width = abs.(is_hi .- is_lo)

    lines = String[]
    push!(lines, "# Conformal Coverage Verification (offline OMNI replay)")
    push!(lines, "")
    push!(lines, "Source: archived NASA OMNI $(YEAR_START)-$(YEAR_END), causal multi-horizon replay.")
    push!(lines, "Nominal interval: $(round(Int, 100*COVERAGE))%. Holdout rows: $(nrow(holdout)) ",
                 "(scored once, never used for selection). ACI is run online over the full ",
                 "time-ordered stream and read out on the holdout rows.")
    push!(lines, "")
    push!(lines, "| stratum | n | interval-scale | static conformal | ACI | median conformal width nT |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: |")
    cov_rows = NamedTuple[]
    function row(label, mask)
        n = count(mask)
        n == 0 && return
        push!(lines, "| $label | $n | $(round(mean(is_hit[mask]); digits=3)) | " *
              "$(round(mean(cf_hit[mask]); digits=3)) | $(round(mean(aci_hit[mask]); digits=3)) | " *
              "$(round(median(cf_width[mask]); digits=1)) |")
        push!(cov_rows, (stratum=label, n=n,
                         interval_scale_cov=mean(is_hit[mask]),
                         static_conformal_cov=mean(cf_hit[mask]),
                         aci_cov=mean(aci_hit[mask]),
                         conformal_median_width_nt=median(cf_width[mask])))
    end
    row("overall", trues(length(obs)))
    for h in sort(unique(hor))
        row("horizon $(Int(h)) h", hor .== h)
    end
    row("quiet (Dst > -30)", latest .> -30.0)
    row("disturbed (Dst <= -30)", latest .<= -30.0)
    CSV.write(joinpath(OUTDIR, "coverage_by_stratum.csv"), DataFrame(cov_rows))

    push!(lines, "")
    push!(lines, "Interpretation: static conformal is well-behaved but can sit below ",
                 "$(round(Int,100*COVERAGE))% under chronological distribution shift; ACI adapts its ",
                 "miscoverage rate online and should track $(round(Int,100*COVERAGE))% more closely. ",
                 "The legacy interval-scale band is shown for comparison. Point forecasts are ",
                 "identical across all three; only the interval differs.")
    write(REPORT, join(lines, "\n") * "\n")

    println("\n=== Holdout coverage summary ===")
    for l in lines
        (startswith(l, "| ") && !startswith(l, "| ---")) && println("  ", l)
    end
    println("\nWrote $REPORT")
    println("Elapsed: $(round((now()-t_start).value/1000; digits=1)) s")
    return nothing
end

main()
