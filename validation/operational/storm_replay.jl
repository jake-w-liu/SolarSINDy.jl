#!/usr/bin/env julia
# storm_replay.jl — Direction 3 (strengthen live evidence): score the Operational-v2 Dst forecaster against
# the physical baselines (persistence, Burton, Burton-full, O'Brien-McPherron) over real SUPERSTORM windows.
#
# Why: the locked-live record is quiet-only (no obs < -50 nT scored yet), so on quiet data persistence and
# O'Brien narrowly edge v2; v2's measured advantage shows up only in disturbed windows. This harness supplies
# the missing storm-time evidence by replaying the strongest storms of 2023-2026 through the SAME verified
# causal replay engine (replay_recent_table) used by the live verifier — no model re-implementation, no package
# change. The locked live record is never touched; this only reads the OMNI archive and the v2 calibration.
#
# The replay engine anchors at the data tail and walks back replay_hours, so we slice OMNI to each storm window
# (tail = storm recovery) and replay across it. Every prediction is causal: drivers come from the previous
# complete hour, the target Dst from the next hour. Scored 1 h ahead (the horizon where v2 carries its value).
#
# Run from the package root: julia --project=. validation/operational/storm_replay.jl

using Dates, DataFrames, Statistics, Printf, CSV
include(joinpath(@__DIR__, "paths.jl"))
include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))  # replay_recent_table, _omni_replay_inputs, LiveVerifyConfig, _load_calibration_for_model

const OMNI = OPERATIONAL_OMNI
const OUT_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "storm_replay_scored.csv")
const OUT_MD  = joinpath(OPERATIONAL_OUTPUT_DIR, "storm_replay_report.md")

# Strongest geomagnetic storms in the OMNI archive overlapping the CME catalog era (well-known dates).
const STORMS = [
    (name="May 2024 (Gannon, G5)", t0=DateTime(2024,5,9,12),  t1=DateTime(2024,5,13,0)),
    (name="Oct 2024 (G4)",         t0=DateTime(2024,10,9,0),  t1=DateTime(2024,10,12,0)),
    (name="Mar 2023 (G4)",         t0=DateTime(2023,3,23,0),  t1=DateTime(2023,3,25,12)),
]
const BASELINES = ["v2_pred_dst_nt", "v1_pred_dst_nt", "persistence_dst_nt",
                   "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt"]
const PRETTY = Dict("v2_pred_dst_nt"=>"V2", "v1_pred_dst_nt"=>"SINDy v1",
                    "persistence_dst_nt"=>"persistence", "burton_dst_nt"=>"Burton",
                    "burton_full_dst_nt"=>"Burton-full", "obrien_dst_nt"=>"O'Brien-McPherron")

const HORIZONS = [1, 2, 3, 6]
rmse(r) = sqrt(mean(abs2, r)); mae(r) = mean(abs.(r))

"Slice the aligned (plasma, mag, dst_times, dst_vals) tuple to a [t0,t1] datetime window."
function slice_window(plasma, mag, dst_times, dst_vals, t0, t1)
    m = (dst_times .>= t0) .& (dst_times .<= t1)
    return plasma[m, :], mag[m, :], dst_times[m], dst_vals[m]
end

"Replay one storm at horizons 1/2/3/6 h; rows are filtered so the ISSUE time lies in the storm window."
function replay_storm(storm, calibration)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr-1, yr)
    # pad 6 h back (driver lookback) and 7 h forward (so the late anchors' +6 h targets exist)
    ps, ms, ts, vs = slice_window(plasma, mag, dst_times, dst_vals, storm.t0 - Hour(6), storm.t1 + Hour(7))
    rh = Int(ceil(Dates.value((storm.t1 + Hour(7)) - (storm.t0 - Hour(6))) / 3_600_000))
    df = replay_recent_table(ps, ms, ts, vs; replay_hours=rh, horizons=HORIZONS, model=:v2, calibration=calibration)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]   # exact storm window by issue time
    df[!, :storm] = fill(storm.name, nrow(df))
    return df
end

"Per-model RMSE/MAE for one horizon subset of a scored frame."
function metrics_table(df, h)
    sub = df[df.model_step_hours .== h, :]
    obs = Float64.(sub.observation_dst_nt)
    rows = DataFrame(model=String[], rmse=Float64[], mae=Float64[], n=Int[])
    isempty(sub) && return rows
    for b in BASELINES
        b in names(sub) || continue
        r = obs .- Float64.(sub[!, b])
        push!(rows, (PRETTY[b], rmse(r), mae(r), length(r)))
    end
    return rows
end

function report_block(io, title, df)
    println("\n", title, "  (n=", nrow(df), ")")
    println(io, "## $title\n")
    for h in HORIZONS
        mt = metrics_table(df, h); isempty(mt) && continue
        sort!(mt, :rmse)
        cov = mean(Bool.(df[df.model_step_hours .== h, :v2_observed_in_90ci]))
        v2rank = findfirst(==("V2"), mt.model)
        @printf("  h=%d h:  best=%-16s  v2 rank %d/%d (RMSE %.2f)  v2 90%%CI cov=%.2f\n",
                h, mt.model[1], v2rank, nrow(mt), mt[mt.model.=="V2", :rmse][1], cov)
        @printf(io, "**Horizon %d h** (v2 90%% CI coverage %.2f):\n\n", h, cov)
        println(io, "| model | RMSE [nT] | MAE [nT] | n |"); println(io, "|---|---|---|---|")
        for r in eachrow(mt); @printf(io, "| %s | %.2f | %.2f | %d |\n", r.model, r.rmse, r.mae, r.n); end
        println(io)
    end
end

# Rapid main-phase subset: the storm hours a Dst forecaster exists to serve, where persistence is weakest.
# A row is main-phase when Dst is dropping faster than 15 nT/h over its forecast interval.
function mainphase_block(io, df)
    rate = (Float64.(df.observation_dst_nt) .- Float64.(df.latest_dst_nt)) ./ Float64.(df.model_step_hours)
    mp = df[rate .< -15.0, :]
    println("\nRAPID MAIN PHASE (Dst dropping >15 nT/h) — the regime a Dst forecaster serves  (n=", nrow(mp), ")")
    println(io, "## Rapid main phase (Dst dropping > 15 nT/h)\n")
    println(io, "The pooled windows are recovery/quiet-dominated, where persistence is nearly unbeatable. This ",
                "subset isolates the rapid main phase — the hours a Dst forecaster exists to serve.\n")
    for h in HORIZONS
        mt = metrics_table(mp, h); isempty(mt) && continue
        sort!(mt, :rmse)
        v2 = mt[mt.model .== "V2", :]; pers = mt[mt.model .== "persistence", :]
        (isempty(v2) || isempty(pers)) && continue
        ob = mt[mt.model .== "O'Brien-McPherron", :]
        @printf("  h=%d h (n=%d):  v2 %.2f  vs persistence %.2f  vs O'Brien %.2f   best=%s\n",
                h, v2.n[1], v2.rmse[1], pers.rmse[1], isempty(ob) ? NaN : ob.rmse[1], mt.model[1])
        @printf(io, "**Horizon %d h** (n=%d): ", h, v2.n[1])
        for r in eachrow(mt); @printf(io, "%s %.1f · ", r.model, r.rmse); end
        println(io, "  → v2 vs persistence: ", v2.rmse[1] < pers.rmse[1] ? "v2 better" : "persistence better")
    end
end

function main()
    println("Storm-time replay: V2 vs physical baselines (horizons ", HORIZONS, ")\n", "="^72)
    cfg = LiveVerifyConfig(model=:v2)            # uses the package's deployed operational calibration
    cal = _load_calibration_for_model(cfg)
    all_scored = DataFrame()
    open(OUT_MD, "w") do io
        println(io, "# Storm-time replay — V2 vs baselines (1/2/3/6 h ahead)\n")
        println(io, "Causal replay of the strongest 2023–2024 storms through the locked `replay_recent_table` ",
                    "engine, scored against persistence / Burton / Burton-full / O'Brien-McPherron. Built from the ",
                    "OMNI archive; the locked live record is untouched. Rows filtered to each storm's issue-time window.\n")
        for s in STORMS
            df = replay_storm(s, cal)
            all_scored = vcat(all_scored, df; cols=:union)
            @printf("\n%-26s observed min Dst=%.0f nT\n", s.name, minimum(Float64.(df.observation_dst_nt)))
            report_block(io, s.name, df)
        end
        report_block(io, "POOLED (all storm windows)", all_scored)
        mainphase_block(io, all_scored)
    end
    CSV.write(OUT_CSV, select(all_scored, Not(:issue_dt)))
    println("\nWrote $(OUT_CSV) and $(OUT_MD)")
    return all_scored
end

# ---- CRC self-test: metric oracles + correct storm windowing ----
function _selftest()
    pass = true
    chk(name, ok) = (pass &= ok; println(ok ? "  ✓ " : "  ✗ ", name))
    chk("rmse(const residual) = |const|", rmse(fill(3.0, 10)) ≈ 3.0)
    chk("mae oracle", mae([-2.0, 4.0]) ≈ 3.0)
    # windowing oracle: the Gannon replay must recover a superstorm-depth observed Dst (<= -300 nT),
    # proving the OMNI slice lands on the real storm and the engine scores it.
    cfg = LiveVerifyConfig(model=:v2); cal = _load_calibration_for_model(cfg)
    df = replay_storm(STORMS[1], cal)
    chk("Gannon window scores rows", nrow(df) > 20)
    chk("Gannon observed min Dst <= -300 nT (correct window)", minimum(Float64.(df.observation_dst_nt)) <= -300)
    chk("all baseline columns present", all(b -> b in names(df), BASELINES))
    chk("all four horizons present", sort(unique(df.model_step_hours)) == HORIZONS)
    chk("issue times confined to the storm window", all(STORMS[1].t0 .<= df.issue_dt .<= STORMS[1].t1))
    return pass
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("storm_replay.jl — CRC self-test:")
    _selftest() || error("self-test failed — do not trust the storm replay")
    println("ALL SELF-TESTS PASS\n")
    main()
end
