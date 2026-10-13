# operational_research_scorecard.jl -- causal replay scorecard for Operational inertial-SINDy candidates.
#
# This is a research gate, not a live-product change. It compares current inertial-SINDy
# against tail variants and simple causal selectors. A candidate is
# promotable only if it improves accuracy versus current inertial-SINDy without
# increasing severe-storm under-warning risk.
#
# Run:
#   julia --project=. validation/operational/operational_research_scorecard.jl

using CSV, DataFrames, Dates, Printf, Statistics

isdefined(@__MODULE__, :_selftest_operational) || include(joinpath(@__DIR__, "operational_replay.jl"))

const OUT_RESEARCH_ROWS = joinpath(OPERATIONAL_OUTPUT_DIR, "operational_research_scorecard_rows.csv")
const OUT_RESEARCH_MD = joinpath(OPERATIONAL_OUTPUT_DIR, "operational_research_scorecard_report.md")

const R0_RESEARCH_VARIANTS = (
    (label="safety-deep-tail", col=:r0_3_75, r0=3.75,
     note="slower relaxation during active deepening; prioritizes severe-storm depth safety"),
    (label="current-inertial-sindy", col=:current_inertial_sindy, r0=R0_INERTIAL,
     note="current inertial-SINDy setting"),
    (label="rmse-lean-tail", col=:r0_15, r0=15.0,
     note="faster relaxation than current during deepening; tests pooled-RMSE pressure"),
    (label="plain-fixed-relax", col=:plain_b, r0=1e9,
     note="fixed B relaxation; known shallow-bias stressor"),
)

const SELECTOR_NOTES = Dict(
    :selector_rmse_lean => "causal selector: use current tail when Dst is already below -75 nT and still deepening; otherwise use rmse-lean tail",
    :selector_safety_lean => "causal selector: use safety-deep tail when Dst is below -50 nT and still deepening; otherwise use rmse-lean tail",
)

_fmt(x) = isfinite(x) ? @sprintf("%.2f", x) : ""
_fmt1(x) = isfinite(x) ? @sprintf("%.1f", x) : ""
_sgn(x) = isfinite(x) ? @sprintf("%+.2f", x) : ""

function _variant_frame(current_core, current_cal, historical_core, historical_cal,
                        luc, replay_inputs, spec)
    rows = _run_operational(
        current_core, current_cal, historical_core, historical_cal, luc, spec.r0;
        replay_inputs=replay_inputs,
    )
    return select(rows, :storm, :issue_utc, :lead, :inertial_sindy => spec.col)
end

function _candidate_rows()
    current_core = load_operational_core(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)
    historical_core = load_operational_core(OPERATIONAL_SINDY_CORE_MODEL_VERSION)
    current_cal = read_inertial_sindy_calibration(
        operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv,
    )
    historical_cal = read_inertial_sindy_calibration(
        operational_calibration_artifacts(OPERATIONAL_SINDY_CORE_MODEL_VERSION).point_csv,
    )
    archive = _load_replay_archive(
        minimum(year(s.t0) for s in STORMS) - 1,
        maximum(year(s.t1) for s in STORMS),
    )
    luc = Dict(year(s.t1) => archive.lookup for s in STORMS)

    base = _run_operational(
        current_core, current_cal, historical_core, historical_cal, luc, R0_INERTIAL;
        replay_inputs=archive.inputs,
    )
    rows = select(
        base, :storm, :issue_utc, :lead, :obs,
        :inertial_sindy_frozen => :frozen_tail_ablation,
        :sindy_core => :historical_sindy_core,
        :persistence, :rate, :inertial_sindy => :current_inertial_sindy,
    )
    key = [:storm, :issue_utc, :lead]
    for spec in R0_RESEARCH_VARIANTS
        spec.col == :current_inertial_sindy && continue
        before = nrow(rows)
        variant = _variant_frame(
            current_core, current_cal, historical_core, historical_cal,
            luc, archive.inputs, spec,
        )
        rows = innerjoin(rows, variant, on=key)
        nrow(rows) == before || error("variant $(spec.label) changed replay row support: $before -> $(nrow(rows))")
    end

    # Causal selectors use only issue-time quantities available in the replay row:
    # latest observed Dst (persistence) and recent Dst rate.
    rows[!, :selector_rmse_lean] = [
        (isfinite(r.rate) && r.rate < MAIN_RATE && r.persistence < -75.0) ? r.current_inertial_sindy : r.r0_15
        for r in eachrow(rows)
    ]
    rows[!, :selector_safety_lean] = [
        (isfinite(r.rate) && r.rate < MAIN_RATE && r.persistence < -50.0) ? r.r0_3_75 : r.r0_15
        for r in eachrow(rows)
    ]
    return rows
end

function _candidate_columns()
    return [:r0_3_75, :current_inertial_sindy, :r0_15, :plain_b, :selector_rmse_lean, :selector_safety_lean]
end

function _candidate_label(col::Symbol)
    col == :r0_3_75 && return "safety-deep-tail"
    col == :current_inertial_sindy && return "current-inertial-sindy"
    col == :r0_15 && return "rmse-lean-tail"
    col == :plain_b && return "plain-fixed-relax"
    col == :selector_rmse_lean && return "selector-rmse-lean"
    col == :selector_safety_lean && return "selector-safety-lean"
    return String(col)
end

function _candidate_note(col::Symbol)
    for spec in R0_RESEARCH_VARIANTS
        spec.col == col && return spec.note
    end
    return get(SELECTOR_NOTES, col, "")
end

function _candidate_cell(rows::DataFrame, col::Symbol)
    nrow(rows) == 0 && return nothing
    pred = Float64.(rows[!, col])
    ef = rows.obs .- rows.frozen_tail_ablation
    eh = rows.obs .- rows.historical_sindy_core
    ec = rows.obs .- pred
    ep = rows.obs .- rows.persistence
    ei = rows.obs .- rows.current_inertial_sindy
    rf, rh, rc, rp, ri = _rmse(ef), _rmse(eh), _rmse(ec), _rmse(ep), _rmse(ei)
    strong_pers = rp <= rf
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ef, ec; storms=rows.storm)
    Δi, loi, hii = col == :current_inertial_sindy ? (0.0, 0.0, 0.0) : paired_improvement(ei, ec; storms=rows.storm)
    return (n=nrow(rows), rmse=rc, rmse_frozen_tail=rf, rmse_sindy_core=rh,
            rmse_persistence=rp, rmse_current=ri,
            stronger=strong_pers ? "persistence" : "inertial-SINDy frozen-tail",
            improve_vs_stronger=Δ, ci_lo=lo, ci_hi=hi,
            improve_vs_current=Δi, current_ci_lo=loi, current_ci_hi=hii)
end

function _metric_table(rows::DataFrame)
    out = DataFrame(candidate=String[], lead=Int[], regime=String[], n=Int[],
                    rmse=Float64[], rmse_current=Float64[], rmse_frozen_tail=Float64[],
                    rmse_sindy_core=Float64[], rmse_persistence=Float64[], stronger=String[],
                    improve_vs_stronger=Float64[], ci_lo=Float64[], ci_hi=Float64[],
                    improve_vs_current=Float64[], current_ci_lo=Float64[], current_ci_hi=Float64[])
    for col in _candidate_columns(), h in LEADS
        sub_h = rows[rows.lead .== h, :]
        for (regime, sub) in (("pooled", sub_h),
                              ("main", sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
            c = _candidate_cell(sub, col)
            c === nothing && continue
            push!(out, (_candidate_label(col), h, regime, c.n, c.rmse, c.rmse_current,
                        c.rmse_frozen_tail, c.rmse_sindy_core, c.rmse_persistence, c.stronger,
                        c.improve_vs_stronger, c.ci_lo, c.ci_hi,
                        c.improve_vs_current, c.current_ci_lo, c.current_ci_hi))
        end
    end
    return out
end

function _safety_table(rows::DataFrame)
    out = DataFrame(candidate=String[], n_deep=Int[], deep_signed_error=Float64[],
                    underwarn_100_count=Int[], underwarn_200_count=Int[],
                    worst_shallow_gap_nt=Float64[])
    sub6 = rows[rows.lead .== 6, :]
    for col in _candidate_columns()
        pred = Float64.(sub6[!, col])
        deep = sub6[isfinite.(sub6.rate) .& (sub6.rate .< MAIN_RATE) .& (sub6.obs .< -100.0), :]
        deep_bias = nrow(deep) == 0 ? NaN : mean(deep.obs .- Float64.(deep[!, col]))
        severe100 = sub6[sub6.obs .<= -100.0, :]
        severe200 = sub6[sub6.obs .<= -200.0, :]
        under100 = nrow(severe100) == 0 ? 0 : count(Float64.(severe100[!, col]) .> -100.0)
        under200 = nrow(severe200) == 0 ? 0 : count(Float64.(severe200[!, col]) .> -200.0)
        shallow_gap = nrow(severe100) == 0 ? NaN : maximum(Float64.(severe100[!, col]) .- severe100.obs)
        push!(out, (_candidate_label(col), nrow(deep), deep_bias, under100, under200, shallow_gap))
    end
    return out
end

function _best_candidate(metrics::DataFrame, safety::DataFrame)
    pool6 = metrics[(metrics.lead .== 6) .& (metrics.regime .== "pooled"), :]
    pool6 = pool6[pool6.candidate .!= "current-inertial-sindy", :]
    nrow(pool6) == 0 && return (candidate="current-inertial-sindy", promotable=false,
                                reason="no non-current candidates were scored")
    sort!(pool6, :rmse)
    best = pool6[1, :]
    current_safety = safety[safety.candidate .== "current-inertial-sindy", :][1, :]
    best_safety = safety[safety.candidate .== best.candidate, :][1, :]
    main6 = metrics[(metrics.candidate .== best.candidate) .& (metrics.lead .== 6) .& (metrics.regime .== "main"), :][1, :]
    early = metrics[(metrics.candidate .== best.candidate) .& (metrics.regime .== "pooled") .& in.(metrics.lead, Ref([1, 2, 3])), :]
    no_early_drop = all(early.improve_vs_current .>= -1.0)
    no_safety_drop = best_safety.underwarn_100_count <= current_safety.underwarn_100_count &&
                     best_safety.underwarn_200_count <= current_safety.underwarn_200_count &&
                     best_safety.worst_shallow_gap_nt <= current_safety.worst_shallow_gap_nt + 5.0
    promotable = best.current_ci_lo > 0.0 && main6.current_ci_lo >= 0.0 && no_early_drop && no_safety_drop
    reason = promotable ? "passes accuracy and severe-underwarning gates" :
             "not promotable: requires 6h pooled CI>0 versus current, nonnegative 6h main CI, no >1 nT early-lead drop, and no worse severe-underwarning profile"
    return (candidate=best.candidate, promotable=promotable, reason=reason)
end

function _write_report(rows::DataFrame, metrics::DataFrame, safety::DataFrame)
    decision = _best_candidate(metrics, safety)
    open(OUT_RESEARCH_MD, "w") do io
        println(io, "# Operational inertial-SINDy research scorecard -- tail variants and causal selectors\n")
        println(io, "Scope: same causal G4/G5 storm replay used by the inertial-SINDy readiness gate. ",
                    "This artifact tests possible successors to current inertial-SINDy; it does not update the live product.\n")
        println(io, "## CRC plan\n")
        println(io, "1. Check: replay each candidate on identical issue times, leads, observations, and inertial-SINDy calibration.")
        println(io, "2. Reflect: compare point RMSE against the inertial-SINDy frozen-tail ablation, persistence, and current inertial-SINDy, then inspect deep-storm under-warning.")
        println(io, "3. Correct: promote only if a candidate clears the stated accuracy and safety gates; otherwise keep current inertial-SINDy.\n")
        println(io, "## Candidate notes\n")
        for col in _candidate_columns()
            println(io, "- ", _candidate_label(col), ": ", _candidate_note(col))
        end
        println(io, "\n## Current decision\n")
        println(io, "**Best 6 h pooled non-current candidate:** ", decision.candidate, ".")
        println(io, "**Promotion verdict:** ", decision.promotable ? "PROMOTABLE" : "NOT PROMOTABLE", " -- ", decision.reason, ".\n")

        println(io, "## Accuracy table\n")
        println(io, "`improve vs current` is paired |error current inertial-SINDy| - |error candidate|; positive means the candidate is better than current inertial-SINDy.\n")
        println(io, "| candidate | lead h | regime | n | RMSE | RMSE current inertial-SINDy | RMSE frozen-tail | RMSE historical SINDy core | stronger base | improve vs stronger (95% CI) | improve vs current (95% CI) |")
        println(io, "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
        for r in eachrow(metrics)
            @printf(io, "| %s | %d | %s | %d | %.2f | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] |\n",
                    r.candidate, r.lead, r.regime, r.n, r.rmse, r.rmse_current,
                    r.rmse_frozen_tail, r.rmse_sindy_core, r.stronger,
                    r.improve_vs_stronger, r.ci_lo, r.ci_hi,
                    r.improve_vs_current, r.current_ci_lo, r.current_ci_hi)
        end

        println(io, "\n## Severe-storm safety table\n")
        println(io, "Deep signed error is mean(obs - pred) on 6 h rows with active deepening and observed Dst below -100 nT. ",
                    "Large negative values indicate shallow under-prediction during severe storms.\n")
        println(io, "| candidate | deep n | deep signed err nT | underwarn <=-100 | underwarn <=-200 | worst shallow gap nT |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in eachrow(safety)
            @printf(io, "| %s | %d | %s | %d | %d | %s |\n",
                    r.candidate, r.n_deep, _fmt1(r.deep_signed_error),
                    r.underwarn_100_count, r.underwarn_200_count, _fmt1(r.worst_shallow_gap_nt))
        end

        println(io, "\n## Research reflection\n")
        println(io, "- The EKF variants use the current 20/11 SINDy core but remain experimental overlays on the frozen-tail ablation; they are not part of served inertial-SINDy.")
        println(io, "- Current inertial-SINDy is a safety-balanced operating point: it fixes the known plain-tail shallow-bias failure while retaining multi-hour skill.")
        println(io, "- If the selector candidates do not clear the promotion gate, the next accuracy work should target earlier and better driver information: sustained southward-Bz stress tests, L1 minute-layer shadow scoring, CME-arrival occurrence priors for uncertainty widening, scored issue-time-resolved same-unit Dst operational baselines, and prospective storm-live collection. Exact Kp/G-scale replay labels, external NOAA Kp archive scoring, Temerin-Li valid-time Dst archive context, and prospective public Dst snapshot collection are now implemented for the current evidence tier.")
    end
    return decision
end

function main()
    rows = _candidate_rows()
    metrics = _metric_table(rows)
    safety = _safety_table(rows)
    CSV.write(OUT_RESEARCH_ROWS, rows)
    decision = _write_report(rows, metrics, safety)
    println("Wrote ", OUT_RESEARCH_ROWS)
    println("Wrote ", OUT_RESEARCH_MD)
    println("Research verdict: ", decision.promotable ? "PROMOTABLE " : "NOT PROMOTABLE ", decision.candidate)
    return (; rows, metrics, safety, decision)
end

function _selftest_research_scorecard()
    rows = DataFrame(storm=["x", "x"], issue_utc=[DateTime(2026), DateTime(2026, 1, 1, 1)],
                     lead=[6, 6], obs=[-120.0, -80.0],
                     frozen_tail_ablation=[-100.0, -75.0], historical_sindy_core=[-98.0, -74.0],
                     persistence=[-90.0, -70.0], rate=[-20.0, 5.0],
                     current_inertial_sindy=[-119.0, -79.0], r0_3_75=[-125.0, -79.0],
                     r0_15=[-110.0, -78.0], plain_b=[-95.0, -77.0],
                     selector_rmse_lean=[-119.0, -78.0], selector_safety_lean=[-125.0, -78.0])
    m = _metric_table(rows)
    s = _safety_table(rows)
    @assert nrow(m) > 0 "metric table is empty"
    @assert s[s.candidate .== "plain-fixed-relax", :underwarn_100_count][1] == 1 "under-warning oracle failed"
    # Promotion-gate CIs must come from the storm-cluster bootstrap (V10-02): two storms whose candidate
    # advantage has opposite signs -> the cell CI equals the storm-labelled paired_improvement CI.
    cobs = fill(-100.0, 8); cbase = [6.0, -5.0, 7.0, -6.0, 6.0, -5.0, 7.0, -6.0]
    ccand = [0.5 .* cbase[1:4]; 1.6 .* cbase[5:8]]
    crows = DataFrame(storm=[fill("a", 4); fill("b", 4)], issue_utc=[DateTime(2026) + Hour(i) for i in 1:8],
                      lead=fill(1, 8), obs=cobs, frozen_tail_ablation=cobs .- cbase,
                      historical_sindy_core=cobs .- cbase, persistence=cobs .- 3 .* cbase, rate=fill(-20.0, 8),
                      current_inertial_sindy=cobs .- cbase, r0_3_75=cobs .- ccand, r0_15=cobs .- ccand,
                      plain_b=cobs .- ccand, selector_rmse_lean=cobs .- ccand, selector_safety_lean=cobs .- ccand)
    cc = _candidate_cell(crows, :r0_15)
    clu = paired_improvement(crows.obs .- crows.frozen_tail_ablation, crows.obs .- crows.r0_15; storms=crows.storm)
    @assert (cc.ci_lo, cc.ci_hi) == clu[2:3] "scorecard CI is not the storm-cluster bootstrap"
    @assert (cc.current_ci_lo, cc.current_ci_hi) == clu[2:3] "scorecard vs-current CI is not the storm-cluster bootstrap"
    println("  ✓ research scorecard self-test: metrics and severe-underwarning oracle")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_research_scorecard()
    main()
end
