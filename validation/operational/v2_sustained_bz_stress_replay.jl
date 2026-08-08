# v2_sustained_bz_stress_replay.jl -- sustained-southward-Bz stress diagnostics for V2.1.
#
# This is a diagnostic research replay. Realized future-Bz labels are used only to
# stratify failure modes after the fact; they are not live selection inputs. Causal
# labels use issue-time and trailing drivers only.
#
# Run:
#   julia --project=. validation/operational/v2_sustained_bz_stress_replay.jl

using CSV, DataFrames, Dates, Printf, Statistics

include(joinpath(@__DIR__, "v2_research_scorecard.jl"))

const OUT_STRESS_ROWS = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_sustained_bz_stress_rows.csv")
const OUT_STRESS_METRICS = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_sustained_bz_stress_metrics.csv")
const OUT_STRESS_MD = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_sustained_bz_stress_report.md")

const BZ_SOUTH_NT = -5.0
const BZ_STRONG_NT = -8.0

function _merged_driver_lookup(rows::DataFrame)
    years = Set{Int}()
    for r in eachrow(rows)
        push!(years, year(r.issue_utc - Hour(3)))
        push!(years, year(r.issue_utc))
        push!(years, year(r.issue_utc + Hour(r.lead)))
    end
    drivers = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    for yr in sort(collect(years))
        merge!(drivers, _driver_lookup(yr))
    end
    return drivers
end

_drv_bz(drivers, t::DateTime) = (d = get(drivers, t, nothing); d === nothing ? NaN : d.Bz)

function _window_bz_stats(drivers, issue::DateTime, lead::Int)
    issue_bz = _drv_bz(drivers, issue)
    past = [_drv_bz(drivers, issue - Hour(k)) for k in 0:2]
    fut = [_drv_bz(drivers, issue + Hour(k)) for k in 1:lead]
    fpast = filter(isfinite, past)
    ffut = filter(isfinite, fut)
    past_south = count(<=(BZ_SOUTH_NT), fpast)
    future_south = count(<=(BZ_SOUTH_NT), ffut)
    future_required = max(1, min(3, lead))
    future_sustained = length(ffut) >= future_required &&
                       future_south >= future_required &&
                       mean(ffut[1:future_required]) <= BZ_SOUTH_NT
    strong_sustained = future_sustained && minimum(ffut[1:future_required]) <= BZ_STRONG_NT
    return (
        issue_bz_nt=issue_bz,
        issue_bs_nt=isfinite(issue_bz) ? max(-issue_bz, 0.0) : NaN,
        past3_mean_bz_nt=isempty(fpast) ? NaN : mean(fpast),
        past3_southward_hours=past_south,
        future_mean_bz_nt=isempty(ffut) ? NaN : mean(ffut),
        future_min_bz_nt=isempty(ffut) ? NaN : minimum(ffut),
        future_southward_hours=future_south,
        future_required_hours=future_required,
        causal_issue_southward=isfinite(issue_bz) && issue_bz <= BZ_SOUTH_NT,
        causal_past_sustained_southward=length(fpast) == 3 && past_south >= 2 && mean(fpast) <= BZ_SOUTH_NT,
        realized_future_sustained_southward=future_sustained,
        realized_future_strong_southward=strong_sustained,
    )
end

function _annotate_sustained_bz(rows::DataFrame; drivers=nothing)
    out = copy(rows)
    drivers === nothing && (drivers = _merged_driver_lookup(out))
    stats = [_window_bz_stats(drivers, r.issue_utc, r.lead) for r in eachrow(out)]
    for name in propertynames(first(stats))
        out[!, name] = [getproperty(s, name) for s in stats]
    end
    out[!, :active_deepening] = isfinite.(out.rate) .& (out.rate .< MAIN_RATE)
    out[!, :deep_observed] = out.obs .<= -100.0
    out[!, :severe_stress] = out.realized_future_sustained_southward .& out.active_deepening .& out.deep_observed
    return out
end

function _regime_masks(rows::DataFrame)
    return [
        (name="all", description="all scored replay rows", mask=trues(nrow(rows))),
        (name="causal_issue_bz_south", description="issue-time Bz <= -5 nT", mask=rows.causal_issue_southward),
        (name="causal_past3_south", description="at least 2 of the trailing 3 issue-time hours southward with mean Bz <= -5 nT", mask=rows.causal_past_sustained_southward),
        (name="realized_future_south", description="first min(lead,3) target hours are sustained southward after issue", mask=rows.realized_future_sustained_southward),
        (name="realized_future_strong", description="realized_future_south with at least one hour Bz <= -8 nT", mask=rows.realized_future_strong_southward),
        (name="active_deepening", description="recent Dst rate below main-phase threshold", mask=rows.active_deepening),
        (name="severe_stress", description="realized_future_south + active deepening + observed Dst <= -100 nT", mask=rows.severe_stress),
    ]
end

function _candidate_cols_for_stress()
    return [:current_v2_1, :r0_3_75, :r0_15, :plain_b, :selector_rmse_lean, :selector_safety_lean]
end

function _stress_cell(rows::DataFrame, col::Symbol)
    nrow(rows) == 0 && return nothing
    pred = Float64.(rows[!, col])
    ef = rows.obs .- rows.frozen_tail_ablation
    eh = rows.obs .- rows.historical_v2_0
    ec = rows.obs .- pred
    ep = rows.obs .- rows.persistence
    ei = rows.obs .- rows.current_v2_1
    rf, rh, rc, rp = _rmse(ef), _rmse(eh), _rmse(ec), _rmse(ep)
    strong_pers = rp <= rf
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ef, ec)
    Δi, loi, hii = col == :current_v2_1 ? (0.0, 0.0, 0.0) : paired_improvement(ei, ec)
    severe = rows[rows.obs .<= -100.0, :]
    deep_bias = nrow(severe) == 0 ? NaN : mean(severe.obs .- Float64.(severe[!, col]))
    under100 = nrow(severe) == 0 ? 0 : count(Float64.(severe[!, col]) .> -100.0)
    return (n=nrow(rows), rmse=rc, rmse_frozen_tail=rf, rmse_v2_0=rh,
            rmse_persistence=rp,
            stronger=strong_pers ? "persistence" : "V2.1 frozen-tail",
            improve_vs_stronger=Δ, ci_lo=lo, ci_hi=hi,
            improve_vs_current=Δi, current_ci_lo=loi, current_ci_hi=hii,
            severe_n=nrow(severe), severe_signed_error=deep_bias, underwarn_100=under100)
end

function _metric_table(rows::DataFrame)
    out = DataFrame(candidate=String[], lead=Int[], regime=String[], n=Int[],
                    rmse=Float64[], rmse_frozen_tail=Float64[], rmse_v2_0=Float64[],
                    rmse_persistence=Float64[],
                    stronger=String[], improve_vs_stronger=Float64[], ci_lo=Float64[], ci_hi=Float64[],
                    improve_vs_current=Float64[], current_ci_lo=Float64[], current_ci_hi=Float64[],
                    severe_n=Int[], severe_signed_error=Float64[], underwarn_100=Int[])
    for col in _candidate_cols_for_stress(), h in LEADS
        sub_h = rows[rows.lead .== h, :]
        for reg in _regime_masks(sub_h)
            sub = sub_h[reg.mask, :]
            c = _stress_cell(sub, col)
            c === nothing && continue
            push!(out, (_candidate_label(col), h, reg.name, c.n, c.rmse,
                        c.rmse_frozen_tail, c.rmse_v2_0, c.rmse_persistence,
                        c.stronger, c.improve_vs_stronger, c.ci_lo, c.ci_hi,
                        c.improve_vs_current, c.current_ci_lo, c.current_ci_hi,
                        c.severe_n, c.severe_signed_error, c.underwarn_100))
        end
    end
    return out
end

function _regime_count_table(rows::DataFrame)
    out = DataFrame(lead=Int[], regime=String[], description=String[], n=Int[], fraction=Float64[])
    for h in LEADS
        sub_h = rows[rows.lead .== h, :]
        denom = max(nrow(sub_h), 1)
        for reg in _regime_masks(sub_h)
            push!(out, (h, reg.name, reg.description, count(reg.mask), count(reg.mask) / denom))
        end
    end
    return out
end

function _decision(metrics::DataFrame)
    stress6 = metrics[(metrics.lead .== 6) .& (metrics.regime .== "realized_future_south") .&
                      (metrics.candidate .!= "current-v2.1"), :]
    nrow(stress6) == 0 && return (candidate="current-v2.1", verdict="INCONCLUSIVE",
                                  reason="no non-current candidate had 6 h realized-future-south rows")
    sort!(stress6, :rmse)
    best = stress6[1, :]
    current = metrics[(metrics.lead .== 6) .& (metrics.regime .== "realized_future_south") .&
                      (metrics.candidate .== "current-v2.1"), :][1, :]
    all6 = metrics[(metrics.lead .== 6) .& (metrics.regime .== "all") .& (metrics.candidate .== best.candidate), :][1, :]
    safety_ok = best.underwarn_100 <= current.underwarn_100 &&
                (!isfinite(best.severe_signed_error) || !isfinite(current.severe_signed_error) ||
                 best.severe_signed_error >= current.severe_signed_error - 5.0)
    promotable = best.current_ci_lo > 0.0 && all6.current_ci_lo >= -1.0 && safety_ok
    verdict = promotable ? "TARGETED PROMOTION CANDIDATE" : "DIAGNOSTIC ONLY"
    reason = promotable ?
        "best stress-regime candidate improves 6 h sustained-southward rows without a pooled or under-warning penalty" :
        "do not alter live behavior: the best stress-regime candidate lacks a clean same-row/safety gate versus current V2.1"
    return (candidate=best.candidate, verdict=verdict, reason=reason)
end

function _write_report(rows::DataFrame, metrics::DataFrame, counts::DataFrame)
    dec = _decision(metrics)
    open(OUT_STRESS_MD, "w") do io
        println(io, "# Sustained southward-Bz stress replay for V2.1\n")
        println(io, "Purpose: isolate whether current V2.1 fails in sustained southward-Bz regimes. ",
                    "Realized future-Bz labels are diagnostic only; live selection must use causal labels.\n")
        println(io, "## CRC plan\n")
        println(io, "1. Check: annotate the existing same-row replay with issue-time Bz, trailing southward Bz, and realized future sustained-southward labels.")
        println(io, "2. Reflect: score current V2.1 and candidate tails inside each stress regime, including severe under-warning counts.")
        println(io, "3. Correct: keep live behavior unchanged unless a candidate clears same-row accuracy and safety gates.\n")
        println(io, "## Regime counts\n")
        println(io, "| lead h | regime | n | fraction | description |")
        println(io, "| ---: | --- | ---: | ---: | --- |")
        for r in eachrow(counts)
            @printf(io, "| %d | %s | %d | %.3f | %s |\n", r.lead, r.regime, r.n, r.fraction, r.description)
        end

        println(io, "\n## Current V2.1 stress performance\n")
        println(io, "| lead h | regime | n | RMSE current | RMSE frozen-tail | RMSE historical V2.0 | RMSE persistence | improve vs stronger (95% CI) | severe n | severe signed err | underwarn <=-100 |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        cur = metrics[metrics.candidate .== "current-v2.1", :]
        for r in eachrow(cur)
            @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %d | %s | %d |\n",
                    r.lead, r.regime, r.n, r.rmse, r.rmse_frozen_tail,
                    r.rmse_v2_0, r.rmse_persistence,
                    r.improve_vs_stronger, r.ci_lo, r.ci_hi, r.severe_n,
                    _fmt1(r.severe_signed_error), r.underwarn_100)
        end

        println(io, "\n## Candidate pressure in realized sustained-southward rows\n")
        println(io, "`improve vs current` is paired |error current V2.1| - |error candidate| within the same regime.\n")
        println(io, "| candidate | lead h | n | RMSE | improve vs current (95% CI) | severe signed err | underwarn <=-100 |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        stress = metrics[(metrics.regime .== "realized_future_south") .& (metrics.candidate .!= "current-v2.1"), :]
        for r in eachrow(stress)
            @printf(io, "| %s | %d | %d | %.2f | %+.2f [%+.2f, %+.2f] | %s | %d |\n",
                    r.candidate, r.lead, r.n, r.rmse, r.improve_vs_current,
                    r.current_ci_lo, r.current_ci_hi, _fmt1(r.severe_signed_error), r.underwarn_100)
        end

        println(io, "\n## Decision\n")
        println(io, "**Best 6 h sustained-southward stress candidate:** ", dec.candidate, ".")
        println(io, "**Verdict:** ", dec.verdict, " -- ", dec.reason, ".\n")
        println(io, "## Research reflection\n")
        println(io, "- If current V2.1 remains competitive in realized sustained-southward rows, the next accuracy bottleneck is likely driver timing/resolution, not tail relaxation tuning.")
        println(io, "- If a candidate improves sustained-southward rows but worsens severe under-warning or all-row 6 h behavior, it stays diagnostic only.")
        println(io, "- The next actionable work after this diagnostic is either larger G3+ event coverage or minute-layer live shadow scoring, depending on which stress rows dominate the residuals.")
    end
    return dec
end

function main()
    rows = _candidate_rows()
    annotated = _annotate_sustained_bz(rows)
    metrics = _metric_table(annotated)
    counts = _regime_count_table(annotated)
    CSV.write(OUT_STRESS_ROWS, annotated)
    CSV.write(OUT_STRESS_METRICS, metrics)
    dec = _write_report(annotated, metrics, counts)
    println("Wrote ", OUT_STRESS_ROWS)
    println("Wrote ", OUT_STRESS_METRICS)
    println("Wrote ", OUT_STRESS_MD)
    println("Stress verdict: ", dec.verdict, " / ", dec.candidate)
    return (; rows=annotated, metrics, counts, decision=dec)
end

function _selftest_sustained_bz()
    t0 = DateTime(2026, 1, 1, 0)
    drivers = Dict(
        t0 - Hour(2) => (V=400.0, Bz=-6.0, By=0.0, n=5.0, Pdyn=2.0),
        t0 - Hour(1) => (V=400.0, Bz=-7.0, By=0.0, n=5.0, Pdyn=2.0),
        t0           => (V=400.0, Bz=-8.0, By=0.0, n=5.0, Pdyn=2.0),
        t0 + Hour(1) => (V=400.0, Bz=-9.0, By=0.0, n=5.0, Pdyn=2.0),
        t0 + Hour(2) => (V=400.0, Bz=-5.5, By=0.0, n=5.0, Pdyn=2.0),
        t0 + Hour(3) => (V=400.0, Bz=-6.0, By=0.0, n=5.0, Pdyn=2.0),
    )
    rows = DataFrame(storm=["x"], issue_utc=[t0], lead=[3], obs=[-120.0],
                     frozen_tail_ablation=[-100.0], historical_v2_0=[-98.0],
                     persistence=[-90.0], rate=[-20.0], current_v2_1=[-118.0],
                     r0_3_75=[-124.0], r0_15=[-112.0], plain_b=[-98.0],
                     selector_rmse_lean=[-118.0], selector_safety_lean=[-124.0])
    ann = _annotate_sustained_bz(rows; drivers=drivers)
    @assert ann.causal_issue_southward[1] "issue southward label failed"
    @assert ann.causal_past_sustained_southward[1] "past sustained label failed"
    @assert ann.realized_future_sustained_southward[1] "future sustained label failed"
    @assert ann.realized_future_strong_southward[1] "future strong label failed"
    m = _metric_table(ann)
    @assert nrow(m) > 0 "stress metric table is empty"
    @assert m[(m.candidate .== "plain-fixed-relax") .& (m.regime .== "severe_stress"), :underwarn_100][1] == 1 "under-warning oracle failed"
    println("  ✓ sustained-Bz stress self-test: labels, metrics, under-warning oracle")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_sustained_bz()
    main()
end
