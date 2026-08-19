#!/usr/bin/env julia

# Evaluate complete-hour causal replay of the served V2.1 point stack on the
# locked 2020--2022 calibration holdout. The point calibration and conformal
# sidecar are already fixed before this script runs. Every interval is the
# deployed static conformal fallback shifted to the complete-hour served center
# after L1 look-ahead, magnetic relaxation, rate projection, and inertia
# safeguards. The replay does not reconstruct fractional subhourly live windows,
# and no adaptive residual from the holdout is used.

using CSV
using DataFrames
using Dates
using Printf
using SHA
using Statistics

isdefined(@__MODULE__, :_selftest_v2) || include(joinpath(@__DIR__, "v2_replay.jl"))

const V21_SERVED_HOLDOUT_SCORED = joinpath(
    OPERATIONAL_OUTPUT_DIR, "v2_1_served_holdout_scored.csv",
)
const V21_SERVED_HOLDOUT_SUMMARY = joinpath(
    OPERATIONAL_OUTPUT_DIR, "v2_1_served_holdout_summary.csv",
)
const V21_SERVED_HOLDOUT_AUDIT = joinpath(
    OPERATIONAL_OUTPUT_DIR, "v2_1_served_holdout_audit.csv",
)
const V21_SERVED_HOLDOUT_REPORT = joinpath(
    OPERATIONAL_OUTPUT_DIR, "v2_1_served_holdout_report.md",
)
const V21_SERVED_HOLDOUT_FLOOR = 0.85
const V21_SERVED_HOLDOUT_NOMINAL = 0.90
const V21_SERVED_HOLDOUT_STORM_DST = -50.0

function _v21_served_holdout_source()
    override = strip(get(ENV, "SOLARSINDY_V21_CALIBRATION_SCORED", ""))
    isempty(override) || return abspath(override)
    return joinpath(
        OPERATIONAL_OUTPUT_DIR, "v2_1_calibration",
        "operational_v2_calibration_scored.csv",
    )
end

function _v21_file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _v21_parse_datetime(value)
    value isa DateTime && return value
    return DateTime(String(value))
end

function _v21_holdout_contract(path::AbstractString)
    isfile(path) || error("deployed calibration split audit is missing: $path")
    split = CSV.read(path, DataFrame)
    required = [
        :split, :rows, :anchors, :minimum_issue_utc, :maximum_issue_utc,
        :minimum_target_utc, :maximum_target_utc, :point_calibration_sha256,
        :source_table_sha256, :conformal_holdout_coverage,
    ]
    all(in(propertynames(split)), required) || error(
        "deployed calibration split audit lacks required fields",
    )
    names = String.(split.split)
    all(name -> count(==(name), names) == 1, ("fit", "validation", "holdout")) ||
        error("deployed calibration split audit must contain one fit/validation/holdout row")
    validation = split[findfirst(==("validation"), names), :]
    holdout = split[findfirst(==("holdout"), names), :]
    validation_max_target = _v21_parse_datetime(validation.maximum_target_utc)
    minimum_issue = _v21_parse_datetime(holdout.minimum_issue_utc)
    maximum_issue = _v21_parse_datetime(holdout.maximum_issue_utc)
    minimum_target = _v21_parse_datetime(holdout.minimum_target_utc)
    maximum_target = _v21_parse_datetime(holdout.maximum_target_utc)
    validation_max_target < minimum_issue || error(
        "served holdout begins before every validation target has matured",
    )
    return (
        rows=Int(holdout.rows), anchors=Int(holdout.anchors),
        validation_max_target=validation_max_target,
        minimum_issue=minimum_issue, maximum_issue=maximum_issue,
        minimum_target=minimum_target, maximum_target=maximum_target,
        point_calibration_sha256=String(holdout.point_calibration_sha256),
        source_table_sha256=String(holdout.source_table_sha256),
        frozen_tail_conformal_coverage=Float64(holdout.conformal_holdout_coverage),
    )
end

function _v21_assert_holdout_rows(rows::DataFrame, contract;
                                  expected_version::AbstractString=OPERATIONAL_V2_1_MODEL_VERSION,
                                  expected_steps::AbstractVector{<:Integer}=
                                      OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS)
    required = [
        :issue_time_utc, :target_time_utc, :model_step_hours, :v2_split,
        :operational_core_version, :operational_candidate_count,
        :operational_active_count, :latest_dst_nt, :observation_dst_nt,
        :v2_pred_dst_nt, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa,
        :dst_delta_1h_nt,
    ]
    all(in(propertynames(rows)), required) || error(
        "served-holdout source lacks required columns",
    )
    nrow(rows) == contract.rows || error(
        "served-holdout row count $(nrow(rows)) does not match split audit $(contract.rows)",
    )
    all(String.(rows.v2_split) .== "holdout") || error(
        "served-holdout source contains a non-holdout row",
    )
    all(String.(rows.operational_core_version) .== expected_version) || error(
        "served-holdout source contains a non-V2.1 core identity",
    )
    all(Int.(rows.operational_candidate_count) .== 20) || error(
        "served-holdout source does not identify the 20-candidate core",
    )
    all(Int.(rows.operational_active_count) .== 11) || error(
        "served-holdout source does not identify the 11-active-term core",
    )
    issues = _v21_parse_datetime.(rows.issue_time_utc)
    targets = _v21_parse_datetime.(rows.target_time_utc)
    leads = Int.(rows.model_step_hours)
    supported = sort!(unique(Int.(expected_steps)))
    !isempty(supported) && all(>(0), supported) || error(
        "served-holdout expected model-step support must contain positive integers",
    )
    sort!(unique(leads)) == supported || error(
        "served-holdout model-step support does not match the issuable support",
    )
    minimum(issues) == contract.minimum_issue || error(
        "served-holdout minimum issue does not match the split audit",
    )
    maximum(issues) == contract.maximum_issue || error(
        "served-holdout maximum issue does not match the split audit",
    )
    minimum(targets) == contract.minimum_target || error(
        "served-holdout minimum target does not match the split audit",
    )
    maximum(targets) == contract.maximum_target || error(
        "served-holdout maximum target does not match the split audit",
    )
    all(issues .> contract.validation_max_target) || error(
        "served holdout includes an issue before validation targets matured",
    )
    all(targets .== issues .+ Hour.(leads)) || error(
        "served-holdout target is not issue plus model-step count",
    )
    return true
end

function _v21_served_center(row, core, calibration, lookup;
                            force_frozen::Bool=false)
    issue = _v21_parse_datetime(row.issue_time_utc)
    h = Int(row.model_step_hours)
    drivers = (
        V=Float64(row.V_kms), Bz=Float64(row.Bz_nt), By=Float64(row.By_nt),
        n=Float64(row.n_cm3), Pdyn=Float64(row.Pdyn_npa),
    )
    latest = Float64(row.latest_dst_nt)
    anchor = pressure_correct_dst([latest], [drivers.Pdyn])[1]
    rate = Float64(row.dst_delta_1h_nt)
    future = k -> get(lookup, issue + Hour(k - 1), nothing)
    return _v2_forecast(
        core.library, core.coefficients, anchor, drivers, future, latest,
        calibration, h, rate; force_frozen=force_frozen,
        calibration_features=row,
    )[2]
end

_v21_holdout_rmse(observed, predicted) =
    sqrt(mean(abs2, Float64.(observed) .- Float64.(predicted)))

function _v21_summary_row(rows::DataFrame, cohort::AbstractString,
                          lead_h::Int, regime::AbstractString;
                          pooled_gate_applies::Bool=false)
    nrow(rows) > 0 || error("served-holdout summary cohort $cohort is empty")
    served_hits = count(Bool.(rows.served_interval_hit))
    frozen_hits = count(Bool.(rows.frozen_tail_interval_hit))
    served_coverage = served_hits / nrow(rows)
    return (
        cohort=String(cohort), lead_h=lead_h, activity_regime=String(regime),
        n_rows=nrow(rows), served_hits=served_hits,
        served_coverage=served_coverage,
        served_rmse_nt=_v21_holdout_rmse(rows.observation_dst_nt, rows.served_v2_1_dst_nt),
        frozen_tail_hits=frozen_hits,
        frozen_tail_coverage=frozen_hits / nrow(rows),
        frozen_tail_rmse_nt=_v21_holdout_rmse(
            rows.observation_dst_nt, rows.frozen_tail_dst_nt,
        ),
        coverage_delta_served_minus_frozen=served_coverage - frozen_hits / nrow(rows),
        rmse_delta_served_minus_frozen_nt=_v21_holdout_rmse(
            rows.observation_dst_nt, rows.served_v2_1_dst_nt,
        ) - _v21_holdout_rmse(rows.observation_dst_nt, rows.frozen_tail_dst_nt),
        nominal_coverage=V21_SERVED_HOLDOUT_NOMINAL,
        promotion_coverage_floor=V21_SERVED_HOLDOUT_FLOOR,
        pooled_gate_applies=pooled_gate_applies,
        pooled_gate_pass=pooled_gate_applies && served_coverage >= V21_SERVED_HOLDOUT_FLOOR,
    )
end

function _v21_served_holdout_summary(rows::DataFrame)
    out = DataFrame()
    push!(out, _v21_summary_row(rows, "overall", 0, "all";
                                pooled_gate_applies=true); cols=:union)
    for h in sort!(unique(Int.(rows.model_step_hours)))
        sub = rows[Int.(rows.model_step_hours) .== h, :]
        push!(out, _v21_summary_row(sub, "lead_$h", h, "all"); cols=:union)
    end
    storm = Float64.(rows.observation_dst_nt) .< V21_SERVED_HOLDOUT_STORM_DST
    for (cohort, mask) in (("quiet", .!storm), ("storm", storm))
        sub = rows[mask, :]
        push!(out, _v21_summary_row(sub, cohort, 0, cohort); cols=:union)
        for h in sort!(unique(Int.(sub.model_step_hours)))
            by_lead = sub[Int.(sub.model_step_hours) .== h, :]
            isempty(by_lead) && continue
            push!(out, _v21_summary_row(
                by_lead, "$(cohort)_lead_$h", h, cohort,
            ); cols=:union)
        end
    end
    return out
end

function _selftest_v21_served_holdout()
    toy = DataFrame(
        model_step_hours=[1, 1, 6, 6],
        observation_dst_nt=[0.0, -60.0, 10.0, -70.0],
        served_v2_1_dst_nt=[1.0, -58.0, 12.0, -66.0],
        frozen_tail_dst_nt=[2.0, -57.0, 11.0, -65.0],
        served_interval_hit=[true, false, true, false],
        frozen_tail_interval_hit=[true, true, false, false],
    )
    summary = _v21_served_holdout_summary(toy)
    overall = summary[summary.cohort .== "overall", :][1, :]
    overall.n_rows == 4 || error("served-holdout summary row count oracle failed")
    overall.served_hits == 2 || error("served-holdout hit-count oracle failed")
    overall.served_coverage == 0.5 || error("served-holdout coverage oracle failed")
    isapprox(overall.served_rmse_nt, 2.5; atol=1e-12) || error(
        "served-holdout RMSE oracle failed",
    )
    storm = summary[summary.cohort .== "storm", :][1, :]
    storm.n_rows == 2 && storm.served_hits == 0 || error(
        "served-holdout storm-stratum oracle failed",
    )

    t0 = DateTime(2020, 1, 1)
    rows = DataFrame(
        issue_time_utc=[t0 + Hour(7)], target_time_utc=[t0 + Hour(8)],
        model_step_hours=[1], v2_split=["holdout"],
        operational_core_version=["v2.1"], operational_candidate_count=[20],
        operational_active_count=[11], latest_dst_nt=[0.0],
        observation_dst_nt=[0.0], v2_pred_dst_nt=[0.0], V_kms=[400.0],
        Bz_nt=[0.0], By_nt=[0.0], n_cm3=[5.0], Pdyn_npa=[1.0],
        dst_delta_1h_nt=[0.0],
    )
    contract = (
        rows=1, anchors=1, validation_max_target=t0 + Hour(6),
        minimum_issue=t0 + Hour(7), maximum_issue=t0 + Hour(7),
        minimum_target=t0 + Hour(8), maximum_target=t0 + Hour(8),
    )
    _v21_assert_holdout_rows(rows, contract; expected_steps=[1])
    broken = copy(rows)
    broken.issue_time_utc[1] = t0 + Hour(6)
    rejected = false
    try
        _v21_assert_holdout_rows(broken, contract; expected_steps=[1])
    catch
        rejected = true
    end
    rejected || error("served-holdout boundary mutation was not rejected")
    println("  ✓ served V2.1 holdout self-test: summary CRC and origin-boundary mutation")
    return true
end

function write_v21_served_holdout_report(path::AbstractString,
                                         summary::DataFrame,
                                         audit::DataFrame)
    overall = summary[summary.cohort .== "overall", :][1, :]
    open(path, "w") do io
        println(io, "# Complete-hour served-stack V2.1 chronological holdout\n")
        println(io, "The locked point calibration and conformal sidecar are evaluated after " *
                    "the complete-hour implementation of the V2.1 tail and safeguards. " *
                    "Fractional subhourly live windows are not reconstructed. Static " *
                    "conformal widths are shifted to the served center; no holdout " *
                    "residual updates the interval.\n")
        @printf(io, "Rows: %d. Served RMSE: %.3f nT. Pooled served coverage: %.3f (%d/%d), against the declared %.2f promotion floor and %.2f nominal target.\n\n",
                overall.n_rows, overall.served_rmse_nt, overall.served_coverage,
                overall.served_hits, overall.n_rows, overall.promotion_coverage_floor,
                overall.nominal_coverage)
        println(io, "| cohort | lead [h] | n | served RMSE [nT] | served coverage | frozen-tail coverage |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for r in eachrow(summary)
            @printf(io, "| %s | %d | %d | %.3f | %.3f | %.3f |\n",
                    r.cohort, r.lead_h, r.n_rows, r.served_rmse_nt,
                    r.served_coverage, r.frozen_tail_coverage)
        end
        println(io, "\nThe pooled floor is the deployment gate. Lead- and activity-stratified " *
                    "coverage is reported as a limitation diagnostic rather than silently " *
                    "promoted to a separate pass criterion.")
        println(io, "\nEvery issuable model step ($(audit.supported_model_steps[1])) is present " *
                    "in the chronological holdout; the minimum lead-specific coverage is " *
                    @sprintf("%.3f", audit.minimum_supported_step_coverage[1]) * ".")
        println(io, "\nAudit identity: `$(audit.model_version[1])`, " *
                    "$(audit.candidate_count[1]) candidates / $(audit.active_count[1]) active terms.")
    end
    return path
end

function main_v21_served_holdout(; self_test_only::Bool=false)
    _selftest_v21_served_holdout()
    self_test_only && return true

    artifacts = operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION)
    split_path = joinpath(dirname(artifacts.point_csv), "operational_v2_calibration_split.csv")
    contract = _v21_holdout_contract(split_path)
    point_sha = _v21_file_sha256(artifacts.point_csv)
    point_sha == contract.point_calibration_sha256 || error(
        "deployed point calibration hash differs from the split audit",
    )
    cfg = LiveVerifyConfig(model=:v2, v2_calibration_path=artifacts.point_csv)
    calibration = _load_calibration_for_model(cfg)
    conformal = _load_conformal_for_model(cfg; calibration=calibration)
    conformal === nothing && error("deployed V2.1 conformal sidecar is missing")
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)

    source = _v21_served_holdout_source()
    isfile(source) || error(
        "V2.1 calibration scored table is missing: $source. " *
        "Run validation/operational/v2_1_calibration.jl first.",
    )
    selected = unique(vcat(
        Symbol[
            :issue_time_utc, :target_time_utc, :model_step_hours, :v2_split,
            :operational_core_version, :operational_candidate_count,
            :operational_active_count, :latest_dst_nt, :observation_dst_nt,
            :v2_pred_dst_nt, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa,
            :dst_delta_1h_nt,
        ],
        calibration.feature_names,
    ))
    scored = CSV.read(source, DataFrame; select=selected)
    holdout = scored[String.(scored.v2_split) .== "holdout", :]
    sort!(holdout, [:issue_time_utc, :model_step_hours])
    _v21_assert_holdout_rows(
        holdout, contract; expected_steps=calibration.supported_model_steps,
    )

    lookup = _driver_lookup_range(
        year(contract.minimum_issue) - 1, year(contract.maximum_target),
    )
    out = DataFrame(
        issue_time_utc=DateTime[], target_time_utc=DateTime[],
        model_step_hours=Int[], latest_dst_nt=Float64[],
        observation_dst_nt=Float64[], dst_delta_1h_nt=Float64[],
        frozen_tail_dst_nt=Float64[], served_v2_1_dst_nt=Float64[],
        served_ci05_nt=Float64[], served_ci95_nt=Float64[],
        served_interval_hit=Bool[], frozen_tail_interval_hit=Bool[],
        activity_regime=String[],
    )
    max_frozen_delta = 0.0
    max_center_delta = 0.0
    for (i, row) in enumerate(eachrow(holdout))
        frozen = _v21_served_center(row, core, calibration, lookup; force_frozen=true)
        frozen_reference = Float64(row.v2_pred_dst_nt)
        max_frozen_delta = max(max_frozen_delta, abs(frozen - frozen_reference))
        served = _v21_served_center(row, core, calibration, lookup)
        h = Int(row.model_step_hours)
        latest = Float64(row.latest_dst_nt)
        lo, hi, source_label = _resolve_interval(
            conformal, served, h, latest, served, served,
        )
        source_label == "conformal" || error("served holdout did not use conformal intervals")
        frozen_lo, frozen_hi = conformal_interval(conformal, frozen_reference, h, latest)
        observed = Float64(row.observation_dst_nt)
        issue = _v21_parse_datetime(row.issue_time_utc)
        target = _v21_parse_datetime(row.target_time_utc)
        max_center_delta = max(max_center_delta, abs((lo + hi) / 2 - served))
        push!(out, (
            issue, target, h, latest, observed, Float64(row.dst_delta_1h_nt),
            frozen_reference, served, lo, hi, lo <= observed <= hi,
            frozen_lo <= observed <= frozen_hi,
            observed < V21_SERVED_HOLDOUT_STORM_DST ? "storm" : "quiet",
        ))
        i % 20_000 == 0 && println("  served holdout rows processed: $i")
    end
    max_frozen_delta <= 1e-9 || error(
        "served-holdout frozen-tail path diverges from the calibration replay: $max_frozen_delta",
    )
    max_center_delta <= 1e-10 || error(
        "served-holdout conformal interval is not centered on the served point: $max_center_delta",
    )
    all(isfinite, Matrix(select(out, Not([:issue_time_utc, :target_time_utc,
                                          :served_interval_hit,
                                          :frozen_tail_interval_hit,
                                          :activity_regime])))) ||
        error("served-holdout output contains a non-finite numeric value")

    summary = _v21_served_holdout_summary(out)
    overall = summary[summary.cohort .== "overall", :][1, :]
    overall.pooled_gate_pass || error(
        "complete-hour served-stack V2.1 holdout coverage $(overall.served_coverage) is below " *
        "the $(V21_SERVED_HOLDOUT_FLOOR) promotion floor",
    )
    lead_rows = summary[(summary.activity_regime .== "all") .&
                        (summary.lead_h .> 0), :]
    supported_steps = sort!(unique(Int.(calibration.supported_model_steps)))
    sort!(Int.(lead_rows.lead_h)) == supported_steps || error(
        "served-holdout summary omits an issuable model step",
    )
    minimum_step_coverage = minimum(Float64.(lead_rows.served_coverage))
    audit = DataFrame(
        model_version=[core.artifacts.version],
        candidate_count=[core.artifacts.candidate_count],
        active_count=[core.artifacts.active_count],
        holdout_rows=[nrow(out)], holdout_anchors=[contract.anchors],
        validation_max_target_utc=[contract.validation_max_target],
        holdout_min_issue_utc=[minimum(out.issue_time_utc)],
        holdout_max_issue_utc=[maximum(out.issue_time_utc)],
        holdout_max_target_utc=[maximum(out.target_time_utc)],
        strict_forecast_origin_separation=[
            contract.validation_max_target < minimum(out.issue_time_utc)
        ],
        interval_policy=["static_conformal_shifted_to_complete_hour_served_center"],
        holdout_residual_updates=[0],
        point_calibration_sha256=[point_sha],
        conformal_calibration_sha256=[_v21_file_sha256(artifacts.conformal_csv)],
        split_audit_sha256=[_v21_file_sha256(split_path)],
        calibration_scored_sha256=[_v21_file_sha256(source)],
        omni_sha256=[_v21_file_sha256(OMNI)],
        maximum_frozen_tail_continuity_error_nt=[max_frozen_delta],
        maximum_interval_center_error_nt=[max_center_delta],
        nominal_coverage=[V21_SERVED_HOLDOUT_NOMINAL],
        pooled_promotion_floor=[V21_SERVED_HOLDOUT_FLOOR],
        served_pooled_coverage=[Float64(overall.served_coverage)],
        served_pooled_gate_pass=[Bool(overall.pooled_gate_pass)],
        supported_model_steps=[join(supported_steps, ";")],
        supported_model_step_count=[length(supported_steps)],
        support_validation_complete=[true],
        minimum_supported_step_coverage=[minimum_step_coverage],
        frozen_tail_pooled_coverage=[Float64(overall.frozen_tail_coverage)],
        heldout_promotion_evidence=[true],
    )

    CSV.write(V21_SERVED_HOLDOUT_SCORED, out)
    CSV.write(V21_SERVED_HOLDOUT_SUMMARY, summary)
    CSV.write(V21_SERVED_HOLDOUT_AUDIT, audit)
    write_v21_served_holdout_report(V21_SERVED_HOLDOUT_REPORT, summary, audit)
    println("Complete-hour served-stack V2.1 holdout: n=$(nrow(out)), RMSE=$(overall.served_rmse_nt), " *
            "coverage=$(overall.served_coverage) ($(overall.served_hits)/$(overall.n_rows))")
    println("  wrote $V21_SERVED_HOLDOUT_SCORED")
    println("  wrote $V21_SERVED_HOLDOUT_SUMMARY")
    println("  wrote $V21_SERVED_HOLDOUT_AUDIT")
    println("  wrote $V21_SERVED_HOLDOUT_REPORT")
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    unknown = [arg for arg in ARGS if arg != "--self-test"]
    isempty(unknown) || error("unknown argument(s): $(join(unknown, ", "))")
    main_v21_served_holdout(; self_test_only=("--self-test" in ARGS))
end
