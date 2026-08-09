#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using HTTP
using JSON3
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "paths.jl"))

const REPO_ROOT = OPERATIONAL_WORKSPACE_ROOT
const LIVE_DIR = OPERATIONAL_EVIDENCE_DIR
const LIVE_LOG_PATH = _operational_path(
    "SOLARSINDY_LIVE_LOG",
    joinpath(LIVE_DIR, "live_forecast_log.csv"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "live_forecast_log.csv"),
    joinpath(LIVE_DIR, "live_forecast_log.csv"),
)
const EXTERNAL_DST_LOG_PATH = _operational_path(
    "SOLARSINDY_EXTERNAL_DST_LOG",
    joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "external_dst_forecast_log.csv"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "external_dst_forecast_log.csv"),
    joinpath(LIVE_DIR, "external_dst_forecast_log.csv"),
)
const EXTERNAL_DST_REPORT_PATH = _operational_path(
    "SOLARSINDY_EXTERNAL_DST_REPORT",
    joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "external_dst_forecast_report.md"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "var", "monitor", "external_dst_forecast_report.md"),
    joinpath(LIVE_DIR, "external_dst_forecast_report.md"),
)
const HISTORICAL_V2_0_LIVE_LOG_PATH = joinpath(
    OPERATIONAL_PACKAGE_ROOT, "data", "historical", "v2_0", "live_forecast_log.csv",
)
const HISTORICAL_V2_0_LIVE_MANIFEST_PATH = joinpath(
    OPERATIONAL_PACKAGE_ROOT, "data", "historical", "v2_0",
    "live_forecast_log_manifest.csv",
)
const V2_1_CALIBRATION_POINT_PATH = joinpath(
    OPERATIONAL_PACKAGE_ROOT, "deploy", "operational_v2_calibration.csv",
)
const V2_1_CALIBRATION_CONFORMAL_PATH = joinpath(
    OPERATIONAL_PACKAGE_ROOT, "deploy", "operational_v2_calibration_conformal.csv",
)
const V2_1_CALIBRATION_SPLIT_PATH = joinpath(
    OPERATIONAL_PACKAGE_ROOT, "deploy", "operational_v2_calibration_split.csv",
)
const V2_1_SERVED_HOLDOUT_DIR = operational_evidence_dir(
    "v2_1_served_holdout_summary.csv", "v2_1_served_holdout_audit.csv",
)
const V2_1_SERVED_HOLDOUT_SUMMARY_PATH = joinpath(
    V2_1_SERVED_HOLDOUT_DIR, "v2_1_served_holdout_summary.csv",
)
const V2_1_SERVED_HOLDOUT_AUDIT_PATH = joinpath(
    V2_1_SERVED_HOLDOUT_DIR, "v2_1_served_holdout_audit.csv",
)
const V2_1_CALIBRATION_COVERAGE_FLOOR = 0.85
const V2_1_NOMINAL_COVERAGE = 0.90
const DEFAULT_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "V2_READINESS.md")
const DEFAULT_API_URL = "http://127.0.0.1:8723/api/status"
const DEFAULT_MAX_ISSUE_AGE_HOURS = 3.0
const DEFAULT_MAX_API_GENERATED_AGE_MIN = 10.0
const REGIME_MIN_ROWS = 40
const MATERIAL_PERSISTENCE_DELTA_NT = 2.0
const REQUIRED_REPLAY_COLS = [:storm, :issue_utc, :lead, :obs, :v2_1,
                              :v2_1_pre_rate_guard, :v2_1_pre_one_hour_inertia,
                              :v2_1_pre_state_inertia,
                              :v2_0, :v2_1_frozen,
                              :persistence, :rate]
const REQUIRED_LIVE_COLS = [
    :issue_time_utc,
    :latest_solar_wind_utc,
    :latest_dst_time_utc,
    :target_time_utc,
    :model_version,
    :observation_dst_nt,
    :served_pred_dst_nt,
    :served_pred_dst_ci05_nt,
    :served_pred_dst_ci95_nt,
    :served_residual_dst_nt,
    :served_observed_in_90ci,
    :v2_pred_dst_nt,
    :persistence_dst_nt,
    :sub_hourly_model_version,
]
const EXPECTED_MODEL_VERSION = "v2.1"
const EXPECTED_LEADS = [1, 2, 3, 6]
const EXPECTED_SUBHOURLY = "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
const EXPECTED_BROAD_STORMS = 193
const MIN_BROAD_REPLAY_ROWS = 70_000
const EXPECTED_GSCALE_EVENTS = 311
const MIN_GSCALE_SCORED_EVENTS = 200
const MIN_GSCALE_REPLAY_ROWS = 25_000
const MIN_NOAA_KP_FORECAST_ROWS = 30_000
const MIN_NOAA_KP_ISSUES = 1_500
const MIN_TEMERIN_DST_ROWS = 10_000
const MIN_TEMERIN_DST_STORMS = 25
const TEMERIN_DST_MATCH_TOL_MIN = 20.0
const TEMERIN_DST_DEFAULT_START = DateTime(2013, 5, 1)
const TEMERIN_DST_DSCOVR_START = DateTime(2016, 12, 1)
const DST_REGIME_ORDER = ["quiet", "minor", "moderate", "intense", "extreme"]
const RATE_REGIME_ORDER = ["recovering", "steady", "deepening", "rapid_deepening"]
const REQUIRED_BROAD_COLS = [:storm_id, :storm, :storm_split, :storm_min_dst_star_nt,
                             :issue_utc, :target_utc, :lead, :obs,
                             :v2_1, :v2_1_pre_rate_guard,
                             :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                             :v2_0, :v2_1_frozen,
                             :persistence, :rate]
const REQUIRED_GSCALE_COLS = [:g_event_id, :storm, :g_level, :peak_kp,
                              :event_start_utc, :event_end_utc,
                              :replay_start_utc, :replay_end_utc,
                              :issue_utc, :target_utc, :lead, :obs,
                              :v2_1, :v2_1_pre_rate_guard,
                              :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                              :v2_0, :v2_1_frozen,
                              :persistence, :rate]
const REQUIRED_NOAA_KP_COLS = [:issue_utc, :target_bin_start_utc, :target_bin_end_utc,
                               :lead_h, :forecast_day, :forecast_kp, :forecast_g_level,
                               :observed_kp, :observed_g_level, :kp_error, :lead_band]
const REQUIRED_TEMERIN_DST_COLS = [:storm_id, :storm, :storm_min_dst_star_nt,
                                   :issue_utc, :target_utc, :lead,
                                   :obs, :v2_1, :v2_0, :persistence,
                                   :temerin_li_dst, :temerin_valid_utc,
                                   :match_abs_gap_min, :temerin_source_file,
                                   :source_epoch]
const REQUIRED_EXTERNAL_DST_COLS = [:source, :issue_utc, :fetched_utc, :target_utc,
                                    :lead_h, :forecast_dst_nt, :forecast_cadence_min,
                                    :issue_basis, :source_url, :raw_sha256, :raw_path,
                                    :source_max_target_utc, :row_role, :observed_dst_nt,
                                    :observed_time_utc, :observed_gap_min, :abs_error_nt,
                                    :scored_utc]
const ACTIVE_PRODUCT_PATHS = [
    joinpath(OPERATIONAL_PACKAGE_ROOT, "app", "src"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "app", "public"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"),
    joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_monitor.jl"),
]
const EKF_DECISION = joinpath(OPERATIONAL_PACKAGE_ROOT, "docs", "src", "ekf-v3-decision.md")

mutable struct AuditState
    checks::Vector{NamedTuple}
    replay_metrics::DataFrame
    broad_metrics::DataFrame
    gscale_metrics::DataFrame
    noaa_kp_metrics::DataFrame
    temerin_dst_metrics::DataFrame
    external_dst_metrics::DataFrame
    regime_metrics::DataFrame
    live_metrics::Dict{Symbol, Any}
    paper_notes::Vector{String}
end

AuditState() = AuditState(
    NamedTuple[],
    DataFrame(lead = Int[], n = Int[], rmse_v2_0 = Float64[], rmse_v2_1 = Float64[],
              rmse_persistence = Float64[], improvement_vs_best = Float64[],
              max_tail_effect = Float64[], max_core_change = Float64[]),
    DataFrame(lead = Int[], n = Int[], n_storms = Int[], rmse_v2_0 = Float64[],
              rmse_v2_1 = Float64[], rmse_persistence = Float64[],
              improvement_vs_best = Float64[], max_tail_effect = Float64[],
              max_core_change = Float64[]),
    DataFrame(cohort = String[], lead = Int[], n = Int[], n_events = Int[],
              rmse_v2_0 = Float64[], rmse_v2_1 = Float64[],
              rmse_persistence = Float64[], improvement_vs_best = Float64[],
              max_tail_effect = Float64[], max_core_change = Float64[]),
    DataFrame(scope = String[], lead_band = String[], threshold_g = Int[],
              n_rows = Int[], hits = Int[], misses = Int[], false_alarms = Int[],
              pod = Float64[], far = Float64[], csi = Float64[],
              rmse_kp = Float64[]),
    DataFrame(scope = String[], lead = Int[], n = Int[], n_storms = Int[],
              rmse_temerin_valid = Float64[], rmse_v2_1 = Float64[],
              rmse_v2_0 = Float64[], rmse_persistence = Float64[],
              v2_1_minus_temerin = Float64[], max_gap_min = Float64[]),
    DataFrame(source = String[], n_rows = Int[], n_scored = Int[],
              n_issues = Int[], max_lead_h = Float64[],
              rmse_nt = Union{Missing, Float64}[], mae_nt = Union{Missing, Float64}[]),
    DataFrame(axis = String[], lead = Int[], regime = String[], n = Int[],
              rmse_v2_0 = Float64[], rmse_v2_1 = Float64[], rmse_persistence = Float64[],
              delta_vs_v2_0 = Float64[], delta_vs_best = Float64[]),
    Dict{Symbol, Any}(),
    String[],
)

function add_check!(state::AuditState, level::Symbol, name::AbstractString, detail::AbstractString)
    push!(state.checks, (level = level, name = String(name), detail = String(detail)))
    return level != :fail
end

function _csv_key_parts(cols...)
    key = fill("", length(first(cols)))
    for col in cols
        key .= key .* "|" .* string.(col)
    end
    return key
end

pass!(state, name, detail) = add_check!(state, :pass, name, detail)
warn!(state, name, detail) = add_check!(state, :warn, name, detail)
fail!(state, name, detail) = add_check!(state, :fail, name, detail)

has_col(df::DataFrame, col::Symbol) = String(col) in names(df)

function finite_value(x)
    ismissing(x) && return false
    try
        return isfinite(Float64(x))
    catch
        return false
    end
end

function finite_mask(df::DataFrame, cols::Vector{Symbol})
    mask = trues(nrow(df))
    for col in cols
        mask .&= [finite_value(x) for x in df[!, col]]
    end
    return mask
end

function rmse(pred, obs)
    p = Float64.(pred)
    o = Float64.(obs)
    return sqrt(mean((p .- o) .^ 2))
end

function target_dst_regime(obs)
    x = Float64(obs)
    x > -30.0 && return "quiet"
    x > -50.0 && return "minor"
    x > -100.0 && return "moderate"
    x > -200.0 && return "intense"
    return "extreme"
end

function issue_rate_regime(rate)
    x = Float64(rate)
    x < -15.0 && return "rapid_deepening"
    x < -5.0 && return "deepening"
    x <= 5.0 && return "steady"
    return "recovering"
end

function newest_cycle_rows(df::DataFrame)
    if has_col(df, :latest_solar_wind_utc)
        parsed = parse_utc_datetime.(df.latest_solar_wind_utc)
        valid = .!ismissing.(parsed)
        if any(valid)
            newest = maximum(parsed[valid])
            return df[coalesce.(parsed .== newest, false), :]
        end
    end
    parsed_issue = has_col(df, :issue_time_utc) ? parse_utc_datetime.(df.issue_time_utc) :
                   Vector{Union{Missing, DateTime}}(missing, nrow(df))
    valid_issue = .!ismissing.(parsed_issue)
    any(valid_issue) || return df[max(1, nrow(df) - 3):nrow(df), :]
    newest_issue = maximum(parsed_issue[valid_issue])
    return df[coalesce.(parsed_issue .== newest_issue, false), :]
end

function nested_get(x, keys::Vector{String}, default = missing)
    cur = x
    for key in keys
        if cur isa AbstractDict
            if haskey(cur, key)
                cur = cur[key]
            elseif haskey(cur, Symbol(key))
                cur = cur[Symbol(key)]
            else
                return default
            end
        else
            return default
        end
    end
    return cur
end

function float_or_missing(x)
    ismissing(x) && return missing
    x === nothing && return missing
    try
        return Float64(x)
    catch
        return missing
    end
end

function parse_utc_datetime(x)
    ismissing(x) && return missing
    x === nothing && return missing
    x isa DateTime && return x
    s = strip(String(x))
    isempty(s) && return missing
    s = replace(s, r"Z$" => "", r"\+00:00$" => "")
    m = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?", s)
    m === nothing && return missing
    base = m.captures[1]
    frac = m.captures[2]
    normalized = frac === nothing ? base : base * "." * rpad(frac[1:min(end, 3)], 3, '0')
    try
        return DateTime(normalized)
    catch
        return missing
    end
end

age_hours(dt::DateTime, now_utc::DateTime) = Dates.value(now_utc - dt) / 3_600_000.0
age_minutes(dt::DateTime, now_utc::DateTime) = Dates.value(now_utc - dt) / 60_000.0

function read_csv_checked!(state::AuditState, path::AbstractString, label::AbstractString;
                           allow_zero::Bool=false)
    if !isfile(path)
        fail!(state, "$label exists", "$(relpath(path, REPO_ROOT)) is missing")
        return nothing
    end
    if filesize(path) == 0
        fail!(state, "$label nonempty", "$(relpath(path, REPO_ROOT)) is empty")
        return nothing
    end
    try
        df = CSV.read(path, DataFrame)
        if nrow(df) == 0 && !allow_zero
            fail!(state, "$label rows", "$(relpath(path, REPO_ROOT)) has zero data rows")
            return nothing
        end
        pass!(state, "$label readable", "$(relpath(path, REPO_ROOT)) rows=$(nrow(df)) cols=$(ncol(df))")
        return df
    catch err
        fail!(state, "$label readable", sprint(showerror, err))
        return nothing
    end
end

function require_columns!(state::AuditState, df::DataFrame, cols::Vector{Symbol}, label::AbstractString)
    missing_cols = [String(c) for c in cols if !has_col(df, c)]
    if isempty(missing_cols)
        pass!(state, "$label schema", "required columns present")
        return true
    else
        fail!(state, "$label schema", "missing columns: $(join(missing_cols, ", "))")
        return false
    end
end

function _v2_1_split_contract(df::DataFrame, deployed_point_sha256::AbstractString;
                              coverage_floor::Real=V2_1_CALIBRATION_COVERAGE_FLOOR)
    required = [
        :split, :rows, :anchors, :minimum_issue_utc, :maximum_issue_utc,
        :minimum_target_utc, :maximum_target_utc,
        :point_calibration_sha256, :source_table_sha256,
        :conformal_holdout_coverage,
    ]
    all(c -> has_col(df, c), required) || return (
        schema=false, partitions=false, causal=false, point_hash=false,
        source_hash=false, frozen_tail_coverage=false,
    )
    expected = ["fit", "validation", "holdout"]
    split_names = String.(df.split)
    partitions = nrow(df) == 3 && sort(split_names) == sort(expected) &&
                 all(Int.(df.rows) .> 0) && all(Int.(df.anchors) .> 0)
    partitions || return (
        schema=true, partitions=false, causal=false, point_hash=false,
        source_hash=false, frozen_tail_coverage=false,
    )
    ordered = [df[findfirst(==(name), split_names), :] for name in expected]
    min_issue = parse_utc_datetime.([r.minimum_issue_utc for r in ordered])
    max_target = parse_utc_datetime.([r.maximum_target_utc for r in ordered])
    causal = all(.!ismissing.(min_issue)) && all(.!ismissing.(max_target)) &&
             max_target[1] < min_issue[2] && max_target[2] < min_issue[3]

    point_hashes = String.(df.point_calibration_sha256)
    point_hash = length(unique(point_hashes)) == 1 &&
                 length(first(point_hashes)) == 64 &&
                 first(point_hashes) == String(deployed_point_sha256)
    source_hashes = String.(df.source_table_sha256)
    source_hash = length(unique(source_hashes)) == 1 &&
                  length(first(source_hashes)) == 64 &&
                  all(isxdigit, first(source_hashes))
    coverages = Float64.(df.conformal_holdout_coverage)
    frozen_tail_coverage = all(isfinite, coverages) &&
                           length(unique(coverages)) == 1 &&
                           first(coverages) >= Float64(coverage_floor)
    return (
        schema=true, partitions=partitions, causal=causal,
        point_hash=point_hash, source_hash=source_hash,
        frozen_tail_coverage=frozen_tail_coverage,
    )
end

function audit_v2_1_calibration_split!(state::AuditState)
    split = read_csv_checked!(
        state, V2_1_CALIBRATION_SPLIT_PATH, "V2.1 calibration split audit",
    )
    split === nothing && return
    if !isfile(V2_1_CALIBRATION_POINT_PATH)
        fail!(state, "V2.1 deployed point calibration", "deployed point calibration is missing")
        return
    end
    point_sha = bytes2hex(sha256(read(V2_1_CALIBRATION_POINT_PATH)))
    contract = _v2_1_split_contract(split, point_sha)
    contract.schema ? pass!(state, "V2.1 calibration split schema", "required fields present") :
                      fail!(state, "V2.1 calibration split schema", "required fields are missing")
    contract.partitions ? pass!(state, "V2.1 calibration split partitions", "fit/validation/holdout are unique and nonempty") :
                          fail!(state, "V2.1 calibration split partitions", "expected exactly three nonempty fit/validation/holdout rows")
    contract.causal ? pass!(state, "V2.1 calibration forecast-origin causality", "fit max target < validation min issue and validation max target < holdout min issue") :
                      fail!(state, "V2.1 calibration forecast-origin causality", "a later split begins before all preceding targets are observable")
    contract.point_hash ? pass!(state, "V2.1 calibration split point hash", "split audit matches deployed point calibration SHA-256") :
                          fail!(state, "V2.1 calibration split point hash", "split point hash does not match the deployed calibration")
    contract.source_hash ? pass!(state, "V2.1 calibration source hash", "one nonempty SHA-256 identifies the calibration source table") :
                           fail!(state, "V2.1 calibration source hash", "source-table SHA-256 is missing or inconsistent")
    contract.frozen_tail_coverage ?
        pass!(state, "V2.1 frozen-tail calibration holdout coverage",
              "frozen-tail conformal coverage meets the $(V2_1_CALIBRATION_COVERAGE_FLOOR) calibration floor") :
        fail!(state, "V2.1 frozen-tail calibration holdout coverage",
              "frozen-tail conformal coverage is inconsistent or below the calibration floor")
    return
end

function _sha256_if_file(path::AbstractString)
    isfile(path) || return ""
    return bytes2hex(sha256(read(path)))
end

function _v2_1_served_holdout_contract(summary::DataFrame, audit::DataFrame;
                                        point_sha256::AbstractString,
                                        conformal_sha256::AbstractString,
                                        split_sha256::AbstractString,
                                        omni_sha256::AbstractString)
    summary_required = [
        :cohort, :lead_h, :activity_regime, :n_rows, :served_hits,
        :served_coverage, :served_rmse_nt, :frozen_tail_hits,
        :frozen_tail_coverage, :frozen_tail_rmse_nt, :nominal_coverage,
        :promotion_coverage_floor, :pooled_gate_applies, :pooled_gate_pass,
    ]
    audit_required = [
        :model_version, :candidate_count, :active_count, :holdout_rows,
        :holdout_anchors, :validation_max_target_utc, :holdout_min_issue_utc,
        :holdout_max_issue_utc, :holdout_max_target_utc,
        :strict_forecast_origin_separation, :interval_policy,
        :holdout_residual_updates, :point_calibration_sha256,
        :conformal_calibration_sha256, :split_audit_sha256,
        :calibration_scored_sha256, :omni_sha256,
        :maximum_frozen_tail_continuity_error_nt,
        :maximum_interval_center_error_nt, :nominal_coverage,
        :pooled_promotion_floor, :served_pooled_coverage,
        :served_pooled_gate_pass, :frozen_tail_pooled_coverage,
        :heldout_promotion_evidence,
    ]
    schema = all(c -> has_col(summary, c), summary_required) &&
             all(c -> has_col(audit, c), audit_required) && nrow(audit) == 1
    schema || return (
        schema=false, identity=false, partition=false, causal=false,
        policy=false, hashes=false, geometry=false, summary_crc=false,
        pooled_gate=false, diagnostics=false,
    )

    a = audit[1, :]
    identity = String(a.model_version) == EXPECTED_MODEL_VERSION &&
               Int(a.candidate_count) == 20 && Int(a.active_count) == 11
    validation_max_target = parse_utc_datetime(a.validation_max_target_utc)
    holdout_min_issue = parse_utc_datetime(a.holdout_min_issue_utc)
    holdout_max_issue = parse_utc_datetime(a.holdout_max_issue_utc)
    holdout_max_target = parse_utc_datetime(a.holdout_max_target_utc)
    partition = Int(a.holdout_rows) > 0 && Int(a.holdout_anchors) > 0 &&
                !ismissing(holdout_max_issue) && !ismissing(holdout_max_target) &&
                holdout_max_target > holdout_max_issue
    causal = Bool(a.strict_forecast_origin_separation) &&
             !ismissing(validation_max_target) && !ismissing(holdout_min_issue) &&
             validation_max_target < holdout_min_issue
    policy = String(a.interval_policy) ==
                 "static_conformal_shifted_to_complete_hour_served_center" &&
             Int(a.holdout_residual_updates) == 0 &&
             Bool(a.heldout_promotion_evidence)
    calibration_scored_sha = String(a.calibration_scored_sha256)
    hashes = String(a.point_calibration_sha256) == point_sha256 &&
             String(a.conformal_calibration_sha256) == conformal_sha256 &&
             String(a.split_audit_sha256) == split_sha256 &&
             String(a.omni_sha256) == omni_sha256 &&
             length(calibration_scored_sha) == 64 &&
             all(isxdigit, calibration_scored_sha)
    geometry = abs(Float64(a.maximum_frozen_tail_continuity_error_nt)) <= 1e-12 &&
               abs(Float64(a.maximum_interval_center_error_nt)) <= 1e-12

    cohorts = String.(summary.cohort)
    overall_indices = findall(==("overall"), cohorts)
    lead_rows = summary[(String.(summary.activity_regime) .== "all") .&
                        (Int.(summary.lead_h) .> 0), :]
    regime_rows = summary[Int.(summary.lead_h) .== 0, :]
    summary_crc = false
    pooled_gate = false
    diagnostics = false
    if length(overall_indices) == 1
        overall = summary[only(overall_indices), :]
        n = Int(overall.n_rows)
        hits = Int(overall.served_hits)
        frozen_hits = Int(overall.frozen_tail_hits)
        summary_crc = n == Int(a.holdout_rows) && 0 <= hits <= n &&
                      0 <= frozen_hits <= n &&
                      Float64(overall.served_coverage) == hits / n &&
                      Float64(overall.frozen_tail_coverage) == frozen_hits / n &&
                      sort(Int.(lead_rows.lead_h)) == EXPECTED_LEADS &&
                      sum(Int.(lead_rows.n_rows)) == n &&
                      sum(Int.(lead_rows.served_hits)) == hits &&
                      all(isfinite, Float64.(summary.served_rmse_nt)) &&
                      all(isfinite, Float64.(summary.frozen_tail_rmse_nt))
        floor = Float64(overall.promotion_coverage_floor)
        coverage = Float64(overall.served_coverage)
        pooled_gate = Bool(overall.pooled_gate_applies) &&
                      Bool(overall.pooled_gate_pass) &&
                      Bool(a.served_pooled_gate_pass) &&
                      floor == V2_1_CALIBRATION_COVERAGE_FLOOR &&
                      Float64(a.pooled_promotion_floor) == floor &&
                      Float64(a.served_pooled_coverage) == coverage &&
                      coverage >= floor &&
                      Float64(overall.nominal_coverage) == V2_1_NOMINAL_COVERAGE &&
                      Float64(a.nominal_coverage) == V2_1_NOMINAL_COVERAGE

        quiet = summary[cohorts .== "quiet", :]
        storm = summary[cohorts .== "storm", :]
        diagnostics = nrow(quiet) == 1 && nrow(storm) == 1 &&
                      Int(quiet.n_rows[1]) + Int(storm.n_rows[1]) == n &&
                      Int(quiet.served_hits[1]) + Int(storm.served_hits[1]) == hits &&
                      sort(Int.(lead_rows.lead_h)) == EXPECTED_LEADS &&
                      nrow(regime_rows) == 3
    end
    return (
        schema=schema, identity=identity, partition=partition, causal=causal,
        policy=policy, hashes=hashes, geometry=geometry,
        summary_crc=summary_crc, pooled_gate=pooled_gate,
        diagnostics=diagnostics,
    )
end

function audit_v2_1_served_holdout!(state::AuditState)
    summary = read_csv_checked!(
        state, V2_1_SERVED_HOLDOUT_SUMMARY_PATH,
        "complete-hour served-stack V2.1 holdout summary",
    )
    audit = read_csv_checked!(
        state, V2_1_SERVED_HOLDOUT_AUDIT_PATH,
        "complete-hour served-stack V2.1 holdout audit",
    )
    (summary === nothing || audit === nothing) && return

    contract = _v2_1_served_holdout_contract(
        summary, audit;
        point_sha256=_sha256_if_file(V2_1_CALIBRATION_POINT_PATH),
        conformal_sha256=_sha256_if_file(V2_1_CALIBRATION_CONFORMAL_PATH),
        split_sha256=_sha256_if_file(V2_1_CALIBRATION_SPLIT_PATH),
        omni_sha256=_sha256_if_file(OPERATIONAL_OMNI),
    )
    checks = (
        (:schema, "complete-hour served-stack V2.1 holdout schema",
         "summary and audit fields are complete",
         "summary or audit fields are missing"),
        (:identity, "complete-hour served-stack V2.1 model identity",
         "v2.1 uses 20 candidates and 11 active terms",
         "holdout identity is not the deployed 20/11 V2.1 core"),
        (:partition, "complete-hour served-stack V2.1 holdout partition",
         "row/anchor counts and terminal issue/target bounds are valid",
         "holdout counts or terminal bounds are invalid"),
        (:causal, "complete-hour served-stack V2.1 forecast-origin causality",
         "every holdout issue follows the last validation target",
         "holdout issues overlap unmatured validation targets"),
        (:policy, "complete-hour served-stack V2.1 interval policy",
         "static conformal offsets are shifted to the served center with zero holdout updates",
         "interval policy, evidence role, or holdout-update count is inconsistent"),
        (:hashes, "complete-hour served-stack V2.1 artifact identity",
         "point, conformal, split, calibration-source, and OMNI hashes are valid",
         "a holdout hash differs from the deployed/source artifact"),
        (:geometry, "complete-hour served-stack V2.1 interval geometry",
         "frozen-tail continuity and served-center errors are <= 1e-12 nT",
         "continuity or interval-center error exceeds 1e-12 nT"),
        (:summary_crc, "complete-hour served-stack V2.1 summary recomputation",
         "pooled, lead, and hit-count identities are internally consistent",
         "summary row, hit, coverage, lead, or finite-value identity failed"),
        (:pooled_gate, "complete-hour served-stack V2.1 pooled holdout coverage gate",
         "pooled static coverage meets the declared 0.85 promotion floor",
         "pooled static coverage is inconsistent or below the 0.85 promotion floor"),
        (:diagnostics, "complete-hour served-stack V2.1 diagnostic strata",
         "lead and quiet/storm strata partition the pooled holdout",
         "lead or quiet/storm diagnostic strata are incomplete"),
    )
    for (field, name, pass_detail, fail_detail) in checks
        getproperty(contract, field) ? pass!(state, name, pass_detail) :
                                       fail!(state, name, fail_detail)
    end
    contract.schema || return

    state.live_metrics[:served_holdout_summary] = summary
    overall = summary[String.(summary.cohort) .== "overall", :][1, :]
    lead6 = summary[String.(summary.cohort) .== "lead_6", :][1, :]
    storm = summary[String.(summary.cohort) .== "storm", :][1, :]
    if Float64(lead6.served_coverage) < V2_1_CALIBRATION_COVERAGE_FLOOR
        warn!(state, "complete-hour served-stack V2.1 6 h coverage boundary",
              @sprintf("coverage %.3f is below the pooled %.2f floor; the declared gate is pooled",
                       Float64(lead6.served_coverage), V2_1_CALIBRATION_COVERAGE_FLOOR))
    else
        pass!(state, "complete-hour served-stack V2.1 6 h coverage boundary",
              @sprintf("coverage %.3f", Float64(lead6.served_coverage)))
    end
    if Float64(storm.served_coverage) < V2_1_CALIBRATION_COVERAGE_FLOOR
        warn!(state, "complete-hour served-stack V2.1 storm coverage boundary",
              @sprintf("Dst < -50 nT coverage %.3f over %d rows; pooled gate coverage %.3f",
                       Float64(storm.served_coverage), Int(storm.n_rows),
                       Float64(overall.served_coverage)))
    else
        pass!(state, "complete-hour served-stack V2.1 storm coverage boundary",
              @sprintf("Dst < -50 nT coverage %.3f over %d rows",
                       Float64(storm.served_coverage), Int(storm.n_rows)))
    end
    return
end

function _regime_detail(rows::DataFrame, limit::Int = 4)
    nrow(rows) == 0 && return ""
    sort!(rows, :delta_vs_best, rev = true)
    parts = String[]
    for r in eachrow(first(rows, min(limit, nrow(rows))))
        push!(parts, @sprintf("%s %dh %s n=%d delta_vs_best=%+.2f nT",
                              r.axis, r.lead, r.regime, r.n, r.delta_vs_best))
    end
    return join(parts, "; ")
end

function _append_regime_metrics!(out::DataFrame, df::DataFrame, axis::AbstractString,
                                 regimes::Vector{String}, labels::Vector{String};
                                 min_n::Int = REGIME_MIN_ROWS)
    work = select(df, :lead, :obs, :v2_0, :v2_1, :persistence)
    work[!, :regime] = labels
    for lead in EXPECTED_LEADS, regime in regimes
        sub = work[(work.lead .== lead) .& (work.regime .== regime), :]
        nrow(sub) >= min_n || continue
        rv20 = rmse(sub.v2_0, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        push!(out, (String(axis), lead, regime, nrow(sub), rv20, rv21, rpers,
                    rv21 - rv20, rv21 - min(rv20, rpers)))
    end
    return out
end

function replay_regime_metrics(df::DataFrame; min_n::Int = REGIME_MIN_ROWS)
    out = DataFrame(axis = String[], lead = Int[], regime = String[], n = Int[],
                    rmse_v2_0 = Float64[], rmse_v2_1 = Float64[], rmse_persistence = Float64[],
                    delta_vs_v2_0 = Float64[], delta_vs_best = Float64[])
    dst_labels = target_dst_regime.(df.obs)
    rate_labels = issue_rate_regime.(df.rate)
    _append_regime_metrics!(out, df, "target_dst", DST_REGIME_ORDER, dst_labels; min_n = min_n)
    _append_regime_metrics!(out, df, "issue_rate", RATE_REGIME_ORDER, rate_labels; min_n = min_n)
    return out
end

function audit_replay_regimes!(state::AuditState, df::DataFrame;
                               min_n::Int = REGIME_MIN_ROWS,
                               material_delta_nt::Float64 = MATERIAL_PERSISTENCE_DELTA_NT,
                               strict_persistence::Bool = false,
                               require_full_coverage::Bool = true)
    metrics = replay_regime_metrics(df; min_n = min_n)
    append!(state.regime_metrics, metrics)

    expected_cells = length(EXPECTED_LEADS) * (length(DST_REGIME_ORDER) + length(RATE_REGIME_ORDER))
    if nrow(metrics) == expected_cells
        pass!(state, "regime scorecard coverage", "all $(expected_cells) target-Dst/rate cells have n >= $(min_n)")
    elseif require_full_coverage
        fail!(state, "regime scorecard coverage", "expected $(expected_cells) populated cells with n >= $(min_n), got $(nrow(metrics))")
    else
        warn!(state, "regime scorecard coverage", "partial synthetic/self-test scorecard cells=$(nrow(metrics))")
    end

    v2_bad = metrics[metrics.delta_vs_v2_0 .> 1e-9, :]
    if nrow(v2_bad) == 0
        pass!(state, "regime historical V2.0 guard", "V2.1 RMSE <= historical V2.0 in every populated target-Dst/rate cell")
    else
        fail!(state, "regime historical V2.0 guard", "V2.1 worse than historical V2.0: $(_regime_detail(v2_bad))")
    end

    # Paper regime criterion (Eq. 16): Δ_C = RMSE_C(V2) − min{baseline, persistence} ≤ 2 nT as a HARD
    # gate. This makes a cell that loses to the stronger comparator (persistence or historical V2.0) by more
    # than the limit FAIL rather than only warn, matching the published promotion criterion. The
    # zero-tolerance baseline guard above remains an additional, stricter check.
    eq16_bad = metrics[metrics.delta_vs_best .> material_delta_nt + 1e-9, :]
    if nrow(eq16_bad) == 0
        pass!(state, "regime criterion (Eq. 16)",
              @sprintf("V2 RMSE within %.1f nT of the stronger baseline in every populated cell", material_delta_nt))
    else
        fail!(state, "regime criterion (Eq. 16)",
              @sprintf("%d cells exceed the %.1f nT regime limit vs the stronger baseline: %s",
                       nrow(eq16_bad), material_delta_nt, _regime_detail(eq16_bad)))
    end

    pers_bad = metrics[(metrics.rmse_persistence .< metrics.rmse_v2_1) .&
                       (metrics.rmse_v2_1 .- metrics.rmse_persistence .> material_delta_nt), :]
    if nrow(pers_bad) == 0
        pass!(state, "regime persistence vulnerability", @sprintf("no populated cell loses to persistence by more than %.1f nT", material_delta_nt))
    else
        detail = @sprintf("%d populated cells lose to persistence by > %.1f nT; worst: %s",
                          nrow(pers_bad), material_delta_nt, _regime_detail(pers_bad))
        strict_persistence ? fail!(state, "regime persistence vulnerability", detail) :
                             warn!(state, "regime persistence vulnerability", detail)
    end
end

function audit_replay!(state::AuditState; strict_regime_persistence::Bool = false)
    path = joinpath(LIVE_DIR, "v2_replay_scored.csv")
    df = read_csv_checked!(state, path, "V2 replay CSV")
    df === nothing && return
    require_columns!(state, df, REQUIRED_REPLAY_COLS, "V2 replay") || return

    if nrow(df) >= 1000
        pass!(state, "V2 replay sample size", "n=$(nrow(df)) scored rows")
    else
        fail!(state, "V2 replay sample size", "expected at least 1000 scored rows, got $(nrow(df))")
    end

    leads = sort(unique(Int.(df.lead)))
    if leads == EXPECTED_LEADS
        pass!(state, "V2 replay leads", "leads=$(join(leads, ",")) h")
    else
        fail!(state, "V2 replay leads", "expected $(join(EXPECTED_LEADS, ",")); got $(join(leads, ","))")
    end

    numeric_cols = [:obs, :v2_1, :v2_1_pre_rate_guard,
                    :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                    :v2_0, :v2_1_frozen, :persistence]
    finite_rows = finite_mask(df, numeric_cols)
    if all(finite_rows)
        pass!(state, "V2 replay finite values", "all $(nrow(df)) rows finite for $(join(String.(numeric_cols), ", "))")
    else
        fail!(state, "V2 replay finite values", "$(count(.!finite_rows)) rows contain non-finite scored values")
    end

    for lead in EXPECTED_LEADS
        sub = df[df.lead .== lead, :]
        if nrow(sub) == 0
            fail!(state, "lead $(lead)h replay coverage", "no scored rows")
            continue
        end
        rv20 = rmse(sub.v2_0, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        tail_effect = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen)))
        core_change = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0)))
        improvement = min(rv20, rpers) - rv21
        push!(state.replay_metrics, (lead, nrow(sub), rv20, rv21, rpers,
                                     improvement, tail_effect, core_change))

        if rv21 < rv20 && rv21 < rpers
            pass!(state, "lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f < historical V2.0 %.2f and persistence %.2f", rv21, rv20, rpers))
        else
            fail!(state, "lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f, historical V2.0 %.2f, persistence %.2f", rv21, rv20, rpers))
        end
        pass!(state, "lead $(lead)h operational-layer decomposition",
              @sprintf("max|V2.1−frozen-tail V2.1| %.2f nT; max|frozen-tail V2.1−V2.0| %.2f nT",
                       tail_effect, core_change))
    end

    deep = df[(df.lead .== 6) .&
              isfinite.(Float64.(df.rate)) .&
              (Float64.(df.rate) .< -15.0) .&
              (Float64.(df.obs) .< -100.0), :]
    if nrow(deep) == 0
        fail!(state, "deep-deepening replay subset", "no 6h rows with rate < -15 nT/h and obs < -100 nT")
    else
        v2_bias = mean(Float64.(deep.obs) .- Float64.(deep.v2_1))
        baseline_bias = mean(Float64.(deep.obs) .- Float64.(deep.v2_0))
        state.live_metrics[:deep_subset_n] = nrow(deep)
        state.live_metrics[:deep_v2_bias] = v2_bias
        state.live_metrics[:deep_baseline_bias] = baseline_bias
        if abs(v2_bias) <= 10.0 && abs(v2_bias) < abs(baseline_bias)
            pass!(state, "deep-deepening bias guard",
                  @sprintf("n=%d, mean(obs−V2.1)=%+.1f nT vs mean(obs−historical V2.0)=%+.1f nT", nrow(deep), v2_bias, baseline_bias))
        else
            fail!(state, "deep-deepening bias guard",
                  @sprintf("n=%d, mean(obs−V2.1)=%+.1f nT vs mean(obs−historical V2.0)=%+.1f nT", nrow(deep), v2_bias, baseline_bias))
        end
    end

    audit_replay_regimes!(state, df; strict_persistence = strict_regime_persistence)
end

function audit_broad_replay!(state::AuditState)
    path = joinpath(LIVE_DIR, "v2_broad_replay_scored.csv")
    df = read_csv_checked!(state, path, "broad Dst-intense replay CSV")
    df === nothing && return
    require_columns!(state, df, REQUIRED_BROAD_COLS, "broad Dst-intense replay") || return

    if nrow(df) >= MIN_BROAD_REPLAY_ROWS
        pass!(state, "broad replay sample size", "n=$(nrow(df)) scored rows")
    else
        fail!(state, "broad replay sample size", "expected at least $(MIN_BROAD_REPLAY_ROWS) scored rows, got $(nrow(df))")
    end

    storm_count = length(unique(Int.(df.storm_id)))
    if storm_count == EXPECTED_BROAD_STORMS
        pass!(state, "broad replay storm coverage", "scored $(storm_count) Dst-intense storms")
    else
        fail!(state, "broad replay storm coverage", "expected $(EXPECTED_BROAD_STORMS) storms, got $(storm_count)")
    end

    if maximum(Float64.(df.storm_min_dst_star_nt)) <= -100.0
        pass!(state, "broad replay Dst*-threshold scope", "all scored storms have catalog min Dst* <= -100 nT")
    else
        fail!(state, "broad replay Dst*-threshold scope", "at least one scored storm has catalog min Dst* > -100 nT")
    end

    leads = sort(unique(Int.(df.lead)))
    if leads == EXPECTED_LEADS
        pass!(state, "broad replay leads", "leads=$(join(leads, ",")) h")
    else
        fail!(state, "broad replay leads", "expected $(join(EXPECTED_LEADS, ",")); got $(join(leads, ","))")
    end

    numeric_cols = [:obs, :v2_1, :v2_1_pre_rate_guard,
                    :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                    :v2_0, :v2_1_frozen, :persistence]
    finite_rows = finite_mask(df, numeric_cols)
    if all(finite_rows)
        pass!(state, "broad replay finite values", "all $(nrow(df)) rows finite for $(join(String.(numeric_cols), ", "))")
    else
        fail!(state, "broad replay finite values", "$(count(.!finite_rows)) rows contain non-finite scored values")
    end

    issue = parse_utc_datetime.(df.issue_utc)
    target = parse_utc_datetime.(df.target_utc)
    time_ok = true
    for i in 1:nrow(df)
        if ismissing(issue[i]) || ismissing(target[i]) ||
           target[i] != issue[i] + Hour(Int(df.lead[i]))
            time_ok = false
            break
        end
    end
    if time_ok
        pass!(state, "broad replay locked-row timing", "target_utc equals issue_utc + lead for all rows")
    else
        fail!(state, "broad replay locked-row timing", "at least one target_utc does not equal issue_utc + lead")
    end

    for lead in EXPECTED_LEADS
        sub = df[df.lead .== lead, :]
        if nrow(sub) == 0
            fail!(state, "broad lead $(lead)h coverage", "no scored rows")
            continue
        end
        rv20 = rmse(sub.v2_0, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        tail_effect = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen)))
        core_change = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0)))
        improvement = min(rv20, rpers) - rv21
        push!(state.broad_metrics, (lead, nrow(sub), length(unique(Int.(sub.storm_id))),
                                    rv20, rv21, rpers, improvement, tail_effect, core_change))
        if rv21 < rv20 && rv21 < rpers
            pass!(state, "broad lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f < historical V2.0 %.2f and persistence %.2f", rv21, rv20, rpers))
        else
            fail!(state, "broad lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f, historical V2.0 %.2f, persistence %.2f", rv21, rv20, rpers))
        end
        pass!(state, "broad lead $(lead)h operational-layer decomposition",
              @sprintf("max|V2.1−frozen-tail V2.1| %.2f nT; max|frozen-tail V2.1−V2.0| %.2f nT",
                       tail_effect, core_change))
    end

    deep = df[(df.lead .== 6) .&
              isfinite.(Float64.(df.rate)) .&
              (Float64.(df.rate) .< -15.0) .&
              (Float64.(df.obs) .< -100.0), :]
    if nrow(deep) == 0
        fail!(state, "broad deep-deepening replay subset", "no 6h rows with rate < -15 nT/h and obs < -100 nT")
    else
        v2_bias = mean(Float64.(deep.obs) .- Float64.(deep.v2_1))
        baseline_bias = mean(Float64.(deep.obs) .- Float64.(deep.v2_0))
        state.live_metrics[:broad_deep_subset_n] = nrow(deep)
        state.live_metrics[:broad_deep_v2_bias] = v2_bias
        state.live_metrics[:broad_deep_baseline_bias] = baseline_bias
        if abs(v2_bias) <= 10.0 && abs(v2_bias) < abs(baseline_bias)
            pass!(state, "broad deep-deepening bias guard",
                  @sprintf("n=%d, mean(obs−V2.1)=%+.1f nT vs mean(obs−historical V2.0)=%+.1f nT",
                            nrow(deep), v2_bias, baseline_bias))
        else
            fail!(state, "broad deep-deepening bias guard",
                  @sprintf("n=%d, mean(obs−V2.1)=%+.1f nT vs mean(obs−historical V2.0)=%+.1f nT",
                            nrow(deep), v2_bias, baseline_bias))
        end
    end
end

function _gscale_summary_consistent(df::DataFrame, summary::DataFrame)
    max_error = 0.0
    for r in eachrow(summary)
        cohort = String(r.cohort)
        sub = cohort == "all_G3plus" ? df :
              df[Int.(df.g_level) .== parse(Int, replace(cohort, "G" => "")), :]
        sub = sub[Int.(sub.lead) .== Int(r.lead_h), :]
        nrow(sub) == Int(r.n_rows) || return (ok = false, detail = "row-count mismatch for $cohort $(r.lead_h)h")
        rv20 = rmse(sub.v2_0, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        tail_effect = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen)))
        core_change = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0)))
        vals = [
            abs(rv20 - Float64(r.rmse_v2_0_nt)),
            abs(rv21 - Float64(r.rmse_v2_1_nt)),
            abs(rpers - Float64(r.rmse_persistence_nt)),
            abs(tail_effect - Float64(r.max_tail_effect_nt)),
            abs(core_change - Float64(r.max_core_change_nt)),
            abs((min(rv20, rpers) - rv21) - Float64(r.improvement_vs_best_nt)),
        ]
        max_error = max(max_error, maximum(vals))
    end
    return (ok = max_error <= 1e-9, detail = @sprintf("max summary recompute error %.3g", max_error))
end

function audit_gscale_replay!(state::AuditState)
    path = joinpath(LIVE_DIR, "v2_gscale_replay_scored.csv")
    df = read_csv_checked!(state, path, "exact Kp/G-scale replay CSV")
    df === nothing && return
    require_columns!(state, df, REQUIRED_GSCALE_COLS, "exact Kp/G-scale replay") || return

    events_path = joinpath(LIVE_DIR, "v2_gscale_event_catalog.csv")
    events = read_csv_checked!(state, events_path, "exact Kp/G-scale event catalog")
    if events !== nothing
        if nrow(events) == EXPECTED_GSCALE_EVENTS
            pass!(state, "exact Kp/G-scale event catalog size", "events=$(nrow(events))")
        else
            fail!(state, "exact Kp/G-scale event catalog size",
                  "expected $(EXPECTED_GSCALE_EVENTS), got $(nrow(events))")
        end
        if all(c -> has_col(events, c), [:replay_start_utc, :replay_end_utc, :peak_kp, :peak_g_level])
            starts = parse_utc_datetime.(events.replay_start_utc)
            ends = parse_utc_datetime.(events.replay_end_utc)
            disjoint = all(.!ismissing.(starts)) && all(.!ismissing.(ends)) &&
                       all(starts[2:end] .> ends[1:end-1])
            if disjoint
                pass!(state, "exact Kp/G-scale replay-window uniqueness", "event replay windows are disjoint")
            else
                fail!(state, "exact Kp/G-scale replay-window uniqueness", "overlapping or unparsable event replay windows")
            end
            if minimum(Float64.(events.peak_kp)) >= 7.0 && minimum(Int.(events.peak_g_level)) >= 3
                pass!(state, "exact Kp/G-scale event scope", "all catalog events have peak Kp >= 7 and NOAA G >= 3")
            else
                fail!(state, "exact Kp/G-scale event scope", "catalog contains an event below G3/Kp 7")
            end
        else
            fail!(state, "exact Kp/G-scale event catalog schema", "missing replay window or peak Kp/G columns")
        end
    end

    if nrow(df) >= MIN_GSCALE_REPLAY_ROWS
        pass!(state, "exact Kp/G-scale replay sample size", "n=$(nrow(df)) scored rows")
    else
        fail!(state, "exact Kp/G-scale replay sample size",
              "expected at least $(MIN_GSCALE_REPLAY_ROWS) scored rows, got $(nrow(df))")
    end

    scored_events = length(unique(Int.(df.g_event_id)))
    if scored_events >= MIN_GSCALE_SCORED_EVENTS
        pass!(state, "exact Kp/G-scale scored-event coverage", "scored $(scored_events) events")
    else
        fail!(state, "exact Kp/G-scale scored-event coverage",
              "expected at least $(MIN_GSCALE_SCORED_EVENTS) scored events, got $(scored_events)")
    end

    if minimum(Int.(df.g_level)) >= 3 && minimum(Float64.(df.peak_kp)) >= 7.0
        pass!(state, "exact Kp/G-scale scored-row scope", "all scored rows are G3+ / Kp >= 7 events")
    else
        fail!(state, "exact Kp/G-scale scored-row scope", "found scored rows below G3/Kp 7")
    end

    leads = sort(unique(Int.(df.lead)))
    if leads == EXPECTED_LEADS
        pass!(state, "exact Kp/G-scale replay leads", "leads=$(join(leads, ",")) h")
    else
        fail!(state, "exact Kp/G-scale replay leads", "expected $(join(EXPECTED_LEADS, ",")); got $(join(leads, ","))")
    end

    numeric_cols = [:obs, :v2_1, :v2_1_pre_rate_guard,
                    :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                    :v2_0, :v2_1_frozen,
                    :persistence, :peak_kp]
    finite_rows = finite_mask(df, numeric_cols)
    if all(finite_rows)
        pass!(state, "exact Kp/G-scale finite values", "all $(nrow(df)) rows finite for $(join(String.(numeric_cols), ", "))")
    else
        fail!(state, "exact Kp/G-scale finite values", "$(count(.!finite_rows)) rows contain non-finite scored values")
    end

    issue = parse_utc_datetime.(df.issue_utc)
    target = parse_utc_datetime.(df.target_utc)
    time_ok = true
    for i in 1:nrow(df)
        if ismissing(issue[i]) || ismissing(target[i]) ||
           target[i] != issue[i] + Hour(Int(df.lead[i]))
            time_ok = false
            break
        end
    end
    time_ok ? pass!(state, "exact Kp/G-scale locked-row timing", "target_utc equals issue_utc + lead for all rows") :
              fail!(state, "exact Kp/G-scale locked-row timing", "at least one target_utc does not equal issue_utc + lead")

    keys = _csv_key_parts(df.issue_utc, df.target_utc, df.lead)
    if length(unique(keys)) == nrow(df)
        pass!(state, "exact Kp/G-scale duplicate-row guard", "no duplicate issue/target/lead rows")
    else
        fail!(state, "exact Kp/G-scale duplicate-row guard", "duplicate issue/target/lead rows detected")
    end

    for lead in EXPECTED_LEADS
        sub = df[df.lead .== lead, :]
        if nrow(sub) == 0
            fail!(state, "exact G3+ lead $(lead)h coverage", "no scored rows")
            continue
        end
        rv20 = rmse(sub.v2_0, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        tail_effect = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen)))
        core_change = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0)))
        improvement = min(rv20, rpers) - rv21
        push!(state.gscale_metrics, ("all_G3plus", lead, nrow(sub), length(unique(Int.(sub.g_event_id))),
                                     rv20, rv21, rpers, improvement, tail_effect, core_change))
        if rv21 < rv20 && rv21 < rpers
            pass!(state, "exact G3+ lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f < historical V2.0 %.2f and persistence %.2f", rv21, rv20, rpers))
        else
            fail!(state, "exact G3+ lead $(lead)h replay skill",
                  @sprintf("V2.1 RMSE %.2f, historical V2.0 %.2f, persistence %.2f", rv21, rv20, rpers))
        end
        pass!(state, "exact G3+ lead $(lead)h operational-layer decomposition",
              @sprintf("max|V2.1−frozen-tail V2.1| %.2f nT; max|frozen-tail V2.1−V2.0| %.2f nT",
                       tail_effect, core_change))
    end

    for g in sort(unique(Int.(df.g_level)))
        subg = df[Int.(df.g_level) .== g, :]
        for lead in EXPECTED_LEADS
            sub = subg[subg.lead .== lead, :]
            nrow(sub) == 0 && continue
            rv20 = rmse(sub.v2_0, sub.obs)
            rv21 = rmse(sub.v2_1, sub.obs)
            rpers = rmse(sub.persistence, sub.obs)
            tail_effect = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen)))
            core_change = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0)))
            improvement = min(rv20, rpers) - rv21
            push!(state.gscale_metrics, ("G$(g)", lead, nrow(sub), length(unique(Int.(sub.g_event_id))),
                                         rv20, rv21, rpers, improvement, tail_effect, core_change))
        end
    end

    subgroup_bad = state.gscale_metrics[(state.gscale_metrics.cohort .!= "all_G3plus") .&
                                        (state.gscale_metrics.improvement_vs_best .< -1e-9), :]
    if nrow(subgroup_bad) == 0
        pass!(state, "exact Kp/G-scale subgroup skill", "V2 beats both baselines in every populated G3/G4/G5 subgroup lead")
    else
        sort!(subgroup_bad, :improvement_vs_best)
        worst = subgroup_bad[1, :]
        warn!(state, "exact Kp/G-scale subgroup skill",
              @sprintf("%d subgroup-lead cell(s) do not strictly beat the best baseline; worst %s %dh shortfall %.2f nT",
                       nrow(subgroup_bad), worst.cohort, worst.lead, -worst.improvement_vs_best))
    end

    summary_path = joinpath(LIVE_DIR, "v2_gscale_replay_summary.csv")
    summary = read_csv_checked!(state, summary_path, "exact Kp/G-scale replay summary")
    if summary !== nothing
        consistency = _gscale_summary_consistent(df, summary)
        consistency.ok ? pass!(state, "exact Kp/G-scale summary CRC", consistency.detail) :
                         fail!(state, "exact Kp/G-scale summary CRC", consistency.detail)
    end
end

function _noaa_kp_summary_consistent(df::DataFrame, summary::DataFrame)
    g3 = summary[(summary.scope .== "all") .& (summary.lead_band .== "all") .&
                 (Int.(summary.threshold_g) .== 3), :]
    nrow(g3) == 1 || return (ok = false, detail = "missing unique all/G3+ NOAA summary row")
    r = g3[1, :]
    obs = Int.(df.observed_g_level) .>= 3
    pred = Int.(df.forecast_g_level) .>= 3
    hits = count(obs .& pred)
    misses = count(obs .& .!pred)
    false_alarms = count(.!obs .& pred)
    rmse_kp = sqrt(mean(Float64.(df.kp_error) .^ 2))
    ok = Int(r.n_rows) == nrow(df) &&
         Int(r.hits) == hits &&
         Int(r.misses) == misses &&
         Int(r.false_alarms) == false_alarms &&
         abs(Float64(r.rmse_kp) - rmse_kp) <= 1e-9
    detail = @sprintf("rows=%d hits=%d misses=%d false_alarms=%d rmse_kp=%.3f",
                      nrow(df), hits, misses, false_alarms, rmse_kp)
    return (ok = ok, detail = detail)
end

function audit_noaa_kp_forecast_archive!(state::AuditState)
    path = joinpath(LIVE_DIR, "noaa_kp_forecast_scored.csv")
    df = read_csv_checked!(state, path, "NOAA 3-day Kp forecast scored CSV")
    df === nothing && return
    require_columns!(state, df, REQUIRED_NOAA_KP_COLS, "NOAA 3-day Kp forecast archive") || return

    if nrow(df) >= MIN_NOAA_KP_FORECAST_ROWS
        pass!(state, "NOAA Kp archive sample size", "n=$(nrow(df)) scored forecast rows")
    else
        fail!(state, "NOAA Kp archive sample size", "expected at least $(MIN_NOAA_KP_FORECAST_ROWS), got $(nrow(df))")
    end

    issue_count = length(unique(df.issue_utc))
    if issue_count >= MIN_NOAA_KP_ISSUES
        pass!(state, "NOAA Kp archive issue coverage", "issues=$(issue_count)")
    else
        fail!(state, "NOAA Kp archive issue coverage", "expected at least $(MIN_NOAA_KP_ISSUES) issues, got $(issue_count)")
    end

    numeric_cols = [:lead_h, :forecast_kp, :observed_kp, :kp_error]
    finite_rows = finite_mask(df, numeric_cols)
    all(finite_rows) ? pass!(state, "NOAA Kp archive finite values", "all rows finite for Kp score columns") :
                       fail!(state, "NOAA Kp archive finite values", "$(count(.!finite_rows)) rows contain non-finite score values")

    if all((Float64.(df.forecast_kp) .>= 0.0) .& (Float64.(df.forecast_kp) .<= 9.0)) &&
       all((Float64.(df.observed_kp) .>= 0.0) .& (Float64.(df.observed_kp) .<= 9.0))
        pass!(state, "NOAA Kp archive physical range", "forecast and observed Kp are inside [0, 9]")
    else
        fail!(state, "NOAA Kp archive physical range", "forecast or observed Kp outside [0, 9]")
    end

    issue = parse_utc_datetime.(df.issue_utc)
    target = parse_utc_datetime.(df.target_bin_start_utc)
    valid_times = .!ismissing.(issue) .& .!ismissing.(target)
    if all(valid_times) && all(target .> issue)
        pass!(state, "NOAA Kp archive causality", "all target bins start after issue time")
    else
        fail!(state, "NOAA Kp archive causality", "at least one target bin is not after issue time or has unparsable time")
    end

    summary_path = joinpath(LIVE_DIR, "noaa_kp_forecast_summary.csv")
    summary = read_csv_checked!(state, summary_path, "NOAA 3-day Kp forecast summary")
    summary === nothing && return
    consistency = _noaa_kp_summary_consistent(df, summary)
    consistency.ok ? pass!(state, "NOAA Kp archive summary CRC", consistency.detail) :
                     fail!(state, "NOAA Kp archive summary CRC", consistency.detail)

    g3 = summary[(summary.scope .== "all") .& (summary.lead_band .== "all") .&
                 (Int.(summary.threshold_g) .== 3), :]
    if nrow(g3) == 1
        r = g3[1, :]
        push!(state.noaa_kp_metrics, ("all", "all", 3, Int(r.n_rows),
                                      Int(r.hits), Int(r.misses),
                                      Int(r.false_alarms), Float64(r.pod),
                                      Float64(r.far), Float64(r.csi),
                                      Float64(r.rmse_kp)))
        pass!(state, "NOAA Kp archive external baseline boundary",
              @sprintf("G3+ POD %.3f, FAR %.3f, CSI %.3f; Kp/G-scale archive is not a Dst RMSE baseline",
                       Float64(r.pod), Float64(r.far), Float64(r.csi)))
    else
        fail!(state, "NOAA Kp archive external baseline boundary", "missing all/G3+ external forecast row")
    end
end

function _temerin_dst_summary_consistent(df::DataFrame, summary::DataFrame)
    max_error = 0.0
    targets = parse_utc_datetime.(df.target_utc)
    for r in eachrow(summary)
        scope = String(r.scope)
        sub = scope == "dscovr_real_time_input_era" ?
              df[coalesce.(targets .>= TEMERIN_DST_DSCOVR_START, false), :] : df
        sub = sub[Int.(sub.lead) .== Int(r.lead_h), :]
        nrow(sub) == Int(r.n_rows) || return (ok = false,
            detail = "row-count mismatch for $scope $(r.lead_h)h")
        rtem = rmse(sub.temerin_li_dst, sub.obs)
        rv21 = rmse(sub.v2_1, sub.obs)
        rv20 = rmse(sub.v2_0, sub.obs)
        rpers = rmse(sub.persistence, sub.obs)
        vals = [
            abs(rtem - Float64(r.rmse_temerin_valid_nt)),
            abs(rv21 - Float64(r.rmse_v2_1_nt)),
            abs(rv20 - Float64(r.rmse_v2_0_nt)),
            abs(rpers - Float64(r.rmse_persistence_nt)),
            abs((rv21 - rtem) - Float64(r.v2_1_minus_temerin_valid_rmse_nt)),
        ]
        max_error = max(max_error, maximum(vals))
    end
    return (ok = max_error <= 1e-9, detail = @sprintf("max summary recompute error %.3g", max_error))
end

function audit_temerin_dst_archive!(state::AuditState)
    path = joinpath(LIVE_DIR, "temerin_dst_archive_scored.csv")
    df = read_csv_checked!(state, path, "Temerin-Li Dst archive scored CSV")
    df === nothing && return
    require_columns!(state, df, REQUIRED_TEMERIN_DST_COLS, "Temerin-Li Dst archive") || return

    if nrow(df) >= MIN_TEMERIN_DST_ROWS
        pass!(state, "Temerin-Li Dst archive sample size", "n=$(nrow(df)) scored rows")
    else
        fail!(state, "Temerin-Li Dst archive sample size",
              "expected at least $(MIN_TEMERIN_DST_ROWS), got $(nrow(df))")
    end

    storm_count = length(unique(Int.(df.storm_id)))
    if storm_count >= MIN_TEMERIN_DST_STORMS
        pass!(state, "Temerin-Li Dst archive storm coverage", "storms=$(storm_count)")
    else
        fail!(state, "Temerin-Li Dst archive storm coverage",
              "expected at least $(MIN_TEMERIN_DST_STORMS) storms, got $(storm_count)")
    end

    numeric_cols = [:obs, :v2_1, :v2_0, :persistence, :temerin_li_dst,
                    :match_abs_gap_min]
    finite_rows = finite_mask(df, numeric_cols)
    all(finite_rows) ? pass!(state, "Temerin-Li Dst archive finite values", "all rows finite for score columns") :
                       fail!(state, "Temerin-Li Dst archive finite values", "$(count(.!finite_rows)) rows contain non-finite score values")

    issue = parse_utc_datetime.(df.issue_utc)
    target = parse_utc_datetime.(df.target_utc)
    valid = parse_utc_datetime.(df.temerin_valid_utc)
    time_ok = true
    for i in 1:nrow(df)
        if ismissing(issue[i]) || ismissing(target[i]) ||
           target[i] != issue[i] + Hour(Int(df.lead[i]))
            time_ok = false
            break
        end
    end
    time_ok ? pass!(state, "Temerin-Li Dst locked-row timing", "V2.1 target_utc equals issue_utc + lead for all scored rows") :
              fail!(state, "Temerin-Li Dst locked-row timing", "at least one V2.1 target_utc does not equal issue_utc + lead")

    if all(.!ismissing.(valid)) && maximum(Float64.(df.match_abs_gap_min)) <= TEMERIN_DST_MATCH_TOL_MIN + 1e-9
        pass!(state, "Temerin-Li Dst valid-time alignment",
              @sprintf("max nearest-valid-time gap %.2f min", maximum(Float64.(df.match_abs_gap_min))))
    else
        fail!(state, "Temerin-Li Dst valid-time alignment",
              "unparsable valid time or match gap exceeds $(TEMERIN_DST_MATCH_TOL_MIN) min")
    end

    if all(coalesce.(target .>= TEMERIN_DST_DEFAULT_START, false))
        pass!(state, "Temerin-Li Dst source-era scope", "all rows are in the LASP real-time-input archive era")
    else
        fail!(state, "Temerin-Li Dst source-era scope", "rows before 2013-05-01 are present")
    end

    summary_path = joinpath(LIVE_DIR, "temerin_dst_archive_summary.csv")
    summary = read_csv_checked!(state, summary_path, "Temerin-Li Dst archive summary")
    summary === nothing && return
    consistency = _temerin_dst_summary_consistent(df, summary)
    consistency.ok ? pass!(state, "Temerin-Li Dst archive summary CRC", consistency.detail) :
                     fail!(state, "Temerin-Li Dst archive summary CRC", consistency.detail)

    all_scope = summary[summary.scope .== "all_operational_input_era", :]
    if nrow(all_scope) > 0
        for r in eachrow(sort(all_scope, :lead_h))
            push!(state.temerin_dst_metrics, (
                String(r.scope), Int(r.lead_h), Int(r.n_rows), Int(r.n_storms),
                Float64(r.rmse_temerin_valid_nt), Float64(r.rmse_v2_1_nt),
                Float64(r.rmse_v2_0_nt), Float64(r.rmse_persistence_nt),
                Float64(r.v2_1_minus_temerin_valid_rmse_nt),
                Float64(r.max_match_gap_min),
            ))
        end
        worst = all_scope[argmax(Float64.(all_scope.v2_1_minus_temerin_valid_rmse_nt)), :]
        pass!(state, "Temerin-Li Dst external baseline boundary",
              @sprintf("same-unit valid-time archive context scored; V2.1-minus-Temerin worst %.2f nT at %dh; not a matched issue-time baseline",
                       Float64(worst.v2_1_minus_temerin_valid_rmse_nt), Int(worst.lead_h)))
    else
        fail!(state, "Temerin-Li Dst external baseline boundary", "missing all_operational_input_era summary rows")
    end
end

function _external_dst_summary_from_log(df::DataFrame)
    out = DataFrame(source = String[], n_rows = Int[], n_scored = Int[],
                    n_issues = Int[], max_lead_h = Float64[],
                    rmse_nt = Union{Missing, Float64}[], mae_nt = Union{Missing, Float64}[])
    for source in sort(unique(String.(df.source)))
        sub = df[String.(df.source) .== source, :]
        scored = .!ismissing.(sub.observed_dst_nt)
        if any(scored)
            err = Float64.(sub.forecast_dst_nt[scored]) .- Float64.(sub.observed_dst_nt[scored])
            rmse_val = sqrt(mean(err .^ 2))
            mae_val = mean(abs.(err))
        else
            rmse_val = missing
            mae_val = missing
        end
        push!(out, (source, nrow(sub), count(scored), length(unique(String.(sub.issue_utc))),
                    maximum(Float64.(sub.lead_h)), rmse_val, mae_val))
    end
    return out
end

function audit_external_dst_snapshots!(state::AuditState)
    path = EXTERNAL_DST_LOG_PATH
    df = read_csv_checked!(state, path, "prospective external Dst forecast snapshot log")
    df === nothing && return
    require_columns!(state, df, REQUIRED_EXTERNAL_DST_COLS, "prospective external Dst forecast snapshot log") || return

    if nrow(df) > 0
        pass!(state, "external Dst snapshot rows", "n=$(nrow(df)) future forecast rows")
    else
        fail!(state, "external Dst snapshot rows", "collector log is empty")
        return
    end

    sources = sort(unique(String.(df.source)))
    if length(sources) >= 2
        pass!(state, "external Dst source coverage", "sources=$(join(sources, ", "))")
    else
        warn!(state, "external Dst source coverage", "only $(length(sources)) source(s): $(join(sources, ", "))")
    end

    numeric_cols = [:lead_h, :forecast_dst_nt, :forecast_cadence_min]
    finite_rows = finite_mask(df, numeric_cols)
    all(finite_rows) ? pass!(state, "external Dst finite forecast values", "all rows finite for lead/cadence/forecast Dst") :
                       fail!(state, "external Dst finite forecast values", "$(count(.!finite_rows)) rows contain non-finite values")

    issue = parse_utc_datetime.(df.issue_utc)
    target = parse_utc_datetime.(df.target_utc)
    fetched = parse_utc_datetime.(df.fetched_utc)
    timing_ok = true
    lead_ok = true
    for i in 1:nrow(df)
        if ismissing(issue[i]) || ismissing(target[i]) || target[i] <= issue[i]
            timing_ok = false
            break
        end
        computed = Dates.value(target[i] - issue[i]) / 3_600_000
        if abs(computed - Float64(df.lead_h[i])) > 1e-6
            lead_ok = false
        end
    end
    timing_ok ? pass!(state, "external Dst issue-target causality", "all logged targets are after issue time") :
                fail!(state, "external Dst issue-target causality", "at least one target is not after issue time")
    lead_ok ? pass!(state, "external Dst lead CRC", "lead_h equals target_utc-issue_utc for all rows") :
              fail!(state, "external Dst lead CRC", "lead_h mismatch against issue/target timestamps")

    if all(.!ismissing.(fetched))
        pass!(state, "external Dst fetch timestamps", "all fetched_utc values parse")
    else
        fail!(state, "external Dst fetch timestamps", "at least one fetched_utc is unparsable")
    end

    if all(length.(String.(df.raw_sha256)) .== 64)
        pass!(state, "external Dst raw SHA-256", "all rows carry 64-character raw-response hashes")
    else
        fail!(state, "external Dst raw SHA-256", "missing or invalid raw-response hashes")
    end

    key = _csv_key_parts(df.source, df.issue_utc, df.target_utc, df.raw_sha256)
    if length(unique(key)) == nrow(df)
        pass!(state, "external Dst duplicate guard", "no duplicate source/issue/target/hash rows")
    else
        fail!(state, "external Dst duplicate guard", "duplicate source/issue/target/hash rows present")
    end

    basis = sort(unique(String.(df.issue_basis)))
    if any(==("fetch_time"), basis)
        warn!(state, "external Dst issue-time basis", "some rows fell back to fetch_time; bases=$(join(basis, ", "))")
    else
        pass!(state, "external Dst issue-time basis", "source issue bases=$(join(basis, ", "))")
    end

    max_lead = maximum(Float64.(df.lead_h))
    if max_lead >= 1.0
        pass!(state, "external Dst prospective lead coverage", @sprintf("max lead %.2f h", max_lead))
    else
        warn!(state, "external Dst prospective lead coverage",
              @sprintf("current public rows max lead %.2f h; collector active but not yet a 1--6 h baseline", max_lead))
    end

    scored = df[.!ismissing.(df.observed_dst_nt), :]
    if nrow(scored) == 0
        warn!(state, "external Dst scored rows", "no external snapshot rows have matured against observed Dst yet")
    else
        err_ok = true
        for r in eachrow(scored)
            err_ok &= abs(abs(Float64(r.forecast_dst_nt) - Float64(r.observed_dst_nt)) -
                          Float64(r.abs_error_nt)) <= 1e-9
        end
        err_ok ? pass!(state, "external Dst scored-row CRC", "checked $(nrow(scored)) scored rows") :
                 fail!(state, "external Dst scored-row CRC", "stored absolute errors disagree with forecast/observation values")
    end

    summary = _external_dst_summary_from_log(df)
    append!(state.external_dst_metrics, summary; cols = :union)
    report = EXTERNAL_DST_REPORT_PATH
    isfile(report) ? pass!(state, "external Dst report", "external_dst_forecast_report.md exists") :
                     warn!(state, "external Dst report", "external_dst_forecast_report.md missing")
end

function audit_live_log!(state::AuditState)
    path = LIVE_LOG_PATH
    df = read_csv_checked!(state, path, "live forecast log"; allow_zero=true)
    df === nothing && return
    require_columns!(state, df, REQUIRED_LIVE_COLS, "live forecast log") || return
    state.live_metrics[:live_log_rows] = nrow(df)
    state.live_metrics[:live_log_mtime] = mtime(path)

    if nrow(df) == 0
        pass!(state, "current V2.1 hot-log boundary",
              "zero-row schema initialized after byte-identical V2.0 archival")
        warn!(state, "current V2.1 live sample maturity",
              "no accumulated V2.1 issue has matured; current skill claims remain replay-only")
        return
    end

    if all(skipmissing(df.model_version) .== EXPECTED_MODEL_VERSION)
        pass!(state, "live model version", "all nonmissing model_version rows are $(EXPECTED_MODEL_VERSION)")
    else
        fail!(state, "live model version", "non-$(EXPECTED_MODEL_VERSION) model_version rows are present")
    end

    recent = newest_cycle_rows(df)
    recent_sub = collect(skipmissing(recent.sub_hourly_model_version))
    if !isempty(recent_sub) && all(recent_sub .== EXPECTED_SUBHOURLY)
        pass!(state, "newest V2 tail", "newest cycle $(length(recent_sub)) rows use $(EXPECTED_SUBHOURLY)")
    else
        fail!(state, "newest V2 tail",
              "expected recent sub_hourly_model_version=$(EXPECTED_SUBHOURLY), got $(join(unique(string.(recent_sub)), ", "))")
    end

    pending = df[ismissing.(df.observation_dst_nt), :]
    if nrow(pending) == 0
        pass!(state, "pending duplicate guard", "no pending rows")
    else
        counts = combine(groupby(pending, [:model_version, :latest_dst_time_utc, :target_time_utc]), nrow => :n)
        dup = counts[counts.n .> 1, :]
        if nrow(dup) == 0
            pass!(state, "pending duplicate guard", "pending rows=$(nrow(pending)), duplicate groups=0")
        else
            fail!(state, "pending duplicate guard", "duplicate pending groups=$(nrow(dup))")
        end
    end

    served_cols = [:observation_dst_nt, :served_pred_dst_nt, :served_pred_dst_ci05_nt,
                   :served_pred_dst_ci95_nt, :v2_pred_dst_nt, :persistence_dst_nt]
    served = df[finite_mask(df, served_cols), :]
    state.live_metrics[:served_n] = nrow(served)
    if nrow(served) >= 48
        pass!(state, "verified V2.1 live rows", "n=$(nrow(served)) rows with finite observations and V2.1 forecast")
    else
        warn!(state, "verified V2.1 live rows",
              "n=$(nrow(served)); V2.1 skill remains replay-only until at least 48 rows mature")
    end
    if nrow(served) < 120
        warn!(state, "live sample maturity", "verified V2 live rows=$(nrow(served)); keep treating live evidence as provisional")
    end

    if nrow(served) > 0
        pred = Float64.(served.served_pred_dst_nt)
        lo = Float64.(served.served_pred_dst_ci05_nt)
        hi = Float64.(served.served_pred_dst_ci95_nt)
        if all(lo .<= pred) && all(pred .<= hi) && all(hi .>= lo)
            pass!(state, "V2 interval geometry", "V2 center lies within [ci05, ci95] for all verified rows")
        else
            fail!(state, "V2 interval geometry", "at least one V2 interval is inverted or excludes its center")
        end

        stored_residual_mask = finite_mask(served, [:served_residual_dst_nt])
        missing_residual = nrow(served) - count(stored_residual_mask)
        if any(stored_residual_mask)
            computed = Float64.(served[stored_residual_mask, :observation_dst_nt]) .-
                       Float64.(served[stored_residual_mask, :served_pred_dst_nt])
            stored = Float64.(served[stored_residual_mask, :served_residual_dst_nt])
            maxerr = maximum(abs.(computed .- stored))
            if maxerr <= 1e-9
                pass!(state, "stored V2 residual consistency",
                      @sprintf("checked %d rows, max error %.3g", count(stored_residual_mask), maxerr))
            else
                fail!(state, "stored V2 residual consistency",
                      @sprintf("checked %d rows, max error %.6g", count(stored_residual_mask), maxerr))
            end
        end

        served_rmse = rmse(served.served_pred_dst_nt, served.observation_dst_nt)
        baseline_rmse = rmse(served.v2_pred_dst_nt, served.observation_dst_nt)
        persistence_rmse = rmse(served.persistence_dst_nt, served.observation_dst_nt)
        state.live_metrics[:served_rmse] = served_rmse
        state.live_metrics[:baseline_rmse] = baseline_rmse
        state.live_metrics[:persistence_rmse] = persistence_rmse
        if served_rmse <= baseline_rmse && served_rmse <= persistence_rmse
            pass!(state, "V2.1 live RMSE",
                  @sprintf("V2.1 %.2f <= V2.1 frozen-tail ablation %.2f and persistence %.2f", served_rmse, baseline_rmse, persistence_rmse))
        else
            warn!(state, "V2.1 live RMSE",
                  @sprintf("V2.1 %.2f, V2.1 frozen-tail ablation %.2f, persistence %.2f on current live sample", served_rmse, baseline_rmse, persistence_rmse))
        end

        obs = Float64.(served.observation_dst_nt)
        state.live_metrics[:obs_min] = minimum(obs)
        state.live_metrics[:obs_max] = maximum(obs)
        if minimum(obs) > -50.0
            warn!(state, "live storm coverage", @sprintf("current verified live sample has no Dst <= -50 nT storm rows: obs range %.1f to %.1f nT", minimum(obs), maximum(obs)))
        else
            pass!(state, "live storm coverage", @sprintf("verified live sample reaches %.1f nT", minimum(obs)))
        end

        inside = (lo .<= obs) .& (obs .<= hi)
        coverage = mean(Float64.(inside))
        state.live_metrics[:served_coverage] = coverage
        if 0.75 <= coverage <= 1.0
            pass!(state, "V2 interval hit rate", @sprintf("coverage %.3f over %d rows", coverage, nrow(served)))
        else
            warn!(state, "V2 interval hit rate", @sprintf("coverage %.3f over %d rows", coverage, nrow(served)))
        end

        stored_hit_mask = .!ismissing.(served.served_observed_in_90ci)
        missing_hit = nrow(served) - count(stored_hit_mask)
        if any(stored_hit_mask)
            stored_hits = Bool.(served[stored_hit_mask, :served_observed_in_90ci])
            if all(stored_hits .== inside[stored_hit_mask])
                pass!(state, "stored interval-hit consistency", "checked $(count(stored_hit_mask)) rows")
            else
                fail!(state, "stored interval-hit consistency", "stored interval-hit flags disagree with interval bounds")
            end
        end

        if missing_residual > 0 || missing_hit > 0
            warn!(state, "stored V2 score completeness",
                  "missing residuals=$(missing_residual), missing interval-hit flags=$(missing_hit); metrics recompute from primary columns")
        end
    end
end

function audit_historical_v2_0_live_log!(state::AuditState)
    df = read_csv_checked!(state, HISTORICAL_V2_0_LIVE_LOG_PATH,
                           "historical V2.0 live forecast log")
    df === nothing && return
    require_columns!(state, df, REQUIRED_LIVE_COLS,
                     "historical V2.0 live forecast log") || return

    versions = sort(unique(String.(collect(skipmissing(df.model_version)))))
    if versions == ["v2"]
        pass!(state, "historical live model boundary",
              "all $(length(collect(skipmissing(df.model_version)))) nonmissing versioned rows are V2.0 (`v2`)")
    else
        fail!(state, "historical live model boundary",
              "unexpected historical model versions: $(join(versions, ", "))")
    end
    served_versions = String.(collect(skipmissing(df.sub_hourly_model_version)))
    if all(!startswith(v, "v2.1") for v in served_versions)
        pass!(state, "historical live V2.1 exclusion",
              "no archived row is labeled as a V2.1 served pipeline")
    else
        fail!(state, "historical live V2.1 exclusion",
              "historical archive contains a V2.1 served label")
    end

    manifest = read_csv_checked!(state, HISTORICAL_V2_0_LIVE_MANIFEST_PATH,
                                 "historical V2.0 live manifest")
    if manifest !== nothing && nrow(manifest) == 1 &&
       all(c -> has_col(manifest, c), [:rows, :verified_rows, :sha256, :operational_version])
        digest = bytes2hex(sha256(read(HISTORICAL_V2_0_LIVE_LOG_PATH)))
        verified = count(!ismissing, df.observation_dst_nt)
        ok = Int(manifest.rows[1]) == nrow(df) &&
             Int(manifest.verified_rows[1]) == verified &&
             String(manifest.sha256[1]) == digest &&
             String(manifest.operational_version[1]) == "v2.0"
        ok ? pass!(state, "historical V2.0 live manifest CRC",
                   "rows=$(nrow(df)), verified=$(verified), sha256=$(digest)") :
             fail!(state, "historical V2.0 live manifest CRC",
                   "manifest values disagree with the archived CSV")
    else
        fail!(state, "historical V2.0 live manifest CRC",
              "manifest is missing or malformed")
    end

    served_cols = [:observation_dst_nt, :served_pred_dst_nt,
                   :served_pred_dst_ci05_nt, :served_pred_dst_ci95_nt,
                   :v2_pred_dst_nt, :persistence_dst_nt]
    served = df[finite_mask(df, served_cols), :]
    nrow(served) >= 1500 ?
        pass!(state, "historical V2.0 verified live rows", "n=$(nrow(served))") :
        fail!(state, "historical V2.0 verified live rows",
              "expected at least 1500 finite served rows, got $(nrow(served))")
    if nrow(served) > 0
        pred = Float64.(served.served_pred_dst_nt)
        obs = Float64.(served.observation_dst_nt)
        lo = Float64.(served.served_pred_dst_ci05_nt)
        hi = Float64.(served.served_pred_dst_ci95_nt)
        state.live_metrics[:historical_v2_0_n] = nrow(served)
        state.live_metrics[:historical_v2_0_rmse] = rmse(pred, obs)
        state.live_metrics[:historical_v2_0_persistence_rmse] =
            rmse(served.persistence_dst_nt, obs)
        state.live_metrics[:historical_v2_0_coverage] =
            mean(Float64.((lo .<= obs) .& (obs .<= hi)))
        all(lo .<= pred) && all(pred .<= hi) ?
            pass!(state, "historical V2.0 interval geometry",
                  "served center lies inside every finite archived interval") :
            fail!(state, "historical V2.0 interval geometry",
                  "an archived finite interval excludes its served center")
    end
end

function audit_v2_1_issue_identity!(state::AuditState)
    path = joinpath(LIVE_DIR, "v2_1_issue_identity.csv")
    df = read_csv_checked!(state, path, "V2.1 issue identity")
    df === nothing && return
    required = [:model_version, :served_model_version, :candidate_count,
                :active_count, :redundant_n_v2_present, :pressure_term_active,
                :pressure_coupling_active, :one_hour_inertia_weight,
                :state_inertia_h1_quiet_weight,
                :state_inertia_h1_deepening_weight,
                :state_inertia_h2_quiet_weight, :state_inertia_h3_quiet_weight,
                :state_inertia_quiet_dst_nt,
                :state_inertia_deepening_lo_nt_per_h,
                :state_inertia_deepening_hi_nt_per_h,
                :rapid_deepening_activation_rate_nt_per_h,
                :rapid_deepening_projection_factor,
                :rapid_deepening_extreme_rate_nt_per_h,
                :rapid_deepening_max_drop_nt,
                :rapid_deepening_extreme_max_drop_nt, :calibration_label,
                :coefficient_sha256, :ensemble_sha256, :draws_sha256,
                :calibration_sha256, :historical_v2_0_requires_explicit_version]
    require_columns!(state, df, required, "V2.1 issue identity") || return
    nrow(df) == 1 || return fail!(state, "V2.1 issue identity cardinality",
                                  "expected one row, got $(nrow(df))")
    r = df[1, :]
    structural = String(r.model_version) == EXPECTED_MODEL_VERSION &&
                 String(r.served_model_version) == EXPECTED_SUBHOURLY &&
                 Int(r.candidate_count) == 20 && Int(r.active_count) == 11 &&
                 !Bool(r.redundant_n_v2_present) && Bool(r.pressure_term_active) &&
                 Bool(r.pressure_coupling_active) &&
                 Float64(r.one_hour_inertia_weight) == 0.75 &&
                 Float64(r.state_inertia_h1_quiet_weight) == 0.75 &&
                 Float64(r.state_inertia_h1_deepening_weight) == 0.0 &&
                 Float64(r.state_inertia_h2_quiet_weight) == 0.625 &&
                 Float64(r.state_inertia_h3_quiet_weight) == 0.875 &&
                 Float64(r.state_inertia_quiet_dst_nt) == -30.0 &&
                 Float64(r.state_inertia_deepening_lo_nt_per_h) == -15.0 &&
                 Float64(r.state_inertia_deepening_hi_nt_per_h) == -5.0 &&
                 Float64(r.rapid_deepening_activation_rate_nt_per_h) == -15.0 &&
                 Float64(r.rapid_deepening_projection_factor) == 0.375 &&
                 Float64(r.rapid_deepening_extreme_rate_nt_per_h) == -60.0 &&
                 Float64(r.rapid_deepening_max_drop_nt) == 50.0 &&
                 Float64(r.rapid_deepening_extreme_max_drop_nt) == 120.0 &&
                 Bool(r.historical_v2_0_requires_explicit_version) &&
                 startswith(String(r.calibration_label), "operational_v2_1_")
    structural ? pass!(state, "V2.1 issue identity contract",
                       "20 candidates, 11 active terms, current calibration and current served-tail label") :
                 fail!(state, "V2.1 issue identity contract",
                       "identity row disagrees with the deployed V2.1 contract")

    paths = (
        coefficient_sha256=joinpath(OPERATIONAL_PACKAGE_ROOT, "data",
                                    "real_sindy_discovery_coefficients.csv"),
        ensemble_sha256=joinpath(OPERATIONAL_PACKAGE_ROOT, "data",
                                 "real_ensemble_inclusion.csv"),
        draws_sha256=joinpath(OPERATIONAL_PACKAGE_ROOT, "data",
                              "real_sindy_ensemble_draws.csv"),
        calibration_sha256=joinpath(OPERATIONAL_PACKAGE_ROOT, "deploy",
                                    "operational_v2_calibration.csv"),
    )
    hash_ok = all(begin
        p = getproperty(paths, field)
        isfile(p) && String(getproperty(r, field)) == bytes2hex(sha256(read(p)))
    end for field in propertynames(paths))
    hash_ok ? pass!(state, "V2.1 issue artifact hashes",
                    "identity hashes match all current core/calibration artifacts") :
              fail!(state, "V2.1 issue artifact hashes",
                    "identity hash differs from a current core/calibration artifact")
end

function refresh_live_log_metrics_for_dashboard!(state::AuditState)
    path = LIVE_LOG_PATH
    old_mtime = get(state.live_metrics, :live_log_mtime, missing)
    try
        df = CSV.read(path, DataFrame)
        served_cols = [:observation_dst_nt, :served_pred_dst_nt, :served_pred_dst_ci05_nt,
                       :served_pred_dst_ci95_nt, :v2_pred_dst_nt, :persistence_dst_nt]
        all(c -> has_col(df, c), served_cols) || return (ok = false, changed = false, detail = "missing V2 product columns")
        served = df[finite_mask(df, served_cols), :]
        nrow(served) > 0 || return (ok = false, changed = false, detail = "no finite V2 product rows")

        state.live_metrics[:served_n] = nrow(served)
        state.live_metrics[:served_rmse] = rmse(served.served_pred_dst_nt, served.observation_dst_nt)
        state.live_metrics[:baseline_rmse] = rmse(served.v2_pred_dst_nt, served.observation_dst_nt)
        state.live_metrics[:persistence_rmse] = rmse(served.persistence_dst_nt, served.observation_dst_nt)

        obs = Float64.(served.observation_dst_nt)
        lo = Float64.(served.served_pred_dst_ci05_nt)
        hi = Float64.(served.served_pred_dst_ci95_nt)
        state.live_metrics[:served_coverage] = mean(Float64.((lo .<= obs) .& (obs .<= hi)))
        state.live_metrics[:obs_min] = minimum(obs)
        state.live_metrics[:obs_max] = maximum(obs)

        new_mtime = mtime(path)
        state.live_metrics[:live_log_rows] = nrow(df)
        state.live_metrics[:live_log_mtime] = new_mtime
        changed = !ismissing(old_mtime) && Float64(old_mtime) != Float64(new_mtime)
        return (ok = true, changed = changed,
                detail = "rows=$(nrow(df)), served=$(nrow(served))")
    catch err
        return (ok = false, changed = false, detail = sprint(showerror, err))
    end
end

function source_files(path::AbstractString)
    if isfile(path)
        return [path]
    elseif isdir(path)
        out = String[]
        for (root, dirs, files) in walkdir(path)
            filter!(d -> !startswith(d, "."), dirs)
            for file in files
                ext = lowercase(splitext(file)[2])
                ext in (".jl", ".js", ".css", ".html", ".md", ".toml", ".json") || continue
                push!(out, joinpath(root, file))
            end
        end
        return out
    else
        return String[]
    end
end

function ekf_hits(paths::Vector{String})
    hits = String[]
    for path in paths
        for file in source_files(path)
            for (line_no, line) in enumerate(eachline(file))
                occursin(r"\b[Ee][Kk][Ff]\b", line) || continue
                push!(hits, "$(relpath(file, REPO_ROOT)):$(line_no)")
            end
        end
    end
    return hits
end

function audit_retired_methods!(state::AuditState)
    hits = ekf_hits(ACTIVE_PRODUCT_PATHS)

    if isempty(hits)
        pass!(state, "retired EKF product isolation", "no EKF references in active V2 dashboard/live paths")
    else
        fail!(state, "retired EKF product isolation", "EKF references found in active product paths: $(join(hits, ", "))")
    end

    if isfile(EKF_DECISION) && filesize(EKF_DECISION) > 0
        text = read(EKF_DECISION, String)
        if occursin("NOT PROMOTABLE", text) && occursin("Retire adaptive-EKF-on-SINDy", text)
            pass!(state, "retired EKF decision record", "EKF_V3_DECISION.md records NOT PROMOTABLE and retirement decision")
        else
            warn!(state, "retired EKF decision record", "EKF_V3_DECISION.md exists but does not contain both required decision phrases")
        end
    else
        warn!(state, "retired EKF decision record", "EKF_V3_DECISION.md missing; archived failed-method rationale is unavailable")
    end
end

function audit_dashboard_api!(state::AuditState, api_url::Union{Nothing, String};
                              require_api::Bool = false,
                              require_fresh::Bool = false,
                              max_issue_age_hours::Float64 = DEFAULT_MAX_ISSUE_AGE_HOURS)
    if api_url === nothing
        warn!(state, "dashboard API check", "skipped; rerun with --api-url=$(DEFAULT_API_URL) to verify V2 dashboard state")
        return
    end

    payload = nothing
    try
        resp = HTTP.get(api_url; connect_timeout = 2, readtimeout = 5, status_exception = false)
        if resp.status != 200
            msg = "HTTP status $(resp.status) from $api_url"
            require_api ? fail!(state, "dashboard API reachable", msg) : warn!(state, "dashboard API reachable", msg)
            return
        end
        payload = JSON3.read(String(resp.body), Dict{String, Any})
    catch err
        msg = "$(api_url) unavailable: $(sprint(showerror, err))"
        require_api ? fail!(state, "dashboard API reachable", msg) : warn!(state, "dashboard API reachable", msg)
        return
    end

    pass!(state, "dashboard API reachable", api_url)
    refresh = refresh_live_log_metrics_for_dashboard!(state)
    if refresh.ok && refresh.changed
        warn!(state, "dashboard comparison snapshot",
              "live log changed during audit; refreshed dashboard comparison metrics ($(refresh.detail))")
    elseif !refresh.ok
        warn!(state, "dashboard comparison snapshot",
              "could not refresh live-log metrics before API comparison: $(refresh.detail)")
    end
    audit_dashboard_payload!(state, payload, api_url;
                             require_fresh = require_fresh,
                             max_issue_age_hours = max_issue_age_hours)
end

function audit_dashboard_payload!(state::AuditState, payload, api_url::AbstractString;
                                  require_fresh::Bool = false,
                                  max_issue_age_hours::Float64 = DEFAULT_MAX_ISSUE_AGE_HOURS,
                                  now_utc::DateTime = now(UTC))
    available = nested_get(payload, ["available"], missing)
    if available === true
        pass!(state, "dashboard API availability", "available=true")
    else
        fail!(state, "dashboard API availability", "available=$(available)")
    end

    model = string(nested_get(payload, ["model_version"], ""))
    if model == EXPECTED_MODEL_VERSION
        pass!(state, "dashboard API model", "model_version=$(EXPECTED_MODEL_VERSION)")
    else
        fail!(state, "dashboard API model", "model_version=$(model)")
    end

    driver = string(nested_get(payload, ["lead_time", "driver_assumption"], ""))
    if occursin("Ballistically propagated L1 forcing", driver) &&
       occursin("regime-aware relaxation", driver) &&
       occursin("causal rate projection", driver) &&
       occursin("one-hour", driver) &&
       occursin("state-conditioned inertia", driver) &&
       occursin("extreme-Dst inertia guard", driver)
        pass!(state, "dashboard API V2-tail assumption", driver)
    else
        fail!(state, "dashboard API V2-tail assumption", "unexpected driver_assumption=$(driver)")
    end

    api_n = nested_get(payload, ["calibration", "v2_n_verified"], missing)
    expected_n = get(state.live_metrics, :served_n, missing)
    if !ismissing(api_n) && !ismissing(expected_n) && Int(api_n) == Int(expected_n)
        pass!(state, "dashboard API V2-row count", "v2_n_verified=$(api_n)")
    else
        fail!(state, "dashboard API V2-row count", "api=$(api_n), log=$(expected_n)")
    end

    api_rmse = float_or_missing(nested_get(payload, ["calibration", "v2_rmse_nt"], missing))
    log_rmse = get(state.live_metrics, :served_rmse, missing)
    zero_verified = !ismissing(expected_n) && Int(expected_n) == 0
    if zero_verified && ismissing(api_rmse) && ismissing(log_rmse)
        pass!(state, "dashboard API V2 RMSE",
              "undefined in both API and log before any V2.1 row matures")
    elseif !zero_verified && !ismissing(api_rmse) && !ismissing(log_rmse) && abs(api_rmse - round(Float64(log_rmse); digits = 2)) <= 0.015
        pass!(state, "dashboard API V2 RMSE", @sprintf("api %.2f matches log %.2f", api_rmse, log_rmse))
    else
        fail!(state, "dashboard API V2 RMSE", "api=$(api_rmse), log=$(log_rmse)")
    end

    api_baseline_rmse = float_or_missing(nested_get(payload, ["calibration", "audit_baseline_rmse_nt"], missing))
    log_baseline_rmse = get(state.live_metrics, :baseline_rmse, missing)
    if zero_verified && ismissing(api_baseline_rmse) && ismissing(log_baseline_rmse)
        pass!(state, "dashboard API audit-baseline RMSE",
              "undefined in both API and log before any V2.1 row matures")
    elseif !zero_verified && !ismissing(api_baseline_rmse) && !ismissing(log_baseline_rmse) && abs(api_baseline_rmse - round(Float64(log_baseline_rmse); digits = 2)) <= 0.015
        pass!(state, "dashboard API audit-baseline RMSE", @sprintf("api %.2f matches log %.2f", api_baseline_rmse, log_baseline_rmse))
    else
        fail!(state, "dashboard API audit-baseline RMSE", "api=$(api_baseline_rmse), log=$(log_baseline_rmse)")
    end

    generated = parse_utc_datetime(nested_get(payload, ["generated_utc"], missing))
    if generated === missing
        fail!(state, "dashboard API generated timestamp", "generated_utc is missing or unparsable")
    else
        generated_age_min = age_minutes(generated, now_utc)
        state.live_metrics[:dashboard_generated_age_min] = generated_age_min
        if -1.0 <= generated_age_min <= DEFAULT_MAX_API_GENERATED_AGE_MIN
            pass!(state, "dashboard API generated freshness",
                  @sprintf("generated_utc age %.2f min", generated_age_min))
        else
            fail!(state, "dashboard API generated freshness",
                  @sprintf("generated_utc age %.2f min exceeds %.1f min or is in the future",
                           generated_age_min, DEFAULT_MAX_API_GENERATED_AGE_MIN))
        end
    end

    issue = parse_utc_datetime(nested_get(payload, ["forecast_issue_utc"], missing))
    if issue === missing
        fail!(state, "dashboard forecast issue timestamp", "forecast_issue_utc is missing or unparsable")
    else
        issue_age_h = age_hours(issue, now_utc)
        state.live_metrics[:dashboard_forecast_issue_age_hours] = issue_age_h
        if -1 / 60 <= issue_age_h <= max_issue_age_hours
            pass!(state, "dashboard forecast issue freshness",
                  @sprintf("forecast_issue_utc age %.2f h", issue_age_h))
        else
            detail = @sprintf("forecast_issue_utc age %.2f h exceeds %.2f h or is in the future",
                              issue_age_h, max_issue_age_hours)
            require_fresh ? fail!(state, "dashboard forecast issue freshness", detail) :
                            warn!(state, "dashboard forecast issue freshness", detail)
        end
    end

    sw = parse_utc_datetime(nested_get(payload, ["latest_solar_wind_utc"], missing))
    if sw === missing
        fail!(state, "dashboard solar-wind timestamp", "latest_solar_wind_utc is missing or unparsable")
    else
        sw_age_h = age_hours(sw, now_utc)
        state.live_metrics[:dashboard_solar_wind_age_hours] = sw_age_h
        max_sw_age_h = max_issue_age_hours + 1.0
        if -1 / 60 <= sw_age_h <= max_sw_age_h
            pass!(state, "dashboard solar-wind freshness",
                  @sprintf("latest_solar_wind_utc age %.2f h", sw_age_h))
        else
            detail = @sprintf("latest_solar_wind_utc age %.2f h exceeds %.2f h or is in the future",
                              sw_age_h, max_sw_age_h)
            require_fresh ? fail!(state, "dashboard solar-wind freshness", detail) :
                            warn!(state, "dashboard solar-wind freshness", detail)
        end
    end

    state.live_metrics[:dashboard_api_url] = api_url
    state.live_metrics[:dashboard_model_version] = model
    state.live_metrics[:dashboard_driver_assumption] = driver
end

function selftest_readiness_audit()
    passed = 0
    fixed_now = DateTime(2026, 6, 26, 7, 15, 0)

    good = Dict{String, Any}(
        "available" => true,
        "model_version" => EXPECTED_MODEL_VERSION,
        "generated_utc" => "2026-06-26T07:14:30.123456Z",
        "forecast_issue_utc" => "2026-06-26T06:30:00Z",
        "latest_solar_wind_utc" => "2026-06-26T06:28:00Z",
        "lead_time" => Dict{String, Any}(
            "driver_assumption" => "Ballistically propagated L1 forcing, then regime-aware relaxation beyond the measured L1 window, followed by a causal rate projection, validation-selected one-hour and state-conditioned inertia blends, and an extreme-Dst inertia guard",
        ),
        "calibration" => Dict{String, Any}(
            "v2_n_verified" => 3,
            "v2_rmse_nt" => 1.23,
            "audit_baseline_rmse_nt" => 2.35,
        ),
    )
    state = AuditState()
    state.live_metrics[:served_n] = 3
    state.live_metrics[:served_rmse] = 1.234
    state.live_metrics[:baseline_rmse] = 2.345
    audit_dashboard_payload!(state, good, "selftest://good";
                             require_fresh = true,
                             max_issue_age_hours = 3.0,
                             now_utc = fixed_now)
    @assert count(c -> c.level == :fail, state.checks) == 0 "good dashboard payload should not fail"
    passed += 1

    zero = deepcopy(good)
    zero["calibration"]["v2_n_verified"] = 0
    zero["calibration"]["v2_rmse_nt"] = nothing
    zero["calibration"]["audit_baseline_rmse_nt"] = nothing
    zero_state = AuditState()
    zero_state.live_metrics[:served_n] = 0
    audit_dashboard_payload!(zero_state, zero, "selftest://zero";
                             require_fresh = true,
                             max_issue_age_hours = 3.0,
                             now_utc = fixed_now)
    @assert count(c -> c.level == :fail, zero_state.checks) == 0 "zero-row dashboard payload should accept undefined sample metrics"
    passed += 1

    fabricated_zero = deepcopy(zero)
    fabricated_zero["calibration"]["v2_rmse_nt"] = 0.0
    fabricated_state = AuditState()
    fabricated_state.live_metrics[:served_n] = 0
    audit_dashboard_payload!(fabricated_state, fabricated_zero, "selftest://fabricated-zero";
                             require_fresh = true,
                             max_issue_age_hours = 3.0,
                             now_utc = fixed_now)
    @assert any(c -> c.level == :fail && c.name == "dashboard API V2 RMSE", fabricated_state.checks) "zero-row dashboard payload must reject fabricated RMSE"
    passed += 1

    bad = deepcopy(good)
    bad["model_version"] = "ekf"
    bad["calibration"]["v2_n_verified"] = 4
    state_bad = AuditState()
    state_bad.live_metrics[:served_n] = 3
    state_bad.live_metrics[:served_rmse] = 1.234
    state_bad.live_metrics[:baseline_rmse] = 2.345
    audit_dashboard_payload!(state_bad, bad, "selftest://bad";
                             require_fresh = true,
                             max_issue_age_hours = 3.0,
                             now_utc = fixed_now)
    @assert count(c -> c.level == :fail, state_bad.checks) >= 2 "bad dashboard payload should fail model and row-count checks"
    passed += 1

    stale = deepcopy(good)
    stale["forecast_issue_utc"] = "2026-06-25T23:30:00Z"
    stale_state = AuditState()
    stale_state.live_metrics[:served_n] = 3
    stale_state.live_metrics[:served_rmse] = 1.234
    stale_state.live_metrics[:baseline_rmse] = 2.345
    audit_dashboard_payload!(stale_state, stale, "selftest://stale";
                             require_fresh = true,
                             max_issue_age_hours = 3.0,
                             now_utc = fixed_now)
    @assert any(c -> c.level == :fail && c.name == "dashboard forecast issue freshness", stale_state.checks) "strict freshness should fail stale forecast issues"
    passed += 1

    df = DataFrame(a = [1.0, missing, NaN], b = [2.0, 3.0, 4.0])
    @assert finite_mask(df, [:a, :b]) == [true, false, false] "finite_mask should reject missing and NaN values"
    passed += 1

    regime_df = DataFrame(
        lead = [1, 1, 1, 1],
        obs = [-210.0, -220.0, -35.0, -40.0],
        v2_0 = [-212.0, -222.0, -36.0, -41.0],
        v2_1 = [-200.0, -199.0, -34.0, -39.0],
        persistence = [-209.0, -221.0, -35.0, -40.0],
        rate = [-20.0, -18.0, 0.0, 1.0],
    )
    regime_state = AuditState()
    audit_replay_regimes!(regime_state, regime_df;
                          min_n = 2,
                          material_delta_nt = 5.0,
                          strict_persistence = true,
                          require_full_coverage = false)
    @assert any(c -> c.level == :fail && c.name == "regime historical V2.0 guard", regime_state.checks) "regime guard should fail historical-comparator regressions"
    @assert any(c -> c.level == :fail && c.name == "regime persistence vulnerability", regime_state.checks) "strict regime guard should fail material persistence losses"
    passed += 1

    split_sha = repeat("a", 64)
    split_fixture = DataFrame(
        split=["fit", "validation", "holdout"],
        rows=[100, 40, 40],
        anchors=[25, 10, 10],
        minimum_issue_utc=[fixed_now - Day(10), fixed_now - Day(6), fixed_now - Day(2)],
        maximum_issue_utc=[fixed_now - Day(8), fixed_now - Day(4), fixed_now],
        minimum_target_utc=[fixed_now - Day(10) + Hour(1), fixed_now - Day(6) + Hour(1), fixed_now - Day(2) + Hour(1)],
        maximum_target_utc=[fixed_now - Day(7), fixed_now - Day(3), fixed_now + Hour(1)],
        point_calibration_sha256=fill(split_sha, 3),
        source_table_sha256=fill(repeat("b", 64), 3),
        conformal_holdout_coverage=fill(0.90, 3),
    )
    @assert all(values(_v2_1_split_contract(split_fixture, split_sha))) "causal split fixture should pass"
    leaky_split = copy(split_fixture)
    leaky_split.minimum_issue_utc[2] = split_fixture.maximum_target_utc[1] - Hour(1)
    leaky_contract = _v2_1_split_contract(leaky_split, split_sha)
    @assert leaky_contract.schema && leaky_contract.partitions && !leaky_contract.causal "forecast-origin overlap mutation should fail only the causal boundary"
    passed += 1

    holdout_summary = DataFrame(
        cohort=["overall", "lead_1", "lead_2", "lead_3", "lead_6", "quiet", "storm"],
        lead_h=[0, 1, 2, 3, 6, 0, 0],
        activity_regime=["all", "all", "all", "all", "all", "quiet", "storm"],
        n_rows=[40, 10, 10, 10, 10, 32, 8],
        served_hits=[36, 9, 9, 9, 9, 30, 6],
        served_coverage=[0.9, 0.9, 0.9, 0.9, 0.9, 30 / 32, 6 / 8],
        served_rmse_nt=fill(2.0, 7),
        frozen_tail_hits=[35, 9, 9, 9, 8, 29, 6],
        frozen_tail_coverage=[35 / 40, 0.9, 0.9, 0.9, 0.8, 29 / 32, 6 / 8],
        frozen_tail_rmse_nt=fill(2.1, 7),
        nominal_coverage=fill(V2_1_NOMINAL_COVERAGE, 7),
        promotion_coverage_floor=fill(V2_1_CALIBRATION_COVERAGE_FLOOR, 7),
        pooled_gate_applies=[true, false, false, false, false, false, false],
        pooled_gate_pass=[true, false, false, false, false, false, false],
    )
    holdout_audit = DataFrame(
        model_version=[EXPECTED_MODEL_VERSION], candidate_count=[20], active_count=[11],
        holdout_rows=[40], holdout_anchors=[10],
        validation_max_target_utc=[fixed_now - Hour(2)],
        holdout_min_issue_utc=[fixed_now - Hour(1)],
        holdout_max_issue_utc=[fixed_now + Hour(8)],
        holdout_max_target_utc=[fixed_now + Hour(14)],
        strict_forecast_origin_separation=[true],
        interval_policy=["static_conformal_shifted_to_complete_hour_served_center"],
        holdout_residual_updates=[0],
        point_calibration_sha256=[repeat("a", 64)],
        conformal_calibration_sha256=[repeat("b", 64)],
        split_audit_sha256=[repeat("c", 64)],
        calibration_scored_sha256=[repeat("e", 64)],
        omni_sha256=[repeat("d", 64)],
        maximum_frozen_tail_continuity_error_nt=[0.0],
        maximum_interval_center_error_nt=[1e-14],
        nominal_coverage=[V2_1_NOMINAL_COVERAGE],
        pooled_promotion_floor=[V2_1_CALIBRATION_COVERAGE_FLOOR],
        served_pooled_coverage=[0.9], served_pooled_gate_pass=[true],
        frozen_tail_pooled_coverage=[35 / 40], heldout_promotion_evidence=[true],
    )
    holdout_contract = _v2_1_served_holdout_contract(
        holdout_summary, holdout_audit;
        point_sha256=repeat("a", 64), conformal_sha256=repeat("b", 64),
        split_sha256=repeat("c", 64), omni_sha256=repeat("d", 64),
    )
    @assert all(values(holdout_contract)) "complete-hour served-stack holdout fixture should pass"
    mutated_summary = copy(holdout_summary)
    mutated_summary.served_hits[2] -= 1
    mutated_contract = _v2_1_served_holdout_contract(
        mutated_summary, holdout_audit;
        point_sha256=repeat("a", 64), conformal_sha256=repeat("b", 64),
        split_sha256=repeat("c", 64), omni_sha256=repeat("d", 64),
    )
    @assert !mutated_contract.summary_crc "lead-hit mutation should fail summary recomputation"
    passed += 1

    mktempdir() do dir
        clean = joinpath(dir, "clean.jl")
        dirty = joinpath(dir, "dirty.jl")
        write(clean, "model = :v2\n")
        write(dirty, "method = :EKF\n")
        @assert isempty(ekf_hits([clean])) "clean file should not produce EKF hits"
        @assert length(ekf_hits([dirty])) == 1 "dirty file should produce one EKF hit"
    end
    passed += 1

    pass_state = AuditState()
    pass!(pass_state, "x", "ok")
    @assert verdict(pass_state) == "PASS"
    warn_state = AuditState()
    warn!(warn_state, "x", "warn")
    @assert verdict(warn_state) == "PASS WITH WARNINGS"
    fail_state = AuditState()
    fail!(fail_state, "x", "fail")
    @assert verdict(fail_state) == "FAIL"
    passed += 1

    println("readiness audit self-test PASS: $(passed) independent checks")
    return true
end

function audit_paper_gate!(state::AuditState)
    contract = joinpath(REPO_ROOT, "VENUE_CONTRACT.md")
    if isfile(contract) && filesize(contract) > 0
        pass!(state, "venue contract", "VENUE_CONTRACT.md exists")
    else
        warn!(state, "venue contract", "VENUE_CONTRACT.md is missing; formal paper submission readiness remains blocked")
        push!(state.paper_notes, "Formal paper-readiness audits cannot be certified until a target journal/article contract is created.")
    end

    declarations = target_declarations()
    targets = sort(unique(values(declarations)))
    if length(targets) <= 1
        pass!(state, "target-journal consistency", isempty(targets) ? "no target declarations found" : "single target declaration: $(only(targets))")
    else
        parts = [string(k, " -> ", v) for (k, v) in sort(collect(declarations); by = x -> x[1])]
        warn!(state, "target-journal consistency", "conflicting target declarations: $(join(parts, "; "))")
        push!(state.paper_notes, "Repository paper artifacts still mention multiple targets; user approval is needed before changing the contract or retargeting manuscript-facing text.")
    end
end

function known_target_from_line(line::AbstractString)
    occursin("Journal of Atmospheric and Solar-Terrestrial Physics", line) && return "Journal of Atmospheric and Solar-Terrestrial Physics"
    occursin("Advances in Space Research", line) && return "Advances in Space Research"
    occursin("Communications Physics", line) && return "Communications Physics"
    return nothing
end

function first_nonempty_after_header(lines::Vector{String}, header::AbstractString)
    idx = findfirst(line -> strip(line) == header, lines)
    idx === nothing && return nothing
    for j in (idx + 1):length(lines)
        line = strip(lines[j])
        isempty(line) && continue
        return line
    end
    return nothing
end

function target_declarations()
    out = Dict{String, String}()

    rp = joinpath(REPO_ROOT, "RESEARCH_PLAN.md")
    if isfile(rp)
        line = first_nonempty_after_header(readlines(rp), "## Target Journal")
        target = line === nothing ? nothing : known_target_from_line(line)
        target !== nothing && (out["RESEARCH_PLAN.md"] = target)
    end

    results = joinpath(REPO_ROOT, "RESULTS_PLAN.md")
    if isfile(results)
        line = first_nonempty_after_header(readlines(results), "## Target Journal")
        target = line === nothing ? nothing : known_target_from_line(line)
        target !== nothing && (out["RESULTS_PLAN.md"] = target)
    end

    maintex = joinpath(REPO_ROOT, "paper_v2_monitor", "main.tex")
    if isfile(maintex)
        text = read(maintex, String)
        m = match(r"\\journal\{([^}]*)\}", text)
        if m !== nothing
            target = known_target_from_line(m.captures[1])
            target !== nothing && (out["paper_v2_monitor/main.tex"] = target)
        end
    end

    review = joinpath(REPO_ROOT, "PAPER_REVIEW_REPORT.md")
    if isfile(review)
        for line in Iterators.take(readlines(review), 30)
            if occursin("Target Journal", line)
                target = known_target_from_line(line)
                target !== nothing && (out["PAPER_REVIEW_REPORT.md"] = target)
                break
            end
        end
    end

    return out
end

function markdown_escape(s)
    return replace(string(s), "|" => "\\|", "\n" => " ")
end

function verdict(state::AuditState)
    nfail = count(c -> c.level == :fail, state.checks)
    nwarn = count(c -> c.level == :warn, state.checks)
    nfail > 0 && return "FAIL"
    nwarn > 0 && return "PASS WITH WARNINGS"
    return "PASS"
end

function write_report(state::AuditState, path::AbstractString)
    open(path, "w") do io
        stamp = Dates.format(now(), DateFormat("yyyy-mm-dd HH:MM:SS"))
        println(io, "# Operational V2.1 Readiness Audit\n")
        println(io, "**Verdict:** $(verdict(state))")
        println(io, "**Generated:** $(stamp) local time\n")
        println(io, "This audit recomputes Operational V2.1 readiness from complete-hour causal replay of the served stack on the chronological holdout, the locked live log, retrospective severe-storm replay artifacts, exact Kp/G-scale replay, broad Dst-intense archive replay, external NOAA Kp forecast archive check, Temerin--Li Dst archive valid-time comparison, and the prospective external Dst issue-time snapshot collector. V2.1 denotes the revised 20-candidate/11-active-term SINDy core; V2.0 denotes the archived 21-candidate/10-active-term comparator. The holdout does not reconstruct fractional subhourly live windows. This audit is an engineering and research guard, not a venue-submission certificate.\n")

        npass = count(c -> c.level == :pass, state.checks)
        nwarn = count(c -> c.level == :warn, state.checks)
        nfail = count(c -> c.level == :fail, state.checks)
        println(io, "## Summary\n")
        println(io, "| PASS | WARN | FAIL |")
        println(io, "|---:|---:|---:|")
        println(io, "| $npass | $nwarn | $nfail |\n")

        println(io, "## Checks\n")
        println(io, "| Level | Check | Detail |")
        println(io, "|---|---|---|")
        for c in state.checks
            println(io, "| $(uppercase(String(c.level))) | $(markdown_escape(c.name)) | $(markdown_escape(c.detail)) |")
        end

        if haskey(state.live_metrics, :served_holdout_summary)
            summary = state.live_metrics[:served_holdout_summary]
            println(io, "\n## Complete-Hour Served-Stack V2.1 Chronological Holdout\n")
            println(io, "The pooled row is the declared static-interval promotion gate. Lead-specific and quiet/storm rows disclose where coverage departs from the pooled result and the 0.90 nominal target.\n")
            println(io, "| Cohort | Lead [h] | n | Served RMSE [nT] | Served coverage | Frozen-tail coverage |")
            println(io, "|---|---:|---:|---:|---:|---:|")
            for r in eachrow(summary)
                @printf(io, "| %s | %d | %d | %.3f | %.3f | %.3f |\n",
                        r.cohort, r.lead_h, r.n_rows, r.served_rmse_nt,
                        r.served_coverage, r.frozen_tail_coverage)
            end
        end

        if nrow(state.replay_metrics) > 0
            println(io, "\n## Replay Metrics\n")
            println(io, "| Lead [h] | n | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | Improve vs best comparator | Max operational-layer effect | Max 20/11-core effect |")
            println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in eachrow(state.replay_metrics)
                @printf(io, "| %d | %d | %.2f | %.2f | %.2f | %+.2f | %.2f | %.2f |\n",
                        r.lead, r.n, r.rmse_v2_0, r.rmse_v2_1, r.rmse_persistence,
                        r.improvement_vs_best, r.max_tail_effect, r.max_core_change)
            end
        end

        if nrow(state.broad_metrics) > 0
            println(io, "\n## Broad Dst-Intense Replay Metrics\n")
            println(io, "Broad replay covers catalog storms with minimum pressure-corrected Dst* <= -100 nT. It is Dst*-threshold evidence, not exact NOAA G-scale classification.\n")
            println(io, "| Lead [h] | n | storms | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | Improve vs best comparator | Max operational-layer effect | Max 20/11-core effect |")
            println(io, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in eachrow(state.broad_metrics)
                @printf(io, "| %d | %d | %d | %.2f | %.2f | %.2f | %+.2f | %.2f | %.2f |\n",
                        r.lead, r.n, r.n_storms, r.rmse_v2_0, r.rmse_v2_1,
                        r.rmse_persistence, r.improvement_vs_best,
                        r.max_tail_effect, r.max_core_change)
            end
        end

        if nrow(state.gscale_metrics) > 0
            println(io, "\n## Exact Kp/G-Scale Replay Metrics\n")
            println(io, "Exact replay selects GFZ three-hour Kp events with Kp >= 7 (NOAA G3+) and scores V2.1 on the same locked issue/target rows as historical V2.0 and persistence. Skipped catalog events are data-coverage skips from unavailable finite OMNI/Dst rows.\n")
            println(io, "| Cohort | Lead [h] | n | events | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | Improve vs best comparator | Max operational-layer effect | Max 20/11-core effect |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            sort!(state.gscale_metrics, [:cohort, :lead])
            for r in eachrow(state.gscale_metrics)
                @printf(io, "| %s | %d | %d | %d | %.2f | %.2f | %.2f | %+.2f | %.2f | %.2f |\n",
                        r.cohort, r.lead, r.n, r.n_events, r.rmse_v2_0,
                        r.rmse_v2_1, r.rmse_persistence, r.improvement_vs_best,
                        r.max_tail_effect, r.max_core_change)
            end
        end

        if nrow(state.noaa_kp_metrics) > 0
            println(io, "\n## External NOAA Kp Forecast Archive Metrics\n")
            println(io, "The NOAA archive check scores official 3-day Kp/G-scale forecast bins against GFZ Kp observations. It is an external operational context check, not a same-unit Dst RMSE comparator for V2.\n")
            println(io, "| Scope | Lead band | Threshold | rows | hits | misses | false alarms | POD | FAR | CSI | Kp RMSE |")
            println(io, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in eachrow(state.noaa_kp_metrics)
                @printf(io, "| %s | %s | G%d+ | %d | %d | %d | %d | %.3f | %.3f | %.3f | %.3f |\n",
                        r.scope, r.lead_band, r.threshold_g, r.n_rows, r.hits,
                        r.misses, r.false_alarms, r.pod, r.far, r.csi, r.rmse_kp)
            end
        end

        if nrow(state.temerin_dst_metrics) > 0
            println(io, "\n## External Temerin-Li Dst Archive Metrics\n")
            println(io, "The Temerin-Li archive check scores same-unit predicted Dst values at archived valid times against the V2.1 target rows. The monthly archive does not expose issue-time/lead rows, so this is a valid-time operational context comparison, not a matched 1--6 h promotion baseline.\n")
            println(io, "| Scope | Lead [h] | n | storms | RMSE Temerin-Li valid-time | RMSE V2.1 | RMSE historical V2.0 | RMSE persistence | V2.1 minus Temerin-Li | Max gap [min] |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in eachrow(state.temerin_dst_metrics)
                @printf(io, "| %s | %d | %d | %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %.2f |\n",
                        r.scope, r.lead, r.n, r.n_storms, r.rmse_temerin_valid,
                        r.rmse_v2_1, r.rmse_v2_0, r.rmse_persistence,
                        r.v2_1_minus_temerin, r.max_gap_min)
            end
        end

        if nrow(state.external_dst_metrics) > 0
            println(io, "\n## Prospective External Dst Snapshot Metrics\n")
            println(io, "The prospective collector stores public same-unit Dst products as issue-time snapshots with raw-response hashes. It starts the missing issue-time archive going forward; it does not backfill unavailable historical issue snapshots.\n")
            println(io, "| Source | rows | scored | issues | max lead [h] | RMSE [nT] | MAE [nT] |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|")
            for r in eachrow(state.external_dst_metrics)
                rmse_s = ismissing(r.rmse_nt) ? "pending" : @sprintf("%.2f", r.rmse_nt)
                mae_s = ismissing(r.mae_nt) ? "pending" : @sprintf("%.2f", r.mae_nt)
                @printf(io, "| %s | %d | %d | %d | %.3f | %s | %s |\n",
                        r.source, r.n_rows, r.n_scored, r.n_issues,
                        r.max_lead_h, rmse_s, mae_s)
            end
        end

        if nrow(state.regime_metrics) > 0
            println(io, "\n## Regime Scorecard\n")
            println(io, "Rows are replay cells with at least $(REGIME_MIN_ROWS) examples. `Delta vs best` is positive when V2.1 is worse than the stronger of historical V2.0 and persistence.\n")
            println(io, "| Axis | Lead [h] | Regime | n | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | Delta vs V2.0 | Delta vs best |")
            println(io, "|---|---:|---|---:|---:|---:|---:|---:|---:|")
            sort!(state.regime_metrics, [:axis, :lead, :regime])
            for r in eachrow(state.regime_metrics)
                @printf(io, "| %s | %d | %s | %d | %.2f | %.2f | %.2f | %+.2f | %+.2f |\n",
                        r.axis, r.lead, r.regime, r.n, r.rmse_v2_0, r.rmse_v2_1,
                        r.rmse_persistence, r.delta_vs_v2_0, r.delta_vs_best)
            end
        end

        if haskey(state.live_metrics, :served_n)
            println(io, "\n## Current V2.1 Live Log Metrics\n")
            println(io, "| Metric | Value |")
            println(io, "|---|---:|")
            println(io, "| Verified V2.1 rows | $(state.live_metrics[:served_n]) |")
            labels = Dict(
                :served_rmse => "V2.1 RMSE",
                :baseline_rmse => "V2.1 frozen-tail ablation RMSE",
                :persistence_rmse => "persistence RMSE",
                :served_coverage => "V2 90% coverage",
                :obs_min => "minimum observed Dst",
                :obs_max => "maximum observed Dst",
                :deep_subset_n => "deep-deepening replay rows",
                :deep_v2_bias => "V2 deep-deepening signed error",
                :deep_baseline_bias => "historical V2.0 deep-deepening signed error",
                :broad_deep_subset_n => "broad deep-deepening replay rows",
                :broad_deep_v2_bias => "broad V2 deep-deepening signed error",
                :broad_deep_baseline_bias => "broad historical V2.0 deep-deepening signed error",
            )
            for key in (:served_rmse, :baseline_rmse, :persistence_rmse, :served_coverage,
                        :obs_min, :obs_max, :deep_subset_n, :deep_v2_bias, :deep_baseline_bias,
                        :broad_deep_subset_n, :broad_deep_v2_bias, :broad_deep_baseline_bias)
                haskey(state.live_metrics, key) || continue
                val = state.live_metrics[key]
                label = get(labels, key, String(key))
                if val isa Integer
                    println(io, "| $label | $val |")
                else
                    @printf(io, "| %s | %.3f |\n", label, Float64(val))
                end
            end
        end

        if haskey(state.live_metrics, :historical_v2_0_n)
            println(io, "\n## Historical V2.0 Live Log Metrics\n")
            println(io, "These archived rows predate the V2.1 migration and are retained only as historical operational evidence; they do not measure V2.1 skill.\n")
            println(io, "| Metric | Value |")
            println(io, "|---|---:|")
            println(io, "| Verified historical V2.0 rows | $(state.live_metrics[:historical_v2_0_n]) |")
            @printf(io, "| Historical V2.0 RMSE [nT] | %.3f |\n",
                    state.live_metrics[:historical_v2_0_rmse])
            @printf(io, "| Historical persistence RMSE [nT] | %.3f |\n",
                    state.live_metrics[:historical_v2_0_persistence_rmse])
            @printf(io, "| Historical V2.0 90%% interval coverage | %.3f |\n",
                    state.live_metrics[:historical_v2_0_coverage])
        end

        if haskey(state.live_metrics, :dashboard_api_url)
            println(io, "\n## Dashboard API Metrics\n")
            println(io, "| Metric | Value |")
            println(io, "|---|---|")
            println(io, "| API URL | $(markdown_escape(state.live_metrics[:dashboard_api_url])) |")
            println(io, "| Model version | $(markdown_escape(state.live_metrics[:dashboard_model_version])) |")
            println(io, "| Driver assumption | $(markdown_escape(state.live_metrics[:dashboard_driver_assumption])) |")
            for key in (:dashboard_generated_age_min, :dashboard_forecast_issue_age_hours,
                        :dashboard_solar_wind_age_hours)
                haskey(state.live_metrics, key) || continue
                unit = key == :dashboard_generated_age_min ? "min" : "h"
                @printf(io, "| %s | %.3f %s |\n", String(key), Float64(state.live_metrics[key]), unit)
            end
        end

        if !isempty(state.paper_notes)
            println(io, "\n## Paper Readiness Notes\n")
            for note in state.paper_notes
                println(io, "- $note")
            end
        end

        println(io, "\n## CRC Interpretation\n")
        println(io, "- Correct: fail on schema drift, non-finite replay values, complete-hour served-stack holdout identity or pooled-gate failure, lost frozen-tail continuity, replay regression, V2 regression inside populated regimes, duplicate pending live rows, dashboard/API mismatch, stale API generation, and retired-method product leakage.")
        println(io, "- Robust: preserve warnings for the current quiet-live persistence edge, limited live sample size, missing Dst <= -50 nT live storm coverage, older score-field backfill, and unresolved venue/target-journal state.")
        println(io, "- Complete: combine the complete-hour served-stack chronological holdout, retrospective storm replay, broad Dst-intense replay, exact Kp/G-scale replay, external NOAA Kp forecast archive context, Temerin-Li valid-time Dst archive context, prospective issue-time external Dst snapshot collection, target-Dst/rate regime scorecards, locked-live log checks, dashboard/API freshness checks, retired-EKF isolation, and paper-readiness caveats in one repeatable audit.")
    end
end

function parse_args(args)
    write = false
    report = DEFAULT_REPORT
    api_url = DEFAULT_API_URL
    require_api = false
    require_fresh = false
    strict_regime_persistence = false
    max_issue_age_hours = DEFAULT_MAX_ISSUE_AGE_HOURS
    self_test = false
    for arg in args
        if arg == "--write-report"
            write = true
        elseif arg == "--require-api"
            require_api = true
        elseif arg == "--require-fresh"
            require_fresh = true
        elseif arg == "--strict-regime-persistence"
            strict_regime_persistence = true
        elseif arg == "--no-api"
            api_url = nothing
        elseif arg == "--self-test"
            self_test = true
        elseif startswith(arg, "--report=")
            report = abspath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--api-url=")
            api_url = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--max-issue-age-hours=")
            max_issue_age_hours = parse(Float64, split(arg, "=", limit = 2)[2])
            max_issue_age_hours > 0 || error("--max-issue-age-hours must be positive")
        elseif arg in ("-h", "--help")
            println("Usage: julia --project=. validation/operational/v2_readiness_audit.jl [--write-report] [--report=PATH] [--api-url=URL|--no-api] [--require-api] [--require-fresh] [--strict-regime-persistence] [--max-issue-age-hours=N] [--self-test]")
            exit(0)
        else
            error("unknown argument: $arg")
        end
    end
    return write, report, api_url, require_api, require_fresh, strict_regime_persistence, max_issue_age_hours, self_test
end

function main(args = ARGS)
    write, report, api_url, require_api, require_fresh, strict_regime_persistence, max_issue_age_hours, self_test = parse_args(args)
    self_test && return selftest_readiness_audit()

    state = AuditState()
    audit_replay!(state; strict_regime_persistence = strict_regime_persistence)
    audit_broad_replay!(state)
    audit_gscale_replay!(state)
    audit_noaa_kp_forecast_archive!(state)
    audit_temerin_dst_archive!(state)
    audit_external_dst_snapshots!(state)
    audit_v2_1_calibration_split!(state)
    audit_v2_1_served_holdout!(state)
    audit_v2_1_issue_identity!(state)
    audit_historical_v2_0_live_log!(state)
    audit_live_log!(state)
    audit_dashboard_api!(state, api_url;
                         require_api = require_api,
                         require_fresh = require_fresh,
                         max_issue_age_hours = max_issue_age_hours)
    audit_retired_methods!(state)
    audit_paper_gate!(state)

    for c in state.checks
        println(rpad(uppercase(String(c.level)), 5), " ", c.name, " - ", c.detail)
    end
    println("Summary: PASS=", count(c -> c.level == :pass, state.checks),
            " WARN=", count(c -> c.level == :warn, state.checks),
            " FAIL=", count(c -> c.level == :fail, state.checks),
            " verdict=", verdict(state))

    if write
        write_report(state, report)
        println("Wrote ", relpath(report, REPO_ROOT))
    end

    return count(c -> c.level == :fail, state.checks) == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = main()
    exit(ok ? 0 : 1)
end
