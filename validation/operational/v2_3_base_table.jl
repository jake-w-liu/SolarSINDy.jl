#!/usr/bin/env julia

# Operational V2.3 base table (2010--2025).
#
# One row per (issue time, model step) of the causal V2.1 calibration replay,
# carrying the served V2.1 center, the frozen-tail V2.1 center, the raw SINDy
# tail center, the causal baselines, the static V2.2 stack center, the
# issue-time drivers, the storm-phase columns the V2.2 regime needs, and every
# V2.1 calibration feature. The served/frozen reconstruction reuses the
# unchanged `_v2_forecast` path of the V2.2 development replay
# (`_v22_hourly_served_centers`), so the 2010--2022 rows must reproduce the
# archived V2.2 development table exactly; that identity is the base-table
# oracle (`--self-test`, `test/test_v2_3_base_table.jl`).
#
# Companion artifacts:
#   v2_3_base_table.csv     one row per (issue, step)
#   v2_3_hourly_frame.csv   causal forward-filled hourly driver frame + Dst
#   v2_3_base_manifest.csv  partition/block row counts, input/output SHA-256,
#                           identity-oracle residual, generation seconds
#
# Run from the package root:
#   julia --project=. validation/operational/v2_3_base_table.jl
#   julia --project=. validation/operational/v2_3_base_table.jl --rebuild
#   julia --project=. validation/operational/v2_3_base_table.jl --self-test

using SolarSINDy
using CSV
using DataFrames
using Dates
using Printf
using SHA
using Statistics

# `_v2_forecast`, `_driver_lookup_range`, `_transit_hours`, `_ffill!`, `OMNI`,
# `OPERATIONAL_OUTPUT_DIR` and the live-forecast-verify replay helpers.
include(joinpath(@__DIR__, "v2_replay.jl"))

# `v2_1_calibration.jl` re-includes `examples/live_forecast_verify.jl`, which has
# no include guard. Loading it into its own module keeps the replay chain above
# byte-identical instead of redefining its types and constants.
module V23CalibrationSource
    include(joinpath(@__DIR__, "v2_1_calibration.jl"))
end

const V23_BASE_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_3_base")
const V23_BASE_TABLE = joinpath(V23_BASE_DIR, "v2_3_base_table.csv")
const V23_BASE_HOURLY_FRAME = joinpath(V23_BASE_DIR, "v2_3_hourly_frame.csv")
const V23_BASE_MANIFEST = joinpath(V23_BASE_DIR, "v2_3_base_manifest.csv")

const V23_BASE_YEAR_START = 2010
const V23_BASE_YEAR_END = 2025
const V23_BASE_HORIZONS = copy(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS)
const V23_BASE_MAX_STEP_HOURS = maximum(V23_BASE_HORIZONS)
const V23_BASE_EMBARGO_HOURS = 168

# Plan section 3: disjoint storm-aware partitions with a 168 h target embargo.
const V23_DEV_FIRST_ISSUE = DateTime(2010, 1, 1, 1)
const V23_DEV_LAST_ISSUE = DateTime(2019, 12, 24, 16)
const V23_DEV_LAST_TARGET = DateTime(2019, 12, 31, 23)
const V23_TEST_FIRST_ISSUE = DateTime(2020, 1, 8, 0)
const V23_TEST_LAST_ISSUE = DateTime(2025, 12, 24, 16)
const V23_TEST_LAST_TARGET = DateTime(2025, 12, 31, 23)
const V23_CV_BLOCK_LABELS = (2013, 2014, 2015, 2016, 2017, 2018)
const V23_NO_CV_BLOCK = 0

function _v23_env_path(name::AbstractString, fallback::AbstractString)
    override = strip(get(ENV, String(name), ""))
    return isempty(override) ? abspath(fallback) : abspath(override)
end

const _V23_V22_DEVELOPMENT_DIR = joinpath(
    OPERATIONAL_WORKSPACE_ROOT, "2026_045_v2_2_package", "validation", "output",
    "operational", "v2_2_development",
)
"Frozen static V2.2 stack (weights fit on 2010--2017), the plan section 5 comparator."
const V23_V22_STACK_PATH = _v23_env_path(
    "SOLARSINDY_V23_V22_STACK",
    joinpath(_V23_V22_DEVELOPMENT_DIR, "operational_v2_2_stack.csv"),
)
"Archived V2.2 development replay table: the 2010--2022 identity oracle."
const V23_V22_ORACLE_TABLE = _v23_env_path(
    "SOLARSINDY_V23_V22_ORACLE",
    joinpath(_V23_V22_DEVELOPMENT_DIR, "v2_2_development_replay.csv"),
)
const V23_ORACLE_LAST_YEAR = 2022
const V23_ORACLE_ATOL_NT = 1e-9
const V23_ORACLE_EXPECTED_ROWS = 679_237
const V23_ORACLE_LAST_TARGET = DateTime(V23_ORACLE_LAST_YEAR, 12, 31, 23)
# The archived V2.2 development replay is a strict subset of the rebuilt
# 2010--2022 rows for two reasons, both verified against the source code:
#   * it was generated with `year_end = 2022`, so the 23 rows whose target falls
#     in 2023-01 are absent;
#   * it inherits the purged V2.1 calibration split, whose `_embargo_splits`
#     drops every anchor issued within the previous split's longest-horizon
#     target window (7 anchors x 6 model steps at each of the two boundaries,
#     2017-10-20T16..22 and 2020-05-22T20..2020-05-23T02).
# 23 + 42 + 42 = 107 rebuilt rows therefore have no archived counterpart; every
# archived row must still be reproduced exactly.
const V23_ORACLE_EXPECTED_EXTRA_ROWS = 107

"Columns the rebuilt table must reproduce from the archived V2.2 development replay."
const V23_ORACLE_SHARED_COLUMNS = [
    :served_v2_1_dst_nt, :frozen_v2_1_dst_nt, :raw_sindy_dst_nt,
    :persistence_dst_nt, :burton_dst_nt, :burton_full_dst_nt, :obrien_dst_nt,
    :observation_dst_nt, :latest_dst_nt, :dst_delta_1h_nt, :coupling_active_mvm,
]

"Explicit, non-calibration columns of the base table, in written order."
const V23_BASE_LEADING_COLUMNS = [
    :issue_time_utc, :target_time_utc, :model_step_hours, :partition, :cv_block,
    :served_v2_1_dst_nt, :frozen_v2_1_dst_nt, :raw_sindy_dst_nt,
    :persistence_dst_nt, :burton_dst_nt, :burton_full_dst_nt, :obrien_dst_nt,
    :static_v2_2_dst_nt, :observation_dst_nt, :latest_dst_nt, :dst_delta_1h_nt,
    :coupling_active_mvm, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa,
]

_v23_file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

# ---------------------------------------------------------------------------
# Partition labels (plan section 3)
# ---------------------------------------------------------------------------

"""
    v2_3_cv_block(issue) -> Int

Inner rolling-origin block label of a DEV issue: calendar years 2013--2017 are
their own blocks, 2018 and 2019 share block `2018`, and every earlier DEV issue
is archive-only (`0`).
"""
function v2_3_cv_block(issue::DateTime)
    y = year(issue)
    2013 <= y <= 2017 && return y
    (y == 2018 || y == 2019) && return 2018
    return V23_NO_CV_BLOCK
end

"""
    v2_3_partition(issue) -> (partition, cv_block)

Classify an issue time into `"DEV"`, `"TEST"` or `"embargo"` using the issue
time and its maximum target `issue + 7 h`. Only DEV issues carry a nonzero
cross-validation block label.
"""
function v2_3_partition(issue::DateTime)
    max_target = issue + Hour(V23_BASE_MAX_STEP_HOURS)
    if V23_DEV_FIRST_ISSUE <= issue <= V23_DEV_LAST_ISSUE &&
       max_target <= V23_DEV_LAST_TARGET
        return ("DEV", v2_3_cv_block(issue))
    elseif V23_TEST_FIRST_ISSUE <= issue <= V23_TEST_LAST_ISSUE &&
           max_target <= V23_TEST_LAST_TARGET
        return ("TEST", V23_NO_CV_BLOCK)
    end
    return ("embargo", V23_NO_CV_BLOCK)
end

"""
    v2_3_block_windows(table) -> DataFrame

Per inner block: the first DEV issue of the block and the training-archive
cutoff `first issue − 168 h`. The block-`b` training archive is every DEV row
whose maximum target does not exceed that cutoff.
"""
function v2_3_block_windows(table::DataFrame)
    rows = NamedTuple[]
    dev = table[table.partition .== "DEV", :]
    for block in V23_CV_BLOCK_LABELS
        block_rows = dev[dev.cv_block .== block, :]
        if nrow(block_rows) == 0
            push!(rows, (
                cv_block=block, block_rows=0, first_issue_utc=DateTime(0),
                training_max_target_utc=DateTime(0), training_rows=0,
                training_issues=0,
            ))
            continue
        end
        first_issue = minimum(block_rows.issue_time_utc)
        cutoff = first_issue - Hour(V23_BASE_EMBARGO_HOURS)
        eligible = (dev.issue_time_utc .+ Hour(V23_BASE_MAX_STEP_HOURS)) .<= cutoff
        push!(rows, (
            cv_block=block,
            block_rows=nrow(block_rows),
            first_issue_utc=first_issue,
            training_max_target_utc=cutoff,
            training_rows=count(eligible),
            training_issues=length(unique(dev.issue_time_utc[eligible])),
        ))
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Hourly driver frame
# ---------------------------------------------------------------------------

"""
    v2_3_hourly_frame(cleaned) -> DataFrame

Build the causal hourly frame `(time_utc, V, Bz, By, n, Pdyn, Dst)` from an OMNI
frame already cleaned with `clean_omni_data!(; causal=true)`. `V`, `Bz`, `By`
and `n` are carried forward across gaps exactly as in `_driver_lookup_range`,
`Pdyn` is recomputed from the filled density and speed so the proton-only
pressure identity holds, and `Dst` is never filled: a missing observation stays
`NaN` so it can neither anchor nor score a forecast.
"""
function v2_3_hourly_frame(cleaned::DataFrame)
    required = (:datetime, :V, :Bz, :By, :n, :Dst)
    missing_columns = [String(c) for c in required if !(String(c) in names(cleaned))]
    isempty(missing_columns) || throw(ArgumentError(
        "cleaned OMNI frame omits: $(join(missing_columns, ", "))",
    ))
    times = DateTime.(cleaned.datetime)
    issorted(times) || throw(ArgumentError(
        "hourly OMNI frame must be chronological before causal forward fill",
    ))
    allunique(times) || throw(ArgumentError("hourly OMNI frame has duplicate timestamps"))
    speed = _ffill!(Float64.(coalesce.(cleaned.V, NaN)))
    bz = _ffill!(Float64.(coalesce.(cleaned.Bz, NaN)))
    by = _ffill!(Float64.(coalesce.(cleaned.By, NaN)))
    density = _ffill!(Float64.(coalesce.(cleaned.n, NaN)))
    pdyn = [dynamic_pressure(density[k], speed[k]) for k in eachindex(speed)]
    dst = Float64.(coalesce.(cleaned.Dst, NaN))
    return DataFrame(
        time_utc=times, V=speed, Bz=bz, By=by, n=density, Pdyn=pdyn, Dst=dst,
    )
end

function v2_3_hourly_frame(year_start::Int, year_end::Int;
                           omni_path::AbstractString=OMNI)
    df = parse_omni2(String(omni_path); year_start=year_start, year_end=year_end)
    clean_omni_data!(df; causal=true)
    return v2_3_hourly_frame(df)
end

"""
    v2_3_frame_driver_lookup(frame) -> Dict{DateTime,NamedTuple}

Issue-time driver lookup built from the hourly frame. A record enters the lookup
only when every driver channel is finite, matching `_driver_lookup_range`.
"""
function v2_3_frame_driver_lookup(frame::DataFrame)
    lookup = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    for i in 1:nrow(frame)
        driver = (
            V=frame.V[i], Bz=frame.Bz[i], By=frame.By[i],
            n=frame.n[i], Pdyn=frame.Pdyn[i],
        )
        all(isfinite, values(driver)) || continue
        lookup[frame.time_utc[i]] = driver
    end
    return lookup
end

# ---------------------------------------------------------------------------
# Static V2.2 stack column
# ---------------------------------------------------------------------------

"""
    v2_3_static_v22_center(stack, model_step_hours, latest_dst_nt,
                           dst_delta_1h_nt, coupling_active_mvm, centers) -> Float64

Static V2.2 stack center for one row. The regime is chosen from issue-time state
only (latest Dst, its one-hour rate, the gated coupling), and `centers` holds the
six V2.2 components of that row.
"""
function v2_3_static_v22_center(stack::OperationalV22Stack,
                                model_step_hours::Integer,
                                latest_dst_nt::Real,
                                dst_delta_1h_nt::Real,
                                coupling_active_mvm::Real,
                                centers)
    return operational_v22_predict(
        stack, model_step_hours, latest_dst_nt, dst_delta_1h_nt,
        coupling_active_mvm, centers,
    ).pred_dst
end

# ---------------------------------------------------------------------------
# Base table
# ---------------------------------------------------------------------------

"""
    build_v2_3_base_table(; year_start, year_end, horizons, stack_path)

Build the 2010--2025 base table and the hourly driver frame. Returns
`(; table, frame, stats)`; `stats` carries the identity residual of the frozen
reconstruction against the deployed V2.1 scored center and the wall-clock cost
of each stage.
"""
function build_v2_3_base_table(; year_start::Int=V23_BASE_YEAR_START,
                               year_end::Int=V23_BASE_YEAR_END,
                               horizons::Vector{Int}=copy(V23_BASE_HORIZONS),
                               stack_path::AbstractString=V23_V22_STACK_PATH)
    started = time()
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    calibration = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
    calibration.selected_component == :v2 || error(
        "V2.3 base table requires the deployed corrected-SINDy V2.1 component",
    )
    stack = read_operational_v22_stack(String(stack_path))
    all(in(stack.supported_model_steps), horizons) || error(
        "static V2.2 stack does not support every model step",
    )

    frame_started = time()
    frame = v2_3_hourly_frame(year_start, year_end)
    lookup = v2_3_frame_driver_lookup(frame)
    lookup == _driver_lookup_range(year_start, year_end) || error(
        "hourly frame driver lookup differs from the replay driver lookup",
    )
    frame_seconds = time() - frame_started

    table_started = time()
    # The uncalibrated replay frame is released as soon as it has been scored;
    # only the scored frame is needed for the rest of the build.
    scored = let raw_table = V23CalibrationSource.build_v2_1_calibration_table(
            OMNI; year_start=year_start, year_end=year_end, horizons=horizons,
        )
        score_operational_v2(raw_table, calibration)
    end
    calibration_seconds = time() - table_started

    replay_started = time()
    n = nrow(scored)
    raw_sindy = Vector{Float64}(undef, n)
    served = Vector{Float64}(undef, n)
    frozen = Float64.(scored.v2_pred_dst_nt)
    static_v22 = Vector{Float64}(undef, n)
    partitions = Vector{String}(undef, n)
    blocks = Vector{Int}(undef, n)
    identity_residual = 0.0
    partition_cache = Dict{DateTime,Tuple{String,Int}}()
    for i in 1:n
        row = scored[i, :]
        issue = DateTime(row.issue_time_utc)
        step = Int(row.model_step_hours)
        drivers = (
            V=Float64(row.V_kms), Bz=Float64(row.Bz_nt), By=Float64(row.By_nt),
            n=Float64(row.n_cm3), Pdyn=Float64(row.Pdyn_npa),
        )
        latest = Float64(row.latest_dst_nt)
        anchor = pressure_correct_dst([latest], [drivers.Pdyn])[1]
        rate = Float64(row.dst_delta_1h_nt)
        future = k -> get(lookup, issue + Hour(k - 1), nothing)
        raw_i, served_i = _v2_forecast(
            core.library, core.coefficients, anchor, drivers, future, latest,
            calibration, step, rate; calibration_features=row,
        )
        _, frozen_check = _v2_forecast(
            core.library, core.coefficients, anchor, drivers, future, latest,
            calibration, step, rate; force_frozen=true, calibration_features=row,
        )
        residual = abs(frozen_check - frozen[i])
        residual <= V23_ORACLE_ATOL_NT || error(
            "frozen-tail reconstruction diverges from the deployed V2.1 center " *
            "at issue=$issue step=$step: $frozen_check vs $(frozen[i])",
        )
        identity_residual = max(identity_residual, residual)
        all(isfinite, (raw_i, served_i)) || error(
            "non-finite served reconstruction at issue=$issue, step=$step",
        )
        raw_sindy[i] = raw_i
        served[i] = served_i
        coupling = Float64(row.coupling_active_mvm)
        static_v22[i] = v2_3_static_v22_center(
            stack, step, latest, rate, coupling,
            (
                served_v2_1=served_i,
                frozen_v2_1=frozen[i],
                persistence=Float64(row.persistence_dst_nt),
                burton=Float64(row.burton_dst_nt),
                burton_full=Float64(row.burton_full_dst_nt),
                obrien=Float64(row.obrien_dst_nt),
            ),
        )
        label = get!(partition_cache, issue) do
            v2_3_partition(issue)
        end
        partitions[i] = label[1]
        blocks[i] = label[2]
    end
    replay_seconds = time() - replay_started

    table = DataFrame(
        issue_time_utc=DateTime.(scored.issue_time_utc),
        target_time_utc=DateTime.(scored.target_time_utc),
        model_step_hours=Int.(scored.model_step_hours),
        partition=partitions,
        cv_block=blocks,
        served_v2_1_dst_nt=served,
        frozen_v2_1_dst_nt=frozen,
        raw_sindy_dst_nt=raw_sindy,
        persistence_dst_nt=Float64.(scored.persistence_dst_nt),
        burton_dst_nt=Float64.(scored.burton_dst_nt),
        burton_full_dst_nt=Float64.(scored.burton_full_dst_nt),
        obrien_dst_nt=Float64.(scored.obrien_dst_nt),
        static_v2_2_dst_nt=static_v22,
        observation_dst_nt=Float64.(scored.observation_dst_nt),
        latest_dst_nt=Float64.(scored.latest_dst_nt),
        dst_delta_1h_nt=Float64.(scored.dst_delta_1h_nt),
        coupling_active_mvm=Float64.(scored.coupling_active_mvm),
        V_kms=Float64.(scored.V_kms),
        Bz_nt=Float64.(scored.Bz_nt),
        By_nt=Float64.(scored.By_nt),
        n_cm3=Float64.(scored.n_cm3),
        Pdyn_npa=Float64.(scored.Pdyn_npa),
    )
    for name in calibration.feature_names
        String(name) in names(table) && continue
        table[!, name] = Float64.(scored[!, name])
    end
    table[!, :pred_dst_nt] = Float64.(scored.pred_dst_nt)
    issorted(collect(zip(table.issue_time_utc, table.model_step_hours))) ||
        sort!(table, [:issue_time_utc, :model_step_hours])
    v2_3_base_table_invariants(table; horizons=horizons, year_end=year_end)
    stats = (
        rows=nrow(table),
        issues=length(unique(table.issue_time_utc)),
        frame_rows=nrow(frame),
        driver_records=length(lookup),
        frozen_identity_max_abs_nt=identity_residual,
        frame_seconds=frame_seconds,
        calibration_seconds=calibration_seconds,
        replay_seconds=replay_seconds,
        generation_seconds=time() - started,
        calibration_label=calibration.label,
        stack_label=stack.label,
    )
    return (; table, frame, stats)
end

"""
    v2_3_base_table_invariants(table; horizons, year_end)

Fail closed on the structural guarantees every downstream V2.3 stage assumes:
target/step consistency, unique keys, finite numerics, the declared model-step
set, the partition/block labelling, and the absence of a target beyond the
declared final year.
"""
function v2_3_base_table_invariants(table::DataFrame;
                                    horizons::Vector{Int}=copy(V23_BASE_HORIZONS),
                                    year_end::Int=V23_BASE_YEAR_END)
    nrow(table) > 0 || error("base table is empty")
    sort(unique(table.model_step_hours)) == sort(horizons) || error(
        "base table does not cover exactly the declared model steps",
    )
    all(table.target_time_utc .== table.issue_time_utc .+ Hour.(table.model_step_hours)) ||
        error("base table target-time invariant failed")
    last_target = DateTime(year_end, 12, 31, 23)
    maximum(table.target_time_utc) <= last_target || error(
        "base table contains a target beyond $last_target",
    )
    row_keys = collect(zip(
        table.issue_time_utc, table.target_time_utc, table.model_step_hours,
    ))
    length(unique(row_keys)) == nrow(table) || error("base table contains duplicate keys")
    for column in names(table, Real)
        all(isfinite, table[!, column]) || error(
            "base table column $column contains a non-finite value",
        )
    end
    allowed = Set(("DEV", "TEST", "embargo"))
    all(in(allowed), table.partition) || error("base table has an unknown partition label")
    for i in 1:nrow(table)
        expected = v2_3_partition(table.issue_time_utc[i])
        (table.partition[i], table.cv_block[i]) == expected || error(
            "partition label disagrees with the issue time at row $i",
        )
    end
    return table
end

# ---------------------------------------------------------------------------
# Identity oracle against the archived V2.2 development replay
# ---------------------------------------------------------------------------

"""
    v2_3_base_oracle_join(; table_path, oracle_path, atol, last_year)

Join the rebuilt base table with the archived V2.2 development replay on
`(issue, target, step)` and return the joined row count together with the
maximum absolute difference of every shared numeric column. Every archived row
must be reproduced. The rebuilt table is a documented superset: rows whose
target lies beyond the archived final year, and rows of the anchors that the
purged V2.1 calibration split discarded at its two split boundaries, have no
archived counterpart, so each unmatched rebuilt row must either carry a target
beyond the archived horizon or belong to an issue time the archive never holds.
"""
function v2_3_base_oracle_join(; table_path::AbstractString=V23_BASE_TABLE,
                               oracle_path::AbstractString=V23_V22_ORACLE_TABLE,
                               atol::Float64=V23_ORACLE_ATOL_NT,
                               last_year::Int=V23_ORACLE_LAST_YEAR)
    isfile(String(table_path)) || error("V2.3 base table is missing: $table_path")
    isfile(String(oracle_path)) || error(
        "archived V2.2 development replay is missing: $oracle_path",
    )
    key_types = Dict(
        "issue_time_utc" => DateTime, "target_time_utc" => DateTime,
    )
    wanted = vcat(
        ["issue_time_utc", "target_time_utc", "model_step_hours"],
        String.(V23_ORACLE_SHARED_COLUMNS),
    )
    rebuilt = CSV.read(String(table_path), DataFrame; select=wanted, types=key_types)
    oracle = CSV.read(String(oracle_path), DataFrame; select=wanted, types=key_types)
    rebuilt = rebuilt[year.(rebuilt.issue_time_utc) .<= last_year, :]
    joined = innerjoin(
        rebuilt, oracle;
        on=[:issue_time_utc, :target_time_utc, :model_step_hours],
        renamecols="" => "_oracle",
    )
    differences = NamedTuple[]
    for column in V23_ORACLE_SHARED_COLUMNS
        left = Float64.(joined[!, column])
        right = Float64.(joined[!, Symbol(column, "_oracle")])
        push!(differences, (
            column=String(column),
            max_abs_difference=nrow(joined) == 0 ? NaN : maximum(abs.(left .- right)),
        ))
    end
    max_difference = nrow(joined) == 0 ? NaN :
        maximum(entry.max_abs_difference for entry in differences)
    archived_keys = Set(zip(
        oracle.issue_time_utc, oracle.target_time_utc, oracle.model_step_hours,
    ))
    archived_issues = Set(oracle.issue_time_utc)
    unmatched = [
        i for i in 1:nrow(rebuilt)
        if !((rebuilt.issue_time_utc[i], rebuilt.target_time_utc[i],
              rebuilt.model_step_hours[i]) in archived_keys)
    ]
    beyond_horizon = count(
        i -> rebuilt.target_time_utc[i] > V23_ORACLE_LAST_TARGET, unmatched,
    )
    unarchived_issue = count(
        i -> !(rebuilt.issue_time_utc[i] in archived_issues), unmatched,
    )
    explained = count(
        i -> rebuilt.target_time_utc[i] > V23_ORACLE_LAST_TARGET ||
             !(rebuilt.issue_time_utc[i] in archived_issues),
        unmatched,
    )
    return (
        joined_rows=nrow(joined),
        rebuilt_rows=nrow(rebuilt),
        oracle_rows=nrow(oracle),
        extra_rows=length(unmatched),
        extra_beyond_archived_horizon=beyond_horizon,
        extra_unarchived_issue=unarchived_issue,
        extra_explained=explained,
        differences=DataFrame(differences),
        max_abs_difference=max_difference,
        agrees=nrow(joined) == nrow(oracle) &&
               length(unmatched) == nrow(rebuilt) - nrow(joined) &&
               explained == length(unmatched) &&
               isfinite(max_difference) && max_difference <= atol,
    )
end

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

function _v23_manifest_rows(table::DataFrame, frame::DataFrame, stats, oracle)
    rows = NamedTuple[]
    push_entry(kind, name, count, value) =
        push!(rows, (entry_type=kind, name=name, count=Float64(count), value=String(value)))
    push_entry("rows", "base_table", nrow(table), "")
    push_entry("rows", "hourly_frame", nrow(frame), "")
    push_entry("rows", "issues", stats.issues, "")
    push_entry("rows", "driver_records", stats.driver_records, "")
    for partition in ("DEV", "TEST", "embargo")
        push_entry("partition_rows", partition, count(==(partition), table.partition), "")
        push_entry(
            "partition_issues", partition,
            length(unique(table.issue_time_utc[table.partition .== partition])), "",
        )
    end
    push_entry(
        "embargo", "dev_test_hours", v23_realised_embargo_hours(),
        "nominal=$(V23_BASE_EMBARGO_HOURS);dev_last_issue=$(V23_DEV_LAST_ISSUE);" *
        "dev_last_target=$(V23_DEV_LAST_ISSUE + Hour(V23_BASE_MAX_STEP_HOURS));" *
        "test_first_issue=$(V23_TEST_FIRST_ISSUE)",
    )
    for window in eachrow(v2_3_block_windows(table))
        push_entry(
            "block_rows", string(window.cv_block), window.block_rows,
            "first_issue=$(window.first_issue_utc);" *
            "training_max_target=$(window.training_max_target_utc);" *
            "training_rows=$(window.training_rows);" *
            "training_issues=$(window.training_issues)",
        )
    end
    inputs = (
        omni=OMNI,
        v2_1_calibration=operational_calibration_artifacts(
            OPERATIONAL_V2_1_MODEL_VERSION,
        ).point_csv,
        sindy_coefficients=operational_core_artifacts().coefficients_csv,
        v2_2_stack=V23_V22_STACK_PATH,
        v2_2_oracle_table=V23_V22_ORACLE_TABLE,
    )
    for (name, path) in pairs(inputs)
        isfile(path) || continue
        push_entry("input_sha256", String(name), NaN, _v23_file_sha256(path))
        push_entry("input_path", String(name), NaN, abspath(path))
    end
    for (name, path) in (("base_table", V23_BASE_TABLE),
                         ("hourly_frame", V23_BASE_HOURLY_FRAME))
        isfile(path) || continue
        push_entry("output_sha256", name, NaN, _v23_file_sha256(path))
        push_entry("output_path", name, NaN, abspath(path))
    end
    push_entry("label", "v2_1_calibration", NaN, stats.calibration_label)
    push_entry("label", "v2_2_stack", NaN, stats.stack_label)
    push_entry(
        "identity", "frozen_v2_1_max_abs_nt", stats.frozen_identity_max_abs_nt, "",
    )
    if oracle !== nothing
        push_entry("identity", "oracle_joined_rows", oracle.joined_rows, "")
        push_entry("identity", "oracle_rebuilt_rows", oracle.rebuilt_rows, "")
        push_entry("identity", "oracle_archived_rows", oracle.oracle_rows, "")
        push_entry("identity", "oracle_extra_rows", oracle.extra_rows, "")
        push_entry(
            "identity", "oracle_extra_beyond_archived_horizon",
            oracle.extra_beyond_archived_horizon, "",
        )
        push_entry(
            "identity", "oracle_extra_unarchived_issue",
            oracle.extra_unarchived_issue, "",
        )
        push_entry("identity", "oracle_max_abs_nt", oracle.max_abs_difference, "")
        for entry in eachrow(oracle.differences)
            push_entry("oracle_column_max_abs_nt", entry.column, entry.max_abs_difference, "")
        end
        push_entry("identity", "oracle_agrees", oracle.agrees ? 1.0 : 0.0, "")
    end
    push_entry("seconds", "hourly_frame", stats.frame_seconds, "")
    push_entry("seconds", "v2_1_calibration_table", stats.calibration_seconds, "")
    push_entry("seconds", "served_reconstruction", stats.replay_seconds, "")
    push_entry("seconds", "generation", stats.generation_seconds, "")
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

"""
    run_v2_3_base_table(; rebuild)

Build (or reuse) the base table, hourly frame and manifest, then run the
2010--2022 identity oracle against the archived V2.2 development replay.
"""
function run_v2_3_base_table(; rebuild::Bool=false)
    mkpath(V23_BASE_DIR)
    if !rebuild && isfile(V23_BASE_TABLE) && isfile(V23_BASE_HOURLY_FRAME)
        println("Reusing cached V2.3 base artifacts (pass --rebuild to regenerate)")
        oracle = v2_3_base_oracle_join()
        _print_oracle(oracle)
        return (; table=nothing, frame=nothing, oracle)
    end
    built = build_v2_3_base_table()
    CSV.write(V23_BASE_TABLE, built.table)
    CSV.write(V23_BASE_HOURLY_FRAME, built.frame)
    oracle = isfile(V23_V22_ORACLE_TABLE) ? v2_3_base_oracle_join() : nothing
    CSV.write(V23_BASE_MANIFEST, _v23_manifest_rows(built.table, built.frame, built.stats, oracle))
    @printf("V2.3 base table: %d rows, %d issues, %d hourly records in %.1f s\n",
            nrow(built.table), built.stats.issues, nrow(built.frame),
            built.stats.generation_seconds)
    @printf("  frozen-tail identity max |Δ| = %.3g nT\n",
            built.stats.frozen_identity_max_abs_nt)
    for partition in ("DEV", "TEST", "embargo")
        @printf("  %-8s rows=%d\n", partition, count(==(partition), built.table.partition))
    end
    if oracle === nothing
        @warn "archived V2.2 development replay not found; identity oracle skipped" path =
            V23_V22_ORACLE_TABLE
    else
        _print_oracle(oracle)
    end
    return (; table=built.table, frame=built.frame, oracle)
end

function _print_oracle(oracle)
    @printf("V2.2 identity oracle: joined %d rows (rebuilt %d, archived %d), max |Δ| = %.3g nT\n",
            oracle.joined_rows, oracle.rebuilt_rows, oracle.oracle_rows,
            oracle.max_abs_difference)
    @printf("  unmatched rebuilt rows %d (%d beyond the archived horizon, %d on an unarchived issue)\n",
            oracle.extra_rows, oracle.extra_beyond_archived_horizon,
            oracle.extra_unarchived_issue)
    for entry in eachrow(oracle.differences)
        @printf("  %-24s max |Δ| = %.3g nT\n", entry.column, entry.max_abs_difference)
    end
    oracle.agrees || error("V2.2 identity oracle failed")
    return oracle
end

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

function _v23_selftest_partitions()
    cases = (
        (DateTime(2010, 1, 1, 0), "embargo", 0),
        (DateTime(2010, 1, 1, 1), "DEV", 0),
        (DateTime(2012, 12, 31, 23), "DEV", 0),
        (DateTime(2013, 1, 1, 0), "DEV", 2013),
        (DateTime(2017, 12, 31, 23), "DEV", 2017),
        (DateTime(2018, 7, 1, 0), "DEV", 2018),
        (DateTime(2019, 12, 24, 16), "DEV", 2018),
        (DateTime(2019, 12, 24, 17), "embargo", 0),
        (DateTime(2020, 1, 7, 23), "embargo", 0),
        (DateTime(2020, 1, 8, 0), "TEST", 0),
        (DateTime(2025, 12, 24, 16), "TEST", 0),
        (DateTime(2025, 12, 24, 17), "embargo", 0),
    )
    for (issue, partition, block) in cases
        v2_3_partition(issue) == (partition, block) || error(
            "partition label wrong at $issue: got $(v2_3_partition(issue)), " *
            "expected ($partition, $block)",
        )
    end
    # Nominal embargo: the partition constants leave exactly 168 h of targets
    # between the last DEV target and the first TEST issue. The *realised* gap is
    # larger, because the last DEV issue precedes `V23_DEV_LAST_TARGET` by a week;
    # `_v23_manifest_rows` records that number so the disclosure does not have to
    # be re-derived from the constants.
    V23_TEST_FIRST_ISSUE - V23_DEV_LAST_TARGET == Hour(V23_BASE_EMBARGO_HOURS + 1) ||
        error("the nominal DEV/TEST embargo is not 168 h of targets")
    v23_realised_embargo_hours() >= V23_BASE_EMBARGO_HOURS ||
        error("the realised DEV/TEST embargo is shorter than $(V23_BASE_EMBARGO_HOURS) h")
    return true
end

"""
    v23_realised_embargo_hours() -> Int

Hours between the last target a DEV issue can carry and the first TEST issue.
The partition constants embargo 168 h nominally; the realised gap is longer
because `V23_DEV_LAST_ISSUE` stops a week before `V23_DEV_LAST_TARGET`, and the
number a reviewer needs is the realised one.
"""
function v23_realised_embargo_hours()
    last_target = V23_DEV_LAST_ISSUE + Hour(V23_BASE_MAX_STEP_HOURS)
    return div(Dates.value(V23_TEST_FIRST_ISSUE - last_target), 3_600_000)
end

function _v23_selftest_hourly_frame()
    times = DateTime(2020, 1, 1) .+ Hour.(0:5)
    cleaned = DataFrame(
        datetime=times,
        V=[NaN, 400.0, NaN, 500.0, NaN, 600.0],
        Bz=[-5.0, -6.0, NaN, -8.0, -9.0, -10.0],
        By=[1.0, NaN, 3.0, 4.0, NaN, 6.0],
        n=[5.0, 6.0, NaN, 8.0, 9.0, NaN],
        Dst=[-10.0, NaN, -30.0, -40.0, -50.0, -60.0],
    )
    frame = v2_3_hourly_frame(cleaned)
    frame.V[2:end] == [400.0, 400.0, 500.0, 500.0, 600.0] || error("driver speed fill wrong")
    isnan(frame.V[1]) || error("a leading gap must not be filled backwards")
    frame.Bz == [-5.0, -6.0, -6.0, -8.0, -9.0, -10.0] || error("Bz fill wrong")
    frame.By == [1.0, 1.0, 3.0, 4.0, 4.0, 6.0] || error("By fill wrong")
    frame.n == [5.0, 6.0, 6.0, 8.0, 9.0, 9.0] || error("density fill wrong")
    # Independent proton-only pressure identity, 1.6726e-6 · n · V² [nPa].
    for i in 2:6
        expected = 1.6726e-6 * frame.n[i] * frame.V[i]^2
        isapprox(frame.Pdyn[i], expected; rtol=1e-12, atol=0.0) ||
            error("dynamic pressure identity broken at $i")
    end
    isnan(frame.Pdyn[1]) || error("pressure must stay NaN while the speed gap is open")
    (isnan(frame.Dst[2]) && frame.Dst[3] == -30.0) || error("Dst must never be filled")
    lookup = v2_3_frame_driver_lookup(frame)
    length(lookup) == 5 || error("driver lookup must drop the non-finite record")
    haskey(lookup, times[1]) && error("a non-finite record entered the driver lookup")
    lookup[times[6]] == (V=600.0, Bz=-10.0, By=6.0, n=9.0, Pdyn=frame.Pdyn[6]) ||
        error("driver lookup tuple wrong")
    return true
end

function _v23_selftest_static_v22()
    pooled = OperationalV22Cell(
        6, :pooled, 60, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
        objective_mse=2.0, iterations=10,
    )
    recovery = OperationalV22Cell(
        6, :recovery, 60, [0.40, 0.20, 0.20, 0.10, 0.05, 0.05];
        objective_mse=1.0, iterations=9,
    )
    stack = OperationalV22Stack([pooled, recovery]; label="v2.3-base-selftest",
                                minimum_cell_rows=48)
    centers = (served_v2_1=-100.0, frozen_v2_1=-90.0, persistence=-80.0,
               burton=-70.0, burton_full=-60.0, obrien=-50.0)
    # Recovery cell (disturbed and no longer falling): hand sum
    # -40 - 18 - 16 - 7 - 3 - 2.5 = -86.5 nT.
    # Tolerance is the rounding budget of a six-term double-precision dot product
    # on |values| <= 100 nT; any wrong weight or component moves it by >= 0.5 nT.
    recovery_center = v2_3_static_v22_center(stack, 6, -50.0, 1.0, 0.0, centers)
    isapprox(recovery_center, -86.5; atol=1e-12, rtol=0.0) ||
        error("static V2.2 recovery blend wrong: $recovery_center")
    # Active deepening has no cell at this lead, so the pooled fallback applies:
    # -30 - 27 - 8 - 7 - 6 - 5 = -83.0 nT.
    deepening_center = v2_3_static_v22_center(stack, 6, -50.0, -1.0, 0.0, centers)
    isapprox(deepening_center, -83.0; atol=1e-12, rtol=0.0) ||
        error("static V2.2 pooled fallback wrong: $deepening_center")
    return true
end

"""
    run_v2_3_base_self_test(; require_artifacts)

Run the base-table oracles: partition labelling, causal hourly-frame
construction, the static V2.2 blend, and — when the rebuilt table and the
archived V2.2 development replay are both present — the 2010--2022 identity
join.
"""
function run_v2_3_base_self_test(; require_artifacts::Bool=true)
    _v23_selftest_partitions()
    _v23_selftest_hourly_frame()
    _v23_selftest_static_v22()
    println("  ✓ V2.3 base self-test: partitions, causal hourly frame, static V2.2 blend")
    available = isfile(V23_BASE_TABLE) && isfile(V23_V22_ORACLE_TABLE)
    if !available
        require_artifacts && error(
            "identity oracle needs $(V23_BASE_TABLE) and $(V23_V22_ORACLE_TABLE)",
        )
        @warn "identity oracle skipped: rebuilt table or archived replay missing"
        return nothing
    end
    oracle = v2_3_base_oracle_join()
    _print_oracle(oracle)
    oracle.joined_rows == V23_ORACLE_EXPECTED_ROWS || error(
        "identity oracle joined $(oracle.joined_rows) rows, expected " *
        "$(V23_ORACLE_EXPECTED_ROWS)",
    )
    oracle.extra_rows == V23_ORACLE_EXPECTED_EXTRA_ROWS || error(
        "identity oracle left $(oracle.extra_rows) rebuilt rows unmatched, " *
        "expected $(V23_ORACLE_EXPECTED_EXTRA_ROWS)",
    )
    println("  ✓ V2.3 base self-test: 2010--2022 identity against the V2.2 development replay")
    return oracle
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--self-test" in ARGS
        run_v2_3_base_self_test()
    else
        run_v2_3_base_table(; rebuild="--rebuild" in ARGS)
    end
end
