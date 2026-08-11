#!/usr/bin/env julia

# Leakage-safe expanding-window primary-stack cross-fitting for the bounded
# nonlinear V2.2 development stage. This script reads only the pinned pre-2023
# residual replay and writes nothing unless executed directly.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA

const V22_CROSSFIT_REPLAY_PATH = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "v2_2_residual_development",
    "v2_2_residual_replay.csv",
))
const V22_CROSSFIT_OUTPUT_DIR = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "v2_2_crossfit",
))
const V22_CROSSFIT_OUTPUT_PATH = joinpath(
    V22_CROSSFIT_OUTPUT_DIR, "v2_2_primary_crossfit_oof.csv",
)
const V22_CROSSFIT_AUDIT_PATH = joinpath(
    V22_CROSSFIT_OUTPUT_DIR, "v2_2_primary_crossfit_audit.csv",
)

# The complete upstream lineage is frozen before nonlinear fitting. Only the
# residual replay itself is opened here; the preceding hashes are provenance.
const V22_CROSSFIT_INPUT_HASHES = (
    upstream_source_sha256=
        "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a",
    primary_replay_sha256=
        "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43",
    primary_stack_sha256=
        "66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d",
    residual_replay_sha256=
        "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f",
    feature_schema_sha256=
        "dbd15d62ad43ae930895768f01570a3d2ed1248a6d01202634d5be246fcf6d09",
)

const V22_CROSSFIT_POINT_FIT_END = DateTime(2017, 10, 20, 15)
const V22_CROSSFIT_POINT_FIT_TARGET_END = DateTime(2017, 10, 20, 22)
const V22_CROSSFIT_POST_2022_START = DateTime(2023, 1, 1)
const V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS = 168
const V22_CROSSFIT_MINIMUM_CELL_ROWS = 48
const V22_CROSSFIT_MODEL_STEPS = Tuple(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS)
const V22_CROSSFIT_BLOCKS = (
    (label="calendar_2013", start=DateTime(2013, 1, 8),
     stop=DateTime(2013, 12, 24, 23)),
    (label="calendar_2014", start=DateTime(2014, 1, 8),
     stop=DateTime(2014, 12, 24, 23)),
    (label="calendar_2015", start=DateTime(2015, 1, 8),
     stop=DateTime(2015, 12, 24, 23)),
    (label="calendar_2016", start=DateTime(2016, 1, 8),
     stop=DateTime(2016, 12, 24, 23)),
    (label="calendar_2017", start=DateTime(2017, 1, 8),
     stop=V22_CROSSFIT_POINT_FIT_END),
)

const V22_CROSSFIT_KEY_COLUMNS = (
    :issue_time_utc, :target_time_utc, :model_step_hours,
)
const V22_CROSSFIT_REQUIRED_COLUMNS = unique((
    V22_CROSSFIT_KEY_COLUMNS...,
    :split_label,
    values(DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS)...,
    :observation_dst_nt,
    :latest_dst_nt,
    :dst_delta_1h_nt,
    :coupling_active_mvm,
    OPERATIONAL_V22_RESIDUAL_FEATURES...,
))

_v22_crossfit_file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _v22_crossfit_feature_schema_sha256(features=OPERATIONAL_V22_RESIDUAL_FEATURES)
    payload = "operational_v2_2_issue_feature_schema_v1\n" *
              join(String.(features), '\n') * "\n"
    return bytes2hex(sha256(codeunits(payload)))
end

function _v22_crossfit_elapsed_hours(later::DateTime, earlier::DateTime)
    milliseconds = Dates.value(later - earlier)
    milliseconds % 3_600_000 == 0 || throw(ArgumentError(
        "cross-fit boundaries must differ by an integer number of hours",
    ))
    return milliseconds ÷ 3_600_000
end

function _v22_crossfit_validate_blocks(blocks)
    isempty(blocks) && throw(ArgumentError("cross-fit requires at least one block"))
    labels = String[]
    previous = nothing
    for block in blocks
        label = String(block.label)
        isempty(strip(label)) && throw(ArgumentError("cross-fit block label is empty"))
        block.start <= block.stop || throw(ArgumentError(
            "cross-fit block $label ends before it starts",
        ))
        floor(block.start, Hour) == block.start && floor(block.stop, Hour) == block.stop ||
            throw(ArgumentError("cross-fit block $label is not on the hourly grid"))
        block.stop <= V22_CROSSFIT_POINT_FIT_END || throw(ArgumentError(
            "cross-fit block $label extends beyond the point-fit boundary",
        ))
        if previous !== nothing
            block.start > previous.stop || throw(ArgumentError(
                "cross-fit blocks overlap or are out of order",
            ))
            gap = _v22_crossfit_elapsed_hours(block.start, previous.stop)
            gap >= V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS || throw(ArgumentError(
                "cross-fit blocks $(previous.label) and $label are separated by " *
                "$gap h, below the 168 h minimum",
            ))
        end
        push!(labels, label)
        previous = block
    end
    length(unique(labels)) == length(labels) || throw(ArgumentError(
        "cross-fit block labels must be unique",
    ))
    return nothing
end

function _v22_crossfit_require_columns(df::DataFrame)
    missing_columns = [
        String(column) for column in V22_CROSSFIT_REQUIRED_COLUMNS
        if !(String(column) in names(df))
    ]
    isempty(missing_columns) || throw(ArgumentError(
        "cross-fit replay omits required column(s): $(join(missing_columns, ", "))",
    ))
    return nothing
end

function _v22_crossfit_datetime_column(df::DataFrame, column::Symbol)
    values = df[!, column]
    all(value -> value isa DateTime, values) || throw(ArgumentError(
        "cross-fit column $column must contain DateTime values",
    ))
    return DateTime.(values)
end

function _v22_crossfit_model_steps(df::DataFrame)
    steps = Vector{Int}(undef, nrow(df))
    for row in 1:nrow(df)
        value = df[row, :model_step_hours]
        value isa Real && !(value isa Bool) && isfinite(value) &&
            value > 0 && isinteger(value) || throw(ArgumentError(
                "cross-fit model_step_hours must contain positive integers",
            ))
        steps[row] = Int(value)
    end
    return steps
end

function _v22_crossfit_key_set(df::DataFrame)
    return Set(zip(
        DateTime.(df.issue_time_utc), DateTime.(df.target_time_utc),
        Int.(df.model_step_hours),
    ))
end

function _v22_crossfit_point_rows(df::DataFrame;
                                  model_steps=V22_CROSSFIT_MODEL_STEPS)
    _v22_crossfit_require_columns(df)
    _v22_crossfit_feature_schema_sha256() ==
        V22_CROSSFIT_INPUT_HASHES.feature_schema_sha256 || error(
        "the frozen V2.2 issue-feature schema hash changed",
    )
    issues = _v22_crossfit_datetime_column(df, :issue_time_utc)
    targets = _v22_crossfit_datetime_column(df, :target_time_utc)
    steps = _v22_crossfit_model_steps(df)
    labels = String.(df.split_label)
    point_indices = findall(==("fit"), labels)
    isempty(point_indices) && throw(ArgumentError("cross-fit replay has no point-fit rows"))
    point = copy(df[point_indices, :])
    point.issue_time_utc = issues[point_indices]
    point.target_time_utc = targets[point_indices]
    point.model_step_hours = steps[point_indices]
    sort!(point, collect(V22_CROSSFIT_KEY_COLUMNS))

    maximum(point.issue_time_utc) <= V22_CROSSFIT_POINT_FIT_END || error(
        "point-fit replay extends beyond its frozen issue boundary",
    )
    maximum(point.target_time_utc) <= V22_CROSSFIT_POINT_FIT_TARGET_END || error(
        "point-fit replay contains an immature target",
    )
    nrow(point) == length(_v22_crossfit_key_set(point)) || throw(ArgumentError(
        "point-fit replay contains duplicate issue/target/lead keys",
    ))

    expected_steps = sort!(unique(Int.(collect(model_steps))))
    expected_steps == collect(model_steps) || throw(ArgumentError(
        "cross-fit model steps must be sorted and unique",
    ))
    for anchor in groupby(point, :issue_time_utc; sort=true)
        anchor_steps = sort!(Int.(anchor.model_step_hours))
        anchor_steps == expected_steps || throw(ArgumentError(
            "cross-fit anchor $(anchor.issue_time_utc[1]) is not whole: " *
            "steps=$(join(anchor_steps, ','))",
        ))
        all(anchor.target_time_utc .==
            anchor.issue_time_utc .+ Hour.(anchor.model_step_hours)) ||
            throw(ArgumentError(
                "cross-fit anchor $(anchor.issue_time_utc[1]) violates target maturity",
            ))
    end
    return point
end

function _v22_crossfit_partition_point(point::DataFrame, block)
    issue_groups = groupby(point, :issue_time_utc; sort=true)
    fit_indices = Int[]
    oof_indices = Int[]
    fit_target_cutoff = block.start - Hour(V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS)
    for anchor in issue_groups
        issue = only(unique(anchor.issue_time_utc))
        indices = parentindices(anchor)[1]
        if issue < block.start && maximum(anchor.target_time_utc) <= fit_target_cutoff
            append!(fit_indices, indices)
        elseif block.start <= issue <= block.stop
            append!(oof_indices, indices)
        end
    end
    isempty(fit_indices) && throw(ArgumentError(
        "cross-fit block $(block.label) has no mature expanding-window fit rows",
    ))
    isempty(oof_indices) && throw(ArgumentError(
        "cross-fit block $(block.label) has no retained OOF rows",
    ))
    return (; fit_indices, oof_indices)
end

"""
    v2_2_crossfit_partition(df, block)

Return keys for the expanding fit set, purged so its last target is at least
168 h before the retained OOF block, and for the retained block itself.
Selection depends only on split labels, issue times, target times, and complete
lead anchors; forecast values and observations cannot affect it.
"""
function v2_2_crossfit_partition(df::DataFrame, block;
                                 model_steps=V22_CROSSFIT_MODEL_STEPS)
    point = _v22_crossfit_point_rows(df; model_steps)
    partition = _v22_crossfit_partition_point(point, block)
    keys = collect(V22_CROSSFIT_KEY_COLUMNS)
    return (
        fit_keys=select(point[partition.fit_indices, :], keys),
        oof_keys=select(point[partition.oof_indices, :], keys),
    )
end

function _v22_crossfit_refresh_primary_features!(scored::DataFrame)
    scored.served_minus_frozen_v2_1_nt =
        scored.served_v2_1_dst_nt .- scored.frozen_v2_1_dst_nt
    scored.primary_minus_served_v2_1_nt =
        scored.v2_2_pred_dst_nt .- scored.served_v2_1_dst_nt
    scored.primary_minus_frozen_v2_1_nt =
        scored.v2_2_pred_dst_nt .- scored.frozen_v2_1_dst_nt
    scored.primary_minus_persistence_nt =
        scored.v2_2_pred_dst_nt .- scored.persistence_dst_nt
    scored.primary_minus_burton_full_nt =
        scored.v2_2_pred_dst_nt .- scored.burton_full_dst_nt
    scored.primary_minus_obrien_nt =
        scored.v2_2_pred_dst_nt .- scored.obrien_dst_nt
    for feature in OPERATIONAL_V22_RESIDUAL_FEATURES
        all(isfinite, Float64.(scored[!, feature])) || error(
            "cross-fit produced a non-finite issue feature $feature",
        )
    end
    return scored
end

"""
    build_v2_2_primary_crossfit(df; kwargs...)

Fit the constrained primary stack on expanding, target-mature point-fit rows
with a 168 h target-to-block purge and score exactly one OOF prediction for
each retained 2013--2017 key. The
returned `oof` table carries fold and fit-cutoff labels; `fold_audit` records
the independently checkable chronology and pinned input hashes.
"""
function build_v2_2_primary_crossfit(df::DataFrame;
        blocks=V22_CROSSFIT_BLOCKS,
        model_steps=V22_CROSSFIT_MODEL_STEPS,
        minimum_cell_rows::Integer=V22_CROSSFIT_MINIMUM_CELL_ROWS)
    _v22_crossfit_validate_blocks(blocks)
    point = _v22_crossfit_point_rows(df; model_steps)
    outputs = DataFrame[]
    expected = DataFrame[]
    audit_rows = NamedTuple[]
    stacks = Dict{String,OperationalV22Stack}()

    for block in blocks
        partition = _v22_crossfit_partition_point(point, block)
        fit_rows = point[partition.fit_indices, :]
        oof_rows = point[partition.oof_indices, :]
        fit_issue_max = maximum(fit_rows.issue_time_utc)
        fit_target_max = maximum(fit_rows.target_time_utc)
        fit_target_gap_hours = _v22_crossfit_elapsed_hours(
            block.start, fit_target_max,
        )
        fit_issue_max < block.start || error(
            "cross-fit fit issue does not strictly predate $(block.label)",
        )
        fit_target_gap_hours >= V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS || error(
            "cross-fit fit target is not purged by 168 h before $(block.label)",
        )

        label = "v2_2_primary_crossfit_$(block.label)_purge168h"
        stack = fit_operational_v22_stack(
            fit_rows;
            minimum_cell_rows=minimum_cell_rows,
            label=label,
        )
        stack.supported_model_steps == Tuple(model_steps) || error(
            "cross-fit stack $label omitted a supported lead",
        )
        scored = score_operational_v22(oof_rows, stack)
        _v22_crossfit_refresh_primary_features!(scored)
        scored[!, :v2_2_crossfit_fold] = fill(String(block.label), nrow(scored))
        scored[!, :v2_2_crossfit_block_start_utc] = fill(block.start, nrow(scored))
        scored[!, :v2_2_crossfit_block_stop_utc] = fill(block.stop, nrow(scored))
        scored[!, :v2_2_crossfit_fit_issue_max_utc] = fill(fit_issue_max, nrow(scored))
        scored[!, :v2_2_crossfit_fit_target_max_utc] = fill(fit_target_max, nrow(scored))
        scored[!, :v2_2_crossfit_fit_target_gap_hours] =
            fill(fit_target_gap_hours, nrow(scored))
        scored[!, :v2_2_crossfit_fit_rows] = fill(nrow(fit_rows), nrow(scored))
        push!(outputs, scored)
        push!(expected, select(oof_rows, collect(V22_CROSSFIT_KEY_COLUMNS)))
        stacks[String(block.label)] = stack
        push!(audit_rows, (
            fold=String(block.label),
            block_start_utc=block.start,
            block_stop_utc=block.stop,
            fit_issue_max_utc=fit_issue_max,
            fit_target_max_utc=fit_target_max,
            fit_target_gap_hours=fit_target_gap_hours,
            fit_rows=nrow(fit_rows),
            fit_anchors=length(unique(fit_rows.issue_time_utc)),
            oof_rows=nrow(oof_rows),
            oof_anchors=length(unique(oof_rows.issue_time_utc)),
            upstream_source_sha256=V22_CROSSFIT_INPUT_HASHES.upstream_source_sha256,
            primary_replay_sha256=V22_CROSSFIT_INPUT_HASHES.primary_replay_sha256,
            primary_stack_sha256=V22_CROSSFIT_INPUT_HASHES.primary_stack_sha256,
            residual_replay_sha256=V22_CROSSFIT_INPUT_HASHES.residual_replay_sha256,
            feature_schema_sha256=V22_CROSSFIT_INPUT_HASHES.feature_schema_sha256,
            post_2022_rows_read=0,
        ))
    end

    oof = vcat(outputs...)
    expected_keys = vcat(expected...)
    sort!(oof, collect(V22_CROSSFIT_KEY_COLUMNS))
    sort!(expected_keys, collect(V22_CROSSFIT_KEY_COLUMNS))
    actual_keys = select(oof, collect(V22_CROSSFIT_KEY_COLUMNS))
    nrow(oof) == length(_v22_crossfit_key_set(oof)) || error(
        "cross-fit OOF output contains duplicate keys",
    )
    nrow(actual_keys) == nrow(expected_keys) && actual_keys == expected_keys || error(
        "cross-fit OOF output does not exactly cover the retained keys",
    )
    return (; oof, fold_audit=DataFrame(audit_rows), stacks)
end

function read_v2_2_crossfit_replay(path::AbstractString=V22_CROSSFIT_REPLAY_PATH)
    abspath(path) == abspath(V22_CROSSFIT_REPLAY_PATH) || throw(ArgumentError(
        "cross-fit may read only the frozen V2.2 residual replay",
    ))
    isfile(path) && !islink(path) || throw(ArgumentError(
        "cross-fit replay must be a regular non-symlink file: $path",
    ))
    _v22_crossfit_file_sha256(path) ==
        V22_CROSSFIT_INPUT_HASHES.residual_replay_sha256 || error(
        "frozen V2.2 residual replay SHA-256 changed",
    )
    table = CSV.read(path, DataFrame; types=Dict(
        :issue_time_utc => DateTime,
        :target_time_utc => DateTime,
    ))
    _v22_crossfit_require_columns(table)
    maximum(table.issue_time_utc) < V22_CROSSFIT_POST_2022_START &&
        maximum(table.target_time_utc) < V22_CROSSFIT_POST_2022_START || error(
        "cross-fit replay contains a post-2022 row",
    )
    return table
end

function _v22_crossfit_atomic_write(path::AbstractString, table::DataFrame)
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    try
        CSV.write(temporary, table)
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return path
end

function run_v2_2_crossfit()
    table = read_v2_2_crossfit_replay()
    result = build_v2_2_primary_crossfit(table)
    _v22_crossfit_atomic_write(V22_CROSSFIT_OUTPUT_PATH, result.oof)
    _v22_crossfit_atomic_write(V22_CROSSFIT_AUDIT_PATH, result.fold_audit)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_v2_2_crossfit()
    println("V2.2 primary cross-fit complete: ", nrow(result.oof), " OOF rows")
end
