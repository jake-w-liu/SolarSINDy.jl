function _archive_dst_checked_shift(tag::DateTime, hours::Integer)
    milliseconds = Base.Checked.checked_mul(Int64(hours), Int64(3_600_000))
    shifted = Base.Checked.checked_add(Dates.value(tag), milliseconds)
    return DateTime(Dates.UTInstant{Millisecond}(Millisecond(shifted)))
end

"""
    archive_dst_forecast_timing(anchor_tag, target_tag; archive_basis,
        extra_dst_delay_hours=0, anchor_received_at=nothing, actual_issue=nothing,
        target_received_at=nothing, require_positive_completion_lead=true)

Interpret immutable OMNI hourly archive tags as means over [tag, tag+1 h).
`archive_basis=:omni_hour_start_legacy_origin` is mandatory: this helper is not
a live-provider timestamp converter. Without measured anchor receipt and actual
issue, issuance is explicitly hypothetical at anchor completion plus the declared
extra delay. Supplied measured times are retained exactly and validated together.
Target completion is a maturity lower bound, never a claimed publication receipt.
"""
function archive_dst_forecast_timing(anchor_tag::DateTime, target_tag::DateTime;
        archive_basis::Symbol, extra_dst_delay_hours::Integer=0,
        anchor_received_at::Union{Nothing,DateTime}=nothing,
        actual_issue::Union{Nothing,DateTime}=nothing,
        target_received_at::Union{Nothing,DateTime}=nothing,
        require_positive_completion_lead::Bool=true)
    archive_basis === :omni_hour_start_legacy_origin || throw(ArgumentError(
        "this helper requires an explicitly declared OMNI hour-start legacy origin"))
    # Test the integer millisecond grid directly: calendar rounding near the
    # DateTime representation limit can itself overflow before this validation.
    mod(Dates.value(anchor_tag), 3_600_000) == 0 && mod(Dates.value(target_tag), 3_600_000) == 0 ||
        throw(ArgumentError("OMNI hourly mean tags must lie on the hour"))
    extra_dst_delay_hours isa Bool && throw(ArgumentError("extra delay must be an integer hour count"))
    extra_dst_delay_hours >= 0 || throw(ArgumentError("extra delay must be nonnegative"))
    target_tag > anchor_tag || throw(ArgumentError("raw target must follow raw anchor"))
    anchor_complete = _archive_dst_checked_shift(anchor_tag, 1)
    target_complete = _archive_dst_checked_shift(target_tag, 1)
    measured = anchor_received_at !== nothing || actual_issue !== nothing
    if measured
        anchor_received_at !== nothing && actual_issue !== nothing || throw(ArgumentError(
            "measured chronology requires both anchor receipt and actual issue"))
        extra_dst_delay_hours == 0 || throw(ArgumentError("measured chronology cannot mix an assumed additional delay"))
        anchor_received_at >= anchor_complete || throw(ArgumentError("anchor receipt predates completed hourly mean"))
        actual_issue >= anchor_received_at || throw(ArgumentError("actual issue predates anchor receipt"))
        issue = actual_issue
        basis = "measured actual issue and anchor receipt; archive interval interpretation"
    else
        issue = _archive_dst_checked_shift(anchor_complete, extra_dst_delay_hours)
        basis = "assumed anchor completion plus declared extra delay; not measured publication or receipt"
    end
    target_received_at === nothing || target_received_at >= target_complete || throw(ArgumentError(
        "target receipt predates completed target hour"))
    positive = target_complete > issue
    require_positive_completion_lead && !positive && throw(ArgumentError(
        "the reconstructed issue has no positive lead to target-hour completion"))
    lead_ms = Base.Checked.checked_sub(Dates.value(target_complete), Dates.value(issue))
    model_ms = Base.Checked.checked_sub(Dates.value(target_tag), Dates.value(anchor_tag))
    return (;raw_anchor_tag_utc=anchor_tag, raw_target_tag_utc=target_tag,
        anchor_hour_complete_utc=anchor_complete,
        anchor_received_at_utc=anchor_received_at === nothing ? missing : anchor_received_at,
        reconstructed_issue_time_utc=issue, issue_time_basis=basis,
        target_hour_start_utc=target_tag, target_hour_complete_utc=target_complete,
        target_received_at_utc=target_received_at === nothing ? missing : target_received_at,
        completion_lead_hours=lead_ms / 3_600_000,
        archive_model_step_hours=div(model_ms, 3_600_000),
        assumed_extra_dst_delay_hours=measured ? missing : Int(extra_dst_delay_hours),
        has_measured_issue_and_anchor_receipt=measured,
        is_positive_completion_lead=positive,
        legacy_origin_year=year(anchor_tag), reconstructed_issue_year=year(issue))
end

_archive_dst_datetime(value::DateTime) = value
_archive_dst_datetime(value::AbstractString) = DateTime(strip(value))
_archive_dst_datetime(value) = throw(ArgumentError("archive timing column must contain DateTime or ISO-8601 text"))

"""
    archive_dst_provenance(table; archive_basis, origin_column=:issue_time_utc,
        step_column=:model_step_hours, target_column=nothing, ...)

Return an annotated copy of an explicitly declared legacy archive table.
Original keys, numerical columns and supplied receipt timestamps are unchanged.
New names distinguish reconstructed issuance from the legacy `issue_time_utc`
raw-origin key. This does not reassign calendar folds, refit, or establish live
availability. A timing-column collision fails rather than overwriting evidence.
"""
function archive_dst_provenance(table::AbstractDataFrame; archive_basis::Symbol,
        origin_column::Symbol=:issue_time_utc, step_column::Symbol=:model_step_hours,
        target_column::Union{Nothing,Symbol}=nothing,
        anchor_receipt_column::Union{Nothing,Symbol}=nothing,
        actual_issue_column::Union{Nothing,Symbol}=nothing,
        target_receipt_column::Union{Nothing,Symbol}=nothing,
        extra_dst_delay_hours::Integer=0, require_positive_completion_lead::Bool=true)
    archive_basis === :omni_hour_start_legacy_origin || throw(ArgumentError("explicit legacy OMNI archive basis is required"))
    required = Symbol[origin_column]
    if target_column === nothing
        push!(required, step_column)
    else
        push!(required, target_column)
    end
    for column in (anchor_receipt_column, actual_issue_column, target_receipt_column)
        column === nothing || push!(required, column)
    end
    all(column -> hasproperty(table, column), required) || throw(ArgumentError("archive table omits a requested timing column"))
    (anchor_receipt_column === nothing) == (actual_issue_column === nothing) ||
        throw(ArgumentError("measured chronology requires both receipt and actual-issue columns"))
    nrow(table) > 0 || throw(ArgumentError("archive timing export requires at least one row"))
    timings = map(eachrow(table)) do row
        origin = _archive_dst_datetime(row[origin_column])
        if target_column === nothing
            step = row[step_column]
            step isa Integer && !(step isa Bool) && step > 0 || throw(ArgumentError("archive model step must be a positive integer"))
            target = _archive_dst_checked_shift(origin, step)
        else
            target = _archive_dst_datetime(row[target_column])
        end
        maybe_time(column) = column === nothing || ismissing(row[column]) ? nothing : _archive_dst_datetime(row[column])
        archive_dst_forecast_timing(origin, target; archive_basis, extra_dst_delay_hours,
            anchor_received_at=maybe_time(anchor_receipt_column), actual_issue=maybe_time(actual_issue_column),
            target_received_at=maybe_time(target_receipt_column), require_positive_completion_lead)
    end
    output = copy(table)
    for name in propertynames(first(timings))
        hasproperty(output, name) && throw(ArgumentError("archive provenance would overwrite existing column $name"))
        output[!, name] = getproperty.(timings, name)
    end
    return output
end
