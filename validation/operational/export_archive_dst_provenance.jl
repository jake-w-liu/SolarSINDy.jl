#!/usr/bin/env julia

# An explicit legacy-archive export, not a live-provider timestamp migration.
using SolarSINDy, CSV, DataFrames, Dates, SHA

_archive_export_sha(path) = open(io -> bytes2hex(sha256(io)), path)

"Publish a completed CSV with an atomic no-replacement hard link on the same filesystem."
function _archive_export_publish(table, destination::AbstractString)
    mktemp(dirname(abspath(destination))) do temporary, io
        CSV.write(io, table)
        flush(io)
        close(io)
        # Unlike an existence check followed by CSV.write or rename, link cannot
        # replace a competing destination created between preparation and publish.
        hardlink(temporary, destination)
    end
    return destination
end

"Write a separately named annotated legacy result; never overwrite its source or prior evidence."
function export_archive_dst_provenance(input::AbstractString, output::AbstractString;
        archive_basis::Symbol, extra_dst_delay_hours::Integer=0,
        require_positive_completion_lead::Bool=true,
        origin_column::Symbol=:issue_time_utc, step_column::Symbol=:model_step_hours,
        target_column::Union{Nothing,Symbol}=nothing)
    archive_basis === :omni_hour_start_legacy_origin || throw(ArgumentError("explicit legacy OMNI archive basis is required"))
    source, destination = abspath(input), abspath(output)
    receipt_path = destination * ".provenance.csv"
    isfile(source) || throw(ArgumentError("legacy archive result does not exist"))
    (ispath(destination) || ispath(receipt_path)) && throw(ArgumentError("refusing to overwrite existing export evidence"))
    isdir(dirname(destination)) || throw(ArgumentError("export directory does not exist"))
    source_digest = _archive_export_sha(source)
    original = CSV.read(source, DataFrame)
    annotated = archive_dst_provenance(original; archive_basis, extra_dst_delay_hours,
        require_positive_completion_lead, origin_column, step_column, target_column)
    isequal(annotated[:, names(original)], original) || error("archive export changed an original value")
    _archive_export_sha(source) == source_digest || error("legacy source changed while export was being prepared")
    _archive_export_publish(annotated, destination)
    _archive_export_sha(source) == source_digest || error("legacy source changed during export")
    entries = [
        (entry_type="input_sha256", name=basename(source), value=source_digest),
        (entry_type="output_sha256", name=basename(destination), value=_archive_export_sha(destination)),
        (entry_type="interval_basis", name="OMNI", value="raw tag is start of hourly mean; raw source keys and predictions unchanged"),
        (entry_type="receipt_basis", name="retrospective", value="assumed minimum completion plus declared extra Dst delay; actual historical publication/receipt times unavailable"),
        (entry_type="issue_basis", name="extra_dst_delay_hours", value=string(extra_dst_delay_hours)),
        (entry_type="column_binding", name="raw_origin", value=String(origin_column)),
        (entry_type="column_binding", name="raw_target", value=target_column === nothing ? "origin + " * String(step_column) : String(target_column)),
        (entry_type="fold_basis", name="legacy", value="original raw-origin fold labels retained; no refitting or calendar-membership reassignment"),
    ]
    _archive_export_publish(DataFrame(entries), receipt_path)
    return (;path=destination, receipt_path, rows=nrow(annotated), source_sha256=source_digest)
end

function _archive_export_main(args=ARGS)
    length(args) >= 3 || throw(ArgumentError(
        "usage: export_archive_dst_provenance.jl INPUT OUTPUT --archive-basis=omni-hour-start-legacy-origin [--extra-dst-delay-hours=N] [--allow-nonpositive-for-audit]"))
    args[3] == "--archive-basis=omni-hour-start-legacy-origin" || throw(ArgumentError("explicit archive basis is required"))
    delay, positive = 0, true
    origin_column, step_column, target_column = :issue_time_utc, :model_step_hours, nothing
    for argument in args[4:end]
        if startswith(argument, "--extra-dst-delay-hours=")
            delay = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--allow-nonpositive-for-audit"
            positive = false
        elseif startswith(argument, "--origin-column=")
            origin_column = Symbol(split(argument, '='; limit=2)[2])
        elseif startswith(argument, "--step-column=")
            step_column = Symbol(split(argument, '='; limit=2)[2])
        elseif startswith(argument, "--target-column=")
            target_column = Symbol(split(argument, '='; limit=2)[2])
        else
            throw(ArgumentError("unknown archive export argument: $argument"))
        end
    end
    result = export_archive_dst_provenance(args[1], args[2];
        archive_basis=:omni_hour_start_legacy_origin, extra_dst_delay_hours=delay,
        require_positive_completion_lead=positive, origin_column, step_column, target_column)
    println("Legacy archive provenance export: $(result.rows) rows; original source SHA256 $(result.source_sha256)")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    _archive_export_main()
end
