#!/usr/bin/env julia

# Fetch the NASA CDAWeb one-minute OMNI HRO months required by the seven-storm
# sub-hourly validation set. Files are kept in the ignored operational source
# cache and are never treated as package-owned inputs.

using Dates
using Downloads
using SHA

include(joinpath(@__DIR__, "v2_subhourly_A_replay.jl"))

function _requested_hro_months(args)
    if isempty(args)
        return sort!(unique(vcat((_hro_months_for_storm(storm) for storm in STORMS)...)))
    end
    months = Date[]
    for arg in args
        occursin(r"^\d{6}$", arg) || error("month must use YYYYMM format: $arg")
        push!(months, Date(parse(Int, arg[1:4]), parse(Int, arg[5:6]), 1))
    end
    return sort!(unique(months))
end

function _validate_downloaded_hro(path::AbstractString, month_start::Date)
    filesize(path) > 1_000_000 || error("downloaded HRO file is unexpectedly small: $path")
    first_row = nothing
    open(path, "r") do io
        for line in eachline(io)
            fields = split(line)
            length(fields) >= 28 || continue
            first_row = fields
            break
        end
    end
    first_row === nothing && error("downloaded HRO file has no parseable data rows: $path")
    parse(Int, first_row[1]) == year(month_start) || error("HRO year mismatch in $path")
    doy = parse(Int, first_row[2])
    row_date = Date(year(month_start), 1, 1) + Day(doy - 1)
    month(row_date) == month(month_start) || error("HRO month mismatch in $path")
    return bytes2hex(open(sha256, path))
end

function fetch_hro_month(month_start::Date)
    destination = _hro_month_path(month_start)
    if isfile(destination)
        digest = _validate_downloaded_hro(destination, month_start)
        println("verified cached $(basename(destination))  sha256=$digest")
        return (month=Dates.format(month_start, "yyyymm"), file=basename(destination), sha256=digest,
                bytes=filesize(destination), source="$HRO_BASE_URL/$(basename(destination))")
    end

    mkpath(dirname(destination))
    source = "$HRO_BASE_URL/$(basename(destination))"
    digest = mktemp(dirname(destination)) do temp_path, temp_io
        close(temp_io)
        println("fetching $source")
        Downloads.download(source, temp_path)
        checked = _validate_downloaded_hro(temp_path, month_start)
        mv(temp_path, destination)
        checked
    end
    println("stored $(basename(destination))  sha256=$digest")
    return (month=Dates.format(month_start, "yyyymm"), file=basename(destination), sha256=digest,
            bytes=filesize(destination), source=source)
end

function main_fetch_hro(args=ARGS)
    records = fetch_hro_month.(_requested_hro_months(args))
    manifest = joinpath(HRO_DIR, "manifest.csv")
    open(manifest, "w") do io
        println(io, "month,file,sha256,bytes,source")
        for record in records
            println(io, join((record.month, record.file, record.sha256, record.bytes, record.source), ','))
        end
    end
    println("wrote $manifest")
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_fetch_hro()
end
