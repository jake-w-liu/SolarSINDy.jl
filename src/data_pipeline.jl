# Real data ingestion: NASA OMNI2 hourly data
# Format reference: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text

using Dates
using Downloads: download

const OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"

# Fill value for OMNI2 Dst (word 41 / extracted-CSV column 10); |Dst| >= this is a gap.
const OMNI2_DST_FILL = 99999.0
# Default freshness window (days). Only gates a re-fetch when coverage is already
# insufficient, so a fully-covered past year is never re-downloaded; it prevents
# thrashing when NASA simply has no newer data than a recently fetched file.
const OMNI2_FRESHNESS_DAYS = 7

"""
    _last_valid_omni_date(path; delim, dst_field, fill=OMNI2_DST_FILL, tail_bytes=16_000_000)

Return the DateTime of the last record in `path` whose Dst field is a real
observation (`abs(Dst) < fill`), or `nothing` if none is found in the scanned tail.

The OMNI2 archive pads the current year to Dec 31 with fill rows, so the file's
final row's year/doy always reads as full-year coverage. Scanning from the end for
the last non-fill Dst recovers the true end of usable data. Only the file tail is
read (default 16 MB, comfortably more than a full year of hourly rows) to avoid
re-reading the ~200 MB raw archive. `delim=nothing` splits on whitespace (raw
archive); a `Char` splits on that delimiter (extracted CSV). Fields 1/2/3 are
year/doy/hour.
"""
function _last_valid_omni_date(path::String; delim::Union{Char,Nothing},
                               dst_field::Int, fill::Float64=OMNI2_DST_FILL,
                               tail_bytes::Int=16_000_000,
                               min_fields::Int=dst_field)
    isfile(path) || return nothing
    dst_field >= 1 && min_fields >= dst_field ||
        throw(ArgumentError("invalid OMNI field-count requirement"))
    fsz = filesize(path)
    return open(path, "r") do io
        start = max(0, fsz - tail_bytes)
        start > 0 && seek(io, start)
        chunk = read(io, String)
        lines = split(chunk, '\n'; keepempty=false)
        # Drop a possibly-partial leading line when the read began mid-file.
        start > 0 && length(lines) > 1 && (lines = lines[2:end])
        for line in Iterators.reverse(lines)
            fields = delim === nothing ? split(line) : split(line, delim)
            length(fields) >= min_fields || continue
            dstv = tryparse(Float64, strip(fields[dst_field]))
            (dstv === nothing || abs(dstv) >= fill) && continue
            y = tryparse(Int, strip(fields[1]))
            d = tryparse(Int, strip(fields[2]))
            h = tryparse(Int, strip(fields[3]))
            (y === nothing || d === nothing || h === nothing) && continue
            1 <= y <= 9999 || continue
            1 <= d <= Dates.daysinyear(Date(y)) || continue
            0 <= h <= 23 || continue
            return DateTime(y, 1, 1) + Day(d - 1) + Hour(h)
        end
        return nothing
    end
end

function _validate_raw_omni(path::AbstractString; minimum_rows::Int=1)
    minimum_rows >= 1 || throw(ArgumentError("minimum_rows must be positive"))
    isfile(path) || throw(ArgumentError("OMNI2 raw archive not found: $path"))
    first_time = nothing
    last_time = nothing
    last_valid_dst_time = nothing
    n_rows = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line)
            length(fields) >= 54 || error(
                "OMNI2 raw row has $(length(fields)) fields; expected at least 54"
            )
            y = tryparse(Int, fields[1])
            d = tryparse(Int, fields[2])
            h = tryparse(Int, fields[3])
            y !== nothing && d !== nothing && h !== nothing ||
                error("OMNI2 raw row has an invalid timestamp")
            1 <= y <= 9999 && 1 <= d <= Dates.daysinyear(Date(y)) && 0 <= h <= 23 ||
                error("OMNI2 raw row has an out-of-range timestamp")
            all(index -> begin
                    value = tryparse(Float64, fields[index])
                    value !== nothing && isfinite(value)
                end, 4:54) ||
                error("OMNI2 raw row contains a nonnumeric or non-finite measurement")
            timestamp = DateTime(y, 1, 1) + Day(d - 1) + Hour(h)
            if last_time !== nothing
                timestamp - last_time == Hour(1) ||
                    error("OMNI2 raw timestamps must be strictly contiguous and hourly")
            end
            first_time === nothing && (first_time = timestamp)
            last_time = timestamp
            dst = parse(Float64, fields[41])
            abs(dst) < OMNI2_DST_FILL && (last_valid_dst_time = timestamp)
            n_rows += 1
        end
    end
    n_rows >= minimum_rows || error(
        "OMNI2 raw archive has $n_rows rows; expected at least $minimum_rows"
    )
    last_valid_dst_time !== nothing ||
        error("OMNI2 raw archive contains no valid Dst observation")
    return (; first_time, last_time, last_valid_dst_time, n_rows)
end

"""
    _omni_cache_stale(last_valid, year_end, now_utc, cache_mtime; freshness_days=OMNI2_FRESHNESS_DAYS)

Decide whether an OMNI2 cache should be re-fetched. Returns `true` (stale) when:

- the cache carries no readable Dst record (`last_valid === nothing`); or
- a fully-elapsed requested year (`year_end < year(now_utc)`) is not covered through
  its final hour AND the cache is older than `freshness_days`; or
- the requested year is the current/future year (coverage end unknowable) AND the
  cache is older than `freshness_days`.

A past year that is already covered through Dec 31 is never re-fetched. The freshness
gate on the uncovered branches prevents re-downloading a file NASA has not yet
extended.
"""
function _omni_cache_stale(last_valid::Union{Nothing,DateTime}, year_end::Int,
                           now_utc::DateTime, cache_mtime::DateTime;
                           freshness_days::Int=OMNI2_FRESHNESS_DAYS)
    last_valid === nothing && return true
    fresh = (now_utc - cache_mtime) <= Day(freshness_days)
    if year_end < year(now_utc)
        # A completed calendar year must be covered through its final hour.
        last_valid >= DateTime(year_end, 12, 31, 23) && return false
        return !fresh
    end
    # Current/future year: cannot judge by coverage, so use file freshness.
    return !fresh
end

"""
    download_omni2(output_path; url=OMNI2_URL, year_end=2025,
                   freshness_days=OMNI2_FRESHNESS_DAYS, now_utc=now(UTC))

Download the OMNI2 all-years hourly data file from NASA SPDF. An existing cache is
reused only when it actually covers the requested `year_end`; a stale cache (raw
archive fetched before the requested year's data was complete) is re-downloaded.
The last valid Dst date is always printed so coverage is visible rather than silent.
"""
function download_omni2(output_path::String; url::String=OMNI2_URL,
                        year_end::Int=2025,
                        freshness_days::Int=OMNI2_FRESHNESS_DAYS,
                        now_utc::DateTime=now(UTC),
                        download_fn::Function=download)
    _require_regular_output_target(output_path)
    previous_last_valid = nothing
    previous_size = 0
    if isfile(output_path) && filesize(output_path) > 1_000_000
        cached_summary = try
            _validate_raw_omni(output_path)
        catch e
            e isa InterruptException && rethrow()
            @warn "existing OMNI2 cache failed validation; downloading a replacement" exception=(e, catch_backtrace())
            nothing
        end
        if cached_summary === nothing
            println("  OMNI2 cache is invalid; re-downloading...")
        else
            last_valid = cached_summary.last_valid_dst_time
            previous_last_valid = last_valid
            previous_size = filesize(output_path)
            lv_str = string(last_valid)
            cache_mtime = unix2datetime(mtime(output_path))
            if _omni_cache_stale(last_valid, year_end, now_utc, cache_mtime;
                                 freshness_days=freshness_days)
                println("  OMNI2 cache is stale (last valid Dst: $(lv_str); requested through $(year_end)); re-downloading...")
            else
                println("  OMNI2 file already exists: $(output_path) ($(round(filesize(output_path)/1e6, digits=1)) MB); last valid Dst: $(lv_str)")
                return output_path
            end
        end
    end
    println("  Downloading OMNI2 hourly data from NASA SPDF...")
    parent = dirname(output_path)
    !isempty(parent) && mkpath(parent)
    tmp_path, tmp_io = mktemp(parent; cleanup=false)
    close(tmp_io)
    try
        download_fn(url, tmp_path)
        filesize(tmp_path) > 0 || error("downloaded OMNI2 archive is empty")
        downloaded_summary = _validate_raw_omni(tmp_path; minimum_rows=8_760)
        downloaded_last = downloaded_summary.last_valid_dst_time
        if year_end < year(now_utc)
            required_last = DateTime(year_end, 12, 31, 23)
            downloaded_last >= required_last || error(
                "downloaded OMNI2 coverage ends at $downloaded_last; " *
                "requested completed year $year_end requires coverage through $required_last"
            )
        end
        if previous_last_valid !== nothing
            downloaded_last >= previous_last_valid || error(
                "downloaded OMNI2 coverage regressed from $previous_last_valid to $downloaded_last"
            )
            filesize(tmp_path) >= previous_size || error(
                "downloaded OMNI2 archive is smaller than the existing verified cache"
            )
        end
        _atomic_replace_regular(tmp_path, output_path)
    finally
        isfile(tmp_path) && rm(tmp_path; force=true)
    end
    println("  Downloaded: $(round(filesize(output_path)/1e6, digits=1)) MB")
    return output_path
end

const OMNI_EXTRACTED_HEADER =
    "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU"

function _validate_extracted_omni(path::AbstractString)
    isfile(path) || throw(ArgumentError("extracted OMNI file not found: $path"))
    first_time = nothing
    last_time = nothing
    last_valid_dst_time = nothing
    n_rows = 0
    open(path, "r") do io
        eof(io) && error("extracted OMNI file is empty")
        strip(readline(io)) == OMNI_EXTRACTED_HEADER ||
            error("extracted OMNI header is invalid")
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line, ','; keepempty=true)
            length(fields) == 13 || error(
                "extracted OMNI row has $(length(fields)) fields; expected 13"
            )
            y = tryparse(Int, strip(fields[1]))
            d = tryparse(Int, strip(fields[2]))
            h = tryparse(Int, strip(fields[3]))
            y !== nothing && d !== nothing && h !== nothing ||
                error("extracted OMNI row has an invalid timestamp")
            1 <= y <= 9999 && 1 <= d <= Dates.daysinyear(Date(y)) && 0 <= h <= 23 ||
                error("extracted OMNI row has an out-of-range timestamp")
            all(index -> begin
                    value = tryparse(Float64, strip(fields[index]))
                    value !== nothing && isfinite(value)
                end, 4:13) ||
                error("extracted OMNI row contains a nonnumeric or non-finite measurement")
            timestamp = DateTime(y, 1, 1) + Day(d - 1) + Hour(h)
            last_time === nothing || timestamp - last_time == Hour(1) ||
                error("extracted OMNI timestamps must be strictly contiguous and hourly")
            first_time === nothing && (first_time = timestamp)
            last_time = timestamp
            dst = parse(Float64, strip(fields[10]))
            isfinite(dst) && abs(dst) < OMNI2_DST_FILL &&
                (last_valid_dst_time = timestamp)
            n_rows += 1
        end
    end
    n_rows >= 1 || error("extracted OMNI file contains no data rows")
    last_valid_dst_time !== nothing ||
        error("extracted OMNI file contains no valid Dst observation")
    return (; first_time, last_time, last_valid_dst_time, n_rows)
end

"""
    extract_omni2_columns(raw_path, extracted_path)

Use shell awk to extract needed columns from OMNI2 and write as CSV.
Columns: year, doy, hour, By_gsm(16), Bz_gsm(17), T(23), n(24), V(25),
         Pdyn(29), Dst(41), AE(42), AL(53), AU(54)
"""
function extract_omni2_columns(raw_path::String, extracted_path::String;
                               year_end::Int=2025,
                               freshness_days::Int=OMNI2_FRESHNESS_DAYS,
                               now_utc::DateTime=now(UTC))
    _require_regular_output_target(extracted_path)
    if isfile(extracted_path) && filesize(extracted_path) > 100_000
        cached_summary = try
            _validate_extracted_omni(extracted_path)
        catch e
            e isa InterruptException && rethrow()
            @warn "existing extracted OMNI cache failed validation; rebuilding it" exception=(e, catch_backtrace())
            nothing
        end
        # Re-extract when the raw archive has been refreshed since this extraction
        # (download_omni2 can update the raw file without invalidating the CSV), or
        # when the extraction does not cover the requested year_end.
        raw_newer = isfile(raw_path) && mtime(raw_path) > mtime(extracted_path)
        last_valid = cached_summary === nothing ? nothing : cached_summary.last_valid_dst_time
        cache_mtime = unix2datetime(mtime(extracted_path))
        stale = cached_summary === nothing ||
                _omni_cache_stale(last_valid, year_end, now_utc, cache_mtime;
                                  freshness_days=freshness_days)
        lv_str = last_valid === nothing ? "unknown" : string(last_valid)
        if raw_newer || stale
            reason = cached_summary === nothing ? "existing extraction is invalid" :
                     raw_newer ? "raw archive is newer than extraction" :
                                 "extraction ends at $(lv_str), before requested $(year_end)"
            println("  Re-extracting OMNI2 columns ($(reason))...")
        else
            println("  Extracted CSV exists: $(extracted_path); last valid Dst: $(lv_str)")
            return extracted_path
        end
    end
    isfile(raw_path) || throw(ArgumentError("OMNI2 raw archive not found: $raw_path"))
    _validate_raw_omni(raw_path)
    println("  Extracting columns with awk → CSV...")
    parent = dirname(extracted_path)
    !isempty(parent) && mkpath(parent)
    tmp_path, tmp_io = mktemp(parent; cleanup=false)
    try
        println(tmp_io, OMNI_EXTRACTED_HEADER)
        close(tmp_io)
        prog = "{OFS=\",\"; print \$1,\$2,\$3,\$16,\$17,\$23,\$24,\$25,\$29,\$41,\$42,\$53,\$54}"
        open(tmp_path, "a") do io
            run(pipeline(Cmd(["awk", prog, raw_path]), stdout=io))
        end
        new_summary = _validate_extracted_omni(tmp_path)
        if isfile(extracted_path)
            old_summary = try
                _validate_extracted_omni(extracted_path)
            catch e
                e isa InterruptException && rethrow()
                nothing
            end
            if old_summary !== nothing
                new_summary.first_time <= old_summary.first_time || error(
                    "OMNI2 extraction would discard earlier historical coverage"
                )
                new_summary.last_valid_dst_time >= old_summary.last_valid_dst_time ||
                    error("OMNI2 extraction would regress valid Dst coverage")
                new_summary.n_rows >= old_summary.n_rows ||
                    error("OMNI2 extraction would discard existing rows")
            end
        end
        _atomic_replace_regular(tmp_path, extracted_path)
    finally
        isopen(tmp_io) && close(tmp_io)
        isfile(tmp_path) && rm(tmp_path; force=true)
    end
    println("  Extracted: $(round(filesize(extracted_path)/1e6, digits=1)) MB")
    return extracted_path
end

"""
    parse_omni2_csv(filepath; year_start=1963, year_end=2025)

Parse the extracted OMNI2 CSV (from `extract_omni2_columns`) into a DataFrame.
Uses CSV.jl for fast columnar reading. Replaces fill values with NaN and
constructs DateTime column from year/doy/hour.
"""
function parse_omni2_csv(filepath::String; year_start::Int=1963, year_end::Int=2025)
    println("  Reading extracted CSV with CSV.jl...")
    raw = CSV.read(filepath, DataFrame)
    println("  Read $(nrow(raw)) rows")

    # Filter year range
    mask = (raw.year .>= year_start) .& (raw.year .<= year_end)
    raw = raw[mask, :]
    println("  After year filter ($(year_start)–$(year_end)): $(nrow(raw)) rows")

    # Build DateTime column
    datetimes = [DateTime(row.year, 1, 1) + Day(row.doy - 1) + Hour(row.hour)
                 for row in eachrow(raw)]

    # Convert to Float64 and replace fill values with NaN
    V    = _fill_to_nan(Float64.(raw.V),    9999.0)
    Bz   = _fill_to_nan(Float64.(raw.Bz),   999.9, abscheck=true)
    By   = _fill_to_nan(Float64.(raw.By),   999.9, abscheck=true)
    n    = _fill_to_nan(Float64.(raw.n),    999.9)
    Pdyn = _fill_to_nan(Float64.(raw.Pdyn), 99.99)
    T    = _fill_to_nan(Float64.(raw.T),    9999999.0)
    Dst  = _fill_to_nan(Float64.(raw.Dst),  99999.0, abscheck=true)
    AE   = _fill_to_nan(Float64.(raw.AE),   9999.0)
    AL   = _fill_to_nan(Float64.(raw.AL),   99999.0, abscheck=true)
    AU   = _fill_to_nan(Float64.(raw.AU),   99999.0, abscheck=true)  # OMNI2 word-54 fill is 99999, not 9999

    println("  Parsed $(length(datetimes)) records")

    return DataFrame(
        datetime=datetimes, V=V, Bz=Bz, By=By,
        n=n, Pdyn=Pdyn, T=T, Dst=Dst,
        AE=AE, AL=AL, AU=AU
    )
end

function _fill_to_nan(x::Vector{Float64}, fill_val::Float64; abscheck::Bool=false)
    for i in eachindex(x)
        if abscheck
            abs(x[i]) >= fill_val && (x[i] = NaN)
        else
            x[i] >= fill_val && (x[i] = NaN)
        end
    end
    return x
end

# Keep old name as alias
const parse_omni2 = parse_omni2_csv

function _clean_parsed_omni!(df::DataFrame; causal::Bool=false)
    add_original_observation_flags!(df)
    clean_omni_data!(df; causal=causal)
    return df
end

"""
    load_omni2_csv(filepath)

Load a previously saved cleaned OMNI2 CSV file.
"""
function load_omni2_csv(filepath::String)
    df = CSV.read(filepath, DataFrame)
    if eltype(df.datetime) <: AbstractString
        df.datetime = DateTime.(df.datetime)
    end
    return df
end

"""
    prepare_omni_data(; data_dir=get_data_dir(), year_start=1963, year_end=2025, force=false)

End-to-end fetch + prep of the NASA OMNI2 hourly dataset so the package is
self-contained: download the raw archive from the public NASA SPDF source (if
absent), extract the needed columns, parse, and clean. Returns the cleaned
`DataFrame`. The large raw/extracted files are written under `data_dir` (the
package `data/` directory by default) and are intentionally gitignored — they are
never committed; this function regenerates them on demand.
"""
function prepare_omni_data(; data_dir::String=get_data_dir(), year_start::Int=1963,
                           year_end::Int=2025, force::Bool=false)
    mkpath(data_dir)
    raw = joinpath(data_dir, "omni_hourly_raw.dat")
    extracted = joinpath(data_dir, "omni_extracted.csv")
    if force
        rm(raw, force=true); rm(extracted, force=true)
    end
    download_omni2(raw; year_end=year_end)
    extract_omni2_columns(raw, extracted; year_end=year_end)
    df = parse_omni2(extracted; year_start=year_start, year_end=year_end)
    _clean_parsed_omni!(df)
    return df
end
