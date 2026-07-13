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
                               tail_bytes::Int=16_000_000)
    isfile(path) || return nothing
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
            length(fields) >= dst_field || continue
            dstv = tryparse(Float64, strip(fields[dst_field]))
            (dstv === nothing || abs(dstv) >= fill) && continue
            y = tryparse(Int, strip(fields[1]))
            d = tryparse(Int, strip(fields[2]))
            h = tryparse(Int, strip(fields[3]))
            (y === nothing || d === nothing || h === nothing) && continue
            return DateTime(y, 1, 1) + Day(d - 1) + Hour(h)
        end
        return nothing
    end
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
                        now_utc::DateTime=now(UTC))
    if isfile(output_path) && filesize(output_path) > 1_000_000
        last_valid = _last_valid_omni_date(output_path; delim=nothing, dst_field=41)
        lv_str = last_valid === nothing ? "unknown" : string(last_valid)
        cache_mtime = unix2datetime(mtime(output_path))
        if _omni_cache_stale(last_valid, year_end, now_utc, cache_mtime;
                             freshness_days=freshness_days)
            println("  OMNI2 cache is stale (last valid Dst: $(lv_str); requested through $(year_end)); re-downloading...")
        else
            println("  OMNI2 file already exists: $(output_path) ($(round(filesize(output_path)/1e6, digits=1)) MB); last valid Dst: $(lv_str)")
            return output_path
        end
    end
    println("  Downloading OMNI2 hourly data from NASA SPDF...")
    download(url, output_path)
    println("  Downloaded: $(round(filesize(output_path)/1e6, digits=1)) MB")
    return output_path
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
    if isfile(extracted_path) && filesize(extracted_path) > 100_000
        # Re-extract when the raw archive has been refreshed since this extraction
        # (download_omni2 can update the raw file without invalidating the CSV), or
        # when the extraction does not cover the requested year_end.
        raw_newer = isfile(raw_path) && mtime(raw_path) > mtime(extracted_path)
        last_valid = _last_valid_omni_date(extracted_path; delim=',', dst_field=10)
        cache_mtime = unix2datetime(mtime(extracted_path))
        stale = _omni_cache_stale(last_valid, year_end, now_utc, cache_mtime;
                                  freshness_days=freshness_days)
        lv_str = last_valid === nothing ? "unknown" : string(last_valid)
        if raw_newer || stale
            reason = raw_newer ? "raw archive is newer than extraction" :
                                 "extraction ends at $(lv_str), before requested $(year_end)"
            println("  Re-extracting OMNI2 columns ($(reason))...")
        else
            println("  Extracted CSV exists: $(extracted_path); last valid Dst: $(lv_str)")
            return extracted_path
        end
    end
    println("  Extracting columns with awk → CSV...")
    # Write header, then append awk-extracted data
    open(extracted_path, "w") do io
        println(io, "year,doy,hour,By,Bz,T,n,V,Pdyn,Dst,AE,AL,AU")
    end
    prog = "{OFS=\",\"; print \$1,\$2,\$3,\$16,\$17,\$23,\$24,\$25,\$29,\$41,\$42,\$53,\$54}"
    run(pipeline(Cmd(["awk", prog, raw_path]), stdout=extracted_path, append=true))
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
    clean_omni_data!(df)
    return df
end
