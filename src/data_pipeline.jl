# Real data ingestion: NASA OMNI2 hourly data
# Format reference: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text

using Dates
using Downloads: download

const OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"

"""
    download_omni2(output_path; url=OMNI2_URL)

Download the OMNI2 all-years hourly data file from NASA SPDF.
"""
function download_omni2(output_path::String; url::String=OMNI2_URL)
    if isfile(output_path) && filesize(output_path) > 1_000_000
        println("  OMNI2 file already exists: $(output_path) ($(round(filesize(output_path)/1e6, digits=1)) MB)")
        return output_path
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
function extract_omni2_columns(raw_path::String, extracted_path::String)
    if isfile(extracted_path) && filesize(extracted_path) > 100_000
        println("  Extracted CSV exists: $(extracted_path)")
        return extracted_path
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
    AU   = _fill_to_nan(Float64.(raw.AU),   9999.0)

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
