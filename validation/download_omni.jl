#!/usr/bin/env julia
# download_omni.jl — Phase A: Download, clean, and catalog real OMNI2 data
#
# Outputs:
#   paper/data/omni_hourly_raw.dat    — raw OMNI2 ASCII (cached)
#   paper/data/storm_catalog.csv      — storm catalog with splits

using SolarSINDy
using CSV, DataFrames, Dates

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATA_DIR)

# ============================================================
# A1: Download OMNI2 hourly data
# ============================================================
println("=" ^ 60)
println("Phase A1: Download OMNI2 hourly data")
println("=" ^ 60)

raw_path = joinpath(DATA_DIR, "omni_hourly_raw.dat")
download_omni2(raw_path)

# ============================================================
# A2: Parse and clean
# ============================================================
println("\n" * "=" ^ 60)
println("Phase A2: Parse and clean OMNI2 data")
println("=" ^ 60)

extracted_path = joinpath(DATA_DIR, "omni_extracted.csv")
extract_omni2_columns(raw_path, extracted_path)
@time df = parse_omni2(extracted_path; year_start=1963, year_end=2025)
println("  DataFrame: $(nrow(df)) rows, $(ncol(df)) columns")
println("  Date range: $(df.datetime[1]) to $(df.datetime[end])")

for col in [:V, :Bz, :Dst, :AE]
    n_valid = count(!isnan, df[!, col])
    println("  $(rpad(string(col), 6)): $(n_valid) valid ($(round(100*n_valid/nrow(df), digits=1))%)")
end

println("\nCleaning...")
@time clean_omni_data!(df)

println("  Dst_star range: $(round(minimum(filter(!isnan, df.Dst_star)), digits=1)) to $(round(maximum(filter(!isnan, df.Dst_star)), digits=1)) nT")
println("  Dst_star < -50: $(count(x -> !isnan(x) && x < -50, df.Dst_star)) hours")
println("  Dst_star < -100: $(count(x -> !isnan(x) && x < -100, df.Dst_star)) hours")

# ============================================================
# A3: Build storm catalog
# ============================================================
println("\n" * "=" ^ 60)
println("Phase A3: Build storm catalog")
println("=" ^ 60)

@time catalog = build_storm_catalog(df; dst_thresh=-50.0, window_pre=24,
                                     window_post=144, min_separation=48)

if isempty(catalog)
    println("  ERROR: No storms found!")
    exit(1)
end

# Summary by solar cycle
println("\n  Storms per solar cycle:")
for cycle in 20:25
    sc = filter(e -> e.solar_cycle == cycle, catalog)
    if !isempty(sc)
        println("    Cycle $(cycle) ($(sc[1].split)): $(length(sc)) storms, " *
                "Dst [$(round(minimum(e.min_dst for e in sc), digits=0)), $(round(maximum(e.min_dst for e in sc), digits=0))] nT")
    end
end

# Notable storms
println("\n  Top 10 strongest storms:")
for e in sort(catalog, by=e->e.min_dst)[1:min(10, length(catalog))]
    println("    #$(lpad(e.storm_id, 4)): $(Dates.format(e.onset_time, "yyyy-mm-dd")) " *
            "Dst=$(lpad(round(Int, e.min_dst), 5)) nT  [$(e.split)]")
end

# May 2024 superstorm
may2024 = filter(e -> Dates.year(e.onset_time) == 2024 && Dates.month(e.onset_time) == 5, catalog)
if !isempty(may2024)
    println("\n  May 2024 superstorm: Storm #$(may2024[1].storm_id), Dst=$(round(may2024[1].min_dst, digits=1)) nT")
else
    println("\n  WARNING: May 2024 superstorm not found")
end

# Save catalog
catalog_path = joinpath(DATA_DIR, "storm_catalog.csv")
save_storm_catalog(catalog, catalog_path)

# ============================================================
# A4: Validate storm extraction
# ============================================================
println("\n" * "=" ^ 60)
println("Phase A4: Validate storm extraction")
println("=" ^ 60)

for split_name in ["train", "val", "test"]
    entries = filter(e -> e.split == split_name, catalog)
    if !isempty(entries)
        swd = extract_storm_data(df, entries[1])
        println("  $(rpad(split_name, 6)) example: Storm #$(entries[1].storm_id), $(length(swd.t)) pts, " *
                "V=[$(round(minimum(swd.V)))–$(round(maximum(swd.V)))] km/s, " *
                "Dst*=[$(round(minimum(swd.Dst_star), digits=1))–$(round(maximum(swd.Dst_star), digits=1))] nT")
    end
end

# Count totals
n_train = count(e -> e.split == "train", catalog)
n_val = count(e -> e.split == "val", catalog)
n_test = count(e -> e.split == "test", catalog)

println("\n" * "=" ^ 60)
println("Phase A complete.")
println("  Storm catalog: $(catalog_path)")
println("  Train/Val/Test: $(n_train)/$(n_val)/$(n_test) storms")
println("=" ^ 60)
