#!/usr/bin/env julia
# phase_sensitivity.jl — Phase threshold sensitivity analysis
#
# Sweeps D_thresh and R_thresh to assess stability of phase-dependent coefficients.
# Output: paper/data/phase_threshold_sensitivity.csv

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")

println("=" ^ 60)
println("Phase Threshold Sensitivity Analysis")
println("=" ^ 60)

# Load data
extracted_path = joinpath(DATA_DIR, "omni_extracted.csv")
println("\nLoading OMNI data...")
@time df = parse_omni2(extracted_path; year_start=1963, year_end=2025)
@time clean_omni_data!(df)

catalog = load_storm_catalog(joinpath(DATA_DIR, "storm_catalog.csv"))
train_entries = filter(e -> e.split == "train", catalog)
println("Training storms: $(length(train_entries))")

# Build library
lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                 include_cross=true, include_known=true)
term_names = get_term_names(lib)

# Prepare all storm data once
all_swd = SolarWindData[]
all_data_dicts = Dict{String,Vector{Float64}}[]
all_dDst = Float64[]
all_phases_cache = Dict{Tuple{Float64,Float64}, Vector{Int}}()

storm_data_pairs = []
for entry in train_entries
    swd = extract_storm_data(df, entry)
    try
        data_dict, dDst_dt = prepare_sindy_data(swd, 1.0; smooth_window=5)
        if count(!isnan, dDst_dt) < 20
            continue
        end
        push!(storm_data_pairs, (swd, data_dict, dDst_dt))
    catch
        continue
    end
end
println("Storms with valid data: $(length(storm_data_pairs))")

# Sweep parameters
D_thresh_values = [-30.0, -25.0, -20.0, -15.0, -10.0]
R_thresh_values = [-4.0, -3.0, -2.0, -1.0, 0.0]

sensitivity_rows = []
phase_lambda = 148.0

for D_thresh in D_thresh_values
    for R_thresh in R_thresh_values
        println("\n--- D_thresh=$D_thresh, R_thresh=$R_thresh ---")

        # Segment data by phase
        phase_data = Dict{String,Vector{Dict{String,Vector{Float64}}}}(
            "quiet" => [], "main" => [], "recovery" => []
        )
        phase_dDst = Dict{String,Vector{Float64}}(
            "quiet" => Float64[], "main" => Float64[], "recovery" => Float64[]
        )

        for (swd, data_dict, dDst_dt) in storm_data_pairs
            # Custom phase classification with these thresholds
            n_pts = length(swd.Dst_star)
            phases = ones(Int, n_pts)
            for k in 1:n_pts
                dst_k = swd.Dst_star[k]
                dDst_k = k <= length(dDst_dt) ? dDst_dt[k] : 0.0
                if dst_k >= D_thresh
                    phases[k] = 1  # quiet
                elseif !isnan(dDst_k) && dDst_k < R_thresh
                    phases[k] = 2  # main
                else
                    phases[k] = 3  # recovery
                end
            end

            for (phase_id, phase_name) in [(1, "quiet"), (2, "main"), (3, "recovery")]
                mask = (phases[1:length(dDst_dt)] .== phase_id) .& .!isnan.(dDst_dt)
                for key in keys(data_dict)
                    mask .&= .!isnan.(data_dict[key])
                end
                n_phase = count(mask)
                if n_phase >= 5
                    pd = Dict{String,Vector{Float64}}()
                    for key in keys(data_dict)
                        pd[key] = data_dict[key][mask]
                    end
                    push!(phase_data[phase_name], pd)
                    append!(phase_dDst[phase_name], dDst_dt[mask])
                end
            end
        end

        # Run SINDy for each phase
        for phase_name in ["quiet", "main", "recovery"]
            dDst_vec = phase_dDst[phase_name]
            if length(dDst_vec) < 100
                println("  $phase_name: insufficient data ($(length(dDst_vec)))")
                continue
            end

            # Concatenate phase data
            concat = Dict{String,Vector{Float64}}()
            if !isempty(phase_data[phase_name])
                for key in keys(phase_data[phase_name][1])
                    concat[key] = vcat([d[key] for d in phase_data[phase_name]]...)
                end
            end

            xi, active, _ = sindy_discover(concat, lib, dDst_vec;
                                            λ=phase_lambda, normalize=true)

            n_active = count(xi .!= 0)
            println("  $phase_name: $(length(dDst_vec)) pts, $n_active terms")

            for (i, name) in enumerate(term_names)
                push!(sensitivity_rows, (
                    D_thresh = D_thresh,
                    R_thresh = R_thresh,
                    phase = phase_name,
                    term = name,
                    coefficient = xi[i],
                    n_data_points = length(dDst_vec),
                    n_active_terms = n_active
                ))
            end
        end
    end
end

sensitivity_df = DataFrame(sensitivity_rows)
CSV.write(joinpath(DATA_DIR, "phase_threshold_sensitivity.csv"), sensitivity_df)
println("\nSaved: phase_threshold_sensitivity.csv ($(nrow(sensitivity_df)) rows)")

# Print summary of key coefficient stability
println("\n--- Key Coefficient Stability ---")
for phase_name in ["quiet", "main", "recovery"]
    println("\n$(uppercase(phase_name)):")
    for key_term in ["Dst_star", "Bs", "sin^(8/3)(θ_c/2)", "n*V", "n*V^2"]
        sub = filter(row -> row.phase == phase_name && row.term == key_term &&
                     row.coefficient != 0.0, sensitivity_df)
        if nrow(sub) > 0
            coefs = sub.coefficient
            println("  $(rpad(key_term, 20)) mean=$(round(mean(coefs), sigdigits=3))  " *
                    "std=$(round(std(coefs), sigdigits=2))  " *
                    "range=[$(round(minimum(coefs), sigdigits=3)), $(round(maximum(coefs), sigdigits=3))]  " *
                    "present in $(nrow(sub))/$(length(D_thresh_values)*length(R_thresh_values)) configs")
        end
    end
end

println("\n" * "=" ^ 60)
println("Sensitivity Analysis Complete")
println("=" ^ 60)
