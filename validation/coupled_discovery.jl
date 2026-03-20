#!/usr/bin/env julia
# coupled_discovery.jl — Phase D: Coupled Dst-AE Multi-Index Dynamics
#
# Following PLAN.md:
#   D1: Coupled system setup (state vector x = [Dst*, AE])
#   D2: Implementation (multi-output SINDy)
#   D3: Validation (coupled vs single-index predictions)
#
# Outputs:
#   paper/data/coupled_coefficients.csv
#   paper/data/coupled_metrics.csv

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random, LinearAlgebra

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATA_DIR)

# ============================================================
# Helper functions
# ============================================================

"""
Build augmented library for coupled Dst-AE system.
Adds AE, AE², Dst*·AE cross terms to the standard solar wind library.
"""
function build_coupled_library()
    base_lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                          include_cross=true, include_known=true)
    base_names = get_term_names(base_lib)
    base_funcs = base_lib.funcs

    # Add AE-related terms
    aug_names = copy(base_names)
    aug_funcs = copy(base_funcs)

    push!(aug_names, "AE")
    push!(aug_funcs, d -> d["AE"])

    push!(aug_names, "AE^2")
    push!(aug_funcs, d -> d["AE"].^2)

    push!(aug_names, "Dst_star*AE")
    push!(aug_funcs, d -> d["Dst_star"] .* d["AE"])

    push!(aug_names, "V*AE")
    push!(aug_funcs, d -> d["V"] .* d["AE"])

    push!(aug_names, "Bs*AE")
    push!(aug_funcs, d -> d["Bs"] .* d["AE"])

    return CandidateLibrary(aug_names, aug_funcs)
end

"""
Prepare coupled system data: add AE and dAE/dt to the data dict.
"""
function prepare_coupled_data(swd::SolarWindData, dt::Real; smooth_window::Int=5)
    data_dict, dDst_dt = prepare_sindy_data(swd, dt; smooth_window=smooth_window)

    # Add AE index (use Dst field from SolarWindData — we need actual AE from DataFrame)
    # For now, we'll handle this in the main script where we have access to the DataFrame
    return data_dict, dDst_dt
end

"""
Forward-simulate coupled Dst-AE system.
"""
function simulate_coupled(ξ_dst::AbstractVector, ξ_ae::AbstractVector,
                          lib::CandidateLibrary, swd::SolarWindData,
                          AE_obs::AbstractVector, dt::Real)
    n_pts = length(swd.t)
    Dst_pred = zeros(n_pts)
    AE_pred = zeros(n_pts)
    Dst_pred[1] = swd.Dst_star[1]
    AE_pred[1] = AE_obs[1]

    for k in 1:(n_pts - 1)
        Bs_k = max(-swd.Bz[k], 0.0)
        theta_c_k = atan(abs(swd.By[k]), swd.Bz[k])
        BT_k = sqrt(swd.By[k]^2 + swd.Bz[k]^2)

        point_data = Dict{String,Vector{Float64}}(
            "V"        => [swd.V[k]],
            "Bs"       => [Bs_k],
            "n"        => [swd.n[k]],
            "Pdyn"     => [swd.Pdyn[k]],
            "Dst_star" => [Dst_pred[k]],
            "theta_c"  => [theta_c_k],
            "BT"       => [BT_k],
            "By"       => [swd.By[k]],
            "Bz"       => [swd.Bz[k]],
            "AE"       => [AE_pred[k]]
        )

        Θ_k = evaluate_library(lib, point_data)

        dDst = clamp((Θ_k * ξ_dst)[1], -200.0, 200.0)
        dAE  = clamp((Θ_k * ξ_ae)[1], -2000.0, 2000.0)

        Dst_pred[k+1] = clamp(Dst_pred[k] + dt * dDst, -2000.0, 50.0)
        AE_pred[k+1]  = clamp(AE_pred[k] + dt * dAE, 0.0, 5000.0)
    end

    return Dst_pred, AE_pred
end

# simulate_sindy is now exported from SolarSINDy — use simulate_sindy() directly

# ============================================================
# Load data
# ============================================================
println("=" ^ 60)
println("Phase D: Coupled Dst-AE Multi-Index Dynamics")
println("=" ^ 60)

extracted_path = joinpath(DATA_DIR, "omni_extracted.csv")
println("\nLoading OMNI data...")
@time df = parse_omni2(extracted_path; year_start=1963, year_end=2025)
@time clean_omni_data!(df)

catalog = load_storm_catalog(joinpath(DATA_DIR, "storm_catalog.csv"))
println("Loaded catalog: $(length(catalog)) storms")

# ============================================================
# D1: Coupled System Setup
# ============================================================
println("\n" * "=" ^ 60)
println("D1: Coupled System Setup")
println("=" ^ 60)

coupled_lib = build_coupled_library()
coupled_names = get_term_names(coupled_lib)
println("Coupled library: $(length(coupled_lib)) terms")
println("Added terms: AE, AE², Dst*·AE, V·AE, Bs·AE")

# Standard library for single-index comparison
std_lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                     include_cross=true, include_known=true)

# ============================================================
# D2: Discover coupled equations from training data
# ============================================================
println("\n" * "=" ^ 60)
println("D2: Coupled Equation Discovery")
println("=" ^ 60)

train_entries = filter(e -> e.split == "train", catalog)

# Prepare coupled training data
coupled_data_list = Dict{String,Vector{Float64}}[]
dDst_list = Float64[]
dAE_list = Float64[]
n_used = 0

for entry in train_entries
    rows = entry.onset_idx:entry.end_idx
    n_pts = length(rows)
    if n_pts < 25
        continue
    end

    # Check AE data availability
    AE_storm = df.AE[rows]
    if count(isnan, AE_storm) > 0.3 * n_pts
        continue  # Skip storms with >30% missing AE
    end

    swd = extract_storm_data(df, entry)
    try
        data_dict, dDst_dt = prepare_sindy_data(swd, 1.0; smooth_window=5)
        if count(!isnan, dDst_dt) < 20
            continue
        end

        # Get AE data and compute dAE/dt
        AE_raw = Float64.(df.AE[rows])
        # Fill NaN with nearest neighbor
        for i in 2:length(AE_raw)
            if isnan(AE_raw[i]) && !isnan(AE_raw[i-1])
                AE_raw[i] = AE_raw[i-1]
            end
        end
        for i in (length(AE_raw)-1):-1:1
            if isnan(AE_raw[i]) && !isnan(AE_raw[i+1])
                AE_raw[i] = AE_raw[i+1]
            end
        end

        # Smooth AE
        AE_smooth = smooth_moving_average(AE_raw, 5)
        dAE_dt = numerical_derivative(AE_smooth, 1.0)

        # Add AE to data dict
        data_dict["AE"] = AE_smooth

        # Valid mask
        valid = .!isnan.(dDst_dt) .& .!isnan.(dAE_dt) .& .!isnan.(AE_smooth)
        for key in keys(data_dict)
            valid .&= .!isnan.(data_dict[key])
        end

        n_v = count(valid)
        if n_v < 20
            continue
        end

        # Filter
        filtered_dict = Dict{String,Vector{Float64}}()
        for key in keys(data_dict)
            filtered_dict[key] = data_dict[key][valid]
        end

        push!(coupled_data_list, filtered_dict)
        append!(dDst_list, dDst_dt[valid])
        append!(dAE_list, dAE_dt[valid])
        global n_used += 1
    catch e
        continue
    end
end

println("Storms with AE data: $(n_used) / $(length(train_entries))")
println("Total coupled data points: $(length(dDst_list))")

if n_used < 10
    println("ERROR: Not enough storms with AE data!")
    exit(1)
end

# Concatenate
coupled_concat = Dict{String,Vector{Float64}}()
for key in keys(coupled_data_list[1])
    coupled_concat[key] = vcat([d[key] for d in coupled_data_list]...)
end

# Evaluate coupled library
Θ_coupled = evaluate_library(coupled_lib, coupled_concat)
println("Coupled library matrix: $(size(Θ_coupled))")

# --- Discover dDst*/dt equation (with AE cross terms) ---
println("\n--- dDst*/dt equation (with AE cross terms) ---")
lambdas = 10.0 .^ range(-1, 4, length=40)
sweep_dst = sweep_lambda(Θ_coupled, dDst_list, lambdas; normalize=true)

println("Pareto front (Dst*):")
for nt in sort(unique([r.n_terms for r in sweep_dst]))
    subset = filter(r -> r.n_terms == nt, sweep_dst)
    best = argmin(r -> r.rmse, subset)
    if nt <= 15
        println("  $(nt) terms: RMSE=$(round(best.rmse, digits=4))")
    end
end

# Find good λ for 8-12 terms
dst_λ = 100.0
for target in [8:12, 6:14, 4:16]
    kr = filter(r -> r.n_terms in target, sweep_dst)
    if !isempty(kr)
        global dst_λ = argmin(r -> r.rmse, kr).λ
        break
    end
end

ξ_dst_coupled, active_dst, _ = sindy_discover(coupled_concat, coupled_lib, dDst_list;
                                                λ=dst_λ, normalize=true)

println("\nDiscovered dDst*/dt ($(count(ξ_dst_coupled .!= 0)) terms):")
for (name, coef) in sort(collect(active_dst), by=x->abs(x[2]), rev=true)
    println("  $(round(coef, sigdigits=4)) × $(name)")
end

# Check if AE appears in Dst equation
ae_in_dst = any(name -> occursin("AE", name), keys(active_dst))
println("\nAE cross-index terms in Dst* equation: $(ae_in_dst ? "YES" : "NO")")

# --- Discover dAE/dt equation ---
println("\n--- dAE/dt equation ---")
sweep_ae = sweep_lambda(Θ_coupled, dAE_list, lambdas; normalize=true)

println("Pareto front (AE):")
for nt in sort(unique([r.n_terms for r in sweep_ae]))
    subset = filter(r -> r.n_terms == nt, sweep_ae)
    best = argmin(r -> r.rmse, subset)
    if nt <= 15
        println("  $(nt) terms: RMSE=$(round(best.rmse, digits=4))")
    end
end

ae_λ = 100.0
for target in [8:12, 6:14, 4:16]
    kr = filter(r -> r.n_terms in target, sweep_ae)
    if !isempty(kr)
        global ae_λ = argmin(r -> r.rmse, kr).λ
        break
    end
end

ξ_ae_coupled, active_ae, _ = sindy_discover(coupled_concat, coupled_lib, dAE_list;
                                              λ=ae_λ, normalize=true)

println("\nDiscovered dAE/dt ($(count(ξ_ae_coupled .!= 0)) terms):")
for (name, coef) in sort(collect(active_ae), by=x->abs(x[2]), rev=true)
    println("  $(round(coef, sigdigits=4)) × $(name)")
end

# Check if Dst appears in AE equation
dst_in_ae = any(name -> occursin("Dst", name), keys(active_ae))
println("\nDst* cross-index terms in AE equation: $(dst_in_ae ? "YES" : "NO")")

# --- Ensemble for coupled system ---
println("\n--- Ensemble SINDy for coupled system ---")
@time med_ξ_dst, inc_dst, all_ξ_dst = ensemble_sindy(
    coupled_concat, coupled_lib, dDst_list;
    λ=dst_λ, n_models=500, subsample_frac=0.8, seed=42
)

@time med_ξ_ae, inc_ae, all_ξ_ae = ensemble_sindy(
    coupled_concat, coupled_lib, dAE_list;
    λ=ae_λ, n_models=500, subsample_frac=0.8, seed=42
)

# Save coupled coefficients
coef_rows = []
for (i, name) in enumerate(coupled_names)
    push!(coef_rows, (
        equation = "dDst/dt",
        term = name,
        coefficient = ξ_dst_coupled[i],
        median_coef = med_ξ_dst[i],
        inclusion_prob = inc_dst[i]
    ))
    push!(coef_rows, (
        equation = "dAE/dt",
        term = name,
        coefficient = ξ_ae_coupled[i],
        median_coef = med_ξ_ae[i],
        inclusion_prob = inc_ae[i]
    ))
end
coef_coupled_df = DataFrame(coef_rows)
CSV.write(joinpath(DATA_DIR, "coupled_coefficients.csv"), coef_coupled_df)
println("Saved: coupled_coefficients.csv")

# ============================================================
# D3: Validation — Coupled vs Single-Index
# ============================================================
println("\n" * "=" ^ 60)
println("D3: Coupled vs Single-Index Validation")
println("=" ^ 60)

# Load single-index coefficients from Phase B
sindy_single_df = CSV.read(joinpath(DATA_DIR, "real_sindy_coefficients.csv"), DataFrame)
ξ_single = sindy_single_df.coefficient

coupled_metrics_rows = []

for split_name in ["val", "test"]
    split_entries = filter(e -> e.split == split_name, catalog)
    println("\n$(uppercase(split_name)) set: $(length(split_entries)) storms")

    for entry in split_entries
        rows = entry.onset_idx:entry.end_idx
        n_pts = length(rows)
        if n_pts < 20
            continue
        end

        # Check AE availability
        AE_obs = Float64.(df.AE[rows])
        if count(isnan, AE_obs) > 0.3 * n_pts
            continue
        end
        # Fill NaN
        for i in 2:length(AE_obs)
            isnan(AE_obs[i]) && !isnan(AE_obs[i-1]) && (AE_obs[i] = AE_obs[i-1])
        end
        for i in (length(AE_obs)-1):-1:1
            isnan(AE_obs[i]) && !isnan(AE_obs[i+1]) && (AE_obs[i] = AE_obs[i+1])
        end

        swd = extract_storm_data(df, entry)
        Dst_obs = swd.Dst_star
        Bs = max.(-swd.Bz, 0.0)
        dt = 1.0

        # 1. Burton
        Dst_burton = simulate_burton(swd.V, Bs, dt; Dst0=Dst_obs[1])

        # 2. Single-index SINDy (standard lib)
        Dst_single = simulate_sindy(ξ_single, std_lib, swd, dt)

        # 3. Coupled SINDy
        Dst_coupled, AE_coupled = simulate_coupled(ξ_dst_coupled, ξ_ae_coupled,
                                                     coupled_lib, swd, AE_obs, dt)

        # Dst metrics
        for (name, pred) in [("Coupled-SINDy", Dst_coupled),
                              ("Single-SINDy", Dst_single),
                              ("Burton", Dst_burton)]
            if length(pred) == length(Dst_obs)
                m = metrics_summary(pred, Dst_obs; name=name)
                push!(coupled_metrics_rows, (
                    split = split_name,
                    storm_id = entry.storm_id,
                    target = "Dst",
                    model = name,
                    rmse = m.rmse,
                    correlation = m.corr,
                    pe = m.pe,
                    min_dst_obs = minimum(Dst_obs)
                ))
            end
        end

        # AE metrics (only for coupled)
        if length(AE_coupled) == length(AE_obs) && !any(isnan, AE_obs)
            m_ae = metrics_summary(AE_coupled, AE_obs; name="Coupled-SINDy")
            push!(coupled_metrics_rows, (
                split = split_name,
                storm_id = entry.storm_id,
                target = "AE",
                model = "Coupled-SINDy",
                rmse = m_ae.rmse,
                correlation = m_ae.corr,
                pe = m_ae.pe,
                min_dst_obs = minimum(Dst_obs)
            ))
        end
    end
end

coupled_metrics_df = DataFrame(coupled_metrics_rows)
CSV.write(joinpath(DATA_DIR, "coupled_metrics.csv"), coupled_metrics_df)

# Summary
println("\n--- Coupled System Summary ---")
for split_name in ["val", "test"]
    println("\n$(uppercase(split_name)) set:")
    split_data = filter(row -> row.split == split_name, coupled_metrics_df)

    # Dst predictions
    println("  Dst* predictions:")
    dst_data = filter(row -> row.target == "Dst", split_data)
    for model in ["Coupled-SINDy", "Single-SINDy", "Burton"]
        ms = filter(row -> row.model == model, dst_data)
        if nrow(ms) > 0
            println("    $(rpad(model, 18)) RMSE=$(round(mean(ms.rmse), digits=2)) ± $(round(std(ms.rmse), digits=2))  " *
                    "PE=$(round(mean(ms.pe), digits=3))  r=$(round(mean(ms.correlation), digits=3))")
        end
    end

    # AE predictions
    ae_data = filter(row -> row.target == "AE", split_data)
    if nrow(ae_data) > 0
        println("  AE predictions:")
        println("    Coupled-SINDy     RMSE=$(round(mean(ae_data.rmse), digits=1)) ± $(round(std(ae_data.rmse), digits=1))  " *
                "PE=$(round(mean(ae_data.pe), digits=3))  r=$(round(mean(ae_data.correlation), digits=3))")
    end
end

# Key question: Does AE improve Dst prediction?
println("\n--- Key Question: Does AE improve Dst* prediction? ---")
for split_name in ["val", "test"]
    dst_data = filter(row -> row.split == split_name && row.target == "Dst", coupled_metrics_df)
    coupled_rmse = mean(filter(row -> row.model == "Coupled-SINDy", dst_data).rmse)
    single_rmse = mean(filter(row -> row.model == "Single-SINDy", dst_data).rmse)
    improvement = (single_rmse - coupled_rmse) / single_rmse * 100
    println("  $(uppercase(split_name)): Coupled RMSE=$(round(coupled_rmse, digits=2)), " *
            "Single RMSE=$(round(single_rmse, digits=2)), " *
            "Improvement=$(round(improvement, digits=1))%")
end

println("\n" * "=" ^ 60)
println("Phase D Complete")
println("=" ^ 60)
