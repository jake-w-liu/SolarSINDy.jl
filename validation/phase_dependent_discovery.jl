#!/usr/bin/env julia
# phase_dependent_discovery.jl — Phase C: State-Dependent SINDy Discovery
#
# Following PLAN.md:
#   C1: Phase segmentation with real data
#   C2: Phase-specific SINDy
#   C3: Switching model (forward simulation)
#   C4: Physical interpretation
#
# Outputs:
#   paper/data/phase_dependent_real_coefficients.csv
#   paper/data/switching_model_metrics.csv

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")
mkpath(DATA_DIR)

# ============================================================
# Helper: Forward-simulate with phase-switching SINDy
# ============================================================

# simulate_sindy is now exported from SolarSINDy — use it directly

"""
    _simulate_switching(ξ_quiet, ξ_main, ξ_recovery, lib, swd, dt;
                        quiet_thresh=-20.0, deriv_thresh=-2.0)

Forward-integrate with phase-switching: select equation based on current state.
"""
function _simulate_switching(ξ_quiet::AbstractVector, ξ_main::AbstractVector,
                             ξ_recovery::AbstractVector, lib::CandidateLibrary,
                             swd::SolarWindData, dt::Real;
                             quiet_thresh::Real=-20.0, deriv_thresh::Real=-2.0)
    n_pts = length(swd.t)
    Dst_pred = zeros(n_pts)
    Dst_pred[1] = swd.Dst_star[1]
    phases = ones(Int, n_pts)  # 1=quiet, 2=main, 3=recovery

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
            "Bz"       => [swd.Bz[k]]
        )

        Θ_k = evaluate_library(lib, point_data)

        # Classify phase based on current predicted Dst*
        # Estimate dDst*/dt from the last two points
        if k >= 2
            dDst_est = (Dst_pred[k] - Dst_pred[k-1]) / dt
        else
            dDst_est = 0.0
        end

        if Dst_pred[k] >= quiet_thresh
            phase = 1  # quiet
        elseif dDst_est < deriv_thresh
            phase = 2  # main phase (rapidly decreasing)
        else
            phase = 3  # recovery
        end
        phases[k] = phase

        # Select equation by phase
        ξ_k = phase == 1 ? ξ_quiet : (phase == 2 ? ξ_main : ξ_recovery)
        dDst = clamp((Θ_k * ξ_k)[1], -200.0, 200.0)
        Dst_pred[k+1] = clamp(Dst_pred[k] + dt * dDst, -2000.0, 50.0)
    end

    return Dst_pred, phases
end

# ============================================================
# Load data
# ============================================================
println("=" ^ 60)
println("Phase C: State-Dependent SINDy Discovery")
println("=" ^ 60)

extracted_path = joinpath(DATA_DIR, "omni_extracted.csv")
println("\nLoading OMNI data...")
@time df = parse_omni2(extracted_path; year_start=1963, year_end=2025)
@time clean_omni_data!(df)

catalog = load_storm_catalog(joinpath(DATA_DIR, "storm_catalog.csv"))
println("Loaded catalog: $(length(catalog)) storms")

# ============================================================
# C1: Phase Segmentation
# ============================================================
println("\n" * "=" ^ 60)
println("C1: Phase Segmentation with Real Data")
println("=" ^ 60)

train_entries = filter(e -> e.split == "train", catalog)
println("Training storms: $(length(train_entries))")

# Build library
lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                 include_cross=true, include_known=true)
term_names = get_term_names(lib)

# Prepare and segment data by phase
quiet_data_list = Dict{String,Vector{Float64}}[]
main_data_list  = Dict{String,Vector{Float64}}[]
recov_data_list = Dict{String,Vector{Float64}}[]
quiet_dDst_list = Float64[]
main_dDst_list  = Float64[]
recov_dDst_list = Float64[]

n_storms_processed = 0
for entry in train_entries
    swd = extract_storm_data(df, entry)
    try
        data_dict, dDst_dt = prepare_sindy_data(swd, 1.0; smooth_window=5)
        if count(!isnan, dDst_dt) < 20
            continue
        end

        # Phase classification
        phases = identify_storm_phases(swd.Dst_star, dDst_dt)

        # Split data by phase
        for (phase_id, data_list, dDst_list) in [
            (1, quiet_data_list, quiet_dDst_list),
            (2, main_data_list,  main_dDst_list),
            (3, recov_data_list, recov_dDst_list)
        ]
            mask = (phases .== phase_id) .& .!isnan.(dDst_dt)
            # Also check all data keys are non-NaN
            for key in keys(data_dict)
                mask .&= .!isnan.(data_dict[key])
            end
            n_phase = count(mask)
            if n_phase >= 5
                phase_data = Dict{String,Vector{Float64}}()
                for key in keys(data_dict)
                    phase_data[key] = data_dict[key][mask]
                end
                push!(data_list, phase_data)
                append!(dDst_list, dDst_dt[mask])
            end
        end
        n_storms_processed += 1
    catch
        continue
    end
end

println("Storms processed: $(n_storms_processed)")

# Concatenate phase data
function _concat_dicts(dlist)
    if isempty(dlist)
        return Dict{String,Vector{Float64}}()
    end
    result = Dict{String,Vector{Float64}}()
    for key in keys(dlist[1])
        result[key] = vcat([d[key] for d in dlist]...)
    end
    return result
end

quiet_data = _concat_dicts(quiet_data_list)
main_data  = _concat_dicts(main_data_list)
recov_data = _concat_dicts(recov_data_list)

println("Phase data points:")
println("  Quiet:    $(length(quiet_dDst_list))")
println("  Main:     $(length(main_dDst_list))")
println("  Recovery: $(length(recov_dDst_list))")

# ============================================================
# C2: Phase-Specific SINDy
# ============================================================
println("\n" * "=" ^ 60)
println("C2: Phase-Specific Ensemble SINDy (500 bootstraps)")
println("=" ^ 60)

# Choose λ for each phase — use the value from B1 analysis
phase_λ = 148.0  # gives 10 terms on full data; may differ per phase

phase_results = Dict{String,Any}()
for (phase_name, phase_data_dict, phase_dDst) in [
    ("quiet",    quiet_data, quiet_dDst_list),
    ("main",     main_data,  main_dDst_list),
    ("recovery", recov_data, recov_dDst_list)
]
    println("\n--- $(uppercase(phase_name)) phase ($(length(phase_dDst)) points) ---")

    if length(phase_dDst) < 100
        println("  Insufficient data — skipping")
        continue
    end

    # Lambda sweep for this phase
    Θ_phase = evaluate_library(lib, phase_data_dict)
    lambdas = 10.0 .^ range(-1, 4, length=40)
    sweep = sweep_lambda(Θ_phase, phase_dDst, lambdas; normalize=true)

    println("  Pareto front:")
    for nt in sort(unique([r.n_terms for r in sweep]))
        subset = filter(r -> r.n_terms == nt, sweep)
        best = argmin(r -> r.rmse, subset)
        if nt <= 15
            println("    $(nt) terms: RMSE=$(round(best.rmse, digits=4)), λ=$(round(best.λ, sigdigits=3))")
        end
    end

    # Find good λ for this phase (8-12 terms)
    phase_best_λ = phase_λ
    for target in [6:10, 4:12, 3:15, 2:21]
        kr = filter(r -> r.n_terms in target, sweep)
        if !isempty(kr)
            phase_best_λ = argmin(r -> r.rmse, kr).λ
            break
        end
    end

    # Ensemble SINDy
    @time med_ξ, inc_prob, all_ξ_phase = ensemble_sindy(
        phase_data_dict, lib, phase_dDst;
        λ=phase_best_λ, n_models=500, subsample_frac=0.8, seed=42
    )

    # Single discovery
    ξ_phase, active, _ = sindy_discover(phase_data_dict, lib, phase_dDst;
                                          λ=phase_best_λ, normalize=true)

    phase_results[phase_name] = (ξ=ξ_phase, median_ξ=med_ξ, inclusion=inc_prob,
                                  all_ξ=all_ξ_phase, λ=phase_best_λ)

    println("  Active terms ($(count(ξ_phase .!= 0))):")
    for (name, coef) in sort(collect(active), by=x->abs(x[2]), rev=true)
        idx = findfirst(==(name), term_names)
        π_val = idx !== nothing ? inc_prob[idx] : 0.0
        println("    $(round(coef, sigdigits=4)) × $(rpad(name, 20))  π=$(round(π_val, digits=3))")
    end

    # Physical interpretation
    dst_idx = findfirst(==("Dst_star"), term_names)
    if dst_idx !== nothing && ξ_phase[dst_idx] != 0.0
        τ = -1.0 / ξ_phase[dst_idx]
        println("  Decay timescale: τ = $(round(τ, digits=1)) hr")
    end
end

# Save phase-dependent coefficients
coef_rows = []
for phase_name in ["quiet", "main", "recovery"]
    if haskey(phase_results, phase_name)
        r = phase_results[phase_name]
        for (i, name) in enumerate(term_names)
            push!(coef_rows, (
                phase = phase_name,
                term = name,
                coefficient = r.ξ[i],
                median_coef = r.median_ξ[i],
                inclusion_prob = r.inclusion[i],
                ci_025 = begin
                    vals = r.all_ξ[i, :]; active = vals[vals .!= 0.0]
                    isempty(active) ? 0.0 : quantile(active, 0.025)
                end,
                ci_975 = begin
                    vals = r.all_ξ[i, :]; active = vals[vals .!= 0.0]
                    isempty(active) ? 0.0 : quantile(active, 0.975)
                end
            ))
        end
    end
end
coef_phase_df = DataFrame(coef_rows)
CSV.write(joinpath(DATA_DIR, "phase_dependent_real_coefficients.csv"), coef_phase_df)
println("\nSaved: phase_dependent_real_coefficients.csv")

# ============================================================
# C3: Switching Model Evaluation
# ============================================================
println("\n" * "=" ^ 60)
println("C3: Switching Model vs Single-Equation vs Baselines")
println("=" ^ 60)

# Get phase-specific ξ vectors
ξ_quiet = haskey(phase_results, "quiet") ? phase_results["quiet"].ξ : zeros(length(lib))
ξ_main  = haskey(phase_results, "main")  ? phase_results["main"].ξ  : zeros(length(lib))
ξ_recov = haskey(phase_results, "recovery") ? phase_results["recovery"].ξ : zeros(length(lib))

# Load single-equation ξ from Phase B
sindy_coef = CSV.read(joinpath(DATA_DIR, "real_sindy_coefficients.csv"), DataFrame)
ξ_single = sindy_coef.coefficient

# Evaluate on validation + test sets
switch_metrics = []
for split_name in ["val", "test"]
    split_entries = filter(e -> e.split == split_name, catalog)
    println("\n$(uppercase(split_name)) set: $(length(split_entries)) storms")

    for entry in split_entries
        swd = extract_storm_data(df, entry)
        if length(swd.t) < 20
            continue
        end

        Dst_obs = swd.Dst_star
        Bs = max.(-swd.Bz, 0.0)
        dt = 1.0

        # 1. Burton
        Dst_burton = simulate_burton(swd.V, Bs, dt; Dst0=Dst_obs[1])

        # 2. O'Brien-McPherron
        Dst_obrien = simulate_obrien(swd.V, Bs, dt; Dst0=Dst_obs[1])

        # 3. Single-equation SINDy (from Phase B)
        Dst_single = simulate_sindy(ξ_single, lib, swd, dt)

        # 4. Switching SINDy
        Dst_switch, _ = _simulate_switching(ξ_quiet, ξ_main, ξ_recov, lib, swd, dt)

        for (name, pred) in [("Switching-SINDy", Dst_switch),
                              ("Single-SINDy", Dst_single),
                              ("Burton", Dst_burton),
                              ("OBrienMcP", Dst_obrien)]
            if length(pred) == length(Dst_obs)
                m = metrics_summary(pred, Dst_obs; name=name)
                push!(switch_metrics, (
                    split = split_name,
                    storm_id = entry.storm_id,
                    model = name,
                    rmse = m.rmse,
                    correlation = m.corr,
                    pe = m.pe,
                    min_dst_obs = minimum(Dst_obs)
                ))
            end
        end
    end
end

switch_df = DataFrame(switch_metrics)
CSV.write(joinpath(DATA_DIR, "switching_model_metrics.csv"), switch_df)

# Summary
println("\n--- Switching Model Summary ---")
for split_name in ["val", "test"]
    println("\n$(uppercase(split_name)) set:")
    split_data = filter(row -> row.split == split_name, switch_df)
    for model in ["Switching-SINDy", "Single-SINDy", "Burton", "OBrienMcP"]
        ms = filter(row -> row.model == model, split_data)
        if nrow(ms) > 0
            println("  $(rpad(model, 18)) RMSE=$(round(mean(ms.rmse), digits=2)) ± $(round(std(ms.rmse), digits=2))  " *
                    "PE=$(round(mean(ms.pe), digits=3))  r=$(round(mean(ms.correlation), digits=3))")
        end
    end
end

# ============================================================
# C4: Physical Interpretation
# ============================================================
println("\n" * "=" ^ 60)
println("C4: Physical Interpretation")
println("=" ^ 60)

println("\nDecay timescales by phase:")
dst_idx = findfirst(==("Dst_star"), term_names)
for phase_name in ["quiet", "main", "recovery"]
    if haskey(phase_results, phase_name)
        ξ = phase_results[phase_name].ξ
        if ξ[dst_idx] != 0.0
            τ = -1.0 / ξ[dst_idx]
            println("  $(rpad(phase_name, 10)) τ = $(round(τ, digits=1)) hr")
        else
            println("  $(rpad(phase_name, 10)) no decay term")
        end
    end
end

println("\nInjection terms by phase:")
bs_idx = findfirst(==("Bs"), term_names)
vbs_idx = findfirst(==("V*Bs"), term_names)
nvbs_idx = findfirst(==("n*V*Bs"), term_names)
for phase_name in ["quiet", "main", "recovery"]
    if haskey(phase_results, phase_name)
        ξ = phase_results[phase_name].ξ
        parts = String[]
        for (name, idx) in [("Bs", bs_idx), ("V*Bs", vbs_idx), ("n*V*Bs", nvbs_idx)]
            if idx !== nothing && ξ[idx] != 0.0
                push!(parts, "$(name)=$(round(ξ[idx], sigdigits=3))")
            end
        end
        println("  $(rpad(phase_name, 10)) $(isempty(parts) ? "none" : join(parts, ", "))")
    end
end

println("\n" * "=" ^ 60)
println("Phase C Complete")
println("=" ^ 60)
println("Outputs:")
for f in ["phase_dependent_real_coefficients.csv", "switching_model_metrics.csv"]
    fp = joinpath(DATA_DIR, f)
    if isfile(fp)
        println("  ✓ $(f) ($(round(filesize(fp)/1e3, digits=1)) KB)")
    else
        println("  ✗ $(f) — MISSING")
    end
end
