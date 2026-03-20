#!/usr/bin/env julia
# fix_synthetic_validation.jl — Address M2 (Pareto knee) and M3 (ensemble sparsity)
#
# 1. Extend lambda sweep to higher values so sweep reaches 2 terms
# 2. Run ensemble SINDy at the lambda that produces 2 terms
# 3. Save updated CSVs and regenerate Pareto figure

using SolarSINDy
using CSV, DataFrames
using PlotlySupply
import PlotlyKaleido
using Statistics, Random, LinearAlgebra

PlotlyKaleido.start()
const savefig = PlotlyKaleido.savefig

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const SINGLE_W = 504
const SINGLE_H = 360
const DOUBLE_W = 1008

# ============================================================
# 1. Regenerate synthetic data (same seed as original)
# ============================================================
println("--- Generating synthetic storm data ---")
swd, event = generate_synthetic_storm(seed=42, dt=0.25, noise_level=0.03)
data, dDst = prepare_sindy_data(swd, 0.25; smooth_window=11)
lib = build_solar_wind_library(max_poly_order=2, include_trig=true,
                                include_cross=true, include_known=true)
term_names = get_term_names(lib)

# Build library matrix
Θ = evaluate_library(lib, data)

# ============================================================
# 2. Extended lambda sweep (up to 10000)
# ============================================================
println("--- Running extended lambda sweep ---")
lambdas = 10.0 .^ range(-3, 4, length=100)  # 0.001 to 10000
sweep_results = sweep_lambda(Θ, dDst, lambdas; normalize=true)

# Print sweep summary
println("  Lambda sweep results:")
for s in sweep_results
    if s.n_terms <= 10 || s.n_terms == 21
        println("    λ=$(round(s.λ, sigdigits=4)), n_terms=$(s.n_terms), rmse=$(round(s.rmse, sigdigits=4))")
    end
end

# Find the lambda that gives exactly 2 terms
two_term_lambdas = filter(s -> s.n_terms == 2, sweep_results)
if !isempty(two_term_lambdas)
    lambda_2term = two_term_lambdas[1].λ
    println("\n  λ for 2-term model: $(round(lambda_2term, sigdigits=4))")
    println("  2-term RMSE: $(round(two_term_lambdas[1].rmse, sigdigits=4))")
    println("  2-term coefficients:")
    ξ_2 = two_term_lambdas[1].ξ
    for (i, name) in enumerate(term_names)
        if abs(ξ_2[i]) > 0
            println("    $name: $(round(ξ_2[i], sigdigits=4))")
        end
    end
else
    println("  WARNING: No lambda produces exactly 2 terms!")
    # Find the minimum number of terms achieved
    min_terms = minimum(s.n_terms for s in sweep_results if s.n_terms > 0)
    println("  Minimum terms achieved: $min_terms")
end

# Save extended lambda sweep
df_sweep = DataFrame(
    lambda = [s.λ for s in sweep_results],
    n_terms = [s.n_terms for s in sweep_results],
    rmse = [s.rmse for s in sweep_results]
)
CSV.write("paper/data/lambda_sweep.csv", df_sweep)
println("  Saved paper/data/lambda_sweep.csv")

# ============================================================
# 3. Ensemble SINDy at high lambda (targeting 2-term model)
# ============================================================
if !isempty(two_term_lambdas)
    lambda_ens_high = lambda_2term
else
    # Use highest lambda that still gives some terms
    nz_results = filter(s -> s.n_terms > 0, sweep_results)
    lambda_ens_high = nz_results[end].λ
end

println("\n--- Running ensemble SINDy at high λ=$(round(lambda_ens_high, sigdigits=4)) ---")
median_ξ_high, inclusion_prob_high, all_ξ_high = ensemble_sindy(
    data, lib, dDst;
    λ=lambda_ens_high, n_models=500, subsample_frac=0.8, seed=42)

println("  Ensemble results at high λ:")
for (i, name) in enumerate(term_names)
    if inclusion_prob_high[i] > 0.01
        println("    $name: π=$(round(inclusion_prob_high[i], digits=3)), ξ=$(round(median_ξ_high[i], sigdigits=4))")
    end
end

# Count terms with pi >= 0.90
n_high_pi = count(inclusion_prob_high .>= 0.90)
println("  Terms with π ≥ 0.90: $n_high_pi")

# Save high-lambda ensemble results
df_ens_high = DataFrame(
    term = term_names,
    median_coefficient = median_ξ_high,
    inclusion_probability = inclusion_prob_high
)
CSV.write("paper/data/ensemble_inclusion_high_lambda.csv", df_ens_high)
println("  Saved paper/data/ensemble_inclusion_high_lambda.csv")

# ============================================================
# 4. Also keep original ensemble at λ=5.0 for reference
# ============================================================
println("\n--- Running ensemble SINDy at original λ=5.0 ---")
median_ξ_orig, inclusion_prob_orig, all_ξ_orig = ensemble_sindy(
    data, lib, dDst;
    λ=5.0, n_models=200, subsample_frac=0.8, seed=42)

df_ens_orig = DataFrame(
    term = term_names,
    median_coefficient = median_ξ_orig,
    inclusion_probability = inclusion_prob_orig
)
CSV.write("paper/data/ensemble_inclusion.csv", df_ens_orig)
println("  Saved paper/data/ensemble_inclusion.csv")

# ============================================================
# 5. Regenerate Pareto figure with extended sweep
# ============================================================
println("\n--- Regenerating Pareto figure ---")
pareto_dict = Dict{Int,Float64}()
for s in sweep_results
    if s.n_terms > 0
        if !haskey(pareto_dict, s.n_terms) || s.rmse < pareto_dict[s.n_terms]
            pareto_dict[s.n_terms] = s.rmse
        end
    end
end
pareto_terms = sort(collect(keys(pareto_dict)))
pareto_rmse = [pareto_dict[k] for k in pareto_terms]

println("  Pareto front:")
for (t, r) in zip(pareto_terms, pareto_rmse)
    println("    $t terms: RMSE=$(round(r, sigdigits=4))")
end

fig = plot_scatter(Float64.(pareto_terms), pareto_rmse;
    xlabel="Number of Active Terms",
    ylabel="RMSE [nT/hr]",
    mode="lines+markers", color=COLORS[1], dash=DASHES[1],
    linewidth=2, marker_size=8)
set_legend!(fig; position=:topright)
savefig(fig, "paper/figs/fig_results_pareto_front.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_pareto_front.pdf")

println("\n=== Synthetic validation fix complete ===")
