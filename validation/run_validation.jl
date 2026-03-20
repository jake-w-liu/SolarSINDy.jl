#!/usr/bin/env julia
# run_validation.jl — Phase 1 heuristic validation of SINDy for solar wind coupling
#
# Generates:
#   paper/figs/heuristic_sindy_recovery.pdf   — SINDy recovers Burton equation from synthetic data
#   paper/figs/heuristic_model_comparison.pdf  — SINDy vs Burton vs O'Brien prediction comparison
#   paper/figs/heuristic_pareto_front.pdf      — Parsimony vs accuracy (lambda sweep)
#   paper/figs/heuristic_ensemble_inclusion.pdf — Ensemble SINDy term inclusion probabilities
#   paper/figs/heuristic_phase_dependent.pdf   — Phase-dependent SINDy discovery
#   paper/data/validation_data.csv             — Synthetic storm time series
#   paper/data/sindy_coefficients.csv          — Discovered SINDy coefficients
#   paper/data/model_metrics.csv               — Model comparison metrics
#   paper/data/lambda_sweep.csv                — Lambda sweep results
#   paper/data/ensemble_inclusion.csv          — Ensemble inclusion probabilities

using SolarSINDy
using CSV, DataFrames
using PlotlySupply
import PlotlyKaleido
using Statistics, Random, LinearAlgebra

PlotlyKaleido.start()

# Resolve savefig ambiguity
const savefig = PlotlyKaleido.savefig

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const SINGLE_W = 504   # IEEE single col width
const SINGLE_H = 360
const DOUBLE_W = 1008

# ============================================================
# 1. Generate synthetic storm data
# ============================================================
println("--- Generating synthetic storm data ---")
swd, event = generate_synthetic_storm(seed=42, dt=0.25, noise_level=0.03)
data, dDst = prepare_sindy_data(swd, 0.25; smooth_window=11)

# Save validation data CSV
df_data = DataFrame(
    time_hr = swd.t,
    V_kms = swd.V,
    Bz_nT = swd.Bz,
    By_nT = swd.By,
    n_cm3 = swd.n,
    Pdyn_nPa = swd.Pdyn,
    Dst_nT = swd.Dst,
    Dst_star_nT = swd.Dst_star,
    dDst_star_dt_nThr = dDst
)
CSV.write("paper/data/validation_data.csv", df_data)
println("  Saved paper/data/validation_data.csv")

# ============================================================
# 2. SINDy Discovery — recover Burton from synthetic data
# ============================================================
println("--- Running SINDy discovery ---")
lib = build_solar_wind_library(max_poly_order=2, include_trig=true,
                                include_cross=true, include_known=true)
term_names = get_term_names(lib)

# Discover with higher sparsity to get sparse solution
ξ, active_terms, Θ = sindy_discover(data, lib, dDst; λ=5.0, normalize=true)

println("  Discovered $(length(active_terms)) active terms:")
for (name, coeff) in sort(collect(active_terms), by=x->abs(x[2]), rev=true)
    println("    $name: $(round(coeff, sigdigits=4))")
end

# Save coefficients
df_coeff = DataFrame(
    term = term_names,
    coefficient = ξ,
    active = abs.(ξ) .> 0
)
CSV.write("paper/data/sindy_coefficients.csv", df_coeff)
println("  Saved paper/data/sindy_coefficients.csv")

# ============================================================
# 3. Model predictions — SINDy vs Burton vs O'Brien
# ============================================================
println("--- Computing model predictions ---")

# SINDy prediction (of dDst/dt)
dDst_sindy = Θ * ξ

# Burton baseline prediction
Bs = halfwave_rectify(swd.Bz)
dDst_burton = burton_model(swd.V, Bs, data["Dst_star"])

# O'Brien-McPherron baseline
dDst_obrien = obrien_mcpherron_model(swd.V, Bs, data["Dst_star"])

# Forward simulation comparison (Dst time series)
Dst_sindy_sim = zeros(length(swd.t))
Dst_sindy_sim[1] = swd.Dst_star[1]
dt = 0.25
for k in 1:length(swd.t)-1
    # Build data dict for single timestep
    d_k = Dict{String,Vector{Float64}}()
    for (key, val) in data
        d_k[key] = [val[k]]
    end
    d_k["Dst_star"] = [Dst_sindy_sim[k]]
    pred_k = sindy_predict(ξ, lib, d_k)
    Dst_sindy_sim[k+1] = Dst_sindy_sim[k] + pred_k[1] * dt
end

Dst_burton_sim = simulate_burton(swd.V, Bs, dt)
Dst_obrien_sim = simulate_obrien(swd.V, Bs, dt)

# Metrics
ms_sindy = metrics_summary(dDst_sindy, dDst; name="SINDy")
ms_burton = metrics_summary(dDst_burton, dDst; name="Burton")
ms_obrien = metrics_summary(dDst_obrien, dDst; name="O'Brien-McPherron")

# Forward sim metrics (against observed Dst*)
ms_sindy_fwd = metrics_summary(Dst_sindy_sim, swd.Dst_star; name="SINDy (fwd)")
ms_burton_fwd = metrics_summary(Dst_burton_sim, swd.Dst_star; name="Burton (fwd)")
ms_obrien_fwd = metrics_summary(Dst_obrien_sim, swd.Dst_star; name="O'Brien (fwd)")

println("\n  === dDst/dt Prediction Metrics ===")
for ms in [ms_sindy, ms_burton, ms_obrien]
    println("  $(ms.name): RMSE=$(round(ms.rmse, digits=3)), r=$(round(ms.corr, digits=4)), PE=$(round(ms.pe, digits=4))")
end
println("\n  === Forward Simulation Metrics ===")
for ms in [ms_sindy_fwd, ms_burton_fwd, ms_obrien_fwd]
    println("  $(ms.name): RMSE=$(round(ms.rmse, digits=3)), r=$(round(ms.corr, digits=4)), PE=$(round(ms.pe, digits=4))")
end

# Save metrics CSV
df_metrics = DataFrame(
    model = [ms_sindy.name, ms_burton.name, ms_obrien.name,
             ms_sindy_fwd.name, ms_burton_fwd.name, ms_obrien_fwd.name],
    rmse = [ms_sindy.rmse, ms_burton.rmse, ms_obrien.rmse,
            ms_sindy_fwd.rmse, ms_burton_fwd.rmse, ms_obrien_fwd.rmse],
    correlation = [ms_sindy.corr, ms_burton.corr, ms_obrien.corr,
                   ms_sindy_fwd.corr, ms_burton_fwd.corr, ms_obrien_fwd.corr],
    prediction_efficiency = [ms_sindy.pe, ms_burton.pe, ms_obrien.pe,
                             ms_sindy_fwd.pe, ms_burton_fwd.pe, ms_obrien_fwd.pe],
    n_points = [ms_sindy.n_points, ms_burton.n_points, ms_obrien.n_points,
                ms_sindy_fwd.n_points, ms_burton_fwd.n_points, ms_obrien_fwd.n_points]
)
CSV.write("paper/data/model_metrics.csv", df_metrics)
println("  Saved paper/data/model_metrics.csv")

# ============================================================
# 4. Lambda sweep — Pareto front
# ============================================================
println("--- Running lambda sweep ---")
lambdas = 10.0 .^ range(-3, 2, length=50)
sweep_results = sweep_lambda(Θ, dDst, lambdas; normalize=true)

n_terms_sweep = [s.n_terms for s in sweep_results]
rmse_sweep = [s.rmse for s in sweep_results]

df_sweep = DataFrame(
    lambda = [s.λ for s in sweep_results],
    n_terms = n_terms_sweep,
    rmse = rmse_sweep
)
CSV.write("paper/data/lambda_sweep.csv", df_sweep)
println("  Saved paper/data/lambda_sweep.csv")

# ============================================================
# 5. Ensemble SINDy
# ============================================================
println("--- Running ensemble SINDy ---")
median_ξ, inclusion_prob, all_ξ = ensemble_sindy(data, lib, dDst;
    λ=5.0, n_models=200, subsample_frac=0.8, seed=42)

df_ensemble = DataFrame(
    term = term_names,
    median_coefficient = median_ξ,
    inclusion_probability = inclusion_prob
)
CSV.write("paper/data/ensemble_inclusion.csv", df_ensemble)
println("  Saved paper/data/ensemble_inclusion.csv")

# ============================================================
# 6. Phase-dependent SINDy
# ============================================================
println("--- Running phase-dependent SINDy ---")
phases = identify_storm_phases(data["Dst_star"], dDst)
phase_names = ["Quiet", "Main Phase", "Recovery"]

phase_results = Dict{String,Dict{String,Float64}}()
for (p, pname) in enumerate(phase_names)
    idx = findall(phases .== p)
    if length(idx) < 10
        println("  Skipping $pname (only $(length(idx)) points)")
        continue
    end
    # Build sub-data
    sub_data = Dict{String,Vector{Float64}}()
    for (key, val) in data
        sub_data[key] = val[idx]
    end
    _, active_p, _ = sindy_discover(sub_data, lib, dDst[idx]; λ=3.0, normalize=true)
    phase_results[pname] = active_p
    println("  $pname: $(length(active_p)) terms — $(join(keys(active_p), ", "))")
end

# ============================================================
# FIGURES
# ============================================================
println("\n--- Generating figures ---")

# --- Figure 1: SINDy Recovery — dDst/dt comparison ---
fig1 = plot_scatter(swd.t, dDst;
    xlabel="Time [hr]", ylabel=raw"$\mathrm{d}Dst^*/\mathrm{d}t$ [nT/hr]",
    mode="lines", color=COLORS[1], dash=DASHES[1],
    legend="Observed", linewidth=2)
plot_scatter!(fig1, swd.t, dDst_sindy;
    color=COLORS[2], dash=DASHES[2], mode="lines",
    legend="SINDy", linewidth=2)
plot_scatter!(fig1, swd.t, dDst_burton;
    color=COLORS[3], dash=DASHES[3], mode="lines",
    legend="Burton", linewidth=2)
set_legend!(fig1; position=:bottomleft)
savefig(fig1, "paper/figs/heuristic_sindy_recovery.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved heuristic_sindy_recovery.pdf")

# --- Figure 2: Forward simulation comparison ---
fig2 = plot_scatter(swd.t, swd.Dst_star;
    xlabel="Time [hr]", ylabel=raw"$Dst^*$ [nT]",
    mode="lines", color=COLORS[1], dash=DASHES[1],
    legend="Observed", linewidth=2)
plot_scatter!(fig2, swd.t, Dst_sindy_sim;
    color=COLORS[2], dash=DASHES[2], mode="lines",
    legend="SINDy", linewidth=2)
plot_scatter!(fig2, swd.t, Dst_burton_sim;
    color=COLORS[3], dash=DASHES[3], mode="lines",
    legend="Burton", linewidth=2)
plot_scatter!(fig2, swd.t, Dst_obrien_sim;
    color=COLORS[4], dash=DASHES[4], mode="lines",
    legend="O'Brien-McPherron", linewidth=2)
set_legend!(fig2; position=:bottomleft)
savefig(fig2, "paper/figs/heuristic_model_comparison.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved heuristic_model_comparison.pdf")

# --- Figure 3: Pareto front ---
# Get unique (n_terms, best rmse) pairs
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

fig3 = plot_scatter(Float64.(pareto_terms), pareto_rmse;
    xlabel="Number of Active Terms",
    ylabel="RMSE [nT/hr]",
    mode="lines+markers", color=COLORS[1], dash=DASHES[1],
    linewidth=2, marker_size=8)
set_legend!(fig3; position=:topright)
savefig(fig3, "paper/figs/heuristic_pareto_front.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved heuristic_pareto_front.pdf")

# --- Figure 4: Ensemble inclusion probabilities ---
# Sort by inclusion probability, show top terms
sort_idx_ens = sortperm(inclusion_prob, rev=true)
top_n = min(12, length(inclusion_prob))
top_idx_ens = sort_idx_ens[1:top_n]

fig4 = plot_bar(term_names[top_idx_ens], inclusion_prob[top_idx_ens];
    xlabel="Library Term",
    ylabel="Inclusion Probability",
    color=COLORS[1])
set_legend!(fig4; position=:topright)
savefig(fig4, "paper/figs/heuristic_ensemble_inclusion.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved heuristic_ensemble_inclusion.pdf")

# --- Figure 5: Phase-dependent — Dst* decay coefficient comparison ---
# Extract the Dst_star coefficient for each phase (key physical parameter: decay rate 1/τ)
# Also show top 5 terms with largest magnitude difference across phases
all_phase_terms = Set{String}()
for (pname, active) in phase_results
    sorted = sort(collect(active), by=x->abs(x[2]), rev=true)
    for (t, c) in sorted[1:min(5, length(sorted))]
        push!(all_phase_terms, t)
    end
end
top_terms = sort(collect(all_phase_terms))
n_top = min(8, length(top_terms))
top_terms = top_terms[1:n_top]

# Plot as grouped scatter (one series per phase)
x_pos = collect(1.0:Float64(n_top))
first_phase = true
global fig5 = plot_scatter(x_pos .- 0.15,
    [abs(get(get(phase_results, "Quiet", Dict()), t, 0.0)) for t in top_terms];
    xlabel="Term Index", ylabel=raw"$|\xi|$",
    mode="markers", color=COLORS[1], marker_size=10,
    marker_symbol="circle", legend="Quiet", yscale="log")
plot_scatter!(fig5, x_pos,
    [abs(get(get(phase_results, "Main Phase", Dict()), t, 0.0)) for t in top_terms];
    color=COLORS[2], mode="markers", marker_size=10,
    marker_symbol="square", legend="Main Phase")
plot_scatter!(fig5, x_pos .+ 0.15,
    [abs(get(get(phase_results, "Recovery", Dict()), t, 0.0)) for t in top_terms];
    color=COLORS[3], mode="markers", marker_size=10,
    marker_symbol="diamond", legend="Recovery")
set_legend!(fig5; position=:topright)
savefig(fig5, "paper/figs/heuristic_phase_dependent.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved heuristic_phase_dependent.pdf")
println("  Phase terms plotted: ", join(top_terms, ", "))

println("\n=== Validation complete ===")
println("Figures: paper/figs/heuristic_*.pdf")
println("Data:    paper/data/*.csv")
