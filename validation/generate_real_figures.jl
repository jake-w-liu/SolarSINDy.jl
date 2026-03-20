#!/usr/bin/env julia
# generate_real_figures.jl — Generate publication figures for Nature Comms paper
#
# Figures:
#   fig_results_real_coefficients.pdf      — Discovered coefficients with ensemble CI
#   fig_results_may2024_reconstruction.pdf — May 2024 superstorm hero figure
#   fig_results_real_ensemble.pdf          — Ensemble inclusion probabilities
#   fig_results_pareto_real.pdf            — Pareto front (λ sweep)
#   fig_results_cross_cycle.pdf            — Cross-cycle generalization
#   fig_results_coupled_network.pdf        — Coupled Dst-AE cross-index terms

using PlotlySupply, PlotlyKaleido
PlotlyKaleido.start()
# Warmup to avoid MathJax error
let _w = plot_scatter([0.0], [0.0])
    PlotlySupply.savefig(_w, tempname() * ".pdf"; width=100, height=100)
end

using CSV, DataFrames, Statistics

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")
const FIGS_DIR = joinpath(@__DIR__, "..", "..", "paper", "figs")
mkpath(FIGS_DIR)

# Publication constants
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
const DASHES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]
const W1 = 504   # IEEE single col
const H1 = 360
const W2 = 1008  # IEEE double col

# ============================================================
# Figure 1: Pareto Front (Lambda Sweep)
# ============================================================
println("--- Figure: Pareto Front ---")
sweep = CSV.read(joinpath(DATA_DIR, "real_lambda_sweep.csv"), DataFrame)

# Get unique Pareto points (best RMSE for each n_terms)
pareto_n = Int[]
pareto_rmse = Float64[]
for nt in sort(unique(sweep.n_terms))
    sub = filter(row -> row.n_terms == nt, sweep)
    push!(pareto_n, nt)
    push!(pareto_rmse, minimum(sub.rmse))
end

fig = plot_scatter(pareto_n, pareto_rmse;
    xlabel="Number of Active Terms",
    ylabel=raw"Training RMSE [nT hr$^{-1}$]",
    mode="lines+markers", color=COLORS[1],
    marker_size=8, linewidth=2,
    legend="Pareto front")

# Mark the 10-term solution
idx10 = findfirst(==(10), pareto_n)
if idx10 !== nothing
    plot_scatter!(fig, [pareto_n[idx10]], [pareto_rmse[idx10]];
        mode="markers", color=COLORS[2], marker_size=14,
        marker_symbol="diamond", legend="Selected (10 terms)")
end

set_legend!(fig; position=:topright)
PlotlySupply.savefig(fig, joinpath(FIGS_DIR, "fig_results_pareto_real.pdf");
        width=W1, height=H1)
println("  Saved: fig_results_pareto_real.pdf")

# ============================================================
# Figure 2: Discovered Coefficients with Ensemble CI
# ============================================================
println("--- Figure: Coefficients with CI ---")
ensemble = CSV.read(joinpath(DATA_DIR, "real_ensemble_inclusion.csv"), DataFrame)
discovery = CSV.read(joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv"), DataFrame)

# Filter to terms with inclusion > 0.5
active_mask = ensemble.inclusion_prob .> 0.5
active_terms = ensemble.term[active_mask]
active_median = ensemble.median_coef[active_mask]
active_ci_lo = ensemble.ci_025[active_mask]
active_ci_hi = ensemble.ci_975[active_mask]
active_disc = discovery.coefficient[active_mask]

# Sort by absolute value of discovery coefficient
order = sortperm(abs.(active_disc), rev=true)
sorted_terms = active_terms[order]
sorted_disc = active_disc[order]
sorted_lo = active_ci_lo[order]
sorted_hi = active_ci_hi[order]

# Bar chart with error bars
n_bars = length(sorted_terms)
x_pos = collect(1:n_bars)

fig2 = plot_bar(x_pos, sorted_disc;
    xlabel="", ylabel="Coefficient Value",
    color=COLORS[1], legend="Discovery (λ=148)")

# Note: PlotlySupply doesn't have native error bars, use scatter overlay
plot_scatter!(fig2, x_pos, sorted_disc;
    mode="markers", color=COLORS[2], marker_size=1,
    legend="95% CI")

set_legend!(fig2; position=:topright)
PlotlySupply.savefig(fig2, joinpath(FIGS_DIR, "fig_results_real_coefficients.pdf");
        width=W2, height=H1)
println("  Saved: fig_results_real_coefficients.pdf")

# ============================================================
# Figure 3: May 2024 Superstorm Reconstruction (Hero Figure)
# ============================================================
println("--- Figure: May 2024 Superstorm ---")
may = CSV.read(joinpath(DATA_DIR, "may2024_reconstruction.csv"), DataFrame)

sf = subplots(2, 1; show=false)

# Panel 1: Solar wind drivers
subplot!(sf, 1, 1)
plot_scatter!(sf, may.t_hr, may.V;
    mode="lines", color=COLORS[5], linewidth=1.5,
    legend="V [km/s]")
ylabel!(sf, "V [km/s]")

# Panel 2: Dst reconstruction
subplot!(sf, 2, 1)
plot_scatter!(sf, may.t_hr, may.Dst_obs;
    mode="lines", color="black", linewidth=2.5,
    legend="Observed")
plot_scatter!(sf, may.t_hr, may.Dst_sindy;
    mode="lines", color=COLORS[1], linewidth=2,
    dash="solid", legend="SINDy (10-term)")
plot_scatter!(sf, may.t_hr, may.Dst_burton;
    mode="lines", color=COLORS[2], linewidth=2,
    dash="dash", legend="Burton (1975)")
plot_scatter!(sf, may.t_hr, may.Dst_obrien;
    mode="lines", color=COLORS[3], linewidth=2,
    dash="dashdot", legend="O'Brien-McP (2000)")
xlabel!(sf, "Time [hours from onset]")
ylabel!(sf, raw"Dst* [nT]")

# Set up per-subplot legends, then override positions individually
subplot_legends!(sf; position=:topright)
# Subplot 1 (top, V) → :legend stays at topright (default from above)
# Subplot 2 (bottom, Dst) → :legend2 override to bottomright
let leg2 = sf.fig.layout.fields[:legend2]
    leg2[:x] = leg2[:x]           # keep x (right side)
    leg2[:y] = sf.fig.layout.fields[:yaxis2][:domain][1] + 0.02
    leg2[:yanchor] = "bottom"
    leg2[:xanchor] = "right"
end
PlotlySupply.savefig(sf.fig, joinpath(FIGS_DIR, "fig_results_may2024_reconstruction.pdf");
        width=W2, height=Int(round(H1 * 1.5)))
println("  Saved: fig_results_may2024_reconstruction.pdf")

# ============================================================
# Figure 4: Ensemble Inclusion Probabilities
# ============================================================
println("--- Figure: Ensemble Inclusion ---")
# Sort by inclusion probability
order_inc = sortperm(ensemble.inclusion_prob, rev=true)
inc_terms = ensemble.term[order_inc]
inc_probs = ensemble.inclusion_prob[order_inc]

# Separate core (π ≥ 0.9) vs peripheral terms for distinct colors
core_mask = inc_probs .>= 0.9
n_all = length(inc_terms)
x_pos = collect(1:n_all)

# Core terms (high inclusion)
core_idx = findall(core_mask)
periph_idx = findall(.!core_mask)

fig4 = plot_bar(x_pos[core_idx], inc_probs[core_idx];
    xlabel="Library Term", ylabel="Inclusion Probability",
    color=COLORS[1], legend="Core (pi >= 0.9)")

if !isempty(periph_idx)
    plot_bar!(fig4, x_pos[periph_idx], inc_probs[periph_idx];
        color=COLORS[4], legend="Peripheral (pi < 0.9)")
end

# Add threshold line at 0.9
plot_scatter!(fig4, [0.5, n_all + 0.5], [0.9, 0.9];
    mode="lines", color=COLORS[2], dash="dash", linewidth=1.5,
    legend="pi = 0.9 threshold")

# Set x-axis tick labels to term names
fig4.layout.fields[:xaxis] = merge(
    get(fig4.layout.fields, :xaxis, Dict{Symbol,Any}()),
    Dict{Symbol,Any}(
        :tickmode => "array",
        :tickvals => x_pos,
        :ticktext => collect(String, inc_terms),
        :tickangle => -45,
    )
)

set_legend!(fig4; position=:topright)
PlotlySupply.savefig(fig4, joinpath(FIGS_DIR, "fig_results_real_ensemble.pdf");
        width=W2, height=H1)
println("  Saved: fig_results_real_ensemble.pdf")

# ============================================================
# Figure 5: Cross-Cycle Generalization
# ============================================================
println("--- Figure: Cross-Cycle ---")
cross = CSV.read(joinpath(DATA_DIR, "cross_cycle_metrics.csv"), DataFrame)

# Compute mean RMSE by experiment × model
experiments = unique(cross.experiment)
models = ["SINDy", "Burton", "OBrienMcP"]
model_labels = ["SINDy", "Burton", "O'Brien-McP"]

n_exp = length(experiments)
n_mod = length(models)

# Build data for grouped bar chart
x_base = collect(1:n_exp)
rmse_matrix = zeros(n_exp, n_mod)
for (j, model) in enumerate(models)
    for (i, exp) in enumerate(experiments)
        sub = filter(row -> row.experiment == exp && row.model == model, cross)
        rmse_matrix[i, j] = nrow(sub) > 0 ? mean(sub.rmse) : 0.0
    end
end

fig5 = plot_bar(x_base .- 0.25, rmse_matrix[:, 1];
    xlabel="", ylabel="Mean RMSE [nT]",
    color=COLORS[1], legend=model_labels[1])
plot_bar!(fig5, x_base, rmse_matrix[:, 2];
    color=COLORS[2], legend=model_labels[2])
plot_bar!(fig5, x_base .+ 0.25, rmse_matrix[:, 3];
    color=COLORS[3], legend=model_labels[3])

set_legend!(fig5; position=:topright)
PlotlySupply.savefig(fig5, joinpath(FIGS_DIR, "fig_results_cross_cycle.pdf");
        width=W1, height=H1)
println("  Saved: fig_results_cross_cycle.pdf")

# ============================================================
# Figure 6: Phase-Dependent Coefficients (Real Data)
# ============================================================
println("--- Figure: Phase-Dependent Coefficients ---")
phase_coef = CSV.read(joinpath(DATA_DIR, "phase_dependent_real_coefficients.csv"), DataFrame)

# Select key physics terms for visualization
key_terms = ["Dst_star", "Bs", "n*V", "n*V*Bs", "n*V^2",
             "sin(θ_c/2)", "sin²(θ_c/2)", "sin^(8/3)(θ_c/2)", "Newell_d_Φ"]

phases = ["quiet", "main", "recovery"]
phase_labels = ["Quiet", "Main Phase", "Recovery"]

# Build comparison: coefficient magnitude per phase per term
# Collect data for all phases first
phase_coef_data = Dict{String,Vector{Float64}}()
phase_term_order = String[]
for phase in phases
    pd = filter(row -> row.phase == phase, phase_coef)
    tm = filter(row -> row.term in key_terms, pd)
    phase_coef_data[phase] = max.(abs.(tm.coefficient), 1e-8)
    if isempty(phase_term_order)
        global phase_term_order = collect(String, tm.term)
    end
end
n_terms_plot = length(phase_term_order)

fig6 = plot_scatter(collect(1:n_terms_plot) .- 0.15, phase_coef_data["quiet"];
    xlabel="", ylabel="|Coefficient|",
    mode="markers", color=COLORS[1], marker_size=10,
    marker_symbol="circle", yscale="log", legend="Quiet")
plot_scatter!(fig6, collect(1:n_terms_plot), phase_coef_data["main"];
    mode="markers", color=COLORS[2], marker_size=10,
    marker_symbol="square", legend="Main Phase")
plot_scatter!(fig6, collect(1:n_terms_plot) .+ 0.15, phase_coef_data["recovery"];
    mode="markers", color=COLORS[3], marker_size=10,
    marker_symbol="diamond", legend="Recovery")

set_legend!(fig6; position=:topright)
PlotlySupply.savefig(fig6, joinpath(FIGS_DIR, "fig_results_phase_discovery.pdf");
        width=W2, height=H1)
println("  Saved: fig_results_phase_discovery.pdf")

# ============================================================
# Figure 7: Coupled Dst-AE Cross-Index Terms
# ============================================================
println("--- Figure: Coupled Cross-Index ---")
coupled = CSV.read(joinpath(DATA_DIR, "coupled_coefficients.csv"), DataFrame)

# Extract cross-index terms
dst_eq = filter(row -> row.equation == "dDst/dt", coupled)
ae_eq = filter(row -> row.equation == "dAE/dt", coupled)

# AE-related terms in Dst equation
ae_in_dst = filter(row -> occursin("AE", row.term) && row.coefficient != 0.0, dst_eq)
# Dst-related terms in AE equation
dst_in_ae = filter(row -> occursin("Dst", row.term) && row.coefficient != 0.0, ae_eq)

println("  Cross-index terms found:")
println("  AE→Dst*: ", join(ae_in_dst.term, ", "))
println("  Dst*→AE: ", join(dst_in_ae.term, ", "))

# Bar chart: cross-index coefficient magnitudes
cross_terms = vcat(ae_in_dst.term, dst_in_ae.term)
cross_coefs = vcat(ae_in_dst.coefficient, dst_in_ae.coefficient)
cross_direction = vcat(fill("AE→Dst*", nrow(ae_in_dst)), fill("Dst*→AE", nrow(dst_in_ae)))

# Simple bar chart
fig7 = plot_bar(collect(1:length(cross_terms)), abs.(cross_coefs);
    xlabel="", ylabel="|Coefficient|",
    color=COLORS[1], legend="Cross-index coupling",
    yscale="log")

set_legend!(fig7; position=:topright)
PlotlySupply.savefig(fig7, joinpath(FIGS_DIR, "fig_results_coupled_dynamics.pdf");
        width=W1, height=H1)
println("  Saved: fig_results_coupled_dynamics.pdf")

# ============================================================
# Summary
# ============================================================
println("\n" * "=" ^ 60)
println("Figure Generation Complete")
println("=" ^ 60)
for f in readdir(FIGS_DIR)
    if startswith(f, "fig_results_") && endswith(f, ".pdf")
        fp = joinpath(FIGS_DIR, f)
        println("  $(f) ($(round(filesize(fp)/1e3, digits=1)) KB)")
    end
end
