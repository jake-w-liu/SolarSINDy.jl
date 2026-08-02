#!/usr/bin/env julia
# generate_real_figures.jl — Generate legacy-name AISR diagnostic figures
#
# Figures:
#   fig_results_real_coefficients.pdf      — Point coefficients with subsample spread
#   fig_results_may2024_reconstruction.pdf — May 2024 superstorm hero figure
#   fig_results_real_ensemble.pdf          — Ensemble inclusion probabilities
#   fig_results_pareto_real.pdf            — Pareto front (λ sweep)
#   fig_results_cross_cycle.pdf            — Cross-cycle generalization
#   fig_results_phase_discovery.pdf         — Phase coefficient diagnostics
#   fig_results_coupled_network.pdf        — Coupled Dst-AE cross-index terms

using PlotlySupply
using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "output_paths.jl"))
const OUTPUT_PATHS = validation_output_paths()
OUTPUT_PATHS.mode == :canonical && error(
    "legacy unmanifested real-figure generation is prohibited in canonical runs",
)
const DATA_DIR = OUTPUT_PATHS.data
const FIGS_DIR = OUTPUT_PATHS.figs

# Publication constants
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
const W1 = 504   # IEEE single col
const H1 = 360
const W2 = 1008  # IEEE double col

const TERM_DISPLAY = Dict(
    "Dst_star" => "Dst<sup>*</sup>",
    "V^2" => "V<sup>2</sup>",
    "Bs^2" => "B<sub>s</sub><sup>2</sup>",
    "n^2" => "n<sup>2</sup>",
    "V*Bs" => "V B<sub>s</sub>",
    "n*V" => "n V",
    "n*Bs" => "n B<sub>s</sub>",
    "Pdyn*Bs" => "P<sub>dyn</sub> B<sub>s</sub>",
    "n*V*Bs" => "n V B<sub>s</sub>",
    "sin(θ_c/2)" => "sin(θ<sub>c</sub>/2)",
    "sin²(θ_c/2)" => "sin<sup>2</sup>(θ<sub>c</sub>/2)",
    "sin⁴(θ_c/2)" => "sin<sup>4</sup>(θ<sub>c</sub>/2)",
    "sin^(8/3)(θ_c/2)" => "sin<sup>8/3</sup>(θ<sub>c</sub>/2)",
    "V*sin²(θ_c/2)" => "V sin<sup>2</sup>(θ<sub>c</sub>/2)",
    "Newell_d_Φ" => "dΦ<sub>N</sub>/dt",
)

display_terms(terms) = [get(TERM_DISPLAY, String(term), String(term)) for term in terms]

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
    push!(pareto_rmse, minimum(sub.mean_validation_rmse_nt))
end

fig = plot_scatter(pareto_n, pareto_rmse;
    xlabel="Number of Active Terms",
    ylabel="Mean validation RMSE [nT]",
    mode="lines+markers", color=COLORS[1],
    marker_size=8, linewidth=2,
    legend="Best per term count")

# Mark the validation candidate selected by the one-standard-error rule.  Its
# full-training refit can contain a different number of terms.
selected_terms = count(!iszero, CSV.read(
    joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv"), DataFrame,
).coefficient)
selected_rows = filter(row -> row.selected, sweep)
nrow(selected_rows) == 1 || error("lambda sweep must contain exactly one selected row")
selected_row = only(eachrow(selected_rows))
plot_scatter!(fig, [selected_row.n_terms], [selected_row.mean_validation_rmse_nt];
    mode="markers", color=COLORS[2], marker_size=14,
    marker_symbol="diamond",
    legend="Selected ($(selected_row.n_terms) validation; $(selected_terms) refit)")

# Eight-position PDF audit: top-right has zero visible trace overlap.
set_legend!(fig; position=:topright)
PlotlySupply.savefig(fig, joinpath(FIGS_DIR, "fig_results_pareto_real.pdf");
        width=W1, height=420)
println("  Saved: fig_results_pareto_real.pdf")

# ============================================================
# Figure 2: Column-normalized coefficients with empirical subsample spread
# ============================================================
println("--- Figure: Normalized coefficients with empirical subsample spread ---")
ensemble = CSV.read(joinpath(DATA_DIR, "real_ensemble_inclusion.csv"), DataFrame)
discovery = CSV.read(joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv"), DataFrame)
design_norms = CSV.read(joinpath(DATA_DIR, "real_design_column_norms.csv"), DataFrame)

# Filter to terms with inclusion > 0.5
active_mask = ensemble.inclusion_probability .> 0.5
active_terms = String.(ensemble.term[active_mask])
coefficient_by_term = Dict(String(row.term) => Float64(row.coefficient)
                           for row in eachrow(discovery))
norm_by_term = Dict(String(row.term) => Float64(row.training_column_l2_norm)
                    for row in eachrow(design_norms) if row.basis == "full")
all(haskey(coefficient_by_term, term) && haskey(norm_by_term, term)
    for term in active_terms) || error("coefficient diagnostic term alignment failed")
active_norms = [norm_by_term[term] for term in active_terms]
active_disc = [coefficient_by_term[term] for term in active_terms] .* active_norms
active_median = Float64.(
    ensemble.conditional_nonzero_median[active_mask],
) .* active_norms
active_ci_lo = Float64.(
    ensemble.conditional_nonzero_empirical_q025[active_mask],
) .* active_norms
active_ci_hi = Float64.(
    ensemble.conditional_nonzero_empirical_q975[active_mask],
) .* active_norms

# Physical coefficients have term-dependent units. Multiplication by the
# training-column L2 norm puts the displayed values on the common scale used
# for STLSQ thresholding and avoids a dimensionally invalid raw comparison.
order = sortperm(abs.(active_disc), rev=true)
sorted_terms = active_terms[order]
sorted_disc = active_disc[order]
sorted_median = active_median[order]
sorted_lo = active_ci_lo[order]
sorted_hi = active_ci_hi[order]

n_bars = length(sorted_terms)
x_pos = collect(1:n_bars)

fig2 = plot_bar(x_pos, sorted_disc;
    xlabel="Library Term", ylabel="Column-normalized coefficient",
    color=COLORS[1], legend="Selected full refit")

range_x = Float64[]
range_y = Float64[]
for (x, lower, upper) in zip(x_pos, sorted_lo, sorted_hi)
    isfinite(lower) && isfinite(upper) || continue
    append!(range_x, (Float64(x), Float64(x), NaN))
    append!(range_y, (lower, upper, NaN))
end
plot_scatter!(fig2, range_x, range_y;
    mode="lines", color=COLORS[6], linewidth=4,
    legend="Conditional 2.5 to 97.5% range")
plot_scatter!(fig2, x_pos, sorted_median;
    mode="markers", color=COLORS[2], marker_size=7,
    marker_symbol="circle", legend="Conditional median")

fig2.layout.fields[:xaxis] = merge(
    get(fig2.layout.fields, :xaxis, Dict{Symbol,Any}()),
    Dict{Symbol,Any}(
        :tickmode => "array",
        :tickvals => x_pos,
        :ticktext => display_terms(sorted_terms),
        :tickangle => -45,
    ),
)

# Eight-position PDF audit: top-right has zero visible trace overlap.
set_legend!(fig2; position=:topright)
PlotlySupply.savefig(fig2, joinpath(FIGS_DIR, "fig_results_real_coefficients.pdf");
        width=W2, height=480)
println("  Saved: fig_results_real_coefficients.pdf")

# ============================================================
# Figure 3: May 2024 Superstorm Reconstruction (Hero Figure)
# ============================================================
println("--- Figure: May 2024 Superstorm ---")
may = CSV.read(joinpath(DATA_DIR, "may2024_reconstruction.csv"), DataFrame)

# title="" suppresses PlotlySupply's default package-name title, which its
# subplot constructor otherwise applies as a visible figure title.
sf = subplots(2, 1; sync=false, show=false, title="")

# Panel 1: Solar wind drivers
subplot!(sf, 1, 1)
plot_scatter!(sf, may.time_hr, may.v_kms;
    mode="lines", color=COLORS[5], linewidth=1.5,
    legend="V [km/s]")
ylabel!(sf, "V [km/s]")

# Panel 2: Dst reconstruction
subplot!(sf, 2, 1)
plot_scatter!(sf, may.time_hr, may.dst_star_observed_nt;
    mode="lines", color="black", linewidth=2.5,
    legend="Observed")
plot_scatter!(sf, may.time_hr, may.dst_star_sindy_nt;
    mode="lines", color=COLORS[1], linewidth=2,
    dash="solid", legend="SINDy ($(selected_terms)-term)")
plot_scatter!(sf, may.time_hr, may.dst_star_burton_published_nt;
    mode="lines", color=COLORS[2], linewidth=2,
    dash="dash", legend="Burton (1975)")
plot_scatter!(sf, may.time_hr, may.dst_star_obrien_nt;
    mode="lines", color=COLORS[3], linewidth=2,
    dash="dashdot", legend="O'Brien-McP (2000)")
xlabel!(sf, "Time [hours from shared anchor]")
ylabel!(sf, raw"Dst* [nT]")

# Set up per-subplot legends, then override positions individually.
subplot_legends!(sf; position=:topright)
# Eight-position PDF audit: top-right (V) and bottom-right (Dst*) have zero
# visible trace overlap.
let leg2 = sf.fig.layout.fields[:legend2]
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
# Sort by inclusion frequency
order_inc = sortperm(ensemble.inclusion_probability, rev=true)
inc_terms = ensemble.term[order_inc]
inc_probs = ensemble.inclusion_probability[order_inc]

# Separate core (π ≥ 0.9) vs peripheral terms for distinct colors
core_mask = inc_probs .>= 0.9
n_all = length(inc_terms)
x_pos = collect(1:n_all)

# Core terms (high inclusion)
core_idx = findall(core_mask)
periph_idx = findall(.!core_mask)

fig4 = plot_bar(x_pos[core_idx], inc_probs[core_idx];
    xlabel="Library Term", ylabel="Inclusion Frequency",
    color=COLORS[1], legend="Core (π ≥ 0.9)")

if !isempty(periph_idx)
    plot_bar!(fig4, x_pos[periph_idx], inc_probs[periph_idx];
        color=COLORS[4], legend="Peripheral (π < 0.9)")
end

# Add threshold line at 0.9
plot_scatter!(fig4, [0.5, n_all + 0.5], [0.9, 0.9];
    mode="lines", color=COLORS[2], dash="dash", linewidth=1.5,
    legend="π = 0.9 threshold")

# Set x-axis tick labels to term names
fig4.layout.fields[:xaxis] = merge(
    get(fig4.layout.fields, :xaxis, Dict{Symbol,Any}()),
    Dict{Symbol,Any}(
        :tickmode => "array",
        :tickvals => x_pos,
        :ticktext => display_terms(inc_terms),
        :tickangle => -45,
    )
)

# Eight-position PDF audit: bottom-right has zero bar or threshold overlap.
set_legend!(fig4; position=:bottomright)
PlotlySupply.savefig(fig4, joinpath(FIGS_DIR, "fig_results_real_ensemble.pdf");
        width=W2, height=H1)
println("  Saved: fig_results_real_ensemble.pdf")

# ============================================================
# Figure 5: Cross-Cycle Generalization
# ============================================================
println("--- Figure: Cross-Cycle ---")
cross = CSV.read(joinpath(DATA_DIR, "cross_cycle_metrics.csv"), DataFrame)

# Compute mean RMSE by experiment × model
experiments = ["C20-22->C23", "even->odd", "C20-23->C25"]
experiment_labels = ["C20-22 to C23", "Even to odd", "C20-23 to C25"]
models = ["SINDy", "Burton", "BurtonFull", "OBrienMcP"]
model_labels = ["SINDy", "Burton", "Burton (full)", "O'Brien-McP"]

n_exp = length(experiments)
n_mod = length(models)

# Build data for grouped bar chart
x_base = collect(1:n_exp)
rmse_matrix = zeros(n_exp, n_mod)
for (j, model) in enumerate(models)
    for (i, exp) in enumerate(experiments)
        sub = filter(row -> row.experiment == exp && row.model == model, cross)
        isempty(sub) && error("cross-cycle figure is missing $model rows for $exp")
        values = Float64.(sub.rmse_nt)
        all(isfinite, values) || error(
            "cross-cycle figure has non-finite $model RMSE values for $exp",
        )
        rmse_matrix[i, j] = mean(values)
    end
end

offsets = [-0.30, -0.10, 0.10, 0.30]
fig5 = plot_bar(x_base .+ offsets[1], rmse_matrix[:, 1];
    xlabel="", ylabel="Mean RMSE [nT]",
    color=COLORS[1], legend=model_labels[1])
plot_bar!(fig5, x_base .+ offsets[2], rmse_matrix[:, 2];
    color=COLORS[2], legend=model_labels[2])
plot_bar!(fig5, x_base .+ offsets[3], rmse_matrix[:, 3];
    color=COLORS[4], legend=model_labels[3])
plot_bar!(fig5, x_base .+ offsets[4], rmse_matrix[:, 4];
    color=COLORS[3], legend=model_labels[4])

fig5.layout.fields[:xaxis] = merge(
    get(fig5.layout.fields, :xaxis, Dict{Symbol,Any}()),
    Dict{Symbol,Any}(
        :tickmode => "array",
        :tickvals => x_base,
        :ticktext => experiment_labels,
        :tickangle => -20,
    ),
)

# All eight standard inside positions overlap at least one bar. A single-row
# legend at the lower edge preserves every bar top, which carries the RMSE
# comparison, and limits overlap to the common zero baseline.
set_legend!(fig5; position=:bottom)
let legend = fig5.layout.fields[:legend]
    fields = legend isa AbstractDict ? legend : legend.fields
    fields[:orientation] = "h"
    fields[:font] = attr(size=9)
    fields[:x] = 0.5
    fields[:xanchor] = "center"
    fields[:y] = 0.01
    fields[:yanchor] = "bottom"
end
PlotlySupply.savefig(fig5, joinpath(FIGS_DIR, "fig_results_cross_cycle.pdf");
        width=W1, height=420)
println("  Saved: fig_results_cross_cycle.pdf")

# ============================================================
# Figure 6: Phase-Dependent Coefficients (Real Data)
# ============================================================
println("--- Figure: Phase-Dependent Coefficients ---")
phase_coef = CSV.read(joinpath(DATA_DIR, "phase_dependent_real_coefficients.csv"), DataFrame)

phases = ["quiet", "main", "recovery"]
phase_labels = ["Quiet", "Main Phase", "Recovery"]

function phase_term_rows(term)
    return [only(filter(row -> row.phase == phase && row.term == term, phase_coef))
            for phase in phases]
end

function phase_interval_vectors(rows)
    x = Float64[]
    y = Float64[]
    for (index, row) in enumerate(rows)
        lower = Float64(row.conditional_nonzero_quantile_025)
        upper = Float64(row.conditional_nonzero_quantile_975)
        if isfinite(lower) && isfinite(upper)
            append!(x, (Float64(index), Float64(index), NaN))
            append!(y, (lower, upper, NaN))
        end
    end
    return x, y
end

dst_rows = phase_term_rows("Dst_star")
clock_rows = phase_term_rows("sin^(8/3)(θ_c/2)")
phase_x = collect(1:length(phases))

fig6 = subplots(2, 1; sync=false, show=false, title="")
for (panel, rows, ylabel_text) in (
    (1, dst_rows, "Dst* coefficient [hr^-1]"),
    (2, clock_rows, "sin<sup>8/3</sup>(θ<sub>c</sub>/2) coefficient [nT/hr]"),
)
    subplot!(fig6, panel, 1)
    phase_range_x, phase_range_y = phase_interval_vectors(rows)
    isempty(phase_range_x) || plot_scatter!(fig6, phase_range_x, phase_range_y;
        mode="lines", color=COLORS[6], linewidth=4,
        legend="Conditional 2.5 to 97.5% range")
    plot_scatter!(fig6, phase_x, Float64.(getproperty.(rows, :point_coefficient));
        mode="markers", color=COLORS[1], marker_size=10,
        marker_symbol="circle", legend="Point refit")
    plot_scatter!(fig6, [0.7, 3.3], [0.0, 0.0];
        mode="lines", color="black", dash="dash", linewidth=1,
        legend="Zero")
    ylabel!(fig6, ylabel_text)
end
xlabel!(fig6, "Phase")
# Eight-position PDF audit: right placement avoids points, intervals, and zero lines.
subplot_legends!(fig6; position=:right)
for key in (:xaxis, :xaxis2)
    axis = fig6.fig.layout.fields[key]
    fields = axis isa AbstractDict ? axis : axis.fields
    fields[:tickmode] = "array"
    fields[:tickvals] = phase_x
    fields[:ticktext] = phase_labels
end
PlotlySupply.savefig(fig6, joinpath(FIGS_DIR, "fig_results_phase_discovery.pdf");
        width=W2, height=720)
println("  Saved: fig_results_phase_discovery.pdf")

# ============================================================
# Figure 7: Coupled Dst-AE Cross-Index Terms
# ============================================================
println("--- Figure: Coupled Cross-Index ---")
coupled = CSV.read(joinpath(DATA_DIR, "coupled_coefficients.csv"), DataFrame)

# Extract cross-index terms
dst_eq = filter(row -> row.equation == "dDst_star/dt", coupled)
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
