#!/usr/bin/env julia
# figure_generation.jl — Phase 3: Generate all results figures and CSV data
#
# Generates:
#   paper/figs/fig_results_multistorm_coefficients.pdf
#   paper/figs/fig_results_storm_reconstruction.pdf
#   paper/figs/fig_results_pareto_front.pdf
#   paper/figs/fig_results_ensemble_inclusion.pdf
#   paper/figs/fig_results_noise_sweep.pdf
#   paper/figs/fig_results_ablation.pdf
#   paper/figs/fig_results_phase_dependent.pdf
#   paper/figs/fig_results_scalability.pdf
#   paper/data/multistorm_*.csv, holdout_metrics.csv, noise_sweep.csv,
#   ablation_metrics.csv, phase_dependent_coefficients.csv, scalability_sweep.csv

using SolarSINDy
using CSV, DataFrames
using PlotlySupply
import PlotlyKaleido
using Statistics, Random, LinearAlgebra

PlotlyKaleido.start()
# Warmup export to avoid MathJax "MathMenu.js" error on first real figure
let _warmup = plot_scatter([0.0], [0.0])
    PlotlyKaleido.savefig(_warmup, tempname() * ".pdf"; width=100, height=100)
end

const savefig = PlotlyKaleido.savefig

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const SINGLE_W = 504   # IEEE single col width (3.5in @144DPI)
const SINGLE_H = 360   # ~2.5in @144DPI
const DOUBLE_W = 1008  # IEEE double col width (7.0in @144DPI)

# --- Helper: concatenate multiple storms ---
function concatenate_data(datasets)
    all_data = Dict{String,Vector{Float64}}()
    all_dDst = Float64[]
    for swd in datasets
        data_k, dDst_k = prepare_sindy_data(swd, 1.0; smooth_window=5)
        if isempty(all_dDst)
            for (key, val) in data_k
                all_data[key] = copy(val)
            end
        else
            for (key, val) in data_k
                append!(all_data[key], val)
            end
        end
        append!(all_dDst, dDst_k)
    end
    return all_data, all_dDst
end

# simulate_sindy is now exported from SolarSINDy — use it directly

# ============================================================
# Generate multi-storm dataset
# ============================================================
# Use FIXED Burton parameters (α=4.559e-3, τ=7.7) so SINDy can recover them.
# Vary solar wind drivers across storms for diversity.
println("=== Generating multi-storm dataset ===")

n_storms = 10
α_true = 4.559e-3
τ_true = 7.7

all_datasets = SolarWindData[]
all_events = StormEvent[]
for k in 1:n_storms
    swd, event = generate_synthetic_storm(
        seed=100+k, dt=1.0, duration=120.0,
        α=α_true, τ=τ_true, noise_level=0.05
    )
    push!(all_datasets, swd)
    push!(all_events, event)
end
println("  Generated $n_storms storms, α=$α_true, τ=$τ_true")
println("  Min Dst range: $(round(minimum(e.min_dst for e in all_events), digits=1)) to $(round(maximum(e.min_dst for e in all_events), digits=1)) nT")

# Concatenate all for training
all_data, all_dDst = concatenate_data(all_datasets)
lib = build_solar_wind_library(max_poly_order=2, include_trig=true,
                                include_cross=true, include_known=true)
term_names = get_term_names(lib)

# ============================================================
# 1. Multi-Storm SINDy Discovery
# ============================================================
println("\n=== Result 1: Multi-Storm SINDy Discovery ===")

# Use a higher λ to promote sparsity on multi-storm data
ξ_multi, active_multi, Θ_multi = sindy_discover(all_data, lib, all_dDst;
    λ=3.0, normalize=true)

n_active_multi = count(abs.(ξ_multi) .> 0)
println("  Multi-storm SINDy (λ=3.0): $n_active_multi active terms")
for (name, coeff) in sort(collect(active_multi), by=x->abs(x[2]), rev=true)
    println("    $name: $(round(coeff, sigdigits=4))")
end

# Save multi-storm coefficients
df_multi_coeff = DataFrame(
    term = term_names,
    coefficient = ξ_multi,
    active = abs.(ξ_multi) .> 0
)
CSV.write("paper/data/multistorm_coefficients.csv", df_multi_coeff)
println("  Saved paper/data/multistorm_coefficients.csv")

# Metrics on concatenated data
dDst_sindy_multi = Θ_multi * ξ_multi
Bs_all = halfwave_rectify(all_data["Bz"])
dDst_burton_multi = burton_model(all_data["V"], Bs_all, all_data["Dst_star"])
dDst_obrien_multi = obrien_mcpherron_model(all_data["V"], Bs_all, all_data["Dst_star"])

ms_s = metrics_summary(dDst_sindy_multi, all_dDst; name="SINDy")
ms_b = metrics_summary(dDst_burton_multi, all_dDst; name="Burton")
ms_o = metrics_summary(dDst_obrien_multi, all_dDst; name="O'Brien-McPherron")

println("  SINDy  RMSE=$(round(ms_s.rmse, digits=3)), r=$(round(ms_s.corr, digits=4)), PE=$(round(ms_s.pe, digits=4))")
println("  Burton RMSE=$(round(ms_b.rmse, digits=3)), r=$(round(ms_b.corr, digits=4)), PE=$(round(ms_b.pe, digits=4))")
println("  O'Brien RMSE=$(round(ms_o.rmse, digits=3)), r=$(round(ms_o.corr, digits=4)), PE=$(round(ms_o.pe, digits=4))")

df_multi_metrics = DataFrame(
    model = [ms_s.name, ms_b.name, ms_o.name],
    rmse = [ms_s.rmse, ms_b.rmse, ms_o.rmse],
    correlation = [ms_s.corr, ms_b.corr, ms_o.corr],
    prediction_efficiency = [ms_s.pe, ms_b.pe, ms_o.pe],
    n_active_terms = [n_active_multi, 2, 3]
)
CSV.write("paper/data/multistorm_metrics.csv", df_multi_metrics)
println("  Saved paper/data/multistorm_metrics.csv")

# Figure 1: Discovered coefficients bar chart
active_idx = findall(abs.(ξ_multi) .> 0)
act_names = term_names[active_idx]
act_vals = ξ_multi[active_idx]
sort_order = sortperm(abs.(act_vals), rev=true)
act_names = act_names[sort_order]
act_vals = act_vals[sort_order]

fig1 = plot_bar(act_names, act_vals;
    xlabel="Library Term",
    ylabel=raw"Coefficient $\xi$",
    color=COLORS[1])
set_legend!(fig1; position=:topright)
savefig(fig1, "paper/figs/fig_results_multistorm_coefficients.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_multistorm_coefficients.pdf")

# ============================================================
# 2. Held-Out Storm Reconstruction (Hero Figure)
# ============================================================
println("\n=== Result 2: Held-Out Storm Reconstruction ===")

# Hold out storm 5 (middle intensity) — train on rest
holdout_idx = 5
train_datasets = [all_datasets[k] for k in 1:n_storms if k != holdout_idx]
train_data, train_dDst = concatenate_data(train_datasets)

ξ_ho, _, _ = sindy_discover(train_data, lib, train_dDst; λ=3.0, normalize=true)

test_swd = all_datasets[holdout_idx]
test_data, test_dDst = prepare_sindy_data(test_swd, 1.0; smooth_window=5)
Bs_test = halfwave_rectify(test_swd.Bz)

# Forward simulations
Dst_sindy_ho = simulate_sindy(ξ_ho, lib, test_swd, 1.0)
Dst_burton_ho = simulate_burton(test_swd.V, Bs_test, 1.0)
Dst_obrien_ho = simulate_obrien(test_swd.V, Bs_test, 1.0)

ms_s_ho = metrics_summary(Dst_sindy_ho, test_swd.Dst_star; name="SINDy")
ms_b_ho = metrics_summary(Dst_burton_ho, test_swd.Dst_star; name="Burton")
ms_o_ho = metrics_summary(Dst_obrien_ho, test_swd.Dst_star; name="O'Brien")

println("  Held-out storm $holdout_idx (min Dst=$(round(all_events[holdout_idx].min_dst, digits=1)) nT):")
println("  SINDy  RMSE=$(round(ms_s_ho.rmse, digits=2)), PE=$(round(ms_s_ho.pe, digits=4))")
println("  Burton RMSE=$(round(ms_b_ho.rmse, digits=2)), PE=$(round(ms_b_ho.pe, digits=4))")
println("  O'Brien RMSE=$(round(ms_o_ho.rmse, digits=2)), PE=$(round(ms_o_ho.pe, digits=4))")

# Save holdout metrics for multiple storms
holdout_storms = [3, 5, 8]  # weak, moderate, strong (by seed variation)
holdout_labels_out = String[]
holdout_min_dst = Float64[]
holdout_rmse_sindy = Float64[]
holdout_rmse_burton = Float64[]
holdout_rmse_obrien = Float64[]
holdout_pe_sindy = Float64[]
holdout_pe_burton = Float64[]
holdout_pe_obrien = Float64[]

for ho in holdout_storms
    tr_ds = [all_datasets[k] for k in 1:n_storms if k != ho]
    tr_data, tr_dDst = concatenate_data(tr_ds)
    ξ_h, _, _ = sindy_discover(tr_data, lib, tr_dDst; λ=3.0, normalize=true)

    ts = all_datasets[ho]
    td, _ = prepare_sindy_data(ts, 1.0; smooth_window=5)
    bs = halfwave_rectify(ts.Bz)

    Dst_s = simulate_sindy(ξ_h, lib, ts, 1.0)
    Dst_b = simulate_burton(ts.V, bs, 1.0)
    Dst_o = simulate_obrien(ts.V, bs, 1.0)

    push!(holdout_labels_out, "Storm $ho")
    push!(holdout_min_dst, all_events[ho].min_dst)
    push!(holdout_rmse_sindy, rmse(Dst_s, ts.Dst_star))
    push!(holdout_rmse_burton, rmse(Dst_b, ts.Dst_star))
    push!(holdout_rmse_obrien, rmse(Dst_o, ts.Dst_star))
    push!(holdout_pe_sindy, prediction_efficiency(Dst_s, ts.Dst_star))
    push!(holdout_pe_burton, prediction_efficiency(Dst_b, ts.Dst_star))
    push!(holdout_pe_obrien, prediction_efficiency(Dst_o, ts.Dst_star))
end

df_holdout = DataFrame(
    storm = holdout_labels_out,
    min_dst_nT = holdout_min_dst,
    sindy_rmse = holdout_rmse_sindy,
    burton_rmse = holdout_rmse_burton,
    obrien_rmse = holdout_rmse_obrien,
    sindy_pe = holdout_pe_sindy,
    burton_pe = holdout_pe_burton,
    obrien_pe = holdout_pe_obrien
)
CSV.write("paper/data/holdout_metrics.csv", df_holdout)
println("  Saved paper/data/holdout_metrics.csv")

# Hero Figure: storm reconstruction (single panel, held-out storm 5)
fig2 = plot_scatter(test_swd.t, test_swd.Dst_star;
    xlabel="Time [hr]", ylabel=raw"$Dst^*$ [nT]",
    mode="lines", color=COLORS[1], dash=DASHES[1],
    legend="Observed", linewidth=2)
plot_scatter!(fig2, test_swd.t, Dst_sindy_ho;
    color=COLORS[2], dash=DASHES[2], mode="lines",
    legend="SINDy", linewidth=2)
plot_scatter!(fig2, test_swd.t, Dst_burton_ho;
    color=COLORS[3], dash=DASHES[3], mode="lines",
    legend="Burton", linewidth=2)
plot_scatter!(fig2, test_swd.t, Dst_obrien_ho;
    color=COLORS[4], dash=DASHES[4], mode="lines",
    legend="O'Brien-McP.", linewidth=2)
set_legend!(fig2; position=:bottomleft)
savefig(fig2, "paper/figs/fig_results_storm_reconstruction.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved fig_results_storm_reconstruction.pdf")

# ============================================================
# 3. Lambda Sweep / Pareto Front on Multi-Storm Data
# ============================================================
println("\n=== Result 3: Lambda Sweep / Pareto Front ===")

lambdas = 10.0 .^ range(-2, 2, length=60)
sweep_res = sweep_lambda(Θ_multi, all_dDst, lambdas; normalize=true)

n_terms_sweep = [s.n_terms for s in sweep_res]
rmse_sweep = [s.rmse for s in sweep_res]

df_sweep = DataFrame(
    lambda = [s.λ for s in sweep_res],
    n_terms = n_terms_sweep,
    rmse = rmse_sweep
)
CSV.write("paper/data/multistorm_lambda_sweep.csv", df_sweep)
println("  Saved paper/data/multistorm_lambda_sweep.csv")

# Pareto front: unique (n_terms, best rmse) pairs
pareto_dict = Dict{Int,Float64}()
for s in sweep_res
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
savefig(fig3, "paper/figs/fig_results_pareto_front.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_pareto_front.pdf")

# ============================================================
# 4. Ensemble SINDy on Multi-Storm Data
# ============================================================
println("\n=== Result 4: Ensemble SINDy ===")

median_ξ_multi, incl_prob_multi, _ = ensemble_sindy(
    all_data, lib, all_dDst;
    λ=3.0, n_models=200, subsample_frac=0.8, seed=42)

df_ensemble = DataFrame(
    term = term_names,
    median_coefficient = median_ξ_multi,
    inclusion_probability = incl_prob_multi
)
CSV.write("paper/data/multistorm_ensemble.csv", df_ensemble)
println("  Saved paper/data/multistorm_ensemble.csv")

# Sort by inclusion probability
sort_idx = sortperm(incl_prob_multi, rev=true)
sorted_names = term_names[sort_idx]
sorted_probs = incl_prob_multi[sort_idx]

fig4 = plot_bar(sorted_names, sorted_probs;
    xlabel="Library Term",
    ylabel="Inclusion Probability",
    color=COLORS[1])
set_legend!(fig4; position=:topright)
savefig(fig4, "paper/figs/fig_results_ensemble_inclusion.pdf";
        width=DOUBLE_W, height=SINGLE_H)
println("  Saved fig_results_ensemble_inclusion.pdf")

# ============================================================
# 5. Noise Robustness Parametric Sweep
# ============================================================
println("\n=== Result 5: Noise Robustness Sweep ===")

noise_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
noise_rmse = Float64[]
noise_n_terms = Int[]
noise_burton_rmse = Float64[]
noise_core_recovered = Bool[]

for σ in noise_levels
    noisy_datasets = SolarWindData[]
    for k in 1:n_storms
        swd_n, _ = generate_synthetic_storm(
            seed=200+k, dt=1.0, duration=120.0,
            α=α_true, τ=τ_true, noise_level=σ
        )
        push!(noisy_datasets, swd_n)
    end

    data_n, dDst_n = concatenate_data(noisy_datasets)
    ξ_n, active_n, Θ_n = sindy_discover(data_n, lib, dDst_n; λ=3.0, normalize=true)
    dDst_pred_n = Θ_n * ξ_n
    Bs_n = halfwave_rectify(data_n["Bz"])
    dDst_burt_n = burton_model(data_n["V"], Bs_n, data_n["Dst_star"])

    push!(noise_rmse, rmse(dDst_pred_n, dDst_n))
    push!(noise_n_terms, count(abs.(ξ_n) .> 0))
    push!(noise_burton_rmse, rmse(dDst_burt_n, dDst_n))
    push!(noise_core_recovered, haskey(active_n, "Dst_star") && haskey(active_n, "V*Bs"))

    println("  σ=$σ: n_terms=$(noise_n_terms[end]), RMSE=$(round(noise_rmse[end], digits=3)), core=$(noise_core_recovered[end])")
end

df_noise = DataFrame(
    noise_level = noise_levels,
    sindy_rmse = noise_rmse,
    burton_rmse = noise_burton_rmse,
    n_active_terms = noise_n_terms,
    core_terms_recovered = noise_core_recovered
)
CSV.write("paper/data/noise_sweep.csv", df_noise)
println("  Saved paper/data/noise_sweep.csv")

fig5 = plot_scatter(noise_levels .* 100, noise_rmse;
    xlabel="Noise Level [% of signal std]",
    ylabel="RMSE [nT/hr]",
    mode="lines+markers", color=COLORS[1], dash=DASHES[1],
    legend="SINDy", linewidth=2, marker_size=8)
plot_scatter!(fig5, noise_levels .* 100, noise_burton_rmse;
    color=COLORS[2], dash=DASHES[2], mode="lines+markers",
    legend="Burton", linewidth=2, marker_size=8)
set_legend!(fig5; position=:topleft)
savefig(fig5, "paper/figs/fig_results_noise_sweep.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_noise_sweep.pdf")

# ============================================================
# 6. Library Ablation Study
# ============================================================
println("\n=== Result 6: Library Ablation Study ===")

lib_full = build_solar_wind_library(max_poly_order=2, include_trig=true,
                                     include_cross=true, include_known=true)
lib_no_trig = build_solar_wind_library(max_poly_order=2, include_trig=false,
                                        include_cross=true, include_known=true)
lib_no_cross = build_solar_wind_library(max_poly_order=2, include_trig=true,
                                         include_cross=false, include_known=true)
lib_minimal = build_minimal_library()

lib_variants = [
    ("Full (21)", lib_full),
    ("No trig (15)", lib_no_trig),
    ("No cross (14)", lib_no_cross),
    ("Minimal (3)", lib_minimal)
]

ablation_names = String[]
ablation_rmse = Float64[]
ablation_n_terms = Int[]
ablation_pe = Float64[]

for (label, lib_var) in lib_variants
    Θ_var = evaluate_library(lib_var, all_data)
    ξ_var = stlsq(Θ_var, all_dDst; λ=3.0, normalize=true)
    pred_var = Θ_var * ξ_var
    n_act = count(abs.(ξ_var) .> 0)

    push!(ablation_names, label)
    push!(ablation_rmse, rmse(pred_var, all_dDst))
    push!(ablation_n_terms, n_act)
    push!(ablation_pe, prediction_efficiency(pred_var, all_dDst))
    println("  $label: n_active=$n_act, RMSE=$(round(ablation_rmse[end], digits=4)), PE=$(round(ablation_pe[end], digits=4))")
end

df_ablation = DataFrame(
    library = ablation_names,
    rmse = ablation_rmse,
    n_active_terms = ablation_n_terms,
    prediction_efficiency = ablation_pe
)
CSV.write("paper/data/ablation_metrics.csv", df_ablation)
println("  Saved paper/data/ablation_metrics.csv")

fig6 = plot_bar(ablation_names, ablation_rmse;
    xlabel="Library Variant",
    ylabel="RMSE [nT/hr]",
    color=COLORS[1])
set_legend!(fig6; position=:topright)
savefig(fig6, "paper/figs/fig_results_ablation.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_ablation.pdf")

# ============================================================
# 7. Phase-Dependent Discovery
# ============================================================
println("\n=== Result 7: Phase-Dependent Discovery ===")

phases_all = identify_storm_phases(all_data["Dst_star"], all_dDst)
phase_labels = ["Quiet", "Main", "Recovery"]

phase_coeff_dict = Dict{String,Vector{Float64}}()
for (p_idx, p_label) in enumerate(phase_labels)
    idx = findall(phases_all .== p_idx)
    if length(idx) < 20
        println("  Skipping $p_label (only $(length(idx)) points)")
        phase_coeff_dict[p_label] = zeros(length(term_names))
        continue
    end
    sub_data = Dict{String,Vector{Float64}}()
    for (key, val) in all_data
        sub_data[key] = val[idx]
    end
    ξ_phase, active_phase, _ = sindy_discover(sub_data, lib, all_dDst[idx]; λ=1.0, normalize=true)
    phase_coeff_dict[p_label] = ξ_phase
    n_act_phase = count(abs.(ξ_phase) .> 0)
    println("  $p_label ($( length(idx)) pts): $n_act_phase active terms")
    sorted_active = sort(collect(active_phase), by=x->abs(x[2]), rev=true)
    for (name, coeff) in sorted_active[1:min(5,length(sorted_active))]
        println("    $name: $(round(coeff, sigdigits=4))")
    end
end

# Save phase-dependent coefficients
df_phase = DataFrame(term = term_names)
for p_label in phase_labels
    df_phase[!, Symbol(p_label)] = phase_coeff_dict[p_label]
end
CSV.write("paper/data/phase_dependent_coefficients.csv", df_phase)
println("  Saved paper/data/phase_dependent_coefficients.csv")

# Figure 7: Phase-dependent — scatter of |ξ| for top terms per phase
# Select terms active in at least one phase
any_active = [any(abs(phase_coeff_dict[pl][j]) > 0 for pl in phase_labels) for j in 1:length(term_names)]
active_term_idx = findall(any_active)
if !isempty(active_term_idx)
    max_abs = [maximum(abs(phase_coeff_dict[pl][j]) for pl in phase_labels) for j in active_term_idx]
    sort_order_p = sortperm(max_abs, rev=true)
    top_n = min(8, length(sort_order_p))
    top_idx = active_term_idx[sort_order_p[1:top_n]]

    # Build matrix for bar chart comparison
    x_labels = term_names[top_idx]
    quiet_vals = [abs(phase_coeff_dict["Quiet"][j]) for j in top_idx]
    main_vals = [abs(phase_coeff_dict["Main"][j]) for j in top_idx]
    recovery_vals = [abs(phase_coeff_dict["Recovery"][j]) for j in top_idx]

    # Clamp zeros to small value for log scale
    eps_floor = 1e-6
    quiet_log = max.(quiet_vals, eps_floor)
    main_log = max.(main_vals, eps_floor)
    recovery_log = max.(recovery_vals, eps_floor)

    x_pos = collect(1.0:Float64(top_n))
    fig7 = plot_scatter(x_pos .- 0.2, quiet_log;
        xlabel="Library Term", ylabel=raw"$|\xi|$",
        mode="markers", color=COLORS[1], marker_size=10,
        marker_symbol="circle", legend="Quiet",
        yscale="log")
    plot_scatter!(fig7, x_pos, main_log;
        color=COLORS[2], mode="markers", marker_size=10,
        marker_symbol="square", legend="Main Phase")
    plot_scatter!(fig7, x_pos .+ 0.2, recovery_log;
        color=COLORS[3], mode="markers", marker_size=10,
        marker_symbol="diamond", legend="Recovery")
    set_legend!(fig7; position=:topright)
    savefig(fig7, "paper/figs/fig_results_phase_dependent.pdf";
            width=DOUBLE_W, height=SINGLE_H)
    println("  Saved fig_results_phase_dependent.pdf")
    println("  Terms: $(join(x_labels, ", "))")
end

# ============================================================
# 8. Scalability with Training Data
# ============================================================
println("\n=== Result 8: Scalability Sweep ===")

n_storm_vals = [1, 2, 3, 5, 7, 10]
scale_rmse = Float64[]
scale_n_terms = Int[]
scale_pe = Float64[]

# Test on a held-out storm (index 10)
test_swd_sc = all_datasets[10]
test_data_sc, _ = prepare_sindy_data(test_swd_sc, 1.0; smooth_window=5)

for ns in n_storm_vals
    train_ds = all_datasets[1:min(ns, 9)]  # never include test storm 10
    tr_data, tr_dDst = concatenate_data(train_ds)
    ξ_sc, _, _ = sindy_discover(tr_data, lib, tr_dDst; λ=3.0, normalize=true)

    Dst_sc = simulate_sindy(ξ_sc, lib, test_swd_sc, 1.0)

    push!(scale_rmse, rmse(Dst_sc, test_swd_sc.Dst_star))
    push!(scale_n_terms, count(abs.(ξ_sc) .> 0))
    push!(scale_pe, prediction_efficiency(Dst_sc, test_swd_sc.Dst_star))
    println("  n_storms=$ns: n_terms=$(scale_n_terms[end]), RMSE=$(round(scale_rmse[end], digits=2)), PE=$(round(scale_pe[end], digits=4))")
end

# Also add Burton baseline for reference
Bs_sc = halfwave_rectify(test_swd_sc.Bz)
Dst_burton_sc = simulate_burton(test_swd_sc.V, Bs_sc, 1.0)
burton_rmse_sc = rmse(Dst_burton_sc, test_swd_sc.Dst_star)

df_scale = DataFrame(
    n_storms = n_storm_vals,
    rmse = scale_rmse,
    n_active_terms = scale_n_terms,
    prediction_efficiency = scale_pe
)
CSV.write("paper/data/scalability_sweep.csv", df_scale)
println("  Saved paper/data/scalability_sweep.csv")
println("  Burton reference RMSE: $(round(burton_rmse_sc, digits=2))")

fig8 = plot_scatter(Float64.(n_storm_vals), scale_rmse;
    xlabel="Number of Training Storms",
    ylabel="Held-Out RMSE [nT]",
    mode="lines+markers", color=COLORS[1], dash=DASHES[1],
    legend="SINDy", linewidth=2, marker_size=8)
# Add Burton reference line
plot_scatter!(fig8, Float64.([1, 10]), [burton_rmse_sc, burton_rmse_sc];
    color=COLORS[2], dash=DASHES[2], mode="lines",
    legend="Burton", linewidth=2)
set_legend!(fig8; position=:topright)
savefig(fig8, "paper/figs/fig_results_scalability.pdf";
        width=SINGLE_W, height=SINGLE_H)
println("  Saved fig_results_scalability.pdf")

println("\n=== All figures and data generated ===")
println("Figures: paper/figs/fig_results_*.pdf")
println("Data:    paper/data/*.csv")
