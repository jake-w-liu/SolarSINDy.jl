#!/usr/bin/env julia
# real_data_discovery.jl — Phase B: Real-Data SINDy Discovery
#
# Following PLAN.md strictly:
#   B1: Single-equation discovery on training storms (cycles 20–23)
#   B2: Compare to all baselines on validation set (cycle 24)
#   B3: Extreme event test (May 2024 superstorm)
#   B4: Cross-solar-cycle generalization
#
# Outputs:
#   data/real_sindy_coefficients.csv
#   data/real_ensemble_inclusion.csv
#   data/real_lambda_sweep.csv
#   data/real_holdout_metrics.csv
#   data/may2024_reconstruction.csv
#   data/cross_cycle_metrics.csv

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const FIGS_DIR = joinpath(@__DIR__, "..", "figs")
mkpath(DATA_DIR)
mkpath(FIGS_DIR)

# ============================================================
# Helper functions (must be defined before use)
# ============================================================

# simulate_sindy is now exported from SolarSINDy — use it directly

"""
    _concat_storm_data(df, entries)

Extract storms, prepare for SINDy, concatenate, and remove NaN AND fill-fabricated
rows. Returns (concat_data::Dict, dDst_clean::Vector, n_storms_used::Int).

Fabricated-row exclusion (finding real_data_discovery.jl:69): `extract_storm_data`
forward/backward-fills every driver, so by the time the concatenated arrays are
built there is no NaN left for a post-fill `isnan` filter to catch — the old filter
was dead code and up to ~40% fill-fabricated (and, at window heads, future-copied)
driver hours entered the SINDy regression. The fix carries a PRE-FILL validity mask
straight from the cleaned DataFrame (which still holds the original NaNs) for each
storm window and drops those rows. Density and Pdyn are checked explicitly because
`clean_omni_data!`'s `:quality` flag covers only V/Bz/Dst, yet n and Pdyn drive
active discovered terms (n·V, n·Bs, n·V·Bs, n·V², Pdyn·Bs).
"""
function _concat_storm_data(df::DataFrame, entries::Vector{StormCatalogEntry})
    all_data_dicts = Dict{String,Vector{Float64}}[]
    all_dDst = Float64[]
    all_rawvalid = Bool[]   # per-row PRE-FILL driver validity, aligned to all_dDst
    n_used = 0

    for entry in entries
        swd = extract_storm_data(df, entry)
        try
            data_dict, dDst_dt = prepare_sindy_data(swd, 1.0; smooth_window=5)
            if count(!isnan, dDst_dt) < 20
                continue
            end
            # Pre-fill validity: rows whose model drivers were originally present.
            # extract_storm_data set swd.t = 0:(n_pts-1) over exactly these rows, and
            # prepare_sindy_data preserves length, so the mask aligns 1:1 with
            # dDst_dt / data_dict. A row filled by _fillnan (forward OR backward, the
            # latter copying future values into leading gaps) has raw_valid = false.
            rows = entry.onset_idx:entry.end_idx
            raw_valid = .!(isnan.(df.V[rows]) .| isnan.(df.Bz[rows]) .|
                           isnan.(df.By[rows]) .| isnan.(df.n[rows]) .|
                           isnan.(df.Pdyn[rows]) .| isnan.(df.Dst_star[rows]))
            push!(all_data_dicts, data_dict)
            append!(all_dDst, dDst_dt)
            append!(all_rawvalid, raw_valid)
            n_used += 1
        catch
            continue
        end
    end

    if isempty(all_data_dicts)
        return Dict{String,Vector{Float64}}(), Float64[], 0
    end

    # Concatenate
    concat = Dict{String,Vector{Float64}}()
    for key in keys(all_data_dicts[1])
        concat[key] = vcat([d[key] for d in all_data_dicts]...)
    end

    # Drop rows that are NaN in the target/drivers (derivative/smoothing edges) OR
    # that were fabricated by gap-filling (all_rawvalid). The pre-fill mask is the
    # load-bearing part: without it the isnan checks alone never fire on filled rows.
    valid = .!isnan.(all_dDst) .& all_rawvalid
    for key in keys(concat)
        valid .&= .!isnan.(concat[key])
    end
    for key in keys(concat)
        concat[key] = concat[key][valid]
    end

    return concat, all_dDst[valid], n_used
end

"""
    _cross_cycle_experiment(df, train_entries, test_entries, lib)

Run SINDy discovery on train_entries, evaluate on test_entries with baselines.
Returns (ξ, metrics_rows).
"""
function _cross_cycle_experiment(df::DataFrame,
                                  train_entries::Vector{StormCatalogEntry},
                                  test_entries::Vector{StormCatalogEntry},
                                  lib::CandidateLibrary;
                                  λ::Real=148.0)
    concat, dDst, n_used = _concat_storm_data(df, train_entries)

    if isempty(dDst)
        println("  No valid training data!")
        return zeros(length(lib)), []
    end

    println("  Training: $(length(train_entries)) storms ($(n_used) used), $(length(dDst)) valid points")

    # Discover
    ξ, _, _ = sindy_discover(concat, lib, dDst; λ=λ, normalize=true)
    n_active = count(ξ .!= 0)
    println("  Discovered: $(n_active) active terms")

    # Evaluate on test storms
    metrics_rows = []
    for entry in test_entries
        swd = extract_storm_data(df, entry)
        if length(swd.t) < 20
            continue
        end

        Dst_obs = swd.Dst_star
        Bs = max.(-swd.Bz, 0.0)
        dt = 1.0

        Dst_burton = simulate_burton(swd.V, Bs, dt; Dst0=Dst_obs[1])
        Dst_burton_full = simulate_burton_full(swd.V, Bs, dt; Dst0=Dst_obs[1])
        Dst_obrien = simulate_obrien(swd.V, Bs, dt; Dst0=Dst_obs[1])
        Dst_sindy = simulate_sindy(ξ, lib, swd, dt)

        for (name, pred) in [("SINDy", Dst_sindy), ("Burton", Dst_burton), ("BurtonFull", Dst_burton_full), ("OBrienMcP", Dst_obrien)]
            if length(pred) == length(Dst_obs)
                m = metrics_summary(pred, Dst_obs; name=name)
                push!(metrics_rows, (
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

    println("  Test: $(length(test_entries)) storms evaluated")
    return ξ, metrics_rows
end

# ============================================================
# Load cleaned data and storm catalog
# ============================================================
println("=" ^ 60)
println("Phase B: Real-Data SINDy Discovery")
println("=" ^ 60)

extracted_path = joinpath(DATA_DIR, "omni_extracted.csv")
println("\nLoading OMNI data...")
@time df = parse_omni2(extracted_path; year_start=1963, year_end=2025)
@time clean_omni_data!(df)

catalog_path = joinpath(DATA_DIR, "storm_catalog.csv")
catalog = load_storm_catalog(catalog_path)
println("Loaded catalog: $(length(catalog)) storms")

# ============================================================
# B1: Single-Equation Discovery (Training Set)
# ============================================================
println("\n" * "=" ^ 60)
println("B1: Single-Equation Discovery on Training Storms")
println("=" ^ 60)

train_entries = filter(e -> e.split == "train", catalog)
println("Training storms: $(length(train_entries))")

concat_data, dDst_clean, n_storms_used = _concat_storm_data(df, train_entries)
println("Storms used for SINDy: $(n_storms_used) / $(length(train_entries))")
println("Valid data points: $(length(dDst_clean))")

# Build candidate library (full 21-term)
lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                 include_cross=true, include_known=true)
println("Library: $(length(lib)) terms")
println("Terms: ", join(get_term_names(lib), ", "))

# Evaluate library on concatenated training data
Θ = evaluate_library(lib, concat_data)
println("Library matrix: $(size(Θ))")

# --- B1a: Lambda sweep (Pareto front) ---
println("\n--- Lambda Sweep ---")
lambdas = 10.0 .^ range(-2, 4, length=60)
sweep_results = sweep_lambda(Θ, dDst_clean, lambdas; normalize=true)

sweep_df = DataFrame(
    lambda = [r.λ for r in sweep_results],
    n_terms = [r.n_terms for r in sweep_results],
    rmse = [r.rmse for r in sweep_results]
)
CSV.write(joinpath(DATA_DIR, "real_lambda_sweep.csv"), sweep_df)
println("Lambda sweep saved ($(nrow(sweep_df)) points)")

println("\nPareto front:")
for nt in sort(unique([r.n_terms for r in sweep_results]))
    subset = filter(r -> r.n_terms == nt, sweep_results)
    best = argmin(r -> r.rmse, subset)
    println("  $(nt) terms: RMSE=$(round(best.rmse, digits=4)), λ=$(round(best.λ, sigdigits=3))")
end

# --- B1b: Ensemble SINDy (500 subsample ensembles) ---
# λ selection procedure (documented in paper §2.4):
#   Sweep over logarithmic λ grid, then select the minimum-RMSE λ within
#   the Pareto-optimal range of 4–8 active terms. On the 1963–2025 OMNI
#   dataset (61,494 points), this yields λ ≈ 236. The value is data-dependent;
#   re-downloading OMNI data may produce a slightly different λ.
# Target: 4–8 active terms (physically motivated: Dst* decay + injection + trig)
#
# DELIBERATE λ MISMATCH (finding real_data_discovery.jl:261). The uncertainty
# ensemble here is fit in the SPARSER 4–8-term regime, whereas the DEPLOYED point
# fit (ξ_best below) is selected in the 8–10-term regime. The two therefore live in
# different sparsity regimes by construction, with two consequences the operational
# uncertainty path inherits: (1) deployed-active terms selected only in the denser
# fit (e.g. n·Bs, n·V·Bs) appear in ≤1 subsample here, so their persisted CI is
# degenerate (ci_025 == ci_975 ⇒ width 0 ⇒ σ = 0 in init_forecast); (2) the
# surviving terms' spreads are estimated around ensemble point values that differ
# ~30% from the deployed ones. A degenerate-CI audit runs after ξ_best is fit
# (below) to make this mismatch visible. The served V2 interval is ACI/conformal and
# does not consume these ensemble widths, so the shipped CI semantics are left
# unchanged; the mismatch is documented rather than silently rebalanced.
#
# NOTE (finding sindy.jl:132): ensemble_sindy draws are 80% subsamples WITHOUT
# replacement (subagging), not n-out-of-n bootstraps — hence "subsample ensembles".
ensemble_λ = 100.0  # default
for target_range in [4:8, 3:10, 2:12]
    kr = filter(r -> r.n_terms in target_range, sweep_results)
    if !isempty(kr)
        global ensemble_λ = argmin(r -> r.rmse, kr).λ
        break
    end
end
println("\n--- Ensemble SINDy (500 subsample ensembles, λ=$(round(ensemble_λ, sigdigits=3))) ---")
@time median_ξ, inclusion_prob, all_ξ = ensemble_sindy(
    concat_data, lib, dDst_clean;
    λ=ensemble_λ, n_models=500, subsample_frac=0.8, seed=42
)

term_names = get_term_names(lib)
println("\nEnsemble results:")
for (i, name) in enumerate(term_names)
    if inclusion_prob[i] > 0.05
        coef_vals = all_ξ[i, :]
        active_vals = coef_vals[coef_vals .!= 0.0]
        ci_lo = isempty(active_vals) ? 0.0 : quantile(active_vals, 0.025)
        ci_hi = isempty(active_vals) ? 0.0 : quantile(active_vals, 0.975)
        println("  $(rpad(name, 25)) π=$(round(inclusion_prob[i], digits=3))  " *
                "ξ=$(round(median_ξ[i], sigdigits=4))  " *
                "95%CI=[$(round(ci_lo, sigdigits=3)), $(round(ci_hi, sigdigits=3))]")
    end
end

# Save ensemble results
ensemble_df = DataFrame(
    term = term_names,
    inclusion_prob = inclusion_prob,
    median_coef = median_ξ,
    ci_025 = [begin
        vals = all_ξ[i, :]; active = vals[vals .!= 0.0]
        isempty(active) ? 0.0 : quantile(active, 0.025)
    end for i in 1:length(term_names)],
    ci_975 = [begin
        vals = all_ξ[i, :]; active = vals[vals .!= 0.0]
        isempty(active) ? 0.0 : quantile(active, 0.975)
    end for i in 1:length(term_names)]
)
CSV.write(joinpath(DATA_DIR, "real_ensemble_inclusion.csv"), ensemble_df)

coef_df = DataFrame(term=term_names, coefficient=median_ξ, inclusion=inclusion_prob)
CSV.write(joinpath(DATA_DIR, "real_sindy_coefficients.csv"), coef_df)

# --- Single best-λ discovery ---
println("\n--- Single Discovery (best λ) ---")
# Need 8+ terms for physical model (Dst_star decay + injection + trig)
# Pareto front shows jump from 4 (trig-only) to 8 terms — knee is at 8
best_λ = ensemble_λ
best_n_terms = 0
for target_range in [8:10, 6:12, 4:15, 2:21]
    kr = filter(r -> r.n_terms in target_range, sweep_results)
    if !isempty(kr)
        global best_λ = argmin(r -> r.rmse, kr).λ
        global best_n_terms = argmin(r -> r.rmse, kr).n_terms
        break
    end
end
println("Best λ: $(round(best_λ, sigdigits=3)) → $(best_n_terms) terms")

ξ_best, active_terms, _ = sindy_discover(concat_data, lib, dDst_clean;
                                           λ=best_λ, normalize=true)
println("Discovered equation (dDst*/dt):")
for (name, coef) in sort(collect(active_terms), by=x->abs(x[2]), rev=true)
    println("  $(round(coef, sigdigits=4)) × $(name)")
end

# --- Persist the served point fit + provenance (finding real_data_discovery.jl:271)
# The operationally served ODE reads data/real_sindy_discovery_coefficients.csv, but
# that file previously had no generator: ξ_best was printed and thrown away, so the
# deployed equation was not reproducible or traceable to code. Write it here with the
# same (term, coefficient) schema the readers (init_forecast, generate_real_figures)
# expect, index-aligned to get_term_names(lib). A companion provenance row records
# best_λ, the active-term count, the training span/size, and the run date so a later
# audit can tell whether a regenerated fit still matches the shipped, version-pinned
# artifact (which is frozen by a package regression test).
discovery_coef_df = DataFrame(term=term_names, coefficient=ξ_best)
CSV.write(joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv"), discovery_coef_df)
provenance_df = DataFrame(
    field = ["best_lambda", "n_active_terms", "n_train_storms_used",
             "n_train_points", "omni_year_start", "omni_year_end",
             "library_terms", "generated_date"],
    value = [string(best_λ), string(count(ξ_best .!= 0)), string(n_storms_used),
             string(length(dDst_clean)), "1963", "2025",
             string(length(lib)), string(Dates.today())]
)
CSV.write(joinpath(DATA_DIR, "real_sindy_discovery_provenance.csv"), provenance_df)
println("Saved: real_sindy_discovery_coefficients.csv + provenance sidecar")

# --- Degenerate-CI audit for the deliberate ensemble/deployed λ mismatch
# (finding real_data_discovery.jl:261). Flag every deployed-active term whose
# ensemble CI is degenerate (single-draw ci_025 == ci_975) or absent, i.e. terms
# that would receive exactly zero coefficient spread in the operational ensemble.
println("\nEnsemble-CI coverage of deployed-active terms (λ mismatch audit):")
for (i, name) in enumerate(term_names)
    ξ_best[i] == 0.0 && continue
    width = ensemble_df.ci_975[i] - ensemble_df.ci_025[i]
    incl = inclusion_prob[i]
    flag = width <= 0 ? "  ⚠ DEGENERATE CI (σ→0 in ensemble)" : ""
    println("  $(rpad(name, 20)) π=$(round(incl, digits=3))  " *
            "CI width=$(round(width, sigdigits=3))$(flag)")
end

# ============================================================
# B2: Validation on Cycle 24
# ============================================================
println("\n" * "=" ^ 60)
n_val_storms = count(e -> e.split == "val", catalog)
println("B2: Validation on Cycle 24 ($n_val_storms storms)")
println("=" ^ 60)

val_data, val_entries = extract_all_storms(df, catalog; split="val")

metrics_rows = []
for (i, swd) in enumerate(val_data)
    dt = 1.0
    n_pts = length(swd.t)
    if n_pts < 20
        continue
    end

    Dst_obs = swd.Dst_star
    Bs = max.(-swd.Bz, 0.0)

    Dst_burton = simulate_burton(swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_burton_full = simulate_burton_full(swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_obrien = simulate_obrien(swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_sindy = simulate_sindy(ξ_best, lib, swd, dt)

    for (name, pred) in [("SINDy", Dst_sindy), ("Burton", Dst_burton), ("BurtonFull", Dst_burton_full), ("OBrienMcP", Dst_obrien)]
        if length(pred) == length(Dst_obs)
            m = metrics_summary(pred, Dst_obs; name=name)
            push!(metrics_rows, (
                storm_id = val_entries[i].storm_id,
                model = name,
                rmse = m.rmse,
                correlation = m.corr,
                pe = m.pe,
                min_dst_obs = minimum(Dst_obs),
                n_points = n_pts
            ))
        end
    end
end

metrics_df = DataFrame(metrics_rows)

println("\nValidation metrics (mean ± std across storms):")
for model in ["SINDy", "Burton", "BurtonFull", "OBrienMcP"]
    subset = filter(row -> row.model == model, metrics_df)
    if nrow(subset) > 0
        println("  $(rpad(model, 12)) RMSE=$(round(mean(subset.rmse), digits=2)) ± $(round(std(subset.rmse), digits=2))  " *
                "PE=$(round(mean(subset.pe), digits=3)) ± $(round(std(subset.pe), digits=3))  " *
                "r=$(round(mean(subset.correlation), digits=3))")
    end
end

CSV.write(joinpath(DATA_DIR, "real_holdout_metrics.csv"), metrics_df)
println("Saved: real_holdout_metrics.csv ($(nrow(metrics_df)) rows)")

# ============================================================
# B3: May 2024 Superstorm (Extreme Event Test)
# ============================================================
println("\n" * "=" ^ 60)
println("B3: May 2024 Superstorm — Extreme Event Test")
println("=" ^ 60)

may2024_entries = filter(e -> Dates.year(e.onset_time) == 2024 &&
                              Dates.month(e.onset_time) == 5, catalog)
if !isempty(may2024_entries)
    may_entry = may2024_entries[1]
    may_swd = extract_storm_data(df, may_entry)
    println("May 2024 storm: $(length(may_swd.t)) hours, " *
            "Dst_star min=$(round(minimum(may_swd.Dst_star), digits=1)) nT")

    dt = 1.0
    Dst_obs = may_swd.Dst_star
    Bs = max.(-may_swd.Bz, 0.0)

    Dst_burton = simulate_burton(may_swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_burton_full = simulate_burton_full(may_swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_obrien = simulate_obrien(may_swd.V, Bs, dt; Dst0=Dst_obs[1])
    Dst_sindy = simulate_sindy(ξ_best, lib, may_swd, dt)

    println("\nMay 2024 superstorm metrics:")
    for (name, pred) in [("SINDy", Dst_sindy), ("Burton", Dst_burton), ("BurtonFull", Dst_burton_full), ("OBrienMcP", Dst_obrien)]
        m = metrics_summary(pred, Dst_obs; name=name)
        println("  $(rpad(name, 12)) RMSE=$(round(m.rmse, digits=1)) nT, " *
                "PE=$(round(m.pe, digits=3)), r=$(round(m.corr, digits=3))")
    end

    may_df = DataFrame(
        t_hr = may_swd.t,
        Dst_obs = Dst_obs,
        Dst_sindy = Dst_sindy,
        Dst_burton = Dst_burton,
        Dst_burton_full = Dst_burton_full,
        Dst_obrien = Dst_obrien,
        V = may_swd.V,
        Bz = may_swd.Bz,
        Pdyn = may_swd.Pdyn
    )
    CSV.write(joinpath(DATA_DIR, "may2024_reconstruction.csv"), may_df)
    println("Saved: may2024_reconstruction.csv")
else
    println("WARNING: May 2024 superstorm not found in catalog!")
end

# ============================================================
# B4: Cross-Solar-Cycle Generalization
# ============================================================
println("\n" * "=" ^ 60)
println("B4: Cross-Solar-Cycle Generalization")
println("=" ^ 60)

cross_cycle_rows = []

# Experiment 1: Train on cycles 20-22, test on cycle 23
println("\n--- Train: cycles 20-22, Test: cycle 23 ---")
train_2022 = filter(e -> e.solar_cycle <= 22, catalog)
test_23 = filter(e -> e.solar_cycle == 23, catalog)
if !isempty(train_2022) && !isempty(test_23)
    _, metrics_2022 = _cross_cycle_experiment(df, train_2022, test_23, lib; λ=best_λ)
    for m in metrics_2022
        push!(cross_cycle_rows, (experiment="C20-22->C23", m...))
    end
end

# Experiment 2: Train on even cycles (20,22,24), test on odd (21,23)
println("\n--- Train: even cycles (20,22,24), Test: odd cycles (21,23) ---")
train_even = filter(e -> e.solar_cycle in [20, 22, 24], catalog)
test_odd = filter(e -> e.solar_cycle in [21, 23], catalog)
if !isempty(train_even) && !isempty(test_odd)
    _, metrics_even = _cross_cycle_experiment(df, train_even, test_odd, lib; λ=best_λ)
    for m in metrics_even
        push!(cross_cycle_rows, (experiment="even->odd", m...))
    end
end

# Experiment 3: Train on cycles 20-23, test on cycle 25 (full OOD)
println("\n--- Train: cycles 20-23, Test: cycle 25 (incl. May 2024) ---")
train_2023 = filter(e -> e.solar_cycle <= 23, catalog)
test_25 = filter(e -> e.solar_cycle == 25, catalog)
if !isempty(train_2023) && !isempty(test_25)
    _, metrics_2023 = _cross_cycle_experiment(df, train_2023, test_25, lib; λ=best_λ)
    for m in metrics_2023
        push!(cross_cycle_rows, (experiment="C20-23->C25", m...))
    end
end

if !isempty(cross_cycle_rows)
    cross_df = DataFrame(cross_cycle_rows)
    CSV.write(joinpath(DATA_DIR, "cross_cycle_metrics.csv"), cross_df)
    println("\nCross-cycle results saved: cross_cycle_metrics.csv")

    println("\nCross-cycle summary (mean RMSE across test storms):")
    for exp in unique(cross_df.experiment)
        subset = filter(row -> row.experiment == exp, cross_df)
        for model in unique(subset.model)
            ms = filter(row -> row.model == model, subset)
            println("  $(rpad(exp, 15)) $(rpad(model, 12)) " *
                    "RMSE=$(round(mean(ms.rmse), digits=2)), PE=$(round(mean(ms.pe), digits=3))")
        end
    end
end

# ============================================================
# Summary
# ============================================================
println("\n" * "=" ^ 60)
println("Phase B Complete")
println("=" ^ 60)
println("Outputs:")
for f in ["real_sindy_coefficients.csv", "real_sindy_discovery_coefficients.csv",
          "real_sindy_discovery_provenance.csv", "real_ensemble_inclusion.csv",
          "real_lambda_sweep.csv", "real_holdout_metrics.csv",
          "may2024_reconstruction.csv", "cross_cycle_metrics.csv"]
    fp = joinpath(DATA_DIR, f)
    if isfile(fp)
        println("  ✓ $(f) ($(round(filesize(fp)/1e3, digits=1)) KB)")
    else
        println("  ✗ $(f) — MISSING")
    end
end
