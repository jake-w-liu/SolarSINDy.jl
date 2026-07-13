#!/usr/bin/env julia
# generate_ensemble_draws.jl — persist the JOINT ensemble coefficient draws.
#
# Finding sindy.jl:47 (uncertainty-quantification defect): init_forecast currently
# perturbs the four near-collinear clock-angle coefficients INDEPENDENTLY, discarding
# their near-perfect negative correlations, which inflates the operational ensemble
# spread. The fix is to sample coefficient VECTORS jointly from the ensemble draw
# matrix (all_ξ), which ensemble_sindy already returns but the pipeline discarded.
#
# This generator reruns ensemble_sindy on the real training pool (cycles 20–23) at
# the DEPLOYED best-λ sparsity regime and writes data/real_sindy_ensemble_draws.csv:
#   * one row per subsample draw (n_models = 500 ≥ 300),
#   * columns = the discovered-coefficient term names in the exact order of
#     data/real_sindy_discovery_coefficients.csv (= get_term_names order),
#   * a fixed RNG seed for reproducibility,
#   * each column RECENTERED so its mean equals the deployed point coefficient
#     (the served fit stays the center) while the within/between-term correlation
#     structure — in particular the clock-angle cancellation manifold — is preserved.
#
# It does NOT touch the deployed coefficient file or any paper_v2_monitor/data artifact; it only
# reads the cached OMNI archive + storm catalog and writes the new draws sidecar.

using SolarSINDy
using CSV, DataFrames, Statistics, Random

const PKG_DIR  = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(PKG_DIR, "data")

# Resolve the cached OMNI archive (never re-download): package data/ first, then the
# project's paper_v2_monitor/data cache, then $OMNI_EXTRACTED.
function _resolve_omni()
    cands = String[
        joinpath(DATA_DIR, "omni_extracted.csv"),
        normpath(joinpath(PKG_DIR, "..", "paper_v2_monitor", "data", "omni_extracted.csv")),
    ]
    haskey(ENV, "OMNI_EXTRACTED") && pushfirst!(cands, ENV["OMNI_EXTRACTED"])
    for c in cands
        isfile(c) && return c
    end
    error("omni_extracted.csv not found in: $(cands)")
end

# Faithful copy of real_data_discovery.jl's ORIGINAL concatenation (fill-included),
# so the draws share the exact training pool that produced the DEPLOYED fit whose
# coefficients we recenter onto. (The regression-side fabricated-row mask fix in
# real_data_discovery.jl is a separate, not-yet-redeployed change.)
function _concat_train(df, entries)
    dicts = Dict{String,Vector{Float64}}[]
    dDst = Float64[]
    for entry in entries
        swd = extract_storm_data(df, entry)
        try
            dd, ddt = prepare_sindy_data(swd, 1.0; smooth_window=5)
            count(!isnan, ddt) < 20 && continue
            push!(dicts, dd); append!(dDst, ddt)
        catch
            continue
        end
    end
    isempty(dicts) && error("no training storms concatenated")
    concat = Dict{String,Vector{Float64}}()
    for key in keys(dicts[1])
        concat[key] = vcat([d[key] for d in dicts]...)
    end
    valid = .!isnan.(dDst)
    for key in keys(concat)
        valid .&= .!isnan.(concat[key])
    end
    for key in keys(concat)
        concat[key] = concat[key][valid]
    end
    return concat, dDst[valid]
end

const SEED = 42          # fixed for reproducibility (matches shipped ensemble seed)
const N_MODELS = 500     # ≥ 300 required

println("Loading cached OMNI archive + catalog ...")
omni_path = _resolve_omni()
println("  OMNI: $(omni_path)")
df = parse_omni2(omni_path; year_start=1963, year_end=2025)
clean_omni_data!(df)
catalog = load_storm_catalog(joinpath(DATA_DIR, "storm_catalog.csv"))
train_entries = filter(e -> e.split == "train", catalog)
println("  train storms: $(length(train_entries))")

lib = build_solar_wind_library(; max_poly_order=2, include_trig=true,
                                include_cross=true, include_known=true)
term_names = get_term_names(lib)

concat_data, dDst_clean = _concat_train(df, train_entries)
Θ = evaluate_library(lib, concat_data)
println("  training points: $(length(dDst_clean)), library terms: $(length(lib))")

# Select the DEPLOYED best-λ (same procedure as real_data_discovery.jl).
lambdas = 10.0 .^ range(-2, 4, length=60)
sweep = sweep_lambda(Θ, dDst_clean, lambdas; normalize=true)
best_λ = lambdas[1]
for target_range in [8:10, 6:12, 4:15, 2:21]
    kr = filter(r -> r.n_terms in target_range, sweep)
    if !isempty(kr)
        global best_λ = argmin(r -> r.rmse, kr).λ
        break
    end
end
println("  best-λ (deployed regime) = $(round(best_λ, sigdigits=4))")

println("Running ensemble_sindy ($(N_MODELS) subsample draws, seed=$(SEED)) ...")
_, _, all_ξ = ensemble_sindy(concat_data, lib, dDst_clean;
                             λ=best_λ, n_models=N_MODELS, subsample_frac=0.8, seed=SEED)
# all_ξ is p × n_models; transpose to draws (rows) × terms (cols).
draws = permutedims(all_ξ)               # N_MODELS × p

# Load the DEPLOYED point coefficients and recenter each column onto them.
deployed = CSV.read(joinpath(DATA_DIR, "real_sindy_discovery_coefficients.csv"), DataFrame)
String.(deployed.term) == term_names ||
    error("deployed term order does not match library order")
ξ_dep = Float64.(deployed.coefficient)
for j in 1:length(term_names)
    draws[:, j] .+= (ξ_dep[j] - mean(view(draws, :, j)))
end

out = DataFrame(draws, Symbol.(term_names))
out_path = joinpath(DATA_DIR, "real_sindy_ensemble_draws.csv")
CSV.write(out_path, out)

# --- Verification: means match deployed to 1e-8, row count ≥ 300 ---
col_means = [mean(out[!, Symbol(t)]) for t in term_names]
max_dev = maximum(abs.(col_means .- ξ_dep))
println("\nWrote $(out_path)")
println("  rows = $(nrow(out)) (≥ 300 required: $(nrow(out) >= 300))")
println("  max |col_mean - deployed| = $(max_dev) (≤ 1e-8 required: $(max_dev <= 1e-8))")
@assert nrow(out) >= 300 "row count < 300"
@assert max_dev <= 1e-8 "column means do not match deployed coefficients to 1e-8"
println("VERIFY OK")
