#!/usr/bin/env julia
# Reproduce the manuscript's paired Wilcoxon signed-rank significance tests
# (SINDy vs. O'Brien--McPherron per-storm RMSE) from the persisted cross-cycle
# and held-out metrics, and persist the p-values. This closes the gap where the
# reported p-values had no in-repository computing code.
#
# Usage:
#   julia --project=SolarSINDy.jl SolarSINDy.jl/validation/significance_tests.jl

using SolarSINDy
using CSV
using DataFrames

const DATA_DIR = SolarSINDy.get_data_dir()
const PAPER_DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

function _sindy_minus_obrien(df::DataFrame, experiment)
    sub = experiment === nothing ? df : df[df.experiment .== experiment, :]
    s = sort(sub[sub.model .== "SINDy", :], :storm_id)
    o = sort(sub[sub.model .== "OBrienMcP", :], :storm_id)
    s.storm_id == o.storm_id ||
        error("storm_id mismatch between SINDy and O'Brien rows for $(experiment)")
    return Float64.(s.rmse) .- Float64.(o.rmse)
end

function main()
    cross = CSV.read(joinpath(DATA_DIR, "cross_cycle_metrics.csv"), DataFrame)
    holdout = CSV.read(joinpath(DATA_DIR, "real_holdout_metrics.csv"), DataFrame)

    rows = NamedTuple[]
    for experiment in unique(cross.experiment)
        d = _sindy_minus_obrien(cross, experiment)
        r = wilcoxon_signed_rank_p(d)
        push!(rows, (experiment=String(experiment), comparison="SINDy_vs_OBrienMcP",
                     n=r.n, statistic_w=r.w, z=r.z, p_value=r.p))
    end
    rh = wilcoxon_signed_rank_p(_sindy_minus_obrien(holdout, nothing))
    push!(rows, (experiment="Validation_C24", comparison="SINDy_vs_OBrienMcP",
                 n=rh.n, statistic_w=rh.w, z=rh.z, p_value=rh.p))

    df = DataFrame(rows)
    for dir in (DATA_DIR, PAPER_DATA_DIR)
        isdir(dir) || continue
        path = joinpath(dir, "wilcoxon_pvalues.csv")
        CSV.write(path, df)
        println("Wrote $path")
    end
    println(df)
    return df
end

main()
