#!/usr/bin/env julia
# gen_sensitivity_fig.jl — Generate phase threshold sensitivity figure (SI)

using PlotlySupply, CSV, DataFrames, Statistics
import PlotlyKaleido

PlotlyKaleido.start()

const DATA_DIR = joinpath(@__DIR__, "..", "..", "paper", "data")
const FIGS_DIR = joinpath(@__DIR__, "..", "..", "paper", "figs")

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

# Load sensitivity data
df = CSV.read(joinpath(DATA_DIR, "phase_threshold_sensitivity.csv"), DataFrame)

# Recovery phase decay coefficient — most important for validation
phase = "recovery"
term = "Dst_star"
R_vals = sort(unique(df.R_thresh))

fig = nothing
for (ri, R_thresh) in enumerate(R_vals)
    sub = filter(row -> row.phase == phase && row.term == term &&
                 row.R_thresh == R_thresh, df)
    sort!(sub, :D_thresh)

    # Convert Dst_star coefficient to decay timescale (τ = -1/ξ)
    tau_vals = [-1.0 / c for c in sub.coefficient]

    label = raw"$R_\mathrm{thresh}$=" * "$(Int(R_thresh)) nT/hr"
    dashes = ["solid", "dash", "dashdot", "dot", "longdash"]

    if fig === nothing
        global fig = plot_scatter(Float64.(sub.D_thresh), tau_vals;
            xlabel=raw"$D_\mathrm{thresh}$ [nT]",
            ylabel=raw"Recovery decay $\tau$ [hr]",
            mode="lines+markers",
            color=COLORS[ri],
            dash=dashes[ri],
            marker_size=6,
            linewidth=2,
            legend=label,
            fontsize=12)
    else
        plot_scatter!(fig, Float64.(sub.D_thresh), tau_vals;
            mode="lines+markers",
            color=COLORS[ri],
            dash=dashes[ri],
            marker_size=6,
            linewidth=2,
            legend=label)
    end
end

set_legend!(fig; position=:topright)
PlotlyKaleido.savefig(fig, joinpath(FIGS_DIR, "fig_results_phase_sensitivity.pdf");
                       width=504, height=360)
println("Saved: fig_results_phase_sensitivity.pdf")
