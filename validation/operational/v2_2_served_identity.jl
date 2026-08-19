#!/usr/bin/env julia

# v2_2_served_identity.jl — offline identity oracle for the served V2.2 static-stack center.
#
# The served point center is the fitted static regime stack applied to six point components. The
# stack was fitted, and its skill measured, on the archived base table; the live engine applies it to
# the components it computes itself. If the application differs — a mis-derived regime, a permuted
# component order, a coupling gate that never disengages — the served center would be a different
# estimator from the one that was scored, and no downstream test would notice.
#
# This oracle drives the *serving* function `v22_serving_center`, the same code the live engine calls,
# with the archived base-table components, and compares the result with the archived
# `static_v2_2_dst_nt` column. It therefore pins the regime derivation, the coupling gate, the
# per-step weight lookup and the weighted sum together.
#
# What it does not pin — and must not be read as pinning — is the *production* of the six components.
# The live engine's own component rollouts admit L1-measured hours, so the live component values are
# not the archived ones; the served center is the stack applied to live components, and this oracle
# establishes only that the stack is applied correctly.
#
# Usage (from the package root):
#   julia --startup-file=no --project=. validation/operational/v2_2_served_identity.jl [options]
#
#   --stack=<path>    stack weights to test (default deploy/operational_v2_2_stack.csv)
#   --rows=<n>        minimum number of archived rows to reproduce (default 5000)
#   --stride=<n>      keep every n-th archived row (default 1, i.e. every row)
#   --out=<path>      per-step summary CSV (default …/v2_2_served_identity.csv)

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

"Identity target: the largest absolute deviation a served center may show against the archived one."
const V22_IDENTITY_TOL_NT = 1e-9

"Minimum archived rows the oracle must reproduce before it may report PASS."
const V22_IDENTITY_MIN_ROWS = 5_000

_v22_identity_log(msg) = (println(msg); flush(stdout))

"Base-table columns the served static-stack oracle needs."
function v22_identity_table_columns()
    return String["issue_time_utc", "model_step_hours", "partition",
                  "latest_dst_nt", "dst_delta_1h_nt", "VBsouth_mvm", "coupling_active_mvm",
                  "served_v2_1_dst_nt", "frozen_v2_1_dst_nt", "persistence_dst_nt",
                  "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt", "static_v2_2_dst_nt"]
end

"""
    run_v2_2_served_identity(; stack_path, table_path, min_rows, stride, out) -> NamedTuple

Reproduce the archived `static_v2_2_dst_nt` centers through `v22_serving_center` and report the
largest absolute deviation per model step and per regime.
"""
function run_v2_2_served_identity(; stack_path::AbstractString,
                                  table_path::AbstractString = V23_BASE_TABLE,
                                  min_rows::Int = V22_IDENTITY_MIN_ROWS,
                                  stride::Int = 1,
                                  out::AbstractString = joinpath(
                                      OPERATIONAL_OUTPUT_DIR, "v2_2_served_identity.csv"))
    stride >= 1 || error("stride must be positive, got $stride")
    stack = load_v22_serving_stack(String(stack_path))
    _v22_identity_log("  stack $(stack.label) steps $(join(stack.supported_model_steps, ","))")
    isfile(table_path) || error("V2.3 base table is missing: $table_path")
    table = CSV.read(table_path, DataFrame; select = v22_identity_table_columns(),
                     types = Dict("issue_time_utc" => DateTime))
    keep = [r for r in 1:nrow(table)
            if String(table.partition[r]) in ("DEV", "TEST") &&
               Int(table.model_step_hours[r]) in stack.supported_model_steps]
    rows = keep[1:stride:end]
    length(rows) >= min_rows || error(
        "the archived table supplies $(length(rows)) scorable rows, fewer than the required $min_rows",
    )
    _v22_identity_log("  archived rows $(length(rows)) of $(nrow(table)) (stride $stride)")

    worst = 0.0
    worst_row = 0
    worst_coupling = 0.0
    worst_projected = 0.0
    projected_rows = 0
    per_step = Dict{Int,NamedTuple{(:n, :worst),Tuple{Int,Float64}}}()
    per_regime = Dict{String,NamedTuple{(:n, :worst),Tuple{Int,Float64}}}()
    pooled_fallback = 0
    for r in rows
        step = Int(table.model_step_hours[r])
        result = v22_serving_center(
            stack;
            model_steps = step,
            latest_dst = Float64(table.latest_dst_nt[r]),
            dst_delta_1h_nt = Float64(table.dst_delta_1h_nt[r]),
            vbsouth_mvm = Float64(table.VBsouth_mvm[r]),
            served_v2_1 = Float64(table.served_v2_1_dst_nt[r]),
            frozen_v2_1 = Float64(table.frozen_v2_1_dst_nt[r]),
            persistence = Float64(table.persistence_dst_nt[r]),
            burton = Float64(table.burton_dst_nt[r]),
            burton_full = Float64(table.burton_full_dst_nt[r]),
            obrien = Float64(table.obrien_dst_nt[r]),
        )
        # The archived column carries the unprojected stack sum. The served center additionally takes
        # the physical projection every other served center takes, so the identity is checked against
        # `raw_center` and the projection is reported separately: it can only differ on rows whose
        # observed Dst already sits above the +50 nT ceiling.
        delta = abs(result.raw_center - Float64(table.static_v2_2_dst_nt[r]))
        projected_delta = abs(result.center - Float64(table.static_v2_2_dst_nt[r]))
        worst_projected = max(worst_projected, projected_delta)
        projected_delta > V22_IDENTITY_TOL_NT && (projected_rows += 1)
        if delta > worst
            worst = delta
            worst_row = r
        end
        # The coupling gate is recomputed from the rectified coupling and the one-hour rate rather
        # than read back, so a gate that silently stayed engaged through recovery would show up here.
        coupling_delta = abs(result.coupling_active_mvm - Float64(table.coupling_active_mvm[r]))
        worst_coupling = max(worst_coupling, coupling_delta)
        result.used_pooled_fallback && (pooled_fallback += 1)
        entry = get(per_step, step, (n = 0, worst = 0.0))
        per_step[step] = (n = entry.n + 1, worst = max(entry.worst, delta))
        regime = String(result.regime)
        rentry = get(per_regime, regime, (n = 0, worst = 0.0))
        per_regime[regime] = (n = rentry.n + 1, worst = max(rentry.worst, delta))
    end

    summary = NamedTuple[]
    for step in sort(collect(keys(per_step)))
        entry = per_step[step]
        push!(summary, (scope = "model_step_hours", key = string(step), n = entry.n,
                        max_abs_delta_nt = entry.worst))
        _v22_identity_log(@sprintf("    step %d h: n=%d  max|Δ| = %.3e nT", step, entry.n,
                                   entry.worst))
    end
    for regime in sort(collect(keys(per_regime)))
        entry = per_regime[regime]
        push!(summary, (scope = "regime", key = regime, n = entry.n,
                        max_abs_delta_nt = entry.worst))
        _v22_identity_log(@sprintf("    regime %-16s n=%d  max|Δ| = %.3e nT", regime, entry.n,
                                   entry.worst))
    end
    push!(summary, (scope = "all", key = "pooled", n = length(rows), max_abs_delta_nt = worst))
    push!(summary, (scope = "all", key = "coupling_gate", n = length(rows),
                    max_abs_delta_nt = worst_coupling))
    push!(summary, (scope = "all", key = "physical_projection", n = projected_rows,
                    max_abs_delta_nt = worst_projected))
    mkpath(dirname(String(out)))
    CSV.write(String(out), DataFrame(summary))

    passed = worst <= V22_IDENTITY_TOL_NT && worst_coupling <= V22_IDENTITY_TOL_NT &&
             length(rows) >= min_rows
    _v22_identity_log(@sprintf(
        "  rows %d, pooled-fallback cells %d, max|Δ| stack sum = %.3e nT, max|Δ| coupling gate = %.3e: %s",
        length(rows), pooled_fallback, worst, worst_coupling, passed ? "PASS" : "FAIL"))
    _v22_identity_log(@sprintf(
        "  physical projection moved %d of %d rows (max %.3f nT); those rows have observed Dst above the +50 nT ceiling",
        projected_rows, length(rows), worst_projected))
    worst > V22_IDENTITY_TOL_NT && _v22_identity_log(
        "  worst row: issue=$(table.issue_time_utc[worst_row]) step=$(table.model_step_hours[worst_row])",
    )
    return (rows = length(rows), worst_nt = worst, worst_coupling_mvm = worst_coupling,
            worst_projected_nt = worst_projected, projected_rows = projected_rows,
            per_step = per_step, per_regime = per_regime, summary = DataFrame(summary),
            pooled_fallback = pooled_fallback, passed = passed, stack_label = stack.label,
            out = String(out))
end

function main_v2_2_served_identity(args = ARGS)
    stack_path = joinpath(OPERATIONAL_PACKAGE_ROOT, "deploy", V22_SERVED_STACK_FILE)
    min_rows = V22_IDENTITY_MIN_ROWS
    stride = 1
    out = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_2_served_identity.csv")
    for arg in args
        if startswith(arg, "--stack=")
            stack_path = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--rows=")
            min_rows = parse(Int, String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--stride=")
            stride = parse(Int, String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--out=")
            out = abspath(String(split(arg, "=", limit = 2)[2]))
        else
            error("unknown option $arg")
        end
    end
    _v22_identity_log("V2.2 served static-stack identity oracle")
    result = run_v2_2_served_identity(; stack_path = stack_path, min_rows = min_rows,
                                      stride = stride, out = out)
    result.passed || error(@sprintf(
        "V2.2 served identity FAILED: max|Δ| = %.3e nT over %d rows",
        result.worst_nt, result.rows))
    _v22_identity_log("V2.2 served static-stack identity PASS")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_2_served_identity()
end
