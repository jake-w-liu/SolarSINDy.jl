#!/usr/bin/env julia

# Development-only feasibility probe for the preregistered V2.2-M1 trajectory
# augmentation. It reads a pinned pre-2018 slice of the historical OMNI file,
# prints metrics, and writes no artifacts. This is not a promotion runner.

using SolarSINDy
using Dates
using LinearAlgebra
using SHA
using Statistics

const V22_M1_PROBE_OMNI_SHA256 =
    "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"
const V22_M1_PROBE_TAUS_H = (2.0, 6.0, 12.0)
const V22_M1_PROBE_LEADS_H = (1, 2, 3, 4, 6, 7)
const V22_M1_PROBE_HISTORY_H = 72
const V22_M1_PROBE_TRAIN_END = DateTime(2016, 12, 31, 23)
const V22_M1_PROBE_TEST_START = DateTime(2017, 1, 8, 0)
const V22_M1_PROBE_TEST_END = DateTime(2017, 10, 20, 15)

_v22_m1_probe_sha256(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

_v22_m1_probe_coupling(driver) =
    1.0e-3 * Float64(driver.V) * max(-Float64(driver.Bz), 0.0)

function _v22_m1_probe_history(driver_at, issue::DateTime, tau_h::Float64)
    rho = exp(-1.0 / tau_h)
    memory = 0.0
    for lag in V22_M1_PROBE_HISTORY_H:-1:1
        driver = get(driver_at, issue - Hour(lag), nothing)
        driver === nothing && return nothing
        memory = rho * memory + (1.0 - rho) * _v22_m1_probe_coupling(driver)
    end
    return memory
end

function _v22_m1_probe_step(core, state::Float64, memory::Float64,
                            driver, eta::NTuple{3,Float64}, tau_h::Float64)
    coupling = _v22_m1_probe_coupling(driver)
    features = (memory, coupling - memory, state * memory)
    augmentation = sum(eta[i] * features[i] for i in 1:3)
    base = operational_core_forecast(core, state, driver, 1)[1]
    next_state = clamp(base + augmentation, -2000.0, 50.0)
    rho = exp(-1.0 / tau_h)
    next_memory = rho * memory + (1.0 - rho) * coupling
    return next_state, next_memory
end

function _v22_m1_probe_constrained_fit(matrix::Matrix{Float64},
                                       target::Vector{Float64})
    size(matrix, 1) == length(target) || throw(DimensionMismatch())
    best = (loss=Inf, eta=(0.0, 0.0, 0.0), support=0)
    # Exhaustive sparse supports plus exact sign-boundary solutions. A boundary
    # coefficient is represented by a smaller support, so no post-fit clipping
    # is needed.
    for support in 0:7
        columns = [i for i in 1:3 if !iszero(support & (1 << (i - 1)))]
        eta = zeros(3)
        if !isempty(columns)
            eta[columns] = matrix[:, columns] \ target
        end
        eta[1] <= 0.0 || continue
        eta[3] <= 0.0 || continue
        residual = target - matrix * eta
        loss = mean(abs2, residual)
        if loss < best.loss - 1.0e-12 ||
           (abs(loss - best.loss) <= 1.0e-12 && count_ones(support) < count_ones(best.support))
            best = (loss=loss, eta=Tuple(eta), support=support)
        end
    end
    isfinite(best.loss) || error("no feasible V2.2-M1 constrained fit")
    return best
end

function _v22_m1_probe_inputs(path::AbstractString)
    isfile(path) && !islink(path) || error("OMNI probe source must be a regular file")
    _v22_m1_probe_sha256(path) == V22_M1_PROBE_OMNI_SHA256 ||
        error("OMNI probe source hash changed")
    frame = parse_omni2(String(path); year_start=2010, year_end=2017)
    clean_omni_data!(frame; causal=true)
    driver_at = Dict{DateTime,NamedTuple}()
    dst_at = Dict{DateTime,Float64}()
    for row in eachrow(frame)
        time = DateTime(row.datetime)
        dst = Float64(row.Dst)
        isfinite(dst) && (dst_at[time] = dst)
        values = Float64.(Tuple((row.V, row.Bz, row.By, row.n, row.Pdyn)))
        all(isfinite, values) || continue
        values[1] >= 0 && values[4] >= 0 && values[5] >= 0 || continue
        driver_at[time] = (
            V=values[1], Bz=values[2], By=values[3],
            n=values[4], Pdyn=values[5],
        )
    end
    return driver_at, dst_at
end

function _v22_m1_probe_anchor(driver_at, dst_at, issue::DateTime,
                              tau_h::Float64)
    issue_driver = get(driver_at, issue - Hour(1), nothing)
    issue_dst = get(dst_at, issue, NaN)
    memory = _v22_m1_probe_history(driver_at, issue, tau_h)
    issue_driver === nothing && return nothing
    memory === nothing && return nothing
    isfinite(issue_dst) || return nothing
    targets = NamedTuple[]
    for lead in V22_M1_PROBE_LEADS_H
        target_time = issue + Hour(lead)
        target_driver = get(driver_at, target_time - Hour(1), nothing)
        target_dst = get(dst_at, target_time, NaN)
        target_driver === nothing && return nothing
        isfinite(target_dst) || return nothing
        push!(targets, (
            lead=lead,
            state=dst_to_dst_star(target_dst, target_driver.Pdyn),
        ))
    end
    state = dst_to_dst_star(issue_dst, issue_driver.Pdyn)
    return (; issue, issue_driver, state, memory, targets)
end

function _v22_m1_probe_fit_rows(core, driver_at, dst_at, tau_h::Float64)
    features = NTuple{3,Float64}[]
    targets = Float64[]
    first_issue = DateTime(2010, 1, 4, 0)
    for issue in first_issue:Hour(1):V22_M1_PROBE_TRAIN_END
        anchor = _v22_m1_probe_anchor(driver_at, dst_at, issue, tau_h)
        anchor === nothing && continue
        coupling = _v22_m1_probe_coupling(anchor.issue_driver)
        push!(features, (
            anchor.memory,
            coupling - anchor.memory,
            anchor.state * anchor.memory,
        ))
        base = operational_core_forecast(
            core, anchor.state, anchor.issue_driver, 1,
        )[1]
        push!(targets, anchor.targets[1].state - base)
    end
    isempty(features) && error("V2.2-M1 probe has no fit rows")
    matrix = Matrix{Float64}(undef, length(features), 3)
    for row in eachindex(features), column in 1:3
        matrix[row, column] = features[row][column]
    end
    return matrix, targets
end

_v22_m1_probe_rmse(observed, predicted) =
    sqrt(mean(abs2, observed .- predicted))

function _v22_m1_probe_score(core, driver_at, dst_at, tau_h::Float64,
                             eta::NTuple{3,Float64})
    observations = Dict(lead => Float64[] for lead in V22_M1_PROBE_LEADS_H)
    base_predictions = Dict(lead => Float64[] for lead in V22_M1_PROBE_LEADS_H)
    m1_predictions = Dict(lead => Float64[] for lead in V22_M1_PROBE_LEADS_H)
    oracle_predictions = Dict(lead => Float64[] for lead in V22_M1_PROBE_LEADS_H)
    for issue in V22_M1_PROBE_TEST_START:Hour(1):V22_M1_PROBE_TEST_END
        anchor = _v22_m1_probe_anchor(driver_at, dst_at, issue, tau_h)
        anchor === nothing && continue
        base_state = anchor.state
        m1_state = anchor.state
        oracle_state = anchor.state
        m1_memory = anchor.memory
        for step in 1:maximum(V22_M1_PROBE_LEADS_H)
            base_state = operational_core_forecast(
                core, base_state, anchor.issue_driver, 1,
            )[1]
            m1_state, m1_memory = _v22_m1_probe_step(
                core, m1_state, m1_memory, anchor.issue_driver, eta, tau_h,
            )
            realized_driver = get(driver_at, issue + Hour(step - 1), nothing)
            realized_driver === nothing && error("complete anchor lost a realized driver")
            oracle_state = operational_core_forecast(
                core, oracle_state, realized_driver, 1,
            )[1]
            step in V22_M1_PROBE_LEADS_H || continue
            target = only(row.state for row in anchor.targets if row.lead == step)
            push!(observations[step], target)
            push!(base_predictions[step], base_state)
            push!(m1_predictions[step], m1_state)
            push!(oracle_predictions[step], oracle_state)
        end
    end
    rows = NamedTuple[]
    for lead in V22_M1_PROBE_LEADS_H
        observed = observations[lead]
        base_rmse = _v22_m1_probe_rmse(observed, base_predictions[lead])
        m1_rmse = _v22_m1_probe_rmse(observed, m1_predictions[lead])
        oracle_rmse = _v22_m1_probe_rmse(observed, oracle_predictions[lead])
        push!(rows, (
            lead=lead,
            rows=length(observed),
            base_rmse=base_rmse,
            m1_rmse=m1_rmse,
            m1_gain=base_rmse - m1_rmse,
            oracle_rmse=oracle_rmse,
            oracle_gain=base_rmse - oracle_rmse,
        ))
    end
    return rows
end

function main_v2_2_m1_probe()
    path = strip(get(ENV, "SOLARSINDY_OMNI_EXTRACTED", ""))
    isempty(path) && error("set SOLARSINDY_OMNI_EXTRACTED to the pinned OMNI CSV")
    driver_at, dst_at = _v22_m1_probe_inputs(abspath(path))
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    for tau_h in V22_M1_PROBE_TAUS_H
        matrix, target = _v22_m1_probe_fit_rows(core, driver_at, dst_at, tau_h)
        fit_result = _v22_m1_probe_constrained_fit(matrix, target)
        println("tau_h=", tau_h, " support=", fit_result.support,
                " eta=", fit_result.eta, " derivative_rmse=", sqrt(fit_result.loss))
        for row in _v22_m1_probe_score(
            core, driver_at, dst_at, tau_h, fit_result.eta,
        )
            println("  lead=", row.lead, " rows=", row.rows,
                    " base=", row.base_rmse, " m1=", row.m1_rmse,
                    " m1_gain=", row.m1_gain,
                    " noncausal_oracle=", row.oracle_rmse,
                    " oracle_gain=", row.oracle_gain)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_2_m1_probe()
end
