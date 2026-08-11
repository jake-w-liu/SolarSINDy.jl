#!/usr/bin/env julia

# Purged pre-2018 cross-fit for the preregistered Operational V2.2-M1 sparse
# trajectory augmentation. This runner reads only two pinned pre-2023 sources.

using SolarSINDy
using CSV
using DataFrames
using Dates
using Random
using SHA
using Statistics

const V22_HISTORY_LEADS_H = (1, 2, 3, 4, 6, 7)
const V22_HISTORY_TAUS_H = (2.0, 6.0, 12.0)
const V22_HISTORY_SUPPORTS = Tuple(
    ntuple(index -> !iszero(mask & (1 << (index - 1))), 3)
    for mask in 0:7
)
const V22_HISTORY_LOOKBACK_H = 72
const V22_HISTORY_PURGE_H = 168
const V22_HISTORY_COUPLING_BOUND_MVM = 50.0
const V22_HISTORY_EFFECT_GATE_NT = 0.25
const V22_HISTORY_AVAILABILITY_GATE = 0.99
const V22_HISTORY_OMNI_SHA256 =
    "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"
const V22_HISTORY_COMPARATOR_SHA256 =
    "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43"
const V22_HISTORY_COMPARATOR_COLUMNS = (
    :served_v2_1_dst_nt,
    :frozen_v2_1_dst_nt,
    :raw_sindy_dst_nt,
    :persistence_dst_nt,
    :burton_dst_nt,
    :burton_full_dst_nt,
    :obrien_dst_nt,
)
const V22_HISTORY_FOLDS = (
    (label="calendar_2013", start=DateTime(2013, 1, 8),
     stop=DateTime(2013, 12, 24, 23)),
    (label="calendar_2014", start=DateTime(2014, 1, 8),
     stop=DateTime(2014, 12, 24, 23)),
    (label="calendar_2015", start=DateTime(2015, 1, 8),
     stop=DateTime(2015, 12, 24, 23)),
    (label="calendar_2016", start=DateTime(2016, 1, 8),
     stop=DateTime(2016, 12, 24, 23)),
    (label="calendar_2017", start=DateTime(2017, 1, 8),
     stop=DateTime(2017, 10, 20, 15)),
)

const V22_HISTORY_OUTPUT_DIR = joinpath(
    @__DIR__, "..", "output", "operational", "v2_2_history",
)
const V22_HISTORY_SUMMARY_CSV = joinpath(
    V22_HISTORY_OUTPUT_DIR, "v2_2_history_crossfit_summary.csv",
)
const V22_HISTORY_DECISION_CSV = joinpath(
    V22_HISTORY_OUTPUT_DIR, "v2_2_history_crossfit_decision.csv",
)
const V22_HISTORY_SELECTED_OOF_CSV = joinpath(
    V22_HISTORY_OUTPUT_DIR, "v2_2_history_selected_oof.csv",
)

_v22_history_sha256(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

_v22_history_rmse(observed, predicted) =
    sqrt(mean(abs2, observed .- predicted))

_v22_history_support_text(support) = join(Int.(support), "")

function _v22_history_regular_pinned_file(path::AbstractString,
                                          expected_sha256::AbstractString,
                                          label::AbstractString)
    source = abspath(String(path))
    isfile(source) && !islink(source) || error(
        "$label must be a regular non-symlink file: $source",
    )
    _v22_history_sha256(source) == expected_sha256 || error(
        "$label SHA-256 changed: $source",
    )
    return source
end

function _v22_history_sources()
    omni = strip(get(ENV, "SOLARSINDY_OMNI_EXTRACTED", ""))
    comparator = strip(get(ENV, "SOLARSINDY_V22_COMPARATOR_REPLAY", ""))
    isempty(omni) && error("set SOLARSINDY_OMNI_EXTRACTED")
    isempty(comparator) && error("set SOLARSINDY_V22_COMPARATOR_REPLAY")
    return (
        omni=_v22_history_regular_pinned_file(
            omni, V22_HISTORY_OMNI_SHA256, "V2.2-M1 OMNI source",
        ),
        comparator=_v22_history_regular_pinned_file(
            comparator, V22_HISTORY_COMPARATOR_SHA256,
            "V2.2-M1 comparator replay",
        ),
    )
end

function _v22_history_comparator_table(path::AbstractString)
    columns = Symbol[
        :issue_time_utc, :target_time_utc, :model_step_hours,
        :observation_dst_nt, :latest_dst_nt, :dst_delta_1h_nt,
        :coupling_active_mvm,
    ]
    append!(columns, V22_HISTORY_COMPARATOR_COLUMNS)
    table = CSV.read(path, DataFrame; select=String.(columns), types=Dict(
        :issue_time_utc => DateTime,
        :target_time_utc => DateTime,
    ))
    missing_columns = setdiff(columns, Symbol.(names(table)))
    isempty(missing_columns) || error(
        "V2.2-M1 comparator replay omits: $(join(missing_columns, ','))",
    )
    select!(table, columns)
    all(table.target_time_utc .== table.issue_time_utc .+
        Hour.(Int.(table.model_step_hours))) || error(
        "V2.2-M1 comparator target semantics are inconsistent",
    )
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) || error(
        "V2.2-M1 comparator replay contains a post-2022 target",
    )
    keys = Tuple.(eachrow(select(
        table, :issue_time_utc, :target_time_utc, :model_step_hours,
    )))
    length(unique(keys)) == nrow(table) || error(
        "V2.2-M1 comparator replay has duplicate keys",
    )
    numeric = names(table, Real)
    all(column -> all(isfinite, Float64.(table[!, column])), numeric) || error(
        "V2.2-M1 comparator replay contains non-finite numeric values",
    )
    sort!(unique(Int.(table.model_step_hours))) == collect(V22_HISTORY_LEADS_H) ||
        error("V2.2-M1 comparator replay does not cover all six leads")
    return table
end

function _v22_history_driver_table(path::AbstractString)
    frame = parse_omni2(String(path); year_start=2010, year_end=2017)
    clean_omni_data!(frame; causal=true)
    driver_at = Dict{DateTime,OperationalV22HistoryDriver}()
    for row in eachrow(frame)
        values = Float64.(Tuple((row.V, row.Bz, row.By, row.n, row.Pdyn)))
        all(isfinite, values) || continue
        values[1] >= 0.0 && values[4] >= 0.0 && values[5] >= 0.0 || continue
        driver_at[DateTime(row.datetime)] = OperationalV22HistoryDriver(values...)
    end
    isempty(driver_at) && error("V2.2-M1 OMNI source has no finite drivers")
    return driver_at
end

function _v22_history_causal_drivers(driver_at, issue::DateTime,
                                     rate_nt_per_h::Real)
    issue_driver = get(driver_at, issue - Hour(1), nothing)
    issue_driver === nothing && return nothing
    rate = isfinite(Float64(rate_nt_per_h)) ? Float64(rate_nt_per_h) : 0.0
    tau_tail_h = min(48.0, 3.0 * (1.0 + max(0.0, -rate) / 7.5))
    drivers = OperationalV22HistoryDriver[]
    for step in 1:maximum(V22_HISTORY_LEADS_H)
        # Earth-shifted OMNI does not retain the upstream observation or receipt
        # chronology needed to prove that a future Earth hour was L1-known at
        # issue time. The deployable retrospective arm therefore uses no
        # post-issue OMNI row. Realized rows are isolated in the oracle below.
        relaxation = exp(-step / tau_tail_h)
        driver = OperationalV22HistoryDriver(
            issue_driver.speed_km_s,
            issue_driver.bz_nt * relaxation,
            issue_driver.by_nt * relaxation,
            issue_driver.density_cm3,
            issue_driver.pdyn_npa,
        )
        push!(drivers, driver)
    end
    return drivers
end

function _v22_history_realized_drivers(driver_at, issue::DateTime)
    drivers = OperationalV22HistoryDriver[]
    for step in 1:maximum(V22_HISTORY_LEADS_H)
        driver = get(driver_at, issue + Hour(step - 1), nothing)
        driver === nothing && return nothing
        push!(drivers, driver)
    end
    return drivers
end

function _v22_history_initial_state(artifact::OperationalV22HistoryArtifact,
                                    driver_at, issue::DateTime,
                                    current_driver::OperationalV22HistoryDriver,
                                    dst_star_nt::Float64)
    -2000.0 <= dst_star_nt <= 50.0 || return nothing
    timestamps = collect(
        (issue - Hour(V22_HISTORY_LOOKBACK_H)):Hour(1):issue,
    )
    history = OperationalV22HistoryDriver[]
    for timestamp in timestamps[1:(end - 1)]
        driver = get(driver_at, timestamp, nothing)
        driver === nothing && return nothing
        operational_v22_history_coupling(driver) <=
            artifact.coupling_bound_mvm || return nothing
        push!(history, driver)
    end
    operational_v22_history_coupling(current_driver) <=
        artifact.coupling_bound_mvm || return nothing
    push!(history, current_driver)
    return init_operational_v22_history_state(
        artifact, timestamps, history, dst_star_nt,
    )
end

function _v22_history_anchor_rows(table::DataFrame)
    grouped = Dict{DateTime,DataFrame}()
    for group in groupby(table, :issue_time_utc)
        frame = DataFrame(group)
        sort!(frame, :model_step_hours)
        Tuple(Int.(frame.model_step_hours)) == V22_HISTORY_LEADS_H || continue
        grouped[DateTime(frame.issue_time_utc[1])] = frame
    end
    isempty(grouped) && error("V2.2-M1 source has no complete six-lead anchors")
    return grouped
end

function _v22_history_zero_artifact(core, tau_h::Float64)
    return OperationalV22HistoryArtifact(
        core, (0.0, 0.0, 0.0);
        tau_memory_hours=tau_h,
        coupling_bound_mvm=V22_HISTORY_COUPLING_BOUND_MVM,
        fit_rows=3,
        support_mask=(false, false, false),
        label="v2_2_m1_zero_tau$(Int(tau_h))",
    )
end

function _v22_history_rollout_after_identity_check(
        core, artifact, initial_state, drivers, theta)
    dst = Vector{Float64}(undef, length(drivers))
    state = initial_state
    for index in eachindex(drivers)
        result = SolarSINDy._operational_v22_history_step_unchecked(
            core, artifact, state, drivers[index], theta,
        )
        state = result.next_state
        dst[index] = state.dst_star_nt
    end
    return dst
end

function _v22_history_fit_artifact(core, anchors, driver_at,
                                   fold, tau_h::Float64, support)
    zero = _v22_history_zero_artifact(core, tau_h)
    SolarSINDy._operational_v22_history_validate_identity(core, zero)
    theta = Vector{Float64}(undef, length(core.library))
    dst = Float64[]
    memory = Float64[]
    coupling = Float64[]
    residual = Float64[]
    # The full six-lead target of the last fitted anchor must mature at least
    # 168 h before the test block begins.
    maximum_fit_issue = fold.start - Hour(
        V22_HISTORY_PURGE_H + maximum(V22_HISTORY_LEADS_H),
    )
    for issue in sort!(collect(keys(anchors)))
        issue <= maximum_fit_issue || break
        rows = anchors[issue]
        lead_one = rows[rows.model_step_hours .== 1, :]
        nrow(lead_one) == 1 || error("V2.2-M1 anchor lacks one lead-one row")
        row = lead_one[1, :]
        drivers = _v22_history_causal_drivers(
            driver_at, issue, row.dst_delta_1h_nt,
        )
        drivers === nothing && continue
        issue_driver = get(driver_at, issue - Hour(1), nothing)
        target_driver = get(driver_at, issue, nothing)
        issue_driver === nothing && continue
        target_driver === nothing && continue
        state_dst = dst_to_dst_star(row.latest_dst_nt, issue_driver.pdyn_npa)
        state = _v22_history_initial_state(
            zero, driver_at, issue, first(drivers), state_dst,
        )
        state === nothing && continue
        target_dst = dst_to_dst_star(
            row.observation_dst_nt, target_driver.pdyn_npa,
        )
        diagnostic = SolarSINDy._operational_v22_history_derivative_unchecked(
            core, zero, state, first(drivers), theta,
        )
        push!(dst, state.dst_star_nt)
        push!(memory, state.memory_mvm)
        push!(coupling, diagnostic.coupling_mvm)
        push!(residual, target_dst - state.dst_star_nt -
                        diagnostic.base_derivative_nt_per_h)
    end
    length(dst) >= 1000 || error(
        "V2.2-M1 fold $(fold.label) has only $(length(dst)) fit rows",
    )
    artifact = fit_operational_v22_history(
        core, dst, memory, coupling, residual;
        tau_memory_hours=tau_h,
        coupling_bound_mvm=V22_HISTORY_COUPLING_BOUND_MVM,
        support_mask=support,
        label="v2_2_m1_$(fold.label)_tau$(Int(tau_h))_s$(_v22_history_support_text(support))",
    )
    return artifact, maximum_fit_issue, length(dst)
end

function _v22_history_score_fold(core, artifact, anchors, driver_at, fold,
                                 maximum_fit_issue::DateTime, fit_rows::Int)
    rows = NamedTuple[]
    eligible_anchors = 0
    scheduled_anchors = 0
    zero = _v22_history_zero_artifact(core, artifact.tau_memory_hours)
    SolarSINDy._operational_v22_history_validate_identity(core, artifact)
    SolarSINDy._operational_v22_history_validate_identity(core, zero)
    artifact_sha256 = operational_v22_history_sha256(artifact)
    theta_candidate = Vector{Float64}(undef, length(core.library))
    theta_zero = Vector{Float64}(undef, length(core.library))
    for issue in sort!(collect(keys(anchors)))
        fold.start <= issue <= fold.stop || continue
        scheduled_anchors += 1
        source = anchors[issue]
        row_one = only(eachrow(source[source.model_step_hours .== 1, :]))
        causal_drivers = _v22_history_causal_drivers(
            driver_at, issue, row_one.dst_delta_1h_nt,
        )
        realized_drivers = _v22_history_realized_drivers(driver_at, issue)
        causal_drivers === nothing && continue
        realized_drivers === nothing && continue
        issue_driver = get(driver_at, issue - Hour(1), nothing)
        issue_driver === nothing && continue
        state_dst = dst_to_dst_star(
            row_one.latest_dst_nt, issue_driver.pdyn_npa,
        )
        state = _v22_history_initial_state(
            artifact, driver_at, issue, first(causal_drivers), state_dst,
        )
        state === nothing && continue
        zero_state = OperationalV22HistoryState(
            state.t_current, state.dst_star_nt, state.memory_mvm,
        )
        candidate_dst = _v22_history_rollout_after_identity_check(
            core, artifact, state, causal_drivers, theta_candidate,
        )
        causal_base_dst = _v22_history_rollout_after_identity_check(
            core, zero, zero_state, causal_drivers, theta_zero,
        )
        realized_oracle_dst = _v22_history_rollout_after_identity_check(
            core, zero, zero_state, realized_drivers, theta_zero,
        )
        eligible_anchors += 1
        regime = operational_v22_regime(
            row_one.latest_dst_nt,
            row_one.dst_delta_1h_nt,
            row_one.coupling_active_mvm,
        )
        for row in eachrow(source)
            lead = Int(row.model_step_hours)
            index = lead
            target_driver = get(driver_at, issue + Hour(lead - 1), nothing)
            target_driver === nothing && error("eligible target driver disappeared")
            target_star = dst_to_dst_star(
                row.observation_dst_nt, target_driver.pdyn_npa,
            )
            candidate_raw = dst_star_to_dst(
                candidate_dst[index], causal_drivers[index].pdyn_npa,
            )
            causal_raw = dst_star_to_dst(
                causal_base_dst[index], causal_drivers[index].pdyn_npa,
            )
            oracle_raw = dst_star_to_dst(
                realized_oracle_dst[index],
                realized_drivers[index].pdyn_npa,
            )
            push!(rows, (
                fold=fold.label,
                issue_time_utc=issue,
                target_time_utc=DateTime(row.target_time_utc),
                model_step_hours=lead,
                tau_memory_hours=artifact.tau_memory_hours,
                support_mask=_v22_history_support_text(artifact.support_mask),
                artifact_sha256=artifact_sha256,
                maximum_fit_issue_utc=maximum_fit_issue,
                fit_rows=fit_rows,
                scheduled_anchors=scheduled_anchors,
                eligible_anchors=eligible_anchors,
                issue_regime=String(regime),
                latest_dst_nt=Float64(row.latest_dst_nt),
                dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                observation_dst_nt=Float64(row.observation_dst_nt),
                observation_dst_star_nt=target_star,
                v2_2_m1_dst_nt=candidate_raw,
                v2_2_m1_dst_star_nt=candidate_dst[index],
                causal_v2_1_core_dst_nt=causal_raw,
                causal_v2_1_core_dst_star_nt=causal_base_dst[index],
                noncausal_input_oracle_dst_nt=oracle_raw,
                noncausal_input_oracle_dst_star_nt=
                    realized_oracle_dst[index],
                served_v2_1_dst_nt=Float64(row.served_v2_1_dst_nt),
                frozen_v2_1_dst_nt=Float64(row.frozen_v2_1_dst_nt),
                raw_sindy_dst_nt=Float64(row.raw_sindy_dst_nt),
                persistence_dst_nt=Float64(row.persistence_dst_nt),
                burton_dst_nt=Float64(row.burton_dst_nt),
                burton_full_dst_nt=Float64(row.burton_full_dst_nt),
                obrien_dst_nt=Float64(row.obrien_dst_nt),
            ))
        end
    end
    scheduled_anchors > 0 || error("V2.2-M1 fold $(fold.label) has no scheduled anchors")
    eligible_anchors > 0 || error("V2.2-M1 fold $(fold.label) has no eligible anchors")
    out = DataFrame(rows)
    out[!, :scheduled_anchors] .= scheduled_anchors
    out[!, :eligible_anchors] .= eligible_anchors
    return out
end

function _v22_history_candidate_oof(core, anchors, driver_at,
                                    tau_h::Float64, support)
    frames = DataFrame[]
    for fold in V22_HISTORY_FOLDS
        artifact, maximum_fit_issue, fit_rows = _v22_history_fit_artifact(
            core, anchors, driver_at, fold, tau_h, support,
        )
        push!(frames, _v22_history_score_fold(
            core, artifact, anchors, driver_at, fold,
            maximum_fit_issue, fit_rows,
        ))
    end
    return reduce(vcat, frames)
end

function _v22_history_best_comparator(rows::DataFrame)
    observed = Float64.(rows.observation_dst_nt)
    metrics = [(column, _v22_history_rmse(observed, Float64.(rows[!, column])))
               for column in V22_HISTORY_COMPARATOR_COLUMNS]
    sort!(metrics; by=last)
    return first(metrics)
end

function _v22_history_metrics(oof::DataFrame)
    output = NamedTuple[]
    total_scheduled = sum(unique(select(
        oof, :fold, :scheduled_anchors,
    )).scheduled_anchors)
    total_eligible = sum(unique(select(
        oof, :fold, :eligible_anchors,
    )).eligible_anchors)
    availability = total_eligible / total_scheduled
    for lead in V22_HISTORY_LEADS_H
        lead_rows = oof[oof.model_step_hours .== lead, :]
        nrow(lead_rows) > 0 || error("V2.2-M1 OOF lacks lead $lead")
        comparator, comparator_rmse = _v22_history_best_comparator(lead_rows)
        observed = Float64.(lead_rows.observation_dst_nt)
        candidate_rmse = _v22_history_rmse(
            observed, Float64.(lead_rows.v2_2_m1_dst_nt),
        )
        causal_rmse = _v22_history_rmse(
            observed, Float64.(lead_rows.causal_v2_1_core_dst_nt),
        )
        oracle_rmse = _v22_history_rmse(
            observed, Float64.(lead_rows.noncausal_input_oracle_dst_nt),
        )
        regime_safe = true
        active_recovery_safe = true
        for regime in ("quiet", "active_deepening", "recovery")
            stratum = lead_rows[lead_rows.issue_regime .== regime, :]
            nrow(stratum) >= 40 || continue
            stratum_observed = Float64.(stratum.observation_dst_nt)
            base = Float64.(stratum[!, comparator])
            candidate = Float64.(stratum.v2_2_m1_dst_nt)
            loss = _v22_history_rmse(stratum_observed, candidate) -
                   _v22_history_rmse(stratum_observed, base)
            regime_safe &= loss <= 0.50
            if lead in (6, 7) && regime in ("active_deepening", "recovery")
                active_recovery_safe &= loss <= 0.0
            end
        end
        push!(output, (
            tau_memory_hours=only(unique(oof.tau_memory_hours)),
            support_mask=only(unique(oof.support_mask)),
            lead_h=lead,
            rows=nrow(lead_rows),
            availability=availability,
            candidate_rmse_nt=candidate_rmse,
            best_comparator=String(comparator),
            best_comparator_rmse_nt=comparator_rmse,
            gain_vs_best_nt=comparator_rmse - candidate_rmse,
            causal_core_rmse_nt=causal_rmse,
            gain_vs_causal_core_nt=causal_rmse - candidate_rmse,
            noncausal_input_oracle_rmse_nt=oracle_rmse,
            noncausal_input_oracle_gain_vs_best_nt=comparator_rmse - oracle_rmse,
            regime_safe=regime_safe,
            active_recovery_safe=active_recovery_safe,
        ))
    end
    return DataFrame(output)
end

function _v22_history_candidate_pass(summary::DataFrame)
    all(summary.availability .>= V22_HISTORY_AVAILABILITY_GATE) || return false
    all(summary.gain_vs_best_nt .>= V22_HISTORY_EFFECT_GATE_NT) || return false
    all(summary.regime_safe) || return false
    all(summary.active_recovery_safe) || return false
    return true
end

function _v22_history_simultaneous_bootstrap(oof::DataFrame;
                                             replicates::Int=10_000,
                                             seed::Int=22_022_026)
    replicates > 0 || throw(ArgumentError("bootstrap replicates must be positive"))
    epoch = DateTime(2010, 1, 1)
    block_ms = V22_HISTORY_PURGE_H * 3_600_000
    block_ids = sort!(unique([
        div(Dates.value(issue - epoch), block_ms)
        for issue in DateTime.(oof.issue_time_utc)
    ]))
    length(block_ids) >= 10 || error("V2.2-M1 bootstrap has too few 168 h blocks")
    block_index = Dict(value => index for (index, value) in enumerate(block_ids))
    n_blocks = length(block_ids)
    n_leads = length(V22_HISTORY_LEADS_H)
    n_methods = length(V22_HISTORY_COMPARATOR_COLUMNS)
    counts = zeros(Int, n_blocks, n_leads)
    candidate_sse = zeros(Float64, n_blocks, n_leads)
    comparator_sse = zeros(Float64, n_blocks, n_leads, n_methods)
    lead_index = Dict(lead => index for (index, lead) in enumerate(V22_HISTORY_LEADS_H))
    for row in eachrow(oof)
        block = block_index[div(Dates.value(row.issue_time_utc - epoch), block_ms)]
        lead = lead_index[Int(row.model_step_hours)]
        counts[block, lead] += 1
        candidate_sse[block, lead] +=
            abs2(Float64(row.observation_dst_nt) - Float64(row.v2_2_m1_dst_nt))
        for (method, column) in enumerate(V22_HISTORY_COMPARATOR_COLUMNS)
            comparator_sse[block, lead, method] +=
                abs2(Float64(row.observation_dst_nt) - Float64(row[column]))
        end
    end
    all(counts .> 0) || error(
        "V2.2-M1 bootstrap requires every 168 h block to cover every lead",
    )
    rng = MersenneTwister(seed)
    minimum_gain = Vector{Float64}(undef, replicates)
    lead_gains = Matrix{Float64}(undef, replicates, n_leads)
    multiplicities = zeros(Int, n_blocks)
    for replicate in 1:replicates
        fill!(multiplicities, 0)
        for sampled in rand(rng, 1:n_blocks, n_blocks)
            multiplicities[sampled] += 1
        end
        for lead in 1:n_leads
            n = sum(multiplicities .* view(counts, :, lead))
            candidate_rmse = sqrt(
                sum(multiplicities .* view(candidate_sse, :, lead)) / n,
            )
            comparator_rmse = minimum(
                sqrt(sum(multiplicities .* view(comparator_sse, :, lead, method)) / n)
                for method in 1:n_methods
            )
            lead_gains[replicate, lead] = comparator_rmse - candidate_rmse
        end
        minimum_gain[replicate] = minimum(view(lead_gains, replicate, :))
    end
    return (
        simultaneous_lower_95_nt=quantile(minimum_gain, 0.05),
        per_lead_lower_95_nt=Tuple(
            quantile(view(lead_gains, :, lead), 0.05) for lead in 1:n_leads
        ),
        replicates=replicates,
        blocks=n_blocks,
        seed=seed,
    )
end

function run_v2_2_history_crossfit(; write_outputs::Bool=true)
    sources = _v22_history_sources()
    table = _v22_history_comparator_table(sources.comparator)
    anchors = _v22_history_anchor_rows(table)
    driver_at = _v22_history_driver_table(sources.omni)
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)

    summaries = DataFrame[]
    candidate_cache = Dict{Tuple{Float64,String},DataFrame}()
    for tau_h in V22_HISTORY_TAUS_H, support in V22_HISTORY_SUPPORTS
        support_text = _v22_history_support_text(support)
        println("V2.2-M1 crossfit tau=", tau_h, " support=", support_text)
        oof = _v22_history_candidate_oof(
            core, anchors, driver_at, tau_h, support,
        )
        summary = _v22_history_metrics(oof)
        push!(summaries, summary)
        candidate_cache[(tau_h, support_text)] = oof
    end
    all_summary = reduce(vcat, summaries)
    candidates = combine(groupby(
        all_summary, [:tau_memory_hours, :support_mask],
    ),
        :gain_vs_best_nt => minimum => :minimum_gain_vs_best_nt,
        :gain_vs_best_nt => mean => :mean_gain_vs_best_nt,
        :candidate_rmse_nt => mean => :mean_lead_rmse_nt,
        :availability => minimum => :minimum_availability,
        :regime_safe => all => :all_regime_safe,
        :active_recovery_safe => all => :all_active_recovery_safe,
    )
    candidates[!, :passes_crossfit] = [
        _v22_history_candidate_pass(all_summary[
            (all_summary.tau_memory_hours .== row.tau_memory_hours) .&
            (all_summary.support_mask .== row.support_mask), :,
        ]) for row in eachrow(candidates)
    ]
    sort!(candidates, [
        order(:passes_crossfit, rev=true),
        order(:minimum_gain_vs_best_nt, rev=true),
        order(:mean_gain_vs_best_nt, rev=true),
        :tau_memory_hours,
        :support_mask,
    ])
    selected = first(candidates)
    selected_oof = candidate_cache[
        (Float64(selected.tau_memory_hours), String(selected.support_mask)),
    ]
    inference = _v22_history_simultaneous_bootstrap(selected_oof)
    inference_pass = inference.simultaneous_lower_95_nt > 0.0
    verdict = Bool(selected.passes_crossfit) && inference_pass ?
        "CROSSFIT_PASS" : "NO_GO_M1"
    decision = DataFrame((
        verdict=[verdict],
        selected_tau_memory_hours=[Float64(selected.tau_memory_hours)],
        selected_support_mask=[String(selected.support_mask)],
        minimum_gain_vs_best_nt=[Float64(selected.minimum_gain_vs_best_nt)],
        mean_gain_vs_best_nt=[Float64(selected.mean_gain_vs_best_nt)],
        minimum_availability=[Float64(selected.minimum_availability)],
        all_regime_safe=[Bool(selected.all_regime_safe)],
        all_active_recovery_safe=[Bool(selected.all_active_recovery_safe)],
        simultaneous_lower_95_nt=[inference.simultaneous_lower_95_nt],
        per_lead_lower_95_nt=[join(inference.per_lead_lower_95_nt, ";")],
        bootstrap_replicates=[inference.replicates],
        bootstrap_blocks=[inference.blocks],
        bootstrap_seed=[inference.seed],
        inference_pass=[inference_pass],
        omni_sha256=[V22_HISTORY_OMNI_SHA256],
        comparator_replay_sha256=[V22_HISTORY_COMPARATOR_SHA256],
        post_2022_rows_read=[0],
        promotion_authorized=[false],
    ))
    if write_outputs
        mkpath(V22_HISTORY_OUTPUT_DIR)
        CSV.write(V22_HISTORY_SUMMARY_CSV, all_summary)
        CSV.write(V22_HISTORY_DECISION_CSV, decision)
        CSV.write(V22_HISTORY_SELECTED_OOF_CSV, selected_oof)
    end
    println(decision)
    return (; summary=all_summary, candidates, decision, selected_oof)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_v2_2_history_crossfit()
end
