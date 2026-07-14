using SolarSINDy
using CSV, DataFrames, Statistics, SHA

"""
Availability-only eligibility thresholds shared by real, phase, and coupled
discovery. A storm needs 20 original-support derivative rows and 20 original
post-anchor Dst* targets by default; the targets must cover at least 60% of the
selected contiguous forward trajectory. The anchor is not a scored target.
"""
struct DiscoveryObservationPolicy
    smooth_window::Int
    min_regression_rows::Int
    min_scoring_rows::Int
    min_scoring_fraction::Float64
    require_ae::Bool

    function DiscoveryObservationPolicy(; smooth_window::Int=5,
            min_regression_rows::Int=20, min_scoring_rows::Int=20,
            min_scoring_fraction::Real=0.60, require_ae::Bool=false)
        smooth_window > 0 && isodd(smooth_window) ||
            throw(ArgumentError("smooth_window must be a positive odd integer"))
        min_regression_rows >= 1 ||
            throw(ArgumentError("min_regression_rows must be positive"))
        min_scoring_rows >= 1 ||
            throw(ArgumentError("min_scoring_rows must be positive"))
        isfinite(min_scoring_fraction) && 0 < min_scoring_fraction <= 1 ||
            throw(ArgumentError("min_scoring_fraction must be in (0, 1]"))
        new(smooth_window, min_regression_rows, min_scoring_rows,
            Float64(min_scoring_fraction), require_ae)
    end
end

const _DISCOVERY_DRIVER_COLUMNS = (:V, :Bz, :By, :n, :Pdyn)
const _DISCOVERY_DRIVER_FLAGS = (:V_observed, :Bz_observed, :By_observed,
                                 :n_observed)
const _DISCOVERY_DRIVER_PAIRS = ((:V, :V_observed), (:Bz, :Bz_observed),
                                 (:By, :By_observed), (:n, :n_observed))
const _DISCOVERY_TARGET_FLAGS = (:Dst_observed, :V_observed, :n_observed)
const _DISCOVERY_MAX_CLEANED_GAP = 3

_finite_real(value) = value isa Real && isfinite(value)

function _resolve_discovery_policy(policy::DiscoveryObservationPolicy,
                                   smooth_window::Union{Nothing,Int})
    smooth_window === nothing && return policy
    smooth_window == policy.smooth_window || throw(ArgumentError(
        "smooth_window conflicts with the supplied observation policy"
    ))
    return policy
end

function _require_discovery_observation_state(df::DataFrame,
                                              policy::DiscoveryObservationPolicy)
    required = (_DISCOVERY_DRIVER_COLUMNS..., :Dst, :Dst_star,
                _DISCOVERY_DRIVER_FLAGS..., _DISCOVERY_TARGET_FLAGS...)
    policy.require_ae && (required = (required..., :AE, :AE_observed))
    missing_columns = filter(name -> !(name in propertynames(df)), unique(required))
    isempty(missing_columns) || throw(ArgumentError(
        "discovery data are missing cleaned values or pre-cleaning observation flags: " *
        join(string.(missing_columns), ", ")
    ))
    for flag in unique((_DISCOVERY_DRIVER_FLAGS..., _DISCOVERY_TARGET_FLAGS...))
        all(value -> value isa Bool, df[!, flag]) ||
            throw(ArgumentError("$flag must contain only Boolean pre-cleaning flags"))
    end
    if policy.require_ae
        all(value -> value isa Bool, df.AE_observed) ||
            throw(ArgumentError("AE_observed must contain only Boolean pre-cleaning flags"))
    end
    return nothing
end

function _discovery_regression_mask(df::DataFrame, rows,
                                    policy::DiscoveryObservationPolicy)
    row_idx = collect(Int, rows)
    mask = original_sindy_mask(df, row_idx;
                               smooth_window=policy.smooth_window,
                               require_ae=policy.require_ae)
    radius = div(policy.smooth_window, 2) + 1
    for i in eachindex(mask)
        mask[i] || continue
        support = max(1, i - radius):min(length(mask), i + radius)
        current_ok = all(column -> _finite_real(df[row_idx[i], column]),
                         _DISCOVERY_DRIVER_COLUMNS)
        support_columns = policy.require_ae ? (:Dst, :V, :n, :Dst_star, :AE) :
                                             (:Dst, :V, :n, :Dst_star)
        target_support_ok = all(
            j -> all(column -> _finite_real(df[row_idx[j], column]), support_columns),
            support,
        )
        mask[i] = current_ok && target_support_ok
    end
    return mask
end

function _short_gap_cleaned_value_is_admissible(df::DataFrame, row::Int,
                                                column::Symbol, flag::Symbol)
    Bool(df[row, flag]) && return true
    lo = row
    while lo > 1 && !Bool(df[lo - 1, flag]) && row - lo < _DISCOVERY_MAX_CLEANED_GAP
        lo -= 1
    end
    hi = row
    while hi < nrow(df) && !Bool(df[hi + 1, flag]) && hi - row < _DISCOVERY_MAX_CLEANED_GAP
        hi += 1
    end
    return hi - lo + 1 <= _DISCOVERY_MAX_CLEANED_GAP && lo > 1 && hi < nrow(df) &&
           Bool(df[lo - 1, flag]) && Bool(df[hi + 1, flag]) &&
           _finite_real(df[lo - 1, column]) && _finite_real(df[hi + 1, column])
end

function _discovery_driver_mask(df::DataFrame, row_idx::Vector{Int})
    return Bool[
        all(column -> _finite_real(df[row, column]), _DISCOVERY_DRIVER_COLUMNS) &&
        all(pair -> _short_gap_cleaned_value_is_admissible(df, row, pair...),
            _DISCOVERY_DRIVER_PAIRS) &&
        df[row, :V] >= 0 && df[row, :n] >= 0 && df[row, :Pdyn] >= 0
        for row in row_idx
    ]
end

function _discovery_original_target_mask(df::DataFrame, row_idx::Vector{Int})
    return Bool[
        all(flag -> Bool(df[row, flag]), _DISCOVERY_TARGET_FLAGS) &&
        _finite_real(df[row, :Dst_star])
        for row in row_idx
    ]
end

function _best_forward_scoring_segment(driver_mask::AbstractVector{Bool},
                                       target_mask::AbstractVector{Bool})
    length(driver_mask) == length(target_mask) ||
        throw(DimensionMismatch("driver and target masks must have equal length"))
    candidates = NamedTuple[]
    i = firstindex(driver_mask)
    while i <= lastindex(driver_mask)
        if !driver_mask[i]
            i += 1
            continue
        end
        run_start = i
        while i <= lastindex(driver_mask) && driver_mask[i]
            i += 1
        end
        run_end = i - 1
        targets = findall(@view target_mask[run_start:run_end]) .+ run_start .- 1
        length(targets) >= 2 || continue
        anchor = first(targets)
        endpoint = last(targets)
        scoring_rows = length(targets) - 1
        opportunities = endpoint - anchor
        push!(candidates, (
            anchor = anchor,
            endpoint = endpoint,
            trajectory_rows = opportunities + 1,
            scoring_rows = scoring_rows,
            scoring_fraction = scoring_rows / opportunities,
        ))
    end
    isempty(candidates) && return (
        anchor = 0, endpoint = 0, trajectory_rows = 0,
        scoring_rows = 0, scoring_fraction = 0.0,
    )
    sort!(candidates; by=candidate -> (-candidate.scoring_rows,
                                       -candidate.scoring_fraction,
                                       candidate.anchor,
                                       candidate.endpoint))
    return first(candidates)
end

function _discovery_window_policy(df::DataFrame, entry::StormCatalogEntry,
                                  policy::DiscoveryObservationPolicy)
    1 <= entry.onset_idx <= entry.end_idx <= nrow(df) || throw(ArgumentError(
        "storm $(entry.storm_id) has an invalid catalog window"
    ))
    rows = entry.onset_idx:entry.end_idx
    row_idx = collect(rows)
    regression_mask = _discovery_regression_mask(df, rows, policy)
    driver_mask = _discovery_driver_mask(df, row_idx)
    target_mask = _discovery_original_target_mask(df, row_idx)
    original_driver_mask = Bool[
        all(flag -> Bool(df[row, flag]), _DISCOVERY_DRIVER_FLAGS)
        for row in row_idx
    ]
    filled_driver_mask = driver_mask .& .!original_driver_mask
    segment = _best_forward_scoring_segment(driver_mask, target_mask)
    segment_range = segment.anchor == 0 ? (1:0) :
        (segment.anchor:segment.endpoint)
    segment_filled = count(@view filled_driver_mask[segment_range])

    regression_rows = count(regression_mask)
    exclusion_reason = if regression_rows < policy.min_regression_rows
        "insufficient_original_support_regression_rows"
    elseif segment.scoring_rows < policy.min_scoring_rows
        "insufficient_original_dst_star_scoring_rows"
    elseif segment.scoring_fraction < policy.min_scoring_fraction
        "insufficient_original_dst_star_scoring_fraction"
    else
        ""
    end
    eligible = isempty(exclusion_reason)
    n_catalog = length(row_idx)
    n_admissible = count(driver_mask)
    n_filled = count(filled_driver_mask)
    scoring_opportunities = max(segment.trajectory_rows - 1, 0)
    record = (
        storm_id = entry.storm_id,
        onset_time = string(entry.onset_time),
        solar_cycle = entry.solar_cycle,
        catalog_split = entry.split,
        onset_idx = entry.onset_idx,
        end_idx = entry.end_idx,
        catalog_rows = n_catalog,
        window_rows = n_catalog,
        admissible_cleaned_driver_rows = n_admissible,
        admissible_cleaned_driver_fraction = n_admissible / n_catalog,
        admissible_filled_driver_rows = n_filled,
        admissible_filled_driver_fraction = n_admissible == 0 ? 0.0 :
            n_filled / n_admissible,
        original_dst_star_rows = count(target_mask),
        regression_rows = regression_rows,
        regression_fraction = regression_rows / n_catalog,
        scoring_trajectory_rows = segment.trajectory_rows,
        scoring_rows = segment.scoring_rows,
        scoring_fraction = segment.scoring_fraction,
        scoring_admissible_filled_driver_rows = segment_filled,
        scoring_admissible_filled_driver_fraction = segment.trajectory_rows == 0 ?
            0.0 : segment_filled / segment.trajectory_rows,
        scoring_start_idx = segment.anchor == 0 ? 0 :
            entry.onset_idx + segment.anchor - 1,
        scoring_end_idx = segment.endpoint == 0 ? 0 :
            entry.onset_idx + segment.endpoint - 1,
        min_regression_rows = policy.min_regression_rows,
        min_scoring_rows = policy.min_scoring_rows,
        min_scoring_fraction = policy.min_scoring_fraction,
        max_cleaned_gap_rows = _DISCOVERY_MAX_CLEANED_GAP,
        require_ae = policy.require_ae,
        original_rows = regression_rows,
        excluded_rows = n_catalog - regression_rows,
        eligible = eligible,
        exclusion_reason = exclusion_reason,
    )
    return (
        record = record,
        regression_mask = regression_mask,
        driver_mask = driver_mask,
        target_mask = target_mask,
        filled_driver_mask = filled_driver_mask,
        scoring_segment = segment,
        scoring_opportunities = scoring_opportunities,
    )
end

function _summarize_discovery_observations(records::AbstractVector)
    summaries = NamedTuple[]
    isempty(records) && return summaries
    for cycle in sort!(unique(Int(record.solar_cycle) for record in records))
        subset = filter(record -> record.solar_cycle == cycle, records)
        catalog_rows = sum(record.catalog_rows for record in subset)
        admissible_rows = sum(record.admissible_cleaned_driver_rows for record in subset)
        filled_rows = sum(record.admissible_filled_driver_rows for record in subset)
        regression_rows = sum(record.regression_rows for record in subset)
        scoring_rows = sum(record.scoring_rows for record in subset)
        scoring_opportunities = sum(max(record.scoring_trajectory_rows - 1, 0)
                                    for record in subset)
        reason_counts = Dict{String,Int}()
        for record in subset
            isempty(record.exclusion_reason) && continue
            reason_counts[record.exclusion_reason] =
                get(reason_counts, record.exclusion_reason, 0) + 1
        end
        reasons = join(("$reason=$(reason_counts[reason])"
                        for reason in sort!(collect(keys(reason_counts)))), ";")
        push!(summaries, (
            solar_cycle = cycle,
            catalog_storms = length(subset),
            eligible_storms = count(record -> record.eligible, subset),
            excluded_storms = count(record -> !record.eligible, subset),
            catalog_rows = catalog_rows,
            admissible_cleaned_driver_rows = admissible_rows,
            admissible_cleaned_driver_fraction = admissible_rows / catalog_rows,
            admissible_filled_driver_rows = filled_rows,
            admissible_filled_driver_fraction = admissible_rows == 0 ? 0.0 :
                filled_rows / admissible_rows,
            regression_rows = regression_rows,
            regression_fraction = regression_rows / catalog_rows,
            scoring_rows = scoring_rows,
            scoring_fraction = scoring_opportunities == 0 ? 0.0 :
                scoring_rows / scoring_opportunities,
            exclusion_reason = reasons,
        ))
    end
    return summaries
end

"""
Predeclare storm eligibility from observation availability, before any split or
model fit. Regression rows require original target-stencil support and original
current drivers. Forward scoring uses finite short-gap-cleaned driver segments
but only original Dst/V/n-derived Dst* target timestamps.
"""
function _audit_discovery_observations(df::DataFrame,
                                       catalog::Vector{StormCatalogEntry};
                                       policy::DiscoveryObservationPolicy=
                                           DiscoveryObservationPolicy(),
                                       smooth_window::Union{Nothing,Int}=nothing)
    policy = _resolve_discovery_policy(policy, smooth_window)
    _require_discovery_observation_state(df, policy)
    storm_ids = getproperty.(catalog, :storm_id)
    length(unique(storm_ids)) == length(storm_ids) ||
        throw(ArgumentError("catalog contains duplicate storm ids"))
    eligible = StormCatalogEntry[]
    records = NamedTuple[]
    for entry in sort(catalog; by=e -> (e.onset_time, e.storm_id))
        state = _discovery_window_policy(df, entry, policy)
        state.record.eligible && push!(eligible, entry)
        push!(records, state.record)
    end
    return (
        eligible_entries = eligible,
        storm_records = records,
        cycle_records = _summarize_discovery_observations(records),
        policy = policy,
    )
end

function _predeclare_discovery_storms(df::DataFrame,
                                      catalog::Vector{StormCatalogEntry};
                                      policy::DiscoveryObservationPolicy=
                                          DiscoveryObservationPolicy(),
                                      smooth_window::Union{Nothing,Int}=nothing)
    audit = _audit_discovery_observations(df, catalog;
                                          policy=policy,
                                          smooth_window=smooth_window)
    return audit.eligible_entries, audit.storm_records
end

function _prepare_discovery_storm(df::DataFrame, entry::StormCatalogEntry,
                                  lib::CandidateLibrary;
                                  policy::DiscoveryObservationPolicy=
                                      DiscoveryObservationPolicy(),
                                  smooth_window::Union{Nothing,Int}=nothing)
    policy = _resolve_discovery_policy(policy, smooth_window)
    _require_discovery_observation_state(df, policy)
    state = _discovery_window_policy(df, entry, policy)
    state.record.eligible || throw(ArgumentError(
        "storm $(entry.storm_id) is ineligible: $(state.record.exclusion_reason)"
    ))
    window_swd = extract_storm_data(df, entry)
    full_data, full_target = prepare_sindy_data(
        window_swd, 1.0; smooth_window=policy.smooth_window,
    )
    regression_mask = copy(state.regression_mask)
    regression_mask .&= isfinite.(full_target)
    for column in values(full_data)
        regression_mask .&= isfinite.(column)
    end
    count(regression_mask) == state.record.regression_rows || throw(ArgumentError(
        "storm $(entry.storm_id) regression-row audit disagrees with prepared data"
    ))
    data = Dict(key => column[regression_mask] for (key, column) in full_data)
    target = full_target[regression_mask]
    theta = evaluate_library(lib, data)
    all(isfinite, theta) ||
        throw(ArgumentError("storm $(entry.storm_id) has a non-finite design matrix"))
    segment = state.scoring_segment
    swd = _slice_solar_wind(window_swd, segment.anchor, segment.endpoint)
    score_mask = @view state.target_mask[segment.anchor:segment.endpoint]
    scoring_observations = fill(NaN, length(score_mask))
    for i in eachindex(score_mask)
        score_mask[i] && (scoring_observations[i] = swd.Dst_star[i])
    end
    return (
        storm_id = entry.storm_id,
        onset_time = entry.onset_time,
        entry = entry,
        swd = swd,
        window_swd = window_swd,
        data = data,
        theta = theta,
        target = target,
        regression_mask = regression_mask,
        scoring_observations = scoring_observations,
        observation_record = state.record,
        library_terms = Tuple(get_term_names(lib)),
    )
end

function _prepare_discovery_storms(df::DataFrame,
                                   entries::Vector{StormCatalogEntry},
                                   lib::CandidateLibrary;
                                   policy::DiscoveryObservationPolicy=
                                       DiscoveryObservationPolicy(),
                                   smooth_window::Union{Nothing,Int}=nothing)
    policy = _resolve_discovery_policy(policy, smooth_window)
    return [_prepare_discovery_storm(df, entry, lib; policy=policy)
            for entry in entries]
end

function _cached_subset_design!(cache::AbstractDict, storms::AbstractVector)
    isempty(storms) && throw(ArgumentError("cannot build an empty storm design"))
    ordered = sort(collect(storms); by=s -> (s.onset_time, s.storm_id))
    ids = string.(getproperty.(ordered, :storm_id))
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("subset contains duplicate storm ids"))
    terms = ordered[1].library_terms
    all(storm -> storm.library_terms == terms, ordered) ||
        throw(ArgumentError("subset storms use different candidate libraries"))
    key = (terms, Tuple(ids))
    return get!(cache, key) do
        theta = reduce(vcat, (storm.theta for storm in ordered))
        target = reduce(vcat, (storm.target for storm in ordered))
        size(theta, 1) == length(target) ||
            throw(DimensionMismatch("cached subset design and target lengths differ"))
        (theta=theta, target=target, storm_ids=Tuple(ids))
    end
end

function _concat_discovery_data(storms::AbstractVector)
    isempty(storms) && throw(ArgumentError("cannot concatenate an empty storm subset"))
    ordered = sort(collect(storms); by=s -> (s.onset_time, s.storm_id))
    keys0 = Set(keys(ordered[1].data))
    all(storm -> Set(keys(storm.data)) == keys0, ordered) ||
        throw(ArgumentError("subset storms expose different library data keys"))
    return Dict(key => reduce(vcat, (storm.data[key] for storm in ordered))
                for key in keys0)
end

function _slice_solar_wind(swd::SolarWindData, first_index::Int,
                           last_index::Int=length(swd.t))
    1 <= first_index <= length(swd.t) || throw(BoundsError(swd.t, first_index))
    first_index <= last_index <= length(swd.t) ||
        throw(BoundsError(swd.t, last_index))
    rows = first_index:last_index
    return SolarWindData(
        swd.t[rows] .- swd.t[first_index], swd.V[rows], swd.Bz[rows],
        swd.By[rows], swd.n[rows], swd.Pdyn[rows], swd.Dst[rows],
        swd.Dst_star[rows],
    )
end

_discovery_scoring_observations(storm) =
    hasproperty(storm, :scoring_observations) ? storm.scoring_observations :
    storm.swd.Dst_star

function _discovery_cohort_identity(storm, anchor::Int,
                                    scored_indices::AbstractVector{Int})
    1 <= anchor <= length(storm.swd.t) || throw(BoundsError(storm.swd.t, anchor))
    all(index -> 2 <= index <= length(storm.swd.t) - anchor + 1,
        scored_indices) || throw(BoundsError(storm.swd.t, scored_indices))
    start_index = Int(storm.observation_record.scoring_start_idx)
    end_index = Int(storm.observation_record.scoring_end_idx)
    expected_end = start_index + length(storm.swd.t) - 1
    end_index == expected_end || error(
        "storm $(storm.storm_id) scoring bounds do not match its driver rows",
    )
    absolute_anchor = start_index + anchor - 1
    absolute_rows = absolute_anchor .+ scored_indices .- 1
    signature_source = join((
        string(storm.storm_id), string(absolute_anchor), string(absolute_anchor),
        string(end_index), join(absolute_rows, ";"),
    ), "|")
    return (
        anchor_catalog_index = absolute_anchor,
        driver_start_catalog_index = absolute_anchor,
        driver_end_catalog_index = end_index,
        scored_catalog_indices = join(absolute_rows, ";"),
        cohort_signature_sha256 = bytes2hex(SHA.sha256(signature_source)),
    )
end

"""Run the fixed storm-level selector while caching each distinct subset design."""
function _select_discovery_lambda(storms::AbstractVector,
                                  lib::CandidateLibrary,
                                  design_cache::AbstractDict)
    terms = get_term_names(lib)
    fit = function (fit_storms, lambda)
        design = _cached_subset_design!(design_cache, fit_storms)
        coefficients = stlsq(design.theta, design.target; λ=lambda, normalize=true)
        support = terms[coefficients .!= 0.0]
        return (model=coefficients, support=support)
    end
    integrate = function (coefficients, storm, anchor_index, anchor_value)
        sliced = _slice_solar_wind(storm.swd, anchor_index)
        return simulate_sindy(coefficients, lib, sliced, 1.0; Dst0=anchor_value)
    end
    return select_storm_lambda(storms;
        fit=fit,
        integrate=integrate,
        observations=_discovery_scoring_observations,
    )
end

"""Score SINDy and all analytical baselines after one shared observed anchor."""
function _score_discovery_storm(storm, coefficients, lib::CandidateLibrary)
    observations = _discovery_scoring_observations(storm)
    observations isa AbstractVector ||
        throw(ArgumentError("scoring observations must be a vector"))
    all(value -> value isa Real && (isfinite(value) || isnan(value)), observations) ||
        throw(ArgumentError("scoring observations must contain finite values or NaN gaps"))
    length(observations) == length(storm.swd.t) ||
        throw(DimensionMismatch("scoring observations and solar-wind rows differ"))
    anchor = findfirst(isfinite, observations)
    anchor === nothing &&
        throw(ArgumentError("storm $(storm.storm_id) has no finite Dst* anchor"))
    swd = _slice_solar_wind(storm.swd, anchor)
    observed = observations[anchor:end]
    length(swd.t) >= 2 ||
        throw(ArgumentError("storm $(storm.storm_id) has no post-anchor sample"))
    all(all(isfinite, field) for field in (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn)) ||
        throw(ArgumentError("storm $(storm.storm_id) contains a non-finite scoring driver"))
    scored_indices = findall(isfinite, @view observed[2:end]) .+ 1
    isempty(scored_indices) && throw(ArgumentError(
        "storm $(storm.storm_id) has no original post-anchor Dst* target"
    ))

    cohort = _discovery_cohort_identity(storm, anchor, scored_indices)
    bs = max.(-swd.Bz, 0.0)
    predictions = (
        SINDy = simulate_sindy(coefficients, lib, swd, 1.0; Dst0=observed[1]),
        Burton = simulate_burton(swd.V, bs, 1.0; Dst0=observed[1]),
        BurtonFull = simulate_burton_full(swd.V, bs, 1.0; Dst0=observed[1]),
        OBrienMcP = simulate_obrien(swd.V, bs, 1.0; Dst0=observed[1]),
    )
    rows = NamedTuple[]
    for model in propertynames(predictions)
        prediction = getproperty(predictions, model)
        first(prediction) == first(observed) ||
            throw(ArgumentError("$model was not initialized from the shared anchor"))
        metrics = metrics_summary(prediction[scored_indices], observed[scored_indices];
                                  name=String(model))
        push!(rows, (
            storm_id = storm.storm_id,
            model = String(model),
            rmse_nt = metrics.rmse,
            mae_nt = metrics.mae,
            correlation = metrics.corr,
            pe = metrics.pe,
            min_dst_star_observed_nt = minimum(observed[isfinite.(observed)]),
            anchor_index = anchor,
            n_points = length(scored_indices),
            cohort...,
        ))
    end
    return (anchor_index=anchor, swd=swd, observations=observed,
            scored_indices=scored_indices, predictions=predictions, metrics=rows)
end

function _write_discovery_csv(path::AbstractString, data)
    mkpath(dirname(path))
    SolarSINDy._require_regular_output_target(path)
    temporary, io = mktemp(dirname(path); cleanup=false)
    close(io)
    try
        CSV.write(temporary, data isa DataFrame ? data : DataFrame(data))
        SolarSINDy._atomic_replace_regular(temporary, path)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end
