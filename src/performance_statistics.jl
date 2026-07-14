# Paired whole-storm performance statistics

_performance_rows(rows::AbstractDataFrame) = eachrow(rows)
_performance_rows(rows) = rows

function _rmse_by_storm(rows, storm_id, rmse_value, label; positive::Bool)
    values = Dict{String,Float64}()
    for row in _performance_rows(rows)
        raw_id = storm_id(row)
        (raw_id === missing || raw_id === nothing ||
         (raw_id isa Real && !isfinite(raw_id))) &&
            throw(ArgumentError("$label contains an invalid storm id"))
        id = strip(string(raw_id))
        isempty(id) && throw(ArgumentError("$label contains an empty storm id"))
        haskey(values, id) && throw(ArgumentError("$label contains duplicate storm id $id"))

        raw_rmse = rmse_value(row)
        raw_rmse isa Real && !(raw_rmse isa Bool) && isfinite(raw_rmse) ||
            throw(ArgumentError("$label RMSE for storm $id must be finite"))
        value = Float64(raw_rmse)
        isfinite(value) || throw(ArgumentError(
            "$label RMSE for storm $id is outside the supported Float64 range"
        ))
        if positive
            value > 0 || throw(ArgumentError(
                "$label RMSE for storm $id must be positive for a relative effect"
            ))
        else
            value >= 0 || throw(ArgumentError("$label RMSE for storm $id must be nonnegative"))
        end
        values[id] = value
    end
    isempty(values) && throw(ArgumentError("$label contains no storms"))
    return values
end

function _bootstrap_effects(rmse_differences, relative_differences, draws, seed)
    rng = MersenneTwister(seed)
    n = length(rmse_differences)
    rmse_means = Vector{Float64}(undef, draws)
    relative_means = similar(rmse_means)
    rmse_scale = maximum(abs, rmse_differences)
    relative_scale = maximum(abs, relative_differences)
    # Preserve the former direct-sum path exactly whenever its worst-case sum is
    # representable. Extreme-but-finite effects use scaled accumulation instead.
    direct_rmse = rmse_scale == 0.0 || rmse_scale <= floatmax(Float64) / n
    direct_relative = relative_scale == 0.0 || relative_scale <= floatmax(Float64) / n
    for draw in 1:draws
        rmse_sum = 0.0
        relative_sum = 0.0
        for _ in 1:n
            index = rand(rng, 1:n)
            rmse_sum += direct_rmse ? rmse_differences[index] :
                        rmse_differences[index] / rmse_scale
            relative_sum += direct_relative ? relative_differences[index] :
                            relative_differences[index] / relative_scale
        end
        isfinite(rmse_sum) && isfinite(relative_sum) ||
            throw(ArgumentError("bootstrap effect sum overflowed"))
        rmse_means[draw] = direct_rmse ? rmse_sum / n :
                           rmse_scale * clamp(rmse_sum / n, -1.0, 1.0)
        relative_means[draw] = direct_relative ? relative_sum / n :
                               relative_scale * clamp(relative_sum / n, -1.0, 1.0)
        isfinite(rmse_means[draw]) && isfinite(relative_means[draw]) ||
            throw(ArgumentError("bootstrap effect mean overflowed"))
    end
    return rmse_means, relative_means
end

function _performance_mean(values::Vector{Float64})
    scale = maximum(abs, values)
    if scale == 0.0 || scale <= floatmax(Float64) / length(values)
        return mean(values)
    end
    normalized = sum(value -> value / scale, values) / length(values)
    return scale * clamp(normalized, -1.0, 1.0)
end

"""
    paired_storm_statistics(model_rows, reference_rows;
                            storm_id, rmse_value, draws=10_000,
                            coverage=0.95, seed=42)

Compute paired whole-storm model-versus-reference statistics. Rows are paired by
unique storm identifier, never by row position. The signed absolute-scale effect
is `model RMSE - reference RMSE`; the per-storm relative effect is that difference
divided by reference RMSE. Negative effects therefore favor the model.

The returned estimates are unweighted storm means. Their intervals are percentile
intervals from `draws` paired, whole-storm bootstrap samples of size `n` with
replacement. A two-sided paired Wilcoxon result from
[`wilcoxon_signed_rank_p`](@ref) is included as a secondary statistic.
"""
function paired_storm_statistics(model_rows, reference_rows;
                                 storm_id = row -> getproperty(row, :storm_id),
                                 rmse_value = row -> getproperty(row, :rmse),
                                 draws=10_000,
                                 coverage=0.95,
                                 seed=42)
    draws isa Integer && !(draws isa Bool) && 1 <= draws <= typemax(Int) ||
        throw(ArgumentError("draws must be a positive integer"))
    coverage isa Real && !(coverage isa Bool) && isfinite(coverage) && 0 < coverage < 1 ||
        throw(ArgumentError("coverage must be finite and lie in (0, 1)"))
    seed isa Integer && !(seed isa Bool) && 0 <= seed <= typemax(UInt32) ||
        throw(ArgumentError("seed must be an integer in [0, $(typemax(UInt32))]"))
    n_draws = Int(draws)
    interval_coverage = Float64(coverage)
    isfinite(interval_coverage) && 0 < interval_coverage < 1 ||
        throw(ArgumentError("coverage must be representable in Float64 and lie in (0, 1)"))
    rng_seed = Int(seed)

    model = _rmse_by_storm(model_rows, storm_id, rmse_value, "model"; positive=false)
    reference = _rmse_by_storm(reference_rows, storm_id, rmse_value, "reference";
                               positive=true)
    model_ids = Set(keys(model))
    reference_ids = Set(keys(reference))
    if model_ids != reference_ids
        model_only = sort!(collect(setdiff(model_ids, reference_ids)))
        reference_only = sort!(collect(setdiff(reference_ids, model_ids)))
        throw(ArgumentError(
            "model and reference storm-id sets differ: model-only=$model_only, " *
            "reference-only=$reference_only"
        ))
    end

    ids = sort!(collect(model_ids))
    rmse_differences = [model[id] - reference[id] for id in ids]
    relative_differences = [rmse_differences[i] / reference[id] for (i, id) in enumerate(ids)]
    all(isfinite, rmse_differences) && all(isfinite, relative_differences) ||
        throw(ArgumentError("paired RMSE effects must be finite"))
    rmse_estimate = _performance_mean(rmse_differences)
    relative_estimate = _performance_mean(relative_differences)
    isfinite(rmse_estimate) && isfinite(relative_estimate) ||
        throw(ArgumentError("mean paired RMSE effects must be finite"))
    rmse_bootstrap, relative_bootstrap =
        _bootstrap_effects(rmse_differences, relative_differences, n_draws, rng_seed)
    tail = (1 - interval_coverage) / 2
    rmse_interval = quantile(rmse_bootstrap, [tail, 1 - tail])
    relative_interval = quantile(relative_bootstrap, [tail, 1 - tail])
    wilcoxon = wilcoxon_signed_rank_p(rmse_differences)

    pair_records = [(
        storm_id = id,
        model_rmse_nt = model[id],
        reference_rmse_nt = reference[id],
        rmse_difference_nt = rmse_differences[i],
        relative_difference_fraction = relative_differences[i],
    ) for (i, id) in enumerate(ids)]
    bootstrap_records = [(
        draw = draw,
        mean_rmse_difference_nt = rmse_bootstrap[draw],
        mean_relative_difference_fraction = relative_bootstrap[draw],
    ) for draw in 1:n_draws]
    summary_records = [
        (
            effect = "rmse_difference_model_minus_reference",
            unit = "nT",
            estimate = rmse_estimate,
            interval_lower = rmse_interval[1],
            interval_upper = rmse_interval[2],
            coverage = interval_coverage,
            n_storms = length(ids),
            bootstrap_draws = n_draws,
            seed = rng_seed,
        ),
        (
            effect = "relative_difference_model_minus_reference",
            unit = "fraction",
            estimate = relative_estimate,
            interval_lower = relative_interval[1],
            interval_upper = relative_interval[2],
            coverage = interval_coverage,
            n_storms = length(ids),
            bootstrap_draws = n_draws,
            seed = rng_seed,
        ),
    ]
    wilcoxon_record = (
        effect = "rmse_difference_model_minus_reference",
        p_value = wilcoxon.p,
        z = wilcoxon.z,
        w = wilcoxon.w,
        n_nonzero = wilcoxon.n,
    )
    return (
        mean_rmse_difference = summary_records[1].estimate,
        rmse_difference_interval = (rmse_interval[1], rmse_interval[2]),
        mean_relative_difference = summary_records[2].estimate,
        relative_difference_interval = (relative_interval[1], relative_interval[2]),
        wilcoxon = wilcoxon,
        pair_records = pair_records,
        bootstrap_records = bootstrap_records,
        summary_records = summary_records,
        wilcoxon_record = wilcoxon_record,
        coverage = interval_coverage,
        draws = n_draws,
        seed = rng_seed,
    )
end

"""
    holm_adjust(p_values; labels=nothing)

Apply Holm's step-down family-wise error adjustment. The supplied vector is the
entire predeclared family. Tied p-values are ordered by their original position,
and returned records preserve the caller's original order.
"""
function holm_adjust(p_values::AbstractVector; labels=nothing)
    isempty(p_values) && throw(ArgumentError("Holm adjustment requires at least one p-value"))
    all(value -> value isa Real && !(value isa Bool) && isfinite(value) &&
                 0 <= value <= 1, p_values) ||
        throw(ArgumentError("Holm p-values must be finite and lie in [0, 1]"))
    labels === nothing || labels isa AbstractVector ||
        throw(ArgumentError("Holm labels must be a vector"))
    names = labels === nothing ? string.(eachindex(p_values)) : string.(labels)
    length(names) == length(p_values) ||
        throw(DimensionMismatch("Holm labels and p-values must have equal length"))
    length(unique(names)) == length(names) ||
        throw(ArgumentError("Holm labels must be unique"))
    all(name -> !isempty(strip(name)), names) ||
        throw(ArgumentError("Holm labels must not be empty"))

    raw = Float64.(p_values)
    order = sortperm(eachindex(raw); by=index -> (raw[index], index))
    adjusted = similar(raw)
    ranks = Vector{Int}(undef, length(raw))
    running = 0.0
    family_size = length(raw)
    for (rank, index) in enumerate(order)
        running = max(running, (family_size - rank + 1) * raw[index])
        adjusted[index] = min(running, 1.0)
        ranks[index] = rank
    end
    return [(
        label = names[index],
        p_value = raw[index],
        holm_p_value = adjusted[index],
        holm_rank = ranks[index],
        family_size = family_size,
    ) for index in eachindex(raw)]
end

"""
    write_paired_storm_statistics(result, output_root; prefix="paired_storm")

Atomically persist paired rows, bootstrap draws, interval summaries, and the
secondary Wilcoxon result beneath an explicit output root.
"""
function write_paired_storm_statistics(result, output_root::AbstractString;
                                       prefix::AbstractString="paired_storm")
    isempty(output_root) && throw(ArgumentError("output_root must not be empty"))
    isempty(prefix) && throw(ArgumentError("prefix must not be empty"))
    basename(prefix) == prefix ||
        throw(ArgumentError("prefix must be a file-name prefix, not a path"))
    mkpath(output_root)
    paths = (
        pairs = joinpath(output_root, "$(prefix)_pairs.csv"),
        bootstrap = joinpath(output_root, "$(prefix)_bootstrap.csv"),
        summary = joinpath(output_root, "$(prefix)_summary.csv"),
        wilcoxon = joinpath(output_root, "$(prefix)_wilcoxon.csv"),
    )
    frames = (
        pairs = DataFrame(result.pair_records),
        bootstrap = DataFrame(result.bootstrap_records),
        summary = DataFrame(result.summary_records),
        wilcoxon = DataFrame([result.wilcoxon_record]),
    )
    return _write_selection_csv_set(paths, frames)
end

"""
    write_holm_adjustment(records, output_root; prefix="holm")

Atomically persist records returned by [`holm_adjust`](@ref).
"""
function write_holm_adjustment(records::AbstractVector, output_root::AbstractString;
                               prefix::AbstractString="holm")
    isempty(records) && throw(ArgumentError("Holm records must not be empty"))
    isempty(output_root) && throw(ArgumentError("output_root must not be empty"))
    isempty(prefix) && throw(ArgumentError("prefix must not be empty"))
    basename(prefix) == prefix ||
        throw(ArgumentError("prefix must be a file-name prefix, not a path"))
    mkpath(output_root)
    path = joinpath(output_root, "$(prefix)_adjusted.csv")
    _write_selection_csv(path, records)
    return path
end
