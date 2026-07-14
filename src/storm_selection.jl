# Leakage-free storm-level regularisation selection

"""
    storm_lambda_grid()

Return the fixed 60-point regularisation grid used for storm-level SINDy model
selection: `10 .^ range(-2, 4, length=60)`.
"""
storm_lambda_grid() = collect(10.0 .^ range(-2, 4, length=60))

function _scaled_mean_standard_error(values::Vector{Float64})
    isempty(values) && throw(ArgumentError("storm-error vector must not be empty"))
    all(isfinite, values) || throw(ArgumentError("storm errors must be finite"))
    scale = maximum(abs, values)
    iszero(scale) && return (mean=0.0, standard_error=0.0)
    normalized = values ./ scale
    scaled_mean = mean(normalized)
    scaled_standard_error = length(values) == 1 ? 0.0 :
        std(normalized) / sqrt(length(values))
    result = (
        mean=scale * scaled_mean,
        standard_error=scale * scaled_standard_error,
    )
    isfinite(result.mean) && isfinite(result.standard_error) ||
        throw(ArgumentError("storm-error summary exceeds the supported range"))
    return result
end

function _selection_fit(fit, storms, lambda)
    result = fit(copy(storms), lambda)
    hasproperty(result, :model) && hasproperty(result, :support) ||
        throw(ArgumentError("fit must return a value with model and support fields"))
    result.support isa AbstractVector ||
        throw(ArgumentError("fit support must be a vector of active-term names"))
    support = sort!(String.(result.support))
    length(unique(support)) == length(support) ||
        throw(ArgumentError("fit returned duplicate active-support terms"))
    return result.model, support
end

function _storm_validation_error(model, storm, integrate, observations,
                                 storm_id, onset_time)
    obs = observations(storm)
    obs isa AbstractVector || throw(ArgumentError("observations must return a vector"))
    all(value -> value isa Real && !(value isa Bool), obs) ||
        throw(ArgumentError("storm $(storm_id(storm)) observations must be real numbers"))
    all(value -> isnan(value) ||
                 (isfinite(value) && isfinite(Float64(value))), obs) ||
        throw(ArgumentError(
            "storm $(storm_id(storm)) observations must be NaN or finite Float64 values",
        ))
    anchor = findfirst(isfinite, obs)
    anchor === nothing && throw(ArgumentError("storm $(storm_id(storm)) has no finite Dst* sample"))
    anchor_value = Float64(obs[anchor])
    pred = integrate(model, storm, anchor, anchor_value)
    pred isa AbstractVector || throw(ArgumentError("integrate must return a vector"))
    expected = length(obs) - anchor + 1
    length(pred) == expected || throw(DimensionMismatch(
        "integrated trajectory for storm $(storm_id(storm)) has length $(length(pred)); expected $expected"
    ))
    all(value -> value isa Real && !(value isa Bool) && isfinite(value), pred) ||
        throw(ArgumentError("integrated trajectory for storm $(storm_id(storm)) is non-finite"))
    Float64(first(pred)) == anchor_value || throw(ArgumentError(
        "integrated trajectory for storm $(storm_id(storm)) is not anchored at its first finite Dst* sample"
    ))

    error_scale = 0.0
    scaled_sum_squares = 0.0
    n_scored = 0
    # The anchor is supplied to the model, not forecast. Score only subsequent
    # points so every error is genuinely out-of-anchor forward prediction.
    for (prediction, observation) in zip(@view(pred[2:end]),
                                         @view(obs[(anchor + 1):end]))
        if isfinite(observation)
            difference = Float64(prediction) - Float64(observation)
            isfinite(difference) || throw(ArgumentError(
                "storm $(storm_id(storm)) scoring difference exceeds Float64 range",
            ))
            magnitude = abs(difference)
            if magnitude != 0.0
                if error_scale < magnitude
                    ratio = error_scale / magnitude
                    scaled_sum_squares = 1.0 + scaled_sum_squares * ratio^2
                    error_scale = magnitude
                else
                    scaled_sum_squares += (magnitude / error_scale)^2
                end
            end
            n_scored += 1
        end
    end
    n_scored >= 1 ||
        throw(ArgumentError("storm $(storm_id(storm)) has no finite post-anchor Dst* sample"))
    return (
        storm_id = string(storm_id(storm)),
        onset_time = string(onset_time(storm)),
        rmse = error_scale == 0.0 ? 0.0 :
            error_scale * sqrt(scaled_sum_squares / n_scored),
        anchor_index = anchor,
        n_scored = n_scored,
    )
end

"""
    select_storm_lambda(training_storms; fit, integrate, storm_id,
                        onset_time, observations)

Select the SINDy regularisation parameter without outer-evaluation contact.
Only `training_storms` are accepted. Whole storms are sorted by onset; the
earliest 80% (rounded down) form the inner-training set and the remainder form
the inner-validation set. Each value from [`storm_lambda_grid`](@ref) is fitted
on the inner-training storms and scored by the unweighted mean of per-storm
forward-integrated Dst* RMSEs.

`fit(storms, lambda)` must return a value with `model` and `support` fields.
`integrate(model, storm, anchor_index, anchor_value)` must return the trajectory
from `anchor_index` through the end of the storm, with its first value exactly
equal to `anchor_value`. The default accessors read `storm_id`, `onset_time`, and
`Dst_star` properties.

The one-standard-error cutoff is the minimum mean storm RMSE plus the
storm-level standard error at that minimum. The largest eligible lambda is
selected; exact ties prefer fewer active terms and then larger lambda. The
selected model is then refitted once on every supplied training storm.

The return value contains the final model and deterministic split, candidate,
per-storm error, support, and decision records suitable for
[`write_storm_lambda_selection`](@ref).
"""
function select_storm_lambda(training_storms::AbstractVector;
                             fit,
                             integrate,
                             storm_id = storm -> getproperty(storm, :storm_id),
                             onset_time = storm -> getproperty(storm, :onset_time),
                             observations = storm -> getproperty(storm, :Dst_star))
    length(training_storms) >= 2 ||
        throw(ArgumentError("storm-level selection requires at least two training storms"))
    ordered = sort(collect(training_storms);
                   by=storm -> (onset_time(storm), string(storm_id(storm))))
    ids = string.(storm_id.(ordered))
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("storm identifiers must be unique"))

    n_inner_train = fld(4 * length(ordered), 5)
    inner_train = ordered[1:n_inner_train]
    inner_validation = ordered[(n_inner_train + 1):end]
    split_records = [(
        storm_id = ids[i],
        onset_time = string(onset_time(ordered[i])),
        inner_split = i <= n_inner_train ? "train" : "validation",
        chronological_index = i,
    ) for i in eachindex(ordered)]

    lambdas = storm_lambda_grid()
    means = Vector{Float64}(undef, length(lambdas))
    standard_errors = similar(means)
    active_counts = Vector{Int}(undef, length(lambdas))
    error_records = NamedTuple[]
    support_records = NamedTuple[]

    for (candidate_index, lambda) in enumerate(lambdas)
        model, support = _selection_fit(fit, inner_train, lambda)
        active_counts[candidate_index] = length(support)
        for term in support
            push!(support_records, (
                stage = "inner_candidate",
                candidate_index = candidate_index,
                lambda = lambda,
                term = term,
            ))
        end

        storm_errors = Vector{Float64}(undef, length(inner_validation))
        for (i, storm) in enumerate(inner_validation)
            score = _storm_validation_error(model, storm, integrate, observations,
                                            storm_id, onset_time)
            storm_errors[i] = score.rmse
            push!(error_records, (
                candidate_index = candidate_index,
                lambda = lambda,
                score...,
            ))
        end
        summary = _scaled_mean_standard_error(storm_errors)
        means[candidate_index] = summary.mean
        standard_errors[candidate_index] = summary.standard_error
    end

    minimum_index = sortperm(eachindex(lambdas);
        by=i -> (means[i], active_counts[i], -lambdas[i]))[1]
    cutoff = means[minimum_index] + standard_errors[minimum_index]
    eligible = findall(means .<= cutoff)
    selected_index = sort(eligible;
        by=i -> (-lambdas[i], active_counts[i], -lambdas[i]))[1]
    selected_lambda = lambdas[selected_index]

    final_model, final_support = _selection_fit(fit, ordered, selected_lambda)
    for term in final_support
        push!(support_records, (
            stage = "full_refit",
            candidate_index = selected_index,
            lambda = selected_lambda,
            term = term,
        ))
    end
    candidate_records = [(
        candidate_index = i,
        lambda = lambdas[i],
        mean_storm_rmse = means[i],
        standard_error = standard_errors[i],
        n_active_terms = active_counts[i],
        eligible = i in eligible,
        selected = i == selected_index,
    ) for i in eachindex(lambdas)]
    decision_record = (
        n_training_storms = length(ordered),
        n_inner_training_storms = length(inner_train),
        n_inner_validation_storms = length(inner_validation),
        minimum_candidate_index = minimum_index,
        minimum_lambda = lambdas[minimum_index],
        minimum_mean_storm_rmse = means[minimum_index],
        minimum_standard_error = standard_errors[minimum_index],
        one_standard_error_cutoff = cutoff,
        selected_candidate_index = selected_index,
        selected_lambda = selected_lambda,
        selection_rule = "largest_lambda_within_one_standard_error_then_fewer_terms_then_larger_lambda",
    )
    return (
        model = final_model,
        support = final_support,
        selected_lambda = selected_lambda,
        split_records = split_records,
        candidate_records = candidate_records,
        error_records = error_records,
        support_records = support_records,
        decision_record = decision_record,
    )
end

function _require_regular_output_target(path::AbstractString)
    islink(path) && throw(ArgumentError(
        "output target must not be a symbolic link: $path",
    ))
    ispath(path) && !isfile(path) && throw(ArgumentError(
        "output target exists but is not a regular file: $path",
    ))
    return path
end

function _atomic_replace_regular(source::AbstractString, target::AbstractString)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "atomic replacement source must be a regular non-symlink file: $source",
    ))
    _require_regular_output_target(target)
    Base.Filesystem.rename(source, target)
    isfile(target) && !islink(target) || throw(ErrorException(
        "atomic replacement did not install a regular file: $target",
    ))
    return target
end

function _write_selection_csv(path, rows, empty_frame=nothing)
    frame = isempty(rows) && empty_frame !== nothing ? empty_frame : DataFrame(rows)
    _require_regular_output_target(path)
    tmp, io = mktemp(dirname(path); cleanup=false)
    close(io)
    try
        CSV.write(tmp, frame)
        _atomic_replace_regular(tmp, path)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return path
end

function _selection_csv_set_lock_path(paths)
    fields = propertynames(paths)
    isempty(fields) && throw(ArgumentError("selection output set must not be empty"))
    targets = String[getproperty(paths, field) for field in fields]
    length(unique(abspath.(targets))) == length(targets) ||
        throw(ArgumentError("selection output set contains duplicate paths"))
    length(unique(dirname.(abspath.(targets)))) == 1 || throw(ArgumentError(
        "selection output files must share one directory",
    ))
    return abspath(first(targets)) * ".set.lock"
end

function _with_selection_csv_set_lock(f::Function, paths)
    lock_path = _selection_csv_set_lock_path(paths)
    mkpath(dirname(lock_path))
    islink(lock_path) && throw(ArgumentError(
        "selection output lock must not be a symbolic link: $lock_path",
    ))
    ispath(lock_path) && !isfile(lock_path) && throw(ArgumentError(
        "selection output lock exists but is not a regular file: $lock_path",
    ))
    return Pidfile.mkpidlock(
        lock_path; wait=true, stale_age=900.0, refresh=450.0,
    ) do
        f()
    end
end

function _write_selection_csv_set(paths, frames)
    propertynames(paths) == propertynames(frames) || throw(ArgumentError(
        "selection output paths and frames must have identical fields",
    ))
    return _with_selection_csv_set_lock(paths) do
        fields = propertynames(paths)
        foreach(field -> _require_regular_output_target(getproperty(paths, field)), fields)
        staging = mktempdir(dirname(getproperty(paths, first(fields))))
        backups = Dict{String,String}()
        installed = String[]
        try
            staged = Dict{Symbol,String}()
            for field in fields
                path = joinpath(staging, String(field) * ".csv")
                CSV.write(path, getproperty(frames, field))
                staged[field] = path
            end
            for field in fields
                target = getproperty(paths, field)
                _require_regular_output_target(target)
                if isfile(target)
                    backup = joinpath(staging, String(field) * ".backup")
                    cp(target, backup)
                    backups[target] = backup
                end
                _atomic_replace_regular(staged[field], target)
                push!(installed, target)
            end
        catch
            for target in reverse(installed)
                backup = get(backups, target, nothing)
                if backup === nothing
                    isfile(target) && !islink(target) && rm(target; force=true)
                elseif isfile(backup)
                    _atomic_replace_regular(backup, target)
                end
            end
            rethrow()
        finally
            rm(staging; recursive=true, force=true)
        end
        return paths
    end
end

function _read_selection_csv_set(paths)
    return _with_selection_csv_set_lock(paths) do
        fields = propertynames(paths)
        frames = map(fields) do field
            path = getproperty(paths, field)
            isfile(path) && !islink(path) || error(
                "selection output must be a regular non-symlink file: $path",
            )
            CSV.read(path, DataFrame)
        end
        NamedTuple{fields}(Tuple(frames))
    end
end

function _atomic_install_regular_file_set(staged_paths::AbstractVector,
                                          target_paths::AbstractVector)
    length(staged_paths) == length(target_paths) || throw(DimensionMismatch(
        "staged and target file sets must have equal length",
    ))
    isempty(staged_paths) && return String[]
    length(unique(abspath.(target_paths))) == length(target_paths) ||
        throw(ArgumentError("target file set contains duplicate paths"))
    all(path -> isfile(path) && !islink(path), staged_paths) ||
        throw(ArgumentError("every staged source must be a regular file"))
    foreach(_require_regular_output_target, target_paths)
    foreach(path -> mkpath(dirname(path)), target_paths)

    backup_root = mktempdir(dirname(first(target_paths)))
    backups = Dict{String,String}()
    installed = String[]
    try
        for (index, target) in enumerate(target_paths)
            if isfile(target)
                backup = joinpath(backup_root, "backup_$index")
                cp(target, backup)
                backups[String(target)] = backup
            end
            _atomic_replace_regular(staged_paths[index], target)
            push!(installed, String(target))
        end
    catch
        for target in reverse(installed)
            backup = get(backups, target, nothing)
            if backup === nothing
                isfile(target) && !islink(target) && rm(target; force=true)
            elseif isfile(backup)
                _atomic_replace_regular(backup, target)
            end
        end
        rethrow()
    finally
        rm(backup_root; recursive=true, force=true)
    end
    return String.(target_paths)
end

function _snapshot_regular_file_set(paths::AbstractVector)
    targets = unique(String.(paths))
    isempty(targets) && throw(ArgumentError("snapshot file set must not be empty"))
    foreach(_require_regular_output_target, targets)
    mkpath(dirname(first(targets)))
    backup_root = mktempdir(dirname(first(targets)))
    backups = Dict{String,String}()
    for (index, target) in enumerate(targets)
        isfile(target) || continue
        backup = joinpath(backup_root, "backup_$index")
        cp(target, backup)
        backups[target] = backup
    end
    return (; targets, backups, backup_root)
end

function _restore_regular_file_set!(snapshot)
    for target in snapshot.targets
        backup = get(snapshot.backups, target, nothing)
        if backup === nothing
            isfile(target) && !islink(target) && rm(target; force=true)
        elseif isfile(backup)
            _atomic_replace_regular(backup, target)
        end
    end
    rm(snapshot.backup_root; recursive=true, force=true)
    return nothing
end

function _discard_regular_file_snapshot!(snapshot)
    rm(snapshot.backup_root; recursive=true, force=true)
    return nothing
end

function _storm_selection_paths(output_root::AbstractString,
                                prefix::AbstractString)
    isempty(output_root) && throw(ArgumentError("output_root must not be empty"))
    isempty(prefix) && throw(ArgumentError("prefix must not be empty"))
    basename(prefix) == prefix ||
        throw(ArgumentError("prefix must be a file-name prefix, not a path"))
    return (
        split = joinpath(output_root, "$(prefix)_inner_split.csv"),
        candidates = joinpath(output_root, "$(prefix)_candidates.csv"),
        errors = joinpath(output_root, "$(prefix)_validation_errors.csv"),
        support = joinpath(output_root, "$(prefix)_support.csv"),
        decision = joinpath(output_root, "$(prefix)_decision.csv"),
    )
end

"""
    write_storm_lambda_selection(result, output_root; prefix="storm_lambda")

Persist a storm-level selection result under the explicit `output_root`. Returns
the five written CSV paths. Files contain the chronological inner split,
candidate scores, per-storm errors, candidate/final support, and final decision.
Concurrent readers must use [`read_storm_lambda_selection`](@ref), which shares
the writer's file-set lock and therefore observes one complete generation.
"""
function write_storm_lambda_selection(result, output_root::AbstractString;
                                      prefix::AbstractString="storm_lambda")
    paths = _storm_selection_paths(output_root, prefix)
    mkpath(output_root)
    candidate_frame = DataFrame(result.candidate_records)
    rename!(candidate_frame,
        :mean_storm_rmse => :mean_storm_rmse_nt,
        :standard_error => :standard_error_nt,
    )
    error_frame = DataFrame(result.error_records)
    :rmse in propertynames(error_frame) && rename!(error_frame, :rmse => :rmse_nt)
    decision_frame = DataFrame([result.decision_record])
    rename!(decision_frame,
        :minimum_mean_storm_rmse => :minimum_mean_storm_rmse_nt,
        :minimum_standard_error => :minimum_standard_error_nt,
        :one_standard_error_cutoff => :one_standard_error_cutoff_nt,
    )
    empty_support = DataFrame(stage=String[], candidate_index=Int[],
                              lambda=Float64[], term=String[])
    support_frame = isempty(result.support_records) ?
        empty_support : DataFrame(result.support_records)
    frames = (
        split=DataFrame(result.split_records),
        candidates=candidate_frame,
        errors=error_frame,
        support=support_frame,
        decision=decision_frame,
    )
    return _write_selection_csv_set(paths, frames)
end

"""
    read_storm_lambda_selection(output_root; prefix="storm_lambda")

Read all five CSV files written by [`write_storm_lambda_selection`](@ref) while
holding the same cross-process lock used by the writer. Using this API prevents
a reader from observing files from different selection generations.
"""
function read_storm_lambda_selection(output_root::AbstractString;
                                     prefix::AbstractString="storm_lambda")
    return _read_selection_csv_set(_storm_selection_paths(output_root, prefix))
end
