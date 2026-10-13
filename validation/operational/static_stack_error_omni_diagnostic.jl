isdefined(@__MODULE__, :run_static_stack_driver_core_omni_diagnostic) ||
    include(joinpath(@__DIR__, "static_stack_driver_core_omni_diagnostic.jl"))

const STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE = joinpath(
    @__DIR__, "..", "output", "operational",
    "static_stack_residual_development", "static_stack_residual_replay.csv",
)
const STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE_SHA256 =
    "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f"
const STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256 =
    "30b06222b8b48329955b6e122aeefc55bd628868b405f9124a3cc5094c976f1b"
const STATIC_STACK_ERROR_OMNI_CORE_ERRORS_SHA256 =
    "dd1f57ddd1c622da436d8f1f59279270fe15f09bf1c9ca7d6246c5a9d7c2b912"
const STATIC_STACK_ERROR_OMNI_HISTORY_HOURS = 24
const STATIC_STACK_ERROR_OMNI_MINIMUM_GAIN_NT = 0.25
const STATIC_STACK_ERROR_OMNI_BOOTSTRAP_SEED = 22_022_026
const STATIC_STACK_ERROR_OMNI_BOOTSTRAP_DRAWS = 10_000
const STATIC_STACK_ERROR_OMNI_DIAGNOSTIC_SCHEMA = "static_stack_error_omni_diagnostic_v1"
const STATIC_STACK_ERROR_OMNI_FIRST_FEATURES =
    STATIC_STACK_RESIDUAL_FEATURES[1:15]
const STATIC_STACK_ERROR_OMNI_BASELINE_SPREAD_FEATURE = :baseline_spread_nt
const STATIC_STACK_ERROR_OMNI_SOURCE_EXPERTS = (
    :served_inertial_sindy_dst_nt,
    :frozen_inertial_sindy_dst_nt,
    :raw_sindy_dst_nt,
    :persistence_dst_nt,
    :burton_dst_nt,
    :burton_full_dst_nt,
    :obrien_dst_nt,
    :static_stack_pred_dst_nt,
    :observation_dst_nt,
)
const STATIC_STACK_ERROR_OMNI_CORE_COMPARATORS = (
    "burton_dense_driver",
    "burton_full_dense_driver",
    "burton_full_persistent_driver",
    "burton_full_sparse_driver",
    "burton_persistent_driver",
    "burton_sparse_driver",
    "dst_persistence",
    "obrien_dense_driver",
    "obrien_persistent_driver",
    "obrien_sparse_driver",
    "sindy_dense_driver",
    "sindy_persistent_driver",
    "sindy_sparse_driver",
)
const STATIC_STACK_ERROR_OMNI_SOURCE_COMPARATORS = (
    "served_inertial_sindy",
    "frozen_inertial_sindy",
    "source_raw_sindy",
    "static_stack_primary",
)
const STATIC_STACK_ERROR_OMNI_COMPARATORS = (
    STATIC_STACK_ERROR_OMNI_CORE_COMPARATORS...,
    STATIC_STACK_ERROR_OMNI_SOURCE_COMPARATORS...,
)

struct StaticStackM3OmniBaseRecord
    issue_time::DateTime
    model_step_hours::Int
    source_row::Int
    split_label::String
    prediction_base_sha256::String
    issue_features::NTuple{length(STATIC_STACK_RESIDUAL_FEATURES),Float64}
    m2_trajectory::Matrix{Float64}
    base_prediction_dst_nt::Float64
    observation_dst_nt::Float64
end

struct StaticStackM3OmniExample
    fold::String
    fit_row::StaticStackErrorExogenousFitRow
    latest_dst_nt::Float64
    dst_delta_1h_nt::Float64
end

function _err_omni_parse_datetime(value)
    value isa DateTime && return value
    return DateTime(String(value))
end

function _err_omni_source_columns()
    return (
        :issue_time_utc,
        :target_time_utc,
        :model_step_hours,
        :split_label,
        STATIC_STACK_ERROR_OMNI_SOURCE_EXPERTS...,
        STATIC_STACK_ERROR_OMNI_FIRST_FEATURES...,
        STATIC_STACK_ERROR_OMNI_BASELINE_SPREAD_FEATURE,
    )
end

function _err_omni_point_row_count(path::AbstractString)
    point_rows = 0
    open(path, "r") do io
        eof(io) && throw(ArgumentError("M3 OMNI source is empty"))
        readline(io)
        while !eof(io)
            prefix = ntuple(4) do field
                value = String(readuntil(io, UInt8(',')))
                isempty(value) && throw(ArgumentError(
                    "M3 OMNI source row has an empty prefix field $field",
                ))
                (occursin('\n', value) || occursin('\r', value) ||
                 occursin('"', value)) && throw(ArgumentError(
                    "M3 OMNI source prefix is not plain unquoted CSV",
                ))
                value
            end
            split_label = prefix[4]
            if split_label == "calibration"
                return point_rows
            elseif split_label == "fit" || split_label == "validation"
                point_rows += 1
                readline(io)
            else
                throw(ArgumentError(
                    "M3 OMNI source contains an unknown split label",
                ))
            end
        end
    end
    point_rows > 0 || throw(ArgumentError(
        "M3 OMNI source contains no point-development rows",
    ))
    return point_rows
end

function static_stack_error_omni_read_source(path::AbstractString;
        expected_sha256::AbstractString=STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE_SHA256)
    _drv_omni_sha256(path) == expected_sha256 || throw(ArgumentError(
        "M3 OMNI diagnostic source checksum changed",
    ))
    columns = _err_omni_source_columns()
    point_rows = _err_omni_point_row_count(path)
    source = CSV.read(path, DataFrame; limit=point_rows)
    SolarSINDy.canonicalize_static_stack_frozen_table!(source)
    Set(columns) ⊆ Set(Symbol.(names(source))) || throw(ArgumentError(
        "M3 OMNI diagnostic source schema changed",
    ))
    select!(source, collect(columns))
    source.issue_time_utc = _err_omni_parse_datetime.(source.issue_time_utc)
    source.target_time_utc = _err_omni_parse_datetime.(source.target_time_utc)
    source.model_step_hours = Int.(source.model_step_hours)
    source.split_label = String.(source.split_label)
    all(source.split_label .∈ Ref(("fit", "validation"))) ||
        throw(ArgumentError("M3 OMNI source contains an unknown split label"))
    count(==("calibration"), source.split_label) == 0 ||
        throw(ArgumentError("M3 OMNI point reader admitted calibration rows"))
    all(source.model_step_hours .∈ Ref(STATIC_STACK_DRIVER_OMNI_LEADS_H)) ||
        throw(ArgumentError("M3 OMNI source contains an unsupported lead"))
    all(source.target_time_utc .== source.issue_time_utc .+
        Hour.(source.model_step_hours)) || throw(ArgumentError(
            "M3 OMNI source target keys do not match their leads",
        ))
    maximum(source.issue_time_utc) < DateTime(2023, 1, 1) &&
        maximum(source.target_time_utc) < DateTime(2023, 1, 1) ||
        throw(ArgumentError("M3 OMNI source crosses the pre-2023 boundary"))
    keys = collect(zip(source.issue_time_utc, source.model_step_hours))
    length(unique(keys)) == length(keys) || throw(ArgumentError(
        "M3 OMNI source contains duplicate issue/lead keys",
    ))
    for column in (STATIC_STACK_ERROR_OMNI_SOURCE_EXPERTS...,
                   STATIC_STACK_ERROR_OMNI_FIRST_FEATURES...)
        all(isfinite, Float64.(source[!, column])) || throw(ArgumentError(
            "M3 OMNI source contains non-finite $column values",
        ))
    end
    return source
end

function _err_omni_source_index(source::DataFrame)
    index = Dict{Tuple{DateTime,Int},Int}()
    for row in 1:nrow(source)
        key = (source.issue_time_utc[row], source.model_step_hours[row])
        haskey(index, key) && throw(ArgumentError(
            "M3 OMNI source key is duplicated",
        ))
        index[key] = row
    end
    return index
end

function _err_omni_issue_features(source::DataFrame, row::Int,
        primary_prediction::Float64)
    values = Float64[Float64(source[row, feature])
                     for feature in STATIC_STACK_ERROR_OMNI_FIRST_FEATURES]
    served = Float64(source.served_inertial_sindy_dst_nt[row])
    frozen = Float64(source.frozen_inertial_sindy_dst_nt[row])
    persistence = Float64(source.persistence_dst_nt[row])
    burton = Float64(source.burton_dst_nt[row])
    burton_full = Float64(source.burton_full_dst_nt[row])
    obrien = Float64(source.obrien_dst_nt[row])
    push!(values, served - frozen)
    push!(values, primary_prediction - served)
    push!(values, primary_prediction - frozen)
    push!(values, primary_prediction - persistence)
    push!(values, primary_prediction - burton_full)
    push!(values, primary_prediction - obrien)
    push!(values, Float64(source.baseline_spread_nt[row]))
    length(values) == length(STATIC_STACK_RESIDUAL_FEATURES) ||
        throw(ErrorException("M3 OMNI issue feature construction is incomplete"))
    all(isfinite, values) || throw(ArgumentError(
        "M3 OMNI issue features are non-finite",
    ))
    return Tuple(values)
end

function _err_omni_hash_values(values)
    io = IOBuffer()
    for value in values
        print(io, ncodeunits(string(value)), ':', value, '|')
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _err_omni_dataframe_sha256(table::DataFrame)
    values = Any[names(table)...]
    for row in eachrow(table), column in names(table)
        push!(values, row[column])
    end
    return _err_omni_hash_values(values)
end

function _err_omni_prediction_base_sha256(core::OperationalCore,
        artifact::StaticStackDriverArtifact)
    return _err_omni_hash_values((
        STATIC_STACK_ERROR_OMNI_DIAGNOSTIC_SCHEMA,
        "mixed_source_omni_hro2_zero_latency_dst_oracle",
        STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256,
        static_stack_driver_sha256(artifact),
        SolarSINDy.STATIC_STACK_CORE_PATH_SCHEMA_VERSION,
        static_stack_core_sha256(core),
        "same_hour_dst_anchor",
        "zero_known_prefix_steps",
    ))
end

function _err_omni_crossfit_manifest_sha256(manifest::DataFrame)
    return _err_omni_hash_values((
        STATIC_STACK_ERROR_OMNI_DIAGNOSTIC_SCHEMA,
        STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE_SHA256,
        STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256,
        STATIC_STACK_ERROR_OMNI_CORE_ERRORS_SHA256,
        _err_omni_dataframe_sha256(sort(manifest, :outer_fold)),
        STATIC_STACK_ERROR_EXOGENOUS_FEATURES...,
    ))
end

function _err_omni_base_records(series::StaticStackM2OmniSeries,
        dst_at::Dict{DateTime,Float64}, source::DataFrame,
        source_index::Dict{Tuple{DateTime,Int},Int},
        artifact::StaticStackDriverArtifact, core::OperationalCore,
        issue_start::DateTime, issue_stop::DateTime,
        prediction_base_sha256::AbstractString)
    occursin(r"^[0-9a-f]{64}$", prediction_base_sha256) ||
        throw(ArgumentError("M3 OMNI prediction-base checksum is invalid"))
    cases = static_stack_driver_omni_cases(series, issue_start, issue_stop).cases
    records = Dict{Tuple{DateTime,Int},StaticStackM3OmniBaseRecord}()
    for case in cases
        issue = case.issue_time
        required_source = all(haskey(source_index, (issue, lead))
                              for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H)
        required_dst = haskey(dst_at, issue) &&
            all(haskey(dst_at, issue + Hour(lead))
                for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H)
        required_source && required_dst || continue
        history = _drv_omni_history(series, case)
        path = _drv_omni_prediction(
            :sparse, artifact, series, case, 0,
        )
        anchor_star = dst_to_dst_star(
            dst_at[issue], _drv_core_anchor_pressure(history),
        )
        core_path = static_stack_core_path_forecast(core, anchor_star, path)
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
            source_row = source_index[(issue, lead)]
            observation = dst_at[issue + Hour(lead)]
            source_observation = Float64(source.observation_dst_nt[source_row])
            observation == source_observation || throw(ArgumentError(
                "M3 OMNI HAPI/source Dst observations disagree",
            ))
            base = core_path.pred_dst_nt[lead]
            records[(issue, lead)] = StaticStackM3OmniBaseRecord(
                issue,
                lead,
                source_row,
                String(source.split_label[source_row]),
                String(prediction_base_sha256),
                _err_omni_issue_features(source, source_row, base),
                path,
                base,
                observation,
            )
        end
    end
    return records
end

function _err_omni_examples(records, fold::AbstractString,
        issue_start::DateTime, issue_stop::DateTime,
        expected_split::AbstractString="fit")
    examples = StaticStackM3OmniExample[]
    required_lags = STATIC_STACK_ERROR_EXOGENOUS_LAGS_H
    issues = sort(unique(key[1] for key in keys(records)
                         if issue_start <= key[1] <= issue_stop))
    for issue in issues
        all(haskey(records, (issue, lead))
            for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H) || continue
        all(records[(issue, lead)].split_label == expected_split
            for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H) || throw(ArgumentError(
                "M3 OMNI $fold current anchor is outside the expected " *
                "$expected_split source split",
            ))
        all(haskey(records, (issue - Hour(lag), 1))
            for lag in 1:STATIC_STACK_ERROR_OMNI_HISTORY_HOURS) || continue
        history_times = Tuple(issue - Hour(lag) for lag in required_lags)
        all(haskey(records, (time, 1)) for time in history_times) || continue
        h1_history = Tuple(records[(time, 1)] for time in history_times)
        all(row.prediction_base_sha256 ==
            records[(issue, 1)].prediction_base_sha256 for row in h1_history) ||
            throw(ArgumentError(
                "M3 OMNI innovation history crosses a base-center identity",
            ))
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
            current = records[(issue, lead)]
            current.prediction_base_sha256 ==
                records[(issue, 1)].prediction_base_sha256 ||
                throw(ArgumentError(
                    "M3 OMNI current lead centers do not share one identity",
                ))
            values = Float64[current.issue_features...]
            for feature_index in (1, 5, 7, 11)
                append!(values, (row.issue_features[feature_index]
                                 for row in h1_history))
            end
            append!(values, (row.observation_dst_nt -
                             row.base_prediction_dst_nt
                             for row in h1_history))
            append!(values, @view current.m2_trajectory[2 * lead, :])
            push!(values, current.base_prediction_dst_nt)
            length(values) == length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES) ||
                throw(ErrorException("M3 OMNI 73-feature row is incomplete"))
            features = StaticStackErrorExogenousFeatures(
                issue,
                lead,
                current.prediction_base_sha256,
                current.base_prediction_dst_nt,
                STATIC_STACK_ERROR_EXOGENOUS_FEATURES,
                Tuple(values),
                true,
                :none,
            )
            fit_row = StaticStackErrorExogenousFitRow(
                features,
                issue + Hour(lead),
                issue + Hour(lead),
                current.observation_dst_nt,
            )
            push!(examples, StaticStackM3OmniExample(
                String(fold), fit_row, current.issue_features[1],
                current.issue_features[2],
            ))
        end
    end
    length(examples) % length(STATIC_STACK_DRIVER_OMNI_LEADS_H) == 0 ||
        throw(ErrorException("M3 OMNI examples do not retain complete lead anchors"))
    return examples
end

function _err_omni_inner_driver_folds(outer_fold)
    outer_year = year(outer_fold.train_stop)
    outer_year >= 2013 || throw(ArgumentError(
        "M3 OMNI outer fold is too early for nested M2 selection",
    ))
    return [
        (
            name="inner_$(year_value)_for_$(outer_fold.name)",
            train_stop=DateTime(year_value, 1, 1),
            validation_start=DateTime(year_value, 1, 8),
            validation_stop=DateTime(year_value, 12, 24, 23),
        )
        for year_value in 2011:(outer_year - 1)
    ]
end

function _err_omni_fold_driver_artifacts(series::StaticStackM2OmniSeries)
    artifacts = StaticStackDriverArtifact[]
    manifest_rows = NamedTuple[]
    selection_tables = DataFrame[]
    for outer_fold in STATIC_STACK_DRIVER_OMNI_FOLDS
        inner_folds = _err_omni_inner_driver_folds(outer_fold)
        selection = static_stack_driver_omni_select_hyperparameters(
            series; folds=inner_folds,
        )
        audit = copy(selection.audit)
        audit[!, :outer_fold] = fill(outer_fold.name, nrow(audit))
        push!(selection_tables, audit)
        inner_selection_sha256 = _err_omni_dataframe_sha256(audit)
        training = static_stack_driver_omni_training_segments(
            series, outer_fold.train_stop,
        )
        training.max_time + Hour(STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS) <=
            outer_fold.validation_start || throw(ArgumentError(
                "nested M2 fold $(outer_fold.name) violates the target embargo",
            ))
        artifact = _drv_omni_trial_artifact(
            training.segments,
            selection.sparse.ridge,
            selection.sparse.threshold,
            "static-stack-m3-omni-oof-$(outer_fold.name)",
        )
        push!(artifacts, artifact)
        push!(manifest_rows, (
            outer_fold=outer_fold.name,
            outer_block_start_utc=outer_fold.validation_start,
            outer_block_stop_utc=outer_fold.validation_stop,
            training_target_max_utc=training.max_time,
            target_to_block_gap_hours=_drv_omni_elapsed_hours(
                training.max_time, outer_fold.validation_start,
            ),
            inner_fold_count=length(inner_folds),
            selected_ridge=selection.sparse.ridge,
            selected_threshold=selection.sparse.threshold,
            active_groups=count(static_stack_driver_support(artifact)),
            retained_training_segments=training.retained_segments,
            retained_training_rows=training.retained_rows,
            artifact_sha256=static_stack_driver_sha256(artifact),
            inner_selection_sha256=inner_selection_sha256,
            driver_bins_sha256=STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256,
            post_2022_rows_read=0,
        ))
    end
    return (
        artifacts=artifacts,
        manifest=DataFrame(manifest_rows),
        selection=reduce((left, right) -> vcat(left, right; cols=:union),
                         selection_tables),
    )
end

function _err_omni_expected_grid_failure(error)
    message = lowercase(sprint(showerror, error))
    return (error isa ArgumentError &&
            occursin("innovation recurrence is unstable", message)) ||
           (error isa ErrorException &&
            occursin("thresholding did not converge", message))
end

function _err_omni_corrected_prediction(
        artifact::StaticStackErrorExogenousArtifact,
        example::StaticStackM3OmniExample)
    features = example.fit_row.features
    raw = SolarSINDy._static_stack_error_exogenous_raw_predict(
        artifact, features.values,
    )
    correction = clamp(
        raw,
        -SolarSINDy._static_stack_error_cap(features.model_step_hours),
        SolarSINDy._static_stack_error_cap(features.model_step_hours),
    )
    prediction = features.base_prediction_dst_nt + correction
    isfinite(prediction) || throw(ErrorException(
        "M3 OMNI corrected prediction is non-finite",
    ))
    return prediction, raw, correction
end

function _err_omni_select_index(audit::DataFrame, indices)
    isempty(indices) && throw(ArgumentError(
        "M3 OMNI one-standard-error selection has no candidates",
    ))
    best = indices[argmin(audit.mean_validation_rmse_nt[indices])]
    limit = audit.mean_validation_rmse_nt[best] +
            audit.fold_standard_error_nt[best]
    eligible = filter(index ->
        audit.mean_validation_rmse_nt[index] <= limit,
        collect(indices),
    )
    sort!(eligible; by=index -> (
        audit.max_active_groups[index],
        -audit.ridge[index],
        -audit.threshold[index],
        audit.mean_validation_rmse_nt[index],
    ))
    return first(eligible), Set(eligible)
end

function _err_omni_selection_gains(
        examples::Vector{StaticStackM3OmniExample}, predictions::Vector{Float64})
    length(examples) == length(predictions) || throw(DimensionMismatch(
        "M3 OMNI selection predictions do not match their examples",
    ))
    observations = Float64[
        example.fit_row.observation_dst_nt for example in examples
    ]
    centers = Float64[
        example.fit_row.features.base_prediction_dst_nt for example in examples
    ]
    latest = Float64[example.latest_dst_nt for example in examples]
    delta = Float64[example.dst_delta_1h_nt for example in examples]
    masks = (
        overall=trues(length(examples)),
        target_storm=observations .<= -50.0,
        active_or_deepening=(latest .<= -30.0) .& (delta .< 0.0),
        recovery=(latest .<= -30.0) .& (delta .>= 0.0),
    )
    values = Dict{String,NamedTuple}()
    for (regime, selected) in pairs(masks)
        rows = count(selected)
        base_rmse = rows == 0 ? NaN :
            sqrt(mean((centers[selected] .- observations[selected]).^2))
        candidate_rmse = rows == 0 ? NaN :
            sqrt(mean((predictions[selected] .- observations[selected]).^2))
        values[String(regime)] = (
            rows=rows,
            base_rmse_nt=base_rmse,
            candidate_rmse_nt=candidate_rmse,
            gain_nt=rows == 0 ? NaN : base_rmse - candidate_rmse,
        )
    end
    safe = all(regime -> begin
        row = values[regime]
        row.rows < 20 || row.gain_nt >= 0.0
    end, ("target_storm", "active_or_deepening", "recovery"))
    return (values=values, regime_safe=safe)
end

function _err_omni_fit_crossfit_error_exogenous(
        examples::Vector{StaticStackM3OmniExample};
        serving_base_center_sha256::AbstractString,
        expected_fold_hashes::Dict{String,String},
        training_manifest_sha256::AbstractString,
        model_step_hours::Int,
        fit_as_of::DateTime,
        ridge::Real,
        threshold::Real,
        minimum_rows::Int=32,
        label::AbstractString="static-stack-m3-omni-diagnostic")
    occursin(r"^[0-9a-f]{64}$", training_manifest_sha256) ||
        throw(ArgumentError("M3 OMNI training-manifest checksum is invalid"))
    lead = Int(model_step_hours)
    lead in STATIC_STACK_DRIVER_OMNI_LEADS_H || throw(ArgumentError(
        "M3 OMNI cross-fit lead is unsupported",
    ))
    ridge_value = Float64(ridge)
    threshold_value = Float64(threshold)
    ridge_value in STATIC_STACK_ERROR_EXOGENOUS_RIDGE_GRID ||
        throw(ArgumentError("M3 OMNI ridge is outside the frozen grid"))
    threshold_value in STATIC_STACK_ERROR_EXOGENOUS_THRESHOLD_GRID ||
        throw(ArgumentError("M3 OMNI threshold is outside the frozen grid"))
    eligible = StaticStackM3OmniExample[]
    whole_anchor_stop = fit_as_of - Hour(maximum(STATIC_STACK_DRIVER_OMNI_LEADS_H))
    for example in examples
        haskey(expected_fold_hashes, example.fold) || throw(ArgumentError(
            "M3 OMNI example has no declared fold-center identity",
        ))
        example.fit_row.features.base_center_sha256 ==
            expected_fold_hashes[example.fold] || throw(ArgumentError(
                "M3 OMNI example center does not match its fold artifact",
            ))
        row = example.fit_row
        row.features.model_step_hours == lead || continue
        row.features.issue_time <= whole_anchor_stop || continue
        row.observation_available_at <= fit_as_of || continue
        push!(eligible, example)
    end
    sort!(eligible; by=example -> example.fit_row.features.issue_time)
    length(unique(example.fit_row.features.issue_time
                  for example in eligible)) == length(eligible) ||
        throw(ArgumentError("M3 OMNI cross-fit rows contain duplicate issues"))
    length(eligible) >= minimum_rows || throw(ArgumentError(
        "M3 OMNI cross-fit has $(length(eligible)) rows; " *
        "minimum_rows=$minimum_rows",
    ))
    nfeatures = length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)
    design = Matrix{Float64}(undef, length(eligible), nfeatures)
    target = Vector{Float64}(undef, length(eligible))
    for (row_index, example) in pairs(eligible)
        row = example.fit_row
        design[row_index, :] .= row.features.values
        target[row_index] = row.observation_dst_nt -
                            row.features.base_prediction_dst_nt
    end
    all(isfinite, design) && all(isfinite, target) || throw(ArgumentError(
        "M3 OMNI cross-fit inputs must be finite",
    ))
    feature_center = vec(mean(design; dims=1))
    feature_scale = vec(std(design; dims=1, corrected=false))
    for index in eachindex(feature_scale)
        iszero(feature_scale[index]) && (feature_scale[index] = 1.0)
    end
    standardized = (design .- transpose(feature_center)) ./
                   transpose(feature_scale)
    intercept, coefficients, support, iterations =
        SolarSINDy._static_stack_error_exogenous_threshold_fit(
            standardized, target, ridge_value, threshold_value,
        )
    artifact_label = String(label) * "-manifest-" *
                     String(training_manifest_sha256)
    return StaticStackErrorExogenousArtifact(
        serving_base_center_sha256,
        lead,
        intercept,
        coefficients;
        support_mask=Tuple(support),
        feature_center=feature_center,
        feature_scale=feature_scale,
        ridge=ridge_value,
        threshold=threshold_value,
        fit_rows=length(eligible),
        threshold_iterations=iterations,
        label=artifact_label,
    )
end

function _err_omni_choose_hyperparameters!(audit::DataFrame)
    selected = Dict{Int,NamedTuple}()
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        indices = findall((audit.model_step_hours .== lead) .& audit.valid .&
                          audit.regime_safe .&
                          (audit.overall_gain_nt .> 0.0))
        if isempty(indices)
            selected[lead] = (
                use_correction=false,
                ridge=100.0,
                threshold=0.0,
                reason="no_regime_safe_positive_oof_gain",
            )
            continue
        end
        index, eligible = _err_omni_select_index(audit, indices)
        audit.one_standard_error_eligible .=
            audit.one_standard_error_eligible .|
            [row in eligible for row in 1:nrow(audit)]
        audit.selected[index] = true
        selected[lead] = (
            use_correction=true,
            ridge=audit.ridge[index],
            threshold=audit.threshold[index],
            reason="selected_regime_safe_one_standard_error",
        )
    end
    return selected
end

function static_stack_error_omni_select_hyperparameters(
        examples::Vector{StaticStackM3OmniExample},
        serving_base_center_sha256::AbstractString,
        fold_hashes::Dict{String,String},
        training_manifest_sha256::AbstractString)
    selection_folds = STATIC_STACK_DRIVER_OMNI_FOLDS[2:end]
    rows = NamedTuple[]
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
            ridge in STATIC_STACK_ERROR_EXOGENOUS_RIDGE_GRID,
            threshold in STATIC_STACK_ERROR_EXOGENOUS_THRESHOLD_GRID
        fold_losses = Float64[]
        fold_base_losses = Float64[]
        active_groups = Int[]
        validation_examples = StaticStackM3OmniExample[]
        validation_predictions = Float64[]
        failure = ""
        for fold in selection_folds
            fit_cutoff = fold.validation_start -
                         Hour(STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS)
            artifact = try
                _err_omni_fit_crossfit_error_exogenous(
                    examples;
                    serving_base_center_sha256=serving_base_center_sha256,
                    expected_fold_hashes=fold_hashes,
                    training_manifest_sha256=training_manifest_sha256,
                    model_step_hours=lead,
                    fit_as_of=fit_cutoff,
                    ridge=ridge,
                    threshold=threshold,
                    minimum_rows=32,
                    label="static-stack-m3-omni-select-$(fold.name)-h$(lead)",
                )
            catch error
                _err_omni_expected_grid_failure(error) || rethrow()
                failure = sprint(showerror, error)
                break
            end
            validation = [
                example for example in examples
                if example.fold == fold.name &&
                   example.fit_row.features.model_step_hours == lead
            ]
            isempty(validation) && throw(ArgumentError(
                "M3 OMNI fold $(fold.name) has no complete lead-$lead examples",
            ))
            predictions = Float64[
                first(_err_omni_corrected_prediction(artifact, example))
                for example in validation
            ]
            observations = Float64[
                example.fit_row.observation_dst_nt for example in validation
            ]
            centers = Float64[
                example.fit_row.features.base_prediction_dst_nt
                for example in validation
            ]
            push!(fold_losses, sqrt(mean((predictions .- observations).^2)))
            push!(fold_base_losses,
                  sqrt(mean((centers .- observations).^2)))
            push!(active_groups, count(artifact.support_mask))
            append!(validation_examples, validation)
            append!(validation_predictions, predictions)
        end
        valid = length(fold_losses) == length(selection_folds)
        gains = valid ? _err_omni_selection_gains(
            validation_examples, validation_predictions,
        ) : nothing
        overall = valid ? gains.values["overall"] : nothing
        push!(rows, (
            model_step_hours=lead,
            ridge=Float64(ridge),
            threshold=Float64(threshold),
            valid=valid,
            mean_validation_rmse_nt=valid ? mean(fold_losses) : NaN,
            mean_base_rmse_nt=valid ? mean(fold_base_losses) : NaN,
            fold_standard_error_nt=valid && length(fold_losses) > 1 ?
                std(fold_losses) / sqrt(length(fold_losses)) : 0.0,
            max_active_groups=valid ? maximum(active_groups) : -1,
            overall_gain_nt=valid ? overall.gain_nt : NaN,
            target_storm_gain_nt=valid ?
                gains.values["target_storm"].gain_nt : NaN,
            active_or_deepening_gain_nt=valid ?
                gains.values["active_or_deepening"].gain_nt : NaN,
            recovery_gain_nt=valid ? gains.values["recovery"].gain_nt : NaN,
            regime_safe=valid ? gains.regime_safe : false,
            fold_losses_nt=valid ? join(string.(fold_losses), ';') : "",
            fold_base_losses_nt=valid ?
                join(string.(fold_base_losses), ';') : "",
            failure_reason=failure,
            one_standard_error_eligible=false,
            selected=false,
        ))
    end
    audit = DataFrame(rows)
    selected = _err_omni_choose_hyperparameters!(audit)
    return (audit=audit, selected=selected)
end

function _err_omni_fit_final_artifacts(
        examples::Vector{StaticStackM3OmniExample}, selection,
        serving_base_center_sha256::AbstractString,
        fold_hashes::Dict{String,String},
        training_manifest_sha256::AbstractString)
    artifacts = StaticStackErrorExogenousArtifact[]
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        hyperparameters = selection.selected[lead]
        if !hyperparameters.use_correction
            eligible_rows = count(example ->
                example.fit_row.features.model_step_hours == lead &&
                example.fit_row.features.issue_time <=
                    STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP -
                    Hour(maximum(STATIC_STACK_DRIVER_OMNI_LEADS_H)) &&
                example.fit_row.observation_available_at <=
                    STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP,
                examples,
            )
            eligible_rows >= 2 || throw(ArgumentError(
                "M3 OMNI null artifact has fewer than two fit rows",
            ))
            push!(artifacts, StaticStackErrorExogenousArtifact(
                serving_base_center_sha256,
                lead,
                0.0,
                zeros(length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES));
                support_mask=ntuple(
                    _ -> false,
                    length(STATIC_STACK_ERROR_EXOGENOUS_GROUPS),
                ),
                ridge=hyperparameters.ridge,
                threshold=hyperparameters.threshold,
                fit_rows=eligible_rows,
                threshold_iterations=1,
                label="static-stack-m3-omni-diagnostic-null-h$(lead)-manifest-" *
                      String(training_manifest_sha256),
            ))
            continue
        end
        push!(artifacts, _err_omni_fit_crossfit_error_exogenous(
            examples;
            serving_base_center_sha256=serving_base_center_sha256,
            expected_fold_hashes=fold_hashes,
            training_manifest_sha256=training_manifest_sha256,
            model_step_hours=lead,
            fit_as_of=STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP,
            ridge=hyperparameters.ridge,
            threshold=hyperparameters.threshold,
            minimum_rows=32,
            label="static-stack-m3-omni-diagnostic-h$(lead)",
        ))
    end
    return artifacts
end

function _err_omni_examples_table(examples::Vector{StaticStackM3OmniExample})
    rows = NamedTuple[]
    for example in examples
        features = example.fit_row.features
        values = NamedTuple{STATIC_STACK_ERROR_EXOGENOUS_FEATURES}(
            features.values,
        )
        push!(rows, merge((
            fold=example.fold,
            issue_time_utc=features.issue_time,
            target_time_utc=example.fit_row.target_time,
            model_step_hours=features.model_step_hours,
            base_center_sha256=features.base_center_sha256,
            base_prediction_dst_nt=features.base_prediction_dst_nt,
            observation_dst_nt=example.fit_row.observation_dst_nt,
        ), values))
    end
    return DataFrame(rows)
end

function static_stack_error_omni_read_core_errors(path::AbstractString;
        expected_sha256::AbstractString=STATIC_STACK_ERROR_OMNI_CORE_ERRORS_SHA256)
    _drv_omni_sha256(path) == expected_sha256 || throw(ArgumentError(
        "M3 OMNI core-error checksum changed",
    ))
    columns = (
        :issue_time_utc, :method, :lead_hours, :latest_dst_nt,
        :dst_delta_1h_nt, :observed_dst_nt, :predicted_dst_nt,
    )
    errors = CSV.read(path, DataFrame; select=collect(String.(columns)))
    Set(Symbol.(names(errors))) == Set(columns) || throw(ArgumentError(
        "M3 OMNI core-error schema changed",
    ))
    errors.issue_time_utc = _err_omni_parse_datetime.(
        errors.issue_time_utc,
    )
    errors.method = String.(errors.method)
    errors.lead_hours = Int.(errors.lead_hours)
    Set(errors.method) == Set(STATIC_STACK_ERROR_OMNI_CORE_COMPARATORS) ||
        throw(ArgumentError("M3 OMNI core comparator set changed"))
    all(errors.lead_hours .∈ Ref(STATIC_STACK_DRIVER_OMNI_LEADS_H)) ||
        throw(ArgumentError("M3 OMNI core errors contain an unsupported lead"))
    all(STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START .<= errors.issue_time_utc .<=
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP) || throw(ArgumentError(
            "M3 OMNI core errors leave the exploratory validation window",
        ))
    for column in (:latest_dst_nt, :dst_delta_1h_nt,
                   :observed_dst_nt, :predicted_dst_nt)
        all(isfinite, Float64.(errors[!, column])) || throw(ArgumentError(
            "M3 OMNI core errors contain non-finite $column values",
        ))
    end
    keys = collect(zip(errors.issue_time_utc, errors.lead_hours,
                       errors.method))
    length(unique(keys)) == length(keys) || throw(ArgumentError(
        "M3 OMNI core errors contain duplicate issue/lead/method rows",
    ))
    return errors
end

function _err_omni_core_error_index(errors::DataFrame)
    return Dict(
        (DateTime(errors.issue_time_utc[row]), Int(errors.lead_hours[row]),
         String(errors.method[row])) => row
        for row in 1:nrow(errors)
    )
end

function _err_omni_ready_index(examples::Vector{StaticStackM3OmniExample})
    ready = Dict{Tuple{DateTime,Int},StaticStackM3OmniExample}()
    for example in examples
        key = (example.fit_row.features.issue_time,
               example.fit_row.features.model_step_hours)
        haskey(ready, key) && throw(ArgumentError(
            "M3 OMNI ready examples contain a duplicate issue/lead",
        ))
        ready[key] = example
    end
    return ready
end

function _err_omni_source_comparator_predictions(source::DataFrame,
        row::Int)
    return Dict(
        "served_inertial_sindy" => Float64(source.served_inertial_sindy_dst_nt[row]),
        "frozen_inertial_sindy" => Float64(source.frozen_inertial_sindy_dst_nt[row]),
        "source_raw_sindy" => Float64(source.raw_sindy_dst_nt[row]),
        "static_stack_primary" => Float64(source.static_stack_pred_dst_nt[row]),
    )
end

function static_stack_error_omni_score(records,
        ready_examples::Vector{StaticStackM3OmniExample},
        artifacts::Vector{StaticStackErrorExogenousArtifact},
        source::DataFrame, core_errors::DataFrame,
        final_base_center_sha256::AbstractString)
    ready_at = _err_omni_ready_index(ready_examples)
    artifact_at = Dict(artifact.model_step_hours => artifact
                       for artifact in artifacts)
    Set(keys(artifact_at)) == Set(STATIC_STACK_DRIVER_OMNI_LEADS_H) ||
        throw(ArgumentError("M3 OMNI final artifact set is incomplete"))
    all(artifact.base_center_sha256 == final_base_center_sha256
        for artifact in artifacts) || throw(ArgumentError(
            "M3 OMNI final artifacts do not bind the final base center",
        ))
    source_index = _err_omni_source_index(source)
    core_index = _err_omni_core_error_index(core_errors)
    base_rows = core_errors[core_errors.method .== "sindy_sparse_driver", :]
    sort!(base_rows, [:issue_time_utc, :lead_hours])
    rows = NamedTuple[]
    for base_row in eachrow(base_rows)
        issue = DateTime(base_row.issue_time_utc)
        lead = Int(base_row.lead_hours)
        key = (issue, lead)
        haskey(records, key) || throw(ArgumentError(
            "M3 OMNI final base record is missing for $issue lead $lead",
        ))
        current = records[key]
        current.split_label == "validation" || throw(ArgumentError(
            "M3 OMNI exploratory score admitted a non-validation point row",
        ))
        current.prediction_base_sha256 == final_base_center_sha256 ||
            throw(ArgumentError(
                "M3 OMNI final base record has the wrong center identity",
            ))
        isapprox(current.base_prediction_dst_nt,
                 Float64(base_row.predicted_dst_nt); atol=1e-12, rtol=1e-12) ||
            throw(ArgumentError(
                "M3 OMNI reconstructed center differs from the frozen core replay",
            ))
        source_row = source_index[key]
        source.split_label[source_row] == "validation" || throw(ArgumentError(
            "M3 OMNI source comparator row is not in point validation",
        ))
        ready = haskey(ready_at, key)
        if ready
            example = ready_at[key]
            example.fit_row.features.base_center_sha256 ==
                final_base_center_sha256 || throw(ArgumentError(
                    "M3 OMNI ready row is bound to the wrong final center",
                ))
            candidate, raw_correction, correction =
                _err_omni_corrected_prediction(artifact_at[lead], example)
            fallback_reason = "none"
        else
            candidate = current.base_prediction_dst_nt
            raw_correction = 0.0
            correction = 0.0
            fallback_reason = "missing_contiguous_24h_history"
        end
        predictions = Dict{String,Float64}(
            "m3_sparse_error_state" => candidate,
        )
        for comparator in STATIC_STACK_ERROR_OMNI_CORE_COMPARATORS
            index_key = (issue, lead, comparator)
            haskey(core_index, index_key) || throw(ArgumentError(
                "M3 OMNI core comparator row is missing",
            ))
            predictions[comparator] = Float64(
                core_errors.predicted_dst_nt[core_index[index_key]],
            )
        end
        merge!(predictions,
               _err_omni_source_comparator_predictions(source, source_row))
        Set(keys(predictions)) == Set(("m3_sparse_error_state",
                                      STATIC_STACK_ERROR_OMNI_COMPARATORS...)) ||
            throw(ErrorException("M3 OMNI comparison set is incomplete"))
        observed = current.observation_dst_nt
        latest = current.issue_features[1]
        delta = current.issue_features[2]
        for method in sort!(collect(keys(predictions)))
            prediction = predictions[method]
            isfinite(prediction) || throw(ErrorException(
                "M3 OMNI prediction is non-finite",
            ))
            push!(rows, (
                issue_time_utc=issue,
                target_time_utc=issue + Hour(lead),
                model_step_hours=lead,
                method=method,
                comparison_role=method == "m3_sparse_error_state" ?
                    "candidate" :
                    (occursin("dense_driver", method) ?
                     "retrospective_driver_oracle_control" : "available_comparator"),
                candidate_ready=ready,
                fallback_reason=fallback_reason,
                split_label=current.split_label,
                predicted_dst_nt=prediction,
                observed_dst_nt=observed,
                squared_error_nt2=(prediction - observed)^2,
                absolute_error_nt=abs(prediction - observed),
                raw_candidate_correction_nt=method == "m3_sparse_error_state" ?
                    raw_correction : 0.0,
                applied_candidate_correction_nt=
                    method == "m3_sparse_error_state" ? correction : 0.0,
                latest_dst_nt=latest,
                dst_delta_1h_nt=delta,
                target_storm=observed <= -50.0,
                active_or_deepening=latest <= -30.0 && delta < 0.0,
                recovery=latest <= -30.0 && delta >= 0.0,
            ))
        end
    end
    errors = DataFrame(rows)
    expected_per_method = nrow(base_rows)
    all(nrow(group) == expected_per_method
        for group in groupby(errors, :method)) || throw(ArgumentError(
            "M3 OMNI methods do not retain identical common rows",
        ))
    count(==("calibration"), errors.split_label) == 0 ||
        throw(ArgumentError("M3 OMNI score admitted calibration rows"))
    return errors
end

function static_stack_error_omni_metrics(errors::DataFrame)
    rows = NamedTuple[]
    for group in groupby(errors, [:method, :model_step_hours])
        regimes = (
            overall=trues(nrow(group)),
            target_storm=group.target_storm,
            active_or_deepening=group.active_or_deepening,
            recovery=group.recovery,
        )
        for (regime, selected) in pairs(regimes)
            retained = count(selected)
            push!(rows, (
                method=String(group.method[1]),
                model_step_hours=Int(group.model_step_hours[1]),
                regime=String(regime),
                rows=retained,
                ready_rows=count(group.candidate_ready[selected]),
                readiness=retained == 0 ? NaN :
                    count(group.candidate_ready[selected]) / retained,
                rmse_nt=retained == 0 ? NaN :
                    sqrt(mean(group.squared_error_nt2[selected])),
                mae_nt=retained == 0 ? NaN :
                    mean(group.absolute_error_nt[selected]),
            ))
        end
    end
    return DataFrame(rows)
end

function _err_omni_calendar_block(time::DateTime)
    return year(time) * 12 + month(time) - 1
end

function static_stack_error_omni_bootstrap(errors::DataFrame;
        draws::Integer=STATIC_STACK_ERROR_OMNI_BOOTSTRAP_DRAWS,
        seed::Integer=STATIC_STACK_ERROR_OMNI_BOOTSTRAP_SEED,
        candidate::AbstractString="m3_sparse_error_state",
        comparators=STATIC_STACK_ERROR_OMNI_COMPARATORS)
    draws >= 100 || throw(ArgumentError(
        "M3 OMNI bootstrap requires at least 100 draws",
    ))
    specifications = NamedTuple[]
    common_blocks = nothing
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H, comparator in comparators
        candidate_rows = errors[(errors.method .== candidate) .&
                                (errors.model_step_hours .== lead),
                                [:issue_time_utc, :squared_error_nt2]]
        reference = errors[(errors.method .== comparator) .&
                           (errors.model_step_hours .== lead),
                           [:issue_time_utc, :squared_error_nt2]]
        sort!(candidate_rows, :issue_time_utc)
        sort!(reference, :issue_time_utc)
        candidate_rows.issue_time_utc == reference.issue_time_utc ||
            throw(ArgumentError("M3 OMNI bootstrap rows are not exactly paired"))
        blocks = _err_omni_calendar_block.(candidate_rows.issue_time_utc)
        unique_blocks = sort(unique(blocks))
        length(unique_blocks) >= 2 || throw(ArgumentError(
            "M3 OMNI bootstrap requires at least two calendar months",
        ))
        if common_blocks === nothing
            common_blocks = unique_blocks
        else
            unique_blocks == common_blocks || throw(ArgumentError(
                "M3 OMNI lead comparisons do not share identical month blocks",
            ))
        end
        candidate_sums = zeros(Float64, length(unique_blocks))
        comparator_sums = zeros(Float64, length(unique_blocks))
        counts = zeros(Int, length(unique_blocks))
        block_index = Dict(block => index
                           for (index, block) in pairs(unique_blocks))
        for row in 1:nrow(candidate_rows)
            index = block_index[blocks[row]]
            candidate_sums[index] += candidate_rows.squared_error_nt2[row]
            comparator_sums[index] += reference.squared_error_nt2[row]
            counts[index] += 1
        end
        push!(specifications, (
            comparator=comparator,
            lead=lead,
            rows=nrow(candidate_rows),
            blocks=length(unique_blocks),
            candidate_sums=candidate_sums,
            comparator_sums=comparator_sums,
            counts=counts,
            point=sqrt(mean(reference.squared_error_nt2)) -
                  sqrt(mean(candidate_rows.squared_error_nt2)),
        ))
    end
    block_count = length(common_blocks)
    differences = Matrix{Float64}(undef, draws, length(specifications))
    rng = MersenneTwister(seed)
    for draw in 1:draws
        sampled = rand(rng, 1:block_count, block_count)
        for (comparison, specification) in pairs(specifications)
            n = sum(specification.counts[sampled])
            n > 0 || throw(ErrorException(
                "M3 OMNI bootstrap sampled no paired rows",
            ))
            candidate_rmse = sqrt(
                sum(specification.candidate_sums[sampled]) / n,
            )
            comparator_rmse = sqrt(
                sum(specification.comparator_sums[sampled]) / n,
            )
            differences[draw, comparison] = comparator_rmse - candidate_rmse
        end
    end
    simultaneous = quantile(vec(minimum(differences; dims=2)), 0.05)
    rows = NamedTuple[]
    for (comparison, specification) in pairs(specifications)
        push!(rows, (
            candidate=String(candidate),
            comparator=specification.comparator,
            model_step_hours=specification.lead,
            rows=specification.rows,
            blocks=specification.blocks,
            draws=Int(draws),
            seed=Int(seed),
            block_policy="calendar_month_only_event_ids_unavailable",
            event_block_complete=false,
            point_improvement_nt=specification.point,
            one_sided_95_lower_nt=quantile(
                @view(differences[:, comparison]), 0.05,
            ),
            simultaneous_one_sided_95_lower_nt=simultaneous,
        ))
    end
    return DataFrame(rows)
end

function _err_omni_metric(metrics::DataFrame, method::AbstractString,
        lead::Int, regime::AbstractString)
    rows = metrics[(metrics.method .== method) .&
                   (metrics.model_step_hours .== lead) .&
                   (metrics.regime .== regime), :]
    nrow(rows) == 1 || throw(ArgumentError(
        "M3 OMNI metric key is missing or duplicated",
    ))
    return only(eachrow(rows))
end

function _err_omni_comparison_role(comparator::AbstractString)
    return occursin("dense_driver", comparator) ?
        "retrospective_driver_oracle_control" : "available_comparator"
end

function static_stack_error_omni_decision(metrics::DataFrame,
        bootstrap::DataFrame, driver_decision::DataFrame;
        candidate::AbstractString="m3_sparse_error_state",
        comparators=STATIC_STACK_ERROR_OMNI_COMPARATORS,
        scheduled_issues::Union{Nothing,Integer}=nothing,
        m2_eligible_issues::Union{Nothing,Integer}=nothing)
    rows = NamedTuple[]
    simultaneous_lower = only(unique(
        bootstrap.simultaneous_one_sided_95_lower_nt,
    ))
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H, comparator in comparators
        evidence_rows = bootstrap[
            (bootstrap.comparator .== comparator) .&
            (bootstrap.model_step_hours .== lead), :]
        nrow(evidence_rows) == 1 || throw(ArgumentError(
            "M3 OMNI bootstrap key is missing or duplicated",
        ))
        evidence = only(eachrow(evidence_rows))
        candidate_overall = _err_omni_metric(
            metrics, candidate, lead, "overall",
        )
        comparator_overall = _err_omni_metric(
            metrics, comparator, lead, "overall",
        )
        mae_safe = candidate_overall.mae_nt <= comparator_overall.mae_nt
        regime_safe = true
        for regime in ("target_storm", "active_or_deepening", "recovery")
            candidate_regime = _err_omni_metric(
                metrics, candidate, lead, regime,
            )
            comparator_regime = _err_omni_metric(
                metrics, comparator, lead, regime,
            )
            if candidate_regime.rows >= 20 && comparator_regime.rows >= 20
                regime_safe &= candidate_regime.rmse_nt <= comparator_regime.rmse_nt
            end
        end
        point_pass = evidence.point_improvement_nt >=
                     STATIC_STACK_ERROR_OMNI_MINIMUM_GAIN_NT
        inference_pass = evidence.one_sided_95_lower_nt > 0.0 &&
                         simultaneous_lower > 0.0
        push!(rows, (
            comparator=comparator,
            comparison_role=_err_omni_comparison_role(comparator),
            model_step_hours=lead,
            point_improvement_nt=evidence.point_improvement_nt,
            one_sided_95_lower_nt=evidence.one_sided_95_lower_nt,
            simultaneous_one_sided_95_lower_nt=simultaneous_lower,
            minimum_required_gain_nt=STATIC_STACK_ERROR_OMNI_MINIMUM_GAIN_NT,
            point_margin_pass=point_pass,
            inference_pass=inference_pass,
            mae_safe=mae_safe,
            supported_regimes_safe=regime_safe,
            available_comparator_diagnostic_pass=
                point_pass && inference_pass && mae_safe && regime_safe,
            full_contract_pass=false,
        ))
    end
    by_comparison = DataFrame(rows)
    available_supported = all(
        by_comparison.available_comparator_diagnostic_pass,
    )
    upstream_supported = nrow(driver_decision) == 1 &&
        String(driver_decision.diagnostic_outcome[1]) ==
        "PREDICTABILITY_SUPPORTED_WITHOUT_PREFIX"
    candidate_metrics = metrics[
        (metrics.method .== candidate) .&
        (metrics.regime .== "overall"), :]
    conditional_readiness = minimum(candidate_metrics.readiness)
    eligible_issues = isnothing(m2_eligible_issues) ?
        minimum(candidate_metrics.rows) : Int(m2_eligible_issues)
    scheduled = isnothing(scheduled_issues) ? eligible_issues :
        Int(scheduled_issues)
    0 < eligible_issues <= scheduled || throw(ArgumentError(
        "M3 OMNI readiness counts are inconsistent",
    ))
    minimum(candidate_metrics.rows) == eligible_issues || throw(ArgumentError(
        "M3 OMNI metrics do not match M2-eligible issue count",
    ))
    effective_readiness = conditional_readiness * eligible_issues / scheduled
    summary = DataFrame(
        available_comparator_diagnostic_supported=[available_supported],
        full_retrospective_contract_pass=[false],
        promotion_eligible=[false],
        upstream_sparse_driver_supported=[upstream_supported],
        exact_receipt_causal=[false],
        zero_latency_dst_oracle=[true],
        mixed_source_oracle=[true],
        exposed_point_validation_window=[true],
        event_block_inference_available=[false],
        complete_mandatory_comparator_battery=[false],
        lag_feature_tree_comparator_included=[false],
        sindy_removal_ablation_included=[false],
        prospective_storm_recovery_cohort_complete=[false],
        scheduled_issues=[scheduled],
        m2_eligible_issues=[eligible_issues],
        m2_eligibility=[eligible_issues / scheduled],
        minimum_conditional_readiness_given_driver=[conditional_readiness],
        minimum_effective_scheduled_readiness=[effective_readiness],
        simultaneous_one_sided_95_lower_nt=[simultaneous_lower],
        missing_contract_evidence=[
            "event_ids;historical_sindy_core;causalized_symbolic;physics_gradient_boost;" *
            "PiRC_style_sequence;direct_causal_temporal;correction_without_DRIVER;" *
            "SINDy_removal;matched_driver_baselines;prospective_storm_recovery",
        ],
        diagnostic_outcome=[available_supported ?
            "AVAILABLE_COMPARATOR_SIGNAL_ONLY" :
            "AVAILABLE_COMPARATOR_SIGNAL_NOT_SUPPORTED"],
    )
    return by_comparison, summary
end

function _err_omni_read_final_driver(driver_output_root::AbstractString)
    artifact = read_static_stack_driver(joinpath(
        driver_output_root, "static_stack_driver_omni_sparse_artifact.csv",
    ))
    split = CSV.read(joinpath(
        driver_output_root, "static_stack_driver_omni_split_audit.csv",
    ), DataFrame)
    nrow(split) == 1 || throw(ArgumentError(
        "M3 OMNI driver split audit must have one row",
    ))
    static_stack_driver_sha256(artifact) ==
        String(split.sparse_artifact_sha256[1]) || throw(ArgumentError(
            "M3 OMNI final driver artifact does not match its split audit",
        ))
    return artifact, split
end

function _err_omni_read_dst(cache_root::AbstractString,
        core_output_root::AbstractString)
    manifest = CSV.read(joinpath(
        core_output_root, "static_stack_driver_core_dst_source.csv",
    ), DataFrame)
    nrow(manifest) == 1 || throw(ArgumentError(
        "M3 OMNI Dst source manifest must have one row",
    ))
    source = _drv_core_cached_dst(cache_root)
    source.sha256 == String(manifest.source_sha256[1]) ||
        throw(ArgumentError("M3 OMNI hourly Dst source checksum changed"))
    table = static_stack_driver_core_read_dst(source.path)
    return source, table
end

function _err_omni_readiness_table(errors::DataFrame,
        scheduled_issues::Integer)
    scheduled_issues > 0 || throw(ArgumentError(
        "M3 OMNI scheduled issue count must be positive",
    ))
    candidate = errors[errors.method .== "m3_sparse_error_state", :]
    rows = NamedTuple[]
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        selected = candidate[candidate.model_step_hours .== lead, :]
        push!(rows, (
            model_step_hours=lead,
            evaluation_rows=nrow(selected),
            scheduled_issues=Int(scheduled_issues),
            ready_rows=count(selected.candidate_ready),
            fallback_rows=count(.!selected.candidate_ready),
            conditional_readiness_given_driver=nrow(selected) == 0 ? NaN :
                count(selected.candidate_ready) / nrow(selected),
            effective_scheduled_readiness=
                count(selected.candidate_ready) / scheduled_issues,
            fallback_policy="unchanged_driver_sindy_center",
        ))
    end
    return DataFrame(rows)
end

function run_static_stack_error_omni_diagnostic(;
        driver_output_root=joinpath(@__DIR__, "..", "output", "operational",
            "static_stack_driver_omni_diagnostic"),
        core_output_root=joinpath(@__DIR__, "..", "output", "operational",
            "static_stack_driver_core_omni_diagnostic"),
        output_root=joinpath(@__DIR__, "..", "output", "operational",
            "static_stack_error_omni_diagnostic"),
        cache_root=joinpath(@__DIR__, "..", "output", "operational",
            "source_cache", "static_stack_driver_core_omni"),
        residual_source=STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE)
    mkpath(output_root)
    bins_path = joinpath(
        driver_output_root, "static_stack_driver_omni_half_hour_bins.csv",
    )
    _drv_omni_sha256(bins_path) == STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256 ||
        throw(ArgumentError("M3 OMNI driver-bin checksum changed"))
    bins = CSV.read(bins_path, DataFrame)
    series = static_stack_driver_omni_series(bins)
    final_driver, driver_split = _err_omni_read_final_driver(
        driver_output_root,
    )
    core_errors_path = joinpath(
        core_output_root, "static_stack_driver_core_paired_errors.csv",
    )
    core_errors = static_stack_error_omni_read_core_errors(core_errors_path)
    source = static_stack_error_omni_read_source(residual_source)
    source_index = _err_omni_source_index(source)
    dst_source, dst_table = _err_omni_read_dst(
        cache_root, core_output_root,
    )
    dst_at = Dict(DateTime(row.issue_time_utc) => Float64(row.dst_nt)
                  for row in eachrow(dst_table))
    core = load_operational_core(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)

    folds = _err_omni_fold_driver_artifacts(series)
    fold_manifest = copy(folds.manifest)
    fold_hash_values = String[
        _err_omni_prediction_base_sha256(core, artifact)
        for artifact in folds.artifacts
    ]
    fold_manifest[!, :prediction_base_sha256] = fold_hash_values
    fold_manifest[!, :frozen_core_sha256] = fill(
        static_stack_core_sha256(core), nrow(fold_manifest),
    )
    fold_hashes = Dict(
        STATIC_STACK_DRIVER_OMNI_FOLDS[index].name => fold_hash_values[index]
        for index in eachindex(STATIC_STACK_DRIVER_OMNI_FOLDS)
    )
    crossfit_manifest_sha256 = _err_omni_crossfit_manifest_sha256(
        fold_manifest,
    )
    final_base_center_sha256 = _err_omni_prediction_base_sha256(
        core, final_driver,
    )

    oof_examples = StaticStackM3OmniExample[]
    for (index, fold) in pairs(STATIC_STACK_DRIVER_OMNI_FOLDS)
        records = _err_omni_base_records(
            series, dst_at, source, source_index, folds.artifacts[index], core,
            fold.validation_start - Hour(STATIC_STACK_ERROR_OMNI_HISTORY_HOURS),
            fold.validation_stop,
            fold_hash_values[index],
        )
        examples = _err_omni_examples(
            records, fold.name, fold.validation_start, fold.validation_stop,
            "fit",
        )
        isempty(examples) && throw(ArgumentError(
            "M3 OMNI fold $(fold.name) has no ready examples",
        ))
        append!(oof_examples, examples)
    end
    selection = static_stack_error_omni_select_hyperparameters(
        oof_examples, final_base_center_sha256, fold_hashes,
        crossfit_manifest_sha256,
    )
    final_artifacts = _err_omni_fit_final_artifacts(
        oof_examples, selection, final_base_center_sha256, fold_hashes,
        crossfit_manifest_sha256,
    )

    final_records = _err_omni_base_records(
        series, dst_at, source, source_index, final_driver, core,
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START -
            Hour(STATIC_STACK_ERROR_OMNI_HISTORY_HOURS),
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP,
        final_base_center_sha256,
    )
    final_ready = _err_omni_examples(
        final_records, "exposed_point_validation",
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START,
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP,
        "validation",
    )
    errors = static_stack_error_omni_score(
        final_records, final_ready, final_artifacts, source, core_errors,
        final_base_center_sha256,
    )
    metrics = static_stack_error_omni_metrics(errors)
    bootstrap = static_stack_error_omni_bootstrap(errors)
    driver_decision = CSV.read(joinpath(
        driver_output_root, "static_stack_driver_omni_decision.csv",
    ), DataFrame)
    scheduled_issues = Int(driver_split.scheduled_validation_issues[1])
    eligible_issues = Int(driver_split.eligible_validation_issues[1])
    by_comparison, decision = static_stack_error_omni_decision(
        metrics, bootstrap, driver_decision;
        scheduled_issues=scheduled_issues,
        m2_eligible_issues=eligible_issues,
    )
    readiness = _err_omni_readiness_table(errors, scheduled_issues)

    selected_training = [
        example for example in oof_examples
        if example.fit_row.features.issue_time <=
               STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP -
               Hour(maximum(STATIC_STACK_DRIVER_OMNI_LEADS_H)) &&
           example.fit_row.observation_available_at <=
               STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP
    ]
    isempty(selected_training) && throw(ArgumentError(
        "M3 OMNI final training set is empty",
    ))
    max_training_issue = maximum(
        example.fit_row.features.issue_time for example in selected_training
    )
    max_training_target = maximum(
        example.fit_row.target_time for example in selected_training
    )
    max_training_issue <= DateTime(2017, 10, 13, 16) ||
        throw(ArgumentError("M3 OMNI final whole-anchor embargo failed"))
    max_training_target <= STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP ||
        throw(ArgumentError("M3 OMNI final target-maturity embargo failed"))
    target_gap = _drv_omni_elapsed_hours(
        max_training_target, STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START,
    )
    target_gap >= STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS || throw(ArgumentError(
        "M3 OMNI final target-to-validation gap is below 168 hours",
    ))

    CSV.write(joinpath(output_root, "static_stack_error_omni_fold_manifest.csv"),
              fold_manifest)
    CSV.write(joinpath(output_root, "static_stack_error_omni_driver_selection.csv"),
              folds.selection)
    CSV.write(joinpath(output_root, "static_stack_error_omni_oof_examples.csv"),
              _err_omni_examples_table(oof_examples))
    CSV.write(joinpath(output_root, "static_stack_error_omni_final_ready_examples.csv"),
              _err_omni_examples_table(final_ready))
    CSV.write(joinpath(output_root, "static_stack_error_omni_selection.csv"),
              selection.audit)
    selection_summary = DataFrame([
        (
            model_step_hours=lead,
            correction_selected=selection.selected[lead].use_correction,
            ridge=selection.selected[lead].ridge,
            threshold=selection.selected[lead].threshold,
            reason=selection.selected[lead].reason,
        )
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
    ])
    CSV.write(joinpath(output_root, "static_stack_error_omni_selected.csv"),
              selection_summary)
    artifact_rows = NamedTuple[]
    for artifact in final_artifacts
        path = joinpath(
            output_root,
            "static_stack_error_omni_artifact_h$(artifact.model_step_hours).csv",
        )
        write_static_stack_error_exogenous(path, artifact)
        push!(artifact_rows, (
            model_step_hours=artifact.model_step_hours,
            ridge=artifact.ridge,
            threshold=artifact.threshold,
            active_groups=count(artifact.support_mask),
            fit_rows=artifact.fit_rows,
            base_center_sha256=artifact.base_center_sha256,
            training_manifest_sha256=crossfit_manifest_sha256,
            artifact_sha256=static_stack_error_exogenous_sha256(artifact),
            artifact_path=basename(path),
        ))
    end
    artifact_manifest = DataFrame(artifact_rows)
    CSV.write(joinpath(output_root, "static_stack_error_omni_artifact_manifest.csv"),
              artifact_manifest)
    CSV.write(joinpath(output_root, "static_stack_error_omni_paired_errors.csv"), errors)
    CSV.write(joinpath(output_root, "static_stack_error_omni_metrics.csv"), metrics)
    CSV.write(joinpath(output_root, "static_stack_error_omni_bootstrap.csv"), bootstrap)
    CSV.write(joinpath(output_root, "static_stack_error_omni_decision_by_comparison.csv"),
              by_comparison)
    CSV.write(joinpath(output_root, "static_stack_error_omni_decision.csv"), decision)
    CSV.write(joinpath(output_root, "static_stack_error_omni_readiness.csv"), readiness)
    source_manifest = DataFrame(
        residual_source_sha256=[STATIC_STACK_ERROR_OMNI_RESIDUAL_SOURCE_SHA256],
        residual_point_rows=[nrow(source)],
        residual_fit_rows=[count(==("fit"), source.split_label)],
        residual_validation_rows=[count(==("validation"), source.split_label)],
        residual_calibration_values_materialized=[0],
        residual_calibration_boundary_rows_scanned=[1],
        m2_scheduled_validation_issues=[scheduled_issues],
        m2_eligible_validation_issues=[eligible_issues],
        driver_bins_sha256=[STATIC_STACK_ERROR_OMNI_DRIVER_BINS_SHA256],
        core_errors_sha256=[STATIC_STACK_ERROR_OMNI_CORE_ERRORS_SHA256],
        hourly_dst_sha256=[dst_source.sha256],
        frozen_core_sha256=[static_stack_core_sha256(core)],
        final_driver_sha256=[static_stack_driver_sha256(final_driver)],
        final_base_center_sha256=[final_base_center_sha256],
        crossfit_manifest_sha256=[crossfit_manifest_sha256],
        oof_examples_sha256=[_drv_omni_sha256(joinpath(
            output_root, "static_stack_error_omni_oof_examples.csv",
        ))],
        final_ready_examples_sha256=[_drv_omni_sha256(joinpath(
            output_root, "static_stack_error_omni_final_ready_examples.csv",
        ))],
        paired_errors_sha256=[_drv_omni_sha256(joinpath(
            output_root, "static_stack_error_omni_paired_errors.csv",
        ))],
        post_2022_rows_read=[0],
        source_role=["mixed_source_omni_hro2_zero_latency_dst_oracle"],
    )
    CSV.write(joinpath(output_root, "static_stack_error_omni_source_manifest.csv"),
              source_manifest)
    split_audit = DataFrame(
        final_driver_train_actual_max_utc=[driver_split.final_train_actual_max_utc[1]],
        final_error_whole_anchor_max_issue_utc=[max_training_issue],
        final_error_target_availability_max_utc=[max_training_target],
        final_validation_start_utc=[STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START],
        final_validation_stop_utc=[STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP],
        target_to_validation_gap_hours=[target_gap],
        frozen_warmup_hours=[STATIC_STACK_ERROR_OMNI_HISTORY_HOURS],
        oof_examples=[length(oof_examples)],
        final_evaluation_rows=[count(errors.method .==
                                     "m3_sparse_error_state")],
        final_ready_rows=[count(errors.candidate_ready[errors.method .==
                                                       "m3_sparse_error_state"])],
        calibration_rows_used=[0],
        exposed_2023_2025_rows_read=[0],
        point_window_previously_exposed=[true],
        event_block_inference_available=[false],
        promotion_eligible=[false],
    )
    CSV.write(joinpath(output_root, "static_stack_error_omni_split_audit.csv"),
              split_audit)
    return (
        decision=decision,
        by_comparison=by_comparison,
        metrics=metrics,
        bootstrap=bootstrap,
        selection=selection.audit,
        readiness=readiness,
        artifacts=artifact_manifest,
        split_audit=split_audit,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_stack_error_omni_diagnostic()
    println("static stack-M3 OMNI diagnostic: ",
            result.decision.diagnostic_outcome[1])
end
