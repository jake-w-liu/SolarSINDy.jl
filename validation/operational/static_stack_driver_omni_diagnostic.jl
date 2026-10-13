using CSV
using DataFrames
using Dates
using Downloads
using Random
using SHA
using SolarSINDy
using Statistics

const STATIC_STACK_DRIVER_OMNI_DATASET = "OMNI_HRO2_1MIN"
const STATIC_STACK_DRIVER_OMNI_PARAMETERS = (
    "Time",
    "percent_interp",
    "Timeshift",
    "Time_btwn_obs",
    "BX_GSE",
    "BY_GSM",
    "BZ_GSM",
    "flow_speed",
    "Vx",
    "proton_density",
)
const STATIC_STACK_DRIVER_OMNI_COLUMNS = (
    :time_utc,
    :percent_interpolated,
    :timeshift_seconds,
    :time_between_observations_seconds,
    :bx_gse_nt,
    :by_gsm_nt,
    :bz_gsm_nt,
    :speed_km_s,
    :vx_gse_km_s,
    :density_cm3,
)
const STATIC_STACK_DRIVER_OMNI_MIN_JOINT_MINUTES = 15
const STATIC_STACK_DRIVER_OMNI_LEADS_H = (1, 2, 3, 4, 6, 7)
const STATIC_STACK_DRIVER_OMNI_BOOTSTRAP_SEED = 22_022_026
const STATIC_STACK_DRIVER_OMNI_BOOTSTRAP_DRAWS = 10_000
const STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS = 168
const STATIC_STACK_DRIVER_OMNI_SELECTION_STRIDE_HOURS = 6
const STATIC_STACK_DRIVER_OMNI_GRID_START = DateTime(2010, 1, 1)
const STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP = DateTime(2017, 10, 13, 23)
const STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START = DateTime(2017, 10, 20, 23)
const STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP = DateTime(2020, 5, 22, 19)
const STATIC_STACK_DRIVER_OMNI_FOLDS = (
    (name="calendar_2013", train_stop=DateTime(2013, 1, 1),
     validation_start=DateTime(2013, 1, 8),
     validation_stop=DateTime(2013, 12, 24, 23)),
    (name="calendar_2014", train_stop=DateTime(2014, 1, 1),
     validation_start=DateTime(2014, 1, 8),
     validation_stop=DateTime(2014, 12, 24, 23)),
    (name="calendar_2015", train_stop=DateTime(2015, 1, 1),
     validation_start=DateTime(2015, 1, 8),
     validation_stop=DateTime(2015, 12, 24, 23)),
    (name="calendar_2016", train_stop=DateTime(2016, 1, 1),
     validation_start=DateTime(2016, 1, 8),
     validation_stop=DateTime(2016, 12, 24, 23)),
    (name="partial_2017", train_stop=DateTime(2017, 1, 1),
     validation_start=DateTime(2017, 1, 8),
     validation_stop=DateTime(2017, 10, 13, 16)),
)

struct StaticStackM2OmniSeries
    times::Vector{DateTime}
    states::Matrix{Float64}
    joint_minutes::Vector{Int}
end

struct StaticStackM2OmniCase
    issue_time::DateTime
    anchor_index::Int
end

function _drv_omni_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function _drv_omni_parse_time(value)
    text = String(value)
    occursin(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", text) ||
        throw(ArgumentError("invalid OMNI HAPI timestamp: $text"))
    return DateTime(text[1:19], dateformat"yyyy-mm-ddTHH:MM:SS")
end

function _drv_omni_month_key(time::DateTime)
    return Dates.format(Date(time), dateformat"yyyymm")
end

function _drv_omni_hapi_url(month_start::DateTime)
    month_stop = month_start + Month(1)
    minimum = Dates.format(month_start, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    maximum = Dates.format(month_stop, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    parameters = join(STATIC_STACK_DRIVER_OMNI_PARAMETERS, ',')
    return "https://cdaweb.gsfc.nasa.gov/hapi/data" *
           "?id=$(STATIC_STACK_DRIVER_OMNI_DATASET)&parameters=$parameters" *
           "&time.min=$minimum&time.max=$maximum&format=csv"
end

function _drv_omni_validate_source_object(path::AbstractString,
        key::AbstractString)
    filesize(path) > 0 || throw(ArgumentError(
        "OMNI HAPI returned an empty source object for month $key",
    ))
    first_record = open(readline, path)
    occursin(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z,",
             first_record) || throw(ArgumentError(
        "OMNI HAPI did not return CSV data for month $key",
    ))
    return true
end

function _drv_omni_cached_month(month_start::DateTime,
        cache_root::AbstractString)
    mkpath(cache_root)
    islink(cache_root) && throw(ArgumentError(
        "OMNI diagnostic cache root must not be a symbolic link",
    ))
    key = _drv_omni_month_key(month_start)
    prefix = "omni_hro2_1min_$(key)_"
    matches = filter(name -> startswith(name, prefix) && endswith(name, ".csv"),
                     readdir(cache_root))
    length(matches) <= 1 || throw(ArgumentError(
        "multiple pinned OMNI source objects exist for month $key",
    ))
    if length(matches) == 1
        path = joinpath(cache_root, only(matches))
        expected = splitext(only(matches))[1][(lastindex(prefix) + 1):end]
        occursin(r"^[0-9a-f]{64}$", expected) || throw(ArgumentError(
            "cached OMNI source filename does not contain a SHA-256 digest",
        ))
        _drv_omni_sha256(path) == expected || throw(ArgumentError(
            "cached OMNI source object failed its filename checksum",
        ))
        _drv_omni_validate_source_object(path, key)
        return (path=path, url=_drv_omni_hapi_url(month_start),
                sha256=expected, reused=true)
    end
    url = _drv_omni_hapi_url(month_start)
    return mktempdir(cache_root) do temporary_root
        temporary = joinpath(temporary_root, "download.csv")
        Downloads.download(url, temporary)
        _drv_omni_validate_source_object(temporary, key)
        checksum = _drv_omni_sha256(temporary)
        destination = joinpath(cache_root, prefix * checksum * ".csv")
        ispath(destination) && throw(ArgumentError(
            "OMNI cache destination already exists unexpectedly",
        ))
        mv(temporary, destination)
        return (path=destination, url=url, sha256=checksum, reused=false)
    end
end

function _drv_omni_read_month(path::AbstractString)
    return CSV.read(path, DataFrame;
        header=collect(STATIC_STACK_DRIVER_OMNI_COLUMNS),
        types=Dict(
            :time_utc => String,
            :bx_gse_nt => Float64,
            :by_gsm_nt => Float64,
            :bz_gsm_nt => Float64,
            :speed_km_s => Float64,
            :vx_gse_km_s => Float64,
            :density_cm3 => Float64,
            :percent_interpolated => Int,
            :timeshift_seconds => Int,
            :time_between_observations_seconds => Int,
        ),
        strict=true,
        silencewarnings=false,
    )
end

function _drv_omni_valid_joint(raw::DataFrame, row::Int)
    bx = raw.bx_gse_nt[row]
    by = raw.by_gsm_nt[row]
    bz = raw.bz_gsm_nt[row]
    speed = raw.speed_km_s[row]
    vx = raw.vx_gse_km_s[row]
    density = raw.density_cm3[row]
    return all(isfinite, (bx, by, bz, speed, vx, density)) &&
           abs(bx) < 9999.99 && abs(by) < 9999.99 && abs(bz) < 9999.99 &&
           0.0 < speed < 99999.9 && -99999.9 < vx < 0.0 &&
           0.0 < density < 999.99
end

"Aggregate one complete HAPI month to half-open 30-minute state bins."
function static_stack_driver_omni_aggregate_month(raw::DataFrame;
        minimum_joint_minutes::Integer=STATIC_STACK_DRIVER_OMNI_MIN_JOINT_MINUTES)
    minimum_joint_minutes isa Bool && throw(ArgumentError(
        "minimum_joint_minutes must be an integer",
    ))
    1 <= minimum_joint_minutes <= 30 || throw(ArgumentError(
        "minimum_joint_minutes must lie in 1:30",
    ))
    Tuple(Symbol.(names(raw))) == STATIC_STACK_DRIVER_OMNI_COLUMNS || throw(ArgumentError(
        "OMNI HAPI columns do not match the frozen diagnostic schema",
    ))
    nrow(raw) > 0 && nrow(raw) % 30 == 0 || throw(ArgumentError(
        "OMNI HAPI month must contain a positive whole number of 30-minute bins",
    ))
    times = _drv_omni_parse_time.(raw.time_utc)
    all(index -> times[index] == times[1] + Minute(index - 1), eachindex(times)) ||
        throw(ArgumentError("OMNI HAPI month is not a complete one-minute grid"))
    minute(times[1]) == 0 && hour(times[1]) == 0 || throw(ArgumentError(
        "OMNI HAPI month must begin at 00:00 UTC",
    ))
    rows = NamedTuple[]
    for first_row in 1:30:nrow(raw)
        selected = [row for row in first_row:(first_row + 29)
                    if _drv_omni_valid_joint(raw, row)]
        length(selected) >= minimum_joint_minutes || continue
        speed = median(raw.speed_km_s[selected])
        density = median(raw.density_cm3[selected])
        interpolation = [raw.percent_interpolated[row] for row in selected
                         if 0 <= raw.percent_interpolated[row] < 999]
        timeshifts = [raw.timeshift_seconds[row] for row in selected
                      if abs(raw.timeshift_seconds[row]) < 999999]
        spacings = [raw.time_between_observations_seconds[row] for row in selected
                    if 0 <= raw.time_between_observations_seconds[row] < 999999]
        push!(rows, (
            bin_end_utc=times[first_row] + Minute(30),
            Bx_nt=median(raw.bx_gse_nt[selected]),
            By_nt=median(raw.by_gsm_nt[selected]),
            Bz_nt=median(raw.bz_gsm_nt[selected]),
            logV=log(speed),
            logn=log(density),
            joint_minutes=length(selected),
            percent_interpolated_median=isempty(interpolation) ? NaN :
                median(Float64.(interpolation)),
            timeshift_seconds_median=isempty(timeshifts) ? NaN :
                median(Float64.(timeshifts)),
            observation_spacing_seconds_median=isempty(spacings) ? NaN :
                median(Float64.(spacings)),
        ))
    end
    return DataFrame(rows)
end

function static_stack_driver_omni_series(bins::DataFrame)
    required = ["bin_end_utc", "Bx_nt", "By_nt", "Bz_nt", "logV", "logn",
                "joint_minutes"]
    all(in(names(bins)), required) || throw(ArgumentError(
        "aggregated OMNI bins are missing a required column",
    ))
    order = sortperm(bins.bin_end_utc)
    times = DateTime.(bins.bin_end_utc[order])
    length(unique(times)) == length(times) || throw(ArgumentError(
        "aggregated OMNI bins contain duplicate end times",
    ))
    states = Matrix{Float64}(bins[order, [:Bx_nt, :By_nt, :Bz_nt, :logV, :logn]])
    all(isfinite, states) || throw(ArgumentError(
        "aggregated OMNI state contains a non-finite value",
    ))
    all(value -> 15 <= value <= 30, bins.joint_minutes[order]) ||
        throw(ArgumentError("aggregated OMNI joint-minute count is invalid"))
    all(index -> times[index] > times[index - 1] &&
                 Dates.value(times[index] - times[index - 1]) %
                    Dates.value(Minute(30)) == 0,
        2:length(times)) || throw(ArgumentError(
            "aggregated OMNI times do not lie on one 30-minute grid",
        ))
    return StaticStackM2OmniSeries(times, states, Int.(bins.joint_minutes[order]))
end

function _drv_omni_contiguous_ranges(times::Vector{DateTime}, indices)
    isempty(indices) && return UnitRange{Int}[]
    ranges = UnitRange{Int}[]
    first_index = first(indices)
    previous = first_index
    for index in Iterators.drop(indices, 1)
        if index != previous + 1 || times[index] != times[previous] + Minute(30)
            push!(ranges, first_index:previous)
            first_index = index
        end
        previous = index
    end
    push!(ranges, first_index:previous)
    return ranges
end

function static_stack_driver_omni_training_segments(series::StaticStackM2OmniSeries,
        train_stop::DateTime; minimum_rows::Integer=26)
    minimum_rows >= 26 || throw(ArgumentError(
        "M2 training segments require at least 26 rows",
    ))
    indices = findall(<=(train_stop), series.times)
    ranges = _drv_omni_contiguous_ranges(series.times, indices)
    retained = filter(range -> length(range) >= minimum_rows, ranges)
    segments = [Matrix(@view series.states[range, :]) for range in retained]
    isempty(segments) && throw(ArgumentError(
        "no complete M2 training segment exists before the fold cutoff",
    ))
    return (
        segments=segments,
        retained_segments=length(retained),
        discarded_segments=length(ranges) - length(retained),
        retained_rows=sum(length, retained),
        discarded_rows=sum(length, ranges) - sum(length, retained),
        max_time=maximum(series.times[last(range)] for range in retained),
    )
end

function static_stack_driver_omni_cases(series::StaticStackM2OmniSeries,
        issue_start::DateTime, issue_stop::DateTime;
        stride_hours::Integer=1)
    issue_start <= issue_stop || throw(ArgumentError(
        "M2 diagnostic issue window is inverted",
    ))
    minute(issue_start) == 0 && second(issue_start) == 0 &&
        minute(issue_stop) == 0 && second(issue_stop) == 0 ||
        throw(ArgumentError("M2 diagnostic issues must lie on the UTC hour"))
    stride_hours isa Bool && throw(ArgumentError(
        "M2 diagnostic issue stride must be an integer",
    ))
    stride_hours >= 1 || throw(ArgumentError(
        "M2 diagnostic issue stride must be positive",
    ))
    index_at = Dict(time => index for (index, time) in pairs(series.times))
    cases = StaticStackM2OmniCase[]
    scheduled = 0
    for issue in issue_start:Hour(stride_hours):issue_stop
        scheduled += 1
        anchor = get(index_at, issue, 0)
        anchor == 0 && continue
        anchor >= 25 && anchor + 14 <= length(series.times) || continue
        series.times[anchor - 24] == issue - Hour(12) || continue
        series.times[anchor + 14] == issue + Hour(7) || continue
        all(offset -> series.times[anchor + offset] == issue + Minute(30 * offset),
            -24:14) || continue
        push!(cases, StaticStackM2OmniCase(issue, anchor))
    end
    return (cases=cases, scheduled=scheduled, eligible=length(cases),
            availability=length(cases) / scheduled)
end

function _drv_omni_history(series::StaticStackM2OmniSeries, case::StaticStackM2OmniCase)
    return Matrix(@view series.states[(case.anchor_index - 24):case.anchor_index, :])
end

function _drv_omni_target(series::StaticStackM2OmniSeries, case::StaticStackM2OmniCase)
    return Matrix(@view series.states[(case.anchor_index + 1):(case.anchor_index + 14), :])
end

function _drv_omni_prediction(method::Symbol,
        artifact::Union{Nothing,StaticStackDriverArtifact},
        series::StaticStackM2OmniSeries, case::StaticStackM2OmniCase,
        known_prefix_steps::Int)
    0 <= known_prefix_steps <= 14 || throw(ArgumentError(
        "known-prefix length must lie in 0:14",
    ))
    history = _drv_omni_history(series, case)
    target = _drv_omni_target(series, case)
    known_prefix_steps == 14 && return target
    augmented = known_prefix_steps == 0 ? history :
        vcat(history, target[1:known_prefix_steps, :])
    current_history = augmented[(end - 24):end, :]
    remaining = 14 - known_prefix_steps
    suffix = if method == :persistence
        repeat(current_history[end:end, :], remaining, 1)
    elseif method in (:sparse, :dense)
        artifact === nothing && throw(ArgumentError(
            "driver artifact is required for sparse or dense prediction",
        ))
        static_stack_driver_rollout(artifact, current_history)[1:remaining, :]
    else
        throw(ArgumentError("unknown M2 diagnostic method: $method"))
    end
    return known_prefix_steps == 0 ? suffix :
        vcat(target[1:known_prefix_steps, :], suffix)
end

function _drv_omni_trial_artifact(segments, ridge::Float64,
        threshold::Float64, label::String)
    converted = [SolarSINDy._static_stack_driver_matrix(segment)
                 for segment in segments]
    return SolarSINDy._static_stack_driver_fit_segments(
        converted, ridge, threshold, label;
        verify_training_rollouts=false,
    )
end

function _drv_omni_standardized_loss(artifact::StaticStackDriverArtifact,
        series::StaticStackM2OmniSeries, cases::Vector{StaticStackM2OmniCase})
    isempty(cases) && throw(ArgumentError(
        "M2 diagnostic loss requires at least one complete issue",
    ))
    scale = reshape(collect(artifact.scale), 1, :)
    squared_error = 0.0
    count_values = 0
    for case in cases
        prediction = _drv_omni_prediction(:sparse, artifact, series, case, 0)
        target = _drv_omni_target(series, case)
        squared_error += sum(abs2, (prediction .- target) ./ scale)
        count_values += length(target)
    end
    return sqrt(squared_error / count_values)
end

function _drv_omni_expected_grid_failure(error)
    message = lowercase(sprint(showerror, error))
    return (error isa ArgumentError && occursin("unstable", message)) ||
           (error isa ErrorException &&
            occursin("thresholding did not converge", message))
end

function _drv_omni_one_se_index(audit::DataFrame, indices)
    isempty(indices) && throw(ArgumentError(
        "one-standard-error selection requires at least one candidate",
    ))
    best = indices[argmin(audit.mean_fourteen_step_standardized_rmse[indices])]
    limit = audit.mean_fourteen_step_standardized_rmse[best] +
            audit.fold_standard_error[best]
    eligible = filter(index ->
        audit.mean_fourteen_step_standardized_rmse[index] <= limit,
        collect(indices),
    )
    sort!(eligible; by=index -> (
        audit.max_active_groups[index],
        -audit.ridge[index],
        -audit.threshold[index],
        audit.mean_fourteen_step_standardized_rmse[index],
    ))
    return first(eligible), Set(eligible)
end

function static_stack_driver_omni_select_hyperparameters(series::StaticStackM2OmniSeries;
        folds=STATIC_STACK_DRIVER_OMNI_FOLDS,
        ridge_grid=STATIC_STACK_DRIVER_RIDGE_GRID,
        threshold_grid=STATIC_STACK_DRIVER_THRESHOLD_GRID)
    fold_inputs = map(folds) do fold
        training = static_stack_driver_omni_training_segments(series, fold.train_stop)
        validation = static_stack_driver_omni_cases(
            series, fold.validation_start, fold.validation_stop;
            stride_hours=STATIC_STACK_DRIVER_OMNI_SELECTION_STRIDE_HOURS,
        )
        validation.eligible > 0 || throw(ArgumentError(
            "fold $(fold.name) has no complete validation issues",
        ))
        training.max_time + Hour(STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS) <=
            fold.validation_start || throw(ArgumentError(
                "fold $(fold.name) violates the 168-hour embargo",
            ))
        (fold=fold, training=training, validation=validation)
    end
    rows = NamedTuple[]
    for ridge in ridge_grid, threshold in threshold_grid
        fold_losses = Float64[]
        active_groups = Int[]
        failure = ""
        for input in fold_inputs
            artifact = try
                _drv_omni_trial_artifact(
                    input.training.segments, Float64(ridge), Float64(threshold),
                    "omni-diagnostic-$(input.fold.name)",
                )
            catch error
                _drv_omni_expected_grid_failure(error) || rethrow()
                failure = sprint(showerror, error)
                break
            end
            push!(fold_losses, _drv_omni_standardized_loss(
                artifact, series, input.validation.cases,
            ))
            push!(active_groups, count(static_stack_driver_support(artifact)))
        end
        valid = length(fold_losses) == length(fold_inputs)
        push!(rows, (
            ridge=Float64(ridge),
            threshold=Float64(threshold),
            valid=valid,
            mean_fourteen_step_standardized_rmse=valid ? mean(fold_losses) : NaN,
            fold_standard_error=valid && length(fold_losses) > 1 ?
                std(fold_losses) / sqrt(length(fold_losses)) : 0.0,
            max_active_groups=valid ? maximum(active_groups) : -1,
            fold_losses=valid ? join(string.(fold_losses), ';') : "",
            failure_reason=failure,
            one_standard_error_eligible=false,
            selected_sparse=false,
            selected_dense=false,
        ))
    end
    audit = DataFrame(rows)
    valid_indices = findall(audit.valid)
    isempty(valid_indices) && throw(ArgumentError(
        "no stable M2 grid candidate completed every chronological fold",
    ))
    sparse_index, sparse_eligible =
        _drv_omni_one_se_index(audit, valid_indices)
    dense_indices = filter(index -> audit.threshold[index] == 0.0, valid_indices)
    isempty(dense_indices) && throw(ArgumentError(
        "no stable dense-ridge comparator completed every fold",
    ))
    dense_index, _ = _drv_omni_one_se_index(audit, dense_indices)
    audit.one_standard_error_eligible .= [index in sparse_eligible
                                          for index in 1:nrow(audit)]
    audit.selected_sparse[sparse_index] = true
    audit.selected_dense[dense_index] = true
    return (
        audit=audit,
        sparse=(ridge=audit.ridge[sparse_index],
                threshold=audit.threshold[sparse_index]),
        dense=(ridge=audit.ridge[dense_index], threshold=0.0),
    )
end

function _drv_omni_ey(state_row)
    return 1.0e-3 * exp(state_row[4]) * max(0.0, -state_row[3])
end

function static_stack_driver_omni_score(series::StaticStackM2OmniSeries,
        cases::Vector{StaticStackM2OmniCase},
        sparse::StaticStackDriverArtifact,
        dense::StaticStackDriverArtifact;
        known_prefix_steps=(0, 1))
    error_rows = NamedTuple[]
    for prefix in known_prefix_steps, case in cases
        target = _drv_omni_target(series, case)
        for (method, artifact) in ((:sparse, sparse), (:dense, dense),
                                   (:persistence, nothing))
            prediction = _drv_omni_prediction(
                method, artifact, series, case, Int(prefix),
            )
            for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
                step = 2 * lead
                truth_bz = target[step, 3]
                pred_bz = prediction[step, 3]
                truth_ey = _drv_omni_ey(@view target[step, :])
                pred_ey = _drv_omni_ey(@view prediction[step, :])
                standardized = (prediction[step, :] .- target[step, :]) ./
                               collect(sparse.scale)
                push!(error_rows, (
                    issue_time_utc=case.issue_time,
                    known_prefix_steps=Int(prefix),
                    method=String(method),
                    lead_hours=lead,
                    bz_squared_error_nt2=(pred_bz - truth_bz)^2,
                    bz_absolute_error_nt=abs(pred_bz - truth_bz),
                    southward_bz_squared_error_nt2=truth_bz < 0.0 ?
                        (pred_bz - truth_bz)^2 : NaN,
                    ey_squared_error_mvm2=(pred_ey - truth_ey)^2,
                    ey_absolute_error_mvm=abs(pred_ey - truth_ey),
                    standardized_state_squared_error=sum(abs2, standardized) / 5,
                    observed_southward_event=truth_bz < -3.0,
                    predicted_southward_event=pred_bz < -3.0,
                ))
            end
        end
    end
    errors = DataFrame(error_rows)
    metric_rows = NamedTuple[]
    for group in groupby(errors, [:known_prefix_steps, :method, :lead_hours])
        south = filter(isfinite, group.southward_bz_squared_error_nt2)
        tp = count(group.observed_southward_event .& group.predicted_southward_event)
        fp = count(.!group.observed_southward_event .& group.predicted_southward_event)
        fn = count(group.observed_southward_event .& .!group.predicted_southward_event)
        precision = tp + fp == 0 ? NaN : tp / (tp + fp)
        recall = tp + fn == 0 ? NaN : tp / (tp + fn)
        f1 = isfinite(precision) && isfinite(recall) && precision + recall > 0 ?
            2 * precision * recall / (precision + recall) : NaN
        push!(metric_rows, (
            known_prefix_steps=group.known_prefix_steps[1],
            method=group.method[1],
            lead_hours=group.lead_hours[1],
            rows=nrow(group),
            bz_rmse_nt=sqrt(mean(group.bz_squared_error_nt2)),
            bz_mae_nt=mean(group.bz_absolute_error_nt),
            southward_bz_rmse_nt=isempty(south) ? NaN : sqrt(mean(south)),
            ey_rmse_mvm=sqrt(mean(group.ey_squared_error_mvm2)),
            ey_mae_mvm=mean(group.ey_absolute_error_mvm),
            standardized_state_rmse=sqrt(mean(group.standardized_state_squared_error)),
            southward_precision=precision,
            southward_recall=recall,
            southward_f1=f1,
            southward_brier=mean((Float64.(group.predicted_southward_event) .-
                                  Float64.(group.observed_southward_event)).^2),
        ))
    end
    return errors, DataFrame(metric_rows)
end

function _drv_omni_block_id(time::DateTime)
    elapsed_ms = Dates.value(time - STATIC_STACK_DRIVER_OMNI_GRID_START)
    block_ms = Dates.value(convert(Millisecond, Hour(168)))
    return fld(elapsed_ms, block_ms)
end

function _drv_omni_elapsed_hours(start::DateTime, stop::DateTime)
    return Dates.value(stop - start) /
           Dates.value(convert(Millisecond, Hour(1)))
end

function static_stack_driver_omni_bootstrap(errors::DataFrame;
        draws::Integer=STATIC_STACK_DRIVER_OMNI_BOOTSTRAP_DRAWS,
        seed::Integer=STATIC_STACK_DRIVER_OMNI_BOOTSTRAP_SEED)
    draws >= 100 || throw(ArgumentError(
        "M2 diagnostic bootstrap requires at least 100 draws",
    ))
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for prefix in sort(unique(errors.known_prefix_steps)),
            lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
            comparator in ("persistence", "dense"),
            (metric, column) in (("bz_rmse", :bz_squared_error_nt2),
                                 ("ey_rmse", :ey_squared_error_mvm2))
        sparse_rows = errors[(errors.known_prefix_steps .== prefix) .&
                             (errors.lead_hours .== lead) .&
                             (errors.method .== "sparse"),
                             [:issue_time_utc, column]]
        comparator_rows = errors[(errors.known_prefix_steps .== prefix) .&
                                 (errors.lead_hours .== lead) .&
                                 (errors.method .== comparator),
                                 [:issue_time_utc, column]]
        sparse_rows.issue_time_utc == comparator_rows.issue_time_utc ||
            throw(ArgumentError("M2 bootstrap rows are not exactly paired"))
        blocks = _drv_omni_block_id.(sparse_rows.issue_time_utc)
        unique_blocks = sort(unique(blocks))
        length(unique_blocks) >= 2 || throw(ArgumentError(
            "M2 bootstrap requires at least two 168-hour blocks",
        ))
        sparse_sums = Dict(block => 0.0 for block in unique_blocks)
        comparator_sums = Dict(block => 0.0 for block in unique_blocks)
        counts = Dict(block => 0 for block in unique_blocks)
        for index in eachindex(blocks)
            block = blocks[index]
            sparse_sums[block] += sparse_rows[index, column]
            comparator_sums[block] += comparator_rows[index, column]
            counts[block] += 1
        end
        differences = Vector{Float64}(undef, draws)
        for draw in 1:draws
            sampled = rand(rng, unique_blocks, length(unique_blocks))
            n = sum(counts[block] for block in sampled)
            sparse_rmse = sqrt(sum(sparse_sums[block] for block in sampled) / n)
            comparator_rmse = sqrt(
                sum(comparator_sums[block] for block in sampled) / n,
            )
            differences[draw] = comparator_rmse - sparse_rmse
        end
        point = sqrt(mean(comparator_rows[!, column])) -
                sqrt(mean(sparse_rows[!, column]))
        push!(rows, (
            known_prefix_steps=prefix,
            lead_hours=lead,
            metric=metric,
            comparator=comparator,
            rows=nrow(sparse_rows),
            blocks=length(unique_blocks),
            draws=draws,
            seed=seed,
            point_improvement=point,
            one_sided_95_lower=quantile(differences, 0.05),
        ))
    end
    return DataFrame(rows)
end

function static_stack_driver_omni_decision(metrics::DataFrame, bootstrap::DataFrame)
    rows = NamedTuple[]
    for prefix in (0, 1), lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        sparse = only(eachrow(metrics[(metrics.known_prefix_steps .== prefix) .&
                                      (metrics.lead_hours .== lead) .&
                                      (metrics.method .== "sparse"), :]))
        passes = true
        for comparator in ("persistence", "dense")
            reference = only(eachrow(metrics[
                (metrics.known_prefix_steps .== prefix) .&
                (metrics.lead_hours .== lead) .&
                (metrics.method .== comparator), :]))
            for (metric, sparse_value, comparator_value) in (
                    ("bz_rmse", sparse.bz_rmse_nt, reference.bz_rmse_nt),
                    ("ey_rmse", sparse.ey_rmse_mvm, reference.ey_rmse_mvm))
                evidence = only(eachrow(bootstrap[
                    (bootstrap.known_prefix_steps .== prefix) .&
                    (bootstrap.lead_hours .== lead) .&
                    (bootstrap.metric .== metric) .&
                    (bootstrap.comparator .== comparator), :]))
                passes &= sparse_value < comparator_value &&
                          evidence.one_sided_95_lower > 0.0
            end
        end
        push!(rows, (known_prefix_steps=prefix, lead_hours=lead,
                     driver_gate_pass=passes))
    end
    by_lead = DataFrame(rows)
    no_prefix = all(by_lead.driver_gate_pass[by_lead.known_prefix_steps .== 0])
    one_step = all(by_lead.driver_gate_pass[by_lead.known_prefix_steps .== 1])
    outcome = no_prefix ? "PREDICTABILITY_SUPPORTED_WITHOUT_PREFIX" :
              one_step ? "PREDICTABILITY_SUPPORTED_ONLY_WITH_ORACLE_PREFIX" :
              "PREDICTABILITY_NOT_SUPPORTED"
    summary = DataFrame(
        diagnostic_outcome=[outcome],
        no_prefix_all_leads_pass=[no_prefix],
        one_step_oracle_prefix_all_leads_pass=[one_step],
        promotion_eligible=[false],
        receipt_causal=[false],
        source_role=["retrospectively_shifted_near_earth_diagnostic_only"],
    )
    return by_lead, summary
end

function run_static_stack_driver_omni_diagnostic(; output_root=joinpath(@__DIR__, "..", "output",
        "operational", "static_stack_driver_omni_diagnostic"),
        cache_root=joinpath(@__DIR__, "..", "output", "operational",
            "source_cache", "static_stack_driver_omni"))
    mkpath(output_root)
    month = DateTime(2010, 1, 1)
    final_month = DateTime(2020, 5, 1)
    bin_tables = DataFrame[]
    source_rows = NamedTuple[]
    while month <= final_month
        source = _drv_omni_cached_month(month, cache_root)
        raw = _drv_omni_read_month(source.path)
        bins = static_stack_driver_omni_aggregate_month(raw)
        push!(bin_tables, bins)
        push!(source_rows, (
            month_utc=Dates.format(month, dateformat"yyyy-mm"),
            source_url=source.url,
            source_path=relpath(source.path, cache_root),
            source_sha256=source.sha256,
            source_bytes=filesize(source.path),
            reused=source.reused,
            source_rows=nrow(raw),
            retained_half_hour_bins=nrow(bins),
        ))
        month += Month(1)
    end
    source_manifest = DataFrame(source_rows)
    bins = reduce((left, right) -> vcat(left, right; cols=:setequal), bin_tables)
    filter!(row -> row.bin_end_utc <= STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP + Hour(7),
            bins)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_source_manifest.csv"),
              source_manifest)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_half_hour_bins.csv"), bins)
    series = static_stack_driver_omni_series(bins)
    selection = static_stack_driver_omni_select_hyperparameters(series)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_selection.csv"),
              selection.audit)

    training = static_stack_driver_omni_training_segments(
        series, STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP,
    )
    training.max_time + Hour(STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS) <=
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START || throw(ArgumentError(
            "final M2 diagnostic split violates the 168-hour embargo",
        ))
    sparse = _drv_omni_trial_artifact(
        training.segments, selection.sparse.ridge,
        selection.sparse.threshold,
        "static-stack-m2-omni-diagnostic-sparse",
    )
    dense = _drv_omni_trial_artifact(
        training.segments, selection.dense.ridge, 0.0,
        "static-stack-m2-omni-diagnostic-dense",
    )
    validation = static_stack_driver_omni_cases(
        series, STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START,
        STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP,
    )
    validation.eligible > 0 || throw(ArgumentError(
        "final M2 diagnostic window contains no complete issue",
    ))
    errors, metrics = static_stack_driver_omni_score(
        series, validation.cases, sparse, dense,
    )
    bootstrap = static_stack_driver_omni_bootstrap(errors)
    by_lead, summary = static_stack_driver_omni_decision(metrics, bootstrap)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_paired_errors.csv"), errors)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_metrics.csv"), metrics)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_bootstrap.csv"), bootstrap)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_decision_by_lead.csv"), by_lead)
    CSV.write(joinpath(output_root, "static_stack_driver_omni_decision.csv"), summary)
    write_static_stack_driver(
        joinpath(output_root, "static_stack_driver_omni_sparse_artifact.csv"), sparse,
    )
    write_static_stack_driver(
        joinpath(output_root, "static_stack_driver_omni_dense_artifact.csv"), dense,
    )
    split_audit = DataFrame(
        final_train_stop_utc=[STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP],
        final_train_actual_max_utc=[training.max_time],
        final_validation_start_utc=[STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START],
        final_validation_stop_utc=[STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_STOP],
        target_embargo_hours=[_drv_omni_elapsed_hours(
            training.max_time, STATIC_STACK_DRIVER_OMNI_FINAL_VALIDATION_START,
        )],
        retained_training_segments=[training.retained_segments],
        discarded_training_segments=[training.discarded_segments],
        retained_training_rows=[training.retained_rows],
        discarded_training_rows=[training.discarded_rows],
        scheduled_validation_issues=[validation.scheduled],
        eligible_validation_issues=[validation.eligible],
        validation_availability=[validation.availability],
        sparse_artifact_sha256=[static_stack_driver_sha256(sparse)],
        dense_artifact_sha256=[static_stack_driver_sha256(dense)],
        exposed_2023_2025_rows_read=[0],
        promotion_eligible=[false],
    )
    CSV.write(joinpath(output_root, "static_stack_driver_omni_split_audit.csv"), split_audit)
    return (summary=summary, by_lead=by_lead, metrics=metrics,
            bootstrap=bootstrap, selection=selection.audit,
            split_audit=split_audit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_stack_driver_omni_diagnostic()
    println("static stack-M2 OMNI diagnostic: ", result.summary.diagnostic_outcome[1])
end
