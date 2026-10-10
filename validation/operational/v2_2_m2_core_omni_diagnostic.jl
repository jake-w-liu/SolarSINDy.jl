isdefined(@__MODULE__, :run_v2_2_m2_omni_diagnostic) ||
    include(joinpath(@__DIR__, "v2_2_m2_omni_diagnostic.jl"))

const V22_M2_CORE_DST_DATASET = "OMNI2_H0_MRG1HR"
const V22_M2_CORE_DST_PARAMETERS = ("Time", "DST1800")
const V22_M2_CORE_MINIMUM_GAIN_NT = 0.25
const V22_M2_CORE_CANDIDATES = (
    "sindy_sparse_driver",
    "sindy_dense_driver",
)
const V22_M2_CORE_COMPARATORS = Dict(
    "sindy_sparse_driver" => (
        "sindy_dense_driver",
        "sindy_persistent_driver",
        "dst_persistence",
        "burton_sparse_driver",
        "burton_full_sparse_driver",
        "obrien_sparse_driver",
    ),
    "sindy_dense_driver" => (
        "sindy_persistent_driver",
        "dst_persistence",
        "burton_dense_driver",
        "burton_full_dense_driver",
        "obrien_dense_driver",
    ),
)

function _v22m2_core_dst_url()
    minimum = Dates.format(V22_M2_OMNI_GRID_START,
                           dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    stop = V22_M2_OMNI_FINAL_VALIDATION_STOP + Hour(8)
    maximum = Dates.format(stop, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    parameters = join(V22_M2_CORE_DST_PARAMETERS, ',')
    return "https://cdaweb.gsfc.nasa.gov/hapi/data" *
           "?id=$(V22_M2_CORE_DST_DATASET)&parameters=$parameters" *
           "&time.min=$minimum&time.max=$maximum&format=csv"
end

function _v22m2_core_cached_dst(cache_root::AbstractString)
    mkpath(cache_root)
    islink(cache_root) && throw(ArgumentError(
        "M2 core diagnostic cache root must not be a symbolic link",
    ))
    prefix = "omni2_h0_dst_20100101_20200523_"
    matches = filter(name -> startswith(name, prefix) && endswith(name, ".csv"),
                     readdir(cache_root))
    length(matches) <= 1 || throw(ArgumentError(
        "multiple pinned hourly Dst source objects exist",
    ))
    if length(matches) == 1
        name = only(matches)
        path = joinpath(cache_root, name)
        checksum = splitext(name)[1][(lastindex(prefix) + 1):end]
        occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
            "cached hourly Dst filename does not contain a SHA-256 digest",
        ))
        _v22m2_omni_sha256(path) == checksum || throw(ArgumentError(
            "cached hourly Dst source failed its filename checksum",
        ))
        _v22m2_omni_validate_source_object(path, "hourly Dst")
        return (path=path, url=_v22m2_core_dst_url(), sha256=checksum,
                reused=true)
    end
    return mktempdir(cache_root) do temporary_root
        temporary = joinpath(temporary_root, "download.csv")
        Downloads.download(_v22m2_core_dst_url(), temporary)
        _v22m2_omni_validate_source_object(temporary, "hourly Dst")
        checksum = _v22m2_omni_sha256(temporary)
        destination = joinpath(cache_root, prefix * checksum * ".csv")
        ispath(destination) && throw(ArgumentError(
            "hourly Dst cache destination already exists unexpectedly",
        ))
        mv(temporary, destination)
        return (path=destination, url=_v22m2_core_dst_url(),
                sha256=checksum, reused=false)
    end
end

function v2_2_m2_core_read_dst(path::AbstractString)
    raw = CSV.read(path, DataFrame;
        header=[:time_midpoint_utc, :dst_nt],
        types=Dict(:time_midpoint_utc => String, :dst_nt => Float64),
        strict=true,
        silencewarnings=false,
    )
    nrow(raw) > 0 || throw(ArgumentError(
        "hourly Dst source contains no rows",
    ))
    midpoint = _v22m2_omni_parse_time.(raw.time_midpoint_utc)
    all(index -> midpoint[index] == midpoint[1] + Hour(index - 1),
        eachindex(midpoint)) || throw(ArgumentError(
            "hourly Dst source is not a complete hourly grid",
        ))
    all(time -> minute(time) == 30 && second(time) == 0, midpoint) ||
        throw(ArgumentError("hourly Dst time tags are not half-hour midpoints"))
    issue_time = midpoint .- Minute(30)
    values = Float64.(raw.dst_nt)
    valid = isfinite.(values) .& (abs.(values) .< 99_999.0)
    table = DataFrame(issue_time_utc=issue_time[valid], dst_nt=values[valid])
    nrow(table) == length(unique(table.issue_time_utc)) || throw(ArgumentError(
        "hourly Dst source contains duplicate valid issue times",
    ))
    return table
end

function _v22m2_core_anchor_pressure(history::AbstractMatrix)
    size(history) == (25, 5) || throw(DimensionMismatch(
        "M2 core diagnostic history must have size (25, 5)",
    ))
    speed = mean(exp.(@view history[24:25, 4]))
    density = mean(exp.(@view history[24:25, 5]))
    pressure = dynamic_pressure(density, speed)
    isfinite(pressure) && pressure >= 0.0 || throw(ArgumentError(
        "M2 core diagnostic anchor pressure is invalid",
    ))
    return pressure
end

function _v22m2_core_advance_baselines(dst_star::Float64, driver)
    southward = max(-driver.Bz, 0.0)
    speed = [driver.V]
    bs = [southward]
    state = [dst_star]
    burton = clamp(burton_model(speed, bs, state)[1], -200.0, 200.0)
    burton_full = clamp(
        burton_model_full(speed, bs, state)[1], -200.0, 200.0,
    )
    obrien = clamp(
        obrien_mcpherron_model(speed, bs, state)[1], -200.0, 200.0,
    )
    return (
        burton=clamp(dst_star + burton, -2000.0, 50.0),
        burton_full=clamp(dst_star + burton_full, -2000.0, 50.0),
        obrien=clamp(dst_star + obrien, -2000.0, 50.0),
    )
end

function _v22m2_core_physics_paths(anchor_dst_star::Float64, drivers)
    states = (burton=anchor_dst_star, burton_full=anchor_dst_star,
              obrien=anchor_dst_star)
    burton = Float64[]
    burton_full = Float64[]
    obrien = Float64[]
    for driver in drivers
        advanced_burton = _v22m2_core_advance_baselines(states.burton, driver)
        advanced_full = _v22m2_core_advance_baselines(
            states.burton_full, driver,
        )
        advanced_obrien = _v22m2_core_advance_baselines(
            states.obrien, driver,
        )
        states = (
            burton=advanced_burton.burton,
            burton_full=advanced_full.burton_full,
            obrien=advanced_obrien.obrien,
        )
        push!(burton, dst_star_to_dst(states.burton, driver.Pdyn))
        push!(burton_full, dst_star_to_dst(states.burton_full, driver.Pdyn))
        push!(obrien, dst_star_to_dst(states.obrien, driver.Pdyn))
    end
    return (burton=burton, burton_full=burton_full, obrien=obrien)
end

function _v22m2_core_complete_cases(series::V22M2OmniSeries,
        dst_table::DataFrame)
    dst_at = Dict(DateTime(row.issue_time_utc) => Float64(row.dst_nt)
                  for row in eachrow(dst_table))
    driver_cases = v2_2_m2_omni_cases(
        series, V22_M2_OMNI_FINAL_VALIDATION_START,
        V22_M2_OMNI_FINAL_VALIDATION_STOP,
    )
    cases = V22M2OmniCase[]
    for case in driver_cases.cases
        required = (case.issue_time - Hour(1), case.issue_time,
                    (case.issue_time + Hour(lead)
                     for lead in V22_M2_OMNI_LEADS_H)...)
        all(haskey(dst_at, time) for time in required) && push!(cases, case)
    end
    return (cases=cases, dst_at=dst_at,
            driver_eligible=driver_cases.eligible,
            complete=length(cases))
end

function v2_2_m2_core_score(core::OperationalCore,
        series::V22M2OmniSeries, complete_cases,
        sparse::OperationalV22DriverArtifact,
        dense::OperationalV22DriverArtifact)
    rows = NamedTuple[]
    for case in complete_cases.cases
        issue = case.issue_time
        latest_dst = complete_cases.dst_at[issue]
        previous_dst = complete_cases.dst_at[issue - Hour(1)]
        history = _v22m2_omni_history(series, case)
        anchor_pressure = _v22m2_core_anchor_pressure(history)
        anchor_star = dst_to_dst_star(latest_dst, anchor_pressure)
        paths = (
            sparse=_v22m2_omni_prediction(:sparse, sparse, series, case, 0),
            dense=_v22m2_omni_prediction(:dense, dense, series, case, 0),
            persistent=_v22m2_omni_prediction(
                :persistence, nothing, series, case, 0,
            ),
        )
        predictions = Dict{String,Vector{Float64}}()
        for (name, path) in pairs(paths)
            core_path = operational_v22_core_path_forecast(core, anchor_star, path)
            suffix = name == :persistent ? "persistent" : String(name)
            predictions["sindy_$(suffix)_driver"] =
                collect(core_path.pred_dst_nt)
            physics = _v22m2_core_physics_paths(
                anchor_star, core_path.hourly_drivers,
            )
            predictions["burton_$(suffix)_driver"] = physics.burton
            predictions["burton_full_$(suffix)_driver"] = physics.burton_full
            predictions["obrien_$(suffix)_driver"] = physics.obrien
        end
        predictions["dst_persistence"] = fill(latest_dst, 7)
        for lead in V22_M2_OMNI_LEADS_H
            observed = complete_cases.dst_at[issue + Hour(lead)]
            for method in sort!(collect(keys(predictions)))
                prediction = predictions[method][lead]
                push!(rows, (
                    issue_time_utc=issue,
                    method=method,
                    lead_hours=lead,
                    latest_dst_nt=latest_dst,
                    dst_delta_1h_nt=latest_dst - previous_dst,
                    observed_dst_nt=observed,
                    predicted_dst_nt=prediction,
                    squared_error_nt2=(prediction - observed)^2,
                    absolute_error_nt=abs(prediction - observed),
                    target_storm=observed <= -50.0,
                    active_or_deepening=latest_dst <= -30.0 &&
                        latest_dst - previous_dst < 0.0,
                    recovery=latest_dst <= -30.0 &&
                        latest_dst - previous_dst >= 0.0,
                ))
            end
        end
    end
    errors = DataFrame(rows)
    metrics = NamedTuple[]
    for group in groupby(errors, [:method, :lead_hours])
        values = (
            overall=trues(nrow(group)),
            target_storm=group.target_storm,
            active_or_deepening=group.active_or_deepening,
            recovery=group.recovery,
        )
        for (regime, selected) in pairs(values)
            count_rows = count(selected)
            push!(metrics, (
                method=group.method[1],
                lead_hours=group.lead_hours[1],
                regime=String(regime),
                rows=count_rows,
                rmse_nt=count_rows == 0 ? NaN :
                    sqrt(mean(group.squared_error_nt2[selected])),
                mae_nt=count_rows == 0 ? NaN :
                    mean(group.absolute_error_nt[selected]),
            ))
        end
    end
    return errors, DataFrame(metrics)
end

function v2_2_m2_core_bootstrap(errors::DataFrame;
        draws::Integer=V22_M2_OMNI_BOOTSTRAP_DRAWS,
        seed::Integer=V22_M2_OMNI_BOOTSTRAP_SEED + 1)
    draws >= 100 || throw(ArgumentError(
        "M2 core diagnostic bootstrap requires at least 100 draws",
    ))
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for candidate in V22_M2_CORE_CANDIDATES,
            comparator in V22_M2_CORE_COMPARATORS[candidate],
            lead in V22_M2_OMNI_LEADS_H
        candidate_rows = errors[(errors.method .== candidate) .&
                                (errors.lead_hours .== lead),
                                [:issue_time_utc, :squared_error_nt2]]
        comparator_rows = errors[(errors.method .== comparator) .&
                                 (errors.lead_hours .== lead),
                                 [:issue_time_utc, :squared_error_nt2]]
        candidate_rows.issue_time_utc == comparator_rows.issue_time_utc ||
            throw(ArgumentError("M2 core bootstrap rows are not exactly paired"))
        blocks = _v22m2_omni_block_id.(candidate_rows.issue_time_utc)
        unique_blocks = sort(unique(blocks))
        length(unique_blocks) >= 2 || throw(ArgumentError(
            "M2 core bootstrap requires at least two 168-hour blocks",
        ))
        candidate_sums = Dict(block => 0.0 for block in unique_blocks)
        comparator_sums = Dict(block => 0.0 for block in unique_blocks)
        counts = Dict(block => 0 for block in unique_blocks)
        for index in eachindex(blocks)
            block = blocks[index]
            candidate_sums[block] += candidate_rows.squared_error_nt2[index]
            comparator_sums[block] += comparator_rows.squared_error_nt2[index]
            counts[block] += 1
        end
        differences = Vector{Float64}(undef, draws)
        for draw in 1:draws
            sampled = rand(rng, unique_blocks, length(unique_blocks))
            n = sum(counts[block] for block in sampled)
            candidate_rmse = sqrt(
                sum(candidate_sums[block] for block in sampled) / n,
            )
            comparator_rmse = sqrt(
                sum(comparator_sums[block] for block in sampled) / n,
            )
            differences[draw] = comparator_rmse - candidate_rmse
        end
        point = sqrt(mean(comparator_rows.squared_error_nt2)) -
                sqrt(mean(candidate_rows.squared_error_nt2))
        push!(rows, (
            candidate=candidate,
            comparator=comparator,
            lead_hours=lead,
            rows=nrow(candidate_rows),
            blocks=length(unique_blocks),
            draws=draws,
            seed=seed,
            point_improvement_nt=point,
            one_sided_95_lower_nt=quantile(differences, 0.05),
        ))
    end
    return DataFrame(rows)
end

function _v22m2_core_metric(metrics::DataFrame, method::AbstractString,
        lead::Int, regime::AbstractString)
    rows = metrics[(metrics.method .== method) .&
                   (metrics.lead_hours .== lead) .&
                   (metrics.regime .== regime), :]
    nrow(rows) == 1 || throw(ArgumentError(
        "M2 core metric key is missing or duplicated",
    ))
    return only(eachrow(rows))
end

function v2_2_m2_core_decision(metrics::DataFrame, bootstrap::DataFrame,
        driver_decision::DataFrame)
    rows = NamedTuple[]
    driver_supported = driver_decision.diagnostic_outcome[1] ==
                       "PREDICTABILITY_SUPPORTED_WITHOUT_PREFIX"
    for candidate in V22_M2_CORE_CANDIDATES, lead in V22_M2_OMNI_LEADS_H
        passes = candidate == "sindy_sparse_driver" ? driver_supported : true
        for comparator in V22_M2_CORE_COMPARATORS[candidate]
            evidence = bootstrap[(bootstrap.candidate .== candidate) .&
                                 (bootstrap.comparator .== comparator) .&
                                 (bootstrap.lead_hours .== lead), :]
            nrow(evidence) == 1 || throw(ArgumentError(
                "M2 core bootstrap key is missing or duplicated",
            ))
            row = only(eachrow(evidence))
            passes &= row.point_improvement_nt >= V22_M2_CORE_MINIMUM_GAIN_NT &&
                      row.one_sided_95_lower_nt > 0.0
            for regime in ("target_storm", "active_or_deepening", "recovery")
                candidate_metric = _v22m2_core_metric(
                    metrics, candidate, lead, regime,
                )
                comparator_metric = _v22m2_core_metric(
                    metrics, comparator, lead, regime,
                )
                if candidate_metric.rows >= 20 && comparator_metric.rows >= 20
                    passes &= candidate_metric.rmse_nt <= comparator_metric.rmse_nt
                end
            end
        end
        push!(rows, (candidate=candidate, lead_hours=lead,
                     end_to_end_gate_pass=passes))
    end
    by_lead = DataFrame(rows)
    sparse = all(by_lead.end_to_end_gate_pass[
        by_lead.candidate .== "sindy_sparse_driver",
    ])
    dense = all(by_lead.end_to_end_gate_pass[
        by_lead.candidate .== "sindy_dense_driver",
    ])
    summary = DataFrame(
        sparse_core_diagnostic_supported=[sparse],
        dense_core_exploratory_supported=[dense],
        minimum_required_gain_nt=[V22_M2_CORE_MINIMUM_GAIN_NT],
        upstream_sparse_driver_supported=[driver_supported],
        receipt_causal=[false],
        promotion_eligible=[false],
        source_role=["retrospectively_shifted_near_earth_core_diagnostic_only"],
    )
    return by_lead, summary
end

function run_v2_2_m2_core_omni_diagnostic(;
        driver_output_root=joinpath(@__DIR__, "..", "output", "operational",
            "v2_2_m2_omni_diagnostic"),
        output_root=joinpath(@__DIR__, "..", "output", "operational",
            "v2_2_m2_core_omni_diagnostic"),
        cache_root=joinpath(@__DIR__, "..", "output", "operational",
            "source_cache", "v2_2_m2_core_omni"))
    mkpath(output_root)
    bins = CSV.read(joinpath(
        driver_output_root, "v2_2_m2_omni_half_hour_bins.csv",
    ), DataFrame)
    series = v2_2_m2_omni_series(bins)
    sparse = read_operational_v22_driver(joinpath(
        driver_output_root, "v2_2_m2_omni_sparse_artifact.csv",
    ))
    dense = read_operational_v22_driver(joinpath(
        driver_output_root, "v2_2_m2_omni_dense_artifact.csv",
    ))
    split = CSV.read(joinpath(
        driver_output_root, "v2_2_m2_omni_split_audit.csv",
    ), DataFrame)
    operational_v22_driver_sha256(sparse) == split.sparse_artifact_sha256[1] ||
        throw(ArgumentError("sparse M2 artifact does not match its split audit"))
    operational_v22_driver_sha256(dense) == split.dense_artifact_sha256[1] ||
        throw(ArgumentError("dense M2 artifact does not match its split audit"))
    driver_decision = CSV.read(joinpath(
        driver_output_root, "v2_2_m2_omni_decision.csv",
    ), DataFrame)

    source = _v22m2_core_cached_dst(cache_root)
    dst_table = v2_2_m2_core_read_dst(source.path)
    complete = _v22m2_core_complete_cases(series, dst_table)
    complete.complete > 0 || throw(ArgumentError(
        "M2 core diagnostic has no complete driver-and-Dst cases",
    ))
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    errors, metrics = v2_2_m2_core_score(
        core, series, complete, sparse, dense,
    )
    bootstrap = v2_2_m2_core_bootstrap(errors)
    by_lead, decision = v2_2_m2_core_decision(
        metrics, bootstrap, driver_decision,
    )
    CSV.write(joinpath(output_root, "v2_2_m2_core_dst_source.csv"), DataFrame(
        source_url=[source.url],
        source_path=[relpath(source.path, cache_root)],
        source_sha256=[source.sha256],
        source_bytes=[filesize(source.path)],
        source_rows=[nrow(dst_table)],
        reused=[source.reused],
    ))
    CSV.write(joinpath(output_root, "v2_2_m2_core_paired_errors.csv"), errors)
    CSV.write(joinpath(output_root, "v2_2_m2_core_metrics.csv"), metrics)
    CSV.write(joinpath(output_root, "v2_2_m2_core_bootstrap.csv"), bootstrap)
    CSV.write(joinpath(output_root, "v2_2_m2_core_decision_by_lead.csv"), by_lead)
    CSV.write(joinpath(output_root, "v2_2_m2_core_decision.csv"), decision)
    CSV.write(joinpath(output_root, "v2_2_m2_core_split_audit.csv"), DataFrame(
        driver_eligible_cases=[complete.driver_eligible],
        complete_driver_dst_cases=[complete.complete],
        first_issue_time_utc=[minimum(case.issue_time for case in complete.cases)],
        last_issue_time_utc=[maximum(case.issue_time for case in complete.cases)],
        sparse_driver_sha256=[operational_v22_driver_sha256(sparse)],
        dense_driver_sha256=[operational_v22_driver_sha256(dense)],
        frozen_core_sha256=[operational_v22_core_sha256(core)],
        exposed_2023_2025_rows_read=[0],
        promotion_eligible=[false],
    ))
    return (decision=decision, by_lead=by_lead, metrics=metrics,
            bootstrap=bootstrap)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_v2_2_m2_core_omni_diagnostic()
    println("V2.2-M2 core OMNI diagnostic: sparse=",
            result.decision.sparse_core_diagnostic_supported[1],
            ", dense=", result.decision.dense_core_exploratory_supported[1])
end
