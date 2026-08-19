"""
Leakage-safe research replay rows for V2.2 development.

This file deliberately reuses the pure V2.1 issuance kernels.  It does not read
an archive on its own and it never writes a production artifact.  Callers must
provide an observation accessor; exposed-benchmark access is rejected before that
    accessor is called unless `benchmark_access=true` is explicit.
"""

using SolarSINDy
using DataFrames
using Dates
using SHA

isdefined(@__MODULE__, :LiveVerifyConfig) ||
    include(joinpath(@__DIR__, "..", "..", "examples", "live_forecast_verify.jl"))

const V22_REPLAY_MODEL_STEPS = copy(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS)
const V22_EXPOSED_BENCHMARK_LOCKED = :exposed_benchmark_locked

# These are the already-audited V2.1 chronological boundaries.  A split owns an
# anchor only when every supported target has matured before its target_end.
const V22_REPLAY_SPLIT_CONTRACT = (
    fit=(
        issue_start=DateTime(2010, 1, 1, 1),
        issue_end=DateTime(2017, 10, 20, 15),
        target_end=DateTime(2017, 10, 20, 22),
    ),
    validation=(
        issue_start=DateTime(2017, 10, 20, 23),
        issue_end=DateTime(2020, 5, 22, 19),
        target_end=DateTime(2020, 5, 23, 2),
    ),
    calibration=(
        issue_start=DateTime(2020, 5, 23, 3),
        issue_end=DateTime(2022, 12, 31, 16),
        target_end=DateTime(2022, 12, 31, 23),
    ),
    exposed_benchmark_start=DateTime(2023, 1, 1),
)

"One retrospective issuance anchor and the causal Dst state available to it."
struct V22ReplayAnchor
    issue_time_utc::DateTime
    latest_dst_time_utc::DateTime

    function V22ReplayAnchor(issue_time_utc::DateTime,
                             latest_dst_time_utc::DateTime=issue_time_utc)
        floor(latest_dst_time_utc, Hour) == latest_dst_time_utc || throw(ArgumentError(
            "latest_dst_time_utc must lie on the hourly Dst grid",
        ))
        latest_dst_time_utc <= issue_time_utc || throw(ArgumentError(
            "latest Dst state cannot postdate issue time",
        ))
        lag = (floor(issue_time_utc, Hour) - latest_dst_time_utc) / Hour(1)
        isinteger(lag) && 0 <= lag <= LIVE_MAX_DST_ANCHOR_LAG_STEPS || throw(ArgumentError(
            "replay anchor exceeds the live Dst-lag admission contract",
        ))
        new(issue_time_utc, latest_dst_time_utc)
    end
end

struct V22BenchmarkAccessError <: Exception
    issue_time_utc::DateTime
end

Base.showerror(io::IO, err::V22BenchmarkAccessError) = print(
    io,
    "V2.2 exposed-benchmark observations are locked at issue ", err.issue_time_utc,
    "; pass benchmark_access=true only in the separately authorized scoring path",
)

"""
    v2_2_replay_split(anchor; benchmark_access=false)

Assign an entire issuance anchor to `:fit`, `:validation`, `:calibration`, or
`:benchmark`.  The maximum target is computed over the complete supported-step
set, so asking for a subset of leads cannot move a boundary anchor into an
earlier split. Absolute issue bounds are checked independently of target
maturity. Purged boundary anchors return `:embargo`; exposed-benchmark anchors
return the `V22_EXPOSED_BENCHMARK_LOCKED` sentinel by default.
"""
function v2_2_replay_split(anchor::V22ReplayAnchor;
                           benchmark_access::Bool=false)
    issue = anchor.issue_time_utc
    latest = anchor.latest_dst_time_utc
    targets = filter(>(issue), latest .+ Hour.(V22_REPLAY_MODEL_STEPS))
    isempty(targets) && return :embargo
    max_target = maximum(targets)
    contract = V22_REPLAY_SPLIT_CONTRACT

    issue >= contract.exposed_benchmark_start &&
        return benchmark_access ? :benchmark : V22_EXPOSED_BENCHMARK_LOCKED
    issue >= contract.calibration.issue_start &&
        issue <= contract.calibration.issue_end &&
        max_target <= contract.calibration.target_end && return :calibration
    issue >= contract.validation.issue_start &&
        issue <= contract.validation.issue_end &&
        max_target <= contract.validation.target_end && return :validation
    issue >= contract.fit.issue_start &&
        issue <= contract.fit.issue_end &&
        max_target <= contract.fit.target_end && return :fit
    return :embargo
end

v2_2_replay_split(issue_time_utc::DateTime; benchmark_access::Bool=false) =
    v2_2_replay_split(V22ReplayAnchor(issue_time_utc); benchmark_access)

function _v22_float(value, context::AbstractString)
    (ismissing(value) || value === nothing) && throw(ArgumentError("missing $context"))
    parsed = try
        Float64(value)
    catch
        throw(ArgumentError("non-numeric $context: $(repr(value))"))
    end
    isfinite(parsed) || throw(ArgumentError("non-finite $context: $parsed"))
    return parsed
end

function _v22_observation(observation_at::Function, t::DateTime,
                          context::AbstractString)
    return _v22_float(observation_at(t), "$context at $t")
end

function _v22_validate_driver(driver, context::AbstractString)
    vals = Float64[
        driver.V, driver.Bz, driver.By, driver.n, driver.Pdyn,
    ]
    all(isfinite, vals) || throw(ArgumentError("$context contains a non-finite driver"))
    vals[1] >= 0 && vals[4] >= 0 && vals[5] >= 0 || throw(ArgumentError(
        "$context contains a negative speed, density, or dynamic pressure",
    ))
    expected = dynamic_pressure(vals[4], vals[1])
    isapprox(vals[5], expected; atol=32eps(max(1.0, abs(expected))), rtol=0.0) ||
        throw(ArgumentError("$context violates the proton dynamic-pressure identity"))
    return (
        V=vals[1], Bz=vals[2], By=vals[3], n=vals[4], Pdyn=vals[5],
    )
end

function _v22_hash(parts...)
    io = IOBuffer()
    for part in parts
        print(io, repr(part), '\n')
    end
    return bytes2hex(sha256(take!(io)))
end

function _v22_driver_hash(sequence, h::Int)
    io = IOBuffer()
    for k in 1:h
        step = sequence[k]
        print(io, k, '|', step.l1_measured ? '1' : '0', '|', step.hours_since_l1)
        for value in (step.driver.V, step.driver.Bz, step.driver.By,
                      step.driver.n, step.driver.Pdyn)
            print(io, '|', bitstring(Float64(value)))
        end
        print(io, '\n')
    end
    return bytes2hex(sha256(take!(io)))
end

function _v22_core_hash(core::OperationalCore)
    return _v22_hash(
        core.artifacts.version,
        get_term_names(core.library),
        bitstring.(core.coefficients),
    )
end

function _v22_calibration_hash(calibration::OperationalV2Calibration)
    return _v22_hash(
        calibration.label,
        calibration.feature_names,
        bitstring.(calibration.feature_mean),
        bitstring.(calibration.feature_scale),
        bitstring.(calibration.coefficients),
        calibration.selected_component,
        calibration.supported_model_steps,
    )
end

function _v22_point_state(core::OperationalCore, t0::DateTime, dst0::Float64)
    # The ensemble does not affect the primary point path.  A one-row copy keeps
    # this research helper exact to `step_forecast!` without reloading 500 draws.
    ensemble = reshape(copy(core.coefficients), 1, :)
    return ForecastState(
        t0, dst0, core.library, core.coefficients, ensemble, 1.0,
        ForecastResult[],
    )
end

function _v22_driver_sequences(plasma::DataFrame, mag::DataFrame,
                               anchor::V22ReplayAnchor, dst_rate::Float64;
                               min_samples::Int=LIVE_MIN_HOURLY_DRIVER_SAMPLES)
    min_samples >= 1 || throw(ArgumentError("min_samples must be at least one"))
    issue = anchor.issue_time_utc
    latest_dst_time = anchor.latest_dst_time_utc
    isempty(plasma.time_tag) && throw(ArgumentError("plasma input is empty"))
    isempty(mag.time_tag) && throw(ArgumentError("magnetic-field input is empty"))
    pidx = _latest_causal_index(DateTime.(plasma.time_tag), issue, "Plasma")
    midx = _latest_causal_index(DateTime.(mag.time_tag), issue, "Magnetic-field")
    latest_common_sw = min(plasma.time_tag[pidx], mag.time_tag[midx])
    issue - latest_common_sw <= Hour(round(Int, LIVE_MAX_SOLAR_WIND_AGE_HOURS)) ||
        throw(ArgumentError("solar-wind input exceeds the live staleness ceiling"))

    recent_start = latest_common_sw - Hour(1)
    counts = (
        _window_finite_count(plasma, :speed, recent_start, latest_common_sw),
        _window_finite_count(mag, :bz_gsm, recent_start, latest_common_sw),
        _window_finite_count(plasma, :density, recent_start, latest_common_sw),
        _window_finite_count(mag, :by_gsm, recent_start, latest_common_sw),
    )
    _driver_gap_status(counts...; min_samples) == :ok || throw(ArgumentError(
        "trailing issue-time driver window is incomplete: counts=$(counts), minimum=$min_samples",
    ))
    recent = _v22_validate_driver(
        _drivers_for_window(
            plasma, mag, recent_start, latest_common_sw; min_samples,
        ),
        "trailing issue-time driver",
    )
    anchor_status = _subhourly_driver_with_status(
        plasma, mag, latest_dst_time, recent, latest_common_sw; min_samples,
    )
    anchor_status.l1_measured || throw(ArgumentError(
        "insufficient propagated L1 coverage for the Dst anchor pressure window",
    ))
    anchor_driver = _v22_validate_driver(anchor_status.driver, "Dst-anchor driver")

    reference = NamedTuple[]
    served = NamedTuple[]
    last_known = recent
    hours_since_l1 = 0
    for k in 1:maximum(V22_REPLAY_MODEL_STEPS)
        step_time = latest_dst_time + Hour(k)
        status = _subhourly_driver_with_status(
            plasma, mag, step_time, recent, latest_common_sw; min_samples,
        )
        reference_driver = _v22_validate_driver(
            status.driver, "V2.1 reference driver at step $k",
        )
        served_driver = if status.l1_measured
            hours_since_l1 = 0
            last_known = reference_driver
        else
            hours_since_l1 += 1
            _v22_validate_driver(
                _relaxed_tail_driver(last_known, hours_since_l1, dst_rate),
                "served driver at step $k",
            )
        end
        push!(reference, (
            driver=reference_driver,
            l1_measured=status.l1_measured,
            hours_since_l1=status.l1_measured ? 0 : 1,
        ))
        push!(served, (
            driver=served_driver,
            l1_measured=status.l1_measured,
            hours_since_l1=hours_since_l1,
        ))
    end
    return (; latest_common_sw, recent, anchor_driver, reference, served, counts)
end

function _v22_rollout(core::OperationalCore, anchor_time::DateTime,
                      anchor_dst_star::Float64, sequence)
    state = _v22_point_state(core, anchor_time, anchor_dst_star)
    burton_star = anchor_dst_star
    burton_full_star = anchor_dst_star
    obrien_star = anchor_dst_star
    out = Vector{NamedTuple}(undef, length(sequence))
    for k in eachindex(sequence)
        driver = sequence[k].driver
        result = step_forecast!(
            state, anchor_time + Hour(k), driver.V, driver.Bz, driver.By,
            driver.n, driver.Pdyn,
        )
        burton_star = _advance_baselines(burton_star, driver).burton
        burton_full_star = _advance_baselines(burton_full_star, driver).burton_full
        obrien_star = _advance_baselines(obrien_star, driver).obrien
        out[k] = (
            raw_sindy=_dst_from_dst_star(result.dst_predicted, driver.Pdyn),
            burton=_dst_from_dst_star(burton_star, driver.Pdyn),
            burton_full=_dst_from_dst_star(burton_full_star, driver.Pdyn),
            obrien=_dst_from_dst_star(obrien_star, driver.Pdyn),
        )
    end
    return out
end

function _v22_anchor_rows(anchor::V22ReplayAnchor, split::Symbol,
                          plasma::DataFrame, mag::DataFrame,
                          observation_at::Function, core::OperationalCore,
                          calibration::OperationalV2Calibration,
                          model_steps::Vector{Int}, core_hash::String,
                          calibration_hash::String, min_samples::Int)
    latest_time = anchor.latest_dst_time_utc
    # Only state at or before the Dst anchor is read before forecasting.
    history_times = collect((latest_time - Hour(3)):Hour(1):latest_time)
    history_values = Float64[
        _v22_observation(observation_at, t, "causal Dst history") for t in history_times
    ]
    latest_dst = history_values[end]
    dst_rate = latest_dst - history_values[end - 1]
    sequences = _v22_driver_sequences(
        plasma, mag, anchor, dst_rate; min_samples,
    )
    anchor_dst_star = pressure_correct_dst(
        [latest_dst], [sequences.anchor_driver.Pdyn],
    )[1]
    isfinite(anchor_dst_star) || throw(ArgumentError("anchor Dst* is non-finite"))

    # The served sequence defines the tail-aware raw SINDy comparator and served
    # V2.1 center. Physical experts use the reference sequence because those are
    # the exact baseline centers already computed and logged by live issuance.
    served_rollout = _v22_rollout(
        core, latest_time, anchor_dst_star, sequences.served,
    )
    reference_rollout = _v22_rollout(
        core, latest_time, anchor_dst_star, sequences.reference,
    )
    memory = _live_v2_memory_features(
        plasma, mag, history_times, history_values, latest_time,
        sequences.anchor_driver, sequences.latest_common_sw,
    )
    coupling = _vb_south(sequences.anchor_driver)
    rows = NamedTuple[]
    anchor_lag_steps = round(
        Int, (floor(anchor.issue_time_utc, Hour) - latest_time) / Hour(1),
    )
    for h in model_steps
        target = latest_time + Hour(h)
        target > anchor.issue_time_utc || throw(ArgumentError(
            "replay target $target must be strictly later than issue $(anchor.issue_time_utc)",
        ))
        product_horizon_hours = round(
            Int, (target - anchor.issue_time_utc) / Hour(1),
        )
        product_horizon_hours > 0 || throw(ArgumentError(
            "replay product horizon must be positive",
        ))
        reference = reference_rollout[h]
        reference_baselines = (
            persistence=latest_dst,
            burton=reference.burton,
            burton_full=reference.burton_full,
            obrien=reference.obrien,
        )
        features = _v2_features(
            latest_dst,
            sequences.anchor_driver;
            memory,
            baselines=reference_baselines,
            v1_pred_dst=reference.raw_sindy,
            model_steps=h,
        )
        selected = _select_model_prediction(
            :v2,
            calibration,
            latest_dst,
            sequences.anchor_driver,
            reference.raw_sindy,
            reference.raw_sindy - 1.0,
            reference.raw_sindy + 1.0;
            baselines=reference_baselines,
            features,
            model_steps=h,
        )
        ismissing(selected.v2_correction) && error(
            "V2.1 calibration did not return a correction at step $h",
        )
        shared = served_rollout[h]
        served_center = _apply_v2_1_safeguards(
            shared.raw_sindy + Float64(selected.v2_correction),
            latest_dst,
            h,
            dst_rate,
        )
        frozen_center = _v22_float(
            selected.v2_pred_dst, "V2.1 frozen-tail center at step $h",
        )
        observation = _v22_observation(
            observation_at, target, "target Dst observation",
        )
        driver_hash = _v22_driver_hash(sequences.served, h)
        reference_hash = _v22_driver_hash(sequences.reference, h)
        input_hash = _v22_hash(
            anchor.issue_time_utc,
            latest_time,
            target,
            h,
            bitstring(latest_dst),
            bitstring(dst_rate),
            driver_hash,
            reference_hash,
            core_hash,
            calibration_hash,
        )
        target_driver = sequences.served[h].driver
        admission = join(
            (sequences.served[k].l1_measured ? "$(k):L1" : "$(k):TAIL"
             for k in 1:h),
            ";",
        )
        values = (
            served_center, frozen_center, shared.raw_sindy, latest_dst,
            reference.burton, reference.burton_full, reference.obrien, observation,
            dst_rate, coupling,
        )
        all(isfinite, values) || error(
            "non-finite replay output for issue=$(anchor.issue_time_utc), target=$target, step=$h",
        )
        push!(rows, (
            issue_time_utc=anchor.issue_time_utc,
            target_time_utc=target,
            model_step_hours=h,
            anchor_lag_steps=anchor_lag_steps,
            product_horizon_hours=product_horizon_hours,
            split_label=String(split),
            served_v2_1_dst_nt=served_center,
            frozen_v2_1_dst_nt=frozen_center,
            raw_sindy_dst_nt=shared.raw_sindy,
            persistence_dst_nt=latest_dst,
            burton_dst_nt=reference.burton,
            burton_full_dst_nt=reference.burton_full,
            obrien_dst_nt=reference.obrien,
            observation_dst_nt=observation,
            latest_dst_time_utc=latest_time,
            latest_dst_nt=latest_dst,
            dst_delta_1h_nt=dst_rate,
            VBsouth_mvm=coupling,
            coupling_active_mvm=coupling > 0.0 && dst_rate < 0.0 ? coupling : 0.0,
            latest_solar_wind_utc=sequences.latest_common_sw,
            feature_driver_basis=_ANCHOR_FEATURE_DRIVER_BASIS,
            driver_assumption=V2_DRIVER_ASSUMPTION,
            V_kms=sequences.anchor_driver.V,
            Bz_nt=sequences.anchor_driver.Bz,
            By_nt=sequences.anchor_driver.By,
            n_cm3=sequences.anchor_driver.n,
            Pdyn_npa=sequences.anchor_driver.Pdyn,
            target_step_V_kms=target_driver.V,
            target_step_Bz_nt=target_driver.Bz,
            target_step_By_nt=target_driver.By,
            target_step_n_cm3=target_driver.n,
            target_step_Pdyn_npa=target_driver.Pdyn,
            l1_admission_pattern=admission,
            n_l1_admitted_steps=count(
                k -> sequences.served[k].l1_measured, 1:h,
            ),
            driver_sequence_sha256=driver_hash,
            v2_1_reference_sequence_sha256=reference_hash,
            core_version=core.artifacts.version,
            core_sha256=core_hash,
            calibration_label=calibration.label,
            calibration_sha256=calibration_hash,
            inputs_sha256=input_hash,
        ))
    end
    return rows
end

"""
    build_v2_2_served_replay(anchors, plasma, mag, observation_at, core, calibration;
                             model_steps=V22_REPLAY_MODEL_STEPS,
                             benchmark_access=false, min_samples=10)

Emit exactly one finite row per `(issue_time_utc, target_time_utc,
model_step_hours)`.  Missing inputs, duplicate keys, embargo anchors, and
exposed-benchmark access without the explicit opt-in all fail loudly; no row is
silently dropped.
"""
function build_v2_2_served_replay(
    anchors::AbstractVector{V22ReplayAnchor},
    plasma::DataFrame,
    mag::DataFrame,
    observation_at::Function,
    core::OperationalCore,
    calibration::OperationalV2Calibration;
    model_steps::AbstractVector{<:Integer}=V22_REPLAY_MODEL_STEPS,
    benchmark_access::Bool=false,
    min_samples::Int=LIVE_MIN_HOURLY_DRIVER_SAMPLES,
)
    core.artifacts.version == OPERATIONAL_V2_1_MODEL_VERSION || throw(ArgumentError(
        "served replay requires the canonical V2.1 SINDy core",
    ))
    steps = sort!(Int.(collect(model_steps)))
    isempty(steps) && throw(ArgumentError("model_steps must not be empty"))
    length(unique(steps)) == length(steps) || throw(ArgumentError(
        "model_steps must not contain duplicates",
    ))
    all(in(V22_REPLAY_MODEL_STEPS), steps) || throw(ArgumentError(
        "model_steps must be a subset of $(V22_REPLAY_MODEL_STEPS)",
    ))
    calibration.supported_model_steps == V22_REPLAY_MODEL_STEPS || throw(ArgumentError(
        "calibration model-step support does not match the V2.1 issuance contract",
    ))
    calibration.selected_component == :v2 || throw(ArgumentError(
        "served replay requires the deployed corrected-SINDy V2.1 component",
    ))
    isempty(anchors) && throw(ArgumentError("anchors must not be empty"))
    for anchor in anchors, step in steps
        anchor.latest_dst_time_utc + Hour(step) > anchor.issue_time_utc ||
            throw(ArgumentError(
                "model step $step does not produce a post-issue target for " *
                "anchor $(anchor.issue_time_utc)",
            ))
    end

    # Split every anchor before the first observation lookup.  A mixed request
    # containing one locked benchmark anchor therefore reads zero observations.
    splits = Symbol[
        v2_2_replay_split(anchor; benchmark_access) for anchor in anchors
    ]
    locked = findfirst(==(V22_EXPOSED_BENCHMARK_LOCKED), splits)
    locked === nothing || throw(V22BenchmarkAccessError(
        anchors[locked].issue_time_utc,
    ))
    embargo = findfirst(==(:embargo), splits)
    embargo === nothing || throw(ArgumentError(
        "anchor $(anchors[embargo].issue_time_utc) lies inside a target-maturity embargo",
    ))

    core_hash = _v22_core_hash(core)
    calibration_hash = _v22_calibration_hash(calibration)
    rows = NamedTuple[]
    for (anchor, split) in zip(anchors, splits)
        append!(rows, _v22_anchor_rows(
            anchor, split, plasma, mag, observation_at, core, calibration,
            steps, core_hash, calibration_hash, min_samples,
        ))
    end
    expected = length(anchors) * length(steps)
    length(rows) == expected || error(
        "strict replay row count failed: expected $expected, got $(length(rows))",
    )
    table = DataFrame(rows)
    keys = Tuple.(eachrow(select(
        table, :issue_time_utc, :target_time_utc, :product_horizon_hours,
    )))
    length(unique(keys)) == expected || throw(ArgumentError(
        "duplicate issue/target/product-horizon replay key",
    ))
    all(
        table.target_time_utc .== table.latest_dst_time_utc .+
            Hour.(table.model_step_hours),
    ) || error("target-time/model-step invariant failed")
    all(
        table.target_time_utc .== table.issue_time_utc .+
            Hour.(table.product_horizon_hours),
    ) || error("target-time/product-horizon invariant failed")
    sort!(table, [:issue_time_utc, :target_time_utc, :model_step_hours])
    return table
end
