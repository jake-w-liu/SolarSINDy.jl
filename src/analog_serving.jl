# analog_serving.jl — Analog-continuation ensemble machinery: the artifact record and the pure
# center-forming functions the live engine and the offline identity oracle share.
#
# The analog ensemble replaces the served inertial-SINDy heuristic tail with an analog driver continuation, an
# analog-core refit of the deployed inertial-SINDy ridge correction (T1r), a lead-aware blend toward the
# frozen inertial-SINDy center (LAT), and a per-step error layer (E). Scoring that pipeline offline and
# serving it live are two different execution environments — an hourly OMNI archive versus a
# minute-cadence L1 feed — but they must produce the same number for the same information set.
#
# This file is the single implementation of the center. It takes the parts of the state that the
# two environments genuinely disagree about (how an hour of solar wind is measured) as arguments,
# and computes everything else once:
#
#   * `analog_serving_features`  — the 18 issue-time analog features from a driver history and the
#     two anchor Dst values (specification step 1);
#   * `analog_serving_members`   — the K nearest archive origins and the ensemble-mean raw center
#     obtained by rolling the frozen SINDy core once per member (step 2);
#   * `analog_serving_center`    — the T1r correction, the LAT blend and the E layer (steps 3-5).
#
# The only environment-specific input is the `step_driver` callback: it answers "what drives
# rollout step k, and was that hour measured at L1 by issue time?". The live engine answers with
# the ballistically propagated sub-hourly L1 window; the identity oracle answers with the hourly
# archive under the same L1 admission gate the served inertial-SINDy tail uses. Everything downstream of
# that callback is shared code, which is what makes the offline identity check meaningful.
#
# Time convention (unchanged from `analog_features.jl`): the anchor is the newest observed
# Dst hour `t`; the freshest driver record is the one tagged `t-1`, covering the at-Earth interval
# [t-1, t); rollout step k targets `t+k`.

import SHA

"Model steps the analog ensemble candidate supports, in artifact column order."
const ANALOG_SERVING_MODEL_STEPS = (1, 2, 3, 4, 6, 7)

"Slot of a model step in the analog ensemble per-step artifact vectors; 0 marks an unsupported step."
const ANALOG_SERVING_STEP_SLOT = let slots = zeros(Int, maximum(ANALOG_SERVING_MODEL_STEPS))
    for (slot, step) in enumerate(ANALOG_SERVING_MODEL_STEPS)
        slots[step] = slot
    end
    Tuple(slots)
end

"""
    analog_serving_identity(weight_set, k) -> String

Pipeline identity of a analog ensemble deployment. It names every stage a reader needs to reproduce the center:
the frozen 20-candidate/11-active SINDy core, the L1 look-ahead admission, the analog driver
continuation with its distance weights and ensemble size, the analog-core refit of the inertial-SINDy ridge
correction, the lead-aware blend toward the frozen inertial-SINDy center, and the error layer.
"""
analog_serving_identity(weight_set, k::Integer) =
    string("analog",
           "+sindy20x11+L1A+ADC($(String(Symbol(weight_set))),K$(Int(k)))+T1rcal+LAT+E")

"""
    ANALOG_SERVING_IDENTITY

Pipeline identity of the analog ensemble candidate: the magnetic-weighted 25-member analog ensemble selected on
the development partition. A deployment directory that loads with a different identity is a different
model.
"""
const ANALOG_SERVING_IDENTITY = analog_serving_identity(:magnetic, 25)

"Number of matured one-step innovations the error layer consumes."
const ANALOG_SERVING_E_INNOVATION_LAGS = 6

"Names of the innovation error-state inputs, in column order (E1 design)."
const ANALOG_SERVING_E_FEATURE_NAMES = (
    :innovation_1h, :innovation_2h, :innovation_3h, :innovation_4h, :innovation_5h,
    :innovation_6h, :dst0, :ddst1, :vbs0,
)

"Cap of an error-layer correction at model step h (nT)."
analog_serving_e_cap(h::Integer) = 5.0 + 5.0 * Int(h)

"Sun-Earth L1 standoff (km); the ballistic transit lag is `L1_DIST_KM / V / 3600` hours."
const ANALOG_SERVING_L1_DIST_KM = 1.5e6

"Ballistic L1 -> Earth transit time (h) for a solar-wind speed `V` (km/s)."
@inline analog_serving_transit_hours(V::Real) =
    (isfinite(V) && V > 0) ? ANALOG_SERVING_L1_DIST_KM / Float64(V) / 3600.0 : 0.0

"Tolerance for the shipped-vs-recomputed analog standardisation check (feature units)."
const ANALOG_SERVING_STATS_ATOL = 1e-9

"Physical projection applied to every reported analog ensemble Dst quantity (nT)."
const ANALOG_SERVING_DST_FLOOR_NT = -2000.0
const ANALOG_SERVING_DST_CEIL_NT = 50.0

"Placeholder interval half-width the T1r feature frame carries; never read back."
const ANALOG_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT = 1.0

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

"""
    AnalogServingRidge

Deployed E1 layer: a ridge fitted on standardised inputs with an intercept. `mean` and `scale`
standardise the design, `beta` are the standardised slopes and `intercept` is the training-target
mean, exactly as the development fit stored them.
"""
struct AnalogServingRidge
    mean::Vector{Float64}
    scale::Vector{Float64}
    beta::Vector{Float64}
    intercept::Float64
    lambda::Float64
end

"""
    AnalogServingELayer

Error layer of one model step. `kind` is `:identity`, `:E1` (ridge) or `:E2` (boosted); `cap` is
the symmetric bound the correction is clamped to; `feature_names` records the design the fitted
object expects, so a schema drift fails at load instead of silently reordering columns.
"""
struct AnalogServingELayer
    model_step_hours::Int
    kind::Symbol
    param::String
    cap::Float64
    feature_names::Vector{String}
    ridge::Union{Nothing,AnalogServingRidge}
    boosted::Any
end

"""
    AnalogServingArtifacts

Everything the analog-continuation expert needs, loaded once and immutable afterwards: the hourly analog
frame and its driver lookup, the rebuilt analog archive with its standardisation, the analog-core
refit calibration, the lead-aware blend weights, the per-step error layers, and the frozen inertial-SINDy
core the ensemble is rolled through.
"""
struct AnalogServingArtifacts
    dir::String
    frame::DataFrame
    lookup::Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}
    origins::Vector{DateTime}
    archive_features::Matrix{Float64}
    archive_standardised::Matrix{Float64}
    feature_mean::Vector{Float64}
    feature_sd::Vector{Float64}
    weight_set::Symbol
    weights::Vector{Float64}
    k::Int
    exclusion_hours::Int
    calibration::InertialSindyCalibration
    lat_weights::Vector{Float64}
    e_layers::Vector{AnalogServingELayer}
    selection::Dict{String,Any}
    core::Any
    identity::String
end

Base.show(io::IO, art::AnalogServingArtifacts) = print(
    io, "AnalogServingArtifacts(", art.identity, ", origins=", length(art.origins),
    ", K=", art.k, ", dir=", art.dir, ")",
)

# ---------------------------------------------------------------------------
# Shared artifact utilities
# ---------------------------------------------------------------------------

"SHA-256 of a regular non-symlink file, as lowercase hex."
function analog_serving_file_sha256(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "analog ensemble serving artifact must be a regular non-symlink file: $source",
    ))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"Driver lookup of the shipped analog frame; a record enters only when every channel is finite."
function analog_serving_frame_lookup(frame::DataFrame)
    lookup = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    sizehint!(lookup, nrow(frame))
    for i in 1:nrow(frame)
        driver = (V = Float64(frame.V[i]), Bz = Float64(frame.Bz[i]), By = Float64(frame.By[i]),
                  n = Float64(frame.n[i]), Pdyn = Float64(frame.Pdyn[i]))
        all(isfinite, values(driver)) || continue
        lookup[frame.time_utc[i]] = driver
    end
    return lookup
end

# ---------------------------------------------------------------------------
# Step 1 — issue-time analog features
# ---------------------------------------------------------------------------

"""
    analog_serving_features(art, anchor_time, driver_history, dst_anchor, dst_prev) -> NamedTuple

Issue-time analog feature vector for the anchor hour `anchor_time` (specification step 1).

`driver_history[j]` is the at-Earth driver record tagged `anchor_time - j` hours — the hourly mean
covering `[anchor_time - j, anchor_time - j + 1)` — or `nothing` when that hour is not available.
At least `ANALOG_HISTORY_LAGS_H` entries must be supplied; the further entries, up to
`ANALOG_SOUTH_RUN_CAP_H`, are consumed only by the consecutive-southward run-length feature and may be
absent, in which case the run length truncates exactly as it does at the start of the archive.
`dst_anchor` and `dst_prev` are the observed Dst at `anchor_time` and one hour earlier.

Returns `(features, ok, reason)`. `ok = false` leaves `features` non-finite and names the missing
input in `reason`, which is what the caller records when no analog center is available.

The construction goes through `analog_feature_matrix` on a short frame rather than reimplementing the
feature definitions, so the live key and the archive keys are produced by one code path.
"""
function analog_serving_features(art::AnalogServingArtifacts, anchor_time::DateTime, driver_history,
                              dst_anchor::Real, dst_prev::Real)
    lags = length(driver_history)
    lags >= ANALOG_HISTORY_LAGS_H || throw(ArgumentError(
        "the analog ensemble key needs at least $(ANALOG_HISTORY_LAGS_H) driver records, got $lags",
    ))
    times = DateTime[anchor_time - Hour(j) for j in lags:-1:1]
    push!(times, anchor_time)
    n = length(times)
    V = fill(NaN, n); Bz = fill(NaN, n); By = fill(NaN, n)
    density = fill(NaN, n); pdyn = fill(NaN, n); dst = fill(NaN, n)
    for j in 1:lags
        record = driver_history[j]
        record === nothing && continue
        row = lags - j + 1
        V[row] = Float64(record.V)
        Bz[row] = Float64(record.Bz)
        By[row] = Float64(record.By)
        density[row] = Float64(record.n)
        pdyn[row] = Float64(record.Pdyn)
    end
    dst[n] = Float64(dst_anchor)
    lags >= 1 && (dst[n - 1] = Float64(dst_prev))
    mini = DataFrame(time_utc = times, V = V, Bz = Bz, By = By, n = density,
                     Pdyn = pdyn, Dst = dst)
    X, ok = analog_feature_matrix(mini, [anchor_time])
    values = vec(X[1, :])
    ok[1] && return (features = values, ok = true, reason = "ok")
    reason = if !isfinite(Float64(dst_anchor))
        "missing_anchor_dst"
    elseif !isfinite(Float64(dst_prev))
        "missing_previous_dst"
    else
        bad = findfirst(j -> driver_history[j] === nothing ||
                             !all(isfinite, (Float64(driver_history[j].V),
                                             Float64(driver_history[j].Bz),
                                             Float64(driver_history[j].By),
                                             Float64(driver_history[j].n),
                                             Float64(driver_history[j].Pdyn))) ||
                             Float64(driver_history[j].n) <= 0.0,
                        1:ANALOG_HISTORY_LAGS_H)
        bad === nothing ? "incomplete_analog_key" : "missing_driver_lag$(bad)"
    end
    return (features = values, ok = false, reason = reason)
end

# ---------------------------------------------------------------------------
# Step 2 — analog retrieval and the ensemble raw center
# ---------------------------------------------------------------------------

"""
    analog_serving_step_driver_from_frame(lookup, anchor_time, issue_drv) -> Function

Rollout-step driver policy of the hourly archive: the L1 admission gate the served inertial-SINDy tail uses,
expressed against an hourly driver lookup. Step `k` is admitted when the issue-time transit window
covers it (`k <= floor(transit(V_issue))`) and the arriving record's own speed still covers it; an
in-window record that fails either test freezes the last admitted driver, and steps past the window
are handed to the analog member.

This is the offline half of the analog path's environment split. The live engine supplies the
minute-cadence equivalent instead, and every stage downstream is shared.
"""
function analog_serving_step_driver_from_frame(lookup::AbstractDict, anchor_time::DateTime, issue_drv)
    kdelta = floor(Int, analog_serving_transit_hours(issue_drv.V))
    return function (k::Int, last_known)
        k <= kdelta || return (driver = last_known, l1_measured = false)
        record = get(lookup, anchor_time + Hour(k - 1), nothing)
        admitted = record !== nothing && k <= analog_serving_transit_hours(record.V)
        return (driver = admitted ? record : last_known, l1_measured = true)
    end
end

"""
    analog_serving_members(art, features; anchor_time, issue_drv, anchor_dst_star, model_steps,
                        step_driver) -> NamedTuple

Retrieve the K nearest admissible archive origins for the analog key `features` and form the raw
analog ensemble center at `model_steps` hours (specification step 2).

Each retrieved origin becomes one ensemble member. A member rolls the frozen SINDy core from the
anchor state: steps the environment reports as L1-measured take the measured driver and advance the
last-known driver, and later steps take the member's analog continuation, which applies the origin's
archived speed and log-density increments to the query's own issue driver and copies the magnetic
field. Each member contributes `Dst* + 7.26*sqrt(Pdyn) - 11` evaluated with its own final-step
pressure, and the raw center is the member mean.

Returns the member origins, the per-member centers, the unprojected mean and the reported raw
center, which is the mean projected to the physical range.
"""
function analog_serving_members(art::AnalogServingArtifacts, features::AbstractVector{<:Real};
                             anchor_time::DateTime, issue_drv, anchor_dst_star::Real,
                             model_steps::Integer, step_driver)
    length(features) == ANALOG_FEATURE_COUNT || throw(DimensionMismatch(
        "the analog ensemble key has $(length(features)) entries, expected $(ANALOG_FEATURE_COUNT)",
    ))
    all(isfinite, features) || throw(ArgumentError(
        "the analog ensemble key must be finite; check `ok` before retrieval",
    ))
    steps = Int(model_steps)
    steps >= 1 || throw(ArgumentError("model step must be positive, got $steps"))
    query = analog_standardize(reshape(Float64.(collect(features)), 1, ANALOG_FEATURE_COUNT),
                            art.feature_mean, art.feature_sd)
    neighbours = analog_knn(query, [anchor_time], art.archive_standardised, art.origins, art.k;
                         weights = art.weights, exclusion_hours = art.exclusion_hours)
    origins = DateTime[art.origins[neighbours[1, m]] for m in 1:art.k]
    lib = art.core.library
    xi = art.core.coefficients
    member_pred = Vector{Float64}(undef, art.k)
    for m in 1:art.k
        last_known = issue_drv
        final_drv = issue_drv
        filter = init_assimilation(lib, xi, Int[], Float64(anchor_dst_star))
        for k in 1:steps
            decision = step_driver(k, last_known)
            drv = if decision.l1_measured
                last_known = decision.driver
                last_known
            else
                analog_member_driver(issue_drv, art.lookup, origins[m], k)
            end
            final_drv = drv
            assimilation_predict!(filter, drv)
            filter.mean[1] = clamp(filter.mean[1], ANALOG_SERVING_DST_FLOOR_NT,
                                   ANALOG_SERVING_DST_CEIL_NT)
        end
        member_pred[m] = current_dst(filter) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
    end
    raw = mean(member_pred)
    return (origins = origins, k = art.k, member_pred = member_pred, raw = raw,
            raw_reported = clamp(raw, ANALOG_SERVING_DST_FLOOR_NT, ANALOG_SERVING_DST_CEIL_NT))
end

# ---------------------------------------------------------------------------
# Steps 3-5 — correction, lead-aware blend, error layer
# ---------------------------------------------------------------------------

"""
    analog_serving_calibration_features(calibration, core_dst, latest_dst, drivers, memory,
                                     baselines, model_steps, anchor_time) -> NamedTuple

Calibration feature tuple of one `(anchor, model step)` row for the ridge layer `calibration`,
evaluated with `core_dst` as the raw core the correction is applied to.

The frame is handed to `add_inertial_sindy_features!`, the package path the deployed calibration was
fitted with, rather than to a re-derivation of the feature definitions: the five core-dependent
features (the three baseline differences, the baseline spread and the lead interaction) then come out
of the same code for the analog core and for the frozen core, and the issue-time state and memory
lags pass through untouched. The six-hour memory columns the fit does not use are the only
quantities a single-row frame cannot reproduce, and they are absent from every deployed feature list.
"""
function analog_serving_calibration_features(calibration::InertialSindyCalibration, core_dst::Real,
                                          latest_dst::Real, drivers, memory, baselines,
                                          model_steps::Integer, anchor_time::DateTime)
    core_value = Float64(core_dst)
    frame = DataFrame(
        issue_time_utc = [anchor_time],
        model_step_hours = [Int(model_steps)],
        latest_dst_nt = [Float64(latest_dst)],
        V_kms = [Float64(drivers.V)],
        Bz_nt = [Float64(drivers.Bz)],
        By_nt = [Float64(drivers.By)],
        n_cm3 = [Float64(drivers.n)],
        Pdyn_npa = [Float64(drivers.Pdyn)],
        dst_delta_1h_nt = [Float64(memory.dst_delta_1h_nt)],
        dst_delta_3h_nt = [Float64(memory.dst_delta_3h_nt)],
        Bz_delta_1h_nt = [Float64(memory.Bz_delta_1h_nt)],
        VBsouth_delta_1h_mvm = [Float64(memory.VBsouth_delta_1h_mvm)],
        VBsouth_mean_3h_mvm = [Float64(memory.VBsouth_mean_3h_mvm)],
        Bsouth_mean_3h_nt = [Float64(memory.Bsouth_mean_3h_nt)],
        persistence_dst_nt = [Float64(baselines.persistence)],
        burton_dst_nt = [Float64(baselines.burton)],
        burton_full_dst_nt = [Float64(baselines.burton_full)],
        obrien_dst_nt = [Float64(baselines.obrien)],
        pred_dst_nt = [core_value],
        pred_dst_ci05_nt = [core_value - ANALOG_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT],
        pred_dst_ci95_nt = [core_value + ANALOG_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT],
    )
    add_inertial_sindy_features!(frame)
    available = Set(names(frame))
    for column in calibration.feature_names
        String(column) in available || error(
            "the analog ensemble correction feature frame omits $(String(column))",
        )
    end
    return NamedTuple{Tuple(calibration.feature_names)}(
        Tuple(Float64(frame[1, column]) for column in calibration.feature_names),
    )
end

"""
    analog_serving_t1r_features(art, raw_reported, latest_dst, drivers, memory, baselines,
                             model_steps, anchor_time) -> NamedTuple

Feature tuple of the analog-core refit: the deployed feature construction evaluated with the analog
ensemble's raw center in place of the frozen core.
"""
analog_serving_t1r_features(art::AnalogServingArtifacts, raw_reported::Real, latest_dst::Real,
                         drivers, memory, baselines, model_steps::Integer,
                         anchor_time::DateTime) =
    analog_serving_calibration_features(art.calibration, raw_reported, latest_dst, drivers, memory,
                                     baselines, model_steps, anchor_time)

"""
    analog_serving_frozen_center(art; inertial_sindy_calibration, issue_drv, anchor_dst_star, latest_dst,
                              memory, baselines, model_steps, anchor_time) -> NamedTuple

Frozen-tail inertial-SINDy center at `model_steps` hours: the blend partner of the lead-aware stage.

The frozen tail holds the issue driver — the record tagged `anchor - 1` — for every rollout step, so
the center is the deployed core rolled under a fixed driver, corrected by the deployed inertial-SINDy ridge
layer and projected to the physical range, with no safeguard operator. That is the quantity the
scored artifact blends against, and it is not the live engine's own `inertial_sindy_pred_dst_nt`, which admits
L1-measured hours into the rollout. Computing it here keeps the live blend equal to the scored one.

Returns `(raw, correction, center)`.
"""
function analog_serving_frozen_center(art::AnalogServingArtifacts;
                                   inertial_sindy_calibration::InertialSindyCalibration,
                                   issue_drv, anchor_dst_star::Real, latest_dst::Real,
                                   memory, baselines, model_steps::Integer,
                                   anchor_time::DateTime)
    steps = Int(model_steps)
    steps >= 1 || throw(ArgumentError("model step must be positive, got $steps"))
    trajectory = operational_core_forecast(art.core, Float64(anchor_dst_star), issue_drv, steps)
    raw = dst_star_to_dst(trajectory[steps], Float64(issue_drv.Pdyn))
    features = analog_serving_calibration_features(inertial_sindy_calibration, raw, latest_dst, issue_drv,
                                                memory, baselines, steps, anchor_time)
    correction = operational_inertial_sindy_correction(inertial_sindy_calibration, features)
    return (raw = raw, correction = correction,
            center = clamp(raw + correction, ANALOG_SERVING_DST_FLOOR_NT, ANALOG_SERVING_DST_CEIL_NT))
end

"""
    analog_serving_innovations_from_step1_centers(step1_centers, observed_dst) -> Dict{DateTime,Float64}

The matured one-step innovation of every anchor whose one-hour center is known and whose one-hour
target has since been observed: `Dst(anchor + 1 h) - center_1h(anchor)`.

`step1_centers` maps an anchor hour to the analog ensemble center at a one-hour model step *before* the error
layer acts, and `observed_dst` maps an hour to the observed Dst at that hour. An anchor whose
one-hour target has not matured, or whose inputs are not finite, contributes nothing, so the layer
sees only innovations that a forecaster could have scored.

This is the one definition both environments use: the live engine builds `step1_centers` from the
one-hour centers it logged and `observed_dst` from the current Kyoto series, and the offline
identity oracle builds both from the scored table. Keeping the rule here is what makes the live
error-layer chain provably the chain the candidate was scored under.
"""
function analog_serving_innovations_from_step1_centers(step1_centers::AbstractDict,
                                                    observed_dst::AbstractDict)
    innovations = Dict{DateTime,Float64}()
    for (anchor, center) in step1_centers
        key = DateTime(anchor)
        value = Float64(center)
        isfinite(value) || continue
        observed = get(observed_dst, key + Hour(1), nothing)
        observed === nothing && continue
        matured = Float64(observed)
        isfinite(matured) || continue
        innovations[key] = matured - value
    end
    return innovations
end

"""
    analog_serving_innovation_lags(anchor_time, innovations) -> NamedTuple

The six matured one-step innovations the error layer consumes, taken from `innovations`, a map from
an anchor hour to the observed-minus-forecast residual of the analog ensemble center issued at that hour for
one model step. The forecast issued at `t - j` matures at `t - j + 1`, so all six lags are observable
at the anchor. Returns `(values, ok)`; a single missing or non-finite lag makes the whole block
unavailable, and the layer then keeps the uncorrected center rather than an imputed one.
"""
function analog_serving_innovation_lags(anchor_time::DateTime, innovations::AbstractDict)
    values = Vector{Float64}(undef, ANALOG_SERVING_E_INNOVATION_LAGS)
    for lag in 1:ANALOG_SERVING_E_INNOVATION_LAGS
        value = get(innovations, anchor_time - Hour(lag), nothing)
        (value !== nothing && isfinite(Float64(value))) ||
            return (values = fill(NaN, ANALOG_SERVING_E_INNOVATION_LAGS), ok = false)
        values[lag] = Float64(value)
    end
    return (values = values, ok = true)
end

"Predict one standardised ridge layer on a single design row."
function _analog_serving_ridge_predict(model::AnalogServingRidge, design::AbstractVector{<:Real})
    length(design) == length(model.beta) || throw(DimensionMismatch(
        "the E1 layer expects $(length(model.beta)) inputs, received $(length(design))",
    ))
    row = reshape(Float64.(collect(design)), 1, length(model.beta))
    z = (row .- transpose(model.mean)) ./ transpose(model.scale)
    return (z * model.beta)[1] + model.intercept
end

"""
    analog_serving_center(art; raw_reported, latest_dst, anchor_drivers, memory, baselines,
                       model_steps, frozen_inertial_sindy, analog_features, anchor_time,
                       innovations = nothing) -> NamedTuple

Turn the raw analog center into the analog-continuation expert center (specification steps 3-5).

The analog-core refit of the deployed inertial-SINDy ridge layer is evaluated on the row's calibration
features and added to the raw core, projected to the physical range and taking no inertial-SINDy safeguard —
the safeguards were selected against the frozen tail and the development run scored this candidate
with them off. The corrected center is then blended with the frozen inertial-SINDy center by the lead-aware
weight of the model step, which is the quantity the error layer's innovation history is defined
against. Finally the step's error layer, when one was selected and six matured innovations exist,
adds a capped correction.

Returns the correction, the corrected center, the blended center (`center`, the value whose one-step
innovations feed later layers) and the final center, together with whether the error layer
acted.
"""
function analog_serving_center(art::AnalogServingArtifacts; raw_reported::Real, latest_dst::Real,
                            anchor_drivers, memory, baselines, model_steps::Integer,
                            frozen_inertial_sindy::Real, analog_features::AbstractVector{<:Real},
                            anchor_time::DateTime, innovations = nothing)
    step = Int(model_steps)
    slot = 1 <= step <= length(ANALOG_SERVING_STEP_SLOT) ? ANALOG_SERVING_STEP_SLOT[step] : 0
    slot == 0 && throw(ArgumentError(
        "the analog ensemble candidate does not cover a $(step) h model step",
    ))
    features = analog_serving_t1r_features(art, raw_reported, latest_dst, anchor_drivers, memory,
                                        baselines, step, anchor_time)
    correction = operational_inertial_sindy_correction(art.calibration, features)
    t1r = clamp(Float64(raw_reported) + correction, ANALOG_SERVING_DST_FLOOR_NT,
                ANALOG_SERVING_DST_CEIL_NT)
    weight = art.lat_weights[slot]
    frozen = Float64(frozen_inertial_sindy)
    isfinite(frozen) || throw(ArgumentError(
        "the lead-aware blend needs a finite frozen inertial-SINDy center",
    ))
    center = weight * t1r + (1 - weight) * frozen
    layer = art.e_layers[slot]
    delta = 0.0
    applied = false
    if layer.kind !== :identity && innovations !== nothing
        block = Float64.(collect(innovations))
        length(block) == ANALOG_SERVING_E_INNOVATION_LAGS || throw(DimensionMismatch(
            "the error layer needs $(ANALOG_SERVING_E_INNOVATION_LAGS) matured innovations",
        ))
        if all(isfinite, block)
            design = if layer.kind === :E1
                vcat(block, Float64[Float64(latest_dst), Float64(memory.dst_delta_1h_nt),
                                    _analog_serving_vbs(anchor_drivers)])
            else
                vcat(block, Float64.(collect(analog_features)), Float64[center])
            end
            raw_delta = layer.kind === :E1 ?
                _analog_serving_ridge_predict(layer.ridge, design) :
                gbm_predict(layer.boosted, reshape(design, 1, length(design)))[1]
            delta = clamp(raw_delta, -layer.cap, layer.cap)
            applied = true
        end
    end
    return (raw_reported = Float64(raw_reported), correction = correction, t1r_center = t1r,
            center = center, e_layer = String(layer.kind), e_layer_applied = applied,
            e_delta = delta, final = center + delta, model_step_hours = step,
            lat_weight = weight)
end

"Rectified coupling proxy of an issue driver, in mV/m."
@inline _analog_serving_vbs(drivers) =
    Float64(drivers.V) * max(0.0, -Float64(drivers.Bz)) / 1000.0
