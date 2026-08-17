# operational_v23_serving.jl — Operational V2.3 shadow forecast: deployable artifacts and the
# pure center-forming functions the live engine and the offline identity oracle share.
#
# The candidate this file serves is a SHADOW forecast. Its single-shot confirmatory scoring returned
# NO_GO, so the live engine computes and logs the center but never publishes it as the served forecast
# and never uses it for severity.
#
# The V2.3 study replaced the served V2.1 heuristic tail with an analog driver continuation, an
# analog-core refit of the deployed V2.1 ridge correction (T1r), a lead-aware blend toward the
# frozen V2.1 center (LAT), and a per-step error layer (E). Scoring that pipeline offline and
# serving it live are two different execution environments — an hourly OMNI archive versus a
# minute-cadence L1 feed — but they must produce the same number for the same information set.
#
# This file is the single implementation of the center. It takes the parts of the state that the
# two environments genuinely disagree about (how an hour of solar wind is measured) as arguments,
# and computes everything else once:
#
#   * `v23_serving_features`  — the 18 issue-time analog features from a driver history and the
#     two anchor Dst values (specification step 1);
#   * `v23_serving_members`   — the K nearest archive origins and the ensemble-mean raw center
#     obtained by rolling the frozen SINDy core once per member (step 2);
#   * `v23_serving_center`    — the T1r correction, the LAT blend and the E layer (steps 3-5).
#
# The only environment-specific input is the `step_driver` callback: it answers "what drives
# rollout step k, and was that hour measured at L1 by issue time?". The live engine answers with
# the ballistically propagated sub-hourly L1 window; the identity oracle answers with the hourly
# archive under the same L1 admission gate the served V2.1 tail uses. Everything downstream of
# that callback is shared code, which is what makes the offline identity check meaningful.
#
# Time convention (unchanged from `operational_v23_features.jl`): the anchor is the newest observed
# Dst hour `t`; the freshest driver record is the one tagged `t-1`, covering the at-Earth interval
# [t-1, t); rollout step k targets `t+k`.

import SHA
import JSON3

"Model steps the V2.3 candidate supports, in artifact column order."
const V23_SERVING_MODEL_STEPS = (1, 2, 3, 4, 6, 7)

"Slot of a model step in the V2.3 per-step artifact vectors; 0 marks an unsupported step."
const V23_SERVING_STEP_SLOT = let slots = zeros(Int, maximum(V23_SERVING_MODEL_STEPS))
    for (slot, step) in enumerate(V23_SERVING_MODEL_STEPS)
        slots[step] = slot
    end
    Tuple(slots)
end

"""
    v23_serving_identity(weight_set, k; shadow = false) -> String

Pipeline identity of a V2.3 deployment. It names every stage a reader needs to reproduce the center:
the frozen 20-candidate/11-active SINDy core, the L1 look-ahead admission, the analog driver
continuation with its distance weights and ensemble size, the analog-core refit of the V2.1 ridge
correction, the lead-aware blend toward the frozen V2.1 center, and the error layer.

`shadow = true` marks the label used for the columns of a shadow deployment: a V2.3 center that is
computed, logged and monitored but never served and never used for alerting. The single-shot
confirmatory scoring returned `NO_GO` for this candidate, so the shadow label is the only one the
product carries.
"""
v23_serving_identity(weight_set, k::Integer; shadow::Bool = false) =
    string(shadow ? "v2.3-shadow" : "v2.3",
           "+sindy20x11+L1A+ADC($(String(Symbol(weight_set))),K$(Int(k)))+T1rcal+LAT+E")

"""
    V23_SERVING_IDENTITY

Pipeline identity of the V2.3 candidate: the magnetic-weighted 25-member analog ensemble selected on
the development partition. A deployment directory that loads with a different identity is a different
model.
"""
const V23_SERVING_IDENTITY = v23_serving_identity(:magnetic, 25)

"""
    V23_SERVING_SHADOW_IDENTITY

Label the V2.3 columns carry in the served log. The candidate failed the confirmatory gates, so it is
integrated as a shadow forecast: recorded for monitoring, never served, never used for severity.
"""
const V23_SERVING_SHADOW_IDENTITY = v23_serving_identity(:magnetic, 25; shadow = true)

"Driver assumption recorded with a V2.3 shadow row."
const V23_SERVING_DRIVER_ASSUMPTION =
    "ballistically_propagated_l1_then_analog_driver_continuation_ensemble_then_analog_refit_" *
    "ridge_correction_then_lead_aware_frozen_blend_then_capped_error_layer"

"Number of matured one-step innovations the error layer consumes."
const V23_SERVING_E_INNOVATION_LAGS = 6

"Names of the innovation error-state inputs, in column order (E1 design)."
const V23_SERVING_E_FEATURE_NAMES = (
    :innovation_1h, :innovation_2h, :innovation_3h, :innovation_4h, :innovation_5h,
    :innovation_6h, :dst0, :ddst1, :vbs0,
)

"Cap of an error-layer correction at model step h (nT)."
v23_serving_e_cap(h::Integer) = 5.0 + 5.0 * Int(h)

"Sun-Earth L1 standoff (km); the ballistic transit lag is `L1_DIST_KM / V / 3600` hours."
const V23_SERVING_L1_DIST_KM = 1.5e6

"Ballistic L1 -> Earth transit time (h) for a solar-wind speed `V` (km/s)."
@inline v23_serving_transit_hours(V::Real) =
    (isfinite(V) && V > 0) ? V23_SERVING_L1_DIST_KM / Float64(V) / 3600.0 : 0.0

"Tolerance for the shipped-vs-recomputed analog standardisation check (feature units)."
const V23_SERVING_STATS_ATOL = 1e-9

"Physical projection applied to every reported V2.3 Dst quantity (nT)."
const V23_SERVING_DST_FLOOR_NT = -2000.0
const V23_SERVING_DST_CEIL_NT = 50.0

"Placeholder interval half-width the T1r feature frame carries; never read back."
const V23_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT = 1.0

"Files every V2.3 deployment directory must carry, independent of which layers were selected."
const V23_SERVING_REQUIRED_FILES = (
    "analog_frame_2010_2019.csv", "analog_origins.csv", "analog_feature_stats.csv",
    "t1r_calibration.csv", "lat_weights.csv", "e_layers.json",
    "selected_configuration.json",
)

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

"""
    V23ServingRidge

Deployed E1 layer: a ridge fitted on standardised inputs with an intercept. `mean` and `scale`
standardise the design, `beta` are the standardised slopes and `intercept` is the training-target
mean, exactly as the development fit stored them.
"""
struct V23ServingRidge
    mean::Vector{Float64}
    scale::Vector{Float64}
    beta::Vector{Float64}
    intercept::Float64
    lambda::Float64
end

"""
    V23ServingELayer

Error layer of one model step. `kind` is `:identity`, `:E1` (ridge) or `:E2` (boosted); `cap` is
the symmetric bound the correction is clamped to; `feature_names` records the design the fitted
object expects, so a schema drift fails at load instead of silently reordering columns.
"""
struct V23ServingELayer
    model_step_hours::Int
    kind::Symbol
    param::String
    cap::Float64
    feature_names::Vector{String}
    ridge::Union{Nothing,V23ServingRidge}
    boosted::Any
end

"""
    V23ServingArtifacts

Everything the V2.3 shadow center needs, loaded once and immutable afterwards: the hourly analog
frame and its driver lookup, the rebuilt analog archive with its standardisation, the analog-core
refit calibration, the lead-aware blend weights, the per-step error layers, and the frozen V2.1
core the ensemble is rolled through.
"""
struct V23ServingArtifacts
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
    calibration::OperationalV2Calibration
    lat_weights::Vector{Float64}
    e_layers::Vector{V23ServingELayer}
    selection::Dict{String,Any}
    manifest::DataFrame
    core::Any
    identity::String
end

Base.show(io::IO, art::V23ServingArtifacts) = print(
    io, "V23ServingArtifacts(", art.identity, ", origins=", length(art.origins),
    ", K=", art.k, ", dir=", art.dir, ")",
)

# ---------------------------------------------------------------------------
# Loading and verification
# ---------------------------------------------------------------------------

"SHA-256 of a regular non-symlink file, as lowercase hex."
function v23_serving_file_sha256(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.3 serving artifact must be a regular non-symlink file: $source",
    ))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"Rows of a V2.3 manifest whose `entry_type` matches `kind`."
function _v23_manifest_rows(manifest::DataFrame, kind::AbstractString)
    return [r for r in eachrow(manifest) if String(r.entry_type) == String(kind)]
end

"The single manifest row `(kind, name)`, or `nothing`."
function _v23_manifest_row(manifest::DataFrame, kind::AbstractString, name::AbstractString)
    hits = [r for r in _v23_manifest_rows(manifest, kind) if String(r.name) == String(name)]
    isempty(hits) && return nothing
    length(hits) == 1 || throw(ArgumentError(
        "V2.3 manifest repeats the entry ($kind, $name)",
    ))
    return hits[1]
end

"Parse a `key=value;key=value` manifest value field."
function _v23_manifest_pairs(value::AbstractString)
    out = Dict{String,String}()
    for piece in split(String(value), ';')
        isempty(strip(piece)) && continue
        parts = split(piece, '=', limit = 2)
        length(parts) == 2 || continue
        out[strip(String(parts[1]))] = strip(String(parts[2]))
    end
    return out
end

"""
    v23_serving_verify_manifest(dir; manifest = nothing) -> DataFrame

Check every `sha256` row of `dir/manifest.csv` against the file it names and check that every
required artifact carries such a row. Returns the manifest. A missing row is as much a failure as a
wrong digest: an unlisted file would be served without provenance.
"""
function v23_serving_verify_manifest(dir::AbstractString; manifest = nothing)
    directory = String(dir)
    path = joinpath(directory, "manifest.csv")
    table = manifest === nothing ?
        CSV.read(path, DataFrame; types = Dict("value" => String, "name" => String,
                                               "entry_type" => String)) : manifest
    for column in ("entry_type", "name", "count", "value")
        column in names(table) || throw(ArgumentError(
            "V2.3 manifest $path lacks the $column column",
        ))
    end
    hashed = Set{String}()
    for row in _v23_manifest_rows(table, "sha256")
        name = String(row.name)
        file = joinpath(directory, name)
        digest = v23_serving_file_sha256(file)
        digest == String(row.value) || error(
            "V2.3 serving artifact $name fails its manifest digest: expected $(row.value), " *
            "computed $digest",
        )
        size = Int(round(Float64(row.count)))
        filesize(file) == size || error(
            "V2.3 serving artifact $name has $(filesize(file)) bytes but the manifest records $size",
        )
        push!(hashed, name)
    end
    for required in V23_SERVING_REQUIRED_FILES
        required in hashed || error(
            "V2.3 manifest $path has no sha256 entry for the required artifact $required",
        )
    end
    return table
end

"""
    v23_serving_manifest_hashed_names(manifest) -> Set{String}

Artifact names the manifest carries a `sha256` row for. Only these names have been digest-verified
by `v23_serving_verify_manifest`, so an artifact named by a configuration file must be checked
against this set before it is parsed; otherwise a manifest with the artifact's digest row removed
would load the file without provenance.
"""
function v23_serving_manifest_hashed_names(manifest::DataFrame)
    return Set{String}(String(row.name) for row in _v23_manifest_rows(manifest, "sha256"))
end

"Read the shipped analog standardisation table into `(mean, sd)` in feature order."
function _v23_serving_read_stats(path::AbstractString)
    table = CSV.read(path, DataFrame)
    names_shipped = String.(table.feature_name)
    names_shipped == String.(collect(V23_FEATURE_NAMES)) || error(
        "shipped analog feature statistics are not in the V2.3 feature order: $path",
    )
    return (mean = Float64.(table.feature_mean), sd = Float64.(table.feature_sd))
end

"Read one deployed E1 ridge artifact."
function _v23_serving_read_e1(path::AbstractString, feature_names::Vector{String})
    table = CSV.read(path, DataFrame)
    String.(table.feature_name) == feature_names || error(
        "E1 artifact $path lists features $(String.(table.feature_name)) but the layer schema is " *
        "$(feature_names)",
    )
    intercepts = unique(Float64.(table.intercept_nt))
    lambdas = unique(Float64.(table.ridge_lambda))
    length(intercepts) == 1 || error("E1 artifact $path carries $(length(intercepts)) intercepts")
    length(lambdas) == 1 || error("E1 artifact $path carries $(length(lambdas)) ridge penalties")
    return V23ServingRidge(
        Float64.(table.feature_mean), Float64.(table.feature_scale),
        Float64.(table.standardised_coefficient), intercepts[1], lambdas[1],
    )
end

"""
Read the per-step error-layer records of a deployment directory.

`hashed` is the set of artifact names `v23_serving_verify_manifest` digest-checked. Every artifact
the configuration names must appear in it: the E-layer models are selected per step by
`e_layers.json` rather than by a fixed file list, so without this check a manifest whose E-layer
digest rows were removed would still load and serve those models unverified.
"""
function _v23_serving_read_e_layers(dir::AbstractString;
                                    hashed::Union{Nothing,AbstractSet{String}} = nothing)
    payload = JSON3.read(read(joinpath(dir, "e_layers.json"), String))
    steps = payload["steps"]
    length(steps) == length(V23_SERVING_MODEL_STEPS) || error(
        "e_layers.json describes $(length(steps)) steps but the candidate covers " *
        "$(length(V23_SERVING_MODEL_STEPS))",
    )
    layers = V23ServingELayer[]
    for (slot, record) in enumerate(steps)
        step = Int(record["model_step_hours"])
        step == V23_SERVING_MODEL_STEPS[slot] || error(
            "e_layers.json step $slot is $(step) h but the candidate covers " *
            "$(V23_SERVING_MODEL_STEPS[slot]) h",
        )
        kind = Symbol(String(record["layer"]))
        cap = Float64(record["correction_cap_nt"])
        cap == v23_serving_e_cap(step) || error(
            "e_layers.json caps the $(step) h layer at $cap nT; the published cap is " *
            "$(v23_serving_e_cap(step)) nT",
        )
        names_json = String[String(n) for n in record["feature_names"]]
        artifact = record["artifact"]
        param = record["param"] === nothing ? "" : String(record["param"])
        if kind === :identity
            artifact === nothing || error("an identity layer must not name an artifact")
            push!(layers, V23ServingELayer(step, :identity, param, cap, String[], nothing, nothing))
            continue
        end
        kind in (:E1, :E2) || error("unknown V2.3 error layer $(kind) at $(step) h")
        artifact === nothing && error("the $(kind) layer at $(step) h names no artifact")
        name = String(artifact)
        hashed === nothing || name in hashed || error(
            "the $(kind) layer at $(step) h names $name, which carries no verified manifest " *
            "digest; refusing to serve an unverified error-layer artifact",
        )
        path = joinpath(dir, name)
        isfile(path) || error("V2.3 error-layer artifact is missing: $path")
        expected = kind === :E1 ?
            String.(collect(V23_SERVING_E_FEATURE_NAMES)) :
            vcat(String.(collect(V23_SERVING_E_FEATURE_NAMES))[1:V23_SERVING_E_INNOVATION_LAGS],
                 String.(collect(V23_FEATURE_NAMES)), ["center_dst_nt"])
        names_json == expected || error(
            "the $(kind) layer at $(step) h lists an unexpected design: $(names_json)",
        )
        if kind === :E1
            push!(layers, V23ServingELayer(step, :E1, param, cap, names_json,
                                           _v23_serving_read_e1(path, names_json), nothing))
        else
            model = v23_load(path)
            stored = String.(model.info[:feature_names])
            stored == names_json || error(
                "the boosted layer at $(step) h was fitted on $(stored) but the record lists " *
                "$(names_json)",
            )
            push!(layers, V23ServingELayer(step, :E2, param, cap, names_json, nothing, model))
        end
    end
    return layers
end

"Driver lookup of the shipped analog frame; a record enters only when every channel is finite."
function v23_serving_frame_lookup(frame::DataFrame)
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

"""
    load_v23_serving_artifacts(dir; core, verify_hashes = true) -> V23ServingArtifacts

Load and verify a V2.3 deployment directory.

Every shipped file is checked against its SHA-256 in `manifest.csv` before it is parsed. The analog
archive is then rebuilt rather than trusted: the issue-time features of every shipped origin are
recomputed from the shipped hourly frame, each origin is re-checked against the eligibility rule the
scoring run used (a complete causal history and seven continuable driver records), the origin count
is compared with the count the scoring run recorded, and the feature standardisation is recomputed
and compared with the shipped table to `V23_SERVING_STATS_ATOL`. A deployment whose archive does not
reproduce the scored archive therefore fails at load, not at the first shadow forecast.

The origin identities are shipped rather than derived because archive membership is not a function
of the hourly frame alone: an origin is an archive member only if it was a V2.1 calibration anchor,
which additionally requires a quality-flagged, non-gap-filled L1 driver record at `t-1`. That
condition lives in the OMNI quality flags, which the causal hourly frame does not carry.
"""
function load_v23_serving_artifacts(dir::AbstractString;
                                    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION),
                                    verify_hashes::Bool = true)
    directory = abspath(String(dir))
    isdir(directory) || error("V2.3 deployment directory not found: $directory")
    manifest = CSV.read(joinpath(directory, "manifest.csv"), DataFrame;
                        types = Dict("entry_type" => String, "name" => String, "value" => String))
    verify_hashes && v23_serving_verify_manifest(directory; manifest = manifest)

    selection = Dict{String,Any}(JSON3.read(
        read(joinpath(directory, "selected_configuration.json"), String), Dict{String,Any},
    ))
    String(get(selection, "family", "")) == "T1r" || error(
        "V2.3 deployment holds family $(get(selection, "family", "?")); the integrated candidate is T1r",
    )
    Bool(get(selection, "safeguards", true)) == false || error(
        "V2.3 deployment records safeguards = true; the scored T1r center takes no V2.1 safeguards",
    )
    params = get(selection, "params", Dict{String,Any}())
    weight_set = Symbol(String(get(params, "weight_set", "magnetic")))
    k = Int(selection["k"])
    k >= 1 || error("V2.3 deployment records a non-positive ensemble size")

    calibration = read_operational_v2_calibration(joinpath(directory, "t1r_calibration.csv"))
    lat_table = CSV.read(joinpath(directory, "lat_weights.csv"), DataFrame)
    Int.(lat_table.model_step_hours) == collect(V23_SERVING_MODEL_STEPS) || error(
        "lat_weights.csv does not list the candidate model steps in order",
    )
    lat_weights = Float64.(lat_table.selected_weight)
    all(w -> isfinite(w) && 0.0 <= w <= 1.0, lat_weights) || error(
        "lead-aware blend weights must lie in [0, 1]",
    )
    selected_lat = get(selection, "lat_weights", nothing)
    selected_lat === nothing || Float64.(collect(selected_lat)) == lat_weights || error(
        "lat_weights.csv disagrees with the selected configuration",
    )

    e_layers = _v23_serving_read_e_layers(
        directory;
        hashed = verify_hashes ? v23_serving_manifest_hashed_names(manifest) : nothing,
    )

    frame = CSV.read(joinpath(directory, "analog_frame_2010_2019.csv"), DataFrame;
                     types = Dict("time_utc" => DateTime))
    issorted(frame.time_utc) || error("the shipped analog frame is not chronological")
    lookup = v23_serving_frame_lookup(frame)

    origins_table = CSV.read(joinpath(directory, "analog_origins.csv"), DataFrame;
                             types = Dict("origin_time_utc" => DateTime))
    origins = collect(DateTime, origins_table.origin_time_utc)
    issorted(origins) || error("the shipped analog origins are not chronological")
    allunique(origins) || error("the shipped analog origins repeat a timestamp")
    length(origins) >= k || error(
        "the shipped analog archive holds $(length(origins)) origins, fewer than K = $k",
    )

    archive_row = _v23_manifest_row(manifest, "analog_archive", "origins")
    archive_row === nothing && error("V2.3 manifest records no analog_archive origin count")
    expected_origins = Int(round(Float64(archive_row.count)))
    length(origins) == expected_origins || error(
        "the shipped analog archive holds $(length(origins)) origins but the scoring run " *
        "recorded $expected_origins",
    )
    bounds = _v23_manifest_pairs(String(archive_row.value))
    haskey(bounds, "first") && DateTime(bounds["first"]) == first(origins) || error(
        "the shipped analog archive starts at $(first(origins)); the manifest records " *
        "$(get(bounds, "first", "?"))",
    )
    haskey(bounds, "last") && DateTime(bounds["last"]) == last(origins) || error(
        "the shipped analog archive ends at $(last(origins)); the manifest records " *
        "$(get(bounds, "last", "?"))",
    )

    features, ok = v23_feature_matrix(frame, origins)
    all(ok) || error(
        "$(count(!, ok)) shipped analog origins do not have a complete causal history in the " *
        "shipped frame; the archive cannot be rebuilt",
    )
    continuable = v23_analog_origin_ok(lookup, origins)
    all(continuable) || error(
        "$(count(!, continuable)) shipped analog origins cannot supply a seven-step driver " *
        "continuation from the shipped frame",
    )

    stats = v23_feature_stats(features)
    shipped = _v23_serving_read_stats(joinpath(directory, "analog_feature_stats.csv"))
    worst_mean = maximum(abs.(stats.mean .- shipped.mean))
    worst_sd = maximum(abs.(stats.sd .- shipped.sd))
    max(worst_mean, worst_sd) <= V23_SERVING_STATS_ATOL || error(
        "the rebuilt analog standardisation disagrees with the shipped table by " *
        "$(max(worst_mean, worst_sd)); the deployed archive is not the scored archive",
    )

    weights = v23_weights(weight_set)
    standardised = v23_standardize(features, stats.mean, stats.sd)
    return V23ServingArtifacts(
        directory, frame, lookup, origins, features, standardised,
        stats.mean, stats.sd, weight_set, weights, k, V23_ANALOG_EXCLUSION_HOURS,
        calibration, lat_weights, e_layers, selection, manifest, core,
        v23_serving_identity(weight_set, k),
    )
end

# ---------------------------------------------------------------------------
# Step 1 — issue-time analog features
# ---------------------------------------------------------------------------

"""
    v23_serving_features(art, anchor_time, driver_history, dst_anchor, dst_prev) -> NamedTuple

Issue-time analog feature vector for the anchor hour `anchor_time` (specification step 1).

`driver_history[j]` is the at-Earth driver record tagged `anchor_time - j` hours — the hourly mean
covering `[anchor_time - j, anchor_time - j + 1)` — or `nothing` when that hour is not available.
At least `V23_HISTORY_LAGS_H` entries must be supplied; the further entries, up to
`V23_SOUTH_RUN_CAP_H`, are consumed only by the consecutive-southward run-length feature and may be
absent, in which case the run length truncates exactly as it does at the start of the archive.
`dst_anchor` and `dst_prev` are the observed Dst at `anchor_time` and one hour earlier.

Returns `(features, ok, reason)`. `ok = false` leaves `features` non-finite and names the missing
input in `reason`, which is what the shadow column records when no analog center is available.

The construction goes through `v23_feature_matrix` on a short frame rather than reimplementing the
feature definitions, so the live key and the archive keys are produced by one code path.
"""
function v23_serving_features(art::V23ServingArtifacts, anchor_time::DateTime, driver_history,
                              dst_anchor::Real, dst_prev::Real)
    lags = length(driver_history)
    lags >= V23_HISTORY_LAGS_H || throw(ArgumentError(
        "the V2.3 analog key needs at least $(V23_HISTORY_LAGS_H) driver records, got $lags",
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
    X, ok = v23_feature_matrix(mini, [anchor_time])
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
                        1:V23_HISTORY_LAGS_H)
        bad === nothing ? "incomplete_analog_key" : "missing_driver_lag$(bad)"
    end
    return (features = values, ok = false, reason = reason)
end

# ---------------------------------------------------------------------------
# Step 2 — analog retrieval and the ensemble raw center
# ---------------------------------------------------------------------------

"""
    v23_serving_step_driver_from_frame(lookup, anchor_time, issue_drv) -> Function

Rollout-step driver policy of the hourly archive: the L1 admission gate the served V2.1 tail uses,
expressed against an hourly driver lookup. Step `k` is admitted when the issue-time transit window
covers it (`k <= floor(transit(V_issue))`) and the arriving record's own speed still covers it; an
in-window record that fails either test freezes the last admitted driver, and steps past the window
are handed to the analog member.

This is the offline half of the shadow path's environment split. The live engine supplies the
minute-cadence equivalent instead, and every stage downstream is shared.
"""
function v23_serving_step_driver_from_frame(lookup::AbstractDict, anchor_time::DateTime, issue_drv)
    kdelta = floor(Int, v23_serving_transit_hours(issue_drv.V))
    return function (k::Int, last_known)
        k <= kdelta || return (driver = last_known, l1_measured = false)
        record = get(lookup, anchor_time + Hour(k - 1), nothing)
        admitted = record !== nothing && k <= v23_serving_transit_hours(record.V)
        return (driver = admitted ? record : last_known, l1_measured = true)
    end
end

"""
    v23_serving_members(art, features; anchor_time, issue_drv, anchor_dst_star, model_steps,
                        step_driver) -> NamedTuple

Retrieve the K nearest admissible archive origins for the analog key `features` and form the raw
V2.3 center at `model_steps` hours (specification step 2).

Each retrieved origin becomes one ensemble member. A member rolls the frozen SINDy core from the
anchor state: steps the environment reports as L1-measured take the measured driver and advance the
last-known driver, and later steps take the member's analog continuation, which applies the origin's
archived speed and log-density increments to the query's own issue driver and copies the magnetic
field. Each member contributes `Dst* + 7.26*sqrt(Pdyn) - 11` evaluated with its own final-step
pressure, and the raw center is the member mean.

Returns the member origins, the per-member centers, the unprojected mean and the reported raw
center, which is the mean projected to the physical range.
"""
function v23_serving_members(art::V23ServingArtifacts, features::AbstractVector{<:Real};
                             anchor_time::DateTime, issue_drv, anchor_dst_star::Real,
                             model_steps::Integer, step_driver)
    length(features) == V23_FEATURE_COUNT || throw(DimensionMismatch(
        "the V2.3 analog key has $(length(features)) entries, expected $(V23_FEATURE_COUNT)",
    ))
    all(isfinite, features) || throw(ArgumentError(
        "the V2.3 analog key must be finite; check `ok` before retrieval",
    ))
    steps = Int(model_steps)
    steps >= 1 || throw(ArgumentError("model step must be positive, got $steps"))
    query = v23_standardize(reshape(Float64.(collect(features)), 1, V23_FEATURE_COUNT),
                            art.feature_mean, art.feature_sd)
    neighbours = v23_knn(query, [anchor_time], art.archive_standardised, art.origins, art.k;
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
                v23_member_driver(issue_drv, art.lookup, origins[m], k)
            end
            final_drv = drv
            assimilation_predict!(filter, drv)
            filter.mean[1] = clamp(filter.mean[1], V23_SERVING_DST_FLOOR_NT,
                                   V23_SERVING_DST_CEIL_NT)
        end
        member_pred[m] = current_dst(filter) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
    end
    raw = mean(member_pred)
    return (origins = origins, k = art.k, member_pred = member_pred, raw = raw,
            raw_reported = clamp(raw, V23_SERVING_DST_FLOOR_NT, V23_SERVING_DST_CEIL_NT))
end

# ---------------------------------------------------------------------------
# Steps 3-5 — correction, lead-aware blend, error layer
# ---------------------------------------------------------------------------

"""
    v23_serving_calibration_features(calibration, core_dst, latest_dst, drivers, memory,
                                     baselines, model_steps, anchor_time) -> NamedTuple

Calibration feature tuple of one `(anchor, model step)` row for the ridge layer `calibration`,
evaluated with `core_dst` as the raw core the correction is applied to.

The frame is handed to `add_operational_v2_features!`, the package path the deployed calibration was
fitted with, rather than to a re-derivation of the feature definitions: the five core-dependent
features (the three baseline differences, the baseline spread and the lead interaction) then come out
of the same code for the analog core and for the frozen core, and the issue-time state and memory
lags pass through untouched. The six-hour memory columns the fit does not use are the only
quantities a single-row frame cannot reproduce, and they are absent from every deployed feature list.
"""
function v23_serving_calibration_features(calibration::OperationalV2Calibration, core_dst::Real,
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
        pred_dst_ci05_nt = [core_value - V23_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT],
        pred_dst_ci95_nt = [core_value + V23_SERVING_T1R_PLACEHOLDER_HALF_WIDTH_NT],
    )
    add_operational_v2_features!(frame)
    available = Set(names(frame))
    for column in calibration.feature_names
        String(column) in available || error(
            "the V2.3 correction feature frame omits $(String(column))",
        )
    end
    return NamedTuple{Tuple(calibration.feature_names)}(
        Tuple(Float64(frame[1, column]) for column in calibration.feature_names),
    )
end

"""
    v23_serving_t1r_features(art, raw_reported, latest_dst, drivers, memory, baselines,
                             model_steps, anchor_time) -> NamedTuple

Feature tuple of the analog-core refit: the deployed feature construction evaluated with the analog
ensemble's raw center in place of the frozen core.
"""
v23_serving_t1r_features(art::V23ServingArtifacts, raw_reported::Real, latest_dst::Real,
                         drivers, memory, baselines, model_steps::Integer,
                         anchor_time::DateTime) =
    v23_serving_calibration_features(art.calibration, raw_reported, latest_dst, drivers, memory,
                                     baselines, model_steps, anchor_time)

"""
    v23_serving_frozen_center(art; v2_1_calibration, issue_drv, anchor_dst_star, latest_dst,
                              memory, baselines, model_steps, anchor_time) -> NamedTuple

Frozen-tail V2.1 center at `model_steps` hours: the blend partner of the lead-aware stage.

The frozen tail holds the issue driver — the record tagged `anchor - 1` — for every rollout step, so
the center is the deployed core rolled under a fixed driver, corrected by the deployed V2.1 ridge
layer and projected to the physical range, with no safeguard operator. That is the quantity the
scored artifact blends against, and it is not the live engine's own `v2_pred_dst_nt`, which admits
L1-measured hours into the rollout. Computing it here keeps the live blend equal to the scored one.

Returns `(raw, correction, center)`.
"""
function v23_serving_frozen_center(art::V23ServingArtifacts;
                                   v2_1_calibration::OperationalV2Calibration,
                                   issue_drv, anchor_dst_star::Real, latest_dst::Real,
                                   memory, baselines, model_steps::Integer,
                                   anchor_time::DateTime)
    steps = Int(model_steps)
    steps >= 1 || throw(ArgumentError("model step must be positive, got $steps"))
    trajectory = operational_core_forecast(art.core, Float64(anchor_dst_star), issue_drv, steps)
    raw = dst_star_to_dst(trajectory[steps], Float64(issue_drv.Pdyn))
    features = v23_serving_calibration_features(v2_1_calibration, raw, latest_dst, issue_drv,
                                                memory, baselines, steps, anchor_time)
    correction = operational_v2_correction(v2_1_calibration, features)
    return (raw = raw, correction = correction,
            center = clamp(raw + correction, V23_SERVING_DST_FLOOR_NT, V23_SERVING_DST_CEIL_NT))
end

"""
    v23_serving_innovations_from_step1_centers(step1_centers, observed_dst) -> Dict{DateTime,Float64}

The matured one-step innovation of every anchor whose one-hour center is known and whose one-hour
target has since been observed: `Dst(anchor + 1 h) - center_1h(anchor)`.

`step1_centers` maps an anchor hour to the V2.3 center at a one-hour model step *before* the error
layer acts, and `observed_dst` maps an hour to the observed Dst at that hour. An anchor whose
one-hour target has not matured, or whose inputs are not finite, contributes nothing, so the layer
sees only innovations that a forecaster could have scored.

This is the one definition both environments use: the live engine builds `step1_centers` from the
one-hour centers it logged and `observed_dst` from the current Kyoto series, and the offline
identity oracle builds both from the scored table. Keeping the rule here is what makes the live
error-layer chain provably the chain the candidate was scored under.
"""
function v23_serving_innovations_from_step1_centers(step1_centers::AbstractDict,
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
    v23_serving_innovation_lags(anchor_time, innovations) -> NamedTuple

The six matured one-step innovations the error layer consumes, taken from `innovations`, a map from
an anchor hour to the observed-minus-forecast residual of the V2.3 center issued at that hour for
one model step. The forecast issued at `t - j` matures at `t - j + 1`, so all six lags are observable
at the anchor. Returns `(values, ok)`; a single missing or non-finite lag makes the whole block
unavailable, and the layer then keeps the uncorrected center rather than an imputed one.
"""
function v23_serving_innovation_lags(anchor_time::DateTime, innovations::AbstractDict)
    values = Vector{Float64}(undef, V23_SERVING_E_INNOVATION_LAGS)
    for lag in 1:V23_SERVING_E_INNOVATION_LAGS
        value = get(innovations, anchor_time - Hour(lag), nothing)
        (value !== nothing && isfinite(Float64(value))) ||
            return (values = fill(NaN, V23_SERVING_E_INNOVATION_LAGS), ok = false)
        values[lag] = Float64(value)
    end
    return (values = values, ok = true)
end

"Predict one standardised ridge layer on a single design row."
function _v23_serving_ridge_predict(model::V23ServingRidge, design::AbstractVector{<:Real})
    length(design) == length(model.beta) || throw(DimensionMismatch(
        "the E1 layer expects $(length(model.beta)) inputs, received $(length(design))",
    ))
    row = reshape(Float64.(collect(design)), 1, length(model.beta))
    z = (row .- transpose(model.mean)) ./ transpose(model.scale)
    return (z * model.beta)[1] + model.intercept
end

"""
    v23_serving_center(art; raw_reported, latest_dst, anchor_drivers, memory, baselines,
                       model_steps, frozen_v2_1, analog_features, anchor_time,
                       innovations = nothing) -> NamedTuple

Turn the raw analog center into the V2.3 shadow center (specification steps 3-5).

The analog-core refit of the deployed V2.1 ridge layer is evaluated on the row's calibration
features and added to the raw core, projected to the physical range and taking no V2.1 safeguard —
the safeguards were selected against the frozen tail and the development run scored this candidate
with them off. The corrected center is then blended with the frozen V2.1 center by the lead-aware
weight of the model step, which is the quantity the error layer's innovation history is defined
against. Finally the step's error layer, when one was selected and six matured innovations exist,
adds a capped correction.

Returns the correction, the corrected center, the blended center (`center`, the value whose one-step
innovations feed later layers) and the shadow center (`final`), together with whether the error layer
acted.
"""
function v23_serving_center(art::V23ServingArtifacts; raw_reported::Real, latest_dst::Real,
                            anchor_drivers, memory, baselines, model_steps::Integer,
                            frozen_v2_1::Real, analog_features::AbstractVector{<:Real},
                            anchor_time::DateTime, innovations = nothing)
    step = Int(model_steps)
    slot = 1 <= step <= length(V23_SERVING_STEP_SLOT) ? V23_SERVING_STEP_SLOT[step] : 0
    slot == 0 && throw(ArgumentError(
        "the V2.3 candidate does not cover a $(step) h model step",
    ))
    features = v23_serving_t1r_features(art, raw_reported, latest_dst, anchor_drivers, memory,
                                        baselines, step, anchor_time)
    correction = operational_v2_correction(art.calibration, features)
    t1r = clamp(Float64(raw_reported) + correction, V23_SERVING_DST_FLOOR_NT,
                V23_SERVING_DST_CEIL_NT)
    weight = art.lat_weights[slot]
    frozen = Float64(frozen_v2_1)
    isfinite(frozen) || throw(ArgumentError(
        "the lead-aware blend needs a finite frozen V2.1 center",
    ))
    center = weight * t1r + (1 - weight) * frozen
    layer = art.e_layers[slot]
    delta = 0.0
    applied = false
    if layer.kind !== :identity && innovations !== nothing
        block = Float64.(collect(innovations))
        length(block) == V23_SERVING_E_INNOVATION_LAGS || throw(DimensionMismatch(
            "the error layer needs $(V23_SERVING_E_INNOVATION_LAGS) matured innovations",
        ))
        if all(isfinite, block)
            design = if layer.kind === :E1
                vcat(block, Float64[Float64(latest_dst), Float64(memory.dst_delta_1h_nt),
                                    _v23_serving_vbs(anchor_drivers)])
            else
                vcat(block, Float64.(collect(analog_features)), Float64[center])
            end
            raw_delta = layer.kind === :E1 ?
                _v23_serving_ridge_predict(layer.ridge, design) :
                v23_predict(layer.boosted, reshape(design, 1, length(design)))[1]
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
@inline _v23_serving_vbs(drivers) =
    Float64(drivers.V) * max(0.0, -Float64(drivers.Bz)) / 1000.0
