# operational_v24_serving.jl — Operational V2.4e served forecast: the deployable artifact bundle and
# the pure center-forming functions the live engine and the offline identity oracle share.
#
# What V2.4e is (rolling-origin study `v2_4_decision.csv`, research plan Amendment A3): a non-negative
# least-squares super-learner over ten causal experts, fitted per (model step, issue-time regime,
# ring-current depth bin) with a 0.60 mass floor on the SINDy family, where the family now counts the
# fixed static V2.2 stack because that product is itself a composition of the deployed SINDy
# operators. There is no boosted residual layer, no innovation chain and no point-forecast guard: the
# static stack entered the stack as an expert, so the optimiser recovers the physics composition in
# the deep cells where it is best instead of a hard minimum imposing it. Intervals are split-conformal
# half-widths stratified by (step, depth bin).
#
# The guard machinery is retained and is driven by the bundle's own `guard.json`, so a later bundle
# whose selection rule returns a guarded variant can be served by this same code path without a source
# change; the currently deployed bundle records `guard_applied = false` and the loader refuses any
# bundle whose guard record and guard reference disagree.
#
# Point forecast versus alerting. Depth safety did not leave the product with the guard: the published
# severity and watch state are taken through `v24_serving_depth_safe_center`, the deepest of the V2.4e
# center, the static V2.2 stack center and the V2.1 served center. The stack may blend toward a
# shallower state, and that must never lower a warning either predecessor would have raised.
#
# Why the center lives here rather than in the engine. The rolling-origin study scored V2.4e on an
# hourly OMNI archive; the live engine forms it from a minute-cadence L1 feed mapped to Earth-arrival
# hours. Those two environments genuinely disagree about how an hour of solar wind is measured and
# about nothing else, so every stage downstream of that disagreement is written once, here, and driven
# by both. `validation/operational/v2_4_serving_identity.jl` drives these functions with archive
# inputs and requires the scored fold table to be reproduced to 1e-9 nT.
#
# Expert order (fixed; the stack weight columns are read in this order):
#   1 served_v2_1   2 frozen_v2_1   3 t1r_analog        <- SINDy family, with expert 10, carrying 0.60
#   4 persistence   5 burton        6 burton_full
#   7 obrien        8 direct_gbm    9 climatology
#  10 static_v2_2                                       <- the fixed 2010-2017 regime stack
#
# Time convention (unchanged from `operational_v23_features.jl`): the anchor is the newest observed
# Dst hour `t`, the freshest driver record is the one tagged `t-1` covering `[t-1, t)`, and rollout
# step `k` targets `t+k`.

import SHA
import JSON3

"Model steps the V2.4e product supports, in artifact column order."
const V24_SERVING_MODEL_STEPS = (1, 2, 3, 4, 6, 7)

"Slot of a model step in the V2.4 per-step artifact vectors; 0 marks an unsupported step."
const V24_SERVING_STEP_SLOT = let slots = zeros(Int, maximum(V24_SERVING_MODEL_STEPS))
    for (slot, step) in enumerate(V24_SERVING_MODEL_STEPS)
        slots[step] = slot
    end
    Tuple(slots)
end

"""
    V24_SERVED_IDENTITY

Served-pipeline identity of the V2.4e product. It names every stage that can move the published
center: the frozen 20-candidate/11-active SINDy core, the ten-expert super-learner with the
SINDy-family mass floor, and the conformal interval. There is no guard stage in this identity because
the deployed bundle serves the stack center unmodified; a bundle that loads under a different identity
is a different model and is refused.
"""
const V24_SERVED_IDENTITY = "v2.4+sindy20x11+superlearner10floor+conformal"

"Study variant name of the served center, as the bundle records it and the study scored it."
const V24_SERVED_VARIANT = "v2_4e"

"Driver assumption recorded with a V2.4e served row."
const V24_SERVED_DRIVER_ASSUMPTION =
    "ballistically_propagated_l1_then_ten_expert_nnls_superlearner_with_sindy_family_floor_over_" *
    "served_frozen_analog_and_the_static_regime_stack_then_depth_stratified_conformal_interval"

"""
    V24_SERVING_EXPERTS

The ten super-learner experts in fixed weight-column order. The first three and the last are the
SINDy family that carries the mass floor: `static_v2_2` is the fixed 2010–2017 regime stack over the
deployed SINDy operators, so counting it in the family keeps the floor's meaning while letting the
optimiser use that composition where it is strongest.
"""
const V24_SERVING_EXPERTS = (
    :served_v2_1, :frozen_v2_1, :t1r_analog,
    :persistence, :burton, :burton_full, :obrien, :direct_gbm, :climatology,
    :static_v2_2,
)
const V24_SERVING_EXPERT_COUNT = length(V24_SERVING_EXPERTS)

"Expert indices carrying the SINDy-family mass floor (served, frozen, analog, static stack)."
const V24_SERVING_SINDY_FAMILY = (1, 2, 3, V24_SERVING_EXPERT_COUNT)

"Expert slot of the static V2.2 stack; it is also the guard reference when a bundle enables a guard."
const V24_SERVING_STATIC_EXPERT = V24_SERVING_EXPERT_COUNT

"SINDy-family mass floor of the served (floor-constrained) stack."
const V24_SERVING_SINDY_FLOOR = 0.60

"Issue-time regimes of `operational_v22_regime`, and the pooled fallback label."
const V24_SERVING_REGIMES = (:quiet, :active_deepening, :recovery)
const V24_SERVING_POOLED = :pooled

"Ring-current depth bins of the stack cells and the conformal strata."
const V24_SERVING_DEPTH_BINS = (:shallow, :moderate, :deep)

"Upper edge of the moderate depth bin (nT); above it a row is shallow."
const V24_SERVING_DEPTH_MODERATE_NT = -30.0

"Upper edge of the deep depth bin (nT); at or below it a row is deep."
const V24_SERVING_DEPTH_DEEP_NT = -70.0

"""
Deepening-cell thresholds. They label the deepening state that is logged with every served row, and
they are the thresholds a bundle's optional point-forecast guard would act in.
"""
const V24_SERVING_GUARD_RATE_NT_PER_H = -15.0
const V24_SERVING_GUARD_DEPTH_NT = -50.0

"""
Guard reference the served code path accepts when a bundle enables the point-forecast guard, and the
value `guard.json` must record when it does not.
"""
const V24_SERVING_GUARD_REFERENCE = "static_v2_2"
const V24_SERVING_GUARD_REFERENCE_NONE = "none"

"Nominal coverage of the served conformal interval."
const V24_SERVING_COVERAGE = 0.90

"Physical projection applied to every reported V2.4 Dst center (nT)."
const V24_SERVING_DST_FLOOR_NT = -2000.0
const V24_SERVING_DST_CEIL_NT = 50.0

"Tolerance on the shipped stack weights' unit mass."
const V24_SERVING_WEIGHT_SUM_ATOL = 1e-9

"Tolerance on the shipped-versus-recomputed analog standardisation (feature units)."
const V24_SERVING_STATS_ATOL = 1e-9

"Deepest Dst lag the direct-GBM design reads, so the live Dst history has to reach back this far."
const V24_SERVING_DST_LAG_MAX_H = maximum(V23_DIRECT_DST_LAG_HOURS)

"Driver-history depth the V2.4 feature blocks need: the analog run-length window covers the rest."
const V24_SERVING_DRIVER_LAGS_H = V23_SOUTH_RUN_CAP_H

"Stack-weight variant the served center reads: the ten-expert floor-constrained fit."
const V24_SERVING_STACK_VARIANT = "L1e"

"Files every V2.4 bundle must carry, independent of which steps were fitted."
const V24_SERVING_REQUIRED_FILES = (
    "stack_weights.csv", "conformal.csv", "climatology_tau.csv", "t1r_calibration.csv",
    "analog_frame.csv", "analog_origins.csv", "analog_feature_stats.csv",
    "guard.json", "selected.json",
)

"Per-step direct-GBM artifact name."
v24_serving_direct_file(step::Integer) = "direct_gbm_step$(Int(step)).bson"

"Identity string of the analog sub-artifact, so a V2.3 deployment can never be mistaken for it."
const V24_SERVING_ANALOG_IDENTITY = "v2.4-analog+ADC(magnetic,K25)+T1rcal"

"Analog ensemble configuration of expert E3, fixed by the study."
const V24_SERVING_ANALOG_K = 25
const V24_SERVING_ANALOG_WEIGHT_SET = :magnetic

# ---------------------------------------------------------------------------
# Cell geometry, depth bins and the guard
# ---------------------------------------------------------------------------

"""
    v24_serving_depth_bin(latest_dst) -> Symbol

Depth bin of a causal issue-time ring-current value. The edges are closed from below
(`latest <= -30` is at least moderate) so a row sitting exactly on an edge belongs to the deeper
bin, which is the safe direction for a guard and for an interval. A non-finite value is treated as
`:shallow`, which is the neutral cell: it selects the coarsest evidence rather than promoting an
unmeasured state into the deep stratum, and the caller has already refused to serve on a non-finite
anchor Dst.
"""
function v24_serving_depth_bin(latest_dst::Real)
    value = Float64(latest_dst)
    isfinite(value) || return :shallow
    value <= V24_SERVING_DEPTH_DEEP_NT && return :deep
    value <= V24_SERVING_DEPTH_MODERATE_NT && return :moderate
    return :shallow
end

"""
    v24_serving_cell_chain(step, regime, depth) -> Vector{Tuple{Int,Symbol,Symbol}}

Cell keys a row may use, from the most specific to the coarsest: its own `(regime, depth)` cell, the
regime-pooled cell over every depth, then the fully pooled cell. This is the only place the fallback
order is written down, and it restates the study's chain exactly.
"""
v24_serving_cell_chain(step::Integer, regime::Symbol, depth::Symbol) =
    Tuple{Int,Symbol,Symbol}[
        (Int(step), regime, depth),
        (Int(step), regime, V24_SERVING_POOLED),
        (Int(step), V24_SERVING_POOLED, V24_SERVING_POOLED),
    ]

"""
    v24_serving_cell_grid(step) -> Vector{Tuple{Int,Symbol,Symbol}}

Every cell key the stack may hold at one model step, in the order the weight table writes them, so a
reader does not depend on dictionary order.
"""
function v24_serving_cell_grid(step::Integer)
    keys = Tuple{Int,Symbol,Symbol}[(Int(step), V24_SERVING_POOLED, V24_SERVING_POOLED)]
    for regime in V24_SERVING_REGIMES
        push!(keys, (Int(step), regime, V24_SERVING_POOLED))
        for depth in V24_SERVING_DEPTH_BINS
            push!(keys, (Int(step), regime, depth))
        end
    end
    return keys
end

"""
    v24_serving_deepening(dst_delta_1h_nt, coupling_active_mvm, latest_dst) -> Bool

Deepening cell of the depth-safe guard: a one-hour fall steeper than `-15` nT/h, or active coupling
with the ring current already at or below `-50` nT. A non-finite rate is neutral, matching the gated
coupling convention, so a driver or Dst gap cannot invent a deepening cell.
"""
function v24_serving_deepening(dst_delta_1h_nt::Real, coupling_active_mvm::Real,
                               latest_dst::Real)
    rate = Float64(dst_delta_1h_nt)
    coupling = Float64(coupling_active_mvm)
    latest = Float64(latest_dst)
    isfinite(rate) && rate < V24_SERVING_GUARD_RATE_NT_PER_H && return true
    return isfinite(coupling) && coupling > 0.0 && isfinite(latest) &&
           latest <= V24_SERVING_GUARD_DEPTH_NT
end

"""
    v24_serving_guard(reference, center, dst_delta_1h_nt, coupling_active_mvm, latest_dst)

The guarded center: in a deepening cell the served center is `min(center, reference)`, so the stack
can deepen the published forecast but can never make it shallower than the static V2.2 stack; elsewhere
it is `center`. This is the arithmetic of the study's guarded variants (`v2_4d`, `v2_4f`). The deployed
V2.4e bundle records `guard_applied = false` and this function is then never reached from
[`v24_serving_center`](@ref); it stays here so a bundle whose selection returns a guarded variant is
servable by the same code path, and so the arithmetic remains under test either way.
"""
v24_serving_guard(reference::Real, center::Real, dst_delta_1h_nt::Real,
                  coupling_active_mvm::Real, latest_dst::Real) =
    v24_serving_deepening(dst_delta_1h_nt, coupling_active_mvm, latest_dst) ?
        min(Float64(center), Float64(reference)) : Float64(center)

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

"""
    V24ServingCell

One fitted super-learner cell: the ten non-negative weights of a `(model step, regime, depth bin)`
population, the row count they were fitted on, and the SINDy-family mass the floor constrains.
"""
struct V24ServingCell
    model_step_hours::Int
    regime::Symbol
    depth::Symbol
    n_rows::Int
    weights::Vector{Float64}
    sindy_family_mass::Float64
end

"""
    V24ServingStratum

One conformal stratum: the half-width of a `(model step, depth bin)` population, the residual count
behind it, the guaranteed finite-sample coverage floor, and whether the bin used its own quantile or
the step's pooled fallback.
"""
struct V24ServingStratum
    model_step_hours::Int
    depth::Symbol
    half_width_nt::Float64
    n::Int
    coverage_floor::Float64
    source::String
end

"""
    V24ServingArtifacts

Everything the V2.4e center needs, loaded once and immutable afterwards: the analog sub-artifact
(archive, standardisation and the refit ridge correction of expert E3), the super-learner cells, the
per-step direct increment-GBM models, the climatology timescale, the conformal strata, the guard and
selection records, and the bundle manifest with its own digest.

`served_variant` is the study variant the bundle publishes and `guard_applied` is that bundle's own
point-forecast guard switch, read from `guard.json` rather than assumed, so the served arithmetic
follows the deployed artifact instead of a source constant.

`analog` is a `V23ServingArtifacts` built in process from this bundle's own files. Reusing that type
is deliberate: the analog retrieval and the member rollout are the V2.3 code path, and V2.4 changes
only *which* archive and *which* refit calibration they use. Its `lat_weights` and `e_layers` fields
are inert here — V2.4 reads the T1r center directly, and the bundle ships neither a blend nor an
error layer — so they are filled with the identity and never consulted.
"""
struct V24ServingArtifacts
    dir::String
    manifest::DataFrame
    manifest_sha256::String
    selection::Dict{String,Any}
    guard::Dict{String,Any}
    analog::V23ServingArtifacts
    cells::Dict{Tuple{Int,Symbol,Symbol},V24ServingCell}
    conformal::Dict{Tuple{Int,Symbol},V24ServingStratum}
    climatology_tau_hours::Float64
    direct_models::Dict{Int,Any}
    direct_labels::Dict{Int,String}
    direct_feature_names::Vector{String}
    fold_year::Int
    training_max_target_utc::DateTime
    oof_pool_years::Vector{Int}
    served_variant::String
    guard_applied::Bool
    guard_reference::String
    identity::String
end

Base.show(io::IO, art::V24ServingArtifacts) = print(
    io, "V24ServingArtifacts(", art.identity, ", fold=", art.fold_year,
    ", origins=", length(art.analog.origins), ", cells=", length(art.cells),
    ", dir=", art.dir, ")",
)

# ---------------------------------------------------------------------------
# Manifest verification
# ---------------------------------------------------------------------------

"""
    _v24_manifest_text(value) -> String

Text of a manifest cell. An empty CSV field reads back as `missing`, and a manifest legitimately
carries empty values (a row that records only a count), so the reader coerces rather than throwing on
a field it does not need.
"""
_v24_manifest_text(value) = value === missing ? "" : String(value)

"Rows of a V2.4 manifest whose `entry_type` matches `kind`."
_v24_manifest_rows(manifest::DataFrame, kind::AbstractString) =
    [r for r in eachrow(manifest) if _v24_manifest_text(r.entry_type) == String(kind)]

"The single manifest row `(kind, name)`, or `nothing`."
function _v24_manifest_row(manifest::DataFrame, kind::AbstractString, name::AbstractString)
    hits = [r for r in _v24_manifest_rows(manifest, kind)
            if _v24_manifest_text(r.name) == String(name)]
    isempty(hits) && return nothing
    length(hits) == 1 || throw(ArgumentError(
        "V2.4 manifest repeats the entry ($kind, $name)",
    ))
    return hits[1]
end

"Value of the single manifest row `(kind, name)`, or `nothing`."
function _v24_manifest_value(manifest::DataFrame, kind::AbstractString, name::AbstractString)
    row = _v24_manifest_row(manifest, kind, name)
    row === nothing && return nothing
    return _v24_manifest_text(row.value)
end

"Parse a `key=value;key=value` manifest value field."
function _v24_manifest_pairs(value::AbstractString)
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
    v24_serving_verify_manifest(dir; manifest = nothing) -> DataFrame

Check every `sha256` row of `dir/manifest.csv` against the file it names, and check that every
required artifact — the fixed list plus one direct-GBM model per supported model step — carries such
a row. A missing row is as much a failure as a wrong digest: an unlisted file would be served without
provenance.
"""
function v24_serving_verify_manifest(dir::AbstractString; manifest = nothing)
    directory = String(dir)
    path = joinpath(directory, "manifest.csv")
    table = manifest === nothing ?
        CSV.read(path, DataFrame; types = Dict("value" => String, "name" => String,
                                               "entry_type" => String)) : manifest
    for column in ("entry_type", "name", "count", "value")
        column in names(table) || throw(ArgumentError(
            "V2.4 manifest $path lacks the $column column",
        ))
    end
    hashed = Set{String}()
    for row in _v24_manifest_rows(table, "sha256")
        name = _v24_manifest_text(row.name)
        file = joinpath(directory, name)
        digest = v23_serving_file_sha256(file)
        digest == _v24_manifest_text(row.value) || error(
            "V2.4 serving artifact $name fails its manifest digest: expected $(row.value), " *
            "computed $digest",
        )
        size = Int(round(Float64(row.count)))
        filesize(file) == size || error(
            "V2.4 serving artifact $name has $(filesize(file)) bytes but the manifest records $size",
        )
        push!(hashed, name)
    end
    required = vcat(collect(String, V24_SERVING_REQUIRED_FILES),
                    [v24_serving_direct_file(step) for step in V24_SERVING_MODEL_STEPS])
    for name in required
        name in hashed || error(
            "V2.4 manifest $path has no sha256 entry for the required artifact $name",
        )
    end
    return table
end

"""
    v24_serving_manifest_hashed_names(manifest) -> Set{String}

Artifact names the manifest carries a `sha256` row for. Only these have been digest-verified, so any
file a configuration record names must be checked against this set before it is parsed.
"""
v24_serving_manifest_hashed_names(manifest::DataFrame) =
    Set{String}(_v24_manifest_text(row.name) for row in _v24_manifest_rows(manifest, "sha256"))

# ---------------------------------------------------------------------------
# Reading the bundle
# ---------------------------------------------------------------------------

"Read the shipped analog standardisation table into `(mean, sd)` in feature order."
function _v24_serving_read_stats(path::AbstractString)
    table = CSV.read(path, DataFrame)
    String.(table.feature_name) == String.(collect(V23_FEATURE_NAMES)) || error(
        "shipped analog feature statistics are not in the V2.3 feature order: $path",
    )
    return (mean = Float64.(table.feature_mean), sd = Float64.(table.feature_sd))
end

"""
    _v24_serving_read_cells(path) -> Dict

Read the served super-learner cells from `stack_weights.csv`. Only the served variant's rows are read,
because that is the fit the served center uses; a bundle may legitimately ship the other fitted stacks
for comparison and they must not be able to reach the served center. Every cell is checked for
non-negativity, unit mass and the SINDy-family floor, and the fully pooled cell of every model step
must exist so a row always has a fallback to walk to.

When the table carries an `expert_set` column — the pipe-joined expert names the cell was fitted on —
it must name exactly the served experts in the served order. Without that check a nine-expert row and a
ten-expert row are indistinguishable once both are widened to the shared weight schema, and a
nine-expert cell would be served with a hard zero on the static stack.
"""
function _v24_serving_read_cells(path::AbstractString)
    table = CSV.read(path, DataFrame; types = Dict("variant" => String, "regime" => String,
                                                   "depth_bin" => String,
                                                   "expert_set" => String))
    for column in ("variant", "model_step_hours", "regime", "depth_bin", "n_rows")
        column in names(table) || error("$path lacks the $column column")
    end
    weight_columns = [Symbol("w_", expert) for expert in V24_SERVING_EXPERTS]
    for column in weight_columns
        String(column) in names(table) || error(
            "$path lacks the weight column $(String(column)); the expert order is " *
            join(String.(collect(V24_SERVING_EXPERTS)), ", "),
        )
    end
    expected_set = join(String.(collect(V24_SERVING_EXPERTS)), "|")
    has_expert_set = "expert_set" in names(table)
    has_expert_count = "n_experts" in names(table)
    cells = Dict{Tuple{Int,Symbol,Symbol},V24ServingCell}()
    for r in 1:nrow(table)
        String(table.variant[r]) == V24_SERVING_STACK_VARIANT || continue
        step = Int(table.model_step_hours[r])
        step in V24_SERVING_MODEL_STEPS || error(
            "$path holds an unsupported model step $step",
        )
        if has_expert_set
            String(table.expert_set[r]) == expected_set || error(
                "$path fitted the $(V24_SERVING_STACK_VARIANT) cell at step $step on the experts " *
                "$(table.expert_set[r]); the served set is $expected_set",
            )
        end
        if has_expert_count
            Int(table.n_experts[r]) == V24_SERVING_EXPERT_COUNT || error(
                "$path records $(table.n_experts[r]) experts in a served cell at step $step; the " *
                "served stack has $(V24_SERVING_EXPERT_COUNT)",
            )
        end
        regime = Symbol(String(table.regime[r]))
        depth = Symbol(String(table.depth_bin[r]))
        key = (step, regime, depth)
        haskey(cells, key) && error("$path repeats the cell $key")
        weights = Float64[Float64(table[r, column]) for column in weight_columns]
        all(w -> isfinite(w) && w >= 0.0, weights) || error(
            "$path holds a negative or non-finite weight in cell $key",
        )
        total = sum(weights)
        abs(total - 1.0) <= V24_SERVING_WEIGHT_SUM_ATOL || error(
            "$path cell $key has weight mass $total; a served stack must be a convex combination",
        )
        family = sum(weights[j] for j in V24_SERVING_SINDY_FAMILY)
        family >= V24_SERVING_SINDY_FLOOR - V24_SERVING_WEIGHT_SUM_ATOL || error(
            "$path cell $key carries SINDy-family mass $family below the served floor " *
            "$(V24_SERVING_SINDY_FLOOR)",
        )
        cells[key] = V24ServingCell(step, regime, depth, Int(table.n_rows[r]), weights, family)
    end
    isempty(cells) && error(
        "$path holds no $(V24_SERVING_STACK_VARIANT) cell; the served center has no weights",
    )
    for step in V24_SERVING_MODEL_STEPS
        haskey(cells, (step, V24_SERVING_POOLED, V24_SERVING_POOLED)) || error(
            "$path has no pooled cell at step $step; a row could then find no cell to use",
        )
    end
    for (key, _) in cells
        step, regime, depth = key
        (regime === V24_SERVING_POOLED || regime in V24_SERVING_REGIMES) || error(
            "$path holds the unknown regime $(regime) at step $step",
        )
        (depth === V24_SERVING_POOLED || depth in V24_SERVING_DEPTH_BINS) || error(
            "$path holds the unknown depth bin $(depth) at step $step",
        )
    end
    return cells
end

"""
    _v24_serving_read_conformal(path, variant) -> Dict

Read the served conformal strata from `conformal.csv`, keeping only the served variant's rows. Every
model step must carry its pooled stratum and all three depth bins, so a row always resolves to a
half-width without inventing one.
"""
function _v24_serving_read_conformal(path::AbstractString, variant::AbstractString)
    table = CSV.read(path, DataFrame; types = Dict("variant" => String, "depth_bin" => String,
                                                   "source" => String))
    for column in ("variant", "model_step_hours", "depth_bin", "half_width_nt", "n",
                   "coverage_floor")
        column in names(table) || error("$path lacks the $column column")
    end
    strata = Dict{Tuple{Int,Symbol},V24ServingStratum}()
    for r in 1:nrow(table)
        String(table.variant[r]) == String(variant) || continue
        step = Int(table.model_step_hours[r])
        step in V24_SERVING_MODEL_STEPS || error("$path holds an unsupported model step $step")
        depth = Symbol(String(table.depth_bin[r]))
        key = (step, depth)
        haskey(strata, key) && error("$path repeats the conformal stratum $key")
        half = Float64(table.half_width_nt[r])
        (isfinite(half) && half > 0.0) || error(
            "$path stratum $key has half-width $half; an interval needs a positive finite width",
        )
        coverage = Float64(table.coverage_floor[r])
        (isfinite(coverage) && 0.0 < coverage < 1.0) || error(
            "$path stratum $key records the coverage floor $coverage",
        )
        source = "source" in names(table) ? String(table.source[r]) : ""
        strata[key] = V24ServingStratum(step, depth, half, Int(table.n[r]), coverage, source)
    end
    isempty(strata) && error("$path holds no conformal stratum for variant $variant")
    for step in V24_SERVING_MODEL_STEPS
        haskey(strata, (step, V24_SERVING_POOLED)) || error(
            "$path has no pooled conformal stratum at step $step",
        )
        for depth in V24_SERVING_DEPTH_BINS
            haskey(strata, (step, depth)) || error(
                "$path has no $(depth) conformal stratum at step $step",
            )
        end
    end
    return strata
end

"Read the climatology relaxation timescale (hours) of the bundle."
function _v24_serving_read_tau(path::AbstractString)
    table = CSV.read(path, DataFrame)
    "tau_hours" in names(table) || error("$path lacks the tau_hours column")
    nrow(table) == 1 || error("$path must hold exactly one row, found $(nrow(table))")
    tau = Float64(table.tau_hours[1])
    (isfinite(tau) && tau > 0.0) || error("$path records the non-physical timescale $tau h")
    return tau
end

"""
    _v24_serving_read_direct(dir, hashed) -> (models, labels)

Read the per-step direct increment-GBM models. Every model must have been fitted on the 29-column
design in `v23_direct_feature_names()` order, and every file must carry a verified manifest digest;
without that check a manifest whose model digests were removed would still serve those models.
"""
function _v24_serving_read_direct(dir::AbstractString,
                                  hashed::Union{Nothing,AbstractSet{String}},
                                  labels_by_step::Dict{Int,String})
    expected = String.(v23_direct_feature_names())
    models = Dict{Int,Any}()
    labels = Dict{Int,String}()
    for step in V24_SERVING_MODEL_STEPS
        name = v24_serving_direct_file(step)
        hashed === nothing || name in hashed || error(
            "the direct-GBM model at $(step) h names $name, which carries no verified manifest " *
            "digest; refusing to serve an unverified expert",
        )
        path = joinpath(dir, name)
        isfile(path) || error("V2.4 direct-GBM artifact is missing: $path")
        model = v23_load(path)
        stored = String.(model.info[:feature_names])
        stored == expected || error(
            "the direct-GBM model at $(step) h was fitted on $(stored); the served design is " *
            "$(expected)",
        )
        models[step] = model
        labels[step] = get(labels_by_step, step, "")
    end
    return (models = models, labels = labels)
end

"Driver lookup of the shipped analog frame; a record enters only when every channel is finite."
_v24_serving_frame_lookup(frame::DataFrame) = v23_serving_frame_lookup(frame)

"""
    load_v24_serving_artifacts(dir; core, verify_hashes = true) -> V24ServingArtifacts

Load and verify a V2.4 deployment bundle.

Every shipped file is checked against its SHA-256 in `manifest.csv` before it is parsed. The analog
archive is then rebuilt rather than trusted: the issue-time features of every shipped origin are
recomputed from the shipped hourly frame, each origin is re-checked against the eligibility rule the
study used (a complete causal history and seven continuable driver records), the origin count and
window are compared with what the builder recorded, and the feature standardisation is recomputed and
compared with the shipped table to `V24_SERVING_STATS_ATOL`. The stack weights are checked for
non-negativity, unit mass, the served expert set and the SINDy-family floor; the conformal grid is
checked for completeness; the guard record must restate the package's own thresholds and must agree
with itself about whether a guard acts and against what; and the selected variant must be the one this
identity publishes. A bundle that fails any of these does not load, so it cannot produce a first
forecast either.
"""
function load_v24_serving_artifacts(dir::AbstractString;
                                    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION),
                                    verify_hashes::Bool = true)
    directory = abspath(String(dir))
    isdir(directory) || error("V2.4 deployment directory not found: $directory")
    manifest_path = joinpath(directory, "manifest.csv")
    isfile(manifest_path) || error("V2.4 deployment bundle has no manifest.csv: $directory")
    manifest = CSV.read(manifest_path, DataFrame;
                        types = Dict("entry_type" => String, "name" => String, "value" => String))
    verify_hashes && v24_serving_verify_manifest(directory; manifest = manifest)
    hashed = verify_hashes ? v24_serving_manifest_hashed_names(manifest) : nothing

    selection = Dict{String,Any}(JSON3.read(
        read(joinpath(directory, "selected.json"), String), Dict{String,Any},
    ))
    guard = Dict{String,Any}(JSON3.read(
        read(joinpath(directory, "guard.json"), String), Dict{String,Any},
    ))
    variant = String(get(selection, "served_variant", ""))
    variant == V24_SERVED_VARIANT || error(
        "V2.4 bundle records the served variant $(repr(variant)); the published product is " *
        "$(V24_SERVED_VARIANT)",
    )
    String(get(selection, "identity", "")) == V24_SERVED_IDENTITY || error(
        "V2.4 bundle records identity $(repr(get(selection, "identity", ""))); this build serves " *
        "$(V24_SERVED_IDENTITY)",
    )
    Bool(get(selection, "residual_applied", false)) == false || error(
        "V2.4 bundle records a boosted residual layer; the served variant carries none",
    )
    String(get(selection, "stack_variant", V24_SERVING_STACK_VARIANT)) ==
        V24_SERVING_STACK_VARIANT || error(
        "V2.4 bundle records the stack variant $(repr(get(selection, "stack_variant", ""))); the " *
        "served center reads $(V24_SERVING_STACK_VARIANT)",
    )
    # The guard is a bundle-level switch, not a source constant, so both fields are read and they must
    # agree: an enabled guard without its reference would serve unguarded while claiming otherwise, and
    # a disabled guard that still names a reference is a record nobody can act on.
    haskey(guard, "guard_applied") || error(
        "V2.4 guard record does not say whether the point-forecast guard acts",
    )
    guard_applied = Bool(guard["guard_applied"])
    guard_reference = String(get(guard, "guard_reference", ""))
    if guard_applied
        guard_reference == V24_SERVING_GUARD_REFERENCE || error(
            "V2.4 guard record enables the guard against $(repr(guard_reference)); the served guard " *
            "is taken against $(V24_SERVING_GUARD_REFERENCE)",
        )
    else
        guard_reference == V24_SERVING_GUARD_REFERENCE_NONE || error(
            "V2.4 guard record disables the guard but names the reference $(repr(guard_reference)); " *
            "an inactive guard must record $(repr(V24_SERVING_GUARD_REFERENCE_NONE))",
        )
    end
    Float64(get(guard, "deepening_rate_nt_per_h_strict_below", NaN)) ==
        V24_SERVING_GUARD_RATE_NT_PER_H || error(
        "V2.4 guard record uses a deepening rate threshold this build does not serve",
    )
    Float64(get(guard, "deepening_depth_nt_at_or_below_with_active_coupling", NaN)) ==
        V24_SERVING_GUARD_DEPTH_NT || error(
        "V2.4 guard record uses a deepening depth threshold this build does not serve",
    )
    Float64(get(guard, "sindy_family_floor", NaN)) == V24_SERVING_SINDY_FLOOR || error(
        "V2.4 guard record uses a SINDy-family floor this build does not serve",
    )
    Float64(get(guard, "depth_bin_moderate_nt_at_or_below", NaN)) ==
        V24_SERVING_DEPTH_MODERATE_NT || error(
        "V2.4 guard record uses a moderate depth edge this build does not serve",
    )
    Float64(get(guard, "depth_bin_deep_nt_at_or_below", NaN)) == V24_SERVING_DEPTH_DEEP_NT ||
        error("V2.4 guard record uses a deep depth edge this build does not serve")
    Float64(get(guard, "conformal_coverage", NaN)) == V24_SERVING_COVERAGE || error(
        "V2.4 guard record uses a conformal coverage this build does not serve",
    )
    order = String[String(name) for name in get(guard, "expert_order", String[])]
    order == String.(collect(V24_SERVING_EXPERTS)) || error(
        "V2.4 guard record lists the expert order $(order); the served order is " *
        "$(String.(collect(V24_SERVING_EXPERTS)))",
    )
    family = String[String(name) for name in
                    get(guard, "sindy_family", [String(V24_SERVING_EXPERTS[j])
                                                for j in V24_SERVING_SINDY_FAMILY])]
    family == [String(V24_SERVING_EXPERTS[j]) for j in V24_SERVING_SINDY_FAMILY] || error(
        "V2.4 guard record puts the mass floor on $(family); the served floor group is " *
        "$([String(V24_SERVING_EXPERTS[j]) for j in V24_SERVING_SINDY_FAMILY])",
    )

    cells = _v24_serving_read_cells(joinpath(directory, "stack_weights.csv"))
    conformal = _v24_serving_read_conformal(joinpath(directory, "conformal.csv"), variant)
    tau = _v24_serving_read_tau(joinpath(directory, "climatology_tau.csv"))
    direct_labels = Dict{Int,String}()
    for step in V24_SERVING_MODEL_STEPS
        value = _v24_manifest_value(manifest, "direct_gbm", "step$(step)")
        value === nothing || (direct_labels[step] = value)
    end
    direct = _v24_serving_read_direct(directory, hashed, direct_labels)

    calibration = read_operational_v2_calibration(joinpath(directory, "t1r_calibration.csv"))
    frame = CSV.read(joinpath(directory, "analog_frame.csv"), DataFrame;
                     types = Dict("time_utc" => DateTime))
    issorted(frame.time_utc) || error("the shipped analog frame is not chronological")
    lookup = _v24_serving_frame_lookup(frame)
    origins_table = CSV.read(joinpath(directory, "analog_origins.csv"), DataFrame;
                             types = Dict("origin_time_utc" => DateTime))
    origins = collect(DateTime, origins_table.origin_time_utc)
    issorted(origins) || error("the shipped analog origins are not chronological")
    allunique(origins) || error("the shipped analog origins repeat a timestamp")
    length(origins) >= V24_SERVING_ANALOG_K || error(
        "the shipped analog archive holds $(length(origins)) origins, fewer than K = " *
        "$(V24_SERVING_ANALOG_K)",
    )
    archive_row = _v24_manifest_row(manifest, "analog_archive", "origins")
    archive_row === nothing && error("V2.4 manifest records no analog_archive origin count")
    expected_origins = Int(round(Float64(archive_row.count)))
    length(origins) == expected_origins || error(
        "the shipped analog archive holds $(length(origins)) origins but the builder recorded " *
        "$expected_origins",
    )
    bounds = _v24_manifest_pairs(_v24_manifest_text(archive_row.value))
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
    shipped = _v24_serving_read_stats(joinpath(directory, "analog_feature_stats.csv"))
    worst = max(maximum(abs.(stats.mean .- shipped.mean)), maximum(abs.(stats.sd .- shipped.sd)))
    worst <= V24_SERVING_STATS_ATOL || error(
        "the rebuilt analog standardisation disagrees with the shipped table by $worst; the " *
        "deployed archive is not the fitted archive",
    )

    fold_year = Int(get(selection, "fold_year", 0))
    fold_year > 0 || error("V2.4 bundle records no fold year")
    training_max_target = DateTime(String(get(selection, "training_max_target_utc", "")))
    pool_years = Int[Int(y) for y in get(selection, "oof_pool_years", Int[])]
    isempty(pool_years) && error("V2.4 bundle records no out-of-fold pool years")
    maximum(pool_years) < fold_year || error(
        "V2.4 bundle records the out-of-fold pool year $(maximum(pool_years)) at or after its " *
        "fold year $(fold_year); the bundle would carry a fit that saw its own scored window",
    )

    analog = V23ServingArtifacts(
        directory, frame, lookup, origins, features,
        v23_standardize(features, stats.mean, stats.sd), stats.mean, stats.sd,
        V24_SERVING_ANALOG_WEIGHT_SET, v23_weights(V24_SERVING_ANALOG_WEIGHT_SET),
        V24_SERVING_ANALOG_K, V23_ANALOG_EXCLUSION_HOURS, calibration,
        ones(Float64, length(V23_SERVING_MODEL_STEPS)),
        V23ServingELayer[
            V23ServingELayer(step, :identity, "", v23_serving_e_cap(step), String[], nothing,
                             nothing)
            for step in V23_SERVING_MODEL_STEPS
        ],
        Dict{String,Any}("family" => "T1r", "safeguards" => false,
                         "k" => V24_SERVING_ANALOG_K,
                         "params" => Dict{String,Any}(
                             "weight_set" => String(V24_SERVING_ANALOG_WEIGHT_SET))),
        manifest, core, V24_SERVING_ANALOG_IDENTITY,
    )
    return V24ServingArtifacts(
        directory, manifest, v23_serving_file_sha256(manifest_path), selection, guard, analog,
        cells, conformal, tau, direct.models, direct.labels,
        String.(v23_direct_feature_names()), fold_year, training_max_target,
        sort(pool_years), variant, guard_applied, guard_reference, V24_SERVED_IDENTITY,
    )
end

# ---------------------------------------------------------------------------
# Expert E3 — analog raw core and the refit ridge correction
# ---------------------------------------------------------------------------

"""
    v24_serving_analog_features(art, anchor_time, driver_history, dst_anchor, dst_prev)

Issue-time analog key of the V2.4 archive. This is `v23_serving_features` evaluated against the V2.4
bundle's own frame statistics: the feature definitions are shared with the V2.3 path, and only the
standardisation and the archive differ.
"""
v24_serving_analog_features(art::V24ServingArtifacts, anchor_time::DateTime, driver_history,
                            dst_anchor::Real, dst_prev::Real) =
    v23_serving_features(art.analog, anchor_time, driver_history, dst_anchor, dst_prev)

"""
    v24_serving_analog_members(art, features; anchor_time, issue_drv, anchor_dst_star,
                               model_steps, step_driver)

Analog ensemble raw core of expert E3 at `model_steps` hours: the K = 25 magnetic-weighted archive
origins nearest to `features`, each rolled through the frozen SINDy core with the origin's archived
driver continuation. Shared with the V2.3 path, evaluated against the V2.4 archive.
"""
v24_serving_analog_members(art::V24ServingArtifacts, features::AbstractVector{<:Real};
                           anchor_time::DateTime, issue_drv, anchor_dst_star::Real,
                           model_steps::Integer, step_driver) =
    v23_serving_members(art.analog, features; anchor_time = anchor_time, issue_drv = issue_drv,
                        anchor_dst_star = anchor_dst_star, model_steps = model_steps,
                        step_driver = step_driver)

"""
    v24_serving_t1r_center(art; raw_reported, latest_dst, anchor_drivers, memory, baselines,
                           model_steps, anchor_time) -> NamedTuple

Expert E3: the analog raw core corrected by this bundle's refit of the deployed V2.1 ridge layer and
projected to the physical range, `clamp(raw + correction, -2000, 50)`, taking no V2.1 safeguard. That
is the `t1r_analog` column of the study's fold tables, formed by the same arithmetic.
"""
function v24_serving_t1r_center(art::V24ServingArtifacts; raw_reported::Real, latest_dst::Real,
                                anchor_drivers, memory, baselines, model_steps::Integer,
                                anchor_time::DateTime)
    features = v23_serving_t1r_features(art.analog, raw_reported, latest_dst, anchor_drivers,
                                        memory, baselines, Int(model_steps), anchor_time)
    correction = operational_v2_correction(art.analog.calibration, features)
    center = clamp(Float64(raw_reported) + correction, V24_SERVING_DST_FLOOR_NT,
                   V24_SERVING_DST_CEIL_NT)
    return (raw_reported = Float64(raw_reported), correction = correction, center = center)
end

# ---------------------------------------------------------------------------
# Expert E8 — direct increment-GBM
# ---------------------------------------------------------------------------

"""
    v24_serving_direct_frame(anchor_time, driver_history, dst_history) -> DataFrame

Hourly frame the direct-GBM design is read from: one row per hour from `anchor - 24` to `anchor`, the
driver channels taken from the at-Earth hourly means of `driver_history` (entry `j` is the record
tagged `anchor - j`, `nothing` for an hour with no measured L1 coverage), and `Dst` taken from
`dst_history`, a map from an hour to the observed Dst at that hour.

The frame exists so the live path can call `v23_direct_features` — the same package function the
study's design matrix came from — instead of restating 29 feature definitions. Twenty-four hours of
Dst are carried because the design reads `dst_lag24`; the driver channels are carried for the
run-length window only, which is the deepest driver lag any V2.4 feature reads.
"""
function v24_serving_direct_frame(anchor_time::DateTime, driver_history, dst_history::AbstractDict)
    lags = length(driver_history)
    lags >= V23_HISTORY_LAGS_H || throw(ArgumentError(
        "the V2.4 direct design needs at least $(V23_HISTORY_LAGS_H) driver records, got $lags",
    ))
    span = max(lags, V24_SERVING_DST_LAG_MAX_H)
    times = DateTime[anchor_time - Hour(j) for j in span:-1:0]
    n = length(times)
    V = fill(NaN, n); Bz = fill(NaN, n); By = fill(NaN, n)
    density = fill(NaN, n); pdyn = fill(NaN, n); dst = fill(NaN, n)
    for j in 1:lags
        record = driver_history[j]
        record === nothing && continue
        row = span - j + 1
        V[row] = Float64(record.V)
        Bz[row] = Float64(record.Bz)
        By[row] = Float64(record.By)
        density[row] = Float64(record.n)
        pdyn[row] = Float64(record.Pdyn)
    end
    for (row, time) in enumerate(times)
        value = get(dst_history, time, nothing)
        value === nothing && continue
        numeric = Float64(value)
        isfinite(numeric) && (dst[row] = numeric)
    end
    return DataFrame(time_utc = times, V = V, Bz = Bz, By = By, n = density, Pdyn = pdyn,
                     Dst = dst)
end

"""
    v24_serving_direct_features(art, anchor_time, driver_history, dst_history; analog_features)

The 29-column direct-GBM design of one anchor, built from the live hourly means and the observed Dst
lag ladder through `v23_direct_features`.

`analog_features`, when given, is the analog key computed by `v24_serving_analog_features`; its 18
values must equal the first 18 columns of the design exactly. Both blocks describe the same issue-time
state, so a disagreement means the two paths saw different hourly means, and the expert is refused
rather than served from an inconsistent state.

Returns `(features, ok, reason)`; `ok = false` leaves the design non-finite and names the missing
input, which is what the log records when no direct center is available.
"""
function v24_serving_direct_features(art::V24ServingArtifacts, anchor_time::DateTime,
                                     driver_history, dst_history::AbstractDict;
                                     analog_features = nothing)
    frame = v24_serving_direct_frame(anchor_time, driver_history, dst_history)
    X, ok = v23_direct_features(frame, [anchor_time])
    values = vec(Matrix{Float64}(X)[1, :])
    if !ok[1]
        reason = if get(dst_history, anchor_time, nothing) === nothing
            "missing_anchor_dst"
        else
            missing_lag = findfirst(
                lag -> !(get(dst_history, anchor_time - Hour(lag), nothing) isa Real) ||
                       !isfinite(Float64(get(dst_history, anchor_time - Hour(lag), NaN))),
                collect(V23_DIRECT_DST_LAG_HOURS),
            )
            if missing_lag !== nothing
                "missing_dst_lag$(V23_DIRECT_DST_LAG_HOURS[missing_lag])"
            else
                bad_driver = findfirst(
                    j -> driver_history[j] === nothing,
                    1:min(length(driver_history),
                          V23_HISTORY_LAGS_H + maximum(V23_DIRECT_VBS_LAG_STEPS)),
                )
                bad_driver === nothing ? "incomplete_direct_design" :
                    "missing_driver_lag$(bad_driver)"
            end
        end
        return (features = values, ok = false, reason = reason)
    end
    if analog_features !== nothing
        key = Float64.(collect(analog_features))
        length(key) == V23_FEATURE_COUNT || throw(DimensionMismatch(
            "the analog key has $(length(key)) entries, expected $(V23_FEATURE_COUNT)",
        ))
        for j in 1:V23_FEATURE_COUNT
            values[j] == key[j] || error(
                "the V2.4 direct design and the analog key disagree on feature " *
                "$(V23_FEATURE_NAMES[j]) at $(anchor_time): $(values[j]) versus $(key[j])",
            )
        end
    end
    return (features = values, ok = true, reason = "ok")
end

"""
    v24_serving_direct_center(art, features, model_steps) -> Float64

Expert E8: the fitted increment of `model_steps` hours added back to the design's own anchor Dst,
`dst0`. The model regresses `Dst(t+h) - Dst(t)`, so inverting it with any other anchor would change
what the expert predicts; the anchor therefore comes from the feature vector the model was scored on.
"""
function v24_serving_direct_center(art::V24ServingArtifacts, features::AbstractVector{<:Real},
                                   model_steps::Integer)
    step = Int(model_steps)
    haskey(art.direct_models, step) || throw(ArgumentError(
        "the V2.4 bundle carries no direct-GBM model at $(step) h",
    ))
    design = Float64.(collect(features))
    length(design) == V23_DIRECT_FEATURE_COUNT || throw(DimensionMismatch(
        "the V2.4 direct design has $(length(design)) entries, expected " *
        "$(V23_DIRECT_FEATURE_COUNT)",
    ))
    all(isfinite, design) || throw(ArgumentError(
        "the V2.4 direct design must be finite; check `ok` before scoring",
    ))
    increment = v23_predict(art.direct_models[step], reshape(design, 1, length(design)))[1]
    return increment + design[v23_feature_index(:dst0)]
end

# ---------------------------------------------------------------------------
# Expert E9 — climatology-relaxed persistence
# ---------------------------------------------------------------------------

"""
    v24_serving_climatology_center(art, latest_dst, model_steps) -> Float64

Expert E9: `Dst(t) · exp(-h/tau)`, with the recovery timescale fitted on the bundle's training
window. This restates the study's climatology comparator exactly.
"""
v24_serving_climatology_center(art::V24ServingArtifacts, latest_dst::Real, model_steps::Integer) =
    Float64(latest_dst) * exp(-Int(model_steps) / art.climatology_tau_hours)

# ---------------------------------------------------------------------------
# The served center
# ---------------------------------------------------------------------------

"""
    v24_serving_cell(art, model_steps, regime, depth) -> NamedTuple

The stack cell a row uses, found by walking `v24_serving_cell_chain`. Returns the cell, its position
on the chain and whether a coarser cell had to be used, so a log can record why a center moved.
"""
function v24_serving_cell(art::V24ServingArtifacts, model_steps::Integer, regime::Symbol,
                          depth::Symbol)
    chain = v24_serving_cell_chain(model_steps, regime, depth)
    for (position, key) in enumerate(chain)
        cell = get(art.cells, key, nothing)
        cell === nothing && continue
        return (cell = cell, position = position, used_pooled_fallback = position > 1)
    end
    error("the V2.4 stack has no cell for step $(Int(model_steps)) on the chain $(chain)")
end

"""
    v24_serving_interval(art, model_steps, depth, center) -> NamedTuple

Conformal interval of a served center: the `(step, depth bin)` half-width, with the step's pooled
stratum as the fallback. The endpoints are the center plus and minus the half-width, unprojected,
which is what the study scored coverage on.
"""
function v24_serving_interval(art::V24ServingArtifacts, model_steps::Integer, depth::Symbol,
                              center::Real)
    step = Int(model_steps)
    stratum = get(art.conformal, (step, depth), nothing)
    if stratum === nothing
        stratum = get(art.conformal, (step, V24_SERVING_POOLED), nothing)
        stratum === nothing &&
            error("the V2.4 conformal strata lack step $step at depth $(depth)")
    end
    value = Float64(center)
    return (half_width_nt = stratum.half_width_nt, lo_nt = value - stratum.half_width_nt,
            hi_nt = value + stratum.half_width_nt, stratum = stratum)
end

"""
    v24_serving_center(art; model_steps, latest_dst, dst_delta_1h_nt, vbsouth_mvm, experts)
        -> NamedTuple

Served V2.4e center of one `(anchor, model step)` row.

`experts` is a named tuple carrying the ten expert centers under the names of
[`V24_SERVING_EXPERTS`](@ref); the static V2.2 stack is one of them, so it is not passed separately.
The issue-time regime comes from the observed Dst, its one-hour rate and the gated coupling proxy,
exactly as the static stack's regime does; the depth bin comes from the observed Dst alone. The row's
cell weights form the L1 center, which the deployed bundle publishes unchanged, and the conformal
stratum of the row's `(step, depth)` supplies the interval. If the loaded bundle enables the
point-forecast guard, the center is additionally taken no shallower than the static stack expert in a
deepening cell.

Returns the served center, the unguarded L1 center, whether the guard acted, whether the physical
projection acted, the cell identity, the weights, the regime and depth labels, and the interval, so a
caller can log every stage that moved the number.

`projection_applied` is the last of those stages: the center is taken through
`clamp(·, -2000, 50)` nT, which is inert on every row the study scored but would act on an anchor
state above roughly `+52` nT. It is reported rather than left implicit, because a projected center is
otherwise indistinguishable from an unprojected one and the `center == l1_center` identity of an
unguarded bundle holds only where the projection did not act.
"""
function v24_serving_center(art::V24ServingArtifacts; model_steps::Integer, latest_dst::Real,
                            dst_delta_1h_nt::Real, vbsouth_mvm::Real, experts)
    step = Int(model_steps)
    step in V24_SERVING_MODEL_STEPS || throw(ArgumentError(
        "the V2.4e product does not cover a $(step) h model step; supported steps are " *
        join(V24_SERVING_MODEL_STEPS, ", "),
    ))
    values = Vector{Float64}(undef, V24_SERVING_EXPERT_COUNT)
    for (j, name) in enumerate(V24_SERVING_EXPERTS)
        haskey(experts, name) || throw(ArgumentError(
            "the V2.4 expert panel is missing $(name)",
        ))
        value = Float64(getproperty(experts, name))
        isfinite(value) || throw(ArgumentError(
            "every V2.4 expert center must be finite; $(name) is $(value)",
        ))
        values[j] = value
    end
    reference = values[V24_SERVING_STATIC_EXPERT]
    latest = Float64(latest_dst)
    isfinite(latest) || throw(ArgumentError("the V2.4 cell needs a finite issue-time Dst"))
    coupling = v22_serving_coupling_active(vbsouth_mvm, dst_delta_1h_nt)
    rate = isfinite(Float64(dst_delta_1h_nt)) ? Float64(dst_delta_1h_nt) : 0.0
    regime = operational_v22_regime(latest, rate, coupling)
    depth = v24_serving_depth_bin(latest)
    resolved = v24_serving_cell(art, step, regime, depth)
    cell = resolved.cell
    l1 = 0.0
    for j in 1:V24_SERVING_EXPERT_COUNT
        l1 += cell.weights[j] * values[j]
    end
    isfinite(l1) || throw(ArgumentError("the V2.4 stack center became non-finite"))
    guarded = art.guard_applied ? v24_serving_guard(reference, l1, rate, coupling, latest) : l1
    center = clamp(guarded, V24_SERVING_DST_FLOOR_NT, V24_SERVING_DST_CEIL_NT)
    interval = v24_serving_interval(art, step, depth, center)
    return (center = center, raw_center = guarded, l1_center = l1,
            guard_applied = guarded != l1, guard_enabled = art.guard_applied,
            projection_applied = center != guarded,
            deepening_cell = v24_serving_deepening(rate, coupling, latest),
            guard_reference = reference, regime = regime, depth_bin = depth,
            cell_regime = cell.regime, cell_depth = cell.depth,
            used_pooled_fallback = resolved.used_pooled_fallback,
            regime_cell = "$(cell.regime)/$(cell.depth)",
            weights = NamedTuple{V24_SERVING_EXPERTS}(Tuple(cell.weights)),
            sindy_family_mass = cell.sindy_family_mass,
            coupling_active_mvm = coupling, model_step_hours = step,
            half_width_nt = interval.half_width_nt, ci05_nt = interval.lo_nt,
            ci95_nt = interval.hi_nt, conformal_source = interval.stratum.source,
            conformal_n = interval.stratum.n)
end
