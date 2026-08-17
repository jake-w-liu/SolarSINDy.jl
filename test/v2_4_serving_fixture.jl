module V24ServingFixture

# Synthetic V2.4e deployment bundle for the serving unit tests.
#
# The real bundle is a 12 MB refit of the rolling study; loading it in a unit test would test the
# study, not the loader. This fixture builds a complete, self-consistent bundle from a short synthetic
# hourly frame instead: an analog archive with its standardisation, a readable correction calibration,
# six tiny boosted models on the real 29-column design schema, a floor-satisfying stack, a complete
# conformal grid, the guard and selection records, and a digest manifest.
#
# `mutate` injects exactly one defect. Every keyword corresponds to a way a bundle could be wrong in a
# deployment — an edited weight, a removed cell, a claimed layer the product does not carry, a drifted
# guard threshold, an inconsistent guard switch, a stack fitted on the wrong expert set, a relabelled
# identity, an unlisted or tampered file — and the tests assert the loader refuses each one. Without
# those cases the loader's checks would be untested code.

using CSV
using DataFrames
using Dates
using JSON3
using SHA
using SolarSINDy

export build_v24_fixture_bundle, v24_fixture_frame, v24_fixture_weights, V24_FIXTURE_MUTATIONS,
       V24_FIXTURE_EXPECTED_ERRORS

"Defects `build_v24_fixture_bundle` can inject, each of which the loader must refuse."
const V24_FIXTURE_MUTATIONS = (
    :sub_floor_mass, :non_unit_mass, :negative_weight, :missing_pooled_cell,
    :missing_conformal_bin, :zero_half_width, :bad_identity, :residual_claimed,
    :guard_rate_drift, :guard_depth_edge_drift, :wrong_expert_order, :wrong_gbm_features,
    :tampered_stack_file, :unlisted_gbm_model, :stale_stats, :origins_count_mismatch,
    :pool_year_at_fold_year, :bad_tau,
    # Amendment A3 defects: the served stack is the ten-expert fit, the served floor group counts the
    # static stack, the served variant and stack label name that fit, and the guard switch and its
    # reference have to agree with each other.
    :wrong_expert_set, :wrong_expert_count, :wrong_floor_group, :wrong_served_variant,
    :wrong_stack_variant, :guard_enabled_without_reference, :guard_disabled_with_reference,
    :guard_switch_absent, :nine_expert_stack, :conformal_variant_mismatch,
)

"""
    V24_FIXTURE_EXPECTED_ERRORS

The check each injected defect must trip, as a fragment of the message that check raises.

A bare "some exception was thrown" assertion passes when a defect is caught by the wrong check, and a
defect caught by the wrong check leaves the intended check untested while looking covered. Two defects
here were caught by a different check than the fixture documented before this table existed: an
unlisted direct-GBM model is refused by the manifest's required-artifact rule rather than by the
reader's own digest gate (which `test_operational_v24_serving.jl` therefore reaches directly), and a
renamed served variant is refused by the selection record rather than by the conformal keying — so
`:conformal_variant_mismatch` was added to exercise that keying on its own.
"""
const V24_FIXTURE_EXPECTED_ERRORS = Dict{Symbol,String}(
    :sub_floor_mass => "carries SINDy-family mass",
    :non_unit_mass => "has weight mass",
    :negative_weight => "holds a negative or non-finite weight in cell",
    :missing_pooled_cell => "has no pooled cell at step",
    :missing_conformal_bin => "conformal stratum at step",
    :zero_half_width => "an interval needs a positive finite width",
    :bad_identity => "records identity",
    :residual_claimed => "records a boosted residual layer",
    :guard_rate_drift => "deepening rate threshold this build does not serve",
    :guard_depth_edge_drift => "moderate depth edge this build does not serve",
    :wrong_expert_order => "lists the expert order",
    :wrong_gbm_features => "the served design is",
    :tampered_stack_file => "fails its manifest digest",
    :unlisted_gbm_model => "has no sha256 entry for the required artifact",
    :stale_stats => "the rebuilt analog standardisation disagrees with the shipped table",
    :origins_count_mismatch => "origins but the builder recorded",
    :pool_year_at_fold_year => "records the out-of-fold pool year",
    :bad_tau => "non-physical timescale",
    :wrong_expert_set => "the served set is",
    :wrong_expert_count => "experts in a served cell at step",
    :wrong_floor_group => "puts the mass floor on",
    :wrong_served_variant => "records the served variant",
    :wrong_stack_variant => "records the stack variant",
    :guard_enabled_without_reference => "enables the guard against",
    :guard_disabled_with_reference => "disables the guard but names the reference",
    :guard_switch_absent => "does not say whether the point-forecast guard acts",
    :nine_expert_stack => "holds no $(V24_SERVING_STACK_VARIANT) cell",
    :conformal_variant_mismatch => "holds no conformal stratum for variant",
)

"""
    v24_fixture_frame(; n, t0) -> DataFrame

Deterministic hourly driver/Dst frame: a smooth, physically admissible solar wind with a southward
excursion and a matching ring-current depression, so the analog archive has structure and the depth
bins are all reachable.
"""
function v24_fixture_frame(; n::Int = 720, t0::DateTime = DateTime(2015, 3, 1, 0))
    times = [t0 + Hour(k - 1) for k in 1:n]
    speed = [420.0 + 60.0 * sin(2π * k / 96) for k in 1:n]
    bz = [-3.0 + 4.0 * cos(2π * k / 72) - 2.0 * exp(-((k - 300)^2) / 800) for k in 1:n]
    by = [1.5 * sin(2π * k / 48) for k in 1:n]
    density = [5.0 + 2.0 * sin(2π * k / 120) for k in 1:n]
    pdyn = [dynamic_pressure(density[k], speed[k]) for k in 1:n]
    dst = [-18.0 - 55.0 * exp(-((k - 305)^2) / 1500) + 6.0 * sin(2π * k / 60) for k in 1:n]
    return DataFrame(time_utc = times, V = speed, Bz = bz, By = by, n = density, Pdyn = pdyn,
                     Dst = dst)
end

"""
    v24_fixture_weights(; sindy_mass, total) -> Vector{Float64}

Ten weights in `V24_SERVING_EXPERTS` order carrying `sindy_mass` on the SINDy family — served, frozen,
analog and the static stack — and summing to `total`. The defaults sit exactly on the served floor,
which is where an off-by-a-hair weight check would show.
"""
function v24_fixture_weights(; sindy_mass::Float64 = V24_SERVING_SINDY_FLOOR,
                             total::Float64 = 1.0)
    weights = zeros(Float64, V24_SERVING_EXPERT_COUNT)
    for j in V24_SERVING_SINDY_FAMILY
        weights[j] = sindy_mass / length(V24_SERVING_SINDY_FAMILY)
    end
    rest = total - sindy_mass
    others = [j for j in 1:V24_SERVING_EXPERT_COUNT if !(j in V24_SERVING_SINDY_FAMILY)]
    for j in others
        weights[j] = rest / length(others)
    end
    return weights
end

_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

"""
One row of the stack-weight table, in the column order the served loader reads. `expert_set` and
`n_experts` describe the fit the row came from, exactly as the study writes them, because a
nine-expert row and a ten-expert row are otherwise indistinguishable once both are widened to the
shared weight schema.
"""
function _stack_row(step::Int, regime::Symbol, depth::Symbol, weights::Vector{Float64};
                    variant::AbstractString = V24_SERVING_STACK_VARIANT, n_rows::Int = 4096,
                    expert_set::AbstractString = join(String.(collect(V24_SERVING_EXPERTS)), "|"),
                    n_experts::Int = V24_SERVING_EXPERT_COUNT)
    named = NamedTuple{Tuple(Symbol("w_", e) for e in V24_SERVING_EXPERTS)}(Tuple(weights))
    return merge((fold_year = 2026, variant = String(variant), model_step_hours = step,
                  regime = String(regime), depth_bin = String(depth), n_rows = n_rows,
                  n_experts = n_experts, expert_set = String(expert_set)), named,
                 (weight_sum = sum(weights),
                  sindy_family_mass = sum(weights[j] for j in V24_SERVING_SINDY_FAMILY),
                  objective_mse = 12.5, support = "fixture",
                  kkt_stationarity = 0.0, kkt_dual_min = 0.0, floor_active = true))
end

"""
Conformal half-width of a `(step, depth)` stratum in the fixture: wider with lead and with depth.

The pooled stratum has its own width, distinct from every depth bin's. A pooled width equal to the
shallow one makes the interval fallback untestable — a row that resolved to the pooled stratum and a
row that resolved to the shallow one would return the same number — so the two are deliberately
different.
"""
_fixture_half_width(step::Int, depth::Symbol) =
    (depth === :deep ? 3.0 : depth === :moderate ? 1.6 :
     depth === :shallow ? 1.0 : 2.2) * (4.0 + 1.5 * step)

"""
    build_v24_fixture_bundle(dir; mutate, fold_year, oof_pool_years, guard) -> NamedTuple

Write a complete synthetic bundle into `dir` and return what was written, so a test can assert against
the values it asked for rather than against the file it just parsed.

`guard` writes a bundle whose `guard.json` enables the point-forecast guard against the static V2.2
stack. The deployed product does not, but the guard code path is retained and configurable, so a test
needs a bundle that exercises it; a bundle built with `guard = true` is well-formed and must load.
"""
function build_v24_fixture_bundle(dir::AbstractString; mutate::Union{Nothing,Symbol} = nothing,
                                  fold_year::Int = 2026,
                                  oof_pool_years::Vector{Int} = collect(2013:2025),
                                  guard::Bool = false,
                                  frame::DataFrame = v24_fixture_frame())
    mutate === nothing || mutate in V24_FIXTURE_MUTATIONS ||
        throw(ArgumentError("unknown fixture mutation $(mutate)"))
    directory = abspath(String(dir))
    mkpath(directory)
    written = String[]
    record(name) = (push!(written, name); joinpath(directory, name))

    # ---- analog archive ----
    lookup = v23_serving_frame_lookup(frame)
    candidates = [t for t in frame.time_utc
                  if t >= frame.time_utc[V23_SOUTH_RUN_CAP_H + 2] &&
                     t <= frame.time_utc[end] - Hour(V23_ANALOG_MAX_STEP + 1)]
    features, ok = v23_feature_matrix(frame, candidates)
    continuable = v23_analog_origin_ok(lookup, candidates)
    keep = [j for j in eachindex(candidates) if ok[j] && continuable[j]]
    length(keep) >= V24_SERVING_ANALOG_K ||
        error("the fixture frame yields only $(length(keep)) eligible analog origins")
    origins = candidates[keep]
    stats = v23_feature_stats(features[keep, :])
    CSV.write(record("analog_frame.csv"), frame)
    CSV.write(record("analog_origins.csv"), DataFrame(origin_time_utc = origins))
    stats_mean = copy(stats.mean)
    mutate === :stale_stats && (stats_mean[1] += 1.0e-3)
    CSV.write(record("analog_feature_stats.csv"), DataFrame(
        feature_name = String.(collect(V23_FEATURE_NAMES)), feature_mean = stats_mean,
        feature_sd = stats.sd,
    ))

    # ---- correction calibration ----
    # The deployed V2.1 point layer is written out as the bundle's correction: the loader's contract is
    # that the file parses as a calibration with the deployed feature list, which this satisfies without
    # the fixture having to fit a ridge on synthetic residuals.
    calibration = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv)
    write_operational_v2_calibration(record("t1r_calibration.csv"), calibration)

    # ---- direct increment-GBM per model step ----
    names_29 = String.(v23_direct_feature_names())
    rows = 240
    design = Matrix{Float64}(undef, rows, V23_DIRECT_FEATURE_COUNT)
    for i in 1:rows, j in 1:V23_DIRECT_FEATURE_COUNT
        design[i, j] = sinpi((i + 3 * j) / 47) * (1.0 + 0.1 * j)
    end
    for step in V24_SERVING_MODEL_STEPS
        target = [0.5 * step * design[i, 1] - 0.2 * design[i, 17] for i in 1:rows]
        used_names = (mutate === :wrong_gbm_features && step == first(V24_SERVING_MODEL_STEPS)) ?
            vcat(names_29[2:end], names_29[1]) : names_29
        model = v23_fit_gbm(design, target; max_depth = 2, nrounds = 3, nbins = 8,
                            min_weight = 4.0, seed = 7, feature_names = used_names)
        path = joinpath(directory, v24_serving_direct_file(step))
        v23_save(model, path)
        # An unlisted model file is a bundle whose expert is served without provenance; the manifest
        # therefore skips its digest row only under that mutation.
        (mutate === :unlisted_gbm_model && step == first(V24_SERVING_MODEL_STEPS)) ||
            push!(written, v24_serving_direct_file(step))
    end

    # ---- stack weights ----
    stack_rows = NamedTuple[]
    base = v24_fixture_weights()
    deep = v24_fixture_weights(; sindy_mass = 0.75)
    for step in V24_SERVING_MODEL_STEPS
        (mutate === :missing_pooled_cell && step == last(V24_SERVING_MODEL_STEPS)) ||
            push!(stack_rows, _stack_row(step, V24_SERVING_POOLED, V24_SERVING_POOLED, base))
        push!(stack_rows, _stack_row(step, :quiet, V24_SERVING_POOLED, base; n_rows = 2048))
        push!(stack_rows, _stack_row(step, :active_deepening, :deep, deep; n_rows = 96))
    end
    if mutate === :sub_floor_mass
        push!(stack_rows, _stack_row(first(V24_SERVING_MODEL_STEPS), :recovery, :moderate,
                                     v24_fixture_weights(; sindy_mass = 0.4)))
    elseif mutate === :non_unit_mass
        push!(stack_rows, _stack_row(first(V24_SERVING_MODEL_STEPS), :recovery, :moderate,
                                     v24_fixture_weights(; total = 0.9)))
    elseif mutate === :negative_weight
        bad = v24_fixture_weights()
        # The static-stack expert is the one the floor group also counts, so a negative weight is put on
        # a non-family slot: this must be refused by the non-negativity check, not by the floor check.
        bad[V24_SERVING_EXPERT_COUNT - 1] = -0.05
        bad[V24_SERVING_EXPERT_COUNT - 2] += 0.05
        push!(stack_rows, _stack_row(first(V24_SERVING_MODEL_STEPS), :recovery, :moderate, bad))
    end
    if mutate === :wrong_expert_set
        # A served cell whose recorded expert set is the nine-expert one: the widened weight columns
        # would still parse, and the static stack would silently be served at weight zero.
        stack_rows[1] = _stack_row(
            first(V24_SERVING_MODEL_STEPS), V24_SERVING_POOLED, V24_SERVING_POOLED, base;
            expert_set = join(String.(collect(V24_SERVING_EXPERTS))[1:end - 1], "|"),
        )
    elseif mutate === :wrong_expert_count
        stack_rows[1] = _stack_row(first(V24_SERVING_MODEL_STEPS), V24_SERVING_POOLED,
                                   V24_SERVING_POOLED, base;
                                   n_experts = V24_SERVING_EXPERT_COUNT - 1)
    elseif mutate === :nine_expert_stack
        # The whole table relabelled as the nine-expert fit: the served variant then has no cells at
        # all, which is a different failure from a single mislabelled row.
        stack_rows = [merge(row, (variant = "L1a",)) for row in stack_rows]
    end
    CSV.write(record("stack_weights.csv"), DataFrame(stack_rows))

    # ---- conformal strata ----
    # `:wrong_served_variant` renames the served variant everywhere the bundle records it. The loader
    # refuses it at the selection record, before any interval is read, so that defect proves the served
    # variant contract rather than the conformal keying.
    #
    # `:conformal_variant_mismatch` is the defect that proves the keying: the conformal rows alone carry
    # another variant's name while the selection record still names the served one, so the bundle loses
    # its interval instead of silently serving half-widths calibrated on a different center.
    served_variant = mutate === :wrong_served_variant ? "v2_4d" : V24_SERVED_VARIANT
    conformal_variant = mutate === :conformal_variant_mismatch ? "v2_4f" : served_variant
    conformal_rows = NamedTuple[]
    for step in V24_SERVING_MODEL_STEPS, depth in (V24_SERVING_POOLED, V24_SERVING_DEPTH_BINS...)
        (mutate === :missing_conformal_bin && step == first(V24_SERVING_MODEL_STEPS) &&
         depth === :deep) && continue
        half = _fixture_half_width(step, depth)
        (mutate === :zero_half_width && step == first(V24_SERVING_MODEL_STEPS) &&
         depth === :shallow) && (half = 0.0)
        push!(conformal_rows, (
            fold_year = fold_year, variant = conformal_variant, model_step_hours = step,
            depth_bin = String(depth), half_width_nt = half, n = 5000,
            coverage_floor = 0.9001, source = "own", calibration_note = "fixture",
        ))
    end
    CSV.write(record("conformal.csv"), DataFrame(conformal_rows))

    # ---- climatology timescale ----
    tau = mutate === :bad_tau ? -7.5 : 7.5
    CSV.write(record("climatology_tau.csv"), DataFrame(
        tau_hours = [tau], fold_year = [fold_year], n_training_anchors = [1234],
        training_max_target_utc = [string(last(frame.time_utc))],
    ))

    # ---- guard and selection records ----
    experts = [String(e) for e in V24_SERVING_EXPERTS]
    mutate === :wrong_expert_order && (experts = reverse(experts))
    family = [String(V24_SERVING_EXPERTS[j]) for j in V24_SERVING_SINDY_FAMILY]
    # The floor group of the served stack counts the static stack; dropping it is a floor that no longer
    # means what the bundle says it means, even though every weight still passes the mass check.
    mutate === :wrong_floor_group && (family = family[1:end - 1])
    guard_applied = guard || mutate === :guard_enabled_without_reference
    guard_reference = if mutate === :guard_disabled_with_reference
        V24_SERVING_GUARD_REFERENCE
    elseif mutate === :guard_enabled_without_reference
        "l1_center"
    elseif guard_applied
        V24_SERVING_GUARD_REFERENCE
    else
        V24_SERVING_GUARD_REFERENCE_NONE
    end
    guard_record = Dict{String,Any}(
        "served_variant" => served_variant,
        "served_stack" => V24_SERVING_STACK_VARIANT,
        "guard_applied" => guard_applied,
        "guard_reference" => guard_reference,
        "guard_rule" => "fixture",
        "center_source" => "ten_expert_stack_with_family_floor",
        "residual_applied" => false,
        "deepening_rate_nt_per_h_strict_below" =>
            mutate === :guard_rate_drift ? -10.0 : V24_SERVING_GUARD_RATE_NT_PER_H,
        "deepening_depth_nt_at_or_below_with_active_coupling" => V24_SERVING_GUARD_DEPTH_NT,
        "sindy_family_floor" => V24_SERVING_SINDY_FLOOR,
        "sindy_family" => family,
        "expert_order" => experts,
        "n_experts" => V24_SERVING_EXPERT_COUNT,
        "model_steps" => collect(V24_SERVING_MODEL_STEPS),
        "depth_bins" => [String(b) for b in V24_SERVING_DEPTH_BINS],
        "depth_bin_moderate_nt_at_or_below" =>
            mutate === :guard_depth_edge_drift ? -25.0 : V24_SERVING_DEPTH_MODERATE_NT,
        "depth_bin_deep_nt_at_or_below" => V24_SERVING_DEPTH_DEEP_NT,
        "conformal_coverage" => V24_SERVING_COVERAGE,
        "climatology_tau_hours" => tau,
        "l1_min_cell_rows" => 48,
    )
    # A guard record that does not say whether the guard acts cannot be served either way.
    mutate === :guard_switch_absent && delete!(guard_record, "guard_applied")
    open(record("guard.json"), "w") do io
        JSON3.pretty(io, guard_record)
        println(io)
    end
    selected = Dict{String,Any}(
        "served_variant" => served_variant,
        "identity" => mutate === :bad_identity ? "v2.4-fixture" : V24_SERVED_IDENTITY,
        "driver_assumption" => V24_SERVED_DRIVER_ASSUMPTION,
        "stack_variant" => mutate === :wrong_stack_variant ? "L1a" : V24_SERVING_STACK_VARIANT,
        "residual_applied" => mutate === :residual_claimed,
        "fold_year" => fold_year,
        "bundle_label" => "fixture",
        "training_max_target_utc" => string(last(frame.time_utc)),
        "oof_pool_years" => mutate === :pool_year_at_fold_year ?
                            vcat(oof_pool_years, [fold_year]) : oof_pool_years,
        "conformal_pool_years" => oof_pool_years,
        "julia_threads" => 1,
    )
    open(record("selected.json"), "w") do io
        JSON3.pretty(io, selected)
        println(io)
    end

    # ---- manifest ----
    manifest_rows = NamedTuple[
        (entry_type = "build", name = "identity", count = NaN, value = V24_SERVED_IDENTITY),
        (entry_type = "fold", name = "year", count = Float64(fold_year), value = "fixture"),
        (entry_type = "analog_archive", name = "origins", count = Float64(length(origins)),
         value = "first=$(first(origins));last=$(last(origins))"),
        (entry_type = "hyperparameter", name = "climatology_tau_hours", count = tau, value = ""),
    ]
    if mutate === :origins_count_mismatch
        manifest_rows[3] = (entry_type = "analog_archive", name = "origins",
                            count = Float64(length(origins) - 1),
                            value = "first=$(first(origins));last=$(last(origins))")
    end
    for step in V24_SERVING_MODEL_STEPS
        push!(manifest_rows, (entry_type = "direct_gbm", name = "step$(step)", count = 240.0,
                              value = "d2_r3_nb8"))
    end
    for name in written
        path = joinpath(directory, name)
        push!(manifest_rows, (entry_type = "sha256", name = name, count = Float64(filesize(path)),
                              value = _sha256_file(path)))
    end
    CSV.write(joinpath(directory, "manifest.csv"), DataFrame(manifest_rows))
    # A tampered file is edited after its digest was recorded, which is exactly the case the manifest
    # exists to catch.
    if mutate === :tampered_stack_file
        open(joinpath(directory, "stack_weights.csv"), "a") do io
            println(io, "# edited after the digest was recorded")
        end
    end

    return (dir = directory, frame = frame, origins = origins, stats = stats,
            files = written, weights = base, deep_weights = deep, tau = tau,
            served_variant = served_variant, guard_applied = guard_applied,
            fold_year = fold_year, oof_pool_years = oof_pool_years,
            half_width = _fixture_half_width, calibration = calibration,
            design_names = names_29)
end

end # module
