# operational_v22_serving.jl — Operational V2.2 static regime stack as the served point center.
#
# The V2.3 analog-driver study returned `NO_GO` on its single-shot confirmatory partition, so the
# served center does not become the analog candidate. The static V2.2 stack does: a fitted, bounded,
# SINDy-dominant convex combination of the six point components the live engine already computes
# (served V2.1, frozen V2.1, persistence, Burton, Burton-full, O'Brien-McPherron), selected per
# model step and per causal issue-time regime.
#
# The stack itself is `operational_v22.jl`; this file is only the serving contract around it, and it
# exists so that one code path forms the served center in both environments:
#
#   * the live engine calls `v22_serving_center` with its own six component values;
#   * the offline identity oracle calls the same function with the archived base-table columns and
#     checks the result against the archived `static_v2_2_dst_nt`.
#
# The two inputs that are easy to get wrong are the regime inputs. The stack's regime is a function
# of the issue-time Dst, the one-hour Dst rate and the *gated* coupling proxy, and the gate is not
# the raw coupling: `coupling_active_mvm` is the rectified `VBs` only while the wind is driving and
# the ring current is deepening, and zero otherwise. `v22_serving_coupling_active` restates that gate
# so the live engine cannot drift from the definition the stack was fitted under.

import SHA

"""
    V22_SERVED_STACK_LABEL

Label of the fitted stack this package serves. The label records the fitting recipe and the fitted
row count, so a stack CSV with a different label is a different model and is refused rather than
served under this identity.
"""
const V22_SERVED_STACK_LABEL = "operational_v2_2_primary_sindy60_fit407598"

"SHA-256 of the served V2.2 stack weights."
const V22_SERVED_STACK_SHA256 =
    "66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d"

"File name of the served stack weights inside the deployment directory."
const V22_SERVED_STACK_FILE = "operational_v2_2_stack.csv"

"Provenance manifest shipped beside the served stack weights."
const V22_SERVED_STACK_MANIFEST = "operational_v2_2_stack_manifest.csv"

"""
    V22_SERVED_IDENTITY

Served-pipeline identity of the V2.2 product: the whole V2.1 served operator (L1 look-ahead,
regime-aware relaxation, rate projection and the three inertia guards) followed by the static regime
stack. Every stage that can move the published center appears in the label.
"""
const V22_SERVED_IDENTITY =
    "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"

"Driver assumption recorded with a V2.2 served row."
const V22_SERVED_DRIVER_ASSUMPTION =
    "ballistically_propagated_l1_then_regime_aware_relaxation_then_rate_projection_then_one_hour_" *
    "inertia_blend_then_state_inertia_then_extreme_inertia_guard_then_static_regime_stack"

"Physical projection applied to the served V2.2 center (nT)."
const V22_SERVED_DST_FLOOR_NT = -2000.0
const V22_SERVED_DST_CEIL_NT = 50.0

"""
    v22_serving_coupling_active(vbsouth_mvm, dst_delta_1h_nt) -> Float64

The gated coupling proxy the stack's regime rule consumes, restating
`add_operational_v2_features!`'s `coupling_active_mvm`: the rectified southward coupling
`VBs = 1e-3 * V * max(-Bz, 0)` counts only while the wind is driving (`VBs > 0`) and the ring current
is deepening (`dDst/dt < 0`), and is zero otherwise. A non-finite input is neutral, matching the
archived definition, so a driver gap cannot promote an issue into the active-deepening regime.
"""
function v22_serving_coupling_active(vbsouth_mvm::Real, dst_delta_1h_nt::Real)
    coupling = Float64(vbsouth_mvm)
    rate = Float64(dst_delta_1h_nt)
    (isfinite(coupling) && coupling > 0.0 && isfinite(rate) && rate < 0.0) || return 0.0
    return coupling
end

"""
    v22_serving_stack_manifest_rows(path; source) -> Vector{NamedTuple}

Provenance rows for the shipped stack weights: where the fitted file came from, its label, its size
and its digest. Written beside the weights so a deployment records the fit it serves.
"""
function v22_serving_stack_manifest_rows(path::AbstractString; source::AbstractString = "")
    file = String(path)
    stack = read_operational_v22_stack(file)
    return NamedTuple[
        (entry_type = "sha256", name = basename(file), count = Float64(filesize(file)),
         value = v22_serving_stack_sha256(file)),
        (entry_type = "stack", name = "label", count = NaN, value = stack.label),
        (entry_type = "stack", name = "supported_model_steps", count = NaN,
         value = join(stack.supported_model_steps, ";")),
        (entry_type = "stack", name = "sindy_mass_floor", count = stack.sindy_mass_floor,
         value = ""),
        (entry_type = "stack", name = "identity", count = NaN, value = V22_SERVED_IDENTITY),
        (entry_type = "source", name = "fitted_stack_path", count = NaN, value = String(source)),
    ]
end

"SHA-256 of a regular non-symlink file, as lowercase hex."
function v22_serving_stack_sha256(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2 stack provenance source must be a regular non-symlink file: $source",
    ))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

"""
    load_v22_serving_stack(path; expect_label, expect_sha256) -> OperationalV22Stack

Read the served stack weights and refuse anything but the fitted stack this package publishes: the
digest and the label must both match. Serving an unpinned stack under the published identity would
make the identity meaningless, so both checks are errors rather than warnings.

Pass `expect_sha256 = ""` to skip the digest check; that is for test fixtures which necessarily carry
their own weights, never for a deployment.
"""
function load_v22_serving_stack(path::AbstractString;
                               expect_label::AbstractString = V22_SERVED_STACK_LABEL,
                               expect_sha256::AbstractString = V22_SERVED_STACK_SHA256)
    file = String(path)
    isfile(file) || error("served V2.2 stack weights not found: $file")
    if !isempty(expect_sha256)
        digest = v22_serving_stack_sha256(file)
        digest == String(expect_sha256) || error(
            "served V2.2 stack $file fails its pinned digest: expected $expect_sha256, computed " *
            "$digest",
        )
    end
    stack = read_operational_v22_stack(file)
    isempty(expect_label) || stack.label == String(expect_label) || error(
        "served V2.2 stack $file carries label $(stack.label); the published product is " *
        "$expect_label",
    )
    return stack
end

"""
    v22_serving_center(stack; model_steps, latest_dst, dst_delta_1h_nt, vbsouth_mvm,
                       served_v2_1, frozen_v2_1, persistence, burton, burton_full,
                       obrien) -> NamedTuple

Served V2.2 point center of one `(anchor, model step)` row.

The regime is chosen from issue-time state only — the observed Dst, its one-hour rate and the gated
coupling proxy — and the stack's per-step, per-regime convex weights are applied to the six component
centers. The result is projected to the physical Dst range; the projection is a no-op on a convex
combination of in-range components and only guards against a non-finite component reaching the log.

Returns the served center together with the regime, the pooled-fallback flag, the component weights
and the stack label, so a caller can log why the center moved.
"""
function v22_serving_center(stack::OperationalV22Stack; model_steps::Integer,
                            latest_dst::Real, dst_delta_1h_nt::Real, vbsouth_mvm::Real,
                            served_v2_1::Real, frozen_v2_1::Real, persistence::Real,
                            burton::Real, burton_full::Real, obrien::Real)
    step = Int(model_steps)
    step in stack.supported_model_steps || throw(ArgumentError(
        "the served V2.2 stack does not support a $(step) h model step; supported steps are " *
        join(stack.supported_model_steps, ", "),
    ))
    components = (served_v2_1 = Float64(served_v2_1), frozen_v2_1 = Float64(frozen_v2_1),
                  persistence = Float64(persistence), burton = Float64(burton),
                  burton_full = Float64(burton_full), obrien = Float64(obrien))
    all(isfinite, values(components)) || throw(ArgumentError(
        "every V2.2 stack component center must be finite, got $components",
    ))
    coupling = v22_serving_coupling_active(vbsouth_mvm, dst_delta_1h_nt)
    rate = isfinite(Float64(dst_delta_1h_nt)) ? Float64(dst_delta_1h_nt) : 0.0
    prediction = operational_v22_predict(stack, step, Float64(latest_dst), rate, coupling,
                                         components)
    center = clamp(prediction.pred_dst, V22_SERVED_DST_FLOOR_NT, V22_SERVED_DST_CEIL_NT)
    return (center = center, raw_center = prediction.pred_dst, regime = prediction.regime,
            cell_regime = prediction.cell_regime,
            used_pooled_fallback = prediction.used_pooled_fallback,
            coupling_active_mvm = coupling, weights = prediction.weights,
            sindy_mass = prediction.sindy_mass, label = prediction.label,
            model_step_hours = step)
end

# The depth-safe alerting center is defined once, in a dependency-free file the dashboard
# application includes as well, so the published severity cannot drift from this contract.
include("serving_depth_safe.jl")

"Path of the shared depth-safe-center definition, so the application can locate it."
const V22_SERVING_DEPTH_SAFE_FILE = "serving_depth_safe.jl"
