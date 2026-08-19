#!/usr/bin/env julia

# v2_3_t1r.jl — Amendment A1 of the Operational V2.3 protocol: the analog raw
# core with a ridge correction refit by the V2.1 recipe.
#
# Why the amendment exists. The deployed V2.1 point calibration
# (`deploy/operational_v2_calibration.csv`, label
# `operational_v2_1_memory_expert_lead_ridge100.0_fit407598`) is a 27-parameter
# linear layer fit on the residuals of the *frozen-driver* SINDy core. Part of
# what it learned is that core's persistent over-injection. The analog driver
# continuation removes much of that over-injection at the source, so applying
# the frozen-core correction to the analog core subtracts a bias that is no
# longer there and cancels most of the analog gain (V2_3_DEVELOPMENT_REPORT.md,
# interim reading 2). T1r keeps the ridge layer but refits it, with the identical
# recipe, on the analog core's out-of-fold residuals.
#
# "Identical recipe" is a hard requirement here, not a convenience: the same 26
# issue-time features, the same ridge strength 100, the same standardization
# (per-feature mean/sd of the fitted rows, unpenalized intercept) and the same
# `fit_operational_v2_calibration` entry point. The only difference is which raw
# core the residual is taken against — which is exactly the quantity Amendment A1
# is testing. Five of the 26 features are functions of that raw core
# (`V23_T1R_RAW_CORE_FEATURES`); for T1r they are recomputed from the analog raw
# core, because the V2.1 recipe defines them relative to the core being
# corrected. The executable oracle for that recomputation is
# [`v23_t1r_feature_oracle`](@ref): feeding the *frozen* raw core back through the
# same path must reproduce the archived base-table feature columns.
#
# Leakage. For DEV selection the correction is cross-fitted leave-one-block-out
# over the six inner blocks, so a block's centers never see that block's
# observations. The layer is a temporal cross-fit (blocks are calendar years
# inside DEV, and blocks after b contribute to b's fit); Amendment A1 discloses
# this. For the confirmatory run the layer is fit once on all six blocks'
# out-of-fold rows, which is what `t1r_fit_all_dev_<config>.csv` records.
#
# Interval layer. T1r is a point layer. The base table carries no ensemble
# interval, so the fit is handed the placeholder half-width
# `V23_T1R_PLACEHOLDER_HALF_WIDTH_NT` and the resulting `interval_scale` is a
# nominal quantile of the absolute residual, not a deployable interval. Nothing
# in this file consumes it, and `v23_t1r_apply` uses only the point coefficients.
#
# Run from the package root:
#   JULIA_NUM_THREADS=4 julia --startup-file=no --project=. validation/operational/v2_3_t1r.jl

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

# ---------------------------------------------------------------------------
# Recipe constants (all pinned to the deployed V2.1 point calibration)
# ---------------------------------------------------------------------------

"Ridge strength of the deployed V2.1 point layer, reused unchanged."
const V23_T1R_RIDGE = 100.0
"Interval coverage argument of the deployed recipe (point layer is unaffected)."
const V23_T1R_INTERVAL_COVERAGE = 0.90
"Selector guard margin of the deployed recipe."
const V23_T1R_GUARD_MARGIN_NT = 0.5
"Label infix that records the recipe and the ridge strength."
const V23_T1R_LABEL_INFIX = "memory_expert_lead_ridge"
"Nominal half-width handed to the fit; the base table carries no ensemble interval."
const V23_T1R_PLACEHOLDER_HALF_WIDTH_NT = 1.0
"Family label written into the summary artifacts."
const V23_T1R_FAMILY = "T1r"

"""
    V23_T1R_RAW_CORE_FEATURES

The calibration features that are functions of the raw core being corrected, as
`SolarSINDy.add_operational_v2_features!` defines them: the expert differences
against the three physical baselines and persistence, the spread of the baseline
panel (which contains the core itself), and the lead interaction built on the
core-minus-persistence difference. Every other calibration feature is issue-time
state and is identical for the frozen and the analog path.
"""
const V23_T1R_RAW_CORE_FEATURES = (:baseline_spread_nt, :v1_minus_persistence_nt,
                                   :obrien_minus_v1_nt, :burton_minus_v1_nt,
                                   :lead_v1_persistence_interaction)

"""
    V23_T1R_PASSTHROUGH_INPUTS

Base-table columns copied unchanged into the feature frame. They are the raw
issue-time state, the timestamp-keyed memory features of the frozen replay
(none of which depend on the raw core), and the baseline panel the expert
features are built from.
"""
const V23_T1R_PASSTHROUGH_INPUTS = (:latest_dst_nt, :V_kms, :Bz_nt, :By_nt, :n_cm3,
                                    :Pdyn_npa, :dst_delta_1h_nt, :dst_delta_3h_nt,
                                    :Bz_delta_1h_nt, :VBsouth_delta_1h_mvm,
                                    :VBsouth_mean_3h_mvm, :Bsouth_mean_3h_nt,
                                    :persistence_dst_nt, :burton_dst_nt,
                                    :burton_full_dst_nt, :obrien_dst_nt)

"Columns `fit_operational_v2_calibration` requires beyond the feature list."
const V23_T1R_FIT_REQUIRED = (:pred_dst_nt, :observation_dst_nt,
                              :pred_dst_ci05_nt, :pred_dst_ci95_nt)

"""
    v23_t1r_base_table_columns() -> Vector{String}

Base-table columns a T1r application needs: the row key, the partition label, the
raw issue-time inputs [`v23_t1r_features`](@ref) passes through, the frozen raw
core, and the 26 archived calibration feature columns that make
[`v23_t1r_feature_oracle`](@ref) non-vacuous. The confirmatory runner reads the
base table with exactly this selection, because the full table is far larger than
the T1r layer needs and it is already resident once inside the run context.
"""
function v23_t1r_base_table_columns()
    columns = String["issue_time_utc", "model_step_hours", "partition"]
    for c in V23_T1R_PASSTHROUGH_INPUTS
        String(c) in columns || push!(columns, String(c))
    end
    for c in v23_t1r_deployed_calibration().feature_names
        String(c) in columns || push!(columns, String(c))
    end
    "pred_dst_nt" in columns || push!(columns, "pred_dst_nt")
    return columns
end

const _V23_T1R_DEPLOYED = Ref{Any}(nothing)

"""
    v23_t1r_deployed_calibration() -> OperationalV2Calibration

The deployed V2.1 point calibration. It is the authority for the feature list,
the label grammar and the selector configuration that T1r must reproduce.
"""
function v23_t1r_deployed_calibration()
    if _V23_T1R_DEPLOYED[] === nothing
        _V23_T1R_DEPLOYED[] = read_operational_v2_calibration(
            operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
        )
    end
    return _V23_T1R_DEPLOYED[]
end

# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------

"""
    v23_t1r_features(base_rows, analog_raw) -> DataFrame

Calibration feature frame of the analog path. `base_rows` are base-table rows
(one per issue and model step) and `analog_raw` is the analog raw core of the
same rows, in the same order.

The frame is built by handing `SolarSINDy.add_operational_v2_features!` the raw
inputs with `pred_dst_nt` set to the analog core, so every derived feature —
including the five raw-core-dependent ones — comes out of the same code path the
deployed V2.1 calibration was fit with. Features that do not depend on the core
(the issue-time state and the timestamp-keyed memory lags) are passed through
from the base table rather than recomputed, because the base table is the
archived frozen-replay record of those quantities.

The returned frame carries the 26 calibration features, the raw core as
`pred_dst_nt`, the placeholder interval columns, the baseline panel, and the
`issue_time_utc` / `model_step_hours` / `cv_block` / `observation_dst_nt` columns
the cross-fit and the metrics need.
"""
function v23_t1r_features(base_rows::DataFrame, analog_raw::AbstractVector{<:Real})
    n = nrow(base_rows)
    length(analog_raw) == n || throw(DimensionMismatch(
        "analog raw core has $(length(analog_raw)) entries for $n base rows",
    ))
    src = DataFrame()
    String(:issue_time_utc) in names(base_rows) || throw(ArgumentError(
        "base rows must carry issue_time_utc: the memory features are timestamp-keyed",
    ))
    src[!, :issue_time_utc] = collect(base_rows.issue_time_utc)
    src[!, :model_step_hours] = Int.(base_rows.model_step_hours)
    for c in V23_T1R_PASSTHROUGH_INPUTS
        String(c) in names(base_rows) || throw(ArgumentError(
            "base rows lack the calibration input column $(String(c))",
        ))
        src[!, c] = Float64.(base_rows[!, c])
    end
    for c in (:cv_block,)
        String(c) in names(base_rows) && (src[!, c] = Int.(base_rows[!, c]))
    end
    String(:observation_dst_nt) in names(base_rows) &&
        (src[!, :observation_dst_nt] = Float64.(base_rows.observation_dst_nt))
    src[!, :pred_dst_nt] = Float64.(analog_raw)
    # Point layer: the placeholder interval satisfies the fit's column contract
    # and is never read back (see the file header).
    src[!, :pred_dst_ci05_nt] = src.pred_dst_nt .- V23_T1R_PLACEHOLDER_HALF_WIDTH_NT
    src[!, :pred_dst_ci95_nt] = src.pred_dst_nt .+ V23_T1R_PLACEHOLDER_HALF_WIDTH_NT
    SolarSINDy.add_operational_v2_features!(src)
    keep = Symbol[:issue_time_utc, :model_step_hours]
    String(:cv_block) in names(src) && push!(keep, :cv_block)
    String(:observation_dst_nt) in names(src) && push!(keep, :observation_dst_nt)
    append!(keep, (:pred_dst_nt, :pred_dst_ci05_nt, :pred_dst_ci95_nt))
    append!(keep, (:persistence_dst_nt, :burton_dst_nt, :burton_full_dst_nt, :obrien_dst_nt))
    for c in v23_t1r_deployed_calibration().feature_names
        c in keep || push!(keep, c)
    end
    return src[!, keep]
end

"""
    v23_t1r_feature_oracle(base_rows; atol = 1e-9) -> NamedTuple

Independent check of [`v23_t1r_features`](@ref): recomputing the feature frame
from the *frozen* raw core (`pred_dst_nt` of the base table) must reproduce every
archived calibration feature column of those base rows. This is an oracle rather
than a restatement — the frozen columns were written by the base-table build from
the frozen V2.1 replay, while the values compared against them are rebuilt here
from the raw inputs through the deployed feature path. A recomputation that
silently dropped, reordered or mis-signed one of the five raw-core-dependent
features would fail here before any analog fit is attempted.

Returns the worst absolute deviation, the column that carries it, and the number
of rows checked.
"""
function v23_t1r_feature_oracle(base_rows::DataFrame; atol::Real=1e-9)
    frozen = v23_t1r_features(base_rows, Float64.(base_rows.pred_dst_nt))
    worst = 0.0
    worst_column = :none
    checked = Symbol[]
    for c in v23_t1r_deployed_calibration().feature_names
        String(c) in names(base_rows) || continue
        push!(checked, c)
        deviation = maximum(abs.(frozen[!, c] .- Float64.(base_rows[!, c])))
        if deviation > worst
            worst = deviation
            worst_column = c
        end
    end
    length(checked) == length(v23_t1r_deployed_calibration().feature_names) || error(
        "base rows carry only $(length(checked)) of the " *
        "$(length(v23_t1r_deployed_calibration().feature_names)) calibration feature columns; " *
        "the frozen-feature oracle would be vacuous",
    )
    all(c in checked for c in V23_T1R_RAW_CORE_FEATURES) || error(
        "the frozen-feature oracle did not cover the raw-core-dependent features",
    )
    worst <= Float64(atol) || error(
        "recomputed frozen features disagree with the base table by $worst at $(worst_column)",
    )
    return (worst_nt=worst, column=worst_column, n=nrow(base_rows), columns=checked)
end

# ---------------------------------------------------------------------------
# Fit and apply
# ---------------------------------------------------------------------------

"""
    v23_t1r_finite_mask(df) -> BitVector

Rows the V2.1 fit would keep: every required column finite. Mirrors the
package's internal row filter so the fitted row count is known before the fit
and can be written into the label the way the deployed label records it
(`…_fit407598`). `test/test_v2_3_t1r.jl` pins this against
`SolarSINDy._finite_rows`.
"""
function v23_t1r_finite_mask(df::DataFrame)
    required = vcat(collect(V23_T1R_FIT_REQUIRED), v23_t1r_deployed_calibration().feature_names)
    mask = trues(nrow(df))
    for c in required
        String(c) in names(df) || throw(ArgumentError(
            "fit frame lacks the required column $(String(c))",
        ))
        column = df[!, c]
        @inbounds for i in 1:nrow(df)
            value = column[i]
            mask[i] &= !ismissing(value) && isfinite(Float64(value))
        end
    end
    return mask
end

"""
    v23_t1r_fit(train_df; label_stem) -> OperationalV2Calibration

Fit the T1r correction on `train_df` with the deployed V2.1 recipe: the deployed
26-feature list, ridge 100, the deployed interval coverage and selector guard
margin, through `fit_operational_v2_calibration`. `train_df` must carry the
analog raw core as `pred_dst_nt`.

The recipe equalities are asserted, not assumed: feature list, coefficient
count, ridge strength recorded in the label, selector component and supported
model steps must all match the deployed calibration, so a drift in the package
recipe fails here instead of producing a T1r layer that is no longer the V2.1
layer refit.
"""
function v23_t1r_fit(train_df::DataFrame;
                     label_stem::AbstractString="operational_v2_3_t1r")
    deployed = v23_t1r_deployed_calibration()
    occursin(string(V23_T1R_LABEL_INFIX, V23_T1R_RIDGE), deployed.label) || error(
        "deployed V2.1 label $(deployed.label) no longer records the " *
        "$(V23_T1R_LABEL_INFIX)$(V23_T1R_RIDGE) recipe T1r is required to reuse",
    )
    mask = v23_t1r_finite_mask(train_df)
    clean = train_df[mask, :]
    n_fit = nrow(clean)
    n_fit > length(deployed.feature_names) + 1 || error(
        "T1r fit has only $n_fit finite rows for $(length(deployed.feature_names)) features",
    )
    label = string(label_stem, "_", V23_T1R_LABEL_INFIX, V23_T1R_RIDGE, "_fit", n_fit)
    cal = fit_operational_v2_calibration(
        clean;
        feature_names=copy(deployed.feature_names),
        ridge=V23_T1R_RIDGE,
        interval_coverage=V23_T1R_INTERVAL_COVERAGE,
        label=label,
        guard_margin_nt=V23_T1R_GUARD_MARGIN_NT,
    )
    cal.feature_names == deployed.feature_names || error(
        "T1r fit returned a different feature list than the deployed V2.1 calibration",
    )
    length(cal.coefficients) == length(deployed.coefficients) || error(
        "T1r fit returned $(length(cal.coefficients)) coefficients, deployed has " *
        "$(length(deployed.coefficients))",
    )
    cal.label == label || error("T1r fit did not carry its label through")
    cal.guard_margin_nt == deployed.guard_margin_nt || error(
        "T1r fit used guard margin $(cal.guard_margin_nt), deployed uses $(deployed.guard_margin_nt)",
    )
    cal.selected_component == deployed.selected_component || error(
        "T1r selector chose $(cal.selected_component) where the deployed recipe serves " *
        "$(deployed.selected_component); the refit layer would not be the V2.1 layer",
    )
    (isempty(cal.supported_model_steps) ||
     cal.supported_model_steps == deployed.supported_model_steps) || error(
        "T1r fit supports steps $(cal.supported_model_steps), deployed supports " *
        "$(deployed.supported_model_steps)",
    )
    return cal
end

# Row-wise correction through the deployed evaluator, so a T1r center is formed
# by exactly the arithmetic the served product uses (same standardization, same
# accumulation order), not by a re-derived dot product.
function _v23_t1r_corrections(cal::OperationalV2Calibration,
                              columns::Vector{Vector{Float64}}, n::Int)
    names_tuple = Tuple(cal.feature_names)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        feats = NamedTuple{names_tuple}(Tuple(columns[j][i] for j in eachindex(columns)))
        out[i] = SolarSINDy.operational_v2_correction(cal, feats)
    end
    return out
end

"""
    v23_t1r_correction_vector(cal, df) -> Vector{Float64}

Row-wise T1r correction of `df` under `cal`, after checking that the frame
carries every calibration feature and the analog raw core. Separated from
[`v23_t1r_apply`](@ref) so that the two safeguard variants of one frame can share
a single evaluation of the 27-parameter layer instead of paying for it twice.
"""
function v23_t1r_correction_vector(cal::OperationalV2Calibration, df::DataFrame)
    for c in cal.feature_names
        String(c) in names(df) || throw(ArgumentError(
            "T1r apply frame lacks the calibration feature $(String(c))",
        ))
    end
    String(:pred_dst_nt) in names(df) || throw(ArgumentError(
        "T1r apply frame lacks pred_dst_nt (the analog raw core)",
    ))
    columns = Vector{Float64}[Float64.(df[!, c]) for c in cal.feature_names]
    return _v23_t1r_corrections(cal, columns, nrow(df))
end

"Form the reported center from a precomputed correction vector."
function _v23_t1r_from_corrections(df::DataFrame, corrections::AbstractVector{Float64};
                                   safeguards::Bool)
    n = nrow(df)
    length(corrections) == n || throw(DimensionMismatch(
        "correction vector has $(length(corrections)) entries for $n rows",
    ))
    raw = Float64.(df.pred_dst_nt)
    centers = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        centers[i] = clamp(raw[i] + corrections[i], -2000.0, 50.0)
    end
    safeguards || return centers
    for c in (:latest_dst_nt, :model_step_hours, :dst_delta_1h_nt)
        String(c) in names(df) || throw(ArgumentError(
            "T1r safeguard variant needs $(String(c))",
        ))
    end
    latest = Float64.(df.latest_dst_nt)
    steps = Int.(df.model_step_hours)
    rate = Float64.(df.dst_delta_1h_nt)
    guarded = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        guarded[i] = _apply_v2_1_safeguards(
            centers[i], latest[i], steps[i], isfinite(rate[i]) ? rate[i] : 0.0,
        )
    end
    return guarded
end

"""
    v23_t1r_apply(cal, df; safeguards = false) -> Vector{Float64}

T1r centers of `df` under `cal`. Without safeguards the center is
`clamp(analog raw + correction, −2000, 50)`; with `safeguards = true` the
published V2.1 point operator `_apply_v2_1_safeguards` is applied on top, using
the row's latest Dst, model step and one-hour Dst rate. The operator projects its
input to the same physical range before acting, so passing the projected center
is identical to passing the unprojected sum the runner's `v23_center` forms;
`test/test_v2_3_t1r.jl` pins that equality.
"""
function v23_t1r_apply(cal::OperationalV2Calibration, df::DataFrame;
                       safeguards::Bool=false)
    corrections = v23_t1r_correction_vector(cal, df)
    return _v23_t1r_from_corrections(df, corrections; safeguards=safeguards)
end

"""
    v23_t1r_centers(cal, df) -> NamedTuple

Both safeguard variants of the T1r center from one evaluation of the correction
layer: `off` is the projected sum and `on` carries the published V2.1 point
operator. Element for element this is `v23_t1r_apply(cal, df; safeguards = …)`
for the two options, which `test/test_v2_3_t1r.jl` pins exactly; the confirmatory
runner uses it because it scores both options for every analog configuration and
the row-wise layer evaluation dominates the cost.
"""
function v23_t1r_centers(cal::OperationalV2Calibration, df::DataFrame)
    corrections = v23_t1r_correction_vector(cal, df)
    return (off=_v23_t1r_from_corrections(df, corrections; safeguards=false),
            on=_v23_t1r_from_corrections(df, corrections; safeguards=true),
            corrections=corrections)
end

# ---------------------------------------------------------------------------
# Leave-one-block-out cross-fit
# ---------------------------------------------------------------------------

"""
    v23_t1r_crossfit(features, blocks; label_stem, fit_mask) -> NamedTuple

Leave-one-block-out cross-fit of the T1r correction over the inner blocks. For
block `b` the layer is fit on the rows of the other blocks and applied to block
`b`, so no block's centers are informed by that block's observations. `fit_mask`,
when given, additionally excludes rows from every fit (used to keep served
fallback rows, whose raw core is not the analog core, out of the fitted
residuals).

Returns the two center vectors, aligned to the rows of `features`, and the
per-block calibrations.
"""
function v23_t1r_crossfit(features::DataFrame, blocks::AbstractVector{<:Integer};
                          label_stem::AbstractString="operational_v2_3_t1r",
                          fit_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    String(:cv_block) in names(features) || throw(ArgumentError(
        "cross-fit needs the cv_block column",
    ))
    n = nrow(features)
    usable = fit_mask === nothing ? trues(n) : BitVector(fit_mask)
    length(usable) == n || throw(DimensionMismatch(
        "fit mask has $(length(usable)) entries for $n rows",
    ))
    block_column = Int.(features.cv_block)
    center_off = fill(NaN, n)
    center_on = fill(NaN, n)
    calibrations = Dict{Int,OperationalV2Calibration}()
    for b in blocks
        query = findall(==(Int(b)), block_column)
        isempty(query) && error("cross-fit block $b has no rows")
        train = [i for i in 1:n if block_column[i] != Int(b) && usable[i]]
        isempty(train) && error("cross-fit block $b has an empty training set")
        cal = v23_t1r_fit(features[train, :]; label_stem="$(label_stem)_xf$(b)")
        query_frame = features[query, :]
        center_off[query] = v23_t1r_apply(cal, query_frame)
        center_on[query] = v23_t1r_apply(cal, query_frame; safeguards=true)
        calibrations[Int(b)] = cal
    end
    return (center_off=center_off, center_on=center_on, calibrations=calibrations)
end

# ---------------------------------------------------------------------------
# Development-run artifacts
# ---------------------------------------------------------------------------

v23_t1r_id(config::AbstractString) = "T1r_$(config)"
v23_t1r_oof_path(outdir, config) = joinpath(outdir, "oof_$(v23_t1r_id(config)).csv")
v23_t1r_summary_path(outdir, config) = joinpath(outdir, "summary_$(v23_t1r_id(config)).csv")
v23_t1r_calibration_path(outdir, config) = joinpath(outdir, "t1r_fit_all_dev_$(config).csv")
v23_t1r_manifest_path(outdir) = joinpath(outdir, "t1r_manifest.csv")
v23_t1r_report_path(outdir) = joinpath(outdir, "t1r_dev_report.csv")
v23_t1r_signature_path(outdir, config) = joinpath(outdir, "sig_T1r_$(config).txt")

"""
    v23_t1r_signature(outdir, config; table_sha, code_sha) -> String

Provenance key of one T1r artifact set: the analog out-of-fold core it corrects,
the base table the calibration features are read from, the V2.3 source signature
and this file. T1r is written outside the stage machinery of
`v2_3_common.jl`, so without this key a summary or an all-DEV calibration left
over from an earlier source revision would be consumed by the selection rule and
by the confirmatory run without anything noticing.
"""
function v23_t1r_signature(outdir::AbstractString, config::AbstractString;
                           table_sha::AbstractString,
                           code_sha::AbstractString=v23_code_signature())
    source_oof = joinpath(outdir, "oof_$(config).csv")
    isfile(source_oof) || error("analog out-of-fold artifact is missing: $source_oof")
    payload = join([
        "config=$(String(config))",
        "source_oof_sha256=$(_v23_t1r_sha256(source_oof))",
        "base_table_sha256=$(String(table_sha))",
        "code_sha256=$(String(code_sha))",
        "v2_3_t1r_sha256=$(_v23_t1r_sha256(joinpath(@__DIR__, "v2_3_t1r.jl")))",
    ], "\n")
    return bytes2hex(sha256(payload))
end

"""
    v23_t1r_signature_ok(outdir, config; table_sha, code_sha) -> Bool

Whether the persisted T1r artifacts of `config` may be consumed: the out-of-fold
rows, the metric summary, the all-DEV calibration and a signature file that still
matches must all be present.
"""
function v23_t1r_signature_ok(outdir::AbstractString, config::AbstractString;
                              table_sha::AbstractString,
                              code_sha::AbstractString=v23_code_signature())
    signature_path = v23_t1r_signature_path(outdir, config)
    for path in (signature_path, v23_t1r_oof_path(outdir, config),
                 v23_t1r_summary_path(outdir, config),
                 v23_t1r_calibration_path(outdir, config))
        isfile(path) || return false
    end
    expected = v23_t1r_signature(outdir, config; table_sha=table_sha, code_sha=code_sha)
    return strip(read(signature_path, String)) == expected
end

"""
    v23_t1r_require_signature(outdir, config; table_sha, code_sha)

Fail closed when a T1r artifact set has no valid signature. The selection rule
and the confirmatory run both consume these files as if they had been produced
by the current sources; a missing or stale key means they were not, and guessing
is how a superseded correction layer reaches the single-shot partition.
"""
function v23_t1r_require_signature(outdir::AbstractString, config::AbstractString;
                                   table_sha::AbstractString,
                                   code_sha::AbstractString=v23_code_signature())
    v23_t1r_signature_ok(outdir, config; table_sha=table_sha, code_sha=code_sha) && return true
    signature_path = v23_t1r_signature_path(outdir, config)
    reason = isfile(signature_path) ? "is stale" : "is missing"
    error(
        "the T1r artifacts of $config cannot be used: $(signature_path) $reason. Rerun " *
        "v2_3_t1r.jl for $config so its out-of-fold rows, summary and all-DEV calibration " *
        "carry a signature over the current analog core, base table and sources.",
    )
end

"""
    v23_t1r_available_configs(dir) -> Vector{String}

Analog configurations whose out-of-fold raw cores are already on disk. T1r is a
correction layer over those cores, so a configuration is available exactly when
its `oof_<config>.csv` exists.
"""
function v23_t1r_available_configs(dir::AbstractString=V23_DEV_DIR)
    isdir(dir) || return String[]
    out = String[]
    for file in readdir(dir)
        m = match(r"^oof_(T1a?_[a-z]+_K\d+)\.csv$", file)
        m === nothing || push!(out, String(m.captures[1]))
    end
    return sort(out)
end

"Parameter record of one T1r configuration, inheriting the analog parameters."
function v23_t1r_params(outdir::AbstractString, config::AbstractString)
    params = Dict{String,Any}(
        "tail" => "T1r",
        "source_config" => String(config),
        "correction" => "refit_on_analog_raw_core",
        "ridge" => V23_T1R_RIDGE,
        "features" => length(v23_t1r_deployed_calibration().feature_names),
        "crossfit" => "leave_one_block_out",
    )
    source_summary = joinpath(outdir, "summary_$(config).csv")
    if isfile(source_summary)
        summary = CSV.read(source_summary, DataFrame; limit=1)
        if nrow(summary) == 1 && String(:params_json) in names(summary)
            for (k, v) in JSON3.read(String(summary.params_json[1]))
                key = String(k)
                key == "tail" && continue
                params[key] = v
            end
        end
    end
    return params
end

_v23_t1r_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

"""
    v23_t1r_run_config!(config, table, anchors, masks, rowindex, blocks; outdir)

Score one analog configuration's T1r layer on DEV and persist its artifacts: the
cross-fitted out-of-fold centers, the per-step and per-cell metrics of both
safeguard variants in the schema the other development configurations use, and
the all-DEV calibration the confirmatory run would carry.
"""
function v23_t1r_run_config!(config::AbstractString, table::DataFrame,
                             anchors::V23Anchors, masks, rowindex::Dict{Tuple{DateTime,Int},Int},
                             blocks::AbstractVector{<:Integer};
                             outdir::AbstractString=V23_DEV_DIR,
                             table_sha::Union{Nothing,AbstractString}=nothing,
                             code_sha::AbstractString=v23_code_signature())
    t0 = time()
    source_oof = joinpath(outdir, "oof_$(config).csv")
    isfile(source_oof) || error("analog out-of-fold artifact is missing: $source_oof")
    oof = CSV.read(source_oof, DataFrame; types=Dict("issue_time_utc" => DateTime))
    nrow(oof) > 0 || error("analog out-of-fold artifact $source_oof is empty")
    rows_index = Vector{Int}(undef, nrow(oof))
    for r in 1:nrow(oof)
        key = (oof.issue_time_utc[r], Int(oof.model_step_hours[r]))
        idx = get(rowindex, key, 0)
        idx == 0 && error("out-of-fold row $(key) of $config has no base-table row")
        rows_index[r] = idx
    end
    base_rows = table[rows_index, :]
    all(Int.(base_rows.cv_block) .== Int.(oof.cv_block)) || error(
        "$config out-of-fold blocks disagree with the base table",
    )
    oracle = v23_t1r_feature_oracle(base_rows)
    features = v23_t1r_features(base_rows, Float64.(oof.raw_dst_nt))
    fallback = Bool.(oof.fallback)
    n_fallback = count(fallback)
    crossfit = v23_t1r_crossfit(
        features, blocks;
        label_stem="operational_v2_3_t1r_$(config)",
        fit_mask=.!fallback,
    )

    n_anchor = length(anchors)
    centers_on = fill(NaN, n_anchor, V23_STEP_COUNT)
    centers_off = fill(NaN, n_anchor, V23_STEP_COUNT)
    fallback_anchor = fill(false, n_anchor)
    seen = falses(n_anchor)
    written = NamedTuple[]
    sizehint!(written, nrow(oof))
    for r in 1:nrow(oof)
        i = get(anchors.index, oof.issue_time_utc[r], 0)
        i == 0 && error("out-of-fold row of $config holds an unknown issue time")
        slot = V23_STEP_SLOT[Int(oof.model_step_hours[r])]
        slot == 0 && error("out-of-fold row of $config holds an unscored model step")
        anchors.present[i, slot] || error(
            "out-of-fold row of $config scores a target the base table does not hold",
        )
        if fallback[r]
            # A served fallback anchor has no analog core to correct; it takes the
            # served product, exactly as `v23_config_centers` does for the analog
            # configurations themselves.
            fallback_anchor[i] = true
            centers_on[i, slot] = anchors.served[i, slot]
            centers_off[i, slot] = anchors.served[i, slot]
        else
            centers_on[i, slot] = crossfit.center_on[r]
            centers_off[i, slot] = crossfit.center_off[r]
        end
        seen[i] = true
        push!(written, (
            issue_time_utc=string(oof.issue_time_utc[r]),
            cv_block=Int(oof.cv_block[r]),
            model_step_hours=Int(oof.model_step_hours[r]),
            center_s_on_dst_nt=centers_on[i, slot],
            center_s_off_dst_nt=centers_off[i, slot],
        ))
    end
    scored_rows = findall(seen)
    mkpath(outdir)
    CSV.write(v23_t1r_oof_path(outdir, config), DataFrame(written))

    params = v23_t1r_params(outdir, config)
    seconds = time() - t0
    metrics = NamedTuple[]
    for (safeguards, centers) in ((true, centers_on), (false, centers_off))
        id = "$(v23_t1r_id(config))_S$(safeguards ? "on" : "off")"
        for row in v23_metric_rows(id, centers, anchors, masks, scored_rows, fallback_anchor)
            push!(metrics, merge(row, (
                family=V23_T1R_FAMILY,
                safeguards=safeguards,
                params_json=JSON3.write(params),
                seconds=Float64(seconds),
            )))
        end
    end
    CSV.write(v23_t1r_summary_path(outdir, config), DataFrame(metrics))

    # Confirmatory-path layer: one fit on every out-of-fold row of all six blocks.
    full = v23_t1r_fit(features[.!fallback, :];
                       label_stem="operational_v2_3_t1r_$(config)_alldev")
    write_operational_v2_calibration(v23_t1r_calibration_path(outdir, config), full)

    # Provenance key, written last so it can only exist beside a complete set.
    signature = table_sha === nothing ? nothing :
        v23_t1r_signature(outdir, config; table_sha=table_sha, code_sha=code_sha)
    signature === nothing || write(v23_t1r_signature_path(outdir, config), signature)

    return (config=String(config), metrics=metrics, centers_on=centers_on,
            centers_off=centers_off, rows=scored_rows, fallback=fallback_anchor,
            n_fallback=n_fallback, calibration=full, crossfit=crossfit.calibrations,
            oracle=oracle, seconds=seconds, params=params, signature=signature)
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

"Pooled RMSE of a comparator matrix over the scored rows at one step slot."
function v23_t1r_pooled(matrix::AbstractMatrix{Float64}, anchors::V23Anchors,
                        rows::AbstractVector{Int}, slot::Int)
    return first(v23_pooled_rmse(matrix, anchors, rows, slot))
end

"""
    v23_t1r_report_rows(result, anchors, masks) -> Vector{NamedTuple}

Per-step comparison of a T1r configuration against the comparators the plan
names for DEV out-of-fold reading: the deployed served V2.1 product, the
information-equivalent frozen-tail V2.1, and the static V2.2 stack, pooled over
all scored rows and over the disturbed cell (latest Dst ≤ −50 nT).
"""
function v23_t1r_report_rows(result, anchors::V23Anchors, masks)
    rows = result.rows
    storm = [i for i in rows if masks.latest_le_m50[i]]
    out = NamedTuple[]
    for (safeguards, centers) in ((true, result.centers_on), (false, result.centers_off))
        for (slot, step) in enumerate(V23_MODEL_STEPS)
            push!(out, (
                config=result.config,
                safeguards=safeguards,
                model_step_hours=step,
                n=count(i -> anchors.present[i, slot], rows),
                rmse_t1r_nt=v23_t1r_pooled(centers, anchors, rows, slot),
                rmse_frozen_nt=v23_t1r_pooled(anchors.frozen, anchors, rows, slot),
                rmse_served_nt=v23_t1r_pooled(anchors.served, anchors, rows, slot),
                rmse_static_v22_nt=v23_t1r_pooled(anchors.static_v22, anchors, rows, slot),
                n_storm=count(i -> anchors.present[i, slot], storm),
                rmse_t1r_storm_nt=v23_t1r_pooled(centers, anchors, storm, slot),
                rmse_frozen_storm_nt=v23_t1r_pooled(anchors.frozen, anchors, storm, slot),
                rmse_served_storm_nt=v23_t1r_pooled(anchors.served, anchors, storm, slot),
                rmse_static_v22_storm_nt=v23_t1r_pooled(anchors.static_v22, anchors, storm, slot),
            ))
        end
    end
    return out
end

"Selection statistic of plan section 4: pooled DEV-OOF RMSE averaged over steps {2,3,6}."
const V23_T1R_SELECTION_STEPS = (2, 3, 6)

function v23_t1r_selection_statistic(report_rows, config::AbstractString, safeguards::Bool)
    values = Float64[]
    for step in V23_T1R_SELECTION_STEPS
        row = findfirst(r -> r.config == config && r.safeguards == safeguards &&
                             r.model_step_hours == step, report_rows)
        row === nothing && error("selection statistic misses step $step of $config")
        push!(values, report_rows[row].rmse_t1r_nt)
    end
    all(isfinite, values) || error("selection statistic of $config is not finite")
    return mean(values)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    v23_t1r_run(; outdir, configs, blocks) -> NamedTuple

Development cross-fit of the T1r layer over every available analog
configuration. Loads the DEV base table once, rebuilds the anchor view and the
cell masks with the shared runner machinery (so the T1r summaries partition
issues exactly as every other development configuration does), checks the cached
V2.1 correction against the archived frozen centers, then scores each
configuration and writes the manifest and the comparison report.
"""
function v23_t1r_run(; outdir::AbstractString=V23_DEV_DIR,
                     configs::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                     blocks::AbstractVector{<:Integer}=V23_DEV_BLOCKS,
                     table_path::AbstractString=V23_BASE_TABLE)
    t_start = time()
    selected = configs === nothing ? v23_t1r_available_configs(outdir) : collect(String.(configs))
    isempty(selected) && error("no analog out-of-fold artifacts found in $outdir")
    _v23_log("T1r configurations: $(join(selected, ", "))")
    cells_source = v23_regime_cells_source()
    cal = v23_t1r_deployed_calibration()
    length(cal.feature_names) == 26 || error(
        "deployed V2.1 calibration carries $(length(cal.feature_names)) features, expected 26",
    )
    _v23_log("  deployed calibration: $(cal.label) ($(length(cal.feature_names)) features)")
    table = v23_read_base_table(; path=table_path, partitions=("DEV",))
    table_sha = _v23_t1r_sha256(table_path)
    code_sha = v23_code_signature()
    _v23_log("  base table: $(nrow(table)) DEV rows")
    anchors = v23_build_anchors(table, cal)
    residual = v23_check_correction!(anchors, table)
    _v23_log("  correction oracle: frozen center == frozen raw + correction to $(residual) nT")
    masks = v23_cell_masks(anchors)
    rowindex = Dict{Tuple{DateTime,Int},Int}()
    sizehint!(rowindex, nrow(table))
    for i in 1:nrow(table)
        rowindex[(table.issue_time_utc[i], Int(table.model_step_hours[i]))] = i
    end

    results = Any[]
    report_rows = NamedTuple[]
    for config in selected
        _v23_log("  T1r on $config …")
        result = v23_t1r_run_config!(config, table, anchors, masks, rowindex, blocks;
                                     outdir=outdir, table_sha=table_sha, code_sha=code_sha)
        push!(results, result)
        append!(report_rows, v23_t1r_report_rows(result, anchors, masks))
        guards_on = v23_guard_flags([r for r in result.metrics if r.safeguards])
        guards_off = v23_guard_flags([r for r in result.metrics if !r.safeguards])
        _v23_log(@sprintf(
            "  + %s: feature oracle %.3g nT, %d fallback rows, guards S=on %s / S=off %s, %.1f s",
            v23_t1r_id(config), result.oracle.worst_nt, result.n_fallback,
            guards_on.guards_ok ? "ok" : "FAIL($(guards_on.guard_failures))",
            guards_off.guards_ok ? "ok" : "FAIL($(guards_off.guard_failures))",
            result.seconds,
        ))
    end

    CSV.write(v23_t1r_report_path(outdir), DataFrame(report_rows))
    manifest = NamedTuple[]
    push!(manifest, (kind="input", name="base_table", path=table_path,
                     sha256=_v23_t1r_sha256(table_path), bytes=filesize(table_path)))
    deployed_path = operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv
    push!(manifest, (kind="input", name="deployed_v2_1_calibration", path=deployed_path,
                     sha256=_v23_t1r_sha256(deployed_path), bytes=filesize(deployed_path)))
    for source in (joinpath(@__DIR__, "v2_3_t1r.jl"), joinpath(@__DIR__, "v2_3_common.jl"),
                   joinpath(@__DIR__, "v2_3_kernel.jl"))
        push!(manifest, (kind="code", name=basename(source), path=source,
                         sha256=_v23_t1r_sha256(source), bytes=filesize(source)))
    end
    for config in selected
        source_oof = joinpath(outdir, "oof_$(config).csv")
        push!(manifest, (kind="input", name="analog_oof_$(config)", path=source_oof,
                         sha256=_v23_t1r_sha256(source_oof), bytes=filesize(source_oof)))
        for path in (v23_t1r_oof_path(outdir, config), v23_t1r_summary_path(outdir, config),
                     v23_t1r_calibration_path(outdir, config),
                     v23_t1r_signature_path(outdir, config))
            isfile(path) || continue
            push!(manifest, (kind="output", name=basename(path), path=path,
                             sha256=_v23_t1r_sha256(path), bytes=filesize(path)))
        end
    end
    push!(manifest, (kind="output", name=basename(v23_t1r_report_path(outdir)),
                     path=v23_t1r_report_path(outdir),
                     sha256=_v23_t1r_sha256(v23_t1r_report_path(outdir)),
                     bytes=filesize(v23_t1r_report_path(outdir))))
    CSV.write(v23_t1r_manifest_path(outdir), DataFrame(manifest))

    elapsed = time() - t_start
    _v23_log(@sprintf("T1r development cross-fit complete in %.1f s (cells: %s)",
                      elapsed, cells_source))
    return (results=results, report=report_rows, anchors=anchors, masks=masks,
            correction_residual_nt=residual, seconds=elapsed, configs=selected)
end

"""
    v23_t1r_print_report(run)

Print the DEV out-of-fold comparison table and the plan section 4 selection
statistic of every scored T1r configuration. The guards-passing minimum is
reported separately from the unconditional minimum, because the selection rule
is "minimum subject to the section 6-A2 regime guards".
"""
function v23_t1r_print_report(run)
    println("\n== pooled DEV-OOF RMSE [nT] ==")
    println(rpad("config", 26), rpad("S", 5), rpad("step", 6), rpad("T1r", 9),
            rpad("frozen", 9), rpad("served", 9), rpad("staticV22", 11),
            rpad("T1r storm", 11), rpad("frozen storm", 14))
    for row in run.report
        println(rpad(row.config, 26), rpad(row.safeguards ? "on" : "off", 5),
                rpad(row.model_step_hours, 6),
                rpad(@sprintf("%.3f", row.rmse_t1r_nt), 9),
                rpad(@sprintf("%.3f", row.rmse_frozen_nt), 9),
                rpad(@sprintf("%.3f", row.rmse_served_nt), 9),
                rpad(@sprintf("%.3f", row.rmse_static_v22_nt), 11),
                rpad(@sprintf("%.3f", row.rmse_t1r_storm_nt), 11),
                rpad(@sprintf("%.3f", row.rmse_frozen_storm_nt), 14))
    end
    println("\n== selection statistic (mean pooled RMSE over steps 2,3,6) ==")
    best = nothing
    unguarded = nothing
    for result in run.results, safeguards in (true, false)
        statistic = v23_t1r_selection_statistic(run.report, result.config, safeguards)
        guards = v23_guard_flags([r for r in result.metrics if r.safeguards == safeguards])
        println(rpad(v23_t1r_id(result.config), 26), rpad(safeguards ? "S=on" : "S=off", 7),
                @sprintf("%.4f", statistic), "  guards=",
                guards.guards_ok ? "ok" : "FAIL($(guards.guard_failures))")
        candidate = (config=result.config, safeguards=safeguards, statistic=statistic)
        (unguarded === nothing || statistic < unguarded.statistic) && (unguarded = candidate)
        guards.guards_ok && (best === nothing || statistic < best.statistic) &&
            (best = candidate)
    end
    unguarded === nothing || println(@sprintf(
        "minimum over the T1r grid: %s S=%s at %.4f nT",
        v23_t1r_id(unguarded.config), unguarded.safeguards ? "on" : "off",
        unguarded.statistic))
    best === nothing ? println("no T1r configuration passes the section 6-A2 guards") :
        println(@sprintf("minimum subject to the section 6-A2 guards: %s S=%s at %.4f nT",
                         v23_t1r_id(best.config), best.safeguards ? "on" : "off",
                         best.statistic))
    return (best=best, unguarded=unguarded)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # `--smoke` cross-fits the layer over the development smoke tree, whose analog
    # artifacts cover two inner blocks. It exercises the same code path on a few
    # thousand anchors and never touches the development selection.
    if "--smoke" in ARGS
        v23_t1r_print_report(v23_t1r_run(; outdir=V23_DEV_SMOKE_DIR,
                                         blocks=v23_dev_plan(; smoke=true).blocks))
    else
        v23_t1r_print_report(v23_t1r_run())
    end
end
