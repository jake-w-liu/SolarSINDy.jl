#!/usr/bin/env julia

# superlearner_contract.jl — the file contract between the Operational super-learner rolling
# engine (`superlearner_rolling.jl`, Task A) and the super-learner/scoring stage
# (`superlearner_learn.jl`, Task B).
#
# Both stages must agree, byte for byte, on the column names of
# `oof_year_<Y>.csv`. A silent disagreement would not fail: the stack would fit
# on whichever columns it happened to find, so the study would report weights
# over an expert set nobody chose. This file is therefore the single definition
# of the schema, plus a validator that reads a persisted fold and says exactly
# which part of the contract it breaks.
#
# It is deliberately light: `using SolarSINDy` for the feature schema and the
# model steps, and nothing from the replay/kernel include chain. Either stage can
# `include` it on its own.
#
# Conventions the schema records (see `SUPERLEARNER_IMPLEMENTATION_SPEC.md`):
#   * one row per (issue time, model step) of the scored calendar year;
#   * `fallback = true` marks an anchor whose issue-time feature key is
#     incomplete; every learned column of that row carries the served inertial-SINDy value
#     and the feature columns are empty, never imputed;
#   * the `f_*` columns are the direct-GBM design vector in *raw physical units*
#     (see `SUPERLEARNER_FEATURE_SCALING`), not standardised, so a row's feature values do
#     not depend on which fold wrote it and years can be pooled;
#   * every column is a Dst value in nT unless its name says otherwise.

using CSV
using DataFrames
using Dates
using SolarSINDy

# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

"""
    SUPERLEARNER_OUTPUT_ROOT

Operational output root, resolved exactly as `validation/operational/paths.jl`
resolves it, so the contract can be included without the replay chain and still
points at the same tree. `superlearner_rolling.jl` asserts the two agree.
"""
const SUPERLEARNER_OUTPUT_ROOT = let override = strip(get(ENV, "SOLARSINDY_OPERATIONAL_OUTPUT_DIR", ""))
    abspath(isempty(override) ? joinpath(@__DIR__, "..", "output", "operational") : override)
end

"Run tree of the super-learner rolling-origin study."
const SUPERLEARNER_ROLLING_DIR = joinpath(SUPERLEARNER_OUTPUT_ROOT, "superlearner_rolling")

"Run tree of the reduced smoke path; never an input to the study."
const SUPERLEARNER_ROLLING_SMOKE_DIR = joinpath(SUPERLEARNER_OUTPUT_ROOT, "superlearner_rolling_smoke")

superlearner_oof_year_path(dir::AbstractString, year::Integer) =
    joinpath(dir, "oof_year_$(Int(year)).csv")
superlearner_manifest_year_path(dir::AbstractString, year::Integer) =
    joinpath(dir, "manifest_year_$(Int(year)).csv")
superlearner_signature_year_path(dir::AbstractString, year::Integer) =
    joinpath(dir, "sig_year_$(Int(year)).txt")

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

"Model steps carried by every fold file, in written order (plan section 3)."
const SUPERLEARNER_MODEL_STEPS = (1, 2, 3, 4, 6, 7)

"Test years of the rolling-origin protocol; 2013 seeds the out-of-fold history."
const SUPERLEARNER_FOLD_YEARS = 2013:2025

"Eras the study reports (plan section 3). 2013 belongs to neither."
const SUPERLEARNER_ERA_E1_YEARS = 2014:2019
const SUPERLEARNER_ERA_E2_YEARS = 2020:2025
const SUPERLEARNER_ERA_ALL_YEARS = 2014:2025

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

"Row key of a fold file."
const SUPERLEARNER_KEY_COLUMNS = ("issue_time_utc", "model_step_hours")

"""
    SUPERLEARNER_STATE_COLUMNS

Observation and causal issue-time state. `latest_dst_nt`, `dst_delta_1h_nt` and
`coupling_active_mvm` are the three quantities `static_stack_regime` needs, so
Task B can label regimes without touching the base table.
"""
const SUPERLEARNER_STATE_COLUMNS = ("observation_dst_nt", "latest_dst_nt", "dst_delta_1h_nt",
                           "coupling_active_mvm", "fallback")

"""
    SUPERLEARNER_EXPERT_COLUMNS

The super-learner experts, in the order the stack weights are written. Every one
is a causal Dst center at the row's model step.

The first nine are plan section 4's E1–E9. Amendment A3 adds the fixed
2010–2017 static stack as E10: the deep storm cells are data-poor in the
early folds, so a stack that cannot use the physics composition the static product
already encodes could not recover it there. The ten-expert set is used by the
`super-learner`/`super-learner-f` variants; the nine-expert set is kept for the other variants, so
both remain scored.
"""
const SUPERLEARNER_EXPERT_COLUMNS = ("served_inertial_sindy", "frozen_inertial_sindy", "t1r_analog", "persistence",
                            "burton", "burton_full", "obrien", "direct_gbm", "climatology",
                            "static_stack")

"""
    SUPERLEARNER_COMPARATOR_COLUMNS

Columns that are scored but are never stack inputs: the uncorrected analog raw
core, the analog ensemble lead-aware composition and its error-layer completion, and the
noncausal realized-driver ceiling. The static stack is both a comparator and
(since Amendment A3) an expert, so it is listed with the experts.

`analog_lat` is the composition before the error layers. It is persisted in
addition to `analog_shadow` so that a resumed run can rebuild the error-layer
fitting pool exactly, and so a verifier can separate the two stages; when the
error layers are not refitted the two columns are equal and the fold manifest
records that.
"""
const SUPERLEARNER_COMPARATOR_COLUMNS = ("t1_analog_raw", "analog_lat", "analog_shadow",
                                "oracle_realized")

"""
    SUPERLEARNER_MODEL_COLUMNS

Every forecast column of a fold file, in written order.
"""
const SUPERLEARNER_MODEL_COLUMNS = ("served_inertial_sindy", "frozen_inertial_sindy", "persistence", "burton",
                           "burton_full", "obrien", "static_stack", "climatology",
                           "t1_analog_raw", "t1r_analog", "direct_gbm", "analog_lat",
                           "analog_shadow", "oracle_realized")

"""
    SUPERLEARNER_FEATURE_NAMES

Feature schema of the persisted design vector: the 18 ADC issue-time features of
`SolarSINDy.ANALOG_FEATURE_NAMES` followed by the 11 Dst/VBs lag columns of
`SolarSINDy.DIRECT_GBM_EXTRA_FEATURE_NAMES`, which is exactly
`SolarSINDy.direct_gbm_feature_names()`. The literal list is checked against the
package at load time, so a package-side reorder fails here instead of silently
relabelling the persisted columns.
"""
const SUPERLEARNER_FEATURE_NAMES = (
    "bz0", "bz1", "bz2", "bz_mean6", "bz_sd6",
    "by0", "bperp0", "bperp_mean6",
    "v0", "dv6",
    "logn0", "logn_mean6", "pdyn0",
    "vbs0", "vbs_mean3", "south_run",
    "dst0", "ddst1",
    "dst_lag1", "dst_lag2", "dst_lag3", "dst_lag4", "dst_lag5", "dst_lag6",
    "dst_lag12", "dst_lag24", "vbs_lag1", "vbs_lag2", "vbs_lag3",
)

let observed = Tuple(String.(SolarSINDy.direct_gbm_feature_names()))
    observed == SUPERLEARNER_FEATURE_NAMES || error(
        "the super-learner persisted feature schema no longer matches " *
        "SolarSINDy.direct_gbm_feature_names(): package says $(observed)",
    )
end

"Column prefix of a persisted feature, so a feature can never collide with a model name."
const SUPERLEARNER_FEATURE_PREFIX = "f_"

"Persisted feature column names, in matrix order."
const SUPERLEARNER_FEATURE_COLUMNS = Tuple(SUPERLEARNER_FEATURE_PREFIX * name for name in SUPERLEARNER_FEATURE_NAMES)

"""
    SUPERLEARNER_FEATURE_SCALING

How the `f_*` columns are scaled. Raw physical units: a per-fold standardisation
would make one anchor's feature values depend on which fold wrote them, so rows
from different years could not be pooled into one residual model. Task B may
standardise on its own pool; the boosted residual of plan section 4 does not need
to.
"""
const SUPERLEARNER_FEATURE_SCALING = "raw_physical_units"

"Full column list of `oof_year_<Y>.csv`, in written order."
const SUPERLEARNER_OOF_COLUMNS = Tuple(vcat(
    collect(SUPERLEARNER_KEY_COLUMNS), collect(SUPERLEARNER_STATE_COLUMNS), collect(SUPERLEARNER_MODEL_COLUMNS),
    collect(SUPERLEARNER_FEATURE_COLUMNS),
))

"Columns whose value must be finite on every row."
const SUPERLEARNER_REQUIRED_FINITE_COLUMNS = Tuple(vcat(
    ["model_step_hours"],
    [c for c in SUPERLEARNER_STATE_COLUMNS if c != "fallback"],
    collect(SUPERLEARNER_MODEL_COLUMNS),
))

let names = collect(SUPERLEARNER_OOF_COLUMNS)
    allunique(names) || error("the super-learner fold schema repeats a column name")
    for expert in SUPERLEARNER_EXPERT_COLUMNS
        expert in names || error("expert column $expert is absent from the super-learner fold schema")
    end
    for comparator in SUPERLEARNER_COMPARATOR_COLUMNS
        comparator in names ||
            error("comparator column $comparator is absent from the super-learner fold schema")
    end
end

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_superlearner_finite(value) = !(value === missing) && isa(value, Real) && isfinite(Float64(value))

"""
    superlearner_validate_oof_year(path; expect_year = nothing, max_problems = 40) -> NamedTuple

Read a persisted fold file and check it against the contract. Returns

    (ok, path, year, n_rows, n_issues, n_fallback, steps, missing_columns,
     extra_columns, problems)

with `ok = isempty(problems)`. Nothing is repaired and nothing is imputed: the
function reports, and the caller decides whether to stop.

Checked: the required columns exist; the file is non-empty; every issue time
parses and falls in one calendar year (the fold's own year when `expect_year` is
given); every model step is one of `SUPERLEARNER_MODEL_STEPS`; no `(issue, step)` pair
repeats; `fallback` is boolean; `SUPERLEARNER_REQUIRED_FINITE_COLUMNS` are finite
everywhere; the `f_*` feature columns are finite on every non-fallback row.
Feature columns are allowed to be empty on a fallback row, because that is what
an incomplete feature key means — the contract forbids imputing them.
"""
function superlearner_validate_oof_year(path::AbstractString; expect_year=nothing,
                               max_problems::Integer=40)
    problems = String[]
    note!(message) = (length(problems) < max_problems && push!(problems, String(message)); nothing)
    empty_result(reason) = begin
        note!(reason)
        (ok=false, path=String(path), year=nothing, n_rows=0, n_issues=0, n_fallback=0,
         steps=Int[], missing_columns=String[], extra_columns=String[], problems=problems)
    end
    isfile(path) || return empty_result("file does not exist: $path")
    table = CSV.read(path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    header = String.(propertynames(table))
    missing_columns = [c for c in SUPERLEARNER_OOF_COLUMNS if !(c in header)]
    extra_columns = [c for c in header if !(c in SUPERLEARNER_OOF_COLUMNS)]
    isempty(missing_columns) ||
        note!("missing columns: $(join(missing_columns, ", "))")
    nrow(table) > 0 || note!("file holds no rows")
    if !isempty(missing_columns) || nrow(table) == 0
        return (ok=false, path=String(path), year=nothing, n_rows=nrow(table), n_issues=0,
                n_fallback=0, steps=Int[], missing_columns=missing_columns,
                extra_columns=extra_columns, problems=problems)
    end

    issues = table.issue_time_utc
    years = unique(year.(issues))
    length(years) == 1 || note!("file mixes calendar years: $(sort(years))")
    observed_year = length(years) == 1 ? first(years) : nothing
    expect_year === nothing || observed_year === nothing ||
        observed_year == Int(expect_year) ||
        note!("file holds year $(observed_year) but $(Int(expect_year)) was expected")

    steps = Int.(table.model_step_hours)
    unexpected = sort(unique([s for s in steps if !(s in SUPERLEARNER_MODEL_STEPS)]))
    isempty(unexpected) || note!("unexpected model steps: $(unexpected)")

    seen = Set{Tuple{DateTime,Int}}()
    duplicates = 0
    for r in 1:nrow(table)
        key = (issues[r], steps[r])
        key in seen ? (duplicates += 1) : push!(seen, key)
    end
    duplicates == 0 || note!("$(duplicates) repeated (issue, model step) rows")

    fallback = table.fallback
    n_fallback = 0
    for r in 1:nrow(table)
        value = fallback[r]
        if value === true || value === false
            n_fallback += value === true ? 1 : 0
        else
            note!("fallback is not boolean at row $r: $(repr(value))")
        end
    end

    for column in SUPERLEARNER_REQUIRED_FINITE_COLUMNS
        values = table[!, column]
        bad = findfirst(v -> !_superlearner_finite(v), values)
        bad === nothing || note!(
            "column $column is not finite at row $(bad) (value $(repr(values[bad])))",
        )
    end

    for column in SUPERLEARNER_FEATURE_COLUMNS
        values = table[!, column]
        bad = findfirst(r -> fallback[r] !== true && !_superlearner_finite(values[r]), 1:nrow(table))
        bad === nothing || note!(
            "feature column $column is not finite on non-fallback row $(bad) " *
            "(value $(repr(values[bad])))",
        )
    end

    return (ok=isempty(problems), path=String(path), year=observed_year, n_rows=nrow(table),
            n_issues=length(unique(issues)), n_fallback=n_fallback,
            steps=sort(unique(steps)), missing_columns=missing_columns,
            extra_columns=extra_columns, problems=problems)
end

"""
    superlearner_require_oof_year(path; expect_year = nothing) -> NamedTuple

[`superlearner_validate_oof_year`](@ref) that raises on the first contract breach. Use it
where a downstream stage would otherwise consume a fold it cannot trust.
"""
function superlearner_require_oof_year(path::AbstractString; expect_year=nothing)
    result = superlearner_validate_oof_year(path; expect_year=expect_year)
    result.ok || error(
        "the super-learner fold file $(path) breaks the contract:\n  " *
        join(result.problems, "\n  "),
    )
    return result
end
