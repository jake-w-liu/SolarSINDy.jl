#!/usr/bin/env julia

# v2_4_learn.jl — Operational V2.4 learning and scoring stage (Task B of
# `V2_4_IMPLEMENTATION_SPEC.md`, governed by `V2_4_RESEARCH_PLAN.md`).
#
# The rolling engine (Task A, `v2_4_rolling.jl`) writes one out-of-fold expert
# table per calendar year, `oof_year_<Y>.csv`. This file reads exactly those
# tables and nothing else, and turns them into the V2.4 candidate, its
# intervals, the preregistered scores and the preregistered gate decision:
#
#   L1  per step x regime non-negative super-learner weights over the nine
#       experts, summing to one, fitted only on years strictly before the
#       scored year (variant L1a adds the 0.60 SINDy-family floor);
#   L2  a boosted residual on `observation - L1`, capped at +/-(10 + 5h) nT,
#       with hyper-parameters, model form and per-step acceptance selected on an
#       inner validation window of the training pool that the inner training
#       block clears by the same 168 h target embargo;
#   L3  the depth-safe guard, which lets the residual deepen a forecast but
#       never lift a deepening one;
#   L4  split-conformal half-widths per step x activity at 0.90 coverage,
#       calibrated on the pool's out-of-fold residuals of the final center.
#
# Every fit for scored year Y sees only rows with issue year < Y. That is the
# one property the whole study rests on, so it is asserted here (pool year
# bounds, checked per fold) and tested by mutation in
# `test/test_v2_4_learn.jl`.
#
# Conventions inherited from the V2.3 study and reused verbatim so that the two
# studies stay comparable: model steps {1,2,3,4,6,7}; issue-time regimes from
# `SolarSINDy.operational_v22_regime`; storm cells from
# `SolarSINDy.v23_regime_cells`; the paired fixed non-overlapping 168 h
# calendar-block bootstrap and the Holm step-down of
# `SolarSINDy.v23_block_bootstrap` / `SolarSINDy.v23_holm`; the matured
# one-step-innovation lag convention of `v23_innovation_features` (a one-step
# forecast issued at t-j matures at t-j+1 and is therefore known at issue t for
# every j >= 1); and the pinned EvoTrees wrapper `SolarSINDy.v23_fit_gbm`.
#
#   julia --project=. validation/operational/v2_4_learn.jl --fixture=<dir>
#   JULIA_NUM_THREADS=8 julia --project=. validation/operational/v2_4_learn.jl
#
# This file defines only; running it as a program is what triggers the study.

using CSV
using DataFrames
using Dates
using JSON3
using LinearAlgebra
using Printf
using SHA
using SolarSINDy
using Statistics

include(joinpath(@__DIR__, "paths.jl"))

# ---------------------------------------------------------------------------
# Preregistered constants (research plan sections 3, 4 and 6)
# ---------------------------------------------------------------------------

"Directory holding Task A's `oof_year_<Y>.csv` tables and this stage's output."
const V24_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_4_rolling")

"Model steps carried by the study, in written order (plan section 1)."
const V24_STEPS = (1, 2, 3, 4, 6, 7)

"Scored folds of the learning stage; the preceding year exists only as the first pool."
const V24_FOLD_YEARS = 2014:2025

"Eras of plan section 3."
const V24_ERAS = (ALL=2014:2025, E1=2014:2019, E2=2020:2025)

"""
    V24_EXPERTS

The nine L1 experts in fixed column order (plan section 4). The first three are
the SINDy family that carries the L1a mass floor, which is why they lead.
"""
const V24_EXPERTS = (
    :served_v2_1, :frozen_v2_1, :t1r_analog,
    :persistence, :burton, :burton_full, :obrien, :direct_gbm, :climatology,
)
const V24_EXPERT_COUNT = length(V24_EXPERTS)

"""
    V24_EXPERTS_TEN

Amendment A3's expert set: the nine of plan section 4 plus E10, the fixed
2010–2017 static V2.2 stack. The deep storm cells are data-poor in the early
folds, so a stack whose inputs exclude the static product cannot recover the
physics composition that product already encodes where it is strongest; giving the
optimiser that column lets it use the composition in the tail and adapt elsewhere.

Both sets are fitted every fold: the nine-expert stack feeds `v2_4a`, `v2_4b`,
`v2_4c`, `v2_4a_floor` and `v2_4d`, and the ten-expert stack feeds `v2_4e` and
`v2_4f`, so the amendment is an addition to the scored family rather than a
replacement.
"""
const V24_EXPERTS_TEN = (V24_EXPERTS..., :static_v2_2)
const V24_EXPERT_TEN_COUNT = length(V24_EXPERTS_TEN)

"Expert indices carrying the floor mass in the nine-expert stack (served, frozen, T1r)."
const V24_SINDY_FAMILY = (1, 2, 3)

"""
    V24_SINDY_FAMILY_TEN

Floor group of the ten-expert stack. Amendment A3 counts the static V2.2 stack in
the floor because that product is itself a combination of the deployed SINDy
operators, so the floor keeps its meaning: at least 0.60 of the mass stays on the
SINDy family rather than on the physics and persistence baselines.
"""
const V24_SINDY_FAMILY_TEN = (1, 2, 3, V24_EXPERT_TEN_COUNT)

"L1a SINDy-family mass floor (plan section 4, ablation only)."
const V24_SINDY_FLOOR = 0.60

"Target embargo of every out-of-fold pool fit (hours; Amendment A3)."
const V24_EMBARGO_HOURS = 168

"""
    v24_pool_cutoff(fold_year) -> DateTime

Latest target an out-of-fold pool row may carry when fitting for `fold_year`:
`Y-01-01T00 - 168 h`, the same bound the rolling engine's training window obeys.
"""
v24_pool_cutoff(fold_year::Integer) =
    DateTime(Int(fold_year), 1, 1) - Hour(V24_EMBARGO_HOURS)

"""
    v24_in_pool(data, i, cutoff) -> Bool

Whether pool row `i` clears the fold's target embargo. `cutoff === nothing`
disables the rule, which is what the unit tests of the pure fitting arithmetic
use; every study fit passes the fold's own cutoff.
"""
v24_in_pool(data, i::Int, cutoff::Union{Nothing,DateTime}) =
    cutoff === nothing || data.issue[i] + Hour(data.step[i]) <= cutoff

"Minimum rows for a resolved L1 cell; thinner cells use the coarser fallback."
const V24_MIN_CELL_ROWS = 48

"Issue-time regimes of `SolarSINDy.operational_v22_regime`, in written order."
const V24_REGIMES = (:quiet, :active_deepening, :recovery)
const V24_POOLED_REGIME = :pooled

"""
    V24_DEPTH_BINS

Ring-current depth bins of Amendment A1, on the causal issue-time `latest_dst_nt`:
`:shallow` above -30 nT, `:moderate` in (-30, -70] nT, `:deep` at or below -70 nT.

The amendment introduces them for two reasons the first run made visible. Stack
weights fitted per regime alone are dominated by moderate rows, so the deep cells
inherited a combination tuned on shallow states and the static V2.2 stack stayed
ahead of the candidate there. Conformal half-widths fitted per "disturbed" row
mixed -30 nT and -400 nT states in one stratum, so storm coverage fell below the
G3 floor while pooled coverage held. Both layers therefore resolve depth
explicitly, with the same bin edges, so a cell and its interval describe the same
population.
"""
const V24_DEPTH_BINS = (:shallow, :moderate, :deep)
const V24_POOLED_DEPTH = :pooled

"Upper edge of the moderate depth bin (nT); above it a row is shallow."
const V24_DEPTH_MODERATE_NT = -30.0

"Upper edge of the deep depth bin (nT); at or below it a row is deep."
const V24_DEPTH_DEEP_NT = -70.0

"""
    v24_depth_bin(latest) -> Symbol

Depth bin of a causal issue-time ring-current value, per Amendment A1. The edges
are closed from below (`latest <= -30` is at least moderate) so a row sitting
exactly on an edge belongs to the deeper bin, which is the safe direction for a
guard and for an interval.
"""
function v24_depth_bin(latest::Real)
    value = Float64(latest)
    value <= V24_DEPTH_DEEP_NT && return :deep
    value <= V24_DEPTH_MODERATE_NT && return :moderate
    return :shallow
end

"L2 EvoTrees grid: `(max_depth, nrounds)` (plan section 4)."
const V24_L2_GRID = ((3, 200), (3, 400), (4, 200), (4, 400))
const V24_L2_ETA = 0.05
const V24_L2_MIN_WEIGHT = 128
const V24_L2_NBINS = 255

"Preregistered seed, shared with every V2.3 boosted fit."
const V24_SEED = 22_022_026

"Inner-validation window of the L2 pool (plan section 3)."
const V24_INNER_VALIDATION_MONTHS = 24

"""
    V24_INNER_MIN_ROWS

Row floor both halves of the inner split must clear before the plan's "last 24
months" rule is used. `min_weight = 128` is the smallest leaf a V2.4 residual
tree may carry, so a usable fitting block needs room for several splits above
it; twenty times the leaf floor is that room. When the pool is too short for the
24 month rule to leave such a block, the split degrades to the chronological
two-thirds/one-third fallback and the rule actually applied is persisted per
fold.
"""
const V24_INNER_MIN_ROWS = 20 * V24_L2_MIN_WEIGHT

"Minimum inner-training span before the 24 month rule is accepted."
const V24_INNER_MIN_TRAIN_MONTHS = 3

"""
    V24_L2_MIN_POOL_ROWS

Eligible residual rows a pool needs before the L2 layer is fitted at all. The
chronological two-thirds fallback split gives the validation half one third of
the pool, so three times [`V24_INNER_MIN_ROWS`](@ref) is the smallest pool for
which both halves are guaranteed to clear that floor. Below it the fold keeps its
L1 center and says so in the persisted selection table.
"""
const V24_L2_MIN_POOL_ROWS = 3 * V24_INNER_MIN_ROWS

"Depth-safe guard thresholds (plan section 4, L3)."
const V24_GUARD_RATE_NT_PER_H = -15.0
const V24_GUARD_DEPTH_NT = -50.0

"Conformal target coverage and activity split (plan section 4, L4)."
const V24_COVERAGE = 0.90
const V24_DISTURBED_NT = -30.0

"""
    V24_CONFORMAL_MIN_STRATUM_N

Rows a `(step, activity)` conformal stratum needs before its own quantile is
used. The deployed V2.1 sidecar uses the same floor for the same reason:
`ceil((20+1)*0.90) = 19 <= 20`, so twenty residuals are the fewest that can
reach the nominal level without falling back on the sample maximum. A thinner
stratum takes the widest populated stratum at its step, which can only widen an
interval.
"""
const V24_CONFORMAL_MIN_STRATUM_N = 20

"Bootstrap tail mass and gate significance level (plan section 6)."
const V24_ALPHA = 0.05

"G1 effect-size floor: `max(0.10 nT, 1 %)` against the strongest comparator."
const V24_G1_MIN_GAIN_NT = 0.10
const V24_G1_MIN_GAIN_FRACTION = 0.01

"""
    V24_G1_HEADROOM_NT

Realized-driver-oracle headroom below which Amendment A2 replaces the G1
superiority-with-margin requirement by non-inferiority. The headroom at a step is
`RMSE(best comparator) - RMSE(noncausal realized-driver oracle)`: the entire
distance a causal method could still travel. Where that distance is under
0.25 nT, requiring a `max(0.10 nT, 1 %)` margin would require most of the
information ceiling itself, which is a mis-specified test rather than a hard one.
"""
const V24_G1_HEADROOM_NT = 0.25

"Non-inferiority bound of the Amendment A2 headroom-limited G1 clause (nT)."
const V24_G1_NONINFERIORITY_NT = -0.05

"The V2.2-era 0.25 nT margin, reported alongside G1 but not gated (plan section 6)."
const V24_G1_REPORTED_MARGIN_NT = 0.25

"G2 storm-safety thresholds (plan section 6)."
const V24_G2_MIN_CELL_ROWS = 40
const V24_G2_MAX_LOSS_NT = 0.50
const V24_G2_INTENSE_BIAS_NT = 10.0
const V24_G2_INTENSE_STEP = 6
const V24_G2_CELLS = (:latest_le_m50, :latest_le_m100, :active_deepening, :recovery)
const V24_G2_INTENSE_CELL = :intense_deepening

"""
    V24_G2_SERVED_TOLERANCE_NT

Numerical slack on the "never loses to served V2.1" clause of G2 and on the
"no RMSE loss" clause of its intense-deepening branch. Both are exact
inequalities between two root mean square errors accumulated over the same rows;
a 1e-9 nT allowance keeps a floating-point tie from being reported as a breach
without admitting any physically meaningful loss.
"""
const V24_G2_SERVED_TOLERANCE_NT = 1e-9

"G3 interval thresholds (plan section 6)."
const V24_G3_COVERAGE_LO = 0.85
const V24_G3_COVERAGE_HI = 0.95
const V24_G3_PER_STEP_COVERAGE_MIN = 0.80
const V24_G3_MAX_WIDTH_RATIO = 1.10
const V24_G3_STORM_DST_NT = -50.0

"Steps whose pooled RMSE decides the served variant (plan section 4)."
const V24_SELECTION_STEPS = (2, 3, 6)

"""
    V24_VARIANTS

Variants scored by this stage, in written order (plan section 4 and Amendment A2).
`v2_4d` is the A2 industrial variant: the floor-constrained stack guarded against
the currently served static V2.2 stack, so in a deepening storm the candidate may
deepen the served forecast but never lift it.
"""
const V24_VARIANTS = (:v2_4a, :v2_4b, :v2_4c, :v2_4a_floor, :v2_4d, :v2_4e, :v2_4f)

"Comparator the Amendment A2 guard of `v2_4d` may never lift in a deepening cell."
const V24_D_GUARD_REFERENCE = :static_v2_2

"""
    V24_SELECTABLE_VARIANTS

Variants the selection rule may choose between. Amendment A1 opened the set to
every variant — the first rolling run showed the stack-only center already had the
lowest pooled RMSE at every step, and a rule that could not select it would serve
a strictly worse center by construction — and Amendment A2 added `v2_4d`.

Amendment A3 fixes the set to the four industrial candidates: the two
floor-constrained stacks and their static-guarded forms. The two residual variants
and the unconstrained stack stay scored and reported but are no longer eligible to
be served, because the residual layer never earned a step at the short leads and
the unconstrained stack has no tail floor.

The order is the tie-break order, guarded first, and among the guarded ones the
stack that already carries the static composition as an expert: `v2_4f`, `v2_4d`,
then the unguarded `v2_4e` and `v2_4a_floor`. Exact ties between different centers
do not occur in practice; the order exists so that the rule is total.
"""
const V24_SELECTABLE_VARIANTS = (:v2_4f, :v2_4d, :v2_4e, :v2_4a_floor)

"""
    V24_GATED_COMPARATORS

The comparator set of plan section 5 that G1 and G2 are evaluated against: the
two deployed SINDy products, the static V2.2 stack, both persisted V2.3
compositions, and each standalone expert E3-E9 including the tuned
increment-GBM.

Both `v2_3_lat` (the lead-aware composition) and `v2_3_shadow` (that composition
completed by the accepted error layers) are gated, because Task A persists both
and either could be the stronger V2.3 product on a given step. On a fold where
the error layers were not refitted the two columns are equal by construction;
they then enter the Holm family twice, which can only make the family-wise
adjustment stricter, never more permissive.
"""
const V24_GATED_COMPARATORS = (
    :served_v2_1, :frozen_v2_1, :static_v2_2, :v2_3_lat, :v2_3_shadow,
    :t1r_analog, :persistence, :burton, :burton_full, :obrien,
    :direct_gbm, :climatology,
)

"""
    V24_REPORTED_COMPARATORS

Columns scored and tabulated but excluded from the gates: the uncorrected analog
core, which `t1r_analog` supersedes, and the noncausal realized-driver oracle,
which plan section 5 admits as a ceiling only.
"""
const V24_REPORTED_COMPARATORS = (:t1_analog_raw, :oracle_realized)

"Every model column this stage reads from an `oof_year_<Y>.csv` table."
const V24_MODEL_COLUMNS = (
    :served_v2_1, :frozen_v2_1, :persistence, :burton, :burton_full, :obrien,
    :static_v2_2, :climatology, :t1_analog_raw, :t1r_analog, :direct_gbm,
)

"""
    V24_SHADOW_COLUMN_CANDIDATES

Task A persists `v2_3_shadow` when the V2.3 error layers were refit for the fold
and `v2_3_lat` when only the lead-aware tail composition was available. Either
satisfies the file contract; which one was present is persisted. When both are
present they are read as two comparators; when only one is, the two are equal, so
the single column supplies both.
"""
const V24_SHADOW_COLUMN_CANDIDATES = (:v2_3_shadow, :v2_3_lat)

"The lead-aware composition column, gated separately from its error-layer completion."
const V24_LAT_COLUMN = :v2_3_lat

const V24_ORACLE_COLUMN = :oracle_realized

"Innovation lags of the L2 error-state block (V2.3 convention)."
const V24_INNOVATION_LAGS = 6

"Storm cells reported per model, in the order `SolarSINDy.v23_regime_cells` emits."
const V24_CELL_NAMES = Symbol.(collect(SolarSINDy.V23_CELL_LABELS))

"Residual cap at model step `h` (plan section 4, L2)."
v24_residual_cap(h::Integer) = 10.0 + 5.0 * Int(h)

"""
    v24_feature_names() -> Vector{String}

The 29 issue-time columns Task A persists: the 18 ADC features in
`SolarSINDy.V23_FEATURE_NAMES` order followed by the Dst and VBs lag ladder in
`SolarSINDy.V23_DIRECT_EXTRA_FEATURE_NAMES` order.
"""
v24_feature_names() = String.(SolarSINDy.v23_direct_feature_names())

const V24_FEATURE_COUNT = SolarSINDy.V23_DIRECT_FEATURE_COUNT

"""
    v24_l2_feature_names() -> Vector{String}

Column names of the L2 residual design matrix, in matrix order: the 29
issue-time columns, the nine expert forecasts, the two expert-spread summaries,
the three regime indicators, the six matured one-step innovations, and the model
step.
"""
function v24_l2_feature_names()
    names = v24_feature_names()
    append!(names, ["expert_" * String(e) for e in V24_EXPERTS])
    append!(names, ["expert_spread_range_nt", "expert_spread_sd_nt"])
    append!(names, ["regime_" * String(r) for r in V24_REGIMES])
    append!(names, ["innovation_$(lag)h" for lag in 1:V24_INNOVATION_LAGS])
    push!(names, "model_step_hours")
    return names
end

const V24_L2_FEATURE_COUNT = V24_FEATURE_COUNT + V24_EXPERT_COUNT + 2 +
    length(V24_REGIMES) + V24_INNOVATION_LAGS + 1

"Slot of a model step in the fixed step order; zero when the step is not scored."
const V24_STEP_SLOT = let slots = zeros(Int, maximum(V24_STEPS))
    for (slot, step) in enumerate(V24_STEPS)
        slots[step] = slot
    end
    tuple(slots...)
end

_v24_log(msg) = (println(msg); flush(stdout))

"""
    _v24_sprintf(format, args...) -> String

Runtime `printf` formatting. `Printf.@sprintf` requires a single literal format
string, which would force the long requirement strings of the gate table onto
unreadable single lines; parsing the format at run time costs nothing on the
few thousand report rows this stage writes.
"""
_v24_sprintf(format::AbstractString, args...) =
    Printf.format(Printf.Format(format), args...)

_v24_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

_v24_fmt(x::Real; digits::Int=3) = isfinite(x) ? string(round(Float64(x); digits=digits)) : "NaN"
_v24_fmt(x) = string(x)

"""
    v24_markdown_table(io, header, rows)

One GitHub-flavoured markdown table. The generated report carries numbers only;
interpretation belongs in the planner's decision record.
"""
function v24_markdown_table(io::IO, header::AbstractVector{<:AbstractString}, rows)
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    println(io)
end

# ---------------------------------------------------------------------------
# Exact non-negative simplex least squares (L1 super-learner)
# ---------------------------------------------------------------------------

"""
    v24_project_simplex(values, mass) -> Vector{Float64}

Exact Euclidean projection of `values` onto `{x >= 0, sum(x) = mass}` by the
sorted-threshold rule. This restates the projection the V2.2 stack uses; the
test suite requires the two to agree on random inputs, which is what makes the
restatement safe rather than a second convention.
"""
function v24_project_simplex(values::AbstractVector{<:Real}, mass::Real)
    m = Float64(mass)
    isfinite(m) && m >= 0.0 || throw(ArgumentError("simplex mass must be finite and nonnegative"))
    v = Float64.(values)
    all(isfinite, v) || throw(ArgumentError("simplex values must be finite"))
    isempty(v) && throw(ArgumentError("cannot project an empty vector"))
    iszero(m) && return zeros(length(v))
    sorted = sort(v; rev=true)
    cumulative = 0.0
    rho = 0
    cumulative_at_rho = 0.0
    for j in eachindex(sorted)
        cumulative += sorted[j]
        if sorted[j] - (cumulative - m) / j > 0.0
            rho = j
            cumulative_at_rho = cumulative
        end
    end
    rho > 0 || error("simplex projection failed to identify an active set")
    theta = (cumulative_at_rho - m) / rho
    projected = max.(v .- theta, 0.0)
    idx = argmax(projected)
    projected[idx] += m - sum(projected)
    projected[idx] >= 0.0 || error("simplex projection produced a negative correction")
    return projected
end

"""
    v24_project_floor(values, floor_mass) -> Vector{Float64}

Exact Euclidean projection onto `{w >= 0, sum(w) = 1, sum(w[SINDy]) >= floor}`.
When the plain simplex projection already satisfies the floor it is the answer;
otherwise the floor is active, the two blocks decouple, and each is projected
onto its own simplex. This is the V2.2 construction with the SINDy family
widened from two experts to three.
"""
function v24_project_floor(values::AbstractVector{<:Real}, floor_mass::Real;
                           family=V24_SINDY_FAMILY)
    p = length(values)
    p >= 2 || throw(DimensionMismatch("V2.4 weight projection needs at least two values"))
    sindy = collect(Int, family)
    (allunique(sindy) && all(j -> 1 <= j <= p, sindy) && length(sindy) < p) ||
        throw(ArgumentError("the V2.4 floor group $(sindy) is not a proper subset of 1:$(p)"))
    floor64 = Float64(floor_mass)
    isfinite(floor64) && 0.0 <= floor64 <= 1.0 || throw(ArgumentError(
        "SINDy-family floor must lie in [0, 1]",
    ))
    ordinary = v24_project_simplex(values, 1.0)
    iszero(floor64) && return ordinary
    physical = setdiff(1:p, sindy)
    cap = 1.0 - floor64
    sum(ordinary[physical]) <= cap + 8eps(Float64) && return ordinary
    out = zeros(Float64, p)
    out[sindy] = v24_project_simplex(view(values, sindy), floor64)
    out[physical] = v24_project_simplex(view(values, physical), cap)
    return out
end

"Index of the constraint group containing coordinate `index`."
function _v24_group_of(index::Int, groups)
    for (g, group) in enumerate(groups)
        index in group && return g
    end
    error("coordinate $index belongs to no constraint group")
end

"""
    _v24_solve_support(Q, c, groups, masses, support) -> Union{Nothing,NamedTuple}

Solve `min 0.5 wᵀQw - cᵀw` restricted to `support`, subject to one sum
constraint per group and ignoring the sign constraints.

Returns `nothing` when the Karush-Kuhn-Tucker system is singular or its solution
does not reproduce the right-hand side. Skipping such a support is safe: a
singular system means the objective is flat along a feasible direction, so the
same objective value is attained on a strictly smaller support that the
enumeration also visits.
"""
function _v24_solve_support(Q::Matrix{Float64}, c::Vector{Float64}, groups, masses,
                            support::Vector{Int})
    k = length(support)
    m = length(groups)
    G = zeros(Float64, m, k)
    for (col, index) in enumerate(support)
        G[_v24_group_of(index, groups), col] = 1.0
    end
    for g in 1:m
        (masses[g] > 0.0 && iszero(sum(view(G, g, :)))) && return nothing
    end
    K = [Q[support, support] Matrix(-transpose(G)); G zeros(Float64, m, m)]
    rhs = vcat(c[support], Float64.(collect(masses)))
    solution = try
        K \ rhs
    catch
        return nothing
    end
    all(isfinite, solution) || return nothing
    norm(K * solution .- rhs) <= 1e-8 * max(1.0, norm(rhs)) || return nothing
    return (w=solution[1:k], mu=solution[(k + 1):end])
end

"""
    _v24_nonempty_subsets(group) -> Vector{Vector{Int}}

Every non-empty subset of `group`, ordered by size and then lexicographically,
so the enumeration below breaks objective ties toward the sparsest and then the
lowest-indexed support.
"""
function _v24_nonempty_subsets(group)
    items = collect(Int, group)
    subsets = Vector{Vector{Int}}()
    for mask in 1:(2^length(items) - 1)
        push!(subsets, [items[b] for b in eachindex(items) if (mask >> (b - 1)) & 1 == 1])
    end
    sort!(subsets; by=s -> (length(s), s))
    return subsets
end

"""
    _v24_grouped_qp(Q, c, groups, masses) -> NamedTuple

Global minimiser of `0.5 wᵀQw - cᵀw` over `{w >= 0}` intersected with one sum
constraint per group, by exhaustive enumeration of the active sets.

At the optimum the sign constraints are inactive on the support, so the optimal
weight vector solves the equality-constrained problem restricted to its own
support. Enumerating every support therefore contains the optimum, and convexity
makes the best feasible candidate global. With nine experts the enumeration is
at most 511 small linear solves, which removes the iterative-convergence failure
mode of a projected-gradient stack: there is no tolerance to trip and no
iteration budget to exhaust.

Returns the weights, the objective, the support, and the Karush-Kuhn-Tucker
certificate (`stationarity`, the largest scaled deviation of the reduced
gradient from its group multiplier on the support, and `dual_min`, the smallest
scaled reduced gradient off the support).
"""
function _v24_grouped_qp(Q::Matrix{Float64}, c::Vector{Float64}, groups, masses)
    p = length(c)
    size(Q) == (p, p) || throw(DimensionMismatch("Gram matrix and cross term disagree"))
    all(isfinite, Q) && all(isfinite, c) || throw(ArgumentError("QP inputs must be finite"))
    objective(w) = 0.5 * dot(w, Q * w) - dot(c, w)
    per_group = [_v24_nonempty_subsets(group) for group in groups]
    best_w = nothing
    best_objective = Inf
    best_support = Int[]
    for combination in Iterators.product(per_group...)
        support = sort!(vcat(collect(combination)...))
        solved = _v24_solve_support(Q, c, groups, masses, support)
        solved === nothing && continue
        # Only a rounding-level negative entry is tolerated; a genuinely negative
        # coordinate means this support is not the optimal one.
        minimum(solved.w) < -1e-11 && continue
        w = zeros(Float64, p)
        for (col, index) in enumerate(support)
            w[index] = max(solved.w[col], 0.0)
        end
        for (group, mass) in zip(groups, masses)
            indices = collect(group)
            deficit = mass - sum(w[indices])
            iszero(deficit) && continue
            w[indices[argmax(w[indices])]] += deficit
        end
        value = objective(w)
        isfinite(value) || continue
        # Ties keep the incumbent, and the enumeration visits supports in size
        # then lexicographic order, so a tie resolves to the sparsest support.
        if best_w === nothing ||
           value < best_objective - 1e-14 * max(1.0, abs(best_objective))
            best_objective = value
            best_w = w
            best_support = support
        end
    end
    best_w === nothing && error("V2.4 stack enumeration found no feasible active set")
    gradient = Q * best_w .- c
    scale = max(1.0, maximum(abs, c), maximum(abs, gradient))
    support_set = Set(best_support)
    mu = zeros(Float64, length(groups))
    for (g, group) in enumerate(groups)
        members = [j for j in group if j in support_set]
        mu[g] = isempty(members) ? 0.0 : mean(gradient[j] for j in members)
    end
    stationarity = 0.0
    for j in best_support
        stationarity = max(stationarity,
                           abs(gradient[j] - mu[_v24_group_of(j, groups)]) / scale)
    end
    dual_min = Inf
    for j in 1:p
        j in support_set && continue
        dual_min = min(dual_min, (gradient[j] - mu[_v24_group_of(j, groups)]) / scale)
    end
    isfinite(dual_min) || (dual_min = 0.0)
    return (weights=best_w, objective=best_objective, support=best_support,
            stationarity=stationarity, dual_min=dual_min)
end

"""
    v24_stack_system(A, y) -> (Q, c)

Gram system of the super-learner objective, centred by each row's mean expert
value. Because the weights sum to one this leaves the objective exactly
unchanged while removing the large common Dst offset, which would otherwise
dominate the Gram spectrum; the V2.2 stack uses the same reduction.
"""
function v24_stack_system(A::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    design = Matrix{Float64}(A)
    target = Vector{Float64}(y)
    n = size(design, 1)
    reference = vec(mean(design; dims=2))
    centered = design .- reference
    centered_target = target .- reference
    Q = Matrix(Symmetric((transpose(centered) * centered) ./ n))
    c = (transpose(centered) * centered_target) ./ n
    all(isfinite, Q) && all(isfinite, c) || error("V2.4 stack Gram system became non-finite")
    return (Q, c)
end

"""
    v24_fit_nnls(A, y; floor_mass=0.0) -> NamedTuple

Non-negative super-learner weights over the columns of `A`, summing to one and
minimising the mean squared error against `y`. With `floor_mass > 0` the columns in
`family` additionally carry at least that much mass, which is the L1a ablation of
plan section 4; `family` is `V24_SINDY_FAMILY` for the nine-expert stack and
`V24_SINDY_FAMILY_TEN` for the ten-expert stack of Amendment A3.

Returns the weights, the mean squared error of the fitted combination, the
support, the Karush-Kuhn-Tucker certificate, and whether the SINDy-family floor
turned out to be active.
"""
function v24_fit_nnls(A::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                      floor_mass::Real=0.0, family=V24_SINDY_FAMILY)
    n, p = size(A)
    p >= 2 || throw(DimensionMismatch("V2.4 stack needs at least two expert columns"))
    group = collect(Int, family)
    (allunique(group) && all(j -> 1 <= j <= p, group)) || throw(ArgumentError(
        "the V2.4 floor group $(group) is not a set of expert indices in 1:$(p)",
    ))
    length(group) < p || throw(ArgumentError(
        "the V2.4 floor group cannot contain every expert; there would be no free mass",
    ))
    length(y) == n || throw(DimensionMismatch("stack design and target disagree on rows"))
    n >= 1 || throw(ArgumentError("V2.4 stack fit needs at least one row"))
    design = Matrix{Float64}(A)
    target = Vector{Float64}(y)
    all(isfinite, design) && all(isfinite, target) || throw(ArgumentError(
        "V2.4 stack inputs must be finite",
    ))
    floor64 = Float64(floor_mass)
    isfinite(floor64) && 0.0 <= floor64 <= 1.0 || throw(ArgumentError(
        "SINDy-family floor must lie in [0, 1]",
    ))
    Q, c = v24_stack_system(design, target)

    result = _v24_grouped_qp(Q, c, (collect(1:p),), (1.0,))
    floor_active = false
    if floor64 > 0.0 && sum(result.weights[group]) < floor64 - 1e-12
        physical = setdiff(1:p, group)
        result = _v24_grouped_qp(Q, c, (group, physical), (floor64, 1.0 - floor64))
        floor_active = true
    end
    weights = result.weights
    residual = design * weights .- target
    mse = sum(abs2, residual) / n
    isfinite(mse) || error("V2.4 stack fit produced a non-finite MSE")
    all(>=(0.0), weights) || error("V2.4 stack produced a negative weight")
    abs(sum(weights) - 1.0) <= 1e-10 || error(
        "V2.4 stack weights sum to $(sum(weights)) instead of one",
    )
    result.stationarity <= 1e-7 || error(_v24_sprintf(
        "V2.4 stack optimality certificate failed: stationarity %.3e", result.stationarity,
    ))
    result.dual_min >= -1e-7 || error(_v24_sprintf(
        "V2.4 stack optimality certificate failed: dual minimum %.3e", result.dual_min,
    ))
    return (weights=weights, objective_mse=mse, support=result.support,
            stationarity=result.stationarity, dual_min=result.dual_min,
            floor_active=floor_active, n_rows=n)
end

"""
    v24_pgd_nnls(A, y; floor_mass=0.0, max_iterations=200_000, tolerance=1e-13)

Accelerated projected-gradient reference for [`v24_fit_nnls`](@ref), used as the
independent optimiser in the test suite. It solves the same problem by a
different route — Nesterov momentum with monotone restart on top of the exact
projection [`v24_project_floor`](@ref) — so agreement between the two is
evidence about the solution rather than about a shared implementation.
"""
function v24_pgd_nnls(A::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                      floor_mass::Real=0.0, family=V24_SINDY_FAMILY,
                      max_iterations::Integer=200_000, tolerance::Real=1e-13)
    design = Matrix{Float64}(A)
    target = Vector{Float64}(y)
    n, p = size(design)
    Q, c = v24_stack_system(design, target)
    project(v) = v24_project_floor(v, Float64(floor_mass); family=family)
    objective(w) = 0.5 * dot(w, Q * w) - dot(c, w)
    lipschitz = eigmax(Symmetric(Q))
    step = lipschitz <= eps(Float64) ? 1.0 : 1.0 / lipschitz
    w = project(fill(1.0 / p, p))
    z = copy(w)
    theta = 1.0
    value = objective(w)
    iterations = 0
    for iteration in 1:Int(max_iterations)
        iterations = iteration
        candidate = project(z .- step .* (Q * z .- c))
        candidate_value = objective(candidate)
        if candidate_value > value
            # Monotone restart: the momentum step overshot, so fall back to the
            # plain projected-gradient step from the last accepted point.
            candidate = project(w .- step .* (Q * w .- c))
            candidate_value = objective(candidate)
            theta = 1.0
        end
        theta_next = (1.0 + sqrt(1.0 + 4.0 * theta^2)) / 2.0
        z = candidate .+ ((theta - 1.0) / theta_next) .* (candidate .- w)
        moved = maximum(abs, candidate .- w)
        improvement = value - candidate_value
        w = candidate
        value = candidate_value
        theta = theta_next
        (moved <= tolerance && abs(improvement) <= tolerance) && break
    end
    residual = design * w .- target
    return (weights=w, objective_mse=sum(abs2, residual) / n, objective=value,
            iterations=iterations)
end

# ---------------------------------------------------------------------------
# Reading Task A's fold tables
# ---------------------------------------------------------------------------

"Path of Task A's fold table for calendar year `year`."
v24_oof_path(year::Integer; dir::AbstractString=V24_DIR) =
    joinpath(dir, "oof_year_$(year).csv")

"Path of Task A's fold manifest for calendar year `year`."
v24_manifest_path(year::Integer; dir::AbstractString=V24_DIR) =
    joinpath(dir, "manifest_year_$(year).csv")

"""
    v24_available_years(years; dir) -> Vector{Int}

The prefix of `years` whose fold tables exist, contiguously from the first. Task
A writes folds in ascending order, so a gap means the run is still in flight and
every later year is unusable rather than merely absent.
"""
function v24_available_years(years; dir::AbstractString=V24_DIR)
    available = Int[]
    for year in years
        isfile(v24_oof_path(year; dir=dir)) || break
        push!(available, Int(year))
    end
    return available
end

function _v24_bool(value, context::AbstractString)
    value === missing && error("$context holds missing where a flag is required")
    value isa Bool && return value
    value isa Integer && return value != 0
    value isa Real && return !iszero(value)
    if value isa AbstractString
        text = lowercase(strip(String(value)))
        text in ("true", "t", "1", "yes") && return true
        text in ("false", "f", "0", "no") && return false
    end
    error("$context holds $(repr(value)), which is not a flag")
end

function _v24_float(value, context::AbstractString; require_finite::Bool=true)
    if value === missing
        require_finite && error("$context holds missing where a finite number is required")
        return NaN
    end
    (value isa Real && !(value isa Bool)) ||
        error("$context holds $(repr(value)), which is not a real number")
    number = Float64(value)
    require_finite && !isfinite(number) && error("$context holds a non-finite value")
    return number
end

"""
    V24YearData

One calendar fold as this stage needs it: the issue-time state, the nine expert
forecasts, the 29 issue-time feature columns, the comparator columns, and the
per-row products this stage adds (innovations, L1 and L1a centers, the residual,
the four variant centers and their conformal half-widths).
"""
mutable struct V24YearData
    year::Int
    issue::Vector{DateTime}
    step::Vector{Int}
    obs::Vector{Float64}
    latest::Vector{Float64}
    rate::Vector{Float64}
    coupling::Vector{Float64}
    fallback::Vector{Bool}
    regime::Vector{Symbol}
    usable::BitVector
    experts::Matrix{Float64}
    features::Matrix{Float64}
    comparators::Dict{Symbol,Vector{Float64}}
    shadow_source::String
    innovations::Matrix{Float64}
    innovation_ok::BitVector
    l1::Vector{Float64}
    l1_floor::Vector{Float64}
    l1_ten::Vector{Float64}
    cell_regime::Vector{Symbol}
    cell_depth::Vector{Symbol}
    used_pooled::BitVector
    residual_raw::Vector{Float64}
    residual::Vector{Float64}
    l2_applied::BitVector
    centers::Dict{Symbol,Vector{Float64}}
    half_widths::Dict{Symbol,Vector{Float64}}
end

Base.length(data::V24YearData) = length(data.issue)

"""
    v24_read_year(year; dir) -> V24YearData

Read one `oof_year_<Y>.csv` under the Task A file contract and validate it.

Fail-closed rules, in the order they are checked: the shadow and feature columns
must be present in one of their contracted spellings; the issue year must be
`year`; `(issue, step)` must be unique; the model steps must be the study's; and
the issue-time state, the observation and every forecast center must be finite on
every row, fallback rows included — Task A writes the served V2.1 product into
every forecast column of a fallback row, so an empty center there is a contract
breach and not a declared gap. The feature block is the part a fallback row is
allowed to leave empty, and a row with an incomplete feature block is marked
unusable instead of being repaired. The realized-driver oracle is the one column
allowed to be absent on a row, because a target beyond the end of the driver
record has no realized continuation; its row count is reported wherever it is
scored.
"""
function v24_read_year(year::Integer; dir::AbstractString=V24_DIR)
    path = v24_oof_path(year; dir=dir)
    isfile(path) || error("V2.4 fold table is missing: $path")
    table = CSV.read(path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    nrow(table) > 0 || error("$path holds no rows")
    present = Set(names(table))

    shadow_column = nothing
    for candidate in V24_SHADOW_COLUMN_CANDIDATES
        String(candidate) in present || continue
        shadow_column = candidate
        break
    end
    shadow_column === nothing && error(
        "$path has neither $(join(String.(V24_SHADOW_COLUMN_CANDIDATES), " nor ")) " *
        "as required by the Task A file contract",
    )

    feature_names = v24_feature_names()
    prefix = if all(name -> ("f_" * name) in present, feature_names)
        "f_"
    elseif all(name -> name in present, feature_names)
        ""
    else
        absent_features = [name for name in feature_names
                           if !(("f_" * name) in present) && !(name in present)]
        error("$path is missing feature column(s): " * join(absent_features, ", "))
    end

    required = String[
        "issue_time_utc", "model_step_hours", "observation_dst_nt", "latest_dst_nt",
        "dst_delta_1h_nt", "coupling_active_mvm", "fallback",
    ]
    append!(required, String.(collect(V24_MODEL_COLUMNS)))
    push!(required, String(shadow_column))
    push!(required, String(V24_ORACLE_COLUMN))
    absent = [name for name in required if !(name in present)]
    isempty(absent) || error("$path is missing required column(s): " * join(absent, ", "))
    return _v24_year_from_table(table, Int(year), path, shadow_column, prefix)
end

function _v24_year_from_table(table::DataFrame, year::Int, path::AbstractString,
                              shadow_column::Symbol, prefix::AbstractString)
    n = nrow(table)
    issue = collect(DateTime, table.issue_time_utc)
    step = Vector{Int}(undef, n)
    for i in 1:n
        value = _v24_float(table.model_step_hours[i], "$path row $i model_step_hours")
        (isinteger(value) && value > 0) ||
            error("$path row $i has a non-integral model step $value")
        Int(value) in V24_STEPS ||
            error("$path row $i carries unscored model step $(Int(value))")
        step[i] = Int(value)
    end
    seen = Set{Tuple{DateTime,Int}}()
    for i in 1:n
        Dates.year(issue[i]) == year ||
            error("$path row $i has issue $(issue[i]) outside fold year $year")
        key = (issue[i], step[i])
        key in seen && error("$path repeats issue $(issue[i]) at step $(step[i])")
        push!(seen, key)
    end

    obs = Vector{Float64}(undef, n)
    latest = Vector{Float64}(undef, n)
    rate = Vector{Float64}(undef, n)
    coupling = Vector{Float64}(undef, n)
    fallback = Vector{Bool}(undef, n)
    for i in 1:n
        obs[i] = _v24_float(table.observation_dst_nt[i], "$path row $i observation_dst_nt")
        latest[i] = _v24_float(table.latest_dst_nt[i], "$path row $i latest_dst_nt")
        rate[i] = _v24_float(table.dst_delta_1h_nt[i], "$path row $i dst_delta_1h_nt")
        coupling[i] = _v24_float(table.coupling_active_mvm[i],
                                 "$path row $i coupling_active_mvm")
        coupling[i] >= 0.0 ||
            error("$path row $i has a negative rectified coupling $(coupling[i])")
        fallback[i] = _v24_bool(table.fallback[i], "$path row $i fallback")
    end

    comparators = Dict{Symbol,Vector{Float64}}()
    for column in V24_MODEL_COLUMNS
        values = Vector{Float64}(undef, n)
        for i in 1:n
            values[i] = _v24_float(table[i, column], "$path row $i $(column)")
        end
        comparators[column] = values
    end
    shadow = Vector{Float64}(undef, n)
    for i in 1:n
        shadow[i] = _v24_float(table[i, shadow_column], "$path row $i $(shadow_column)")
    end
    comparators[:v2_3_shadow] = shadow
    # The lead-aware composition is a comparator in its own right. Task A persists
    # it next to its error-layer completion; where only one composition column
    # exists the two are equal, so that column supplies both.
    if shadow_column !== V24_LAT_COLUMN && String(V24_LAT_COLUMN) in names(table)
        lat = Vector{Float64}(undef, n)
        for i in 1:n
            lat[i] = _v24_float(table[i, V24_LAT_COLUMN],
                                "$path row $i $(V24_LAT_COLUMN)")
        end
        comparators[V24_LAT_COLUMN] = lat
    else
        comparators[V24_LAT_COLUMN] = copy(shadow)
    end
    oracle = Vector{Float64}(undef, n)
    for i in 1:n
        oracle[i] = _v24_float(table[i, V24_ORACLE_COLUMN],
                               "$path row $i $(V24_ORACLE_COLUMN)"; require_finite=false)
    end
    comparators[V24_ORACLE_COLUMN] = oracle

    feature_names = v24_feature_names()
    features = Matrix{Float64}(undef, n, length(feature_names))
    for (j, name) in enumerate(feature_names)
        column = Symbol(prefix * name)
        for i in 1:n
            features[i, j] = _v24_float(table[i, column], "$path row $i $(column)";
                                        require_finite=false)
        end
    end

    experts = Matrix{Float64}(undef, n, V24_EXPERT_COUNT)
    for (j, expert) in enumerate(V24_EXPERTS)
        experts[:, j] = comparators[expert]
    end

    regime = Vector{Symbol}(undef, n)
    usable = falses(n)
    for i in 1:n
        regime[i] = SolarSINDy.operational_v22_regime(latest[i], rate[i], coupling[i])
        # Every forecast column was already required to be finite on every row, so
        # eligibility turns on the feature block alone: Task A leaves it empty
        # exactly where the issue-time key was incomplete, and such a row is
        # scored with the served product but never fitted on.
        usable[i] = !fallback[i] && all(isfinite, view(features, i, :))
    end

    return V24YearData(
        year, issue, step, obs, latest, rate, coupling, fallback, regime, usable,
        experts, features, comparators, String(shadow_column),
        fill(NaN, n, V24_INNOVATION_LAGS), falses(n),
        fill(NaN, n), fill(NaN, n), fill(NaN, n), fill(V24_POOLED_REGIME, n),
        fill(V24_POOLED_DEPTH, n), falses(n),
        zeros(n), zeros(n), falses(n),
        Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}(),
    )
end

# ---------------------------------------------------------------------------
# L1: super-learner stack
# ---------------------------------------------------------------------------

"""
    V24StackCell

One fitted L1 weight cell, keyed by model step, issue-time regime and depth bin.
`experts` records which expert set the weights are over, so a cell can never be
applied to the wrong design: the nine-expert and ten-expert stacks of Amendment A3
are fitted side by side and share this type.
"""
struct V24StackCell
    model_step_hours::Int
    regime::Symbol
    depth::Symbol
    n_rows::Int
    experts::Vector{Symbol}
    weights::Vector{Float64}
    objective_mse::Float64
    support::Vector{Int}
    stationarity::Float64
    dual_min::Float64
    floor_active::Bool
end

"Key type of the fitted L1 stack: `(model step, regime, depth bin)`."
const V24CellKey = Tuple{Int,Symbol,Symbol}

"""
    v24_cell_chain(step, regime, depth) -> Vector{V24CellKey}

Cell keys a row may use, from the most specific to the coarsest: its own
`(regime, depth)` cell, then the regime-pooled cell over every depth, then the
fully pooled cell. Amendment A1 fixes this order, and it is the only place the
fallback is written down.
"""
v24_cell_chain(step::Integer, regime::Symbol, depth::Symbol) = V24CellKey[
    (Int(step), regime, depth),
    (Int(step), regime, V24_POOLED_DEPTH),
    (Int(step), V24_POOLED_REGIME, V24_POOLED_DEPTH),
]

"""
    v24_cell_grid() -> Vector{V24CellKey}

Every cell key the stack may hold at one model step, in written order, so the
persisted weight table does not depend on dictionary order.
"""
function v24_cell_grid(step::Integer)
    keys = V24CellKey[(Int(step), V24_POOLED_REGIME, V24_POOLED_DEPTH)]
    for regime in V24_REGIMES
        push!(keys, (Int(step), regime, V24_POOLED_DEPTH))
        for depth in V24_DEPTH_BINS
            push!(keys, (Int(step), regime, depth))
        end
    end
    return keys
end

"""
    v24_fit_l1(pool; floor_mass=0.0, minimum_cell_rows=48) -> Dict

Fit the L1 super-learner on the fold's training pool. Amendment A1 resolves the
cells by issue-time regime *and* depth bin: for every model step the fully pooled
cell, a regime-pooled cell per regime, and a `(regime, depth)` cell for every
combination that reaches `minimum_cell_rows`. `pool` is a vector of
[`V24YearData`](@ref); only rows flagged `usable` (complete features, complete
experts, not a Task A fallback) enter a fit. Returns a dictionary keyed by
`(step, regime, depth)`.

`minimum_cell_rows` is the threshold a *resolved* cell must reach to be fitted at
all. The fully pooled cell is the fallback every row is entitled to, so it is not
optional and its own floor stays at `V24_MIN_CELL_ROWS`; raising
`minimum_cell_rows` therefore coarsens the stack along the chain of
[`v24_cell_chain`](@ref) instead of leaving a row with no cell to use.
"""
function v24_fit_l1(pool::AbstractVector{V24YearData}; floor_mass::Real=0.0,
                    minimum_cell_rows::Integer=V24_MIN_CELL_ROWS,
                    experts=V24_EXPERTS, family=V24_SINDY_FAMILY,
                    cutoff::Union{Nothing,DateTime}=nothing)
    isempty(pool) && error("V2.4 L1 fit needs at least one pool year")
    pooled_minimum = min(Int(minimum_cell_rows), V24_MIN_CELL_ROWS)
    fully_pooled = (V24_POOLED_REGIME, V24_POOLED_DEPTH)
    names = collect(Symbol, experts)
    cells = Dict{V24CellKey,V24StackCell}()
    for step in V24_STEPS
        selectors = Dict{Tuple{Symbol,Symbol},Vector{Tuple{Int,Int}}}()
        for key in ((V24_POOLED_REGIME, V24_POOLED_DEPTH),
                    ((regime, V24_POOLED_DEPTH) for regime in V24_REGIMES)...,
                    ((regime, depth) for regime in V24_REGIMES,
                     depth in V24_DEPTH_BINS)...)
            selectors[key] = Tuple{Int,Int}[]
        end
        for (y, data) in enumerate(pool)
            for i in 1:length(data)
                (data.step[i] == step && data.usable[i]) || continue
                v24_in_pool(data, i, cutoff) || continue
                regime = data.regime[i]
                depth = v24_depth_bin(data.latest[i])
                push!(selectors[fully_pooled], (y, i))
                push!(selectors[(regime, V24_POOLED_DEPTH)], (y, i))
                push!(selectors[(regime, depth)], (y, i))
            end
        end
        length(selectors[fully_pooled]) >= pooled_minimum || error(
            "V2.4 L1 pooled cell at step $step has " *
            "$(length(selectors[fully_pooled])) rows, fewer than the " *
            "$(pooled_minimum) required",
        )
        for (_, regime, depth) in v24_cell_grid(step)
            rows = selectors[(regime, depth)]
            threshold = (regime, depth) == fully_pooled ? pooled_minimum :
                        Int(minimum_cell_rows)
            length(rows) >= threshold || continue
            A = Matrix{Float64}(undef, length(rows), length(names))
            y = Vector{Float64}(undef, length(rows))
            for (r, (yi, i)) in enumerate(rows)
                for (j, name) in enumerate(names)
                    A[r, j] = pool[yi].comparators[name][i]
                end
                y[r] = pool[yi].obs[i]
            end
            fit = v24_fit_nnls(A, y; floor_mass=floor_mass, family=family)
            cells[(step, regime, depth)] = V24StackCell(
                step, regime, depth, length(rows), names, fit.weights, fit.objective_mse,
                fit.support, fit.stationarity, fit.dual_min, fit.floor_active,
            )
        end
    end
    return cells
end

"""
    v24_l1_centers(data, cells) -> NamedTuple

Apply fitted L1 cells to a fold. A row walks the chain of
[`v24_cell_chain`](@ref) and uses the first cell that was fitted; a row that is
not `usable` keeps the served V2.1 value, which is the plan section 3 rule for
fallback rows and which the returned `cell_regime` marks as `:served`.
"""
function v24_l1_centers(data::V24YearData, cells::Dict{V24CellKey,V24StackCell})
    n = length(data)
    centers = Vector{Float64}(undef, n)
    cell_regime = Vector{Symbol}(undef, n)
    cell_depth = Vector{Symbol}(undef, n)
    used_pooled = falses(n)
    served = data.comparators[:served_v2_1]
    for i in 1:n
        if !data.usable[i]
            centers[i] = served[i]
            cell_regime[i] = :served
            cell_depth[i] = :served
            continue
        end
        step = data.step[i]
        chain = v24_cell_chain(step, data.regime[i], v24_depth_bin(data.latest[i]))
        cell = nothing
        for (position, key) in enumerate(chain)
            candidate = get(cells, key, nothing)
            candidate === nothing && continue
            cell = candidate
            used_pooled[i] = position > 1
            break
        end
        cell === nothing && error(
            "V2.4 L1 has no cell for step $step on the chain $(chain)",
        )
        total = 0.0
        for (j, name) in enumerate(cell.experts)
            total += cell.weights[j] * data.comparators[name][i]
        end
        isfinite(total) || error("V2.4 L1 center became non-finite at $(data.issue[i])")
        centers[i] = total
        cell_regime[i] = cell.regime
        cell_depth[i] = cell.depth
    end
    return (centers=centers, cell_regime=cell_regime, cell_depth=cell_depth,
            used_pooled=used_pooled)
end

"""
    v24_apply_l1!(data, cells; floor=false)

Attach one fitted stack's centers to a fold. `target` selects the slot: `:l1` for
the plain nine-expert stack, `:l1_floor` for its floor-constrained form, `:l1_ten`
for the ten-expert floor stack of Amendment A3. The cell bookkeeping is recorded
from the plain fit only, because every fit shares the same cell geometry and only
the plain one feeds the residual layer.
"""
function v24_apply_l1!(data::V24YearData, cells::Dict{V24CellKey,V24StackCell};
                       target::Symbol=:l1)
    applied = v24_l1_centers(data, cells)
    if target === :l1
        copyto!(data.l1, applied.centers)
        copyto!(data.cell_regime, applied.cell_regime)
        copyto!(data.cell_depth, applied.cell_depth)
        data.used_pooled .= applied.used_pooled
    elseif target === :l1_floor
        copyto!(data.l1_floor, applied.centers)
    elseif target === :l1_ten
        copyto!(data.l1_ten, applied.centers)
    else
        error("V2.4 L1 has no center slot named $target")
    end
    return data
end

"""
    v24_l1_weight_rows(cells, year, label) -> Vector{NamedTuple}

Persistable form of a fitted L1 stack: one row per cell, keyed by step, regime
and depth bin, with every weight, the in-pool objective, and the optimality
certificate.
"""
function v24_l1_weight_rows(cells::Dict{V24CellKey,V24StackCell}, year::Integer,
                            label::AbstractString)
    rows = NamedTuple[]
    for step in V24_STEPS, key in v24_cell_grid(step)
        cell = get(cells, key, nothing)
        cell === nothing && continue
        # One numeric column per expert of the widest set, so the nine-expert and
        # ten-expert stacks share a schema; an expert that was not an input of this
        # cell carries a hard zero and `expert_set` says which set was fitted.
        index = Dict(name => j for (j, name) in enumerate(cell.experts))
        weights = NamedTuple{Tuple(Symbol("w_", e) for e in V24_EXPERTS_TEN)}(
            Tuple(haskey(index, e) ? cell.weights[index[e]] : 0.0
                  for e in V24_EXPERTS_TEN),
        )
        family = cell.experts == collect(V24_EXPERTS_TEN) ? V24_SINDY_FAMILY_TEN :
                 V24_SINDY_FAMILY
        push!(rows, merge((
            fold_year=Int(year), variant=String(label), model_step_hours=step,
            regime=String(cell.regime), depth_bin=String(cell.depth),
            n_rows=cell.n_rows, n_experts=length(cell.experts),
            expert_set=join(String.(cell.experts), "|"),
        ), weights, (
            weight_sum=sum(cell.weights),
            sindy_family_mass=sum(cell.weights[j] for j in family),
            objective_mse=cell.objective_mse,
            support=join([String(cell.experts[j]) for j in cell.support], "|"),
            kkt_stationarity=cell.stationarity, kkt_dual_min=cell.dual_min,
            floor_active=cell.floor_active,
        )))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Innovations and the L2 residual design
# ---------------------------------------------------------------------------

"""
    v24_fill_innovations!(data, innovation_store)

Attach the six matured one-step innovations of the L1 center to each row of a
fold. `innovation_store` maps an issue time to the innovation of the one-step L1
forecast issued then, `observation(t+1) - L1_step1(t)`, which matures at `t + 1`
and is therefore known at every issue `t + j`, `j >= 1` — the causal convention
of `v23_innovation_features`. A row whose six predecessors are not all present
keeps `innovation_ok = false` and later keeps its uncorrected L1 center rather
than an imputed correction.
"""
function v24_fill_innovations!(data::V24YearData,
                               innovation_store::Dict{DateTime,Float64})
    for i in 1:length(data)
        complete = true
        for lag in 1:V24_INNOVATION_LAGS
            value = get(innovation_store, data.issue[i] - Hour(lag), NaN)
            if !isfinite(value)
                complete = false
                break
            end
            data.innovations[i, lag] = value
        end
        data.innovation_ok[i] = complete
        complete || (data.innovations[i, :] .= NaN)
    end
    return data
end

"""
    v24_record_innovations!(store, data)

Record this fold's one-step L1 innovations into the shared store, so the next
fold's first hours can read across the calendar boundary.
"""
function v24_record_innovations!(store::Dict{DateTime,Float64}, data::V24YearData)
    for i in 1:length(data)
        data.step[i] == 1 || continue
        (isfinite(data.obs[i]) && isfinite(data.l1[i])) || continue
        store[data.issue[i]] = data.obs[i] - data.l1[i]
    end
    return store
end

"""
    v24_l2_eligible(data, i) -> Bool

Whether row `i` can carry a boosted residual: complete features and experts, not
a Task A fallback, a finite L1 center, a finite observation, and the six matured
innovations.
"""
v24_l2_eligible(data::V24YearData, i::Int) =
    data.usable[i] && data.innovation_ok[i] && isfinite(data.l1[i]) && isfinite(data.obs[i])

"""
    v24_l2_design(rows) -> Matrix{Float64}

Assemble the L2 residual design matrix for `rows`, a vector of
`(V24YearData, row index)` pairs, in [`v24_l2_feature_names`](@ref) order.
"""
function v24_l2_design(rows::AbstractVector{Tuple{V24YearData,Int}})
    X = Matrix{Float64}(undef, length(rows), V24_L2_FEATURE_COUNT)
    for (r, (data, i)) in enumerate(rows)
        column = 0
        for j in 1:V24_FEATURE_COUNT
            column += 1
            X[r, column] = data.features[i, j]
        end
        minimum_expert = Inf
        maximum_expert = -Inf
        total = 0.0
        for j in 1:V24_EXPERT_COUNT
            column += 1
            value = data.experts[i, j]
            X[r, column] = value
            minimum_expert = min(minimum_expert, value)
            maximum_expert = max(maximum_expert, value)
            total += value
        end
        mean_expert = total / V24_EXPERT_COUNT
        variance = 0.0
        for j in 1:V24_EXPERT_COUNT
            variance += abs2(data.experts[i, j] - mean_expert)
        end
        column += 1
        X[r, column] = maximum_expert - minimum_expert
        column += 1
        # Corrected (sample) standard deviation of the nine expert forecasts.
        X[r, column] = sqrt(variance / (V24_EXPERT_COUNT - 1))
        for regime in V24_REGIMES
            column += 1
            X[r, column] = data.regime[i] === regime ? 1.0 : 0.0
        end
        for lag in 1:V24_INNOVATION_LAGS
            column += 1
            X[r, column] = data.innovations[i, lag]
        end
        column += 1
        X[r, column] = Float64(data.step[i])
        column == V24_L2_FEATURE_COUNT || error(
            "V2.4 L2 design wrote $column of $(V24_L2_FEATURE_COUNT) columns",
        )
    end
    all(isfinite, X) || error("V2.4 L2 design matrix became non-finite")
    return X
end

"Residual target `observation - L1` for `rows`."
v24_l2_target(rows::AbstractVector{Tuple{V24YearData,Int}}) =
    Float64[data.obs[i] - data.l1[i] for (data, i) in rows]

"""
    v24_cap_residual(value, step) -> Float64

Clamp a boosted residual to `+/-(10 + 5h)` nT at model step `h`.
"""
function v24_cap_residual(value::Real, step::Integer)
    cap = v24_residual_cap(step)
    number = Float64(value)
    isfinite(number) || return 0.0
    return clamp(number, -cap, cap)
end

"""
    v24_inner_split(rows) -> NamedTuple

Chronological inner split of an L2 pool. The plan's rule is the last
[`V24_INNER_VALIDATION_MONTHS`](@ref) months of the pool as validation; it is
used whenever it leaves a training block of at least
[`V24_INNER_MIN_TRAIN_MONTHS`](@ref) months and [`V24_INNER_MIN_ROWS`](@ref)
rows on each side. Otherwise the split degrades to the last third of the pool by
issue time, and the rule that was applied is returned so it can be persisted.

The two halves are separated by the same [`V24_EMBARGO_HOURS`](@ref) target
embargo that Amendment A3 puts between a fold's pool and the year it scores. A
contiguous split would let the inner training block carry targets that mature
after the inner validation window opens, so the hyper-parameter, joint-versus-
per-step and per-step acceptance decisions would be taken with a sliver of the
validation period's own outcome inside the fitting rows. The training half is
therefore `v24_in_pool` against the inner cutoff — evaluated per row with that
row's own model step, because the pool carries all six steps at once and a 7 h
row matures six hours later than a 1 h row issued beside it. Rows falling in the
gap belong to neither half and are counted in `n_embargoed`.
"""
function v24_inner_split(rows::AbstractVector{Tuple{V24YearData,Int}})
    isempty(rows) && error("V2.4 inner split needs rows")
    issues = [data.issue[i] for (data, i) in rows]
    first_issue = minimum(issues)
    last_issue = maximum(issues)
    # A candidate split is described entirely by where its validation window
    # opens: validation is every row issued at or after the boundary, training is
    # every row whose target clears the boundary by the embargo.
    function halves(boundary::DateTime)
        cutoff = boundary - Hour(V24_EMBARGO_HOURS)
        train = Int[k for k in eachindex(rows)
                    if v24_in_pool(rows[k][1], rows[k][2], cutoff)]
        return (train=train, validate=findall(>=(boundary), issues), cutoff=cutoff)
    end

    boundary = last_issue - Month(V24_INNER_VALIDATION_MONTHS)
    rule = "last_$(V24_INNER_VALIDATION_MONTHS)_months"
    half = halves(boundary)
    span_ok = !isempty(half.train) &&
        maximum(issues[half.train]) >= first_issue + Month(V24_INNER_MIN_TRAIN_MONTHS)
    if !(span_ok && length(half.train) >= V24_INNER_MIN_ROWS &&
         length(half.validate) >= V24_INNER_MIN_ROWS)
        span = Dates.value(last_issue - first_issue)
        boundary = first_issue + Millisecond(div(2 * span, 3))
        rule = "chronological_two_thirds"
        half = halves(boundary)
    end
    (isempty(half.train) || isempty(half.validate)) && error(
        "V2.4 inner split produced an empty half over $(length(rows)) rows",
    )
    return (train=half.train, validate=half.validate, boundary=boundary,
            cutoff=half.cutoff, rule=rule,
            n_embargoed=length(rows) - length(half.train) - length(half.validate),
            first_issue=first_issue, last_issue=last_issue)
end

"""
    v24_fit_l2(pool_rows; grid, seed) -> NamedTuple

Fit the boosted residual layer of plan section 4.

Hyper-parameters and the joint-versus-per-step question are both decided on the
inner validation window of the pool: for each `(max_depth, nrounds)` in `grid`,
both a joint model carrying the model step as a feature and one model per step
are fitted on the inner-training block and scored by the pooled root mean square
error of `observation - (L1 + capped residual)` on the inner-validation block.
The joint form is kept when it is not worse, which is the tie-break the
specification names. The winning configuration is then refitted on the whole
pool.

The inner-training block is embargoed against the inner-validation window by
[`V24_EMBARGO_HOURS`](@ref) (see [`v24_inner_split`](@ref)), so no decision taken
here — grid point, model form, or per-step acceptance — is informed by a target
that matures inside the window it is scored on. The refit on the whole pool is
unaffected: the pool is already embargoed against the fold year it will score.
"""
function v24_fit_l2(pool_rows::AbstractVector{Tuple{V24YearData,Int}};
                    grid=V24_L2_GRID, seed::Integer=V24_SEED)
    isempty(grid) && error("V2.4 L2 grid must not be empty")
    split = v24_inner_split(pool_rows)
    names = v24_l2_feature_names()
    inner_train = pool_rows[split.train]
    inner_validate = pool_rows[split.validate]
    X_train = v24_l2_design(inner_train)
    y_train = v24_l2_target(inner_train)
    X_validate = v24_l2_design(inner_validate)
    y_validate = v24_l2_target(inner_validate)
    steps_train = Int[data.step[i] for (data, i) in inner_train]
    steps_validate = Int[data.step[i] for (data, i) in inner_validate]

    function score(prediction)
        total = 0.0
        for r in eachindex(prediction)
            total += abs2(y_validate[r] - v24_cap_residual(prediction[r], steps_validate[r]))
        end
        return sqrt(total / length(prediction))
    end

    trace = NamedTuple[]
    best = nothing
    best_prediction = Float64[]
    for (depth, rounds) in grid
        joint = SolarSINDy.v23_fit_gbm(
            X_train, y_train; max_depth=depth, nrounds=rounds, eta=V24_L2_ETA,
            min_weight=V24_L2_MIN_WEIGHT, seed=seed, nbins=V24_L2_NBINS,
            feature_names=names,
        )
        joint_prediction = SolarSINDy.v23_predict(joint, X_validate)
        joint_rmse = score(joint_prediction)
        per_step_prediction = Vector{Float64}(undef, length(y_validate))
        per_step_ok = true
        for step in V24_STEPS
            train_rows = findall(==(step), steps_train)
            validate_rows = findall(==(step), steps_validate)
            if length(train_rows) < 2 * V24_L2_MIN_WEIGHT || isempty(validate_rows)
                per_step_ok = false
                break
            end
            model = SolarSINDy.v23_fit_gbm(
                X_train[train_rows, :], y_train[train_rows]; max_depth=depth,
                nrounds=rounds, eta=V24_L2_ETA, min_weight=V24_L2_MIN_WEIGHT,
                seed=seed, nbins=V24_L2_NBINS, feature_names=names,
            )
            per_step_prediction[validate_rows] =
                SolarSINDy.v23_predict(model, X_validate[validate_rows, :])
        end
        per_step_rmse = per_step_ok ? score(per_step_prediction) : NaN
        joint_wins = !isfinite(per_step_rmse) || joint_rmse <= per_step_rmse
        chosen_rmse = joint_wins ? joint_rmse : per_step_rmse
        push!(trace, (
            max_depth=depth, nrounds=rounds, joint_inner_rmse_nt=joint_rmse,
            per_step_inner_rmse_nt=per_step_rmse, form=joint_wins ? "joint" : "per_step",
            inner_rmse_nt=chosen_rmse,
        ))
        if best === nothing || chosen_rmse < best.inner_rmse_nt
            best = (max_depth=depth, nrounds=rounds,
                    form=joint_wins ? "joint" : "per_step", inner_rmse_nt=chosen_rmse)
            best_prediction = joint_wins ? joint_prediction : per_step_prediction
        end
    end
    best === nothing && error("V2.4 L2 grid produced no configuration")

    # Amendment A1: the residual is accepted at a step only where it improved the
    # inner validation of that step. The comparison is between the capped
    # correction and the identity (no correction) on the same inner-validation
    # rows, so a step where the boosted layer only adds variance keeps its L1
    # center instead of being corrected because other steps liked the layer.
    accepted = Dict{Int,Bool}()
    acceptance = NamedTuple[]
    for step in V24_STEPS
        rows = findall(==(step), steps_validate)
        if isempty(rows)
            accepted[step] = false
            push!(acceptance, (
                model_step_hours=step, n_inner_validate=0, rmse_identity_nt=NaN,
                rmse_residual_nt=NaN, gain_nt=NaN, accepted=false,
                reason="no_inner_validation_row",
            ))
            continue
        end
        identity_rmse = sqrt(sum(abs2, view(y_validate, rows)) / length(rows))
        corrected = 0.0
        for r in rows
            corrected += abs2(y_validate[r] - v24_cap_residual(best_prediction[r], step))
        end
        residual_rmse = sqrt(corrected / length(rows))
        gain = identity_rmse - residual_rmse
        accepted[step] = isfinite(gain) && gain > 0.0
        push!(acceptance, (
            model_step_hours=step, n_inner_validate=length(rows),
            rmse_identity_nt=identity_rmse, rmse_residual_nt=residual_rmse,
            gain_nt=gain, accepted=accepted[step],
            reason=accepted[step] ? "inner_gain_positive" : "inner_gain_not_positive",
        ))
    end

    models = Dict{Int,Any}()
    if any(values(accepted))
        X_pool = v24_l2_design(pool_rows)
        y_pool = v24_l2_target(pool_rows)
        steps_pool = Int[data.step[i] for (data, i) in pool_rows]
        if best.form == "joint"
            models[0] = SolarSINDy.v23_fit_gbm(
                X_pool, y_pool; max_depth=best.max_depth, nrounds=best.nrounds,
                eta=V24_L2_ETA, min_weight=V24_L2_MIN_WEIGHT, seed=seed,
                nbins=V24_L2_NBINS, feature_names=names,
            )
        else
            for step in V24_STEPS
                accepted[step] || continue
                rows = findall(==(step), steps_pool)
                length(rows) >= 2 * V24_L2_MIN_WEIGHT || error(
                    "V2.4 L2 per-step refit at step $step has only $(length(rows)) " *
                    "pool rows",
                )
                models[step] = SolarSINDy.v23_fit_gbm(
                    X_pool[rows, :], y_pool[rows]; max_depth=best.max_depth,
                    nrounds=best.nrounds, eta=V24_L2_ETA, min_weight=V24_L2_MIN_WEIGHT,
                    seed=seed, nbins=V24_L2_NBINS, feature_names=names,
                )
            end
        end
    end
    return (models=models, form=best.form, max_depth=best.max_depth,
            nrounds=best.nrounds, inner_rmse_nt=best.inner_rmse_nt, trace=trace,
            split=split, n_pool_rows=length(pool_rows),
            n_inner_train=length(split.train), n_inner_validate=length(split.validate),
            n_inner_embargoed=split.n_embargoed,
            accepted=accepted, acceptance=acceptance,
            accepted_steps=[step for step in V24_STEPS if accepted[step]])
end

"""
    v24_apply_l2!(data, layer)

Score a fold with a fitted residual layer. Only eligible rows at a step the
layer's inner validation accepted receive a correction; every other row keeps its
L1 center and is recorded with `l2_applied = false`.
"""
function v24_apply_l2!(data::V24YearData, layer)
    rows = Tuple{V24YearData,Int}[(data, i) for i in 1:length(data)
                                  if v24_l2_eligible(data, i) &&
                                     get(layer.accepted, data.step[i], false)]
    fill!(data.residual_raw, 0.0)
    fill!(data.residual, 0.0)
    data.l2_applied .= false
    isempty(rows) && return data
    X = v24_l2_design(rows)
    prediction = if layer.form == "joint"
        SolarSINDy.v23_predict(layer.models[0], X)
    else
        out = Vector{Float64}(undef, length(rows))
        steps = Int[data.step[i] for (_, i) in rows]
        for step in V24_STEPS
            selected = findall(==(step), steps)
            isempty(selected) && continue
            haskey(layer.models, step) ||
                error("V2.4 L2 per-step layer lacks a model for step $step")
            out[selected] = SolarSINDy.v23_predict(layer.models[step], X[selected, :])
        end
        out
    end
    for (r, (_, i)) in enumerate(rows)
        data.residual_raw[i] = prediction[r]
        data.residual[i] = v24_cap_residual(prediction[r], data.step[i])
        data.l2_applied[i] = true
    end
    return data
end

# ---------------------------------------------------------------------------
# L3 guard and variant centers
# ---------------------------------------------------------------------------

"""
    v24_deepening(rate, coupling, latest) -> Bool

Deepening cell of the L3 depth-safe guard: a one-hour fall steeper than
`-15` nT/h, or active coupling with the ring current already at or below
`-50` nT.
"""
v24_deepening(rate::Real, coupling::Real, latest::Real) =
    Float64(rate) < V24_GUARD_RATE_NT_PER_H ||
    (Float64(coupling) > 0.0 && Float64(latest) <= V24_GUARD_DEPTH_NT)

"""
    v24_guard(l1, l2, rate, coupling, latest) -> Float64

The guarded center: in a deepening cell the residual may deepen the forecast but
never lift it, so the center is `min(l2, l1)`; elsewhere it is `l2`.
"""
v24_guard(l1::Real, l2::Real, rate::Real, coupling::Real, latest::Real) =
    v24_deepening(rate, coupling, latest) ? min(Float64(l2), Float64(l1)) : Float64(l2)

"""
    v24_build_centers!(data)

Materialise the preregistered variant centers from the L1, L1a and residual
columns already attached to `data`.

`v2_4d` is the Amendment A2 industrial center: the same guard arithmetic applied
to the floor-constrained stack against the served static V2.2 stack rather than
against its own uncorrected center, so in a deepening cell the candidate is
`min(V2.4a-floor, static V2.2)`. `v2_4e` is the Amendment A3 ten-expert floor
stack and `v2_4f` is that stack under the same static guard.
"""
function v24_build_centers!(data::V24YearData)
    n = length(data)
    b = Vector{Float64}(undef, n)
    c = Vector{Float64}(undef, n)
    d = Vector{Float64}(undef, n)
    f = Vector{Float64}(undef, n)
    reference = data.comparators[V24_D_GUARD_REFERENCE]
    for i in 1:n
        b[i] = data.l1[i] + data.residual[i]
        c[i] = v24_guard(data.l1[i], b[i], data.rate[i], data.coupling[i], data.latest[i])
        d[i] = v24_guard(reference[i], data.l1_floor[i], data.rate[i], data.coupling[i],
                         data.latest[i])
        f[i] = v24_guard(reference[i], data.l1_ten[i], data.rate[i], data.coupling[i],
                         data.latest[i])
    end
    data.centers[:v2_4a] = copy(data.l1)
    data.centers[:v2_4b] = b
    data.centers[:v2_4c] = c
    data.centers[:v2_4a_floor] = copy(data.l1_floor)
    data.centers[:v2_4d] = d
    data.centers[:v2_4e] = copy(data.l1_ten)
    data.centers[:v2_4f] = f
    for variant in V24_VARIANTS
        all(isfinite, data.centers[variant]) || error(
            "V2.4 variant $variant produced a non-finite center in $(data.year)",
        )
    end
    return data
end

# ---------------------------------------------------------------------------
# L4: split conformal
# ---------------------------------------------------------------------------

"""
    v24_conformal_halfwidth(residuals, coverage) -> NamedTuple

Finite-sample split-conformal half-width: the `k`-th smallest absolute residual
with `k = ceil((n+1)·coverage)` clamped to `n`. The guaranteed marginal coverage
`k/(n+1)` is returned alongside so a sample too small to reach the nominal level
is visible rather than overstated. This restates the deployed V2.1 rule; the
test suite requires agreement with `SolarSINDy._conformal_quantile`.
"""
function v24_conformal_halfwidth(residuals::AbstractVector{<:Real}, coverage::Real)
    finite = Float64[abs(r) for r in residuals if isfinite(r)]
    n = length(finite)
    n >= 1 || throw(ArgumentError("a conformal quantile needs at least one finite residual"))
    0.0 < Float64(coverage) < 1.0 ||
        throw(ArgumentError("conformal coverage must lie strictly between 0 and 1"))
    sorted = sort(finite)
    k = clamp(ceil(Int, (n + 1) * Float64(coverage)), 1, n)
    return (half_width=sorted[k], coverage_floor=k / (n + 1), n=n)
end

"""
    v24_activity(latest) -> Symbol

Activity label of the plan's original conformal split: disturbed at or below
-30 nT. Amendment A1 replaced it as the stratum key with
[`v24_depth_bin`](@ref) — one "disturbed" stratum mixed -30 nT and -400 nT states,
so its quantile was set by the shallow majority and storm coverage fell below the
G3 floor. The label is kept because the -30 nT boundary still describes the
disturbed population the intervals are reported over.
"""
v24_activity(latest::Real) = Float64(latest) <= V24_DISTURBED_NT ? :disturbed : :quiet

"""
    v24_fit_conformal(sources; coverage) -> Dict

Split-conformal half-widths per `(step, depth bin)` with a per-step pooled
fallback, which is the Amendment A1 stratification. `sources` is a vector of
`(V24YearData, center vector)` pairs drawn from years strictly before the scored
fold.

A depth stratum thinner than [`V24_CONFORMAL_MIN_STRATUM_N`](@ref) cannot support
its own quantile and takes the step's pooled stratum instead. The pooled stratum
is always fitted, so a step always has a usable quantile, and which one a stratum
used is recorded in `source`. Falling back on the pooled stratum can narrow a deep
bin's interval rather than widen it, which is the price of the amendment's choice:
a quantile estimated from fewer than twenty residuals is not an estimate of the
bin's spread either, and the fallback is visible in the artifact. In the study's
folds every bin carries thousands of residuals per step, so the fallback is a
guard rather than a working path.
"""
function v24_fit_conformal(sources::AbstractVector{<:Tuple{V24YearData,Vector{Float64}}};
                           coverage::Real=V24_COVERAGE,
                           cutoff::Union{Nothing,DateTime}=nothing)
    buckets = Dict{Tuple{Int,Symbol},Vector{Float64}}()
    for (data, centers) in sources
        length(centers) == length(data) || throw(DimensionMismatch(
            "conformal source centers do not match the fold row count",
        ))
        for i in 1:length(data)
            (isfinite(data.obs[i]) && isfinite(centers[i])) || continue
            v24_in_pool(data, i, cutoff) || continue
            residual = data.obs[i] - centers[i]
            step = data.step[i]
            push!(get!(buckets, (step, v24_depth_bin(data.latest[i])), Float64[]), residual)
            push!(get!(buckets, (step, V24_POOLED_DEPTH), Float64[]), residual)
        end
    end
    isempty(buckets) && error("V2.4 conformal calibration found no residuals")
    fitted = Dict{Tuple{Int,Symbol},NamedTuple}()
    for (key, residuals) in buckets
        fitted[key] = v24_conformal_halfwidth(residuals, coverage)
    end
    resolved = Dict{Tuple{Int,Symbol},NamedTuple}()
    for step in V24_STEPS
        pooled = get(fitted, (step, V24_POOLED_DEPTH), nothing)
        pooled === nothing &&
            error("V2.4 conformal calibration has no pooled stratum at step $step")
        resolved[(step, V24_POOLED_DEPTH)] = merge(pooled, (source="own",))
        for depth in V24_DEPTH_BINS
            own = get(fitted, (step, depth), nothing)
            if own !== nothing && own.n >= V24_CONFORMAL_MIN_STRATUM_N
                resolved[(step, depth)] = merge(own, (source="own",))
                continue
            end
            label = own === nothing ? "pooled_fallback_absent" :
                    "pooled_fallback_thin_$(own.n)"
            resolved[(step, depth)] = merge(pooled, (source=label,))
        end
    end
    return resolved
end

"Per-row conformal half-widths of a fold under fitted `strata`."
function v24_apply_conformal(data::V24YearData, strata::Dict{Tuple{Int,Symbol},NamedTuple})
    out = Vector{Float64}(undef, length(data))
    for i in 1:length(data)
        key = (data.step[i], v24_depth_bin(data.latest[i]))
        stratum = get(strata, key, nothing)
        if stratum === nothing
            stratum = get(strata, (data.step[i], V24_POOLED_DEPTH), nothing)
            stratum === nothing && error("V2.4 conformal strata lack $(key)")
        end
        out[i] = stratum.half_width
    end
    return out
end

"""
    v24_conformal_rows(strata, year, label; note) -> Vector{NamedTuple}

Persistable form of one fitted conformal calibration, emitted over the fixed
`(step, depth bin)` grid — the pooled stratum first — so the artifact does not
depend on dictionary order.
"""
function v24_conformal_rows(strata::Dict{Tuple{Int,Symbol},NamedTuple}, year::Integer,
                            label::AbstractString; note::AbstractString="")
    rows = NamedTuple[]
    for step in V24_STEPS, depth in (V24_POOLED_DEPTH, V24_DEPTH_BINS...)
        stratum = strata[(step, depth)]
        push!(rows, (
            fold_year=Int(year), variant=String(label), model_step_hours=step,
            depth_bin=String(depth), half_width_nt=stratum.half_width, n=stratum.n,
            coverage_floor=stratum.coverage_floor, source=stratum.source,
            calibration_note=String(note),
        ))
    end
    return rows
end

"""
    v24_interval_score(observation, lower, upper, alpha) -> Float64

Winkler interval score of a central `1 - alpha` interval: the width plus a
`2/alpha` penalty on the distance by which the observation falls outside.
Smaller is better, and the score rewards a narrow interval only while it keeps
covering.
"""
function v24_interval_score(observation::Real, lower::Real, upper::Real, alpha::Real)
    y = Float64(observation)
    lo = Float64(lower)
    hi = Float64(upper)
    a = Float64(alpha)
    0.0 < a < 1.0 || throw(ArgumentError("interval score alpha must lie in (0, 1)"))
    hi >= lo || throw(ArgumentError("interval score needs upper >= lower"))
    score = hi - lo
    y < lo && (score += 2.0 * (lo - y) / a)
    y > hi && (score += 2.0 * (y - hi) / a)
    return score
end

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

"Every column the summary tables score, in written order."
v24_scored_models() = collect(Symbol, (V24_VARIANTS..., V24_GATED_COMPARATORS...,
                                       V24_REPORTED_COMPARATORS...))

"Center vector of any scored model name within one fold."
function v24_center_of(data::V24YearData, model::Symbol)
    haskey(data.centers, model) && return data.centers[model]
    haskey(data.comparators, model) && return data.comparators[model]
    error("V2.4 fold $(data.year) has no column for model $model")
end

"Folds whose calendar year lies in `scope`."
v24_scope_rows(years::AbstractVector{V24YearData}, scope) =
    [data for data in years if data.year in scope]

"Active eras: those `eras` entries covered by at least one scored fold."
function v24_active_eras(eras, scored_years)
    keep = Tuple(k for k in keys(eras)
                 if any(y -> y in getproperty(eras, k), scored_years))
    isempty(keep) && error("V2.4 scoring found no era covered by the folds")
    return NamedTuple{keep}(Tuple(getproperty(eras, k) for k in keep))
end

"""
    v24_summary_rows(years, eras) -> Vector{NamedTuple}

Pooled root mean square error, bias and mean absolute error of every scored
model at every model step, on each era of `eras`. One pass over the rows per era
accumulates every model at once.
"""
function v24_summary_rows(years::AbstractVector{V24YearData}, eras)
    models = v24_scored_models()
    rows = NamedTuple[]
    for scope_name in keys(eras)
        folds = v24_scope_rows(years, getproperty(eras, scope_name))
        isempty(folds) && continue
        squared = zeros(Float64, length(models), length(V24_STEPS))
        absolute = zeros(Float64, length(models), length(V24_STEPS))
        signed = zeros(Float64, length(models), length(V24_STEPS))
        counts = zeros(Int, length(models), length(V24_STEPS))
        for data in folds
            centers = [v24_center_of(data, model) for model in models]
            for i in 1:length(data)
                isfinite(data.obs[i]) || continue
                slot = V24_STEP_SLOT[data.step[i]]
                for m in eachindex(models)
                    value = centers[m][i]
                    isfinite(value) || continue
                    error_nt = data.obs[i] - value
                    squared[m, slot] += abs2(error_nt)
                    absolute[m, slot] += abs(error_nt)
                    signed[m, slot] += error_nt
                    counts[m, slot] += 1
                end
            end
        end
        for m in eachindex(models), slot in eachindex(V24_STEPS)
            n = counts[m, slot]
            push!(rows, (
                scope=String(scope_name), model=String(models[m]),
                model_step_hours=V24_STEPS[slot], n=n,
                rmse_nt=(n == 0 ? NaN : sqrt(squared[m, slot] / n)),
                bias_nt=(n == 0 ? NaN : signed[m, slot] / n),
                mae_nt=(n == 0 ? NaN : absolute[m, slot] / n),
            ))
        end
    end
    return rows
end

"""
    v24_cell_rows(years, eras) -> Vector{NamedTuple}

Per-cell metrics of every scored model. Cells are the plan section 6-A2 storm
cells of `SolarSINDy.v23_regime_cells`, evaluated from issue-time state only.
"""
function v24_cell_rows(years::AbstractVector{V24YearData}, eras)
    models = v24_scored_models()
    cell_index = Dict(cell => c for (c, cell) in enumerate(V24_CELL_NAMES))
    # Cell membership depends on issue-time state alone, so it is derived once per
    # fold and shared by the eras that contain that fold.
    membership = Dict{Int,Vector{Vector{Int}}}()
    for data in years
        membership[data.year] = [
            [cell_index[Symbol(label)] for label in
             SolarSINDy.v23_regime_cells(data.latest[i], data.rate[i], data.coupling[i])]
            for i in 1:length(data)
        ]
    end
    rows = NamedTuple[]
    for scope_name in keys(eras)
        folds = v24_scope_rows(years, getproperty(eras, scope_name))
        isempty(folds) && continue
        dims = (length(V24_CELL_NAMES), length(models), length(V24_STEPS))
        squared = zeros(Float64, dims)
        signed = zeros(Float64, dims)
        counts = zeros(Int, dims)
        for data in folds
            centers = [v24_center_of(data, model) for model in models]
            labels_of = membership[data.year]
            for i in 1:length(data)
                isfinite(data.obs[i]) || continue
                slot = V24_STEP_SLOT[data.step[i]]
                labels = labels_of[i]
                for m in eachindex(models)
                    value = centers[m][i]
                    isfinite(value) || continue
                    error_nt = data.obs[i] - value
                    for c in labels
                        squared[c, m, slot] += abs2(error_nt)
                        signed[c, m, slot] += error_nt
                        counts[c, m, slot] += 1
                    end
                end
            end
        end
        for c in eachindex(V24_CELL_NAMES), m in eachindex(models),
            slot in eachindex(V24_STEPS)
            n = counts[c, m, slot]
            n == 0 && continue
            push!(rows, (
                scope=String(scope_name), cell=String(V24_CELL_NAMES[c]),
                model=String(models[m]), model_step_hours=V24_STEPS[slot], n=n,
                rmse_nt=sqrt(squared[c, m, slot] / n),
                bias_nt=signed[c, m, slot] / n,
            ))
        end
    end
    return rows
end

"""
    v24_paired_rows(years, scope, candidate, comparator, step) -> NamedTuple

Squared errors of a candidate and a comparator on the rows where both are
scored, ordered by issue time so that the block sums — and therefore the
reported bootstrap bound — do not depend on how the caller assembled the fold
list.
"""
function v24_paired_rows(years::AbstractVector{V24YearData}, scope, candidate::Symbol,
                         comparator::Symbol, step::Integer)
    issues = DateTime[]
    candidate_se = Float64[]
    comparator_se = Float64[]
    n_candidate = 0
    n_comparator = 0
    for data in v24_scope_rows(years, scope)
        candidate_centers = v24_center_of(data, candidate)
        comparator_centers = v24_center_of(data, comparator)
        for i in 1:length(data)
            (data.step[i] == Int(step) && isfinite(data.obs[i])) || continue
            isfinite(candidate_centers[i]) && (n_candidate += 1)
            isfinite(comparator_centers[i]) && (n_comparator += 1)
            (isfinite(candidate_centers[i]) && isfinite(comparator_centers[i])) || continue
            push!(issues, data.issue[i])
            push!(candidate_se, abs2(data.obs[i] - candidate_centers[i]))
            push!(comparator_se, abs2(data.obs[i] - comparator_centers[i]))
        end
    end
    order = sortperm(issues; alg=MergeSort)
    return (issues=issues[order], candidate_se=candidate_se[order],
            comparator_se=comparator_se[order], n_candidate=n_candidate,
            n_comparator=n_comparator)
end

"""
    v24_bootstrap_rows(years, eras, candidate; comparators, replicates) -> Vector{NamedTuple}

Paired 168 h calendar-block bootstrap of the candidate against every gated
comparator, at every model step, on each era. The Holm step-down is applied
within each era across the whole `steps × comparators` family, which is the
family plan section 6 names for G1.
"""
function v24_bootstrap_rows(years::AbstractVector{V24YearData}, eras, candidate::Symbol;
                            comparators=V24_GATED_COMPARATORS,
                            replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES)
    rows = NamedTuple[]
    for scope_name in keys(eras)
        scope = getproperty(eras, scope_name)
        isempty(v24_scope_rows(years, scope)) && continue
        family = NamedTuple[]
        for comparator in comparators, step in V24_STEPS
            paired = v24_paired_rows(years, scope, candidate, comparator, step)
            isempty(paired.issues) && continue
            result = SolarSINDy.v23_block_bootstrap(
                paired.comparator_se, paired.candidate_se, paired.issues;
                replicates=replicates,
            )
            push!(family, (
                scope=String(scope_name), candidate=String(candidate),
                comparator=String(comparator), model_step_hours=step,
                n=length(paired.issues), n_candidate_scored=paired.n_candidate,
                n_comparator_scored=paired.n_comparator,
                rows_matched=(paired.n_candidate == length(paired.issues) &&
                              paired.n_comparator == length(paired.issues)),
                rmse_candidate_nt=sqrt(mean(paired.candidate_se)),
                rmse_comparator_nt=sqrt(mean(paired.comparator_se)),
                gain_nt=result.point, lower_nt=result.lower,
                p_one_sided=result.p_one_sided, n_blocks=result.n_blocks,
            ))
        end
        isempty(family) && continue
        holm = SolarSINDy.v23_holm([Float64(row.p_one_sided) for row in family])
        for (index, row) in enumerate(family)
            push!(rows, merge(row, (holm_p=holm[index], family_size=length(family))))
        end
    end
    return rows
end

"""
    v24_interval_rows(years, eras, variants) -> Vector{NamedTuple}

Coverage, mean width and mean Winkler interval score of every variant's
conformal intervals and of the matched-recipe V2.1 intervals: pooled and per
step, on each era, over the whole sample and over the two storm subsets G3
names.
"""
function v24_interval_rows(years::AbstractVector{V24YearData}, eras, variants)
    alpha = 1.0 - V24_COVERAGE
    subset_names = (:pooled, :storm_le_m50, :storm_le_m100)
    labels = collect(Symbol, variants)
    rows = NamedTuple[]
    for scope_name in keys(eras)
        folds = v24_scope_rows(years, getproperty(eras, scope_name))
        isempty(folds) && continue
        dims = (length(labels), length(subset_names), length(V24_STEPS))
        counts = zeros(Int, dims)
        covered = zeros(Int, dims)
        width = zeros(Float64, dims)
        score = zeros(Float64, dims)
        for data in folds
            for (v, label) in enumerate(labels)
                haskey(data.half_widths, label) || continue
                centers = v24_center_of(data, label)
                half = data.half_widths[label]
                for i in 1:length(data)
                    (isfinite(data.obs[i]) && isfinite(centers[i]) &&
                        isfinite(half[i])) || continue
                    slot = V24_STEP_SLOT[data.step[i]]
                    lo = centers[i] - half[i]
                    hi = centers[i] + half[i]
                    inside = data.obs[i] >= lo && data.obs[i] <= hi
                    interval = v24_interval_score(data.obs[i], lo, hi, alpha)
                    memberships = (true, data.latest[i] <= V24_G3_STORM_DST_NT,
                                   data.latest[i] <= -100.0)
                    for s in eachindex(subset_names)
                        memberships[s] || continue
                        counts[v, s, slot] += 1
                        inside && (covered[v, s, slot] += 1)
                        width[v, s, slot] += hi - lo
                        score[v, s, slot] += interval
                    end
                end
            end
        end
        for v in eachindex(labels), s in eachindex(subset_names)
            for slot in (0, eachindex(V24_STEPS)...)
                n = slot == 0 ? sum(view(counts, v, s, :)) : counts[v, s, slot]
                n == 0 && continue
                covered_n = slot == 0 ? sum(view(covered, v, s, :)) : covered[v, s, slot]
                width_sum = slot == 0 ? sum(view(width, v, s, :)) : width[v, s, slot]
                score_sum = slot == 0 ? sum(view(score, v, s, :)) : score[v, s, slot]
                push!(rows, (
                    scope=String(scope_name), variant=String(labels[v]),
                    subset=String(subset_names[s]),
                    model_step_hours=(slot == 0 ? 0 : V24_STEPS[slot]), n=n,
                    coverage=covered_n / n, mean_width_nt=width_sum / n,
                    mean_interval_score_nt=score_sum / n,
                ))
            end
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# Selection rule (plan section 4)
# ---------------------------------------------------------------------------

"Root mean square error of one model pooled over `steps` on `scope`."
function v24_pooled_rmse(years::AbstractVector{V24YearData}, scope, model::Symbol, steps)
    total = 0.0
    n = 0
    for data in v24_scope_rows(years, scope)
        centers = v24_center_of(data, model)
        for i in 1:length(data)
            (data.step[i] in steps && isfinite(centers[i]) && isfinite(data.obs[i])) ||
                continue
            total += abs2(data.obs[i] - centers[i])
            n += 1
        end
    end
    return (rmse_nt=(n == 0 ? NaN : sqrt(total / n)), n=n)
end

"""
    v24_cell_paired_rows(years, scope, cell, candidate, comparator, step) -> NamedTuple

Squared errors of a candidate and a comparator on the rows of one storm cell at
one model step, ordered by issue time. This is [`v24_paired_rows`](@ref)
restricted to the cell membership of `SolarSINDy.v23_regime_cells`, which is what
the Amendment A2 bootstrap support for G2 needs: the cell's own rows, not the
pooled sample.
"""
function v24_cell_paired_rows(years::AbstractVector{V24YearData}, scope, cell::Symbol,
                              candidate::Symbol, comparator::Symbol, step::Integer)
    issues = DateTime[]
    candidate_se = Float64[]
    comparator_se = Float64[]
    for data in v24_scope_rows(years, scope)
        candidate_centers = v24_center_of(data, candidate)
        comparator_centers = v24_center_of(data, comparator)
        for i in 1:length(data)
            (data.step[i] == Int(step) && isfinite(data.obs[i])) || continue
            (isfinite(candidate_centers[i]) && isfinite(comparator_centers[i])) || continue
            cell in SolarSINDy.v23_regime_cells(data.latest[i], data.rate[i],
                                                data.coupling[i]) || continue
            push!(issues, data.issue[i])
            push!(candidate_se, abs2(data.obs[i] - candidate_centers[i]))
            push!(comparator_se, abs2(data.obs[i] - comparator_centers[i]))
        end
    end
    order = sortperm(issues; alg=MergeSort)
    return (issues=issues[order], candidate_se=candidate_se[order],
            comparator_se=comparator_se[order])
end

"""
    v24_cell_loss_lower(years, scope, cell, candidate, comparator, step; replicates)

One-sided 95 % lower bound of the candidate's root-mean-square-error *loss* to a
comparator inside one storm cell, by the same fixed 168 h calendar-block bootstrap
the pooled gates use. The roles are passed reversed on purpose: the primitive
returns `RMSE(first) - RMSE(second)`, so `(candidate, comparator)` makes the
statistic the loss and its `alpha` quantile the lower bound of the loss. A bound
at or below zero means the cell's rows cannot distinguish the loss from noise.

Returns `NaN` when the cell has too few 168 h blocks to bootstrap, which is
reported rather than treated as a pass or a failure.
"""
function v24_cell_loss_lower(years::AbstractVector{V24YearData}, scope, cell::Symbol,
                             candidate::Symbol, comparator::Symbol, step::Integer;
                             replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES)
    paired = v24_cell_paired_rows(years, scope, cell, candidate, comparator, step)
    isempty(paired.issues) && return (lower_nt=NaN, point_nt=NaN, n=0, n_blocks=0)
    try
        result = SolarSINDy.v23_block_bootstrap(
            paired.candidate_se, paired.comparator_se, paired.issues;
            replicates=replicates,
        )
        return (lower_nt=result.lower, point_nt=result.point, n=length(paired.issues),
                n_blocks=result.n_blocks)
    catch
        # Fewer than two 168 h blocks: the cell is real but unbootstrappable.
        return (lower_nt=NaN, point_nt=NaN, n=length(paired.issues), n_blocks=0)
    end
end

"""
    v24_storm_guard(cell_rows, scope, candidate; years, replicates) -> NamedTuple

The plan section 6 G2 storm guards for one candidate on one era, evaluated from
the per-cell metric rows. A guarded cell with at least
[`V24_G2_MIN_CELL_ROWS`](@ref) rows may lose at most
[`V24_G2_MAX_LOSS_NT`](@ref) nT to the best gated comparator in that cell and
may never lose to served V2.1; the intense-deepening cell at six hours may lose
no root mean square error at all and must keep its mean signed error inside
`+/-10` nT.

Amendment A2 adds bootstrap support: a loss counts as a breach only when its
one-sided 95 % lower bound over that cell's own rows is positive. Without it a
73-row cell of 60 nT-scale errors can fail the gate on a difference the sample
cannot resolve. Every candidate breach is reported either way — as `breach` when
the bound supports it and as `within_noise` when it does not — so nothing is
hidden by the amendment.

`scope` is the label written into the returned rows; `scope_years` is the calendar
range those rows cover. Both `years` and `scope_years` are needed to run the
bootstrap. Without them the deterministic losses are reported unchanged and every
one of them counts, which is the conservative reading used by unit tests that
supply cell tables only.
"""
function v24_storm_guard(cell_rows, scope, candidate::Symbol;
                         years::Union{Nothing,AbstractVector{V24YearData}}=nothing,
                         scope_years=nothing,
                         replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES)
    bootstrap_years = (years === nothing || scope_years === nothing ||
                       isempty(years)) ? nothing : years
    scope_label = String(scope)
    index = Dict{Tuple{String,String,Int},Any}()
    for row in cell_rows
        row.scope == scope_label || continue
        index[(row.cell, row.model, row.model_step_hours)] = row
    end
    function best_comparator(cell::AbstractString, step::Int, n::Int)
        name = ""
        rmse = Inf
        for comparator in V24_GATED_COMPARATORS
            row = get(index, (cell, String(comparator), step), nothing)
            row === nothing && continue
            row.n == n || continue
            row.rmse_nt < rmse || continue
            rmse = row.rmse_nt
            name = String(comparator)
        end
        return (name=name, rmse=rmse)
    end
    failures = String[]
    within_noise = String[]
    detail = NamedTuple[]
    evaluated = 0
    # A deterministic loss becomes a breach only if the cell's own rows support it.
    function record!(cell::Symbol, step::Int, comparator::AbstractString, loss::Float64,
                     allowance::Float64, clause::AbstractString, n::Int)
        label = _v24_sprintf("%s@%dh_vs_%s:%+.3f", cell, step, comparator, loss)
        support = bootstrap_years === nothing ?
            (lower_nt=NaN, point_nt=loss, n=n, n_blocks=0) :
            v24_cell_loss_lower(bootstrap_years, scope_years, cell, candidate,
                                Symbol(comparator), step; replicates=replicates)
        supported = bootstrap_years === nothing || (isfinite(support.lower_nt) &&
                                                    support.lower_nt > 0.0)
        supported ? push!(failures, label) :
            push!(within_noise, label * _v24_sprintf("(lower=%+.3f)", support.lower_nt))
        push!(detail, (
            scope=scope_label, candidate=String(candidate), cell=String(cell),
            model_step_hours=step, clause=String(clause), comparator=String(comparator),
            n=n, n_blocks=support.n_blocks, loss_nt=loss, allowance_nt=allowance,
            bootstrap_loss_nt=support.point_nt, lower_nt=support.lower_nt,
            counted=supported,
        ))
        return nothing
    end
    for cell in V24_G2_CELLS, step in V24_STEPS
        row = get(index, (String(cell), String(candidate), step), nothing)
        row === nothing && continue
        row.n >= V24_G2_MIN_CELL_ROWS || continue
        best = best_comparator(String(cell), step, row.n)
        isfinite(best.rmse) || continue
        evaluated += 1
        loss = row.rmse_nt - best.rmse
        loss <= V24_G2_MAX_LOSS_NT ||
            record!(cell, step, best.name, loss, V24_G2_MAX_LOSS_NT, "best_comparator",
                    row.n)
        served = get(index, (String(cell), "served_v2_1", step), nothing)
        served === nothing && continue
        served_loss = row.rmse_nt - served.rmse_nt
        served_loss <= V24_G2_SERVED_TOLERANCE_NT ||
            record!(cell, step, "served_v2_1", served_loss, V24_G2_SERVED_TOLERANCE_NT,
                    "never_below_served", row.n)
    end
    intense_rows = 0
    intense = get(index, (String(V24_G2_INTENSE_CELL), String(candidate),
                          V24_G2_INTENSE_STEP), nothing)
    if intense !== nothing
        intense_rows = intense.n
        best = best_comparator(String(V24_G2_INTENSE_CELL), V24_G2_INTENSE_STEP, intense.n)
        if isfinite(best.rmse)
            loss = intense.rmse_nt - best.rmse
            loss <= V24_G2_SERVED_TOLERANCE_NT ||
                record!(V24_G2_INTENSE_CELL, V24_G2_INTENSE_STEP, best.name, loss,
                        V24_G2_SERVED_TOLERANCE_NT, "intense_no_loss", intense.n)
        end
        # The bias clause is a mean, not a difference of two root mean squares, so
        # the bootstrap support of the amendment does not apply to it.
        abs(intense.bias_nt) <= V24_G2_INTENSE_BIAS_NT || push!(failures, _v24_sprintf(
            "intense_deepening@%dh_bias:%+.2f", V24_G2_INTENSE_STEP, intense.bias_nt,
        ))
    end
    return (pass=isempty(failures), failures=join(failures, "|"),
            within_noise=join(within_noise, "|"), detail=detail,
            cells_evaluated=evaluated, intense_rows=intense_rows)
end

"""
    v24_select_variant(years, cell_rows, eras) -> NamedTuple

Selection rule of Amendment A1: among all four variants, the one with the lowest
mean over steps `{2, 3, 6}` of the per-step pooled root mean square error on era
E1, subject to the G2 storm guards on E1; ties go to the safer center in the order
of [`V24_SELECTABLE_VARIANTS`](@ref). When no variant clears the E1 guards the
whole set stays in contention, the safest best-scoring center is returned and the
guard failure is recorded, so the gate table reports the breach instead of the
selection hiding it.

The row-pooled root mean square error over the same steps is computed and
persisted next to the mean, because the two orderings can in principle disagree
and a reader is entitled to see both.
"""
function v24_select_variant(years::AbstractVector{V24YearData}, cell_rows, eras;
                            replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES,
                            bootstrap_guards::Bool=true)
    haskey(eras, :E1) || error("V2.4 selection needs era E1")
    scope = getproperty(eras, :E1)
    trace = NamedTuple[]
    eligible = Symbol[]
    scores = Dict{Symbol,Float64}()
    guard_detail = NamedTuple[]
    for variant in V24_SELECTABLE_VARIANTS
        per_step = [v24_pooled_rmse(years, scope, variant, (step,))
                    for step in V24_SELECTION_STEPS]
        mean_rmse = mean([entry.rmse_nt for entry in per_step])
        pooled = v24_pooled_rmse(years, scope, variant, V24_SELECTION_STEPS)
        guard = v24_storm_guard(cell_rows, :E1, variant;
                                years=(bootstrap_guards ? years : nothing),
                                scope_years=scope, replicates=replicates)
        append!(guard_detail, guard.detail)
        scores[variant] = mean_rmse
        guard.pass && push!(eligible, variant)
        push!(trace, merge((
            variant=String(variant), era="E1",
            selection_steps=join(V24_SELECTION_STEPS, "|"),
            mean_step_rmse_nt=mean_rmse, pooled_rmse_nt=pooled.rmse_nt, n=pooled.n,
        ), NamedTuple{Tuple(Symbol("rmse_step", step, "_nt")
                            for step in V24_SELECTION_STEPS)}(
            Tuple(entry.rmse_nt for entry in per_step),
        ), (
            guards_pass=guard.pass, guard_failures=guard.failures,
            guard_within_noise=guard.within_noise,
            cells_evaluated=guard.cells_evaluated, eligible=guard.pass,
        )))
    end
    pool = isempty(eligible) ? collect(V24_SELECTABLE_VARIANTS) : eligible
    # Visiting the safest center first makes an exact tie resolve toward it.
    selected = nothing
    for variant in V24_SELECTABLE_VARIANTS
        (variant in pool && isfinite(scores[variant])) || continue
        (selected === nothing || scores[variant] < scores[selected]) && (selected = variant)
    end
    selected === nothing && error(
        "V2.4 selection has no finite pooled RMSE for any selectable variant",
    )
    return (selected=selected, eligible=eligible, trace=trace, scores=scores,
            guard_detail=guard_detail, guards_all_failed=isempty(eligible))
end

# ---------------------------------------------------------------------------
# Gates (plan section 6)
# ---------------------------------------------------------------------------

"""
    v24_gate_rows(summary_rows, cell_rows, bootstrap_rows, interval_rows, eras, candidate;
                  years, replicates)

Evaluate gates G1, G2 and G3 for the selected candidate on every era. One row per
requirement, with the observed quantity written out so the artifact can be
re-derived by hand.

G1 follows Amendment A2. At a step whose realized-driver-oracle headroom over the
best comparator, `RMSE(best comparator) - RMSE(oracle)`, is smaller than
[`V24_G1_HEADROOM_NT`](@ref), no causal method can be asked for a
`max(0.10 nT, 1 %)` margin: the whole distance to the noncausal ceiling is
smaller than the margin. Such a step requires non-inferiority instead — a lower
bound above `-0.05` nT against the strongest comparator. Every other step keeps
the superiority requirement: lower RMSE than every comparator on identical rows
and a positive Holm-adjusted lower bound. The margin flag is reported at every
step and gated at none, and the headroom itself is persisted per step.

`years` enables the Amendment A2 bootstrap support for G2.
"""
function v24_gate_rows(summary_rows, cell_rows, bootstrap_rows, interval_rows, eras,
                       candidate::Symbol;
                       years::Union{Nothing,AbstractVector{V24YearData}}=nothing,
                       replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES)
    rows = NamedTuple[]
    emit(gate, scope, family, step, comparator, requirement, observed, holm, pass;
         rule="", candidate_rmse_nt=NaN, best_comparator_rmse_nt=NaN,
         oracle_rmse_nt=NaN, headroom_nt=NaN, gain_nt=NaN, lower_nt=NaN,
         margin_required_nt=NaN, margin_pass=false, beat_all=false) =
        push!(rows, (
            gate=String(gate), scope=String(scope), family=String(family),
            model_step_hours=step, comparator=String(comparator),
            requirement=String(requirement), observed=String(observed),
            holm_p=holm, pass=pass, rule=String(rule),
            candidate_rmse_nt=candidate_rmse_nt,
            best_comparator_rmse_nt=best_comparator_rmse_nt,
            oracle_rmse_nt=oracle_rmse_nt, headroom_nt=headroom_nt, gain_nt=gain_nt,
            lower_nt=lower_nt, margin_required_nt=margin_required_nt,
            margin_pass=margin_pass, beat_all=beat_all,
        ))
    verdicts = Dict{Tuple{String,String},Bool}()

    summary = Dict{Tuple{String,String,Int},Any}()
    for row in summary_rows
        summary[(row.scope, row.model, row.model_step_hours)] = row
    end
    boot = Dict{Tuple{String,String,Int},Any}()
    for row in bootstrap_rows
        row.candidate == String(candidate) || continue
        boot[(row.scope, row.comparator, row.model_step_hours)] = row
    end
    intervals = Dict{Tuple{String,String,String,Int},Any}()
    for row in interval_rows
        intervals[(row.scope, row.variant, row.subset, row.model_step_hours)] = row
    end

    for scope_name in keys(eras)
        scope = String(scope_name)
        haskey(summary, (scope, String(candidate), first(V24_STEPS))) || continue

        # --- G1: lower RMSE than every comparator, with margin against the best ---
        g1_pass = true
        for step in V24_STEPS
            candidate_row = summary[(scope, String(candidate), step)]
            best_name = ""
            best_rmse = Inf
            beaten_all = true
            for comparator in V24_GATED_COMPARATORS
                comparator_row = get(summary, (scope, String(comparator), step), nothing)
                if comparator_row === nothing
                    beaten_all = false
                    continue
                end
                comparator_row.n == candidate_row.n || (beaten_all = false)
                candidate_row.rmse_nt < comparator_row.rmse_nt || (beaten_all = false)
                if comparator_row.rmse_nt < best_rmse
                    best_rmse = comparator_row.rmse_nt
                    best_name = String(comparator)
                end
            end
            required = max(V24_G1_MIN_GAIN_NT, V24_G1_MIN_GAIN_FRACTION * best_rmse)
            bootstrap_row = get(boot, (scope, best_name, step), nothing)
            gain = bootstrap_row === nothing ? NaN : bootstrap_row.gain_nt
            lower = bootstrap_row === nothing ? NaN : bootstrap_row.lower_nt
            holm = bootstrap_row === nothing ? NaN : bootstrap_row.holm_p
            matched = bootstrap_row === nothing ? false : bootstrap_row.rows_matched
            oracle_row = get(summary, (scope, String(V24_ORACLE_COLUMN), step), nothing)
            oracle_rmse = oracle_row === nothing ? NaN : oracle_row.rmse_nt
            headroom = best_rmse - oracle_rmse
            headroom_limited = isfinite(headroom) && headroom < V24_G1_HEADROOM_NT
            margin_pass = isfinite(gain) && gain >= required
            pass = if headroom_limited
                matched && isfinite(lower) && lower > V24_G1_NONINFERIORITY_NT
            else
                beaten_all && matched && isfinite(gain) && lower > 0.0 && holm < V24_ALPHA
            end
            g1_pass &= pass
            requirement = headroom_limited ? _v24_sprintf(
                "oracle headroom %.3f nT < %.2f nT: non-inferiority only, lower > %.2f nT",
                headroom, V24_G1_HEADROOM_NT, V24_G1_NONINFERIORITY_NT,
            ) : _v24_sprintf(
                "lower RMSE than all %d comparators on identical rows; lower > 0; " *
                "Holm p < %.2f (margin >= %.3f nT reported, not gated)",
                length(V24_GATED_COMPARATORS), V24_ALPHA, required,
            )
            emit("G1", scope, "beats_all", step, best_name, requirement, _v24_sprintf(
                "rmse=%.3f best=%.3f oracle=%.3f headroom=%+.3f gain=%+.3f lower=%+.3f " *
                "holm=%.4f beat_all=%s rows_matched=%s margin_%.3f=%s margin_0p25=%s",
                candidate_row.rmse_nt, best_rmse, oracle_rmse, headroom, gain, lower, holm,
                beaten_all, matched, required, margin_pass,
                isfinite(gain) && gain >= V24_G1_REPORTED_MARGIN_NT,
            ), holm, pass; rule=(headroom_limited ? "non_inferiority" : "superiority"),
               candidate_rmse_nt=candidate_row.rmse_nt, best_comparator_rmse_nt=best_rmse,
               oracle_rmse_nt=oracle_rmse, headroom_nt=headroom, gain_nt=gain,
               lower_nt=lower, margin_required_nt=required, margin_pass=margin_pass,
               beat_all=beaten_all)
        end
        verdicts[(scope, "G1")] = g1_pass

        # --- G2: storm safety, with the Amendment A2 bootstrap support ---
        guard = v24_storm_guard(cell_rows, scope_name, candidate; years=years,
                                scope_years=getproperty(eras, scope_name),
                                replicates=replicates)
        verdicts[(scope, "G2")] = guard.pass
        emit("G2", scope, "storm_cells", -1, "best_comparator", _v24_sprintf(
            "guarded cells with >= %d rows lose <= %.2f nT to the best comparator and " *
            "never to served V2.1; intense deepening at %d h loses nothing and keeps " *
            "|bias| <= %.1f nT; a loss counts only with a positive one-sided 95%% lower " *
            "bound on the cell's own rows", V24_G2_MIN_CELL_ROWS, V24_G2_MAX_LOSS_NT,
            V24_G2_INTENSE_STEP, V24_G2_INTENSE_BIAS_NT,
        ), (isempty(guard.failures) ?
            "no cell breached ($(guard.cells_evaluated) evaluated, intense rows " *
            "$(guard.intense_rows))" : guard.failures) *
           (isempty(guard.within_noise) ? "" :
            " [within noise: " * guard.within_noise * "]"), NaN, guard.pass)
        for row in guard.detail
            emit("G2", scope, "cell_detail", row.model_step_hours, row.comparator,
                 _v24_sprintf("%s in %s: loss <= %.2f nT or lower bound <= 0", row.clause,
                          row.cell, row.allowance_nt),
                 _v24_sprintf("n=%d blocks=%d loss=%+.3f bootstrap_loss=%+.3f lower=%+.3f",
                          row.n, row.n_blocks, row.loss_nt, row.bootstrap_loss_nt,
                          row.lower_nt),
                 NaN, !row.counted; rule=row.cell, lower_nt=row.lower_nt,
                 gain_nt=-row.loss_nt)
        end

        # --- G3: intervals ---
        g3_pass = true
        pooled = get(intervals, (scope, String(candidate), "pooled", 0), nothing)
        reference = get(intervals, (scope, "served_v2_1", "pooled", 0), nothing)
        if pooled === nothing || reference === nothing
            g3_pass = false
            emit("G3", scope, "pooled", 0, "served_v2_1",
                 "pooled coverage, width and interval score require both interval rows",
                 "absent", NaN, false)
        else
            coverage_ok = V24_G3_COVERAGE_LO <= pooled.coverage <= V24_G3_COVERAGE_HI
            width_ok = pooled.mean_width_nt <=
                V24_G3_MAX_WIDTH_RATIO * reference.mean_width_nt
            score_ok = pooled.mean_interval_score_nt <= reference.mean_interval_score_nt
            pass = coverage_ok && width_ok && score_ok
            g3_pass &= pass
            emit("G3", scope, "pooled", 0, "served_v2_1", _v24_sprintf(
                "coverage in [%.2f, %.2f]; mean width <= %.2f x V2.1; interval score " *
                "<= V2.1", V24_G3_COVERAGE_LO, V24_G3_COVERAGE_HI, V24_G3_MAX_WIDTH_RATIO,
            ), _v24_sprintf(
                "coverage=%.4f width=%.3f v2_1_width=%.3f score=%.3f v2_1_score=%.3f",
                pooled.coverage, pooled.mean_width_nt, reference.mean_width_nt,
                pooled.mean_interval_score_nt, reference.mean_interval_score_nt,
            ), NaN, pass)
        end
        for step in V24_STEPS
            row = get(intervals, (scope, String(candidate), "pooled", step), nothing)
            row === nothing && continue
            pass = row.coverage >= V24_G3_PER_STEP_COVERAGE_MIN
            g3_pass &= pass
            emit("G3", scope, "coverage_per_step", step, "-",
                 _v24_sprintf("coverage >= %.2f", V24_G3_PER_STEP_COVERAGE_MIN),
                 _v24_sprintf("coverage=%.4f n=%d", row.coverage, row.n), NaN, pass)
        end
        storm = get(intervals, (scope, String(candidate), "storm_le_m50", 0), nothing)
        if storm !== nothing
            pass = storm.coverage >= V24_G3_PER_STEP_COVERAGE_MIN
            g3_pass &= pass
            emit("G3", scope, "coverage_storm", 0, "-", _v24_sprintf(
                "coverage >= %.2f on rows with latest Dst <= %.0f nT",
                V24_G3_PER_STEP_COVERAGE_MIN, V24_G3_STORM_DST_NT,
            ), _v24_sprintf("coverage=%.4f n=%d", storm.coverage, storm.n), NaN, pass)
        end
        verdicts[(scope, "G3")] = g3_pass
    end
    return (rows=rows, verdicts=verdicts)
end

"""
    V24_SERVE_RULE_ERAS

Eras the Amendment A3 operational serve rule is decided on. E1 is scored and
persisted with `decides = false`: the fixed static V2.2 stack was fitted on
2010–2017, so on 2014–2017 it is partly in-sample and a comparison there is
biased in its favour. That bias is a reason to disclose E1, not to serve on it.
"""
const V24_SERVE_RULE_ERAS = (:ALL, :E2)

"Storm cells the serve rule checks against the served product (Amendment A3)."
const V24_SERVE_RULE_CELLS = (:latest_le_m50, :latest_le_m100, :active_deepening,
                              :recovery, :intense_deepening)

"The served product the serve rule compares against."
const V24_SERVE_REFERENCE = :static_v2_2

"""
    v24_serve_rule_rows(years, eras, candidate, interval_rows, g3_verdicts; replicates)

Evaluate the Amendment A3 operational serve rule and return one row per checked
quantity.

The rule is narrower than G1 and G2 on purpose: replacing the served product is a
comparison against *that* product, not against the strongest of twelve
comparators. On each deciding era the candidate must (1) beat the served static
V2.2 stack pooled at every model step with a positive one-sided 95 % lower bound,
(2) lose to it in no storm cell of at least 40 rows with a bootstrap-supported
loss, and (3) pass G3. G4 belongs to the serving integration and is carried as
`PENDING`.

Rows carry `decides = false` for E1 so a reader can see the era without it
entering the verdict.
"""
function v24_serve_rule_rows(years::AbstractVector{V24YearData}, eras, candidate::Symbol,
                             interval_rows, g3_verdicts;
                             replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES)
    rows = NamedTuple[]
    emit(scope, decides, check, cell, step, n, gain, lower, pass, note) = push!(rows, (
        scope=String(scope), decides=decides, check=String(check), cell=String(cell),
        model_step_hours=step, comparator=String(V24_SERVE_REFERENCE), n=n,
        gain_nt=gain, lower_nt=lower, pass=pass, note=String(note),
    ))
    verdicts = Dict{String,Bool}()
    cell_index = Dict{Tuple{String,Symbol,Int},Int}()
    for scope_name in keys(eras)
        scope = getproperty(eras, scope_name)
        isempty(v24_scope_rows(years, scope)) && continue
        decides = scope_name in V24_SERVE_RULE_ERAS
        era_pass = true

        # (1) pooled superiority against the served product at every step
        for step in V24_STEPS
            paired = v24_paired_rows(years, scope, candidate, V24_SERVE_REFERENCE, step)
            if isempty(paired.issues)
                era_pass = false
                emit(scope_name, decides, "pooled_gain", "all", step, 0, NaN, NaN, false,
                     "no paired row")
                continue
            end
            result = SolarSINDy.v23_block_bootstrap(
                paired.comparator_se, paired.candidate_se, paired.issues;
                replicates=replicates,
            )
            pass = result.lower > 0.0
            era_pass &= pass
            emit(scope_name, decides, "pooled_gain", "all", step, length(paired.issues),
                 result.point, result.lower, pass,
                 _v24_sprintf("rmse_candidate=%.4f rmse_served=%.4f blocks=%d",
                          sqrt(mean(paired.candidate_se)), sqrt(mean(paired.comparator_se)),
                          result.n_blocks))
        end

        # (2) no bootstrap-supported storm-cell loss to the served product
        for cell in V24_SERVE_RULE_CELLS, step in V24_STEPS
            paired = v24_cell_paired_rows(years, scope, cell, candidate,
                                          V24_SERVE_REFERENCE, step)
            n = length(paired.issues)
            cell_index[(String(scope_name), cell, step)] = n
            n >= V24_G2_MIN_CELL_ROWS || continue
            support = v24_cell_loss_lower(years, scope, cell, candidate,
                                          V24_SERVE_REFERENCE, step;
                                          replicates=replicates)
            loss = sqrt(mean(paired.candidate_se)) - sqrt(mean(paired.comparator_se))
            supported_loss = isfinite(support.lower_nt) && support.lower_nt > 0.0
            era_pass &= !supported_loss
            emit(scope_name, decides, "storm_cell", cell, step, n, -loss,
                 support.lower_nt, !supported_loss,
                 _v24_sprintf("loss=%+.4f blocks=%d", loss, support.n_blocks))
        end

        # (3) G3 on the same era
        g3 = get(g3_verdicts, (String(scope_name), "G3"), false)
        era_pass &= g3
        pooled = nothing
        for row in interval_rows
            (row.scope == String(scope_name) && row.variant == String(candidate) &&
             row.subset == "pooled" && row.model_step_hours == 0) || continue
            pooled = row
        end
        emit(scope_name, decides, "intervals_G3", "all", 0,
             pooled === nothing ? 0 : pooled.n, NaN, NaN, g3,
             pooled === nothing ? "no pooled interval row" :
             _v24_sprintf("coverage=%.4f width=%.3f score=%.3f", pooled.coverage,
                      pooled.mean_width_nt, pooled.mean_interval_score_nt))
        verdicts[String(scope_name)] = era_pass
        emit(scope_name, decides, "era_verdict", "all", -1, 0, NaN, NaN, era_pass,
             decides ? "counts toward the serve decision" :
             "disclosed only; the served product is partly in-sample here")
    end
    deciding = [scope for scope in String.(collect(V24_SERVE_RULE_ERAS))
                if haskey(verdicts, scope)]
    serve = !isempty(deciding) && all(verdicts[scope] for scope in deciding)
    emit("-", true, "serve_rule", "all", -1, length(deciding), NaN, NaN, serve,
         "G4 pending at integration; deciding eras " * join(deciding, "|"))
    return (rows=rows, verdicts=verdicts, serve=serve, deciding=deciding)
end

"""
    v24_decision(verdicts, eras) -> NamedTuple

Plan section 6 decision arithmetic over the gates this stage can evaluate.

G4 is the integrity and industrial gate: finiteness and the no-year-Y-in-any-fit
property are asserted here and by the test suite, but its availability, latency
and live-versus-offline identity requirements are properties of the serving
integration and cannot be established from the fold tables. The decision is
therefore reported as `SERVE_PENDING_G4` when G1-G3 pass on every scored scope,
which is the plan's `SERVE` branch with its remaining precondition named rather
than assumed; `SHADOW` when G1 fails on exactly one era or G2 fails anywhere;
and `NO_GO` otherwise.
"""
function v24_decision(verdicts, eras)
    scopes = [String(name) for name in keys(eras) if haskey(verdicts, (String(name), "G1"))]
    isempty(scopes) && error("V2.4 decision has no scored era")
    g1 = Dict(scope => verdicts[(scope, "G1")] for scope in scopes)
    g2 = Dict(scope => verdicts[(scope, "G2")] for scope in scopes)
    g3 = Dict(scope => verdicts[(scope, "G3")] for scope in scopes)
    era_scopes = [scope for scope in scopes if scope != "ALL"]
    era_g1_failures = count(scope -> !g1[scope], era_scopes)
    state = if all(values(g1)) && all(values(g2)) && all(values(g3))
        "SERVE_PENDING_G4"
    elseif !all(values(g2)) || era_g1_failures == 1
        "SHADOW"
    else
        "NO_GO"
    end
    failing = String[]
    for scope in scopes
        g1[scope] || push!(failing, "G1@$(scope)")
        g2[scope] || push!(failing, "G2@$(scope)")
        g3[scope] || push!(failing, "G3@$(scope)")
    end
    return (state=state, failing=failing, scopes=scopes, g1=g1, g2=g2, g3=g3)
end

# ---------------------------------------------------------------------------
# Deployment bundle
# ---------------------------------------------------------------------------

"Subdirectory of the run tree holding the deployable final-fold models."
const V24_DEPLOY_SUBDIR = "v2_4_deploy"

"""
    v24_write_deployment(outdir, bundle) -> NamedTuple

Persist the last fold's fitted objects as a deployable bundle.

The last fold is the one whose fits use every scored year before the end of the
study, so it — and only it — is the model a serving path would load. What a
serving path needs is exactly what this writes: the stack weights per step and
regime, the boosted residual with the feature order it was fitted on and the cap
it is clamped by, the conformal half-widths of the selected variant, and the
guard thresholds together with the variant the selection rule chose. Every file
is plain text apart from the boosted models, which go through the pinned EvoTrees
writer, and each one is checksummed into `deploy_manifest.csv`.

Nothing here is scoring evidence: the bundle is a copy of fitted state, and the
study's numbers come from the per-fold tables.
"""
function v24_write_deployment(outdir::AbstractString, bundle)
    directory = joinpath(outdir, V24_DEPLOY_SUBDIR)
    mkpath(directory)
    written = String[]
    record(path) = (push!(written, path); path)

    # Which fitted stack the selected variant's center comes from. The bundle ships
    # every fitted stack of the final fold, and `served_stack` names the one a
    # serving path must evaluate; shipping only that one would make a later
    # comparison impossible, shipping it unlabelled would make deployment guesswork.
    served_stack = bundle.selected_variant in (:v2_4e, :v2_4f) ? "L1e" :
                   (bundle.selected_variant in (:v2_4a_floor, :v2_4d) ? "L1a" : "L1")
    served_experts = served_stack == "L1e" ? V24_EXPERTS_TEN : V24_EXPERTS
    served_family = served_stack == "L1e" ? V24_SINDY_FAMILY_TEN : V24_SINDY_FAMILY
    stack_path = record(joinpath(directory, "stack_weights.csv"))
    stack_rows = v24_l1_weight_rows(bundle.l1_cells, bundle.fold_year, "L1")
    append!(stack_rows, v24_l1_weight_rows(bundle.l1_floor_cells, bundle.fold_year, "L1a"))
    append!(stack_rows, v24_l1_weight_rows(bundle.l1_ten_cells, bundle.fold_year, "L1e"))
    stack_rows = [merge(row, (served=(row.variant == served_stack),)) for row in stack_rows]
    CSV.write(stack_path, DataFrame(stack_rows))

    feature_path = record(joinpath(directory, "residual_features.csv"))
    CSV.write(feature_path, DataFrame(
        column_index=collect(1:V24_L2_FEATURE_COUNT),
        feature_name=v24_l2_feature_names(),
    ))
    cap_path = record(joinpath(directory, "residual_cap.csv"))
    CSV.write(cap_path, DataFrame(
        model_step_hours=collect(V24_STEPS),
        residual_cap_nt=[v24_residual_cap(step) for step in V24_STEPS],
    ))

    residual_files = String[]
    if bundle.l2_layer === nothing
        residual_form = "none"
    else
        residual_form = bundle.l2_layer.form
        if residual_form == "joint"
            path = record(joinpath(directory, "residual_model_joint.bson"))
            SolarSINDy.v23_save(bundle.l2_layer.models[0], path)
            push!(residual_files, basename(path))
        else
            for step in V24_STEPS
                path = record(joinpath(directory, "residual_model_step$(step).bson"))
                SolarSINDy.v23_save(bundle.l2_layer.models[step], path)
                push!(residual_files, basename(path))
            end
        end
    end

    conformal_path = record(joinpath(directory, "conformal.csv"))
    CSV.write(conformal_path, DataFrame(v24_conformal_rows(
        bundle.conformal, bundle.fold_year, String(bundle.selected_variant);
        note="deployment_bundle",
    )))

    guarded = bundle.selected_variant in (:v2_4c, :v2_4d, :v2_4f)
    static_guarded = bundle.selected_variant in (:v2_4d, :v2_4f)
    guard = Dict{String,Any}(
        "selected_variant" => String(bundle.selected_variant),
        "served_stack" => served_stack,
        "served_expert_order" => [String(e) for e in served_experts],
        "served_floor_group" => [String(served_experts[j]) for j in served_family],
        "guard_applied" => guarded,
        "guard_reference" => static_guarded ? String(V24_D_GUARD_REFERENCE) :
                             (bundle.selected_variant === :v2_4c ? "l1_center" : "none"),
        "guard_rule" => static_guarded ?
            "final = min($(served_stack) stack, static V2.2 stack) in a deepening cell" :
            (bundle.selected_variant === :v2_4c ?
             "final = min(L1 + residual, L1) in a deepening cell" : "no guard"),
        "center_source" => served_stack == "L1e" ?
                           "ten_expert_stack_with_family_floor" :
                           (served_stack == "L1a" ?
                            "nine_expert_stack_with_family_floor" :
                            (bundle.selected_variant === :v2_4a ? "l1_stack" :
                             "l1_stack_plus_boosted_residual")),
        "pool_target_embargo_hours" => V24_EMBARGO_HOURS,
        "pool_target_cutoff_utc" => string(bundle.pool_cutoff),
        "residual_used_by_selected_variant" =>
            bundle.selected_variant in (:v2_4b, :v2_4c),
        "deepening_rate_nt_per_h_strict_below" => V24_GUARD_RATE_NT_PER_H,
        "deepening_depth_nt_at_or_below_with_active_coupling" => V24_GUARD_DEPTH_NT,
        "residual_cap_nt" => Dict(string(step) => v24_residual_cap(step)
                                  for step in V24_STEPS),
        "residual_cap_rule" => "+/-(10 + 5h) nT",
        "residual_form" => residual_form,
        "residual_model_files" => residual_files,
        "residual_max_depth" => bundle.l2_layer === nothing ? 0 :
                                bundle.l2_layer.max_depth,
        "residual_nrounds" => bundle.l2_layer === nothing ? 0 : bundle.l2_layer.nrounds,
        "residual_eta" => V24_L2_ETA,
        "residual_min_weight" => V24_L2_MIN_WEIGHT,
        "residual_nbins" => V24_L2_NBINS,
        "expert_order" => [String(expert) for expert in V24_EXPERTS],
        "expert_order_ten" => [String(expert) for expert in V24_EXPERTS_TEN],
        "model_steps" => collect(V24_STEPS),
        "regimes" => [String(regime) for regime in (V24_POOLED_REGIME, V24_REGIMES...)],
        "conformal_coverage" => V24_COVERAGE,
        "disturbed_reporting_boundary_nt" => V24_DISTURBED_NT,
        "depth_bins" => [String(bin) for bin in V24_DEPTH_BINS],
        "depth_bin_moderate_nt_at_or_below" => V24_DEPTH_MODERATE_NT,
        "depth_bin_deep_nt_at_or_below" => V24_DEPTH_DEEP_NT,
        "l1_cell_fallback_chain" => ["regime_x_depth", "regime_pooled", "pooled"],
        "l1_min_cell_rows" => V24_MIN_CELL_ROWS,
        "residual_accepted_steps" => bundle.l2_layer === nothing ? Int[] :
                                     collect(bundle.l2_layer.accepted_steps),
        "sindy_family_floor_variant_only" => V24_SINDY_FLOOR,
        "fold_year" => bundle.fold_year,
        "l1_pool_years" => collect(bundle.l1_pool_years),
        "l2_pool_years" => collect(bundle.l2_pool_years),
        "conformal_pool_years" => collect(bundle.conformal_pool_years),
        "seed" => bundle.seed,
        "evotrees_version" => SolarSINDy.V23_GBM_EVOTREES_VERSION,
        "provenance" => "V2.4 rolling-origin study, last fold; fits use only years " *
                        "strictly before the fold year",
    )
    guard_path = record(joinpath(directory, "guard.json"))
    open(guard_path, "w") do io
        JSON3.pretty(io, guard)
        println(io)
    end

    manifest_path = joinpath(directory, "deploy_manifest.csv")
    CSV.write(manifest_path, DataFrame(
        file=[basename(path) for path in written],
        bytes=[Float64(filesize(path)) for path in written],
        sha256=[_v24_sha256_file(path) for path in written],
    ))
    return (directory=directory, files=written, manifest=manifest_path,
            residual_form=residual_form)
end

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

"""
    v24_learn_year_rows(data) -> DataFrame

Per-row artifact of one scored fold: the issue-time state, the four variant
centers with their conformal intervals, the matched-recipe V2.1 half-width, and
every comparator column copied through from Task A.
"""
function v24_learn_year_rows(data::V24YearData)
    n = length(data)
    frame = DataFrame(
        issue_time_utc=data.issue,
        model_step_hours=data.step,
        observation_dst_nt=data.obs,
        latest_dst_nt=data.latest,
        dst_delta_1h_nt=data.rate,
        coupling_active_mvm=data.coupling,
        fallback=data.fallback,
        usable=collect(data.usable),
        regime=String.(data.regime),
        depth_bin=[String(v24_depth_bin(data.latest[i])) for i in 1:n],
        l1_cell_regime=String.(data.cell_regime),
        l1_cell_depth=String.(data.cell_depth),
        l1_used_pooled_fallback=collect(data.used_pooled),
        deepening_cell=[v24_deepening(data.rate[i], data.coupling[i], data.latest[i])
                        for i in 1:n],
        innovation_ok=collect(data.innovation_ok),
        l2_applied=collect(data.l2_applied),
        l2_residual_raw_nt=data.residual_raw,
        l2_residual_capped_nt=data.residual,
    )
    for variant in V24_VARIANTS
        frame[!, Symbol(variant)] = data.centers[variant]
        half = get(data.half_widths, variant, nothing)
        half === nothing && continue
        frame[!, Symbol(variant, "_half_width_nt")] = half
        frame[!, Symbol(variant, "_lo_nt")] = data.centers[variant] .- half
        frame[!, Symbol(variant, "_hi_nt")] = data.centers[variant] .+ half
    end
    reference = get(data.half_widths, :served_v2_1, nothing)
    reference === nothing || (frame[!, :served_v2_1_half_width_nt] = reference)
    for column in (V24_MODEL_COLUMNS..., V24_LAT_COLUMN, :v2_3_shadow, V24_ORACLE_COLUMN)
        frame[!, Symbol(column)] = data.comparators[column]
    end
    return frame
end

"""
    run_v2_4_learn(; kwargs...) -> NamedTuple

Run the V2.4 learning and scoring stage over the available folds.

Keyword arguments exist so the test suite can exercise the whole pipeline on a
small fixture; the defaults are the preregistered protocol and must not be
overridden in a study run. `years` is the fold list, `eras` the scoring scopes,
`indir` the directory holding Task A's tables, `outdir` where this stage writes,
`l2_grid` the residual hyper-parameter grid, and `bootstrap_replicates` the
replicate count.
"""
function run_v2_4_learn(; years=V24_FOLD_YEARS, eras=V24_ERAS,
                        indir::AbstractString=V24_DIR, outdir::AbstractString=V24_DIR,
                        l2_grid=V24_L2_GRID,
                        bootstrap_replicates::Integer=SolarSINDy.V23_BOOTSTRAP_REPLICATES,
                        seed::Integer=V24_SEED, write_report::Bool=true)
    total_started = time()
    requested = collect(Int, years)
    isempty(requested) && error("V2.4 learning stage needs at least one fold year")
    seed_year = minimum(requested) - 1
    isfile(v24_oof_path(seed_year; dir=indir)) || error(
        "V2.4 learning stage needs the seed fold table " *
        v24_oof_path(seed_year; dir=indir),
    )
    available = v24_available_years(requested; dir=indir)
    isempty(available) && error(
        "V2.4 learning stage found no fold table for $(first(requested)) in $indir",
    )
    length(available) == length(requested) || _v24_log(_v24_sprintf(
        "  ! Task A has delivered %d of %d requested folds (%d..%d); scoring the " *
        "contiguous prefix", length(available), length(requested), first(available),
        last(available),
    ))
    mkpath(outdir)

    manifest = NamedTuple[]
    record!(kind, name, count, value) = push!(manifest, (
        entry_type=String(kind), name=String(name), count=Float64(count),
        value=String(value),
    ))

    _v24_log(_v24_sprintf("V2.4 learning stage: seed %d, folds %d..%d", seed_year,
                      first(available), last(available)))
    seed_path = v24_oof_path(seed_year; dir=indir)
    seed_data = v24_read_year(seed_year; dir=indir)
    record!("input_sha256", basename(seed_path), length(seed_data),
            _v24_sha256_file(seed_path))
    record!("fold", "seed_shadow_column", seed_year, seed_data.shadow_source)

    pool = V24YearData[seed_data]
    scored = V24YearData[]
    innovation_store = Dict{DateTime,Float64}()
    weight_rows = NamedTuple[]
    l2_rows = NamedTuple[]
    l2_acceptance_rows = NamedTuple[]
    conformal_rows = NamedTuple[]
    fold_rows = NamedTuple[]
    # Fitted state of whichever fold turns out to be the last one; the deployable
    # bundle is written from it once the selection rule has named a variant.
    final_state = nothing

    for year in available
        fold_started = time()
        data = v24_read_year(year; dir=indir)
        oof_path = v24_oof_path(year; dir=indir)
        record!("input_sha256", basename(oof_path), length(data),
                _v24_sha256_file(oof_path))
        fold_manifest = v24_manifest_path(year; dir=indir)
        isfile(fold_manifest) && record!("input_sha256", basename(fold_manifest), NaN,
                                         _v24_sha256_file(fold_manifest))

        # No fit may see a row of the scored year; the pool is the audit surface.
        for member in pool
            member.year < year ||
                error("V2.4 fold $year would fit on pool year $(member.year)")
        end
        for member in scored
            member.year < year || error(
                "V2.4 fold $year would fit its residual layer on pool year $(member.year)",
            )
        end

        # Amendment A3: every pool fit for this fold sees only rows whose target
        # clears the 168 h embargo, the same bound the rolling engine applies to its
        # training window.
        pool_cutoff = v24_pool_cutoff(year)
        l1_started = time()
        plain_cells = v24_fit_l1(pool; cutoff=pool_cutoff)
        floor_cells = v24_fit_l1(pool; floor_mass=V24_SINDY_FLOOR, cutoff=pool_cutoff)
        ten_cells = v24_fit_l1(pool; floor_mass=V24_SINDY_FLOOR, cutoff=pool_cutoff,
                               experts=V24_EXPERTS_TEN, family=V24_SINDY_FAMILY_TEN)
        v24_apply_l1!(data, plain_cells)
        v24_apply_l1!(data, floor_cells; target=:l1_floor)
        v24_apply_l1!(data, ten_cells; target=:l1_ten)
        l1_seconds = time() - l1_started
        append!(weight_rows, v24_l1_weight_rows(plain_cells, year, "L1"))
        append!(weight_rows, v24_l1_weight_rows(floor_cells, year, "L1a"))
        append!(weight_rows, v24_l1_weight_rows(ten_cells, year, "L1e"))

        # The fold's own one-step innovations are recorded before its innovation
        # lags are read: the one-step forecast issued at t-j uses weights fitted
        # on years < Y and matures at t-j+1, so at issue t it is observed history
        # even when t-j lies inside year Y. This is the "current year's already
        # scored earlier issues" clause of the specification, and it is causal
        # because nothing later than the issue hour enters.
        v24_record_innovations!(innovation_store, data)
        v24_fill_innovations!(data, innovation_store)

        l2_started = time()
        layer = nothing
        pool_rows = Tuple{V24YearData,Int}[]
        for member in scored, i in 1:length(member)
            (v24_l2_eligible(member, i) && v24_in_pool(member, i, pool_cutoff)) &&
                push!(pool_rows, (member, i))
        end
        if length(pool_rows) >= V24_L2_MIN_POOL_ROWS
            layer = v24_fit_l2(pool_rows; grid=l2_grid, seed=seed)
            v24_apply_l2!(data, layer)
        else
            _v24_log(_v24_sprintf(
                "  ! fold %d has %d eligible residual rows in its pool, fewer than the " *
                "%d needed; V2.4b and V2.4c equal V2.4a on this fold", year,
                length(pool_rows), V24_L2_MIN_POOL_ROWS,
            ))
        end
        l2_seconds = time() - l2_started
        if layer === nothing
            push!(l2_rows, (
                fold_year=year, available=false, form="none", max_depth=0, nrounds=0,
                inner_rule="none", inner_boundary="", inner_target_cutoff_utc="",
                n_pool_rows=length(pool_rows),
                n_inner_train=0, n_inner_validate=0, n_inner_embargoed=0,
                inner_rmse_nt=NaN,
                joint_inner_rmse_nt=NaN, per_step_inner_rmse_nt=NaN, selected=true,
                accepted_steps="",
            ))
            for step in V24_STEPS
                push!(l2_acceptance_rows, (
                    fold_year=year, model_step_hours=step, n_inner_validate=0,
                    rmse_identity_nt=NaN, rmse_residual_nt=NaN, gain_nt=NaN,
                    accepted=false, reason="no_residual_layer",
                ))
            end
        else
            append!(l2_acceptance_rows,
                    [merge((fold_year=year,), row) for row in layer.acceptance])
            for row in layer.trace
                push!(l2_rows, (
                    fold_year=year, available=true, form=row.form,
                    max_depth=row.max_depth, nrounds=row.nrounds,
                    inner_rule=layer.split.rule,
                    inner_boundary=string(layer.split.boundary),
                    inner_target_cutoff_utc=string(layer.split.cutoff),
                    n_pool_rows=layer.n_pool_rows, n_inner_train=layer.n_inner_train,
                    n_inner_validate=layer.n_inner_validate,
                    n_inner_embargoed=layer.n_inner_embargoed,
                    inner_rmse_nt=row.inner_rmse_nt,
                    joint_inner_rmse_nt=row.joint_inner_rmse_nt,
                    per_step_inner_rmse_nt=row.per_step_inner_rmse_nt,
                    selected=(row.max_depth == layer.max_depth &&
                              row.nrounds == layer.nrounds && row.form == layer.form),
                    accepted_steps=join(layer.accepted_steps, "|"),
                ))
            end
        end

        v24_build_centers!(data)

        conformal_started = time()
        # Calibration sources are years strictly before the fold. The first scored
        # fold has no predecessor with out-of-fold variant centers, so its
        # calibration uses the seed year with the fold's own L1 cells applied to
        # it. Those seed residuals are in-sample for a nine-weight convex
        # combination over thousands of rows per step, which shifts the residual
        # scale by order p/n; the alternative is no interval at all on the first
        # fold. The substitution is stamped into the artifact.
        seed_note = isempty(scored) ?
            "in_sample_seed_$(seed_year)_l1_residuals" : ""
        fold_strata = Dict{Symbol,Dict{Tuple{Int,Symbol},NamedTuple}}()
        for variant in V24_VARIANTS
            sources = if isempty(scored)
                cells = if variant in (:v2_4e, :v2_4f)
                    ten_cells
                elseif variant in (:v2_4a_floor, :v2_4d)
                    floor_cells
                else
                    plain_cells
                end
                Tuple{V24YearData,Vector{Float64}}[
                    (seed_data, v24_l1_centers(seed_data, cells).centers),
                ]
            else
                Tuple{V24YearData,Vector{Float64}}[
                    (member, member.centers[variant]) for member in scored
                ]
            end
            strata = v24_fit_conformal(sources; cutoff=pool_cutoff)
            fold_strata[variant] = strata
            data.half_widths[variant] = v24_apply_conformal(data, strata)
            append!(conformal_rows,
                    v24_conformal_rows(strata, year, String(variant); note=seed_note))
        end
        # Matched-recipe V2.1 reference intervals: identical strata, pool and
        # target coverage applied to the served center. The deployed V2.1 sidecar
        # is calibrated on a fixed 2020-2022 window that overlaps this study's
        # scored years, so reusing it would break the rolling rule; recalibrating
        # V2.1 under the same rolling recipe isolates the effect of the center,
        # which is the quantity G3 compares.
        reference_pool = isempty(scored) ? V24YearData[seed_data] : scored
        reference_strata = v24_fit_conformal(Tuple{V24YearData,Vector{Float64}}[
            (member, member.comparators[:served_v2_1]) for member in reference_pool
        ]; cutoff=pool_cutoff)
        data.half_widths[:served_v2_1] = v24_apply_conformal(data, reference_strata)
        append!(conformal_rows, v24_conformal_rows(
            reference_strata, year, "served_v2_1";
            note=isempty(scored) ? "seed_$(seed_year)_served_residuals" : "",
        ))
        conformal_seconds = time() - conformal_started

        CSV.write(joinpath(outdir, "learn_year_$(year).csv"), v24_learn_year_rows(data))

        final_state = (fold_year=year, l1_cells=plain_cells, l1_floor_cells=floor_cells,
                       l1_ten_cells=ten_cells, pool_cutoff=pool_cutoff,
                       l2_layer=layer, strata=fold_strata, seed=Int(seed),
                       l1_pool_years=[member.year for member in pool],
                       l2_pool_years=[member.year for member in scored],
                       conformal_pool_years=(isempty(scored) ? [seed_year] :
                                             [member.year for member in scored]))
        push!(scored, data)
        push!(pool, data)
        fold_seconds = time() - fold_started
        push!(fold_rows, (
            fold_year=year, n_rows=length(data), n_usable=count(data.usable),
            n_fallback=count(data.fallback),
            n_feature_incomplete=count(i -> !data.fallback[i] &&
                                       !all(isfinite, view(data.features, i, :)),
                                       1:length(data)),
            n_innovation_ok=count(data.innovation_ok),
            n_l2_applied=count(data.l2_applied),
            pool_target_cutoff_utc=string(pool_cutoff),
            # Rows of the *earlier* years that the target embargo removed from this
            # fold's pool. The current year is already in `pool` at this point and
            # is excluded by construction, not by the embargo, so counting it here
            # would report the fold's own size instead of the rule's effect.
            n_prior_pool_rows_embargoed=sum(
                count(i -> !v24_in_pool(member, i, pool_cutoff), 1:length(member))
                for member in pool if member.year < year; init=0),
            l2_accepted_steps=(layer === nothing ? "" :
                               join(layer.accepted_steps, "|")),
            n_deep_rows=count(i -> v24_depth_bin(data.latest[i]) === :deep,
                              1:length(data)),
            n_moderate_rows=count(i -> v24_depth_bin(data.latest[i]) === :moderate,
                                  1:length(data)),
            n_l1_pooled_fallback=count(data.used_pooled),
            n_deepening=count(i -> v24_deepening(data.rate[i], data.coupling[i],
                                                 data.latest[i]), 1:length(data)),
            shadow_column=data.shadow_source,
            lat_equals_shadow=(data.comparators[V24_LAT_COLUMN] ==
                               data.comparators[:v2_3_shadow]),
            l1_seconds=l1_seconds,
            l2_seconds=l2_seconds, conformal_seconds=conformal_seconds,
            fold_seconds=fold_seconds,
        ))
        _v24_log(_v24_sprintf(
            "  fold %d: %d rows (%d usable, %d fallback, %d residual-corrected), " *
            "L1 %.1f s, L2 %.1f s, conformal %.1f s, total %.1f s", year, length(data),
            count(data.usable), count(data.fallback), count(data.l2_applied),
            l1_seconds, l2_seconds, conformal_seconds, fold_seconds,
        ))
    end

    scored_years = [data.year for data in scored]
    active_eras = v24_active_eras(eras, scored_years)

    scoring_started = time()
    summary_rows = v24_summary_rows(scored, active_eras)
    cell_rows = v24_cell_rows(scored, active_eras)
    interval_rows = v24_interval_rows(scored, active_eras, (V24_VARIANTS..., :served_v2_1))
    selection = v24_select_variant(scored, cell_rows, active_eras;
                                   replicates=bootstrap_replicates)
    bootstrap_rows = v24_bootstrap_rows(
        scored, active_eras, selection.selected; replicates=bootstrap_replicates,
    )
    gates = v24_gate_rows(summary_rows, cell_rows, bootstrap_rows, interval_rows,
                          active_eras, selection.selected; years=scored,
                          replicates=bootstrap_replicates)
    decision = v24_decision(gates.verdicts, active_eras)
    # Amendment A3: the gates above are the served candidate's. Every variant is
    # gated as well, so a later reader can see what the gates would have said for a
    # different served choice instead of transferring one variant's verdict.
    variant_gate_rows = NamedTuple[]
    variant_bootstrap_rows = NamedTuple[]
    for variant in V24_VARIANTS
        boot = variant === selection.selected ? bootstrap_rows :
            v24_bootstrap_rows(scored, active_eras, variant;
                               replicates=bootstrap_replicates)
        variant === selection.selected || append!(variant_bootstrap_rows, boot)
        variant_gates = variant === selection.selected ? gates :
            v24_gate_rows(summary_rows, cell_rows, boot, interval_rows, active_eras,
                          variant; years=scored, replicates=bootstrap_replicates)
        for row in variant_gates.rows
            push!(variant_gate_rows, merge((variant=String(variant),
                                            served_candidate=(variant ===
                                                              selection.selected)), row))
        end
    end
    serve_rule = v24_serve_rule_rows(scored, active_eras, selection.selected,
                                     interval_rows, gates.verdicts;
                                     replicates=bootstrap_replicates)
    scoring_seconds = time() - scoring_started

    CSV.write(joinpath(outdir, "v2_4_l1_weights.csv"), DataFrame(weight_rows))
    CSV.write(joinpath(outdir, "v2_4_l2_selection.csv"), DataFrame(l2_rows))
    CSV.write(joinpath(outdir, "v2_4_l2_acceptance.csv"), DataFrame(l2_acceptance_rows))
    CSV.write(joinpath(outdir, "v2_4_conformal.csv"), DataFrame(conformal_rows))
    CSV.write(joinpath(outdir, "v2_4_folds.csv"), DataFrame(fold_rows))
    CSV.write(joinpath(outdir, "v2_4_summary.csv"), DataFrame(summary_rows))
    CSV.write(joinpath(outdir, "v2_4_cells.csv"), DataFrame(cell_rows))
    CSV.write(joinpath(outdir, "v2_4_intervals.csv"), DataFrame(interval_rows))
    CSV.write(joinpath(outdir, "v2_4_selection.csv"), DataFrame(selection.trace))
    isempty(selection.guard_detail) ||
        CSV.write(joinpath(outdir, "v2_4_selection_guards.csv"),
                  DataFrame(selection.guard_detail))
    CSV.write(joinpath(outdir, "v2_4_bootstrap.csv"), DataFrame(bootstrap_rows))
    isempty(variant_bootstrap_rows) ||
        CSV.write(joinpath(outdir, "v2_4_bootstrap_by_variant.csv"),
                  DataFrame(variant_bootstrap_rows))
    CSV.write(joinpath(outdir, "v2_4_gates.csv"), DataFrame(gates.rows))
    CSV.write(joinpath(outdir, "v2_4_gates_by_variant.csv"), DataFrame(variant_gate_rows))
    CSV.write(joinpath(outdir, "v2_4_serve_rule.csv"), DataFrame(serve_rule.rows))

    decision_rows = NamedTuple[]
    for scope in decision.scopes, gate in ("G1", "G2", "G3")
        push!(decision_rows, (
            item=gate, scope=scope, status=string(gates.verdicts[(scope, gate)]), value="",
        ))
    end
    push!(decision_rows, (
        item="G4", scope="-", status="PENDING",
        value="integrity and industrial gate: finiteness and the no-scored-year-in-any-fit " *
              "property are asserted by this stage and its tests; availability, latency " *
              "and live-versus-offline identity belong to the serving integration",
    ))
    push!(decision_rows, (
        item="selected_variant", scope="E1", status=String(selection.selected),
        value=join([_v24_sprintf("%s=%.4f", k, v)
                    for (k, v) in sort(collect(selection.scores); by=first)], "|"),
    ))
    push!(decision_rows, (
        item="decision", scope=join(decision.scopes, "|"), status=decision.state,
        value=isempty(decision.failing) ? "no gate failing" : join(decision.failing, "|"),
    ))
    push!(decision_rows, (
        item="serve_rule", scope=join(serve_rule.deciding, "|"),
        status=serve_rule.serve ? "SERVE_ELIGIBLE_PENDING_G4" : "NOT_SERVE_ELIGIBLE",
        value="Amendment A3 rule versus the served static V2.2 stack: pooled gain with a " *
              "positive lower bound at every step, no bootstrap-supported storm-cell " *
              "loss, and G3, on " * join(serve_rule.deciding, " and ") *
              "; E1 disclosed only (the served product is partly in-sample there)",
    ))
    for (scope, verdict) in sort(collect(serve_rule.verdicts); by=first)
        push!(decision_rows, (
            item="serve_rule_era", scope=scope, status=string(verdict),
            value=(Symbol(scope) in V24_SERVE_RULE_ERAS ? "deciding" : "disclosed_only"),
        ))
    end
    push!(decision_rows, (
        item="data_boundary", scope="-", status="DISCLOSED",
        value="rolling-origin evidence only; the 2020-2025 partition was already scored " *
              "once by the V2.3 confirmatory run, so no untouched multi-year window " *
              "remains (research plan section 1)",
    ))
    CSV.write(joinpath(outdir, "v2_4_decision.csv"), DataFrame(decision_rows))

    final_state === nothing && error("V2.4 learning stage scored no fold")
    deployment = v24_write_deployment(outdir, (
        fold_year=final_state.fold_year, l1_cells=final_state.l1_cells,
        l1_floor_cells=final_state.l1_floor_cells,
        l1_ten_cells=final_state.l1_ten_cells, pool_cutoff=final_state.pool_cutoff,
        l2_layer=final_state.l2_layer,
        conformal=final_state.strata[selection.selected],
        selected_variant=selection.selected, seed=final_state.seed,
        l1_pool_years=final_state.l1_pool_years,
        l2_pool_years=final_state.l2_pool_years,
        conformal_pool_years=final_state.conformal_pool_years,
    ))

    write_report && _v24_write_report(
        joinpath(outdir, "v2_4_report.md"), summary_rows, cell_rows, bootstrap_rows,
        interval_rows, gates.rows, decision_rows, selection, active_eras, scored_years,
        serve_rule.rows,
    )

    record!("environment", "julia_threads", Threads.nthreads(), "")
    record!("environment", "evotrees_version", NaN,
            SolarSINDy.V23_GBM_EVOTREES_VERSION)
    record!("input_sha256", "v2_4_learn.jl", NaN, _v24_sha256_file(@__FILE__))
    record!("protocol", "experts_ten", V24_EXPERT_TEN_COUNT,
            join(String.(V24_EXPERTS_TEN), "|"))
    record!("protocol", "pool_target_embargo_hours", V24_EMBARGO_HOURS,
            "every out-of-fold pool fit; Amendment A3")
    record!("protocol", "inner_target_embargo_hours", V24_EMBARGO_HOURS,
            "L2 inner train/validate split, same target rule as the pool cutoff; " *
            "per-fold cutoff and dropped-row count in v2_4_l2_selection.csv " *
            "(inner_target_cutoff_utc, n_inner_embargoed)")
    record!("protocol", "serve_rule", length(V24_SERVE_RULE_ERAS),
            "deciding eras " * join(String.(V24_SERVE_RULE_ERAS), "|") *
            "; reference " * String(V24_SERVE_REFERENCE))
    record!("protocol", "experts", V24_EXPERT_COUNT, join(String.(V24_EXPERTS), "|"))
    record!("protocol", "gated_comparators", length(V24_GATED_COMPARATORS),
            join(String.(V24_GATED_COMPARATORS), "|"))
    record!("protocol", "folds_scored", length(scored_years),
            join(string.(scored_years), "|"))
    record!("protocol", "folds_requested", length(requested),
            join(string.(requested), "|"))
    record!("protocol", "eras", length(keys(active_eras)),
            join([_v24_sprintf("%s=%d..%d", k, first(getproperty(active_eras, k)),
                           last(getproperty(active_eras, k)))
                  for k in keys(active_eras)], "|"))
    record!("protocol", "l2_grid", length(l2_grid),
            join([_v24_sprintf("d%d_r%d", d, r) for (d, r) in l2_grid], "|"))
    record!("protocol", "bootstrap_replicates", bootstrap_replicates,
            _v24_sprintf("seed=%d block=%dh alpha=%.2f", SolarSINDy.V23_BOOTSTRAP_SEED,
                     SolarSINDy.V23_BOOTSTRAP_BLOCK_HOURS, V24_ALPHA))
    record!("protocol", "seed", seed, "")
    record!("protocol", "l2_feature_count", V24_L2_FEATURE_COUNT,
            join(v24_l2_feature_names(), "|"))
    record!("protocol", "amendment", 3,
            "A3: out-of-fold pool fits target-embargoed by " *
            string(V24_EMBARGO_HOURS) * " h, the L2 inner train/validate split included; " *
            "expert E10 = static_v2_2 (ten-expert stack " *
            "with the floor over served+frozen+t1r+static); variants v2_4e and v2_4f; " *
            "selection set " * join(String.(V24_SELECTABLE_VARIANTS), "|") * "; gates " *
            "evaluated for the served candidate and, separately, for every variant; " *
            "operational serve rule versus static_v2_2 on ALL and E2")
    record!("protocol", "amendment", 2,
            "A2: variant v2_4d = L1a stack guarded against the static V2.2 stack in " *
            "deepening cells; G1 effect size is headroom-aware (non-inferiority where " *
            "the realized-driver-oracle headroom is below " *
            _v24_sprintf("%.2f nT", V24_G1_HEADROOM_NT) * ", superiority elsewhere, the " *
            "margin reported not gated); a G2 cell loss counts only with a positive " *
            "one-sided 95% lower bound on that cell's rows")
    record!("protocol", "amendment", 1,
            "A1: L1 cells regime x depth bin with regime-pooled then pooled fallback; " *
            "L2 accepted per step on positive inner-validation gain; conformal strata " *
            "step x depth bin with pooled fallback; selection over all four variants on " *
            "E1 by mean pooled RMSE over steps " *
            join(V24_SELECTION_STEPS, "/") * "; gates unchanged")
    record!("protocol", "depth_bins", length(V24_DEPTH_BINS),
            _v24_sprintf("moderate<=%.0f;deep<=%.0f", V24_DEPTH_MODERATE_NT,
                     V24_DEPTH_DEEP_NT))
    record!("protocol", "selectable_variants", length(V24_SELECTABLE_VARIANTS),
            join(String.(V24_SELECTABLE_VARIANTS), "|"))
    record!("selection", "variant", NaN, String(selection.selected))
    record!("selection", "guards_all_failed", NaN, string(selection.guards_all_failed))
    record!("decision", "state", NaN, decision.state)
    record!("deployment", "final_fold", final_state.fold_year,
            "$(V24_DEPLOY_SUBDIR);residual=$(deployment.residual_form)")
    for path in vcat(deployment.files, [deployment.manifest])
        record!("output_sha256", joinpath(V24_DEPLOY_SUBDIR, basename(path)),
                filesize(path), _v24_sha256_file(path))
    end
    record!("seconds", "scoring", scoring_seconds, "")
    record!("seconds", "total", time() - total_started, "")
    manifest_out = joinpath(outdir, "v2_4_learn_manifest.csv")
    for file in sort(readdir(outdir))
        full = joinpath(outdir, file)
        (isfile(full) && full != abspath(manifest_out) &&
            (startswith(file, "v2_4_") || startswith(file, "learn_year_"))) || continue
        record!("output_sha256", file, filesize(full), _v24_sha256_file(full))
    end
    CSV.write(manifest_out, DataFrame(manifest))

    _v24_log(_v24_sprintf(
        "V2.4 learning stage complete in %.1f s -> %s (selected %s, decision %s)",
        time() - total_started, outdir, selection.selected, decision.state,
    ))
    return (selected=selection.selected, selection=selection, decision=decision,
            gates=gates, variant_gates=variant_gate_rows, serve_rule=serve_rule,
            summary=summary_rows, cells=cell_rows,
            bootstrap=bootstrap_rows, intervals=interval_rows, weights=weight_rows,
            l2=l2_rows, conformal=conformal_rows, folds=fold_rows, scored=scored,
            years=scored_years, outdir=outdir, deployment=deployment)
end

function _v24_write_report(path::AbstractString, summary_rows, cell_rows, bootstrap_rows,
                           interval_rows, gate_rows, decision_rows, selection, eras,
                           scored_years, serve_rule_rows=NamedTuple[])
    open(path, "w") do io
        println(io, "# Operational V2.4 rolling-origin scores\n")
        println(io, "Folds scored: ", join(string.(scored_years), ", "), ".\n")
        println(io, "Eras: ", join([_v24_sprintf("%s = %d-%d", k, first(getproperty(eras, k)),
                                             last(getproperty(eras, k)))
                                    for k in keys(eras)], "; "), ".\n")
        println(io, "## Pooled RMSE by model and step\n")
        models = unique([row.model for row in summary_rows])
        for scope in keys(eras)
            println(io, "### ", String(scope), "\n")
            header = vcat(["model"], [_v24_sprintf("%d h", s) for s in V24_STEPS], ["n (1 h)"])
            table = Vector{String}[]
            for model in models
                cells = String[model]
                for step in V24_STEPS
                    hit = [row for row in summary_rows if row.scope == String(scope) &&
                           row.model == model && row.model_step_hours == step]
                    push!(cells, isempty(hit) ? "-" : _v24_fmt(first(hit).rmse_nt))
                end
                hit = [row for row in summary_rows if row.scope == String(scope) &&
                       row.model == model && row.model_step_hours == first(V24_STEPS)]
                push!(cells, isempty(hit) ? "-" : string(first(hit).n))
                push!(table, cells)
            end
            v24_markdown_table(io, header, table)
        end
        println(io, "## Selection trace (plan section 4, Amendment A1)\n")
        v24_markdown_table(io,
            ["variant", "era", "steps", "mean step RMSE [nT]", "pooled RMSE [nT]", "n",
             "guards", "failures"],
            [[row.variant, row.era, row.selection_steps,
              _v24_fmt(row.mean_step_rmse_nt), _v24_fmt(row.pooled_rmse_nt),
              string(row.n), string(row.guards_pass), row.guard_failures]
             for row in selection.trace])
        println(io, "Selected variant: `", String(selection.selected), "`.\n")
        println(io, "## Bootstrap against every gated comparator\n")
        v24_markdown_table(io,
            ["scope", "comparator", "step", "n", "RMSE cand.", "RMSE comp.", "gain",
             "lower", "Holm p"],
            [[row.scope, row.comparator, string(row.model_step_hours), string(row.n),
              _v24_fmt(row.rmse_candidate_nt), _v24_fmt(row.rmse_comparator_nt),
              _v24_fmt(row.gain_nt), _v24_fmt(row.lower_nt),
              _v24_fmt(row.holm_p; digits=4)]
             for row in bootstrap_rows])
        println(io, "## Storm cells\n")
        v24_markdown_table(io,
            ["scope", "cell", "model", "step", "n", "RMSE [nT]", "bias [nT]"],
            [[row.scope, row.cell, row.model, string(row.model_step_hours), string(row.n),
              _v24_fmt(row.rmse_nt), _v24_fmt(row.bias_nt)]
             for row in cell_rows if Symbol(row.cell) in V24_G2_CELLS ||
                 Symbol(row.cell) === V24_G2_INTENSE_CELL])
        println(io, "## Intervals\n")
        v24_markdown_table(io,
            ["scope", "variant", "subset", "step", "n", "coverage", "mean width [nT]",
             "interval score [nT]"],
            [[row.scope, row.variant, row.subset, string(row.model_step_hours),
              string(row.n), _v24_fmt(row.coverage; digits=4),
              _v24_fmt(row.mean_width_nt), _v24_fmt(row.mean_interval_score_nt)]
             for row in interval_rows])
        println(io, "## Gates\n")
        v24_markdown_table(io,
            ["gate", "scope", "family", "step", "comparator", "requirement", "observed",
             "pass"],
            [[row.gate, row.scope, row.family, string(row.model_step_hours),
              row.comparator, row.requirement, row.observed, string(row.pass)]
             for row in gate_rows])
        println(io, "## Operational serve rule versus the served static stack\n")
        v24_markdown_table(io,
            ["scope", "decides", "check", "cell", "step", "n", "gain [nT]", "lower [nT]",
             "pass"],
            [[row.scope, string(row.decides), row.check, row.cell,
              string(row.model_step_hours), string(row.n), _v24_fmt(row.gain_nt),
              _v24_fmt(row.lower_nt), string(row.pass)] for row in serve_rule_rows])
        println(io, "## Decision\n")
        v24_markdown_table(io, ["item", "scope", "status", "value"],
            [[row.item, row.scope, row.status, row.value] for row in decision_rows])
    end
    return path
end

"Extract `--flag=value` from a command line, or `nothing` when absent."
function _v24_flag(arguments, flag::AbstractString)
    prefix = "$(flag)="
    for argument in arguments
        startswith(argument, prefix) || continue
        value = String(argument[(length(prefix) + 1):end])
        isempty(strip(value)) && error("$flag needs a non-empty value")
        return value
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    let fixture = _v24_flag(ARGS, "--fixture")
        if fixture === nothing
            run_v2_4_learn()
        else
            run_v2_4_learn(; indir=fixture, outdir=fixture)
        end
    end
end
