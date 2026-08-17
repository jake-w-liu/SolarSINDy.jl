#!/usr/bin/env julia

# v2_4_rolling.jl — Operational V2.4 rolling-origin engine (Task A of
# `V2_4_IMPLEMENTATION_SPEC.md`, protocol in `V2_4_RESEARCH_PLAN.md`).
#
# What this file produces and why. The V2.4 study needs one honest per-row table
# per calendar year in which *every* model — the nine super-learner experts, the
# comparators and the noncausal ceiling — was fitted only on data that precedes
# that year by at least the 168 h target embargo. Section 1 of the plan is
# explicit that no untouched multi-year window is left, so the rolling-origin
# refit is the evidence; a single fit reused across years would not be.
# Everything downstream (the stack, the boosted residual, the guard, the
# conformal intervals, the gates) reads only these files, which is why the
# schema lives in `v2_4_contract.jl` and is validated on write and on reuse.
#
# What is reused, deliberately. The V2.3 machinery already defines the anchor
# view of the base table, the multi-step rollout, the analog retrieval, the
# refit ridge correction (Amendment A1), the error layers, the lead-aware
# composition, the cells and the metrics. This runner includes
# `v2_3_confirmatory.jl` — which pulls in `v2_3_t1r.jl` → `v2_3_common.jl` →
# the kernel and base-table chain — and calls those functions rather than
# restating them, so a V2.4 expert and its V2.3 ancestor differ in *when* they
# were fitted and in nothing else. The include is definition-only: the
# confirmatory runner executes only under `abspath(PROGRAM_FILE) == @__FILE__`.
#
# Fold rule (plan section 3). For test year Y the training window is every
# base-table anchor whose maximum target `issue + 7 h` does not exceed
# `Y-01-01T00 − 168 h`; the scored rows are every anchor issued in year Y.
# DEV and TEST partitions are both admissible because this is a rolling study,
# and the base table's `embargo` partition (2019-12-24T17 … 2020-01-07T23) is
# dropped, so 2019 ends and 2020 begins where the partition labels end and
# begin. Fold 2013 seeds the out-of-fold history and belongs to no era.
#
# Run from the package root:
#   JULIA_NUM_THREADS=8 julia --startup-file=no --project=. \
#       validation/operational/v2_4_rolling.jl --smoke
#   JULIA_NUM_THREADS=8 julia --startup-file=no --project=. \
#       validation/operational/v2_4_rolling.jl
#   … --years=2015,2016      score a subset of folds into the study tree
#   … --reuse-stale-folds=<reason>   consume persisted folds whose signature no
#                                    longer matches, recording the reason

include(joinpath(@__DIR__, "v2_3_confirmatory.jl"))
include(joinpath(@__DIR__, "v2_4_contract.jl"))

V24_OUTPUT_ROOT == OPERATIONAL_OUTPUT_DIR || error(
    "the V2.4 contract resolves the operational output root to $(V24_OUTPUT_ROOT) but the " *
    "replay chain resolves it to $(OPERATIONAL_OUTPUT_DIR); the fold files would be written " *
    "outside the tree every other artifact lives in",
)
Tuple(V24_MODEL_STEPS) == V23_MODEL_STEPS || error(
    "the V2.4 contract carries model steps $(V24_MODEL_STEPS) but the V2.3 base table carries " *
    "$(V23_MODEL_STEPS)",
)

# ---------------------------------------------------------------------------
# Preregistered configuration of the rolling engine
# ---------------------------------------------------------------------------

"Analog ensemble of expert E3 (plan section 4): K = 25, magnetic weights, S = off."
const V24_ANALOG_K = 25
const V24_ANALOG_WEIGHT_SET = :magnetic
const V24_ANALOG_SAFEGUARDS = false
const V24_ANALOG_FAMILY = "T1"

"""
    V24DirectConfig

One grid point of expert E8, the tuned direct increment-GBM. `nbins` is part of
the identity because the V2.3 audit showed the histogram resolution is what lets
a boosted learner resolve the deep tail of `dst0`, so the plan's grid varies it
on the same footing as depth and rounds.
"""
struct V24DirectConfig
    depth::Int
    rounds::Int
    nbins::Int
end

v24_direct_label(config::V24DirectConfig) =
    "d$(config.depth)_r$(config.rounds)_nb$(config.nbins)"

"""
    V24_DIRECT_GRID

Preregistered direct-GBM grid of plan section 4: depth {4, 6} × rounds
{200, 400} × nbins {64, 255}, the full eight-point product. The strongest
data-driven comparator must be tuned over the grid the plan promised, not over a
subset of it: an inner search that never offers 64 bins to depth 4 would let the
candidate win a comparison the comparator was not allowed to enter.
"""
const V24_DIRECT_GRID = (
    V24DirectConfig(4, 200, 64), V24DirectConfig(4, 200, 255),
    V24DirectConfig(4, 400, 64), V24DirectConfig(4, 400, 255),
    V24DirectConfig(6, 200, 64), V24DirectConfig(6, 200, 255),
    V24DirectConfig(6, 400, 64), V24DirectConfig(6, 400, 255),
)

"""
    V24_DIRECT_SELECTION

Which model steps inherit which inner-validation decision: the step-1 winner is
applied to steps {1, 2} and the step-6 winner to steps {3, 4, 6, 7}. Selecting
per step at every step would spend six inner searches per fold on a decision the
spec fixes at two; selecting once for all six would hide that the short and long
leads prefer different capacities.
"""
const V24_DIRECT_SELECTION = ((1, (1, 2)), (6, (3, 4, 6, 7)))

"Length of the inner validation window at the end of a training window (months)."
const V24_DIRECT_INNER_MONTHS = 24

"Ridge penalties and boosted setting the per-fold error layers are adjudicated over."
const V24_E1_PARAMS = collect(V23_E1_LAMBDA_GRID)
const V24_E2_PARAMS = [(V23_E2_DEPTH, V23_E2_ROUNDS)]

"Lead-aware composition weight vector that reproduces the tail center exactly."
const V24_LAT_IDENTITY_WEIGHTS = ones(Float64, V23_STEP_COUNT)

"Largest absolute deviation tolerated by the frozen-feature oracle of a fold (nT)."
const V24_T1R_ORACLE_ATOL_NT = 1e-9

# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

"""
    V24Plan

Scale of a run. Only the smoke path reduces anything, and
[`v24_assert_full_plan`](@ref) fails closed if a study run would.
`max_anchors` caps the *scored* anchors of a fold and never the training window,
so a smoke fold still searches the archive the study fold would search — that is
what makes the smoke identity against the archived V2.3 out-of-fold core
meaningful.
"""
struct V24Plan
    label::String
    outdir::String
    smoke::Bool
    years::Vector{Int}
    max_anchors::Int                     # 0 = every anchor of the year
    direct_grid::Vector{V24DirectConfig}
    direct_max_train::Int                # 0 = the whole training window
    seed_max_anchors::Int                # 0 = the whole training window
    e_layers::Bool
end

"Full preregistered rolling plan: 13 folds, full grid, every anchor."
v24_full_plan(; years=collect(V24_FOLD_YEARS)) = V24Plan(
    "v2_4_rolling", V24_ROLLING_DIR, false, collect(Int, years), 0,
    collect(V24_DIRECT_GRID), 0, 0, true,
)

"""
    v24_smoke_plan(; year, max_anchors, …)

Reduced path: one fold, capped queries, one boosted grid point, a capped
in-sample seed fit and no error layers. It exercises every stage on a few
thousand anchors and writes to its own tree; nothing it produces is an input to
the study.
"""
v24_smoke_plan(; year::Integer=2015, max_anchors::Integer=2000,
               direct_grid=[V24DirectConfig(4, 200, 64)],
               direct_max_train::Integer=6000, seed_max_anchors::Integer=3000) = V24Plan(
    "v2_4_rolling_smoke", V24_ROLLING_SMOKE_DIR, true, [Int(year)], Int(max_anchors),
    collect(V24DirectConfig, direct_grid), Int(direct_max_train), Int(seed_max_anchors),
    false,
)

"""
    v24_assert_full_plan(plan) -> plan

Refuse a study run that would trim the protocol: a reduced boosted grid, capped
anchors, a capped seed fit or disabled error layers all change what the fold
files mean, and the files carry no field that would make the reduction visible
to Task B.
"""
function v24_assert_full_plan(plan::V24Plan)
    plan.smoke && return plan
    Tuple(plan.direct_grid) == V24_DIRECT_GRID || error(
        "run plan $(plan.label) trims the preregistered direct-GBM grid",
    )
    plan.max_anchors == 0 || error("run plan $(plan.label) caps the scored anchors")
    plan.direct_max_train == 0 || error(
        "run plan $(plan.label) caps the direct-GBM training window",
    )
    plan.seed_max_anchors == 0 || error("run plan $(plan.label) caps the seed fit")
    plan.e_layers || error("run plan $(plan.label) disables the V2.3 shadow error layers")
    issubset(Set(plan.years), Set(V24_FOLD_YEARS)) || error(
        "run plan $(plan.label) asks for years outside $(V24_FOLD_YEARS)",
    )
    return plan
end

"Context plan handed to `v23_build_context`; it only records scale in the manifest."
v24_context_plan(plan::V24Plan) = V23RunPlan(
    plan.label, plan.outdir, plan.smoke, copy(plan.years), plan.max_anchors,
    [V24_ANALOG_WEIGHT_SET], [V24_ANALOG_K], [(4, 200)], [(4, 200)],
)

# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

"""
    V24Fold

One rolling-origin fold: the training window, the scored anchors of the test
year, and the cutoff that separates them. `cutoff` is the latest target a
training anchor may have, not the latest issue, because the embargo of plan
section 3 is stated on targets.
"""
struct V24Fold
    year::Int
    cutoff::DateTime
    train::Vector{Int}
    query::Vector{Int}
end

"""
    v24_fold_cutoff(test_year; embargo_hours) -> DateTime

Latest target a fold's training window may contain: `Y-01-01T00 − 168 h`.
"""
v24_fold_cutoff(test_year::Integer; embargo_hours::Integer=V23_BASE_EMBARGO_HOURS) =
    DateTime(Int(test_year), 1, 1) - Hour(Int(embargo_hours))

"""
    v24_fold_rows(issues, test_year; embargo_hours, max_step) -> (train, query, cutoff)

Row indices of one fold, from issue times alone. Separated from the context so
the fold rule can be checked against an explicit list of issue times rather than
against the runner's own output: `train` holds every index whose maximum target
`issue + max_step` does not exceed the cutoff, and `query` holds every index
issued in the test year.
"""
function v24_fold_rows(issues::AbstractVector{DateTime}, test_year::Integer;
                       embargo_hours::Integer=V23_BASE_EMBARGO_HOURS,
                       max_step::Integer=V23_MAX_STEP)
    cutoff = v24_fold_cutoff(test_year; embargo_hours=embargo_hours)
    train = [i for i in eachindex(issues) if issues[i] + Hour(Int(max_step)) <= cutoff]
    query = [i for i in eachindex(issues) if Dates.year(issues[i]) == Int(test_year)]
    return (train=train, query=sort(query; by=i -> issues[i]), cutoff=cutoff)
end

"""
    v24_folds(ctx, plan) -> Vector{V24Fold}

Every fold of the run, with the embargo verified on targets by the confirmatory
runner's own check. A fold with an empty training window or no scored anchor is
an error: the plan names the years, so a missing one is a broken base table and
not a fold to skip quietly.
"""
function v24_folds(ctx::V23Context, plan::V24Plan)
    issues = ctx.anchors.issue
    folds = V24Fold[]
    for test_year in sort(plan.years)
        rows = v24_fold_rows(issues, test_year)
        isempty(rows.train) && error("fold $(test_year) has an empty training window")
        isempty(rows.query) && error("fold $(test_year) has no scored anchor")
        query = rows.query
        if plan.max_anchors > 0 && length(query) > plan.max_anchors
            query = query[1:plan.max_anchors]
        end
        _v23_assert_embargo(ctx.anchors, rows.train, query)
        maximum(issues[rows.train]) + Hour(V23_MAX_STEP) <= rows.cutoff || error(
            "fold $(test_year) training window carries a target beyond $(rows.cutoff)",
        )
        push!(folds, V24Fold(test_year, rows.cutoff, rows.train, query))
    end
    return folds
end

"`V23Fold` view of a rolling fold, for the reused V2.3 stage runners."
v24_stage_fold(fold::V24Fold; label::AbstractString="year") =
    V23Fold("$(label)_$(fold.year)", fold.train, fold.query)

# ---------------------------------------------------------------------------
# Out-of-fold store
# ---------------------------------------------------------------------------

"""
    V24Store

Accumulated out-of-fold history: for every year already scored, which anchors it
scored, which of them were served fallbacks, and the three center matrices later
folds fit on — the analog raw core (the T1r ridge target), the corrected T1r
center (the lead-aware composition target) and the composed center (the error
layers' baseline).

The store is the only channel by which one fold's result reaches another, and
[`v24_prior_rows`](@ref) is the only way to read it, so the "strictly earlier
years only" rule has exactly one place to hold.
"""
mutable struct V24Store
    years::Vector{Int}
    rows::Dict{Int,Vector{Int}}
    fallback::Vector{Bool}
    analog_raw::Matrix{Float64}
    t1r::Matrix{Float64}
    lat::Matrix{Float64}
end

V24Store(n::Integer) = V24Store(
    Int[], Dict{Int,Vector{Int}}(), fill(false, Int(n)),
    fill(NaN, Int(n), V23_STEP_COUNT), fill(NaN, Int(n), V23_STEP_COUNT),
    fill(NaN, Int(n), V23_STEP_COUNT),
)

"""
    v24_store_add!(store, year, rows, fallback, analog_raw, t1r, lat) -> store

Register one scored year. Registering a year twice is an error: the second copy
would silently double the fitting pool of every later fold.
"""
function v24_store_add!(store::V24Store, year::Integer, rows::AbstractVector{Int},
                        fallback::AbstractVector{Bool}, analog_raw::AbstractMatrix{Float64},
                        t1r::AbstractMatrix{Float64}, lat::AbstractMatrix{Float64})
    Int(year) in store.years && error("year $(year) is already in the out-of-fold store")
    for i in rows
        store.fallback[i] = fallback[i]
        for slot in 1:V23_STEP_COUNT
            store.analog_raw[i, slot] = analog_raw[i, slot]
            store.t1r[i, slot] = t1r[i, slot]
            store.lat[i, slot] = lat[i, slot]
        end
    end
    store.rows[Int(year)] = sort(collect(Int, rows))
    push!(store.years, Int(year))
    sort!(store.years)
    return store
end

"""
    v24_prior_rows(store, anchors, test_year; nonfallback, years) -> Vector{Int}

Anchors of the out-of-fold pool a fold may fit on: rows of years strictly earlier
than `test_year` whose *targets* also clear the fold's 168 h embargo, served
fallbacks removed unless asked for.

The target rule is Amendment A3. Before it, the pool was cut on the issue year
alone, so an anchor issued on 31 December carried targets up to `Y-01-01T06` —
inside the embargo the plan states on targets, which is the same rule the training
window of `v24_fold_rows` already obeyed. The pool is assembled per anchor rather
than per `(anchor, step)` pair, so the bound is applied at the longest step: an
admitted anchor then satisfies `issue + step <= cutoff` at every step it carries.
Every returned row is re-checked from the anchor view itself, so a store filled
with the wrong rows fails here instead of producing a leaked fit.
"""
function v24_prior_rows(store::V24Store, anchors::V23Anchors, test_year::Integer;
                        nonfallback::Bool=true, years=nothing,
                        embargo_hours::Integer=V23_BASE_EMBARGO_HOURS,
                        max_step::Integer=V23_MAX_STEP)
    boundary = DateTime(Int(test_year), 1, 1)
    cutoff = v24_fold_cutoff(test_year; embargo_hours=embargo_hours)
    wanted = years === nothing ? store.years : collect(Int, years)
    out = Int[]
    for y in sort(wanted)
        y < Int(test_year) || error(
            "the out-of-fold pool of fold $(test_year) may not contain year $y",
        )
        haskey(store.rows, y) || error("year $y is not in the out-of-fold store")
        for i in store.rows[y]
            (nonfallback && store.fallback[i]) && continue
            anchors.issue[i] < boundary || error(
                "out-of-fold row $(anchors.issue[i]) is not strictly earlier than " *
                "$(boundary); the pool of fold $(test_year) would leak",
            )
            anchors.issue[i] + Hour(Int(max_step)) <= cutoff || continue
            push!(out, i)
        end
    end
    return sort(out)
end

# ---------------------------------------------------------------------------
# Row eligibility
# ---------------------------------------------------------------------------

"""
    v24_usable_mask(ctx) -> (usable, direct_features, adc_only)

One eligibility mask for every learned column. An anchor is usable when both
issue-time feature blocks are complete: the 18 ADC features that key the analog
retrieval and the 29-column direct-GBM design that carries the Dst/VBs lag
ladder. The V2.3 runners let the analog tail and the direct comparator fall back
on different rows; a super-learner cannot, because a stack weight is only
meaningful when every expert answered the same row. `adc_only` counts the
anchors that lose their analog center to this stricter rule, so the cost of the
choice is reported rather than absorbed.
"""
function v24_usable_mask(ctx::V23Context)
    features, direct_ok = v23_direct_features!(ctx)
    usable = BitVector(undef, length(ctx.anchors))
    adc_only = 0
    for i in eachindex(usable)
        usable[i] = ctx.ok[i] & direct_ok[i]
        (ctx.ok[i] && !direct_ok[i]) && (adc_only += 1)
    end
    return (usable=usable, features=features, adc_only=adc_only)
end

"Substitute the served V2.1 product on every fallback row of a center matrix."
function v24_serve_fallback!(centers::AbstractMatrix{Float64}, anchors::V23Anchors,
                             rows::AbstractVector{Int}, usable::AbstractVector{Bool})
    for i in rows
        usable[i] && continue
        for slot in 1:V23_STEP_COUNT
            anchors.present[i, slot] || continue
            centers[i, slot] = anchors.served[i, slot]
        end
    end
    return centers
end

# ---------------------------------------------------------------------------
# Base-table row view for the refit ridge correction
# ---------------------------------------------------------------------------

"""
    V24RowView

Base-table rows of every `(issue, model step)` pair the run can touch, in one
read, with a key index. The Amendment A1 correction is a function of the
archived issue-time calibration features, which the anchor view keeps only per
issue, so they have to come from the table; reading the 419 MB artifact once per
fold would cost more than every fit in the study put together.
"""
struct V24RowView
    table::DataFrame
    index::Dict{Tuple{DateTime,Int},Int}
end

"Base-table columns the refit correction and its frozen-feature oracle need."
function v24_t1r_columns()
    columns = v23_t1r_base_table_columns()
    for extra in ("observation_dst_nt", "cv_block")
        extra in columns || push!(columns, extra)
    end
    return columns
end

"""
    v24_row_view(; table_path, partitions) -> V24RowView

Read the base table once with the correction's column selection. `pred_dst_nt`
(the frozen raw core) and the 26 archived calibration feature columns are kept
so that [`v23_t1r_feature_oracle`](@ref) can rebuild them from the raw inputs and
compare, which is the check that the correction is being applied to the state the
recipe was fitted on.
"""
function v24_row_view(; table_path::AbstractString=V23_BASE_TABLE,
                      partitions=("DEV", "TEST"))
    isfile(table_path) || error("V2.3 base table is missing: $table_path")
    table = CSV.read(table_path, DataFrame; select=v24_t1r_columns(),
                     types=Dict("issue_time_utc" => DateTime))
    wanted = Set(String.(partitions))
    table = table[[String(p) in wanted for p in table.partition], :]
    nrow(table) > 0 || error("the base table holds no rows for partitions $(partitions)")
    index = Dict{Tuple{DateTime,Int},Int}()
    sizehint!(index, nrow(table))
    for r in 1:nrow(table)
        key = (table.issue_time_utc[r], Int(table.model_step_hours[r]))
        haskey(index, key) && error("the base table repeats the row $key")
        index[key] = r
    end
    return V24RowView(table, index)
end

"""
    v24_pairs(anchors, rows) -> Vector{Tuple{Int,Int}}

Every `(anchor, step slot)` the base table actually holds for `rows`, in
`(issue, step)` order. Scoring a pair the table does not hold would invent an
observation, so the pair list is derived from `present` and never assumed.
"""
function v24_pairs(anchors::V23Anchors, rows::AbstractVector{Int})
    pairs = Tuple{Int,Int}[]
    for i in rows, slot in 1:V23_STEP_COUNT
        anchors.present[i, slot] || continue
        push!(pairs, (i, slot))
    end
    return pairs
end

"""
    v24_row_frame(rowview, anchors, pairs) -> DataFrame

Base-table rows of `pairs`, in the order of `pairs`.
"""
function v24_row_frame(rowview::V24RowView, anchors::V23Anchors,
                       pairs::AbstractVector{Tuple{Int,Int}})
    order = Vector{Int}(undef, length(pairs))
    for (j, (i, slot)) in enumerate(pairs)
        key = (anchors.issue[i], V23_MODEL_STEPS[slot])
        r = get(rowview.index, key, 0)
        r == 0 && error("the base table has no row for $key; the correction cannot be applied")
        order[j] = r
    end
    return rowview.table[order, :]
end

"""
    v24_t1r_centers!(centers, cal, rowview, anchors, pairs, raw) -> centers

Apply the fold's refit ridge correction to a raw core. `raw[j]` is the core at
`pairs[j]`; the center is `clamp(raw + correction, −2000, 50)`, the S = off
convention of expert E3.
"""
function v24_t1r_centers!(centers::AbstractMatrix{Float64}, cal::OperationalV2Calibration,
                          rowview::V24RowView, anchors::V23Anchors,
                          pairs::AbstractVector{Tuple{Int,Int}},
                          raw::AbstractVector{Float64})
    length(raw) == length(pairs) || throw(DimensionMismatch(
        "the raw core has $(length(raw)) entries for $(length(pairs)) rows",
    ))
    frame = v23_t1r_features(v24_row_frame(rowview, anchors, pairs), raw)
    values = v23_t1r_apply(cal, frame; safeguards=V24_ANALOG_SAFEGUARDS)
    for (j, (i, slot)) in enumerate(pairs)
        centers[i, slot] = values[j]
    end
    return centers
end

# ---------------------------------------------------------------------------
# Expert E3 — analog raw core and the refit ridge correction
# ---------------------------------------------------------------------------

"""
    v24_analog_stage(ctx, stage_fold) -> (result, seconds)

Analog driver-continuation raw core of one fold: K = 25 magnetic-weighted
neighbours drawn from the fold's training window, rolled out once per neighbour
through the unchanged SINDy core. The ±168 h retrieval exclusion of
`v23_knn` is inside the reused stage, and the fold rule already puts every
origin at least 168 h before the year, so an origin can neither be the query nor
overlap its target window.
"""
function v24_analog_stage(ctx::V23Context, stage_fold::V23Fold)
    started = time()
    results = v23_run_analog_stage!(ctx, [stage_fold], V24_ANALOG_WEIGHT_SET;
                                   ks=[V24_ANALOG_K], direct=false,
                                   family=V24_ANALOG_FAMILY)
    result = results[V24_ANALOG_K]
    return (result=result, seconds=time() - started)
end

"""
    v24_reported_raw(result, anchors, rows, usable) -> Matrix{Float64}

Reported analog raw core: the ensemble mean projected to the physical range, and
the frozen raw core on every row the analog family cannot forecast — the served
fallback convention of `v23_config_centers`, applied to the raw column so that
`t1_analog_raw` and `t1r_analog` describe the same rows.
"""
function v24_reported_raw(result::V23ConfigResult, anchors::V23Anchors,
                          rows::AbstractVector{Int}, usable::AbstractVector{Bool})
    raw = fill(NaN, length(anchors), V23_STEP_COUNT)
    for i in rows, slot in 1:V23_STEP_COUNT
        anchors.present[i, slot] || continue
        raw[i, slot] = usable[i] ? clamp(result.raw[i, slot], -2000.0, 50.0) :
                                   anchors.raw_served[i, slot]
    end
    return raw
end

"""
    v24_t1r_fit_rows(store, ctx, fold, usable, plan) -> (pairs, raw, in_sample)

Rows the fold's ridge correction is fitted on, and the analog raw core at those
rows.

With an accumulated out-of-fold history the fit uses the analog cores of the
earlier folds, which is the leakage-safe choice: those cores were produced by
archives that end before their own years, and all of them end before this
fold's year. The seed fold has no such history, so its layer is fitted
*in sample* on the training window — the analog core there is produced by an
archive that contains the fitted rows themselves. That is disclosed on the fold
manifest (`t1r_fit_in_sample = 1`) and is why 2013 seeds the history instead of
entering an era.
"""
function v24_t1r_fit_rows(store::V24Store, ctx::V23Context, fold::V24Fold,
                          usable::AbstractVector{Bool}, plan::V24Plan)
    anchors = ctx.anchors
    if !isempty(store.years)
        rows = v24_prior_rows(store, anchors, fold.year)
        isempty(rows) && error("fold $(fold.year) has an empty out-of-fold correction pool")
        pairs = [(i, slot) for (i, slot) in v24_pairs(anchors, rows)
                 if isfinite(store.analog_raw[i, slot])]
        raw = [store.analog_raw[i, slot] for (i, slot) in pairs]
        return (pairs=pairs, raw=raw, in_sample=false, seed_rows=Int[], seconds=0.0)
    end
    seed = [i for i in fold.train if usable[i]]
    isempty(seed) && error("fold $(fold.year) has no usable training anchor to seed the layer")
    if plan.seed_max_anchors > 0 && length(seed) > plan.seed_max_anchors
        seed = seed[end - plan.seed_max_anchors + 1:end]
    end
    started = time()
    stage = v23_run_analog_stage!(ctx, [V23Fold("seed_$(fold.year)", fold.train, seed)],
                                 V24_ANALOG_WEIGHT_SET; ks=[V24_ANALOG_K], direct=false,
                                 family=V24_ANALOG_FAMILY)[V24_ANALOG_K]
    pairs = [(i, slot) for (i, slot) in v24_pairs(anchors, seed)
             if isfinite(stage.raw[i, slot])]
    raw = [clamp(stage.raw[i, slot], -2000.0, 50.0) for (i, slot) in pairs]
    return (pairs=pairs, raw=raw, in_sample=true, seed_rows=seed, seconds=time() - started)
end

# ---------------------------------------------------------------------------
# Expert E8 — tuned direct increment-GBM
# ---------------------------------------------------------------------------

"""
    v24_direct_select(ctx, fold, usable, features, plan) -> NamedTuple

Inner-validation choice of the direct increment-GBM hyper-parameters, made
entirely inside the fold's own training window: the last 24 months of that window
are held out, every grid point is fitted on the rest and scored on the holdout,
and the winner of the step-1 search is applied to steps {1, 2} while the winner
of the step-6 search is applied to steps {3, 4, 6, 7}.

Separated from [`v24_direct_stage!`](@ref) so that a deployment builder can reuse
the selection and the fit without a scored year to predict on; the rolling engine
calls both in sequence and its arithmetic is unchanged.
"""
function v24_direct_select(ctx::V23Context, fold::V24Fold, usable::AbstractVector{Bool},
                           features::AbstractMatrix{Float64}, plan::V24Plan)
    started = time()
    anchors = ctx.anchors
    names = String.(v23_direct_feature_names())

    valid_start = fold.cutoff - Month(V24_DIRECT_INNER_MONTHS)
    inner_cutoff = valid_start - Hour(V23_BASE_EMBARGO_HOURS)
    audit = NamedTuple[]
    chosen = Dict{Int,V24DirectConfig}()
    for (selection_step, applies) in V24_DIRECT_SELECTION
        slot = V23_STEP_SLOT[selection_step]
        pool = [i for i in fold.train if usable[i] && anchors.present[i, slot]]
        inner_train = [i for i in pool
                       if anchors.issue[i] + Hour(V23_MAX_STEP) <= inner_cutoff]
        inner_valid = [i for i in pool if anchors.issue[i] >= valid_start]
        if plan.direct_max_train > 0 && length(inner_train) > plan.direct_max_train
            inner_train = inner_train[end - plan.direct_max_train + 1:end]
        end
        length(inner_train) >= 2 * V23_GBM_MIN_WEIGHT || error(
            "fold $(fold.year) has only $(length(inner_train)) inner-training rows at step " *
            "$(selection_step)",
        )
        isempty(inner_valid) && error(
            "fold $(fold.year) has no inner-validation row at step $(selection_step)",
        )
        design = features[inner_train, :]
        target = v23_direct_target(anchors, inner_train, slot)
        validation = features[inner_valid, :]
        best = nothing
        best_rmse = Inf
        for config in plan.direct_grid
            model = v23_fit_gbm(design, target; max_depth=config.depth,
                                nrounds=config.rounds, nbins=config.nbins,
                                eta=V23_GBM_ETA, min_weight=V23_GBM_MIN_WEIGHT,
                                seed=V23_SEED, feature_names=names)
            increments = v23_predict(model, validation)
            errors = [anchors.obs[i, slot] - v23_direct_center(anchors, i, increments[j])
                      for (j, i) in enumerate(inner_valid)]
            rmse = v23_rmse(errors)
            push!(audit, (
                selection_step=selection_step, config=v24_direct_label(config),
                depth=config.depth, rounds=config.rounds, nbins=config.nbins,
                n_inner_train=length(inner_train), n_inner_valid=length(inner_valid),
                inner_rmse_nt=rmse,
            ))
            (isfinite(rmse) && rmse < best_rmse) && (best_rmse = rmse; best = config)
        end
        best === nothing && error(
            "fold $(fold.year) inner validation produced no finite RMSE at step " *
            "$(selection_step)",
        )
        for step in applies
            chosen[step] = best
        end
        _v23_log(@sprintf(
            "    direct-GBM fold %d: step %d chose %s (inner RMSE %.4f nT over %d rows)",
            fold.year, selection_step, v24_direct_label(best), best_rmse, length(inner_valid),
        ))
    end
    length(chosen) == V23_STEP_COUNT || error(
        "the direct-GBM selection covers $(length(chosen)) of $(V23_STEP_COUNT) model steps",
    )
    return (chosen=chosen, audit=audit, seconds=time() - started)
end

"""
    v24_direct_fit(ctx, fold, usable, features, chosen, plan) -> NamedTuple

Fit one direct increment-GBM per model step on the fold's whole training window
with the hyper-parameters `chosen` by [`v24_direct_select`](@ref). Returns the
fitted models and the row count behind each, keyed by model step, so the caller
can either predict with them or persist them.
"""
function v24_direct_fit(ctx::V23Context, fold::V24Fold, usable::AbstractVector{Bool},
                        features::AbstractMatrix{Float64}, chosen::Dict{Int,V24DirectConfig},
                        plan::V24Plan)
    started = time()
    anchors = ctx.anchors
    names = String.(v23_direct_feature_names())
    models = Dict{Int,Any}()
    train_rows = Dict{Int,Int}()
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        haskey(chosen, step) || error(
            "the direct-GBM selection carries no configuration at step $step",
        )
        config = chosen[step]
        train = [i for i in fold.train if usable[i] && anchors.present[i, slot]]
        if plan.direct_max_train > 0 && length(train) > plan.direct_max_train
            train = train[end - plan.direct_max_train + 1:end]
        end
        length(train) >= 2 * V23_GBM_MIN_WEIGHT || error(
            "fold $(fold.year) has only $(length(train)) direct-GBM training rows at step $step",
        )
        v23_direct_check_anchor(features, anchors, train)
        models[step] = v23_fit_gbm(features[train, :], v23_direct_target(anchors, train, slot);
                                   max_depth=config.depth, nrounds=config.rounds,
                                   nbins=config.nbins, eta=V23_GBM_ETA,
                                   min_weight=V23_GBM_MIN_WEIGHT, seed=V23_SEED,
                                   feature_names=names)
        train_rows[step] = length(train)
    end
    return (models=models, train_rows=train_rows, seconds=time() - started)
end

"""
    v24_direct_stage!(ctx, fold, usable, features, plan) -> NamedTuple

The direct increment-GBM of plan section 4, refitted inside the fold with its
hyper-parameters chosen inside the fold's own training window.

Two things differ from `v23_run_direct_stage!` and both are required by the
spec: the grid carries the histogram resolution (`nbins`), which the V2.3 stage
cannot express because it takes the package default of 64, and the choice is
made by an inner validation on the last 24 months of the training window instead
of being fixed in advance. Everything else is the reused V2.3 arithmetic — the
same 29-column design, the same increment target `Dst(t+h) − Dst(t)` through
`v23_direct_target`, the same inversion through `v23_direct_center`, the same
anchor check, the same seed.
"""
function v24_direct_stage!(ctx::V23Context, fold::V24Fold, usable::AbstractVector{Bool},
                           features::AbstractMatrix{Float64}, plan::V24Plan)
    started = time()
    anchors = ctx.anchors
    query = [i for i in fold.query if usable[i]]
    isempty(query) && error("fold $(fold.year) has no usable scored anchor")
    v23_direct_check_anchor(features, anchors, query)
    query_features = features[query, :]
    selection = v24_direct_select(ctx, fold, usable, features, plan)
    fitted = v24_direct_fit(ctx, fold, usable, features, selection.chosen, plan)

    centers = fill(NaN, length(anchors), V23_STEP_COUNT)
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        increments = v23_predict(fitted.models[step], query_features)
        for (j, i) in enumerate(query)
            centers[i, slot] = v23_direct_center(anchors, i, increments[j])
        end
    end
    v24_serve_fallback!(centers, anchors, fold.query, usable)
    return (centers=centers, chosen=selection.chosen, audit=selection.audit,
            models=fitted.models, seconds=time() - started)
end

# ---------------------------------------------------------------------------
# V2.3 shadow — lead-aware composition and the error layers
# ---------------------------------------------------------------------------

"""
    v24_lat_weights(store, ctx, fold) -> (weights, table, n_rows)

Lead-aware composition weights of the fold, fitted on the corrected analog
centers of the earlier folds only. With no earlier fold the composition is the
identity (`w = 1`, the tail center), which is the disclosed seed behaviour.
"""
function v24_lat_weights(store::V24Store, ctx::V23Context, fold::V24Fold)
    isempty(store.years) && return (weights=copy(V24_LAT_IDENTITY_WEIGHTS),
                                    table=NamedTuple[], n_rows=0)
    rows = v24_prior_rows(store, ctx.anchors, fold.year)
    isempty(rows) && return (weights=copy(V24_LAT_IDENTITY_WEIGHTS), table=NamedTuple[],
                             n_rows=0)
    lat = v23_lat_weights(ctx, store.t1r, rows)
    return (weights=lat.weights, table=lat.table, n_rows=length(rows))
end

"""
    v24_e_layer_folds(store, ctx, fold) -> Vector{NamedTuple}

Expanding-window folds the error layers are adjudicated on: year `y` of the
accumulated pool is corrected by a layer fitted on the earlier pooled years,
embargoed by 168 h of targets. The first pooled year has no predecessor and is
therefore not adjudicated, exactly as `v23_e_layer_folds` treats the first inner
block.
"""
function v24_e_layer_folds(store::V24Store, ctx::V23Context, fold::V24Fold)
    out = NamedTuple[]
    years = sort(store.years)
    for (position, y) in enumerate(years)
        position == 1 && continue
        query = v24_prior_rows(store, ctx.anchors, fold.year; years=[y])
        isempty(query) && continue
        cutoff = minimum(ctx.anchors.issue[query]) - Hour(V23_BASE_EMBARGO_HOURS)
        train = [i for i in v24_prior_rows(store, ctx.anchors, fold.year;
                                           years=years[1:position - 1])
                 if ctx.anchors.issue[i] + Hour(V23_MAX_STEP) <= cutoff]
        isempty(train) && continue
        push!(out, (name="year_$(y)", train=train, query=query))
    end
    return out
end

"""
    v24_shadow_layers(store, ctx, fold, lat_centers, year_rows) -> NamedTuple

The V2.3 shadow of the fold: the composed center completed by whichever error
layer the accumulated pool says helps at each step.

The adjudication runs on the pool alone — layers are fitted on earlier pooled
years and scored on later ones — and only the layer a step accepts there is
refitted on the whole pool and applied to the scored year. Re-adjudicating on the
scored year would make the shadow a selection on its own test rows, which is the
mistake the confirmatory runner avoids by fixing acceptance on DEV.
"""
function v24_shadow_layers(store::V24Store, ctx::V23Context, fold::V24Fold,
                           lat_centers::AbstractMatrix{Float64},
                           year_rows::AbstractVector{Int})
    started = time()
    identity_result = (centers=copy(lat_centers), audit=NamedTuple[],
                       layers=Vector{Any}(nothing, V23_STEP_COUNT), applied=NamedTuple[],
                       note="identity", seconds=time() - started)
    isempty(store.years) && return merge(identity_result, (note="no_prior_out_of_fold",))
    pool = v24_prior_rows(store, ctx.anchors, fold.year)
    isempty(pool) && return merge(identity_result, (note="empty_pool",))
    e_folds = v24_e_layer_folds(store, ctx, fold)
    isempty(e_folds) && return merge(identity_result, (note="no_adjudication_fold",))
    eval_rows = sort(unique(reduce(vcat, [f.query for f in e_folds]; init=Int[])))
    isempty(eval_rows) && return merge(identity_result, (note="no_adjudication_row",))

    e1 = v23_run_e_layer(ctx, e_folds, store.lat, pool, :E1; params=V24_E1_PARAMS,
                         label="E1", eval_rows=eval_rows)
    e2 = v23_run_e_layer(ctx, e_folds, store.lat, pool, :E2; params=V24_E2_PARAMS,
                         label="E2", eval_rows=eval_rows)
    audit = vcat(e1.audit, e2.audit)
    layers = Vector{Any}(nothing, V23_STEP_COUNT)
    for slot in 1:V23_STEP_COUNT
        step = V23_MODEL_STEPS[slot]
        gain_of(entries, label) = begin
            hit = [r for r in entries
                   if r.model_step_hours == step && r.accepted && r.layer == label]
            isempty(hit) ? -Inf : first(hit).gain_nt
        end
        g1 = gain_of(e1.audit, "E1")
        g2 = gain_of(e2.audit, "E2")
        max(g1, g2) <= 0 && continue
        layers[slot] = g1 >= g2 ? (kind=:E1, param=e1.params[slot], gain_nt=g1) :
                                  (kind=:E2, param=e2.params[slot], gain_nt=g2)
    end
    if all(layer -> layer === nothing, layers)
        return (centers=copy(lat_centers), audit=audit, layers=layers, applied=NamedTuple[],
                note="no_layer_accepted", seconds=time() - started)
    end

    combined = copy(store.lat)
    for i in year_rows, slot in 1:V23_STEP_COUNT
        ctx.anchors.present[i, slot] || continue
        combined[i, slot] = lat_centers[i, slot]
    end
    applied = _v23_apply_fixed_e_layers(ctx, pool, year_rows, combined, layers)
    centers = copy(lat_centers)
    for i in year_rows, slot in 1:V23_STEP_COUNT
        ctx.anchors.present[i, slot] || continue
        isfinite(applied.centers[i, slot]) || continue
        centers[i, slot] = applied.centers[i, slot]
    end
    return (centers=centers, audit=audit, layers=layers, applied=applied.applied,
            note="layers_applied", seconds=time() - started)
end

# ---------------------------------------------------------------------------
# Fold artifacts
# ---------------------------------------------------------------------------

"""
    v24_fold_signature(ctx, plan, fold) -> String

Resume key of one fold: the fold itself, the scale of the run, the base table,
the hourly frame, the V2.3 sources, this runner and the contract. A persisted
fold is consumed only when this string still matches, so a changed input or a
changed stage invalidates it instead of mixing two generations of centers inside
one study.
"""
function v24_fold_signature(ctx::V23Context, plan::V24Plan, fold::V24Fold)
    payload = Dict{String,Any}(
        "year" => fold.year,
        "cutoff" => string(fold.cutoff),
        "train_anchors" => length(fold.train),
        "query_anchors" => length(fold.query),
        "plan" => plan.label,
        "max_anchors" => plan.max_anchors,
        "direct_grid" => [v24_direct_label(c) for c in plan.direct_grid],
        "direct_max_train" => plan.direct_max_train,
        "seed_max_anchors" => plan.seed_max_anchors,
        "e_layers" => plan.e_layers,
        "analog" => "$(V24_ANALOG_WEIGHT_SET)_K$(V24_ANALOG_K)_S$(V24_ANALOG_SAFEGUARDS)",
        "columns" => collect(V24_OOF_COLUMNS),
        "base_table_sha256" => ctx.base_table_sha,
        "hourly_frame_sha256" => ctx.frame_sha,
        "code_sha256" => ctx.code_sha,
        "rolling_sha256" => _v23_sha256_file(joinpath(@__DIR__, "v2_4_rolling.jl")),
        "contract_sha256" => _v23_sha256_file(joinpath(@__DIR__, "v2_4_contract.jl")),
        "cells" => ctx.cells_source,
    )
    return _v23_sha256_text(JSON3.write(payload))
end

"""
    v24_fold_is_current(ctx, plan, fold) -> Bool

Whether the persisted fold artifacts can be consumed as this run's own output.
"""
function v24_fold_is_current(ctx::V23Context, plan::V24Plan, fold::V24Fold)
    signature_path = v24_signature_year_path(plan.outdir, fold.year)
    for path in (signature_path, v24_oof_year_path(plan.outdir, fold.year),
                 v24_manifest_year_path(plan.outdir, fold.year))
        isfile(path) || return false
    end
    return strip(read(signature_path, String)) == v24_fold_signature(ctx, plan, fold)
end

"""
    v24_write_fold(plan, ctx, fold, usable, columns, features) -> NamedTuple

Write `oof_year_<Y>.csv` in the contract's column order and validate it by
reading it back. The validation is not ceremony: Task B fits stack weights over
these column names, and a fold that silently lost a column would produce weights
over an expert set nobody chose.
"""
function v24_write_fold(plan::V24Plan, ctx::V23Context, fold::V24Fold,
                        usable::AbstractVector{Bool},
                        centers::Dict{String,Matrix{Float64}},
                        features::AbstractMatrix{Float64})
    anchors = ctx.anchors
    pairs = v24_pairs(anchors, fold.query)
    n = length(pairs)
    data = Dict{String,Vector}()
    data["issue_time_utc"] = [string(anchors.issue[i]) for (i, _) in pairs]
    data["model_step_hours"] = [V23_MODEL_STEPS[slot] for (_, slot) in pairs]
    data["observation_dst_nt"] = [anchors.obs[i, slot] for (i, slot) in pairs]
    data["latest_dst_nt"] = [anchors.latest[i] for (i, _) in pairs]
    data["dst_delta_1h_nt"] = [anchors.rate[i] for (i, _) in pairs]
    data["coupling_active_mvm"] = [anchors.coupling[i] for (i, _) in pairs]
    data["fallback"] = [!usable[i] for (i, _) in pairs]
    for name in V24_MODEL_COLUMNS
        matrix = centers[name]
        column = Vector{Float64}(undef, n)
        for (j, (i, slot)) in enumerate(pairs)
            value = matrix[i, slot]
            isfinite(value) || error(
                "fold $(fold.year) column $name is not finite at " *
                "$(anchors.issue[i]) step $(V23_MODEL_STEPS[slot])",
            )
            column[j] = value
        end
        data[name] = column
    end
    for (index, name) in enumerate(V24_FEATURE_COLUMNS)
        column = Vector{Float64}(undef, n)
        for (j, (i, _)) in enumerate(pairs)
            column[j] = usable[i] ? features[i, index] : NaN
        end
        data[name] = column
    end
    path = v24_oof_year_path(plan.outdir, fold.year)
    CSV.write(path, DataFrame([name => data[name] for name in V24_OOF_COLUMNS]))
    check = v24_require_oof_year(path; expect_year=fold.year)
    return (path=path, n_rows=n, check=check)
end

"Per-fold manifest: inputs, window, counts, hyper-parameters, identities, timings."
function v24_write_fold_manifest(plan::V24Plan, ctx::V23Context, fold::V24Fold,
                                 manifest::V23Manifest)
    path = v24_manifest_year_path(plan.outdir, fold.year)
    v23_manifest!(manifest, "input_sha256", "v2_3_base_table.csv", NaN, ctx.base_table_sha)
    v23_manifest!(manifest, "input_sha256", "v2_3_hourly_frame.csv", NaN, ctx.frame_sha)
    v23_manifest!(manifest, "input_sha256", "v2_3_source_code", NaN, ctx.code_sha)
    v23_manifest!(manifest, "input_sha256", "v2_4_rolling.jl", NaN,
                  _v23_sha256_file(joinpath(@__DIR__, "v2_4_rolling.jl")))
    v23_manifest!(manifest, "input_sha256", "v2_4_contract.jl", NaN,
                  _v23_sha256_file(joinpath(@__DIR__, "v2_4_contract.jl")))
    v23_manifest!(manifest, "environment", "julia_threads", Threads.nthreads(), "")
    v23_manifest!(manifest, "environment", "cells", NaN, ctx.cells_source)
    v23_manifest!(manifest, "environment", "feature_scaling", NaN, V24_FEATURE_SCALING)
    v23_manifest!(manifest, "identity", "v2_1_correction_max_abs_nt",
                  ctx.correction_residual_nt, "")
    oof = v24_oof_year_path(plan.outdir, fold.year)
    isfile(oof) && v23_manifest!(manifest, "output_sha256", basename(oof), filesize(oof),
                                 _v23_sha256_file(oof))
    CSV.write(path, DataFrame(manifest.rows))
    return path
end

# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

"""
    v24_run_fold!(ctx, plan, fold, store, rowview, usable, features) -> NamedTuple

Score one rolling-origin fold and persist it. Order matters only in one place:
the refit ridge correction is fitted before it is applied to the analog core and
to the ceiling, and its frozen-feature oracle runs before either application, so
a drift in the archived calibration state stops the fold instead of silently
changing what the correction corrects.
"""
function v24_run_fold!(ctx::V23Context, plan::V24Plan, fold::V24Fold, store::V24Store,
                       rowview::V24RowView, usable::AbstractVector{Bool},
                       features::AbstractMatrix{Float64})
    total_started = time()
    anchors = ctx.anchors
    manifest = V23Manifest()
    rows = fold.query
    scored_pairs = v24_pairs(anchors, rows)
    n_fallback = count(i -> !usable[i], rows)
    _v23_log(@sprintf(
        "  fold %d: train %d anchors (targets <= %s), scored %d anchors / %d rows, %d served fallback",
        fold.year, length(fold.train), fold.cutoff, length(rows), length(scored_pairs),
        n_fallback,
    ))
    v23_manifest!(manifest, "plan", "label", NaN, plan.label)
    v23_manifest!(manifest, "plan", "smoke", plan.smoke ? 1 : 0, "")
    v23_manifest!(manifest, "fold", "test_year", fold.year, "")
    v23_manifest!(manifest, "window", "training_max_target_utc", NaN, string(fold.cutoff))
    v23_manifest!(manifest, "window", "training_first_issue_utc", NaN,
                  string(minimum(anchors.issue[fold.train])))
    v23_manifest!(manifest, "window", "training_last_issue_utc", NaN,
                  string(maximum(anchors.issue[fold.train])))
    v23_manifest!(manifest, "window", "training_last_target_utc", NaN,
                  string(maximum(anchors.issue[fold.train]) + Hour(V23_MAX_STEP)))
    v23_manifest!(manifest, "window", "scored_first_issue_utc", NaN,
                  string(minimum(anchors.issue[rows])))
    v23_manifest!(manifest, "window", "scored_last_issue_utc", NaN,
                  string(maximum(anchors.issue[rows])))
    v23_manifest!(manifest, "window", "embargo_hours",
                  _v23_assert_embargo(anchors, fold.train, rows), "")
    v23_manifest!(manifest, "rows", "training_anchors", length(fold.train), "")
    v23_manifest!(manifest, "rows", "training_anchors_usable",
                  count(i -> usable[i], fold.train), "")
    v23_manifest!(manifest, "rows", "scored_anchors", length(rows), "")
    v23_manifest!(manifest, "rows", "scored_rows", length(scored_pairs), "")
    v23_manifest!(manifest, "rows", "scored_anchors_fallback", n_fallback, "")
    v23_manifest!(manifest, "rows", "scored_anchors_latest_le_m50",
                  count(i -> ctx.masks.latest_le_m50[i], rows), "")
    v23_manifest!(manifest, "rows", "scored_anchors_latest_le_m100",
                  count(i -> ctx.masks.latest_le_m100[i], rows), "")

    centers = Dict{String,Matrix{Float64}}()
    base_matrix(field) = begin
        matrix = fill(NaN, length(anchors), V23_STEP_COUNT)
        source = getproperty(anchors, field)
        for i in rows, slot in 1:V23_STEP_COUNT
            anchors.present[i, slot] || continue
            matrix[i, slot] = source[i, slot]
        end
        matrix
    end
    for (name, field) in (("served_v2_1", :served), ("frozen_v2_1", :frozen),
                          ("persistence", :persistence), ("burton", :burton),
                          ("burton_full", :burton_full), ("obrien", :obrien),
                          ("static_v2_2", :static_v22))
        centers[name] = base_matrix(field)
    end

    # ---- expert E9, climatology-relaxed persistence -------------------------
    started = time()
    tau = _v23_climatology_tau(ctx, fold.train)
    centers["climatology"] = _v23_climatology_centers(ctx, rows, tau)
    v23_manifest!(manifest, "hyperparameter", "climatology_tau_hours", tau, "")
    v23_manifest!(manifest, "seconds", "stage_climatology", time() - started, "")

    # ---- expert E3, analog raw core ----------------------------------------
    analog = v24_analog_stage(ctx, v24_stage_fold(fold))
    centers["t1_analog_raw"] = v24_reported_raw(analog.result, anchors, rows, usable)
    v23_manifest!(manifest, "hyperparameter", "analog_k", V24_ANALOG_K,
                  "weights=$(V24_ANALOG_WEIGHT_SET);safeguards=$(V24_ANALOG_SAFEGUARDS)")
    v23_manifest!(manifest, "seconds", "stage_analog", analog.seconds, "")

    # ---- expert E3, refit ridge correction ---------------------------------
    started = time()
    fit = v24_t1r_fit_rows(store, ctx, fold, usable, plan)
    fit_frame = v23_t1r_features(v24_row_frame(rowview, anchors, fit.pairs), fit.raw)
    cal = v23_t1r_fit(fit_frame; label_stem="operational_v2_4_t1r_year$(fold.year)")
    oracle_frame = v24_row_frame(rowview, anchors, scored_pairs)
    feature_oracle = v23_t1r_feature_oracle(oracle_frame; atol=V24_T1R_ORACLE_ATOL_NT)
    t1r = fill(NaN, length(anchors), V23_STEP_COUNT)
    v24_t1r_centers!(t1r, cal, rowview, anchors, scored_pairs,
                     [centers["t1_analog_raw"][i, slot] for (i, slot) in scored_pairs])
    v24_serve_fallback!(t1r, anchors, rows, usable)
    centers["t1r_analog"] = t1r
    v23_manifest!(manifest, "hyperparameter", "t1r_label", NaN, cal.label)
    v23_manifest!(manifest, "hyperparameter", "oof_pool_target_embargo_hours",
                  V23_BASE_EMBARGO_HOURS,
                  "amendment_A3;pool_rows_need_issue_plus_$(V23_MAX_STEP)h_le_cutoff")
    v23_manifest!(manifest, "hyperparameter", "t1r_fit_rows", length(fit.pairs), "")
    v23_manifest!(manifest, "hyperparameter", "t1r_fit_in_sample", fit.in_sample ? 1 : 0,
                  fit.in_sample ? "seed_fold_training_window" : "prior_fold_out_of_fold")
    fit.in_sample && v23_manifest!(manifest, "hyperparameter", "t1r_seed_anchors",
                                   length(fit.seed_rows), "")
    v23_manifest!(manifest, "identity", "t1r_feature_oracle_max_abs_nt",
                  feature_oracle.worst_nt, String(feature_oracle.column))
    v23_manifest!(manifest, "seconds", "stage_t1r", time() - started + fit.seconds, "")

    # ---- expert E8, tuned direct increment-GBM -----------------------------
    direct = v24_direct_stage!(ctx, fold, usable, features, plan)
    centers["direct_gbm"] = direct.centers
    v23_manifest!(manifest, "hyperparameter", "direct_gbm_grid_points",
                  length(plan.direct_grid),
                  join([v24_direct_label(config) for config in plan.direct_grid], ";"))
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        v23_manifest!(manifest, "hyperparameter", "direct_gbm_step$(step)", NaN,
                      v24_direct_label(direct.chosen[step]))
    end
    for row in direct.audit
        v23_manifest!(manifest, "direct_gbm_inner",
                      "step$(row.selection_step)_$(row.config)", row.inner_rmse_nt,
                      "train=$(row.n_inner_train);valid=$(row.n_inner_valid)")
    end
    v23_manifest!(manifest, "seconds", "stage_direct_gbm", direct.seconds, "")

    # ---- noncausal ceiling -------------------------------------------------
    started = time()
    ceiling = v23_run_oracle_stage!(ctx, [v24_stage_fold(fold; label="oracle")]).result
    oracle_centers = fill(NaN, length(anchors), V23_STEP_COUNT)
    v24_t1r_centers!(oracle_centers, cal, rowview, anchors, scored_pairs,
                     [clamp(ceiling.raw[i, slot], -2000.0, 50.0) for (i, slot) in scored_pairs])
    centers["oracle_realized"] = oracle_centers
    v23_manifest!(manifest, "comparator", "oracle_correction", NaN,
                  "t1r_year$(fold.year)_S$(V24_ANALOG_SAFEGUARDS)")
    v23_manifest!(manifest, "seconds", "stage_oracle", time() - started, "")

    # ---- V2.3 shadow -------------------------------------------------------
    started = time()
    lat = v24_lat_weights(store, ctx, fold)
    tail_rows = [i for i in rows if usable[i]]
    lat_centers = v23_apply_lat(ctx, t1r, lat.weights, tail_rows)
    v24_serve_fallback!(lat_centers, anchors, rows, usable)
    centers["v2_3_lat"] = lat_centers
    v23_manifest!(manifest, "hyperparameter", "lat_weights", lat.n_rows,
                  join(string.(lat.weights), ";"))
    if plan.e_layers
        shadow = v24_shadow_layers(store, ctx, fold, lat_centers, tail_rows)
        shadow_centers = shadow.centers
        v24_serve_fallback!(shadow_centers, anchors, rows, usable)
        centers["v2_3_shadow"] = shadow_centers
        v23_manifest!(manifest, "hyperparameter", "e_layer_state", NaN, shadow.note)
        for (slot, step) in enumerate(V23_MODEL_STEPS)
            layer = shadow.layers[slot]
            v23_manifest!(manifest, "hyperparameter", "e_layer_step$(step)",
                          layer === nothing ? 0.0 : layer.gain_nt,
                          layer === nothing ? "identity" :
                          "$(layer.kind):$(layer.param)")
        end
        isempty(shadow.audit) || CSV.write(
            joinpath(plan.outdir, "e_layer_audit_year_$(fold.year).csv"),
            DataFrame(shadow.audit),
        )
        v23_manifest!(manifest, "seconds", "stage_shadow_layers", shadow.seconds, "")
    else
        centers["v2_3_shadow"] = copy(lat_centers)
        v23_manifest!(manifest, "hyperparameter", "e_layer_state", NaN,
                      "disabled_by_run_plan")
    end
    v23_manifest!(manifest, "seconds", "stage_lat", time() - started, "")

    # ---- persist -----------------------------------------------------------
    started = time()
    written = v24_write_fold(plan, ctx, fold, usable, centers, features)
    v23_manifest!(manifest, "rows", "written_rows", written.n_rows, "")
    v23_manifest!(manifest, "seconds", "stage_write", time() - started, "")
    v23_manifest!(manifest, "seconds", "fold_total", time() - total_started, "")
    v24_write_fold_manifest(plan, ctx, fold, manifest)
    write(v24_signature_year_path(plan.outdir, fold.year),
          v24_fold_signature(ctx, plan, fold))
    v24_store_add!(store, fold.year, rows, [!usable[i] for i in eachindex(usable)],
                   centers["t1_analog_raw"], t1r, lat_centers)
    _v23_log(@sprintf("  + fold %d written: %d rows in %.1f s → %s", fold.year,
                      written.n_rows, time() - total_started, basename(written.path)))
    return (year=fold.year, rows=written.n_rows, seconds=time() - total_started,
            path=written.path, fallback=n_fallback, tau_hours=tau,
            direct=Dict(step => v24_direct_label(config)
                        for (step, config) in direct.chosen),
            t1r_in_sample=fit.in_sample, t1r_fit_rows=length(fit.pairs),
            lat_weights=copy(lat.weights), feature_oracle_nt=feature_oracle.worst_nt)
end

"""
    v24_load_fold!(store, ctx, plan, fold) -> NamedTuple

Rebuild one fold's contribution to the out-of-fold store from its persisted file,
without recomputing it. The columns read back are exactly the three the later
folds fit on, which is why `v2_3_lat` is persisted next to `v2_3_shadow`.
"""
function v24_load_fold!(store::V24Store, ctx::V23Context, plan::V24Plan, fold::V24Fold)
    path = v24_oof_year_path(plan.outdir, fold.year)
    check = v24_require_oof_year(path; expect_year=fold.year)
    table = CSV.read(path, DataFrame;
                     select=["issue_time_utc", "model_step_hours", "fallback",
                             "t1_analog_raw", "t1r_analog", "v2_3_lat"],
                     types=Dict("issue_time_utc" => DateTime))
    anchors = ctx.anchors
    n = length(anchors)
    analog_raw = fill(NaN, n, V23_STEP_COUNT)
    t1r = fill(NaN, n, V23_STEP_COUNT)
    lat = fill(NaN, n, V23_STEP_COUNT)
    fallback = fill(false, n)
    seen = falses(n)
    for r in 1:nrow(table)
        i = get(anchors.index, table.issue_time_utc[r], 0)
        i == 0 && error("persisted fold $(fold.year) holds an unknown issue time")
        Dates.year(anchors.issue[i]) == fold.year || error(
            "persisted fold $(fold.year) holds an issue from $(anchors.issue[i])",
        )
        slot = V23_STEP_SLOT[Int(table.model_step_hours[r])]
        slot == 0 && error("persisted fold $(fold.year) holds an unscored model step")
        analog_raw[i, slot] = Float64(table.t1_analog_raw[r])
        t1r[i, slot] = Float64(table.t1r_analog[r])
        lat[i, slot] = Float64(table.v2_3_lat[r])
        fallback[i] = Bool(table.fallback[r])
        seen[i] = true
    end
    rows = findall(seen)
    isempty(rows) && error("persisted fold $(fold.year) holds no usable row")
    v24_store_add!(store, fold.year, rows, fallback, analog_raw, t1r, lat)
    return (year=fold.year, rows=nrow(table), anchors=length(rows), check=check)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    v24_run(plan; reuse_stale_reason) -> NamedTuple

Score every fold of `plan` in ascending year order, resuming any fold whose
persisted signature still matches. The order is not a convenience: fold `Y` fits
its correction layer, its composition weights and its error layers on the
accumulated out-of-fold history of the earlier folds, so a later fold cannot be
computed before an earlier one.

`reuse_stale_reason` consumes persisted folds whose signature no longer matches
and records the written reason on the run manifest. It exists for the same
narrow case as the confirmatory runner's `--accept-code-drift`: a source change
that provably cannot move the persisted centers. Without it a stale fold is
recomputed.
"""
function v24_run(plan::V24Plan; reuse_stale_reason::Union{Nothing,String}=nothing)
    total_started = time()
    v24_assert_full_plan(plan)
    mkpath(plan.outdir)
    _v23_log("V2.4 rolling engine ($(plan.label)); threads=$(Threads.nthreads()); " *
             "years=$(join(sort(plan.years), ","))")

    started = time()
    ctx = v23_build_context(v24_context_plan(plan); partitions=("DEV", "TEST"))
    mask = v24_usable_mask(ctx)
    usable = mask.usable
    rowview = v24_row_view()
    _v23_log(@sprintf(
        "  context: %d anchors, %d usable (%d ADC-complete, lag-incomplete), %d base rows, %.1f s",
        length(ctx.anchors), count(usable), mask.adc_only, nrow(rowview.table),
        time() - started,
    ))

    folds = v24_folds(ctx, plan)
    store = V24Store(length(ctx.anchors))
    summary = NamedTuple[]
    run_manifest = V23Manifest()
    v23_manifest!(run_manifest, "plan", "label", NaN, plan.label)
    v23_manifest!(run_manifest, "plan", "years", length(plan.years),
                  join(sort(plan.years), ";"))
    v23_manifest!(run_manifest, "environment", "julia_threads", Threads.nthreads(), "")
    v23_manifest!(run_manifest, "environment", "feature_scaling", NaN, V24_FEATURE_SCALING)
    v23_manifest!(run_manifest, "rows", "anchors", length(ctx.anchors), "")
    v23_manifest!(run_manifest, "rows", "anchors_usable", count(usable), "")
    v23_manifest!(run_manifest, "rows", "anchors_adc_only", mask.adc_only, "")
    reuse_stale_reason === nothing || v23_manifest!(
        run_manifest, "plan", "reuse_stale_folds_reason", NaN, reuse_stale_reason,
    )

    for fold in folds
        fold_started = time()
        current = v24_fold_is_current(ctx, plan, fold)
        stale = !current && isfile(v24_oof_year_path(plan.outdir, fold.year)) &&
                reuse_stale_reason !== nothing
        if current || stale
            loaded = v24_load_fold!(store, ctx, plan, fold)
            state = current ? "reused" : "reused_stale"
            _v23_log(@sprintf("  = fold %d %s from disk: %d rows, %d anchors, %.1f s",
                              fold.year, state, loaded.rows, loaded.anchors,
                              time() - fold_started))
            stale && @warn "consuming a fold whose signature no longer matches" year =
                fold.year reason = reuse_stale_reason
            push!(summary, (year=fold.year, state=state, rows=loaded.rows,
                            seconds=time() - fold_started))
            v23_manifest!(run_manifest, "fold", "year_$(fold.year)", loaded.rows, state)
            continue
        end
        result = v24_run_fold!(ctx, plan, fold, store, rowview, usable, mask.features)
        push!(summary, (year=result.year, state="computed", rows=result.rows,
                        seconds=result.seconds))
        v23_manifest!(run_manifest, "fold", "year_$(fold.year)", result.rows,
                      @sprintf("computed;seconds=%.1f;fallback=%d;tau=%.2f;t1r_in_sample=%d",
                               result.seconds, result.fallback, result.tau_hours,
                               result.t1r_in_sample ? 1 : 0))
    end

    v23_manifest!(run_manifest, "seconds", "total", time() - total_started, "")
    for file in sort(readdir(plan.outdir))
        full = joinpath(plan.outdir, file)
        isfile(full) || continue
        startswith(file, "run_manifest") && continue
        v23_manifest!(run_manifest, "output_sha256", file, filesize(full),
                      _v23_sha256_file(full))
    end
    manifest_path = joinpath(plan.outdir, "run_manifest.csv")
    CSV.write(manifest_path, DataFrame(run_manifest.rows))
    CSV.write(joinpath(plan.outdir, "run_summary.csv"), DataFrame(summary))
    _v23_log(@sprintf("V2.4 rolling engine complete in %.1f s → %s",
                      time() - total_started, plan.outdir))
    return (summary=summary, outdir=plan.outdir, store=store, ctx=ctx, folds=folds,
            seconds=time() - total_started, manifest=manifest_path)
end

"Print the per-fold table a run returns."
function v24_print_summary(run)
    println("\n== V2.4 rolling folds ==")
    println(rpad("year", 6), rpad("state", 14), rpad("rows", 9), "seconds")
    for row in run.summary
        println(rpad(row.year, 6), rpad(row.state, 14), rpad(row.rows, 9),
                @sprintf("%.1f", row.seconds))
    end
    println(@sprintf("total %.1f s", run.seconds))
    return run
end

"Years requested on the command line, or every preregistered fold."
function _v24_years(arguments)
    for argument in arguments
        startswith(argument, "--years=") || continue
        text = argument[length("--years=") + 1:end]
        years = [parse(Int, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
        isempty(years) && error("--years needs at least one calendar year")
        return years
    end
    return collect(V24_FOLD_YEARS)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--smoke" in ARGS
        v24_print_summary(v24_run(v24_smoke_plan()))
    else
        v24_print_summary(v24_run(
            v24_full_plan(; years=_v24_years(ARGS));
            reuse_stale_reason=_v23_written_reason(ARGS, "--reuse-stale-folds"),
        ))
    end
end
