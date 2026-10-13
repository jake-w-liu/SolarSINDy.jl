module SuperlearnerRollingTests

# Unit oracles for the Operational super-learner rolling engine
# (`validation/operational/superlearner_rolling.jl`) and the fold contract
# (`validation/operational/superlearner_contract.jl`).
#
# The rolling study has four properties that no downstream table can repair, so
# each one is pinned here against an independently stated expectation:
#
#   1. the fold window is defined on *targets* — a training anchor whose target
#      lands one hour past the cutoff must be gone, not merely rare;
#   2. the out-of-fold pool a fold fits on contains strictly earlier years, and a
#      store that was filled with the wrong rows must fail rather than leak;
#   3. the direct comparator regresses an increment, so its target and its
#      inversion have to be exact inverses or every one of its centers is wrong
#      by the issue-time Dst;
#   4. a row the feature key cannot describe is served, never imputed, and the
#      persisted schema is exactly what Task B reads.
#
# The heavy identity — the smoke fold reproducing the archived analog ensemble development
# analog core on shared anchors — needs generated artifacts. It runs when they
# exist and is reported as skipped when they do not, in the same way as the
# base-table identity oracle.

using Test
using CSV
using DataFrames
using Dates
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "superlearner_rolling.jl")))

# ---------------------------------------------------------------------------
# Controlled anchor view and context (same shape as `test_analog_runners.jl`)
# ---------------------------------------------------------------------------

"""
Anchor view over `issues` with fully controlled observations and comparators.
Every issue-level quantity is explicit so that a test can predict the arithmetic
of the stage under test without consulting the base table.
"""
function tiny_anchors(issues::Vector{DateTime}; obs=nothing, served=nothing, frozen=nothing,
                      raw_served=nothing, latest=nothing, rate=nothing, coupling=nothing)
    count = length(issues)
    index = Dict(t => i for (i, t) in enumerate(issues))
    driver = [(V=400.0, Bz=-5.0, By=2.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
              for _ in 1:count]
    filled(value) = fill(Float64(value), count, ANALOG_STEP_COUNT)
    return AnalogAnchors(
        issues, index, fill("DEV", count), [Dates.year(t) for t in issues], driver,
        latest === nothing ? fill(-20.0, count) : latest,
        rate === nothing ? zeros(count) : rate,
        coupling === nothing ? zeros(count) : coupling,
        fill(-20.0, count), trues(count, ANALOG_STEP_COUNT),
        obs === nothing ? filled(-30.0) : obs,
        served === nothing ? filled(-25.0) : served,
        frozen === nothing ? filled(-24.0) : frozen,
        raw_served === nothing ? filled(-23.0) : raw_served,
        filled(-22.0), filled(-21.0), filled(-20.5), filled(-19.0), filled(-18.0),
        filled(0.0),
    )
end

function tiny_context(anchors::AnalogAnchors; outdir::AbstractString=mktempdir())
    plan = AnalogRunPlan("unit", outdir, true, [2015], 0, [SUPERLEARNER_ANALOG_WEIGHT_SET],
                      [SUPERLEARNER_ANALOG_K], [(4, 200)], [(4, 200)])
    count = length(anchors)
    features = zeros(count, ANALOG_FEATURE_COUNT)
    for i in 1:count
        features[i, analog_feature_index(:dst0)] = anchors.latest[i]
        features[i, analog_feature_index(:ddst1)] = anchors.rate[i]
    end
    return AnalogContext(plan, anchors, DataFrame(), Dict{DateTime,_ANALOG_DRIVER_NT}(),
                      nothing, nothing, features, trues(count), trues(count),
                      analog_cell_masks(anchors), nothing, "table-sha", "frame-sha",
                      "code-sha", "unit", 0.0)
end

"Hourly issue times over a closed interval."
hourly(first_issue::DateTime, last_issue::DateTime) =
    collect(first_issue:Hour(1):last_issue)

# ---------------------------------------------------------------------------
# 1. Fold window
# ---------------------------------------------------------------------------

@testset "the fold window is bounded by targets, not by issue ordering" begin
    cutoff = superlearner_fold_cutoff(2015)
    @test cutoff == DateTime(2014, 12, 25, 0)

    # The training rule admits `issue + 7 h <= cutoff`. The two anchors on either
    # side of that boundary are the whole rule, so they are stated explicitly.
    boundary = cutoff - Hour(ANALOG_MAX_STEP)
    issues = [boundary - Hour(1), boundary, boundary + Hour(1),
              DateTime(2015, 1, 1, 0), DateTime(2015, 6, 1, 0), DateTime(2016, 1, 1, 0)]
    rows = superlearner_fold_rows(issues, 2015)
    @test rows.cutoff == cutoff
    @test issues[rows.train] == [boundary - Hour(1), boundary]
    @test issues[rows.query] == [DateTime(2015, 1, 1, 0), DateTime(2015, 6, 1, 0)]

    # A one-hour shift of the boundary anchor removes it, which is what makes the
    # rule a target embargo rather than an ordering check.
    shifted = superlearner_fold_rows([boundary + Hour(1)], 2015)
    @test isempty(shifted.train)

    # Same rule read through the confirmatory embargo check, which measures the
    # gap between the last training target and the first scored issue.
    anchors = tiny_anchors(issues)
    @test _analog_assert_embargo(anchors, rows.train, rows.query) >=
          ANALOG_BASE_EMBARGO_HOURS

    # The 2013 seed fold and the era boundaries are protocol constants, not
    # choices this runner makes.
    @test first(SUPERLEARNER_FOLD_YEARS) == 2013
    @test collect(SUPERLEARNER_ERA_E1_YEARS) == collect(2014:2019)
    @test collect(SUPERLEARNER_ERA_E2_YEARS) == collect(2020:2025)
    @test !(2013 in SUPERLEARNER_ERA_ALL_YEARS)
end

@testset "folds are built for every requested year and refuse a broken window" begin
    issues = vcat(hourly(DateTime(2014, 12, 20, 0), DateTime(2014, 12, 31, 23)),
                  hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 5, 23)))
    ctx = tiny_context(tiny_anchors(issues))
    plan = superlearner_smoke_plan(; year=2015, max_anchors=3)
    folds = superlearner_folds(ctx, plan)
    @test length(folds) == 1
    fold = first(folds)
    @test fold.year == 2015
    @test length(fold.query) == 3                       # the cap applies to scored rows
    @test all(Dates.year(ctx.anchors.issue[i]) == 2015 for i in fold.query)
    @test maximum(ctx.anchors.issue[fold.train]) + Hour(ANALOG_MAX_STEP) <= fold.cutoff

    # A study run may not trim the grid, cap the anchors, cap the seed fit or
    # switch the shadow layers off; each of those is refused on its own.
    @test superlearner_assert_full_plan(superlearner_full_plan()).years == collect(SUPERLEARNER_FOLD_YEARS)
    trimmed_grid = SuperlearnerPlan("bad", plan.outdir, false, [2015], 0,
                           [SuperlearnerDirectConfig(4, 200, 64)], 0, 0, true)
    @test_throws ErrorException superlearner_assert_full_plan(trimmed_grid)
    capped = SuperlearnerPlan("bad", plan.outdir, false, [2015], 2000, collect(SUPERLEARNER_DIRECT_GRID), 0,
                     0, true)
    @test_throws ErrorException superlearner_assert_full_plan(capped)
    no_layers = SuperlearnerPlan("bad", plan.outdir, false, [2015], 0, collect(SUPERLEARNER_DIRECT_GRID), 0,
                        0, false)
    @test_throws ErrorException superlearner_assert_full_plan(no_layers)

    # A year the base table cannot serve is an error, not a silently skipped fold.
    @test_throws ErrorException superlearner_folds(ctx, superlearner_smoke_plan(; year=2017))
end

# ---------------------------------------------------------------------------
# 2. Out-of-fold accumulation
# ---------------------------------------------------------------------------

"Store filled with one anchor per month of `years`, all centers set to the year."
function seeded_store(years::Vector{Int})
    issues = DateTime[]
    for y in years, m in 1:12
        push!(issues, DateTime(y, m, 1, 0))
    end
    anchors = tiny_anchors(issues)
    store = SuperlearnerStore(length(issues))
    for y in years
        rows = [i for i in eachindex(issues) if Dates.year(issues[i]) == y]
        centers = fill(NaN, length(issues), ANALOG_STEP_COUNT)
        for i in rows, slot in 1:ANALOG_STEP_COUNT
            centers[i, slot] = Float64(y)
        end
        superlearner_store_add!(store, y, rows, fill(false, length(issues)), centers, centers,
                       centers)
    end
    return (store=store, anchors=anchors, issues=issues)
end

@testset "the out-of-fold pool holds strictly earlier years and cannot be doubled" begin
    seeded = seeded_store([2013, 2014])
    store, anchors = seeded.store, seeded.anchors
    @test store.years == [2013, 2014]

    pool = superlearner_prior_rows(store, anchors, 2015)
    @test length(pool) == 24
    @test all(anchors.issue[i] < DateTime(2015, 1, 1) for i in pool)
    @test issorted(pool)

    # Restricting to one year returns that year and nothing else.
    @test length(superlearner_prior_rows(store, anchors, 2015; years=[2014])) == 12
    @test all(Dates.year(anchors.issue[i]) == 2014
              for i in superlearner_prior_rows(store, anchors, 2015; years=[2014]))

    # The fold's own year, and any later year, are refused.
    @test_throws ErrorException superlearner_prior_rows(store, anchors, 2014)
    @test_throws ErrorException superlearner_prior_rows(store, anchors, 2013)
    @test_throws ErrorException superlearner_prior_rows(store, anchors, 2015; years=[2015])

    # Registering a year twice would double every later fold's fitting pool.
    @test_throws ErrorException superlearner_store_add!(
        store, 2014, Int[], fill(false, length(anchors)),
        fill(NaN, length(anchors), ANALOG_STEP_COUNT),
        fill(NaN, length(anchors), ANALOG_STEP_COUNT),
        fill(NaN, length(anchors), ANALOG_STEP_COUNT),
    )

    # A store whose row list disagrees with the calendar is caught by the pool,
    # not by the fit that would have consumed it.
    leaked = seeded_store([2013, 2014])
    later = findfirst(t -> Dates.year(t) == 2014, leaked.issues)
    push!(leaked.store.rows[2013], later)
    @test_throws ErrorException superlearner_prior_rows(leaked.store, leaked.anchors, 2014)

    # Served fallbacks are outside the fitting pool but still countable.
    masked = seeded_store([2013])
    masked.store.fallback[1] = true
    @test length(superlearner_prior_rows(masked.store, masked.anchors, 2015)) == 11
    @test length(superlearner_prior_rows(masked.store, masked.anchors, 2015; nonfallback=false)) == 12
end

@testset "an out-of-fold pool row whose target enters the embargo is excluded" begin
    # A December of hourly anchors around the fold-2016 cutoff: the pool may keep only
    # anchors whose longest target still clears Y-01-01T00 - 168 h.
    cutoff = superlearner_fold_cutoff(2016)
    @test cutoff == DateTime(2016, 1, 1) - Hour(ANALOG_BASE_EMBARGO_HOURS)
    issues = hourly(DateTime(2015, 12, 20, 0), DateTime(2015, 12, 31, 23))
    anchors = tiny_anchors(issues)
    store = SuperlearnerStore(length(issues))
    rows = collect(eachindex(issues))
    matrix = fill(-30.0, length(issues), ANALOG_STEP_COUNT)
    superlearner_store_add!(store, 2015, rows, fill(false, length(issues)), matrix, matrix, matrix)
    pool = superlearner_prior_rows(store, anchors, 2016)
    admissible = [i for i in rows if issues[i] + Hour(ANALOG_MAX_STEP) <= cutoff]
    @test pool == admissible
    @test !isempty(pool)
    @test length(pool) < length(rows)
    @test maximum(issues[pool]) + Hour(ANALOG_MAX_STEP) <= cutoff
    # Every dropped anchor is dropped for that reason and no other.
    for i in setdiff(rows, pool)
        @test issues[i] + Hour(ANALOG_MAX_STEP) > cutoff
        @test issues[i] < DateTime(2016, 1, 1)
    end
    # The last admissible anchor sits exactly on the bound, so the rule is not
    # accidentally strict by an hour.
    @test maximum(issues[pool]) == cutoff - Hour(ANALOG_MAX_STEP)
    # A pool row from the scored year itself is still a hard error, not a filter.
    forward = SuperlearnerStore(length(issues))
    superlearner_store_add!(forward, 2016, rows, fill(false, length(issues)), matrix, matrix, matrix)
    @test_throws ErrorException superlearner_prior_rows(forward, anchors, 2016)
end

@testset "the correction layer is fitted on prior-fold rows only" begin
    seeded = seeded_store([2013, 2014])
    ctx = tiny_context(seeded.anchors)
    issues = seeded.issues
    fold = SuperlearnerFold(2015, superlearner_fold_cutoff(2015), collect(eachindex(issues)), Int[])
    usable = trues(length(issues))

    fit = superlearner_t1r_fit_rows(seeded.store, ctx, fold, usable, superlearner_full_plan())
    @test fit.in_sample == false
    @test length(fit.pairs) == 24 * ANALOG_STEP_COUNT
    @test all(Dates.year(issues[i]) < 2015 for (i, _) in fit.pairs)
    @test all(isfinite, fit.raw)
    @test Set(fit.raw) == Set([2013.0, 2014.0])

    # A row without a persisted core cannot be fitted on, and disappears from the
    # fit instead of entering it as an imputed value.
    seeded.store.analog_raw[1, 1] = NaN
    trimmed = superlearner_t1r_fit_rows(seeded.store, ctx, fold, usable, superlearner_full_plan())
    @test length(trimmed.pairs) == 24 * ANALOG_STEP_COUNT - 1

    # With no accumulated history the layer must take the disclosed in-sample
    # branch, which needs a usable training anchor to fit on.
    empty_store = SuperlearnerStore(length(issues))
    @test_throws ErrorException superlearner_t1r_fit_rows(empty_store, ctx, fold,
                                                falses(length(issues)), superlearner_full_plan())
end

# ---------------------------------------------------------------------------
# 3. Direct comparator increment
# ---------------------------------------------------------------------------

@testset "the direct comparator target and its inversion are exact inverses" begin
    issues = hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 1, 9))
    count = length(issues)
    latest = [-10.0 * i for i in 1:count]
    obs = [(-10.0 * i) - 3.0 * slot for i in 1:count, slot in 1:ANALOG_STEP_COUNT]
    anchors = tiny_anchors(issues; obs=obs, latest=latest)
    rows = collect(1:count)

    for slot in 1:ANALOG_STEP_COUNT
        target = direct_gbm_target(anchors, rows, slot)
        @test target == [anchors.obs[i, slot] - anchors.latest[i] for i in rows]
        for (j, i) in enumerate(rows)
            @test direct_gbm_center(anchors, i, target[j]) == anchors.obs[i, slot]
        end
        # Dropping the anchor is exactly the failure mode the increment target
        # exists to prevent: the center would be wrong by the issue-time Dst.
        @test all(abs(target[j] - anchors.obs[i, slot]) == abs(anchors.latest[i])
                  for (j, i) in enumerate(rows))
    end

    # A level target would have been the alternative; the runner records the
    # increment convention so a loader cannot use a persisted model as a center.
    @test DIRECT_GBM_TARGET == "increment"
    @test DIRECT_GBM_TARGET_ANCHOR == "latest_dst_nt"

    # The contract's expert list is what Task B fits weights over, so it must match
    # the learner's ten-expert order of Amendment A3 exactly.
    @test SUPERLEARNER_EXPERT_COLUMNS ==
          ("served_inertial_sindy", "frozen_inertial_sindy", "t1r_analog", "persistence", "burton",
           "burton_full", "obrien", "direct_gbm", "climatology", "static_stack")
    @test all(name -> name in SUPERLEARNER_MODEL_COLUMNS, SUPERLEARNER_EXPERT_COLUMNS)
    @test !("static_stack" in SUPERLEARNER_COMPARATOR_COLUMNS)

    # The preregistered grid carries the histogram resolution, and the 24-month
    # inner window is what selects among its points. Plan section 4 fixes the
    # full product depth {4,6} x rounds {200,400} x nbins {64,255}: a subset
    # would tune the strongest comparator on less than it was promised.
    @test length(SUPERLEARNER_DIRECT_GRID) == 8
    @test Set((c.depth, c.rounds, c.nbins) for c in SUPERLEARNER_DIRECT_GRID) ==
          Set((d, r, b) for d in (4, 6), r in (200, 400), b in (64, 255))
    @test SUPERLEARNER_DIRECT_INNER_MONTHS == 24
    @test Set(vcat([collect(applies) for (_, applies) in SUPERLEARNER_DIRECT_SELECTION]...)) ==
          Set(ANALOG_MODEL_STEPS)
end

# ---------------------------------------------------------------------------
# 4. Fallback convention and the persisted schema
# ---------------------------------------------------------------------------

@testset "an unusable row is served, never imputed" begin
    issues = hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 1, 5))
    count = length(issues)
    served = fill(-55.0, count, ANALOG_STEP_COUNT)
    raw_served = fill(-44.0, count, ANALOG_STEP_COUNT)
    anchors = tiny_anchors(issues; served=served, raw_served=raw_served)
    usable = trues(count)
    usable[2] = false
    usable[5] = false
    rows = collect(1:count)

    centers = fill(-1.0, count, ANALOG_STEP_COUNT)
    superlearner_serve_fallback!(centers, anchors, rows, usable)
    @test all(centers[i, :] == served[i, :] for i in (2, 5))
    @test all(centers[i, :] == fill(-1.0, ANALOG_STEP_COUNT) for i in (1, 3, 4, 6))

    # The reported raw core follows the same convention and stays inside the
    # physical range, so a runaway ensemble mean cannot be written to a fold file.
    raw = fill(-31.0, count, ANALOG_STEP_COUNT)
    raw[3, 1] = 5.0e3
    raw[4, 2] = -5.0e3
    result = AnalogConfigResult("T1_magnetic_K25", "T1", Dict{String,Any}(), raw,
                             fill(false, count), rows, true)
    reported = superlearner_reported_raw(result, anchors, rows, usable)
    @test all(reported[i, :] == raw_served[i, :] for i in (2, 5))
    @test reported[1, 1] == -31.0
    @test reported[3, 1] == 50.0
    @test reported[4, 2] == -2000.0
end

"One synthetic fold file that satisfies the contract."
function synthetic_fold_file(directory::AbstractString, test_year::Int; rows::Int=12)
    issues = [DateTime(test_year, 1, 1, 0) + Hour(div(r - 1, ANALOG_STEP_COUNT))
              for r in 1:rows]
    data = Dict{String,Vector}()
    data["issue_time_utc"] = [string(t) for t in issues]
    data["model_step_hours"] = [ANALOG_MODEL_STEPS[mod1(r, ANALOG_STEP_COUNT)] for r in 1:rows]
    data["fallback"] = fill(false, rows)
    for name in ("observation_dst_nt", "latest_dst_nt", "dst_delta_1h_nt",
                 "coupling_active_mvm")
        data[name] = [-1.0 * r for r in 1:rows]
    end
    for name in SUPERLEARNER_MODEL_COLUMNS
        data[name] = [-10.0 - r for r in 1:rows]
    end
    for name in SUPERLEARNER_FEATURE_COLUMNS
        data[name] = [0.5 * r for r in 1:rows]
    end
    path = superlearner_oof_year_path(directory, test_year)
    CSV.write(path, DataFrame([name => data[name] for name in SUPERLEARNER_OOF_COLUMNS]))
    return (path=path, data=data, issues=issues)
end

@testset "the fold contract accepts the schema and names every breach" begin
    directory = mktempdir()
    file = synthetic_fold_file(directory, 2015)
    ok = superlearner_validate_oof_year(file.path; expect_year=2015)
    @test ok.ok
    @test isempty(ok.problems)
    @test ok.year == 2015
    @test ok.n_rows == 12
    @test ok.steps == sort(collect(ANALOG_MODEL_STEPS))
    @test ok.n_fallback == 0
    @test isempty(ok.extra_columns)
    @test superlearner_require_oof_year(file.path; expect_year=2015).ok

    # Every expert Task B stacks over, and every comparator it scores, is present.
    # Amendment A3 made the static stack the tenth expert.
    @test length(SUPERLEARNER_EXPERT_COLUMNS) == 10
    @test all(name in SUPERLEARNER_OOF_COLUMNS for name in SUPERLEARNER_EXPERT_COLUMNS)
    @test all(name in SUPERLEARNER_OOF_COLUMNS for name in SUPERLEARNER_COMPARATOR_COLUMNS)
    @test length(SUPERLEARNER_FEATURE_COLUMNS) == 29
    @test SUPERLEARNER_FEATURE_COLUMNS ==
          Tuple("f_" * String(name) for name in SolarSINDy.direct_gbm_feature_names())

    # A missing expert column is named, not tolerated.
    table = CSV.read(file.path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    trimmed = joinpath(directory, "trimmed.csv")
    CSV.write(trimmed, select(table, Not(:direct_gbm)))
    dropped = superlearner_validate_oof_year(trimmed)
    @test !dropped.ok
    @test dropped.missing_columns == ["direct_gbm"]
    @test_throws ErrorException superlearner_require_oof_year(trimmed)

    # A non-finite forecast is a breach wherever it appears.
    holed = joinpath(directory, "holed.csv")
    broken = copy(table)
    broken.t1r_analog[3] = NaN
    CSV.write(holed, broken)
    @test !superlearner_validate_oof_year(holed).ok
    @test any(occursin("t1r_analog", p) for p in superlearner_validate_oof_year(holed).problems)

    # A feature is allowed to be empty only on a served-fallback row, which is
    # what "never imputed" means in the schema.
    featured = joinpath(directory, "featured.csv")
    partial = copy(table)
    partial.f_bz0[4] = NaN
    CSV.write(featured, partial)
    @test !superlearner_validate_oof_year(featured).ok
    partial.fallback[4] = true
    CSV.write(featured, partial)
    @test superlearner_validate_oof_year(featured).ok
    @test superlearner_validate_oof_year(featured).n_fallback == 1

    # One file is one calendar year.
    mixed = joinpath(directory, "mixed.csv")
    spread = copy(table)
    spread.issue_time_utc[1] = DateTime(2016, 1, 1)
    CSV.write(mixed, spread)
    @test !superlearner_validate_oof_year(mixed).ok
    @test !superlearner_validate_oof_year(file.path; expect_year=2016).ok

    # A repeated (issue, step) row would double-count that observation.
    repeated = joinpath(directory, "repeated.csv")
    CSV.write(repeated, vcat(table, table[1:1, :]))
    @test !superlearner_validate_oof_year(repeated).ok
    @test any(occursin("repeated", p) for p in superlearner_validate_oof_year(repeated).problems)
end

@testset "a fold file rebuilds the store it came from" begin
    directory = mktempdir()
    file = synthetic_fold_file(directory, 2015)
    anchors = tiny_anchors(unique(file.issues))
    ctx = tiny_context(anchors)
    plan = SuperlearnerPlan("unit", directory, true, [2015], 0, [SuperlearnerDirectConfig(4, 200, 64)], 0, 0,
                   false)
    store = SuperlearnerStore(length(anchors))
    loaded = superlearner_load_fold!(store, ctx, plan, SuperlearnerFold(2015, superlearner_fold_cutoff(2015), Int[],
                                                      collect(eachindex(anchors.issue))))
    @test loaded.rows == 12
    @test store.years == [2015]
    # The three columns a later fold fits on come back exactly as written.
    expected = sort([-10.0 - r for r in 1:12])
    @test sort(collect(filter(isfinite, store.t1r))) == expected
    @test sort(collect(filter(isfinite, store.lat))) == expected
    @test sort(collect(filter(isfinite, store.analog_raw))) == expected
    @test superlearner_prior_rows(store, anchors, 2016) == store.rows[2015]
    @test !any(store.fallback)
end

@testset "a --years subset never fits a later fold on a truncated history (V06-02)" begin
    issues = vcat([hourly(DateTime(y, 12, 20, 0), DateTime(y, 12, 31, 23)) for y in 2012:2015]...,
                  hourly(DateTime(2016, 1, 1, 0), DateTime(2016, 1, 5, 23)))
    ctx = tiny_context(tiny_anchors(issues))
    outdir = mktempdir()
    plan = SuperlearnerPlan("superlearner_rolling", outdir, false, [2015, 2016], 0, collect(SUPERLEARNER_DIRECT_GRID),
                   0, 0, true)
    folds = superlearner_run_folds(ctx, plan)
    # The earlier preregistered folds are walked as history, in year order.
    @test [f.year for f in folds] == [2013, 2014, 2015, 2016]
    # On an empty tree an unrequested history fold is refused, not computed or skipped.
    @test_throws ErrorException superlearner_fold_action(ctx, plan, folds[1], Int[])
    @test superlearner_fold_action(ctx, plan, folds[3], [2013, 2014]) === :compute

    # The signature records the out-of-fold history the fold was fitted on: a fold a subset
    # run computed as a seed (no prior years) is not current for the full chain.
    fold_2015 = folds[3]
    seed_signature = superlearner_fold_signature(ctx, plan, fold_2015, Int[])
    @test seed_signature != superlearner_fold_signature(ctx, plan, fold_2015, [2013, 2014])
    write_fold_stub(year, prior) = begin
        write(superlearner_oof_year_path(outdir, year), "stub")
        write(superlearner_manifest_year_path(outdir, year), "stub")
        write(superlearner_signature_year_path(outdir, year),
              superlearner_fold_signature(ctx, plan, folds[year - 2012], prior))
    end
    write_fold_stub(2015, Int[])
    @test superlearner_fold_is_current(ctx, plan, fold_2015, Int[])
    @test !superlearner_fold_is_current(ctx, plan, fold_2015, [2013, 2014])
    @test superlearner_fold_action(ctx, plan, fold_2015, [2013, 2014]) === :compute

    # History folds that are current on disk are reused.
    write_fold_stub(2013, Int[])
    write_fold_stub(2014, [2013])
    @test superlearner_fold_action(ctx, plan, folds[1], Int[]) === :reused
    @test superlearner_fold_action(ctx, plan, folds[2], [2013]) === :reused
    # ...but a history fold persisted on another history is not.
    @test_throws ErrorException superlearner_fold_action(ctx, plan, folds[2], Int[])

    # A plan that starts at the seed year walks only what it asks for; the smoke plan stays a
    # single seed fold in its own tree.
    seeded = SuperlearnerPlan("superlearner_rolling", outdir, false, [2013, 2014], 0,
                     collect(SUPERLEARNER_DIRECT_GRID), 0, 0, true)
    @test [f.year for f in superlearner_run_folds(ctx, seeded)] == [2013, 2014]
    @test [f.year for f in superlearner_run_folds(ctx, superlearner_smoke_plan(; year=2015))] == [2015]
end

# ---------------------------------------------------------------------------
# 5. Artifact identity: the smoke fold against the archived analog ensemble core
# ---------------------------------------------------------------------------

const SUPERLEARNER_SMOKE_OOF = superlearner_oof_year_path(SUPERLEARNER_ROLLING_SMOKE_DIR, 2015)
const SUPERLEARNER_DEV_ANALOG_OOF = joinpath(ANALOG_DEV_DIR, "oof_T1_magnetic_K25.csv")

@testset "the smoke fold reproduces the archived analog ensemble core" begin
    if !(isfile(SUPERLEARNER_SMOKE_OOF) && isfile(SUPERLEARNER_DEV_ANALOG_OOF))
        @info "super-learner smoke identity skipped: generated artifact missing" smoke =
            SUPERLEARNER_SMOKE_OOF archive = SUPERLEARNER_DEV_ANALOG_OOF
    else
        smoke = superlearner_require_oof_year(SUPERLEARNER_SMOKE_OOF; expect_year=2015)
        @test smoke.ok
        rolling = CSV.read(SUPERLEARNER_SMOKE_OOF, DataFrame;
                           select=["issue_time_utc", "model_step_hours", "fallback",
                                   "t1_analog_raw"],
                           types=Dict("issue_time_utc" => DateTime))
        archived = CSV.read(SUPERLEARNER_DEV_ANALOG_OOF, DataFrame;
                            types=Dict("issue_time_utc" => DateTime))
        reference = Dict{Tuple{DateTime,Int},Float64}()
        for r in 1:nrow(archived)
            Bool(archived.fallback[r]) && continue
            reference[(archived.issue_time_utc[r], Int(archived.model_step_hours[r]))] =
                Float64(archived.raw_dst_nt[r])
        end
        shared = 0
        worst = 0.0
        for r in 1:nrow(rolling)
            Bool(rolling.fallback[r]) && continue
            key = (rolling.issue_time_utc[r], Int(rolling.model_step_hours[r]))
            haskey(reference, key) || continue
            shared += 1
            worst = max(worst, abs(Float64(rolling.t1_analog_raw[r]) - reference[key]))
        end
        # The rolling fold for 2015 and the archived inner block 2015 search the
        # same archive under the same rule, so the analog raw core is the same
        # number; a different archive, a different K or a different weight set
        # would move it by far more than the tolerance.
        @test shared > 1_000
        @test worst <= 1e-9

        manifest = CSV.read(superlearner_manifest_year_path(SUPERLEARNER_ROLLING_SMOKE_DIR, 2015), DataFrame)
        cutoff = only(manifest.value[manifest.name .== "training_max_target_utc"])
        @test DateTime(cutoff) == superlearner_fold_cutoff(2015)
        embargo = only(manifest.count[manifest.name .== "embargo_hours"])
        @test embargo >= ANALOG_BASE_EMBARGO_HOURS
        last_target = only(manifest.value[manifest.name .== "training_last_target_utc"])
        @test DateTime(last_target) <= superlearner_fold_cutoff(2015)
    end
end

@testset "every persisted study fold satisfies the contract" begin
    files = isdir(SUPERLEARNER_ROLLING_DIR) ?
        [f for f in readdir(SUPERLEARNER_ROLLING_DIR) if startswith(f, "oof_year_")] : String[]
    if isempty(files)
        @info "super-learner study folds skipped: no rolling artifact present" directory =
            SUPERLEARNER_ROLLING_DIR
    else
        for file in sort(files)
            test_year = parse(Int, match(r"^oof_year_(\d+)\.csv$", file).captures[1])
            @test test_year in SUPERLEARNER_FOLD_YEARS
            check = superlearner_require_oof_year(joinpath(SUPERLEARNER_ROLLING_DIR, file);
                                         expect_year=test_year)
            @test check.ok
            @test check.n_rows > 0
            @test check.n_issues * length(ANALOG_MODEL_STEPS) >= check.n_rows

            manifest_path = superlearner_manifest_year_path(SUPERLEARNER_ROLLING_DIR, test_year)
            @test isfile(manifest_path)
            manifest = CSV.read(manifest_path, DataFrame)
            last_target = only(manifest.value[manifest.name .== "training_last_target_utc"])
            @test DateTime(last_target) <= superlearner_fold_cutoff(test_year)
            scored_first = only(manifest.value[manifest.name .== "scored_first_issue_utc"])
            @test Dates.year(DateTime(scored_first)) == test_year

            # The fold records which direct-GBM grid it searched, and the inner
            # validation must have scored every point of it at both selection
            # steps: a fold written under a trimmed grid would tune the strongest
            # comparator on less than plan section 4 promises.
            grid = only(manifest.value[manifest.name .== "direct_gbm_grid_points"])
            @test only(manifest.count[manifest.name .== "direct_gbm_grid_points"]) ==
                  length(SUPERLEARNER_DIRECT_GRID)
            @test Set(split(grid, ";")) ==
                  Set(superlearner_direct_label(config) for config in SUPERLEARNER_DIRECT_GRID)
            inner = manifest[manifest.entry_type .== "direct_gbm_inner", :]
            @test nrow(inner) == length(SUPERLEARNER_DIRECT_SELECTION) * length(SUPERLEARNER_DIRECT_GRID)
            for (selection_step, _) in SUPERLEARNER_DIRECT_SELECTION, config in SUPERLEARNER_DIRECT_GRID
                @test any(inner.name .== "step$(selection_step)_$(superlearner_direct_label(config))")
            end
            for (_, applies) in SUPERLEARNER_DIRECT_SELECTION, step in applies
                chosen = only(manifest.value[manifest.name .== "direct_gbm_step$(step)"])
                @test chosen in [superlearner_direct_label(c) for c in SUPERLEARNER_DIRECT_GRID]
            end
        end
    end
end

end # module
