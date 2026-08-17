module V24RollingTests

# Unit oracles for the Operational V2.4 rolling engine
# (`validation/operational/v2_4_rolling.jl`) and the fold contract
# (`validation/operational/v2_4_contract.jl`).
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
# The heavy identity — the smoke fold reproducing the archived V2.3 development
# analog core on shared anchors — needs generated artifacts. It runs when they
# exist and is reported as skipped when they do not, in the same way as the
# base-table identity oracle.

using Test
using CSV
using DataFrames
using Dates
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "v2_4_rolling.jl")))

# ---------------------------------------------------------------------------
# Controlled anchor view and context (same shape as `test_v2_3_runners.jl`)
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
    filled(value) = fill(Float64(value), count, V23_STEP_COUNT)
    return V23Anchors(
        issues, index, fill("DEV", count), [Dates.year(t) for t in issues], driver,
        latest === nothing ? fill(-20.0, count) : latest,
        rate === nothing ? zeros(count) : rate,
        coupling === nothing ? zeros(count) : coupling,
        fill(-20.0, count), trues(count, V23_STEP_COUNT),
        obs === nothing ? filled(-30.0) : obs,
        served === nothing ? filled(-25.0) : served,
        frozen === nothing ? filled(-24.0) : frozen,
        raw_served === nothing ? filled(-23.0) : raw_served,
        filled(-22.0), filled(-21.0), filled(-20.5), filled(-19.0), filled(-18.0),
        filled(0.0),
    )
end

function tiny_context(anchors::V23Anchors; outdir::AbstractString=mktempdir())
    plan = V23RunPlan("unit", outdir, true, [2015], 0, [V24_ANALOG_WEIGHT_SET],
                      [V24_ANALOG_K], [(4, 200)], [(4, 200)])
    count = length(anchors)
    features = zeros(count, V23_FEATURE_COUNT)
    for i in 1:count
        features[i, v23_feature_index(:dst0)] = anchors.latest[i]
        features[i, v23_feature_index(:ddst1)] = anchors.rate[i]
    end
    return V23Context(plan, anchors, DataFrame(), Dict{DateTime,_V23_DRIVER_NT}(),
                      nothing, nothing, features, trues(count), trues(count),
                      v23_cell_masks(anchors), nothing, "table-sha", "frame-sha",
                      "code-sha", "unit", 0.0)
end

"Hourly issue times over a closed interval."
hourly(first_issue::DateTime, last_issue::DateTime) =
    collect(first_issue:Hour(1):last_issue)

# ---------------------------------------------------------------------------
# 1. Fold window
# ---------------------------------------------------------------------------

@testset "the fold window is bounded by targets, not by issue ordering" begin
    cutoff = v24_fold_cutoff(2015)
    @test cutoff == DateTime(2014, 12, 25, 0)

    # The training rule admits `issue + 7 h <= cutoff`. The two anchors on either
    # side of that boundary are the whole rule, so they are stated explicitly.
    boundary = cutoff - Hour(V23_MAX_STEP)
    issues = [boundary - Hour(1), boundary, boundary + Hour(1),
              DateTime(2015, 1, 1, 0), DateTime(2015, 6, 1, 0), DateTime(2016, 1, 1, 0)]
    rows = v24_fold_rows(issues, 2015)
    @test rows.cutoff == cutoff
    @test issues[rows.train] == [boundary - Hour(1), boundary]
    @test issues[rows.query] == [DateTime(2015, 1, 1, 0), DateTime(2015, 6, 1, 0)]

    # A one-hour shift of the boundary anchor removes it, which is what makes the
    # rule a target embargo rather than an ordering check.
    shifted = v24_fold_rows([boundary + Hour(1)], 2015)
    @test isempty(shifted.train)

    # Same rule read through the confirmatory embargo check, which measures the
    # gap between the last training target and the first scored issue.
    anchors = tiny_anchors(issues)
    @test _v23_assert_embargo(anchors, rows.train, rows.query) >=
          V23_BASE_EMBARGO_HOURS

    # The 2013 seed fold and the era boundaries are protocol constants, not
    # choices this runner makes.
    @test first(V24_FOLD_YEARS) == 2013
    @test collect(V24_ERA_E1_YEARS) == collect(2014:2019)
    @test collect(V24_ERA_E2_YEARS) == collect(2020:2025)
    @test !(2013 in V24_ERA_ALL_YEARS)
end

@testset "folds are built for every requested year and refuse a broken window" begin
    issues = vcat(hourly(DateTime(2014, 12, 20, 0), DateTime(2014, 12, 31, 23)),
                  hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 5, 23)))
    ctx = tiny_context(tiny_anchors(issues))
    plan = v24_smoke_plan(; year=2015, max_anchors=3)
    folds = v24_folds(ctx, plan)
    @test length(folds) == 1
    fold = first(folds)
    @test fold.year == 2015
    @test length(fold.query) == 3                       # the cap applies to scored rows
    @test all(Dates.year(ctx.anchors.issue[i]) == 2015 for i in fold.query)
    @test maximum(ctx.anchors.issue[fold.train]) + Hour(V23_MAX_STEP) <= fold.cutoff

    # A study run may not trim the grid, cap the anchors, cap the seed fit or
    # switch the shadow layers off; each of those is refused on its own.
    @test v24_assert_full_plan(v24_full_plan()).years == collect(V24_FOLD_YEARS)
    trimmed_grid = V24Plan("bad", plan.outdir, false, [2015], 0,
                           [V24DirectConfig(4, 200, 64)], 0, 0, true)
    @test_throws ErrorException v24_assert_full_plan(trimmed_grid)
    capped = V24Plan("bad", plan.outdir, false, [2015], 2000, collect(V24_DIRECT_GRID), 0,
                     0, true)
    @test_throws ErrorException v24_assert_full_plan(capped)
    no_layers = V24Plan("bad", plan.outdir, false, [2015], 0, collect(V24_DIRECT_GRID), 0,
                        0, false)
    @test_throws ErrorException v24_assert_full_plan(no_layers)

    # A year the base table cannot serve is an error, not a silently skipped fold.
    @test_throws ErrorException v24_folds(ctx, v24_smoke_plan(; year=2017))
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
    store = V24Store(length(issues))
    for y in years
        rows = [i for i in eachindex(issues) if Dates.year(issues[i]) == y]
        centers = fill(NaN, length(issues), V23_STEP_COUNT)
        for i in rows, slot in 1:V23_STEP_COUNT
            centers[i, slot] = Float64(y)
        end
        v24_store_add!(store, y, rows, fill(false, length(issues)), centers, centers,
                       centers)
    end
    return (store=store, anchors=anchors, issues=issues)
end

@testset "the out-of-fold pool holds strictly earlier years and cannot be doubled" begin
    seeded = seeded_store([2013, 2014])
    store, anchors = seeded.store, seeded.anchors
    @test store.years == [2013, 2014]

    pool = v24_prior_rows(store, anchors, 2015)
    @test length(pool) == 24
    @test all(anchors.issue[i] < DateTime(2015, 1, 1) for i in pool)
    @test issorted(pool)

    # Restricting to one year returns that year and nothing else.
    @test length(v24_prior_rows(store, anchors, 2015; years=[2014])) == 12
    @test all(Dates.year(anchors.issue[i]) == 2014
              for i in v24_prior_rows(store, anchors, 2015; years=[2014]))

    # The fold's own year, and any later year, are refused.
    @test_throws ErrorException v24_prior_rows(store, anchors, 2014)
    @test_throws ErrorException v24_prior_rows(store, anchors, 2013)
    @test_throws ErrorException v24_prior_rows(store, anchors, 2015; years=[2015])

    # Registering a year twice would double every later fold's fitting pool.
    @test_throws ErrorException v24_store_add!(
        store, 2014, Int[], fill(false, length(anchors)),
        fill(NaN, length(anchors), V23_STEP_COUNT),
        fill(NaN, length(anchors), V23_STEP_COUNT),
        fill(NaN, length(anchors), V23_STEP_COUNT),
    )

    # A store whose row list disagrees with the calendar is caught by the pool,
    # not by the fit that would have consumed it.
    leaked = seeded_store([2013, 2014])
    later = findfirst(t -> Dates.year(t) == 2014, leaked.issues)
    push!(leaked.store.rows[2013], later)
    @test_throws ErrorException v24_prior_rows(leaked.store, leaked.anchors, 2014)

    # Served fallbacks are outside the fitting pool but still countable.
    masked = seeded_store([2013])
    masked.store.fallback[1] = true
    @test length(v24_prior_rows(masked.store, masked.anchors, 2015)) == 11
    @test length(v24_prior_rows(masked.store, masked.anchors, 2015; nonfallback=false)) == 12
end

@testset "an out-of-fold pool row whose target enters the embargo is excluded" begin
    # A December of hourly anchors around the fold-2016 cutoff: the pool may keep only
    # anchors whose longest target still clears Y-01-01T00 - 168 h.
    cutoff = v24_fold_cutoff(2016)
    @test cutoff == DateTime(2016, 1, 1) - Hour(V23_BASE_EMBARGO_HOURS)
    issues = hourly(DateTime(2015, 12, 20, 0), DateTime(2015, 12, 31, 23))
    anchors = tiny_anchors(issues)
    store = V24Store(length(issues))
    rows = collect(eachindex(issues))
    matrix = fill(-30.0, length(issues), V23_STEP_COUNT)
    v24_store_add!(store, 2015, rows, fill(false, length(issues)), matrix, matrix, matrix)
    pool = v24_prior_rows(store, anchors, 2016)
    admissible = [i for i in rows if issues[i] + Hour(V23_MAX_STEP) <= cutoff]
    @test pool == admissible
    @test !isempty(pool)
    @test length(pool) < length(rows)
    @test maximum(issues[pool]) + Hour(V23_MAX_STEP) <= cutoff
    # Every dropped anchor is dropped for that reason and no other.
    for i in setdiff(rows, pool)
        @test issues[i] + Hour(V23_MAX_STEP) > cutoff
        @test issues[i] < DateTime(2016, 1, 1)
    end
    # The last admissible anchor sits exactly on the bound, so the rule is not
    # accidentally strict by an hour.
    @test maximum(issues[pool]) == cutoff - Hour(V23_MAX_STEP)
    # A pool row from the scored year itself is still a hard error, not a filter.
    forward = V24Store(length(issues))
    v24_store_add!(forward, 2016, rows, fill(false, length(issues)), matrix, matrix, matrix)
    @test_throws ErrorException v24_prior_rows(forward, anchors, 2016)
end

@testset "the correction layer is fitted on prior-fold rows only" begin
    seeded = seeded_store([2013, 2014])
    ctx = tiny_context(seeded.anchors)
    issues = seeded.issues
    fold = V24Fold(2015, v24_fold_cutoff(2015), collect(eachindex(issues)), Int[])
    usable = trues(length(issues))

    fit = v24_t1r_fit_rows(seeded.store, ctx, fold, usable, v24_full_plan())
    @test fit.in_sample == false
    @test length(fit.pairs) == 24 * V23_STEP_COUNT
    @test all(Dates.year(issues[i]) < 2015 for (i, _) in fit.pairs)
    @test all(isfinite, fit.raw)
    @test Set(fit.raw) == Set([2013.0, 2014.0])

    # A row without a persisted core cannot be fitted on, and disappears from the
    # fit instead of entering it as an imputed value.
    seeded.store.analog_raw[1, 1] = NaN
    trimmed = v24_t1r_fit_rows(seeded.store, ctx, fold, usable, v24_full_plan())
    @test length(trimmed.pairs) == 24 * V23_STEP_COUNT - 1

    # With no accumulated history the layer must take the disclosed in-sample
    # branch, which needs a usable training anchor to fit on.
    empty_store = V24Store(length(issues))
    @test_throws ErrorException v24_t1r_fit_rows(empty_store, ctx, fold,
                                                falses(length(issues)), v24_full_plan())
end

# ---------------------------------------------------------------------------
# 3. Direct comparator increment
# ---------------------------------------------------------------------------

@testset "the direct comparator target and its inversion are exact inverses" begin
    issues = hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 1, 9))
    count = length(issues)
    latest = [-10.0 * i for i in 1:count]
    obs = [(-10.0 * i) - 3.0 * slot for i in 1:count, slot in 1:V23_STEP_COUNT]
    anchors = tiny_anchors(issues; obs=obs, latest=latest)
    rows = collect(1:count)

    for slot in 1:V23_STEP_COUNT
        target = v23_direct_target(anchors, rows, slot)
        @test target == [anchors.obs[i, slot] - anchors.latest[i] for i in rows]
        for (j, i) in enumerate(rows)
            @test v23_direct_center(anchors, i, target[j]) == anchors.obs[i, slot]
        end
        # Dropping the anchor is exactly the failure mode the increment target
        # exists to prevent: the center would be wrong by the issue-time Dst.
        @test all(abs(target[j] - anchors.obs[i, slot]) == abs(anchors.latest[i])
                  for (j, i) in enumerate(rows))
    end

    # A level target would have been the alternative; the runner records the
    # increment convention so a loader cannot use a persisted model as a center.
    @test V23_DIRECT_TARGET == "increment"
    @test V23_DIRECT_TARGET_ANCHOR == "latest_dst_nt"

    # The contract's expert list is what Task B fits weights over, so it must match
    # the learner's ten-expert order of Amendment A3 exactly.
    @test V24_EXPERT_COLUMNS ==
          ("served_v2_1", "frozen_v2_1", "t1r_analog", "persistence", "burton",
           "burton_full", "obrien", "direct_gbm", "climatology", "static_v2_2")
    @test all(name -> name in V24_MODEL_COLUMNS, V24_EXPERT_COLUMNS)
    @test !("static_v2_2" in V24_COMPARATOR_COLUMNS)

    # The preregistered grid carries the histogram resolution, and the 24-month
    # inner window is what selects among its points. Plan section 4 fixes the
    # full product depth {4,6} x rounds {200,400} x nbins {64,255}: a subset
    # would tune the strongest comparator on less than it was promised.
    @test length(V24_DIRECT_GRID) == 8
    @test Set((c.depth, c.rounds, c.nbins) for c in V24_DIRECT_GRID) ==
          Set((d, r, b) for d in (4, 6), r in (200, 400), b in (64, 255))
    @test V24_DIRECT_INNER_MONTHS == 24
    @test Set(vcat([collect(applies) for (_, applies) in V24_DIRECT_SELECTION]...)) ==
          Set(V23_MODEL_STEPS)
end

# ---------------------------------------------------------------------------
# 4. Fallback convention and the persisted schema
# ---------------------------------------------------------------------------

@testset "an unusable row is served, never imputed" begin
    issues = hourly(DateTime(2015, 1, 1, 0), DateTime(2015, 1, 1, 5))
    count = length(issues)
    served = fill(-55.0, count, V23_STEP_COUNT)
    raw_served = fill(-44.0, count, V23_STEP_COUNT)
    anchors = tiny_anchors(issues; served=served, raw_served=raw_served)
    usable = trues(count)
    usable[2] = false
    usable[5] = false
    rows = collect(1:count)

    centers = fill(-1.0, count, V23_STEP_COUNT)
    v24_serve_fallback!(centers, anchors, rows, usable)
    @test all(centers[i, :] == served[i, :] for i in (2, 5))
    @test all(centers[i, :] == fill(-1.0, V23_STEP_COUNT) for i in (1, 3, 4, 6))

    # The reported raw core follows the same convention and stays inside the
    # physical range, so a runaway ensemble mean cannot be written to a fold file.
    raw = fill(-31.0, count, V23_STEP_COUNT)
    raw[3, 1] = 5.0e3
    raw[4, 2] = -5.0e3
    result = V23ConfigResult("T1_magnetic_K25", "T1", Dict{String,Any}(), raw,
                             fill(false, count), rows, true)
    reported = v24_reported_raw(result, anchors, rows, usable)
    @test all(reported[i, :] == raw_served[i, :] for i in (2, 5))
    @test reported[1, 1] == -31.0
    @test reported[3, 1] == 50.0
    @test reported[4, 2] == -2000.0
end

"One synthetic fold file that satisfies the contract."
function synthetic_fold_file(directory::AbstractString, test_year::Int; rows::Int=12)
    issues = [DateTime(test_year, 1, 1, 0) + Hour(div(r - 1, V23_STEP_COUNT))
              for r in 1:rows]
    data = Dict{String,Vector}()
    data["issue_time_utc"] = [string(t) for t in issues]
    data["model_step_hours"] = [V23_MODEL_STEPS[mod1(r, V23_STEP_COUNT)] for r in 1:rows]
    data["fallback"] = fill(false, rows)
    for name in ("observation_dst_nt", "latest_dst_nt", "dst_delta_1h_nt",
                 "coupling_active_mvm")
        data[name] = [-1.0 * r for r in 1:rows]
    end
    for name in V24_MODEL_COLUMNS
        data[name] = [-10.0 - r for r in 1:rows]
    end
    for name in V24_FEATURE_COLUMNS
        data[name] = [0.5 * r for r in 1:rows]
    end
    path = v24_oof_year_path(directory, test_year)
    CSV.write(path, DataFrame([name => data[name] for name in V24_OOF_COLUMNS]))
    return (path=path, data=data, issues=issues)
end

@testset "the fold contract accepts the schema and names every breach" begin
    directory = mktempdir()
    file = synthetic_fold_file(directory, 2015)
    ok = v24_validate_oof_year(file.path; expect_year=2015)
    @test ok.ok
    @test isempty(ok.problems)
    @test ok.year == 2015
    @test ok.n_rows == 12
    @test ok.steps == sort(collect(V23_MODEL_STEPS))
    @test ok.n_fallback == 0
    @test isempty(ok.extra_columns)
    @test v24_require_oof_year(file.path; expect_year=2015).ok

    # Every expert Task B stacks over, and every comparator it scores, is present.
    # Amendment A3 made the static V2.2 stack the tenth expert.
    @test length(V24_EXPERT_COLUMNS) == 10
    @test all(name in V24_OOF_COLUMNS for name in V24_EXPERT_COLUMNS)
    @test all(name in V24_OOF_COLUMNS for name in V24_COMPARATOR_COLUMNS)
    @test length(V24_FEATURE_COLUMNS) == 29
    @test V24_FEATURE_COLUMNS ==
          Tuple("f_" * String(name) for name in SolarSINDy.v23_direct_feature_names())

    # A missing expert column is named, not tolerated.
    table = CSV.read(file.path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    trimmed = joinpath(directory, "trimmed.csv")
    CSV.write(trimmed, select(table, Not(:direct_gbm)))
    dropped = v24_validate_oof_year(trimmed)
    @test !dropped.ok
    @test dropped.missing_columns == ["direct_gbm"]
    @test_throws ErrorException v24_require_oof_year(trimmed)

    # A non-finite forecast is a breach wherever it appears.
    holed = joinpath(directory, "holed.csv")
    broken = copy(table)
    broken.t1r_analog[3] = NaN
    CSV.write(holed, broken)
    @test !v24_validate_oof_year(holed).ok
    @test any(occursin("t1r_analog", p) for p in v24_validate_oof_year(holed).problems)

    # A feature is allowed to be empty only on a served-fallback row, which is
    # what "never imputed" means in the schema.
    featured = joinpath(directory, "featured.csv")
    partial = copy(table)
    partial.f_bz0[4] = NaN
    CSV.write(featured, partial)
    @test !v24_validate_oof_year(featured).ok
    partial.fallback[4] = true
    CSV.write(featured, partial)
    @test v24_validate_oof_year(featured).ok
    @test v24_validate_oof_year(featured).n_fallback == 1

    # One file is one calendar year.
    mixed = joinpath(directory, "mixed.csv")
    spread = copy(table)
    spread.issue_time_utc[1] = DateTime(2016, 1, 1)
    CSV.write(mixed, spread)
    @test !v24_validate_oof_year(mixed).ok
    @test !v24_validate_oof_year(file.path; expect_year=2016).ok

    # A repeated (issue, step) row would double-count that observation.
    repeated = joinpath(directory, "repeated.csv")
    CSV.write(repeated, vcat(table, table[1:1, :]))
    @test !v24_validate_oof_year(repeated).ok
    @test any(occursin("repeated", p) for p in v24_validate_oof_year(repeated).problems)
end

@testset "a fold file rebuilds the store it came from" begin
    directory = mktempdir()
    file = synthetic_fold_file(directory, 2015)
    anchors = tiny_anchors(unique(file.issues))
    ctx = tiny_context(anchors)
    plan = V24Plan("unit", directory, true, [2015], 0, [V24DirectConfig(4, 200, 64)], 0, 0,
                   false)
    store = V24Store(length(anchors))
    loaded = v24_load_fold!(store, ctx, plan, V24Fold(2015, v24_fold_cutoff(2015), Int[],
                                                      collect(eachindex(anchors.issue))))
    @test loaded.rows == 12
    @test store.years == [2015]
    # The three columns a later fold fits on come back exactly as written.
    expected = sort([-10.0 - r for r in 1:12])
    @test sort(collect(filter(isfinite, store.t1r))) == expected
    @test sort(collect(filter(isfinite, store.lat))) == expected
    @test sort(collect(filter(isfinite, store.analog_raw))) == expected
    @test v24_prior_rows(store, anchors, 2016) == store.rows[2015]
    @test !any(store.fallback)
end

# ---------------------------------------------------------------------------
# 5. Artifact identity: the smoke fold against the archived V2.3 core
# ---------------------------------------------------------------------------

const V24_SMOKE_OOF = v24_oof_year_path(V24_ROLLING_SMOKE_DIR, 2015)
const V24_DEV_ANALOG_OOF = joinpath(V23_DEV_DIR, "oof_T1_magnetic_K25.csv")

@testset "the smoke fold reproduces the archived V2.3 analog core" begin
    if !(isfile(V24_SMOKE_OOF) && isfile(V24_DEV_ANALOG_OOF))
        @info "V2.4 smoke identity skipped: generated artifact missing" smoke =
            V24_SMOKE_OOF archive = V24_DEV_ANALOG_OOF
    else
        smoke = v24_require_oof_year(V24_SMOKE_OOF; expect_year=2015)
        @test smoke.ok
        rolling = CSV.read(V24_SMOKE_OOF, DataFrame;
                           select=["issue_time_utc", "model_step_hours", "fallback",
                                   "t1_analog_raw"],
                           types=Dict("issue_time_utc" => DateTime))
        archived = CSV.read(V24_DEV_ANALOG_OOF, DataFrame;
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

        manifest = CSV.read(v24_manifest_year_path(V24_ROLLING_SMOKE_DIR, 2015), DataFrame)
        cutoff = only(manifest.value[manifest.name .== "training_max_target_utc"])
        @test DateTime(cutoff) == v24_fold_cutoff(2015)
        embargo = only(manifest.count[manifest.name .== "embargo_hours"])
        @test embargo >= V23_BASE_EMBARGO_HOURS
        last_target = only(manifest.value[manifest.name .== "training_last_target_utc"])
        @test DateTime(last_target) <= v24_fold_cutoff(2015)
    end
end

@testset "every persisted study fold satisfies the contract" begin
    files = isdir(V24_ROLLING_DIR) ?
        [f for f in readdir(V24_ROLLING_DIR) if startswith(f, "oof_year_")] : String[]
    if isempty(files)
        @info "V2.4 study folds skipped: no rolling artifact present" directory =
            V24_ROLLING_DIR
    else
        for file in sort(files)
            test_year = parse(Int, match(r"^oof_year_(\d+)\.csv$", file).captures[1])
            @test test_year in V24_FOLD_YEARS
            check = v24_require_oof_year(joinpath(V24_ROLLING_DIR, file);
                                         expect_year=test_year)
            @test check.ok
            @test check.n_rows > 0
            @test check.n_issues * length(V23_MODEL_STEPS) >= check.n_rows

            manifest_path = v24_manifest_year_path(V24_ROLLING_DIR, test_year)
            @test isfile(manifest_path)
            manifest = CSV.read(manifest_path, DataFrame)
            last_target = only(manifest.value[manifest.name .== "training_last_target_utc"])
            @test DateTime(last_target) <= v24_fold_cutoff(test_year)
            scored_first = only(manifest.value[manifest.name .== "scored_first_issue_utc"])
            @test Dates.year(DateTime(scored_first)) == test_year

            # The fold records which direct-GBM grid it searched, and the inner
            # validation must have scored every point of it at both selection
            # steps: a fold written under a trimmed grid would tune the strongest
            # comparator on less than plan section 4 promises.
            grid = only(manifest.value[manifest.name .== "direct_gbm_grid_points"])
            @test only(manifest.count[manifest.name .== "direct_gbm_grid_points"]) ==
                  length(V24_DIRECT_GRID)
            @test Set(split(grid, ";")) ==
                  Set(v24_direct_label(config) for config in V24_DIRECT_GRID)
            inner = manifest[manifest.entry_type .== "direct_gbm_inner", :]
            @test nrow(inner) == length(V24_DIRECT_SELECTION) * length(V24_DIRECT_GRID)
            for (selection_step, _) in V24_DIRECT_SELECTION, config in V24_DIRECT_GRID
                @test any(inner.name .== "step$(selection_step)_$(v24_direct_label(config))")
            end
            for (_, applies) in V24_DIRECT_SELECTION, step in applies
                chosen = only(manifest.value[manifest.name .== "direct_gbm_step$(step)"])
                @test chosen in [v24_direct_label(c) for c in V24_DIRECT_GRID]
            end
        end
    end
end

end # module
