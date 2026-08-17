#!/usr/bin/env julia

# v2_4_build_deploy.jl — assemble a deployable Operational V2.4e bundle from the rolling-origin study.
#
# What a bundle is. The rolling study fits every expert inside every fold and scores the folds; it
# never fits the model a live engine would load, because no fold's training window reaches the end of
# the data. This script produces that model: a "fold Y" refit in which the training window is every
# base-table anchor whose maximum target precedes the fold boundary by the 168 h embargo, and the
# out-of-fold pool is the persisted fold tables of every earlier year. For the deployment bundle
# Y = 2026 and the boundary is the end of the base table; for the identity oracle a replica bundle is
# built at Y = 2025, whose fits must therefore reproduce the rolling fold-2025 fits exactly.
#
# What is reused, deliberately. Every fitted object comes from the code path that produced the study:
# the analog archive rule and the refit ridge correction from `v2_4_rolling.jl` (which pulls the V2.3
# chain), the inner-validation choice and the per-step fit of the direct increment-GBM from
# `v24_direct_select`/`v24_direct_fit`, the climatology timescale from `_v23_climatology_tau`, and the
# super-learner and conformal fits from `v2_4_learn.jl`. Nothing here restates a fit. The learning
# stage is included inside its own module because it and the file contract legitimately disagree about
# two constant names (`V24_FOLD_YEARS`, `V24_MODEL_COLUMNS`), and a top-level double include would let
# one definition silently win over the other.
#
# What is verified before a bundle is written out. When the fold year is one the study scored, the
# fitted stack weights, conformal half-widths, climatology timescale, direct-GBM configurations and
# correction label are compared against the study's own artifacts and must agree; then the bundle is
# loaded back through `load_v24_serving_artifacts`, the same path the live engine uses. A bundle that
# cannot be reproduced from the study or cannot be loaded is not published.
#
# Boosted fits are thread-count sensitive: EvoTrees accumulates histogram sums in parallel, so a fit
# is bit-identical only at a fixed thread count. The study ran with 8 threads and this script refuses
# to build at a different count unless the operator states the intent.
#
# Usage (from the package root):
#   JULIA_NUM_THREADS=8 julia --startup-file=no --project=. \
#       validation/operational/v2_4_build_deploy.jl [options]
#
#   --folds=2026,2025     bundles to build (default both: the deployment and the oracle replica)
#   --indir=<dir>         rolling study tree holding oof_year_*.csv / learn_year_*.csv
#   --deploy-out=<dir>    deployment bundle directory (default deploy/v2_4)
#   --replica-out=<dir>   oracle replica directory (default …/output/operational/v2_4_replica_2025)
#   --force               overwrite a non-empty output directory
#   --allow-thread-drift  build at a thread count other than the study's, recording the deviation

include(joinpath(@__DIR__, "v2_4_rolling.jl"))

"""
    V24LearnStage

The V2.4 learning stage, loaded into its own module so its `V24_FOLD_YEARS` and `V24_MODEL_COLUMNS`
cannot collide with the file contract's constants of the same names. Only the fitting primitives are
used from here: the fold reader, the NNLS super-learner, the conformal calibration and their
persistable row forms.
"""
module V24LearnStage
include(joinpath(@__DIR__, "v2_4_learn.jl"))
end

"Thread count the study's boosted fits ran under; a bundle is bit-reproducible only at this count."
const V24_DEPLOY_STUDY_THREADS = 8

"Deployment bundle file names."
const V24_DEPLOY_STACK = "stack_weights.csv"
const V24_DEPLOY_CONFORMAL = "conformal.csv"
const V24_DEPLOY_TAU = "climatology_tau.csv"
const V24_DEPLOY_CALIBRATION = "t1r_calibration.csv"
const V24_DEPLOY_FRAME = "analog_frame.csv"
const V24_DEPLOY_ORIGINS = "analog_origins.csv"
const V24_DEPLOY_STATS = "analog_feature_stats.csv"
const V24_DEPLOY_GUARD = "guard.json"
const V24_DEPLOY_SELECTED = "selected.json"
const V24_DEPLOY_MANIFEST = "manifest.csv"

"""
Served variant and the study stack its weights come from. Amendment A3's selection returned `v2_4e`:
the ten-expert floor-constrained stack, served without a point-forecast guard, so the bundle's own
`guard.json` records the guard as inactive and the loader refuses to apply one.
"""
const V24_DEPLOY_SERVED_VARIANT = "v2_4e"
const V24_DEPLOY_STACK_VARIANT = "L1e"

"Largest deviation tolerated when a bundle is checked against the study's own fitted artifacts."
const V24_DEPLOY_STUDY_ATOL = 1e-12

"Default output directories."
const V24_DEPLOY_DEFAULT_DIR = normpath(joinpath(OPERATIONAL_PACKAGE_ROOT, "deploy", "v2_4"))
const V24_DEPLOY_REPLICA_DEFAULT_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_4_replica_2025")

"Fold year of the deployable bundle: the first year no base-table anchor can belong to."
const V24_DEPLOY_FOLD_YEAR = 2026

_v24_deploy_log(msg) = (println(msg); flush(stdout))

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

"""
    V24DeployRequest

One bundle to build: its fold year, where it goes, which persisted fold years feed the out-of-fold
pool of the correction and the super-learner, and which learn-stage years calibrate the interval.

`calendar_cutoff` distinguishes the two kinds of bundle. A replica of a scored fold takes the study's
calendar rule (`Y-01-01T00 − 168 h`) so its fits are the fold's fits; the deployable bundle takes the
end of the base table instead, because there is no later year to embargo against.
"""
struct V24DeployRequest
    fold_year::Int
    label::String
    outdir::String
    oof_years::Vector{Int}
    conformal_years::Vector{Int}
    calendar_cutoff::Bool
    replica_of_study_fold::Bool
end

"""
    v24_deploy_request(fold_year; outdir, indir) -> V24DeployRequest

Build the request for `fold_year` from the persisted study tree: the out-of-fold pool is every fold
table strictly before the fold year, and the conformal pool is every learn-stage year strictly before
it (the seed fold has no variant centers, so it calibrates nothing).
"""
function v24_deploy_request(fold_year::Integer; outdir::AbstractString,
                            indir::AbstractString=V24_ROLLING_DIR)
    year = Int(fold_year)
    oof = Int[]
    for candidate in V24_FOLD_YEARS
        candidate < year || continue
        isfile(v24_oof_year_path(indir, candidate)) || continue
        push!(oof, candidate)
    end
    isempty(oof) && error(
        "no persisted fold table precedes $year in $indir; the bundle would have no pool to fit on",
    )
    conformal = Int[]
    for candidate in oof
        candidate == minimum(oof) && continue
        isfile(joinpath(indir, "learn_year_$(candidate).csv")) || continue
        push!(conformal, candidate)
    end
    isempty(conformal) && error(
        "no learn-stage year precedes $year in $indir; the bundle would have no interval",
    )
    replica = isfile(joinpath(indir, "learn_year_$(year).csv"))
    # A replica of a scored fold takes the study's calendar cutoff so its fits are that fold's fits;
    # the deployable bundle has no later year to embargo against and takes the end of the base table.
    return V24DeployRequest(year, "fold$(year)", abspath(String(outdir)), oof, conformal,
                            replica, replica)
end

# ---------------------------------------------------------------------------
# The fold a bundle is fitted on
# ---------------------------------------------------------------------------

"""
    v24_deploy_cutoff(anchors, request) -> DateTime

Latest target a bundle's training window may contain.

A replica of a scored fold takes the study's calendar rule, so its training window is the fold's
window. The deployable bundle takes the last issue hour the base table admits: the base-table build
already withholds the trailing 168 h of data from its DEV/TEST partitions, so that hour is the newest
state a deployment may be fitted against, and using it as a *target* bound rather than an issue bound
keeps a further seven hours of margin — which is the bound the integration specification states.
"""
function v24_deploy_cutoff(anchors::V23Anchors, request::V24DeployRequest)
    request.calendar_cutoff && return v24_fold_cutoff(request.fold_year)
    cutoff = maximum(anchors.issue)
    cutoff < v24_fold_cutoff(request.fold_year) || error(
        "the base table reaches $(cutoff), which is not strictly before the calendar embargo " *
        "$(v24_fold_cutoff(request.fold_year)) of fold $(request.fold_year)",
    )
    return cutoff
end

"""
    v24_deploy_fold(ctx, request) -> V24Fold

The fold a bundle is fitted on: the training window under [`v24_deploy_cutoff`](@ref) and, for a
replica of a scored fold, that fold's scored anchors so the reused stages see the same rows. The
deployable bundle has no scored year, so its query list is empty and no stage may depend on one.
"""
function v24_deploy_fold(ctx::V23Context, request::V24DeployRequest)
    issues = ctx.anchors.issue
    cutoff = v24_deploy_cutoff(ctx.anchors, request)
    train = [i for i in eachindex(issues) if issues[i] + Hour(V23_MAX_STEP) <= cutoff]
    isempty(train) && error("bundle $(request.label) has an empty training window")
    query = [i for i in eachindex(issues) if Dates.year(issues[i]) == request.fold_year]
    if request.replica_of_study_fold
        isempty(query) && error(
            "bundle $(request.label) replicates a scored fold but the base table holds no anchor " *
            "issued in $(request.fold_year)",
        )
        rows = v24_fold_rows(issues, request.fold_year)
        rows.train == train || error(
            "the replica training window of $(request.fold_year) is not the study fold's window",
        )
        _v23_assert_embargo(ctx.anchors, train, sort(query; by=i -> issues[i]))
    else
        isempty(query) || error(
            "bundle $(request.label) is not a replica but the base table holds anchors issued in " *
            "$(request.fold_year)",
        )
    end
    maximum(issues[train]) + Hour(V23_MAX_STEP) <= cutoff || error(
        "bundle $(request.label) training window carries a target beyond $(cutoff)",
    )
    return V24Fold(request.fold_year, cutoff, train, sort(query; by=i -> issues[i]))
end

# ---------------------------------------------------------------------------
# Out-of-fold pool
# ---------------------------------------------------------------------------

"""
    v24_deploy_store(ctx, plan, request) -> V24Store

Rebuild the out-of-fold store from the persisted fold tables of `request.oof_years` through
`v24_load_fold!`, the same reader the rolling engine resumes with. Loading is what makes the
correction's fitting pool the study's pool rather than a recomputation of it.
"""
function v24_deploy_store(ctx::V23Context, plan::V24Plan, request::V24DeployRequest)
    store = V24Store(length(ctx.anchors))
    for year in request.oof_years
        loaded = v24_load_fold!(store, ctx, plan, V24Fold(year, v24_fold_cutoff(year), Int[], Int[]))
        _v24_deploy_log("    out-of-fold $(year): $(loaded.rows) rows, $(loaded.anchors) anchors")
    end
    return store
end

# ---------------------------------------------------------------------------
# Super-learner and conformal pools from the persisted tables
# ---------------------------------------------------------------------------

"""
    v24_deploy_pool(years; indir) -> Vector

The learning stage's view of the persisted fold tables, in ascending year order. This is the pool the
super-learner is fitted on, and it is read by the learning stage's own validating reader so a bundle
cannot be fitted on a fold table that breaks the file contract.
"""
function v24_deploy_pool(years::AbstractVector{Int}; indir::AbstractString=V24_ROLLING_DIR)
    pool = V24LearnStage.V24YearData[]
    for year in sort(years)
        push!(pool, V24LearnStage.v24_read_year(year; dir=indir))
    end
    return pool
end

"""
    v24_deploy_conformal_sources(pool, years; indir) -> Vector

Calibration sources of the served interval: for every year in `years`, that fold's rows paired with
the out-of-fold `v2_4e` centers the learning stage persisted for them. The centers are read back
rather than recomputed, because the interval must be calibrated on the residuals of the centers the
study actually scored; the alignment of every `(issue, step)` pair is checked, so a table written in a
different order cannot silently shift a residual onto the wrong row.
"""
function v24_deploy_conformal_sources(pool::AbstractVector, years::AbstractVector{Int};
                                      indir::AbstractString=V24_ROLLING_DIR,
                                      variant::AbstractString=V24_DEPLOY_SERVED_VARIANT)
    by_year = Dict(data.year => data for data in pool)
    sources = Tuple{V24LearnStage.V24YearData,Vector{Float64}}[]
    for year in sort(years)
        haskey(by_year, year) || error(
            "the conformal pool asks for year $year, which is not in the super-learner pool",
        )
        data = by_year[year]
        path = joinpath(indir, "learn_year_$(year).csv")
        isfile(path) || error("the learning stage wrote no table for year $year: $path")
        table = CSV.read(path, DataFrame;
                         select=["issue_time_utc", "model_step_hours", String(variant)],
                         types=Dict("issue_time_utc" => DateTime))
        nrow(table) == length(data) || error(
            "$path holds $(nrow(table)) rows but the fold table holds $(length(data))",
        )
        centers = Vector{Float64}(undef, length(data))
        index = Dict{Tuple{DateTime,Int},Int}()
        for r in 1:nrow(table)
            key = (table.issue_time_utc[r], Int(table.model_step_hours[r]))
            haskey(index, key) && error("$path repeats the row $key")
            index[key] = r
        end
        for i in 1:length(data)
            key = (data.issue[i], data.step[i])
            r = get(index, key, 0)
            r == 0 && error("$path has no row for $key; the interval pool would be misaligned")
            value = Float64(table[r, Symbol(variant)])
            isfinite(value) || error("$path holds a non-finite $(variant) center at $key")
            centers[i] = value
        end
        push!(sources, (data, centers))
    end
    return sources
end

# ---------------------------------------------------------------------------
# Checks against the study's own artifacts
# ---------------------------------------------------------------------------

"Rows of a study artifact matching `fold_year` and `variant`."
function _v24_deploy_study_rows(path::AbstractString, fold_year::Integer, variant::AbstractString)
    isfile(path) || error("study artifact is missing: $path")
    table = CSV.read(path, DataFrame)
    keep = [Int(table.fold_year[r]) == Int(fold_year) &&
            String(table.variant[r]) == String(variant) for r in 1:nrow(table)]
    subset = table[keep, :]
    nrow(subset) > 0 || error(
        "$path holds no ($fold_year, $variant) row; the bundle cannot be checked against the study",
    )
    return subset
end

"""
    v24_deploy_check_stack(cells, fold_year; indir) -> NamedTuple

Compare freshly fitted super-learner cells against the weights the study persisted for the same fold.
Every cell must be present in both and every weight must agree to `V24_DEPLOY_STUDY_ATOL`; a
disagreement means the rebuilt pool is not the fold's pool, which would make the bundle a different
model from the one the study scored.
"""
function v24_deploy_check_stack(cells::Dict, fold_year::Integer;
                                indir::AbstractString=V24_ROLLING_DIR)
    table = _v24_deploy_study_rows(joinpath(indir, "v2_4_l1_weights.csv"), fold_year,
                                   V24_DEPLOY_STACK_VARIANT)
    columns = [Symbol("w_", expert) for expert in V24_SERVING_EXPERTS]
    worst = 0.0
    checked = 0
    for r in 1:nrow(table)
        key = (Int(table.model_step_hours[r]), Symbol(String(table.regime[r])),
               Symbol(String(table.depth_bin[r])))
        haskey(cells, key) || error(
            "the study fitted the cell $key but the rebuilt stack does not carry it",
        )
        cell = cells[key]
        cell.n_rows == Int(table.n_rows[r]) || error(
            "cell $key was fitted on $(cell.n_rows) rows; the study recorded $(table.n_rows[r])",
        )
        for (j, column) in enumerate(columns)
            worst = max(worst, abs(cell.weights[j] - Float64(table[r, column])))
        end
        checked += 1
    end
    checked == length(cells) || error(
        "the rebuilt stack holds $(length(cells)) cells but the study recorded $checked",
    )
    worst <= V24_DEPLOY_STUDY_ATOL || error(@sprintf(
        "the rebuilt super-learner weights deviate from the study's fold-%d fit by %.3e",
        Int(fold_year), worst))
    return (cells=checked, worst=worst)
end

"""
    v24_deploy_check_conformal(strata, fold_year; indir) -> NamedTuple

Compare freshly fitted conformal strata against the ones the study persisted for the same fold.
"""
function v24_deploy_check_conformal(strata::Dict, fold_year::Integer;
                                    indir::AbstractString=V24_ROLLING_DIR)
    table = _v24_deploy_study_rows(joinpath(indir, "v2_4_conformal.csv"), fold_year,
                                   V24_DEPLOY_SERVED_VARIANT)
    worst = 0.0
    checked = 0
    for r in 1:nrow(table)
        key = (Int(table.model_step_hours[r]), Symbol(String(table.depth_bin[r])))
        haskey(strata, key) || error(
            "the study fitted the conformal stratum $key but the rebuilt calibration lacks it",
        )
        stratum = strata[key]
        stratum.n == Int(table.n[r]) || error(
            "stratum $key was fitted on $(stratum.n) residuals; the study recorded $(table.n[r])",
        )
        worst = max(worst, abs(stratum.half_width - Float64(table.half_width_nt[r])))
        checked += 1
    end
    checked == length(strata) || error(
        "the rebuilt calibration holds $(length(strata)) strata but the study recorded $checked",
    )
    worst <= V24_DEPLOY_STUDY_ATOL || error(@sprintf(
        "the rebuilt conformal half-widths deviate from the study's fold-%d calibration by %.3e nT",
        Int(fold_year), worst))
    return (strata=checked, worst=worst)
end

"Value of a `(entry_type, name)` row of a persisted fold manifest."
function _v24_deploy_fold_manifest(indir::AbstractString, fold_year::Integer)
    path = v24_manifest_year_path(indir, fold_year)
    isfile(path) || error("the study wrote no fold manifest for $fold_year: $path")
    table = CSV.read(path, DataFrame; types=Dict("entry_type" => String, "name" => String,
                                                 "value" => String))
    text(value) = value === missing ? "" : String(value)
    number(value) = value === missing ? NaN : Float64(value)
    out = Dict{Tuple{String,String},Tuple{Float64,String}}()
    for r in 1:nrow(table)
        out[(text(table.entry_type[r]), text(table.name[r]))] =
            (number(table.count[r]), text(table.value[r]))
    end
    return out
end

"""
    v24_deploy_check_fold_state(fold_manifest, tau, chosen, calibration, fold_year)

Compare the reused stages' fitted state against the persisted fold manifest of the same fold: the
climatology timescale, the direct-GBM configuration of every model step, and the correction label
(which records the recipe, the ridge strength and the fitted row count).
"""
function v24_deploy_check_fold_state(fold_manifest, tau::Real, chosen, calibration,
                                     fold_year::Integer)
    recorded_tau = get(fold_manifest, ("hyperparameter", "climatology_tau_hours"), nothing)
    recorded_tau === nothing &&
        error("the fold-$(fold_year) manifest records no climatology timescale")
    abs(Float64(tau) - recorded_tau[1]) <= V24_DEPLOY_STUDY_ATOL || error(@sprintf(
        "the rebuilt climatology timescale is %.12f h; the study recorded %.12f h",
        Float64(tau), recorded_tau[1]))
    for step in V23_MODEL_STEPS
        recorded = get(fold_manifest, ("hyperparameter", "direct_gbm_step$(step)"), nothing)
        recorded === nothing &&
            error("the fold-$(fold_year) manifest records no direct-GBM choice at step $step")
        label = v24_direct_label(chosen[step])
        label == recorded[2] || error(
            "the inner validation chose $label at step $step; the study chose $(recorded[2])",
        )
    end
    recorded_label = get(fold_manifest, ("hyperparameter", "t1r_label"), nothing)
    recorded_label === nothing &&
        error("the fold-$(fold_year) manifest records no correction label")
    calibration.label == recorded_label[2] || error(
        "the rebuilt correction carries the label $(calibration.label); the study recorded " *
        "$(recorded_label[2])",
    )
    return nothing
end

# ---------------------------------------------------------------------------
# Writing a bundle
# ---------------------------------------------------------------------------

"Append one manifest row."
function _v24_deploy_push!(rows::Vector{NamedTuple}, entry_type, name, count, value)
    push!(rows, (entry_type=String(entry_type), name=String(name), count=Float64(count),
                 value=String(value)))
    return rows
end

"""
    build_v2_4_bundle(ctx, rowview, usable, features, plan, request; indir, force,
                      thread_note) -> NamedTuple

Fit and write one V2.4e bundle. Every fitted object comes from the study's own code path; the fold
year decides only which training window and which out-of-fold pool those paths see.
"""
function build_v2_4_bundle(ctx::V23Context, rowview::V24RowView, usable::AbstractVector{Bool},
                           features::AbstractMatrix{Float64}, plan::V24Plan,
                           request::V24DeployRequest; indir::AbstractString=V24_ROLLING_DIR,
                           force::Bool=false, thread_note::AbstractString="")
    started = time()
    outdir = request.outdir
    if isdir(outdir) && !isempty(readdir(outdir)) && !force
        error("bundle directory $outdir is not empty; pass --force to overwrite it")
    end
    mkpath(outdir)
    anchors = ctx.anchors
    fold = v24_deploy_fold(ctx, request)
    _v24_deploy_log(@sprintf(
        "  %s: training targets <= %s, %d anchors (%d usable), out-of-fold pool %d..%d",
        request.label, fold.cutoff, length(fold.train), count(i -> usable[i], fold.train),
        first(request.oof_years), last(request.oof_years),
    ))

    store = v24_deploy_store(ctx, plan, request)

    # ---- expert E3: the analog archive and the refit ridge correction ----
    archive = [i for i in fold.train if ctx.ok[i] && ctx.origin_ok[i]]
    length(archive) >= V24_ANALOG_K || error(
        "bundle $(request.label) archive holds $(length(archive)) eligible origins, fewer than " *
        "K = $(V24_ANALOG_K)",
    )
    origins = anchors.issue[archive]
    issorted(origins) || error("the anchor view is not chronological; the archive would be unordered")
    stats = v23_feature_stats(ctx.X[archive, :])
    fit = v24_t1r_fit_rows(store, ctx, fold, usable, plan)
    fit.in_sample && error(
        "bundle $(request.label) would fit its correction in sample; a deployable bundle needs an " *
        "out-of-fold pool",
    )
    fit_frame = v23_t1r_features(v24_row_frame(rowview, anchors, fit.pairs), fit.raw)
    calibration = v23_t1r_fit(fit_frame; label_stem="operational_v2_4_t1r_year$(fold.year)")
    _v24_deploy_log("    correction: $(calibration.label) on $(length(fit.pairs)) out-of-fold rows")

    # ---- expert E9: the climatology timescale ----
    tau = _v23_climatology_tau(ctx, fold.train)

    # ---- expert E8: the direct increment-GBM ----
    selection = v24_direct_select(ctx, fold, usable, features, plan)
    fitted = v24_direct_fit(ctx, fold, usable, features, selection.chosen, plan)
    _v24_deploy_log(@sprintf(
        "    direct-GBM: %s, %.1f s selection + %.1f s fit",
        join([@sprintf("step%d=%s", step, v24_direct_label(selection.chosen[step]))
              for step in V23_MODEL_STEPS], " "), selection.seconds, fitted.seconds,
    ))

    # ---- L1: the ten-expert floor-constrained super-learner, and L4: the conformal interval ----
    #
    # Amendment A3 embargoes the out-of-fold pool by target, not only by issue year, so a pool row is
    # admitted to a fit only when `issue + step` clears the same bound the training window clears. For
    # a replica that bound is the study fold's calendar cutoff, which is what makes the replica's
    # weights the fold's weights; for the deployable bundle it is the end of the base table, which is
    # stricter than the calendar rule and therefore never admits a row the study would have withheld.
    pool = v24_deploy_pool(request.oof_years; indir=indir)
    pool_cutoff = fold.cutoff
    cells = V24LearnStage.v24_fit_l1(pool; floor_mass=V24LearnStage.V24_SINDY_FLOOR,
                                     experts=V24LearnStage.V24_EXPERTS_TEN,
                                     family=V24LearnStage.V24_SINDY_FAMILY_TEN,
                                     cutoff=pool_cutoff)
    sources = v24_deploy_conformal_sources(pool, request.conformal_years; indir=indir)
    strata = V24LearnStage.v24_fit_conformal(sources; cutoff=pool_cutoff)
    _v24_deploy_log(@sprintf("    super-learner: %d cells over pool %s; conformal pool %s",
                             length(cells), join(request.oof_years, ","),
                             join(request.conformal_years, ",")))

    # ---- checks against the study, where the study scored this fold ----
    study = (stack=nothing, conformal=nothing)
    if request.replica_of_study_fold
        study = (stack=v24_deploy_check_stack(cells, fold.year; indir=indir),
                 conformal=v24_deploy_check_conformal(strata, fold.year; indir=indir))
        v24_deploy_check_fold_state(_v24_deploy_fold_manifest(indir, fold.year), tau,
                                    selection.chosen, calibration, fold.year)
        _v24_deploy_log(@sprintf(
            "    study agreement: %d stack cells max|Δ| = %.3e, %d conformal strata max|Δ| = %.3e nT",
            study.stack.cells, study.stack.worst, study.conformal.strata, study.conformal.worst,
        ))
    end

    # ---- write ----
    written = String[]
    record(name) = (push!(written, name); joinpath(outdir, name))

    CSV.write(record(V24_DEPLOY_STACK),
              DataFrame(V24LearnStage.v24_l1_weight_rows(cells, fold.year,
                                                         V24_DEPLOY_STACK_VARIANT)))
    CSV.write(record(V24_DEPLOY_CONFORMAL),
              DataFrame(V24LearnStage.v24_conformal_rows(
                  strata, fold.year, V24_DEPLOY_SERVED_VARIANT;
                  note="deploy_bundle_$(request.label)")))
    CSV.write(record(V24_DEPLOY_TAU), DataFrame(
        tau_hours=[tau], fold_year=[fold.year], n_training_anchors=[length(fold.train)],
        training_max_target_utc=[string(fold.cutoff)],
    ))
    write_operational_v2_calibration(record(V24_DEPLOY_CALIBRATION), calibration)
    CSV.write(record(V24_DEPLOY_FRAME), ctx.frame)
    CSV.write(record(V24_DEPLOY_ORIGINS), DataFrame(origin_time_utc=origins))
    CSV.write(record(V24_DEPLOY_STATS), DataFrame(
        feature_name=String.(collect(V23_FEATURE_NAMES)), feature_mean=stats.mean,
        feature_sd=stats.sd,
    ))
    for step in V23_MODEL_STEPS
        SolarSINDy.v23_save(fitted.models[step], record(v24_serving_direct_file(step)))
    end

    guard = Dict{String,Any}(
        "served_variant" => V24_DEPLOY_SERVED_VARIANT,
        "served_stack" => V24_DEPLOY_STACK_VARIANT,
        "guard_applied" => false,
        "guard_reference" => V24_SERVING_GUARD_REFERENCE_NONE,
        "guard_rule" => "no guard: the served center is the ten-expert stack center, and the static " *
                        "V2.2 stack enters that center as an expert rather than as a minimum on it",
        "guard_disclosure" => "the deepening-cell thresholds are recorded because every served row " *
                              "is labelled with them; depth safety of the published severity is " *
                              "taken as the deepest of this center, the static V2.2 stack and the " *
                              "V2.1 served center",
        "center_source" => "ten_expert_stack_with_family_floor",
        "residual_applied" => false,
        "deepening_rate_nt_per_h_strict_below" => V24_SERVING_GUARD_RATE_NT_PER_H,
        "deepening_depth_nt_at_or_below_with_active_coupling" => V24_SERVING_GUARD_DEPTH_NT,
        "sindy_family_floor" => V24_SERVING_SINDY_FLOOR,
        "sindy_family" => [String(V24_SERVING_EXPERTS[j]) for j in V24_SERVING_SINDY_FAMILY],
        "expert_order" => [String(expert) for expert in V24_SERVING_EXPERTS],
        "n_experts" => V24_SERVING_EXPERT_COUNT,
        "pool_target_embargo_hours" => V24LearnStage.V24_EMBARGO_HOURS,
        "pool_target_cutoff_utc" => string(pool_cutoff),
        "model_steps" => collect(V24_SERVING_MODEL_STEPS),
        "regimes" => [String(regime) for regime in (V24_SERVING_POOLED, V24_SERVING_REGIMES...)],
        "depth_bins" => [String(bin) for bin in V24_SERVING_DEPTH_BINS],
        "depth_bin_moderate_nt_at_or_below" => V24_SERVING_DEPTH_MODERATE_NT,
        "depth_bin_deep_nt_at_or_below" => V24_SERVING_DEPTH_DEEP_NT,
        "l1_cell_fallback_chain" => ["regime_x_depth", "regime_pooled", "pooled"],
        "l1_min_cell_rows" => V24LearnStage.V24_MIN_CELL_ROWS,
        "conformal_coverage" => V24_SERVING_COVERAGE,
        "conformal_strata" => "model_step_x_depth_bin_with_pooled_fallback",
        "climatology_tau_hours" => tau,
        "analog_k" => V24_ANALOG_K,
        "analog_weight_set" => String(V24_ANALOG_WEIGHT_SET),
        "analog_safeguards" => V24_ANALOG_SAFEGUARDS,
        "dst_floor_nt" => V24_SERVING_DST_FLOOR_NT,
        "dst_ceil_nt" => V24_SERVING_DST_CEIL_NT,
    )
    open(record(V24_DEPLOY_GUARD), "w") do io
        JSON3.pretty(io, guard)
        println(io)
    end

    selected = Dict{String,Any}(
        "served_variant" => V24_DEPLOY_SERVED_VARIANT,
        "identity" => V24_SERVED_IDENTITY,
        "driver_assumption" => V24_SERVED_DRIVER_ASSUMPTION,
        "stack_variant" => V24_DEPLOY_STACK_VARIANT,
        "residual_applied" => false,
        "fold_year" => fold.year,
        "bundle_label" => request.label,
        "training_max_target_utc" => string(fold.cutoff),
        "training_first_issue_utc" => string(minimum(anchors.issue[fold.train])),
        "training_last_issue_utc" => string(maximum(anchors.issue[fold.train])),
        "training_anchors" => length(fold.train),
        "training_anchors_usable" => count(i -> usable[i], fold.train),
        "oof_pool_years" => request.oof_years,
        "conformal_pool_years" => request.conformal_years,
        "embargo_hours" => V23_BASE_EMBARGO_HOURS,
        "pool_target_embargo_hours" => V24LearnStage.V24_EMBARGO_HOURS,
        "pool_target_cutoff_utc" => string(pool_cutoff),
        "replica_of_study_fold" => request.replica_of_study_fold,
        "study_selection_rule_variant" => V24_DEPLOY_SERVED_VARIANT,
        "guard_applied" => false,
        "served_variant_disclosure" =>
            "the study's preregistered selection rule returned the ten-expert floor-constrained " *
            "stack, which is served without a point-forecast guard; the static V2.2 stack is an " *
            "expert of that stack, and the published severity is additionally taken no shallower " *
            "than the static stack and the V2.1 served center",
        "julia_threads" => Threads.nthreads(),
        "evotrees_version" => SolarSINDy.V23_GBM_EVOTREES_VERSION,
        "seed" => V23_SEED,
    )
    isempty(thread_note) || (selected["thread_deviation"] = thread_note)
    open(record(V24_DEPLOY_SELECTED), "w") do io
        JSON3.pretty(io, selected)
        println(io)
    end

    rows = NamedTuple[]
    _v24_deploy_push!(rows, "build", "utc", NaN, string(now(UTC)))
    _v24_deploy_push!(rows, "build", "package_version", NaN, string(pkgversion(SolarSINDy)))
    _v24_deploy_push!(rows, "build", "julia_version", NaN, string(VERSION))
    _v24_deploy_push!(rows, "build", "julia_threads", Threads.nthreads(), thread_note)
    _v24_deploy_push!(rows, "build", "evotrees_version", NaN,
                      SolarSINDy.V23_GBM_EVOTREES_VERSION)
    _v24_deploy_push!(rows, "build", "identity", NaN, V24_SERVED_IDENTITY)
    _v24_deploy_push!(rows, "build", "bundle_label", NaN, request.label)
    _v24_deploy_push!(rows, "fold", "year", fold.year, request.label)
    _v24_deploy_push!(rows, "window", "training_max_target_utc", NaN, string(fold.cutoff))
    _v24_deploy_push!(rows, "window", "training_first_issue_utc", NaN,
                      string(minimum(anchors.issue[fold.train])))
    _v24_deploy_push!(rows, "window", "training_last_issue_utc", NaN,
                      string(maximum(anchors.issue[fold.train])))
    _v24_deploy_push!(rows, "window", "embargo_hours", V23_BASE_EMBARGO_HOURS, "")
    _v24_deploy_push!(rows, "rows", "training_anchors", length(fold.train), "")
    _v24_deploy_push!(rows, "rows", "training_anchors_usable",
                      count(i -> usable[i], fold.train), "")
    _v24_deploy_push!(rows, "pool", "oof_years", length(request.oof_years),
                      join(request.oof_years, ";"))
    _v24_deploy_push!(rows, "pool", "conformal_years", length(request.conformal_years),
                      join(request.conformal_years, ";"))
    _v24_deploy_push!(rows, "pool", "t1r_fit_rows", length(fit.pairs), calibration.label)
    _v24_deploy_push!(rows, "source", "base_table_sha256", NaN, ctx.base_table_sha)
    _v24_deploy_push!(rows, "source", "hourly_frame_sha256", NaN, ctx.frame_sha)
    _v24_deploy_push!(rows, "source", "study_tree", NaN, abspath(String(indir)))
    for year in request.oof_years
        path = v24_oof_year_path(indir, year)
        _v24_deploy_push!(rows, "source_sha256", basename(path), filesize(path),
                          _v23_sha256_file(path))
    end
    for year in request.conformal_years
        path = joinpath(indir, "learn_year_$(year).csv")
        _v24_deploy_push!(rows, "source_sha256", basename(path), filesize(path),
                          _v23_sha256_file(path))
    end
    _v24_deploy_push!(rows, "analog_archive", "origins", length(origins),
                      "first=$(first(origins));last=$(last(origins))")
    _v24_deploy_push!(rows, "analog_archive", "frame_window", nrow(ctx.frame),
                      "first=$(first(ctx.frame.time_utc));last=$(last(ctx.frame.time_utc))")
    _v24_deploy_push!(rows, "hyperparameter", "climatology_tau_hours", tau, "")
    for step in V23_MODEL_STEPS
        _v24_deploy_push!(rows, "direct_gbm", "step$(step)", fitted.train_rows[step],
                          v24_direct_label(selection.chosen[step]))
    end
    _v24_deploy_push!(rows, "stack", "cells", length(cells), V24_DEPLOY_STACK_VARIANT)
    _v24_deploy_push!(rows, "conformal", "strata", length(strata), V24_DEPLOY_SERVED_VARIANT)
    if request.replica_of_study_fold
        _v24_deploy_push!(rows, "study_agreement", "stack_max_abs_delta",
                          study.stack.worst, "cells=$(study.stack.cells)")
        _v24_deploy_push!(rows, "study_agreement", "conformal_max_abs_delta_nt",
                          study.conformal.worst, "strata=$(study.conformal.strata)")
    end
    for name in written
        path = joinpath(outdir, name)
        _v24_deploy_push!(rows, "sha256", name, filesize(path), _v23_sha256_file(path))
    end
    CSV.write(joinpath(outdir, V24_DEPLOY_MANIFEST), DataFrame(rows))

    artifacts = load_v24_serving_artifacts(outdir; core=ctx.core)
    _v24_deploy_log(@sprintf(
        "  + %s written in %.1f s: %d files, %d origins, %d cells, identity %s",
        request.label, time() - started, length(written) + 1, length(artifacts.analog.origins),
        length(artifacts.cells), artifacts.identity,
    ))
    return (request=request, dir=outdir, files=written, manifest=DataFrame(rows),
            artifacts=artifacts, fold=fold, cells=cells, strata=strata, tau=tau,
            calibration=calibration, chosen=selection.chosen, origins=origins,
            study=study, seconds=time() - started)
end

"""
    build_v2_4_deploy(; folds, indir, deploy_out, replica_out, force, allow_thread_drift)

Build every requested bundle in one process, sharing the base-table context and the design matrix
between them. Returns one result per bundle in the requested order.
"""
function build_v2_4_deploy(; folds::AbstractVector{Int}=[V24_DEPLOY_FOLD_YEAR, 2025],
                           indir::AbstractString=V24_ROLLING_DIR,
                           deploy_out::AbstractString=V24_DEPLOY_DEFAULT_DIR,
                           replica_out::AbstractString=V24_DEPLOY_REPLICA_DEFAULT_DIR,
                           force::Bool=false, allow_thread_drift::Bool=false)
    isempty(folds) && error("no bundle was requested")
    isdir(indir) || error("V2.4 study tree not found: $indir")
    thread_note = ""
    if Threads.nthreads() != V24_DEPLOY_STUDY_THREADS
        allow_thread_drift || error(
            "the V2.4 study fitted its boosted experts with $(V24_DEPLOY_STUDY_THREADS) threads " *
            "and this process has $(Threads.nthreads()); a boosted fit is bit-identical only at a " *
            "fixed thread count. Rerun with JULIA_NUM_THREADS=$(V24_DEPLOY_STUDY_THREADS), or " *
            "pass --allow-thread-drift to record the deviation instead of matching it.",
        )
        thread_note = "study_threads=$(V24_DEPLOY_STUDY_THREADS);build_threads=" *
                      "$(Threads.nthreads());accepted_by_operator"
        @warn "building a V2.4 bundle at a thread count the study did not use" note = thread_note
    end
    requests = V24DeployRequest[]
    for year in folds
        out = Int(year) == V24_DEPLOY_FOLD_YEAR ? deploy_out : replica_out
        push!(requests, v24_deploy_request(year; outdir=out, indir=indir))
    end

    started = time()
    # The plan's output directory is where `v24_load_fold!` reads the persisted folds from, so it is
    # the requested study tree rather than the default one; nothing else about the plan is reduced.
    plan = v24_assert_full_plan(V24Plan(
        "v2_4_deploy", abspath(String(indir)), false, collect(Int, V24_FOLD_YEARS), 0,
        collect(V24_DIRECT_GRID), 0, 0, true,
    ))
    _v24_deploy_log("V2.4 deployment build; threads=$(Threads.nthreads()); " *
                    "folds=$(join(folds, ","))")
    ctx = v23_build_context(v24_context_plan(plan); partitions=("DEV", "TEST"))
    mask = v24_usable_mask(ctx)
    rowview = v24_row_view()
    _v24_deploy_log(@sprintf(
        "  context: %d anchors (%d usable), %d base rows, %.1f s",
        length(ctx.anchors), count(mask.usable), nrow(rowview.table), time() - started,
    ))

    results = NamedTuple[]
    for request in requests
        push!(results, build_v2_4_bundle(ctx, rowview, mask.usable, mask.features, plan, request;
                                         indir=indir, force=force, thread_note=thread_note))
    end
    _v24_deploy_log(@sprintf("V2.4 deployment build complete in %.1f s", time() - started))
    return results
end

function main(args=ARGS)
    folds = Int[V24_DEPLOY_FOLD_YEAR, 2025]
    indir = V24_ROLLING_DIR
    deploy_out = V24_DEPLOY_DEFAULT_DIR
    replica_out = V24_DEPLOY_REPLICA_DEFAULT_DIR
    force = false
    allow_thread_drift = false
    for arg in args
        if startswith(arg, "--folds=")
            folds = [parse(Int, strip(piece))
                     for piece in split(String(split(arg, "=", limit=2)[2]), ',')
                     if !isempty(strip(piece))]
        elseif startswith(arg, "--indir=")
            indir = abspath(String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--deploy-out=")
            deploy_out = abspath(String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--replica-out=")
            replica_out = abspath(String(split(arg, "=", limit=2)[2]))
        elseif arg == "--force"
            force = true
        elseif arg == "--allow-thread-drift"
            allow_thread_drift = true
        else
            error("unknown option $arg")
        end
    end
    return build_v2_4_deploy(; folds=folds, indir=indir, deploy_out=deploy_out,
                             replica_out=replica_out, force=force,
                             allow_thread_drift=allow_thread_drift)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
