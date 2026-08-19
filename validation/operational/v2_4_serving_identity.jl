#!/usr/bin/env julia

# v2_4_serving_identity.jl — offline identity oracle for the served Operational V2.4e center.
#
# The served product and the scored artifact are produced in different environments: the live engine
# maps a minute-cadence L1 feed onto Earth-arrival hours, while the rolling-origin study reads an
# hourly OMNI archive. If the two disagree, the deployed forecast is not the model the study scored,
# and the study is not evidence about what is being served.
#
# This oracle removes that risk by driving the *serving* functions — `v24_serving_analog_features`,
# `v24_serving_analog_members`, `v24_serving_t1r_center`, `v24_serving_direct_features`,
# `v24_serving_direct_center`, `v24_serving_climatology_center` and `v24_serving_center`, the same code
# the live engine calls — with hourly archive inputs at zero anchor lag, and comparing every stage
# against the fold table the study wrote for 2025. Only the rollout-step driver policy is
# environment-specific, and it is supplied here by the package's own
# `v23_serving_step_driver_from_frame`, which restates the served L1 admission gate against an hourly
# lookup.
#
# The bundle under test is the fold-2025 *replica*: `v2_4_build_deploy.jl` refits the deployable
# bundle's own code path with the rolling fold-2025 training window and out-of-fold pool, so its
# experts are the fold's experts. The deployable bundle differs from it only in that window, which is
# why reproducing 2025 through the replica is a statement about the deployed serving path.
#
# Every column is compared, not only the final center: the seven experts the base table supplies
# (including the static V2.2 stack, which Amendment A3 made expert ten), the three the serving path
# computes, the super-learner center, the published center and both interval endpoints. A per-column
# table of the largest absolute deviation is written out, so a regression names the stage that moved.
#
# Usage (from the package root):
#   JULIA_NUM_THREADS=8 julia --startup-file=no --project=. \
#       validation/operational/v2_4_serving_identity.jl [options]
#
#   --bundle=<dir>   replica bundle to test (default …/output/operational/v2_4_replica_2025)
#   --tree=<dir>     rolling study tree holding learn_year_2025.csv / oof_year_2025.csv
#   --year=<Y>       scored year to reproduce (default 2025)
#   --anchors=<n>    minimum number of scored anchors to reproduce (default 500)
#   --out=<path>     per-row output CSV (default …/output/operational/v2_4_serving_identity.csv)

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

"Identity target: the largest absolute deviation a served stage may show against the scored one."
const V24_IDENTITY_TOL_NT = 1e-9

"Driver-history depth handed to the feature builders: the analog run-length window."
const V24_IDENTITY_HISTORY_LAGS = V23_SOUTH_RUN_CAP_H

"Default replica bundle and output location."
const V24_IDENTITY_DEFAULT_BUNDLE = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_4_replica_2025")
const V24_IDENTITY_DEFAULT_OUT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_4_serving_identity.csv")
const V24_IDENTITY_DEFAULT_TREE = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_4_rolling")

"""
    V24_IDENTITY_BASE_EXPERTS

Expert columns the base table supplies rather than the serving path computing them. They are compared
anyway: the served center consumes them by name, so a mapping error between a base-table column and an
expert slot would silently reweight the stack.
"""
const V24_IDENTITY_BASE_EXPERTS = (
    (:served_v2_1, :served_v2_1_dst_nt), (:frozen_v2_1, :frozen_v2_1_dst_nt),
    (:persistence, :persistence_dst_nt), (:burton, :burton_dst_nt),
    (:burton_full, :burton_full_dst_nt), (:obrien, :obrien_dst_nt),
)

"""
    V24_IDENTITY_SERVED_VARIANT

Study column the served center must reproduce. The served variant carries no residual layer and no
point-forecast guard, so its column is the ten-expert stack center itself; the oracle compares the
serving path's L1 center against it as well, and the two must agree with each other for the published
row to be the scored row.
"""
const V24_IDENTITY_SERVED_VARIANT = V24_SERVED_VARIANT

"Columns the oracle reports a per-column maximum deviation for."
const V24_IDENTITY_COLUMNS = (
    "served_v2_1", "frozen_v2_1", "persistence", "burton", "burton_full", "obrien",
    "static_v2_2", "t1_analog_raw", "t1r_analog", "direct_gbm", "climatology",
    "l1_center", "v2_4e", "v2_4e_lo_nt", "v2_4e_hi_nt",
)

_v24_identity_log(msg) = (println(msg); flush(stdout))

"Base-table columns the oracle needs to reconstruct the issue-time state of an anchor."
function v24_identity_table_columns()
    return String["issue_time_utc", "model_step_hours", "partition",
                  "latest_dst_nt", "V_kms", "Bz_nt", "By_nt", "n_cm3", "Pdyn_npa",
                  "dst_delta_1h_nt", "dst_delta_3h_nt", "Bz_delta_1h_nt",
                  "VBsouth_delta_1h_mvm", "VBsouth_mean_3h_mvm", "Bsouth_mean_3h_nt",
                  "VBsouth_mvm", "coupling_active_mvm",
                  "persistence_dst_nt", "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt",
                  "frozen_v2_1_dst_nt", "served_v2_1_dst_nt", "static_v2_2_dst_nt",
                  "observation_dst_nt"]
end

"""
    v24_identity_driver_history(frame_index, frame, anchor; lags) -> Vector

Driver records tagged `anchor - 1 … anchor - lags`, in lag order, exactly as the hourly frame holds
them: a record with a non-finite channel is passed through with that channel non-finite rather than
being replaced by `nothing`, and `nothing` marks only an hour the frame has no row for.

That distinction matters for identity. The study built its feature blocks from the whole frame, where
the run-length feature and the coupling lags read a record's field even when another field of the same
record is unusable. Filtering such a record out here would truncate the run length or drop a coupling
lag and produce a design the study never scored.
"""
function v24_identity_driver_history(frame_index::Dict{DateTime,Int}, frame::DataFrame,
                                     anchor::DateTime; lags::Integer=V24_IDENTITY_HISTORY_LAGS)
    history = Vector{Any}(undef, Int(lags))
    for lag in 1:Int(lags)
        row = get(frame_index, anchor - Hour(lag), 0)
        if row == 0
            history[lag] = nothing
            continue
        end
        history[lag] = (V=Float64(frame.V[row]), Bz=Float64(frame.Bz[row]),
                        By=Float64(frame.By[row]), n=Float64(frame.n[row]),
                        Pdyn=Float64(frame.Pdyn[row]))
    end
    return history
end

"""
    v24_identity_dst_history(frame_index, frame, anchor; span) -> Dict

Observed Dst of the hours `anchor - span … anchor`, as the direct-GBM design's lag ladder reads them.
Non-finite values are omitted, which is how a missing observation reaches the design.
"""
function v24_identity_dst_history(frame_index::Dict{DateTime,Int}, frame::DataFrame,
                                  anchor::DateTime; span::Integer=V24_SERVING_DST_LAG_MAX_H)
    history = Dict{DateTime,Float64}()
    for lag in 0:Int(span)
        time = anchor - Hour(lag)
        row = get(frame_index, time, 0)
        row == 0 && continue
        value = Float64(frame.Dst[row])
        isfinite(value) || continue
        history[time] = value
    end
    return history
end

"""
    v24_identity_anchor_selection(anchors, complete, latest, n_target) -> Vector{DateTime}

Deterministic anchor sample: a uniform stride over the scored anchors that carry all six model steps,
unioned with the deepest-Dst anchors so the storm branch of the stack — the deep cells and the guard —
is exercised rather than sampled away. No randomness, so a rerun compares the same rows.
"""
function v24_identity_anchor_selection(anchors::Vector{DateTime}, complete::Set{DateTime},
                                       latest::Dict{DateTime,Float64}, n_target::Int)
    usable = [t for t in anchors if t in complete]
    isempty(usable) && error("the scored table holds no anchor with all six model steps")
    quota = min(length(usable), max(n_target + n_target ÷ 5, n_target))
    stride = max(1, length(usable) ÷ quota)
    sampled = Set(usable[1:stride:end])
    deep = sort(usable; by=t -> get(latest, t, 0.0))
    for t in deep[1:min(150, length(deep))]
        push!(sampled, t)
    end
    return sort(collect(sampled))
end

"""
    run_v2_4_serving_identity(; bundle, tree, year, n_anchors, table_path, frame_path, out)

Reproduce the scored V2.4 fold table through the serving functions and report the largest absolute
deviation of every stage, per model step and per column.
"""
function run_v2_4_serving_identity(; bundle::AbstractString=V24_IDENTITY_DEFAULT_BUNDLE,
                                   tree::AbstractString=V24_IDENTITY_DEFAULT_TREE,
                                   year::Integer=2025, n_anchors::Int=500,
                                   table_path::AbstractString=V23_BASE_TABLE,
                                   frame_path::AbstractString=V23_BASE_HOURLY_FRAME,
                                   out::AbstractString=V24_IDENTITY_DEFAULT_OUT)
    scored_path = joinpath(String(tree), "learn_year_$(Int(year)).csv")
    oof_path = joinpath(String(tree), "oof_year_$(Int(year)).csv")
    isfile(scored_path) || error("scored learn-stage table is missing: $scored_path")
    isfile(oof_path) || error("scored fold table is missing: $oof_path")

    _v24_identity_log("  bundle $bundle")
    artifacts = load_v24_serving_artifacts(String(bundle))
    artifacts.fold_year == Int(year) || error(
        "the bundle was fitted for fold $(artifacts.fold_year); this oracle reproduces $(Int(year))",
    )
    _v24_identity_log(
        "  identity $(artifacts.identity), archive $(length(artifacts.analog.origins)) origins, " *
        "$(length(artifacts.cells)) stack cells, tau $(round(artifacts.climatology_tau_hours; digits=4)) h",
    )

    scored = CSV.read(scored_path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    for column in ("issue_time_utc", "model_step_hours", "usable", "fallback",
                   "v2_4e", "v2_4e_lo_nt", "v2_4e_hi_nt", "t1_analog_raw", "t1r_analog",
                   "direct_gbm", "climatology", "static_v2_2", "served_v2_1", "frozen_v2_1",
                   "persistence", "burton", "burton_full", "obrien", "regime", "depth_bin",
                   "l1_cell_regime", "l1_cell_depth", "deepening_cell")
        column in names(scored) || error("$scored_path lacks the $column column")
    end

    frame = CSV.read(frame_path, DataFrame; types=Dict("time_utc" => DateTime))
    frame_index = Dict(frame.time_utc[i] => i for i in 1:nrow(frame))
    frame_lookup = v23_serving_frame_lookup(frame)

    table = CSV.read(table_path, DataFrame; select=v24_identity_table_columns(),
                     types=Dict("issue_time_utc" => DateTime))
    state = Dict{DateTime,Any}()
    per_step = Dict{Tuple{DateTime,Int},NamedTuple}()
    for r in 1:nrow(table)
        anchor = table.issue_time_utc[r]
        step = Int(table.model_step_hours[r])
        per_step[(anchor, step)] = (
            persistence=Float64(table.persistence_dst_nt[r]),
            burton=Float64(table.burton_dst_nt[r]),
            burton_full=Float64(table.burton_full_dst_nt[r]),
            obrien=Float64(table.obrien_dst_nt[r]),
            frozen_v2_1=Float64(table.frozen_v2_1_dst_nt[r]),
            served_v2_1=Float64(table.served_v2_1_dst_nt[r]),
            static_v2_2=Float64(table.static_v2_2_dst_nt[r]),
        )
        haskey(state, anchor) && continue
        state[anchor] = (
            latest_dst=Float64(table.latest_dst_nt[r]),
            drivers=(V=Float64(table.V_kms[r]), Bz=Float64(table.Bz_nt[r]),
                     By=Float64(table.By_nt[r]), n=Float64(table.n_cm3[r]),
                     Pdyn=Float64(table.Pdyn_npa[r])),
            memory=(dst_delta_1h_nt=Float64(table.dst_delta_1h_nt[r]),
                    dst_delta_3h_nt=Float64(table.dst_delta_3h_nt[r]),
                    Bz_delta_1h_nt=Float64(table.Bz_delta_1h_nt[r]),
                    VBsouth_delta_1h_mvm=Float64(table.VBsouth_delta_1h_mvm[r]),
                    VBsouth_mean_3h_mvm=Float64(table.VBsouth_mean_3h_mvm[r]),
                    Bsouth_mean_3h_nt=Float64(table.Bsouth_mean_3h_nt[r])),
            vbsouth_mvm=Float64(table.VBsouth_mvm[r]),
            coupling_active_mvm=Float64(table.coupling_active_mvm[r]),
        )
    end
    table = DataFrame()

    target = Dict{Tuple{DateTime,Int},DataFrameRow}()
    step_count = Dict{DateTime,Int}()
    for r in 1:nrow(scored)
        row = scored[r, :]
        anchor = row.issue_time_utc
        step = Int(row.model_step_hours)
        target[(anchor, step)] = row
        step_count[anchor] = get(step_count, anchor, 0) + 1
    end
    anchors = sort(unique(scored.issue_time_utc))
    complete = Set(t for t in anchors
                   if get(step_count, t, 0) == length(V24_SERVING_MODEL_STEPS))
    latest_by_anchor = Dict(t => (haskey(state, t) ? state[t].latest_dst : 0.0) for t in anchors)
    sample = v24_identity_anchor_selection(anchors, complete, latest_by_anchor, n_anchors)
    _v24_identity_log("  scored anchors $(length(anchors)), complete $(length(complete)), " *
                      "sampled $(length(sample))")

    rows = NamedTuple[]
    fallback_anchors = 0
    skipped = 0
    guard_rows = 0
    for anchor in sample
        haskey(state, anchor) || (skipped += 1; continue)
        issue = state[anchor]
        record = get(frame_lookup, anchor - Hour(1), nothing)
        record === nothing && (skipped += 1; continue)
        max(abs(record.V - issue.drivers.V), abs(record.Bz - issue.drivers.Bz),
            abs(record.By - issue.drivers.By), abs(record.n - issue.drivers.n),
            abs(record.Pdyn - issue.drivers.Pdyn)) == 0.0 || error(
            "the archive record at $(anchor - Hour(1)) is not the base table's issue driver at $anchor",
        )
        anchor_row = get(frame_index, anchor, 0)
        anchor_row == 0 && (skipped += 1; continue)
        dst_anchor = Float64(frame.Dst[anchor_row])
        dst_anchor == issue.latest_dst || error(
            "the archive Dst at $anchor is $dst_anchor but the base table records $(issue.latest_dst)",
        )
        previous_row = get(frame_index, anchor - Hour(1), 0)
        dst_previous = previous_row == 0 ? NaN : Float64(frame.Dst[previous_row])

        history = v24_identity_driver_history(frame_index, frame, anchor)
        dst_history = v24_identity_dst_history(frame_index, frame, anchor)
        key = v24_serving_analog_features(artifacts, anchor, history, dst_anchor, dst_previous)
        design = key.ok ?
            v24_serving_direct_features(artifacts, anchor, history, dst_history;
                                        analog_features=key.features) :
            v24_serving_direct_features(artifacts, anchor, history, dst_history)
        usable = key.ok && design.ok
        # The study marks a row usable when both issue-time blocks are complete, and substitutes the
        # served V2.1 product into every learned column when they are not. The oracle therefore has to
        # agree with the study about which rows are servable, in both directions: a disagreement means
        # the live path would serve a row the study scored as a fallback, or refuse one it scored.
        scored_usable = false
        for step in V24_SERVING_MODEL_STEPS
            haskey(target, (anchor, step)) || continue
            scored_usable = Bool(target[(anchor, step)].usable)
            break
        end
        if usable != scored_usable
            error(
                "the serving path and the study disagree about anchor $anchor: serving " *
                "$(usable ? "can" : "cannot") build its blocks (analog $(key.reason), direct " *
                "$(design.reason)) while the study scored it as " *
                "$(scored_usable ? "usable" : "a served fallback")",
            )
        end
        if !usable
            fallback_anchors += 1
            for step in V24_SERVING_MODEL_STEPS
                haskey(target, (anchor, step)) || continue
                row = target[(anchor, step)]
                abs(Float64(row.v2_4e) - Float64(row.served_v2_1)) <= V24_IDENTITY_TOL_NT || error(
                    "anchor $anchor has an incomplete issue-time key but its scored center is not " *
                    "the served V2.1 value at $(step) h",
                )
            end
            continue
        end

        anchor_star = pressure_correct_dst([issue.latest_dst], [issue.drivers.Pdyn])[1]
        step_driver = v23_serving_step_driver_from_frame(frame_lookup, anchor, issue.drivers)
        for step in V24_SERVING_MODEL_STEPS
            haskey(target, (anchor, step)) || continue
            haskey(per_step, (anchor, step)) || continue
            row = target[(anchor, step)]
            panel = per_step[(anchor, step)]
            baselines = (persistence=panel.persistence, burton=panel.burton,
                         burton_full=panel.burton_full, obrien=panel.obrien)
            ensemble = v24_serving_analog_members(
                artifacts, key.features; anchor_time=anchor, issue_drv=issue.drivers,
                anchor_dst_star=anchor_star, model_steps=step, step_driver=step_driver,
            )
            t1r = v24_serving_t1r_center(
                artifacts; raw_reported=ensemble.raw_reported, latest_dst=issue.latest_dst,
                anchor_drivers=issue.drivers, memory=issue.memory, baselines=baselines,
                model_steps=step, anchor_time=anchor,
            )
            direct = v24_serving_direct_center(artifacts, design.features, step)
            climatology = v24_serving_climatology_center(artifacts, issue.latest_dst, step)
            experts = (served_v2_1=panel.served_v2_1, frozen_v2_1=panel.frozen_v2_1,
                       t1r_analog=t1r.center, persistence=panel.persistence,
                       burton=panel.burton, burton_full=panel.burton_full,
                       obrien=panel.obrien, direct_gbm=direct, climatology=climatology,
                       static_v2_2=panel.static_v2_2)
            center = v24_serving_center(
                artifacts; model_steps=step, latest_dst=issue.latest_dst,
                dst_delta_1h_nt=issue.memory.dst_delta_1h_nt, vbsouth_mvm=issue.vbsouth_mvm,
                experts=experts,
            )
            # The published center carries the physical projection the rest of the served product
            # carries; the study's column does not. They agree unless the projection acts, so the
            # comparison is made on the unprojected center and the projection is asserted inert.
            center.center == center.raw_center || error(
                "the physical projection moved the served center at $anchor / $(step) h from " *
                "$(center.raw_center) to $(center.center); the served center would not be the " *
                "scored center",
            )
            center.coupling_active_mvm == issue.coupling_active_mvm || error(
                "the gated coupling proxy recomputed at $anchor is $(center.coupling_active_mvm) " *
                "mV/m but the base table archives $(issue.coupling_active_mvm) mV/m",
            )
            String(center.regime) == String(row.regime) || error(
                "the serving regime at $anchor / $(step) h is $(center.regime); the study scored " *
                "$(row.regime)",
            )
            String(center.depth_bin) == String(row.depth_bin) || error(
                "the serving depth bin at $anchor / $(step) h is $(center.depth_bin); the study " *
                "scored $(row.depth_bin)",
            )
            String(center.cell_regime) == String(row.l1_cell_regime) || error(
                "the serving cell regime at $anchor / $(step) h is $(center.cell_regime); the " *
                "study used $(row.l1_cell_regime)",
            )
            String(center.cell_depth) == String(row.l1_cell_depth) || error(
                "the serving cell depth at $anchor / $(step) h is $(center.cell_depth); the study " *
                "used $(row.l1_cell_depth)",
            )
            center.deepening_cell == Bool(row.deepening_cell) || error(
                "the serving deepening flag at $anchor / $(step) h is $(center.deepening_cell); " *
                "the study scored $(row.deepening_cell)",
            )
            # The served bundle disables the point-forecast guard, so the published center must be the
            # stack center on every row, deepening or not. Asserting it here is what keeps a bundle that
            # silently re-enabled the guard from passing the oracle on the rows where it never acts.
            artifacts.guard_applied || center.raw_center == center.l1_center || error(
                "the served bundle records no point-forecast guard, but the center at $anchor / " *
                "$(step) h moved from $(center.l1_center) to $(center.raw_center)",
            )
            center.guard_applied && (guard_rows += 1)
            push!(rows, (
                issue_time_utc=string(anchor), model_step_hours=step,
                regime=String(center.regime), depth_bin=String(center.depth_bin),
                deepening_cell=center.deepening_cell, guard_applied=center.guard_applied,
                d_served_v2_1=abs(panel.served_v2_1 - Float64(row.served_v2_1)),
                d_frozen_v2_1=abs(panel.frozen_v2_1 - Float64(row.frozen_v2_1)),
                d_persistence=abs(panel.persistence - Float64(row.persistence)),
                d_burton=abs(panel.burton - Float64(row.burton)),
                d_burton_full=abs(panel.burton_full - Float64(row.burton_full)),
                d_obrien=abs(panel.obrien - Float64(row.obrien)),
                d_static_v2_2=abs(panel.static_v2_2 - Float64(row.static_v2_2)),
                d_t1_analog_raw=abs(ensemble.raw_reported - Float64(row.t1_analog_raw)),
                d_t1r_analog=abs(t1r.center - Float64(row.t1r_analog)),
                d_direct_gbm=abs(direct - Float64(row.direct_gbm)),
                d_climatology=abs(climatology - Float64(row.climatology)),
                d_l1_center=abs(center.l1_center - Float64(row.v2_4e)),
                d_v2_4e=abs(center.raw_center - Float64(row.v2_4e)),
                d_v2_4e_lo_nt=abs(center.ci05_nt - Float64(row.v2_4e_lo_nt)),
                d_v2_4e_hi_nt=abs(center.ci95_nt - Float64(row.v2_4e_hi_nt)),
                serving_v2_4e_nt=center.raw_center, scored_v2_4e_nt=Float64(row.v2_4e),
                serving_l1_center_nt=center.l1_center,
                half_width_nt=center.half_width_nt,
            ))
        end
    end
    isempty(rows) && error("the identity oracle reproduced no scored center")
    result = DataFrame(rows)
    mkpath(dirname(String(out)))
    CSV.write(String(out), result)

    per_column = NamedTuple[]
    worst = 0.0
    worst_column = "none"
    for column in V24_IDENTITY_COLUMNS
        values = result[!, Symbol("d_", column)]
        value = maximum(values)
        push!(per_column, (column=column, n=length(values), max_abs_delta_nt=value))
        if value > worst
            worst = value
            worst_column = column
        end
    end
    per_step_rows = NamedTuple[]
    for step in V24_SERVING_MODEL_STEPS
        subset = result[result.model_step_hours .== step, :]
        nrow(subset) == 0 && continue
        push!(per_step_rows, (
            model_step_hours=step, n=nrow(subset),
            max_abs_delta_center_nt=maximum(subset.d_v2_4e),
            max_abs_delta_l1_nt=maximum(subset.d_l1_center),
            max_abs_delta_t1r_nt=maximum(subset.d_t1r_analog),
            max_abs_delta_direct_nt=maximum(subset.d_direct_gbm),
            max_abs_delta_interval_nt=max(maximum(subset.d_v2_4e_lo_nt),
                                          maximum(subset.d_v2_4e_hi_nt)),
            guard_rows=count(subset.guard_applied),
        ))
    end
    anchors_checked = length(unique(result.issue_time_utc))
    _v24_identity_log("  anchors reproduced $anchors_checked, rows $(nrow(result)), " *
                      "served-fallback anchors $fallback_anchors, skipped $skipped, " *
                      "guarded rows $guard_rows")
    for row in per_step_rows
        _v24_identity_log(@sprintf(
            "    step %d h: n=%d  max|Δ| center = %.3e  L1 = %.3e  T1r = %.3e  GBM = %.3e  interval = %.3e nT  guarded=%d",
            row.model_step_hours, row.n, row.max_abs_delta_center_nt, row.max_abs_delta_l1_nt,
            row.max_abs_delta_t1r_nt, row.max_abs_delta_direct_nt,
            row.max_abs_delta_interval_nt, row.guard_rows))
    end
    for row in per_column
        _v24_identity_log(@sprintf("    column %-16s n=%d  max|Δ| = %.3e nT", row.column, row.n,
                                   row.max_abs_delta_nt))
    end
    passed = worst <= V24_IDENTITY_TOL_NT && anchors_checked >= n_anchors
    _v24_identity_log(@sprintf(
        "  overall max|Δ| = %.3e nT on %s (target %.0e), anchors %d (target %d): %s",
        worst, worst_column, V24_IDENTITY_TOL_NT, anchors_checked, n_anchors,
        passed ? "PASS" : "FAIL"))
    return (rows=result, per_column=DataFrame(per_column), per_step=DataFrame(per_step_rows),
            worst_nt=worst, worst_column=worst_column, anchors=anchors_checked,
            fallback=fallback_anchors, skipped=skipped, guard_rows=guard_rows, passed=passed,
            bundle=String(bundle), tree=String(tree), out=String(out))
end

function main_v2_4_serving_identity(args=ARGS)
    bundle = V24_IDENTITY_DEFAULT_BUNDLE
    tree = V24_IDENTITY_DEFAULT_TREE
    year = 2025
    n_anchors = 500
    out = V24_IDENTITY_DEFAULT_OUT
    for arg in args
        if startswith(arg, "--bundle=")
            bundle = abspath(String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--tree=")
            tree = abspath(String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--year=")
            year = parse(Int, String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--anchors=")
            n_anchors = parse(Int, String(split(arg, "=", limit=2)[2]))
        elseif startswith(arg, "--out=")
            out = abspath(String(split(arg, "=", limit=2)[2]))
        else
            error("unknown option $arg")
        end
    end
    _v24_identity_log("V2.4e served identity oracle")
    _v24_identity_log("  tree $tree")
    result = run_v2_4_serving_identity(; bundle=bundle, tree=tree, year=year,
                                      n_anchors=n_anchors, out=out)
    result.passed || error(@sprintf(
        "V2.4e served identity FAILED: max|Δ| = %.3e nT on %s over %d anchors",
        result.worst_nt, result.worst_column, result.anchors))
    _v24_identity_log("V2.4e served identity PASS")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_4_serving_identity()
end
