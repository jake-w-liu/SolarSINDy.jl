#!/usr/bin/env julia

# v2_3_serving_identity.jl — offline identity oracle for the V2.3 shadow center.
#
# The shadow product and the scored artifact are produced in different environments: the live engine
# maps a minute-cadence L1 feed onto Earth-arrival hours, while the confirmatory runner reads an
# hourly OMNI archive. If the two disagree, the monitored shadow forecast is not the model that was
# scored, and the shadow record would not be evidence about the scored candidate at all.
#
# This oracle removes that risk by driving the *serving* functions — `v23_serving_features`,
# `v23_serving_members`, `v23_serving_center`, the same code the live engine calls — with hourly
# archive inputs at zero anchor lag, and comparing the result with the `V2_3_final` column the
# scoring run wrote. Only the rollout-step driver policy is environment-specific, and it is supplied
# here by the package's own `v23_serving_step_driver_from_frame`, which restates the served L1
# admission gate against an hourly lookup.
#
# The error-layer innovation history is taken from the same scored table: the layer is defined
# against the one-step center *before* the layer acts, which is the `V2_3_LAT` column, so the history
# is `observation - V2_3_LAT` at `model_step_hours = 1` over the six preceding anchors.
#
# Usage (from the package root):
#   julia --startup-file=no --project=. validation/operational/v2_3_serving_identity.jl [options]
#
#   --deploy=<dir>    deployment directory to test (default deploy/v2_3_shadow)
#   --tree=<dir>      scored run tree holding test_scores.csv (default …/v2_3_test)
#   --anchors=<n>     minimum number of scored anchors to reproduce (default 500)
#   --out=<path>      per-row output CSV (default …/v2_3_serving_identity.csv)
#   --wait            poll for logs/test_confirm.exit before reading the confirmatory tree, and
#                     fall back to the smoke tree (and its deployment) if it never appears

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

"Identity target: the largest absolute deviation a served center may show against the scored one."
const V23_IDENTITY_TOL_NT = 1e-9

"Driver-history depth handed to the analog key: seven mandatory lags plus the run-length window."
const V23_IDENTITY_HISTORY_LAGS = V23_SOUTH_RUN_CAP_H

"Anchors the live error-layer innovation rule must reproduce against the scored history."
const V23_IDENTITY_MIN_CHAIN_ANCHORS = 300

"Sentinel poll: 90 attempts one minute apart bounds the wait at 90 minutes."
const V23_IDENTITY_SENTINEL_ATTEMPTS = 90
const V23_IDENTITY_SENTINEL_SLEEP_S = 60.0

_v23_identity_log(msg) = (println(msg); flush(stdout))

"Base-table columns the offline oracle needs to reconstruct the issue-time state of an anchor."
function v23_identity_table_columns()
    return String["issue_time_utc", "model_step_hours", "partition",
                  "latest_dst_nt", "V_kms", "Bz_nt", "By_nt", "n_cm3", "Pdyn_npa",
                  "dst_delta_1h_nt", "dst_delta_3h_nt", "Bz_delta_1h_nt",
                  "VBsouth_delta_1h_mvm", "VBsouth_mean_3h_mvm", "Bsouth_mean_3h_nt",
                  "persistence_dst_nt", "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt",
                  "frozen_v2_1_dst_nt", "served_v2_1_dst_nt", "observation_dst_nt"]
end

"""
    v23_identity_driver_history(frame_index, frame, anchor; lags) -> Vector

Driver records tagged `anchor - 1 … anchor - lags`, in lag order, as the serving feature builder
expects them. A record whose channels are not all finite (or whose density is not positive) is
returned as `nothing`, which is how the live path reports an unmeasured hour.
"""
function v23_identity_driver_history(frame_index::Dict{DateTime,Int}, frame::DataFrame,
                                     anchor::DateTime; lags::Integer = V23_IDENTITY_HISTORY_LAGS)
    history = Vector{Any}(undef, Int(lags))
    for lag in 1:Int(lags)
        row = get(frame_index, anchor - Hour(lag), 0)
        if row == 0
            history[lag] = nothing
            continue
        end
        record = (V = Float64(frame.V[row]), Bz = Float64(frame.Bz[row]),
                  By = Float64(frame.By[row]), n = Float64(frame.n[row]),
                  Pdyn = Float64(frame.Pdyn[row]))
        history[lag] = (all(isfinite, values(record)) && record.n > 0) ? record : nothing
    end
    return history
end

"""
    v23_identity_anchor_selection(anchors, scores_by_step, n_target) -> Vector{DateTime}

Deterministic anchor sample: a uniform stride over the scored anchors that carry all six model
steps, unioned with the deepest-Dst anchors so the storm branch of the pipeline is exercised rather
than sampled away. No randomness, so a rerun compares the same rows.
"""
function v23_identity_anchor_selection(anchors::Vector{DateTime}, complete::Set{DateTime},
                                       latest::Dict{DateTime,Float64}, n_target::Int)
    usable = [t for t in anchors if t in complete]
    isempty(usable) && error("the scored table holds no anchor with all six model steps")
    quota = min(length(usable), max(n_target + n_target ÷ 5, n_target))
    stride = max(1, length(usable) ÷ quota)
    sampled = Set(usable[1:stride:end])
    deep = sort(usable; by = t -> get(latest, t, 0.0))
    for t in deep[1:min(100, length(deep))]
        push!(sampled, t)
    end
    return sort(collect(sampled))
end

"""
    run_v2_3_serving_identity(; deploy, tree, n_anchors, table_path, frame_path, out) -> NamedTuple

Reproduce the scored `V2_3_final` centers through the serving functions and report the largest
absolute deviation at every model step.
"""
function run_v2_3_serving_identity(; deploy::AbstractString,
                                   tree::AbstractString,
                                   n_anchors::Int = 500,
                                   table_path::AbstractString = V23_BASE_TABLE,
                                   frame_path::AbstractString = V23_BASE_HOURLY_FRAME,
                                   out::AbstractString = joinpath(
                                       OPERATIONAL_OUTPUT_DIR, "v2_3_serving_identity.csv"))
    scores_path = joinpath(String(tree), "test_scores.csv")
    isfile(scores_path) || error("scored table is missing: $scores_path")
    _v23_identity_log("  deployment $deploy")
    artifacts = load_v23_serving_artifacts(String(deploy))
    _v23_identity_log("  identity $(artifacts.identity), archive $(length(artifacts.origins)) origins")

    v2_1_calibration = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv)
    scores = CSV.read(scores_path, DataFrame; types = Dict("issue_time_utc" => DateTime))
    for column in ("V2_3_final", "V2_3_LAT", "served_v2_1", "observation_dst_nt",
                   "model_step_hours", "issue_time_utc")
        column in names(scores) || error("$scores_path lacks the $column column")
    end
    frame = CSV.read(frame_path, DataFrame; types = Dict("time_utc" => DateTime))
    frame_index = Dict(frame.time_utc[i] => i for i in 1:nrow(frame))
    frame_lookup = v23_serving_frame_lookup(frame)

    table = CSV.read(table_path, DataFrame; select = v23_identity_table_columns(),
                     types = Dict("issue_time_utc" => DateTime))
    state = Dict{DateTime,Any}()
    frozen = Dict{Tuple{DateTime,Int},Float64}()
    # The physical baselines are advanced along the rollout, so `burton`, `burton_full` and `obrien`
    # are per `(anchor, model step)` quantities: caching them per anchor would silently hand every
    # step the one-hour baseline panel and shift the correction's expert features at every longer
    # lead. Only the issue-time state and the timestamp-keyed memory lags are per anchor.
    baselines = Dict{Tuple{DateTime,Int},NamedTuple}()
    for r in 1:nrow(table)
        anchor = table.issue_time_utc[r]
        step = Int(table.model_step_hours[r])
        frozen[(anchor, step)] = Float64(table.frozen_v2_1_dst_nt[r])
        baselines[(anchor, step)] = (
            persistence = Float64(table.persistence_dst_nt[r]),
            burton = Float64(table.burton_dst_nt[r]),
            burton_full = Float64(table.burton_full_dst_nt[r]),
            obrien = Float64(table.obrien_dst_nt[r]),
        )
        haskey(state, anchor) && continue
        state[anchor] = (
            latest_dst = Float64(table.latest_dst_nt[r]),
            drivers = (V = Float64(table.V_kms[r]), Bz = Float64(table.Bz_nt[r]),
                       By = Float64(table.By_nt[r]), n = Float64(table.n_cm3[r]),
                       Pdyn = Float64(table.Pdyn_npa[r])),
            memory = (dst_delta_1h_nt = Float64(table.dst_delta_1h_nt[r]),
                      dst_delta_3h_nt = Float64(table.dst_delta_3h_nt[r]),
                      Bz_delta_1h_nt = Float64(table.Bz_delta_1h_nt[r]),
                      VBsouth_delta_1h_mvm = Float64(table.VBsouth_delta_1h_mvm[r]),
                      VBsouth_mean_3h_mvm = Float64(table.VBsouth_mean_3h_mvm[r]),
                      Bsouth_mean_3h_nt = Float64(table.Bsouth_mean_3h_nt[r])),
        )
    end
    table = DataFrame()

    scored = Dict{Tuple{DateTime,Int},NamedTuple}()
    innovations = Dict{DateTime,Float64}()
    step_count = Dict{DateTime,Int}()
    for r in 1:nrow(scores)
        anchor = scores.issue_time_utc[r]
        step = Int(scores.model_step_hours[r])
        final = Float64(scores.V2_3_final[r])
        lat = Float64(scores.V2_3_LAT[r])
        served = Float64(scores.served_v2_1[r])
        scored[(anchor, step)] = (final = final, lat = lat, served = served)
        step_count[anchor] = get(step_count, anchor, 0) + 1
        if step == 1
            observation = scores.observation_dst_nt[r]
            if !ismissing(observation) && isfinite(Float64(observation)) && isfinite(lat)
                innovations[anchor] = Float64(observation) - lat
            end
        end
    end
    # The live engine cannot filter the log for one-hour rows: it issues wall horizons at an anchor
    # lag, so no logged row carries a one-hour model step. It instead records the one-hour pre-layer
    # center of every anchor and forms the innovation as `Dst(anchor + 1 h) - center_1h(anchor)`
    # through `v23_serving_innovations_from_step1_centers`. That rule must reproduce the history built
    # above from the scored table, or the live error layer is a different layer from the scored one.
    step1_centers = Dict{DateTime,Float64}()
    matured_dst = Dict{DateTime,Float64}()
    for r in 1:nrow(scores)
        Int(scores.model_step_hours[r]) == 1 || continue
        anchor = scores.issue_time_utc[r]
        lat = Float64(scores.V2_3_LAT[r])
        isfinite(lat) || continue
        step1_centers[anchor] = lat
        observation = scores.observation_dst_nt[r]
        (!ismissing(observation) && isfinite(Float64(observation))) || continue
        matured_dst[anchor + Hour(1)] = Float64(observation)
    end
    live_rule = v23_serving_innovations_from_step1_centers(step1_centers, matured_dst)
    chain_anchors = length(live_rule)
    chain_worst = 0.0
    for (anchor, value) in live_rule
        haskey(innovations, anchor) || error(
            "the live innovation rule produced an anchor the scored history does not have: $anchor",
        )
        chain_worst = max(chain_worst, abs(value - innovations[anchor]))
    end
    length(innovations) == chain_anchors || error(
        "the scored innovation history has $(length(innovations)) anchors but the live rule " *
        "produced $chain_anchors",
    )
    chain_worst <= V23_IDENTITY_TOL_NT || error(@sprintf(
        "the live innovation rule deviates from the scored history by %.3e nT", chain_worst))
    _v23_identity_log(@sprintf(
        "  error-layer chain: live rule reproduces %d scored innovations, max|Δ| = %.3e nT",
        chain_anchors, chain_worst))

    anchors = sort(unique(scores.issue_time_utc))
    complete = Set(t for t in anchors
                   if get(step_count, t, 0) == length(V23_SERVING_MODEL_STEPS))
    latest_by_anchor = Dict(t => (haskey(state, t) ? state[t].latest_dst : 0.0) for t in anchors)
    sample = v23_identity_anchor_selection(anchors, complete, latest_by_anchor, n_anchors)
    _v23_identity_log("  scored anchors $(length(anchors)), complete $(length(complete)), " *
                      "sampled $(length(sample))")

    rows = NamedTuple[]
    reproduced = 0
    fallback = 0
    skipped = 0
    for anchor in sample
        if !haskey(state, anchor)
            skipped += 1
            continue
        end
        issue = state[anchor]
        # Anchor-time inputs of the offline path must be the same information the serving functions
        # would see live: the archive record tagged t-1 and the observed Dst at t and t-1.
        record = get(frame_lookup, anchor - Hour(1), nothing)
        if record === nothing
            skipped += 1
            continue
        end
        max(abs(record.V - issue.drivers.V), abs(record.Bz - issue.drivers.Bz),
            abs(record.By - issue.drivers.By), abs(record.n - issue.drivers.n),
            abs(record.Pdyn - issue.drivers.Pdyn)) == 0.0 || error(
            "the archive record at $(anchor - Hour(1)) is not the base table's issue driver at $anchor",
        )
        anchor_row = get(frame_index, anchor, 0)
        previous_row = get(frame_index, anchor - Hour(1), 0)
        if anchor_row == 0 || previous_row == 0
            skipped += 1
            continue
        end
        dst_anchor = Float64(frame.Dst[anchor_row])
        dst_previous = Float64(frame.Dst[previous_row])
        dst_anchor == issue.latest_dst || error(
            "the archive Dst at $anchor is $dst_anchor but the base table records $(issue.latest_dst)",
        )
        history = v23_identity_driver_history(frame_index, frame, anchor)
        key = v23_serving_features(artifacts, anchor, history, dst_anchor, dst_previous)
        if !key.ok
            fallback += 1
            for step in V23_SERVING_MODEL_STEPS
                haskey(scored, (anchor, step)) || continue
                target = scored[(anchor, step)]
                abs(target.final - target.served) <= V23_IDENTITY_TOL_NT || error(
                    "anchor $anchor has an incomplete analog key but its scored center is not the " *
                    "served V2.1 value at $(step) h",
                )
            end
            continue
        end
        anchor_star = pressure_correct_dst([issue.latest_dst], [issue.drivers.Pdyn])[1]
        step_driver = v23_serving_step_driver_from_frame(frame_lookup, anchor, issue.drivers)
        innovation = v23_serving_innovation_lags(anchor, innovations)
        for step in V23_SERVING_MODEL_STEPS
            haskey(scored, (anchor, step)) || continue
            haskey(frozen, (anchor, step)) || continue
            haskey(baselines, (anchor, step)) || continue
            target = scored[(anchor, step)]
            step_baselines = baselines[(anchor, step)]
            ensemble = v23_serving_members(
                artifacts, key.features; anchor_time = anchor, issue_drv = issue.drivers,
                anchor_dst_star = anchor_star, model_steps = step, step_driver = step_driver,
            )
            # The lead-aware blend partner is recomputed rather than read back, because the live
            # shadow path has to recompute it too: the served engine's own `v2_pred_dst_nt` admits
            # L1-measured hours into the core rollout, while the scored blend partner holds the issue
            # driver for every step. Checking the recomputation against the archived column here is
            # what makes the live blend partner trustworthy.
            frozen_center = v23_serving_frozen_center(
                artifacts; v2_1_calibration = v2_1_calibration, issue_drv = issue.drivers,
                anchor_dst_star = anchor_star, latest_dst = issue.latest_dst,
                memory = issue.memory, baselines = step_baselines, model_steps = step,
                anchor_time = anchor,
            )
            frozen_delta = abs(frozen_center.center - frozen[(anchor, step)])
            center = v23_serving_center(
                artifacts; raw_reported = ensemble.raw_reported, latest_dst = issue.latest_dst,
                anchor_drivers = issue.drivers, memory = issue.memory,
                baselines = step_baselines, model_steps = step,
                frozen_v2_1 = frozen_center.center, analog_features = key.features,
                anchor_time = anchor,
                innovations = innovation.ok ? innovation.values : nothing,
            )
            push!(rows, (
                issue_time_utc = string(anchor),
                model_step_hours = step,
                scored_final_nt = target.final,
                serving_final_nt = center.final,
                abs_delta_nt = abs(center.final - target.final),
                scored_lat_nt = target.lat,
                serving_lat_nt = center.center,
                abs_delta_lat_nt = abs(center.center - target.lat),
                raw_dst_nt = ensemble.raw_reported,
                scored_frozen_v2_1_nt = frozen[(anchor, step)],
                serving_frozen_v2_1_nt = frozen_center.center,
                abs_delta_frozen_nt = frozen_delta,
                e_layer = center.e_layer,
                e_layer_applied = center.e_layer_applied,
                e_delta_nt = center.e_delta,
            ))
            reproduced += 1
        end
    end
    isempty(rows) && error("the identity oracle reproduced no scored center")
    result = DataFrame(rows)
    mkpath(dirname(String(out)))
    CSV.write(String(out), result)

    per_step = NamedTuple[]
    for step in V23_SERVING_MODEL_STEPS
        subset = result[result.model_step_hours .== step, :]
        nrow(subset) == 0 && continue
        push!(per_step, (model_step_hours = step, n = nrow(subset),
                         max_abs_delta_nt = maximum(subset.abs_delta_nt),
                         max_abs_delta_lat_nt = maximum(subset.abs_delta_lat_nt),
                         max_abs_delta_frozen_nt = maximum(subset.abs_delta_frozen_nt),
                         e_layer_rows = count(subset.e_layer_applied)))
    end
    worst = maximum(result.abs_delta_nt)
    worst_frozen = maximum(result.abs_delta_frozen_nt)
    anchors_checked = length(unique(result.issue_time_utc))
    _v23_identity_log("  anchors reproduced $anchors_checked, rows $(nrow(result)), " *
                      "served-fallback anchors $fallback, skipped $skipped")
    for row in per_step
        _v23_identity_log(@sprintf(
            "    step %d h: n=%d  max|Δ| final = %.3e  pre-layer = %.3e  frozen = %.3e nT  layer rows=%d",
            row.model_step_hours, row.n, row.max_abs_delta_nt, row.max_abs_delta_lat_nt,
            row.max_abs_delta_frozen_nt, row.e_layer_rows))
    end
    passed = worst <= V23_IDENTITY_TOL_NT && worst_frozen <= V23_IDENTITY_TOL_NT &&
             anchors_checked >= n_anchors &&
             chain_worst <= V23_IDENTITY_TOL_NT &&
             chain_anchors >= V23_IDENTITY_MIN_CHAIN_ANCHORS
    _v23_identity_log(@sprintf("  overall max|Δ| = %.3e nT (target %.0e), anchors %d (target %d): %s",
                               worst, V23_IDENTITY_TOL_NT, anchors_checked, n_anchors,
                               passed ? "PASS" : "FAIL"))
    _v23_identity_log(@sprintf("  frozen V2.1 blend partner max|Δ| = %.3e nT", worst_frozen))
    return (rows = result, per_step = DataFrame(per_step), worst_nt = worst,
            worst_frozen_nt = worst_frozen,
            chain_anchors = chain_anchors, chain_worst_nt = chain_worst,
            anchors = anchors_checked, fallback = fallback, skipped = skipped,
            passed = passed, tree = String(tree), deploy = String(deploy), out = String(out))
end

"""
    v23_identity_wait_for_sentinel(path; attempts, sleep_s) -> Bool

Bounded wait for the confirmatory run's exit sentinel. The loop is capped by attempt count, so it
terminates whether the run succeeds, fails, or dies without writing anything.
"""
function v23_identity_wait_for_sentinel(path::AbstractString;
                                        attempts::Int = V23_IDENTITY_SENTINEL_ATTEMPTS,
                                        sleep_s::Real = V23_IDENTITY_SENTINEL_SLEEP_S)
    for attempt in 1:attempts
        isfile(path) && return true
        attempt == attempts && break
        _v23_identity_log("  waiting for $(basename(path)) ($attempt/$attempts)")
        sleep(Float64(sleep_s))
    end
    return isfile(path)
end

function main_v2_3_serving_identity(args = ARGS)
    deploy = joinpath(OPERATIONAL_PACKAGE_ROOT, "deploy", "v2_3_shadow")
    tree = V23_TEST_DIR
    n_anchors = 500
    out = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_3_serving_identity.csv")
    wait_for_sentinel = false
    for arg in args
        if startswith(arg, "--deploy=")
            deploy = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--tree=")
            tree = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--anchors=")
            n_anchors = parse(Int, String(split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--out=")
            out = abspath(String(split(arg, "=", limit = 2)[2]))
        elseif arg == "--wait"
            wait_for_sentinel = true
        else
            error("unknown option $arg")
        end
    end
    if wait_for_sentinel
        sentinel = joinpath(OPERATIONAL_PACKAGE_ROOT, "logs", "test_confirm.exit")
        if !v23_identity_wait_for_sentinel(sentinel)
            _v23_identity_log(
                "  ! $(sentinel) never appeared; running against the pseudo-TEST smoke tree instead",
            )
            tree = V23_TEST_SMOKE_DIR
        end
    end
    _v23_identity_log("V2.3 shadow identity oracle")
    _v23_identity_log("  tree $tree")
    result = run_v2_3_serving_identity(; deploy = deploy, tree = tree, n_anchors = n_anchors,
                                       out = out)
    result.passed || error(@sprintf(
        "V2.3 shadow identity FAILED: max|Δ| = %.3e nT (frozen %.3e nT) over %d anchors; error-layer chain %.3e nT over %d anchors",
        result.worst_nt, result.worst_frozen_nt, result.anchors,
        result.chain_worst_nt, result.chain_anchors))
    _v23_identity_log("V2.3 shadow identity PASS")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_3_serving_identity()
end
