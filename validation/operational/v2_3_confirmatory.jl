#!/usr/bin/env julia

# v2_3_confirmatory.jl — Operational V2.3 single-shot confirmatory runner
# (plan sections 5 and 6).
#
# Refits the analog archive, the boosted driver continuation, the direct
# machine-learning comparator and the error layers on all of DEV with the
# hyperparameters the development stage selected, scores every preregistered
# variant and every comparator on the TEST partition, and evaluates the
# preregistered gates with the paired fixed non-overlapping 168 h calendar-block
# bootstrap (V2.2 convention) and the Holm step-down of
# `src/operational_v23_stats.jl`.
#
# The preregistered family includes the Amendment A1 members: for every analog
# configuration whose all-DEV refit calibration the development stage persisted,
# the analog raw core scored here is corrected by that calibration instead of by
# the deployed V2.1 layer, in both safeguard options. When the selection names a
# T1r member, the candidate center is that corrected core composed with the
# frozen tail by the development weights and finished by the fixed error layers.
#
# TEST is scored once. The runner therefore refuses to touch TEST rows without
# `--confirm`, and refuses a second confirmed run unless it is given an explicit
# written reason, which it appends to `rerun_log.csv` alongside the decision it
# is about to overwrite.
#
#   julia --project=. validation/operational/v2_3_confirmatory.jl --smoke
#   JULIA_NUM_THREADS=8 julia --project=. validation/operational/v2_3_confirmatory.jl --confirm
#
# `--smoke` scores a late DEV slice as a pseudo-TEST partition into a separate
# output tree. It exercises the whole pipeline, including the gates, without
# reading a single TEST row, and its decision record is labelled as such.

# `v2_3_t1r.jl` pulls in `v2_3_common.jl` and adds the Amendment A1 correction
# layer: the feature frame of the analog path, the refit-calibration reader and
# the two safeguard variants of a T1r center. The confirmatory runner scores that
# family alongside the original one, so it loads the layer rather than restating
# it.
include(joinpath(@__DIR__, "v2_3_t1r.jl"))

const V23_CONFIRM_SELECTION_JSON = "selected_configuration.json"
"""
    V23_SMOKE_PSEUDO_TEST_FIRST_ISSUE

Pseudo-TEST slice of the smoke path: late DEV issues, capped. The window starts
in May 2019 so that the slice contains disturbed hours (11 and 14 May, 5 August
and 1–2 September 2019 all reach Dst ≤ −50 nT). A quiet slice would run the
depth cells and the intense-deepening guard on an empty sample, which exercises
the gate arithmetic without ever exercising the storm branch it exists for.
"""
const V23_SMOKE_PSEUDO_TEST_FIRST_ISSUE = DateTime(2019, 5, 1)
const V23_SMOKE_PSEUDO_TEST_ANCHORS = 3000
"Recovery-phase thresholds of the climatology-relaxed persistence ablation."
const V23_CLIMATOLOGY_MIN_DEPTH_NT = -20.0
"Model steps whose gain carries the Tier A accuracy requirement."
const V23_GATE_A1_STEPS = (3, 4, 6, 7)
"Model steps that only have to be non-inferior."
const V23_GATE_A1_NONINFERIOR_STEPS = (1, 2)
const V23_GATE_A1_MIN_GAIN_NT = 0.25
const V23_GATE_A1_MIN_GAIN_FRACTION = 0.02
const V23_GATE_A1_NONINFERIORITY_NT = -0.10
"Model steps of the Tier B1 SINDy-line claim."
const V23_GATE_B1_STEPS = (6, 7)
const V23_GATE_B1_MIN_GAIN_NT = 0.25
"Largest deficit against the direct comparator that still permits a local-superiority claim."
const V23_GATE_B2_TOLERANCE_NT = 0.25
"Family-wise level of every Holm-adjusted gate."
const V23_GATE_ALPHA = 0.05
"Anchors sampled for the in-run post-issue mutation-invariance check."
const V23_INTEGRITY_SAMPLE = 500

"""
    v23_test_plan(; smoke) -> V23RunPlan

Scale of a confirmatory run. The confirmed run carries the full preregistered
grid and every eligible anchor; the smoke run reduces both, and writes to its
own tree.
"""
function v23_test_plan(; smoke::Bool=false)
    smoke && return V23RunPlan(
        "test_smoke", V23_TEST_SMOKE_DIR, true, Int[], V23_SMOKE_PSEUDO_TEST_ANCHORS,
        collect(V23_WEIGHT_SETS), [25, 50], [(4, 200)], [(4, 200)],
    )
    return V23RunPlan(
        "test", V23_TEST_DIR, false, Int[], 0, collect(V23_WEIGHT_SETS),
        collect(V23_T1_K_GRID), collect(V23_GDC_GRID), collect(V23_DIRECT_GRID),
    )
end

"""
    V23_CONFIRM_SELECTION_TRACE

Selection trace the contract is digested against. `v2_3_select.jl` writes the
trace and the contract in the same pass, so a contract whose recorded digest no
longer matches the trace beside it was written by a different selection than the
one on disk.
"""
const V23_CONFIRM_SELECTION_TRACE = "selection_trace.csv"
const V23_CONFIRM_DEV_COMPOSITION = "oof_V2_3_LAT.csv"

"""
    _v23_selection_digest_check(name, recorded, observed, drift_reason, mismatches)

One digest comparison of the input contract. A mismatch is fatal unless the
operator supplied a written reason, in which case it is collected for the
manifest instead: the single-shot rule makes an unexplained input change a
protocol failure, not a warning.
"""
function _v23_selection_digest_check(name::AbstractString, recorded, observed,
                                     drift_reason, mismatches::Vector{String})
    recorded === nothing && error(
        "the development selection records no $name; rerun v2_3_select.jl so the " *
        "confirmatory input contract can be verified",
    )
    String(recorded) == String(observed) && return nothing
    message = "$name: contract $(String(recorded)) but this run observes $(String(observed))"
    drift_reason === nothing && error(
        "the development selection does not describe this run — $message. Rerun " *
        "v2_3_select.jl against the current sources and inputs, or pass " *
        "--accept-code-drift=<reason> to score TEST anyway and record why.",
    )
    push!(mismatches, message)
    return nothing
end

"""
    _v23_read_selection(dev_dir, base_table_sha; frame_sha, code_sha, accept_code_drift)
        -> NamedTuple

Read the development selection and refuse it unless it describes *this* run: the
same base table, the same hourly frame, the same V2.3 sources, the selection
trace that produced it and the composed out-of-fold rows the fixed error layers
are refitted from. The configuration scored on TEST must be the configuration
that DEV selected, on the same data and by the same code.

Digest mismatches are fatal unless `accept_code_drift` carries a written reason,
which the caller records in the run manifest. A contract that does not carry the
digests at all is a pre-Amendment contract and is refused outright: it names a
selection made over a smaller family than the one the amendment preregistered.
"""
function _v23_read_selection(dev_dir::AbstractString, base_table_sha::AbstractString;
                             frame_sha::Union{Nothing,AbstractString}=nothing,
                             code_sha::Union{Nothing,AbstractString}=nothing,
                             accept_code_drift::Union{Nothing,String}=nothing)
    path = joinpath(dev_dir, V23_CONFIRM_SELECTION_JSON)
    isfile(path) || error(
        "development selection is missing: $path (run v2_3_development.jl first)",
    )
    record = JSON3.read(read(path, String))
    mismatches = String[]
    String(record["base_table_sha256"]) == base_table_sha || error(
        "development selection was made against base table " *
        "$(record["base_table_sha256"]) but this run reads $(base_table_sha)",
    )
    trace_path = joinpath(dev_dir, V23_CONFIRM_SELECTION_TRACE)
    isfile(trace_path) || error(
        "the selection trace the contract is digested against is missing: $trace_path",
    )
    _v23_selection_digest_check(
        "selection_trace_sha256", get(record, :selection_trace_sha256, nothing),
        _v23_sha256_file(trace_path), accept_code_drift, mismatches,
    )
    composition_path = joinpath(dev_dir, V23_CONFIRM_DEV_COMPOSITION)
    isfile(composition_path) || error(
        "the development composition the error layers are refitted from is missing: " *
        "$composition_path",
    )
    _v23_selection_digest_check(
        "oof_v2_3_lat_sha256", get(record, :oof_v2_3_lat_sha256, nothing),
        _v23_sha256_file(composition_path), accept_code_drift, mismatches,
    )
    code_sha === nothing || _v23_selection_digest_check(
        "code_sha256", get(record, :code_sha256, nothing), code_sha,
        accept_code_drift, mismatches,
    )
    frame_sha === nothing || _v23_selection_digest_check(
        "hourly_frame_sha256", get(record, :hourly_frame_sha256, nothing), frame_sha,
        accept_code_drift, mismatches,
    )
    layers = Vector{Any}(undef, V23_STEP_COUNT)
    fill!(layers, nothing)
    for (slot, entry) in enumerate(record["e_layers"])
        entry === nothing && continue
        kind = Symbol(String(entry["layer"]))
        # E1 stores its ridge penalty; E2 has a single preregistered setting, so
        # the recorded label is enough to reconstruct it.
        param = kind === :E1 ? parse(Float64, String(entry["param"])) :
                (V23_E2_DEPTH, V23_E2_ROUNDS)
        gain = Float64(get(entry, :gain_nt, NaN))
        layers[slot] = (kind=kind, param=param, gain_nt=gain)
    end
    family = String(record["family"])
    base_config = String(record["base_config"])
    # Amendment A1 members carry two extra fields: the analog configuration whose
    # raw core they correct, and the all-DEV refit calibration that corrects it.
    # A selection that names the T1r family without them cannot be reproduced on
    # the confirmatory partition, so it is refused rather than approximated.
    analog_raw = get(record, :analog_config, nothing)
    analog_config = analog_raw === nothing ? base_config : String(analog_raw)
    calibration_raw = get(record, :t1r_calibration_csv, nothing)
    calibration = calibration_raw === nothing ? nothing : String(calibration_raw)
    if family == V23_T1R_FAMILY
        (calibration !== nothing && !isempty(calibration)) || error(
            "the development selection is a $(V23_T1R_FAMILY) member but records no " *
            "t1r_calibration_csv; rerun v2_3_select.jl",
        )
        analog_config == base_config && error(
            "the development selection is a $(V23_T1R_FAMILY) member whose analog_config " *
            "equals its base_config; the analog raw core cannot be identified",
        )
        v23_t1r_id(analog_config) == base_config || error(
            "the development selection names base config $(base_config) for analog " *
            "configuration $(analog_config)",
        )
        isfile(joinpath(dev_dir, calibration)) || error(
            "the recorded T1r calibration is missing: $(joinpath(dev_dir, calibration))",
        )
    end
    trace_raw = get(record, :selection_trace_sha256, nothing)
    composition_raw = get(record, :oof_v2_3_lat_sha256, nothing)
    return (
        config=String(record["selected_config"]),
        base_config=base_config,
        family=family,
        analog_config=analog_config,
        t1r_calibration_csv=calibration,
        selection_trace_sha=trace_raw === nothing ? "" : String(trace_raw),
        dev_composition_sha=composition_raw === nothing ? "" : String(composition_raw),
        drift_accepted=mismatches,
        safeguards=Bool(record["safeguards"]),
        params=record["params"],
        lat_weights=Float64.(collect(record["lat_weights"])),
        e_layers=layers,
        code_sha=String(record["code_sha256"]),
    )
end

"""
    _v23_select_direct_gbm(dev_dir, grid) -> Dict{Int,String}

Per model step, the direct-GBM grid point with the lowest pooled DEV
out-of-fold RMSE. Plan section 5 makes the direct comparator a per-lead model,
so the comparator scored on TEST is the per-step best of the development grid
and not a single hyperparameter setting chosen by hand.
"""
function _v23_select_direct_gbm(dev_dir::AbstractString, grid)
    best = Dict{Int,String}()
    scores = Dict{Int,Float64}()
    for (depth, rounds) in grid
        id = v23_direct_id(depth, rounds)
        path = joinpath(dev_dir, "summary_$(id).csv")
        isfile(path) || error("development direct-GBM summary is missing: $path")
        table = CSV.read(path, DataFrame)
        for row in eachrow(table)
            (row.cell == "all" && row.config == "$(id)_Son") || continue
            step = Int(row.model_step_hours)
            value = Float64(row.rmse_nt)
            isfinite(value) || continue
            if !haskey(scores, step) || value < scores[step]
                scores[step] = value
                best[step] = id
            end
        end
    end
    length(best) == V23_STEP_COUNT || error(
        "direct-GBM selection covered $(length(best)) of $(V23_STEP_COUNT) model steps",
    )
    return best
end

"""
    _v23_confirmatory_fold(ctx; smoke) -> V23Fold

The single fit/score split. The confirmed run fits on every DEV anchor and
scores every TEST anchor; the smoke run treats a late DEV slice as a pseudo-TEST
partition and fits only on anchors whose maximum target precedes it by the 168 h
embargo, so the smoke path has the same causal structure without reading TEST.
"""
function _v23_confirmatory_fold(ctx::V23Context; smoke::Bool)
    anchors = ctx.anchors
    if !smoke
        train = [i for i in eachindex(anchors.issue) if anchors.partition[i] == "DEV"]
        query = [i for i in eachindex(anchors.issue) if anchors.partition[i] == "TEST"]
        isempty(query) && error("confirmatory run found no TEST anchors")
        maximum(anchors.issue[train]) < minimum(anchors.issue[query]) || error(
            "the DEV archive is not strictly earlier than the TEST queries",
        )
        _v23_assert_embargo(anchors, train, query)
        return V23Fold("test", train, query)
    end
    query = [i for i in eachindex(anchors.issue)
             if anchors.partition[i] == "DEV" &&
                anchors.issue[i] >= V23_SMOKE_PSEUDO_TEST_FIRST_ISSUE]
    isempty(query) && error("smoke run found no pseudo-TEST anchors")
    query = query[1:min(length(query), ctx.plan.max_anchors)]
    cutoff = minimum(anchors.issue[query]) - Hour(V23_BASE_EMBARGO_HOURS)
    train = [i for i in eachindex(anchors.issue)
             if anchors.partition[i] == "DEV" &&
                anchors.issue[i] + Hour(V23_MAX_STEP) <= cutoff]
    _v23_assert_embargo(anchors, train, query)
    # The smoke slice exists to exercise the storm branch of the gates, so a
    # window that never leaves quiet conditions is a broken smoke run, not a
    # passing one.
    disturbed = count(i -> anchors.latest[i] <= V23_CELL_DST_50_NT, query)
    disturbed > 0 || error(
        "the smoke pseudo-TEST slice starting $(V23_SMOKE_PSEUDO_TEST_FIRST_ISSUE) holds no " *
        "anchor with latest Dst <= $(V23_CELL_DST_50_NT) nT, so the depth cells and the " *
        "intense-deepening guard would never be evaluated",
    )
    return V23Fold("pseudo_test", train, query)
end

"""
    _v23_assert_embargo(anchors, train, query) -> Int

Hours between the last target the fitting partition can see and the first issue
it is scored against. Ordering alone is not the preregistered condition: plan
section 3 embargoes 168 h of *targets*, so a fitting anchor issued seven hours
before the first query would satisfy an ordering check and still leak.
"""
function _v23_assert_embargo(anchors::V23Anchors, train::AbstractVector{Int},
                             query::AbstractVector{Int})
    (isempty(train) || isempty(query)) && error(
        "the confirmatory embargo check needs a non-empty fitting and scoring partition",
    )
    last_target = maximum(anchors.issue[train]) + Hour(V23_MAX_STEP)
    first_issue = minimum(anchors.issue[query])
    gap = div(Dates.value(first_issue - last_target), 3_600_000)
    gap >= V23_BASE_EMBARGO_HOURS || error(
        "the fitting partition ends at target $last_target and the first scored issue is " *
        "$first_issue, a gap of $gap h; plan section 3 requires at least " *
        "$(V23_BASE_EMBARGO_HOURS) h",
    )
    return gap
end

"""
    _v23_climatology_tau(ctx, rows) -> Float64

Mean ring-current recovery timescale of the fitting partition, estimated as
`τ = −1 / mean log(Dst(t+1) / Dst(t))` over recovery hours: hours whose Dst is
below −20 nT and whose next hour is closer to zero but still negative. It
parameterises the climatology-relaxed persistence ablation of plan section 5 and
nothing else.
"""
function _v23_climatology_tau(ctx::V23Context, rows::AbstractVector{Int})
    anchors = ctx.anchors
    frame = ctx.frame
    dst = Dict(frame.time_utc[i] => frame.Dst[i] for i in 1:nrow(frame))
    horizon = maximum(anchors.issue[rows])
    ratios = Float64[]
    for i in 1:nrow(frame) - 1
        time = frame.time_utc[i]
        time <= horizon || continue
        current = frame.Dst[i]
        following = get(dst, time + Hour(1), NaN)
        (isfinite(current) && isfinite(following)) || continue
        (current <= V23_CLIMATOLOGY_MIN_DEPTH_NT && following > current && following < 0.0) ||
            continue
        push!(ratios, log(following / current))
    end
    length(ratios) >= 100 || error(
        "climatology timescale needs at least 100 recovery hours, found $(length(ratios))",
    )
    tau = -1.0 / mean(ratios)
    isfinite(tau) && tau > 0 || error("climatology timescale is not positive and finite: $tau")
    return tau
end

"Climatology-relaxed persistence centers: `Dst(t)·exp(−h/τ)`."
function _v23_climatology_centers(ctx::V23Context, rows::AbstractVector{Int}, tau::Float64)
    centers = fill(NaN, length(ctx.anchors), V23_STEP_COUNT)
    for i in rows, (slot, step) in enumerate(V23_MODEL_STEPS)
        ctx.anchors.present[i, slot] || continue
        centers[i, slot] = ctx.anchors.latest[i] * exp(-step / tau)
    end
    return centers
end

"""
    _v23_adc_spec(selection) -> NamedTuple

Analog configuration used by the SINDy-removal ablation and the post-issue
invariance check. When the development run selected an analog tail this is that
tail; a T1r member changes only the correction layer, so its ablation driver is
the analog configuration it corrects. When the boosted tail was selected there is
no analog member to feed Burton and O'Brien, so the ablation falls back to the
preregistered uniform-weight, K = $(V23_T1A_K) ensemble and says so in the
manifest.
"""
function _v23_adc_spec(selection)
    if selection.family in ("T1", "T1a", V23_T1R_FAMILY)
        return (weight_set=Symbol(String(selection.params["weight_set"])),
                k=Int(selection.params["k"]), direct=Bool(selection.params["direct"]),
                source=selection.family == V23_T1R_FAMILY ? selection.analog_config :
                       selection.base_config)
    end
    return (weight_set=:uniform, k=V23_T1A_K, direct=false,
            source="preregistered_uniform_K$(V23_T1A_K)")
end

"""
    _v23_analog_ids(plan; smoke) -> Set{String}

Analog configurations this run will score, known before any stage executes. The
T1r layer is a correction over those cores, so the set also decides which refit
calibrations are eligible — and it has to be available up front, because the
frozen-feature oracle over those calibrations must run before the first center
is registered.
"""
function _v23_analog_ids(plan::V23RunPlan; smoke::Bool)
    ids = Set{String}()
    for weight_set in plan.weight_sets, k in plan.t1_ks
        push!(ids, v23_analog_id("T1", weight_set, k))
    end
    push!(ids, v23_analog_id("T1a", :uniform, smoke ? maximum(plan.t1_ks) : V23_T1A_K))
    return ids
end

"""
    _v23_t1r_configs(dev_dir) -> Vector{String}

Analog configurations whose all-DEV T1r calibration the development stage
persisted. Amendment A1 fits that layer once on every out-of-fold row of the six
inner blocks, so the presence of `t1r_fit_all_dev_<config>.csv` is exactly the
condition under which the corrected variant of `<config>` can be scored on the
confirmatory partition.
"""
function _v23_t1r_configs(dev_dir::AbstractString)
    isdir(dev_dir) || return String[]
    out = String[]
    for file in readdir(dev_dir)
        m = match(r"^t1r_fit_all_dev_(.+)\.csv$", file)
        m === nothing || push!(out, String(m.captures[1]))
    end
    return sort(out)
end

"""
    _v23_t1r_row_view(ctx, rows; table_path, partitions) -> NamedTuple

Base-table rows of the scored `(issue, model step)` pairs in a fixed order,
together with the anchor index and step slot each row belongs to. The T1r
calibration features are functions of the archived issue-time state, which the
anchor view keeps only per issue, so they are read back from the base table with
the column selection [`v23_t1r_base_table_columns`](@ref) declares. A scored pair
without a base-table row is an error: correcting a row whose archived state is
unknown would invent the state.
"""
function _v23_t1r_row_view(ctx::V23Context, rows::AbstractVector{Int};
                           table_path::AbstractString=V23_BASE_TABLE, partitions)
    isfile(table_path) || error("V2.3 base table is missing: $table_path")
    table = CSV.read(table_path, DataFrame; select=v23_t1r_base_table_columns(),
                     types=Dict("issue_time_utc" => DateTime))
    wanted = Set(String.(partitions))
    table = table[[String(p) in wanted for p in table.partition], :]
    nrow(table) > 0 || error(
        "the base table holds no rows for partitions $(collect(partitions))",
    )
    index = Dict{Tuple{DateTime,Int},Int}()
    sizehint!(index, nrow(table))
    for r in 1:nrow(table)
        index[(table.issue_time_utc[r], Int(table.model_step_hours[r]))] = r
    end
    order = Int[]
    anchor = Int[]
    slot = Int[]
    for i in rows, (s, step) in enumerate(V23_MODEL_STEPS)
        ctx.anchors.present[i, s] || continue
        key = (ctx.anchors.issue[i], step)
        r = get(index, key, 0)
        r == 0 && error("the base table has no row for $key; T1r cannot be applied there")
        push!(order, r)
        push!(anchor, i)
        push!(slot, s)
    end
    return (frame=table[order, :], anchor=anchor, slot=slot)
end

"""
    _v23_t1r_variant_centers(ctx, view, raw, fallback, cal) -> NamedTuple

Both safeguard variants of one T1r configuration on the scored partition: the
analog raw core of `raw` is handed to the Amendment A1 feature path and corrected
by the all-DEV refit calibration `cal`. Served-fallback anchors have no analog
core to correct and take the deployed product, the same convention the analog
configurations and the development cross-fit use, so every variant is scored on
the same rows.
"""
function _v23_t1r_variant_centers(ctx::V23Context, view, raw::AbstractMatrix{Float64},
                                  fallback::AbstractVector{Bool},
                                  cal::OperationalV2Calibration)
    values = [raw[view.anchor[j], view.slot[j]] for j in eachindex(view.anchor)]
    features = v23_t1r_features(view.frame, values)
    centers = v23_t1r_centers(cal, features)
    n = length(ctx.anchors)
    on = fill(NaN, n, V23_STEP_COUNT)
    off = fill(NaN, n, V23_STEP_COUNT)
    for j in eachindex(view.anchor)
        i = view.anchor[j]
        s = view.slot[j]
        if fallback[i]
            on[i, s] = ctx.anchors.served[i, s]
            off[i, s] = ctx.anchors.served[i, s]
        else
            on[i, s] = centers.on[j]
            off[i, s] = centers.off[j]
        end
    end
    return (on=on, off=off)
end

"""
    _v23_run_sindy_removal!(ctx, fold, spec) -> NamedTuple

SINDy-removal ablation (plan section 6-B3): the selected analog driver ensemble
is integrated through Burton, Burton-full and O'Brien–McPherron instead of the
SINDy core. The driver sequence is produced by the same retrieval and the same
rollout loop as the candidate, so the ablation isolates the integrator.
"""
function _v23_run_sindy_removal!(ctx::V23Context, fold::V23Fold, spec)
    anchors = ctx.anchors
    weight_set, k, direct = spec.weight_set, spec.k, spec.direct
    weights = v23_weights(weight_set)
    archive = [i for i in fold.train if ctx.ok[i] && ctx.origin_ok[i]]
    query = sort(fold.query)
    query_ok = [i for i in query if ctx.ok[i]]
    stats = v23_feature_stats(ctx.X[archive, :])
    archive_z = v23_standardize(ctx.X[archive, :], stats.mean, stats.sd)
    query_z = v23_standardize(ctx.X[query_ok, :], stats.mean, stats.sd)
    archive_times = anchors.issue[archive]
    neighbours = v23_knn(query_z, anchors.issue[query_ok], archive_z, archive_times, k;
                         weights=weights)
    n = length(anchors)
    centers = (burton=fill(NaN, n, V23_STEP_COUNT), burton_full=fill(NaN, n, V23_STEP_COUNT),
               obrien=fill(NaN, n, V23_STEP_COUNT))
    lib, xi = ctx.core.library, ctx.core.coefficients
    lookup = ctx.lookup
    Threads.@threads :static for j in eachindex(query_ok)
        i = query_ok[j]
        drv = anchors.drv[i]
        issue = anchors.issue[i]
        future = kk -> get(lookup, issue + Hour(kk - 1), nothing)
        members = [v23_analog_member(drv, lookup, archive_times[neighbours[j, m]];
                                     direct=direct) for m in 1:k]
        raw = Matrix{Float64}(undef, k, V23_STEP_COUNT)
        scratch = (burton=Matrix{Float64}(undef, k, V23_STEP_COUNT),
                   burton_full=Matrix{Float64}(undef, k, V23_STEP_COUNT),
                   obrien=Matrix{Float64}(undef, k, V23_STEP_COUNT))
        v23_member_step_raw!(raw, lib, xi, anchors.anchor_star[i], drv, future, members;
                             baselines=scratch)
        for slot in 1:V23_STEP_COUNT
            centers.burton[i, slot] = mean(scratch.burton[:, slot])
            centers.burton_full[i, slot] = mean(scratch.burton_full[:, slot])
            centers.obrien[i, slot] = mean(scratch.obrien[:, slot])
        end
    end
    # Anchors without an analog key keep the frozen-driver baseline of the base
    # table, the same served-fallback convention the candidates use.
    for i in query, slot in 1:V23_STEP_COUNT
        anchors.present[i, slot] || continue
        isnan(centers.burton[i, slot]) || continue
        centers.burton[i, slot] = anchors.burton[i, slot]
        centers.burton_full[i, slot] = anchors.burton_full[i, slot]
        centers.obrien[i, slot] = anchors.obrien[i, slot]
    end
    return centers
end

"""
    _v23_apply_fixed_e_layers(ctx, train_rows, query_rows, base_centers, layers) -> Matrix

Apply the error layers the development run accepted, refitted on the fitting
partition with the recorded hyperparameters. No layer is re-adjudicated here:
acceptance was decided on DEV, and re-deciding it on TEST would turn the
single-shot test into a selection step.
"""
function _v23_apply_fixed_e_layers(ctx::V23Context, train_rows::AbstractVector{Int},
                                   query_rows::AbstractVector{Int},
                                   base_centers::AbstractMatrix{Float64}, layers;
                                   on_model=nothing)
    centers = copy(base_centers)
    applied = NamedTuple[]
    kinds = unique([layer.kind for layer in layers if layer !== nothing])
    isempty(kinds) && return (centers=centers, applied=applied)
    rows = sort(unique(vcat(collect(train_rows), collect(query_rows))))
    features, ok = v23_innovation_features(ctx, base_centers, rows)
    folds = [(name="confirmatory", train=collect(train_rows), query=collect(query_rows))]
    for kind in kinds
        params = unique([layer.param for layer in layers
                         if layer !== nothing && layer.kind === kind])
        for param in params
            sink = on_model === nothing ? nothing :
                (fold, slot, model, cap) -> on_model(kind, param, slot, model, cap)
            delta = v23_e_layer_predictions(ctx, folds, base_centers, features, ok, kind, param;
                                            on_model=sink)
            for (slot, layer) in enumerate(layers)
                (layer !== nothing && layer.kind === kind && layer.param == param) || continue
                corrected = 0
                for i in query_rows
                    isfinite(delta[i, slot]) || continue
                    centers[i, slot] = base_centers[i, slot] + delta[i, slot]
                    corrected += 1
                end
                push!(applied, (layer=String(kind), model_step_hours=V23_MODEL_STEPS[slot],
                                param=string(param), rows_corrected=corrected,
                                rows_scored=count(i -> ctx.anchors.present[i, slot],
                                                  query_rows)))
            end
        end
    end
    return (centers=centers, applied=applied)
end

"""
    _v23_bootstrap_rows(label, comparator, centers, comparator_centers, ctx, rows) -> Vector

Paired block bootstrap of the RMSE gain against one comparator at every model
step, over the preregistered fixed non-overlapping 168 h calendar blocks of
issue time (V2.2 convention), 10,000 replicates and seed 22,022,026 from
`src/operational_v23_stats.jl`.
"""
function _v23_bootstrap_rows(label::AbstractString, comparator::AbstractString,
                             centers::AbstractMatrix{Float64},
                             comparator_centers::AbstractMatrix{Float64},
                             ctx::V23Context, rows::AbstractVector{Int})
    anchors = ctx.anchors
    out = NamedTuple[]
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        scored = [i for i in rows if anchors.present[i, slot] &&
                                     isfinite(centers[i, slot]) &&
                                     isfinite(comparator_centers[i, slot])]
        isempty(scored) && continue
        # The block sums are accumulated in row order, so the reported bound is
        # order-dependent at the last bit. Fixing the order by (issue, step)
        # makes the artifact reproducible independently of how the caller
        # assembled its row list.
        sort!(scored; by=i -> (anchors.issue[i], step))
        candidate_se = [abs2(anchors.obs[i, slot] - centers[i, slot]) for i in scored]
        comparator_se = [abs2(anchors.obs[i, slot] - comparator_centers[i, slot])
                         for i in scored]
        issues = [anchors.issue[i] for i in scored]
        result = SolarSINDy.v23_block_bootstrap(comparator_se, candidate_se, issues)
        push!(out, (
            variant=String(label), comparator=String(comparator), model_step_hours=step,
            n=length(scored), rmse_variant_nt=sqrt(mean(candidate_se)),
            rmse_comparator_nt=sqrt(mean(comparator_se)), gain_nt=result.point,
            lower_nt=result.lower, p_one_sided=result.p_one_sided,
            n_blocks=result.n_blocks,
        ))
    end
    return out
end

"Holm-adjust the p-values of a family of bootstrap rows, returned in input order."
function _v23_holm_family(rows)
    isempty(rows) && return Float64[]
    return SolarSINDy.v23_holm([Float64(row.p_one_sided) for row in rows])
end

"""
    _v23_run_stage!(ctx, ids, params_of, compute) -> Vector{V23ConfigResult}

Resume-aware stage execution with the same contract as the development runner:
a configuration is recomputed unless its persisted signature still matches the
base table, the hourly frame and the V2.3 sources.
"""
function _v23_run_stage!(ctx::V23Context, ids::Vector{String}, params_of, compute)
    if all(id -> v23_config_is_current(ctx, id, params_of(id)), ids)
        _v23_log("  = reusing $(join(ids, ", "))")
        return [v23_read_config(ctx, id) for id in ids]
    end
    started = time()
    results = compute()
    elapsed = time() - started
    for result in results
        v23_write_config!(ctx, result, elapsed / length(results))
    end
    _v23_log(@sprintf("  + %s written in %.1f s", join(ids, ", "), elapsed))
    return [v23_read_config(ctx, id) for id in ids]
end

"""
    _v23_feature_invariance(ctx, issues) -> Bool

Post-issue mutation invariance of the analog key: every frame value strictly
after the issue hour is destroyed and the issue-time feature vector is rebuilt.
A feature that moved would mean the retrieval key had read the future.
"""
function _v23_feature_invariance(ctx::V23Context, issues::AbstractVector{DateTime})
    for issue in issues
        anchor = ctx.anchors.index[issue]
        mutated = DataFrame(
            time_utc=copy(ctx.frame.time_utc), V=copy(ctx.frame.V), Bz=copy(ctx.frame.Bz),
            By=copy(ctx.frame.By), n=copy(ctx.frame.n), Pdyn=copy(ctx.frame.Pdyn),
            Dst=copy(ctx.frame.Dst),
        )
        later = mutated.time_utc .> issue
        for column in (:V, :Bz, :By, :n, :Pdyn, :Dst)
            mutated[!, column][later] .= NaN
        end
        features, ok = v23_feature_matrix(mutated, [issue])
        ok[1] || return false
        vec(features[1, :]) == vec(ctx.X[anchor, :]) || return false
    end
    return true
end

"""
    _v23_tail_invariance(ctx, fold, spec, sample) -> Bool

Post-issue driver invariance of the rollout: every driver record beyond the
L1-admitted window is replaced by an extreme fabricated record and the ensemble
must return the identical center. Records inside the window are left alone,
because they were L1-measured before issue and the served product admits them
by the same rule.
"""
function _v23_tail_invariance(ctx::V23Context, fold::V23Fold, spec, sample::Int)
    anchors = ctx.anchors
    weight_set, k, direct = spec.weight_set, spec.k, spec.direct
    archive = [i for i in fold.train if ctx.ok[i] && ctx.origin_ok[i]]
    query_ok = [i for i in sort(fold.query) if ctx.ok[i]]
    isempty(query_ok) && return false
    step = max(1, div(length(query_ok), sample))
    sampled = query_ok[1:step:end][1:min(sample, length(query_ok[1:step:end]))]
    stats = v23_feature_stats(ctx.X[archive, :])
    archive_z = v23_standardize(ctx.X[archive, :], stats.mean, stats.sd)
    sampled_z = v23_standardize(ctx.X[sampled, :], stats.mean, stats.sd)
    archive_times = anchors.issue[archive]
    neighbours = v23_knn(sampled_z, anchors.issue[sampled], archive_z, archive_times, k;
                         weights=v23_weights(weight_set))
    fabricated = (V=999.0, Bz=-999.0, By=999.0, n=99.0, Pdyn=99.0)
    lib, xi = ctx.core.library, ctx.core.coefficients
    for (j, i) in enumerate(sampled)
        drv = anchors.drv[i]
        issue = anchors.issue[i]
        admitted = floor(Int, _transit_hours(drv.V))
        truthful = kk -> get(ctx.lookup, issue + Hour(kk - 1), nothing)
        corrupted = kk -> kk <= admitted ? truthful(kk) : fabricated
        members = [v23_analog_member(drv, ctx.lookup, archive_times[neighbours[j, m]];
                                     direct=direct) for m in 1:k]
        before = Matrix{Float64}(undef, k, V23_STEP_COUNT)
        after = Matrix{Float64}(undef, k, V23_STEP_COUNT)
        v23_member_step_raw!(before, lib, xi, anchors.anchor_star[i], drv, truthful, members)
        v23_member_step_raw!(after, lib, xi, anchors.anchor_star[i], drv, corrupted, members)
        before == after || return false
    end
    return true
end

"""
    V23_T1R_ORACLE_HOOK

Test seam for the frozen-feature oracle. Production leaves it `nothing`; a test
sets it to a callable that raises, to prove that an oracle failure aborts the
single-shot run instead of being downgraded to a failed gate.
"""
const V23_T1R_ORACLE_HOOK = Ref{Any}(nothing)

_v23_t1r_oracle(frame) = begin
    hook = V23_T1R_ORACLE_HOOK[]
    hook === nothing ? v23_t1r_feature_oracle(frame) : hook(frame)
end

"""
    _v23_t1r_oracle_or_fail(frame, manifest) -> NamedTuple

Rebuild the archived calibration features of the scored partition from the raw
inputs and require them to reproduce the base table's own columns.

A drift here changes what the refit correction layer is applied to, so it is an
integrity failure of the run, not of a gate: the failure is recorded in the
manifest and rethrown, which is what keeps the single shot unspent. The previous
behaviour caught the exception, scored every T1r variant through the drifted
feature path and wrote a decision anyway.
"""
function _v23_t1r_oracle_or_fail(frame, manifest::V23Manifest)
    oracle = try
        _v23_t1r_oracle(frame)
    catch
        v23_manifest!(manifest, "identity", "t1r_feature_oracle_max_abs_nt", NaN, "FAILED")
        rethrow()
    end
    v23_manifest!(manifest, "identity", "t1r_feature_oracle_max_abs_nt",
                  oracle.worst_nt, String(oracle.column))
    return oracle
end

"""
    _v23_assert_integrity(integrity)

Fail closed on a section 6-A4 breach. A4 asks whether the run itself is sound —
finite centers, an archive that precedes every query, a retrieval key that
cannot see past the issue hour, a tail that cannot see past the L1 window. A run
that fails one of those did not measure the candidate, so it must not leave a
decision record behind for the single-shot partition.
"""
function _v23_assert_integrity(integrity)
    failed = [String(name) for (name, value, _) in integrity if value !== true]
    isempty(failed) && return true
    error(
        "section 6-A4 integrity failed on this run: $(join(failed, ", ")). No decision is " *
        "written: the scored centers cannot be attributed to the preregistered candidate.",
    )
end

"""
    _v23_preregistered_variant(name) -> Bool

Whether a scored column belongs to the preregistered family. Amendment A1 puts
the refit correction on the T1 grid, so the T1r layer over the T1a direct-value
ablation is scored for reporting only and can never be the candidate. Labelling
it keeps that visible in the artifacts instead of only in the selection trace.
"""
_v23_preregistered_variant(name::AbstractString) =
    !startswith(String(name), "$(V23_T1R_FAMILY)_T1a_")

"""
    _v23_gate_rows(ctx, rows, bootstrap, final_metrics, integrity) -> (rows, verdicts)

Evaluate the preregistered gates of plan section 6 from the bootstrap table and
the cell metrics. Every gate row records the requirement it applied and the
number it observed, so a verdict can be re-derived from the artifact alone.
"""
function _v23_gate_rows(bootstrap, final_metrics, integrity, ablation_variants=String[])
    lookup(variant, comparator, step) = begin
        hit = [r for r in bootstrap if r.variant == variant && r.comparator == comparator &&
                                       r.model_step_hours == step]
        length(hit) == 1 || error(
            "bootstrap table has $(length(hit)) rows for $variant vs $comparator at step $step",
        )
        first(hit)
    end
    rows = NamedTuple[]
    emit(gate, family, step, comparator, requirement, observed, holm, pass) = push!(rows, (
        gate=String(gate), family=String(family), model_step_hours=step,
        comparator=String(comparator), requirement=String(requirement),
        observed=String(observed), holm_p=holm, pass=pass,
    ))

    # --- A1 accuracy versus the deployed product -----------------------------
    a1 = [lookup("V2_3_final", "served_v2_1", step) for step in V23_GATE_A1_STEPS]
    a1_holm = _v23_holm_family(a1)
    a1_pass = true
    for (index, row) in enumerate(a1)
        required = max(V23_GATE_A1_MIN_GAIN_NT,
                       V23_GATE_A1_MIN_GAIN_FRACTION * row.rmse_comparator_nt)
        pass = row.gain_nt >= required && row.lower_nt > 0.0 && a1_holm[index] < V23_GATE_ALPHA
        a1_pass &= pass
        emit("A1", "steps_3_4_6_7", row.model_step_hours, "served_v2_1",
             @sprintf("gain >= %.3f nT, lower > 0, Holm p < %.2f", required, V23_GATE_ALPHA),
             @sprintf("gain=%+.3f lower=%+.3f p=%.4f", row.gain_nt, row.lower_nt,
                      row.p_one_sided), a1_holm[index], pass)
    end
    for step in V23_GATE_A1_NONINFERIOR_STEPS
        row = lookup("V2_3_final", "served_v2_1", step)
        pass = row.lower_nt > V23_GATE_A1_NONINFERIORITY_NT && row.gain_nt >= 0.0
        a1_pass &= pass
        emit("A1", "non_inferiority", step, "served_v2_1",
             @sprintf("lower > %.2f nT and gain >= 0", V23_GATE_A1_NONINFERIORITY_NT),
             @sprintf("gain=%+.3f lower=%+.3f", row.gain_nt, row.lower_nt), NaN, pass)
    end

    # --- A2 storm safety -----------------------------------------------------
    guards = v23_guard_flags(final_metrics)
    emit("A2", "storm_cells", -1, "served_v2_1",
         "no guarded cell with >= $(V23_GUARD_MIN_CELL_ROWS) rows loses more than " *
         "$(V23_GUARD_MAX_CELL_LOSS_NT) nT; intense deepening at " *
         "$(V23_GUARD_INTENSE_STEP) h loses nothing and keeps |bias| <= " *
         "$(V23_GUARD_INTENSE_BIAS_NT) nT",
         isempty(guards.guard_failures) ? "no cell breached" : guards.guard_failures,
         NaN, guards.guards_ok)

    # --- A3 baselines --------------------------------------------------------
    a3_pass = true
    for comparator in ("persistence", "burton", "burton_full", "obrien")
        family = [lookup("V2_3_final", comparator, step) for step in V23_MODEL_STEPS]
        holm = _v23_holm_family(family)
        for (index, row) in enumerate(family)
            pass = row.gain_nt > 0.0 && row.lower_nt > 0.0 && holm[index] < V23_GATE_ALPHA
            a3_pass &= pass
            emit("A3", comparator, row.model_step_hours, comparator,
                 @sprintf("gain > 0, lower > 0, Holm p < %.2f", V23_GATE_ALPHA),
                 @sprintf("gain=%+.3f lower=%+.3f p=%.4f", row.gain_nt, row.lower_nt,
                          row.p_one_sided), holm[index], pass)
        end
    end

    # --- A4 integrity --------------------------------------------------------
    for (name, value, requirement) in integrity
        emit("A4", "integrity", -1, "-", requirement, string(value), NaN, value === true)
    end
    a4_pass = all(value === true for (_, value, _) in integrity)

    # --- B1 SINDy-line claim against the information-equivalent frozen tail ---
    b1 = [lookup("V2_3_final", "frozen_v2_1", step) for step in V23_GATE_B1_STEPS]
    b1_holm = _v23_holm_family(b1)
    b1_pass = true
    for (index, row) in enumerate(b1)
        pass = row.gain_nt >= V23_GATE_B1_MIN_GAIN_NT && row.lower_nt > 0.0 &&
               b1_holm[index] < V23_GATE_ALPHA
        b1_pass &= pass
        emit("B1", "steps_6_7", row.model_step_hours, "frozen_v2_1",
             @sprintf("gain >= %.2f nT, lower > 0, Holm p < %.2f", V23_GATE_B1_MIN_GAIN_NT,
                      V23_GATE_ALPHA),
             @sprintf("gain=%+.3f lower=%+.3f p=%.4f", row.gain_nt, row.lower_nt,
                      row.p_one_sided), b1_holm[index], pass)
    end

    # --- B2 direct machine-learning comparator (reported, not a pass/fail bar) ---
    b2_claim_ok = true
    for step in V23_GATE_A1_STEPS
        row = lookup("V2_3_final", "direct_gbm", step)
        within = row.gain_nt >= -V23_GATE_B2_TOLERANCE_NT
        b2_claim_ok &= within
        emit("B2", "steps_3_4_6_7", step, "direct_gbm",
             "deficit <= $(V23_GATE_B2_TOLERANCE_NT) nT for a local-superiority claim",
             @sprintf("gain=%+.3f lower=%+.3f", row.gain_nt, row.lower_nt), NaN, within)
    end

    # --- storm-cell gains, the alternative PIVOT route ------------------------
    storm_cells = ("latest_le_m50", "latest_le_m100", "active_deepening")
    storm_gain_pass = true
    observed_cells = 0
    for row in final_metrics
        (row.cell in storm_cells && row.model_step_hours in V23_GATE_A1_STEPS) || continue
        row.n >= V23_GUARD_MIN_CELL_ROWS || continue
        observed_cells += 1
        storm_gain_pass &= (isfinite(row.loss_vs_served_nt) && row.loss_vs_served_nt < 0.0)
    end
    storm_gain_pass &= observed_cells > 0
    emit("PIVOT_ROUTE", "storm_cell_gains", -1, "served_v2_1",
         "every storm cell with >= $(V23_GUARD_MIN_CELL_ROWS) rows at steps 3/4/6/7 " *
         "gains RMSE against served V2.1",
         "$(observed_cells) cells evaluated", NaN, storm_gain_pass)

    # --- ablation labelling, carried in the artifact rather than only in prose ---
    emit("ABLATION_LABEL", "not_preregistered", -1, "-",
         "Amendment A1 preregisters the refit correction over the T1 grid only; the " *
         "variants named here are scored for reporting and are never the candidate",
         isempty(ablation_variants) ? "none scored" : join(ablation_variants, ";"),
         NaN, true)

    verdicts = (A1=a1_pass, A2=guards.guards_ok, A3=a3_pass, A4=a4_pass, B1=b1_pass,
                B2_claim_ok=b2_claim_ok, storm_cell_gains=storm_gain_pass)
    return (rows=rows, verdicts=verdicts)
end

"""
    _v23_decision(verdicts) -> (state, failing)

Decision states of plan section 6: `SHADOW_READY` when Tier A passes,
`PIVOT` when Tier A fails but storm safety survives and either the SINDy-line
claim or the storm-cell gains hold, `NO_GO` otherwise. `GO` is not reachable
without prospective evidence and is never emitted here.
"""
function _v23_decision(verdicts)
    tier_a = verdicts.A1 && verdicts.A2 && verdicts.A3 && verdicts.A4
    failing = String[name for name in ("A1", "A2", "A3", "A4")
                     if !getproperty(verdicts, Symbol(name))]
    tier_a && return ("SHADOW_READY", failing)
    (verdicts.A2 && (verdicts.B1 || verdicts.storm_cell_gains)) &&
        return ("PIVOT", failing)
    return ("NO_GO", failing)
end

"""
    _v23_read_dev_composition(ctx, dev_dir) -> NamedTuple

Development out-of-fold centers of the composed configuration. They are the
fitting rows of the confirmatory error layers: the only DEV rows for which an
out-of-fold center of the selected configuration exists are the inner-block
rows, so the 2010–2012 archive-only anchors contribute to the analog archive but
not to the layer fit.
"""
function _v23_read_dev_composition(ctx::V23Context, dev_dir::AbstractString)
    path = joinpath(dev_dir, "oof_V2_3_LAT.csv")
    isfile(path) || error("development composition is missing: $path")
    centers = fill(NaN, length(ctx.anchors), V23_STEP_COUNT)
    fallback = fill(false, length(ctx.anchors))
    seen = falses(length(ctx.anchors))
    table = CSV.read(path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    for r in 1:nrow(table)
        anchor = get(ctx.anchors.index, table.issue_time_utc[r], 0)
        anchor == 0 && continue
        slot = V23_STEP_SLOT[Int(table.model_step_hours[r])]
        centers[anchor, slot] = Float64(table.center_s_on_dst_nt[r])
        fallback[anchor] = Bool(table.fallback[r])
        seen[anchor] = true
    end
    rows = findall(seen)
    isempty(rows) && error("development composition holds no rows this run can use")
    return (centers=centers, fallback=fallback, rows=rows)
end

"""
    _v23_append_rerun_log(plan, reason, decision_path; stage, replacement)

One row of the forced-rerun record. The `intent` row is written before the run
starts, while the previous decision still exists to be captured; the `completed`
row is written only after a new decision has actually replaced it. A forced run
that dies in between therefore leaves an intent with no completion, which is
what distinguishes "the decision was overwritten" from "someone tried".
"""
function _v23_append_rerun_log(plan::V23RunPlan, reason::AbstractString;
                               stage::AbstractString="intent",
                               decision_path::Union{Nothing,AbstractString}=nothing,
                               replacement::AbstractString="")
    path = joinpath(plan.outdir, "rerun_log.csv")
    previous = (decision_path !== nothing && isfile(decision_path)) ?
        JSON3.write([NamedTuple(row) for row in eachrow(CSV.read(decision_path, DataFrame))]) :
        ""
    row = DataFrame(rerun_utc=[string(now(UTC))], stage=[String(stage)],
                    reason=[String(reason)], overwritten_decision=[previous],
                    replacement_decision=[String(replacement)])
    CSV.write(path, row; append=isfile(path), writeheader=!isfile(path))
    return path
end

# ---------------------------------------------------------------------------
# Deployment artifacts of the confirmatory fit
# ---------------------------------------------------------------------------

"""
    V23_CONFIRM_ARTIFACT_DIR

Subdirectory of the run tree holding the fitted objects a live engine needs to
reproduce the scored forecast: the error layers refitted on all of DEV, the
per-step direct comparator, and the definition of the analog archive the
retrieval searched. Persisting them is a record of the run, not an input to it —
nothing in the scoring path reads these files back.
"""
const V23_CONFIRM_ARTIFACT_DIR = "artifacts"

"The `(depth, rounds)` grid point behind a direct-GBM configuration identity."
function _v23_direct_grid_point(id::AbstractString, grid)
    for (depth, rounds) in grid
        v23_direct_id(depth, rounds) == String(id) && return (depth, rounds)
    end
    error("direct-GBM identity $id is not a point of this run's grid")
end

"Write one E1 ridge layer as a readable table: standardisation, slopes, intercept, cap."
function _v23_write_e1_artifact(path::AbstractString, model::V23Ridge,
                                feature_names::AbstractVector{<:AbstractString}, cap::Real)
    length(feature_names) == length(model.beta) || error(
        "E1 artifact has $(length(model.beta)) coefficients for " *
        "$(length(feature_names)) feature names",
    )
    CSV.write(path, DataFrame(
        feature_name=String.(collect(feature_names)),
        feature_mean=model.mean,
        feature_scale=model.sd,
        standardised_coefficient=model.beta,
        intercept_nt=fill(model.intercept, length(model.beta)),
        ridge_lambda=fill(model.lambda, length(model.beta)),
        correction_cap_nt=fill(Float64(cap), length(model.beta)),
    ))
    return path
end

"""
    _v23_write_analog_archive!(ctx, fold, directory) -> NamedTuple

Definition of the analog archive the confirmatory retrieval searched: how many
issue-time origins were eligible, which hours bound them, and the per-feature
standardisation the distance metric used. A live engine that rebuilds the
archive from the same partition can check itself against these numbers before it
serves anything.
"""
function _v23_write_analog_archive!(ctx::V23Context, fold::V23Fold, directory::AbstractString)
    archive = [i for i in fold.train if ctx.ok[i] && ctx.origin_ok[i]]
    isempty(archive) && error("the confirmatory fold has no eligible analog origin")
    stats = v23_feature_stats(ctx.X[archive, :])
    path = joinpath(directory, "analog_feature_stats.csv")
    CSV.write(path, DataFrame(
        feature_name=String.(collect(V23_FEATURE_NAMES)),
        feature_mean=stats.mean,
        feature_sd=stats.sd,
    ))
    return (path=path, origins=length(archive),
            first_origin=minimum(ctx.anchors.issue[archive]),
            last_origin=maximum(ctx.anchors.issue[archive]))
end

"""
    run_v2_3_confirmatory(; smoke, confirm, force_reason) -> NamedTuple

Score the preregistered family on the confirmatory partition and evaluate the
plan section 6 gates. `confirm = true` is required before any TEST row is read.
"""
function run_v2_3_confirmatory(; smoke::Bool=false, confirm::Bool=false,
                               force_reason::Union{Nothing,String}=nothing,
                               accept_code_drift::Union{Nothing,String}=nothing)
    total_started = time()
    (smoke || confirm) && !(smoke && confirm) || error(
        "the confirmatory runner needs exactly one of --smoke (pseudo-TEST on a DEV slice) " *
        "or --confirm (the single-shot TEST scoring)",
    )
    plan = v23_test_plan(; smoke=smoke)
    v23_assert_full_grid(plan)
    mkpath(plan.outdir)
    decision_path = joinpath(plan.outdir, "decision.csv")
    if !smoke && isfile(decision_path)
        force_reason === nothing && error(
            "TEST has already been scored once: $decision_path exists. The plan allows a " *
            "single confirmatory run; pass --force-rerun-with-reason=<text> to overwrite it " *
            "and record why.",
        )
        _v23_append_rerun_log(plan, force_reason; stage="intent", decision_path=decision_path)
    end
    dev_dir = smoke ? V23_DEV_SMOKE_DIR : V23_DEV_DIR
    manifest = V23Manifest()
    v23_manifest!(manifest, "plan", "label", NaN, plan.label)
    v23_manifest!(manifest, "plan", "development_dir", NaN, dev_dir)
    force_reason === nothing ||
        v23_manifest!(manifest, "plan", "force_rerun_reason", NaN, force_reason)

    _v23_log("V2.3 confirmatory runner ($(plan.label)); threads=$(Threads.nthreads())")
    # The input contract is verified before the base table is loaded: reading the
    # TEST partition is the irreversible step of this protocol, so a selection
    # that does not describe this run must stop the process before it happens.
    isfile(V23_BASE_TABLE) || error("V2.3 base table is missing: $(V23_BASE_TABLE)")
    isfile(V23_BASE_HOURLY_FRAME) ||
        error("V2.3 hourly frame is missing: $(V23_BASE_HOURLY_FRAME)")
    base_table_sha = _v23_sha256_file(V23_BASE_TABLE)
    frame_sha = _v23_sha256_file(V23_BASE_HOURLY_FRAME)
    code_sha = v23_code_signature()
    selection = _v23_read_selection(dev_dir, base_table_sha; frame_sha=frame_sha,
                                    code_sha=code_sha,
                                    accept_code_drift=accept_code_drift)
    for message in selection.drift_accepted
        @warn "accepted input drift on the confirmatory contract" message reason =
            accept_code_drift
        v23_manifest!(manifest, "accepted_drift", message, NaN, String(accept_code_drift))
    end
    accept_code_drift === nothing ||
        v23_manifest!(manifest, "plan", "accept_code_drift_reason", NaN, accept_code_drift)

    started = time()
    ctx = v23_build_context(plan; partitions=(smoke ? ("DEV",) : ("DEV", "TEST")))
    fold = _v23_confirmatory_fold(ctx; smoke=smoke)
    folds = [fold]
    rows = sort(fold.query)
    v23_manifest!(manifest, "embargo", "fit_to_scored_hours",
                  _v23_assert_embargo(ctx.anchors, fold.train, rows), "")
    v23_manifest!(manifest, "seconds", "context", time() - started, "")
    v23_manifest!(manifest, "rows", "fit_anchors", length(fold.train), "")
    v23_manifest!(manifest, "rows", "scored_anchors", length(rows), "")
    v23_manifest!(manifest, "rows", "scored_anchors_latest_le_m50",
                  count(i -> ctx.masks.latest_le_m50[i], rows), "")
    v23_manifest!(manifest, "selection", "config", NaN, selection.config)
    v23_manifest!(manifest, "selection", "family", NaN, selection.family)
    v23_manifest!(manifest, "selection", "analog_config", NaN, selection.analog_config)
    selection.t1r_calibration_csv === nothing || v23_manifest!(
        manifest, "selection", "t1r_calibration_csv", NaN, selection.t1r_calibration_csv,
    )
    isempty(selection.selection_trace_sha) || v23_manifest!(
        manifest, "selection", "selection_trace_sha256", NaN, selection.selection_trace_sha,
    )
    isempty(selection.dev_composition_sha) || v23_manifest!(
        manifest, "selection", "oof_v2_3_lat_sha256", NaN, selection.dev_composition_sha,
    )
    v23_manifest!(manifest, "selection", "development_code_sha256", NaN, selection.code_sha)
    _v23_log(@sprintf("  fit anchors %d, scored anchors %d, selected %s (family %s)",
                      length(fold.train), length(rows), selection.config, selection.family))

    # ---- Amendment A1 integrity, before a single center is scored ----
    # The frozen-feature oracle decides whether the refit correction is being
    # applied to the state the calibration was fitted on. It is checked here,
    # while nothing has been registered yet, so that a failure aborts the single
    # shot instead of producing a scored decision with a failed gate.
    scored_analog_ids = _v23_analog_ids(plan; smoke=smoke)
    t1r_configs = [config for config in _v23_t1r_configs(dev_dir)
                   if config in scored_analog_ids]
    for config in t1r_configs
        v23_t1r_require_signature(dev_dir, config; table_sha=ctx.base_table_sha,
                                  code_sha=ctx.code_sha)
    end
    t1r_view = isempty(t1r_configs) ? nothing :
        _v23_t1r_row_view(ctx, rows; partitions=(smoke ? ("DEV",) : ("TEST",)))
    t1r_view === nothing || _v23_t1r_oracle_or_fail(t1r_view.frame, manifest)

    scores = Dict{String,Matrix{Float64}}()
    fallbacks = Dict{String,Vector{Bool}}()
    order = String[]
    metrics = NamedTuple[]
    no_fallback = fill(false, length(ctx.anchors))
    function register!(name, centers, fallback)
        haskey(scores, name) && error("score set $name registered twice")
        scores[name] = centers
        fallbacks[name] = fallback
        push!(order, name)
        append!(metrics, v23_metric_rows(name, centers, ctx.anchors, ctx.masks, rows, fallback))
        return name
    end

    # ---- preregistered candidate family ----
    stage_started = time()
    analog_entries = Dict{String,Any}()
    for weight_set in plan.weight_sets
        ids = [v23_analog_id("T1", weight_set, k) for k in plan.t1_ks]
        stored = _v23_run_stage!(
            ctx, ids,
            id -> v23_analog_params("T1", weight_set, parse(Int, split(id, "_K")[end]), false),
            () -> begin
                results = v23_run_analog_stage!(ctx, folds, weight_set; ks=plan.t1_ks,
                                                direct=false, family="T1")
                [results[k] for k in sort(plan.t1_ks)]
            end,
        )
        for (id, entry) in zip(ids, stored)
            analog_entries[id] = entry
            register!("$(id)_Son", entry.centers_on, entry.fallback)
            register!("$(id)_Soff", entry.centers_off, entry.fallback)
        end
    end
    t1a_k = smoke ? maximum(plan.t1_ks) : V23_T1A_K
    t1a_id = v23_analog_id("T1a", :uniform, t1a_k)
    for entry in _v23_run_stage!(
            ctx, [t1a_id], _ -> v23_analog_params("T1a", :uniform, t1a_k, true),
            () -> [v23_run_analog_stage!(ctx, folds, :uniform; ks=[t1a_k], direct=true,
                                         family="T1a")[t1a_k]])
        analog_entries[t1a_id] = entry
        register!("$(t1a_id)_Son", entry.centers_on, entry.fallback)
        register!("$(t1a_id)_Soff", entry.centers_off, entry.fallback)
    end
    for (depth, rounds) in plan.gdc_grid
        id = v23_gdc_id(depth, rounds)
        for entry in _v23_run_stage!(ctx, [id], _ -> v23_gdc_params(depth, rounds),
                                     () -> [v23_run_gdc_stage!(ctx, folds, depth, rounds)])
            register!("$(id)_Son", entry.centers_on, entry.fallback)
            register!("$(id)_Soff", entry.centers_off, entry.fallback)
        end
    end

    # ---- Amendment A1: the refit correction over each analog raw core ----
    # The eligible configurations, their signatures and the frozen-feature oracle
    # were all settled before the first center was registered.
    all(config -> haskey(analog_entries, config), t1r_configs) || error(
        "a T1r calibration names an analog configuration this run did not score",
    )
    t1r_oracle_ok = isempty(t1r_configs) ? nothing : true
    if !isempty(t1r_configs)
        for config in t1r_configs
            calibration_path = joinpath(dev_dir, "t1r_fit_all_dev_$(config).csv")
            cal = read_operational_v2_calibration(calibration_path)
            entry = analog_entries[config]
            variant = _v23_t1r_variant_centers(ctx, t1r_view, entry.raw, entry.fallback, cal)
            id = v23_t1r_id(config)
            register!("$(id)_Son", variant.on, entry.fallback)
            register!("$(id)_Soff", variant.off, entry.fallback)
            v23_manifest!(manifest, "t1r_calibration", id, NaN, cal.label)
        end
        _v23_log("  T1r variants scored: $(join(v23_t1r_id.(t1r_configs), ", "))")
    end
    v23_manifest!(manifest, "t1r", "configs", length(t1r_configs),
                  join(t1r_configs, ";"))
    v23_manifest!(manifest, "seconds", "stage_candidates", time() - stage_started, "")

    # ---- comparators ----
    stage_started = time()
    direct_by_step = _v23_select_direct_gbm(dev_dir, plan.direct_grid)
    direct_centers = fill(NaN, length(ctx.anchors), V23_STEP_COUNT)
    direct_fallback = fill(false, length(ctx.anchors))
    direct_models = Dict{Tuple{String,Int},Any}()
    direct_feature_names = String.(v23_direct_feature_names())
    for (depth, rounds) in plan.direct_grid
        id = v23_direct_id(depth, rounds)
        capture = (_, slot, model, _) -> (direct_models[(id, slot)] = model)
        entry = first(_v23_run_stage!(ctx, [id], _ -> v23_direct_params(depth, rounds),
                                      () -> [v23_run_direct_stage!(ctx, folds, depth, rounds;
                                                                   on_model=capture)]))
        register!(id, entry.centers_on, entry.fallback)
        for (slot, step) in enumerate(V23_MODEL_STEPS)
            direct_by_step[step] == id || continue
            for i in rows
                direct_centers[i, slot] = entry.centers_on[i, slot]
            end
            direct_fallback .|= entry.fallback
        end
        v23_manifest!(manifest, "direct_gbm_selection", id, NaN,
                      join([string(step) for step in V23_MODEL_STEPS
                            if direct_by_step[step] == id], ","))
    end
    v23_manifest!(manifest, "direct_gbm_target", V23_DIRECT_TARGET, NaN,
                  "center = model + $(V23_DIRECT_TARGET_ANCHOR)")
    register!("direct_gbm", direct_centers, direct_fallback)
    oracle = first(_v23_run_stage!(ctx, [V23_ORACLE_ID], _ -> v23_oracle_params(),
                                   () -> [v23_run_oracle_stage!(ctx, folds).result]))
    register!("realized_driver_oracle", oracle.centers_on, oracle.fallback)

    base_centers(field) = begin
        matrix = fill(NaN, length(ctx.anchors), V23_STEP_COUNT)
        source = getproperty(ctx.anchors, field)
        for i in rows, slot in 1:V23_STEP_COUNT
            ctx.anchors.present[i, slot] || continue
            matrix[i, slot] = source[i, slot]
        end
        matrix
    end
    for (name, field) in (("served_v2_1", :served), ("frozen_v2_1", :frozen),
                          ("raw_sindy", :raw_served), ("persistence", :persistence),
                          ("burton", :burton), ("burton_full", :burton_full),
                          ("obrien", :obrien), ("static_v2_2", :static_v22))
        register!(name, base_centers(field), no_fallback)
    end
    tau = _v23_climatology_tau(ctx, fold.train)
    v23_manifest!(manifest, "comparator", "climatology_tau_hours", tau, "")
    register!("climatology_relaxed_persistence", _v23_climatology_centers(ctx, rows, tau),
              no_fallback)
    adc_spec = _v23_adc_spec(selection)
    v23_manifest!(manifest, "ablation", "sindy_removal_driver", adc_spec.k, adc_spec.source)
    removal = _v23_run_sindy_removal!(ctx, fold, adc_spec)
    register!("adc_burton", removal.burton, no_fallback)
    register!("adc_burton_full", removal.burton_full, no_fallback)
    register!("adc_obrien", removal.obrien, no_fallback)
    v23_manifest!(manifest, "seconds", "stage_comparators", time() - stage_started, "")

    # ---- composition with the development-selected weights and layers ----
    stage_started = time()
    selected_name = "$(selection.base_config)_S$(selection.safeguards ? "on" : "off")"
    haskey(scores, selected_name) || error(
        "the development selection $(selected_name) was not scored on this partition",
    )
    selected_centers = scores[selected_name]
    selected_fallback = fallbacks[selected_name]
    tail_rows = [i for i in rows if !selected_fallback[i]]
    lat_centers = v23_apply_lat(ctx, selected_centers, selection.lat_weights, tail_rows)
    for i in rows, slot in 1:V23_STEP_COUNT
        (selected_fallback[i] && ctx.anchors.present[i, slot]) || continue
        lat_centers[i, slot] = ctx.anchors.served[i, slot]
    end
    register!("V2_3_LAT", lat_centers, selected_fallback)

    development = _v23_read_dev_composition(ctx, dev_dir)
    isempty(intersect(Set(development.rows), Set(rows))) || error(
        "the development composition overlaps the scored partition; the error layers would " *
        "be fitted on rows they are about to correct",
    )
    combined = copy(development.centers)
    for i in rows, slot in 1:V23_STEP_COUNT
        combined[i, slot] = lat_centers[i, slot]
    end
    layer_train = [i for i in development.rows if !development.fallback[i]]
    e_models = Dict{Int,Any}()
    # Every (kind, parameter) pair is fitted at every step; only the fit a step
    # actually takes is the deployable layer for that step.
    capture_layer = (kind, param, slot, model, cap) -> begin
        chosen = selection.e_layers[slot]
        (chosen !== nothing && chosen.kind === kind && chosen.param == param) || return nothing
        e_models[slot] = (kind=kind, param=param, model=model, cap=Float64(cap))
        return nothing
    end
    layer = _v23_apply_fixed_e_layers(ctx, layer_train, tail_rows, combined,
                                      selection.e_layers; on_model=capture_layer)
    final_centers = fill(NaN, length(ctx.anchors), V23_STEP_COUNT)
    for i in rows, slot in 1:V23_STEP_COUNT
        ctx.anchors.present[i, slot] || continue
        final_centers[i, slot] = layer.centers[i, slot]
    end
    register!("V2_3_final", final_centers, selected_fallback)
    CSV.write(joinpath(plan.outdir, "e_layer_application.csv"),
              DataFrame(isempty(layer.applied) ?
                        [(layer="identity", model_step_hours=-1, param="none",
                          rows_corrected=0, rows_scored=length(rows))] : layer.applied))
    v23_manifest!(manifest, "seconds", "stage_composition", time() - stage_started, "")

    # ---- deployment artifacts of this fit ----
    # Written after the centers are final and read back by nothing in this
    # runner, so a live engine can reconstruct the scored forecast without
    # refitting it, and no score can depend on the persistence succeeding.
    stage_started = time()
    artifact_dir = joinpath(plan.outdir, V23_CONFIRM_ARTIFACT_DIR)
    mkpath(artifact_dir)
    artifact_paths = String[]
    e_layer_records = Vector{Any}(undef, V23_STEP_COUNT)
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        chosen = selection.e_layers[slot]
        if chosen === nothing
            e_layer_records[slot] = Dict{String,Any}(
                "model_step_hours" => step, "layer" => "identity", "param" => nothing,
                "development_gain_nt" => 0.0, "artifact" => nothing,
                "feature_names" => String[], "correction_cap_nt" => v23_e_cap(step),
            )
            continue
        end
        fitted = get(e_models, slot, nothing)
        fitted === nothing && error(
            "the fixed $(chosen.kind) layer at $(step) h produced no fit to persist; the " *
            "confirmatory fold cannot have applied it either",
        )
        names = v23_e_feature_names(chosen.kind)
        file = "e$(chosen.kind === :E1 ? 1 : 2)_step$(step)." *
               (chosen.kind === :E1 ? "csv" : "bson")
        path = joinpath(artifact_dir, file)
        if chosen.kind === :E1
            _v23_write_e1_artifact(path, fitted.model, names, fitted.cap)
        else
            v23_save(fitted.model, path)
        end
        push!(artifact_paths, path)
        e_layer_records[slot] = Dict{String,Any}(
            "model_step_hours" => step, "layer" => String(chosen.kind),
            "param" => string(chosen.param), "development_gain_nt" => chosen.gain_nt,
            "artifact" => file, "feature_names" => names,
            "correction_cap_nt" => fitted.cap,
        )
    end
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        id = direct_by_step[step]
        model = get(direct_models, (id, slot), nothing)
        if model === nothing
            # The stage was resumed from a persisted configuration, so no fit
            # passed through the sink. Refitting reproduces it exactly — the
            # design, the target and the seed are fixed — and touches no score,
            # because the centers were read back from the persisted artifact.
            _v23_log("  ! direct-GBM $(id) at $(step) h was reused from a persisted " *
                     "configuration; refitting it once so the artifact exists")
            features, usable = v23_direct_features!(ctx)
            fit_rows = [i for i in fold.train if usable[i] && ctx.anchors.present[i, slot]]
            depth, rounds = _v23_direct_grid_point(id, plan.direct_grid)
            v23_direct_check_anchor(features, ctx.anchors, fit_rows)
            model = v23_fit_gbm(features[fit_rows, :],
                                v23_direct_target(ctx.anchors, fit_rows, slot);
                                max_depth=depth, nrounds=rounds, eta=V23_GBM_ETA,
                                min_weight=V23_GBM_MIN_WEIGHT, seed=V23_SEED,
                                feature_names=direct_feature_names)
        end
        path = joinpath(artifact_dir, "direct_gbm_step$(step).bson")
        v23_save(model, path)
        push!(artifact_paths, path)
    end
    open(joinpath(artifact_dir, "e_layers.json"), "w") do io
        JSON3.pretty(io, Dict{String,Any}(
            "selected_config" => selection.config,
            "lat_weights" => selection.lat_weights,
            "direct_gbm_feature_names" => direct_feature_names,
            "direct_gbm_by_step" => Dict(string(step) => direct_by_step[step]
                                         for step in V23_MODEL_STEPS),
            # The persisted comparator regresses the change of Dst over the lead,
            # so a loader must add the issue-time Dst back to obtain a center.
            "direct_gbm_target" => V23_DIRECT_TARGET,
            "direct_gbm_target_anchor" => V23_DIRECT_TARGET_ANCHOR,
            "steps" => e_layer_records,
        ))
    end
    push!(artifact_paths, joinpath(artifact_dir, "e_layers.json"))
    archive = _v23_write_analog_archive!(ctx, fold, artifact_dir)
    push!(artifact_paths, archive.path)
    v23_manifest!(manifest, "analog_archive", "origins", archive.origins,
                  "first=$(archive.first_origin);last=$(archive.last_origin)")
    for path in artifact_paths
        v23_manifest!(manifest, "artifact_sha256",
                      joinpath(V23_CONFIRM_ARTIFACT_DIR, basename(path)),
                      filesize(path), _v23_sha256_file(path))
    end
    v23_manifest!(manifest, "seconds", "stage_artifacts", time() - stage_started, "")

    # ---- bootstrap ----
    stage_started = time()
    comparator_names = ["served_v2_1", "frozen_v2_1", "raw_sindy", "persistence", "burton",
                        "burton_full", "obrien", "static_v2_2", "direct_gbm",
                        "climatology_relaxed_persistence", "adc_burton", "adc_burton_full",
                        "adc_obrien", "realized_driver_oracle"]
    bootstrap = NamedTuple[]
    for comparator in comparator_names
        append!(bootstrap, _v23_bootstrap_rows("V2_3_final", comparator, final_centers,
                                               scores[comparator], ctx, rows))
    end
    for name in order
        (name == "V2_3_final" || name == "served_v2_1") && continue
        append!(bootstrap, _v23_bootstrap_rows(name, "served_v2_1", scores[name],
                                               scores["served_v2_1"], ctx, rows))
    end
    CSV.write(joinpath(plan.outdir, "bootstrap.csv"), DataFrame(bootstrap))
    v23_manifest!(manifest, "seconds", "stage_bootstrap", time() - stage_started, "")

    # ---- integrity, gates, decision ----
    stage_started = time()
    finite_ok = all(isfinite(final_centers[i, slot]) for i in rows
                    for slot in 1:V23_STEP_COUNT if ctx.anchors.present[i, slot])
    archive_ordered = maximum(ctx.anchors.issue[fold.train]) <
                      minimum(ctx.anchors.issue[rows])
    sampled_issues = ctx.anchors.issue[rows[1:max(1, div(length(rows), 25)):end]][1:min(25, length(rows))]
    integrity = [
        ("all_forecasts_finite", finite_ok, "every scored center is finite"),
        ("archive_precedes_queries", archive_ordered,
         "every analog origin precedes every scored issue"),
        ("feature_mutation_invariance", _v23_feature_invariance(ctx, sampled_issues),
         "destroying every frame value after the issue hour leaves the analog key unchanged"),
        ("tail_mutation_invariance",
         _v23_tail_invariance(ctx, fold, adc_spec, V23_INTEGRITY_SAMPLE),
         "replacing every driver record beyond the L1-admitted window leaves the center unchanged"),
    ]
    t1r_oracle_ok === nothing || push!(integrity, (
        "t1r_frozen_feature_oracle", t1r_oracle_ok,
        "rebuilding the archived calibration features from the raw inputs reproduces the base table",
    ))
    final_metrics = [row for row in metrics if row.config == "V2_3_final"]
    ablation_variants = [name for name in order if !_v23_preregistered_variant(name)]
    v23_manifest!(manifest, "ablation", "not_preregistered_variants",
                  length(ablation_variants), join(ablation_variants, ";"))
    gates = _v23_gate_rows(bootstrap, final_metrics, integrity, ablation_variants)
    CSV.write(joinpath(plan.outdir, "gates.csv"), DataFrame(gates.rows))
    # Fail closed before a decision exists: a run that breaches A4 has not
    # measured the candidate, and on the single-shot partition a decision record
    # written from unusable centers is worse than no record at all.
    _v23_assert_integrity(integrity)
    state, failing = _v23_decision(gates.verdicts)
    decision = DataFrame([(
        decision_state=state,
        failing_gates=join(failing, "|"),
        plan=plan.label,
        smoke=smoke,
        selected_config=selection.config,
        scored_anchors=length(rows),
        scored_rows=sum(count(i -> ctx.anchors.present[i, slot], rows)
                        for slot in 1:V23_STEP_COUNT),
        a1_pass=gates.verdicts.A1, a2_pass=gates.verdicts.A2, a3_pass=gates.verdicts.A3,
        a4_pass=gates.verdicts.A4, b1_pass=gates.verdicts.B1,
        b2_local_superiority_claim_allowed=gates.verdicts.B2_claim_ok,
        storm_cell_gains=gates.verdicts.storm_cell_gains,
        exposure_disclosure="TEST 2020-2022 was the V2.1 conformal holdout and TEST " *
            "2023-2025 was used in V2.1 development and safeguard selection; the static " *
            "V2.2 stack was fit on 2010-2017. Every comparator is exposure-favoured on " *
            "TEST and the V2.3 candidate is not.",
        base_table_sha256=ctx.base_table_sha,
        code_sha256=ctx.code_sha,
    )])
    CSV.write(decision_path, decision)
    v23_manifest!(manifest, "decision", "state", NaN, state)
    v23_manifest!(manifest, "decision", "failing_gates", NaN, join(failing, "|"))
    v23_manifest!(manifest, "seconds", "stage_gates", time() - stage_started, "")

    # ---- row-level scores and summaries ----
    stage_started = time()
    columns = Dict{Symbol,Vector}(
        :issue_time_utc => String[], :model_step_hours => Int[], :partition => String[],
        :cv_block => Int[], :observation_dst_nt => Float64[], :latest_dst_nt => Float64[],
        :dst_delta_1h_nt => Float64[], :coupling_active_mvm => Float64[],
    )
    for name in order
        columns[Symbol(name)] = Float64[]
    end
    for i in rows, (slot, step) in enumerate(V23_MODEL_STEPS)
        ctx.anchors.present[i, slot] || continue
        push!(columns[:issue_time_utc], string(ctx.anchors.issue[i]))
        push!(columns[:model_step_hours], step)
        push!(columns[:partition], ctx.anchors.partition[i])
        push!(columns[:cv_block], ctx.anchors.cv_block[i])
        push!(columns[:observation_dst_nt], ctx.anchors.obs[i, slot])
        push!(columns[:latest_dst_nt], ctx.anchors.latest[i])
        push!(columns[:dst_delta_1h_nt], ctx.anchors.rate[i])
        push!(columns[:coupling_active_mvm], ctx.anchors.coupling[i])
        for name in order
            push!(columns[Symbol(name)], scores[name][i, slot])
        end
    end
    header = vcat([:issue_time_utc, :model_step_hours, :partition, :cv_block,
                   :observation_dst_nt, :latest_dst_nt, :dst_delta_1h_nt,
                   :coupling_active_mvm], Symbol.(order))
    CSV.write(joinpath(plan.outdir, "test_scores.csv"),
              DataFrame([name => columns[name] for name in header]))
    # `preregistered` marks which scored columns belong to the family Amendment
    # A1 authorised; the T1r layer over the T1a ablation is reported, never a
    # candidate, and the summary now says so on every one of its rows.
    labelled = [merge(row, (preregistered=_v23_preregistered_variant(row.config),))
                for row in metrics]
    CSV.write(joinpath(plan.outdir, "test_summary.csv"), DataFrame(labelled))
    v23_manifest!(manifest, "seconds", "stage_scores", time() - stage_started, "")

    # ---- provenance of every input this decision rests on ----
    v23_manifest!(manifest, "input_sha256", "v2_3_provenance", NaN,
                  v23_provenance_signature())
    provenance_inputs = Any[
        ("research_plan", V23_RESEARCH_PLAN_PATH),
        ("selected_configuration.json", joinpath(dev_dir, V23_CONFIRM_SELECTION_JSON)),
        ("selection_trace.csv", joinpath(dev_dir, V23_CONFIRM_SELECTION_TRACE)),
        ("oof_V2_3_LAT.csv", joinpath(dev_dir, V23_CONFIRM_DEV_COMPOSITION)),
    ]
    for config in t1r_configs
        push!(provenance_inputs, ("t1r_fit_all_dev_$(config).csv",
                                  joinpath(dev_dir, "t1r_fit_all_dev_$(config).csv")))
        push!(provenance_inputs, ("sig_T1r_$(config).txt",
                                  v23_t1r_signature_path(dev_dir, config)))
    end
    for (depth, rounds) in plan.direct_grid
        name = "summary_$(v23_direct_id(depth, rounds)).csv"
        push!(provenance_inputs, (name, joinpath(dev_dir, name)))
    end
    for (name, path) in provenance_inputs
        v23_manifest!(manifest, "input_sha256", String(name), NaN,
                      isfile(path) ? _v23_sha256_file(path) : "absent")
    end
    v23_manifest!(manifest, "seconds", "total", time() - total_started, "")
    v23_write_manifest(joinpath(plan.outdir, "manifest.csv"), manifest, ctx)
    force_reason === nothing || _v23_append_rerun_log(
        plan, force_reason; stage="completed",
        replacement="$(state)|$(join(failing, "|"))",
    )
    _v23_log(@sprintf("V2.3 confirmatory complete in %.1f s → %s (decision %s)",
                      time() - total_started, plan.outdir, state))
    return (state=state, failing=failing, gates=gates, bootstrap=bootstrap,
            metrics=metrics, outdir=plan.outdir)
end

"The written reason carried by `flag=<text>`, or `nothing` when the flag is absent."
function _v23_written_reason(arguments, flag::AbstractString)
    prefix = "$(flag)="
    for argument in arguments
        startswith(argument, prefix) || continue
        reason = String(argument[length(prefix) + 1:end])
        isempty(strip(reason)) && error("$flag needs a non-empty reason")
        return reason
    end
    flag in arguments && error("$flag needs a reason, written as $(flag)=<text>")
    return nothing
end

"Extract the written reason of a forced rerun, or `nothing` when none was given."
_v23_force_reason(arguments) = _v23_written_reason(arguments, "--force-rerun-with-reason")

"Extract the written reason for accepting an input-contract mismatch."
_v23_accept_code_drift(arguments) = _v23_written_reason(arguments, "--accept-code-drift")

if abspath(PROGRAM_FILE) == @__FILE__
    run_v2_3_confirmatory(; smoke=("--smoke" in ARGS), confirm=("--confirm" in ARGS),
                          force_reason=_v23_force_reason(ARGS),
                          accept_code_drift=_v23_accept_code_drift(ARGS))
end
