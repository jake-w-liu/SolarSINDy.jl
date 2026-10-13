#!/usr/bin/env julia

# analog_select.jl — plan section 4 selection over the *full* preregistered
# candidate family, including the Amendment A1 T1r members, followed by the
# lead-aware tail composition (C1) and the two optional error layers (E1, E2) on
# the selected member.
#
# Why this is a separate stage. `analog_development.jl` scored and selected over
# the original section 4 family before Amendment A1 was logged, so its
# `selected_configuration.json` predates T1r. T1r is a post-processing layer over
# the analog raw cores that `analog_t1r.jl` scores from the same out-of-fold
# artifacts, in the same cell partition, with the same guard evaluator. Re-running
# the whole development pass to fold it in would recompute 50 minutes of
# ensembles that have not changed; reading the persisted per-configuration
# summaries and applying the fixed rule to all of them is the same selection and
# is reproducible from the artifacts alone.
#
# What the rule is (plan section 4, unchanged): among the preregistered family,
# minimise the mean pooled DEV out-of-fold RMSE over model steps {2, 3, 6}
# subject to the section 6-A2 storm-safety guards evaluated on DEV out-of-fold
# rows against served inertial-SINDy; ties break toward the smaller ensemble and then
# toward safeguards on. The guard evaluator, the cell definitions and the pooled
# statistic are the ones `analog_common.jl` already owns, so a member selected here
# is scored on TEST by the identical machinery.
#
# The pre-Amendment artifacts are copied to `*_pre_t1r.*` before anything is
# rewritten, so the earlier selection stays auditable.
#
#   julia --project=. validation/operational/analog_select.jl --smoke
#   JULIA_NUM_THREADS=8 julia --project=. validation/operational/analog_select.jl
#
# `--smoke` works on the development smoke tree, whose grid is reduced and whose
# storm cells are too small for the guards to mean anything; it exercises the code
# path and is never a selection under the protocol.

include(joinpath(@__DIR__, "analog_t1r.jl"))

const ANALOG_SELECT_JSON = "selected_configuration.json"
const ANALOG_SELECT_TRACE = "selection_trace.csv"
const ANALOG_SELECT_LAT = "lat_weights.csv"
const ANALOG_SELECT_E_AUDIT = "e_layer_audit.csv"
const ANALOG_SELECT_MANIFEST = "selection_manifest.csv"

"Artifacts of the pre-Amendment selection that this stage rewrites."
const ANALOG_SELECT_PRESERVED = (ANALOG_SELECT_JSON, ANALOG_SELECT_TRACE, ANALOG_SELECT_LAT,
                              ANALOG_SELECT_E_AUDIT)
const ANALOG_SELECT_PRE_T1R_TAG = "_pre_t1r"

"""
    ANALOG_SELECT_COMPOSED

Composed artifacts the post-Amendment stage rewrites in place: the lead-aware
and finished out-of-fold rows, their metric summaries and their resume
signatures. They are preserved on the same `_pre_t1r` rule as the selection
contract, but only when the rewrite actually changes them, so a reproducible
rerun does not manufacture a spurious "earlier" copy of its own output.
"""
const ANALOG_SELECT_COMPOSED = ("oof_ANALOG_LAT.csv", "oof_ANALOG_final.csv",
                             "summary_ANALOG_LAT.csv", "summary_ANALOG_final.csv",
                             "sig_ANALOG_LAT.txt", "sig_ANALOG_final.txt")

"Marker written beside the pre-Amendment development report and manifest."
const ANALOG_SELECT_SUPERSEDED_MARKER = "SUPERSEDED_BY_selection_manifest.txt"

"Development artifacts whose recorded composition predates this stage."
const ANALOG_SELECT_SUPERSEDED_FILES = ("dev_summary.csv", "dev_report.md", "dev_manifest.csv")

"""
    ANALOG_SELECT_FAMILIES

Families whose persisted summaries are candidate configurations. `DGBM` and
`ORACLE` are comparators, and `C1` / `ANALOG` are composed outputs of an earlier
selection; none of them is selectable, and reading them in would let a
comparator win the candidate rule.
"""
const ANALOG_SELECT_FAMILIES = ("T1", "T1a", "T2", ANALOG_T1R_FAMILY)

# ---------------------------------------------------------------------------
# Artifact handling
# ---------------------------------------------------------------------------

"Path of the preserved pre-Amendment copy of one artifact."
function analog_select_pre_t1r_path(outdir::AbstractString, file::AbstractString)
    stem, extension = splitext(file)
    return joinpath(outdir, string(stem, ANALOG_SELECT_PRE_T1R_TAG, extension))
end

"""
    analog_select_preserve!(outdir, files) -> Vector{String}

Copy the pre-Amendment selection artifacts aside before they are rewritten. A
destination that already exists is left alone, so a second run of this stage
preserves the original record rather than overwriting it with its own output.
"""
function analog_select_preserve!(outdir::AbstractString, files=ANALOG_SELECT_PRESERVED)
    kept = String[]
    for file in files
        source = joinpath(outdir, file)
        isfile(source) || continue
        destination = analog_select_pre_t1r_path(outdir, file)
        isfile(destination) || cp(source, destination)
        push!(kept, destination)
    end
    return kept
end

"""
    analog_select_stage!(outdir, files; scratch) -> Vector{NamedTuple}

Copy the composed artifacts of the previous selection aside *before* this stage
rewrites them, recording the digest each one had. The copies live outside the
run directory until [`analog_select_commit_staged!`](@ref) decides whether they are
worth keeping.
"""
function analog_select_stage!(outdir::AbstractString, files=ANALOG_SELECT_COMPOSED;
                           scratch::AbstractString=mktempdir())
    staged = NamedTuple[]
    for file in files
        source = joinpath(outdir, file)
        isfile(source) || continue
        copy = joinpath(scratch, file)
        cp(source, copy; force=true)
        push!(staged, (file=String(file), copy=copy, sha=_analog_sha256_file(source)))
    end
    return staged
end

"""
    analog_select_commit_staged!(outdir, staged) -> Vector{String}

Keep a staged copy as `*_pre_t1r` only when this stage actually changed the
artifact and no earlier copy exists; otherwise discard it. A reproducible rerun
therefore leaves the audit trail exactly as it found it instead of relabelling
its own output as the pre-Amendment record.
"""
function analog_select_commit_staged!(outdir::AbstractString, staged)
    kept = String[]
    for entry in staged
        current = joinpath(outdir, entry.file)
        changed = isfile(current) && _analog_sha256_file(current) != entry.sha
        destination = analog_select_pre_t1r_path(outdir, entry.file)
        if changed && !isfile(destination)
            cp(entry.copy, destination)
            push!(kept, destination)
        end
        rm(entry.copy; force=true)
    end
    return kept
end

"""
    analog_select_write_superseded_marker!(outdir, manifest_name) -> String

State, next to the pre-Amendment development report and manifest, that they
describe the selection this stage replaced. Those files record the SHA-256 of
composed artifacts that no longer exist in that form, and a manifest that names
a digest the file on disk no longer has is a false audit trail unless the reader
is told which manifest is authoritative.
"""
function analog_select_write_superseded_marker!(outdir::AbstractString,
                                             manifest_name::AbstractString=ANALOG_SELECT_MANIFEST)
    path = joinpath(outdir, ANALOG_SELECT_SUPERSEDED_MARKER)
    present = [file for file in ANALOG_SELECT_SUPERSEDED_FILES if isfile(joinpath(outdir, file))]
    open(path, "w") do io
        println(io, "Superseded development artifacts")
        println(io)
        println(io, "The files below were written by analog_development.jl before Amendment A1 ",
                "was folded in. They describe the pre-Amendment selection and the composed ",
                "artifacts that belonged to it, so the SHA-256 values they record for ",
                "oof_ANALOG_LAT.csv, oof_ANALOG_final.csv, summary_ANALOG_*.csv, sig_ANALOG_*.txt, ",
                "selection_trace.csv, lat_weights.csv, e_layer_audit.csv and ",
                "selected_configuration.json no longer match the files on disk.")
        println(io)
        for file in present
            println(io, "  ", file)
        end
        println(io)
        println(io, "Authoritative manifest for the current composition: ", manifest_name, ".")
        println(io, "Pre-Amendment copies of the rewritten artifacts carry the ",
                ANALOG_SELECT_PRE_T1R_TAG, " suffix.")
        println(io, "Written ", now(UTC), " UTC.")
    end
    return path
end

"""
    analog_select_candidate_metrics(outdir; families) -> NamedTuple

Every persisted per-configuration metric row whose family is a candidate family,
read back from `summary_*.csv`. A summary whose rows disagree about their family
is a corrupted artifact and fails here rather than silently entering the rule.

When `table_sha` is given, every T1r summary must additionally carry a valid
`sig_T1r_<config>.txt`: those artifacts are written outside the stage machinery,
so without the key a summary produced by superseded sources or by a superseded
analog core would enter the selection rule unchallenged.

When `ctx` is given, every T1, T1a and T2 summary must likewise still match its
resume signature `sig_<config>.txt` (`analog_config_is_current`, the key the
development runner reuses artifacts by). The selection record is stamped with the
current code and input digests, so a stale artifact admitted here would pass the
confirmatory drift check under a provenance it does not have.
"""
function analog_select_candidate_metrics(outdir::AbstractString;
                                      families=ANALOG_SELECT_FAMILIES,
                                      table_sha::Union{Nothing,AbstractString}=nothing,
                                      code_sha::AbstractString=analog_code_signature(),
                                      ctx::Union{Nothing,AnalogContext}=nothing)
    isdir(outdir) || error("development output directory is missing: $outdir")
    wanted = Set(String.(families))
    metrics = NamedTuple[]
    files = String[]
    for file in sort(readdir(outdir))
        (startswith(file, "summary_") && endswith(file, ".csv")) || continue
        table = CSV.read(joinpath(outdir, file), DataFrame)
        "family" in names(table) || continue
        nrow(table) > 0 || continue
        family = String(table.family[1])
        all(String(f) == family for f in table.family) || error(
            "$file mixes configuration families",
        )
        family in wanted || continue
        if family == ANALOG_T1R_FAMILY && table_sha !== nothing
            prefix = "summary_$(ANALOG_T1R_FAMILY)_"
            startswith(file, prefix) || error(
                "$file declares the $(ANALOG_T1R_FAMILY) family but does not name a member",
            )
            analog_t1r_require_signature(outdir, file[length(prefix) + 1:end - 4];
                                      table_sha=table_sha, code_sha=code_sha)
        end
        if family != ANALOG_T1R_FAMILY && ctx !== nothing
            id = file[length("summary_") + 1:end - 4]
            analog_config_is_current(ctx, id, analog_select_config_params(id)) || error(
                "the artifacts of $id cannot be used: $(analog_signature_path(ctx.plan, id)) is " *
                "missing or stale. Rerun analog_development.jl so its out-of-fold rows and " *
                "summary carry a signature over the current base table, frame and sources.",
            )
        end
        append!(metrics, analog_metric_namedtuples(table))
        push!(files, file)
    end
    isempty(metrics) && error(
        "no candidate summaries of families $(join(collect(families), ", ")) in $outdir",
    )
    return (metrics=metrics, files=files)
end

"""
    analog_select_config_params(id) -> Dict{String,Any}

Parameter record of a persisted T1, T1a or T2 configuration, rebuilt from its
identity with the same constructors the development runner signs it with. The
identity is checked against the forward map rather than assumed.
"""
function analog_select_config_params(id::AbstractString)
    analog = match(r"^(T1a?)_([a-z]+)_K(\d+)$", String(id))
    if analog !== nothing
        family, weight_set, k = analog.captures[1], analog.captures[2],
                                parse(Int, analog.captures[3])
        analog_id(family, weight_set, k) == String(id) || throw(ArgumentError(
            "$(id) does not invert to an analog configuration",
        ))
        return analog_params(family, weight_set, k, family == "T1a")
    end
    gdc = match(r"^T2_d(\d+)_r(\d+)$", String(id))
    if gdc !== nothing
        depth, rounds = parse(Int, gdc.captures[1]), parse(Int, gdc.captures[2])
        gbm_driver_id(depth, rounds) == String(id) || throw(ArgumentError(
            "$(id) does not invert to a T2 configuration",
        ))
        return gbm_driver_params(depth, rounds)
    end
    throw(ArgumentError("$(id) is not a T1, T1a or T2 configuration"))
end

"""
    analog_select_analog_config(base_id) -> String

The analog configuration a T1r member corrects. `analog_t1r_id` prefixes the analog
identity, so the inverse is a prefix strip that is checked against the forward
map rather than assumed.
"""
function analog_select_analog_config(base_id::AbstractString)
    startswith(String(base_id), "T1r_") || throw(ArgumentError(
        "$(base_id) is not a T1r configuration",
    ))
    analog = String(base_id)[5:end]
    analog_t1r_id(analog) == String(base_id) || throw(ArgumentError(
        "$(base_id) does not invert to an analog configuration",
    ))
    return analog
end

"""
    analog_select_eligible(plan) -> Set{String}

Base configurations the protocol authorises as selectable for this run scale:
the plan section 4 family `{T1 × W × K, T1a, T2}` and, from Amendment A1, the
T1r layer over the T1 grid. The T1r layer over the T1a ablation is produced by
the cross-fit stage for reporting, but the amendment puts the refit correction on
the T1 grid, so it is not selectable and the trace records that.
"""
function analog_select_eligible(plan::AnalogRunPlan)
    t1a_k = plan.smoke ? maximum(plan.t1_ks) : ANALOG_T1A_K
    ids = String[]
    for weight_set in plan.weight_sets, k in plan.t1_ks
        push!(ids, analog_id("T1", weight_set, k))
    end
    push!(ids, analog_id("T1a", :uniform, t1a_k))
    for (depth, rounds) in plan.gdc_grid
        push!(ids, gbm_driver_id(depth, rounds))
    end
    for weight_set in plan.weight_sets, k in plan.t1_ks
        push!(ids, analog_t1r_id(analog_id("T1", weight_set, k)))
    end
    return Set(ids)
end

"""
    analog_select_trace_rows(candidates, selected, eligible) -> Vector{NamedTuple}

Selection trace in ranked order: every scored configuration, its selection
statistic, its per-step components, its guard verdict and failures, whether the
protocol authorises it as selectable, and which one the rule chose.
"""
function analog_select_trace_rows(candidates, selected, eligible)
    rows = NamedTuple[]
    for (rank, c) in enumerate(candidates)
        push!(rows, (
            rank=rank, config=c.id, base_config=c.base_id, family=c.family,
            safeguards=c.safeguards, k=(c.k == typemax(Int) ? -1 : c.k),
            preregistered=(c.base_id in eligible),
            mean_rmse_steps_2_3_6_nt=c.mean_rmse_nt,
            rmse_step_2_nt=c.step_rmse[1], rmse_step_3_nt=c.step_rmse[2],
            rmse_step_6_nt=c.step_rmse[3],
            cell_guard_ok=c.guards.cell_guard_ok,
            intense_guard_ok=c.guards.intense_guard_ok,
            guards_ok=c.guards.guards_ok,
            intense_rows=c.guards.intense_rows,
            guard_failures=c.guards.guard_failures,
            selected=(selected !== nothing && c.id == selected.id),
        ))
    end
    return rows
end

"""
    analog_select_apply_rule(candidates, eligible; require_guards) -> NamedTuple

Plan section 4 rule restricted to the authorised family. The unrestricted
minimum and the unrestricted guard-passing minimum are returned alongside, so a
run can state what the rule excluded instead of hiding it.
"""
function analog_select_apply_rule(candidates, eligible; require_guards::Bool=true)
    inside = [c for c in candidates if c.base_id in eligible]
    isempty(inside) && error(
        "no scored configuration belongs to the preregistered family; the summaries " *
        "and the run plan disagree",
    )
    finite = [c for c in candidates if isfinite(c.mean_rmse_nt)]
    unrestricted = isempty(finite) ? nothing : first(sort(finite; by=_analog_candidate_key))
    passing = [c for c in finite if c.guards.guards_ok]
    unrestricted_guarded = isempty(passing) ? nothing :
        first(sort(passing; by=_analog_candidate_key))
    return (selected=analog_select(inside; require_guards=require_guards),
            eligible=inside, unrestricted=unrestricted,
            unrestricted_guarded=unrestricted_guarded)
end

# ---------------------------------------------------------------------------
# Centers of the selected member
# ---------------------------------------------------------------------------

"""
    analog_select_member_centers(ctx, candidate) -> NamedTuple

Out-of-fold centers, scored anchors and served-fallback flags of one family
member. A T1, T1a or T2 member is read straight from its persisted configuration.
A T1r member takes its centers from the cross-fit artifact of `analog_t1r.jl` and
its scored anchors and fallback flags from the analog configuration it corrects,
because T1r replaces only the correction layer and inherits the analog run's row
set; the two are checked to agree before either is used.
"""
function analog_select_member_centers(ctx::AnalogContext, candidate::AnalogCandidate)
    if candidate.family != ANALOG_T1R_FAMILY
        stored = analog_read_config(ctx, candidate.base_id)
        return (centers_on=stored.centers_on, centers_off=stored.centers_off,
                rows=stored.rows, fallback=stored.fallback,
                analog_config=candidate.base_id, calibration=nothing)
    end
    analog = analog_select_analog_config(candidate.base_id)
    analog_t1r_require_signature(ctx.plan.outdir, analog; table_sha=ctx.base_table_sha,
                              code_sha=ctx.code_sha)
    stored = analog_read_config(ctx, analog)
    path = analog_t1r_oof_path(ctx.plan.outdir, analog)
    isfile(path) || error("T1r out-of-fold artifact is missing: $path")
    calibration = analog_t1r_calibration_path(ctx.plan.outdir, analog)
    isfile(calibration) || error(
        "T1r all-DEV calibration is missing: $calibration (the confirmatory run needs it)",
    )
    n = length(ctx.anchors)
    centers_on = fill(NaN, n, ANALOG_STEP_COUNT)
    centers_off = fill(NaN, n, ANALOG_STEP_COUNT)
    seen = falses(n)
    table = CSV.read(path, DataFrame; types=Dict("issue_time_utc" => DateTime))
    for r in 1:nrow(table)
        i = get(ctx.anchors.index, table.issue_time_utc[r], 0)
        i == 0 && error("T1r artifact $path holds an unknown issue time")
        slot = ANALOG_STEP_SLOT[Int(table.model_step_hours[r])]
        slot == 0 && error("T1r artifact $path holds an unscored model step")
        centers_on[i, slot] = Float64(table.center_s_on_dst_nt[r])
        centers_off[i, slot] = Float64(table.center_s_off_dst_nt[r])
        seen[i] = true
    end
    rows = findall(seen)
    rows == stored.rows || error(
        "T1r artifact $path scores $(length(rows)) anchors but the analog configuration " *
        "$analog scores $(length(stored.rows)); the layer and its core disagree",
    )
    return (centers_on=centers_on, centers_off=centers_off, rows=rows,
            fallback=stored.fallback, analog_config=analog, calibration=calibration)
end

# ---------------------------------------------------------------------------
# Composition stages (identical to the development runner, on the new member)
# ---------------------------------------------------------------------------

"""
    analog_select_lat!(ctx, centers, rows, fallback) -> NamedTuple

Lead-aware tail composition of the selected member: per model step the weight
minimising the pooled out-of-fold RMSE of `w · member + (1 − w) · frozen inertial-SINDy`,
fitted on the rows the member actually forecast. Served-fallback anchors keep the
deployed product, because there is no tail center to blend for them.
"""
function analog_select_lat!(ctx::AnalogContext, centers::AbstractMatrix{Float64},
                         rows::AbstractVector{Int}, fallback::AbstractVector{Bool})
    tail_rows = [i for i in rows if !fallback[i]]
    isempty(tail_rows) && error("the selected member forecast no anchor of its own")
    lat = analog_lat_weights(ctx, centers, tail_rows)
    lat_centers = analog_apply_lat(ctx, centers, lat.weights, tail_rows)
    for i in rows, slot in 1:ANALOG_STEP_COUNT
        (fallback[i] && ctx.anchors.present[i, slot]) || continue
        lat_centers[i, slot] = ctx.anchors.served[i, slot]
    end
    return (weights=lat.weights, table=lat.table, centers=lat_centers, tail_rows=tail_rows)
end

"""
    analog_select_e_layers!(ctx, folds, lat_centers, rows, fallback) -> NamedTuple

Adjudicate E1 and E2 on the composed center with the expanding-window folds the
development runner uses: block `b` is corrected by a layer fitted on the pooled
out-of-fold rows of the earlier blocks, embargoed by 168 h, so the first block has
no layer and is excluded from the layer evaluation. Both layers are adjudicated
against the same baseline and a step takes whichever accepted layer removed more
out-of-fold error, the identity when neither was accepted.
"""
function analog_select_e_layers!(ctx::AnalogContext, folds, lat_centers::AbstractMatrix{Float64},
                              rows::AbstractVector{Int}, fallback::AbstractVector{Bool})
    e_folds_all = analog_e_layer_folds(ctx, folds)
    e_folds = [(name=f.name,
                train=[i for i in f.train if !fallback[i]],
                query=[i for i in f.query if !fallback[i]]) for f in e_folds_all]
    e_eval_rows = sort(unique(reduce(vcat, [f.query for f in e_folds]; init=Int[])))
    # Innovation chain over non-fallback rows only, the row set the confirmatory
    # runner forms; a served-fallback anchor has no tail center to innovate on.
    e_rows = [i for i in rows if !fallback[i]]
    audit = NamedTuple[]
    final_centers = copy(lat_centers)
    choice = Vector{Any}(undef, ANALOG_STEP_COUNT)
    fill!(choice, nothing)
    if isempty(e_eval_rows)
        _analog_log("  E-layers skipped: no block has an embargoed predecessor block")
    else
        e1 = analog_run_e_layer(ctx, e_folds, lat_centers, e_rows, :E1;
                             params=collect(ANALOG_E1_LAMBDA_GRID), label="E1",
                             eval_rows=e_eval_rows)
        e2 = analog_run_e_layer(ctx, e_folds, lat_centers, e_rows, :E2;
                             params=[(ANALOG_E2_DEPTH, ANALOG_E2_ROUNDS)], label="E2",
                             eval_rows=e_eval_rows)
        append!(audit, e1.audit)
        append!(audit, e2.audit)
        for slot in 1:ANALOG_STEP_COUNT
            step = ANALOG_MODEL_STEPS[slot]
            gain_of(layer_audit, layer) = begin
                hit = [r for r in layer_audit
                       if r.model_step_hours == step && r.accepted && r.layer == layer]
                isempty(hit) ? -Inf : first(hit).gain_nt
            end
            g1 = gain_of(e1.audit, "E1")
            g2 = gain_of(e2.audit, "E2")
            if max(g1, g2) <= 0
                continue
            elseif g1 >= g2
                choice[slot] = ("E1", e1.params[slot], g1)
                final_centers[:, slot] = e1.centers[:, slot]
            else
                choice[slot] = ("E2", e2.params[slot], g2)
                final_centers[:, slot] = e2.centers[:, slot]
            end
        end
    end
    for slot in 1:ANALOG_STEP_COUNT
        chosen = choice[slot]
        push!(audit, (
            layer="selected", model_step_hours=ANALOG_MODEL_STEPS[slot],
            param=chosen === nothing ? "identity" : string(chosen[2]),
            n_evaluable=length(e_eval_rows), rmse_base_nt=NaN, rmse_layer_nt=NaN,
            gain_nt=chosen === nothing ? 0.0 : chosen[3],
            selected_param=true, accepted=chosen !== nothing,
            guards_ok=true,
            guard_failures=chosen === nothing ? "identity" : chosen[1],
        ))
    end
    return (centers=final_centers, audit=audit, choice=choice, eval_rows=e_eval_rows)
end

"""
    analog_select_record(...) -> Dict{String,Any}

Contents of `selected_configuration.json`, the input contract of the confirmatory
runner. It keeps every key the development runner wrote and adds the four
Amendment A1 fields the confirmatory path needs: which family the selected
member belongs to, which analog configuration supplies its raw core, which
persisted calibration carries the refit correction, and the digest of the
selection trace the member was chosen from.
"""
function analog_select_record(ctx::AnalogContext, plan::AnalogRunPlan, selected::AnalogCandidate,
                           member, lat, choice, params, trace_sha::AbstractString,
                           lat_oof_sha::AbstractString="")
    return Dict{String,Any}(
        "plan" => plan.label,
        "selected_config" => selected.id,
        "base_config" => selected.base_id,
        "family" => selected.family,
        "analog_config" => member.analog_config,
        "t1r_calibration_csv" => member.calibration === nothing ? nothing :
                                 basename(member.calibration),
        "safeguards" => selected.safeguards,
        "k" => selected.k == typemax(Int) ? nothing : selected.k,
        "params" => params,
        "mean_rmse_steps_2_3_6_nt" => selected.mean_rmse_nt,
        "lat_weights" => lat.weights,
        "e_layers" => [chosen === nothing ? nothing :
                       Dict{String,Any}("layer" => chosen[1], "param" => string(chosen[2]),
                                        "gain_nt" => chosen[3]) for chosen in choice],
        "model_steps" => collect(ANALOG_MODEL_STEPS),
        "base_table_sha256" => ctx.base_table_sha,
        "hourly_frame_sha256" => ctx.frame_sha,
        "code_sha256" => ctx.code_sha,
        "selection_trace_sha256" => String(trace_sha),
        # Digest of the composed out-of-fold rows the confirmatory run reads as
        # the fitting sample of the fixed error layers. Without it a development
        # rerun that died between writing the composition and writing this
        # contract would leave a member and a composition that do not belong to
        # each other, and nothing downstream would notice.
        "oof_analog_lat_sha256" => String(lat_oof_sha),
    )
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    run_analog_selection(; smoke) -> NamedTuple

Apply the plan section 4 rule to every persisted candidate configuration,
including the Amendment A1 T1r members, then fit the lead-aware weights and
adjudicate the error layers on the selected member and rewrite the confirmatory
input contract.
"""
function run_analog_selection(; smoke::Bool=false)
    total_started = time()
    plan = analog_dev_plan(; smoke=smoke)
    analog_assert_full_grid(plan)
    isdir(plan.outdir) || error(
        "development artifacts are missing: $(plan.outdir) (run analog_development.jl first)",
    )
    manifest = AnalogManifest()
    analog_manifest!(manifest, "plan", "label", NaN, plan.label)
    analog_manifest!(manifest, "plan", "stage", NaN, "select_with_t1r")

    _analog_log("analog ensemble selection over the full family ($(plan.label)); " *
             "threads=$(Threads.nthreads())")
    started = time()
    ctx = analog_build_context(plan; partitions=("DEV",))
    folds = analog_dev_folds(ctx)
    analog_manifest!(manifest, "seconds", "context", time() - started, "")
    _analog_log(@sprintf("  anchors %d, folds %d, context %.1f s",
                      length(ctx.anchors), length(folds), time() - started))

    loaded = analog_select_candidate_metrics(plan.outdir; table_sha=ctx.base_table_sha,
                                          code_sha=ctx.code_sha, ctx=ctx)
    candidates = analog_candidates(loaded.metrics)
    eligible = analog_select_eligible(plan)
    rule = analog_select_apply_rule(candidates, eligible; require_guards=!plan.smoke)
    selected = rule.selected
    preserved = analog_select_preserve!(plan.outdir)
    scratch = mktempdir()
    staged = analog_select_stage!(plan.outdir; scratch=scratch)
    for path in preserved
        analog_manifest!(manifest, "preserved", basename(path), NaN, path)
    end
    trace_path = joinpath(plan.outdir, ANALOG_SELECT_TRACE)
    CSV.write(trace_path, DataFrame(analog_select_trace_rows(candidates, selected, eligible)))
    analog_manifest!(manifest, "selection", "summaries_read", length(loaded.files),
                  join(loaded.files, ";"))
    analog_manifest!(manifest, "selection", "candidates", length(candidates), "")
    analog_manifest!(manifest, "selection", "preregistered_candidates",
                  length(rule.eligible), "")
    analog_manifest!(manifest, "selection", "guards_required", plan.smoke ? 0 : 1, "")
    if selected === nothing
        analog_write_manifest(joinpath(plan.outdir, ANALOG_SELECT_MANIFEST), manifest, ctx)
        error("no preregistered configuration passed the storm-safety guards on DEV; " *
              "see $trace_path")
    end
    rule.unrestricted === nothing || analog_manifest!(
        manifest, "selection", "unrestricted_minimum", rule.unrestricted.mean_rmse_nt,
        rule.unrestricted.id,
    )
    rule.unrestricted_guarded === nothing || analog_manifest!(
        manifest, "selection", "unrestricted_guarded_minimum",
        rule.unrestricted_guarded.mean_rmse_nt, rule.unrestricted_guarded.id,
    )
    analog_manifest!(manifest, "selection", "config", selected.mean_rmse_nt, selected.id)
    _analog_log(@sprintf(
        "  selected %s (family %s, S=%s, mean pooled OOF RMSE over steps 2/3/6 = %.4f nT)",
        selected.id, selected.family, selected.safeguards ? "on" : "off",
        selected.mean_rmse_nt,
    ))
    rule.unrestricted === nothing || _analog_log(@sprintf(
        "  unrestricted minimum over every scored configuration: %s at %.4f nT (guards %s)",
        rule.unrestricted.id, rule.unrestricted.mean_rmse_nt,
        rule.unrestricted.guards.guards_ok ? "ok" : "FAIL",
    ))

    # ---- composition: lead-aware tail composition ----
    stage_started = time()
    member = analog_select_member_centers(ctx, selected)
    centers = selected.safeguards ? member.centers_on : member.centers_off
    lat = analog_select_lat!(ctx, centers, member.rows, member.fallback)
    CSV.write(joinpath(plan.outdir, ANALOG_SELECT_LAT), DataFrame(lat.table))
    lat_params = Dict{String,Any}(
        "base_config" => selected.base_id, "family" => selected.family,
        "analog_config" => member.analog_config,
        "safeguards" => selected.safeguards, "lat_weights" => lat.weights,
    )
    lat_metrics = analog_write_final_config!(ctx, "ANALOG_LAT", "C1", lat_params, lat.centers,
                                          member.rows, member.fallback,
                                          selected.safeguards, time() - stage_started)
    analog_manifest!(manifest, "seconds", "stage_lat", time() - stage_started, "")
    _analog_log("  LAT weights per step " * join(string.(ANALOG_MODEL_STEPS), "/") * ": " *
             join(string.(lat.weights), ", "))

    # ---- composition: optional error layers ----
    stage_started = time()
    layers = analog_select_e_layers!(ctx, folds, lat.centers, member.rows, member.fallback)
    CSV.write(joinpath(plan.outdir, ANALOG_SELECT_E_AUDIT), DataFrame(layers.audit))
    final_params = Dict{String,Any}(
        "base_config" => selected.base_id, "family" => selected.family,
        "analog_config" => member.analog_config,
        "safeguards" => selected.safeguards, "lat_weights" => lat.weights,
        "e_layers" => [chosen === nothing ? "identity" : "$(chosen[1]):$(chosen[2])"
                       for chosen in layers.choice],
    )
    final_metrics = analog_write_final_config!(ctx, "ANALOG_final", "ANALOG", final_params,
                                            layers.centers, member.rows, member.fallback,
                                            selected.safeguards, time() - stage_started)
    analog_manifest!(manifest, "seconds", "stage_e_layers", time() - stage_started, "")
    for (slot, step) in enumerate(ANALOG_MODEL_STEPS)
        chosen = layers.choice[slot]
        _analog_log(@sprintf("  E-layer at %d h: %s", step,
                          chosen === nothing ? "identity" :
                          @sprintf("%s(%s), gain %.3f nT", chosen[1], string(chosen[2]),
                                   chosen[3])))
    end

    # ---- contract, summary, manifest ----
    params = JSON3.read(String(first(r for r in loaded.metrics
                                     if r.config == selected.id).params_json))
    lat_oof_path = analog_oof_path(plan, "ANALOG_LAT")
    record = analog_select_record(ctx, plan, selected, member, lat, layers.choice, params,
                               _analog_sha256_file(trace_path), _analog_sha256_file(lat_oof_path))
    open(joinpath(plan.outdir, ANALOG_SELECT_JSON), "w") do io
        JSON3.pretty(io, record)
    end
    CSV.write(joinpath(plan.outdir, "selection_summary.csv"),
              DataFrame(vcat(lat_metrics, final_metrics)))
    for path in analog_select_commit_staged!(plan.outdir, staged)
        analog_manifest!(manifest, "preserved", basename(path), NaN, path)
    end
    rm(scratch; force=true, recursive=true)
    marker = analog_select_write_superseded_marker!(plan.outdir)
    analog_manifest!(manifest, "superseded", basename(marker), NaN,
                  join(ANALOG_SELECT_SUPERSEDED_FILES, ";"))
    analog_manifest!(manifest, "input_sha256", "analog_provenance", NaN,
                  analog_provenance_signature())
    analog_manifest!(manifest, "seconds", "total", time() - total_started, "")
    analog_write_manifest(joinpath(plan.outdir, ANALOG_SELECT_MANIFEST), manifest, ctx)
    _analog_log(@sprintf("analog ensemble selection complete in %.1f s → %s",
                      time() - total_started, plan.outdir))
    return (selected=selected, candidates=candidates, eligible=eligible, rule=rule,
            member=member, lat=lat, layers=layers, record=record, outdir=plan.outdir,
            metrics=vcat(lat_metrics, final_metrics))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analog_selection(; smoke=("--smoke" in ARGS))
end
