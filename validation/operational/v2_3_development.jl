#!/usr/bin/env julia

# v2_3_development.jl — Operational V2.3 development runner (plan sections 3–4).
#
# Scores every preregistered tail-driver configuration on the six inner
# rolling-origin blocks of DEV, applies the fixed selection rule, then fits the
# lead-aware tail composition and adjudicates the two optional error layers. No
# TEST row is read: the runner loads the DEV partition only, so the single-shot
# confirmatory rule cannot be violated by accident here.
#
# Stage order is chosen so that partial work survives an interruption — each
# configuration writes its out-of-fold rows, its metrics and its resume
# signature as soon as it finishes, and a rerun reuses any configuration whose
# signature still matches the base table, the hourly frame and the V2.3 sources.
#
#   julia --project=. validation/operational/v2_3_development.jl --smoke
#   JULIA_NUM_THREADS=8 julia --project=. validation/operational/v2_3_development.jl
#
# `--smoke` runs the whole pipeline on one block, 2,000 anchors and a reduced
# grid, into a separate output tree; it exists to exercise the code path, never
# to produce a selection.
#
# Selection scope. This runner selects over the original plan section 4 family.
# Amendment A1 added the T1r members, which `v2_3_t1r.jl` scores from the
# out-of-fold artifacts this runner persists, so the authoritative selection over
# the full family — and the `selected_configuration.json` the confirmatory runner
# reads — is produced by `v2_3_select.jl` afterwards. Running this file again
# rewrites that contract with the pre-Amendment selection; rerun
# `v2_3_select.jl` after it.

isdefined(@__MODULE__, :V23Context) || include(joinpath(@__DIR__, "v2_3_common.jl"))

const V23_DEV_SELECTION_JSON = "selected_configuration.json"

"""
    _v23_guard_selection_contract(outdir; rewrite_selection)

Refuse to start when the output directory already holds a selection contract
that was written by the post-Amendment selection stage.

`v2_3_select.jl` writes `selected_configuration.json` with a
`selection_trace_sha256` key; this runner writes the pre-Amendment contract, so
completing here would silently replace the authoritative selection with an
earlier one that the confirmatory runner would then score on TEST. That is a
protocol failure, not a resume, so it needs the operator to say so in writing.
"""
function _v23_guard_selection_contract(outdir::AbstractString; rewrite_selection::Bool)
    path = joinpath(outdir, V23_DEV_SELECTION_JSON)
    isfile(path) || return nothing
    record = try
        JSON3.read(read(path, String))
    catch
        return nothing
    end
    haskey(record, :selection_trace_sha256) || return nothing
    rewrite_selection && begin
        _v23_log("  ! --rewrite-selection given: the post-Amendment contract at $path will " *
                 "be replaced by this runner's pre-Amendment selection; rerun " *
                 "v2_3_select.jl afterwards")
        return nothing
    end
    error(
        "$path carries selection_trace_sha256, so it was written by v2_3_select.jl over the " *
        "full Amendment A1 family. This runner would overwrite it with the pre-Amendment " *
        "selection and the confirmatory run would then score " *
        "$(get(record, :selected_config, "the earlier configuration")) instead of the " *
        "selected member. Pass --rewrite-selection to overwrite it deliberately (and rerun " *
        "v2_3_select.jl afterwards), or run v2_3_select.jl if you only need the selection.",
    )
end

"""
    _v23_reuse_or_run!(ctx, ids, params_of, compute) -> Vector{NamedTuple}

Resume-aware execution of one stage. When every configuration the stage would
produce already has a matching signature, the persisted metrics are read back
and nothing is recomputed; otherwise the stage runs and each configuration is
written as it completes.
"""
function _v23_reuse_or_run!(ctx::V23Context, ids::Vector{String}, params_of, compute)
    if all(id -> v23_config_is_current(ctx, id, params_of(id)), ids)
        _v23_log("  = reusing $(join(ids, ", "))")
        return reduce(vcat, [v23_metric_namedtuples(v23_read_config(ctx, id).metrics)
                             for id in ids])
    end
    started = time()
    results = compute()
    elapsed = time() - started
    metrics = NamedTuple[]
    for result in results
        written = v23_write_config!(ctx, result, elapsed / length(results))
        append!(metrics, written.metrics)
    end
    _v23_log(@sprintf("  + %s written in %.1f s", join(ids, ", "), elapsed))
    return metrics
end

"""
    run_v2_3_development(; smoke) -> NamedTuple

Execute the development protocol end to end and return the selection, the
lead-aware weights, the error-layer verdicts and the paths of every artifact.
"""
function run_v2_3_development(; smoke::Bool=false, rewrite_selection::Bool=false)
    total_started = time()
    plan = v23_dev_plan(; smoke=smoke)
    v23_assert_full_grid(plan)
    mkpath(plan.outdir)
    _v23_guard_selection_contract(plan.outdir; rewrite_selection=rewrite_selection)
    manifest = V23Manifest()
    v23_manifest!(manifest, "plan", "label", NaN, plan.label)
    v23_manifest!(manifest, "plan", "blocks", length(plan.blocks), join(plan.blocks, ","))
    v23_manifest!(manifest, "plan", "max_anchors", plan.max_anchors, "")

    _v23_log("V2.3 development runner ($(plan.label)); threads=$(Threads.nthreads())")
    context_started = time()
    ctx = v23_build_context(plan; partitions=("DEV",))
    folds = v23_dev_folds(ctx)
    v23_manifest!(manifest, "seconds", "context", time() - context_started, "")
    v23_manifest!(manifest, "rows", "dev_anchors", length(ctx.anchors), "")
    v23_manifest!(manifest, "rows", "feature_complete_anchors", count(ctx.ok), "")
    v23_manifest!(manifest, "rows", "origin_eligible_anchors", count(ctx.ok .& ctx.origin_ok), "")
    for fold in folds
        v23_manifest!(manifest, "fold", fold.name, length(fold.query),
                      "train_anchors=$(length(fold.train))")
    end
    _v23_log(@sprintf("  anchors %d (features complete %d), folds %d, context %.1f s",
                      length(ctx.anchors), count(ctx.ok), length(folds),
                      time() - context_started))

    candidate_metrics = NamedTuple[]
    comparator_metrics = NamedTuple[]

    # ---- Stage 1 and 2: analog driver continuation, one pass per weight set ----
    for weight_set in plan.weight_sets
        stage_started = time()
        ids = [v23_analog_id("T1", weight_set, k) for k in plan.t1_ks]
        metrics = _v23_reuse_or_run!(
            ctx, ids,
            id -> v23_analog_params("T1", weight_set,
                                    parse(Int, split(id, "_K")[end]), false),
            () -> begin
                results = v23_run_analog_stage!(ctx, folds, weight_set;
                                                ks=plan.t1_ks, direct=false, family="T1")
                [results[k] for k in sort(plan.t1_ks)]
            end,
        )
        append!(candidate_metrics, metrics)
        v23_manifest!(manifest, "seconds", "stage_T1_$(weight_set)",
                      time() - stage_started, "")
    end

    # ---- Stage 3: T1a direct-value ablation (uniform weights, K = 100) ----
    stage_started = time()
    t1a_k = smoke ? maximum(plan.t1_ks) : V23_T1A_K
    t1a_id = v23_analog_id("T1a", :uniform, t1a_k)
    append!(candidate_metrics, _v23_reuse_or_run!(
        ctx, [t1a_id], _ -> v23_analog_params("T1a", :uniform, t1a_k, true),
        () -> [v23_run_analog_stage!(ctx, folds, :uniform; ks=[t1a_k], direct=true,
                                     family="T1a")[t1a_k]],
    ))
    v23_manifest!(manifest, "seconds", "stage_T1a", time() - stage_started, "")

    # ---- Stage 4: direct machine-learning comparator ----
    stage_started = time()
    for (depth, rounds) in plan.direct_grid
        append!(comparator_metrics, _v23_reuse_or_run!(
            ctx, [v23_direct_id(depth, rounds)], _ -> v23_direct_params(depth, rounds),
            () -> [v23_run_direct_stage!(ctx, folds, depth, rounds)],
        ))
    end
    v23_manifest!(manifest, "direct_gbm_target", V23_DIRECT_TARGET, NaN,
                  "center = model + $(V23_DIRECT_TARGET_ANCHOR)")
    v23_manifest!(manifest, "seconds", "stage_direct_gbm", time() - stage_started, "")

    # ---- Stage 5: realized-driver oracle (noncausal ceiling) ----
    stage_started = time()
    append!(comparator_metrics, _v23_reuse_or_run!(
        ctx, [V23_ORACLE_ID], _ -> v23_oracle_params(),
        () -> [v23_run_oracle_stage!(ctx, folds).result],
    ))
    v23_manifest!(manifest, "seconds", "stage_oracle", time() - stage_started, "")

    # ---- Stage 6: boosted driver continuation ----
    stage_started = time()
    for (depth, rounds) in plan.gdc_grid
        append!(candidate_metrics, _v23_reuse_or_run!(
            ctx, [v23_gdc_id(depth, rounds)], _ -> v23_gdc_params(depth, rounds),
            () -> [v23_run_gdc_stage!(ctx, folds, depth, rounds)],
        ))
    end
    v23_manifest!(manifest, "seconds", "stage_T2", time() - stage_started, "")

    # ---- Selection rule ----
    candidates = v23_candidates(candidate_metrics)
    # The guards are a hard gate for the real run. A smoke run scores too few
    # storm rows for the intense-deepening cell to mean anything, so it selects
    # on RMSE alone and stamps the bypass into the manifest and the report.
    selected = v23_select(candidates; require_guards=!plan.smoke)
    v23_manifest!(manifest, "selection", "guards_required", plan.smoke ? 0 : 1, "")
    CSV.write(joinpath(plan.outdir, "selection_trace.csv"),
              DataFrame(v23_selection_rows(candidates, selected)))
    selected === nothing && begin
        v23_write_manifest(joinpath(plan.outdir, "dev_manifest.csv"), manifest, ctx)
        error("no preregistered configuration passed the storm-safety guards on DEV; " *
              "see $(joinpath(plan.outdir, "selection_trace.csv"))")
    end
    _v23_log(@sprintf("  selected %s (mean RMSE over steps 2/3/6 = %.4f nT)",
                      selected.id, selected.mean_rmse_nt))
    v23_manifest!(manifest, "selection", "config", selected.mean_rmse_nt, selected.id)

    # ---- Composition: lead-aware tail composition ----
    stage_started = time()
    stored = v23_read_config(ctx, selected.base_id)
    centers = selected.safeguards ? stored.centers_on : stored.centers_off
    rows = stored.rows
    fallback = stored.fallback
    tail_rows = [i for i in rows if !fallback[i]]
    lat = v23_lat_weights(ctx, centers, tail_rows)
    lat_centers = v23_apply_lat(ctx, centers, lat.weights, tail_rows)
    for i in rows, slot in 1:V23_STEP_COUNT
        (fallback[i] && ctx.anchors.present[i, slot]) || continue
        lat_centers[i, slot] = ctx.anchors.served[i, slot]
    end
    CSV.write(joinpath(plan.outdir, "lat_weights.csv"), DataFrame(lat.table))
    lat_params = Dict{String,Any}(
        "base_config" => selected.base_id, "safeguards" => selected.safeguards,
        "lat_weights" => lat.weights,
    )
    lat_metrics = v23_write_final_config!(ctx, "V2_3_LAT", "C1", lat_params, lat_centers,
                                          rows, fallback, selected.safeguards,
                                          time() - stage_started)
    v23_manifest!(manifest, "seconds", "stage_lat", time() - stage_started, "")
    _v23_log("  LAT weights per step: " * join(string.(lat.weights), ", "))

    # ---- Composition: optional error layers ----
    stage_started = time()
    e_folds_all = v23_e_layer_folds(ctx, folds)
    e_folds = [(name=f.name,
                train=[i for i in f.train if !fallback[i]],
                query=[i for i in f.query if !fallback[i]]) for f in e_folds_all]
    e_eval_rows = sort(unique(reduce(vcat, [f.query for f in e_folds]; init=Int[])))
    # The innovation chain is built over the same row set the confirmatory runner
    # uses: non-fallback rows only. A served-fallback anchor carries no tail
    # center, so including it here would make the six anchors after it see an
    # innovation the TEST path never forms.
    e_rows = [i for i in rows if !fallback[i]]
    audit = NamedTuple[]
    final_centers = copy(lat_centers)
    e_choice = Vector{Any}(undef, V23_STEP_COUNT)
    fill!(e_choice, nothing)
    if isempty(e_eval_rows)
        _v23_log("  E-layers skipped: no block has an embargoed predecessor block")
    else
        e1 = v23_run_e_layer(ctx, e_folds, lat_centers, e_rows, :E1;
                             params=collect(V23_E1_LAMBDA_GRID), label="E1",
                             eval_rows=e_eval_rows)
        e2 = v23_run_e_layer(ctx, e_folds, lat_centers, e_rows, :E2;
                             params=[(V23_E2_DEPTH, V23_E2_ROUNDS)], label="E2",
                             eval_rows=e_eval_rows)
        append!(audit, e1.audit)
        append!(audit, e2.audit)
        # Both layers are adjudicated against the same composed baseline; a step
        # takes whichever accepted layer removed more out-of-fold error, and the
        # identity when neither was accepted.
        for slot in 1:V23_STEP_COUNT
            step = V23_MODEL_STEPS[slot]
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
                e_choice[slot] = ("E1", e1.params[slot], g1)
                final_centers[:, slot] = e1.centers[:, slot]
            else
                e_choice[slot] = ("E2", e2.params[slot], g2)
                final_centers[:, slot] = e2.centers[:, slot]
            end
        end
    end
    for slot in 1:V23_STEP_COUNT
        chosen = e_choice[slot]
        push!(audit, (
            layer="selected", model_step_hours=V23_MODEL_STEPS[slot],
            param=chosen === nothing ? "identity" : string(chosen[2]),
            n_evaluable=length(e_eval_rows), rmse_base_nt=NaN, rmse_layer_nt=NaN,
            gain_nt=chosen === nothing ? 0.0 : chosen[3],
            selected_param=true, accepted=chosen !== nothing,
            guards_ok=true,
            guard_failures=chosen === nothing ? "identity" : chosen[1],
        ))
    end
    CSV.write(joinpath(plan.outdir, "e_layer_audit.csv"), DataFrame(audit))
    final_params = Dict{String,Any}(
        "base_config" => selected.base_id, "safeguards" => selected.safeguards,
        "lat_weights" => lat.weights,
        "e_layers" => [chosen === nothing ? "identity" : "$(chosen[1]):$(chosen[2])"
                       for chosen in e_choice],
    )
    final_metrics = v23_write_final_config!(ctx, "V2_3_final", "V2_3", final_params,
                                            final_centers, rows, fallback,
                                            selected.safeguards, time() - stage_started)
    v23_manifest!(manifest, "seconds", "stage_e_layers", time() - stage_started, "")

    # ---- Summary, report, manifest ----
    all_metrics = vcat(candidate_metrics, comparator_metrics, lat_metrics, final_metrics)
    CSV.write(joinpath(plan.outdir, "dev_summary.csv"), DataFrame(all_metrics))
    selection_record = Dict{String,Any}(
        "plan" => plan.label,
        "selected_config" => selected.id,
        "base_config" => selected.base_id,
        "family" => selected.family,
        "safeguards" => selected.safeguards,
        "k" => selected.k == typemax(Int) ? nothing : selected.k,
        "params" => JSON3.read(String(first(r for r in candidate_metrics
                                           if r.config == selected.id).params_json)),
        "mean_rmse_steps_2_3_6_nt" => selected.mean_rmse_nt,
        "lat_weights" => lat.weights,
        "e_layers" => [chosen === nothing ? nothing :
                       Dict{String,Any}("layer" => chosen[1], "param" => string(chosen[2]),
                                        "gain_nt" => chosen[3]) for chosen in e_choice],
        "model_steps" => collect(V23_MODEL_STEPS),
        "base_table_sha256" => ctx.base_table_sha,
        "hourly_frame_sha256" => ctx.frame_sha,
        "code_sha256" => ctx.code_sha,
    )
    open(joinpath(plan.outdir, V23_DEV_SELECTION_JSON), "w") do io
        JSON3.pretty(io, selection_record)
    end
    _v23_write_dev_report(ctx, plan, candidates, selected, lat, audit, all_metrics)
    v23_manifest!(manifest, "seconds", "total", time() - total_started, "")
    v23_write_manifest(joinpath(plan.outdir, "dev_manifest.csv"), manifest, ctx)
    _v23_log(@sprintf("V2.3 development complete in %.1f s → %s",
                      time() - total_started, plan.outdir))
    return (selected=selected, candidates=candidates, lat=lat, audit=audit,
            outdir=plan.outdir)
end

"Escape the markdown column separator so a multi-part guard string stays in one cell."
_v23_cell(text) = replace(String(text), "|" => "; ")

"Development report: tables only, so every interpretation stays in the decision record."
function _v23_write_dev_report(ctx::V23Context, plan::V23RunPlan, candidates, selected,
                               lat, audit, all_metrics)
    path = joinpath(plan.outdir, "dev_report.md")
    open(path, "w") do io
        println(io, "# Operational V2.3 development tables ($(plan.label))\n")
        println(io, "Blocks: ", join(plan.blocks, ", "),
                "; anchors: ", length(ctx.anchors),
                "; feature-complete anchors: ", count(ctx.ok),
                "; cells: ", ctx.cells_source, "\n")
        plan.smoke && println(io,
            "Smoke tables: reduced grid, capped anchors, storm-safety guards not required " *
            "for selection. Not a selection under the protocol.\n")

        println(io, "## Selection trace (mean pooled DEV-OOF RMSE over steps 2, 3, 6)\n")
        v23_markdown_table(io,
            ["rank", "config", "mean RMSE [nT]", "step 2", "step 3", "step 6",
             "guards", "intense rows", "failures", "selected"],
            [[string(rank), c.id, _v23_fmt(c.mean_rmse_nt), _v23_fmt(c.step_rmse[1]),
              _v23_fmt(c.step_rmse[2]), _v23_fmt(c.step_rmse[3]),
              string(c.guards.guards_ok), string(c.guards.intense_rows),
              isempty(c.guards.guard_failures) ? "-" : _v23_cell(c.guards.guard_failures),
              string(c.id == selected.id)]
             for (rank, c) in enumerate(candidates)])

        println(io, "## Pooled RMSE by model step\n")
        pooled = [r for r in all_metrics if r.cell == "all"]
        configs = unique([r.config for r in pooled])
        header = vcat(["config"], ["step $(h) [nT]" for h in V23_MODEL_STEPS],
                      ["fallback fraction"])
        rows = Vector{String}[]
        for config in configs
            entry = [config]
            for h in V23_MODEL_STEPS
                hit = [r for r in pooled if r.config == config && r.model_step_hours == h]
                push!(entry, isempty(hit) ? "-" : _v23_fmt(first(hit).rmse_nt))
            end
            hit = [r for r in pooled if r.config == config]
            push!(entry, isempty(hit) ? "-" : _v23_fmt(first(hit).fallback_fraction; digits=5))
            push!(rows, entry)
        end
        v23_markdown_table(io, header, rows)

        println(io, "## Storm-cell RMSE of the selected configuration versus served V2.1\n")
        cells = [r for r in all_metrics if r.config == "V2_3_final"]
        v23_markdown_table(io,
            ["cell", "step", "n", "RMSE V2.3 [nT]", "RMSE served [nT]", "loss [nT]",
             "bias V2.3 [nT]", "bias served [nT]"],
            [[r.cell, string(r.model_step_hours), string(r.n), _v23_fmt(r.rmse_nt),
              _v23_fmt(r.rmse_served_nt), _v23_fmt(r.loss_vs_served_nt),
              _v23_fmt(r.bias_nt), _v23_fmt(r.bias_served_nt)] for r in cells])

        println(io, "## Lead-aware tail composition weights\n")
        v23_markdown_table(io,
            ["step", "n", "w", "RMSE(w) [nT]", "w=0", "w=0.25", "w=0.5", "w=0.75", "w=1"],
            [[string(r.model_step_hours), string(r.n), _v23_fmt(r.selected_weight; digits=2),
              _v23_fmt(r.rmse_selected_nt), _v23_fmt(r.rmse_w0_00_nt),
              _v23_fmt(r.rmse_w0_25_nt), _v23_fmt(r.rmse_w0_50_nt),
              _v23_fmt(r.rmse_w0_75_nt), _v23_fmt(r.rmse_w1_00_nt)] for r in lat.table])

        println(io, "## Error-layer audit\n")
        v23_markdown_table(io,
            ["layer", "step", "parameter", "n", "RMSE base [nT]", "RMSE layer [nT]",
             "gain [nT]", "selected", "accepted", "guards"],
            [[r.layer, string(r.model_step_hours), r.param, string(r.n_evaluable),
              _v23_fmt(r.rmse_base_nt), _v23_fmt(r.rmse_layer_nt), _v23_fmt(r.gain_nt),
              string(r.selected_param), string(r.accepted), string(r.guards_ok)]
             for r in audit])
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_v2_3_development(; smoke=("--smoke" in ARGS),
                         rewrite_selection=("--rewrite-selection" in ARGS))
end
