module V23SelectTests

# Unit oracles for the full-family selection stage
# (`validation/operational/v2_3_select.jl`).
#
# This stage decides which member of the preregistered family becomes V2.3, so
# the ways it can be wrong are the ways the study can be wrong: it can read a
# comparator as if it were a candidate, it can let a configuration the protocol
# does not authorise win, it can lose the storm-safety guards or the tie-break,
# it can pair a T1r layer with the wrong analog core, or it can overwrite the
# pre-Amendment record it is supposed to preserve. Each test below fixes one of
# those against an expectation written independently of the code under test.

using Test
using CSV
using DataFrames
using Dates
using JSON3
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "v2_3_select.jl")))

const CELLS = collect(V23_CELL_NAMES)

"""
Metric rows of one configuration in the schema `v23_write_config!` persists:
every cell at every model step, with the pooled `all` cell carrying the score
that the selection statistic averages.
"""
function metric_rows(id, family, safeguards, k, rmse; loss=-1.0, bias=0.0,
                     intense_loss=nothing, intense_bias=nothing, n=500)
    rows = NamedTuple[]
    for step in V23_MODEL_STEPS, cell in CELLS
        cell_loss = (cell === :intense_deepening && intense_loss !== nothing &&
                     step == V23_GUARD_INTENSE_STEP) ? intense_loss : loss
        cell_bias = (cell === :intense_deepening && intense_bias !== nothing &&
                     step == V23_GUARD_INTENSE_STEP) ? intense_bias : bias
        push!(rows, (
            config="$(id)_S$(safeguards ? "on" : "off")", model_step_hours=step,
            cell=String(cell), n=n, rmse_nt=rmse, bias_nt=cell_bias,
            rmse_served_nt=rmse - cell_loss, bias_served_nt=0.0,
            loss_vs_served_nt=cell_loss, fallback_fraction=0.0, family=String(family),
            safeguards=safeguards, params_json="{\"k\":$(k)}", seconds=1.0,
        ))
    end
    return rows
end

write_summary(dir, id, family, safeguards, k, rmse; kwargs...) =
    CSV.write(joinpath(dir, "summary_$(id)_S$(safeguards ? "on" : "off").csv"),
              DataFrame(metric_rows(id, family, safeguards, k, rmse; kwargs...)))

@testset "the authorised family is the section 4 grid plus Amendment A1 on the T1 grid" begin
    eligible = v23_select_eligible(v23_dev_plan())
    expected = Set([
        "T1_uniform_K25", "T1_uniform_K50", "T1_uniform_K100", "T1_uniform_K200",
        "T1_magnetic_K25", "T1_magnetic_K50", "T1_magnetic_K100", "T1_magnetic_K200",
        "T1a_uniform_K100",
        "T2_d4_r200", "T2_d4_r400", "T2_d6_r200", "T2_d6_r400",
        "T1r_T1_uniform_K25", "T1r_T1_uniform_K50", "T1r_T1_uniform_K100",
        "T1r_T1_uniform_K200", "T1r_T1_magnetic_K25", "T1r_T1_magnetic_K50",
        "T1r_T1_magnetic_K100", "T1r_T1_magnetic_K200",
    ])
    @test eligible == expected
    @test length(eligible) == 21
    # Amendment A1 puts the refit correction on the T1 grid; the layer over the
    # T1a ablation is scored and reported but is not selectable.
    @test !("T1r_T1a_uniform_K100" in eligible)
    # The smoke plan carries a reduced grid and the eligibility follows it.
    smoke = v23_select_eligible(v23_dev_plan(; smoke=true))
    @test smoke == Set(["T1_uniform_K25", "T1_uniform_K50", "T1_magnetic_K25",
                        "T1_magnetic_K50", "T1a_uniform_K50", "T2_d4_r200",
                        "T1r_T1_uniform_K25", "T1r_T1_uniform_K50",
                        "T1r_T1_magnetic_K25", "T1r_T1_magnetic_K50"])
    @test issubset(Set(["T1_uniform_K25", "T1_magnetic_K50"]), eligible)
end

@testset "a T1r identity inverts to the analog configuration it corrects" begin
    for config in ("T1_uniform_K25", "T1_magnetic_K200", "T1a_uniform_K100")
        @test v23_select_analog_config(v23_t1r_id(config)) == config
    end
    @test_throws ArgumentError v23_select_analog_config("T1_uniform_K25")
    @test_throws ArgumentError v23_select_analog_config("T2_d4_r200")
end

@testset "only candidate families are read back as candidates" begin
    directory = mktempdir()
    write_summary(directory, "T1_uniform_K25", "T1", true, 25, 8.0)
    write_summary(directory, "T1r_T1_uniform_K25", "T1r", false, 25, 7.0)
    write_summary(directory, "T2_d4_r200", "T2", true, -1, 9.0)
    write_summary(directory, "T1a_uniform_K100", "T1a", true, 100, 9.5)
    # Comparators and composed outputs must not enter the candidate rule, even
    # though they live in the same directory and share the schema.
    write_summary(directory, "DGBM_d4_r200", "DGBM", true, -1, 1.0)
    write_summary(directory, "ORACLE_realized", "ORACLE", true, -1, 0.5)
    write_summary(directory, "V2_3_LAT", "C1", true, 25, 0.6)
    write_summary(directory, "V2_3_final", "V2_3", true, 25, 0.7)
    loaded = v23_select_candidate_metrics(directory)
    families = unique([String(row.family) for row in loaded.metrics])
    @test sort(families) == ["T1", "T1a", "T1r", "T2"]
    @test length(loaded.files) == 4
    configs = unique([String(row.config) for row in loaded.metrics])
    @test !any(occursin("DGBM", c) || occursin("ORACLE", c) || occursin("V2_3", c)
               for c in configs)

    # A summary whose rows disagree about their family is corrupt and fails closed.
    mixed = vcat(metric_rows("T1_uniform_K50", "T1", true, 50, 8.0),
                 metric_rows("T1_uniform_K50", "T2", true, 50, 8.0))
    CSV.write(joinpath(directory, "summary_mixed.csv"), DataFrame(mixed))
    @test_throws ErrorException v23_select_candidate_metrics(directory)
    rm(joinpath(directory, "summary_mixed.csv"))
    @test length(v23_select_candidate_metrics(directory).files) == 4

    empty_directory = mktempdir()
    write_summary(empty_directory, "DGBM_d4_r200", "DGBM", true, -1, 1.0)
    @test_throws ErrorException v23_select_candidate_metrics(empty_directory)
end

@testset "the rule minimises inside the authorised family subject to the guards" begin
    # An unauthorised configuration that both scores best and passes the guards
    # must not win, and the trace must say why it was not eligible.
    metrics = vcat(
        metric_rows("T1r_T1a_uniform_K100", "T1r", false, 100, 6.0),   # best, not eligible
        metric_rows("T1r_T1_uniform_K200", "T1r", false, 200, 7.0;
                    intense_loss=0.5, intense_bias=0.0),               # eligible, guard fails
        metric_rows("T1r_T1_magnetic_K25", "T1r", false, 25, 7.5),     # eligible, guards pass
        metric_rows("T1_magnetic_K50", "T1", true, 50, 8.0),
    )
    candidates = v23_candidates(metrics)
    eligible = v23_select_eligible(v23_dev_plan())
    rule = v23_select_apply_rule(candidates, eligible)
    @test rule.selected.id == "T1r_T1_magnetic_K25_Soff"
    @test rule.selected.family == "T1r"
    @test rule.selected.safeguards == false
    @test rule.unrestricted.id == "T1r_T1a_uniform_K100_Soff"
    @test rule.unrestricted_guarded.id == "T1r_T1a_uniform_K100_Soff"
    @test length(rule.eligible) == 3

    trace = v23_select_trace_rows(candidates, rule.selected, eligible)
    @test [row.rank for row in trace] == collect(1:length(candidates))
    @test first(trace).config == "T1r_T1a_uniform_K100_Soff"
    @test first(trace).preregistered == false
    chosen = only([row for row in trace if row.selected])
    @test chosen.config == "T1r_T1_magnetic_K25_Soff"
    @test chosen.preregistered
    breached = only([row for row in trace if row.config == "T1r_T1_uniform_K200_Soff"])
    @test breached.preregistered
    @test !breached.guards_ok
    @test occursin("intense_deepening", breached.guard_failures)

    # Ties break toward the smaller ensemble and then toward safeguards on.
    tied = vcat(metric_rows("T1r_T1_magnetic_K200", "T1r", true, 200, 7.5),
                metric_rows("T1r_T1_magnetic_K25", "T1r", false, 25, 7.5),
                metric_rows("T1r_T1_magnetic_K25", "T1r", true, 25, 7.5))
    @test v23_select_apply_rule(v23_candidates(tied), eligible).selected.id ==
          "T1r_T1_magnetic_K25_Son"

    # When every authorised configuration breaches a guard the rule returns
    # nothing rather than relaxing the guard.
    failing = vcat(
        metric_rows("T1r_T1_magnetic_K25", "T1r", false, 25, 7.5;
                    intense_loss=0.5, intense_bias=0.0),
        metric_rows("T1_magnetic_K50", "T1", true, 50, 8.0;
                    intense_loss=0.5, intense_bias=0.0),
    )
    @test v23_select_apply_rule(v23_candidates(failing), eligible).selected === nothing
    @test v23_select_apply_rule(v23_candidates(failing), eligible;
                                require_guards=false).selected.id ==
          "T1r_T1_magnetic_K25_Soff"

    # A family with no authorised member at all is a disagreement between the
    # artifacts and the run plan, not a selection.
    outside = metric_rows("T1r_T1a_uniform_K100", "T1r", false, 100, 6.0)
    @test_throws ErrorException v23_select_apply_rule(v23_candidates(outside), eligible)
end

@testset "the pre-Amendment artifacts are preserved and never overwritten twice" begin
    directory = mktempdir()
    write(joinpath(directory, "selection_trace.csv"), "rank,config\n1,old\n")
    write(joinpath(directory, "lat_weights.csv"), "model_step_hours\n1\n")
    # `selected_configuration.json` and `e_layer_audit.csv` are absent here; a
    # missing artifact is skipped rather than fabricated.
    kept = v23_select_preserve!(directory)
    @test sort(basename.(kept)) == ["lat_weights_pre_t1r.csv", "selection_trace_pre_t1r.csv"]
    @test read(joinpath(directory, "selection_trace_pre_t1r.csv"), String) ==
          "rank,config\n1,old\n"
    @test v23_select_pre_t1r_path(directory, "selected_configuration.json") ==
          joinpath(directory, "selected_configuration_pre_t1r.json")

    # A second pass must keep the original record, not replace it with its own
    # output.
    write(joinpath(directory, "selection_trace.csv"), "rank,config\n1,new\n")
    v23_select_preserve!(directory)
    @test read(joinpath(directory, "selection_trace_pre_t1r.csv"), String) ==
          "rank,config\n1,old\n"
end

@testset "composed artifacts are preserved only when the rewrite changes them" begin
    directory = mktempdir()
    scratch = mktempdir()
    write(joinpath(directory, "oof_V2_3_LAT.csv"), "issue_time_utc\n2015-01-01T00:00:00\n")
    write(joinpath(directory, "sig_V2_3_LAT.txt"), "old-signature")
    staged = v23_select_stage!(directory, ("oof_V2_3_LAT.csv", "sig_V2_3_LAT.txt",
                                           "summary_V2_3_final.csv"); scratch=scratch)
    # An artifact that does not exist yet is skipped rather than fabricated.
    @test sort([entry.file for entry in staged]) == ["oof_V2_3_LAT.csv", "sig_V2_3_LAT.txt"]

    # A reproducible rerun writes byte-identical output, so nothing is relabelled
    # as the earlier record and no staged copy survives.
    kept = v23_select_commit_staged!(directory, staged)
    @test isempty(kept)
    @test !isfile(joinpath(directory, "oof_V2_3_LAT_pre_t1r.csv"))
    @test all(!isfile(entry.copy) for entry in staged)

    # A rewrite that changes the artifact keeps the version it replaced.
    staged = v23_select_stage!(directory, ("oof_V2_3_LAT.csv",); scratch=mktempdir())
    write(joinpath(directory, "oof_V2_3_LAT.csv"), "issue_time_utc\n2016-01-01T00:00:00\n")
    kept = v23_select_commit_staged!(directory, staged)
    @test basename.(kept) == ["oof_V2_3_LAT_pre_t1r.csv"]
    @test read(joinpath(directory, "oof_V2_3_LAT_pre_t1r.csv"), String) ==
          "issue_time_utc\n2015-01-01T00:00:00\n"

    # A third pass keeps the original record even when the artifact changes again.
    staged = v23_select_stage!(directory, ("oof_V2_3_LAT.csv",); scratch=mktempdir())
    write(joinpath(directory, "oof_V2_3_LAT.csv"), "issue_time_utc\n2017-01-01T00:00:00\n")
    @test isempty(v23_select_commit_staged!(directory, staged))
    @test read(joinpath(directory, "oof_V2_3_LAT_pre_t1r.csv"), String) ==
          "issue_time_utc\n2015-01-01T00:00:00\n"
end

@testset "the superseded development artifacts are marked, not left to mislead" begin
    directory = mktempdir()
    for file in ("dev_summary.csv", "dev_report.md")
        write(joinpath(directory, file), "superseded content\n")
    end
    path = v23_select_write_superseded_marker!(directory)
    @test basename(path) == V23_SELECT_SUPERSEDED_MARKER
    text = read(path, String)
    @test occursin("dev_summary.csv", text)
    @test occursin("dev_report.md", text)
    # A file that is not there is not claimed to be.
    @test !occursin("  dev_manifest.csv", text)
    @test occursin(V23_SELECT_MANIFEST, text)
    @test occursin(V23_SELECT_PRE_T1R_TAG, text)
end

@testset "a T1r summary without a valid signature never enters the rule" begin
    directory = mktempdir()
    config = "T1_magnetic_K25"
    write(joinpath(directory, "oof_$(config).csv"), "issue_time_utc\n2015-01-01T00:00:00\n")
    summary = DataFrame(config=["$(v23_t1r_id(config))_Soff"], model_step_hours=[2],
                        cell=["all"], n=[10], rmse_nt=[7.0], bias_nt=[0.0],
                        rmse_served_nt=[8.0], bias_served_nt=[0.0],
                        loss_vs_served_nt=[-1.0], fallback_fraction=[0.0],
                        family=[V23_T1R_FAMILY], safeguards=[false], params_json=["{}"],
                        seconds=[1.0])
    CSV.write(v23_t1r_summary_path(directory, config), summary)
    touch(v23_t1r_oof_path(directory, config))
    touch(v23_t1r_calibration_path(directory, config))

    # Without a key the artifacts are refused; the unverified path is only taken
    # when the caller explicitly asks for no verification.
    @test_throws ErrorException v23_select_candidate_metrics(directory; table_sha="table-sha",
                                                             code_sha="code-sha")
    @test length(v23_select_candidate_metrics(directory).metrics) == 1

    write(v23_t1r_signature_path(directory, config),
          v23_t1r_signature(directory, config; table_sha="table-sha", code_sha="code-sha"))
    @test length(v23_select_candidate_metrics(directory; table_sha="table-sha",
                                              code_sha="code-sha").metrics) == 1
    # A key written against another base table, another source revision or
    # another analog core is stale, not acceptable.
    @test_throws ErrorException v23_select_candidate_metrics(directory; table_sha="other-sha",
                                                             code_sha="code-sha")
    @test_throws ErrorException v23_select_candidate_metrics(directory; table_sha="table-sha",
                                                             code_sha="other-code")
    write(joinpath(directory, "oof_$(config).csv"), "issue_time_utc\n2016-01-01T00:00:00\n")
    @test_throws ErrorException v23_select_candidate_metrics(directory; table_sha="table-sha",
                                                             code_sha="code-sha")
end

"Anchor view with fully controlled numbers, for the member and composition tests."
function tiny_anchors(count::Int; obs, served, frozen)
    issues = [DateTime(2015, 6, 1) + Hour(i) for i in 0:count - 1]
    index = Dict(t => i for (i, t) in enumerate(issues))
    driver = [(V=400.0, Bz=-5.0, By=2.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
              for _ in 1:count]
    zeros_matrix() = zeros(count, V23_STEP_COUNT)
    return V23Anchors(
        issues, index, fill("DEV", count), fill(2015, count), driver,
        fill(-10.0, count), zeros(count), zeros(count), fill(-30.0, count),
        trues(count, V23_STEP_COUNT), obs, served, frozen,
        zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(),
        zeros_matrix(), zeros_matrix(),
    )
end

function tiny_context(anchors::V23Anchors, outdir::AbstractString)
    plan = V23RunPlan("unit", outdir, true, [2015], 0, [:uniform], [25], [(4, 200)],
                      [(4, 200)])
    count = length(anchors)
    return V23Context(plan, anchors, DataFrame(), Dict{DateTime,_V23_DRIVER_NT}(),
                      nothing, nothing, zeros(count, V23_FEATURE_COUNT), trues(count),
                      trues(count), v23_cell_masks(anchors), nothing, "table-sha",
                      "frame-sha", "code-sha", "unit", 0.0)
end

candidate(id, base, family, safeguards, k) = V23Candidate(
    id, base, family, safeguards, k, 7.5, [7.0, 7.5, 8.0], v23_guard_flags(NamedTuple[]),
)

@testset "a T1r member takes its centers from the layer and its rows from the core" begin
    outdir = mktempdir()
    count = 24
    anchors = tiny_anchors(count; obs=zeros(count, V23_STEP_COUNT),
                           served=fill(-7.0, count, V23_STEP_COUNT),
                           frozen=fill(-3.0, count, V23_STEP_COUNT))
    ctx = tiny_context(anchors, outdir)
    analog = "T1_magnetic_K25"
    fallback = [i <= 4 for i in 1:count]
    params = v23_analog_params("T1", :magnetic, 25, false)
    core = V23ConfigResult(analog, "T1", params, fill(-11.0, count, V23_STEP_COUNT),
                           fallback, collect(1:count), false)
    v23_write_config!(ctx, core, 1.0)

    # The layer artifact carries its own centers; the analog artifact carries the
    # row set and the served-fallback flags.
    layer_rows = NamedTuple[]
    for i in 1:count, (slot, step) in enumerate(V23_MODEL_STEPS)
        push!(layer_rows, (issue_time_utc=string(anchors.issue[i]), cv_block=2015,
                           model_step_hours=step,
                           center_s_on_dst_nt=-20.0 - slot, center_s_off_dst_nt=-30.0 - slot))
    end
    CSV.write(v23_t1r_oof_path(outdir, analog), DataFrame(layer_rows))
    touch(v23_t1r_calibration_path(outdir, analog))
    touch(v23_t1r_summary_path(outdir, analog))
    signature_path = v23_t1r_signature_path(outdir, analog)
    write(signature_path, v23_t1r_signature(outdir, analog; table_sha=ctx.base_table_sha,
                                            code_sha=ctx.code_sha))

    selected = candidate("T1r_$(analog)_Soff", "T1r_$(analog)", "T1r", false, 25)
    member = v23_select_member_centers(ctx, selected)
    @test member.analog_config == analog
    @test member.rows == collect(1:count)
    @test member.fallback == fallback
    @test basename(member.calibration) == "t1r_fit_all_dev_$(analog).csv"
    for slot in 1:V23_STEP_COUNT
        @test all(member.centers_off[:, slot] .== -30.0 - slot)
        @test all(member.centers_on[:, slot] .== -20.0 - slot)
    end

    # A non-T1r member is read straight from its own configuration.
    plain = v23_select_member_centers(ctx, candidate("$(analog)_Soff", analog, "T1", false, 25))
    @test plain.analog_config == analog
    @test plain.calibration === nothing
    @test all(plain.centers_off[5:count, :] .== -11.0)
    @test all(plain.centers_off[1:4, :] .== -7.0)      # served-fallback anchors

    # A missing confirmatory calibration is a blocker, because the layer could
    # not then be reproduced on TEST.
    rm(v23_t1r_calibration_path(outdir, analog))
    @test_throws ErrorException v23_select_member_centers(
        ctx, candidate("T1r_$(analog)_Soff", "T1r_$(analog)", "T1r", false, 25))
    touch(v23_t1r_calibration_path(outdir, analog))

    # A layer artifact that scores a different row set than its core is refused.
    CSV.write(v23_t1r_oof_path(outdir, analog),
              DataFrame([r for r in layer_rows
                         if r.issue_time_utc != string(anchors.issue[count])]))
    @test_throws ErrorException v23_select_member_centers(
        ctx, candidate("T1r_$(analog)_Soff", "T1r_$(analog)", "T1r", false, 25))
    CSV.write(v23_t1r_oof_path(outdir, analog), DataFrame(layer_rows))
    @test v23_select_member_centers(ctx, selected).rows == collect(1:count)
    # Recomputing the analog core without recomputing its layer leaves the
    # provenance key stale, and a stale key stops the member before any center
    # is read.
    write(v23_oof_path(ctx.plan, analog),
          read(v23_oof_path(ctx.plan, analog), String) * "\n")
    @test_throws ErrorException v23_select_member_centers(ctx, selected)
    write(signature_path, v23_t1r_signature(outdir, analog; table_sha=ctx.base_table_sha,
                                            code_sha=ctx.code_sha))
    @test v23_select_member_centers(ctx, selected).rows == collect(1:count)
    rm(signature_path)
    @test_throws ErrorException v23_select_member_centers(ctx, selected)
end

@testset "the lead-aware stage minimises the blend and serves the fallback anchors" begin
    outdir = mktempdir()
    count = 60
    served = fill(9.0, count, V23_STEP_COUNT)
    anchors = tiny_anchors(count; obs=zeros(count, V23_STEP_COUNT), served=served,
                           frozen=fill(2.0, count, V23_STEP_COUNT))
    ctx = tiny_context(anchors, outdir)
    centers = fill(-2.0, count, V23_STEP_COUNT)
    fallback = [i <= 10 for i in 1:count]
    lat = v23_select_lat!(ctx, centers, collect(1:count), fallback)
    # Blending 2 nT of frozen with −2 nT of member gives 2 − 4w, minimised at 0.5.
    @test all(lat.weights .== 0.5)
    @test lat.tail_rows == collect(11:count)
    @test all(isapprox(row.rmse_selected_nt, 0.0; atol=1e-12) for row in lat.table)
    @test all(row.n == count - 10 for row in lat.table)
    @test all(lat.centers[1:10, :] .== 9.0)
    @test all(isapprox.(lat.centers[11:count, :], 0.0; atol=1e-12))

    # With every anchor served there is no tail to compose and the stage stops.
    @test_throws ErrorException v23_select_lat!(ctx, centers, collect(1:count),
                                                trues(count))
end

@testset "the confirmatory contract records the Amendment A1 fields" begin
    outdir = mktempdir()
    anchors = tiny_anchors(4; obs=zeros(4, V23_STEP_COUNT), served=zeros(4, V23_STEP_COUNT),
                          frozen=zeros(4, V23_STEP_COUNT))
    ctx = tiny_context(anchors, outdir)
    plan = v23_dev_plan()
    selected = candidate("T1r_T1_magnetic_K25_Soff", "T1r_T1_magnetic_K25", "T1r", false, 25)
    member = (analog_config="T1_magnetic_K25",
              calibration=joinpath(outdir, "t1r_fit_all_dev_T1_magnetic_K25.csv"))
    lat = (weights=[1.0, 1.0, 1.0, 1.0, 0.75, 0.75],)
    choice = Any[("E2", (3, 128), 0.2), nothing, nothing, nothing, ("E1", 1000.0, 0.05),
                 nothing]
    params = Dict{String,Any}("weight_set" => "magnetic", "k" => 25, "direct" => false)
    record = v23_select_record(ctx, plan, selected, member, lat, choice, params, "trace-sha",
                               "lat-oof-sha")
    @test record["oof_v2_3_lat_sha256"] == "lat-oof-sha"
    @test record["family"] == "T1r"
    @test record["base_config"] == "T1r_T1_magnetic_K25"
    @test record["analog_config"] == "T1_magnetic_K25"
    @test record["t1r_calibration_csv"] == "t1r_fit_all_dev_T1_magnetic_K25.csv"
    @test record["safeguards"] == false
    @test record["selection_trace_sha256"] == "trace-sha"
    @test record["lat_weights"] == [1.0, 1.0, 1.0, 1.0, 0.75, 0.75]
    @test record["e_layers"][1]["layer"] == "E2"
    @test record["e_layers"][2] === nothing
    @test record["e_layers"][5]["param"] == "1000.0"
    @test record["model_steps"] == collect(V23_MODEL_STEPS)
    @test record["base_table_sha256"] == "table-sha"
    # Every key the pre-Amendment contract carried is still present.
    for key in ("plan", "selected_config", "base_config", "family", "safeguards", "k",
                "params", "mean_rmse_steps_2_3_6_nt", "lat_weights", "e_layers",
                "model_steps", "base_table_sha256", "hourly_frame_sha256", "code_sha256")
        @test haskey(record, key)
    end
    # The confirmatory runner refuses a contract that carries no composition
    # digest, so every contract this stage writes must carry one.
    @test haskey(record, "oof_v2_3_lat_sha256")

    # A non-T1r member records no calibration, so the confirmatory path stays on
    # the deployed correction.
    plain = v23_select_record(ctx, plan, candidate("T1_magnetic_K50_Son", "T1_magnetic_K50",
                                                   "T1", true, 50),
                              (analog_config="T1_magnetic_K50", calibration=nothing),
                              lat, choice, params, "trace-sha")
    @test plain["t1r_calibration_csv"] === nothing
    @test plain["analog_config"] == "T1_magnetic_K50"
    @test plain["safeguards"] == true
end

end # module
