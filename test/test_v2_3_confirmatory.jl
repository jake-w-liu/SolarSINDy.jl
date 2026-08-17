module V23ConfirmatoryTests

# Unit oracles for the confirmatory decision path
# (`validation/operational/v2_3_confirmatory.jl`).
#
# The end-to-end smoke run shows that the pipeline executes; it cannot show that
# the gates encode the preregistered thresholds, because a smoke slice never
# lands on them. These tests place synthetic bootstrap and cell tables exactly on
# each boundary of plan section 6 and check which side the gate falls on, then
# check that the decision states map from the verdicts as the plan defines them.

using Test
using CSV
using DataFrames
using Dates
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational",
                          "v2_3_confirmatory.jl")))

"One bootstrap row with fully controlled numbers."
bootstrap_row(; variant="V2_3_final", comparator, step, gain, lower, p, rmse_comparator=10.0) = (
    variant=variant, comparator=comparator, model_step_hours=step, n=1000,
    rmse_variant_nt=rmse_comparator - gain, rmse_comparator_nt=rmse_comparator,
    gain_nt=gain, lower_nt=lower, p_one_sided=p, n_blocks=40,
)

"One cell-metric row with fully controlled numbers."
metric_row(; cell, step, n=1000, loss=-1.0, bias=0.0) = (
    config="V2_3_final", model_step_hours=step, cell=String(cell), n=n,
    rmse_nt=10.0 + loss, bias_nt=bias, rmse_served_nt=10.0, bias_served_nt=0.0,
    loss_vs_served_nt=loss, fallback_fraction=0.0, family="V2_3", safeguards=true,
    params_json="{}", seconds=0.0,
)

"""
A bootstrap table on which every gate passes comfortably. Individual tests then
perturb one entry and check that exactly the intended gate flips.
"""
function passing_bootstrap(; gain=1.0, lower=0.5, p=1e-4)
    rows = NamedTuple[]
    for step in V23_MODEL_STEPS
        push!(rows, bootstrap_row(; comparator="served_v2_1", step=step, gain=gain,
                                  lower=lower, p=p))
        for comparator in ("persistence", "burton", "burton_full", "obrien")
            push!(rows, bootstrap_row(; comparator=comparator, step=step, gain=3.0,
                                      lower=2.0, p=p))
        end
        push!(rows, bootstrap_row(; comparator="frozen_v2_1", step=step, gain=1.0,
                                  lower=0.5, p=p))
        push!(rows, bootstrap_row(; comparator="direct_gbm", step=step, gain=0.1,
                                  lower=-0.1, p=0.3))
    end
    return rows
end

function passing_metrics()
    rows = NamedTuple[]
    for step in V23_MODEL_STEPS, cell in V23_CELL_NAMES
        push!(rows, metric_row(; cell=cell, step=step, loss=-1.0, bias=1.0))
    end
    return rows
end

const PASSING_INTEGRITY = [("all_forecasts_finite", true, "finite"),
                           ("archive_precedes_queries", true, "ordered")]

@testset "Tier A accuracy gate sits exactly on the preregistered thresholds" begin
    clean = _v23_gate_rows(passing_bootstrap(), passing_metrics(), PASSING_INTEGRITY)
    @test clean.verdicts.A1
    @test clean.verdicts.A2
    @test clean.verdicts.A3
    @test clean.verdicts.A4
    @test clean.verdicts.B1

    # The A1 requirement is max(0.25 nT, 2 % of the comparator RMSE); at a
    # comparator RMSE of 10 nT that is 0.25 nT, and 0.24 must fail.
    replace_row(rows, comparator, step, replacement) = [
        (row.comparator == comparator && row.model_step_hours == step &&
         row.variant == "V2_3_final") ? replacement : row for row in rows
    ]
    at_threshold = replace_row(passing_bootstrap(), "served_v2_1", 6,
                               bootstrap_row(; comparator="served_v2_1", step=6, gain=0.25,
                                             lower=0.01, p=1e-4))
    @test _v23_gate_rows(at_threshold, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    below = replace_row(passing_bootstrap(), "served_v2_1", 6,
                        bootstrap_row(; comparator="served_v2_1", step=6, gain=0.24,
                                      lower=0.01, p=1e-4))
    @test !_v23_gate_rows(below, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    # With a 20 nT comparator the 2 % branch binds and 0.25 nT is no longer enough.
    fraction = replace_row(passing_bootstrap(), "served_v2_1", 4,
                           bootstrap_row(; comparator="served_v2_1", step=4, gain=0.3,
                                         lower=0.01, p=1e-4, rmse_comparator=20.0))
    @test !_v23_gate_rows(fraction, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    # A non-positive lower bound fails even with a large point gain.
    negative_lower = replace_row(passing_bootstrap(), "served_v2_1", 7,
                                 bootstrap_row(; comparator="served_v2_1", step=7, gain=2.0,
                                               lower=0.0, p=1e-4))
    @test !_v23_gate_rows(negative_lower, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    # Non-inferiority at the short leads: the bound is strict at −0.10 nT.
    tolerated = replace_row(passing_bootstrap(), "served_v2_1", 1,
                            bootstrap_row(; comparator="served_v2_1", step=1, gain=0.0,
                                          lower=-0.099, p=0.5))
    @test _v23_gate_rows(tolerated, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    breached = replace_row(passing_bootstrap(), "served_v2_1", 2,
                           bootstrap_row(; comparator="served_v2_1", step=2, gain=0.0,
                                         lower=-0.10, p=0.5))
    @test !_v23_gate_rows(breached, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
    negative_point = replace_row(passing_bootstrap(), "served_v2_1", 2,
                                 bootstrap_row(; comparator="served_v2_1", step=2,
                                               gain=-0.001, lower=0.5, p=1e-4))
    @test !_v23_gate_rows(negative_point, passing_metrics(), PASSING_INTEGRITY).verdicts.A1

    # Holm over the four accuracy steps: four p-values of 0.02 adjust to 0.08,
    # 0.06, 0.04 and 0.02, so the largest two no longer clear the 5 % level.
    marginal = [row.comparator == "served_v2_1" && row.model_step_hours in V23_GATE_A1_STEPS ?
                bootstrap_row(; comparator="served_v2_1", step=row.model_step_hours,
                              gain=1.0, lower=0.5, p=0.02) : row
                for row in passing_bootstrap()]
    @test !_v23_gate_rows(marginal, passing_metrics(), PASSING_INTEGRITY).verdicts.A1
end

@testset "baseline, storm-safety, integrity and SINDy-line gates" begin
    # A3 needs every step of every baseline; one flat step sinks it.
    weak = [row.comparator == "burton" && row.model_step_hours == 3 ?
            bootstrap_row(; comparator="burton", step=3, gain=0.0, lower=-0.1, p=0.4) : row
            for row in passing_bootstrap()]
    @test !_v23_gate_rows(weak, passing_metrics(), PASSING_INTEGRITY).verdicts.A3

    # A2 reads the cell table, not the bootstrap.
    breached = [row.cell == "latest_le_m100" && row.model_step_hours == 4 ?
                metric_row(; cell=:latest_le_m100, step=4, loss=0.60) : row
                for row in passing_metrics()]
    result = _v23_gate_rows(passing_bootstrap(), breached, PASSING_INTEGRITY)
    @test !result.verdicts.A2
    @test result.verdicts.A1                     # the accuracy gate is unaffected

    # A4 is the conjunction of the integrity checks handed to it.
    failed = [("all_forecasts_finite", false, "finite")]
    @test !_v23_gate_rows(passing_bootstrap(), passing_metrics(), failed).verdicts.A4

    # B1 requires 0.25 nT at both steps 6 and 7 against the frozen tail.
    frozen_weak = [row.comparator == "frozen_v2_1" && row.model_step_hours == 7 ?
                   bootstrap_row(; comparator="frozen_v2_1", step=7, gain=0.24, lower=0.1,
                                 p=1e-4) : row for row in passing_bootstrap()]
    @test !_v23_gate_rows(frozen_weak, passing_metrics(), PASSING_INTEGRITY).verdicts.B1

    # B2 is a claim permission, not a pass/fail bar: a deficit beyond 0.25 nT
    # against the direct comparator withdraws the local-superiority claim only.
    outgunned = [row.comparator == "direct_gbm" && row.model_step_hours == 6 ?
                 bootstrap_row(; comparator="direct_gbm", step=6, gain=-0.26, lower=-0.6,
                               p=0.9) : row for row in passing_bootstrap()]
    outgunned_result = _v23_gate_rows(outgunned, passing_metrics(), PASSING_INTEGRITY)
    @test !outgunned_result.verdicts.B2_claim_ok
    @test outgunned_result.verdicts.A1

    # Storm-cell gains are the alternative route into PIVOT.
    @test _v23_gate_rows(passing_bootstrap(), passing_metrics(),
                         PASSING_INTEGRITY).verdicts.storm_cell_gains
    flat = [row.cell == "active_deepening" && row.model_step_hours == 6 ?
            metric_row(; cell=:active_deepening, step=6, loss=0.0) : row
            for row in passing_metrics()]
    @test !_v23_gate_rows(passing_bootstrap(), flat, PASSING_INTEGRITY).verdicts.storm_cell_gains
    # Cells below the minimum size do not count toward the route either way.
    small = [row.cell in ("latest_le_m50", "latest_le_m100", "active_deepening") ?
             metric_row(; cell=Symbol(row.cell), step=row.model_step_hours, n=39, loss=5.0) :
             row for row in passing_metrics()]
    @test !_v23_gate_rows(passing_bootstrap(), small,
                          PASSING_INTEGRITY).verdicts.storm_cell_gains
end

@testset "decision states follow the preregistered map" begin
    verdicts(; A1, A2, A3, A4, B1, storm) = (A1=A1, A2=A2, A3=A3, A4=A4, B1=B1,
                                             B2_claim_ok=true, storm_cell_gains=storm)
    @test _v23_decision(verdicts(; A1=true, A2=true, A3=true, A4=true, B1=false,
                                 storm=false))[1] == "SHADOW_READY"
    @test _v23_decision(verdicts(; A1=false, A2=true, A3=true, A4=true, B1=true,
                                 storm=false))[1] == "PIVOT"
    @test _v23_decision(verdicts(; A1=false, A2=true, A3=true, A4=true, B1=false,
                                 storm=true))[1] == "PIVOT"
    @test _v23_decision(verdicts(; A1=false, A2=true, A3=true, A4=true, B1=false,
                                 storm=false))[1] == "NO_GO"
    # Storm safety is a precondition of the PIVOT route.
    @test _v23_decision(verdicts(; A1=false, A2=false, A3=true, A4=true, B1=true,
                                 storm=true))[1] == "NO_GO"
    # A3 or A4 alone can block Tier A while the PIVOT route stays open.
    @test _v23_decision(verdicts(; A1=true, A2=true, A3=false, A4=true, B1=true,
                                 storm=false))[1] == "PIVOT"
    state, failing = _v23_decision(verdicts(; A1=false, A2=true, A3=false, A4=true,
                                            B1=false, storm=false))
    @test state == "NO_GO"
    @test failing == ["A1", "A3"]
    @test isempty(_v23_decision(verdicts(; A1=true, A2=true, A3=true, A4=true, B1=true,
                                         storm=true))[2])
end

@testset "ablation driver falls back when the boosted tail is selected" begin
    analog = (family="T1", base_config="T1_magnetic_K200",
              params=Dict("weight_set" => "magnetic", "k" => 200, "direct" => false))
    spec = _v23_adc_spec(analog)
    @test spec.weight_set === :magnetic
    @test spec.k == 200
    @test spec.direct == false
    @test spec.source == "T1_magnetic_K200"
    boosted = (family="T2", base_config="T2_d6_r400", params=Dict{String,Any}())
    fallback = _v23_adc_spec(boosted)
    @test fallback.weight_set === :uniform
    @test fallback.k == V23_T1A_K
    @test fallback.direct == false
    @test occursin("preregistered", fallback.source)
end

@testset "forced-rerun reason is parsed and empty reasons are rejected" begin
    @test _v23_force_reason(["--confirm"]) === nothing
    @test _v23_force_reason(["--confirm", "--force-rerun-with-reason=base table rebuilt"]) ==
          "base table rebuilt"
    @test _v23_force_reason(["--force-rerun-with-reason=a=b"]) == "a=b"
    @test_throws ErrorException _v23_force_reason(["--force-rerun-with-reason=  "])
end

@testset "direct comparator is chosen per model step from the development scores" begin
    directory = mktempdir()
    function write_summary(depth, rounds, rmse_by_step)
        id = v23_direct_id(depth, rounds)
        rows = [(config="$(id)_Son", model_step_hours=step, cell="all", n=100,
                 rmse_nt=rmse_by_step[step]) for step in V23_MODEL_STEPS]
        # A decoy safeguard variant and a decoy cell must not be read.
        append!(rows, [(config="$(id)_Soff", model_step_hours=step, cell="all", n=100,
                        rmse_nt=0.0) for step in V23_MODEL_STEPS])
        append!(rows, [(config="$(id)_Son", model_step_hours=step, cell="latest_le_m50",
                        n=100, rmse_nt=0.0) for step in V23_MODEL_STEPS])
        CSV.write(joinpath(directory, "summary_$(id).csv"), DataFrame(rows))
    end
    shallow = Dict(1 => 4.0, 2 => 6.0, 3 => 8.0, 4 => 9.0, 6 => 11.0, 7 => 12.0)
    deep = Dict(1 => 5.0, 2 => 5.5, 3 => 8.5, 4 => 8.5, 6 => 11.5, 7 => 11.0)
    write_summary(4, 200, shallow)
    write_summary(6, 400, deep)
    chosen = _v23_select_direct_gbm(directory, [(4, 200), (6, 400)])
    @test chosen[1] == v23_direct_id(4, 200)
    @test chosen[2] == v23_direct_id(6, 400)
    @test chosen[3] == v23_direct_id(4, 200)
    @test chosen[4] == v23_direct_id(6, 400)
    @test chosen[6] == v23_direct_id(4, 200)
    @test chosen[7] == v23_direct_id(6, 400)
    @test_throws ErrorException _v23_select_direct_gbm(directory, [(4, 400)])
end

"""
A development directory holding the two artifacts the input contract is digested
against, plus a builder for contracts whose digests are correct by construction.
"""
function contract_fixture()
    directory = mktempdir()
    calibration = "t1r_fit_all_dev_T1_magnetic_K25.csv"
    touch(joinpath(directory, calibration))
    trace_path = joinpath(directory, "selection_trace.csv")
    write(trace_path, "rank,config,selected\n1,T1r_T1_magnetic_K25_Soff,true\n")
    composition_path = joinpath(directory, "oof_V2_3_LAT.csv")
    write(composition_path, "issue_time_utc,model_step_hours,center_s_on_dst_nt\n" *
                            "2019-01-01T00:00:00,1,-12.5\n")
    base_record(; overrides...) = merge(Dict{String,Any}(
        "plan" => "dev",
        "selected_config" => "T1r_T1_magnetic_K25_Soff",
        "base_config" => "T1r_T1_magnetic_K25",
        "family" => "T1r",
        "analog_config" => "T1_magnetic_K25",
        "t1r_calibration_csv" => calibration,
        "safeguards" => false,
        "k" => 25,
        "params" => Dict("weight_set" => "magnetic", "k" => 25, "direct" => false),
        "mean_rmse_steps_2_3_6_nt" => 7.305,
        "lat_weights" => [1.0, 1.0, 1.0, 1.0, 0.75, 0.75],
        "e_layers" => Any[Dict("layer" => "E2", "param" => "(3, 128)", "gain_nt" => 0.1),
                          nothing, nothing, nothing,
                          Dict("layer" => "E1", "param" => "1000.0", "gain_nt" => 0.01),
                          nothing],
        "model_steps" => collect(V23_MODEL_STEPS),
        "base_table_sha256" => "table-sha",
        "hourly_frame_sha256" => "frame-sha",
        "code_sha256" => "code-sha",
        "selection_trace_sha256" => _v23_sha256_file(trace_path),
        "oof_v2_3_lat_sha256" => _v23_sha256_file(composition_path),
    ), Dict{String,Any}(String(k) => v for (k, v) in overrides))
    write_record(record) = write(joinpath(directory, "selected_configuration.json"),
                                 JSON3.write(record))
    return (directory=directory, calibration=calibration, trace=trace_path,
            composition=composition_path, base_record=base_record,
            write_record=write_record)
end

@testset "the development contract carries the Amendment A1 fields to the TEST run" begin
    fixture = contract_fixture()
    directory = fixture.directory

    fixture.write_record(fixture.base_record())
    selection = _v23_read_selection(directory, "table-sha")
    @test selection.family == "T1r"
    @test selection.base_config == "T1r_T1_magnetic_K25"
    @test selection.analog_config == "T1_magnetic_K25"
    @test selection.t1r_calibration_csv == fixture.calibration
    @test selection.selection_trace_sha == _v23_sha256_file(fixture.trace)
    @test selection.dev_composition_sha == _v23_sha256_file(fixture.composition)
    @test isempty(selection.drift_accepted)
    @test selection.safeguards == false
    @test selection.lat_weights == [1.0, 1.0, 1.0, 1.0, 0.75, 0.75]
    @test selection.e_layers[1].kind === :E2
    @test selection.e_layers[1].param == (V23_E2_DEPTH, V23_E2_ROUNDS)
    @test selection.e_layers[1].gain_nt == 0.1
    @test selection.e_layers[2] === nothing
    @test selection.e_layers[5].kind === :E1
    @test selection.e_layers[5].param == 1000.0

    # The ablation driver of a T1r member is the analog core it corrects.
    spec = _v23_adc_spec(selection)
    @test spec.weight_set === :magnetic
    @test spec.k == 25
    @test spec.direct == false
    @test spec.source == "T1_magnetic_K25"

    # A T1r selection that cannot be reproduced on TEST is refused, not guessed.
    fixture.write_record(fixture.base_record(; t1r_calibration_csv=nothing))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    fixture.write_record(fixture.base_record(; analog_config="T1r_T1_magnetic_K25"))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    fixture.write_record(fixture.base_record(; analog_config="T1_uniform_K200"))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    fixture.write_record(fixture.base_record(; t1r_calibration_csv="t1r_fit_all_dev_absent.csv"))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    # The base-table guard still binds.
    fixture.write_record(fixture.base_record())
    @test_throws ErrorException _v23_read_selection(directory, "another-sha")
end

@testset "a contract without the selection digests is refused, not read as legacy" begin
    fixture = contract_fixture()
    directory = fixture.directory

    # A pre-Amendment contract names a selection made over a smaller family than
    # Amendment A1 preregistered, and a rerun of v2_3_development.jl is exactly
    # how one reappears. It must never be scored on TEST.
    legacy = fixture.base_record(; family="T1", base_config="T1_magnetic_K50",
                                 selected_config="T1_magnetic_K50_Son", safeguards=true)
    delete!(legacy, "analog_config")
    delete!(legacy, "t1r_calibration_csv")
    delete!(legacy, "selection_trace_sha256")
    delete!(legacy, "oof_v2_3_lat_sha256")
    fixture.write_record(legacy)
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")

    # Present but wrong is equally fatal: the trace on disk is not the trace the
    # contract was written from.
    fixture.write_record(fixture.base_record(; selection_trace_sha256="0" ^ 64))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    fixture.write_record(fixture.base_record(; oof_v2_3_lat_sha256="0" ^ 64))
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
    # Mutating the composition after the contract was written is the failure
    # mode of a development rerun that died between the two writes.
    fixture.write_record(fixture.base_record())
    write(fixture.composition, "issue_time_utc,model_step_hours,center_s_on_dst_nt\n" *
                               "2019-01-01T00:00:00,1,-99.0\n")
    @test_throws ErrorException _v23_read_selection(directory, "table-sha")
end

@testset "code and frame drift are refused unless a written reason is given" begin
    fixture = contract_fixture()
    directory = fixture.directory
    fixture.write_record(fixture.base_record())

    # Matching digests pass.
    ok = _v23_read_selection(directory, "table-sha"; frame_sha="frame-sha",
                             code_sha="code-sha")
    @test isempty(ok.drift_accepted)

    @test_throws ErrorException _v23_read_selection(directory, "table-sha";
                                                    code_sha="a-different-code-sha")
    @test_throws ErrorException _v23_read_selection(directory, "table-sha";
                                                    frame_sha="a-different-frame-sha")

    accepted = _v23_read_selection(directory, "table-sha";
                                   frame_sha="a-different-frame-sha",
                                   code_sha="a-different-code-sha",
                                   accept_code_drift="rebuilt frame, verified identical rows")
    @test length(accepted.drift_accepted) == 2
    @test any(occursin("code_sha256", message) for message in accepted.drift_accepted)
    @test any(occursin("hourly_frame_sha256", message) for message in accepted.drift_accepted)

    # The override is a reason for a *mismatch*, never a licence to run without
    # the digests at all.
    missing_key = fixture.base_record()
    delete!(missing_key, "oof_v2_3_lat_sha256")
    fixture.write_record(missing_key)
    @test_throws ErrorException _v23_read_selection(directory, "table-sha";
                                                    accept_code_drift="any reason")
end

@testset "an in-run integrity failure aborts before a decision exists" begin
    manifest = V23Manifest()
    # The frozen-feature oracle decides what the refit layer is applied to, so a
    # failure must leave the run, not become a failed gate on a scored decision.
    V23_T1R_ORACLE_HOOK[] = _ -> error("frozen feature drifted by 4.2 nT")
    try
        @test_throws ErrorException _v23_t1r_oracle_or_fail(DataFrame(), manifest)
    finally
        V23_T1R_ORACLE_HOOK[] = nothing
    end
    failure = [row for row in manifest.rows
               if row.name == "t1r_feature_oracle_max_abs_nt"]
    @test length(failure) == 1
    @test failure[1].value == "FAILED"

    # With the hook cleared the recorded worst deviation is the oracle's own.
    clean = V23Manifest()
    V23_T1R_ORACLE_HOOK[] = _ -> (worst_nt=0.0, column=:latest_dst_nt)
    try
        @test _v23_t1r_oracle_or_fail(DataFrame(), clean).worst_nt == 0.0
    finally
        V23_T1R_ORACLE_HOOK[] = nothing
    end

    # Every A4 item passing is the only state that lets a decision be written.
    @test _v23_assert_integrity(PASSING_INTEGRITY)
    breached = vcat(PASSING_INTEGRITY,
                    [("t1r_frozen_feature_oracle", false, "oracle reproduces the base table")])
    @test_throws ErrorException _v23_assert_integrity(breached)
    @test_throws ErrorException _v23_assert_integrity(
        [("tail_mutation_invariance", false, "the tail cannot see past the L1 window")])
end

@testset "the T1r layer over the T1a ablation is labelled, never a candidate" begin
    @test _v23_preregistered_variant("T1r_T1_magnetic_K25_Soff")
    @test _v23_preregistered_variant("T1a_uniform_K100_Son")
    @test _v23_preregistered_variant("V2_3_final")
    @test !_v23_preregistered_variant("T1r_T1a_uniform_K100_Son")
    @test !_v23_preregistered_variant("T1r_T1a_uniform_K100_Soff")

    labelled = _v23_gate_rows(passing_bootstrap(), passing_metrics(), PASSING_INTEGRITY,
                              ["T1r_T1a_uniform_K100_Son", "T1r_T1a_uniform_K100_Soff"])
    row = only([r for r in labelled.rows if r.gate == "ABLATION_LABEL"])
    @test row.family == "not_preregistered"
    @test occursin("T1r_T1a_uniform_K100_Son", row.observed)
    @test row.pass
    # Labelling changes no verdict.
    @test labelled.verdicts == _v23_gate_rows(passing_bootstrap(), passing_metrics(),
                                              PASSING_INTEGRITY).verdicts
end

@testset "the forced-rerun log separates intent from completion" begin
    directory = mktempdir()
    plan = V23RunPlan("test", directory, false, Int[], 0, collect(V23_WEIGHT_SETS),
                      collect(V23_T1_K_GRID), collect(V23_GDC_GRID), collect(V23_DIRECT_GRID))
    decision_path = joinpath(directory, "decision.csv")
    CSV.write(decision_path, DataFrame(decision_state=["NO_GO"], failing_gates=["A1"]))

    _v23_append_rerun_log(plan, "recomputed after a corrected base table";
                          stage="intent", decision_path=decision_path)
    log = CSV.read(joinpath(directory, "rerun_log.csv"), DataFrame)
    @test nrow(log) == 1
    @test log.stage[1] == "intent"
    @test occursin("NO_GO", log.overwritten_decision[1])
    @test log.replacement_decision[1] === missing || log.replacement_decision[1] == ""

    # A forced run that dies before writing a decision leaves the intent alone,
    # which is what distinguishes an attempt from an overwrite.
    _v23_append_rerun_log(plan, "recomputed after a corrected base table";
                          stage="completed", replacement="SHADOW_READY|")
    log = CSV.read(joinpath(directory, "rerun_log.csv"), DataFrame)
    @test nrow(log) == 2
    @test log.stage[2] == "completed"
    @test log.replacement_decision[2] == "SHADOW_READY|"
end

@testset "a written reason is parsed from the flag or refused" begin
    @test _v23_written_reason(["--accept-code-drift=frame rebuilt"],
                              "--accept-code-drift") == "frame rebuilt"
    @test _v23_written_reason(String[], "--accept-code-drift") === nothing
    @test_throws ErrorException _v23_written_reason(["--accept-code-drift"],
                                                    "--accept-code-drift")
    @test_throws ErrorException _v23_written_reason(["--accept-code-drift="],
                                                    "--accept-code-drift")
end

@testset "T1r variants are discovered from the persisted all-DEV calibrations" begin
    directory = mktempdir()
    for name in ("t1r_fit_all_dev_T1_magnetic_K25.csv", "t1r_fit_all_dev_T1_uniform_K200.csv",
                 "t1r_fit_all_dev_T1a_uniform_K100.csv")
        touch(joinpath(directory, name))
    end
    # Neighbouring artifacts of the same stage must not be mistaken for a
    # configuration.
    for name in ("t1r_manifest.csv", "t1r_dev_report.csv", "oof_T1r_T1_magnetic_K25.csv",
                 "summary_T1_magnetic_K25.csv", "t1r_fit_all_dev_T1_uniform_K200.txt")
        touch(joinpath(directory, name))
    end
    @test _v23_t1r_configs(directory) ==
          ["T1_magnetic_K25", "T1_uniform_K200", "T1a_uniform_K100"]
    @test _v23_t1r_configs(joinpath(directory, "absent")) == String[]
end

"""
Base-table-shaped rows in the schema `v23_t1r_base_table_columns` selects: the
issue-time state, the baseline panel, the frozen raw core, the partition label and
the archived calibration feature columns produced by the package's own feature
path.
"""
function synthetic_partitioned_table(issues::Int; test_from::Int)
    rows = NamedTuple[]
    start = DateTime(2019, 4, 1)
    for i in 0:issues - 1
        t = start + Hour(i)
        latest = -30.0 + 40.0 * sin(2π * i / 47) - 9.0 * cos(2π * i / 13)
        speed = 390.0 + 150.0 * sin(2π * i / 31)
        bz = -7.0 * sin(2π * i / 19) + 2.0 * cos(2π * i / 7)
        by = 3.5 * cos(2π * i / 23)
        density = 5.0 + 2.5 * sin(2π * i / 17)
        pdyn = dynamic_pressure(density, speed)
        for h in V23_MODEL_STEPS
            core = latest - 3.0 * sin(2π * (i + 2h) / 37) - 0.5 * h
            push!(rows, (
                issue_time_utc=t, model_step_hours=h,
                partition=(i + 1 >= test_from ? "TEST" : "DEV"),
                latest_dst_nt=latest, V_kms=speed, Bz_nt=bz, By_nt=by, n_cm3=density,
                Pdyn_npa=pdyn, persistence_dst_nt=latest,
                burton_dst_nt=latest - 1.2 * h * max(-bz, 0.0) / 6,
                burton_full_dst_nt=latest - 1.0 * h * max(-bz, 0.0) / 6,
                obrien_dst_nt=latest - 0.8 * h * max(-bz, 0.0) / 6,
                observation_dst_nt=core + 2.0 * cos(2π * (i + h) / 11),
                pred_dst_nt=core, pred_dst_ci05_nt=core - 8.0, pred_dst_ci95_nt=core + 8.0,
            ))
        end
    end
    frame = DataFrame(rows)
    SolarSINDy.add_operational_v2_features!(frame)
    return frame
end

"Anchor view whose issues match a synthetic base table row for row."
function table_anchors(issues::Vector{DateTime}; served)
    count = length(issues)
    index = Dict(t => i for (i, t) in enumerate(issues))
    driver = [(V=400.0, Bz=-5.0, By=2.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
              for _ in 1:count]
    zeros_matrix() = zeros(count, V23_STEP_COUNT)
    return V23Anchors(
        issues, index, fill("DEV", count), fill(2018, count), driver,
        fill(-10.0, count), zeros(count), zeros(count), fill(-30.0, count),
        trues(count, V23_STEP_COUNT), zeros_matrix(), served, zeros_matrix(),
        zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(),
        zeros_matrix(), zeros_matrix(),
    )
end

function table_context(anchors::V23Anchors)
    plan = v23_test_plan(; smoke=true)
    count = length(anchors)
    return V23Context(plan, anchors, DataFrame(), Dict{DateTime,_V23_DRIVER_NT}(),
                      nothing, nothing, zeros(count, V23_FEATURE_COUNT), trues(count),
                      trues(count), v23_cell_masks(anchors), nothing, "t", "f", "c",
                      "unit", 0.0)
end

@testset "the T1r row view aligns base-table rows with the scored anchors" begin
    directory = mktempdir()
    path = joinpath(directory, "base.csv")
    frame = synthetic_partitioned_table(30; test_from=21)
    CSV.write(path, frame)
    issues = unique(frame.issue_time_utc)
    @test length(issues) == 30
    anchors = table_anchors(collect(issues); served=fill(-6.0, 30, V23_STEP_COUNT))
    ctx = table_context(anchors)
    rows = collect(21:30)

    view = _v23_t1r_row_view(ctx, rows; table_path=path, partitions=("TEST",))
    @test view.anchor == [i for i in rows for _ in 1:V23_STEP_COUNT]
    @test view.slot == [s for _ in rows for s in 1:V23_STEP_COUNT]
    @test view.frame.issue_time_utc == [anchors.issue[i] for i in rows
                                        for _ in 1:V23_STEP_COUNT]
    @test view.frame.model_step_hours == [h for _ in rows for h in V23_MODEL_STEPS]
    @test nrow(view.frame) == length(rows) * V23_STEP_COUNT
    @test Set(String.(propertynames(view.frame))) == Set(v23_t1r_base_table_columns())
    # The archived feature columns survive the selection, so the frozen-feature
    # oracle stays available on the scored partition.
    @test v23_t1r_feature_oracle(view.frame).worst_nt <= 1e-9

    # A target the base table does not hold is not scored.
    anchors.present[25, 3] = false
    reduced = _v23_t1r_row_view(ctx, rows; table_path=path, partitions=("TEST",))
    @test nrow(reduced.frame) == length(rows) * V23_STEP_COUNT - 1
    @test !any(reduced.anchor[j] == 25 && reduced.slot[j] == 3 for j in eachindex(reduced.anchor))
    anchors.present[25, 3] = true

    # Reading the wrong partition leaves the scored rows unmatched, which fails
    # closed rather than correcting a row whose archived state is unknown.
    @test_throws ErrorException _v23_t1r_row_view(ctx, rows; table_path=path,
                                                  partitions=("DEV",))
    @test_throws ErrorException _v23_t1r_row_view(ctx, rows; table_path=path,
                                                  partitions=("NOWHERE",))
    @test_throws ErrorException _v23_t1r_row_view(ctx, rows;
                                                  table_path=joinpath(directory, "absent.csv"),
                                                  partitions=("TEST",))
end

@testset "T1r centers correct the analog core and serve the fallback anchors" begin
    directory = mktempdir()
    path = joinpath(directory, "base.csv")
    frame = synthetic_partitioned_table(24; test_from=13)
    CSV.write(path, frame)
    issues = collect(unique(frame.issue_time_utc))
    served = fill(-6.0, 24, V23_STEP_COUNT)
    anchors = table_anchors(issues; served=served)
    ctx = table_context(anchors)
    rows = collect(13:24)
    view = _v23_t1r_row_view(ctx, rows; table_path=path, partitions=("TEST",))

    raw = fill(NaN, 24, V23_STEP_COUNT)
    for i in rows, slot in 1:V23_STEP_COUNT
        raw[i, slot] = -40.0 - i - 2.0 * slot
    end
    fallback = [i <= 15 for i in 1:24]
    deployed_names = copy(v23_t1r_deployed_calibration().feature_names)
    zeroed = default_operational_v2_calibration(feature_names=deployed_names,
                                                label="zero_correction")
    @test all(iszero, zeroed.coefficients)
    variant = _v23_t1r_variant_centers(ctx, view, raw, fallback, zeroed)
    for i in rows, slot in 1:V23_STEP_COUNT
        if fallback[i]
            @test variant.off[i, slot] == served[i, slot]
            @test variant.on[i, slot] == served[i, slot]
        else
            @test variant.off[i, slot] == clamp(raw[i, slot], -2000.0, 50.0)
        end
    end
    # Anchors outside the scored set are untouched.
    @test all(isnan, variant.off[1:12, :])

    # A nonzero intercept shifts every corrected center by exactly that amount,
    # so the correction is genuinely applied and not dropped.
    shifted = OperationalV2Calibration(
        copy(zeroed.feature_names), copy(zeroed.feature_mean), copy(zeroed.feature_scale),
        vcat([2.5], zeros(length(deployed_names))), 1.0, "intercept_only",
    )
    moved = _v23_t1r_variant_centers(ctx, view, raw, fallback, shifted)
    for i in 16:24, slot in 1:V23_STEP_COUNT
        @test isapprox(moved.off[i, slot], clamp(raw[i, slot] + 2.5, -2000.0, 50.0);
                       atol=1e-12)
    end
    @test all(moved.off[13:15, :] .== served[13:15, :])

    # The center follows the analog raw core it is handed; if it did not, the
    # Amendment A1 comparison would be vacuous.
    perturbed = copy(raw)
    perturbed[20, 4] += 11.0
    after = _v23_t1r_variant_centers(ctx, view, perturbed, fallback, zeroed)
    @test after.off[20, 4] != variant.off[20, 4]
    @test after.off[19, 4] == variant.off[19, 4]
end

@testset "climatology timescale recovers a known exponential recovery" begin
    tau = 45.0
    hours = 900
    start = DateTime(2012, 1, 1)
    times = [start + Hour(i) for i in 0:hours - 1]
    # Three consecutive recoveries from −120 nT with the same decay constant, so
    # the estimator sees several hundred qualifying hours.
    dst = [-120.0 * exp(-(mod(i, 300)) / tau) for i in 0:hours - 1]
    frame = DataFrame(time_utc=times, V=fill(400.0, hours), Bz=fill(-2.0, hours),
                      By=fill(1.0, hours), n=fill(5.0, hours),
                      Pdyn=fill(dynamic_pressure(5.0, 400.0), hours), Dst=dst)
    # The anchors span the frame, because the estimator only looks at recovery
    # hours up to the last scored issue.
    issues = times[[1, 300, 600, hours]]
    count_anchors = length(issues)
    anchors = V23Anchors(
        collect(issues), Dict(t => i for (i, t) in enumerate(issues)),
        fill("DEV", count_anchors), fill(2015, count_anchors),
        [(V=400.0, Bz=-2.0, By=1.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
         for _ in 1:count_anchors],
        fill(-50.0, count_anchors), zeros(count_anchors), zeros(count_anchors),
        fill(-40.0, count_anchors), trues(count_anchors, V23_STEP_COUNT),
        (zeros(count_anchors, V23_STEP_COUNT) for _ in 1:10)...,
    )
    plan = v23_test_plan(; smoke=true)
    ctx = V23Context(plan, anchors, frame, Dict{DateTime,_V23_DRIVER_NT}(), nothing, nothing,
                     zeros(count_anchors, V23_FEATURE_COUNT), trues(count_anchors),
                     trues(count_anchors), v23_cell_masks(anchors), nothing, "t", "f", "c",
                     "unit", 0.0)
    # The estimator only sees hours whose Dst is at or below −20 nT, but the decay
    # constant of an exact exponential is the same on every such hour.
    @test isapprox(_v23_climatology_tau(ctx, collect(1:count_anchors)), tau; rtol=1e-9)
    centers = _v23_climatology_centers(ctx, collect(1:count_anchors), tau)
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        @test centers[1, slot] ≈ -50.0 * exp(-step / tau)
    end
end

@testset "the persisted E1 layer carries everything needed to reapply it" begin
    # A deployment artifact is only useful if the correction can be rebuilt from
    # it alone: standardisation, slopes, intercept and the step's cap.
    design = [1.0 2.0; 2.0 1.0; 3.0 5.0; 4.0 3.0; 5.0 8.0; 6.0 4.0; 7.0 9.0; 8.0 6.0]
    target = 1.5 .* design[:, 1] .- 0.5 .* design[:, 2] .+ 2.0
    model = v23_ridge_fit(design, target, 10.0)
    directory = mktempdir()
    path = joinpath(directory, "e1_step6.csv")
    _v23_write_e1_artifact(path, model, ["innovation_1h", "innovation_2h"], v23_e_cap(6))
    table = CSV.read(path, DataFrame)
    @test table.feature_name == ["innovation_1h", "innovation_2h"]
    @test table.feature_mean ≈ model.mean
    @test table.feature_scale ≈ model.sd
    @test table.standardised_coefficient ≈ model.beta
    @test all(table.intercept_nt .≈ model.intercept)
    @test all(table.ridge_lambda .== 10.0)
    @test all(table.correction_cap_nt .== v23_e_cap(6))

    # Rebuilding the prediction from the artifact alone reproduces the model.
    query = [2.5 3.5; 6.5 7.5]
    rebuilt = ((query .- transpose(table.feature_mean)) ./ transpose(table.feature_scale)) *
              table.standardised_coefficient .+ table.intercept_nt[1]
    @test rebuilt ≈ v23_ridge_predict(model, query)

    # A name list that does not match the fitted width is a corrupt artifact.
    @test_throws ErrorException _v23_write_e1_artifact(path, model, ["only_one"],
                                                       v23_e_cap(6))
end

@testset "the fit/score embargo is measured in targets, not in ordering" begin
    start = DateTime(2019, 1, 1)
    issues = [start + Hour(i) for i in 0:399]
    anchors = table_anchors(issues; served=zeros(length(issues), V23_STEP_COUNT))
    train = collect(1:100)                                    # last issue index 100
    # The last target a fitting anchor carries is its issue plus the longest
    # model step, so the embargo starts there and not at the last issue.
    last_target_index = 100 + V23_MAX_STEP
    exact = [last_target_index + V23_BASE_EMBARGO_HOURS]
    @test _v23_assert_embargo(anchors, train, exact) == V23_BASE_EMBARGO_HOURS
    @test _v23_assert_embargo(anchors, train, [last_target_index + 200]) == 200

    # One hour short of the embargo fails, although the archive still strictly
    # precedes every query and the old ordering check would have passed.
    short = [last_target_index + V23_BASE_EMBARGO_HOURS - 1]
    @test anchors.issue[maximum(train)] < anchors.issue[short[1]]
    @test_throws ErrorException _v23_assert_embargo(anchors, train, short)
    # The 7 h between the last fitting issue and its last target is exactly the
    # window an ordering check cannot see.
    @test_throws ErrorException _v23_assert_embargo(
        anchors, train, [100 + V23_BASE_EMBARGO_HOURS])
    @test_throws ErrorException _v23_assert_embargo(anchors, Int[], [200])
end

end # module
