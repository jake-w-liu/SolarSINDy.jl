using Test
using DataFrames
using Dates
using Statistics

isdefined(Main, :V24LiveCalibration) ||
    include(joinpath(@__DIR__, "..", "validation", "operational",
                     "v2_4_live_calibration.jl"))
using .V24LiveCalibration

function _live_calibration_fixture(residuals::Vector{Float64};
                                   identity::String=EXACT_V24_IDENTITY,
                                   step::Int=1)
    base = DateTime(2026, 1, 1)
    n = length(residuals)
    issue = [base + Hour(index - 1) + Minute(30) for index in 1:n]
    latest = [base + Hour(index - 1) for index in 1:n]
    target = [base + Hour(index - 1 + step) for index in 1:n]
    point = fill(-10.0, n)
    observation = point .+ residuals
    return DataFrame(
        issue_time_utc=issue,
        latest_dst_time_utc=latest,
        latest_dst_nt=fill(-10.0, n),
        target_time_utc=target,
        model_step_hours=fill(step, n),
        observation_dst_nt=observation,
        served_pred_dst_nt=point,
        served_pred_dst_ci05_nt=point .- 2.0,
        served_pred_dst_ci95_nt=point .+ 2.0,
        sub_hourly_model_version=fill(identity, n),
        v24_status=fill("ok", n),
        v24_manifest_sha256=fill(EXACT_V24_MANIFEST_SHA256, n),
    )
end

@testset "V2.4e live-calibration order statistics and score" begin
    values = [4.0, 1.0, 3.0, 2.0]
    @test conformal_upper(values, 0.50) == 3.0 # ceil((4 + 1) * 0.5) = 3
    @test conformal_upper(values, 0.90) == 4.0 # clipped finite-sample endpoint
    @test empirical_lower(values, 0.20) == 1.0
    @test empirical_upper(values, 0.80) == 4.0
    @test interval_score(-1.0, 1.0, 0.0) == 2.0
    @test interval_score(-1.0, 1.0, 2.0) == 22.0
    @test_throws ArgumentError conformal_upper([NaN, Inf], 0.9)
    @test_throws ArgumentError interval_score(1.0, -1.0, 0.0)
end

@testset "V2.4e exact identity, retry, and strict-future cohort" begin
    source = _live_calibration_fixture([1.0, 2.0, 3.0, 4.0])
    wrong = copy(source[1:1, :])
    wrong.sub_hourly_model_version .= "v2.1"
    wrong.observation_dst_nt .= 999.0
    nonfuture = copy(source[2:2, :])
    nonfuture.target_time_utc .= nonfuture.latest_dst_time_utc
    retry = copy(source[1:1, :])
    retry.issue_time_utc .+= Minute(10)
    retry.served_pred_dst_nt .= -11.0
    retry.served_pred_dst_ci05_nt .= -13.0
    retry.served_pred_dst_ci95_nt .= -9.0
    retry.observation_dst_nt .= -10.0
    combined = vcat(source, wrong, retry)

    canonical = canonical_live_rows(combined)
    reversed = canonical_live_rows(combined[end:-1:1, :])
    @test nrow(canonical) == 4
    @test canonical.row_key == reversed.row_key
    @test canonical.point_dst_nt == reversed.point_dst_nt
    @test canonical.point_dst_nt[1] == -11.0 # latest retry in the issue hour wins
    @test all(canonical.served_identity .== EXACT_V24_IDENTITY)
    @test all(canonical.target_time_utc .> canonical.latest_dst_time_utc)
    @test_throws ArgumentError canonical_live_rows(vcat(source, nonfuture))

    leaked_anchor = copy(source)
    leaked_anchor.latest_dst_time_utc[1] = leaked_anchor.issue_time_utc[1] + Minute(1)
    @test_throws ArgumentError canonical_live_rows(leaked_anchor)

    wrong_manifest = copy(source)
    wrong_manifest.v24_manifest_sha256[1] = "0"^64
    @test_throws ArgumentError canonical_live_rows(wrong_manifest)

    boolean_step = copy(source)
    boolean_step[!, :model_step_hours] = Any[true, 1, 1, 1]
    @test_throws ArgumentError canonical_live_rows(boolean_step)

    wrong_step = copy(source)
    wrong_step.model_step_hours[1] = 2
    @test_throws ArgumentError canonical_live_rows(wrong_step)

    fractional_step = copy(source)
    fractional_step[!, :model_step_hours] = [1.0 + 1.0e-10, 1.0, 1.0, 1.0]
    @test_throws ArgumentError canonical_live_rows(fractional_step)

    outside_interval = copy(source)
    outside_interval.served_pred_dst_ci95_nt[1] = -11.0
    @test_throws ArgumentError canonical_live_rows(outside_interval)

    corrupt = copy(source)
    corrupt.served_pred_dst_nt[1] = Inf
    @test_throws ArgumentError canonical_live_rows(corrupt)

    bad_status = copy(source)
    bad_status.v24_status[1] = "fallback:test"
    @test_throws ArgumentError canonical_live_rows(bad_status)
end

@testset "V2.4e availability-causal replay" begin
    source = _live_calibration_fixture(collect(1.0:8.0))
    symmetric = CandidateSpec("C2", "fixture-symmetric", :symmetric, 3, 0.0, 2)
    replay = replay_candidates(source; specs=[symmetric], warmup=2)

    issued_at_two = replay[replay.issue_time_utc .== DateTime(2026, 1, 1, 2, 30), :]
    @test nrow(issued_at_two) == 1
    @test issued_at_two.history_n[1] == 2
    @test issued_at_two.shadow_available[1]
    # Only residuals 1 and 2 have targets no later than the issue's observed-Dst cutoff.
    @test issued_at_two.shadow_lo_dst_nt[1] == -12.0
    @test issued_at_two.shadow_hi_dst_nt[1] == -8.0

    mutated = copy(source)
    mutated.observation_dst_nt[end] = 10_000.0
    replay_mutated = replay_candidates(mutated; specs=[symmetric], warmup=2)
    before_last_target = replay.target_time_utc .< source.target_time_utc[end]
    @test isequal(replay.shadow_lo_dst_nt[before_last_target],
                  replay_mutated.shadow_lo_dst_nt[before_last_target])
    @test isequal(replay.shadow_hi_dst_nt[before_last_target],
                  replay_mutated.shadow_hi_dst_nt[before_last_target])

    contamination = _live_calibration_fixture([500.0]; identity="v2.1")
    contaminated = vcat(source, contamination)
    replay_contaminated = replay_candidates(contaminated; specs=[symmetric], warmup=2)
    @test replay.row_key == replay_contaminated.row_key
    @test isequal(replay.shadow_lo_dst_nt, replay_contaminated.shadow_lo_dst_nt)
    @test isequal(replay.shadow_hi_dst_nt, replay_contaminated.shadow_hi_dst_nt)

    duplicated = vcat(source, source[3:3, :])
    replay_duplicated = replay_candidates(duplicated[end:-1:1, :];
                                          specs=[symmetric], warmup=2)
    @test replay.row_key == replay_duplicated.row_key
    @test isequal(replay.shadow_lo_dst_nt, replay_duplicated.shadow_lo_dst_nt)

    asymmetric = copy(source)
    asymmetric.served_pred_dst_ci05_nt .= -12.0
    asymmetric.served_pred_dst_ci95_nt .= -7.0
    transported = CandidateSpec("C5", "fixture-transport", :translated_static,
                                2, 0.0, 5, 1.5)
    transported_replay = replay_candidates(asymmetric; specs=[transported], warmup=2)
    transported_at_two = transported_replay[
        transported_replay.issue_time_utc .== DateTime(2026, 1, 1, 2, 30), :]
    # Median residual location is 1.5; deployed half-widths are 2 and 3 nT.
    @test transported_at_two.location_shift_nt[1] == 1.5
    @test transported_at_two.shadow_lo_dst_nt[1] == -11.5
    @test transported_at_two.shadow_hi_dst_nt[1] == -4.0
end

function _gate_fixture(candidate::String, hits::Int; score_delta::Float64=-1.0,
                       width_ratio::Float64=1.0, simplicity::Int=1,
                       family::String="C1")
    n = 50
    covered = [index <= hits for index in 1:n]
    static_width = fill(10.0, n)
    shadow_width = fill(10.0 * width_ratio, n)
    return DataFrame(
        candidate=fill(candidate, n), candidate_family=fill(family, n),
        method=fill("fixture", n), window=fill(48, n), gamma=fill(0.0, n),
        width_scale=fill(1.0, n), simplicity=fill(simplicity, n),
        model_step_hours=fill(1, n),
        shadow_available=fill(true, n), shadow_covered=covered,
        shadow_lo_dst_nt=fill(-5.0, n), shadow_hi_dst_nt=fill(5.0, n),
        shadow_width_nt=shadow_width,
        shadow_interval_score=fill(10.0 + score_delta, n),
        static_covered=fill(true, n), static_width_nt=static_width,
        static_interval_score=fill(10.0, n),
    )
end

@testset "V2.4e frozen development gates" begin
    passing = _gate_fixture("pass", 45)
    failing_coverage = _gate_fixture("fail-coverage", 40)
    failing_score = _gate_fixture("fail-score", 45; score_delta=0.1)
    failing_width = _gate_fixture("fail-width", 45; width_ratio=1.6)
    tables = summarize_candidates(vcat(passing, failing_coverage,
                                       failing_score, failing_width))
    verdict = Dict(row.candidate => row.gate_pass for row in eachrow(tables.summary))
    @test verdict == Dict("pass" => true, "fail-coverage" => false,
                          "fail-score" => false, "fail-width" => false)
    @test select_shadow_candidate(tables.summary) == "pass"

    # A2 is selected over the diagnostic A1 whenever both pass because A1 cannot
    # satisfy the separately frozen prospective width ceiling.
    a1 = _gate_fixture("a1", 45; score_delta=-2.0, width_ratio=1.50,
                       simplicity=5, family="C5")
    a2 = _gate_fixture("a2", 45; score_delta=-1.0, width_ratio=1.25,
                       simplicity=6, family="C6")
    amended = summarize_candidates(vcat(a1, a2))
    @test select_shadow_candidate(amended.summary) == "a2"

    # A3 is the final gate-aligned family and takes precedence over A2 when both pass.
    a3 = _gate_fixture("a3", 45; score_delta=-0.5, width_ratio=1.24,
                       simplicity=7, family="C7")
    final_amendment = summarize_candidates(vcat(a1, a2, a3))
    @test select_shadow_candidate(final_amendment.summary) == "a3"
end

@testset "Frozen V2.4e development receipt" begin
    snapshot = joinpath(@__DIR__, "..", "validation", "input", "operational",
                        "v2_4_live_calibration_dev_20260824T213044Z.csv")
    @test isfile(snapshot)
    if isfile(snapshot)
        rows = canonical_live_rows(snapshot)
        @test nrow(rows) == 652
        @test minimum(rows.observation_dst_nt) == -47.0
        @test count(value -> value <= -50.0, rows.observation_dst_nt) == 0
        covered = (rows.static_lo_dst_nt .<= rows.observation_dst_nt) .&
                  (rows.observation_dst_nt .<= rows.static_hi_dst_nt)
        @test mean(covered) ≈ 431 / 652
        mktempdir() do output
            result = run_study(
                snapshot, output;
                expected_sha="b8355c7f41dfc3e308fc92f7135f76d8d3417c6cb7b02baee07f528143a516a2",
            )
            @test result.winner == "C7-static-transport-w24-s1.50-others1.20"
            @test result.digest ==
                  "31d67e5077ae6fe69cee133fa07dceec3aa639903a17861d179d3877ed0c21af"
        end
    end
end
