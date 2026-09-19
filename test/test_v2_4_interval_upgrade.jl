module V24IntervalUpgradeTests

using Test, CSV, DataFrames, Dates, Random, SHA
include(joinpath(@__DIR__, "..", "validation", "operational", "v2_4_interval_upgrade.jl"))
const U = V24IntervalUpgrade

function fixture(; n=12, steps=(1,))
    base = DateTime(2026, 1, 1)
    records = NamedTuple[]
    for index in 0:n-1, step in steps
        push!(records, (
            issue_time_utc=base + Hour(index) + Minute(30),
            latest_dst_time_utc=base + Hour(index), latest_dst_nt=2.0index,
            target_time_utc=base + Hour(index + step), model_step_hours=step,
            served_pred_dst_nt=0.0, served_pred_dst_ci05_nt=-2.0,
            served_pred_dst_ci95_nt=4.0, observation_dst_nt=2.0(index + step),
            sub_hourly_model_version=U.V24LiveCalibration.EXACT_V24_IDENTITY,
            v24_status="ok", v24_manifest_sha256=U.V24LiveCalibration.EXACT_V24_MANIFEST_SHA256,
            v24_cal_shadow_model_version=U.A3_ID, v24_cal_shadow_config_sha256=U.A3_SHA,
            v24_cal_shadow_status="ok", v24_cal_shadow_ci05_nt=-3.0,
            v24_cal_shadow_ci95_nt=5.0, dst_delta_1h_nt=2.0,
            VBsouth_mvm=0.0, driver_data_gap=false,
        ))
    end
    return DataFrame(records)
end

function band(result, candidate, hour; step=1)
    rows = result.replay
    return only(eachrow(rows[(rows.candidate .== candidate) .&
        (rows.issue_time_utc .== DateTime(2026, 1, 1, hour, 30)) .&
        (rows.model_step_hours .== step), :]))
end

@testset verbose=true "Fixed-grid interval arithmetic" begin
    # The history median is 3; static left/right half-widths are 2 and 4.
    for candidate in ("L24", "L48")
        value = U.candidate_interval(candidate, 0, -2, 4, [2, 4], Float64[]; warmup=2)
        @test (value.lo, value.hi) == (0.5, 8.0)
        @test value.location == 3.0
    end
    for candidate in ("T48", "T96")
        value = U.candidate_interval(candidate, 0, -2, 4, [4, -3, 2, 8], []; warmup=2)
        # floor(5*.05) clips to the minimum; ceil(5*.95) clips to the maximum.
        @test (value.lo, value.hi) == (-3.0, 8.0)
    end
    for candidate in ("S48", "S96")
        value = U.candidate_interval(candidate, 0, -2, 4, [2, 4], [0.5, 2.0]; warmup=2)
        @test (value.lo, value.hi) == (-1.0, 11.0) # q=2; retain unequal half-widths.
        @test !U.candidate_interval(candidate, 0, -2, 4, [2, 4], [0.5]; warmup=2).available
    end
    for candidate in U.CANDIDATES
        @test !U.candidate_interval(candidate, 0, -2, 4, [1], []; warmup=2).available
        @test_throws ArgumentError U.candidate_interval(candidate, 0, 0, 4, [1, 2], [1, 2]; warmup=2)
        @test_throws ArgumentError U.candidate_interval(candidate, 0, -2, 4, [1, NaN], [1, 2]; warmup=2)
        @test_throws ArgumentError U.candidate_interval(candidate, 0, -2, 4, [1, 2], [-1, 2]; warmup=2)
        @test_throws ArgumentError U.candidate_interval(candidate, 0, -2, 4, [1, 2], [1, Inf]; warmup=2)
    end
    @test_throws ArgumentError U.candidate_interval("unknown", 0, -2, 4, [1, 2], [1, 2])
    @test_throws ArgumentError U.candidate_interval("L24", 0, -2, 4, [1, 2], []; warmup=0)
    # Exact ranks on nonconstant windows catch an off-by-one tail or wrong window.
    large = collect(1.0:100.0)
    expected = Dict("L24"=>(86.0, 93.5), "L48"=>(74.0, 81.5),
        "S48"=>(-105.5, 476.5), "S96"=>(-95.5, 456.5),
        "T48"=>(54.0, 99.0), "T96"=>(8.0, 97.0))
    for candidate in U.CANDIDATES
        value = U.candidate_interval(candidate, 0, -2, 4, large, large)
        @test (value.lo, value.hi) == expected[candidate]
    end
    for seed in 1:5
        rng = MersenneTwister(seed)
        residuals = randn(rng, 120)
        scores = abs.(randn(rng, 120))
        for candidate in U.CANDIDATES
            original = U.candidate_interval(candidate, 2, -1, 6, residuals, scores)
            scaled = U.candidate_interval(candidate, 6, -3, 18, 3 .* residuals, scores)
            # Fixed positive unit conversion commutes; roundoff only (no fitted solver).
            @test isapprox(scaled.lo, 3original.lo; atol=1e-12, rtol=1e-14)
            @test isapprox(scaled.hi, 3original.hi; atol=1e-12, rtol=1e-14)
        end
    end
end

@testset verbose=true "Witness-time replay and independent expectations" begin
    input = fixture()
    original_input = copy(input)
    result = U.replay_upgrade(input; warmup=2)
    @test isequal(input, original_input) # Input/source archive is immutable.
    @test nrow(result.replay) == 12 * 8
    @test length(result.witnesses) == 12
    @test (band(result, "L24", 2).lower_dst_nt, band(result, "L24", 2).upper_dst_nt) == (0.5, 8.0)
    @test (band(result, "T48", 2).lower_dst_nt, band(result, "T48", 2).upper_dst_nt) == (2.0, 4.0)
    @test !band(result, "S48", 3).available # Only one issued translated center has matured.
    # Prior translated errors are (6-3)/4=.75 and (8-4)/4=1; current median=5.
    @test (band(result, "S48", 4).lower_dst_nt, band(result, "S48", 4).upper_dst_nt) == (3.0, 9.0)
    @test band(result, "static", 2).upper_dst_nt == 4.0
    @test band(result, "issued_A3", 2).upper_dst_nt == 5.0
    @test band(result, "static", 2).interval_score_nt == 46.0 # width 6 + 20*(6-4).
    @test band(result, "issued_A3", 2).interval_score_nt == 28.0 # width 8 + 20*(6-5).
    @test all(==(0.0), result.replay.point_dst_nt)

    revised_final = copy(input)
    revised_final.observation_dst_nt .+= 1000.0
    changed = U.replay_upgrade(revised_final; warmup=2)
    @test isequal(result.replay[:, [:lower_dst_nt, :upper_dst_nt]],
                  changed.replay[:, [:lower_dst_nt, :upper_dst_nt]])
    @test result.replay.interval_score_nt != changed.replay.interval_score_nt

    future = copy(input)
    future.latest_dst_nt[6:end] .+= 100.0
    future_result = U.replay_upgrade(future; warmup=2)
    before = result.replay.issue_time_utc .< DateTime(2026, 1, 1, 5)
    @test isequal(result.replay[before, [:lower_dst_nt, :upper_dst_nt]],
                  future_result.replay[before, [:lower_dst_nt, :upper_dst_nt]])

    pending = copy(input)
    allowmissing!(pending, :observation_dst_nt)
    pending.observation_dst_nt[1] = missing
    pending_result = U.replay_upgrade(vcat(pending, pending[1:1, :]); warmup=2)
    @test nrow(pending_result.rows) == nrow(input) # Pending duplicate remains a single forecast.
    @test !band(pending_result, "static", 0).scored
    @test band(pending_result, "L24", 2).lower_dst_nt == 0.5 # A witnessed target can calibrate even if its final score field is pending.
    @test nrow(U.V24LiveCalibration.canonical_live_rows(pending)) == nrow(input) - 1

    unknown = U.replay_upgrade(input[[1; 3:12], :]; warmup=2)
    @test !band(unknown, "L24", 2).available # Target 01:00 has no witness; final outcome cannot stand in.
    @test !band(unknown, "L24", 3).available
    @test band(unknown, "L24", 4).available

    revision = copy(input[1:1, :])
    revision.sub_hourly_model_version .= "witness-only"
    revision.issue_time_utc .= DateTime(2026, 1, 1, 3, 45)
    revision.latest_dst_time_utc .= DateTime(2026, 1, 1, 1)
    revision.latest_dst_nt .= 20.0
    revised = U.replay_upgrade(vcat(input, revision); warmup=2)
    @test band(revised, "L24", 3).lower_dst_nt == band(result, "L24", 3).lower_dst_nt
    @test band(revised, "L24", 4).lower_dst_nt == 4.5 # median([20,4,6,8])=7, less 2.5.
    @test band(revised, "L24", 4).upper_dst_nt == 12.0

    for seed in 1:5
        order = randperm(MersenneTwister(seed), nrow(input))
        reordered = U.replay_upgrade(vcat(input[order, :], input[1:2, :]); warmup=2)
        @test isequal(result.replay, reordered.replay)
    end
    multi = U.replay_upgrade(fixture(n=12, steps=(1, 2)); warmup=2)
    @test band(multi, "L24", 2; step=1).available
    @test !band(multi, "L24", 2; step=2).available # No cross-step pooling.

    conflicting = vcat(input, input[1:1, :])
    conflicting.latest_dst_nt[end] = 99.0
    @test_throws ArgumentError U.replay_upgrade(conflicting; warmup=2)
    bad = copy(input); bad.latest_dst_nt[1] = Inf
    @test_throws ArgumentError U.replay_upgrade(bad)
    bad = copy(input); bad.v24_cal_shadow_config_sha256[1] = "0"^64
    @test_throws ArgumentError U.replay_upgrade(bad)
    bad = copy(input); bad.v24_cal_shadow_ci05_nt[1] = 6.0
    @test_throws ArgumentError U.replay_upgrade(bad)
    @test_throws ArgumentError U.replay_upgrade(input; warmup=0)
    @test isempty(U.replay_upgrade(input[1:0, :]).replay)
end

@testset verbose=true "Paired summaries, day blocks, and frozen selection" begin
    values = [1.0, 1.0, 0.0]
    days = [Date(2026, 1, 1), Date(2026, 1, 1), Date(2026, 1, 2)]
    samples = U.bootstrap_days(values, days; reps=200, seed=7)
    @test samples.lower == 0.0
    @test samples.upper == 1.0
    @test all(value -> value in (0.0, 2 / 3, 1.0), samples.draws)
    @test 2 / 3 in samples.draws # A two-row day is not given one-row weight.
    @test samples == U.bootstrap_days(values, days; reps=200, seed=7)
    @test isempty(U.bootstrap_days([1.0], [days[1]]).draws)
    @test_throws DimensionMismatch U.bootstrap_days([1.0], days)
    @test_throws ArgumentError U.bootstrap_days(values, days; reps=99)
    @test_throws ArgumentError U.bootstrap_days(values, days; seed=-1)
    @test_throws ArgumentError U.bootstrap_days([NaN, 1.0, 0.0], days)

    result = U.replay_upgrade(fixture(); warmup=2)
    start, stop = DateTime(2026, 1, 1, 4), DateTime(2026, 1, 1, 5)
    common = U.common_rows(result.replay; start, stop)
    @test nrow(common) == 8 # The one hour has all six candidates and both controls.
    tables = U.summarize_upgrade(result.replay; start, stop, reps=100)
    static = only(eachrow(tables.summary[tables.summary.candidate .== "static", :]))
    @test static.n == 1
    @test static.coverage == 0.0
    @test static.mean_width_nt == 6.0
    @test static.mean_interval_score_nt == 126.0 # y=10, upper=4: 6+20*6.
    @test static.width_ratio == 1.0
    @test static.score_difference_nt == 0.0
    @test static.point_rmse_nt == 10.0
    @test U.select_candidate(tables.summary) === nothing
    @test !any(tables.summary.advances)
    early = U.common_rows(result.replay; start=DateTime(2026, 1, 1), stop=start)
    @test isempty(early) # Standardized warm-up removes the row for every comparator.
    missing_candidate = result.replay[2:end, :]
    @test_throws ArgumentError U.common_rows(missing_candidate; start=DateTime(2026, 1, 1), stop)
    mismatched = copy(result.replay)
    mismatched.point_dst_nt[1] = 1.0
    @test_throws ArgumentError U.common_rows(mismatched; start=DateTime(2026, 1, 1), stop)

    metrics = (n=1000, coverage=0.9, width_ratio=1.25, score_difference_nt=-2.0)
    steps = [(n=100, coverage=0.9) for _ in U.STEPS]
    ci, score_ci = (lower=0.86, upper=0.92), (lower=-3.0, upper=-1.0)
    @test isempty(U._advancement_reasons(metrics, steps, 7, 0.85, ci, score_ci))
    for changed in ((; metrics..., coverage=0.87), (; metrics..., width_ratio=1.251),
                    (; metrics..., score_difference_nt=0.1))
        @test !isempty(U._advancement_reasons(changed, steps, 7, 0.85, ci, score_ci))
    end
    @test !isempty(U._advancement_reasons(metrics, [(n=39, coverage=0.9); steps[2:end]], 7, 0.85, ci, score_ci))
    @test !isempty(U._advancement_reasons(metrics, steps, 6, 0.85, ci, score_ci))
    @test !isempty(U._advancement_reasons(metrics, steps, 7, 0.79, ci, score_ci))
    @test !isempty(U._advancement_reasons(metrics, steps, 7, 0.85, (lower=0.86, upper=0.89), score_ci))
    @test !isempty(U._advancement_reasons(metrics, steps, 7, 0.85, ci, (lower=-3.0, upper=0.1)))
    choice = DataFrame(candidate=["S48", "L24", "static"], advances=[true, true, true],
        mean_interval_score_nt=[10.0, 10.0, 0.0], mean_width_nt=[4.0, 4.0, 1.0])
    @test U.select_candidate(choice) == "L24" # Frozen table order breaks exact ties; controls never win.
    choice.mean_interval_score_nt[1] = 9.0
    @test U.select_candidate(choice) == "S48"
    @test choice.candidate == ["S48", "L24", "static"] # Selection does not mutate its input.

    dates = [DateTime(2026, 1, k, 12) for k in 1:8]
    small = DataFrame(issue_time_utc=dates)
    full = U._complete_days(small, DateTime(2026, 1, 1), DateTime(2026, 1, 8, 13))
    @test full == collect(Date(2026, 1, 1):Day(1):Date(2026, 1, 7))

    mktempdir() do dir
        path = joinpath(dir, "input.csv")
        CSV.write(path, fixture(n=120))
        digest = bytes2hex(sha256(read(path)))
        output = joinpath(dir, "development")
        study = U.run_upgrade(path, output; stage=:development, expected_sha=digest)
        @test study.receipt.input_sha256 == digest
        @test study.selected === nothing
        @test !study.receipt.prospective
        @test !study.receipt.deployment_authorized
        @test CSV.read(joinpath(output, "summary.csv"), DataFrame).candidate == [U.CANDIDATES...; U.CONTROLS...]
        @test isfile(joinpath(output, "witness_support.csv"))
        @test isfile(joinpath(output, "diagnostics.csv"))
        @test_throws ArgumentError U.run_upgrade(path, output; stage=:development, expected_sha=digest)
        @test_throws ArgumentError U.run_upgrade(path, joinpath(dir, "wrong"); stage=:development, expected_sha="0"^64)
        @test_throws ArgumentError U.run_upgrade(path, joinpath(dir, "wrong-stage"); stage=:test, expected_sha=digest)
        @test bytes2hex(sha256(read(path))) == digest
    end
end

end # module
