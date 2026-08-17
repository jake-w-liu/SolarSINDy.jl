module V22ServedReplayTests

using Test
using CSV
using DataFrames
using Dates
using SolarSINDy

include(joinpath(
    @__DIR__, "..", "validation", "operational", "v2_2_served_replay.jl",
))

function _synthetic_feeds(issue::DateTime; include_future::Bool=false)
    stop = include_future ? issue + Hour(8) : issue
    times = collect((issue - Hour(14)):Minute(1):stop)
    causal = times .<= issue
    plasma = DataFrame(
        time_tag=times,
        speed=ifelse.(causal, 500.0, 700.0),
        density=ifelse.(causal, 6.0, 20.0),
    )
    mag = DataFrame(
        time_tag=times,
        bz_gsm=ifelse.(causal, -4.0, -30.0),
        by_gsm=ifelse.(causal, 1.0, 12.0),
    )
    return plasma, mag
end

function _synthetic_dst(issue::DateTime)
    return Dict(
        issue + Hour(h) => -44.0 - 2.0h + 0.25h^2
        for h in -16:8
    )
end

function _lookup(dst::Dict{DateTime,Float64})
    return t -> get(dst, t, missing)
end

const CORE = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
const CALIBRATION = read_operational_v2_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
)
const CONFORMAL = read_conformal_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).conformal_csv,
)

@testset verbose=true "V2.2 causal served replay" begin
    @testset "whole-anchor split and target-maturity embargo" begin
        fit_last = V22ReplayAnchor(DateTime(2017, 10, 20, 15))
        val_last = V22ReplayAnchor(DateTime(2020, 5, 22, 19))
        cal_last = V22ReplayAnchor(DateTime(2022, 12, 31, 16))
        @test v2_2_replay_split(fit_last) == :fit
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2017, 10, 20, 16))) == :embargo
        @test v2_2_replay_split(V22ReplayAnchor(
            DateTime(2017, 10, 20, 16), DateTime(2017, 10, 20, 15),
        )) == :embargo
        @test v2_2_replay_split(V22ReplayAnchor(
            DateTime(2017, 10, 20, 15), DateTime(2017, 10, 20, 14),
        )) == :fit
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2017, 10, 20, 23))) == :validation
        @test v2_2_replay_split(val_last) == :validation
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2020, 5, 22, 20))) == :embargo
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2020, 5, 23, 3))) == :calibration
        @test v2_2_replay_split(cal_last) == :calibration
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2022, 12, 31, 17))) == :embargo

        benchmark = V22ReplayAnchor(DateTime(2023, 1, 1))
        @test v2_2_replay_split(benchmark) == V22_EXPOSED_BENCHMARK_LOCKED
        @test v2_2_replay_split(benchmark; benchmark_access=true) == :benchmark

        # The split is computed from all six supported targets, not a requested
        # subset: the 16:00 fit-boundary anchor remains embargoed even for h=1.
        @test v2_2_replay_split(V22ReplayAnchor(DateTime(2017, 10, 20, 16))) ==
              :embargo
    end

    @testset "strict key, target, row, and schema invariants" begin
        issue = DateTime(2022, 7, 15, 12)
        plasma, mag = _synthetic_feeds(issue)
        dst = _synthetic_dst(issue)
        anchor = V22ReplayAnchor(issue)
        table = build_v2_2_served_replay(
            [anchor], plasma, mag, _lookup(dst), CORE, CALIBRATION,
        )

        @test nrow(table) == length(V22_REPLAY_MODEL_STEPS)
        @test table.model_step_hours == V22_REPLAY_MODEL_STEPS
        @test table.target_time_utc ==
              fill(issue, nrow(table)) .+ Hour.(table.model_step_hours)
        @test table.anchor_lag_steps == zeros(Int, nrow(table))
        @test table.product_horizon_hours == table.model_step_hours
        @test length(unique(Tuple.(eachrow(select(
            table, :issue_time_utc, :target_time_utc, :model_step_hours,
        ))))) == nrow(table)
        @test all(==("calibration"), table.split_label)
        @test_throws ArgumentError build_v2_2_served_replay(
            [V22ReplayAnchor(issue, issue - Hour(1))],
            plasma,
            mag,
            _lookup(dst),
            CORE,
            CALIBRATION;
            model_steps=[1],
        )

        required = Set([
            :served_v2_1_dst_nt, :frozen_v2_1_dst_nt,
            :raw_sindy_dst_nt, :persistence_dst_nt, :burton_dst_nt,
            :burton_full_dst_nt, :obrien_dst_nt, :observation_dst_nt,
            :latest_dst_nt, :dst_delta_1h_nt, :VBsouth_mvm, :coupling_active_mvm,
            :driver_sequence_sha256, :inputs_sha256, :split_label,
        ])
        @test issubset(required, Set(Symbol.(names(table))))
        for name in (
            :served_v2_1_dst_nt, :frozen_v2_1_dst_nt,
            :raw_sindy_dst_nt, :persistence_dst_nt, :burton_dst_nt,
            :burton_full_dst_nt, :obrien_dst_nt, :observation_dst_nt,
            :latest_dst_nt, :dst_delta_1h_nt, :VBsouth_mvm, :coupling_active_mvm,
        )
            @test all(isfinite, table[!, name])
        end
    end

    @testset "benchmark sentinel precedes every observation access" begin
        issue = DateTime(2022, 7, 15, 12)
        plasma, mag = _synthetic_feeds(issue)
        calls = Ref(0)
        forbidden = _ -> (calls[] += 1; error("observation accessor was called"))
        @test_throws V22BenchmarkAccessError build_v2_2_served_replay(
            [V22ReplayAnchor(DateTime(2023, 1, 1))],
            plasma,
            mag,
            forbidden,
            CORE,
            CALIBRATION,
        )
        @test calls[] == 0
    end

    @testset "post-issue mutation cannot enter the admitted information set" begin
        issue = DateTime(2022, 7, 15, 12)
        plasma, mag = _synthetic_feeds(issue; include_future=true)
        dst = _synthetic_dst(issue)
        anchor = V22ReplayAnchor(issue)
        clean = build_v2_2_served_replay(
            [anchor], plasma, mag, _lookup(dst), CORE, CALIBRATION,
        )

        mutated_plasma = copy(plasma)
        mutated_mag = copy(mag)
        post = mutated_plasma.time_tag .> issue
        mutated_plasma.speed[post] .= 9_999.0
        mutated_plasma.density[post] .= 999.0
        mutated_mag.bz_gsm[post] .= -999.0
        mutated_mag.by_gsm[post] .= 777.0
        mutated = build_v2_2_served_replay(
            [anchor], mutated_plasma, mutated_mag, _lookup(dst), CORE, CALIBRATION,
        )
        @test isequal(clean, mutated)
    end

    @testset "missing and non-finite values fail instead of dropping rows" begin
        issue = DateTime(2022, 7, 15, 12)
        plasma, mag = _synthetic_feeds(issue)
        dst = _synthetic_dst(issue)
        anchor = V22ReplayAnchor(issue)
        missing_target = t -> t == issue + Hour(1) ? missing : get(dst, t, missing)
        nan_target = t -> t == issue + Hour(1) ? NaN : get(dst, t, missing)
        @test_throws ArgumentError build_v2_2_served_replay(
            [anchor], plasma, mag, missing_target, CORE, CALIBRATION;
            model_steps=[1],
        )
        @test_throws ArgumentError build_v2_2_served_replay(
            [anchor], plasma, mag, nan_target, CORE, CALIBRATION;
            model_steps=[1],
        )

        broken_plasma = copy(plasma)
        trailing = (broken_plasma.time_tag .>= issue - Hour(1)) .&
                   (broken_plasma.time_tag .< issue)
        broken_plasma.speed[trailing] .= NaN
        @test_throws ArgumentError build_v2_2_served_replay(
            [anchor], broken_plasma, mag, _lookup(dst), CORE, CALIBRATION;
            model_steps=[1],
        )
    end

    @testset "helper is exactly identical to the live V2.1 kernel" begin
        issue = DateTime(2022, 7, 15, 12)
        plasma, mag = _synthetic_feeds(issue)
        dst = _synthetic_dst(issue)
        lookup = _lookup(dst)
        lag0 = build_v2_2_served_replay(
            [V22ReplayAnchor(issue)], plasma, mag, lookup, CORE, CALIBRATION;
            model_steps=[1, 2, 3, 6],
        )
        lag1 = build_v2_2_served_replay(
            [V22ReplayAnchor(issue, issue - Hour(1))],
            plasma,
            mag,
            lookup,
            CORE,
            CALIBRATION;
            model_steps=[2, 3, 4, 7],
        )
        replay_by_lag = Dict(0 => lag0, 1 => lag1)
        cases = (
            (step=1, lag=0, product_horizon=1),
            (step=2, lag=0, product_horizon=2),
            (step=3, lag=0, product_horizon=3),
            (step=4, lag=1, product_horizon=3),
            (step=6, lag=0, product_horizon=6),
            (step=7, lag=1, product_horizon=6),
        )

        mktempdir() do dir
            for case in cases
                latest = issue - Hour(case.lag)
                dst_times = sort!([t for t in keys(dst) if t <= latest])
                dst_values = Float64[dst[t] for t in dst_times]
                cfg = LiveVerifyConfig(;
                    model=:v2,
                    horizon_hours=case.product_horizon,
                    log_path=joinpath(dir, "live_step_$(case.step).csv"),
                    report_path=joinpath(dir, "report_step_$(case.step).md"),
                )
                prepared = (
                    issue_time=issue,
                    plasma=plasma,
                    mag=mag,
                    dst=(dst_times, dst_values),
                    calibration=CALIBRATION,
                    conformal=CONFORMAL,
                    model=:v2,
                    calibration_path=abspath(cfg.v2_calibration_path),
                )
                redirect_stdout(devnull) do
                    issue_forecast(
                        cfg;
                        inputs=prepared,
                        write_trajectory=false,
                        verbose=false,
                        interval_policy=:static,
                    )
                end
                live = CSV.read(cfg.log_path, DataFrame)[1, :]
                replay = only(eachrow(replay_by_lag[case.lag][
                    replay_by_lag[case.lag].model_step_hours .== case.step,
                    :,
                ]))

                # Exact equality is required: both paths call the same V2.1
                # driver admission, point step, correction, and safeguard kernels.
                # The V2.1 center is now the `v2_1_served_pred_dst_nt` continuity column, because
                # `served_pred_dst_nt` carries the static-stack stage that runs on top of it.
                @test replay.served_v2_1_dst_nt == live.v2_1_served_pred_dst_nt
                @test replay.frozen_v2_1_dst_nt == live.v2_pred_dst_nt
                # The stack stage is the only difference between the two served columns, and it must
                # be the stack the log discloses rather than an unexplained divergence.
                if live.sub_hourly_model_version == V2_2_SERVED_TAIL_VERSION
                    stack = load_v22_serving_stack(V2_2_DEFAULT_STACK_PATH)
                    stacked = v22_serving_center(
                        stack; model_steps=Int(live.model_step_hours),
                        latest_dst=Float64(live.latest_dst_nt),
                        dst_delta_1h_nt=Float64(live.dst_delta_1h_nt),
                        vbsouth_mvm=Float64(live.VBsouth_mvm),
                        served_v2_1=Float64(live.v2_1_served_pred_dst_nt),
                        frozen_v2_1=Float64(live.v2_pred_dst_nt),
                        persistence=Float64(live.persistence_dst_nt),
                        burton=Float64(live.burton_dst_nt),
                        burton_full=Float64(live.burton_full_dst_nt),
                        obrien=Float64(live.obrien_dst_nt),
                    )
                    @test Float64(live.served_pred_dst_nt) ≈ stacked.center atol=1e-12
                else
                    @test live.sub_hourly_model_version == V2_SERVED_TAIL_VERSION
                    @test Float64(live.served_pred_dst_nt) ==
                          Float64(live.v2_1_served_pred_dst_nt)
                end
                @test replay.persistence_dst_nt == live.persistence_dst_nt
                @test replay.burton_dst_nt == live.burton_dst_nt
                @test replay.burton_full_dst_nt == live.burton_full_dst_nt
                @test replay.obrien_dst_nt == live.obrien_dst_nt
                @test replay.target_time_utc == DateTime(live.target_time_utc)
                @test replay.model_step_hours == live.model_step_hours == case.step
                @test replay.anchor_lag_steps == case.lag
                @test replay.product_horizon_hours == case.product_horizon
            end
        end
    end
end

end # module
