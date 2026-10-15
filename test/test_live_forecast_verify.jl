module LiveForecastVerificationTests

using Test
using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "examples", "live_forecast_verify.jl"))

struct _InterruptingForecastText end
Base.String(::_InterruptingForecastText) = throw(InterruptException())

include(joinpath(@__DIR__, "superlearner_serving_fixture.jl"))
using .SuperlearnerServingFixture

@testset "Live Forecast Verification Workflow" begin
    @testset "Kyoto transport preserves TLS verification and deadlines" begin
        calls = Ref(0)
        response(url; kwargs...) = begin
            calls[] += 1
            @test kwargs[:socket_type_tls] === MbedTLS.SSLContext
            @test !haskey(kwargs, :require_ssl_verification)
            @test (kwargs[:connect_timeout], kwargs[:readtimeout]) == (15, 30)
            @test kwargs[:retries] == 0
            HTTP.Response(200, """[{"time_tag":"2026-01-01T00:00:00","dst":-10}]""")
        end
        times, values = _fetch_dst(; max_retries=1, fetch_fn=response)
        @test calls[] == 1
        @test times == [DateTime(2026, 1, 1)]
        @test values == [-10.0]
    end

    @testset "OMNI replay windows preserve independent driver and Dst support" begin
        t0 = DateTime(2026, 1, 1)
        plasma = DataFrame(
            time_tag=t0 .+ Hour.([0, 2, 4]),
            speed=[400.0, 420.0, 440.0],
            density=fill(5.0, 3),
        )
        mag = DataFrame(
            time_tag=t0 .+ Hour.([0, 1, 2, 4]),
            bz_gsm=fill(-2.0, 4),
            by_gsm=fill(1.0, 4),
        )
        dst_times = t0 .+ Hour.(0:4)
        dst_vals = Float64.(-10:-10:-50)

        ps, ms, ts, vs = _slice_replay_window(
            plasma, mag, dst_times, dst_vals, t0 + Hour(1), t0 + Hour(3),
        )
        @test ps.time_tag == [t0 + Hour(2)]
        @test ms.time_tag == [t0 + Hour(1), t0 + Hour(2)]
        @test ts == t0 .+ Hour.(1:3)
        @test vs == [-20.0, -30.0, -40.0]
        @test_throws DimensionMismatch _slice_replay_window(
            plasma, mag, dst_times, dst_vals[1:end-1], t0, t0 + Hour(1),
        )
    end

    @testset "A/D: target time is strictly future relative to issue time" begin
        issue_time = DateTime(2026, 6, 6, 4, 0, 34)
        latest_dst_time = DateTime(2026, 6, 6, 3)
        target = _next_hourly_target(issue_time, 1, latest_dst_time)

        @test target == DateTime(2026, 6, 6, 5)
        @test target > issue_time
        @test target > latest_dst_time

        exact_hour_issue = DateTime(2026, 6, 6, 4)
        @test _next_hourly_target(exact_hour_issue, 1, latest_dst_time) ==
              DateTime(2026, 6, 6, 5)

        interrupted = DataFrame(
            issue_time_utc=Any[_InterruptingForecastText()],
            target_time_utc=Any["2026-06-06T05:00:00Z"],
        )
        @test_throws InterruptException _row_is_strictly_future(interrupted, 1)
    end

    @testset "A/D: feed retries never swallow process interruption" begin
        attempts = Ref(0)
        interrupting_fetch = function (_url; kwargs...)
            attempts[] += 1
            throw(InterruptException())
        end
        @test_throws InterruptException _fetch_dst(
            max_retries=3, retry_delay_sec=0, fetch_fn=interrupting_fetch,
        )
        @test attempts[] == 1
    end

    @testset "A/D: future-dated feeds fail closed at the clock-skew boundary" begin
        issue_time = DateTime(2026, 7, 15, 12)
        at_tolerance = issue_time + LIVE_FUTURE_CLOCK_TOLERANCE
        @test _validate_feed_timestamps(
            issue_time, at_tolerance, at_tolerance, at_tolerance,
        ) === nothing
        @test_throws ErrorException _validate_feed_timestamps(
            issue_time, issue_time + Hour(1), issue_time, issue_time,
        )
        @test_throws ErrorException _validate_feed_timestamps(
            issue_time, issue_time, issue_time + Hour(1), issue_time,
        )
        @test_throws ErrorException _validate_feed_timestamps(
            issue_time, issue_time, issue_time, issue_time + Hour(24),
        )
        @test_throws ArgumentError _validate_feed_timestamps(
            issue_time, issue_time, issue_time, issue_time; tolerance=Minute(-1),
        )
        # Between the tolerance and the sanity limit the skew is reported, not fatal: post-issue
        # samples are already excluded by the causal selection, so failing the whole issuance here
        # turned local clock drift into a 4/4-horizon outage and then a dead-man restart loop that
        # a restart cannot fix. The hard failure survives one second past the sanity limit.
        @test LIVE_MAX_FUTURE_CLOCK_SKEW > LIVE_FUTURE_CLOCK_TOLERANCE
        at_sanity = issue_time + LIVE_MAX_FUTURE_CLOCK_SKEW
        for skewed in (
            (at_sanity, issue_time, issue_time),
            (issue_time, at_sanity, issue_time),
            (issue_time, issue_time, at_sanity),
        )
            @test (@test_logs (:warn,) match_mode = :any _validate_feed_timestamps(
                issue_time, skewed...,
            )) === nothing
            @test_throws ErrorException _validate_feed_timestamps(
                issue_time, (skewed .+ Second(1))...,
            )
        end
        # A single call reports every skewed feed, not just the first.
        @test_logs (:warn,) (:warn,) (:warn,) match_mode = :any _validate_feed_timestamps(
            issue_time, at_sanity, at_sanity, at_sanity,
        )
        @test_throws ArgumentError _validate_feed_timestamps(
            issue_time, issue_time, issue_time, issue_time;
            tolerance=Minute(2), sanity_limit=Minute(1),
        )
        @test _latest_causal_index(
            [issue_time - Minute(1), issue_time + Minute(1), issue_time],
            issue_time,
            "test",
        ) == 3
        @test_throws ErrorException _latest_causal_index(
            [issue_time + Second(30)], issue_time, "test",
        )
    end

    @testset "A/D: tolerated source-clock skew never enters an issued forecast" begin
        mktempdir() do dir
            issue_time = DateTime(2026, 7, 15, 12, 30)
            sw_times = collect((issue_time - Hour(3)):Minute(1):issue_time)
            causal_plasma = DataFrame(
                time_tag=sw_times,
                speed=fill(500.0, length(sw_times)),
                density=fill(6.0, length(sw_times)),
            )
            causal_mag = DataFrame(
                time_tag=sw_times,
                bz_gsm=fill(-4.0, length(sw_times)),
                by_gsm=fill(1.0, length(sw_times)),
            )
            future_times = [issue_time + Second(30), issue_time + Minute(1)]
            skewed_plasma = vcat(
                causal_plasma,
                DataFrame(time_tag=future_times, speed=[1200.0, 1200.0], density=[50.0, 50.0]),
            )
            skewed_mag = vcat(
                causal_mag,
                DataFrame(time_tag=future_times, bz_gsm=[-100.0, -100.0], by_gsm=[50.0, 50.0]),
            )
            latest_dst_time = floor(issue_time, Hour)
            dst_times = collect((latest_dst_time - Hour(6)):Hour(1):latest_dst_time)
            dst_values = collect(range(-20.0, -44.0; length=length(dst_times)))
            skewed_dst = (vcat(dst_times, [issue_time + Minute(1)]),
                          vcat(dst_values, [-800.0]))

            # Beyond the 2 min tolerance but inside the sanity limit: the feed still carries causal
            # samples, so issuance must proceed on those rather than failing the whole cycle.
            beyond_tolerance = [issue_time + Minute(3), issue_time + Minute(4)]
            drifted_plasma = vcat(
                causal_plasma,
                DataFrame(time_tag=beyond_tolerance, speed=[1200.0, 1200.0],
                          density=[50.0, 50.0]),
            )
            drifted_mag = vcat(
                causal_mag,
                DataFrame(time_tag=beyond_tolerance, bz_gsm=[-100.0, -100.0],
                          by_gsm=[50.0, 50.0]),
            )
            drifted_dst = (vcat(dst_times, [issue_time + Minute(3)]),
                           vcat(dst_values, [-800.0]))

            configs = [LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=1,
                log_path=joinpath(dir, "forecast_$i.csv"),
                report_path=joinpath(dir, "report_$i.md"),
            ) for i in 1:3]
            causal_inputs = prepare_issue_inputs(
                configs[1]; issue_time,
                plasma_fn=() -> causal_plasma,
                mag_fn=() -> causal_mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            skewed_inputs = prepare_issue_inputs(
                configs[2]; issue_time,
                plasma_fn=() -> skewed_plasma,
                mag_fn=() -> skewed_mag,
                dst_fn=() -> skewed_dst,
            )
            drifted_inputs = prepare_issue_inputs(
                configs[3]; issue_time,
                plasma_fn=() -> drifted_plasma,
                mag_fn=() -> drifted_mag,
                dst_fn=() -> drifted_dst,
            )
            redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    issue_forecast(configs[1]; inputs=causal_inputs, write_trajectory=false, verbose=false)
                    issue_forecast(configs[2]; inputs=skewed_inputs, write_trajectory=false, verbose=false)
                    issue_forecast(configs[3]; inputs=drifted_inputs, write_trajectory=false, verbose=false)
                end
            end
            causal_row = CSV.read(configs[1].log_path, DataFrame)[1, :]
            skewed_row = CSV.read(configs[2].log_path, DataFrame)[1, :]
            drifted_row = CSV.read(configs[3].log_path, DataFrame)[1, :]

            @test DateTime(skewed_row.latest_solar_wind_utc) == issue_time
            @test DateTime(skewed_row.latest_dst_time_utc) == latest_dst_time
            @test skewed_row.latest_dst_nt == last(dst_values)
            # Causality is the invariant the widened tolerance must not touch: the drifted row was
            # admitted, and every driver it used is at or before its own issue time.
            @test DateTime(drifted_row.latest_solar_wind_utc) <= issue_time
            @test DateTime(drifted_row.latest_solar_wind_utc) == issue_time
            @test DateTime(drifted_row.latest_dst_time_utc) <= issue_time
            @test DateTime(drifted_row.latest_dst_time_utc) == latest_dst_time
            @test drifted_row.latest_dst_nt == last(dst_values)
            for column in (
                :anchor_dst_star_nt, :target_time_utc, :V_kms, :Bz_nt, :By_nt,
                :n_cm3, :Pdyn_npa, :target_step_V_kms, :target_step_Bz_nt,
                :target_step_By_nt, :target_step_n_cm3, :target_step_Pdyn_npa,
                :sindy_core_pred_dst_nt, :inertial_sindy_pred_dst_nt, :served_pred_dst_nt,
                :served_pred_dst_ci05_nt, :served_pred_dst_ci95_nt,
            )
                @test isequal(skewed_row[column], causal_row[column])
                @test isequal(drifted_row[column], causal_row[column])
            end

            # A feed whose samples are ALL post-issue still fails closed: there is no causal
            # sample to fall back to, whatever the skew band says.
            all_future = LiveVerifyConfig(;
                model=:inertial_sindy, horizon_hours=1,
                log_path=joinpath(dir, "forecast_all_future.csv"),
                report_path=joinpath(dir, "report_all_future.md"),
            )
            future_only_inputs = prepare_issue_inputs(
                all_future; issue_time,
                plasma_fn=() -> DataFrame(time_tag=[issue_time + Minute(3)],
                                          speed=[500.0], density=[6.0]),
                mag_fn=() -> causal_mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            @test_throws ErrorException redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    issue_forecast(all_future; inputs=future_only_inputs,
                                   write_trajectory=false, verbose=false)
                end
            end
            @test !isfile(all_future.log_path)
        end
    end

    @testset "Monitor-selected ACI policy fails closed without both residual streams" begin
        calls = Ref(0)
        unavailable = (args...; kwargs...) -> (calls[] += 1; nothing)
        @test _aci_interval_for_policy(
            :static, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:inertial_sindy_pred_dst_nt, interval_fn=unavailable,
        ) === nothing
        @test calls[] == 0
        @test _aci_interval_for_policy(
            :auto, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:inertial_sindy_pred_dst_nt, interval_fn=unavailable,
        ) === nothing
        @test calls[] == 1
        @test_throws ErrorException _aci_interval_for_policy(
            :aci, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:served_pred_dst_nt, interval_fn=unavailable,
        )
        available = (args...; kwargs...) -> (-35.0, -5.0)
        @test _aci_interval_for_policy(
            :aci, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:served_pred_dst_nt, interval_fn=available,
        ) == (-35.0, -5.0)
        @test_throws ArgumentError _aci_interval_for_policy(
            :mixed, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:inertial_sindy_pred_dst_nt, interval_fn=available,
        )
    end

    @testset "A/D: materially stale Dst state fails closed" begin
        mktempdir() do dir
            issue_time = DateTime(2026, 7, 15, 12, 30)
            sw_times = collect((issue_time - Hour(12)):Minute(1):issue_time)
            plasma = DataFrame(
                time_tag=sw_times,
                speed=fill(500.0, length(sw_times)),
                density=fill(6.0, length(sw_times)),
            )
            mag = DataFrame(
                time_tag=sw_times,
                bz_gsm=fill(-4.0, length(sw_times)),
                by_gsm=fill(1.0, length(sw_times)),
            )
            latest_dst_time = floor(issue_time, Hour) - Hour(2)
            dst_times = collect((latest_dst_time - Hour(6)):Hour(1):latest_dst_time)
            dst_values = collect(range(-20.0, -44.0; length=length(dst_times)))
            cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=1,
                log_path=joinpath(dir, "live_forecast_log.csv"),
                report_path=joinpath(dir, "live_comparison_report.md"),
            )
            inputs = prepare_issue_inputs(
                cfg; issue_time,
                plasma_fn=() -> plasma,
                mag_fn=() -> mag,
                dst_fn=() -> (dst_times, dst_values),
            )

            @test LIVE_MAX_DST_ANCHOR_LAG_STEPS == 1
            @test_throws ErrorException issue_forecast(
                cfg; inputs, write_trajectory=false, verbose=false,
            )
            @test !isfile(cfg.log_path)
        end
    end

    @testset "D: argument parser accepts operational workflow options" begin
        cfg = _parse_args([
            "--fit-operational-calibration",
            "--model=operational",
            "--table=/tmp/replay.csv",
            "--report=/tmp/live_report.md",
            "--operational-calibration=/tmp/operational.csv",
            "--operational-train-fraction=0.6",
            "--operational-validation-fraction=0.2",
            "--operational-ridge=10",
            "--operational-coverage=0.8",
            "--operational-selector-margin=1.25",
        ])
        @test cfg.mode == :fit_inertial_sindy_calibration
        @test cfg.model == :inertial_sindy
        @test cfg.table_path == "/tmp/replay.csv"
        @test cfg.report_path == "/tmp/live_report.md"
        @test cfg.inertial_sindy_calibration_path == "/tmp/operational.csv"
        @test cfg.operational_train_fraction == 0.6
        @test cfg.operational_validation_fraction == 0.2
        @test cfg.operational_ridge == 10.0
        @test cfg.operational_ridge_grid == [10.0]
        @test cfg.inertial_sindy_interval_coverage == 0.8
        @test cfg.operational_selector_margin_nt == 1.25

        grid_cfg = _parse_args(["--fit-operational-calibration", "--operational-ridge-grid=0,10,100"])
        @test grid_cfg.operational_ridge_grid == [0.0, 10.0, 100.0]

        refresh_cfg = _parse_args(["--refresh-observations", "--log=/tmp/live.csv"])
        @test refresh_cfg.mode == :refresh_observations
        @test refresh_cfg.log_path == "/tmp/live.csv"

        omni_cfg = _parse_args([
            "--replay-omni",
            "--omni=/tmp/omni.csv",
            "--omni-year-start=2024",
            "--omni-year-end=2025",
            "--replay-hours=100",
        ])
        @test omni_cfg.mode == :replay_omni
        @test omni_cfg.omni_path == "/tmp/omni.csv"
        @test omni_cfg.omni_year_start == 2024
        @test omni_cfg.omni_year_end == 2025
        @test omni_cfg.replay_hours == 100
        @test_throws ArgumentError _parse_args([
            "--replay-omni",
            "--omni-year-start=2026",
            "--omni-year-end=2025",
        ])

        campaign = _parse_args([
            "--campaign",
            "--campaign-horizons=1,3,6",
            "--poll-seconds=1",
            "--timeout-hours=0.1",
        ])
        @test campaign.mode == :campaign
        @test campaign.model == :inertial_sindy
        @test campaign.campaign_horizons == [1, 3, 6]

        default_issue = _parse_args(String[])
        @test default_issue.mode == :issue
        @test default_issue.model == :inertial_sindy

        explicit_core = _parse_args(["--campaign", "--model=sindy-core"])
        @test explicit_core.model == :sindy_core
        @test _parse_args(["--campaign", "--model=v1"]).model == :sindy_core
        @test_throws ArgumentError _parse_args(["--campaign-horizons=1,0"])
    end

    @testset "A/D: operational derived features are causal and deterministic" begin
        features = _inertial_sindy_features(
            -20.0,
            (; V=500.0, Bz=-4.0, By=3.0, n=5.0, Pdyn=2.25),
        )
        @test features.latest_dst_nt == -20.0
        @test features.Bsouth_nt == 4.0
        @test features.VBsouth_mvm == 2.0
        @test features.Bperp_nt == 5.0
        @test features.clock_angle_sin2 ≈ 0.9 atol=1e-12
        @test features.sqrt_Pdyn_npa == 1.5
        @test features.dst_delta_1h_nt == 0.0
        @test features.baseline_spread_nt == 0.0

        expert = _inertial_sindy_features(
            -20.0,
            (; V=500.0, Bz=-4.0, By=3.0, n=5.0, Pdyn=2.25);
            memory=(;
                dst_delta_1h_nt=-2.0,
                dst_delta_3h_nt=-5.0,
                Bz_delta_1h_nt=-1.0,
                VBsouth_delta_1h_mvm=0.4,
                VBsouth_mean_3h_mvm=1.5,
                Bsouth_mean_3h_nt=3.0,
            ),
            baselines=(; persistence=-22.0, burton=-18.0, burton_full=-19.0, obrien=-21.0),
            sindy_core_pred_dst=-20.0,
            model_steps=3,
        )
        @test expert.dst_delta_3h_nt == -5.0
        @test expert.baseline_spread_nt == 4.0
        @test expert.v1_minus_persistence_nt == 2.0
        @test expert.obrien_minus_v1_nt == -1.0
        @test expert.lead_2h_indicator == 0.0
        @test expert.lead_3h_indicator == 1.0
        @test expert.lead_6h_indicator == 0.0
        @test expert.lead_latest_dst_interaction == -60.0
        @test expert.lead_v1_persistence_interaction == 6.0
        @test_throws ArgumentError _inertial_sindy_features(
            -20.0,
            (; V=500.0, Bz=-4.0, By=3.0, n=5.0, Pdyn=2.25);
            model_steps=0,
        )

        # The fitted residual layer uses the driver window aligned with the Dst
        # anchor, never the final rollout-step driver.  Pin both the correction
        # input and the log-schema distinction so an arriving shock cannot move
        # the live feature basis away from the replay/training convention.
        anchor = (; V=420.0, Bz=-3.0, By=1.0, n=5.0,
                  Pdyn=dynamic_pressure(5.0, 420.0))
        target_step = (; V=800.0, Bz=-20.0, By=6.0, n=18.0,
                       Pdyn=dynamic_pressure(18.0, 800.0))
        cal = InertialSindyCalibration(
            Symbol[:V_kms], Float64[0.0], Float64[1.0], Float64[0.0, 0.01],
            1.0, "anchor-feature-regression",
        )
        anchor_features = _inertial_sindy_features(-40.0, anchor)
        target_features = _inertial_sindy_features(-40.0, target_step)
        @test SolarSINDy.operational_inertial_sindy_correction(cal, anchor_features) ≈ 4.2
        @test SolarSINDy.operational_inertial_sindy_correction(cal, target_features) ≈ 8.0
        selected = _select_model_prediction(
            :inertial_sindy, cal, -40.0, anchor, -45.0, -55.0, -35.0;
            features=anchor_features,
        )
        @test selected.inertial_sindy_correction ≈ 4.2

        audit = _driver_audit_fields(anchor, target_step)
        @test _ANCHOR_FEATURE_DRIVER_BASIS ==
              "anchor_aligned_ballistically_propagated_l1_hour"
        @test audit.V_kms == anchor.V
        @test audit.Pdyn_npa == anchor.Pdyn
        @test audit.target_step_V_kms == target_step.V
        @test audit.target_step_Pdyn_npa == target_step.Pdyn
    end

    @testset "Issued target-step audit follows the served relaxed tail" begin
        mktempdir() do dir
            issue_time = DateTime(2026, 7, 15, 12, 30)
            sw_times = collect((issue_time - Hour(12)):Minute(1):issue_time)
            plasma = DataFrame(
                time_tag=sw_times,
                speed=fill(500.0, length(sw_times)),
                density=fill(6.0, length(sw_times)),
            )
            mag = DataFrame(
                time_tag=sw_times,
                bz_gsm=fill(-4.0, length(sw_times)),
                by_gsm=fill(1.0, length(sw_times)),
            )
            latest_dst_time = _floor_hour(issue_time) - Hour(1)
            dst_times = collect((latest_dst_time - Hour(12)):Hour(1):latest_dst_time)
            dst_values = collect(range(-20.0, -44.0; length=length(dst_times)))
            cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=6,
                log_path=joinpath(dir, "live_forecast_log.csv"),
                report_path=joinpath(dir, "live_comparison_report.md"),
            )
            inputs = prepare_issue_inputs(
                cfg;
                issue_time,
                plasma_fn=() -> plasma,
                mag_fn=() -> mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            redirect_stdout(devnull) do
                issue_forecast(cfg; inputs, write_trajectory=false, verbose=false)
            end

            row = CSV.read(cfg.log_path, DataFrame)[1, :]
            # The final four target hours lie beyond measured L1 coverage. With
            # dDst/dt = -2 nT/h, the served driver uses the exact four-hour
            # relaxation rather than the frozen-tail driver.
            relax = exp(-4 / _operational_tail_tau(-2.0))
            @test row.target_step_Bz_nt ≈ -4.0 * relax atol=1e-12
            @test row.target_step_By_nt ≈ 1.0 * relax atol=1e-12
            @test row.target_step_Bz_nt != -4.0
            @test row.Bz_nt == -4.0  # anchor-feature provenance remains unchanged

            # The logged served-model identity promises that the operational tail ran.
            # If that computation fails, issuance must fail closed before a row exists.
            failed_cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=6,
                log_path=joinpath(dir, "failed_tail_log.csv"),
                report_path=joinpath(dir, "failed_tail_report.md"),
            )
            throwing_tail = (args...) -> error("injected tail failure")
            @test_throws ErrorException redirect_stdout(devnull) do
                issue_forecast(
                    failed_cfg; inputs, write_trajectory=false, verbose=false,
                    tail_step_fn=throwing_tail,
                )
            end
            @test !isfile(failed_cfg.log_path)
        end
    end

    @testset "Served static stack columns" begin
        # The served stage is the super-learner, and these cases isolate the stage *below* it: the
        # static stack and its fallbacks. Pointing the bundle at a path that does not exist makes the
        # super-learner stage disclose `fallback:deployment_absent`, so the served center is the stack
        # center and each case still tests the stage it names. The super-learner stage has its own
        # testset below, where the bundle is present.
        absent_superlearner = joinpath(mktempdir(), "no_superlearner_bundle")
        # Synthetic feed with a full day of minute-cadence L1 coverage: enough history for the twelve
        # driver lags the analog key can consume, and enough forward coverage that the served tail
        # exercises both the measured and the relaxed branch.
        issue_time = DateTime(2026, 7, 15, 12, 30)
        sw_times = collect((issue_time - Hour(24)):Minute(1):issue_time)
        plasma = DataFrame(
            time_tag=sw_times,
            speed=[470.0 + 20.0 * sin(2π * k / 613) for k in eachindex(sw_times)],
            density=[6.0 + 0.8 * sin(2π * k / 421) for k in eachindex(sw_times)],
        )
        mag = DataFrame(
            time_tag=sw_times,
            bz_gsm=[-6.0 + 3.0 * sin(2π * k / 517) for k in eachindex(sw_times)],
            by_gsm=[1.5 * cos(2π * k / 379) for k in eachindex(sw_times)],
        )
        latest_dst_time = _floor_hour(issue_time) - Hour(1)
        dst_times = collect((latest_dst_time - Hour(24)):Hour(1):latest_dst_time)
        dst_values = collect(range(-30.0, -78.0; length=length(dst_times)))

        function _issue_row(dir::AbstractString, name::AbstractString)
            cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=6,
                log_path=joinpath(dir, "$(name).csv"),
                report_path=joinpath(dir, "$(name).md"),
            )
            inputs = prepare_issue_inputs(
                cfg; issue_time,
                plasma_fn=() -> plasma, mag_fn=() -> mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            redirect_stdout(devnull) do
                issue_forecast(cfg; inputs, write_trajectory=false, verbose=false)
            end
            return CSV.read(cfg.log_path, DataFrame)[1, :]
        end

        @testset "stack stage serves the published identity" begin
            mktempdir() do dir
                row = withenv("SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    _issue_row(dir, "served_stack")
                end
                @test row.model_version == "inertial_sindy"
                @test row.sub_hourly_model_version == STATIC_STACK_SERVED_TAIL_VERSION
                @test row.sub_hourly_model_version ==
                      "staticstack+sindy60_fit407598+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
                @test row.driver_assumption == STATIC_STACK_DRIVER_ASSUMPTION
                @test row.static_stack_regime in ("quiet", "active_deepening", "recovery")
                @test isfinite(row.inertial_sindy_served_pred_dst_nt)
                @test row.static_stack_pooled_fallback == false
                # Independent recomputation: the served center is the stack cell applied to the six
                # logged components, then projected.
                stack = load_static_stack(STATIC_STACK_DEFAULT_STACK_PATH)
                expected = static_stack_serving_center(
                    stack; model_steps=Int(row.model_step_hours),
                    latest_dst=Float64(row.latest_dst_nt),
                    dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                    vbsouth_mvm=Float64(row.VBsouth_mvm),
                    served_inertial_sindy=Float64(row.inertial_sindy_served_pred_dst_nt),
                    frozen_inertial_sindy=Float64(row.inertial_sindy_pred_dst_nt),
                    persistence=Float64(row.persistence_dst_nt),
                    burton=Float64(row.burton_dst_nt),
                    burton_full=Float64(row.burton_full_dst_nt),
                    obrien=Float64(row.obrien_dst_nt),
                )
                @test Float64(row.served_pred_dst_nt) ≈ expected.center atol=1e-12
                @test String(row.static_stack_regime) == String(expected.regime)
                @test Float64(row.static_stack_coupling_active_mvm) == expected.coupling_active_mvm
                # The coupling gate must be the archived gate, not the raw coupling.
                @test Float64(row.static_stack_coupling_active_mvm) == static_stack_serving_coupling_active(
                    Float64(row.VBsouth_mvm), Float64(row.dst_delta_1h_nt))
                # The served band is shifted onto the served center, so the center stays inside it.
                @test row.served_pred_dst_ci05_nt <= row.served_pred_dst_nt <=
                      row.served_pred_dst_ci95_nt
            end
        end

        @testset "artifact caches retry on a bounded cool-down" begin
            # A weights file that is briefly absent during a redeploy must heal without a daemon
            # restart, so the remembered failure has to expire.
            @test STATIC_STACK_RETRY_SECONDS > 0.0
            mktempdir() do dir
                staged = joinpath(dir, "static_stack.csv")
                withenv("SOLARSINDY_STATIC_STACK" => staged,
                        "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!()
                    absent = redirect_stderr(devnull) do
                        _static_stack()
                    end
                    @test absent.stack === nothing
                    @test absent.status == "fallback_inertial_sindy:stack_absent"
                    cp(STATIC_STACK_DEFAULT_STACK_PATH, staged)
                    # Inside the cool-down the remembered failure is reused, so the file is not re-read.
                    @test _static_stack().stack === nothing
                    # Expiring the cool-down lets the now-present file load.
                    _STATIC_STACK_STATUS_TIME[] -= 2 * STATIC_STACK_RETRY_SECONDS
                    healed = _static_stack()
                    @test healed.status == "ok"
                    @test healed.stack.label == STATIC_STACK_SERVED_LABEL
                end
                reset_static_stack!()
            end
        end

        @testset "absent stack weights fall back to the inertial-SINDy center and say so" begin
            mktempdir() do dir
                row = withenv("SOLARSINDY_STATIC_STACK" => joinpath(dir, "no_such_stack.csv"),
                              "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    _issue_row(dir, "stack_absent")
                end
                reset_static_stack!(); reset_superlearner_serving!()
                @test row.sub_hourly_model_version == OPERATIONAL_SERVED_TAIL_VERSION
                @test row.driver_assumption == OPERATIONAL_DRIVER_ASSUMPTION
                @test Float64(row.served_pred_dst_nt) == Float64(row.inertial_sindy_served_pred_dst_nt)
                @test ismissing(row.static_stack_regime)
            end
        end

        @testset "tampered stack weights fall back rather than serve unpinned weights" begin
            mktempdir() do dir
                tampered = joinpath(dir, "static_stack.csv")
                cp(STATIC_STACK_DEFAULT_STACK_PATH, tampered)
                open(tampered, "a") do io
                    write(io, "\n")
                end
                row = withenv("SOLARSINDY_STATIC_STACK" => tampered,
                              "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_tampered")
                    end
                end
                reset_static_stack!(); reset_superlearner_serving!()
                @test row.sub_hourly_model_version == OPERATIONAL_SERVED_TAIL_VERSION
                @test Float64(row.served_pred_dst_nt) == Float64(row.inertial_sindy_served_pred_dst_nt)
            end
        end

        @testset "an unpinned stack digest is refused unless the operator accepts it" begin
            # An empty digest override removes the only check that ties the weights to the fitted
            # stack the served identity names, because the label check passes for any file carrying
            # the label. The default must therefore be refusal, and an accepted unpinned load must not
            # be published under the pinned identity.
            mktempdir() do dir
                staged = joinpath(dir, "static_stack.csv")
                cp(STATIC_STACK_DEFAULT_STACK_PATH, staged)
                chmod(staged, 0o644)
                refused = withenv("SOLARSINDY_STATIC_STACK" => staged,
                                  "SOLARSINDY_STATIC_STACK_SHA256" => "",
                                  "SOLARSINDY_ALLOW_UNPINNED_STACK" => nothing,
                                  "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_unpinned_refused")
                    end
                end
                @test String(refused.static_stack_status) == "fallback_inertial_sindy:stack_unpinned"
                @test refused.sub_hourly_model_version == OPERATIONAL_SERVED_TAIL_VERSION
                @test Float64(refused.served_pred_dst_nt) ==
                      Float64(refused.inertial_sindy_served_pred_dst_nt)
                @test ismissing(refused.static_stack_regime)

                accepted = withenv("SOLARSINDY_STATIC_STACK" => staged,
                                   "SOLARSINDY_STATIC_STACK_SHA256" => "",
                                   "SOLARSINDY_ALLOW_UNPINNED_STACK" => "1",
                                   "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => absent_superlearner) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_unpinned_accepted")
                    end
                end
                reset_static_stack!(); reset_superlearner_serving!()
                @test String(accepted.static_stack_status) == STATIC_STACK_OK_UNPINNED_STATUS
                # The center is a stack center, so the driver assumption is the stack assumption, but
                # the identity must not claim the pinned product.
                @test accepted.sub_hourly_model_version == STATIC_STACK_UNPINNED_SERVED_TAIL_VERSION
                @test accepted.sub_hourly_model_version != STATIC_STACK_SERVED_TAIL_VERSION
                @test accepted.driver_assumption == STATIC_STACK_DRIVER_ASSUMPTION
                @test accepted.static_stack_regime in ("quiet", "active_deepening", "recovery")
                @test Float64(accepted.served_pred_dst_nt) !=
                      Float64(accepted.inertial_sindy_served_pred_dst_nt)

                # The static-stack center is expert ten of the super-learner stack. Weights that carry no
                # digest pin are not the fitted stack the super-learner was trained against, so the
                # combination cannot be formed from them: a stage that served it anyway would publish
                # unpinned weights under the pinned super-learner identity, which is exactly the laundering the
                # separate unpinned label exists to prevent.
                bundle = build_superlearner_fixture_bundle(joinpath(dir, "superlearner_unpinned_bundle"))
                laundered = withenv("SOLARSINDY_STATIC_STACK" => staged,
                                    "SOLARSINDY_STATIC_STACK_SHA256" => "",
                                    "SOLARSINDY_ALLOW_UNPINNED_STACK" => "1",
                                    "SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => bundle.dir) do
                    reset_static_stack!(); reset_superlearner_serving!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_unpinned_with_superlearner")
                    end
                end
                reset_static_stack!(); reset_superlearner_serving!()
                @test String(laundered.static_stack_status) == STATIC_STACK_OK_UNPINNED_STATUS
                @test String(laundered.superlearner_status) == "fallback:static_expert_unpinned"
                @test ismissing(laundered.superlearner_pred_dst_nt)
                @test laundered.sub_hourly_model_version == STATIC_STACK_UNPINNED_SERVED_TAIL_VERSION
                @test laundered.sub_hourly_model_version != SUPERLEARNER_SERVED_TAIL_VERSION
                @test laundered.driver_assumption == STATIC_STACK_DRIVER_ASSUMPTION
                @test Float64(laundered.served_pred_dst_nt) ==
                      Float64(laundered.static_stack_pred_dst_nt)
            end
        end

    end

    @testset "Served super-learner stage" begin
        # The served stage is the super-learner over the ten experts, the static stack among
        # them. These cases drive it on a synthetic bundle (the real bundle's numbers are the study's
        # evidence, and the offline identity oracle is what ties this code to them) and then confirm
        # that the deployed bundle serves at all.
        #
        # A day of minute-cadence L1 coverage and 24 h of hourly Dst are the minimum the stage needs:
        # twelve driver lags for the analog key and the run-length window, and a Dst ladder that reaches
        # `dst_lag24`.
        issue_time = DateTime(2026, 7, 15, 12, 30)
        sw_times = collect((issue_time - Hour(24)):Minute(1):issue_time)
        plasma = DataFrame(
            time_tag=sw_times,
            speed=[470.0 + 20.0 * sin(2π * k / 613) for k in eachindex(sw_times)],
            density=[6.0 + 0.8 * sin(2π * k / 421) for k in eachindex(sw_times)],
        )
        mag = DataFrame(
            time_tag=sw_times,
            bz_gsm=[-6.0 + 3.0 * sin(2π * k / 517) for k in eachindex(sw_times)],
            by_gsm=[1.5 * cos(2π * k / 379) for k in eachindex(sw_times)],
        )
        latest_dst_time = _floor_hour(issue_time) - Hour(1)
        dst_times = collect((latest_dst_time - Hour(24)):Hour(1):latest_dst_time)
        dst_values = collect(range(-30.0, -78.0; length=length(dst_times)))

        # One synthetic feed, shared by the end-to-end rows and by the served-stage status matrix, so
        # both exercise the same information set.
        function _superlearner_feed(; span::Period=Hour(24), dst_span::Period=Hour(24))
            feed_times = collect((issue_time - span):Minute(1):issue_time)
            feed_plasma = DataFrame(
                time_tag=feed_times,
                speed=[470.0 + 20.0 * sin(2π * k / 613) for k in eachindex(feed_times)],
                density=[6.0 + 0.8 * sin(2π * k / 421) for k in eachindex(feed_times)],
            )
            feed_mag = DataFrame(
                time_tag=feed_times,
                bz_gsm=[-6.0 + 3.0 * sin(2π * k / 517) for k in eachindex(feed_times)],
                by_gsm=[1.5 * cos(2π * k / 379) for k in eachindex(feed_times)],
            )
            feed_dst_times = collect((latest_dst_time - dst_span):Hour(1):latest_dst_time)
            feed_dst_values = collect(range(-30.0, -78.0; length=length(feed_dst_times)))
            return (plasma=feed_plasma, mag=feed_mag, dst_times=feed_dst_times,
                    dst_values=feed_dst_values)
        end

        function _superlearner_row(dir::AbstractString, name::AbstractString; bundle_dir,
                          stack_path=STATIC_STACK_DEFAULT_STACK_PATH, span::Period=Hour(24),
                          dst_span::Period=Hour(24), model::Symbol=:inertial_sindy, horizon::Int=6)
            feed = _superlearner_feed(; span=span, dst_span=dst_span)
            feed_plasma = feed.plasma
            feed_mag = feed.mag
            feed_dst_times = feed.dst_times
            feed_dst_values = feed.dst_values
            cfg = LiveVerifyConfig(;
                model=model, horizon_hours=horizon, log_path=joinpath(dir, "$(name).csv"),
                report_path=joinpath(dir, "$(name).md"),
            )
            return withenv("SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => bundle_dir,
                           "SOLARSINDY_STATIC_STACK" => stack_path) do
                reset_static_stack!(); reset_superlearner_serving!()
                inputs = prepare_issue_inputs(
                    cfg; issue_time,
                    plasma_fn=() -> feed_plasma, mag_fn=() -> feed_mag,
                    dst_fn=() -> (feed_dst_times, feed_dst_values),
                )
                row = redirect_stdout(devnull) do
                    redirect_stderr(devnull) do
                        issue_forecast(cfg; inputs, write_trajectory=false, verbose=false)
                    end
                    CSV.read(cfg.log_path, DataFrame)[1, :]
                end
                reset_static_stack!(); reset_superlearner_serving!()
                return row
            end
        end

        mktempdir() do root
            fixture = build_superlearner_fixture_bundle(joinpath(root, "bundle"))
            artifacts = load_superlearner_serving_artifacts(fixture.dir)
            inertial_sindy_calibration = read_inertial_sindy_calibration(
                operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv)

            @testset "the super-learner serves the published identity" begin
                row = _superlearner_row(root, "superlearner_served"; bundle_dir=fixture.dir)
                @test row.model_version == "inertial_sindy"
                @test row.sub_hourly_model_version == SUPERLEARNER_SERVED_TAIL_VERSION
                @test row.sub_hourly_model_version ==
                      "dst-superlearner+sindyfloor+conformal"
                @test row.driver_assumption == SUPERLEARNER_DRIVER_ASSUMPTION
                @test row.superlearner_status == "ok"
                @test row.superlearner_model_version == SUPERLEARNER_SERVED_TAIL_VERSION
                @test row.superlearner_manifest_sha256 == artifacts.manifest_sha256
                # Every stage of the fallback chain stays logged, which is what keeps the published
                # severity depth-safe and the change auditable row by row.
                @test isfinite(row.inertial_sindy_served_pred_dst_nt)
                @test isfinite(row.static_stack_pred_dst_nt)
                @test row.static_stack_status == "ok"
                @test isfinite(row.superlearner_l1_center_dst_nt)
                @test isfinite(row.superlearner_t1r_pred_dst_nt)
                @test isfinite(row.direct_gbm_pred_dst_nt)
                @test isfinite(row.climatology_pred_dst_nt)
                @test Int(row.superlearner_history_hours) == SUPERLEARNER_SERVING_DRIVER_LAGS_H
                @test row.superlearner_regime_cell isa AbstractString
                @test occursin("/", String(row.superlearner_regime_cell))
                # The served center is the super-learner center, not the stack center it replaced.
                @test Float64(row.served_pred_dst_nt) == Float64(row.superlearner_pred_dst_nt)
                @test Float64(row.served_pred_dst_nt) != Float64(row.static_stack_pred_dst_nt)
                # The served band is the study's conformal band for this center, disclosed as such.
                @test row.interval_source == SUPERLEARNER_INTERVAL_SOURCE
                # The super-learner override rewrites `interval_source`, so the batch policy behind the
                # frozen-tail band columns is disclosed separately. On a cold log that policy is the
                # static conformal fallback; the adaptive counterpart is pinned end to end by the
                # mature-log cycle test beside the monitor.
                @test row.frozen_tail_interval_source == "conformal"
                @test row.frozen_tail_interval_source != row.interval_source
                @test Float64(row.served_pred_dst_ci05_nt) == Float64(row.superlearner_ci05_nt)
                @test Float64(row.served_pred_dst_ci95_nt) == Float64(row.superlearner_ci95_nt)
                @test row.served_pred_dst_ci05_nt < row.served_pred_dst_nt <
                      row.served_pred_dst_ci95_nt
                # The physical projection is logged, not inferred: the +50/-2000 nT clamp can act on a
                # real anchor state, and without this column a projected center would be
                # indistinguishable from an unprojected one.
                @test row.superlearner_projection_applied == false
                # Each predecessor's own band edge is logged, because the published watch edge is the
                # deepest of them and the served edge. Both partners carry the same half-width, which
                # is the band the pre-super-learner machinery would have served either center under.
                @test isfinite(row.inertial_sindy_served_ci05_dst_nt)
                @test isfinite(row.static_stack_ci05_dst_nt)
                @test Float64(row.inertial_sindy_served_pred_dst_nt) -
                      Float64(row.inertial_sindy_served_ci05_dst_nt) ≈
                      Float64(row.static_stack_pred_dst_nt) -
                      Float64(row.static_stack_ci05_dst_nt) atol=1e-12
                @test Float64(row.inertial_sindy_served_ci05_dst_nt) <
                      Float64(row.inertial_sindy_served_pred_dst_nt)
                # The served conformal half-width is a different number from the band the predecessors
                # served under, which is exactly why the watch edge cannot be the served edge shifted
                # by the change in the center.
                @test !isapprox(Float64(row.served_pred_dst_nt) -
                                Float64(row.served_pred_dst_ci05_nt),
                                Float64(row.inertial_sindy_served_pred_dst_nt) -
                                Float64(row.inertial_sindy_served_ci05_dst_nt); atol=1e-9)
                # Independent recomputation of the whole served stage from the logged state: the frozen
                # expert is the held-driver rollout (recomputed here, as the engine recomputes it), the
                # other experts are the logged panel — the static stack among them — and the cell and
                # interval follow.
                memory = (dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                          dst_delta_3h_nt=Float64(row.dst_delta_3h_nt),
                          Bz_delta_1h_nt=Float64(row.Bz_delta_1h_nt),
                          VBsouth_delta_1h_mvm=Float64(row.VBsouth_delta_1h_mvm),
                          VBsouth_mean_3h_mvm=Float64(row.VBsouth_mean_3h_mvm),
                          Bsouth_mean_3h_nt=Float64(row.Bsouth_mean_3h_nt))
                drivers = (V=Float64(row.V_kms), Bz=Float64(row.Bz_nt), By=Float64(row.By_nt),
                           n=Float64(row.n_cm3), Pdyn=Float64(row.Pdyn_npa))
                baselines = (persistence=Float64(row.persistence_dst_nt),
                             burton=Float64(row.burton_dst_nt),
                             burton_full=Float64(row.burton_full_dst_nt),
                             obrien=Float64(row.obrien_dst_nt))
                frozen = analog_serving_frozen_center(
                    artifacts.analog; inertial_sindy_calibration=inertial_sindy_calibration, issue_drv=drivers,
                    anchor_dst_star=Float64(row.anchor_dst_star_nt),
                    latest_dst=Float64(row.latest_dst_nt), memory=memory, baselines=baselines,
                    model_steps=Int(row.model_step_hours),
                    anchor_time=_parse_dt(row.latest_dst_time_utc),
                )
                experts = (served_inertial_sindy=Float64(row.inertial_sindy_served_pred_dst_nt),
                           frozen_inertial_sindy=frozen.center,
                           t1r_analog=Float64(row.superlearner_t1r_pred_dst_nt),
                           persistence=Float64(row.persistence_dst_nt),
                           burton=Float64(row.burton_dst_nt),
                           burton_full=Float64(row.burton_full_dst_nt),
                           obrien=Float64(row.obrien_dst_nt),
                           direct_gbm=Float64(row.direct_gbm_pred_dst_nt),
                           climatology=Float64(row.climatology_pred_dst_nt),
                           static_stack=Float64(row.static_stack_pred_dst_nt))
                expected = superlearner_serving_center(
                    artifacts; model_steps=Int(row.model_step_hours),
                    latest_dst=Float64(row.latest_dst_nt),
                    dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                    vbsouth_mvm=Float64(row.VBsouth_mvm), experts=experts,
                )
                @test Float64(row.superlearner_l1_center_dst_nt) ≈ expected.l1_center atol=1e-12
                @test Float64(row.superlearner_pred_dst_nt) ≈ expected.center atol=1e-12
                @test Float64(row.superlearner_ci05_nt) ≈ expected.ci05_nt atol=1e-12
                @test Float64(row.superlearner_ci95_nt) ≈ expected.ci95_nt atol=1e-12
                @test row.superlearner_guard_applied == expected.guard_applied
                @test row.superlearner_projection_applied == expected.projection_applied
                # The deployed bundle carries no point-forecast guard, so the published center is the
                # stack center itself — unless the physical projection acted, which is a separate
                # stage and is logged separately. The invariant is therefore conditional on the
                # projection flag rather than assumed unconditionally.
                @test row.superlearner_guard_applied == false
                if !row.superlearner_projection_applied
                    @test Float64(row.superlearner_pred_dst_nt) ≈
                          Float64(row.superlearner_l1_center_dst_nt) atol=1e-12
                end
                @test row.superlearner_deepening_cell == expected.deepening_cell
                @test String(row.superlearner_regime_cell) == expected.regime_cell
                @test row.superlearner_pooled_fallback == expected.used_pooled_fallback
                # The frozen expert is the held-driver rollout, not the logged core center that admits
                # L1-measured hours: substituting the logged column moves the stack center, which is
                # exactly the mistake the recomputation above rules out.
                @test frozen.center != Float64(row.inertial_sindy_pred_dst_nt)
                @test superlearner_serving_center(
                    artifacts; model_steps=Int(row.model_step_hours),
                    latest_dst=Float64(row.latest_dst_nt),
                    dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                    vbsouth_mvm=Float64(row.VBsouth_mvm),
                    experts=merge(experts, (frozen_inertial_sindy=Float64(row.inertial_sindy_pred_dst_nt),)),
                ).l1_center != expected.l1_center
                # The static stack is a weighted input of the served center: perturbing it alone must
                # move the center, which is what distinguishes an expert from a discarded reference.
                @test superlearner_serving_center(
                    artifacts; model_steps=Int(row.model_step_hours),
                    latest_dst=Float64(row.latest_dst_nt),
                    dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                    vbsouth_mvm=Float64(row.VBsouth_mvm),
                    experts=merge(experts,
                                  (static_stack=Float64(row.static_stack_pred_dst_nt) - 25.0,)),
                ).l1_center != expected.l1_center
            end

            @testset "an absent bundle falls back to the static stack and says so" begin
                row = _superlearner_row(root, "superlearner_absent"; bundle_dir=joinpath(root, "no_bundle"))
                @test row.superlearner_status == "fallback:deployment_absent"
                @test row.sub_hourly_model_version == STATIC_STACK_SERVED_TAIL_VERSION
                @test row.driver_assumption == STATIC_STACK_DRIVER_ASSUMPTION
                @test Float64(row.served_pred_dst_nt) == Float64(row.static_stack_pred_dst_nt)
                @test ismissing(row.superlearner_pred_dst_nt)
                @test ismissing(row.superlearner_ci05_nt)
                @test row.superlearner_guard_applied == false
                @test row.superlearner_projection_applied == false
                @test row.interval_source != SUPERLEARNER_INTERVAL_SOURCE
                # The stack is the stage actually serving this row, so its logged edge must be the
                # published edge itself: the depth-safe minimum is then idempotent by construction.
                @test Float64(row.static_stack_ci05_dst_nt) ≈
                      Float64(row.served_pred_dst_ci05_nt) atol=1e-12
            end

            @testset "a tampered bundle is refused rather than served" begin
                tampered = build_superlearner_fixture_bundle(joinpath(root, "tampered");
                                                    mutate=:tampered_stack_file)
                row = _superlearner_row(root, "superlearner_tampered"; bundle_dir=tampered.dir)
                @test row.superlearner_status == "fallback:deployment_invalid"
                @test row.sub_hourly_model_version == STATIC_STACK_SERVED_TAIL_VERSION
                @test Float64(row.served_pred_dst_nt) == Float64(row.static_stack_pred_dst_nt)
            end

            @testset "without the static-stack expert the stage cannot act" begin
                # The static-stack center is expert ten of the served stack, so a cycle whose stack
                # stage fell back cannot form the combination at all and must serve the inertial-SINDy operator
                # rather than a center over an incomplete panel.
                row = _superlearner_row(root, "superlearner_no_reference"; bundle_dir=fixture.dir,
                               stack_path=joinpath(root, "no_such_stack.csv"))
                @test row.static_stack_status == "fallback_inertial_sindy:stack_absent"
                @test ismissing(row.static_stack_pred_dst_nt)
                @test row.superlearner_status == "fallback:static_expert_unavailable"
                @test row.sub_hourly_model_version == OPERATIONAL_SERVED_TAIL_VERSION
                @test row.driver_assumption == OPERATIONAL_DRIVER_ASSUMPTION
                @test Float64(row.served_pred_dst_nt) == Float64(row.inertial_sindy_served_pred_dst_nt)
                @test ismissing(row.superlearner_pred_dst_nt)
                # A stage that could not act publishes no edge either, so its partner column is
                # missing rather than carrying an edge nothing served.
                @test ismissing(row.static_stack_ci05_dst_nt)
                @test Float64(row.inertial_sindy_served_ci05_dst_nt) ≈
                      Float64(row.served_pred_dst_ci05_nt) atol=1e-12
            end

            @testset "a short L1 feed fails the analog key closed" begin
                row = _superlearner_row(root, "superlearner_short_feed"; bundle_dir=fixture.dir, span=Hour(4))
                @test startswith(String(row.superlearner_status), "fallback:missing_driver_lag")
                @test Int(row.superlearner_history_hours) < ANALOG_HISTORY_LAGS_H
                @test ismissing(row.superlearner_pred_dst_nt)
                @test row.sub_hourly_model_version == STATIC_STACK_SERVED_TAIL_VERSION
                @test isfinite(row.served_pred_dst_nt)
            end

            @testset "a short Dst ladder fails the direct expert closed" begin
                # The analog key needs twelve driver hours; the direct design additionally needs the Dst
                # ladder back to `dst_lag24`. A feed that satisfies the first but not the second must
                # name the missing lag instead of serving a stack with an imputed expert.
                row = _superlearner_row(root, "superlearner_short_dst"; bundle_dir=fixture.dir, dst_span=Hour(12))
                @test startswith(String(row.superlearner_status), "fallback:missing_dst_lag")
                @test ismissing(row.superlearner_pred_dst_nt)
                @test row.sub_hourly_model_version == STATIC_STACK_SERVED_TAIL_VERSION
                @test isfinite(row.served_pred_dst_nt)
            end

            @testset "every documented served-stage status is produced by a code path" begin
                # `superlearner_status` is the only per-row account of why the published center is not the
                # super-learner's, and the readiness audit, the dashboard payload and the fallback-rate
                # window all read it. A status no test ever produces is a branch nobody has seen
                # execute, so each documented value is reached here rather than assumed reachable.
                #
                # The early refusals need no valid physics — they are decided before any expert is
                # formed — so they are driven directly with the state a served row logged.
                good = _superlearner_row(root, "superlearner_status_base"; bundle_dir=fixture.dir)
                feed = _superlearner_feed()
                anchor_time = _parse_dt(good.latest_dst_time_utc)
                drivers = (V=Float64(good.V_kms), Bz=Float64(good.Bz_nt), By=Float64(good.By_nt),
                           n=Float64(good.n_cm3), Pdyn=Float64(good.Pdyn_npa))
                memory = (dst_delta_1h_nt=Float64(good.dst_delta_1h_nt),
                          dst_delta_3h_nt=Float64(good.dst_delta_3h_nt),
                          Bz_delta_1h_nt=Float64(good.Bz_delta_1h_nt),
                          VBsouth_delta_1h_mvm=Float64(good.VBsouth_delta_1h_mvm),
                          VBsouth_mean_3h_mvm=Float64(good.VBsouth_mean_3h_mvm),
                          Bsouth_mean_3h_nt=Float64(good.Bsouth_mean_3h_nt))
                baselines = (persistence=Float64(good.persistence_dst_nt),
                             burton=Float64(good.burton_dst_nt),
                             burton_full=Float64(good.burton_full_dst_nt),
                             obrien=Float64(good.obrien_dst_nt))
                features = (dst_delta_1h_nt=Float64(good.dst_delta_1h_nt),
                            VBsouth_mvm=Float64(good.VBsouth_mvm))
                calibration = read_inertial_sindy_calibration(
                    operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv)
                function _stage(; bundle_dir=fixture.dir, model_steps=Int(good.model_step_hours),
                                calibration=calibration, static_stack=Float64(good.static_stack_pred_dst_nt),
                                static_pinned=true, latest_dst=Float64(good.latest_dst_nt),
                                dst_times=feed.dst_times, dst_vals=feed.dst_values,
                                recent=drivers)
                    return withenv("SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => bundle_dir) do
                        reset_superlearner_serving!()
                        result = redirect_stderr(devnull) do
                            _superlearner_served_center(
                                plasma=feed.plasma, mag=feed.mag, dst_times=dst_times,
                                dst_vals=dst_vals, latest_dst_time=anchor_time,
                                latest_dst=latest_dst, anchor_drivers=drivers,
                                anchor_dst_star=Float64(good.anchor_dst_star_nt), recent=recent,
                                latest_common_sw=_parse_dt(good.latest_solar_wind_utc),
                                model_steps=model_steps, features=features, memory=memory,
                                baselines=baselines,
                                served_inertial_sindy=Float64(good.inertial_sindy_served_pred_dst_nt),
                                static_stack=static_stack, calibration=calibration,
                                static_pinned=static_pinned,
                            )
                        end
                        reset_superlearner_serving!()
                        return result
                    end
                end

                produced = String[]
                record(status) = (push!(produced, status); status)

                @test record(String(_stage().status)) == "ok"
                @test record(String(_stage(; model_steps=5).status)) ==
                      "fallback:unsupported_model_step"
                @test record(String(_stage(; calibration=nothing).status)) ==
                      "fallback:calibration_absent"
                @test record(String(_stage(; static_stack=missing).status)) ==
                      "fallback:static_expert_unavailable"
                @test record(String(_stage(; static_stack=NaN).status)) ==
                      "fallback:static_expert_unavailable"
                @test record(String(_stage(; static_pinned=false).status)) ==
                      "fallback:static_expert_unpinned"
                @test record(String(_stage(; bundle_dir=joinpath(root, "no_status_bundle")).status)) ==
                      "fallback:deployment_absent"
                broken = build_superlearner_fixture_bundle(joinpath(root, "status_invalid");
                                                  mutate=:tampered_stack_file)
                @test record(String(_stage(; bundle_dir=broken.dir).status)) ==
                      "fallback:deployment_invalid"
                # The analog key's own refusals, in the precedence the key reports them.
                @test record(String(_stage(; latest_dst=NaN).status)) ==
                      "fallback:missing_anchor_dst"
                @test record(String(_stage(; dst_times=[anchor_time],
                                           dst_vals=[Float64(good.latest_dst_nt)]).status)) ==
                      "fallback:missing_previous_dst"
                # Any exception inside the stage is a status, never a raised error: the served stage
                # must never interrupt an issuance.
                @test record(String(_stage(; recent=nothing).status)) == "fallback:serving_error"
                # A short feed names the missing driver hour, and a short Dst ladder the missing lag.
                @test startswith(record(String(_superlearner_row(root, "superlearner_status_driver";
                                                        bundle_dir=fixture.dir,
                                                        span=Hour(4)).superlearner_status)),
                                 "fallback:missing_driver_lag")
                @test startswith(record(String(_superlearner_row(root, "superlearner_status_dst";
                                                        bundle_dir=fixture.dir,
                                                        dst_span=Hour(12)).superlearner_status)),
                                 "fallback:missing_dst_lag")
                # A SINDy-core issuance never runs the stage at all, which is the row-level default
                # status. A SINDy-core row is issued at a single step, the only lead its band is defined at.
                core_row = _superlearner_row(root, "superlearner_status_sindy_core"; bundle_dir=fixture.dir,
                                             model=:sindy_core, horizon=1)
                @test core_row.model_version != OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION
                @test record(String(core_row.superlearner_status)) == "fallback:model_not_inertial_sindy"
                @test ismissing(core_row.superlearner_pred_dst_nt)

                # `fallback:incomplete_analog_key` is the key's reason of last resort: it fires only if
                # the feature block rejects an anchor for a reason none of the named inputs accounts
                # for. Every single-input defect is enumerated here and every one lands on a named
                # reason instead, which is what makes that reason unreachable rather than untested — the
                # key's own defect finder and the feature block's usability predicate are the same
                # condition on the same five channels. A new rejection condition in the feature block
                # would make it reachable, and would show up here as a defect with no named reason.
                usable = (V=drivers.V, Bz=drivers.Bz, By=drivers.By, n=drivers.n, Pdyn=drivers.Pdyn)
                anchor_dst = Float64(good.latest_dst_nt)
                key_reason(history, dst_anchor, dst_prev) =
                    superlearner_serving_analog_features(artifacts, anchor_time, history, dst_anchor,
                                               dst_prev).reason
                complete_history = Any[usable for _ in 1:SUPERLEARNER_SERVING_DRIVER_LAGS_H]
                @test key_reason(complete_history, anchor_dst, anchor_dst - 1.0) == "ok"
                @test key_reason(complete_history, NaN, anchor_dst - 1.0) == "missing_anchor_dst"
                @test key_reason(complete_history, anchor_dst, NaN) == "missing_previous_dst"
                for lag in (1, 4, ANALOG_HISTORY_LAGS_H)
                    holed = copy(complete_history)
                    holed[lag] = nothing
                    @test key_reason(holed, anchor_dst, anchor_dst - 1.0) ==
                          "missing_driver_lag$(lag)"
                    for defect in (:V, :Bz, :By, :n, :Pdyn)
                        broken_history = copy(complete_history)
                        broken_history[lag] = merge(usable, NamedTuple{(defect,)}((NaN,)))
                        @test key_reason(broken_history, anchor_dst, anchor_dst - 1.0) ==
                              "missing_driver_lag$(lag)"
                    end
                    nonpositive = copy(complete_history)
                    nonpositive[lag] = merge(usable, (n=0.0,))
                    @test key_reason(nonpositive, anchor_dst, anchor_dst - 1.0) ==
                          "missing_driver_lag$(lag)"
                end

                # Every documented status with a reachable code path has now been produced here. Three
                # remain defence-in-depth branches under the deployed loader, and are reported as such
                # rather than faked: `fallback:non_finite_center` (a non-finite expert or combination is
                # refused earlier and a conformal half-width must load positive and finite),
                # `fallback:incomplete_analog_key` (shown above to be unreachable while the feature
                # block's rejection condition is the key's own defect condition), and
                # `fallback:incomplete_direct_design` (the analog key is formed first and refuses the
                # same driver defects at the lags the design's own block reads).
                documented = ["ok", "fallback:model_not_inertial_sindy", "fallback:unsupported_model_step",
                              "fallback:calibration_absent", "fallback:static_expert_unavailable",
                              "fallback:static_expert_unpinned", "fallback:deployment_absent",
                              "fallback:deployment_invalid", "fallback:missing_anchor_dst",
                              "fallback:missing_previous_dst", "fallback:serving_error"]
                for status in documented
                    @test status in produced
                end
                @test any(s -> startswith(s, "fallback:missing_driver_lag"), produced)
                @test any(s -> startswith(s, "fallback:missing_dst_lag"), produced)
                for unreached in ("fallback:non_finite_center", "fallback:incomplete_analog_key",
                                  "fallback:incomplete_direct_design")
                    @test !_superlearner_served_acted(unreached)
                end
            end

            @testset "bundle load failures are remembered for a bounded cool-down" begin
                @test SUPERLEARNER_RETRY_SECONDS > 0.0
                staged = joinpath(root, "staged_bundle")
                withenv("SOLARSINDY_SUPERLEARNER_DEPLOY_DIR" => staged) do
                    reset_superlearner_serving!()
                    absent = redirect_stderr(devnull) do
                        _superlearner_artifacts()
                    end
                    @test absent.artifacts === nothing
                    @test absent.status == "fallback:deployment_absent"
                    @test !_superlearner_served_acted(absent.status)
                    cp(fixture.dir, staged)
                    # Inside the cool-down the remembered failure is reused, so the bundle is not re-read.
                    @test _superlearner_artifacts().artifacts === nothing
                    _SUPERLEARNER_STATUS_TIME[] -= 2 * SUPERLEARNER_RETRY_SECONDS
                    healed = _superlearner_artifacts()
                    @test healed.status == "ok"
                    @test _superlearner_served_acted(healed.status)
                    @test healed.artifacts.identity == SUPERLEARNER_SERVED_IDENTITY
                    @test healed.manifest_sha256 == artifacts.manifest_sha256
                end
                reset_superlearner_serving!()
            end
        end

        @testset "the deployed bundle serves the published product" begin
            deploy_dir = normpath(joinpath(@__DIR__, "..", "deploy", "dst_superlearner"))
            if !isdir(deploy_dir)
                @test_skip "deploy/dst_superlearner is absent; run validation/operational/superlearner_build_deploy.jl"
            else
                mktempdir() do dir
                    row = _superlearner_row(dir, "superlearner_deployed"; bundle_dir=deploy_dir)
                    @test row.superlearner_status == "ok"
                    @test row.sub_hourly_model_version == SUPERLEARNER_SERVED_TAIL_VERSION
                    @test isfinite(row.superlearner_pred_dst_nt)
                    @test isfinite(row.superlearner_l1_center_dst_nt)
                    @test Float64(row.served_pred_dst_nt) == Float64(row.superlearner_pred_dst_nt)
                    @test row.interval_source == SUPERLEARNER_INTERVAL_SOURCE
                    # The deployed bundle carries no point-forecast guard, so the published center is
                    # the stack center itself even in a deepening cell — unless the physical
                    # projection acted, which the row records separately.
                    @test row.superlearner_guard_applied == false
                    if !row.superlearner_projection_applied
                        @test Float64(row.superlearner_pred_dst_nt) ≈
                              Float64(row.superlearner_l1_center_dst_nt) atol=1e-12
                    end
                    # The published severity partners are all present, so the dashboard can take the
                    # deepest of the three without a missing column silently dropping one.
                    for column in (:superlearner_pred_dst_nt, :static_stack_pred_dst_nt,
                                   :inertial_sindy_served_pred_dst_nt)
                        @test isfinite(Float64(getproperty(row, column)))
                    end
                    # Depth safety now lives at the alerting layer: whatever the combination reports,
                    # the severity center is no shallower than either stage it replaced.
                    severity = superlearner_serving_depth_safe_center(
                        Float64(row.superlearner_pred_dst_nt), Float64(row.static_stack_pred_dst_nt),
                        Float64(row.inertial_sindy_served_pred_dst_nt),
                    )
                    @test severity <= Float64(row.static_stack_pred_dst_nt) + 1e-12
                    @test severity <= Float64(row.inertial_sindy_served_pred_dst_nt) + 1e-12
                end
            end
        end
    end

    @testset "Monitor cycle inputs are fetched once and keep one issue timestamp" begin
        issue_time = DateTime(2026, 7, 15, 12, 34, 56)
        plasma_calls = Ref(0)
        mag_calls = Ref(0)
        dst_calls = Ref(0)
        plasma = DataFrame(time_tag=[issue_time], speed=[500.0], density=[5.0])
        mag = DataFrame(time_tag=[issue_time], bz_gsm=[-4.0], by_gsm=[1.0])
        dst = ([issue_time - Hour(1)], [-40.0])
        prepared = prepare_issue_inputs(
            LiveVerifyConfig(; model=:sindy_core);
            issue_time=issue_time,
            plasma_fn=() -> (plasma_calls[] += 1; plasma),
            mag_fn=() -> (mag_calls[] += 1; mag),
            dst_fn=() -> (dst_calls[] += 1; dst),
        )

        @test (plasma_calls[], mag_calls[], dst_calls[]) == (1, 1, 1)
        @test prepared.issue_time == issue_time
        @test prepared.plasma === plasma
        @test prepared.mag === mag
        @test prepared.dst === dst
    end

    @testset "Issue clock follows retrieval and exact upstream evidence survives parsing" begin
        cfg = LiveVerifyConfig(; model=:sindy_core)
        finished = Ref(DateTime(2000))
        prepared = prepare_issue_inputs(cfg; plasma_fn=() -> nothing, mag_fn=() -> nothing,
            dst_fn=() -> (sleep(0.02); finished[]=now(UTC); nothing))
        @test prepared.issue_time >= finished[]
        fixed = DateTime(2026, 1, 1)
        prepared = prepare_issue_inputs(cfg; issue_time=fixed,
            plasma_fn=() -> nothing, mag_fn=() -> nothing, dst_fn=() -> nothing)
        @test prepared.issue_time == fixed
        wind = JSON3.write([(; time_tag=string(fixed + Minute(k)), active=true,
            proton_speed=400.0 + k, proton_density=5.0, proton_temperature=100_000.0)
            for k in 0:1])
        mag = JSON3.write([(; time_tag=string(fixed + Minute(k)), active=true,
            bx_gsm=1.0, by_gsm=2.0, bz_gsm=-3.0, bt=4.0) for k in 0:1])
        dst = JSON3.write([(; time_tag=string(fixed), dst=-10.0)])
        calls = String[]
        response(url; kwargs...) = begin
            push!(calls, url)
            body = occursin("wind_1m", url) ? wind : occursin("mag_1m", url) ? mag : dst
            HTTP.Response(200, body)
        end
        mktempdir() do directory
            isolated_cfg = _copy_config(cfg; log_path=joinpath(directory, "live.csv"))
            result = prepare_issue_inputs(isolated_cfg; snapshot_dir=directory, http_get=response)
            @test result.plasma.speed == [400.0, 401.0]
            @test result.mag.bz_gsm == [-3.0, -3.0]
            @test result.dst == ([fixed], [-10.0])
            @test length(calls) == 3
            path = only(readdir(joinpath(directory, "issues"); join=true))
            receipt = JSON3.read(read(path, String))
            @test DateTime(receipt.issue_time_utc) == result.issue_time
            @test receipt.explicit_replay_time === false
            @test length(receipt.responses) == 3
            for (item, expected) in zip(receipt.responses, (wind, mag, dst))
                bytes = read(joinpath(directory, item.path))
                @test bytes == collect(codeunits(expected))
                @test bytes2hex(sha256(bytes)) == item.sha256
                @test length(bytes) == item.bytes
                @test DateTime(item.started_utc) <= DateTime(item.received_utc) <= result.issue_time
            end
            @test length(readdir(joinpath(directory, "retrievals"))) == 3
            # Content-addressed inputs are immutable, and symlinks cannot redirect evidence.
            raw_path = joinpath(directory, first(receipt.responses).path)
            original = read(raw_path)
            @test _persist_issue_bytes(raw_path, original) == raw_path
            @test_throws ErrorException _persist_issue_bytes(raw_path, UInt8[0x00])
            @test read(raw_path) == original
            link = joinpath(directory, "link")
            # Windows requires Administrator/Developer Mode to create symlinks; when the OS
            # denies creation the symlink-rejection assertions cannot run here and are skipped.
            can_symlink = try
                symlink(raw_path, link)
                true
            catch e
                e isa Base.IOError || rethrow()
                false
            end
            if can_symlink
                @test_throws ArgumentError _persist_issue_bytes(link, original)
                @test_throws ErrorException prepare_issue_inputs(cfg; snapshot_dir=directory,
                    plasma_fn=() -> nothing, mag_fn=() -> nothing, dst_fn=() -> nothing)
            else
                @warn "symlink creation denied by the OS; skipping symlink evidence checks"
            end
        end
    end

    @testset "Forked forecast states share immutable model data and preserve predictions" begin
        t0 = DateTime(2026, 7, 15, 12)
        coefficients = joinpath(get_data_dir(), "real_sindy_discovery_coefficients.csv")
        ensemble = joinpath(get_data_dir(), "real_ensemble_inclusion.csv")
        template = init_forecast(
            coefficients_csv=coefficients,
            ensemble_csv=ensemble,
            t0=t0,
            dst0=-40.0,
        )
        forked = _fork_forecast_state(template, t0, -40.0)

        @test forked.lib === template.lib
        @test forked.ξ_primary === template.ξ_primary
        @test forked.ξ_ensemble === template.ξ_ensemble
        @test forked.history !== template.history
        direct = step_forecast!(template, t0 + Hour(1), 500.0, -6.0, 2.0, 7.0,
                                dynamic_pressure(7.0, 500.0))
        cloned = step_forecast!(forked, t0 + Hour(1), 500.0, -6.0, 2.0, 7.0,
                                dynamic_pressure(7.0, 500.0))
        @test (direct.t, direct.dst_predicted, direct.dst_median,
               direct.dst_ci_05, direct.dst_ci_95) ==
              (cloned.t, cloned.dst_predicted, cloned.dst_median,
               cloned.dst_ci_05, cloned.dst_ci_95)
        @test isnan(direct.dst_observed) && isnan(cloned.dst_observed)
    end

    @testset "Deployed conformal sidecar is paired to the point calibration bytes" begin
        mktempdir() do tmp
            point_path = joinpath(tmp, "point.csv")
            nfeatures = length(OPERATIONAL_MEMORY_EXPERT_LEAD_FEATURES)
            point_cal = InertialSindyCalibration(
                copy(OPERATIONAL_MEMORY_EXPERT_LEAD_FEATURES),
                zeros(nfeatures),
                ones(nfeatures),
                zeros(nfeatures + 1),
                1.0,
                "operational_inertial_sindy_sidecar_test",
                supported_model_steps=copy(OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS),
            )
            write_inertial_sindy_calibration(point_path, point_cal)
            cfg = LiveVerifyConfig(; model=:inertial_sindy, inertial_sindy_calibration_path=point_path)
            @test _load_calibration_for_model(cfg).label == point_cal.label
            @test _load_conformal_for_model(cfg) === nothing

            cal = fit_conformal(
                zeros(20), fill(6.0, 20), ones(20), fill(-10.0, 20),
            )
            sidecar_path = _conformal_path(point_path)
            point_sha = bytes2hex(sha256(read(point_path)))
            write_conformal_calibration(
                sidecar_path, cal; point_calibration_sha256=point_sha,
                supported_model_steps=join(point_cal.supported_model_steps, ";"),
            )
            loaded = _load_conformal_for_model(cfg)
            @test loaded.coverage == cal.coverage
            @test loaded.global_stratum.half_width == cal.global_stratum.half_width

            changed_point = InertialSindyCalibration(
                copy(point_cal.feature_names),
                copy(point_cal.feature_mean),
                copy(point_cal.feature_scale),
                vcat(0.25, point_cal.coefficients[2:end]),
                point_cal.interval_scale,
                point_cal.label,
                supported_model_steps=copy(point_cal.supported_model_steps),
            )
            write_inertial_sindy_calibration(point_path, changed_point)
            @test_throws ArgumentError _load_conformal_for_model(cfg)

            # A byte-matched sidecar with a different discrete support contract
            # is still incompatible and must fail closed.
            write_inertial_sindy_calibration(point_path, point_cal)
            point_sha = bytes2hex(sha256(read(point_path)))
            write_conformal_calibration(
                sidecar_path, cal; point_calibration_sha256=point_sha,
                supported_model_steps="1;2;3;6",
            )
            @test_throws ArgumentError _load_conformal_for_model(cfg)

            # A self-consistent historical SINDy core point/sidecar pair remains valid
            # for explicit offline replay, but the live inertial-SINDy loader rejects it.
            historical = operational_calibration_artifacts(:sindy_core)
            historical_cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                inertial_sindy_calibration_path=historical.point_csv,
            )
            @test isfile(historical.point_csv)
            @test isfile(historical.conformal_csv)
            @test_throws ArgumentError _load_calibration_for_model(historical_cfg)
            @test_throws ArgumentError _load_conformal_for_model(historical_cfg)
        end
    end

    @testset "A/D: append preserves old log rows while adding baseline columns" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            old_row = DataFrame(
                issue_time_utc = ["2026-06-06T02:17:54.992"],
                target_time_utc = ["2026-06-06T03:00:00"],
                pred_dst_nt = [-49.56],
                pred_dst_ci05_nt = [-54.52],
                pred_dst_ci95_nt = [-44.48],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
            )
            CSV.write(log_path, old_row)

            new_row = DataFrame(
                issue_time_utc = ["2026-06-06T04:15:00"],
                target_time_utc = ["2026-06-06T05:00:00"],
                pred_dst_nt = [-50.0],
                pred_dst_ci05_nt = [-55.0],
                pred_dst_ci95_nt = [-45.0],
                persistence_dst_nt = [-49.0],
                burton_dst_nt = [-48.0],
                burton_full_dst_nt = [-47.0],
                obrien_dst_nt = [-46.0],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
            )
            row_idx = _append_forecast!(log_path, new_row)
            df = CSV.read(log_path, DataFrame)

            @test row_idx == 2
            @test nrow(df) == 2
            @test :obrien_dst_nt in propertynames(df)
            @test ismissing(df.obrien_dst_nt[1])
            @test df.obrien_dst_nt[2] == -46.0
        end
    end

    @testset "C0-4: pending duplicate forecast rows are idempotent" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            row = DataFrame(
                issue_time_utc=["2026-06-06T04:10:00"],
                latest_dst_time_utc=["2026-06-06T04:00:00"],
                target_time_utc=["2026-06-06T06:00:00"],
                model_version=["sindy_core"],
                pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-30.0],
                pred_dst_ci95_nt=[-10.0],
                observation_dst_nt=[missing],
            )

            first = _append_forecast!(log_path, row; return_status=true)
            @test first.row_idx == 1
            @test first.appended
            duplicate = copy(row)
            duplicate.issue_time_utc .= "2026-06-06T04:11:00"

            second = _append_forecast!(log_path, duplicate; return_status=true)
            @test second.row_idx == 1
            @test !second.appended
            written = CSV.read(log_path, DataFrame)
            @test nrow(written) == 1
            @test string(written[1, :issue_time_utc]) == "2026-06-06T04:10:00"

            # A later hourly product may forecast the same target from the same provisional
            # Dst anchor. That reissuance is distinct even though its target and model match.
            next_cycle = copy(row)
            next_cycle.issue_time_utc .= "2026-06-06T05:01:00"
            third = _append_forecast!(log_path, next_cycle; return_status=true)
            @test third == (row_idx=2, appended=true)
            @test nrow(CSV.read(log_path, DataFrame)) == 2
        end
    end

    @testset "inertial-SINDy append refuses an unmigrated historical hot log" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "legacy.csv")
            legacy = DataFrame(
                issue_time_utc=["2026-06-06T04:10:00"],
                latest_dst_time_utc=["2026-06-06T04:00:00"],
                target_time_utc=["2026-06-06T06:00:00"],
                model_version=["sindy_core"],
                pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-30.0],
                pred_dst_ci95_nt=[-10.0],
                observation_dst_nt=[-22.0],
            )
            CSV.write(log_path, legacy)
            current = copy(legacy)
            current.issue_time_utc .= "2026-06-06T05:10:00"
            current.target_time_utc .= "2026-06-06T07:00:00"
            current.model_version .= OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION
            before = read(log_path)
            @test_throws ArgumentError _append_forecast!(log_path, current)
            @test read(log_path) == before
        end
    end

    @testset "LOG-01: durable append state recovers and remains idempotent" begin
        mktempdir() do tmp
            empty_path = joinpath(tmp, "empty_current.csv")
            CSV.write(empty_path, DataFrame(
                issue_time_utc=String[], latest_dst_time_utc=String[],
                target_time_utc=String[], model_version=String[],
                pred_dst_nt=Float64[], pred_dst_ci05_nt=Float64[],
                pred_dst_ci95_nt=Float64[], observation_dst_nt=Float64[],
            ))
            empty_state = _load_or_rebuild_live_state!(empty_path)
            @test empty_state["version"] == 3
            @test empty_state["row_count"] == 0
            @test isempty(empty_state["pending"])
            @test isempty(empty_state["aci_streams"])
            @test empty_state["pending_cache_complete"]
            @test !empty_state["has_current_model"]
            @test !empty_state["has_historical_model"]
            @test !empty_state["has_ambiguous_model"]
            @test _state_matches_log(empty_state, empty_path)

            log_path = joinpath(tmp, "live_forecast_log.csv")
            forecast_row(issue, target) = DataFrame(
                issue_time_utc=[string(issue)],
                latest_dst_time_utc=["2026-06-06T04:00:00"],
                target_time_utc=[string(target)],
                model_version=["sindy_core"],
                pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-30.0],
                pred_dst_ci95_nt=[-10.0],
                observation_dst_nt=[missing],
            )
            first = forecast_row(DateTime(2026, 6, 6, 4, 10),
                                 DateTime(2026, 6, 6, 6))
            @test _append_forecast!(log_path, first) == 1

            # A truncated checkpoint is disposable: rebuild its row count and
            # pending-key index from the authoritative CSV, then keep idempotency.
            open(_live_state_path(log_path), "w") do io
                write(io, "{\"version\":")
            end
            duplicate = copy(first)
            duplicate.issue_time_utc .= "2026-06-06T04:11:00"
            result = _append_forecast!(log_path, duplicate; return_status=true)
            @test result == (row_idx=1, appended=false)
            @test nrow(CSV.read(log_path, DataFrame)) == 1
            @test _state_matches_log(_read_live_state(log_path), log_path)

            # Simulate a process death halfway through the one-row append. The
            # durable transaction restores the exact row before duplicate lookup.
            second = forecast_row(DateTime(2026, 6, 6, 5, 10),
                                  DateTime(2026, 6, 6, 7))
            state = _load_or_rebuild_live_state!(log_path)
            schema = Tuple(Symbol.(state["columns"]))
            buffer = IOBuffer()
            CSV.write(buffer, [_project_row(second, schema)]; header=false)
            row_bytes = take!(buffer)
            pre_size = filesize(log_path)
            _atomic_json(_append_transaction_path(log_path), Dict(
                "version" => 1,
                "pre_size" => pre_size,
                "pre_tail_sha256" => _tail_digest(log_path),
                "row_hex" => bytes2hex(row_bytes),
                "row_sha256" => bytes2hex(sha256(row_bytes)),
            ))
            open(log_path, "a") do io
                write(io, row_bytes[1:max(1, length(row_bytes) ÷ 2)])
            end
            GC.gc()          # release mmap'd reads of the log before the recovery truncate
            recovered = _append_forecast!(log_path, second; return_status=true)
            @test recovered == (row_idx=2, appended=false)
            @test !isfile(_append_transaction_path(log_path))
            @test nrow(CSV.read(log_path, DataFrame)) == 2

            # An unreadable transaction is ambiguous and therefore fails closed;
            # the authoritative log is left byte-for-byte unchanged.
            before = read(log_path)
            open(_append_transaction_path(log_path), "w") do io
                write(io, "{")
            end
            @test_throws ErrorException _append_forecast!(
                log_path,
                forecast_row(DateTime(2026, 6, 6, 6, 10),
                             DateTime(2026, 6, 6, 8)),
            )
            @test read(log_path) == before
        end
    end

    @testset "LOG-01: steady-state append allocation is independent of log length" begin
        mktempdir() do tmp
            function verified_log(path, n)
                base = DateTime(2026, 1, 1)
                issues = base .+ Hour.(0:n-1)
                CSV.write(path, DataFrame(
                    issue_time_utc=string.(issues),
                    latest_dst_time_utc=string.(issues),
                    target_time_utc=string.(issues .+ Hour(1)),
                    model_version=fill("sindy_core", n),
                    pred_dst_nt=fill(-20.0, n),
                    pred_dst_ci05_nt=fill(-30.0, n),
                    pred_dst_ci95_nt=fill(-10.0, n),
                    observation_dst_nt=fill(-19.0, n),
                ))
                _load_or_rebuild_live_state!(path)
            end
            function next_row(n)
                issue = DateTime(2026, 1, 1) + Hour(n + 1)
                return DataFrame(
                    issue_time_utc=[string(issue)],
                    latest_dst_time_utc=[string(issue)],
                    target_time_utc=[string(issue + Hour(1))],
                    model_version=["sindy_core"],
                    pred_dst_nt=[-20.0], pred_dst_ci05_nt=[-30.0],
                    pred_dst_ci95_nt=[-10.0], observation_dst_nt=[missing],
                )
            end

            warm = joinpath(tmp, "warm.csv")
            verified_log(warm, 2)
            _append_forecast!(warm, next_row(2))

            small = joinpath(tmp, "small.csv")
            large = joinpath(tmp, "large.csv")
            verified_log(small, 100)
            verified_log(large, 20_000)
            inode_small = stat(small).inode
            inode_large = stat(large).inode
            GC.gc()
            alloc_small = @allocated _append_forecast!(small, next_row(100))
            GC.gc()
            alloc_large = @allocated _append_forecast!(large, next_row(20_000))
            @test stat(small).inode == inode_small
            @test stat(large).inode == inode_large
            @test alloc_large <= 2 * alloc_small + 1_000_000
        end
    end

    @testset "LOG-01: unresolved identity cache is bounded and resolves cleanly" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "pending.csv")
            n = _LIVE_PENDING_CACHE_LIMIT + 200
            base = DateTime(2026, 1, 1)
            anchors = base .+ Hour.(0:n-1)
            targets = anchors .+ Hour(1)
            log = DataFrame(
                issue_time_utc=string.(anchors),
                latest_dst_time_utc=string.(anchors),
                target_time_utc=string.(targets),
                model_version=fill("sindy_core", n),
                pred_dst_nt=fill(-20.0, n),
                pred_dst_ci05_nt=fill(-30.0, n),
                pred_dst_ci95_nt=fill(-10.0, n),
                observation_dst_nt=fill(missing, n),
            )
            # Repeat the oldest identity at the end. Once the bounded cache is
            # incomplete, duplicate lookup must still return the earliest row.
            log[n, :issue_time_utc] = log[1, :issue_time_utc]
            log[n, :latest_dst_time_utc] = log[1, :latest_dst_time_utc]
            log[n, :target_time_utc] = log[1, :target_time_utc]
            CSV.write(log_path, log)
            state = _load_or_rebuild_live_state!(log_path)
            @test length(state["pending"]) == _LIVE_PENDING_CACHE_LIMIT
            @test !state["pending_cache_complete"]
            @test minimum(Int.(values(state["pending"]))) == n - _LIVE_PENDING_CACHE_LIMIT + 1
            @test filesize(_live_state_path(log_path)) < 100_000

            # The oldest identity is outside the cache, but the bounded-memory
            # CSV fallback still preserves exact idempotency.
            old_retry = log[1:1, :]
            old_retry.issue_time_utc .= string(base + Minute(1))
            retry = _append_forecast!(log_path, old_retry; return_status=true)
            @test retry == (row_idx=1, appended=false)

            # Scoring a cached pending row removes it from both the checkpoint
            # and the authoritative pending scan.
            resolved_idx = n - 1
            resolved_key = _pending_key(log[resolved_idx, :issue_time_utc],
                                        log[resolved_idx, :target_time_utc], "inertial_sindy")
            @test verify_pending!(LiveVerifyConfig(; log_path=log_path);
                                  dst_times=[targets[resolved_idx]],
                                  dst_vals=[-19.0]) == 1
            resolved_state = _read_live_state(log_path)
            @test length(resolved_state["pending"]) == _LIVE_PENDING_CACHE_LIMIT
            @test !haskey(resolved_state["pending"], resolved_key)
            @test _find_pending_row(log_path, resolved_key) === nothing
        end
    end

    @testset "C0-5: forecast log lock creates and releases an owned pidfile" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_path = log_path * ".lock"

            result = _with_forecast_log_lock(log_path) do
                @test isfile(lock_path)
                :locked
            end

            @test result == :locked
            @test !ispath(lock_path)
        end
    end

    @testset "C0-6: forecast log lock recovers a dead stale pidfile" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_path = log_path * ".lock"
            write(lock_path, "999999999 dead-test-host")
            sleep(0.03)

            result = _with_forecast_log_lock(
                log_path; stale_after_sec=0.01, poll_sec=0.005,
            ) do
                @test isfile(lock_path)
                :recovered
            end

            @test result == :recovered
            @test !ispath(lock_path)
        end
    end

    @testset "C0-6b: a live owner is refreshed and never reaped" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            entered = Channel{Symbol}(2)
            release = Channel{Nothing}(1)
            first_owner = @async _with_forecast_log_lock(
                log_path; timeout_sec=1.0, stale_after_sec=0.04, poll_sec=0.005,
            ) do
                put!(entered, :first)
                take!(release)
            end
            @test take!(entered) == :first
            sleep(0.08)
            @test _forecast_pidfile_has_local_live_owner(log_path * ".lock")
            second_owner = @async _with_forecast_log_lock(
                log_path; timeout_sec=1.0, stale_after_sec=0.04, poll_sec=0.005,
            ) do
                put!(entered, :second)
            end
            sleep(0.08)
            @test !isready(entered)
            put!(release, nothing)
            wait(first_owner)
            wait(second_owner)
            @test take!(entered) == :second
        end
    end

    @testset "C0-6e: a live foreign owner past the staleness horizon is reclaimed" begin
        # PID-reuse shape: the lock names this process — a live, local PID that is not a forecast
        # daemon — exactly as a lock left by a SIGKILLed daemon whose PID was later recycled would.
        # The ownership short-circuit is the ONLY place Pidfile's staleness rule can be consulted, so
        # it has to apply that rule itself: a fresh lock still holds (unchanged), an unrefreshed one
        # past 5 * stale_after_sec is reclaimed instead of wedging every locked step forever.
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_path = log_path * ".lock"
            write(lock_path, string(getpid(), " ", gethostname()))

            @test _forecast_pidfile_has_local_live_owner(lock_path)
            @test _forecast_pidfile_has_local_live_owner(lock_path; stale_after_sec=900.0)
            @test_throws ErrorException _with_forecast_log_lock(
                log_path; timeout_sec=0.2, stale_after_sec=900.0, poll_sec=0.02,
            ) do
                :must_not_run
            end
            @test isfile(lock_path)

            # Age the same live-PID lock past 5 * stale_after_sec without touching it.
            sleep(0.25)
            @test !_forecast_pidfile_has_local_live_owner(lock_path; stale_after_sec=0.02)
            @test _with_forecast_log_lock(
                log_path; timeout_sec=1.0, stale_after_sec=0.02, poll_sec=0.02,
            ) do
                isfile(lock_path) ? :reclaimed : :missing
            end == :reclaimed
            @test !ispath(lock_path)

            # stale_after_sec == 0 disables Pidfile's staleness rule; mirror that exactly.
            write(lock_path, string(getpid(), " ", gethostname()))
            sleep(0.05)
            @test _forecast_pidfile_has_local_live_owner(lock_path; stale_after_sec=0)
        end
    end

    @testset "C0-6f: the staleness mirror matches Pidfile, future-dated mtime included" begin
        # The ownership short-circuit is the only place Pidfile's staleness rule can run, so the
        # mirror's contract is exactly `!Pidfile.stale_pidfile` for a local live PID. Pin that as a
        # differential oracle against the stdlib rather than restating the rule, and cover the
        # future-dated case (`age < -stale_age`) the age clause alone does not reach: on the pinned
        # Julia the stdlib only WARNS there and does not reclaim, so a mirror that reported the lock
        # stale would diverge from the authority without clearing anything.
        # `open_exclusive` consults `stale_pidfile` only when `stale_age > 0`, so `stale_age == 0`
        # disables the rule at the CALL SITE rather than inside it. The oracle models the call site
        # `_with_forecast_log_lock` actually uses, refresh included.
        mirror_matches_stdlib = (lock_path, stale) -> begin
            expected = !(stale > 0 && Pidfile.stale_pidfile(String(lock_path), stale, stale / 2))
            return _forecast_pidfile_has_local_live_owner(lock_path; stale_after_sec=stale) ==
                   expected
        end

        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_path = log_path * ".lock"
            write(lock_path, string(getpid(), " ", gethostname()))
            # A backwards clock step / skewed filesystem: the lock's mtime lies in the future.
            run(`touch -t 203001010000 $lock_path`)
            _, _, age = Pidfile.parse_pidfile(String(lock_path))
            @test age < 0

            for stale in (0.0, 0.02, 900.0)
                @test mirror_matches_stdlib(lock_path, stale)
            end
            # The stdlib does not reclaim a future-dated lock by age, and the exclusive open — the
            # authority the mirror stands in for — agrees.
            @test Pidfile.trymkpidlock(lock_path; stale_age=900.0, refresh=450.0) === false
            @test _forecast_pidfile_has_local_live_owner(lock_path; stale_after_sec=900.0)

            # The short-circuit swallows the stdlib's own skew warning, so the wedge would otherwise
            # be a silent 30 s stall on every locked step. It has to name its cause.
            wedge = try
                _with_forecast_log_lock(
                    log_path; timeout_sec=0.1, stale_after_sec=900.0, poll_sec=0.02,
                ) do
                    :must_not_run
                end
                nothing
            catch e
                e
            end
            @test wedge isa ErrorException
            @test occursin("FUTURE", wedge.msg)
            @test occursin(lock_path, wedge.msg)
            @test isfile(lock_path)
        end

        # The same oracle in the reclaim direction: an unrefreshed live-PID lock past 5 * stale_age
        # is stale for the stdlib and unowned for the mirror, so a mutation of either clause shows.
        mktempdir() do tmp
            lock_path = joinpath(tmp, "live_forecast_log.csv.lock")
            write(lock_path, string(getpid(), " ", gethostname()))
            sleep(0.25)
            @test Pidfile.stale_pidfile(String(lock_path), 0.02, 0.01)
            for stale in (0.0, 0.02, 900.0)
                @test mirror_matches_stdlib(lock_path, stale)
            end
        end
    end

    @testset "T-17: a malformed clock-skew setting warns and keeps the documented default" begin
        # These resolve at load time in the engine behind the supervised daemon. `error()` here made
        # one typo in a launchd EnvironmentVariables entry abort the load, which launchd turns into
        # a 60 s restart loop that issues nothing. Fail-safe, not fail-to-start; the installer is
        # where the same typo is fatal.
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => nothing) do
            @test _live_minutes_from_env("LIVE_FUTURE_CLOCK_TOLERANCE_MIN", 2) == Minute(2)
        end
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => " 7 ") do
            @test _live_minutes_from_env("LIVE_FUTURE_CLOCK_TOLERANCE_MIN", 2) == Minute(7)
        end
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "0") do
            @test _live_minutes_from_env("LIVE_FUTURE_CLOCK_TOLERANCE_MIN", 2) == Minute(0)
        end
        for bad in ("two", "", "3.5", "-1", "2 minutes")
            withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => bad) do
                value = @test_logs (:warn,) match_mode = :any _live_minutes_from_env(
                    "LIVE_FUTURE_CLOCK_TOLERANCE_MIN", 2,
                )
                @test value == Minute(2)
            end
        end

        # The band is resolved as a pair. An inconsistent pair leaves the "report and exclude" band
        # empty, so it falls back to BOTH documented defaults rather than widening the sanity limit,
        # which would relax a fail-closed guard the operator never asked to relax.
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "30",
                "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "10") do
            band = @test_logs (:warn,) match_mode = :any _live_clock_skew_band()
            @test band.tolerance == Minute(LIVE_FUTURE_CLOCK_TOLERANCE_DEFAULT_MIN)
            @test band.limit == Minute(LIVE_MAX_FUTURE_CLOCK_SKEW_DEFAULT_MIN)
            @test band.limit >= band.tolerance
        end
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => "3",
                "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "30") do
            band = _live_clock_skew_band()
            @test band.tolerance == Minute(3)
            @test band.limit == Minute(30)
        end
        withenv("LIVE_FUTURE_CLOCK_TOLERANCE_MIN" => nothing,
                "LIVE_MAX_FUTURE_CLOCK_SKEW_MIN" => "nope") do
            band = @test_logs (:warn,) match_mode = :any _live_clock_skew_band()
            @test band.tolerance == Minute(LIVE_FUTURE_CLOCK_TOLERANCE_DEFAULT_MIN)
            @test band.limit == Minute(LIVE_MAX_FUTURE_CLOCK_SKEW_DEFAULT_MIN)
        end
        # Whatever the environment held at load, the loaded band is usable.
        @test LIVE_MAX_FUTURE_CLOCK_SKEW >= LIVE_FUTURE_CLOCK_TOLERANCE
        @test LIVE_FUTURE_CLOCK_TOLERANCE >= Minute(0)
    end

    @testset "T-30: an unobservable row is neither a coverage hit nor a coverage miss" begin
        # `min(ci05, NaN) <= NaN <= max(...)` is `false`, indistinguishable from an interval that
        # genuinely excluded the observation. A frame carrying not-yet-verified rows therefore
        # reported them as coverage FAILURES, and every consumer that averaged the column depressed
        # its coverage by the unverified fraction of the frame.
        center = [-20.0, -20.0, -20.0, -20.0]
        frame = DataFrame(
            pred_dst_nt=center,
            pred_dst_ci05_nt=center .- 5.0,
            pred_dst_ci95_nt=center .+ 5.0,
            #      inside      unobserved  outside     unobserved
            observation_dst_nt=[-21.0, NaN, -40.0, NaN],
            Bz_nt=[0.0, 0.0, 0.0, 0.0],
        )
        cal = InertialSindyCalibration([:Bz_nt], [0.0], [1.0], [0.0, 1.0], 1.0, "t30")
        scored = score_inertial_sindy(frame, cal)

        @test eltype(scored.operational_observed_in_90ci) == Union{Missing,Bool}
        @test scored.operational_observed_in_90ci[1] === true
        @test scored.operational_observed_in_90ci[3] === false
        @test ismissing(scored.operational_observed_in_90ci[2])
        @test ismissing(scored.operational_observed_in_90ci[4])
        # An unobservable row is not silently promoted to a hit either.
        @test !any(x -> !ismissing(x) && x, scored.operational_observed_in_90ci[[2, 4]])

        # The denominator excludes it: coverage equals the observable-only average, not 1/4.
        coverage = _observable_coverage(scored.operational_observed_in_90ci)
        @test coverage.n == 2
        @test coverage.fraction == 0.5
        @test coverage.fraction ==
              mean(Bool[scored.operational_observed_in_90ci[i] for i in (1, 3)])

        # A frame with nothing verified yet is reported as unknown, never as zero coverage.
        unverified = score_inertial_sindy(frame[[2, 4], :], cal)
        blank = _observable_coverage(unverified.operational_observed_in_90ci)
        @test blank.n == 0
        @test isnan(blank.fraction)

        # `missing` serialises as an empty cell, matching how the live logger already records an
        # unverified row.
        mktempdir() do tmp
            path = joinpath(tmp, "scored.csv")
            CSV.write(path, scored)
            reread = CSV.read(path, DataFrame)
            @test ismissing(reread.operational_observed_in_90ci[2])
            @test reread.operational_observed_in_90ci[1] === true
            @test reread.operational_observed_in_90ci[3] === false
        end
    end

    @testset "C0-6c: atomic writers preserve directories and symlinks" begin
        mktempdir() do tmp
            directory_target = joinpath(tmp, "output.csv")
            mkdir(directory_target)
            keep = joinpath(directory_target, "keep")
            write(keep, "preserve")
            @test_throws ArgumentError _atomic_csv(
                directory_target, DataFrame(value=[1]),
            )
            @test isdir(directory_target)
            @test read(keep, String) == "preserve"

            json_directory = joinpath(tmp, "state.json")
            mkdir(json_directory)
            json_keep = joinpath(json_directory, "keep")
            write(json_keep, "preserve")
            @test_throws ArgumentError _atomic_json(json_directory, Dict("value" => 1))
            @test isdir(json_directory)
            @test read(json_keep, String) == "preserve"

            if !Sys.iswindows()
                referent = joinpath(tmp, "referent.csv")
                write(referent, "old")
                link = joinpath(tmp, "link.csv")
                symlink(referent, link)
                @test_throws ArgumentError _atomic_csv(link, DataFrame(value=[1]))
                @test islink(link)
                @test read(referent, String) == "old"
            end
        end
    end

    @testset "C0-6d: duplicate cycles preserve the atomic sub-hour trajectory" begin
        mktempdir() do tmp
            path = joinpath(tmp, "subhour_trajectory.json")
            _atomic_json(path, Dict("generation" => "old"))
            old_bytes = read(path)
            payload_calls = Ref(0)

            skipped = _write_subhour_trajectory!(path, false) do
                payload_calls[] += 1
                Dict("generation" => "duplicate")
            end
            @test !skipped
            @test payload_calls[] == 0
            @test read(path) == old_bytes

            written = _write_subhour_trajectory!(path, true) do
                payload_calls[] += 1
                Dict("generation" => "new")
            end
            @test written
            @test payload_calls[] == 1
            @test String(JSON3.read(read(path, String))["generation"]) == "new"
        end
    end

    @testset "sub-hour display trajectory ends at the h=6 target under a lagged Dst anchor" begin
        mktempdir() do dir
            issue_time = DateTime(2026, 7, 15, 12, 30)
            sw_times = collect((issue_time - Hour(12)):Minute(1):issue_time)
            plasma = DataFrame(
                time_tag=sw_times,
                speed=fill(500.0, length(sw_times)),
                density=fill(6.0, length(sw_times)),
            )
            mag = DataFrame(
                time_tag=sw_times,
                bz_gsm=fill(-4.0, length(sw_times)),
                by_gsm=fill(1.0, length(sw_times)),
            )
            # Anchor lags the issue hour by 1 h (the normal live condition): floor(issue)=12:00 but the
            # freshest Kyoto Dst is 11:00, so the served h=6 target 18:00 sits 7 model-steps from the
            # anchor. A fixed 6 h display window would end at 17:00, one hour short of the furthest
            # issued horizon; the trajectory must span the full anchor->target lead.
            latest_dst_time = floor(issue_time, Hour) - Hour(1)
            dst_times = collect((latest_dst_time - Hour(12)):Hour(1):latest_dst_time)
            dst_values = collect(range(-20.0, -44.0; length=length(dst_times)))
            cfg = LiveVerifyConfig(;
                model=:inertial_sindy,
                horizon_hours=6,
                log_path=joinpath(dir, "live_forecast_log.csv"),
                report_path=joinpath(dir, "live_comparison_report.md"),
            )
            inputs = prepare_issue_inputs(
                cfg; issue_time,
                plasma_fn=() -> plasma,
                mag_fn=() -> mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            target_time = _next_hourly_target(issue_time, 6, latest_dst_time)
            @test (target_time - latest_dst_time) / Hour(1) == 7
            redirect_stdout(devnull) do
                issue_forecast(cfg; inputs, write_trajectory=true, verbose=false)
            end
            sidecar = joinpath(dir, "subhour_trajectory.json")
            @test isfile(sidecar)
            payload = JSON3.read(read(sidecar, String))
            points = payload["points"]
            @test !isempty(points)
            @test DateTime(String(payload["anchor_time_utc"])) == latest_dst_time
            @test DateTime(String(last(points)["t"])) == target_time
        end
    end

    @testset "C0-7: verify_pending! re-reads under the lock; a concurrent append is not lost" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_dir = log_path * ".lock"
            past = DateTime(2026, 6, 6, 5)
            future = DateTime(2026, 6, 6, 9)
            row(target) = DataFrame(
                issue_time_utc=["2026-06-06T04:00:00"],
                latest_dst_time_utc=["2026-06-06T04:00:00"],
                target_time_utc=[string(target)],
                model_version=["sindy_core"],
                pred_dst_nt=[-30.0], pred_dst_ci05_nt=[-50.0], pred_dst_ci95_nt=[-10.0],
                observation_dst_nt=[missing],
            )
            CSV.write(log_path, row(past))                       # P1, pending

            # Simulate an in-progress locked writer (e.g. the live monitor appending a forecast) that
            # already holds the directory lock. verify_pending! must block on it, then re-read the log.
            mkdir(lock_dir)
            task = @async verify_pending!(LiveVerifyConfig(; log_path=log_path);
                                          dst_times=[past], dst_vals=[-40.0])
            sleep(0.3)                                           # let the async verify reach the lock poll loop
            @test !istaskdone(task)                              # blocked on the held lock (did not clobber yet)

            # Concurrent append of P2 while the lock is held here.
            df2 = vcat(CSV.read(log_path, DataFrame), row(future); cols=:union)
            _atomic_csv(log_path, df2)
            rm(lock_dir; recursive=true, force=true)             # release: verify now proceeds
            verified = fetch(task)

            out = CSV.read(log_path, DataFrame)
            @test verified == 1                                  # P1 scored
            @test nrow(out) == 2                                 # P2 survived — no lost update
            # CSV round-trips target_time_utc back to DateTime, so match by parsed time.
            tgt = _parse_dt.(out.target_time_utc)
            p1 = out[tgt .== past, :]
            p2 = out[tgt .== future, :]
            @test nrow(p1) == 1 && nrow(p2) == 1
            @test p1.observation_dst_nt[1] == -40.0              # P1 got the observation
            @test ismissing(p2.observation_dst_nt[1])            # P2 still pending, not clobbered
        end
    end

    @testset "A/D: refresh_observations! reconciles revised Dst without changing predictions" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            target = DateTime(2026, 6, 7, 16)
            log = DataFrame(
                issue_time_utc=["2026-06-07T15:26:39.864"],
                latest_dst_time_utc=["2026-06-07T14:00:00"],
                target_time_utc=[string(target)],
                model_version=["sindy_core"],
                pred_dst_nt=[-23.4],
                pred_dst_ci05_nt=[-45.0],
                pred_dst_ci95_nt=[-1.0],
                observation_dst_nt=[-22.0],
                residual_dst_nt=[1.4],
                observed_in_90ci=[true],
                inertial_sindy_pred_dst_nt=[-23.4],
                inertial_sindy_pred_dst_ci05_nt=[-45.0],
                inertial_sindy_pred_dst_ci95_nt=[-1.0],
                persistence_dst_nt=[-25.0],
                burton_dst_nt=[-30.0],
                burton_full_dst_nt=[-30.0],
                obrien_dst_nt=[-28.0],
            )
            CSV.write(log_path, log)

            cfg = LiveVerifyConfig(; mode=:refresh_observations, log_path=log_path)
            updated = refresh_observations!(
                cfg;
                dst_times=[target],
                dst_vals=[-21.0],
            )
            df = CSV.read(log_path, DataFrame)

            @test updated == 1
            @test df.pred_dst_nt[1] == -23.4
            @test df.observation_dst_nt[1] == -21.0
            @test df.residual_dst_nt[1] ≈ 2.4 atol=1e-12
            @test df.operational_residual_dst_nt[1] ≈ 2.4 atol=1e-12
            @test df.operational_observed_in_90ci[1] == true
            @test df.persistence_residual_dst_nt[1] == 4.0
        end
    end

    @testset "A/D: comparison report uses verified rows and separates pending rows" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            report_path = joinpath(tmp, "comparison.md")
            log = DataFrame(
                issue_time_utc=[
                    "2026-06-06T06:06:43.957",
                    "2026-06-06T07:23:05.548",
                    "2026-06-06T09:15:00",
                ],
                target_time_utc=[
                    "2026-06-06T08:00:00",
                    "2026-06-06T08:00:00",
                    "2026-06-06T10:00:00",
                ],
                model_version=["sindy_core", "sindy_core", "sindy_core"],
                wall_clock_lead_hours=[1.89, 0.62, 0.75],
                pred_dst_nt=[-39.38, -39.12, -35.0],
                pred_dst_ci05_nt=[-44.94, -42.23, -40.0],
                pred_dst_ci95_nt=[-33.82, -36.01, -30.0],
                observation_dst_nt=Union{Missing,Float64}[-33.0, -33.0, missing],
                residual_dst_nt=Union{Missing,Float64}[6.38, 6.12, missing],
                observed_in_90ci=Union{Missing,Bool}[false, false, missing],
                sindy_core_pred_dst_nt=[-40.63, -40.42, -36.0],
                inertial_sindy_pred_dst_nt=[-39.38, -39.12, -35.0],
                inertial_sindy_selected_component=["sindy_core", "sindy_core", "sindy_core"],
                persistence_dst_nt=[-44.0, -44.0, -34.0],
                burton_dst_nt=[-34.30, -34.20, -33.0],
                burton_full_dst_nt=[-34.30, -34.20, -33.0],
                obrien_dst_nt=[-39.91, -39.81, -35.0],
            )
            CSV.write(log_path, log)

            out = write_live_comparison_report(log_path, report_path)
            text = read(report_path, String)

            @test out == report_path
            @test occursin("Verified rows used across all served identities: 2", text)
            @test occursin("Invalid verified rows excluded: 0", text)
            @test occursin("Pending rows: 1", text)
            @test occursin("Same-row current-identity comparison rows: 2", text)
            @test occursin("## Same-Row Model Comparison", text)
            @test occursin("| Historical SINDy core | 2 |", text)
            @test occursin("| SINDy core | 2 |", text)
            @test occursin("## Pending Rows", text)
            @test occursin("2026-06-06T10:00:00", text)
            @test occursin("| sindy_core |", text)
            @test occursin("## Worst Historical SINDy core Misses", text)
            @test occursin("## Operational SINDy core Audit", text)
            @test !occursin("| Historical SINDy core | 3 |", text)
            @test !occursin("| Selected |", text)
        end
    end

    @testset "A/D: verify_pending! scores SINDy and baseline residuals" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            target = DateTime(2026, 6, 6, 5)
            row = DataFrame(
                issue_time_utc = ["2026-06-06T04:15:00"],
                target_time_utc = [string(target)],
                pred_dst_nt = [-50.0],
                pred_dst_ci05_nt = [-55.0],
                pred_dst_ci95_nt = [-45.0],
                persistence_dst_nt = [-49.0],
                burton_dst_nt = [-48.0],
                burton_full_dst_nt = [-47.0],
                obrien_dst_nt = [-46.0],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
                persistence_residual_dst_nt = [missing],
                burton_residual_dst_nt = [missing],
                burton_full_residual_dst_nt = [missing],
                obrien_residual_dst_nt = [missing],
            )
            CSV.write(log_path, row)

            cfg = LiveVerifyConfig(mode=:verify_pending, log_path=log_path)
            n_verified = verify_pending!(cfg;
                dst_times=[target],
                dst_vals=[-63.0],
            )
            df = CSV.read(log_path, DataFrame)

            @test n_verified == 1
            @test df.observation_dst_nt[1] == -63.0
            @test df.residual_dst_nt[1] == -13.0
            @test df.observed_in_90ci[1] == false
            @test df.persistence_residual_dst_nt[1] == -14.0
            @test df.burton_residual_dst_nt[1] == -15.0
            @test df.burton_full_residual_dst_nt[1] == -16.0
            @test df.obrien_residual_dst_nt[1] == -17.0
        end
    end

    @testset "A/D: backfill_baselines! upgrades legacy verified rows" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            row = DataFrame(
                issue_time_utc = ["2026-06-06T04:00:34.31"],
                latest_solar_wind_utc = ["2026-06-06T03:57:00"],
                latest_dst_time_utc = ["2026-06-06T03:00:00"],
                latest_dst_nt = [-49.0],
                anchor_dst_star_nt = [-44.66383751003725],
                target_time_utc = ["2026-06-06T04:00:00"],
                horizon_hours = [-0.009530555555555556],
                driver_assumption = ["legacy"],
                V_kms = [584.2931034482758],
                Bz_nt = [-1.821551724137931],
                By_nt = [2.879655172413793],
                n_cm3 = [1.494655172413793],
                Pdyn_npa = [0.8534825033123322],
                pred_dst_star_nt = [-43.64275511399697],
                pred_dst_nt = [-47.93566823580722],
                pred_dst_ci05_nt = [-52.31545743478508],
                pred_dst_ci95_nt = [-43.37255925892829],
                observation_dst_nt = [-63.0],
                residual_dst_nt = [-15.06433176419278],
                observed_in_90ci = [false],
            )
            CSV.write(log_path, row)

            n_backfilled = backfill_baselines!(log_path)
            df = CSV.read(log_path, DataFrame)

            @test n_backfilled == 1
            @test df.model_step_hours[1] == 1
            @test df.persistence_dst_nt[1] == -49.0
            @test isfinite(df.burton_dst_nt[1])
            @test isfinite(df.burton_full_dst_nt[1])
            @test isfinite(df.obrien_dst_nt[1])
            @test df.persistence_residual_dst_nt[1] == -14.0
            @test df.burton_residual_dst_nt[1] ≈ -63.0 - df.burton_dst_nt[1] atol=1e-12
            @test df.burton_full_residual_dst_nt[1] ≈ -63.0 - df.burton_full_dst_nt[1] atol=1e-12
            @test df.obrien_residual_dst_nt[1] ≈ -63.0 - df.obrien_dst_nt[1] atol=1e-12
        end
    end

    @testset "M9: multi-step backfill baselines match an independent Euler oracle" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            V, Bz, Pdyn, anchor = 500.0, -8.0, 2.5, -60.0
            Bs = max(-Bz, 0.0)
            row = DataFrame(
                issue_time_utc = ["2026-06-06T00:10:00"],
                latest_dst_time_utc = ["2026-06-06T00:00:00"],
                latest_dst_nt = [-50.0],
                anchor_dst_star_nt = [anchor],
                target_time_utc = ["2026-06-06T03:00:00"],   # 3 model steps
                V_kms = [V], Bz_nt = [Bz], By_nt = [2.0], n_cm3 = [6.0], Pdyn_npa = [Pdyn],
                pred_dst_nt = [-55.0],
                pred_dst_ci05_nt = [-60.0],
                pred_dst_ci95_nt = [-50.0],
                observation_dst_nt = [-58.0],
            )
            CSV.write(log_path, row)
            @test backfill_baselines!(log_path) == 1
            df = CSV.read(log_path, DataFrame)
            @test df.model_step_hours[1] == 3

            # Independent forward-Euler oracle (dt = 1 hr, same clamps as _advance_baselines).
            n_steps = 3
            advance(model, star) = begin
                for _ in 1:n_steps
                    d = clamp(model([V], [Bs], [star])[1], -200.0, 200.0)
                    star = clamp(star + d, -2000.0, 50.0)
                end
                star
            end
            to_dst(star) = star + 7.26 * sqrt(max(Pdyn, 0.0)) - 11.0
            @test df.persistence_dst_nt[1] == -50.0
            @test df.burton_dst_nt[1] ≈ to_dst(advance(burton_model, anchor)) atol = 1e-9
            @test df.burton_full_dst_nt[1] ≈ to_dst(advance(burton_model_full, anchor)) atol = 1e-9
            @test df.obrien_dst_nt[1] ≈ to_dst(advance(obrien_mcpherron_model, anchor)) atol = 1e-9

            # Fill-if-missing (M6): a second backfill must NOT change issued values.
            burton_before = df.burton_dst_nt[1]
            @test backfill_baselines!(log_path) == 0
            df2 = CSV.read(log_path, DataFrame)
            @test df2.burton_dst_nt[1] == burton_before
        end
    end

    @testset "A/D: replay_recent_table builds causal predicted-vs-observed rows" begin
        t0 = DateTime(2026, 6, 6, 0)
        times = collect(t0:Hour(1):t0 + Hour(4))
        minute_times = [t + Minute(m) for t in times for m in 0:9]
        hour_idx = repeat(0:4; inner=10)
        plasma = DataFrame(
            time_tag=minute_times,
            density=4.0 .+ 0.2 .* hour_idx,
            speed=410.0 .+ 10.0 .* hour_idx,
            temperature=fill(100_000.0, length(minute_times)),
        )
        mag = DataFrame(
            time_tag=minute_times,
            bx_gsm=zeros(length(minute_times)),
            by_gsm=1.0 .+ 0.1 .* hour_idx,
            bz_gsm=-1.0 .- 0.2 .* hour_idx,
            bt=1.4 .+ 0.2 .* hour_idx,
        )
        dst_vals = [-20.0, -21.0, -23.0, -22.0, -24.0]

        df = replay_recent_table(plasma, mag, times, dst_vals; replay_hours=24)

        @test nrow(df) == 3
        @test all(df.model_version .== "sindy_core")
        @test df.issue_time_utc[1] == string(t0 + Hour(1))
        @test df.source_driver_start_utc[1] == string(t0)
        @test df.source_driver_end_utc[1] == df.issue_time_utc[1]
        @test df.target_time_utc[1] == string(t0 + Hour(2))
        @test all(df.target_time_utc .> df.issue_time_utc)
        @test all(isfinite, df.pred_dst_nt)
        @test df.pred_dst_nt == df.sindy_core_pred_dst_nt
        @test all(ismissing, df.inertial_sindy_pred_dst_nt)
        @test all(isfinite, df.obrien_dst_nt)
        @test :dst_delta_1h_nt in propertynames(df)
        @test :baseline_spread_nt in propertynames(df)
        @test df.dst_delta_1h_nt[1] == 0.0
        @test df.residual_dst_nt[1] ≈ df.observation_dst_nt[1] - df.pred_dst_nt[1] atol=1e-12
        @test df.persistence_residual_dst_nt[1] ≈ df.observation_dst_nt[1] - df.persistence_dst_nt[1] atol=1e-12

        cal = default_inertial_sindy_calibration()
        df_operational = replay_recent_table(
            plasma,
            mag,
            times,
            dst_vals;
            replay_hours=24,
            model=:inertial_sindy,
            calibration=cal,
        )
        @test all(df_operational.model_version .== OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)
        @test df_operational.pred_dst_nt == df_operational.inertial_sindy_pred_dst_nt
        @test df_operational.pred_dst_nt == df_operational.sindy_core_pred_dst_nt
        @test all(df_operational.inertial_sindy_correction_dst_nt .== 0.0)
        @test all(df_operational.inertial_sindy_selected_component .== "inertial_sindy")
        @test df_operational.inertial_sindy_selected_component_pred_nt == df_operational.inertial_sindy_pred_dst_nt

        mktempdir() do tmp
            md_path = joinpath(tmp, "replay.md")
            write_markdown_table(md_path, df; limit=2)
            text = read(md_path, String)
            @test occursin("target_time_utc", text)
            @test count(==('\n'), text) == 4
        end

        # 1b-iii / M7: multi-horizon replay emits one row per (anchor, horizon)
        # whose target observation exists, tags model_step_hours, and leaves the
        # h=1 forecast numerically identical to the single-horizon table (the
        # forecast_ahead refactor is equivalent for one step).
        df_mh = replay_recent_table(plasma, mag, times, dst_vals;
                                    replay_hours=24, horizons=[1, 2])
        @test Set(unique(df_mh.model_step_hours)) == Set([1, 2])
        h1 = df_mh[df_mh.model_step_hours .== 1, :]
        h2 = df_mh[df_mh.model_step_hours .== 2, :]
        @test nrow(h1) == 3                       # anchors t0+1..t0+3, target +1 present
        @test nrow(h2) == 2                       # anchors t0+1..t0+2, target +2 present
        @test sort(h1.pred_dst_nt) ≈ sort(df.pred_dst_nt) atol = 1e-9
        # Longer lead departs STRICTLY further from the anchor persistence value. A regression that
        # returns the 1-step forecast for every horizon (e.g. fc[1] instead of fc[n_steps], or a
        # dropped primary-trajectory update) leaves h=2 == h=1, so a finiteness-only check would miss
        # it; the strict inequality on the matching anchors pins the multi-step propagation.
        for r2 in eachrow(h2)
            @test isfinite(r2.pred_dst_nt)
            r1 = h1[h1.issue_time_utc .== r2.issue_time_utc, :]
            @test nrow(r1) == 1
            dep2 = abs(r2.pred_dst_nt - r2.persistence_dst_nt)
            dep1 = abs(r1.pred_dst_nt[1] - r1.persistence_dst_nt[1])
            @test dep2 > dep1
        end
        @test_throws ArgumentError replay_recent_table(plasma, mag, times, dst_vals;
                                                       replay_hours=24, horizons=Int[])

        mktempdir() do tmp
            omni_path = joinpath(tmp, "omni.csv")
            raw = DataFrame(
                year=fill(2026, 5),
                doy=fill(157, 5),
                hour=0:4,
                By=[1.0, 1.1, 1.2, 1.3, 1.4],
                Bz=[-1.0, -1.2, -1.4, -1.6, -1.8],
                T=fill(100_000.0, 5),
                n=[4.0, 4.2, 4.4, 4.6, 4.8],
                V=[410.0, 420.0, 430.0, 440.0, 450.0],
                Pdyn=[1.13, 1.24, 1.36, 1.49, 1.62],
                Dst=[-20.0, -21.0, -23.0, -22.0, -24.0],
                AE=fill(100.0, 5),
                AL=fill(-50.0, 5),
                AU=fill(50.0, 5),
            )
            CSV.write(omni_path, raw)

            plasma_omni, mag_omni, dst_times_omni, dst_vals_omni =
                _omni_replay_inputs(omni_path, 2026, 2026)

            @test nrow(plasma_omni) == 5
            @test nrow(mag_omni) == 5
            @test dst_times_omni[1] == t0
            @test dst_vals_omni[end] == -24.0
            @test plasma_omni.speed[2] == 420.0
            @test mag_omni.bz_gsm[3] == -1.4
        end
    end

    @testset "A/D: replay gap gate is cadence-aware (hourly archival replay)" begin
        # Regression for the archival-replay blocker (adv-paper-dst-superlearner MINOR): the
        # driver-gap gate demanded >=10 finite samples per hourly window, which hourly
        # OMNI (1 sample/hour) can never meet, so replay_recent_table dropped every
        # anchor and run_conformal_coverage_test.jl could not run end-to-end. The gate is
        # now cadence-aware: a 1-min feed still resolves to the live floor (never
        # weakened), while hourly replay resolves to 1.
        t0 = DateTime(2026, 6, 6, 0)

        # 1-min cadence fixture: median inter-sample spacing 1 min -> live floor retained.
        hours5 = collect(t0:Hour(1):t0 + Hour(4))
        minute_times = [t + Minute(m) for t in hours5 for m in 0:9]
        plasma_1m = DataFrame(time_tag=minute_times,
                              density=fill(4.0, length(minute_times)),
                              speed=fill(410.0, length(minute_times)),
                              temperature=fill(1.0e5, length(minute_times)))
        @test _replay_min_hourly_samples(plasma_1m) == LIVE_MIN_HOURLY_DRIVER_SAMPLES

        # Hourly (archival OMNI) cadence: 1 sample/hour -> cadence-aware floor 1.
        hours = collect(t0:Hour(1):t0 + Hour(6))
        plasma_h = DataFrame(time_tag=hours, density=fill(4.0, 7),
                             speed=fill(410.0, 7), temperature=fill(1.0e5, 7))
        mag_h = DataFrame(time_tag=hours, bx_gsm=zeros(7), by_gsm=fill(1.0, 7),
                          bz_gsm=fill(-2.0, 7), bt=fill(2.2, 7))
        dst_vals = collect(-20.0:-1.0:-26.0)
        @test _replay_min_hourly_samples(plasma_h) == 1

        df = replay_recent_table(plasma_h, mag_h, hours, dst_vals;
                                 replay_hours=24, horizons=[1])
        @test nrow(df) >= 1                       # archival hourly replay now produces rows

        # Explicit live floor (parameterized override) still rejects every hourly anchor,
        # so no row is scored and the function raises the same "no replay rows" error that
        # blocked run_conformal_coverage_test.jl before the cadence-aware default — proof
        # the gate binds when configured for 1-min cadence.
        @test_throws ErrorException replay_recent_table(
            plasma_h, mag_h, hours, dst_vals;
            replay_hours=24, horizons=[1], min_samples=LIVE_MIN_HOURLY_DRIVER_SAMPLES)
        @test_throws ArgumentError replay_recent_table(plasma_h, mag_h, hours, dst_vals;
                                                       replay_hours=24, min_samples=0)
    end

    @testset "A/D: fit_inertial_sindy_calibration! writes calibration and scored replay rows" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "inertial_sindy_calibration.csv")
            pred = collect(-30.0:1.0:-11.0)
            bz = collect(-10.0:1.0:9.0)
            observed = pred .+ 2.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:length(pred)],
                pred_dst_nt=pred .+ 20.0,
                pred_dst_ci05_nt=pred .+ 17.0,
                pred_dst_ci95_nt=pred .+ 23.0,
                observation_dst_nt=observed,
                sindy_core_pred_dst_nt=pred,
                sindy_core_pred_dst_ci05_nt=pred .- 3.0,
                sindy_core_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=pred .- 1.0,
                V_kms=fill(420.0, length(pred)),
                Bz_nt=bz,
                By_nt=fill(1.0, length(pred)),
                n_cm3=fill(5.0, length(pred)),
                Pdyn_npa=fill(1.5, length(pred)),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_inertial_sindy_calibration,
                table_path=table_path,
                inertial_sindy_calibration_path=cal_path,
                operational_train_fraction=0.8,
                operational_ridge_grid=[0.0],
                operational_ridge=0.0,
            )
            cal = fit_inertial_sindy_calibration!(cfg)
            @test isfile(cal_path)
            scored_path = replace(cal_path, r"\.csv$" => "_scored.csv")
            selection_path = replace(cal_path, r"\.csv$" => "_selection.csv")
            @test isfile(scored_path)
            @test isfile(selection_path)
            # A constant +2 correction generalizes, so operational is selected and deployed.
            @test startswith(cal.label, "operational_inertial_sindy_")
            @test cal.selected_component == :inertial_sindy
            reread = read_inertial_sindy_calibration(cal_path)
            scored = CSV.read(scored_path, DataFrame)
            selection = CSV.read(selection_path, DataFrame)
            @test maximum(abs.(scored.operational_residual_dst_nt)) < 0.5
            @test Set(scored.operational_split) == Set(["fit", "validation", "holdout"])
            # Leakage-free audit schema: no ensemble/holdout-shrink columns.
            @test :gate_pass in propertynames(selection)
            @test :acceptance_gate_pass in propertynames(selection)
            @test :holdout_coverage in propertynames(selection)
            @test :beats_preupgrade in propertynames(selection)
            @test :beats_persistence in propertynames(selection)
            @test :holdout_shrink_alpha ∉ propertynames(selection)
            @test any(selection.selected_by_validation)
            @test any(selection.deployed)          # passed the gate → deployed
            @test all(selection.acceptance_gate_pass)
            pred_operational = operational_inertial_sindy_predict(
                reread,
                pred[end],
                pred[end] - 3.0,
                pred[end] + 3.0,
                _inertial_sindy_features(pred[end] - 1.0, (; V=420.0, Bz=bz[end], By=1.0, n=5.0, Pdyn=1.5)),
            )
            @test pred_operational.pred_dst == observed[end]

            # N1: a conformal calibration sidecar is written and round-trips.
            conformal_path = replace(cal_path, r"\.csv$" => "_conformal.csv")
            @test isfile(conformal_path)
            cc = read_conformal_calibration(conformal_path)
            @test cc.coverage == cfg.inertial_sindy_interval_coverage
            @test cc.global_stratum.n >= 1
            # Half-width is a nonnegative finite interval radius.
            hw = conformal_halfwidth(cc, 1.0, pred[end] - 1.0)
            @test isfinite(hw) && hw >= 0.0

            # 1b-ii: the issue-time interval resolver uses conformal when present.
            center, latest = -45.0, -47.0
            lo, hi, src = _resolve_interval(cc, center, 1, latest, -999.0, 999.0)
            @test src == "conformal"
            @test (lo, hi) == conformal_interval(cc, center, 1.0, latest)
            # Interval is centered on the point (width ≥ 0; this toy has an exact
            # +2 correction, so residuals ≈ 0 → a valid degenerate zero-width band).
            @test lo <= center <= hi
            @test isapprox((lo + hi) / 2, center; atol=1e-9)
            # Without conformal, it passes through the supplied interval unchanged.
            lo0, hi0, src0 = _resolve_interval(nothing, center, 1, latest, -50.0, -40.0)
            @test src0 == "interval_scale"
            @test (lo0, hi0) == (-50.0, -40.0)

            # Non-degenerate check: a conformal calibration with a known 6 nT
            # half-width yields a ±6 nT interval around the center.
            cc6 = fit_conformal(zeros(40), vcat(fill(6.0, 38), [6.0, 6.0]),
                                fill(1.0, 40), fill(0.0, 40); coverage=0.90, min_stratum_n=5)
            lo6, hi6, _ = _resolve_interval(cc6, -20.0, 1, 0.0, NaN, NaN)
            @test hi6 - lo6 ≈ 12.0 atol = 1e-9
            @test isapprox((lo6 + hi6) / 2, -20.0; atol=1e-9)
        end
    end

    @testset "A/D: acceptance gate deploys SINDy-core-equivalent fallback when correction fails validation" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "inertial_sindy_calibration.csv")
            n = 24
            pred = collect(-50.0:1.0:-27.0)
            observed = copy(pred)
            # +4 correction on the training portion only; it does NOT generalize
            # to the later validation rows, so operational must fail the acceptance gate.
            observed[1:14] .= pred[1:14] .+ 4.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                sindy_core_pred_dst_nt=pred,
                sindy_core_pred_dst_ci05_nt=pred .- 3.0,
                sindy_core_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_inertial_sindy_calibration,
                table_path=table_path,
                inertial_sindy_calibration_path=cal_path,
                operational_train_fraction=0.6,
                operational_validation_fraction=0.2,
                operational_ridge_grid=[0.0],
                operational_ridge=0.0,
            )
            cal = fit_inertial_sindy_calibration!(cfg)
            scored = CSV.read(replace(cal_path, r"\.csv$" => "_scored.csv"), DataFrame)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            # Gate failed → a SINDy-core-equivalent (zero-correction) fallback is deployed.
            @test cal.label == "operational_inertial_sindy_fallback_v1_equiv"
            @test all(cal.coefficients .== 0.0)            # no correction applied
            @test !any(selection.deployed)                 # nothing passed the gate
            @test all(.!selection.acceptance_gate_pass)
            @test !any(selection.gate_pass)
            # Deployed operational reduces exactly to the SINDy core (the correction was rejected).
            @test scored.inertial_sindy_pred_dst_nt == scored.pred_dst_nt
        end
    end

    @testset "F1+F2: Operational conformal interval undercoverage on holdout blocks deploy" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "inertial_sindy_calibration.csv")
            n = 30
            pred = collect(-60.0:1.0:-31.0)
            observed = copy(pred)
            # +4 holds through train AND validation, but not the holdout. The legacy
            # validation gate passes (the interval_scale band over-covers there), yet
            # the OPERATIONALLY-SERVED conformal interval — fit on near-zero validation
            # residuals — under-covers the untouched holdout where the +4 correction
            # breaks down. F1+F2 gates the served-interval holdout coverage, so operational must
            # NOT deploy; the SINDy-core-equivalent fallback ships instead.
            observed[1:21] .= pred[1:21] .+ 4.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                sindy_core_pred_dst_nt=pred,
                sindy_core_pred_dst_ci05_nt=pred .- 3.0,
                sindy_core_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_inertial_sindy_calibration,
                table_path=table_path,
                inertial_sindy_calibration_path=cal_path,
                operational_train_fraction=0.5,
                operational_validation_fraction=0.2,
                operational_ridge_grid=[0.0],
                operational_ridge=0.0,
            )
            cal = fit_inertial_sindy_calibration!(cfg)
            scored = CSV.read(replace(cal_path, r"\.csv$" => "_scored.csv"), DataFrame)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            holdout = scored[scored.operational_split .== "holdout", :]
            # Honest holdout (scored once for selection, gated once for the served
            # interval) is much worse than validation and undercovers — the served
            # conformal interval gate fires and the SINDy-core-equivalent fallback deploys.
            @test cal.label == "operational_inertial_sindy_fallback_v1_equiv"
            @test all(cal.coefficients .== 0.0)            # no correction applied
            @test !any(selection.deployed)                 # served-interval gate blocked deploy
            @test all(.!selection.acceptance_gate_pass)
            @test selection.deploy_block_reason[1] == "conformal_holdout_undercover"
            @test selection.conformal_holdout_coverage[1] < cfg.operational_coverage_floor
            @test !selection.conformal_gate_pass[1]
            # The Operational conformal interval's holdout coverage drove the block: the
            # candidate operational (+4) under-covers the untouched holdout where the +4
            # correction breaks down, so the gate refused it.
            @test selection.conformal_holdout_coverage[1] == 0.0
            # The deployed (fallback) operational reduces exactly to the SINDy core.
            @test scored.inertial_sindy_pred_dst_nt == scored.pred_dst_nt
        end
    end

    @testset "A/D: campaign mode issues, verifies, and reports requested horizons" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "campaign.csv")
            report_path = joinpath(tmp, "campaign.md")
            targets = DateTime(2026, 6, 6, 10):Hour(1):DateTime(2026, 6, 6, 12)
            cfg = LiveVerifyConfig(;
                mode=:campaign,
                model=:sindy_core,
                poll_seconds=1,
                timeout_hours=0.1,
                campaign_horizons=[1, 2],
                log_path=log_path,
                report_path=report_path,
            )

            function fake_issue(issue_cfg)
                row_idx = isfile(log_path) ? nrow(CSV.read(log_path, DataFrame)) + 1 : 1
                target = collect(targets)[row_idx]
                pred = -30.0 - row_idx
                row = DataFrame(
                    issue_time_utc=["2026-06-06T09:00:00"],
                    latest_dst_time_utc=["2026-06-06T09:00:00"],
                    target_time_utc=[string(target)],
                    model_version=["sindy_core"],
                    wall_clock_lead_hours=[Float64(issue_cfg.horizon_hours)],
                    pred_dst_nt=[pred],
                    pred_dst_ci05_nt=[pred - 10.0],
                    pred_dst_ci95_nt=[pred + 10.0],
                    observation_dst_nt=[missing],
                    residual_dst_nt=[missing],
                    observed_in_90ci=[missing],
                    sindy_core_pred_dst_nt=[pred - 1.0],
                    inertial_sindy_pred_dst_nt=[pred],
                    inertial_sindy_pred_dst_ci05_nt=[pred - 10.0],
                    inertial_sindy_pred_dst_ci95_nt=[pred + 10.0],
                    inertial_sindy_selected_component=["sindy_core"],
                    persistence_dst_nt=[-30.0],
                    burton_dst_nt=[-29.0],
                    burton_full_dst_nt=[-29.0],
                    obrien_dst_nt=[-31.0],
                )
                idx = _append_forecast!(log_path, row)
                return (; row_idx=idx, latest_dst_time=DateTime("2026-06-06T09:00:00"),
                        target_time=target, pred_dst=pred,
                        ci05_dst=pred - 10.0, ci95_dst=pred + 10.0,
                        model_version="sindy_core")
            end

            function fake_verify(verify_cfg)
                df = CSV.read(verify_cfg.log_path, DataFrame)
                n_verified = 0
                for row_idx in 1:nrow(df)
                    ismissing(df[row_idx, :observation_dst_nt]) || continue
                    _score_row!(df, row_idx, df[row_idx, :pred_dst_nt] + 1.0)
                    n_verified += 1
                end
                CSV.write(verify_cfg.log_path, df)
                return n_verified
            end

            result = run_campaign(
                cfg;
                issue_fn=fake_issue,
                verify_fn=fake_verify,
                sleep_fn=_ -> nothing,
            )
            df = CSV.read(log_path, DataFrame)
            text = read(report_path, String)
            @test result.rows == [1, 2]
            @test nrow(df) == 2
            @test all(!ismissing, df.observation_dst_nt)
            @test occursin("Same-row current-identity comparison rows: 2", text)
            @test occursin("Historical SINDy core is the archived operational method", text)
        end
    end

    @testset "C0-3: live report headlines the exact current served identity" begin
        mktempdir() do dir
            log_path = joinpath(dir, "operational_log.csv")
            report_path = joinpath(dir, "operational_report.md")
            df = DataFrame(
                issue_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                latest_dst_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                target_time_utc=["2026-06-06T11:00:00", "2026-06-06T12:00:00"],
                model_version=fill(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, 2),
                model_step_hours=[1, 2],
                wall_clock_lead_hours=[2.0, 2.0],
                horizon_hours=[2.0, 2.0],
                pred_dst_nt=[-40.0, -45.0],
                pred_dst_ci05_nt=[-50.0, -55.0],
                pred_dst_ci95_nt=[-30.0, -35.0],
                observation_dst_nt=[-48.0, -49.0],
                residual_dst_nt=[-8.0, -4.0],
                observed_in_90ci=[true, true],
                sindy_core_pred_dst_nt=[-38.0, -43.0],
                inertial_sindy_pred_dst_nt=[-40.0, -45.0],
                inertial_sindy_pred_dst_ci05_nt=[-50.0, -55.0],
                inertial_sindy_pred_dst_ci95_nt=[-30.0, -35.0],
                served_pred_dst_nt=[-47.0, -50.0],
                served_pred_dst_ci05_nt=[-57.0, -60.0],
                served_pred_dst_ci95_nt=[-37.0, -40.0],
                served_residual_dst_nt=[-1.0, 1.0],
                served_observed_in_90ci=[true, true],
                static_stack_pred_dst_nt=[-46.0, -48.0],
                sub_hourly_model_version=fill(SUPERLEARNER_SERVED_TAIL_VERSION, 2),
                inertial_sindy_selected_component=["inertial_sindy", "inertial_sindy"],
                persistence_dst_nt=[-39.0, -44.0],
                burton_dst_nt=[-41.0, -44.0],
                burton_full_dst_nt=[-41.0, -44.0],
                obrien_dst_nt=[-46.0, -48.0],
            )
            CSV.write(log_path, df)
            write_live_comparison_report(log_path, report_path)
            text = read(report_path, String)
            @test occursin("Current served identity: super-learner (`$(SUPERLEARNER_SERVED_TAIL_VERSION)`)", text)
            @test occursin("super-learner 90% interval coverage", text)
            @test occursin("| super-learner | 2 |", text)
            @test occursin("| Static stack predecessor | 2 | 1.58 | 1.5 | -1.5 |", text)
            @test occursin("| inertial-SINDy frozen-tail ablation | 2 |", text)
            @test occursin("## Same-Row Model Comparison by Internal Step", text)
            @test occursin("| 1 | super-learner | 1 | 1.0 | 1.0 | -1.0 |", text)
            @test occursin("| 1 | Static stack predecessor | 1 | 2.0 | 2.0 | -2.0 |", text)
            @test occursin("| 2 | super-learner | 1 | 1.0 | 1.0 | 1.0 |", text)
            @test occursin("inertial-SINDy frozen-tail pred", text)
            @test !occursin("inertial-SINDy is the dashboard forecast", text)

            # Earlier served identities stay separate from the current headline cohort.
            mixed_history = vcat(first(df, 1), df)
            mixed_history.issue_time_utc[1] = "2026-06-06T08:00:00"
            mixed_history.latest_dst_time_utc[1] = "2026-06-06T08:00:00"
            mixed_history.target_time_utc[1] = "2026-06-06T10:00:00"
            mixed_history.sub_hourly_model_version[1] = OPERATIONAL_SERVED_TAIL_VERSION
            mixed_history_path = joinpath(dir, "mixed_served_history.csv")
            mixed_history_report = joinpath(dir, "mixed_served_history.md")
            CSV.write(mixed_history_path, mixed_history)
            write_live_comparison_report(mixed_history_path, mixed_history_report)
            mixed_text = read(mixed_history_report, String)
            @test occursin("Verified rows used across all served identities: 3", mixed_text)
            @test occursin("Verified rows under the current served identity set: 2", mixed_text)
            @test occursin("Same-row current-identity comparison rows: 2", mixed_text)
            @test occursin("| inertial-SINDy | `$(OPERATIONAL_SERVED_TAIL_VERSION)` | 1 |", mixed_text)
            @test occursin("| super-learner | `$(SUPERLEARNER_SERVED_TAIL_VERSION)` | 2 |", mixed_text)

            # Expanded legacy rows can contain populated served_* columns. The
            # persisted model identity, not column presence, controls labeling.
            historical = copy(df)
            historical.model_version .= "sindy_core"
            select!(historical, Not(:sub_hourly_model_version))
            historical_path = joinpath(dir, "historical_with_served.csv")
            historical_report = joinpath(dir, "historical_with_served.md")
            CSV.write(historical_path, historical)
            write_live_comparison_report(historical_path, historical_report)
            historical_text = read(historical_report, String)
            @test occursin("Historical SINDy core is the archived operational method", historical_text)
            @test occursin("| Historical SINDy core | 2 |", historical_text)
            @test occursin("| Historical SINDy core frozen-tail ablation | 2 |", historical_text)
            @test !occursin("inertial-SINDy is the dashboard forecast", historical_text)

            empty_current_path = joinpath(dir, "empty_current.csv")
            empty_current_report = joinpath(dir, "empty_current.md")
            CSV.write(empty_current_path, first(df, 0))
            write_live_comparison_report(
                empty_current_path,
                empty_current_report;
                empty_identity=:inertial_sindy,
            )
            empty_current_text = read(empty_current_report, String)
            @test occursin("Newest issued forecast: none", empty_current_text)
            @test occursin("inertial-SINDy is the recorded base method", empty_current_text)
            @test occursin("Verified rows used across all served identities: 0", empty_current_text)
            @test !occursin(
                "Historical SINDy core is the archived operational method",
                empty_current_text,
            )
            @test_throws ArgumentError write_live_comparison_report(
                empty_current_path,
                empty_current_report;
                empty_identity=:unknown,
            )

            mixed = copy(df)
            mixed.model_version[2] = "sindy_core"
            @test_throws ArgumentError write_live_comparison_report(
                log_path,
                report_path;
                df=mixed,
            )
        end
    end

    @testset "G09-03: inertial-SINDy-fallback cohort is compared without the absent static-static stack center" begin
        mktempdir() do dir
            n = 4
            issue = [DateTime(2026, 7, 1, 10, 5) + Hour(i) for i in 1:n]
            df = DataFrame(
                issue_time_utc=string.(issue),
                latest_dst_time_utc=string.(_floor_hour.(issue)),
                target_time_utc=string.(_floor_hour.(issue) .+ Hour(1)),
                model_version=fill(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, n),
                model_step_hours=fill(1, n), wall_clock_lead_hours=fill(0.9, n),
                sub_hourly_model_version=fill(OPERATIONAL_SERVED_TAIL_VERSION, n),
                observation_dst_nt=[-20.0, -22, -25, -30],
                pred_dst_nt=[-19.0, -21, -24, -28],
                pred_dst_ci05_nt=fill(-40.0, n), pred_dst_ci95_nt=fill(0.0, n),
                observed_in_90ci=fill(true, n),
                inertial_sindy_pred_dst_nt=[-19.0, -21, -24, -28],
                inertial_sindy_pred_dst_ci05_nt=fill(-40.0, n), inertial_sindy_pred_dst_ci95_nt=fill(0.0, n),
                sindy_core_pred_dst_nt=[-18.0, -20, -23, -27],
                persistence_dst_nt=[-19.0, -20, -22, -25],
                burton_dst_nt=[-19.0, -20, -22, -25],
                burton_full_dst_nt=[-19.0, -20, -22, -25],
                obrien_dst_nt=[-19.0, -20, -22, -25],
                served_pred_dst_nt=[-20.0, -22, -25, -29],
                served_pred_dst_ci05_nt=fill(-40.0, n), served_pred_dst_ci95_nt=fill(0.0, n),
                static_stack_status=fill("fallback_inertial_sindy:stack_absent", n),
                static_stack_pred_dst_nt=Vector{Union{Missing,Float64}}(missing, n),
            )
            log_path = joinpath(dir, "fallback_log.csv")
            report_path = joinpath(dir, "fallback_report.md")
            CSV.write(log_path, df)
            write_live_comparison_report(log_path, report_path)
            text = read(report_path, String)
            @test occursin("Same-row current-identity comparison rows: 4", text)
            @test occursin("| inertial-SINDy | 4 |", text)
            @test occursin("| Persistence | 4 |", text)
            @test !occursin("| Static stack predecessor |", text)

            summary_text = open(joinpath(dir, "fallback_summary.txt"), "w+") do io
                redirect_stdout(io) do
                    summarize_log(log_path)
                end
                seekstart(io)
                read(io, String)
            end
            @test occursin(r"inertial-SINDy\s+n=4 RMSE=", summary_text)
            @test occursin(r"Persistence\s+n=4 RMSE=", summary_text)
            @test !occursin("no verified rows", summary_text)
        end
    end

    @testset "Historical archive report preserves served/frozen SINDy core roles" begin
        mktempdir() do dir
            historical_log = joinpath(
                get_data_dir(), "historical", "sindy_core", "live_forecast_log.csv",
            )
            report_path = joinpath(dir, "historical_report.md")
            write_live_comparison_report(historical_log, report_path)
            text = read(report_path, String)
            @test occursin("Same-row current-identity comparison rows: 1533", text)
            @test occursin("Historical SINDy core 90% interval coverage: 0.88", text)
            @test occursin("| Historical SINDy core | 1533 | 9.54 |", text)
            @test occursin(
                "| Historical SINDy core frozen-tail ablation | 1533 | 9.85 |",
                text,
            )
            @test !occursin("inertial-SINDy is the dashboard forecast", text)

            summary_text = open(joinpath(dir, "historical_summary.txt"), "w+") do io
                redirect_stdout(io) do
                    summarize_log(historical_log)
                end
                seekstart(io)
                read(io, String)
            end
            @test occursin(r"Historical SINDy core\s+n=1533 RMSE=9\.54", summary_text)
            @test occursin(
                r"Historical SINDy core frozen-tail ablation\s+n=1533 RMSE=9\.85",
                summary_text,
            )
            @test occursin("Historical SINDy core 90% coverage n=1533 coverage=0.88", summary_text)
        end
    end

    @testset "C1: _window_finite_count detects solar-wind data gaps" begin
        plasma = DataFrame(
            time_tag = [DateTime(2026, 6, 6, 0, 5), DateTime(2026, 6, 6, 0, 35),
                        DateTime(2026, 6, 6, 1, 5)],
            speed = [400.0, NaN, 420.0],
            density = [5.0, 5.0, 5.0],
        )
        t0 = DateTime(2026, 6, 6, 0)
        t1 = DateTime(2026, 6, 6, 1)
        # One finite speed sample in [00:00, 01:00) (the NaN does not count).
        @test _window_finite_count(plasma, :speed, t0, t1) == 1
        # Empty window → zero finite samples (the data-gap signal that makes
        # issue_forecast refuse rather than fabricate quiet drivers).
        @test _window_finite_count(plasma, :speed, DateTime(2026, 6, 6, 3),
                                   DateTime(2026, 6, 6, 4)) == 0
        # Missing column → zero, never an error.
        @test _window_finite_count(plasma, :bz_gsm, t0, t1) == 0

        # The live hourly mean needs ten finite minute samples. Pin the exact
        # brownout boundary: 0/1/9 are insufficient; 10 is accepted.
        @test LIVE_MIN_HOURLY_DRIVER_SAMPLES == 10
        for n in (0, 1, 9)
            @test _driver_gap_status(n, 10) == :hard
            @test _driver_gap_status(10, n) == :hard
            @test _driver_gap_status(10, 10, n, 10) == :partial
            @test _driver_gap_status(10, 10, 10, n) == :partial
        end
        @test _driver_gap_status(10, 10) == :ok
        @test _driver_gap_status(0, 0) == :hard
        @test _driver_gap_status(10, 10, 10, 10) == :ok
        @test_throws ArgumentError _driver_gap_status(-1, 10)
    end

    @testset "P1-1: an all-NaN density/By trailing window flags a driver gap" begin
        # Trailing hour [00:00, 01:00): finite speed and Bz, but density and By are
        # entirely NaN. Pre-fix, the gap classifier ignored density/By and reported
        # :ok, so `_drivers_for_window` silently substituted n=5/By=0 quiet defaults
        # and fabricated Pdyn/clock-angle terms with no flag. Post-fix the missing
        # density/By trailing samples must classify the window as a (partial) gap.
        t0 = DateTime(2026, 6, 6, 0)
        t1 = DateTime(2026, 6, 6, 1)
        times = t0 .+ Minute.(0:9)
        plasma = DataFrame(time_tag=times, speed=fill(415.0, 10), density=fill(NaN, 10))
        mag = DataFrame(time_tag=times, bz_gsm=fill(-2.5, 10), by_gsm=fill(NaN, 10))

        n_speed = _window_finite_count(plasma, :speed, t0, t1)
        n_bz = _window_finite_count(mag, :bz_gsm, t0, t1)
        n_density = _window_finite_count(plasma, :density, t0, t1)
        n_by = _window_finite_count(mag, :by_gsm, t0, t1)
        @test n_speed == 10 && n_bz == 10
        # The logged finite-counts for the fabricated drivers are exactly zero.
        @test n_density == 0
        @test n_by == 0
        # Speed+Bz present but density empty → flagged as a data gap (not :ok).
        @test _driver_gap_status(n_speed, n_bz, n_density, n_by) != :ok
        @test _driver_gap_status(10, 10, 0, 10) == :partial   # density-only gap
        @test _driver_gap_status(10, 10, 10, 0) == :partial   # By-only gap
        # No gap when all four drivers have finite trailing samples.
        @test _driver_gap_status(10, 10, 10, 10) == :ok
    end

    @testset "P1-2: intermediate all-NaN driver hours increment the fallback counter" begin
        # Mirror the issuance multi-step loop predicate: each intermediate hour whose
        # window has no finite speed OR no finite Bz falls back to frozen persistence
        # drivers and must be counted (silent pre-fix). n_steps = 3 with one all-NaN
        # intermediate hour ⇒ count > 0; the same span with all hours finite ⇒ 0.
        anchor = DateTime(2026, 6, 6, 0)
        n_steps = 3
        # Hours [0,1),[1,2),[2,3). Each good hour has exactly ten samples;
        # hour [1,2) is all-NaN for every driver.
        ptimes = [anchor + Hour(h) + Minute(m) for h in 0:2 for m in 0:9]
        hour_idx = repeat(0:2; inner=10)
        middle_gap = hour_idx .== 1
        plasma_gap = DataFrame(
            time_tag=ptimes,
            speed=ifelse.(middle_gap, NaN, 410.0 .+ 10.0 .* hour_idx),
            density=ifelse.(middle_gap, NaN, 5.0),
        )
        mag_gap = DataFrame(
            time_tag=ptimes,
            bz_gsm=ifelse.(middle_gap, NaN, -1.0 .- hour_idx),
            by_gsm=ifelse.(middle_gap, NaN, 1.0),
        )

        # Sum the source-of-truth per-step predicate the issuance loop uses.
        count_fallback(plasma, mag) = sum(
            _step_driver_fallback(plasma, mag, anchor + Hour(step - 1)) ? 1 : 0
            for step in 1:n_steps
        )
        @test count_fallback(plasma_gap, mag_gap) > 0
        @test count_fallback(plasma_gap, mag_gap) == 1   # exactly the one all-NaN hour
        # The middle hour is exactly the flagged one; the finite hours are not.
        @test _step_driver_fallback(plasma_gap, mag_gap, anchor + Hour(1))
        @test !_step_driver_fallback(plasma_gap, mag_gap, anchor)

        plasma_ok = DataFrame(time_tag=ptimes, speed=410.0 .+ 10.0 .* hour_idx,
                              density=fill(5.0, length(ptimes)))
        mag_ok = DataFrame(time_tag=ptimes, bz_gsm=-1.0 .- hour_idx,
                           by_gsm=fill(1.0, length(ptimes)))
        @test count_fallback(plasma_ok, mag_ok) == 0

        # Exact per-hour threshold for the shared fallback predicate.
        for n in (0, 1, 9)
            @test _step_driver_fallback(plasma_ok[1:n, :], mag_ok[1:n, :], anchor)
        end
        @test !_step_driver_fallback(plasma_ok[1:10, :], mag_ok[1:10, :], anchor)
    end

    @testset "P1-3/C4-PKG-03: issuance remains within the calibrated product support" begin
        # Multi-step SINDy core loops step_forecast!, whose band is ~5× too narrow vs the
        # forecast_ahead propagation, so it must be refused at issuance.
        @test_throws ArgumentError _assert_issuable_model(:sindy_core, 2)
        @test_throws ArgumentError _assert_issuable_model(:sindy_core, 6)
        @test _assert_issuable_model(:sindy_core, 1) === nothing    # single-step SINDy-core issuance is fine
        @test _assert_issuable_model(:inertial_sindy, 6) === nothing    # operational serves a conformal band
        @test _assert_issuable_model(:inertial_sindy, 1) === nothing
        @test_throws ArgumentError _assert_issuable_model(:inertial_sindy, 4)
        @test_throws ArgumentError _assert_issuable_model(:inertial_sindy, 99)

        supported = default_inertial_sindy_calibration(
            supported_model_steps=copy(OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS),
        )
        for step in OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS
            @test _assert_supported_model_step(supported, step) === nothing
        end
        @test_throws ArgumentError _assert_supported_model_step(supported, 0)
        @test_throws ArgumentError _assert_supported_model_step(supported, 5)
        @test_throws ArgumentError _assert_supported_model_step(supported, 8)
        @test _supported_horizon_edges([1, 2, 3, 6]) ==
              [0.0, 1.5, 2.5, 4.5, Inf]
        @test _supported_horizon_edges(OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS) ==
              [0.0, 1.5, 2.5, 3.5, 5.0, 6.5, Inf]
        @test_throws ArgumentError _supported_horizon_edges(Int[])
        @test_throws ArgumentError _supported_horizon_edges([0, 1])

        issue = DateTime(2026, 8, 10, 12)
        actual_steps = Int[]
        for anchor_lag in 0:LIVE_MAX_DST_ANCHOR_LAG_STEPS,
            product_horizon in OPERATIONAL_PRODUCT_HORIZONS
            anchor = issue - Hour(anchor_lag)
            target = _next_hourly_target(issue, product_horizon, anchor)
            step = Int((target - anchor) / Hour(1))
            @test step in OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS
            push!(actual_steps, step)
        end
        @test sort(unique(actual_steps)) == OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS
    end

    @testset "Operational tail: regime-aware relaxation and finite interval shift" begin
        driver = (V=420.0, Bz=-12.0, By=4.0, n=6.0, Pdyn=2.5)
        tau_recovery = _operational_tail_tau(5.0)
        tau_deepening = _operational_tail_tau(-30.0)
        @test tau_deepening > tau_recovery
        @test tau_deepening <= OPERATIONAL_TAIL_TAU_MAX_H
        @test OPERATIONAL_SERVED_TAIL_VERSION == "inertial-sindy+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"

        # Closed-form tau law (all exactly representable, so == is safe): recovery => no scaling,
        # deepening => tau0*(1+|rate|/r0), saturation => cap. Pins the formula SHAPE and constants so
        # a sqrt-shape or an r0/tau0 rescale (which preserve the orderings above) is caught.
        @test _operational_tail_tau(5.0) == 3.0                       # OPERATIONAL_TAIL_TAU0_H
        @test _operational_tail_tau(-30.0) == 15.0                    # 3*(1 + 30/7.5)
        @test _operational_tail_tau(-10_000.0) == OPERATIONAL_TAIL_TAU_MAX_H   # cap saturation

        relaxed_recovery = _relaxed_tail_driver(driver, 1, 5.0)
        relaxed_deepening = _relaxed_tail_driver(driver, 1, -30.0)
        # Recovery relaxes transverse IMF toward quiet; active deepening preserves
        # more southward/east-west field from the same last-known driver.
        @test relaxed_recovery.V == driver.V
        @test relaxed_recovery.n == driver.n
        @test relaxed_recovery.Pdyn == driver.Pdyn
        @test abs(relaxed_recovery.Bz) < abs(driver.Bz)
        @test abs(relaxed_recovery.By) < abs(driver.By)
        @test abs(relaxed_deepening.Bz) > abs(relaxed_recovery.Bz)
        @test abs(relaxed_deepening.By) > abs(relaxed_recovery.By)
        # Exact e-folding: relax = exp(-hours/tau); pins the hours exponent and tau together.
        @test relaxed_deepening.Bz ≈ driver.Bz * exp(-1 / 15) atol = 1e-12
        @test relaxed_deepening.By ≈ driver.By * exp(-1 / 15) atol = 1e-12
        @test _relaxed_tail_driver(driver, 2, -30.0).Bz ≈ driver.Bz * exp(-2 / 15) atol = 1e-12

        lo, hi = _shift_interval_to_center(-90.0, -80.0, -100.0, -60.0)
        @test (lo, hi) == (-110.0, -70.0)
        @test_throws ArgumentError _shift_interval_to_center(NaN, -80.0, -100.0, -60.0)
        @test _served_interval_with_source(-110.0, -70.0, "conformal", nothing) ==
              (-110.0, -70.0, "conformal")
        @test _served_interval_with_source(-110.0, -70.0, "conformal", (-120.0, -60.0)) ==
              (-120.0, -60.0, "aci")
        @test_throws ArgumentError _served_interval_with_source(
            -110.0, -70.0, "conformal", (NaN, -60.0),
        )
        @test_throws ArgumentError _served_interval_with_source(
            -110.0, -70.0, "conformal", (-60.0, -120.0),
        )
        @test _served_driver_assumption(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION) ==
              OPERATIONAL_DRIVER_ASSUMPTION
        @test _served_driver_assumption("sindy_core") == SINDY_CORE_DRIVER_ASSUMPTION
        @test _served_driver_assumption("v1") == SINDY_CORE_DRIVER_ASSUMPTION
        @test_throws ArgumentError _served_driver_assumption("v0")
        @test _near_term_extreme_inertia_guard(-250.0, 1)
        @test _near_term_extreme_inertia_guard(-250.0, 2)
        @test _near_term_extreme_inertia_guard(-240.0, 2)      # exact inclusive (<=) threshold boundary
        @test !_near_term_extreme_inertia_guard(-250.0, 0)     # 0 < model_steps lower bound
        @test !_near_term_extreme_inertia_guard(-250.0, 3)
        @test !_near_term_extreme_inertia_guard(-239.9, 2)
        @test _one_hour_inertia_blend(-120.0, -100.0, 1) == -115.0
        @test _one_hour_inertia_blend(-120.0, -100.0, 2) == -120.0
        @test _one_hour_inertia_blend(-120.0, -100.0, 1; weight=0.0) == -100.0
        @test _one_hour_inertia_blend(-120.0, -100.0, 1; weight=1.0) == -120.0
        @test_throws ArgumentError _one_hour_inertia_blend(-120.0, -100.0, 1; weight=1.01)
        # Order-sensitive oracle: Eq. (13) projects raw 100 nT to 50 nT
        # before the rapid-rate and one-hour inertia maps, yielding -65 nT.
        # Applying the safeguards to raw 100 first would incorrectly give -27.5.
        @test _apply_inertial_sindy_safeguards(100.0, -50.0, 1, -60.0) == -65.0
        @test _apply_inertial_sindy_safeguards(
            100.0, -50.0, 1, -60.0; apply_rate_guard=false,
        ) == 25.0

        t0 = DateTime(2026, 6, 6, 0)
        plasma = DataFrame(time_tag=[t0 + Minute(5)], speed=[410.0], density=[5.0])
        mag = DataFrame(time_tag=[t0 + Minute(5)], bz_gsm=[-3.0], by_gsm=[1.0])
        s = _subhourly_driver_with_status(plasma, mag, t0 + Hour(2), driver, t0)
        @test !s.l1_measured
        @test s.driver == driver
    end

    @testset "Operational tail: L1 look-ahead measured branch blends the ballistic source window" begin
        # recent.V = 500 km/s gives an exact ballistic lag of 50 min (L1_DIST_KM/V/3600 = 0.8333 h).
        recent = (V=500.0, Bz=-2.0, By=1.0, n=5.0, Pdyn=1.6726e-6 * 5.0 * 500.0^2)
        step_time = DateTime(2026, 6, 6, 3)
        # Source window for the step is [step_time-1h-lag, min(step_time-lag, latest_common_sw)).
        # latest_common_sw = step_time-80min caps src_hi so the window is [01:10, 01:40) → 30 min → f=0.5.
        latest_common_sw = step_time - Minute(80)
        inside = DateTime(2026, 6, 6, 1, 11) .+ Minute.(0:9)
        # Leakage bait: rows at/after latest_common_sw with wild values must never enter the result.
        outside = [DateTime(2026, 6, 6, 1, 45), DateTime(2026, 6, 6, 2, 0)]
        plasma = DataFrame(time_tag=vcat(inside, outside),
                           speed=vcat(fill(600.0, 10), fill(9999.0, 2)),
                           density=vcat(fill(8.0, 10), fill(999.0, 2)))
        mag = DataFrame(time_tag=vcat(inside, outside),
                        bz_gsm=vcat(fill(-10.0, 10), fill(50.0, 2)),
                        by_gsm=vcat(fill(3.0, 10), fill(-40.0, 2)))

        s = _subhourly_driver_with_status(plasma, mag, step_time, recent, latest_common_sw)
        @test s.l1_measured
        f = 0.5
        @test s.driver.V ≈ f * 600.0 + (1 - f) * recent.V atol = 1e-9     # 550
        @test s.driver.Bz ≈ f * (-10.0) + (1 - f) * recent.Bz atol = 1e-9  # -6
        @test s.driver.By ≈ f * 3.0 + (1 - f) * recent.By atol = 1e-9      # 2
        @test s.driver.n ≈ f * 8.0 + (1 - f) * recent.n atol = 1e-9        # 6.5
        # Pressure is derived from the blended density and speed, preserving the
        # canonical proton dynamic-pressure identity used by the model.
        meas_pdyn = 1.6726e-6 * 8.0 * 600.0^2
        @test s.driver.Pdyn ≈ dynamic_pressure(s.driver.n, s.driver.V) atol = 1e-12
        @test s.driver.Pdyn != f * meas_pdyn + (1 - f) * recent.Pdyn

        # Leakage guard: dropping the post-latest_common_sw rows leaves the result bitwise identical.
        plasma_clean = DataFrame(time_tag=inside, speed=fill(600.0, 10), density=fill(8.0, 10))
        mag_clean = DataFrame(time_tag=inside, bz_gsm=fill(-10.0, 10), by_gsm=fill(3.0, 10))
        s_clean = _subhourly_driver_with_status(plasma_clean, mag_clean, step_time, recent, latest_common_sw)
        @test s_clean.driver == s.driver

        # Full coverage: latest_common_sw >= step_time-lag makes f→1, so the driver is the pure window mean.
        full_common = step_time                                   # well past step_time - lag
        wide = DateTime(2026, 6, 6, 1, 15) .+ Minute.(0:5:45)  # inside [01:10,02:10)
        plasma_full = DataFrame(time_tag=wide, speed=fill(700.0, 10), density=fill(9.0, 10))
        mag_full = DataFrame(time_tag=wide, bz_gsm=fill(-14.0, 10), by_gsm=fill(5.0, 10))
        s_full = _subhourly_driver_with_status(plasma_full, mag_full, step_time, recent, full_common)
        @test s_full.l1_measured
        @test s_full.driver.V ≈ 700.0 atol = 1e-9                 # pure window mean at f=1
        @test s_full.driver.Bz ≈ -14.0 atol = 1e-9
    end

    @testset "Operational tail: subhour L1 branch rejects sparse, NaN, and partial windows" begin
        recent = (V=500.0, Bz=-2.0, By=1.0, n=5.0,
                  Pdyn=dynamic_pressure(5.0, 500.0))
        step_time = DateTime(2026, 6, 6, 3)
        latest_common_sw = step_time - Minute(80)
        times = DateTime(2026, 6, 6, 1, 11) .+ Minute.(0:9)
        plasma = DataFrame(time_tag=times, speed=fill(600.0, 10), density=fill(8.0, 10))
        mag = DataFrame(time_tag=times, bz_gsm=fill(-10.0, 10), by_gsm=fill(3.0, 10))

        for n in (0, 1, 9)
            sparse = _subhourly_driver_with_status(
                plasma[1:n, :], mag[1:n, :], step_time, recent, latest_common_sw,
            )
            @test !sparse.l1_measured
            @test sparse.driver == recent
        end
        exact = _subhourly_driver_with_status(
            plasma, mag, step_time, recent, latest_common_sw,
        )
        @test exact.l1_measured

        plasma_nan = copy(plasma)
        mag_nan = copy(mag)
        plasma_nan.speed .= NaN
        plasma_nan.density .= NaN
        mag_nan.bz_gsm .= NaN
        mag_nan.by_gsm .= NaN
        all_nan = _subhourly_driver_with_status(
            plasma_nan, mag_nan, step_time, recent, latest_common_sw,
        )
        @test !all_nan.l1_measured
        @test all_nan.driver == recent

        partial_mag = copy(mag)
        partial_mag.by_gsm .= NaN
        partial = _subhourly_driver_with_status(
            plasma, partial_mag, step_time, recent, latest_common_sw,
        )
        @test !partial.l1_measured
        @test partial.driver == recent
    end

    @testset "Operational tail: ballistic arrival is identical across completed and look-ahead steps" begin
        recent = (V=500.0, Bz=-1.0, By=0.0, n=5.0,
                  Pdyn=dynamic_pressure(5.0, 500.0))
        # At 500 km/s, L1 transit is 50 min. A shock measured at 01:15 belongs
        # to Earth hour [02:00,03:00), not the preceding completed hour.
        quiet_times = DateTime(2026, 6, 6, 0, 15) .+ Minute.(0:9)
        shock_times = DateTime(2026, 6, 6, 1, 15) .+ Minute.(0:9)
        times = vcat(quiet_times, shock_times)
        plasma = DataFrame(time_tag=times,
                           speed=vcat(fill(500.0, 10), fill(800.0, 10)),
                           density=vcat(fill(5.0, 10), fill(20.0, 10)))
        mag = DataFrame(time_tag=times,
                        bz_gsm=vcat(fill(-1.0, 10), fill(-20.0, 10)),
                        by_gsm=zeros(20))
        latest_common_sw = DateTime(2026, 6, 6, 1, 40)

        before_arrival = _subhourly_driver_with_status(
            plasma, mag, DateTime(2026, 6, 6, 2), recent, latest_common_sw,
        )
        at_arrival = _subhourly_driver_with_status(
            plasma, mag, DateTime(2026, 6, 6, 3), recent, latest_common_sw,
        )
        @test before_arrival.l1_measured
        @test before_arrival.driver.V == 500.0
        @test before_arrival.driver.Bz == -1.0
        @test at_arrival.l1_measured
        @test at_arrival.driver.V > before_arrival.driver.V
        @test at_arrival.driver.Bz < before_arrival.driver.Bz
        @test at_arrival.driver.Pdyn ≈
              dynamic_pressure(at_arrival.driver.n, at_arrival.driver.V) atol=1e-12
    end

    @testset "M-mem: memory features and freshest-pair tail rate degrade loudly on Dst gaps" begin
        t = DateTime(2026, 6, 6, 5)
        dst_times = [t - Hour(3), t - Hour(2), t - Hour(1), t]
        dst_vals = [-20.0, -25.0, -40.0, -70.0]                  # deepening series
        drv = (V=450.0, Bz=-8.0, By=2.0, n=6.0, Pdyn=1.6726e-6 * 6.0 * 450.0^2)
        plasma = DataFrame(time_tag=[t - Minute(30)], speed=[450.0], density=[6.0])
        mag = DataFrame(time_tag=[t - Minute(30)], bz_gsm=[-8.0], by_gsm=[2.0])

        # Signs/values pinned against the training-side _lagged_difference convention (v[t]-v[t-lag]).
        m = _live_operational_memory_features(plasma, mag, dst_times, dst_vals, t, drv, t)
        @test m.dst_delta_1h_nt == -30.0                         # -70 - (-40)
        @test m.dst_delta_3h_nt == -50.0                         # -70 - (-20)
        @test m.dst_delta_3h_nt != m.dst_delta_1h_nt

        # Driver memory follows consecutive anchor-aligned, ballistically
        # propagated source hours, matching `add_inertial_sindy_features!`.
        # With V=500 km/s the source windows end 50 min before each Earth-hour
        # anchor.  Distinct values in three consecutive windows make an
        # accidental target-step-minus-anchor difference immediately visible.
        source_values = (
            (t - Hour(3) - Minute(50), 500.0, 1.0, 5.0),
            (t - Hour(2) - Minute(50), 500.0, -2.0, 5.0),
            (t - Hour(1) - Minute(50), 500.0, -8.0, 5.0),
        )
        mem_times = DateTime[]
        mem_speed = Float64[]
        mem_density = Float64[]
        mem_bz = Float64[]
        mem_by = Float64[]
        for (start, speed, bz, density) in source_values
            append!(mem_times, start .+ Minute.(0:9))
            append!(mem_speed, fill(speed, 10))
            append!(mem_density, fill(density, 10))
            append!(mem_bz, fill(bz, 10))
            append!(mem_by, zeros(10))
        end
        mem_plasma = DataFrame(
            time_tag=mem_times, speed=mem_speed, density=mem_density,
        )
        mem_mag = DataFrame(time_tag=mem_times, bz_gsm=mem_bz, by_gsm=mem_by)
        anchor_drv = (V=500.0, Bz=-8.0, By=0.0, n=5.0,
                      Pdyn=dynamic_pressure(5.0, 500.0))
        aligned = _live_operational_memory_features(
            mem_plasma, mem_mag, dst_times, dst_vals, t, anchor_drv, t,
        )
        @test aligned.Bz_delta_1h_nt == -6.0
        @test aligned.VBsouth_delta_1h_mvm == 3.0
        @test aligned.VBsouth_mean_3h_mvm ≈ 5 / 3 atol=1e-12
        @test aligned.Bsouth_mean_3h_nt ≈ 10 / 3 atol=1e-12

        # Interior gap at t-3h zeros the calibrated memory tuple (guard), and the 3h delta is NOT
        # silently forced equal to the 1h delta by a cascading fallback (the documented regression).
        gap3_t = [t - Hour(2), t - Hour(1), t]
        gap3_v = [-25.0, -40.0, -70.0]
        z = _live_operational_memory_features(plasma, mag, gap3_t, gap3_v, t, drv, t)
        @test z == _zero_operational_memory_features()
        @test z.dst_delta_1h_nt == 0.0 && z.dst_delta_3h_nt == 0.0

        # The tail relaxation rate is DECOUPLED from that zeroing: the freshest contiguous pair
        # (t, t-1h) still yields the true -30 nT/h even when t-3h is missing, so tau stays long
        # (> tau0) for a deepening storm instead of collapsing to the quiet default.
        @test _freshest_dst_rate(gap3_t, gap3_v, t) == -30.0
        @test _operational_tail_tau(_freshest_dst_rate(gap3_t, gap3_v, t)) > OPERATIONAL_TAIL_TAU0_H
        @test _dst_memory_fallback(gap3_t, gap3_v, t)            # interior gap is flagged (logged/warned)
        @test !_dst_memory_fallback(dst_times, dst_vals, t)      # complete history is not flagged

        # Missing t-1h: freshest pair scans back to (t-2h, t-3h) = -25-(-20) = -5 nT/h; still finite.
        gap1_t = [t - Hour(3), t - Hour(2), t]
        @test _freshest_dst_rate(gap1_t, [-20.0, -25.0, -70.0], t) == -5.0
        # No contiguous pair at all → 0.0 fallback.
        @test _freshest_dst_rate([t], [-70.0], t) == 0.0
    end

    @testset "F3: anchor-aware split keeps issue_time sets pairwise disjoint" begin
        # Small multi-horizon table: each anchor contributes two rows (h=1, h=2). A
        # raw-index cut can straddle an anchor across splits (leakage); the anchor-
        # aware split must assign each anchor's full block to a single split.
        anchors = [string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:6]
        issue = vcat(anchors, anchors)                       # 12 rows, 6 anchors × 2
        df = DataFrame(
            issue_time_utc=issue,
            model_step_hours=vcat(fill(1, 6), fill(2, 6)),
            pred_dst_nt=collect(1.0:12.0),
        )
        train, validation, holdout = _chronological_train_validation_test(df, 0.5, 0.25)
        train_a = Set(train.issue_time_utc)
        val_a = Set(validation.issue_time_utc)
        hold_a = Set(holdout.issue_time_utc)
        @test isempty(intersect(train_a, val_a))
        @test isempty(intersect(val_a, hold_a))
        @test isempty(intersect(train_a, hold_a))
        # Each split must carry whole anchor blocks (both horizons per anchor).
        for split in (train, validation, holdout)
            for a in unique(split.issue_time_utc)
                @test count(==(a), split.issue_time_utc) == 2
            end
        end
        # Every anchor and every row is placed exactly once.
        @test union(train_a, val_a, hold_a) == Set(anchors)
        @test nrow(train) + nrow(validation) + nrow(holdout) == 12
    end

    @testset "F3: embargo starts each later forecast after all earlier targets" begin
        t0 = DateTime(2026, 1, 1)
        anchors = t0 .+ Hour.(0:19)
        horizons = (1, 2)
        df = DataFrame(
            issue_time_utc=[string(anchor) for anchor in anchors for _ in horizons],
            target_time_utc=[string(anchor + Hour(h)) for anchor in anchors for h in horizons],
            model_step_hours=[h for _ in anchors for h in horizons],
            pred_dst_nt=collect(1.0:(length(anchors) * length(horizons))),
        )
        train, validation, holdout = _chronological_train_validation_test(df, 0.30, 0.35)

        @test maximum(_parse_dt.(train.target_time_utc)) <
              minimum(_parse_dt.(validation.issue_time_utc))
        @test maximum(_parse_dt.(validation.target_time_utc)) <
              minimum(_parse_dt.(holdout.issue_time_utc))
        # Purging is anchor-wise: every retained issue still has both horizons.
        for split in (train, validation, holdout)
            @test nrow(split) > 0
            for anchor in unique(split.issue_time_utc)
                @test count(==(anchor), split.issue_time_utc) == length(horizons)
            end
        end
        @test nrow(validation) < floor(Int, 0.35 * length(anchors)) * length(horizons)
        @test nrow(holdout) <
              (length(anchors) - floor(Int, 0.30 * length(anchors)) -
               floor(Int, 0.35 * length(anchors))) * length(horizons)
    end

    @testset "F5: a thin validation split deploys the fallback, not a operational gate on one row" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "inertial_sindy_calibration.csv")
            # Tiny table whose 0.15 validation fraction yields a single validation
            # row — a degenerate 0/1 coverage check. The gate must not be trusted;
            # the SINDy-core-equivalent fallback must deploy regardless of that one row.
            n = 8
            pred = collect(-30.0:1.0:-23.0)
            observed = pred .+ 2.0                            # a correction that would "pass"
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                sindy_core_pred_dst_nt=pred,
                sindy_core_pred_dst_ci05_nt=pred .- 3.0,
                sindy_core_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_inertial_sindy_calibration,
                table_path=table_path,
                inertial_sindy_calibration_path=cal_path,
                operational_train_fraction=0.7,
                operational_validation_fraction=0.15,           # floor(0.15*8)=1 validation row
                operational_ridge_grid=[0.0],
                operational_ridge=0.0,
            )
            _, validation, _ = _chronological_train_validation_test(
                SolarSINDy.add_inertial_sindy_features!(_operational_base_prediction_table(copy(replay))),
                cfg.operational_train_fraction, cfg.operational_validation_fraction,
            )
            @test nrow(validation) < OPERATIONAL_MIN_VALIDATION_ROWS    # degenerate gate input
            cal = fit_inertial_sindy_calibration!(cfg)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            @test cal.label == "operational_inertial_sindy_fallback_v1_equiv"
            @test !any(selection.deployed)
            @test selection.deploy_block_reason[1] == "validation_split_too_thin"
            @test !selection.validation_trusted[1]
        end
    end

    @testset "G09-02: embargo-emptied validation purges holdout vs train and fails closed by name" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "inertial_sindy_calibration.csv")
            # 30 hourly anchors x horizons {1,6}: the 6 h purge after train deletes every
            # validation anchor (0.15 * 30 = 4 anchors < 6 h).
            rows = NamedTuple[]
            for i in 1:30, h in (1, 6)
                issue = DateTime(2026, 1, 1) + Hour(i)
                pred = -30.0 + 0.5i + h
                push!(rows, (issue_time_utc=string(issue), target_time_utc=string(issue + Hour(h)),
                    latest_dst_time_utc=string(issue), model_step_hours=h,
                    pred_dst_nt=pred, pred_dst_ci05_nt=pred - 3, pred_dst_ci95_nt=pred + 3,
                    observation_dst_nt=pred + 2 + 0.3 * sin(i), sindy_core_pred_dst_nt=pred,
                    sindy_core_pred_dst_ci05_nt=pred - 3, sindy_core_pred_dst_ci95_nt=pred + 3,
                    latest_dst_nt=pred - 1, V_kms=420.0, Bz_nt=-2.0 + 0.1i,
                    By_nt=1.0, n_cm3=5.0, Pdyn_npa=1.5))
            end
            replay = DataFrame(rows)
            CSV.write(table_path, replay)
            cfg = LiveVerifyConfig(;
                mode=:fit_inertial_sindy_calibration, table_path=table_path, inertial_sindy_calibration_path=cal_path,
                operational_train_fraction=0.7, operational_validation_fraction=0.15,
                operational_ridge_grid=[0.0], operational_ridge=0.0,
            )
            train, validation, holdout = _chronological_train_validation_test(
                SolarSINDy.add_inertial_sindy_features!(_operational_base_prediction_table(copy(replay))),
                cfg.operational_train_fraction, cfg.operational_validation_fraction,
            )
            @test nrow(validation) == 0
            @test nrow(holdout) > 0
            # Holdout is purged against the latest PRIOR target (train's, validation being empty).
            @test maximum(_parse_dt.(train.target_time_utc)) <
                  minimum(_parse_dt.(holdout.issue_time_utc))
            err = try
                fit_inertial_sindy_calibration!(cfg)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("validation_split_too_thin", sprint(showerror, err))
            @test !occursin("no common finite rows", sprint(showerror, err))
            @test !isfile(cal_path)
        end
    end

    @testset "F6: gate metrics share one finite-row mask across baselines" begin
        # A scored validation frame with one `missing` and one `NaN` baseline row.
        # Per-column finite filters would give operational more rows than the baseline
        # comparators (unpaired); the common-mask metrics must report equal n.
        scored = DataFrame(
            observation_dst_nt=[-30.0, -31.0, -32.0, -33.0],
            inertial_sindy_pred_dst_nt=[-29.0, -30.0, -31.0, -32.0],
            pred_dst_nt=[-29.5, -30.5, -31.5, -32.5],
            latest_dst_nt=Union{Missing,Float64}[-28.0, missing, -33.0, -34.0],
            obrien_dst_nt=[-27.0, -29.0, NaN, -35.0],
        )
        metrics = _paired_gate_metrics(
            scored, Symbol[:inertial_sindy_pred_dst_nt, :pred_dst_nt, :latest_dst_nt, :obrien_dst_nt],
        )
        # Rows 2 (missing persistence) and 3 (NaN O'Brien) drop from ALL metrics.
        @test metrics[:inertial_sindy_pred_dst_nt].n == 2
        @test metrics[:inertial_sindy_pred_dst_nt].n == metrics[:pred_dst_nt].n ==
              metrics[:latest_dst_nt].n ==
              metrics[:obrien_dst_nt].n
        # The surviving rows are exactly the fully finite ones (rows 1 and 4).
        @test metrics[:inertial_sindy_pred_dst_nt].rmse ≈ 1.0 atol = 1e-12
    end

    @testset "F7: promotion gate requires strict uncorrected-center improvement" begin
        @test !_operational_gate_pass((rmse=1.0, mae=1.0), (rmse=1.0, mae=1.0))

        n = 20
        pred = collect(-50.0:1.0:-31.0)
        replay = DataFrame(
            issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
            pred_dst_nt=pred,
            pred_dst_ci05_nt=pred .- 3.0,
            pred_dst_ci95_nt=pred .+ 3.0,
            observation_dst_nt=vcat(pred[1:14] .+ 2.0, pred[15:end]),
            sindy_core_pred_dst_nt=pred,
            sindy_core_pred_dst_ci05_nt=pred .- 3.0,
            sindy_core_pred_dst_ci95_nt=pred .+ 3.0,
            latest_dst_nt=pred .+ 10.0,
            V_kms=fill(420.0, n),
            Bz_nt=collect(-10.0:1.0:9.0),
            By_nt=fill(1.0, n),
            n_cm3=fill(5.0, n),
            Pdyn_npa=fill(1.5, n),
        )
        df = SolarSINDy.add_inertial_sindy_features!(_operational_base_prediction_table(replay))
        selection = _select_validated_inertial_sindy_calibration(
            df[1:14, :],
            df[15:20, :],
            LiveVerifyConfig(; operational_ridge_grid=[0.0], operational_ridge=0.0),
        )
        @test :beats_preupgrade in propertynames(selection.candidates)
        @test !selection.row.beats_preupgrade
        @test selection.row.beats_persistence
        @test !selection.row.gate_pass
    end

    @testset "ACI interval: lead-keyed (model_step) and regime-conditional" begin
        # Regression for the horizon-key bug: the query keys on model_step_hours
        # (target − anchor), but the residual pool was filtered on horizon_hours
        # (target − issue). When the anchor lags issue time the two differ, so long
        # leads matched an empty pool and fell through to the over-wide static band.
        mktempdir() do dir
            log = joinpath(dir, "log.csv")
            # All rows are model_step 7 but wall-clock horizon ~5.5 h (rounds to 6 ≠ 7):
            # the bug filter (round(horizon_hours)==7) would find NONE of them.
            n = 60
            ms = fill(7.0, 2n); hh = fill(5.5, 2n)
            pred = zeros(2n); obs = zeros(2n); ld = zeros(2n); iss = String[]
            for k in 1:n                                   # quiet regime: tiny ±5 residuals
                obs[k] = isodd(k) ? 5.0 : -5.0; ld[k] = 5.0
            end
            for k in 1:n                                   # disturbed regime: large ±60 residuals
                j = n + k; obs[j] = isodd(k) ? 60.0 : -60.0; ld[j] = -80.0
            end
            for i in 1:2n; push!(iss, "2026-01-01T" * lpad(string(i ÷ 60), 2, '0') * ":" * lpad(string(i % 60), 2, '0') * ":00"); end
            aci_rows = DataFrame(model_step_hours=ms, horizon_hours=hh,
                                     inertial_sindy_pred_dst_nt=pred, observation_dst_nt=obs,
                                     latest_dst_nt=ld, issue_time_utc=iss,
                                     model_version=fill(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, 2n))
            CSV.write(log, aci_rows)
            # Quiet query: pools only the ±5 rows -> narrow band (NOT the old ~nothing/fallback).
            q = _aci_interval_from_log(log, 0.0, 7; latest_dst=5.0)
            q_unlimited = _aci_interval_from_log(log, 0.0, 7; latest_dst=5.0,
                                                  history_window=typemax(Int))
            @test q !== nothing                            # would be `nothing` under the horizon-key bug
            @test q == q_unlimited                         # exact while support is below 500 rows
            hw_q = (q[2] - q[1]) / 2
            @test 3.0 <= hw_q <= 12.0                       # tracks the ±5 quiet residuals
            # Disturbed query: pools the ±60 rows -> much wider band (regime conditioning).
            d = _aci_interval_from_log(log, 0.0, 7; latest_dst=-80.0)
            @test d !== nothing
            hw_d = (d[2] - d[1]) / 2
            @test hw_d > 3 * hw_q                           # storm regime band is far wider than quiet

            legacy_log = joinpath(dir, "legacy.csv")
            legacy_rows = copy(aci_rows)
            legacy_rows.model_version .= "sindy_core"
            CSV.write(legacy_log, legacy_rows)
            @test _aci_interval_from_log(legacy_log, 0.0, 7; latest_dst=5.0) === nothing
        end
    end

    @testset "ACI interval: residual pool is keyed on the issued model (SINDy core vs operational)" begin
        # A SINDy core issuance must be banded from SINDy core residuals, not the
        # operational-calibrated pool. Build a log whose SINDy core and operational residual
        # magnitudes differ sharply and confirm the band tracks the queried column.
        mktempdir() do dir
            log = joinpath(dir, "log.csv")
            n = 45                                            # > warmup+5 = 35
            obs = zeros(n)
            inertial_sindy_pred = [isodd(k) ? -3.0 : 3.0 for k in 1:n]    # |operational residual| = 3
            sindy_core_pred = [isodd(k) ? -15.0 : 15.0 for k in 1:n]  # |SINDy core residual| = 15
            iss = ["2026-01-01T" * lpad(string(i ÷ 60), 2, '0') * ":" * lpad(string(i % 60), 2, '0') * ":00" for i in 1:n]
            CSV.write(log, DataFrame(model_step_hours=fill(1.0, n),
                                     sindy_core_pred_dst_nt=sindy_core_pred, inertial_sindy_pred_dst_nt=inertial_sindy_pred,
                                     observation_dst_nt=obs, latest_dst_nt=fill(-10.0, n),
                                     issue_time_utc=iss,
                                     model_version=fill(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, n)))
            b_operational = _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:inertial_sindy_pred_dst_nt)
            b_core = _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:sindy_core_pred_dst_nt)
            @test b_operational !== nothing && b_core !== nothing
            hw_operational = (b_operational[2] - b_operational[1]) / 2
            hw_core = (b_core[2] - b_core[1]) / 2
            @test hw_core > 2 * hw_operational                        # core band tracks the larger core residuals
            # A log lacking the requested column returns nothing (no cross-model borrowing).
            GC.gc()          # release the mmap'd log before rewriting it (Windows keeps it locked)
            CSV.write(log, DataFrame(model_step_hours=fill(1.0, n), inertial_sindy_pred_dst_nt=inertial_sindy_pred,
                                     observation_dst_nt=obs, latest_dst_nt=fill(-10.0, n),
                                     issue_time_utc=iss))
            @test _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:sindy_core_pred_dst_nt) === nothing
        end
    end

    @testset "super-learner A3 calibration shadow is exact-identity and availability causal" begin
        @test _superlearner_calibration_shadow_width_scale(1) == 1.50
        @test _superlearner_calibration_shadow_width_scale(2) == 1.20
        @test _superlearner_calibration_shadow_width_scale(3) == 1.20
        @test _superlearner_calibration_shadow_width_scale(4) == 1.20
        @test _superlearner_calibration_shadow_width_scale(6) == 1.20
        @test _superlearner_calibration_shadow_width_scale(7) == 1.20
        @test_throws ArgumentError _superlearner_calibration_shadow_width_scale(5)
        mktempdir() do dir
            log = joinpath(dir, "log.csv")
            base = DateTime(2026, 1, 1)
            n = 35
            issues = base .+ Hour.(0:n-1)
            targets = issues .+ Hour(1)
            point = fill(-10.0, n)
            residuals = collect(1.0:n)
            rows = DataFrame(
                issue_time_utc=string.(issues),
                latest_dst_time_utc=string.(issues),
                target_time_utc=string.(targets),
                model_step_hours=fill(1.0, n),
                observation_dst_nt=point .+ residuals,
                served_pred_dst_nt=point,
                sub_hourly_model_version=fill(SUPERLEARNER_SERVED_TAIL_VERSION, n),
                superlearner_status=fill("ok", n),
                superlearner_manifest_sha256=fill(
                    SUPERLEARNER_CALIBRATION_SHADOW_SERVED_MANIFEST_SHA256, n,
                ),
            )
            CSV.write(log, rows)
            cutoff = targets[end]
            issue = cutoff + Hour(1)
            shadow = _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, cutoff, issue,
            )
            @test shadow.status == "ok"
            @test shadow.history_n == n
            @test shadow.location == 23.5 # median of trailing residuals 12:35
            @test shadow.lo == 10.5
            @test shadow.hi == 18.0

            wide = copy(rows)
            wide[!, :unrelated_payload] = fill("unused,\"quoted\"", n)
            CSV.write(log, wide)
            @test _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, cutoff, issue,
            ) == shadow

            for invalid_history in (:fractional_step, :wrong_anchor, :nonfuture)
                malformed = copy(rows)
                if invalid_history == :fractional_step
                    malformed.model_step_hours[1] = 1.1
                elseif invalid_history == :wrong_anchor
                    malformed.latest_dst_time_utc[1] = string(issues[1] - Hour(1))
                else
                    malformed.issue_time_utc[1] = malformed.target_time_utc[1]
                end
                CSV.write(log, malformed)
                checked = _superlearner_calibration_shadow_from_log(
                    log, -10.0, -12.0, -7.0, 1, cutoff, issue,
                )
                @test checked.status == "ok"
                @test checked.history_n == n - 1
                @test checked.location == 23.5 # The excluded first residual is outside the trailing 24.
            end

            # Issuance already holds the current causal Dst snapshot. Use it for
            # newly matured and revised outcomes instead of the pre-refresh log.
            stale = copy(rows)
            allowmissing!(stale, :observation_dst_nt)
            stale.observation_dst_nt[end-10:end] .= -1_000.0
            stale.observation_dst_nt[end] = missing
            CSV.write(log, stale)
            current = _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, cutoff, issue;
                dst_times=targets, dst_vals=point .+ residuals,
            )
            @test current == shadow

            # A wrong-identity row and an exact-identity outcome beyond the issue-time cutoff cannot
            # enter the current history, even though both observations are present in the snapshot.
            contaminated = vcat(rows, rows[1:1, :], rows[2:2, :])
            contaminated.sub_hourly_model_version[end-1] = OPERATIONAL_SERVED_TAIL_VERSION
            contaminated.observation_dst_nt[end-1] = 10_000.0
            contaminated.issue_time_utc[end] = string(issue - Minute(10))
            contaminated.latest_dst_time_utc[end] = string(issue - Hour(1))
            contaminated.target_time_utc[end] = string(issue + Hour(6))
            contaminated.observation_dst_nt[end] = 20_000.0
            CSV.write(log, contaminated)
            unchanged = _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, cutoff, issue,
            )
            @test unchanged == shadow

            wrong_manifest = copy(rows)
            wrong_manifest.superlearner_manifest_sha256 .= "0"^64
            CSV.write(log, wrong_manifest)
            manifest_blocked = _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, cutoff, issue,
            )
            @test manifest_blocked.status == "warmup:0/30"

            CSV.write(log, first(rows, 29))
            warmup = _superlearner_calibration_shadow_from_log(
                log, -10.0, -12.0, -7.0, 1, targets[29], issue,
            )
            @test warmup.status == "warmup:29/30"
            @test ismissing(warmup.lo)

            scored = DataFrame(
                pred_dst_nt=[-10.0], pred_dst_ci05_nt=[-12.0], pred_dst_ci95_nt=[-8.0],
                served_pred_dst_nt=[-10.0], served_pred_dst_ci05_nt=[-12.0],
                served_pred_dst_ci95_nt=[-8.0],
                superlearner_cal_shadow_status=["ok"], superlearner_cal_shadow_ci05_nt=[-13.0],
                superlearner_cal_shadow_ci95_nt=[-7.0],
            )
            _score_row!(scored, 1, -12.5)
            @test scored.superlearner_cal_shadow_observed_in_90ci[1]
            @test !scored.served_observed_in_90ci[1]
        end
    end

    @testset "LOG-01: ACI checkpoint is bounded and restart-equivalent" begin
        mktempdir() do dir
            log = joinpath(dir, "log.csv")
            n = 520
            base = DateTime(2026, 1, 1)
            issues = base .+ Hour.(0:n-1)
            pred = fill(-20.0, n)
            obs = pred .+ [isodd(i) ? Float64(i % 17) : -Float64(i % 17)
                           for i in 1:n]
            CSV.write(log, DataFrame(
                issue_time_utc=string.(issues),
                latest_dst_time_utc=string.(issues),
                target_time_utc=string.(issues .+ Hour(1)),
                model_version=fill(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION, n),
                model_step_hours=fill(1.0, n),
                latest_dst_nt=fill(-10.0, n),
                pred_dst_nt=pred,
                pred_dst_ci05_nt=fill(-40.0, n),
                pred_dst_ci95_nt=fill(0.0, n),
                inertial_sindy_pred_dst_nt=pred,
                observation_dst_nt=obs,
            ))

            initial = _aci_interval_from_log(log, -20.0, 1; latest_dst=-10.0)
            @test initial !== nothing
            state = _read_live_state(log)
            entry = first(values(state["aci_streams"]))
            @test length(entry["all"]["history"]) == _ACI_HISTORY_WINDOW
            @test length(entry["quiet"]["history"]) == _ACI_HISTORY_WINDOW
            @test isempty(entry["disturbed"]["history"])

            issue = base + Hour(n)
            target = issue + Hour(1)
            pending = DataFrame(
                issue_time_utc=[string(issue)],
                latest_dst_time_utc=[string(issue)],
                target_time_utc=[string(target)],
                model_version=[OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION], model_step_hours=[1.0],
                latest_dst_nt=[-10.0], pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-40.0], pred_dst_ci95_nt=[0.0],
                inertial_sindy_pred_dst_nt=[-20.0], observation_dst_nt=[missing],
            )
            @test _append_forecast!(log, pending) == n + 1
            @test verify_pending!(LiveVerifyConfig(; log_path=log);
                                  dst_times=[target], dst_vals=[-27.0]) == 1
            incremental = _aci_interval_from_log(log, -20.0, 1; latest_dst=-10.0)

            # Deleting the checkpoint emulates a fresh process with no cached
            # state. Full chronological replay is the independent oracle.
            rm(_live_state_path(log); force=true)
            rebuilt = _aci_interval_from_log(log, -20.0, 1; latest_dst=-10.0)
            @test incremental == rebuilt
            rebuilt_state = _read_live_state(log)
            rebuilt_entry = first(values(rebuilt_state["aci_streams"]))
            @test length(rebuilt_entry["all"]["history"]) == _ACI_HISTORY_WINDOW
        end
    end
end

end # module LiveForecastVerificationTests
