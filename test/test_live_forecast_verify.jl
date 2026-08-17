module LiveForecastVerificationTests

using Test
using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "examples", "live_forecast_verify.jl"))

struct _InterruptingForecastText end
Base.String(::_InterruptingForecastText) = throw(InterruptException())

@testset "Live Forecast Verification Workflow" begin
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

            configs = [LiveVerifyConfig(;
                model=:v2,
                horizon_hours=1,
                log_path=joinpath(dir, "forecast_$i.csv"),
                report_path=joinpath(dir, "report_$i.md"),
            ) for i in 1:2]
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
            redirect_stdout(devnull) do
                issue_forecast(configs[1]; inputs=causal_inputs, write_trajectory=false, verbose=false)
                issue_forecast(configs[2]; inputs=skewed_inputs, write_trajectory=false, verbose=false)
            end
            causal_row = CSV.read(configs[1].log_path, DataFrame)[1, :]
            skewed_row = CSV.read(configs[2].log_path, DataFrame)[1, :]

            @test DateTime(skewed_row.latest_solar_wind_utc) == issue_time
            @test DateTime(skewed_row.latest_dst_time_utc) == latest_dst_time
            @test skewed_row.latest_dst_nt == last(dst_values)
            for column in (
                :anchor_dst_star_nt, :target_time_utc, :V_kms, :Bz_nt, :By_nt,
                :n_cm3, :Pdyn_npa, :target_step_V_kms, :target_step_Bz_nt,
                :target_step_By_nt, :target_step_n_cm3, :target_step_Pdyn_npa,
                :v1_pred_dst_nt, :v2_pred_dst_nt, :served_pred_dst_nt,
                :served_pred_dst_ci05_nt, :served_pred_dst_ci95_nt,
            )
                @test isequal(skewed_row[column], causal_row[column])
            end
        end
    end

    @testset "Monitor-selected ACI policy fails closed without both residual streams" begin
        calls = Ref(0)
        unavailable = (args...; kwargs...) -> (calls[] += 1; nothing)
        @test _aci_interval_for_policy(
            :static, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:v2_pred_dst_nt, interval_fn=unavailable,
        ) === nothing
        @test calls[] == 0
        @test _aci_interval_for_policy(
            :auto, "unused.csv", -20.0, 3;
            latest_dst=-30.0, pred_col=:v2_pred_dst_nt, interval_fn=unavailable,
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
            latest_dst=-30.0, pred_col=:v2_pred_dst_nt, interval_fn=available,
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
                model=:v2,
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

    @testset "D: argument parser accepts v2 workflow options" begin
        cfg = _parse_args([
            "--fit-v2-calibration",
            "--model=v2",
            "--table=/tmp/replay.csv",
            "--report=/tmp/live_report.md",
            "--v2-calibration=/tmp/v2.csv",
            "--v2-train-fraction=0.6",
            "--v2-validation-fraction=0.2",
            "--v2-ridge=10",
            "--v2-coverage=0.8",
            "--v2-selector-margin=1.25",
        ])
        @test cfg.mode == :fit_v2_calibration
        @test cfg.model == :v2
        @test cfg.table_path == "/tmp/replay.csv"
        @test cfg.report_path == "/tmp/live_report.md"
        @test cfg.v2_calibration_path == "/tmp/v2.csv"
        @test cfg.v2_train_fraction == 0.6
        @test cfg.v2_validation_fraction == 0.2
        @test cfg.v2_ridge == 10.0
        @test cfg.v2_ridge_grid == [10.0]
        @test cfg.v2_interval_coverage == 0.8
        @test cfg.v2_selector_margin_nt == 1.25

        grid_cfg = _parse_args(["--fit-v2-calibration", "--v2-ridge-grid=0,10,100"])
        @test grid_cfg.v2_ridge_grid == [0.0, 10.0, 100.0]

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
        @test campaign.model == :v2
        @test campaign.campaign_horizons == [1, 3, 6]

        default_issue = _parse_args(String[])
        @test default_issue.mode == :issue
        @test default_issue.model == :v2

        explicit_v1 = _parse_args(["--campaign", "--model=v1"])
        @test explicit_v1.model == :v1
        @test_throws ArgumentError _parse_args(["--campaign-horizons=1,0"])
    end

    @testset "A/D: v2 derived features are causal and deterministic" begin
        features = _v2_features(
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

        expert = _v2_features(
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
            v1_pred_dst=-20.0,
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
        @test_throws ArgumentError _v2_features(
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
        cal = OperationalV2Calibration(
            Symbol[:V_kms], Float64[0.0], Float64[1.0], Float64[0.0, 0.01],
            1.0, "anchor-feature-regression",
        )
        anchor_features = _v2_features(-40.0, anchor)
        target_features = _v2_features(-40.0, target_step)
        @test SolarSINDy.operational_v2_correction(cal, anchor_features) ≈ 4.2
        @test SolarSINDy.operational_v2_correction(cal, target_features) ≈ 8.0
        selected = _select_model_prediction(
            :v2, cal, -40.0, anchor, -45.0, -55.0, -35.0;
            features=anchor_features,
        )
        @test selected.v2_correction ≈ 4.2

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
                model=:v2,
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
            relax = exp(-4 / _v2_tail_tau(-2.0))
            @test row.target_step_Bz_nt ≈ -4.0 * relax atol=1e-12
            @test row.target_step_By_nt ≈ 1.0 * relax atol=1e-12
            @test row.target_step_Bz_nt != -4.0
            @test row.Bz_nt == -4.0  # anchor-feature provenance remains unchanged

            # The logged served-model identity promises that the operational tail ran.
            # If that computation fails, issuance must fail closed before a row exists.
            failed_cfg = LiveVerifyConfig(;
                model=:v2,
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

    @testset "Served static stack and V2.3 shadow columns" begin
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
                model=:v2,
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
                # The shadow deployment is deliberately pointed away so this case isolates the served
                # stage; the shadow path is exercised separately below.
                row = withenv("SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(dir, "absent")) do
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    _issue_row(dir, "served_stack")
                end
                reset_v2_3_shadow!()
                @test row.model_version == "v2.1"
                @test row.sub_hourly_model_version == V2_2_SERVED_TAIL_VERSION
                @test row.sub_hourly_model_version ==
                      "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"
                @test row.driver_assumption == V2_2_DRIVER_ASSUMPTION
                @test row.v2_2_regime in ("quiet", "active_deepening", "recovery")
                @test isfinite(row.v2_1_served_pred_dst_nt)
                @test row.v2_2_pooled_fallback == false
                # Independent recomputation: the served center is the stack cell applied to the six
                # logged components, then projected.
                stack = load_v22_serving_stack(V2_2_DEFAULT_STACK_PATH)
                expected = v22_serving_center(
                    stack; model_steps=Int(row.model_step_hours),
                    latest_dst=Float64(row.latest_dst_nt),
                    dst_delta_1h_nt=Float64(row.dst_delta_1h_nt),
                    vbsouth_mvm=Float64(row.VBsouth_mvm),
                    served_v2_1=Float64(row.v2_1_served_pred_dst_nt),
                    frozen_v2_1=Float64(row.v2_pred_dst_nt),
                    persistence=Float64(row.persistence_dst_nt),
                    burton=Float64(row.burton_dst_nt),
                    burton_full=Float64(row.burton_full_dst_nt),
                    obrien=Float64(row.obrien_dst_nt),
                )
                @test Float64(row.served_pred_dst_nt) ≈ expected.center atol=1e-12
                @test String(row.v2_2_regime) == String(expected.regime)
                @test Float64(row.v2_2_coupling_active_mvm) == expected.coupling_active_mvm
                # The coupling gate must be the archived gate, not the raw coupling.
                @test Float64(row.v2_2_coupling_active_mvm) == v22_serving_coupling_active(
                    Float64(row.VBsouth_mvm), Float64(row.dst_delta_1h_nt))
                # The served band is shifted onto the served center, so the center stays inside it.
                @test row.served_pred_dst_ci05_nt <= row.served_pred_dst_nt <=
                      row.served_pred_dst_ci95_nt
                # The shadow columns exist and disclose why no shadow center was produced.
                @test row.v23_shadow_model_version == V2_3_SHADOW_TAIL_VERSION
                @test startswith(String(row.v23_status), "unavailable:")
                @test ismissing(row.v23_shadow_pred_dst_nt)
                @test ismissing(row.v23_raw_dst_nt)
                @test row.v23_e_layer_applied == false
            end
        end

        @testset "artifact caches retry on a bounded cool-down" begin
            # A weights file that is briefly absent during a redeploy must heal without a daemon
            # restart, so the remembered failure has to expire.
            @test V2_2_STACK_RETRY_SECONDS > 0.0
            @test V2_3_SHADOW_RETRY_SECONDS >= V2_2_STACK_RETRY_SECONDS
            mktempdir() do dir
                staged = joinpath(dir, "operational_v2_2_stack.csv")
                withenv("SOLARSINDY_V2_2_STACK" => staged) do
                    reset_v2_2_stack!()
                    absent = redirect_stderr(devnull) do
                        _v2_2_stack()
                    end
                    @test absent.stack === nothing
                    @test absent.status == "fallback_v2_1:stack_absent"
                    cp(V2_2_DEFAULT_STACK_PATH, staged)
                    # Inside the cool-down the remembered failure is reused, so the file is not re-read.
                    @test _v2_2_stack().stack === nothing
                    # Expiring the cool-down lets the now-present file load.
                    _V2_2_STACK_STATUS_TIME[] -= 2 * V2_2_STACK_RETRY_SECONDS
                    healed = _v2_2_stack()
                    @test healed.status == "ok"
                    @test healed.stack.label == V22_SERVED_STACK_LABEL
                end
                reset_v2_2_stack!()
            end
        end

        @testset "absent stack weights fall back to the V2.1 center and say so" begin
            mktempdir() do dir
                row = withenv("SOLARSINDY_V2_2_STACK" => joinpath(dir, "no_such_stack.csv"),
                              "SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(dir, "absent")) do
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    _issue_row(dir, "stack_absent")
                end
                reset_v2_2_stack!(); reset_v2_3_shadow!()
                @test row.sub_hourly_model_version == V2_SERVED_TAIL_VERSION
                @test row.driver_assumption == V2_DRIVER_ASSUMPTION
                @test Float64(row.served_pred_dst_nt) == Float64(row.v2_1_served_pred_dst_nt)
                @test ismissing(row.v2_2_regime)
            end
        end

        @testset "tampered stack weights fall back rather than serve unpinned weights" begin
            mktempdir() do dir
                tampered = joinpath(dir, "operational_v2_2_stack.csv")
                cp(V2_2_DEFAULT_STACK_PATH, tampered)
                open(tampered, "a") do io
                    write(io, "\n")
                end
                row = withenv("SOLARSINDY_V2_2_STACK" => tampered,
                              "SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(dir, "absent")) do
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_tampered")
                    end
                end
                reset_v2_2_stack!(); reset_v2_3_shadow!()
                @test row.sub_hourly_model_version == V2_SERVED_TAIL_VERSION
                @test Float64(row.served_pred_dst_nt) == Float64(row.v2_1_served_pred_dst_nt)
            end
        end

        @testset "shadow center is logged and never served" begin
            shadow_dir = normpath(joinpath(@__DIR__, "..", "deploy", "v2_3_shadow"))
            if !isdir(shadow_dir)
                @test_skip "deploy/v2_3_shadow is absent; run validation/operational/v2_3_build_deploy.jl"
            else
                mktempdir() do dir
                    row = withenv("SOLARSINDY_V2_3_SHADOW_DIR" => shadow_dir) do
                        reset_v2_2_stack!(); reset_v2_3_shadow!()
                        _issue_row(dir, "shadow_ok")
                    end
                    # The one-hour pre-layer center is cached per anchor and reused by every horizon of
                    # the cycle, so its key must carry every input the center depends on. The
                    # issue-anchor drivers and the memory features are both recomputed from the L1
                    # stream at each issuance and both enter the blend, so the key is read here before
                    # the reset drops it.
                    step1_key = _V2_3_STEP1_CACHE[] === nothing ? nothing : _V2_3_STEP1_CACHE[][1]
                    reset_v2_3_shadow!()
                    @test step1_key !== nothing
                    @test length(step1_key) == 11
                    @test step1_key[end - 1] isa UInt          # issue-anchor drivers
                    @test step1_key[end] isa UInt              # memory features
                    # The key entries are content hashes: different values, and the same values under
                    # different names, are different keys; integer and float spellings are not.
                    @test _v2_3_named_float_hash((V = 400.0, Bz = -5.0)) !=
                          _v2_3_named_float_hash((V = 400.0, Bz = -5.5))
                    @test _v2_3_named_float_hash((V = 400.0, Bz = -5.0)) !=
                          _v2_3_named_float_hash((Bz = 400.0, V = -5.0))
                    @test _v2_3_named_float_hash((V = 400, Bz = -5)) ==
                          _v2_3_named_float_hash((V = 400.0, Bz = -5.0))
                    # The step-7 layer is a fitted ridge, and a fresh log has no matured one-hour
                    # innovation, so the row is available and says the layer is still pending.
                    @test row.v23_status == V2_3_SHADOW_STATUS_E_LAYER_PENDING
                    @test row.v23_analog_k == 25
                    @test isfinite(row.v23_raw_dst_nt)
                    @test isfinite(row.v23_center_dst_nt)
                    @test isfinite(row.v23_shadow_pred_dst_nt)
                    @test row.v23_shadow_model_version == V2_3_SHADOW_TAIL_VERSION
                    # A fresh log has no matured one-step innovation, so the error layer must be the
                    # identity and the shadow center must equal the pre-layer center.
                    @test row.v23_e_layer_applied == false
                    @test Float64(row.v23_shadow_pred_dst_nt) ==
                          Float64(row.v23_center_dst_nt)
                    # The shadow forecast must not reach the served center or the served label.
                    @test row.sub_hourly_model_version == V2_2_SERVED_TAIL_VERSION
                    @test Float64(row.served_pred_dst_nt) != Float64(row.v23_shadow_pred_dst_nt)
                    # The one-hour pre-layer center is recorded even though one hour is not an issued
                    # horizon of this row: it is the quantity the error layer's history is defined on.
                    @test isfinite(row.v23_step1_center_dst_nt)
                    @test Float64(row.v23_step1_center_dst_nt) != Float64(row.v23_center_dst_nt)
                    # Hourly L1 coverage the analog key drew on, and the shipped manifest digest, so a
                    # short feed and a silent redeploy are both visible in the row.
                    @test Int(row.v23_history_hours) == V23_SOUTH_RUN_CAP_H
                    @test String(row.v23_manifest_sha256) ==
                          v23_serving_file_sha256(joinpath(shadow_dir, "manifest.csv"))
                    # A fresh log holds one anchor and no matured one-hour target, so the history is
                    # empty and the step's fitted layer is still pending.
                    @test isempty(_v2_3_innovation_history(joinpath(dir, "shadow_ok.csv"),
                                                          dst_times, dst_values))
                    @test row.v23_status == V2_3_SHADOW_STATUS_E_LAYER_PENDING
                    @test startswith(String(row.v23_status), V2_3_SHADOW_STATUS_OK)
                    # The one-hour centers are keyed by anchor and read back from the log.
                    centers = _v2_3_step1_centers(joinpath(dir, "shadow_ok.csv"))
                    @test collect(keys(centers)) == [latest_dst_time]
                    @test centers[latest_dst_time] == Float64(row.v23_step1_center_dst_nt)
                end
            end
        end

        @testset "the live error layer engages once six one-hour innovations mature" begin
            # This is the chain the deployed candidate's error layer needs. Production issues wall
            # horizons 1/2/3/6 at an anchor lag of one hour, so its model steps are 2/3/4/7 and no
            # logged row ever carries a one-hour step: an error layer keyed on one-hour *rows* can
            # never engage. Eight consecutive anchors are issued here so the sixth-lag block of the
            # last anchor is complete, and the steps that carry a fitted layer must then apply it while
            # the steps whose selected layer is the identity must not.
            #
            # Maturity comes from the observed Dst series the issuance already holds, not from a
            # verification pass, so the cycles are issue-only.
            shadow_dir = normpath(joinpath(@__DIR__, "..", "deploy", "v2_3_shadow"))
            if !isdir(shadow_dir)
                @test_skip "deploy/v2_3_shadow is absent; run validation/operational/v2_3_build_deploy.jl"
            else
                mktempdir() do dir
                    log_path = joinpath(dir, "chain.csv")
                    base_issue = DateTime(2026, 7, 15, 6, 30)
                    # Deterministic, smooth, storm-scale Dst so consecutive anchors share values at
                    # overlapping hours and the innovations are neither constant nor degenerate.
                    dst_at(t) = -35.0 - 25.0 * sin(2π * (Dates.value(t - DateTime(2026, 1, 1)) /
                                                         3_600_000) / 29)
                    feed_start = base_issue - Hour(40)
                    feed_stop = base_issue + Hour(12)
                    all_sw = collect(feed_start:Minute(1):feed_stop)
                    full_plasma = DataFrame(
                        time_tag=all_sw,
                        speed=[470.0 + 25.0 * sin(2π * k / 613) for k in eachindex(all_sw)],
                        density=[6.0 + 0.9 * sin(2π * k / 421) for k in eachindex(all_sw)],
                    )
                    full_mag = DataFrame(
                        time_tag=all_sw,
                        bz_gsm=[-6.0 + 3.0 * sin(2π * k / 517) for k in eachindex(all_sw)],
                        by_gsm=[1.5 * cos(2π * k / 379) for k in eachindex(all_sw)],
                    )
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    anchors = DateTime[]
                    withenv("SOLARSINDY_V2_3_SHADOW_DIR" => shadow_dir) do
                        for cycle in 1:8
                            issue = base_issue + Hour(cycle - 1)
                            anchor = _floor_hour(issue) - Hour(1)
                            push!(anchors, anchor)
                            # Truncating the feed per cycle keeps each issuance causal: a later cycle's
                            # upstream wind must not be visible to an earlier one.
                            cycle_plasma = full_plasma[full_plasma.time_tag .<= issue, :]
                            cycle_mag = full_mag[full_mag.time_tag .<= issue, :]
                            cycle_dst_times = collect((anchor - Hour(24)):Hour(1):anchor)
                            cycle_dst_values = dst_at.(cycle_dst_times)
                            # The warm-up cycles only need to record their anchor's one-hour center;
                            # the final cycle issues the full requested horizon set.
                            horizons = cycle == 8 ? [1, 2, 3, 6] : [6]
                            for h in horizons
                                cfg = LiveVerifyConfig(;
                                    model=:v2, horizon_hours=h, log_path=log_path,
                                    report_path=joinpath(dir, "chain.md"),
                                )
                                inputs = prepare_issue_inputs(
                                    cfg; issue_time=issue,
                                    plasma_fn=() -> cycle_plasma, mag_fn=() -> cycle_mag,
                                    dst_fn=() -> (cycle_dst_times, cycle_dst_values),
                                )
                                redirect_stdout(devnull) do
                                    issue_forecast(cfg; inputs, write_trajectory=false,
                                                   verbose=false)
                                end
                            end
                        end
                    end
                    reset_v2_3_shadow!()
                    log = CSV.read(log_path, DataFrame)
                    @test nrow(log) == 7 + 4

                    # The chain the layer consumes: one center per anchor, and six matured innovations
                    # at the final anchor.
                    final_anchor = last(anchors)
                    final_dst_times = collect((final_anchor - Hour(24)):Hour(1):final_anchor)
                    centers = _v2_3_step1_centers(log_path)
                    @test length(centers) == 8
                    @test all(a -> haskey(centers, a), anchors)
                    history = _v2_3_innovation_history(log_path, final_dst_times,
                                                       dst_at.(final_dst_times))
                    block = v23_serving_innovation_lags(final_anchor, history)
                    @test block.ok
                    @test length(unique(block.values)) > 1
                    # Independent recomputation of one lag from its two logged ingredients.
                    lag1 = final_anchor - Hour(1)
                    @test history[lag1] ≈ dst_at(lag1 + Hour(1)) - centers[lag1] atol=0

                    logged_anchor = _parse_dt.(string.(log.latest_dst_time_utc))
                    final = log[logged_anchor .== final_anchor, :]
                    @test nrow(final) == 4
                    by_step = Dict(Int(r.model_step_hours) => r for r in eachrow(final))
                    @test sort(collect(keys(by_step))) == [2, 3, 4, 7]
                    # Steps 2 and 7 carry a fitted layer; steps 3 and 4 are the identity by selection.
                    for step in (2, 7)
                        @test by_step[step].v23_e_layer_applied == true
                        @test by_step[step].v23_status == "ok"
                        @test Float64(by_step[step].v23_shadow_pred_dst_nt) !=
                              Float64(by_step[step].v23_center_dst_nt)
                    end
                    for step in (3, 4)
                        @test by_step[step].v23_e_layer_applied == false
                        @test by_step[step].v23_status == "ok"
                        @test Float64(by_step[step].v23_shadow_pred_dst_nt) ==
                              Float64(by_step[step].v23_center_dst_nt)
                    end
                    # Every row of the cycle shares one anchor, hence one one-hour center.
                    @test length(unique(Float64.(final.v23_step1_center_dst_nt))) == 1
                    # The shadow center never reaches the served center or the served label.
                    @test all(final.sub_hourly_model_version .== V2_2_SERVED_TAIL_VERSION)
                    @test all(Float64.(final.served_pred_dst_nt) .!=
                              Float64.(final.v23_shadow_pred_dst_nt))

                    # An earlier cycle, whose sixth lag does not exist yet, must still be pending: the
                    # layer must not engage on a partial block.
                    early = log[logged_anchor .== first(anchors), :]
                    @test all(early.v23_e_layer_applied .== false)
                    @test all(early.v23_status .== V2_3_SHADOW_STATUS_E_LAYER_PENDING)
                end
            end
        end

        @testset "an unpinned stack digest is refused unless the operator accepts it" begin
            # An empty digest override removes the only check that ties the weights to the fitted
            # stack the served identity names, because the label check passes for any file carrying
            # the label. The default must therefore be refusal, and an accepted unpinned load must not
            # be published under the pinned identity.
            mktempdir() do dir
                staged = joinpath(dir, "operational_v2_2_stack.csv")
                cp(V2_2_DEFAULT_STACK_PATH, staged)
                chmod(staged, 0o644)
                refused = withenv("SOLARSINDY_V2_2_STACK" => staged,
                                  "SOLARSINDY_V2_2_STACK_SHA256" => "",
                                  "SOLARSINDY_ALLOW_UNPINNED_STACK" => nothing,
                                  "SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(dir, "absent")) do
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_unpinned_refused")
                    end
                end
                @test String(refused.v2_2_status) == "fallback_v2_1:stack_unpinned"
                @test refused.sub_hourly_model_version == V2_SERVED_TAIL_VERSION
                @test Float64(refused.served_pred_dst_nt) ==
                      Float64(refused.v2_1_served_pred_dst_nt)
                @test ismissing(refused.v2_2_regime)

                accepted = withenv("SOLARSINDY_V2_2_STACK" => staged,
                                   "SOLARSINDY_V2_2_STACK_SHA256" => "",
                                   "SOLARSINDY_ALLOW_UNPINNED_STACK" => "1",
                                   "SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(dir, "absent")) do
                    reset_v2_2_stack!(); reset_v2_3_shadow!()
                    redirect_stderr(devnull) do
                        _issue_row(dir, "stack_unpinned_accepted")
                    end
                end
                reset_v2_2_stack!(); reset_v2_3_shadow!()
                @test String(accepted.v2_2_status) == V2_2_STACK_OK_UNPINNED_STATUS
                # The center is a stack center, so the driver assumption is the stack assumption, but
                # the identity must not claim the pinned product.
                @test accepted.sub_hourly_model_version == V2_2_UNPINNED_SERVED_TAIL_VERSION
                @test accepted.sub_hourly_model_version != V2_2_SERVED_TAIL_VERSION
                @test accepted.driver_assumption == V2_2_DRIVER_ASSUMPTION
                @test accepted.v2_2_regime in ("quiet", "active_deepening", "recovery")
                @test Float64(accepted.served_pred_dst_nt) !=
                      Float64(accepted.v2_1_served_pred_dst_nt)
            end
        end

        @testset "a short L1 feed fails the analog key closed and records its depth" begin
            # The analog key needs hourly at-Earth driver means for its seven mandatory lags, which at
            # a one-hour anchor lag reach back to the issue hour minus seven plus the ballistic transit
            # and the hourly averaging window: roughly nine and a half hours of upstream minute data.
            # A feed that stops short of that must fail closed with a named missing lag rather than
            # impute a key that silently retrieves different archive analogs, and the recorded coverage
            # depth must show how far the feed actually reached.
            shadow_dir = normpath(joinpath(@__DIR__, "..", "deploy", "v2_3_shadow"))
            if !isdir(shadow_dir)
                @test_skip "deploy/v2_3_shadow is absent; run validation/operational/v2_3_build_deploy.jl"
            else
                function _short_feed_row(dir::AbstractString, name::AbstractString, span::Hour)
                    issue = DateTime(2026, 7, 15, 12, 30)
                    feed_times = collect((issue - span):Minute(1):issue)
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
                    feed_anchor = _floor_hour(issue) - Hour(1)
                    feed_dst_times = collect((feed_anchor - Hour(24)):Hour(1):feed_anchor)
                    feed_dst_values = collect(range(-30.0, -78.0;
                                                   length=length(feed_dst_times)))
                    cfg = LiveVerifyConfig(;
                        model=:v2, horizon_hours=6, log_path=joinpath(dir, "$(name).csv"),
                        report_path=joinpath(dir, "$(name).md"),
                    )
                    return withenv("SOLARSINDY_V2_3_SHADOW_DIR" => shadow_dir) do
                        reset_v2_2_stack!(); reset_v2_3_shadow!()
                        inputs = prepare_issue_inputs(
                            cfg; issue_time=issue,
                            plasma_fn=() -> feed_plasma, mag_fn=() -> feed_mag,
                            dst_fn=() -> (feed_dst_times, feed_dst_values),
                        )
                        redirect_stdout(devnull) do
                            issue_forecast(cfg; inputs, write_trajectory=false, verbose=false)
                        end
                        CSV.read(cfg.log_path, DataFrame)[1, :]
                    end
                end

                mktempdir() do dir
                    short = _short_feed_row(dir, "feed_8h", Hour(8))
                    reset_v2_3_shadow!()
                    @test startswith(String(short.v23_status), "unavailable:missing_driver_lag")
                    @test ismissing(short.v23_shadow_pred_dst_nt)
                    @test ismissing(short.v23_step1_center_dst_nt)
                    # The recorded depth is short of the mandatory lags, which is why the key failed.
                    @test Int(short.v23_history_hours) < V23_HISTORY_LAGS_H
                    @test Int(short.v23_history_hours) < V23_SOUTH_RUN_CAP_H
                    # A short shadow feed must not touch the served product.
                    @test short.sub_hourly_model_version == V2_2_SERVED_TAIL_VERSION
                    @test isfinite(short.served_pred_dst_nt)

                    # Just past the boundary the key is admissible again, and the recorded depth says
                    # the run-length window is still truncated: a ten-hour feed supplies the mandatory
                    # lags but not all twelve.
                    boundary = _short_feed_row(dir, "feed_10h", Hour(10))
                    reset_v2_3_shadow!()
                    @test startswith(String(boundary.v23_status), V2_3_SHADOW_STATUS_OK)
                    @test Int(boundary.v23_history_hours) >= V23_HISTORY_LAGS_H
                    @test Int(boundary.v23_history_hours) < V23_SOUTH_RUN_CAP_H
                    @test isfinite(boundary.v23_step1_center_dst_nt)
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
            LiveVerifyConfig(; model=:v1);
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
            nfeatures = length(V2_MEMORY_EXPERT_LEAD_FEATURES)
            point_cal = OperationalV2Calibration(
                copy(V2_MEMORY_EXPERT_LEAD_FEATURES),
                zeros(nfeatures),
                ones(nfeatures),
                zeros(nfeatures + 1),
                1.0,
                "operational_v2_1_sidecar_test",
                supported_model_steps=copy(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS),
            )
            write_operational_v2_calibration(point_path, point_cal)
            cfg = LiveVerifyConfig(; model=:v2, v2_calibration_path=point_path)
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

            changed_point = OperationalV2Calibration(
                copy(point_cal.feature_names),
                copy(point_cal.feature_mean),
                copy(point_cal.feature_scale),
                vcat(0.25, point_cal.coefficients[2:end]),
                point_cal.interval_scale,
                point_cal.label,
                supported_model_steps=copy(point_cal.supported_model_steps),
            )
            write_operational_v2_calibration(point_path, changed_point)
            @test_throws ArgumentError _load_conformal_for_model(cfg)

            # A byte-matched sidecar with a different discrete support contract
            # is still incompatible and must fail closed.
            write_operational_v2_calibration(point_path, point_cal)
            point_sha = bytes2hex(sha256(read(point_path)))
            write_conformal_calibration(
                sidecar_path, cal; point_calibration_sha256=point_sha,
                supported_model_steps="1;2;3;6",
            )
            @test_throws ArgumentError _load_conformal_for_model(cfg)

            # A self-consistent historical V2.0 point/sidecar pair remains valid
            # for explicit offline replay, but the live V2.1 loader rejects it.
            historical = operational_calibration_artifacts(:v2_0)
            historical_cfg = LiveVerifyConfig(;
                model=:v2,
                v2_calibration_path=historical.point_csv,
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
                model_version=["v2"],
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

    @testset "V2.1 append refuses an unmigrated historical hot log" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "legacy.csv")
            legacy = DataFrame(
                issue_time_utc=["2026-06-06T04:10:00"],
                latest_dst_time_utc=["2026-06-06T04:00:00"],
                target_time_utc=["2026-06-06T06:00:00"],
                model_version=["v2"],
                pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-30.0],
                pred_dst_ci95_nt=[-10.0],
                observation_dst_nt=[-22.0],
            )
            CSV.write(log_path, legacy)
            current = copy(legacy)
            current.issue_time_utc .= "2026-06-06T05:10:00"
            current.target_time_utc .= "2026-06-06T07:00:00"
            current.model_version .= OPERATIONAL_V2_1_MODEL_VERSION
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
                model_version=["v2"],
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
                    model_version=fill("v2", n),
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
                    model_version=["v2"],
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
                model_version=fill("v2", n),
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
                                        log[resolved_idx, :target_time_utc], "v2")
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
                model=:v2,
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
                model_version=["v2"],
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
                model_version=["v2"],
                pred_dst_nt=[-23.4],
                pred_dst_ci05_nt=[-45.0],
                pred_dst_ci95_nt=[-1.0],
                observation_dst_nt=[-22.0],
                residual_dst_nt=[1.4],
                observed_in_90ci=[true],
                v2_pred_dst_nt=[-23.4],
                v2_pred_dst_ci05_nt=[-45.0],
                v2_pred_dst_ci95_nt=[-1.0],
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
            @test df.v2_residual_dst_nt[1] ≈ 2.4 atol=1e-12
            @test df.v2_observed_in_90ci[1] == true
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
                model_version=["v2", "v2", "v2"],
                wall_clock_lead_hours=[1.89, 0.62, 0.75],
                pred_dst_nt=[-39.38, -39.12, -35.0],
                pred_dst_ci05_nt=[-44.94, -42.23, -40.0],
                pred_dst_ci95_nt=[-33.82, -36.01, -30.0],
                observation_dst_nt=Union{Missing,Float64}[-33.0, -33.0, missing],
                residual_dst_nt=Union{Missing,Float64}[6.38, 6.12, missing],
                observed_in_90ci=Union{Missing,Bool}[false, false, missing],
                v1_pred_dst_nt=[-40.63, -40.42, -36.0],
                v2_pred_dst_nt=[-39.38, -39.12, -35.0],
                v2_selected_component=["v2", "v2", "v2"],
                persistence_dst_nt=[-44.0, -44.0, -34.0],
                burton_dst_nt=[-34.30, -34.20, -33.0],
                burton_full_dst_nt=[-34.30, -34.20, -33.0],
                obrien_dst_nt=[-39.91, -39.81, -35.0],
            )
            CSV.write(log_path, log)

            out = write_live_comparison_report(log_path, report_path)
            text = read(report_path, String)

            @test out == report_path
            @test occursin("Verified rows used: 2", text)
            @test occursin("Invalid verified rows excluded: 0", text)
            @test occursin("Pending rows: 1", text)
            @test occursin("Same-row forecast comparison rows: 2", text)
            @test occursin("## Same-Row Model Comparison", text)
            @test occursin("| Historical V2.0 | 2 |", text)
            @test occursin("| SINDy v1 | 2 |", text)
            @test occursin("## Pending Rows", text)
            @test occursin("2026-06-06T10:00:00", text)
            @test occursin("| v2 |", text)
            @test occursin("## Worst Historical V2.0 Misses", text)
            @test occursin("## Operational V2.0 Audit", text)
            @test !occursin("| Historical V2.0 | 3 |", text)
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
        @test all(df.model_version .== "v1")
        @test df.issue_time_utc[1] == string(t0 + Hour(1))
        @test df.source_driver_start_utc[1] == string(t0)
        @test df.source_driver_end_utc[1] == df.issue_time_utc[1]
        @test df.target_time_utc[1] == string(t0 + Hour(2))
        @test all(df.target_time_utc .> df.issue_time_utc)
        @test all(isfinite, df.pred_dst_nt)
        @test df.pred_dst_nt == df.v1_pred_dst_nt
        @test all(ismissing, df.v2_pred_dst_nt)
        @test all(isfinite, df.obrien_dst_nt)
        @test :dst_delta_1h_nt in propertynames(df)
        @test :baseline_spread_nt in propertynames(df)
        @test df.dst_delta_1h_nt[1] == 0.0
        @test df.residual_dst_nt[1] ≈ df.observation_dst_nt[1] - df.pred_dst_nt[1] atol=1e-12
        @test df.persistence_residual_dst_nt[1] ≈ df.observation_dst_nt[1] - df.persistence_dst_nt[1] atol=1e-12

        cal = default_operational_v2_calibration()
        df_v2 = replay_recent_table(
            plasma,
            mag,
            times,
            dst_vals;
            replay_hours=24,
            model=:v2,
            calibration=cal,
        )
        @test all(df_v2.model_version .== OPERATIONAL_V2_1_MODEL_VERSION)
        @test df_v2.pred_dst_nt == df_v2.v2_pred_dst_nt
        @test df_v2.pred_dst_nt == df_v2.v1_pred_dst_nt
        @test all(df_v2.v2_correction_dst_nt .== 0.0)
        @test all(df_v2.v2_selected_component .== "v2")
        @test df_v2.v2_selected_component_pred_nt == df_v2.v2_pred_dst_nt

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
        # Regression for the archival-replay blocker (adv-paper-v2monitor MINOR): the
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

    @testset "A/D: fit_v2_calibration! writes calibration and scored replay rows" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            pred = collect(-30.0:1.0:-11.0)
            bz = collect(-10.0:1.0:9.0)
            observed = pred .+ 2.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:length(pred)],
                pred_dst_nt=pred .+ 20.0,
                pred_dst_ci05_nt=pred .+ 17.0,
                pred_dst_ci95_nt=pred .+ 23.0,
                observation_dst_nt=observed,
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred .- 3.0,
                v1_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=pred .- 1.0,
                V_kms=fill(420.0, length(pred)),
                Bz_nt=bz,
                By_nt=fill(1.0, length(pred)),
                n_cm3=fill(5.0, length(pred)),
                Pdyn_npa=fill(1.5, length(pred)),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_v2_calibration,
                table_path=table_path,
                v2_calibration_path=cal_path,
                v2_train_fraction=0.8,
                v2_ridge_grid=[0.0],
                v2_ridge=0.0,
            )
            cal = fit_v2_calibration!(cfg)
            @test isfile(cal_path)
            scored_path = replace(cal_path, r"\.csv$" => "_scored.csv")
            selection_path = replace(cal_path, r"\.csv$" => "_selection.csv")
            @test isfile(scored_path)
            @test isfile(selection_path)
            # A constant +2 correction generalizes, so v2 is selected and deployed.
            @test startswith(cal.label, "operational_v2_")
            @test cal.selected_component == :v2
            reread = read_operational_v2_calibration(cal_path)
            scored = CSV.read(scored_path, DataFrame)
            selection = CSV.read(selection_path, DataFrame)
            @test maximum(abs.(scored.v2_residual_dst_nt)) < 0.5
            @test Set(scored.v2_split) == Set(["fit", "validation", "holdout"])
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
            pred_v2 = operational_v2_predict(
                reread,
                pred[end],
                pred[end] - 3.0,
                pred[end] + 3.0,
                _v2_features(pred[end] - 1.0, (; V=420.0, Bz=bz[end], By=1.0, n=5.0, Pdyn=1.5)),
            )
            @test pred_v2.pred_dst == observed[end]

            # N1: a conformal calibration sidecar is written and round-trips.
            conformal_path = replace(cal_path, r"\.csv$" => "_conformal.csv")
            @test isfile(conformal_path)
            cc = read_conformal_calibration(conformal_path)
            @test cc.coverage == cfg.v2_interval_coverage
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

    @testset "A/D: acceptance gate deploys v1-equivalent fallback when correction fails validation" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            n = 24
            pred = collect(-50.0:1.0:-27.0)
            observed = copy(pred)
            # +4 correction on the training portion only; it does NOT generalize
            # to the later validation rows, so v2 must fail the acceptance gate.
            observed[1:14] .= pred[1:14] .+ 4.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred .- 3.0,
                v1_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_v2_calibration,
                table_path=table_path,
                v2_calibration_path=cal_path,
                v2_train_fraction=0.6,
                v2_validation_fraction=0.2,
                v2_ridge_grid=[0.0],
                v2_ridge=0.0,
            )
            cal = fit_v2_calibration!(cfg)
            scored = CSV.read(replace(cal_path, r"\.csv$" => "_scored.csv"), DataFrame)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            # Gate failed → a v1-equivalent (zero-correction) fallback is deployed.
            @test cal.label == "operational_v2_1_fallback_v1_equiv"
            @test all(cal.coefficients .== 0.0)            # no correction applied
            @test !any(selection.deployed)                 # nothing passed the gate
            @test all(.!selection.acceptance_gate_pass)
            @test !any(selection.gate_pass)
            # Deployed v2 reduces exactly to v1 (the correction was rejected).
            @test scored.v2_pred_dst_nt == scored.pred_dst_nt
        end
    end

    @testset "F1+F2: V2 conformal interval undercoverage on holdout blocks deploy" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            n = 30
            pred = collect(-60.0:1.0:-31.0)
            observed = copy(pred)
            # +4 holds through train AND validation, but not the holdout. The legacy
            # validation gate passes (the interval_scale band over-covers there), yet
            # the OPERATIONALLY-SERVED conformal interval — fit on near-zero validation
            # residuals — under-covers the untouched holdout where the +4 correction
            # breaks down. F1+F2 gates the served-interval holdout coverage, so v2 must
            # NOT deploy; the v1-equivalent fallback ships instead.
            observed[1:21] .= pred[1:21] .+ 4.0
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred .- 3.0,
                v1_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_v2_calibration,
                table_path=table_path,
                v2_calibration_path=cal_path,
                v2_train_fraction=0.5,
                v2_validation_fraction=0.2,
                v2_ridge_grid=[0.0],
                v2_ridge=0.0,
            )
            cal = fit_v2_calibration!(cfg)
            scored = CSV.read(replace(cal_path, r"\.csv$" => "_scored.csv"), DataFrame)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            holdout = scored[scored.v2_split .== "holdout", :]
            # Honest holdout (scored once for selection, gated once for the served
            # interval) is much worse than validation and undercovers — the served
            # conformal interval gate fires and the v1-equivalent fallback deploys.
            @test cal.label == "operational_v2_1_fallback_v1_equiv"
            @test all(cal.coefficients .== 0.0)            # no correction applied
            @test !any(selection.deployed)                 # served-interval gate blocked deploy
            @test all(.!selection.acceptance_gate_pass)
            @test selection.deploy_block_reason[1] == "conformal_holdout_undercover"
            @test selection.conformal_holdout_coverage[1] < cfg.v2_coverage_floor
            @test !selection.conformal_gate_pass[1]
            # The V2 conformal interval's holdout coverage drove the block: the
            # candidate v2 (+4) under-covers the untouched holdout where the +4
            # correction breaks down, so the gate refused it.
            @test selection.conformal_holdout_coverage[1] == 0.0
            # The deployed (fallback) v2 reduces exactly to v1.
            @test scored.v2_pred_dst_nt == scored.pred_dst_nt
        end
    end

    @testset "A/D: campaign mode issues, verifies, and reports requested horizons" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "campaign.csv")
            report_path = joinpath(tmp, "campaign.md")
            targets = DateTime(2026, 6, 6, 10):Hour(1):DateTime(2026, 6, 6, 12)
            cfg = LiveVerifyConfig(;
                mode=:campaign,
                model=:v2,
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
                    model_version=["v2"],
                    wall_clock_lead_hours=[Float64(issue_cfg.horizon_hours)],
                    pred_dst_nt=[pred],
                    pred_dst_ci05_nt=[pred - 10.0],
                    pred_dst_ci95_nt=[pred + 10.0],
                    observation_dst_nt=[missing],
                    residual_dst_nt=[missing],
                    observed_in_90ci=[missing],
                    v1_pred_dst_nt=[pred - 1.0],
                    v2_pred_dst_nt=[pred],
                    v2_pred_dst_ci05_nt=[pred - 10.0],
                    v2_pred_dst_ci95_nt=[pred + 10.0],
                    v2_selected_component=["v2"],
                    persistence_dst_nt=[-30.0],
                    burton_dst_nt=[-29.0],
                    burton_full_dst_nt=[-29.0],
                    obrien_dst_nt=[-31.0],
                )
                idx = _append_forecast!(log_path, row)
                return (; row_idx=idx, latest_dst_time=DateTime("2026-06-06T09:00:00"),
                        target_time=target, pred_dst=pred,
                        ci05_dst=pred - 10.0, ci95_dst=pred + 10.0,
                        model_version="v2")
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
            @test occursin("Same-row forecast comparison rows: 2", text)
            @test occursin("Historical V2.0 is the archived operational method", text)
        end
    end

    @testset "C0-3: live report headlines Operational V2.1 when available" begin
        mktempdir() do dir
            log_path = joinpath(dir, "v2_log.csv")
            report_path = joinpath(dir, "v2_report.md")
            df = DataFrame(
                issue_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                latest_dst_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                target_time_utc=["2026-06-06T11:00:00", "2026-06-06T12:00:00"],
                model_version=fill(OPERATIONAL_V2_1_MODEL_VERSION, 2),
                wall_clock_lead_hours=[2.0, 2.0],
                horizon_hours=[2.0, 2.0],
                pred_dst_nt=[-40.0, -45.0],
                pred_dst_ci05_nt=[-50.0, -55.0],
                pred_dst_ci95_nt=[-30.0, -35.0],
                observation_dst_nt=[-48.0, -49.0],
                residual_dst_nt=[-8.0, -4.0],
                observed_in_90ci=[true, true],
                v1_pred_dst_nt=[-38.0, -43.0],
                v2_pred_dst_nt=[-40.0, -45.0],
                v2_pred_dst_ci05_nt=[-50.0, -55.0],
                v2_pred_dst_ci95_nt=[-30.0, -35.0],
                served_pred_dst_nt=[-47.0, -50.0],
                served_pred_dst_ci05_nt=[-57.0, -60.0],
                served_pred_dst_ci95_nt=[-37.0, -40.0],
                served_residual_dst_nt=[-1.0, 1.0],
                served_observed_in_90ci=[true, true],
                v2_selected_component=["v2", "v2"],
                persistence_dst_nt=[-39.0, -44.0],
                burton_dst_nt=[-41.0, -44.0],
                burton_full_dst_nt=[-41.0, -44.0],
                obrien_dst_nt=[-46.0, -48.0],
            )
            CSV.write(log_path, df)
            write_live_comparison_report(log_path, report_path)
            text = read(report_path, String)
            @test occursin("V2.1 is the dashboard forecast", text)
            @test occursin("V2.1 90% interval coverage", text)
            @test occursin("| V2.1 | 2 |", text)
            @test occursin("| V2.1 frozen-tail ablation | 2 |", text)
            @test occursin("V2.1 frozen-tail pred", text)

            # Expanded legacy rows can contain populated served_* columns. The
            # persisted model identity, not column presence, controls labeling.
            historical = copy(df)
            historical.model_version .= "v2"
            historical_path = joinpath(dir, "historical_with_served.csv")
            historical_report = joinpath(dir, "historical_with_served.md")
            CSV.write(historical_path, historical)
            write_live_comparison_report(historical_path, historical_report)
            historical_text = read(historical_report, String)
            @test occursin("Historical V2.0 is the archived operational method", historical_text)
            @test occursin("| Historical V2.0 | 2 |", historical_text)
            @test occursin("| Historical V2.0 frozen-tail ablation | 2 |", historical_text)
            @test !occursin("V2.1 is the dashboard forecast", historical_text)

            empty_current_path = joinpath(dir, "empty_current.csv")
            empty_current_report = joinpath(dir, "empty_current.md")
            CSV.write(empty_current_path, first(df, 0))
            write_live_comparison_report(
                empty_current_path,
                empty_current_report;
                empty_identity=:v2_1,
            )
            empty_current_text = read(empty_current_report, String)
            @test occursin("Newest issued forecast: none", empty_current_text)
            @test occursin("V2.1 is the dashboard forecast", empty_current_text)
            @test occursin("Verified rows used: 0", empty_current_text)
            @test !occursin(
                "Historical V2.0 is the archived operational method",
                empty_current_text,
            )
            @test_throws ArgumentError write_live_comparison_report(
                empty_current_path,
                empty_current_report;
                empty_identity=:unknown,
            )

            mixed = copy(df)
            mixed.model_version[2] = "v2"
            @test_throws ArgumentError write_live_comparison_report(
                log_path,
                report_path;
                df=mixed,
            )
        end
    end

    @testset "Historical archive report preserves served/frozen V2.0 roles" begin
        mktempdir() do dir
            historical_log = joinpath(
                get_data_dir(), "historical", "v2_0", "live_forecast_log.csv",
            )
            report_path = joinpath(dir, "historical_report.md")
            write_live_comparison_report(historical_log, report_path)
            text = read(report_path, String)
            @test occursin("Same-row forecast comparison rows: 1533", text)
            @test occursin("Historical V2.0 90% interval coverage: 0.88", text)
            @test occursin("| Historical V2.0 | 1533 | 9.54 |", text)
            @test occursin(
                "| Historical V2.0 frozen-tail ablation | 1533 | 9.85 |",
                text,
            )
            @test !occursin("V2.1 is the dashboard forecast", text)

            summary_text = open(joinpath(dir, "historical_summary.txt"), "w+") do io
                redirect_stdout(io) do
                    summarize_log(historical_log)
                end
                seekstart(io)
                read(io, String)
            end
            @test occursin(r"Historical V2\.0\s+n=1533 RMSE=9\.54", summary_text)
            @test occursin(
                r"Historical V2\.0 frozen-tail ablation\s+n=1533 RMSE=9\.85",
                summary_text,
            )
            @test occursin("Historical V2.0 90% coverage n=1533 coverage=0.88", summary_text)
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
        # Multi-step v1 loops step_forecast!, whose band is ~5× too narrow vs the
        # forecast_ahead propagation, so it must be refused at issuance.
        @test_throws ArgumentError _assert_issuable_model(:v1, 2)
        @test_throws ArgumentError _assert_issuable_model(:v1, 6)
        @test _assert_issuable_model(:v1, 1) === nothing    # single-step v1 is fine
        @test _assert_issuable_model(:v2, 6) === nothing    # v2 serves a conformal band
        @test _assert_issuable_model(:v2, 1) === nothing
        @test_throws ArgumentError _assert_issuable_model(:v2, 4)
        @test_throws ArgumentError _assert_issuable_model(:v2, 99)

        supported = default_operational_v2_calibration(
            supported_model_steps=copy(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS),
        )
        for step in OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS
            @test _assert_supported_model_step(supported, step) === nothing
        end
        @test_throws ArgumentError _assert_supported_model_step(supported, 0)
        @test_throws ArgumentError _assert_supported_model_step(supported, 5)
        @test_throws ArgumentError _assert_supported_model_step(supported, 8)
        @test _supported_horizon_edges([1, 2, 3, 6]) ==
              [0.0, 1.5, 2.5, 4.5, Inf]
        @test _supported_horizon_edges(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS) ==
              [0.0, 1.5, 2.5, 3.5, 5.0, 6.5, Inf]
        @test_throws ArgumentError _supported_horizon_edges(Int[])
        @test_throws ArgumentError _supported_horizon_edges([0, 1])

        issue = DateTime(2026, 8, 10, 12)
        actual_steps = Int[]
        for anchor_lag in 0:LIVE_MAX_DST_ANCHOR_LAG_STEPS,
            product_horizon in V2_PRODUCT_HORIZONS
            anchor = issue - Hour(anchor_lag)
            target = _next_hourly_target(issue, product_horizon, anchor)
            step = Int((target - anchor) / Hour(1))
            @test step in OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS
            push!(actual_steps, step)
        end
        @test sort(unique(actual_steps)) == OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS
    end

    @testset "V2 tail: regime-aware relaxation and finite interval shift" begin
        driver = (V=420.0, Bz=-12.0, By=4.0, n=6.0, Pdyn=2.5)
        tau_recovery = _v2_tail_tau(5.0)
        tau_deepening = _v2_tail_tau(-30.0)
        @test tau_deepening > tau_recovery
        @test tau_deepening <= V2_TAIL_TAU_MAX_H
        @test V2_SERVED_TAIL_VERSION == "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"

        # Closed-form tau law (all exactly representable, so == is safe): recovery => no scaling,
        # deepening => tau0*(1+|rate|/r0), saturation => cap. Pins the formula SHAPE and constants so
        # a sqrt-shape or an r0/tau0 rescale (which preserve the orderings above) is caught.
        @test _v2_tail_tau(5.0) == 3.0                       # V2_TAIL_TAU0_H
        @test _v2_tail_tau(-30.0) == 15.0                    # 3*(1 + 30/7.5)
        @test _v2_tail_tau(-10_000.0) == V2_TAIL_TAU_MAX_H   # cap saturation

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
        @test _served_driver_assumption(OPERATIONAL_V2_1_MODEL_VERSION) ==
              V2_DRIVER_ASSUMPTION
        @test _served_driver_assumption("v1") == V1_DRIVER_ASSUMPTION
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
        @test _apply_v2_1_safeguards(100.0, -50.0, 1, -60.0) == -65.0
        @test _apply_v2_1_safeguards(
            100.0, -50.0, 1, -60.0; apply_rate_guard=false,
        ) == 25.0

        t0 = DateTime(2026, 6, 6, 0)
        plasma = DataFrame(time_tag=[t0 + Minute(5)], speed=[410.0], density=[5.0])
        mag = DataFrame(time_tag=[t0 + Minute(5)], bz_gsm=[-3.0], by_gsm=[1.0])
        s = _subhourly_driver_with_status(plasma, mag, t0 + Hour(2), driver, t0)
        @test !s.l1_measured
        @test s.driver == driver
    end

    @testset "V2 tail: L1 look-ahead measured branch blends the ballistic source window" begin
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

    @testset "V2 tail: subhour L1 branch rejects sparse, NaN, and partial windows" begin
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

    @testset "V2 tail: ballistic arrival is identical across completed and look-ahead steps" begin
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
        m = _live_v2_memory_features(plasma, mag, dst_times, dst_vals, t, drv, t)
        @test m.dst_delta_1h_nt == -30.0                         # -70 - (-40)
        @test m.dst_delta_3h_nt == -50.0                         # -70 - (-20)
        @test m.dst_delta_3h_nt != m.dst_delta_1h_nt

        # Driver memory follows consecutive anchor-aligned, ballistically
        # propagated source hours, matching `add_operational_v2_features!`.
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
        aligned = _live_v2_memory_features(
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
        z = _live_v2_memory_features(plasma, mag, gap3_t, gap3_v, t, drv, t)
        @test z == _zero_v2_memory_features()
        @test z.dst_delta_1h_nt == 0.0 && z.dst_delta_3h_nt == 0.0

        # The tail relaxation rate is DECOUPLED from that zeroing: the freshest contiguous pair
        # (t, t-1h) still yields the true -30 nT/h even when t-3h is missing, so tau stays long
        # (> tau0) for a deepening storm instead of collapsing to the quiet default.
        @test _freshest_dst_rate(gap3_t, gap3_v, t) == -30.0
        @test _v2_tail_tau(_freshest_dst_rate(gap3_t, gap3_v, t)) > V2_TAIL_TAU0_H
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

    @testset "F5: a thin validation split deploys the fallback, not a v2 gate on one row" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            # Tiny table whose 0.15 validation fraction yields a single validation
            # row — a degenerate 0/1 coverage check. The gate must not be trusted;
            # the v1-equivalent fallback must deploy regardless of that one row.
            n = 8
            pred = collect(-30.0:1.0:-23.0)
            observed = pred .+ 2.0                            # a correction that would "pass"
            replay = DataFrame(
                issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
                pred_dst_nt=pred,
                pred_dst_ci05_nt=pred .- 3.0,
                pred_dst_ci95_nt=pred .+ 3.0,
                observation_dst_nt=observed,
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred .- 3.0,
                v1_pred_dst_ci95_nt=pred .+ 3.0,
                latest_dst_nt=fill(-40.0, n),
                V_kms=fill(420.0, n),
                Bz_nt=fill(-2.0, n),
                By_nt=fill(1.0, n),
                n_cm3=fill(5.0, n),
                Pdyn_npa=fill(1.5, n),
            )
            CSV.write(table_path, replay)

            cfg = LiveVerifyConfig(;
                mode=:fit_v2_calibration,
                table_path=table_path,
                v2_calibration_path=cal_path,
                v2_train_fraction=0.7,
                v2_validation_fraction=0.15,           # floor(0.15*8)=1 validation row
                v2_ridge_grid=[0.0],
                v2_ridge=0.0,
            )
            _, validation, _ = _chronological_train_validation_test(
                SolarSINDy.add_operational_v2_features!(_v2_base_prediction_table(copy(replay))),
                cfg.v2_train_fraction, cfg.v2_validation_fraction,
            )
            @test nrow(validation) < V2_MIN_VALIDATION_ROWS    # degenerate gate input
            cal = fit_v2_calibration!(cfg)
            selection = CSV.read(replace(cal_path, r"\.csv$" => "_selection.csv"), DataFrame)
            @test cal.label == "operational_v2_1_fallback_v1_equiv"
            @test !any(selection.deployed)
            @test selection.deploy_block_reason[1] == "validation_split_too_thin"
            @test !selection.validation_trusted[1]
        end
    end

    @testset "F6: gate metrics share one finite-row mask across baselines" begin
        # A scored validation frame with one `missing` and one `NaN` baseline row.
        # Per-column finite filters would give v2 more rows than the baseline
        # comparators (unpaired); the common-mask metrics must report equal n.
        scored = DataFrame(
            observation_dst_nt=[-30.0, -31.0, -32.0, -33.0],
            v2_pred_dst_nt=[-29.0, -30.0, -31.0, -32.0],
            pred_dst_nt=[-29.5, -30.5, -31.5, -32.5],
            latest_dst_nt=Union{Missing,Float64}[-28.0, missing, -33.0, -34.0],
            obrien_dst_nt=[-27.0, -29.0, NaN, -35.0],
        )
        metrics = _paired_gate_metrics(
            scored, Symbol[:v2_pred_dst_nt, :pred_dst_nt, :latest_dst_nt, :obrien_dst_nt],
        )
        # Rows 2 (missing persistence) and 3 (NaN O'Brien) drop from ALL metrics.
        @test metrics[:v2_pred_dst_nt].n == 2
        @test metrics[:v2_pred_dst_nt].n == metrics[:pred_dst_nt].n ==
              metrics[:latest_dst_nt].n ==
              metrics[:obrien_dst_nt].n
        # The surviving rows are exactly the fully finite ones (rows 1 and 4).
        @test metrics[:v2_pred_dst_nt].rmse ≈ 1.0 atol = 1e-12
    end

    @testset "F7: promotion gate requires strict uncorrected-center improvement" begin
        @test !_v2_gate_pass((rmse=1.0, mae=1.0), (rmse=1.0, mae=1.0))

        n = 20
        pred = collect(-50.0:1.0:-31.0)
        replay = DataFrame(
            issue_time_utc=[string(DateTime(2026, 1, 1) + Hour(i)) for i in 1:n],
            pred_dst_nt=pred,
            pred_dst_ci05_nt=pred .- 3.0,
            pred_dst_ci95_nt=pred .+ 3.0,
            observation_dst_nt=vcat(pred[1:14] .+ 2.0, pred[15:end]),
            v1_pred_dst_nt=pred,
            v1_pred_dst_ci05_nt=pred .- 3.0,
            v1_pred_dst_ci95_nt=pred .+ 3.0,
            latest_dst_nt=pred .+ 10.0,
            V_kms=fill(420.0, n),
            Bz_nt=collect(-10.0:1.0:9.0),
            By_nt=fill(1.0, n),
            n_cm3=fill(5.0, n),
            Pdyn_npa=fill(1.5, n),
        )
        df = SolarSINDy.add_operational_v2_features!(_v2_base_prediction_table(replay))
        selection = _select_validated_v2_calibration(
            df[1:14, :],
            df[15:20, :],
            LiveVerifyConfig(; v2_ridge_grid=[0.0], v2_ridge=0.0),
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
                                     v2_pred_dst_nt=pred, observation_dst_nt=obs,
                                     latest_dst_nt=ld, issue_time_utc=iss,
                                     model_version=fill(OPERATIONAL_V2_1_MODEL_VERSION, 2n))
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
            legacy_rows.model_version .= "v2"
            CSV.write(legacy_log, legacy_rows)
            @test _aci_interval_from_log(legacy_log, 0.0, 7; latest_dst=5.0) === nothing
        end
    end

    @testset "ACI interval: residual pool is keyed on the issued model (v1 vs v2)" begin
        # A v1 issuance must be banded from v1 residuals, not the v2-calibrated pool. Build a log
        # whose v1 and v2 residual magnitudes differ sharply and confirm the band tracks the queried
        # column.
        mktempdir() do dir
            log = joinpath(dir, "log.csv")
            n = 45                                            # > warmup+5 = 35
            obs = zeros(n)
            v2_pred = [isodd(k) ? -3.0 : 3.0 for k in 1:n]    # |v2 residual| = 3
            v1_pred = [isodd(k) ? -15.0 : 15.0 for k in 1:n]  # |v1 residual| = 15
            iss = ["2026-01-01T" * lpad(string(i ÷ 60), 2, '0') * ":" * lpad(string(i % 60), 2, '0') * ":00" for i in 1:n]
            CSV.write(log, DataFrame(model_step_hours=fill(1.0, n),
                                     v1_pred_dst_nt=v1_pred, v2_pred_dst_nt=v2_pred,
                                     observation_dst_nt=obs, latest_dst_nt=fill(-10.0, n),
                                     issue_time_utc=iss,
                                     model_version=fill(OPERATIONAL_V2_1_MODEL_VERSION, n)))
            b_v2 = _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:v2_pred_dst_nt)
            b_v1 = _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:v1_pred_dst_nt)
            @test b_v2 !== nothing && b_v1 !== nothing
            hw_v2 = (b_v2[2] - b_v2[1]) / 2
            hw_v1 = (b_v1[2] - b_v1[1]) / 2
            @test hw_v1 > 2 * hw_v2                           # v1 band tracks the larger v1 residuals
            # A log lacking the requested column returns nothing (no cross-model borrowing).
            CSV.write(log, DataFrame(model_step_hours=fill(1.0, n), v2_pred_dst_nt=v2_pred,
                                     observation_dst_nt=obs, latest_dst_nt=fill(-10.0, n),
                                     issue_time_utc=iss))
            @test _aci_interval_from_log(log, 0.0, 1; latest_dst=-10.0, pred_col=:v1_pred_dst_nt) === nothing
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
                model_version=fill(OPERATIONAL_V2_1_MODEL_VERSION, n),
                model_step_hours=fill(1.0, n),
                latest_dst_nt=fill(-10.0, n),
                pred_dst_nt=pred,
                pred_dst_ci05_nt=fill(-40.0, n),
                pred_dst_ci95_nt=fill(0.0, n),
                v2_pred_dst_nt=pred,
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
                model_version=[OPERATIONAL_V2_1_MODEL_VERSION], model_step_hours=[1.0],
                latest_dst_nt=[-10.0], pred_dst_nt=[-20.0],
                pred_dst_ci05_nt=[-40.0], pred_dst_ci95_nt=[0.0],
                v2_pred_dst_nt=[-20.0], observation_dst_nt=[missing],
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
