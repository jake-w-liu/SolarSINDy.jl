using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "examples", "live_forecast_verify.jl"))

@testset "Live Forecast Verification Workflow" begin
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
        )
        @test expert.dst_delta_3h_nt == -5.0
        @test expert.baseline_spread_nt == 4.0
        @test expert.v1_minus_persistence_nt == 2.0
        @test expert.obrien_minus_v1_nt == -1.0
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
        end
    end

    @testset "C0-5: forecast log lock creates and releases lock directory" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_dir = log_path * ".lock"

            result = _with_forecast_log_lock(log_path) do
                @test isdir(lock_dir)
                :locked
            end

            @test result == :locked
            @test !isdir(lock_dir)
        end
    end

    @testset "C0-6: forecast log lock recovers stale lock directory" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            lock_dir = log_path * ".lock"
            reap_dir = lock_dir * ".reap"
            mkdir(lock_dir)

            result = _with_forecast_log_lock(log_path; stale_after_sec=0.0) do
                @test isdir(lock_dir)
                @test !isdir(reap_dir)
                :recovered
            end

            @test result == :recovered
            @test !isdir(lock_dir)
            @test !isdir(reap_dir)
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
                    "2026-06-06T07:06:43.957",
                    "2026-06-06T07:23:05.548",
                    "2026-06-06T09:15:00",
                ],
                target_time_utc=[
                    "2026-06-06T08:00:00",
                    "2026-06-06T08:00:00",
                    "2026-06-06T10:00:00",
                ],
                model_version=["v2", "v2", "v2"],
                wall_clock_lead_hours=[0.89, 0.62, 0.75],
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
            @test occursin("| V2 | 2 |", text)
            @test occursin("| SINDy v1 | 2 |", text)
            @test occursin("## Pending Rows", text)
            @test occursin("2026-06-06T10:00:00", text)
            @test occursin("| v2 |", text)
            @test occursin("## Worst V2 Misses", text)
            @test occursin("## Operational V2 Audit", text)
            @test !occursin("| V2 | 3 |", text)
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
        plasma = DataFrame(
            time_tag=times,
            density=[4.0, 4.2, 4.4, 4.6, 4.8],
            speed=[410.0, 420.0, 430.0, 440.0, 450.0],
            temperature=fill(100_000.0, length(times)),
        )
        mag = DataFrame(
            time_tag=times,
            bx_gsm=zeros(length(times)),
            by_gsm=[1.0, 1.1, 1.2, 1.3, 1.4],
            bz_gsm=[-1.0, -1.2, -1.4, -1.6, -1.8],
            bt=[1.4, 1.6, 1.8, 2.0, 2.2],
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
        @test all(df_v2.model_version .== "v2")
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
        # Longer lead departs further from the anchor persistence value.
        for r in eachrow(h2)
            @test isfinite(r.pred_dst_nt)
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
            @test cal.label == "operational_v2_fallback_v1_equiv"
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
            @test cal.label == "operational_v2_fallback_v1_equiv"
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
                return (; row_idx=idx, target_time=target, pred_dst=pred,
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
            @test occursin("V2 is the operational method", text)
        end
    end

    @testset "C0-3: live report headlines upgraded V2 when available" begin
        mktempdir() do dir
            log_path = joinpath(dir, "v2_log.csv")
            report_path = joinpath(dir, "v2_report.md")
            df = DataFrame(
                issue_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                latest_dst_time_utc=["2026-06-06T09:00:00", "2026-06-06T10:00:00"],
                target_time_utc=["2026-06-06T11:00:00", "2026-06-06T12:00:00"],
                model_version=["v2", "v2"],
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
            @test occursin("V2 is the dashboard forecast", text)
            @test occursin("V2 90% interval coverage", text)
            @test occursin("| V2 | 2 |", text)
            @test occursin("| Pre-upgrade baseline | 2 |", text)
            @test occursin("pre-upgrade baseline pred", text)
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

        # Gap classification (the issue/refuse decision, isolated for testing).
        @test _driver_gap_status(3, 2) == :ok
        @test _driver_gap_status(0, 2) == :partial
        @test _driver_gap_status(3, 0) == :partial
        @test _driver_gap_status(0, 0) == :hard
    end

    @testset "P1-1: an all-NaN density/By trailing window flags a driver gap" begin
        # Trailing hour [00:00, 01:00): finite speed and Bz, but density and By are
        # entirely NaN. Pre-fix, the gap classifier ignored density/By and reported
        # :ok, so `_drivers_for_window` silently substituted n=5/By=0 quiet defaults
        # and fabricated Pdyn/clock-angle terms with no flag. Post-fix the missing
        # density/By trailing samples must classify the window as a (partial) gap.
        t0 = DateTime(2026, 6, 6, 0)
        t1 = DateTime(2026, 6, 6, 1)
        times = [DateTime(2026, 6, 6, 0, 5), DateTime(2026, 6, 6, 0, 35)]
        plasma = DataFrame(time_tag=times, speed=[410.0, 420.0], density=[NaN, NaN])
        mag = DataFrame(time_tag=times, bz_gsm=[-2.0, -3.0], by_gsm=[NaN, NaN])

        n_speed = _window_finite_count(plasma, :speed, t0, t1)
        n_bz = _window_finite_count(mag, :bz_gsm, t0, t1)
        n_density = _window_finite_count(plasma, :density, t0, t1)
        n_by = _window_finite_count(mag, :by_gsm, t0, t1)
        @test n_speed == 2 && n_bz == 2
        # The logged finite-counts for the fabricated drivers are exactly zero.
        @test n_density == 0
        @test n_by == 0
        # Speed+Bz present but density empty → flagged as a data gap (not :ok).
        @test _driver_gap_status(n_speed, n_bz, n_density, n_by) != :ok
        @test _driver_gap_status(2, 2, 0, 3) == :partial   # density-only gap
        @test _driver_gap_status(2, 2, 3, 0) == :partial   # By-only gap
        # No gap when all four drivers have finite trailing samples.
        @test _driver_gap_status(2, 2, 3, 3) == :ok
    end

    @testset "P1-2: intermediate all-NaN driver hours increment the fallback counter" begin
        # Mirror the issuance multi-step loop predicate: each intermediate hour whose
        # window has no finite speed OR no finite Bz falls back to frozen persistence
        # drivers and must be counted (silent pre-fix). n_steps = 3 with one all-NaN
        # intermediate hour ⇒ count > 0; the same span with all hours finite ⇒ 0.
        anchor = DateTime(2026, 6, 6, 0)
        n_steps = 3
        # Hours [0,1),[1,2),[2,3). Make hour [1,2) all-NaN for speed and Bz.
        ptimes = [DateTime(2026, 6, 6, 0, 30), DateTime(2026, 6, 6, 1, 30),
                  DateTime(2026, 6, 6, 2, 30)]
        plasma_gap = DataFrame(time_tag=ptimes, speed=[410.0, NaN, 430.0],
                               density=[5.0, 5.0, 5.0])
        mag_gap = DataFrame(time_tag=ptimes, bz_gsm=[-1.0, NaN, -3.0],
                            by_gsm=[1.0, 1.0, 1.0])

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

        plasma_ok = DataFrame(time_tag=ptimes, speed=[410.0, 420.0, 430.0],
                              density=[5.0, 5.0, 5.0])
        mag_ok = DataFrame(time_tag=ptimes, bz_gsm=[-1.0, -2.0, -3.0],
                           by_gsm=[1.0, 1.0, 1.0])
        @test count_fallback(plasma_ok, mag_ok) == 0
    end

    @testset "P1-3: multi-step v1 issuance is rejected; v1 h=1 and any v2 are allowed" begin
        # Multi-step v1 loops step_forecast!, whose band is ~5× too narrow vs the
        # forecast_ahead propagation, so it must be refused at issuance.
        @test_throws ArgumentError _assert_issuable_model(:v1, 2)
        @test_throws ArgumentError _assert_issuable_model(:v1, 6)
        @test _assert_issuable_model(:v1, 1) === nothing    # single-step v1 is fine
        @test _assert_issuable_model(:v2, 6) === nothing    # v2 serves a conformal band
        @test _assert_issuable_model(:v2, 1) === nothing
    end

    @testset "V2 tail: regime-aware relaxation and finite interval shift" begin
        driver = (V=420.0, Bz=-12.0, By=4.0, n=6.0, Pdyn=2.5)
        tau_recovery = _v2_tail_tau(5.0)
        tau_deepening = _v2_tail_tau(-30.0)
        @test tau_deepening > tau_recovery
        @test tau_deepening <= V2_TAIL_TAU_MAX_H
        @test V2_SERVED_TAIL_VERSION == "v2+L1A+Bregime+Pinertia"

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

        lo, hi = _shift_interval_to_center(-90.0, -80.0, -100.0, -60.0)
        @test (lo, hi) == (-110.0, -70.0)
        @test_throws ArgumentError _shift_interval_to_center(NaN, -80.0, -100.0, -60.0)
        @test _near_term_extreme_inertia_guard(-250.0, 1)
        @test _near_term_extreme_inertia_guard(-250.0, 2)
        @test !_near_term_extreme_inertia_guard(-250.0, 3)
        @test !_near_term_extreme_inertia_guard(-239.9, 2)

        t0 = DateTime(2026, 6, 6, 0)
        plasma = DataFrame(time_tag=[t0 + Minute(5)], speed=[410.0], density=[5.0])
        mag = DataFrame(time_tag=[t0 + Minute(5)], bz_gsm=[-3.0], by_gsm=[1.0])
        s = _subhourly_driver_with_status(plasma, mag, t0 + Hour(2), driver, t0)
        @test !s.l1_measured
        @test s.driver == driver
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
            @test cal.label == "operational_v2_fallback_v1_equiv"
            @test !any(selection.deployed)
            @test selection.deploy_block_reason[1] == "validation_split_too_thin"
            @test !selection.validation_trusted[1]
        end
    end

    @testset "F6: gate metrics share one finite-row mask across baselines" begin
        # A scored validation frame with one `missing` and one `NaN` baseline row.
        # Per-column finite filters would give v2 more rows than persistence/O'Brien
        # (unpaired); the common-mask metrics must report equal n across all three.
        scored = DataFrame(
            observation_dst_nt=[-30.0, -31.0, -32.0, -33.0],
            v2_pred_dst_nt=[-29.0, -30.0, -31.0, -32.0],
            latest_dst_nt=Union{Missing,Float64}[-28.0, missing, -33.0, -34.0],
            obrien_dst_nt=[-27.0, -29.0, NaN, -35.0],
        )
        metrics = _paired_gate_metrics(
            scored, Symbol[:v2_pred_dst_nt, :latest_dst_nt, :obrien_dst_nt],
        )
        # Rows 2 (missing persistence) and 3 (NaN O'Brien) drop from ALL metrics.
        @test metrics[:v2_pred_dst_nt].n == 2
        @test metrics[:v2_pred_dst_nt].n == metrics[:latest_dst_nt].n ==
              metrics[:obrien_dst_nt].n
        # The surviving rows are exactly the fully finite ones (rows 1 and 4).
        @test metrics[:v2_pred_dst_nt].rmse ≈ 1.0 atol = 1e-12
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
            CSV.write(log, DataFrame(model_step_hours=ms, horizon_hours=hh,
                                     v2_pred_dst_nt=pred, observation_dst_nt=obs,
                                     latest_dst_nt=ld, issue_time_utc=iss))
            # Quiet query: pools only the ±5 rows -> narrow band (NOT the old ~nothing/fallback).
            q = _aci_interval_from_log(log, 0.0, 7; latest_dst=5.0)
            @test q !== nothing                            # would be `nothing` under the horizon-key bug
            hw_q = (q[2] - q[1]) / 2
            @test 3.0 <= hw_q <= 12.0                       # tracks the ±5 quiet residuals
            # Disturbed query: pools the ±60 rows -> much wider band (regime conditioning).
            d = _aci_interval_from_log(log, 0.0, 7; latest_dst=-80.0)
            @test d !== nothing
            hw_d = (d[2] - d[1]) / 2
            @test hw_d > 3 * hw_q                           # storm regime band is far wider than quiet
        end
    end
end
