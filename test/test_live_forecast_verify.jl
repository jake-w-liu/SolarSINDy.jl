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
            @test occursin("Same-row v2 comparison rows: 2", text)
            @test occursin("## Same-Row Model Comparison", text)
            @test occursin("| Operational v2 | 2 |", text)
            @test occursin("| SINDy v1 | 2 |", text)
            @test occursin("## Pending Rows", text)
            @test occursin("2026-06-06T10:00:00", text)
            @test occursin("| v2 |", text)
            @test occursin("## Worst Operational V2 Misses", text)
            @test occursin("## Operational V2 Audit", text)
            @test !occursin("| Operational v2 | 3 |", text)
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
            @test startswith(cal.label, "operational_v2_validated_")
            @test cal.selected_component == :ensemble
            reread = read_operational_v2_calibration(cal_path)
            scored = CSV.read(scored_path, DataFrame)
            selection = CSV.read(selection_path, DataFrame)
            @test maximum(abs.(scored.v2_residual_dst_nt)) < 0.5
            @test all(scored.v2_selected_component .== "ensemble")
            @test Set(scored.v2_split) == Set(["fit", "validation", "holdout"])
            @test nrow(selection) == 4
            @test any(selection.selected_by_validation)
            @test all(selection.holdout_gate_pass)
            @test :final_benchmark_component in propertynames(selection)
            @test all(occursin(";", w) for w in selection.selector_weights)
            @test sum(reread.selector_weights) ≈ 1.0 atol=1e-12
            pred_v2 = operational_v2_predict(
                reread,
                pred[end],
                pred[end] - 3.0,
                pred[end] + 3.0,
                _v2_features(pred[end] - 1.0, (; V=420.0, Bz=bz[end], By=1.0, n=5.0, Pdyn=1.5)),
            )
            @test pred_v2.pred_dst ≈ observed[end] atol=0.5
        end
    end

    @testset "A/D: validated ensemble weights favor SINDy v1 when correction fails validation" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            n = 24
            pred = collect(-50.0:1.0:-27.0)
            observed = copy(pred)
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
            idx_v1 = findfirst(==(:sindy_v1), cal.selector_names)
            idx_v2 = findfirst(==(:v2), cal.selector_names)
            @test cal.selected_component == :ensemble
            @test all(scored.v2_selected_component .== "ensemble")
            @test cal.selector_weights[idx_v1] > cal.selector_weights[idx_v2]
            @test all(selection.selected_component .== "ensemble")
            @test all(selection.final_component .== "ensemble")
            @test all(selection.holdout_gate_pass)
            @test minimum(selection.validation_rmse_nt) <= 1.0
            @test selection.validation_v1_rmse_nt[1] == 0.0
        end
    end

    @testset "A/D: validated v2 shrinks ensemble toward SINDy v1 when holdout gate fails" begin
        mktempdir() do tmp
            table_path = joinpath(tmp, "replay.csv")
            cal_path = joinpath(tmp, "v2_calibration.csv")
            n = 30
            pred = collect(-60.0:1.0:-31.0)
            observed = copy(pred)
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
            @test cal.selected_component == :ensemble
            @test occursin("holdout_shrink", cal.label)
            @test all(selection.final_component .== "ensemble")
            @test all(selection.holdout_gate_pass)
            @test all(selection.holdout_shrink_alpha .== 0.0)
            @test any(selection.selected_by_validation)
            @test holdout.v2_pred_dst_nt == holdout.pred_dst_nt
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
            @test occursin("Same-row v2 comparison rows: 2", text)
            @test occursin("Operational v2 is the upgraded method", text)
        end
    end
end
