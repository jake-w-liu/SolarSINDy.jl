using Test

module ExternalDstCollectorTestHarness
using Test

# In-package collector (examples/). It is a committed part of the package, so a missing file is a
# real regression, not an environment-specific skip.
const EXTERNAL_DST_COLLECTOR_SCRIPT = normpath(joinpath(@__DIR__, "..", "examples",
                                                        "external_dst_snapshot_collector.jl"))

if isfile(EXTERNAL_DST_COLLECTOR_SCRIPT)
    include(EXTERNAL_DST_COLLECTOR_SCRIPT)
end

end

module LiveMonitorRetentionTestHarness
using Test
include(joinpath(@__DIR__, "..", "examples", "live_monitor.jl"))
end

@testset "Prospective external Dst snapshot collector" begin
    @test isfile(ExternalDstCollectorTestHarness.EXTERNAL_DST_COLLECTOR_SCRIPT)
    C = ExternalDstCollectorTestHarness
    @test C._parse_http_last_modified(["Last-Modified" => "Sat, 27 Jun 2026 05:10:00 GMT"]) ==
          C.DateTime(2026, 6, 27, 5, 10, 0)
    @test C._parse_temerin_model_run("Time of model run:     2026/178-05:05:44") ==
          C.DateTime(2026, 6, 27, 5, 5, 44)
    @test C._sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    @test C._selftest_external_dst_collector()

    @testset "configurable observation tolerance and stable RMSE" begin
        log = C._external_empty_log()
        push!(log, (
            "mock", "2026-07-01T05:00:00Z", "2026-07-01T05:00:00Z",
            "2026-07-01T06:00:00Z", 1.0, -30.0, 60.0, "fetch_time",
            "forecast", repeat("a", 64), "raw/mock.raw", missing, missing,
            "2026-07-01T06:00:00Z", "future_forecast", missing, missing,
            missing, missing, missing,
        ))
        observations = C.DataFrame(
            observed_time_utc=[C.DateTime(2026, 7, 1, 6, 45)],
            observed_dst_nt=[-31.0],
        )
        @test C.score_external_dst_rows!(
            log, observations; max_obs_gap_min=60.0,
        ) == 1
        @test C._validate_external_dst_log(log; max_obs_gap_min=60.0)
        @test_throws ErrorException C._validate_external_dst_log(log)
        @test_throws ArgumentError C.score_external_dst_rows!(
            log, observations; max_obs_gap_min=Inf,
        )

        wide = C.DataFrame(
            source=["mock"], issue_utc=["2026-07-01T05:00:00Z"],
            lead_h=[1.0], forecast_dst_nt=[floatmax(Float64) / 2],
            observed_dst_nt=Union{Missing, Float64}[0.0],
        )
        summary = C.external_dst_summary(wide)
        @test isfinite(summary.rmse_nt[1])
        @test summary.rmse_nt[1] == floatmax(Float64) / 2

        mktempdir() do dir
            report_path = joinpath(dir, "report.md")
            C._write_external_dst_report(
                report_path, log; max_obs_gap_min=60.0,
            )
            @test occursin("within 60.0 min", read(report_path, String))
        end
    end

    @testset "atomic replacement and corrupt raw recovery" begin
        mktempdir() do dir
            target = joinpath(dir, "target.txt")
            write(target, "old-complete")
            failing_writer = function (io)
                write(io, "partial")
                error("injected writer failure")
            end
            @test_throws ErrorException C._external_atomic_file(failing_writer, target)
            @test read(target, String) == "old-complete"

            body = "complete raw response"
            sha = C._sha256_hex(body)
            raw_dir = joinpath(dir, "raw")
            rel = C._write_raw_snapshot(
                raw_dir, "source", C.DateTime(2026, 7, 1), sha, body, dir,
            )
            raw_path = joinpath(dir, rel)
            write(raw_path, "torn")
            @test C._write_raw_snapshot(
                raw_dir, "source", C.DateTime(2026, 7, 1), sha, body, dir,
            ) == rel
            @test read(raw_path, String) == body
            @test C.bytes2hex(C.sha256(read(raw_path))) == sha
        end
    end

    @testset "log/report file-set rollback" begin
        mktempdir() do dir
            log_path = joinpath(dir, "external.csv")
            report_path = joinpath(dir, "external.md")
            old_log = Vector{UInt8}(codeunits("OLD LOG\n"))
            old_report = Vector{UInt8}(codeunits("OLD REPORT\n"))
            write(log_path, old_log)
            write(report_path, old_report)
            @test_throws ErrorException C._external_transactional_log_report!(
                log_path, report_path, C._external_empty_log();
                after_log_commit=() -> error("injected post-log failure"),
            )
            @test read(log_path) == old_log
            @test read(report_path) == old_report

            rm(report_path)
            symlink(joinpath(dir, "elsewhere.md"), report_path)
            @test_throws ArgumentError C._external_transactional_log_report!(
                log_path, report_path, C._external_empty_log(),
            )
            @test read(log_path) == old_log
        end


        mktempdir() do dir
            real_parent = joinpath(dir, "real")
            alias_parent = joinpath(dir, "alias")
            mkpath(real_parent)
            symlink(real_parent, alias_parent)
            log_path = joinpath(real_parent, "same-output")
            report_path = joinpath(alias_parent, "same-output")
            write(log_path, "UNCHANGED\n")
            @test C._external_targets_alias(log_path, report_path)
            @test_throws ArgumentError C._external_transactional_log_report!(
                log_path, report_path, C._external_empty_log(),
            )
            @test read(log_path, String) == "UNCHANGED\n"
        end


        mktempdir() do dir
            log_path = joinpath(dir, "same-output")
            report_path = joinpath(dir, "SAME-OUTPUT")
            @test C._external_targets_alias(log_path, report_path)
            @test_throws ArgumentError C._external_transactional_log_report!(
                log_path, report_path, C._external_empty_log(),
            )
            @test !ispath(log_path)
            @test !ispath(report_path)
        end
    end

    @testset "collector lock serializes concurrent read-merge-write" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            obs_body = """[
              {"time_tag":"2026-07-01T06:00:00","dst":-31.0},
              {"time_tag":"2026-07-01T07:00:00","dst":-41.0}
            ]"""
            bodies = (
                """[{"time_tag":"2026-07-01T06:00:00","dst":-30.0}]""",
                """[{"time_tag":"2026-07-01T07:00:00","dst":-40.0}]""",
            )
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            ready = Channel{Nothing}(2)
            release = Channel{Nothing}(2)
            fake_get(i) = function (url; kwargs...)
                if String(url) == "forecast"
                    put!(ready, nothing)
                    take!(release)
                    return C._mock_response(bodies[i])
                end
                return C._mock_response(obs_body)
            end
            task1 = @async C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get(1),
            )
            task2 = @async C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 6), http_get=fake_get(2),
            )
            take!(ready); take!(ready)
            put!(release, nothing); put!(release, nothing)
            wait(task1); wait(task2)
            out = C.CSV.read(cfg.log_path, C.DataFrame)
            @test C.nrow(out) == 2
            @test sort(out.forecast_dst_nt) == [-40.0, -30.0]
            @test !ispath(cfg.log_path * ".lock")
        end
    end

    @testset "row/raw retention keeps every retained raw reference" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            generation = Ref(1)
            forecast_bodies = (
                """[{"time_tag":"2026-07-01T06:00:00","dst":-30.0}]""",
                """[{"time_tag":"2026-07-01T07:00:00","dst":-40.0}]""",
            )
            obs_body = """[
              {"time_tag":"2026-07-01T06:00:00","dst":-31.0},
              {"time_tag":"2026-07-01T07:00:00","dst":-41.0}
            ]"""
            fake_get = function (url; kwargs...)
                return C._mock_response(
                    String(url) == "forecast" ? forecast_bodies[generation[]] : obs_body,
                )
            end
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=1,
            )
            C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            generation[] = 2
            result = C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 6), http_get=fake_get,
            )
            out = C.CSV.read(cfg.log_path, C.DataFrame)
            @test result.rows_total == 1
            @test result.rows_dropped == 1
            @test result.raw_pruned == 1
            @test length(filter(name -> endswith(name, ".raw"), readdir(cfg.raw_dir))) == 1
            @test all(isfile(joinpath(cfg.repo_root, String(path))) for path in out.raw_path)
        end
    end


    @testset "retained raw corruption fails closed" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            generation = Ref(1)
            forecast_bodies = (
                """[{"time_tag":"2026-07-01T06:00:00","dst":-30.0}]""",
                """[{"time_tag":"2026-07-01T07:00:00","dst":-40.0}]""",
            )
            obs_body = """[
              {"time_tag":"2026-07-01T06:00:00","dst":-31.0},
              {"time_tag":"2026-07-01T07:00:00","dst":-41.0}
            ]"""
            fake_get = function (url; kwargs...)
                body = String(url) == "forecast" ?
                    forecast_bodies[generation[]] : obs_body
                return C._mock_response(body)
            end
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=2,
            )
            C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            first_log = C.CSV.read(cfg.log_path, C.DataFrame)
            corrupt_path = joinpath(cfg.repo_root, String(only(first_log.raw_path)))
            write(corrupt_path, "corrupted retained response")
            old_log = read(cfg.log_path)
            old_report = read(cfg.report_path)

            generation[] = 2
            failure = try
                C.capture_and_score_external_dst_snapshot!(
                    cfg; fetched_utc=C.DateTime(2026, 7, 1, 6), http_get=fake_get,
                )
                nothing
            catch err
                err
            end
            @test failure isa ErrorException
            @test occursin("retained raw snapshot SHA-256 mismatch",
                           sprint(showerror, failure))
            @test read(cfg.log_path) == old_log
            @test read(cfg.report_path) == old_report
        end
    end

    bad_cfg = C.ExternalDstCollectorConfig(max_log_rows=0)
    @test_throws ArgumentError C.capture_and_score_external_dst_snapshot!(
        bad_cfg; http_get=(args...; kwargs...) -> error("must not fetch"),
    )
end

@testset "Live monitor forecast-log retention" begin
    L = LiveMonitorRetentionTestHarness
    mktempdir() do dir
        path = joinpath(dir, "live.csv")
        base = L.DateTime(2026, 7, 1)
        rows = L.DataFrame(
            issue_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            latest_dst_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            target_time_utc=string.([base + L.Hour(i + 1) for i in 0:5]),
            model_version=fill("v2", 6),
            observation_dst_nt=fill(missing, 6),
            marker=collect(1:6),
        )
        L.CSV.write(path, rows)
        L._load_or_rebuild_live_state!(path)
        @test L._retain_live_forecast_log!(path, 3) == 3
        retained = L.CSV.read(path, L.DataFrame)
        @test retained.marker == [4, 5, 6]
        state = L._read_live_state(path)
        @test state !== nothing
        @test state["row_count"] == 3
        @test L._state_matches_log(state, path)
        @test isempty(state["aci_streams"])
        @test L._retain_live_forecast_log!(path, 3) == 0
        @test_throws ArgumentError L._retain_live_forecast_log!(path, 0)
    end
end
