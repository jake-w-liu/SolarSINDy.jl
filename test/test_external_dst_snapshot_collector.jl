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

function _two_point_swpc_fixture(target::AbstractString, dst::Real)
    return """[
      {"time_tag":"2026-07-01T04:59:00","dst":0.0},
      {"time_tag":"$target","dst":$dst}
    ]"""
end

@testset "Prospective external Dst snapshot collector" begin
    @test isfile(ExternalDstCollectorTestHarness.EXTERNAL_DST_COLLECTOR_SCRIPT)
    C = ExternalDstCollectorTestHarness
    @test C._parse_http_last_modified(["Last-Modified" => "Sat, 27 Jun 2026 05:10:00 GMT"]) ==
          C.DateTime(2026, 6, 27, 5, 10, 0)
    @test C._parse_temerin_model_run("Time of model run:     2026/178-05:05:44") ==
          C.DateTime(2026, 6, 27, 5, 5, 44)
    @test C._parse_temerin_model_run("Time of model run: 2024/060-12:34:56") ==
          C.DateTime(2024, 2, 29, 12, 34, 56)
    leap_row = C._parse_temerin_ascii("2024/060-12:34:56 -42.5")
    @test C.nrow(leap_row) == 1
    @test leap_row.target_utc[1] == C.DateTime(2024, 2, 29, 12, 34, 56)
    for timestamp in (
        "0000/001-00:00:00", "2026/000-00:00:00", "2026/366-00:00:00",
        "2024/367-00:00:00",
        "2026/178-24:00:00", "2026/178-23:60:00", "2026/178-23:59:60",
        "2026/999-99:99:99",
    )
        @test ismissing(C._parse_temerin_model_run("Time of model run: $timestamp"))
        @test isempty(C._parse_temerin_ascii("$timestamp -42.0"))
    end
    @test C._sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    @test C._selftest_external_dst_collector()

    cadence_start = C.DateTime(2026, 7, 1)
    @test C._median_cadence_min([cadence_start, cadence_start + C.Minute(10)]) == 10.0
    @test isnan(C._median_cadence_min([cadence_start]))
    @test isnan(C._median_cadence_min([cadence_start, cadence_start]))
    @test isnan(C._median_cadence_min([cadence_start + C.Minute(1), cadence_start]))

    cadence_source = (; name="cadence", url="forecast", kind="swpc_geospace_json")
    one_row_get = (url; kwargs...) -> C._mock_response(
        """[{"time_tag":"2026-07-01T08:00:00","dst":-30.0}]""",
    )
    @test_throws ErrorException C._staged_future_rows_for_source(
        cadence_source; fetched_utc=C.DateTime(2026, 7, 1, 5),
        http_get=one_row_get,
    )
    duplicate_get = (url; kwargs...) -> C._mock_response("""[
      {"time_tag":"2026-07-01T08:00:00","dst":-30.0},
      {"time_tag":"2026-07-01T08:00:00","dst":-31.0}
    ]""")
    @test_throws ErrorException C._staged_future_rows_for_source(
        cadence_source; fetched_utc=C.DateTime(2026, 7, 1, 5),
        http_get=duplicate_get,
    )

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
        for invalid_cadence in (NaN, Inf, 0.0, -1.0)
            bad_cadence = copy(log)
            bad_cadence.forecast_cadence_min[1] = invalid_cadence
            @test_throws ErrorException C._validate_external_dst_log(
                bad_cadence; max_obs_gap_min=60.0,
            )
        end
        wrong_lead = copy(log)
        wrong_lead.lead_h[1] = 9.0
        @test_throws ErrorException C._validate_external_dst_log(
            wrong_lead; max_obs_gap_min=60.0,
        )
        bad_fetch = copy(log)
        bad_fetch.fetched_utc[1] = "not-a-time"
        @test_throws ErrorException C._validate_external_dst_log(
            bad_fetch; max_obs_gap_min=60.0,
        )

        wrong_gap = copy(log)
        wrong_gap.observed_gap_min[1] = 44.0
        @test_throws ErrorException C._validate_external_dst_log(
            wrong_gap; max_obs_gap_min=60.0,
        )
        wrong_error = copy(log)
        wrong_error.abs_error_nt[1] = 1.001
        @test_throws ErrorException C._validate_external_dst_log(
            wrong_error; max_obs_gap_min=60.0,
        )
        partial_score = copy(log)
        partial_score.scored_utc[1] = missing
        @test_throws ErrorException C._validate_external_dst_log(
            partial_score; max_obs_gap_min=60.0,
        )
        unscored_partial = C._external_empty_log()
        push!(unscored_partial, (
            "mock", "2026-07-01T05:00:00Z", "2026-07-01T05:00:00Z",
            "2026-07-01T06:00:00Z", 1.0, -30.0, 60.0, "fetch_time",
            "forecast", repeat("a", 64), "raw/mock.raw", missing, missing,
            "2026-07-01T06:00:00Z", "future_forecast", missing, missing,
            missing, missing, "2026-07-01T06:00:00Z",
        ))
        @test_throws ErrorException C._validate_external_dst_log(unscored_partial)

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

    @testset "scoring waits for target and observation maturity" begin
        log = C._external_empty_log()
        push!(log, (
            "mock", "2026-07-01T05:00:00Z", "2026-07-01T05:00:00Z",
            "2026-07-01T06:00:00Z", 1.0, -30.0, 60.0, "fetch_time",
            "forecast", repeat("a", 64), "raw/mock.raw", missing, missing,
            "2026-07-01T06:00:00Z", "future_forecast", missing, missing,
            missing, missing, missing,
        ))
        observations = C.DataFrame(
            observed_time_utc=[C.DateTime(2026, 7, 1, 6)],
            observed_dst_nt=[-31.0],
        )
        @test C.score_external_dst_rows!(
            log, observations; scored_utc=C.DateTime(2026, 7, 1, 5, 59),
        ) == 0
        @test ismissing(log.observed_dst_nt[1])
        @test C.score_external_dst_rows!(
            log, observations; scored_utc=C.DateTime(2026, 7, 1, 6),
        ) == 1
        @test log.observed_dst_nt[1] == -31.0
        @test C._validate_external_dst_log(log)
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
            @test C._external_file_sha256_hex(raw_path) == sha

            large_raw = joinpath(raw_dir, "large.raw")
            large_bytes = fill(UInt8(0x5a), 2_000_000)
            write(large_raw, large_bytes)
            expected_large = C.bytes2hex(C.sha256(large_bytes))
            @test C._external_file_sha256_hex(large_raw) == expected_large
            C._external_file_sha256_hex(large_raw)  # compile before measuring steady-state work
            GC.gc()
            streamed_allocations = @allocated C._external_file_sha256_hex(large_raw)
            @test streamed_allocations < 256_000
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
                _two_point_swpc_fixture("2026-07-01T06:00:00", -30.0),
                _two_point_swpc_fixture("2026-07-01T07:00:00", -40.0),
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
            @test !ispath(C._external_raw_store_lock_path(cfg.raw_dir))
        end
    end

    @testset "raw store binds one canonical log and repo root" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            forecast_body = _two_point_swpc_fixture("2026-07-01T08:00:00", -30.0)
            obs_body = """[{"time_tag":"2026-07-01T08:00:00","dst":-31.0}]"""
            fake_get = (url; kwargs...) -> C._mock_response(
                String(url) == "forecast" ? forecast_body : obs_body,
            )
            raw_dir = joinpath(dir, "raw")
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "owner.csv"),
                report_path=joinpath(dir, "owner.md"), raw_dir, repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            alias_root = joinpath(dir, "root-alias")
            symlink(dir, alias_root)
            alias_cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(alias_root, "owner.csv"),
                report_path=joinpath(alias_root, "owner.md"),
                raw_dir=joinpath(alias_root, "raw"), repo_root=alias_root,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            alias_result = C.capture_and_score_external_dst_snapshot!(
                alias_cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            @test alias_result.rows_total == 1
            marker = C._external_raw_store_owner_path(raw_dir)
            log_marker = C._external_log_raw_store_owner_path(cfg.log_path)
            @test isfile(marker)
            @test !islink(marker)
            @test isfile(log_marker)
            @test !islink(log_marker)
            old_marker = read(marker)
            old_log_marker = read(log_marker)
            old_log = read(cfg.log_path)
            old_report = read(cfg.report_path)
            old_raw = Set(readdir(raw_dir))

            conflicting_log = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "other.csv"),
                report_path=joinpath(dir, "other.md"), raw_dir, repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            log_failure = try
                C.capture_and_score_external_dst_snapshot!(
                    conflicting_log; fetched_utc=C.DateTime(2026, 7, 1, 6),
                    http_get=fake_get,
                )
                nothing
            catch err
                err
            end
            @test log_failure isa ArgumentError
            @test occursin("different canonical storage identity",
                           sprint(showerror, log_failure))
            @test !ispath(conflicting_log.log_path)
            @test !ispath(conflicting_log.report_path)

            other_raw_dir = joinpath(dir, "other-raw")
            conflicting_raw = C.ExternalDstCollectorConfig(;
                log_path=cfg.log_path, report_path=cfg.report_path,
                raw_dir=other_raw_dir, repo_root=dir, sources=[source], obs_url="obs",
                max_log_rows=10, max_raw_snapshots=10,
            )
            raw_failure = try
                C.capture_and_score_external_dst_snapshot!(
                    conflicting_raw; fetched_utc=C.DateTime(2026, 7, 1, 7),
                    http_get=fake_get,
                )
                nothing
            catch err
                err
            end
            @test raw_failure isa ArgumentError
            @test occursin("different canonical raw store", sprint(showerror, raw_failure))
            @test isempty(readdir(other_raw_dir))

            other_root = joinpath(dir, "other-root")
            mkpath(other_root)
            conflicting_root = C.ExternalDstCollectorConfig(;
                log_path=cfg.log_path, report_path=cfg.report_path,
                raw_dir, repo_root=other_root, sources=[source], obs_url="obs",
                max_log_rows=10, max_raw_snapshots=10,
            )
            root_failure = try
                C.capture_and_score_external_dst_snapshot!(
                    conflicting_root; fetched_utc=C.DateTime(2026, 7, 1, 8),
                    http_get=fake_get,
                )
                nothing
            catch err
                err
            end
            @test root_failure isa ArgumentError
            @test occursin("canonical", sprint(showerror, root_failure))
            @test read(marker) == old_marker
            @test read(log_marker) == old_log_marker
            @test read(cfg.log_path) == old_log
            @test read(cfg.report_path) == old_report
            @test Set(readdir(raw_dir)) == old_raw
            @test !ispath(C._external_raw_store_lock_path(raw_dir))
            @test !ispath(cfg.log_path * ".lock")
            @test !ispath(conflicting_log.log_path * ".lock")
        end
    end

    @testset "raw store rejects concurrent different logs" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            forecast_body = _two_point_swpc_fixture("2026-07-01T08:00:00", -30.0)
            obs_body = """[{"time_tag":"2026-07-01T08:00:00","dst":-31.0}]"""
            ready = Channel{Nothing}(2)
            release = Channel{Nothing}(2)
            fake_get = function (url; kwargs...)
                if String(url) == "forecast"
                    put!(ready, nothing)
                    take!(release)
                    return C._mock_response(forecast_body)
                end
                return C._mock_response(obs_body)
            end
            raw_dir = joinpath(dir, "raw")
            cfgs = ntuple(2) do i
                C.ExternalDstCollectorConfig(;
                    log_path=joinpath(dir, "collector-$i.csv"),
                    report_path=joinpath(dir, "collector-$i.md"), raw_dir,
                    repo_root=dir, sources=[source], obs_url="obs", max_log_rows=10,
                    max_raw_snapshots=10,
                )
            end
            run_capture = function (cfg, hour)
                try
                    return C.capture_and_score_external_dst_snapshot!(
                        cfg; fetched_utc=C.DateTime(2026, 7, 1, hour),
                        http_get=fake_get,
                    )
                catch err
                    return err
                end
            end
            tasks = (
                @async(run_capture(cfgs[1], 5)),
                @async(run_capture(cfgs[2], 6)),
            )
            take!(ready); take!(ready)
            put!(release, nothing); put!(release, nothing)
            results = fetch.(tasks)
            @test count(result -> result isa NamedTuple, results) == 1
            @test count(result -> result isa ArgumentError, results) == 1
            failure = only(filter(result -> result isa ArgumentError, results))
            @test occursin("different canonical storage identity", sprint(showerror, failure))
            @test count(cfg -> isfile(cfg.log_path), cfgs) == 1
            @test count(cfg -> isfile(cfg.report_path), cfgs) == 1
            @test length(filter(name -> endswith(name, ".raw"), readdir(raw_dir))) == 1
            winner = isfile(cfgs[1].log_path) ? cfgs[1] : cfgs[2]
            out = C.CSV.read(winner.log_path, C.DataFrame)
            @test C.nrow(out) == 1
            @test all(isfile(joinpath(winner.repo_root, String(path))) for path in out.raw_path)
            @test isfile(C._external_raw_store_owner_path(raw_dir))
            @test isfile(C._external_log_raw_store_owner_path(winner.log_path))
            @test !ispath(C._external_raw_store_lock_path(raw_dir))
            @test all(!ispath(cfg.log_path * ".lock") for cfg in cfgs)
        end
    end

    @testset "log rejects concurrent different raw stores" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            forecast_body = _two_point_swpc_fixture("2026-07-01T08:00:00", -30.0)
            obs_body = """[{"time_tag":"2026-07-01T08:00:00","dst":-31.0}]"""
            ready = Channel{Nothing}(2)
            release = Channel{Nothing}(2)
            fake_get = function (url; kwargs...)
                if String(url) == "forecast"
                    put!(ready, nothing)
                    take!(release)
                    return C._mock_response(forecast_body)
                end
                return C._mock_response(obs_body)
            end
            log_path = joinpath(dir, "external.csv")
            report_path = joinpath(dir, "external.md")
            cfgs = ntuple(2) do i
                C.ExternalDstCollectorConfig(;
                    log_path, report_path, raw_dir=joinpath(dir, "raw-$i"),
                    repo_root=dir, sources=[source], obs_url="obs", max_log_rows=10,
                    max_raw_snapshots=10,
                )
            end
            run_capture = function (cfg, hour)
                try
                    return C.capture_and_score_external_dst_snapshot!(
                        cfg; fetched_utc=C.DateTime(2026, 7, 1, hour),
                        http_get=fake_get,
                    )
                catch err
                    return err
                end
            end
            tasks = (
                @async(run_capture(cfgs[1], 5)),
                @async(run_capture(cfgs[2], 6)),
            )
            take!(ready); take!(ready)
            put!(release, nothing); put!(release, nothing)
            results = fetch.(tasks)
            @test count(result -> result isa NamedTuple, results) == 1
            @test count(result -> result isa ArgumentError, results) == 1
            failure = only(filter(result -> result isa ArgumentError, results))
            @test occursin("different canonical raw store", sprint(showerror, failure))
            @test isfile(log_path)
            @test isfile(report_path)
            out = C.CSV.read(log_path, C.DataFrame)
            @test C.nrow(out) == 1
            @test sum(length(filter(name -> endswith(name, ".raw"),
                                    readdir(cfg.raw_dir))) for cfg in cfgs) == 1
            @test count(cfg -> isfile(C._external_raw_store_owner_path(cfg.raw_dir)),
                        cfgs) == 1
            @test isfile(C._external_log_raw_store_owner_path(log_path))
            @test all(!ispath(C._external_raw_store_lock_path(cfg.raw_dir)) for cfg in cfgs)
            @test !ispath(log_path * ".lock")
        end
    end

    @testset "raw-store ownership marker fails closed" begin
        source = (; name="mock", url="forecast", kind="swpc_geospace_json")
        forecast_body = _two_point_swpc_fixture("2026-07-01T08:00:00", -30.0)
        obs_body = """[{"time_tag":"2026-07-01T08:00:00","dst":-31.0}]"""
        fake_get = (url; kwargs...) -> C._mock_response(
            String(url) == "forecast" ? forecast_body : obs_body,
        )
        for marker_side in (:raw, :log), unsafe_kind in (:symlink, :directory)
            mktempdir() do dir
                raw_dir = joinpath(dir, "raw")
                cfg = C.ExternalDstCollectorConfig(;
                    log_path=joinpath(dir, "external.csv"),
                    report_path=joinpath(dir, "external.md"), raw_dir,
                    repo_root=dir, sources=[source], obs_url="obs",
                    max_log_rows=10, max_raw_snapshots=10,
                )
                marker = marker_side == :raw ?
                    C._external_raw_store_owner_path(raw_dir) :
                    C._external_log_raw_store_owner_path(cfg.log_path)
                if unsafe_kind == :symlink
                    target = joinpath(dir, "elsewhere.owner")
                    write(target, "must remain unchanged")
                    symlink(target, marker)
                else
                    mkdir(marker)
                end
                @test_throws ArgumentError C.capture_and_score_external_dst_snapshot!(
                    cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
                )
                @test !ispath(cfg.log_path)
                @test !ispath(cfg.report_path)
                @test isempty(filter(name -> endswith(name, ".raw"), readdir(raw_dir)))
                @test !ispath(C._external_raw_store_lock_path(raw_dir))
                @test !ispath(cfg.log_path * ".lock")
                other_marker = marker_side == :raw ?
                    C._external_log_raw_store_owner_path(cfg.log_path) :
                    C._external_raw_store_owner_path(raw_dir)
                @test !ispath(other_marker)
                if unsafe_kind == :symlink
                    @test read(joinpath(dir, "elsewhere.owner"), String) ==
                          "must remain unchanged"
                end
            end
        end

        for marker_side in (:raw, :log)
            mktempdir() do dir
                raw_dir = joinpath(dir, "raw")
                cfg = C.ExternalDstCollectorConfig(;
                    log_path=joinpath(dir, "external.csv"),
                    report_path=joinpath(dir, "external.md"), raw_dir,
                    repo_root=dir, sources=[source], obs_url="obs",
                    max_log_rows=10, max_raw_snapshots=10,
                )
                C.capture_and_score_external_dst_snapshot!(
                    cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
                )
                marker = marker_side == :raw ?
                    C._external_raw_store_owner_path(raw_dir) :
                    C._external_log_raw_store_owner_path(cfg.log_path)
                write(marker, "corrupt marker")
                old_log = read(cfg.log_path)
                old_report = read(cfg.report_path)
                old_raw = Set(readdir(raw_dir))
                @test_throws ArgumentError C.capture_and_score_external_dst_snapshot!(
                    cfg; fetched_utc=C.DateTime(2026, 7, 1, 6), http_get=fake_get,
                )
                @test read(cfg.log_path) == old_log
                @test read(cfg.report_path) == old_report
                @test Set(readdir(raw_dir)) == old_raw
                @test read(marker, String) == "corrupt marker"
                @test !ispath(C._external_raw_store_lock_path(raw_dir))
                @test !ispath(cfg.log_path * ".lock")
            end
        end
    end

    @testset "unmarked nonempty raw stores require log provenance" begin
        source = (; name="mock", url="forecast", kind="swpc_geospace_json")
        forecast_body = _two_point_swpc_fixture("2026-07-01T08:00:00", -30.0)
        obs_body = """[{"time_tag":"2026-07-01T08:00:00","dst":-31.0}]"""
        fake_get = (url; kwargs...) -> C._mock_response(
            String(url) == "forecast" ? forecast_body : obs_body,
        )

        for log_state in (:missing, :empty)
            @testset "$log_state log" begin
                mktempdir() do dir
                    raw_dir = joinpath(dir, "raw")
                    mkpath(raw_dir)
                    foreign_raw = joinpath(raw_dir, "foreign.raw")
                    write(foreign_raw, "pre-existing raw payload")
                    cfg = C.ExternalDstCollectorConfig(;
                        log_path=joinpath(dir, "external.csv"),
                        report_path=joinpath(dir, "external.md"), raw_dir,
                        repo_root=dir, sources=[source], obs_url="obs",
                        max_log_rows=10, max_raw_snapshots=10,
                    )
                    if log_state == :empty
                        C.CSV.write(cfg.log_path, C._external_empty_log())
                    end
                    old_log = isfile(cfg.log_path) ? read(cfg.log_path) : nothing

                    failure = try
                        C.capture_and_score_external_dst_snapshot!(
                            cfg; fetched_utc=C.DateTime(2026, 7, 1, 5),
                            http_get=fake_get,
                        )
                        nothing
                    catch err
                        err
                    end
                    @test failure isa ArgumentError
                    @test occursin("unmarked raw store is nonempty",
                                   sprint(showerror, failure))
                    @test read(foreign_raw, String) == "pre-existing raw payload"
                    @test Set(readdir(raw_dir)) == Set(["foreign.raw"])
                    @test !ispath(cfg.report_path)
                    @test !ispath(C._external_raw_store_owner_path(raw_dir))
                    @test !ispath(C._external_log_raw_store_owner_path(cfg.log_path))
                    @test !ispath(C._external_raw_store_lock_path(raw_dir))
                    @test !ispath(cfg.log_path * ".lock")
                    if log_state == :missing
                        @test !ispath(cfg.log_path)
                    else
                        @test read(cfg.log_path) == old_log
                    end
                end
            end
        end

        mktempdir() do dir
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            result = C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            @test result.rows_total == 1
            @test isfile(C._external_raw_store_owner_path(cfg.raw_dir))
            @test isfile(C._external_log_raw_store_owner_path(cfg.log_path))
            @test length(filter(name -> endswith(name, ".raw"),
                                readdir(cfg.raw_dir))) == 1
        end
    end

    @testset "row/raw retention keeps every retained raw reference" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            generation = Ref(1)
            forecast_bodies = (
                _two_point_swpc_fixture("2026-07-01T06:00:00", -30.0),
                _two_point_swpc_fixture("2026-07-01T07:00:00", -40.0),
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

    @testset "duplicate snapshots retain their logged raw" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            forecast_body = _two_point_swpc_fixture("2026-07-01T06:00:00", -30.0)
            obs_body = """[{"time_tag":"2026-07-01T06:00:00","dst":-31.0}]"""
            fake_get = function (url; kwargs...)
                String(url) == "forecast" && return C._mock_response(
                    forecast_body;
                    last_modified="Wed, 01 Jul 2026 05:00:00 GMT",
                )
                return C._mock_response(obs_body)
            end
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=1,
            )
            first_result = C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5, 1), http_get=fake_get,
            )
            first_log = C.CSV.read(cfg.log_path, C.DataFrame)
            first_raw = String(only(first_log.raw_path))
            second_result = C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5, 2), http_get=fake_get,
            )
            second_log = C.CSV.read(cfg.log_path, C.DataFrame)
            @test first_result.rows_total == second_result.rows_total == 1
            @test second_result.rows_added == 0
            @test second_result.rows_dropped == 0
            @test second_result.raw_pruned == 0
            @test C.nrow(second_log) == 1
            @test String(only(second_log.raw_path)) == first_raw
            @test length(filter(name -> endswith(name, ".raw"), readdir(cfg.raw_dir))) == 1
            @test isfile(joinpath(cfg.repo_root, first_raw))
        end
    end

    @testset "partial upstream failures do not install raw snapshots" begin
        mktempdir() do dir
            good = (; name="good", url="good", kind="swpc_geospace_json")
            bad = (; name="bad", url="bad", kind="swpc_geospace_json")
            forecast_body = _two_point_swpc_fixture("2026-07-02T00:00:00", -30.0)
            obs_body = """[{"time_tag":"2026-07-02T00:00:00","dst":-31.0}]"""
            good_cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[good], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=2,
            )
            successful_get = (url; kwargs...) -> C._mock_response(
                String(url) == "good" ? forecast_body : obs_body,
            )
            C.capture_and_score_external_dst_snapshot!(
                good_cfg; fetched_utc=C.DateTime(2026, 7, 1, 5),
                http_get=successful_get,
            )
            old_log = read(good_cfg.log_path)
            old_report = read(good_cfg.report_path)
            old_raw = Set(readdir(good_cfg.raw_dir))

            failing_cfg = C.ExternalDstCollectorConfig(;
                log_path=good_cfg.log_path, report_path=good_cfg.report_path,
                raw_dir=good_cfg.raw_dir, repo_root=dir,
                sources=[good, bad], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=2,
            )
            failing_get = function (url; kwargs...)
                String(url) == "bad" && error("injected second-source failure")
                return C._mock_response(String(url) == "good" ? forecast_body : obs_body)
            end
            for hour in 6:10
                @test_throws ErrorException C.capture_and_score_external_dst_snapshot!(
                    failing_cfg; fetched_utc=C.DateTime(2026, 7, 1, hour),
                    http_get=failing_get,
                )
                @test Set(readdir(good_cfg.raw_dir)) == old_raw
                @test read(good_cfg.log_path) == old_log
                @test read(good_cfg.report_path) == old_report
                @test !ispath(good_cfg.log_path * ".lock")
            end

            failing_obs_get = function (url; kwargs...)
                String(url) == "obs" && error("injected observation-feed failure")
                return C._mock_response(forecast_body)
            end
            for hour in 11:15
                @test_throws ErrorException C.capture_and_score_external_dst_snapshot!(
                    good_cfg; fetched_utc=C.DateTime(2026, 7, 1, hour),
                    http_get=failing_obs_get,
                )
                @test Set(readdir(good_cfg.raw_dir)) == old_raw
                @test read(good_cfg.log_path) == old_log
                @test read(good_cfg.report_path) == old_report
                @test !ispath(good_cfg.log_path * ".lock")
            end
        end
    end


    @testset "retained raw corruption fails closed" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            generation = Ref(1)
            forecast_bodies = (
                _two_point_swpc_fixture("2026-07-01T06:00:00", -30.0),
                _two_point_swpc_fixture("2026-07-01T07:00:00", -40.0),
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
            old_raw = Set(readdir(cfg.raw_dir))

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
            @test Set(readdir(cfg.raw_dir)) == old_raw
            @test !ispath(cfg.log_path * ".lock")
        end
    end

    @testset "missing retained raw fails closed before installing a replacement" begin
        mktempdir() do dir
            source = (; name="mock", url="forecast", kind="swpc_geospace_json")
            generation = Ref(1)
            forecast_bodies = (
                _two_point_swpc_fixture("2026-07-01T06:00:00", -30.0),
                _two_point_swpc_fixture("2026-07-01T07:00:00", -40.0),
            )
            obs_body = """[
              {"time_tag":"2026-07-01T06:00:00","dst":-31.0},
              {"time_tag":"2026-07-01T07:00:00","dst":-41.0}
            ]"""
            fake_get = (url; kwargs...) -> C._mock_response(
                String(url) == "forecast" ? forecast_bodies[generation[]] : obs_body,
            )
            cfg = C.ExternalDstCollectorConfig(;
                log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"),
                raw_dir=joinpath(dir, "raw"), repo_root=dir,
                sources=[source], obs_url="obs", max_log_rows=10,
                max_raw_snapshots=10,
            )
            C.capture_and_score_external_dst_snapshot!(
                cfg; fetched_utc=C.DateTime(2026, 7, 1, 5), http_get=fake_get,
            )
            first_log = C.CSV.read(cfg.log_path, C.DataFrame)
            rm(joinpath(cfg.repo_root, String(only(first_log.raw_path))))
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
            @test occursin("retained raw snapshot is missing",
                           sprint(showerror, failure))
            @test read(cfg.log_path) == old_log
            @test read(cfg.report_path) == old_report
            @test isempty(filter(name -> endswith(name, ".raw"), readdir(cfg.raw_dir)))
            @test !ispath(cfg.log_path * ".lock")
            @test !ispath(C._external_raw_store_lock_path(cfg.raw_dir))
        end
    end

    bad_cfg = C.ExternalDstCollectorConfig(max_log_rows=0)
    @test_throws ArgumentError C.capture_and_score_external_dst_snapshot!(
        bad_cfg; http_get=(args...; kwargs...) -> error("must not fetch"),
    )
end

@testset "Live monitor forecast-log retention" begin
    L = LiveMonitorRetentionTestHarness
    @test L.HORIZONS == (1, 2, 3, 6)
    @test L._advance_cycle_deadline(100.0, 120.0, 60.0) ==
          (deadline=160.0, skipped=0)
    @test L._advance_cycle_deadline(100.0, 160.0, 60.0) ==
          (deadline=160.0, skipped=0)
    @test L._advance_cycle_deadline(100.0, 161.0, 60.0) ==
          (deadline=160.0, skipped=0)
    @test L._advance_cycle_deadline(100.0, 281.0, 60.0) ==
          (deadline=280.0, skipped=2)
    @test L._advance_cycle_deadline(280.0, 282.0, 60.0) ==
          (deadline=340.0, skipped=0)
    @test_throws ArgumentError L._advance_cycle_deadline(0.0, 1.0, 0.0)
    @test_throws ArgumentError L._advance_cycle_deadline(0.0, Inf, 1.0)
    issue = L.DateTime(2026, 7, 15, 12, 10)
    targets = L.floor(issue, L.Hour) .+ L.Hour.(collect(L.HORIZONS))
    cycle_rows = L.DataFrame(
        issue_time_utc=fill(string(issue), length(L.HORIZONS)),
        latest_solar_wind_utc=fill(string(issue - L.Minute(1)), length(L.HORIZONS)),
        latest_dst_time_utc=fill(string(L.floor(issue, L.Hour) - L.Hour(1)), length(L.HORIZONS)),
        target_time_utc=string.(targets),
        horizon_hours=[(target - issue) / L.Millisecond(3_600_000) for target in targets],
        latest_dst_nt=fill(-20.0, length(L.HORIZONS)),
        observation_dst_nt=fill(missing, length(L.HORIZONS)),
        served_pred_dst_nt=fill(-25.0, length(L.HORIZONS)),
        served_pred_dst_ci05_nt=fill(-35.0, length(L.HORIZONS)),
        served_pred_dst_ci95_nt=fill(-15.0, length(L.HORIZONS)),
        v2_pred_dst_nt=fill(-25.0, length(L.HORIZONS)),
        v2_pred_dst_ci05_nt=fill(-35.0, length(L.HORIZONS)),
        v2_pred_dst_ci95_nt=fill(-15.0, length(L.HORIZONS)),
        interval_source=fill("aci", length(L.HORIZONS)),
        model_version=fill(L.CURRENT_V2_MODEL_VERSION, length(L.HORIZONS)),
        sub_hourly_model_version=fill(
            L.CURRENT_V2_SERVED_MODEL_VERSION,
            length(L.HORIZONS),
        ),
    )
    mktempdir() do dir
        log_path = joinpath(dir, "cycle.csv")
        L.CSV.write(log_path, cycle_rows)
        @test L._complete_issuance_cycle(log_path, issue)
        @test L._complete_issuance_cycle(log_path, issue, :aci)
        @test !L._complete_issuance_cycle(log_path, issue, :static)
        @test !L._complete_issuance_cycle(log_path, issue + L.Hour(1))
        static_rows = copy(cycle_rows)
        static_rows.interval_source .= "conformal"
        L.CSV.write(log_path, static_rows)
        @test L._complete_issuance_cycle(log_path, issue, :static)
        @test !L._complete_issuance_cycle(log_path, issue, :aci)
        L.CSV.write(log_path, cycle_rows[1:3, :])
        @test !L._complete_issuance_cycle(log_path, issue)
    end

    # The issued record is a NamedTuple; the monitor counts the guarded call result and then
    # validates the resulting log cycle independently of the payload type.
    called = Int[]
    inputs = (issue_time=issue,)
    policies = Symbol[]
    fake_issue = function (cfg; interval_policy, kwargs...)
        push!(called, cfg.horizon_hours)
        push!(policies, interval_policy)
        return (row_idx=length(called),)
    end
    issued = L._issue_horizon_cycle!(
        inputs; issue_fn=fake_issue,
        log_path="unused.csv", calibration_path=L.V2_CALIB,
        complete_fn=(path, timestamp, policy) -> true,
        interval_policy=:static,
    )
    @test issued == (succeeded=4, complete=true)
    @test called == collect(L.HORIZONS)
    @test policies == fill(:static, length(L.HORIZONS))
    @test_throws ArgumentError L._issue_horizon_cycle!(
        inputs; issue_fn=fake_issue,
        log_path="unused.csv", calibration_path=L.V2_CALIB,
        complete_fn=(path, timestamp, policy) -> true,
        interval_policy=:auto,
    )

    policy_inputs = (
        issue_time=issue,
        dst=([issue - L.Hour(1), L.floor(issue, L.Hour)], [-22.0, -20.0]),
    )
    readiness_calls = Tuple{Int,Symbol}[]
    all_ready = function (path, model_steps, latest_dst, pred_col)
        push!(readiness_calls, (model_steps, pred_col))
        return true
    end
    @test L._monitor_interval_policy(
        policy_inputs; log_path="unused.csv", readiness_fn=all_ready,
    ) == :aci
    @test Set(readiness_calls) == Set(
        (steps, pred_col) for steps in L.HORIZONS
        for pred_col in (:v2_pred_dst_nt, :served_pred_dst_nt)
    )
    partial_served = (path, model_steps, latest_dst, pred_col) ->
        !(model_steps == maximum(L.HORIZONS) && pred_col == :served_pred_dst_nt)
    @test L._monitor_interval_policy(
        policy_inputs; log_path="unused.csv", readiness_fn=partial_served,
    ) == :static

    mktempdir() do dir
        sentinel = joinpath(dir, "OUTAGE.md")
        L.write_outage_sentinel("2026-07-15T12:00:00Z", 3; path=sentinel)
        body = read(sentinel, String)
        @test occursin("/api/health", body)
        @test occursin("/api/swpc", body)
        @test occursin("Kyoto Dst", body)
        @test !occursin("stdout log", body)
    end
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
        @test L._retain_live_forecast_log!(path, 4) == 2
        retained = L.CSV.read(path, L.DataFrame)
        @test retained.marker == [3, 4, 5, 6]
        state = L._read_live_state(path)
        @test state !== nothing
        @test state["row_count"] == 4
        @test L._state_matches_log(state, path)
        @test isempty(state["aci_streams"])
        @test L._retain_live_forecast_log!(path, 4) == 0
        @test_throws ArgumentError L._retain_live_forecast_log!(path, 0)
        @test_throws ArgumentError L._retain_live_forecast_log!(path, 3)
    end
end
