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
const TEST_MONITOR_DIR = mktempdir()
withenv("SOLARSINDY_MONITOR_DIR" => TEST_MONITOR_DIR) do
    include(joinpath(@__DIR__, "..", "examples", "live_monitor.jl"))
end
end

# Synthetic V2.4e deployment bundle, so the monitor-side cycle test drives the real served stage
# instead of the fallback chain. That combination — a mature adaptive-conformal batch policy over a
# V2.4e-served cycle — is the one the `1ee9ab0` incident broke and no test covered.
include(joinpath(@__DIR__, "v2_4_serving_fixture.jl"))
using .V24ServingFixture: build_v24_fixture_bundle

# Verified-residual history deep enough for the adaptive-conformal streams to be servable at each
# listed model step. Both the baseline-center and served-center streams read the same rows, so one
# frame matures both; the selector needs `warmup + margin` rows in a pool before a stream may be
# used, and the counts are per model step, which is why the required step set (and therefore the
# cadence phase) decides whether the batch can use ACI at all.
function _mature_aci_log_frame(anchor; model_steps, per_step::Int)
    L = LiveMonitorRetentionTestHarness
    total = per_step * length(model_steps)
    issues = L.DateTime[]
    targets = L.DateTime[]
    steps = Int[]
    latest = Float64[]
    v2_pred = Float64[]
    served_pred = Float64[]
    observed = Float64[]
    for slot in 1:total
        step = model_steps[mod1(slot, length(model_steps))]
        issue = anchor - L.Hour(total - slot + 2)
        push!(issues, issue)
        push!(targets, issue + L.Hour(step))
        push!(steps, step)
        push!(latest, -35.0 - 3.0 * sinpi(2 * slot / 13))
        center = -42.0 - 4.0 * sinpi(2 * slot / 17)
        push!(v2_pred, center)
        push!(served_pred, center - 1.5)
        push!(observed, center + 2.0 + 3.0 * sinpi(2 * slot / 11) + 0.4 * step)
    end
    return L.DataFrame(
        issue_time_utc=string.(issues),
        latest_dst_time_utc=string.(issues),
        target_time_utc=string.(targets),
        model_version=fill(L.OPERATIONAL_V2_1_MODEL_VERSION, total),
        model_step_hours=steps,
        latest_dst_nt=latest,
        v2_pred_dst_nt=v2_pred,
        served_pred_dst_nt=served_pred,
        observation_dst_nt=observed,
    )
end

# Diagnostic lines the monitor appended to its bounded ring during `f`. The daemon's operational
# record is the artifact under test for the policy diagnostic, not an incidental print.
function _monitor_diag_during(f)
    L = LiveMonitorRetentionTestHarness
    before = isfile(L.DIAG_LOG) ? filesize(L.DIAG_LOG) : 0
    value = redirect_stdout(devnull) do
        f()
    end
    text = isfile(L.DIAG_LOG) ? open(L.DIAG_LOG) do io
        seek(io, before)
        read(io, String)
    end : ""
    return (value=value, diag=text)
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
    @test L.MONITOR_DIR == L.TEST_MONITOR_DIR
    @test L.DIAG_LOG == joinpath(L.TEST_MONITOR_DIR, "logs", "monitor.log")
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
        # A V2.4e-served cycle carries the study's conformal band whatever the batch's fallback
        # interval policy is; it is complete under both policies, but only when every row's V2.4
        # status is `ok`. Regression: every V2.4e cycle failed the aci-policy check, tripping the
        # issuance dead-man after six cycles and restarting the daemon.
        v24_rows = copy(cycle_rows)
        v24_rows.interval_source .= L.V2_4_INTERVAL_SOURCE
        v24_rows.v24_status = fill(L.V2_4_STATUS_OK, length(L.HORIZONS))
        L.CSV.write(log_path, v24_rows)
        @test L._complete_issuance_cycle(log_path, issue, :aci)
        @test L._complete_issuance_cycle(log_path, issue, :static)
        @test L._complete_issuance_cycle(log_path, issue)
        mixed_rows = copy(v24_rows)
        mixed_rows.v24_status[end] = "fallback:deployment_absent"
        L.CSV.write(log_path, mixed_rows)
        @test !L._complete_issuance_cycle(log_path, issue, :aci)
        @test !L._complete_issuance_cycle(log_path, issue, :static)
        no_status_rows = copy(cycle_rows)
        no_status_rows.interval_source .= L.V2_4_INTERVAL_SOURCE
        L.CSV.write(log_path, no_status_rows)
        @test !L._complete_issuance_cycle(log_path, issue, :aci)
        @test !L._complete_issuance_cycle(log_path, issue, :static)
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
        @test L.clear_outage_sentinel(sentinel)
        @test !isfile(sentinel)

        write(sentinel, "# OUTAGE\n\nSource: external watchdog\n")
        @test !L.clear_outage_sentinel(sentinel)
        @test isfile(sentinel)
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

@testset "Live monitor cold-archive schema-drift guard" begin
    L = LiveMonitorRetentionTestHarness
    base = L.DateTime(2026, 7, 1)
    make_log = (log_path) -> begin
        rows = L.DataFrame(
            issue_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            latest_dst_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            target_time_utc=string.([base + L.Hour(i + 1) for i in 0:5]),
            model_version=fill("v2", 6),
            observation_dst_nt=fill(missing, 6),
            marker=collect(1:6),
        )
        L.CSV.write(log_path, rows)
        L._load_or_rebuild_live_state!(log_path)
        return rows
    end

    # A pre-existing archive whose header does not match the hot-log schema (the v2.0->v2.1
    # column-addition drift class) must never receive misaligned rows. It is left byte-identical and
    # retention rolls to a new numbered segment instead of refusing: the old behaviour threw on every
    # cycle from the first schema upgrade onwards, so rows were never pruned again and the hot log
    # grew past LIVE_MONITOR_MAX_LOG_ROWS without bound.
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        make_log(log_path)
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        mkpath(dirname(archive_path))
        write(archive_path, "issue_time_utc,target_time_utc,marker\n")  # 3 columns vs the 6-col log
        archive_before = read(archive_path)
        @test L._retain_live_forecast_log!(log_path, 4) == 2
        # Fail-closed guarantee preserved: the mismatched archive is untouched.
        @test read(archive_path) == archive_before
        @test !ispath(string(archive_path, ".manifest.json"))
        # The pruned rows landed in a new segment carrying their own header.
        segment = joinpath(dir, "archive", "live_forecast_log_archive.1.csv")
        @test isfile(segment)
        segment_rows = L.CSV.read(segment, L.DataFrame)
        @test names(segment_rows) ==
              ["issue_time_utc", "latest_dst_time_utc", "target_time_utc",
               "model_version", "observation_dst_nt", "marker"]
        @test segment_rows.marker == [1, 2]
        manifest = L.JSON3.read(read(string(segment, ".manifest.json"), String))
        @test Int(manifest.archived_rows) == 2
        @test Int(manifest.segment_index) == 1
        @test Int(manifest.archive_bytes) == filesize(segment)
        @test L.CSV.read(log_path, L.DataFrame).marker == [3, 4, 5, 6]
    end

    # Full schema-upgrade lifecycle: an archive created under schema A, a hot log upgraded to
    # A + column, and two further retentions. Both prune, the schema-A segment stays byte-identical
    # after its last matching append, the schema-B rows accumulate in their own segment with a
    # consistent manifest, and the hot log ends at the cap.
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        segment_path = joinpath(dir, "archive", "live_forecast_log_archive.1.csv")

        make_log(log_path)
        @test L._retain_live_forecast_log!(log_path, 4) == 2
        @test isfile(archive_path)
        @test L.CSV.read(archive_path, L.DataFrame).marker == [1, 2]
        base_archive_bytes = read(archive_path)

        upgraded = (markers) -> begin
            rows = L.DataFrame(
                issue_time_utc=string.([base + L.Hour(i) for i in markers]),
                latest_dst_time_utc=string.([base + L.Hour(i) for i in markers]),
                target_time_utc=string.([base + L.Hour(i + 1) for i in markers]),
                model_version=fill("v2", length(markers)),
                observation_dst_nt=fill(missing, length(markers)),
                marker=collect(markers),
                v24_status=fill("ok", length(markers)),   # the routine column-addition drift
            )
            L.CSV.write(log_path, rows)
            rm(L._live_state_path(log_path); force=true)
            L._load_or_rebuild_live_state!(log_path)
            return rows
        end

        upgraded(10:17)
        @test L._retain_live_forecast_log!(log_path, 4) == 4
        @test read(archive_path) == base_archive_bytes
        @test isfile(segment_path)
        @test L.CSV.read(segment_path, L.DataFrame).marker == collect(10:13)

        upgraded(20:27)
        @test L._retain_live_forecast_log!(log_path, 4) == 4
        @test read(archive_path) == base_archive_bytes
        segment_rows = L.CSV.read(segment_path, L.DataFrame)
        @test segment_rows.marker == vcat(collect(10:13), collect(20:23))
        @test "v24_status" in names(segment_rows)
        manifest = L.JSON3.read(read(string(segment_path, ".manifest.json"), String))
        @test Int(manifest.archived_rows) == 8
        @test Int(manifest.archive_bytes) == filesize(segment_path)
        @test Int(manifest.last_segment_rows) == 4
        @test Int(manifest.segment_index) == 1
        @test !ispath(joinpath(dir, "archive", "live_forecast_log_archive.2.csv"))
        @test L.nrow(L.CSV.read(log_path, L.DataFrame)) == 4
    end

    # Retention derives the archive path from the log's own directory and, when no archive exists yet,
    # creates one with the matching header. The guard is not a blanket refusal: a fresh/matching
    # archive still accepts the prune, and the dropped rows land in the sibling archive.
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        make_log(log_path)
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        @test !ispath(archive_path)
        @test L._retain_live_forecast_log!(log_path, 4) == 2
        @test isfile(archive_path)
        archived = L.CSV.read(archive_path, L.DataFrame)
        @test names(archived) ==
              ["issue_time_utc", "latest_dst_time_utc", "target_time_utc",
               "model_version", "observation_dst_nt", "marker"]
        @test archived.marker == [1, 2]
        @test L.CSV.read(log_path, L.DataFrame).marker == [3, 4, 5, 6]
        # Isolation: the derived archive stays beside its own log, never the module-const production
        # archive.
        @test dirname(dirname(archive_path)) == dir
        @test archive_path != L.FORECAST_ARCHIVE
    end
end

@testset "Live monitor cold-archive appends only to the highest-index segment" begin
    L = LiveMonitorRetentionTestHarness
    seg = (dir, i) -> joinpath(dir, "archive", "live_forecast_log_archive.$(i).csv")
    schema_a = markers -> L.DataFrame(a=collect(markers), b=string.(collect(markers)))
    schema_b = markers -> L.DataFrame(a=collect(markers), b=string.(collect(markers)),
                                      c=collect(markers))

    # The archive's stated contract is that segment order is append order, so reading the segments
    # in index order reproduces the write order. Resolving to the FIRST header-matching segment
    # broke exactly that whenever the hot-log schema returned to an earlier shape — a reverted
    # column, an operator running the previous release — because rows written third would be
    # appended to segment 0, ahead of rows written second in segment 1. Only the highest-index
    # segment is a candidate, so a schema revert opens a new segment instead of writing backwards.
    mktempdir() do dir
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        manifest_path = string(archive_path, ".manifest.json")
        archive = frame -> L._archive_pruned_rows!(
            frame; archive_path=archive_path, manifest_path=manifest_path,
        )

        @test archive(schema_a(1:2)) == 2
        @test archive(schema_b(3:4)) == 2
        @test isfile(seg(dir, 1))
        @test archive(schema_a(5:6)) == 2

        @test isfile(seg(dir, 2))
        @test L.CSV.read(archive_path, L.DataFrame).a == [1, 2]
        @test L.CSV.read(seg(dir, 1), L.DataFrame).a == [3, 4]
        @test L.CSV.read(seg(dir, 2), L.DataFrame).a == [5, 6]
        # The invariant itself: concatenating the segments in index order is the append order.
        @test reduce(vcat, [L.CSV.read(p, L.DataFrame).a
                            for p in (archive_path, seg(dir, 1), seg(dir, 2))]) == collect(1:6)

        # A further schema-A frame extends the LAST segment, never segment 0.
        @test archive(schema_a(7:8)) == 2
        @test L.CSV.read(seg(dir, 2), L.DataFrame).a == [5, 6, 7, 8]
        @test !ispath(seg(dir, 3))
        @test L.CSV.read(archive_path, L.DataFrame).a == [1, 2]
        manifest = L.JSON3.read(read(string(seg(dir, 2), ".manifest.json"), String))
        @test Int(manifest.segment_index) == 2
        @test Int(manifest.archived_rows) == 4
        @test Int(manifest.archive_bytes) == filesize(seg(dir, 2))
    end

    # A manually removed middle segment must not make a later, populated segment invisible: the
    # resolver scans the whole index range, so appends still go forward instead of into the hole.
    mktempdir() do dir
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        manifest_path = string(archive_path, ".manifest.json")
        mkpath(dirname(archive_path))
        L.CSV.write(archive_path, schema_a(1:2))
        L.CSV.write(seg(dir, 2), schema_a(3:4))
        write(string(seg(dir, 2), ".manifest.json"),
              "{\"archived_rows\":2,\"archive_bytes\":$(filesize(seg(dir, 2)))}")

        @test !ispath(seg(dir, 1))
        @test L._archive_pruned_rows!(
            schema_a(5:6); archive_path=archive_path, manifest_path=manifest_path) == 2
        @test !ispath(seg(dir, 1))
        @test L.CSV.read(seg(dir, 2), L.DataFrame).a == [3, 4, 5, 6]
        @test L.CSV.read(archive_path, L.DataFrame).a == [1, 2]
    end
end

@testset "Live monitor cold-archive manifest-completeness guard" begin
    L = LiveMonitorRetentionTestHarness
    base = L.DateTime(2026, 7, 1)
    make_log = (log_path) -> begin
        rows = L.DataFrame(
            issue_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            latest_dst_time_utc=string.([base + L.Hour(i) for i in 0:5]),
            target_time_utc=string.([base + L.Hour(i + 1) for i in 0:5]),
            model_version=fill("v2", 6),
            observation_dst_nt=fill(missing, 6),
            marker=collect(1:6),
        )
        L.CSV.write(log_path, rows)
        L._load_or_rebuild_live_state!(log_path)
        return rows
    end
    # Exact header of the hot-log schema make_log writes, so the schema-drift guard passes and the
    # manifest-completeness guard is the one under test.
    matched_header =
        "issue_time_utc,latest_dst_time_utc,target_time_utc,model_version,observation_dst_nt,marker\n"

    # A non-empty archive whose header MATCHES the hot log but has NO manifest (no verified byte
    # baseline) must abort retention before any append. Before this guard the byte-completeness check
    # fired only AFTER CSV.write had appended the segment, so every retry re-appended the same rows and
    # grew the archive (12->20->28) while retention never completed. The hot log stayed intact, but the
    # archive accumulated duplicates. Now retention refuses before touching the file: archive
    # byte-identical, hot log unchanged, manifest still absent, and retries never grow the archive.
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        make_log(log_path)
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        manifest_path = string(archive_path, ".manifest.json")
        mkpath(dirname(archive_path))
        write(archive_path, matched_header)
        archive_before = read(archive_path)
        @test !ispath(manifest_path)
        @test_throws ErrorException L._retain_live_forecast_log!(log_path, 4)
        @test read(archive_path) == archive_before
        @test !ispath(manifest_path)
        @test L.CSV.read(log_path, L.DataFrame).marker == collect(1:6)
        # Absorbing-duplicate check: a second retry still refuses and never grows the archive.
        @test_throws ErrorException L._retain_live_forecast_log!(log_path, 4)
        @test read(archive_path) == archive_before
        @test !ispath(manifest_path)
    end

    # Corrupt-JSON manifest beside the matched-header non-empty archive is swallowed to prev_bytes==0
    # and must fail closed identically: refuse before append, archive and manifest byte-identical, hot
    # log intact.
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        make_log(log_path)
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        manifest_path = string(archive_path, ".manifest.json")
        mkpath(dirname(archive_path))
        write(archive_path, matched_header)
        write(manifest_path, "{not valid json")
        archive_before = read(archive_path)
        manifest_before = read(manifest_path)
        @test_throws ErrorException L._retain_live_forecast_log!(log_path, 4)
        @test read(archive_path) == archive_before
        @test read(manifest_path) == manifest_before
        @test L.CSV.read(log_path, L.DataFrame).marker == collect(1:6)
    end

    # Healthy-path regression: when the manifest matches the archive, the append path still works.
    # First call creates archive+manifest (existed=false, guard not applicable); second call appends
    # under the matching manifest (existed=true, prev_bytes>0), and no rows are duplicated.
    mktempdir() do dir
        archive_path = joinpath(dir, "archive", "live_forecast_log_archive.csv")
        manifest_path = string(archive_path, ".manifest.json")
        frame1 = L.DataFrame(a=[1, 2], b=["p", "q"])
        frame2 = L.DataFrame(a=[3, 4], b=["r", "s"])
        @test L._archive_pruned_rows!(
            frame1; archive_path=archive_path, manifest_path=manifest_path) == 2
        @test isfile(archive_path)
        @test isfile(manifest_path)
        @test L._archive_pruned_rows!(
            frame2; archive_path=archive_path, manifest_path=manifest_path) == 2
        archived = L.CSV.read(archive_path, L.DataFrame)
        @test archived.a == [1, 2, 3, 4]
        @test archived.b == ["p", "q", "r", "s"]
    end
end

@testset "Live monitor cycle never skips the observation-side steps" begin
    L = LiveMonitorRetentionTestHarness
    # A solar-wind feed outage must not also stall Kyoto verification, hot-log retention, the
    # prospective external Dst snapshot (an hourly record that cannot be backfilled) or the
    # comparison report. Before the split every one of those was skipped by an early return, which
    # is what happened in production on 2026-07-29 (cycles 15/16, ECONNRESET).
    make_log = (log_path) -> begin
        L.CSV.write(log_path, L.DataFrame(
            issue_time_utc=["2026-07-01T00:00:00", "2026-07-01T01:00:00"],
            latest_dst_time_utc=["2026-07-01T00:00:00", "2026-07-01T01:00:00"],
            target_time_utc=["2026-07-01T01:00:00", "2026-07-01T02:00:00"],
            model_version=fill(L.OPERATIONAL_V2_1_MODEL_VERSION, 2),
            observation_dst_nt=[-20.0, missing],
        ))
        return log_path
    end

    run_cycle = (log_path, prepare_fn, policy_fn) -> begin
        called = String[]
        seen_refresh = Ref{Any}(:unset)
        seen_snapshot = Ref{Any}(:unset)
        record = name -> (args...; kwargs...) -> (push!(called, name); nothing)
        issuance = redirect_stdout(devnull) do
            L.cycle!(;
                prepare_fn=prepare_fn,
                policy_fn=policy_fn,
                issue_cycle_fn=(args...; kwargs...) -> begin
                    push!(called, "issue")
                    (succeeded=length(L.HORIZONS), complete=true)
                end,
                dst_fn=() -> begin
                    push!(called, "dst")
                    ([L.DateTime(2026, 7, 1, 2)], [-21.0])
                end,
                refresh_fn=(cfg; dst_times=nothing, dst_vals=nothing) -> begin
                    push!(called, "refresh")
                    seen_refresh[] = (dst_times, dst_vals)
                    nothing
                end,
                verify_fn=record("verify"),
                retention_fn=record("retention"),
                snapshot_fn=(cfg; observations=nothing) -> begin
                    push!(called, "snapshot")
                    seen_snapshot[] = observations
                    nothing
                end,
                report_fn=record("report"),
                log_path=log_path,
                report_path=string(log_path, ".md"),
                calibration_path=L.V2_CALIB,
                max_log_rows=length(L.HORIZONS),
            )
        end
        return (issuance=issuance, called=called,
                refresh=seen_refresh[], snapshot=seen_snapshot[])
    end

    mktempdir() do dir
        # Issuance inputs unavailable (feed outage).
        outcome = run_cycle(make_log(joinpath(dir, "outage.csv")),
                            _cfg -> error("solar wind feed unavailable"),
                            _inputs -> error("interval policy must not be reached"))
        @test outcome.issuance == (succeeded=0, complete=false)
        @test "issue" ∉ outcome.called
        # The observation feed is fetched independently and every later step still runs, in order.
        @test outcome.called == ["dst", "refresh", "retention", "snapshot", "report"]
        # The independently fetched feed actually reaches the steps that consume it, rather than
        # each of them silently refetching or being handed nothing.
        @test outcome.refresh == ([L.DateTime(2026, 7, 1, 2)], [-21.0])
        @test outcome.snapshot isa L.DataFrame
        @test outcome.snapshot.observed_time_utc == [L.DateTime(2026, 7, 1, 2)]
        @test outcome.snapshot.observed_dst_nt == [-21.0]

        # Interval-policy preflight failure: same contract.
        outcome = run_cycle(make_log(joinpath(dir, "policy.csv")),
                            _cfg -> (issue_time=L.DateTime(2026, 7, 1, 2),
                                     dst=([L.DateTime(2026, 7, 1, 2)], [-21.0])),
                            _inputs -> error("residual stream unreadable"))
        @test outcome.issuance == (succeeded=0, complete=false)
        @test "issue" ∉ outcome.called
        # Issuance inputs existed, so their Dst is reused rather than refetched.
        @test outcome.called == ["refresh", "retention", "snapshot", "report"]
        @test outcome.refresh == ([L.DateTime(2026, 7, 1, 2)], [-21.0])
        @test outcome.snapshot.observed_dst_nt == [-21.0]

        # Healthy control: issuance runs and the same observation-side steps follow it.
        outcome = run_cycle(make_log(joinpath(dir, "healthy.csv")),
                            _cfg -> (issue_time=L.DateTime(2026, 7, 1, 2),
                                     dst=([L.DateTime(2026, 7, 1, 2)], [-21.0])),
                            _inputs -> :static)
        @test outcome.issuance == (succeeded=length(L.HORIZONS), complete=true)
        @test outcome.called == ["issue", "refresh", "retention", "snapshot", "report"]

        # The narrower verifier is still the fallback when the refresh itself fails.
        called = String[]
        record = name -> (args...; kwargs...) -> (push!(called, name); nothing)
        redirect_stdout(devnull) do
            L.cycle!(;
                prepare_fn=_cfg -> error("solar wind feed unavailable"),
                policy_fn=_inputs -> :static,
                issue_cycle_fn=(args...; kwargs...) -> (succeeded=0, complete=false),
                dst_fn=() -> ([L.DateTime(2026, 7, 1, 2)], [-21.0]),
                refresh_fn=(args...; kwargs...) -> error("refresh failed"),
                verify_fn=record("verify"),
                retention_fn=record("retention"),
                snapshot_fn=record("snapshot"),
                report_fn=record("report"),
                log_path=make_log(joinpath(dir, "fallback.csv")),
                report_path=joinpath(dir, "fallback.md"),
                calibration_path=L.V2_CALIB,
                max_log_rows=length(L.HORIZONS),
            )
        end
        @test called == ["verify", "retention", "snapshot", "report"]
    end
end

@testset "Live monitor interval policy names the immature streams behind :static" begin
    L = LiveMonitorRetentionTestHarness
    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        anchor = L.DateTime(2026, 7, 15, 12)
        # Residual history only at model steps {2,3,4,7}: the anchor-lag-1 step set. At lag 0 the
        # cycle needs {1,2,3,6}, so exactly the ms=1 and ms=6 streams are immature — the cadence
        # phase alone decides the batch policy, which is the dependence the diagnostic must expose.
        L.CSV.write(log_path, _mature_aci_log_frame(anchor; model_steps=(2, 3, 4, 7), per_step=40))
        L._load_or_rebuild_live_state!(log_path)

        issue = anchor + L.Minute(5)                        # anchor lag 0 -> steps {1,2,3,6}
        inputs = (issue_time=issue, dst=([anchor - L.Hour(1), anchor], [-45.0, -44.0]))
        immature = Tuple{Int,Symbol}[]
        outcome = _monitor_diag_during() do
            L._monitor_interval_policy(inputs; log_path=log_path, immature=immature)
        end
        @test outcome.value == :static
        @test Set(immature) == Set([(1, :v2_pred_dst_nt), (1, :served_pred_dst_nt),
                                    (6, :v2_pred_dst_nt), (6, :served_pred_dst_nt)])
        @test occursin("interval policy :static", outcome.diag)
        @test occursin("lags issue hour", outcome.diag)
        @test occursin("model steps 1/2/3/6", outcome.diag)
        @test occursin("v2_pred_dst_nt@ms=1 n=0(all)", outcome.diag)
        @test occursin("served_pred_dst_nt@ms=6 n=0(all)", outcome.diag)
        @test occursin(string(L._ACI_WARMUP + L._ACI_POOL_MARGIN, " verified rows"), outcome.diag)
        @test !occursin("@ms=2", outcome.diag)
        @test !occursin("@ms=3", outcome.diag)

        # Same log, anchor lag 1: the required step set becomes {2,3,4,7}, every stream is mature,
        # and the identical history now yields the adaptive policy. Nothing but the phase changed.
        lagged_issue = anchor + L.Hour(1) + L.Minute(5)
        lagged_inputs = (issue_time=lagged_issue,
                         dst=([anchor - L.Hour(1), anchor], [-45.0, -44.0]))
        lagged_immature = Tuple{Int,Symbol}[]
        lagged = _monitor_diag_during() do
            L._monitor_interval_policy(lagged_inputs; log_path=log_path,
                                       immature=lagged_immature)
        end
        @test lagged.value == :aci
        @test isempty(lagged_immature)
        @test !occursin("interval policy :static", lagged.diag)
    end
end

@testset "Live monitor completes a V2.4e cycle under a mature ACI policy" begin
    L = LiveMonitorRetentionTestHarness
    mktempdir() do root
        bundle = build_v24_fixture_bundle(joinpath(root, "bundle"))
        log_path = joinpath(root, "live.csv")
        issue_time = L.DateTime(2026, 7, 15, 12, 30)
        anchor = L.floor(issue_time, L.Hour) - L.Hour(1)     # anchor lag 1 -> steps {2,3,4,7}
        L.CSV.write(log_path,
                    _mature_aci_log_frame(anchor; model_steps=(1, 2, 3, 4, 6, 7), per_step=40))
        L._load_or_rebuild_live_state!(log_path)

        feed_times = collect((issue_time - L.Hour(24)):L.Minute(1):issue_time)
        plasma = L.DataFrame(
            time_tag=feed_times,
            speed=[470.0 + 20.0 * sinpi(2 * k / 613) for k in eachindex(feed_times)],
            density=[6.0 + 0.8 * sinpi(2 * k / 421) for k in eachindex(feed_times)],
        )
        mag = L.DataFrame(
            time_tag=feed_times,
            bz_gsm=[-6.0 + 3.0 * sinpi(2 * k / 517) for k in eachindex(feed_times)],
            by_gsm=[1.5 * cospi(2 * k / 379) for k in eachindex(feed_times)],
        )
        dst_times = collect((anchor - L.Hour(24)):L.Hour(1):anchor)
        dst_values = collect(range(-30.0, -78.0; length=length(dst_times)))

        cfg = L.LiveVerifyConfig(; mode=:issue, model=:v2, horizon_hours=first(L.HORIZONS),
                                 log_path=log_path, v2_calibration_path=L.V2_CALIB)
        withenv("SOLARSINDY_V2_4_DEPLOY_DIR" => bundle.dir,
                "SOLARSINDY_V2_2_STACK" => L.V2_2_DEFAULT_STACK_PATH,
                "SOLARSINDY_V2_3_SHADOW_DIR" => joinpath(root, "no_shadow")) do
            L.reset_v2_2_stack!(); L.reset_v2_3_shadow!(); L.reset_v2_4_serving!()
            inputs = L.prepare_issue_inputs(
                cfg; issue_time=issue_time,
                plasma_fn=() -> plasma, mag_fn=() -> mag,
                dst_fn=() -> (dst_times, dst_values),
            )
            policy = redirect_stdout(devnull) do
                L._monitor_interval_policy(inputs; log_path=log_path)
            end
            @test policy == :aci
            issuance = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    L._issue_horizon_cycle!(
                        inputs; log_path=log_path, calibration_path=L.V2_CALIB,
                        interval_policy=policy,
                    )
                end
            end
            @test issuance.succeeded == length(L.HORIZONS)
            # The incident: a V2.4e cycle under an aci batch policy was reported incomplete, the
            # dead-man tripped after six cycles and the supervisor restarted the daemon.
            @test issuance.complete
            @test L._complete_issuance_cycle(log_path, issue_time, :aci)
            @test L._complete_issuance_cycle(log_path, issue_time, :static)
            L.reset_v2_2_stack!(); L.reset_v2_3_shadow!(); L.reset_v2_4_serving!()
        end

        df = L.CSV.read(log_path, L.DataFrame)
        cycle = df[(L.nrow(df) - length(L.HORIZONS) + 1):L.nrow(df), :]
        @test L.floor.(L._parse_dt.(cycle.issue_time_utc), L.Hour) ==
              fill(L.floor(issue_time, L.Hour), length(L.HORIZONS))
        @test all(==(L.V2_4_STATUS_OK), cycle.v24_status)
        @test all(==(L.V2_4_INTERVAL_SOURCE), cycle.interval_source)
        @test sort(Int.(cycle.model_step_hours)) == [2, 3, 4, 7]
        # T-21: the served band is the study's depth band on every row, while the frozen-tail band
        # columns came from the batch's adaptive policy. Only the new column records that.
        @test all(==("aci"), cycle.frozen_tail_interval_source)
        @test all(cycle.frozen_tail_interval_source .!= cycle.interval_source)
        @test all(isfinite, Float64.(cycle.pred_dst_ci05_nt))
        @test all(isfinite, Float64.(cycle.served_pred_dst_ci05_nt))

        # Flipping one row's V2.4 status must make the cycle incomplete under both policies:
        # a cycle carrying the conformal-depth source on a row the V2.4 stage did not serve is
        # incoherent, whatever the batch policy was.
        flipped_path = joinpath(root, "flipped.csv")
        flipped = copy(df)
        # Widen the column first: the CSV reader pools the served statuses into a fixed-width
        # inline string type that cannot hold a longer fallback reason.
        statuses = Vector{Union{Missing,String}}(undef, L.nrow(df))
        for idx in 1:L.nrow(df)
            value = df[idx, :v24_status]
            statuses[idx] = ismissing(value) ? missing : String(value)
        end
        statuses[end] = "fallback:deployment_absent"
        flipped.v24_status = statuses
        L.CSV.write(flipped_path, flipped)
        @test !L._complete_issuance_cycle(flipped_path, issue_time, :aci)
        @test !L._complete_issuance_cycle(flipped_path, issue_time, :static)
    end
end
