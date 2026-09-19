module ExternalDstReceiptTimingTests

using Test, Dates, DataFrames, CSV
include(joinpath(@__DIR__, "..", "examples", "external_dst_snapshot_collector.jl"))

const SOURCE = (name="receipt-fixture", kind="swpc_geospace_json", url="forecast")
const BODY = """[{"time_tag":"2026-09-07T10:00:00","dst":100.0},
                 {"time_tag":"2026-09-07T11:00:00","dst":3.0}]"""
fixture_get(url; kwargs...) = _mock_response(BODY;
    last_modified="Mon, 07 Sep 2026 09:00:00 GMT")
observations() = DataFrame(
    observed_time_utc=[DateTime(2026, 9, 7, 10), DateTime(2026, 9, 7, 11)],
    observed_dst_nt=[0.0, 0.0])

function fixture_rows(; fetched=DateTime(2026, 9, 7, 9, 59),
                        completed=DateTime(2026, 9, 7, 10, 30))
    return _staged_future_rows_for_source(SOURCE;
        fetched_utc=fetched, receipt_clock=() -> completed, http_get=fixture_get).rows
end

@testset "External Dst receipt-time boundary" begin
    rows = fixture_rows()
    @test rows.target_utc == ["2026-09-07T10:00:00Z", "2026-09-07T11:00:00Z"]
    @test rows.receipt_completed_utc == fill("2026-09-07T10:30:00Z", 2)
    @test external_dst_timing.(eachrow(rows)) == [:late, :eligible]
    @test score_external_dst_rows!(rows, observations();
        scored_utc=DateTime(2026, 9, 7, 12)) == 1
    @test ismissing(rows.observed_dst_nt[1])
    @test rows.abs_error_nt[2] == 3.0
    summary = only(eachrow(external_dst_summary(rows)))
    @test (summary.n_rows, summary.n_scored, summary.n_eligible, summary.n_late,
           summary.n_legacy, summary.n_invalid) == (2, 1, 1, 1, 0, 0)
    @test summary.rmse_nt == summary.mae_nt == 3.0
    @test summary.max_receipt_lead_h == 0.5
    @test _validate_external_dst_log(rows)

    @testset "all-missing score columns survive CSV round trip" begin
        mktempdir() do dir
            path = joinpath(dir, "unscored.csv")
            CSV.write(path, fixture_rows())
            unscored = CSV.read(path, DataFrame)
            @test eltype(unscored.observed_dst_nt) === Missing
            @test score_external_dst_rows!(unscored, observations();
                scored_utc=DateTime(2026, 9, 7, 12)) == 1
            @test _validate_external_dst_log(unscored)
            @test ismissing(unscored.observed_dst_nt[1])
            @test unscored.abs_error_nt[2] == 3.0
        end
    end

    @testset "strict boundary and millisecond preservation" begin
        boundary = DateTime(2026, 9, 7, 10)
        for (offset, expected) in ((-1, 2), (0, 1), (1, 1))
            completed = boundary + Millisecond(offset)
            frame = fixture_rows(; completed)
            @test all(_parse_external_time.(frame.receipt_completed_utc) .== completed)
            @test score_external_dst_rows!(frame, observations();
                scored_utc=DateTime(2026, 9, 7, 12)) == expected
            @test _validate_external_dst_log(frame)
        end
        fetched = DateTime(2026, 9, 7, 9, 59, 59, 123)
        frame = fixture_rows(; fetched, completed=fetched + Millisecond(1))
        @test frame.fetched_utc == fill("2026-09-07T09:59:59.123Z", 2)
        @test frame.receipt_completed_utc == fill("2026-09-07T09:59:59.124Z", 2)
        fallback_get(url; kwargs...) = _mock_response(BODY)
        fallback = _staged_future_rows_for_source(SOURCE; fetched_utc=fetched,
            receipt_clock=() -> fetched + Millisecond(1), http_get=fallback_get).rows
        @test fallback.issue_utc == fill("2026-09-07T09:59:59.123Z", 2)
        @test _validate_external_dst_log(fallback)
    end

    @testset "completion follows body and source-run metadata" begin
        clock_value = Ref(DateTime(2026, 9, 7, 9, 59))
        calls = String[]
        source = (name="metadata", kind="temerin_li_ascii", url="body", run_url="run")
        function delayed_get(url; kwargs...)
            push!(calls, url)
            if url == "body"
                clock_value[] = DateTime(2026, 9, 7, 9, 59, 59, 999)
                return _mock_response("2026/250-10:00:00 100.0\n2026/250-11:00:00 3.0\n")
            end
            clock_value[] = DateTime(2026, 9, 7, 10, 0, 0, 1)
            return _mock_response("Time of model run: 2026/250-09:00:00")
        end
        frame = _staged_future_rows_for_source(source;
            fetched_utc=DateTime(2026, 9, 7, 9, 59), http_get=delayed_get,
            receipt_clock=() -> begin
                push!(calls, "clock")
                clock_value[]
            end).rows
        @test calls == ["body", "run", "clock"]
        @test frame.receipt_completed_utc == fill("2026-09-07T10:00:00.001Z", 2)
        @test external_dst_timing.(eachrow(frame)) == [:late, :eligible]
        @test score_external_dst_rows!(frame, observations();
            scored_utc=DateTime(2026, 9, 7, 12)) == 1

        before = now(UTC)
        current_body = """[{"time_tag":"$(before + Hour(1))","dst":1.0},
                            {"time_tag":"$(before + Hour(2))","dst":2.0}]"""
        current_get(url; kwargs...) = _mock_response(current_body)
        current = _staged_future_rows_for_source(SOURCE;
            fetched_utc=before - Day(1), http_get=current_get).rows
        after = now(UTC)
        @test all(before <= time <= after
            for time in _parse_external_time.(current.receipt_completed_utc))
        @test all(external_dst_timing.(eachrow(current)) .== :eligible)
    end

    @testset "legacy scores stay intact but do not become prospective" begin
        legacy = fixture_rows()
        legacy.receipt_completed_utc .= missing
        legacy.observed_dst_nt .= 0.0
        legacy.observed_time_utc .= legacy.target_utc
        legacy.observed_gap_min .= 0.0
        legacy.abs_error_nt .= [100.0, 3.0]
        legacy.scored_utc .= "2026-09-07T12:00:00Z"
        before = copy(legacy)
        @test score_external_dst_rows!(legacy, observations();
            scored_utc=DateTime(2026, 9, 7, 12)) == 0
        @test isequal(before, legacy)
        @test _validate_external_dst_log(legacy)
        old_summary = only(eachrow(external_dst_summary(legacy)))
        @test old_summary.n_legacy == 2
        @test old_summary.n_scored == old_summary.n_eligible == 0
        @test ismissing(old_summary.rmse_nt)
        @test ismissing(old_summary.max_receipt_lead_h)
        legacy.fetched_utc[1] = "2026-09-07T10:00:00Z"
        @test external_dst_timing.(eachrow(legacy)) == [:late, :legacy]
        legacy.fetched_utc[2] = "2026-09-07T08:59:00Z"
        @test external_dst_timing(legacy[2, :]) == :legacy
        mixed = vcat(legacy, rows)
        mixed_summary = only(eachrow(external_dst_summary(mixed)))
        @test (mixed_summary.n_scored, mixed_summary.n_late,
               mixed_summary.n_legacy) == (1, 2, 1)
        @test mixed_summary.rmse_nt == mixed_summary.mae_nt == 3.0
    end

    @testset "invalid receipts fail closed" begin
        for completed in ("not-a-time", "2026-09-07T09:58:59Z",
                          "2026-09-07T10:30:00+08:00")
            frame = fixture_rows()
            frame.receipt_completed_utc .= completed
            @test all(external_dst_timing.(eachrow(frame)) .== :invalid)
            @test_throws ErrorException _validate_external_dst_log(frame)
            @test score_external_dst_rows!(frame, observations();
                scored_utc=DateTime(2026, 9, 7, 12)) == 0
        end
        frame = fixture_rows()
        frame.issue_utc[2] = "2026-09-07T10:31:00Z"
        frame.lead_h[2] = 29 / 60
        @test external_dst_timing(frame[2, :]) == :invalid
        @test_throws ErrorException _validate_external_dst_log(frame)
        frame = copy(rows)
        frame.receipt_completed_utc[2] = "2026-09-07T11:00:00Z"
        @test_throws ErrorException _validate_external_dst_log(frame)
        @test_throws ArgumentError fixture_rows(completed=DateTime(2026, 9, 7, 9, 58))
        @test_throws ArgumentError _staged_future_rows_for_source(SOURCE;
            http_get=fixture_get, receipt_clock=() -> "not a DateTime")
    end

    @testset "migration, duplicate receipt, and exact rollback" begin
        mktempdir() do dir
            cfg = ExternalDstCollectorConfig(log_path=joinpath(dir, "external.csv"),
                report_path=joinpath(dir, "external.md"), raw_dir=joinpath(dir, "raw"),
                repo_root=dir, sources=[SOURCE], max_log_rows=100, max_raw_snapshots=100)
            first_capture = capture_and_score_external_dst_snapshot!(cfg;
                fetched_utc=DateTime(2026, 9, 7, 9, 58), http_get=fixture_get,
                receipt_clock=() -> DateTime(2026, 9, 7, 9, 58, 1),
                observations=observations())
            @test first_capture.summary.n_eligible == [2]
            original = CSV.read(cfg.log_path, DataFrame)
            raw_bytes = read(joinpath(cfg.repo_root, original.raw_path[1]))
            again = capture_and_score_external_dst_snapshot!(cfg;
                fetched_utc=DateTime(2026, 9, 7, 9, 59), http_get=fixture_get,
                receipt_clock=() -> DateTime(2026, 9, 7, 10, 30),
                observations=observations())
            @test again.rows_added == 0
            @test isequal(CSV.read(cfg.log_path, DataFrame), original)
            @test read(joinpath(cfg.repo_root, original.raw_path[1])) == raw_bytes

            legacy = select(original, Not(:receipt_completed_utc))
            CSV.write(cfg.log_path, legacy)
            old_log, old_report = read(cfg.log_path), read(cfg.report_path)
            migrated = _load_external_log(cfg.log_path)
            @test all(ismissing, migrated.receipt_completed_utc)
            @test isequal(select(migrated, Not(:receipt_completed_utc)), legacy)
            @test read(cfg.log_path) == old_log
            @test_throws ErrorException _external_transactional_log_report!(
                cfg.log_path, cfg.report_path, migrated;
                after_log_commit=() -> error("injected migration failure"))
            @test read(cfg.log_path) == old_log
            @test read(cfg.report_path) == old_report
            again = capture_and_score_external_dst_snapshot!(cfg;
                fetched_utc=DateTime(2026, 9, 7, 12), http_get=fixture_get,
                receipt_clock=() -> DateTime(2026, 9, 7, 12, 0, 1),
                observations=observations())
            final = CSV.read(cfg.log_path, DataFrame)
            @test again.rows_added == again.rows_scored_now == 0
            @test isequal(select(final, Not(:receipt_completed_utc)), legacy)
            @test all(ismissing, final.receipt_completed_utc)
            @test read(joinpath(cfg.repo_root, original.raw_path[1])) == raw_bytes
            @test occursin("Unknown completion", read(cfg.report_path, String))
            @test occursin("strictly before target", read(cfg.report_path, String))
        end
    end
end

end
