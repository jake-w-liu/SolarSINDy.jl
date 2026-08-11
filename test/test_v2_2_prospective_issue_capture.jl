using Test
using Dates
using JSON3
using SHA

include(joinpath(
    @__DIR__, "..", "examples", "v2_2_prospective_issue_capture.jl",
))
const PIC = V22ProspectiveIssueCapture
const PICL1 = PIC.L1

_pic_response(body::AbstractString; status=200, headers=Pair{String,String}[]) = (
    status=Int(status),
    headers=collect(headers),
    body=Vector{UInt8}(codeunits(body)),
)

function _pic_clock(values)
    remaining = collect(values)
    return () -> begin
        isempty(remaining) && error("prospective-capture test clock exhausted")
        popfirst!(remaining)
    end
end

function _pic_ephemeris()
    return _pic_response(JSON3.write([
        (
            time_tag="2026-08-12T00:00:00Z", source="DSCOVR", active=true,
            x_gse=1_500_000.0, y_gse=10.0, z_gse=20.0,
        ),
        (
            time_tag="2026-08-12T01:00:00Z", source="DSCOVR", active=true,
            x_gse=1_499_000.0, y_gse=20.0, z_gse=30.0,
        ),
        (
            time_tag="2026-08-12T02:00:00Z", source="DSCOVR", active=true,
            x_gse=1_498_000.0, y_gse=30.0, z_gse=40.0,
        ),
    ]))
end

function _pic_mag(time_tag::AbstractString; bx=1.0, by=2.0, bz=-4.0)
    return _pic_response(JSON3.write([(
        time_tag=String(time_tag), source="DSCOVR", active=true,
        overall_quality=0, bx_gsm=bx, by_gsm=by, bz_gsm=bz,
    )]))
end

function _pic_wind(time_tag::AbstractString; speed=420.0, density=6.0,
                   vx=-415.0)
    return _pic_response(JSON3.write([(
        time_tag=String(time_tag), source="DSCOVR", active=true,
        overall_quality=0, proton_speed=speed, proton_density=density,
        proton_vx_gse=vx,
    )]))
end

function _pic_capture_l1!(root, receipt_start::DateTime,
                          measurement_time::AbstractString;
                          bx=1.0, speed=420.0,
                          wind_time::AbstractString=measurement_time)
    sources = PICL1.V22_L1_RECEIPT_SOURCES
    utc = [
        receipt_start,
        receipt_start + Millisecond(1),
        receipt_start + Millisecond(10),
        receipt_start + Millisecond(11),
        receipt_start + Millisecond(20),
        receipt_start + Millisecond(21),
    ]
    serial = Dates.value(receipt_start - DateTime(2026, 8, 12))
    monotonic = [100, 101, 110, 111, 120, 121] .+ serial
    return PICL1.capture_v2_2_l1_receipts!(
        root;
        http_get=(url; kwargs...) -> begin
            url == sources[1].url && return _pic_mag(measurement_time; bx=bx)
            url == sources[2].url && return _pic_wind(wind_time; speed=speed)
            error("unexpected synthetic L1 URL")
        end,
        ephemeris_http_get=(url; kwargs...) -> begin
            @test url == PICL1.V22_L1_NOAA_EPHEMERIS_URL
            _pic_ephemeris()
        end,
        utc_clock=_pic_clock(utc),
        monotonic_clock=_pic_clock(monotonic),
    )
end

function _pic_dst_body(time_tag::AbstractString, dst::Real)
    return JSON3.write([(
        time_tag=String(time_tag), dst=Float64(dst),
    )])
end

function _pic_capture_dst!(root, started::DateTime, completed::DateTime,
                           body::AbstractString; status=200,
                           headers=["Date" => "Wed, 12 Aug 2026 00:29:00 GMT",
                                    "ETag" => "\"synthetic\""])
    response = _pic_response(body; status=status, headers=headers)
    return PIC.capture_v2_2_dst_receipt!(
        root;
        http_get=(url; kwargs...) -> begin
            @test url == PIC.V22_DST_SOURCE.url
            @test kwargs[:status_exception] === false
            response
        end,
        utc_clock=_pic_clock([started, completed]),
        monotonic_clock=_pic_clock([1_000, 1_010]),
    )
end

function _pic_issue!(root, issue::DateTime;
                     prepared=issue + Second(30), monotonic=10_000,
                     crash_hook=stage -> nothing)
    return PIC.capture_v2_2_research_issue!(
        root, issue;
        preparation_utc_clock=() -> prepared,
        monotonic_clock=() -> monotonic,
        crash_hook=crash_hook,
    )
end

function _pic_raw_path(root, record)
    return joinpath(root, String(record.raw_relative_path))
end

@testset verbose=true "V2.2 off-by-default prospective issue capture" begin
    issue = DateTime(2026, 8, 12, 0, 30)
    measurement = "2026-08-12T00:28:00Z"

    @testset "strict half-hour schedule" begin
        @test_throws ErrorException PIC.main_v2_2_prospective_issue_capture(
            String[],
        )
        @test PIC.next_v2_2_issue_time(issue) == issue
        @test PIC.next_v2_2_issue_time(issue - Millisecond(1)) == issue
        @test PIC.next_v2_2_issue_time(issue + Millisecond(1)) ==
              issue + Minute(30)
        mktempdir() do root
            @test_throws ArgumentError PIC.capture_v2_2_research_issue!(
                root, issue - Second(1),
            )
            @test_throws ArgumentError PIC.capture_v2_2_research_issue!(
                root, issue + Millisecond(1),
            )
            @test isempty(readdir(root))
        end
    end

    @testset "explicit scheduler preserves the registered issue clock" begin
        mktempdir() do root
            calls = Symbol[]
            inputs = PIC.capture_v2_2_prospective_inputs!(
                root;
                l1_capture! = storage -> begin
                    push!(calls, :l1)
                    :synthetic_l1
                end,
                dst_capture! = storage -> begin
                    push!(calls, :dst)
                    :synthetic_dst
                end,
            )
            @test calls == [:l1, :dst]
            @test inputs == (l1=:synthetic_l1, dst=:synthetic_dst)

            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            scheduled = PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Second(1),
                monotonic_clock=() -> 20_000,
                input_capture! = storage ->
                    error("polled after issue became due"),
                sleeper=seconds -> error("slept after bounded scheduler run"),
                max_iterations=1,
            )
            @test scheduled.iterations == 1
            @test scheduled.next_issue_time_utc ==
                  "2026-08-12T01:00:00.000Z"
            @test !ispath(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test isfile(joinpath(
                root, PIC._issue_completion_relative(issue),
            ))
            scheduled_record = PIC.verify_v2_2_research_issue(root, issue)
            @test scheduled_record.issue_time_utc ==
                  "2026-08-12T00:30:00.000Z"
            @test scheduled_record.capture_mode ==
                  "scheduled_fail_closed_guard"
            @test scheduled_record.scheduler_completion_status == "required"
            @test scheduled_record.scheduler_pending_record_sha256 !=
                  PIC._ZERO_SHA256

            completion_path = joinpath(
                root, PIC._issue_completion_relative(issue),
            )
            completion_backup = completion_path * ".backup"
            mv(completion_path, completion_backup)
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            mv(completion_backup, completion_path)

            archive_path = joinpath(root, PIC._pending_archive_relative(
                issue, String(
                    scheduled_record.scheduler_pending_record_sha256,
                ),
            ))
            @test isfile(archive_path)
            archive_backup = archive_path * ".backup"
            mv(archive_path, archive_backup)
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            mv(archive_backup, archive_path)
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == scheduled_record.issue_record_sha256
        end

        mktempdir() do root
            @test_throws ArgumentError PIC.run_v2_2_research_capture_scheduler!(
                root, issue + Second(1); max_iterations=1,
            )
            @test_throws ArgumentError PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Minute(6),
                input_capture! = storage ->
                    error("late scheduler polled inputs"),
                sleeper=seconds -> nothing,
                max_iterations=1,
            )
        end

        mktempdir() do root
            @test_throws ErrorException PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Second(1),
                monotonic_clock=() -> 30_000,
                input_capture! = storage ->
                    error("uncommitted retry test polled inputs"),
                max_iterations=1,
            )
            @test !ispath(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test !ispath(joinpath(
                root, PIC._issue_record_relative(issue),
            ))

            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            retried = PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Second(1),
                monotonic_clock=() -> 30_000,
                input_capture! = storage ->
                    error("uncommitted retry test polled inputs"),
                max_iterations=1,
            )
            @test retried.next_issue_time_utc ==
                  "2026-08-12T01:00:00.000Z"
            @test !ispath(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_sequence == 1
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            completion_clock = _pic_clock([
                issue + Minute(4) + Second(59), issue + Minute(6),
            ])
            @test_throws ArgumentError PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=completion_clock,
                input_capture! = storage ->
                    error("deadline test polled inputs"),
                sleeper=seconds -> nothing,
                max_iterations=1,
            )
            @test isfile(joinpath(
                root, PIC._issue_record_relative(issue),
            ))
            marker_path = joinpath(root, PIC._INVALID_COHORT_RELATIVE)
            @test isfile(marker_path)
            marker = JSON3.read(read(marker_path, String))
            @test marker.invalid_issue_time_utc ==
                  "2026-08-12T00:30:00.000Z"
            @test marker.reason ==
                  "durable_issue_after_five_minute_window"
            @test isfile(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            @test_throws ArgumentError PIC.verify_v2_2_research_issues(root)
            @test_throws ArgumentError _pic_issue!(
                root, issue + Minute(30),
            )
            input_calls = Symbol[]
            @test_throws ArgumentError PIC.capture_v2_2_prospective_inputs!(
                root;
                l1_capture! = storage -> push!(input_calls, :l1),
                dst_capture! = storage -> push!(input_calls, :dst),
            )
            @test isempty(input_calls)
            dst_http_called = Ref(false)
            @test_throws ArgumentError PIC.capture_v2_2_dst_receipt!(
                root;
                http_get=(url; kwargs...) -> begin
                    dst_http_called[] = true
                    _pic_response(_pic_dst_body(
                        "2026-08-12T00:00:00", -1.0,
                    ))
                end,
            )
            @test !dst_http_called[]
            restarted_polls = Ref(0)
            @test_throws ArgumentError PIC.run_v2_2_research_capture_scheduler!(
                root, issue + Minute(30);
                wall_clock=() -> issue + Minute(24),
                input_capture! = storage -> (restarted_polls[] += 1),
                max_iterations=1,
            )
            @test restarted_polls[] == 0
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            completion_clock = _pic_clock([
                issue + Minute(4) + Second(59), issue + Minute(6),
            ])
            @test_throws ErrorException PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=completion_clock,
                input_capture! = storage ->
                    error("invalidation failure test polled inputs"),
                invalidation_crash_hook=stage -> begin
                    @test stage == :before_invalid_cohort_marker
                    error("synthetic invalidation persistence failure")
                end,
                max_iterations=1,
            )
            @test !ispath(joinpath(root, PIC._INVALID_COHORT_RELATIVE))
            @test isfile(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test isfile(joinpath(
                root, PIC._issue_record_relative(issue),
            ))
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            pending_path = joinpath(root, PIC._PENDING_ISSUE_RELATIVE)
            pending_backup = pending_path * ".backup"
            mv(pending_path, pending_backup)
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            mv(pending_backup, pending_path)
            @test_throws ArgumentError _pic_issue!(
                root, issue + Minute(30),
            )
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            @test_throws ArgumentError PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Second(1),
                monotonic_clock=_pic_clock([100, 99]),
                input_capture! = storage ->
                    error("monotonic regression test polled inputs"),
                max_iterations=1,
            )
            @test !ispath(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test !ispath(joinpath(
                root, PIC._issue_record_relative(issue),
            ))
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -11.0),
            )
            @test_throws ErrorException PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue + Second(1),
                input_capture! = storage ->
                    error("issue crash test polled inputs"),
                issue_crash_hook=stage -> begin
                    @test stage == :after_issue_record
                    error("synthetic scheduler issue crash")
                end,
                max_iterations=1,
            )
            @test isfile(joinpath(root, PIC._PENDING_ISSUE_RELATIVE))
            @test isfile(joinpath(
                root, PIC._issue_record_relative(issue),
            ))
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            @test_throws ArgumentError _pic_issue!(
                root, issue + Minute(30),
            )
        end


        mktempdir() do root
            polls = Ref(0)
            waiting = PIC.run_v2_2_research_capture_scheduler!(
                root, issue;
                wall_clock=() -> issue - Minute(6),
                input_capture! = storage -> (polls[] += 1),
                sleeper=seconds -> error("slept after bounded scheduler run"),
                max_iterations=1,
            )
            @test polls[] == 1
            @test waiting.next_issue_time_utc ==
                  "2026-08-12T00:30:00.000Z"
            @test !ispath(joinpath(root, "research_issues"))
        end
    end

    @testset "immutable cutoff, Dst anchor, and unavailable issuance" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            dst = _pic_capture_dst!(
                root,
                DateTime(2026, 8, 12, 0, 29, 10),
                DateTime(2026, 8, 12, 0, 29, 11),
                _pic_dst_body("2026-08-12T00:00:00", -18.0),
            )
            record = _pic_issue!(root, issue)

            @test record.schema_version == PIC.V22_RESEARCH_ISSUE_SCHEMA_VERSION
            @test record.issue_time_utc == "2026-08-12T00:30:00.000Z"
            @test record.issue_sequence == 1
            @test record.commit_witness_status ==
                  "unavailable_research_capture_only"
            @test record.capture_mode == "manual_research_capture"
            @test record.scheduler_pending_record_sha256 == PIC._ZERO_SHA256
            @test record.scheduler_completion_status == "not_applicable"
            @test record.target_times_utc == [
                "2026-08-12T01:30:00.000Z",
                "2026-08-12T02:30:00.000Z",
                "2026-08-12T03:30:00.000Z",
                "2026-08-12T04:30:00.000Z",
                "2026-08-12T06:30:00.000Z",
                "2026-08-12T07:30:00.000Z",
            ]
            @test record.l1_pair_schema_version ==
                  PICL1.V22_L1_ISSUE_PAIR_SCHEMA_VERSION
            @test record.l1_pair_status ==
                  "available_verified_cutoff_pair"
            @test record.l1_pair_source == "DSCOVR"
            @test record.l1_pair_measurement_time_utc ==
                  "2026-08-12T00:28:00.000Z"
            @test record.l1_pair_contract_sha256 != PIC._ZERO_SHA256
            @test record.dst_source_url == PIC.V22_DST_SOURCE.url
            @test record.dst_cutoff_sequence == 1
            @test record.dst_cutoff_record_sha256 == dst.record_sha256
            @test record.dst_anchor_record_sha256 == dst.record_sha256
            @test record.dst_anchor_time_utc == "2026-08-12T00:00:00.000Z"
            @test record.dst_anchor_age_seconds == 1_800
            @test record.dst_anchor_status ==
                  "available_receipt_causal_anchor"
            @test record.model_component_status ==
                  "unavailable_no_fitted_gated_v2_2"
            @test record.issuance_status ==
                  "research_capture_only_unavailable"
            @test record.numeric_forecast_status == "not_emitted"
            @test !hasproperty(record, :forecast_dst_nt)
            @test !hasproperty(record, :prediction)
            @test all(name -> begin
                !occursin("forecast", String(name)) ||
                    getproperty(record, name) isa AbstractString
            end, propertynames(record))
            @test bytes2hex(sha256(codeunits(JSON3.write(
                PIC._issue_payload(record),
            )))) == record.issue_record_sha256

            verified = PIC.verify_v2_2_research_issue(root, issue)
            @test verified.issue_record_sha256 == record.issue_record_sha256
            all_records = PIC.verify_v2_2_research_issues(root)
            @test length(all_records) == 1

            duplicate = PIC.capture_v2_2_research_issue!(
                root, issue;
                preparation_utc_clock=() ->
                    error("duplicate used preparation clock"),
                monotonic_clock=() -> error("duplicate used monotonic clock"),
            )
            @test duplicate.issue_record_sha256 == record.issue_record_sha256

            second_issue = issue + Minute(30)
            second = _pic_issue!(root, second_issue)
            @test second.issue_sequence == 2
            @test second.previous_issue_record_sha256 ==
                  record.issue_record_sha256
            @test second.previous_issue_record_relative_path ==
                  record.issue_record_relative_path
            @test length(PIC.verify_v2_2_research_issues(root)) == 2
            @test_throws ArgumentError _pic_issue!(
                root, second_issue + Hour(1),
            )
        end
    end

    @testset "Dst headers, first receipt, and revision lineage" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            first_body = _pic_dst_body("2026-08-12T00:00:00", -10.0)
            first = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 10),
                DateTime(2026, 8, 12, 0, 10, 1), first_body,
            )
            repeat_record = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 11),
                DateTime(2026, 8, 12, 0, 11, 1), first_body,
            )
            revision = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 20),
                DateTime(2026, 8, 12, 0, 20, 1),
                _pic_dst_body("2026-08-12T00:00:00", -20.0),
            )
            @test first.revision_status == "first_observation"
            @test first.revision_ordinal == 1
            @test repeat_record.revision_status == "unchanged_repeat"
            @test repeat_record.revision_ordinal == 2
            @test repeat_record.revision_of_record_sha256 == first.record_sha256
            @test repeat_record.first_body_receipt_completed_utc ==
                  first.receipt_completed_utc
            @test repeat_record.first_body_monotonic_completed_ns ==
                  first.monotonic_completed_ns
            @test revision.revision_status == "revised_observation"
            @test revision.revision_ordinal == 3
            @test revision.revision_of_record_sha256 ==
                  repeat_record.record_sha256
            @test revision.response_headers_sha256 ==
                  PIC._headers_sha256(revision.response_headers)
            @test revision.parser_decision == "accept_latest_unique_dst_row"
            @test revision.provider_row_sha256 != PIC._ZERO_SHA256
            @test read(_pic_raw_path(root, revision)) ==
                  Vector{UInt8}(codeunits(
                      _pic_dst_body("2026-08-12T00:00:00", -20.0),
                  ))

            issued = _pic_issue!(root, issue)
            @test issued.dst_cutoff_sequence == 3
            @test issued.dst_anchor_record_sha256 == revision.record_sha256
            @test PIC.verify_v2_2_research_issue(
                root, issue,
            ).issue_record_sha256 == issued.issue_record_sha256
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            preissue = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 20),
                DateTime(2026, 8, 12, 0, 20, 1),
                _pic_dst_body("2026-08-12T00:00:00", -14.0),
            )
            postissue_revision = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 31),
                DateTime(2026, 8, 12, 0, 31, 1),
                _pic_dst_body("2026-08-12T00:00:00", -22.0),
            )
            @test postissue_revision.revision_status ==
                  "revised_observation"

            issued = _pic_issue!(root, issue; prepared=issue + Minute(2))
            @test issued.dst_cutoff_sequence == preissue.sequence
            @test issued.dst_cutoff_record_sha256 == preissue.record_sha256
            @test issued.dst_anchor_status ==
                  "available_receipt_causal_anchor"
            @test issued.dst_anchor_record_sha256 == preissue.record_sha256
            @test issued.dst_anchor_record_sha256 !=
                  postissue_revision.record_sha256
            @test issued.dst_anchor_time_utc ==
                  "2026-08-12T00:00:00.000Z"
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == issued.issue_record_sha256
        end
    end

    @testset "late Dst is excluded without moving the issue clock" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            late = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 30, 30),
                DateTime(2026, 8, 12, 0, 30, 31),
                _pic_dst_body("2026-08-12T00:00:00", -15.0),
            )
            record = _pic_issue!(root, issue; prepared=issue + Minute(1))
            @test late.sequence == 1
            @test record.issue_time_utc == "2026-08-12T00:30:00.000Z"
            @test record.dst_cutoff_sequence == 0
            @test record.dst_cutoff_record_sha256 == PIC._ZERO_SHA256
            @test isempty(record.dst_cutoff_record_relative_path)
            @test record.dst_anchor_status ==
                  "unavailable_no_preissue_dst_anchor"
            @test isempty(record.dst_anchor_time_utc)
            @test record.dst_anchor_age_seconds == -1
            @test record.issuance_status ==
                  "research_capture_only_unavailable"
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == record.issue_record_sha256
        end
    end

    @testset "an issue binds explicit L1-pair unavailability" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement;
                wind_time="2026-08-12T00:27:00Z",
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -12.0),
            )
            record = _pic_issue!(root, issue)
            @test record.l1_pair_status ==
                  "unavailable_no_exact_admitted_pair"
            @test record.l1_pair_contract_sha256 == PIC._ZERO_SHA256
            @test isempty(record.l1_pair_measurement_time_utc)
            @test isempty(record.l1_pair_source)
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == record.issue_record_sha256
        end
    end

    @testset "post-issue objects cannot alter historical verification" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            bound_dst = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -12.0),
            )
            original = _pic_issue!(root, issue)

            future_dst = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 40),
                DateTime(2026, 8, 12, 0, 40, 1),
                _pic_dst_body("2026-08-12T00:30:00", -30.0),
            )
            future_l1 = _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 40, 10),
                "2026-08-12T00:38:00Z"; bx=9.0, speed=500.0,
            )
            future_dst_path = joinpath(
                root, PIC._dst_record_relative(future_dst.sequence),
            )
            future_dst_bytes = read(future_dst_path)
            write(future_dst_path, "future corruption")
            future_mag_path = joinpath(
                root, String(first(future_l1).raw_relative_path),
            )
            future_mag_bytes = read(future_mag_path)
            write(future_mag_path, "future corruption")

            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == original.issue_record_sha256

            write(future_dst_path, future_dst_bytes)
            write(future_mag_path, future_mag_bytes)
            bound_path = _pic_raw_path(root, bound_dst)
            bound_bytes = read(bound_path)
            write(bound_path, "bound corruption")
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            write(bound_path, bound_bytes)
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == original.issue_record_sha256

            issue_path = joinpath(root, String(original.issue_record_relative_path))
            issue_bytes = read(issue_path)
            stored = JSON3.read(String(issue_bytes))
            truncated_payload = merge(
                PIC._issue_payload(stored),
                (
                    dst_cutoff_sequence=0,
                    dst_cutoff_record_relative_path="",
                    dst_cutoff_record_sha256=PIC._ZERO_SHA256,
                    dst_anchor_status="unavailable_no_preissue_dst_anchor",
                    dst_anchor_record_relative_path="",
                    dst_anchor_record_sha256=PIC._ZERO_SHA256,
                    dst_anchor_time_utc="",
                    dst_anchor_age_seconds=-1,
                ),
            )
            truncated_sha = PIC._issue_record_sha256(truncated_payload)
            write(issue_path, JSON3.write(merge(
                truncated_payload, (issue_record_sha256=truncated_sha,),
            )))
            @test PIC._issue_record_sha256(JSON3.read(read(
                issue_path, String,
            ))) == truncated_sha
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )

            postissue_payload = merge(
                PIC._issue_payload(stored),
                (
                    dst_cutoff_sequence=Int(future_dst.sequence),
                    dst_cutoff_record_relative_path=
                        PIC._dst_record_relative(Int(future_dst.sequence)),
                    dst_cutoff_record_sha256=String(future_dst.record_sha256),
                ),
            )
            write(issue_path, JSON3.write(merge(
                postissue_payload,
                (issue_record_sha256=PIC._issue_record_sha256(postissue_payload),),
            )))
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )

            semantic_payload = merge(
                PIC._issue_payload(stored), (issuance_status="available",),
            )
            semantic_sha = PIC._issue_record_sha256(semantic_payload)
            write(issue_path, JSON3.write(merge(
                semantic_payload, (issue_record_sha256=semantic_sha,),
            )))
            @test PIC._issue_record_sha256(JSON3.read(read(
                issue_path, String,
            ))) == semantic_sha
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )

            pair_payload = merge(
                PIC._issue_payload(stored),
                (l1_pair_contract_sha256=repeat("f", 64),),
            )
            write(issue_path, JSON3.write(merge(
                pair_payload,
                (issue_record_sha256=PIC._issue_record_sha256(pair_payload),),
            )))
            @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                root, issue,
            )
            write(issue_path, issue_bytes)
        end
    end

    @testset "crash recovery is idempotent" begin
        mktempdir() do root
            response = _pic_response(
                _pic_dst_body("2026-08-12T00:00:00", -8.0),
            )
            @test_throws ErrorException PIC.capture_v2_2_dst_receipt!(
                root;
                http_get=(url; kwargs...) -> response,
                utc_clock=_pic_clock([
                    DateTime(2026, 8, 12, 0, 20),
                    DateTime(2026, 8, 12, 0, 20, 1),
                ]),
                monotonic_clock=_pic_clock([100, 101]),
                crash_hook=stage -> begin
                    @test stage == :after_dst_record
                    error("synthetic Dst crash")
                end,
            )
            @test !ispath(joinpath(root, "dst", "latest.json"))
            recovered_dst = PIC.capture_v2_2_dst_receipt!(
                root;
                http_get=(url; kwargs...) -> error("retry fetched Dst"),
                utc_clock=() -> error("retry used UTC clock"),
                monotonic_clock=() -> error("retry used monotonic clock"),
            )
            @test recovered_dst.sequence == 1
            @test recovered_dst.parser_decision ==
                  "accept_latest_unique_dst_row"
            @test isfile(joinpath(root, "dst", "latest.json"))
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 25),
                DateTime(2026, 8, 12, 0, 25, 1),
                _pic_dst_body("2026-08-12T00:00:00", -12.0),
            )
            injected = Ref(false)
            @test_throws ErrorException _pic_issue!(
                root, issue;
                crash_hook=stage -> begin
                    @test stage == :after_issue_record
                    injected[] = true
                    error("synthetic crash")
                end,
            )
            @test injected[]
            @test !ispath(joinpath(root, "research_issues", "latest.json"))
            @test_throws ArgumentError PIC.verify_v2_2_research_issues(root)
            recovered = PIC.capture_v2_2_research_issue!(
                root, issue;
                preparation_utc_clock=() -> error("retry recaptured issue"),
                monotonic_clock=() -> error("retry recaptured clock"),
            )
            @test recovered.issue_sequence == 1
            @test isfile(joinpath(root, "research_issues", "latest.json"))
            @test length(PIC.verify_v2_2_research_issues(root)) == 1
        end
    end

    @testset "parser and path failures close" begin
        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            rejected = _pic_capture_dst!(
                root, DateTime(2026, 8, 12, 0, 20),
                DateTime(2026, 8, 12, 0, 20, 1),
                _pic_dst_body("2026-08-12T00:21:00", -10.0),
            )
            @test rejected.parser_decision == "reject_no_admissible_dst_row"
            @test rejected.parser_rejections == ["row[1]:post_receipt_time_tag"]
            unavailable = _pic_issue!(root, issue)
            @test unavailable.dst_anchor_status ==
                  "unavailable_no_preissue_dst_anchor"
        end

        mktempdir() do root
            _pic_capture_l1!(
                root, DateTime(2026, 8, 12, 0, 29), measurement,
            )
            transport = PIC.capture_v2_2_dst_receipt!(
                root;
                http_get=(url; kwargs...) -> error("synthetic transport loss"),
                utc_clock=_pic_clock([
                    DateTime(2026, 8, 12, 0, 20),
                    DateTime(2026, 8, 12, 0, 20, 1),
                ]),
                monotonic_clock=_pic_clock([200, 201]),
            )
            @test transport.capture_outcome == "transport_error"
            @test transport.parser_decision == "reject_transport_error"
            @test transport.body_sha256 == PIC._ZERO_SHA256
            transport_issue = _pic_issue!(root, issue)
            @test transport_issue.dst_anchor_status ==
                  "unavailable_no_preissue_dst_anchor"
            @test PIC.verify_v2_2_research_issue(root, issue).
                  issue_record_sha256 == transport_issue.issue_record_sha256
        end

        if !Sys.iswindows()
            mktempdir() do parent
                real_root = joinpath(parent, "real")
                mkpath(real_root)
                linked_root = joinpath(parent, "linked")
                symlink(real_root, linked_root)
                @test_throws ArgumentError PIC.capture_v2_2_dst_receipt!(
                    linked_root;
                    http_get=(url; kwargs...) ->
                        _pic_response(_pic_dst_body(
                            "2026-08-12T00:00:00Z", -1.0,
                        )),
                )
            end

            mktempdir() do root
                _pic_capture_l1!(
                    root, DateTime(2026, 8, 12, 0, 29), measurement,
                )
                dst = _pic_capture_dst!(
                    root, DateTime(2026, 8, 12, 0, 20),
                    DateTime(2026, 8, 12, 0, 20, 1),
                    _pic_dst_body("2026-08-12T00:00:00", -9.0),
                )
                _pic_issue!(root, issue)
                raw_path = _pic_raw_path(root, dst)
                backup_path = raw_path * ".backup"
                mv(raw_path, backup_path)
                symlink(backup_path, raw_path)
                @test_throws ArgumentError PIC.verify_v2_2_research_issue(
                    root, issue,
                )
            end
        end
    end
end
