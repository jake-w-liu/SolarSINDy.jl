using Test
using Dates
using JSON3

module V22L1ReceiptCollectorHarness
include(joinpath(@__DIR__, "..", "examples", "v2_2_l1_receipt_collector.jl"))
end

const L1C = V22L1ReceiptCollectorHarness

function _v22_l1_fake_response(status::Integer, body::AbstractString;
                               headers=Pair{String,String}[])
    return (
        status=Int(status),
        headers=collect(headers),
        body=Vector{UInt8}(codeunits(body)),
    )
end

function _v22_l1_sequence_clock(values)
    remaining = collect(values)
    return () -> begin
        isempty(remaining) && error("test clock exhausted")
        popfirst!(remaining)
    end
end

function _v22_l1_test_latest!(root, source, sequence, relative, checksum)
    path = joinpath(root, "latest", String(source.name) * ".json")
    write(path, JSON3.write((
        schema_version=L1C.V22_L1_RECEIPT_SCHEMA_VERSION,
        source_name=String(source.name),
        sequence=Int(sequence),
        record_relative_path=String(relative),
        record_sha256=String(checksum),
    )))
    return path
end

function _v22_l1_capture_at(root, source, response, start::DateTime, serial::Int;
                            http_get=(url; kwargs...) -> response)
    return only(L1C.capture_v2_2_l1_receipts!(
        root;
        sources=(source,),
        http_get=http_get,
        utc_clock=_v22_l1_sequence_clock([
            start,
            start + Millisecond(1),
        ]),
        monotonic_clock=_v22_l1_sequence_clock([serial, serial + 1]),
    ))
end

@testset "V2.2 prospective L1 receipt collector" begin
    source = (
        name="test_swpc_mag",
        url="https://example.invalid/rtsw_mag_1m.json",
    )
    body = """[{"time_tag":"2022-12-31T23:58:00Z","source":"DSCOVR","active":true}]"""
    response = _v22_l1_fake_response(
        200, body;
        headers=[
            "Date" => "Sat, 31 Dec 2022 23:59:00 GMT",
            "ETag" => "\"frozen\"",
            "Last-Modified" => "Sat, 31 Dec 2022 23:58:30 GMT",
        ],
    )
    fake_get = (url; kwargs...) -> begin
        @test url == source.url
        @test kwargs[:status_exception] === false
        response
    end

    mktempdir() do root
        utc_clock = _v22_l1_sequence_clock([
            DateTime(2022, 12, 31, 23, 59, 0, 100),
            DateTime(2022, 12, 31, 23, 59, 0, 140),
            DateTime(2022, 12, 31, 23, 59, 1, 100),
            DateTime(2022, 12, 31, 23, 59, 1, 150),
        ])
        monotonic_clock = _v22_l1_sequence_clock([100, 140, 1_100, 1_150])

        first_capture = L1C.capture_v2_2_l1_receipts!(
            root; sources=(source,), http_get=fake_get,
            utc_clock=utc_clock, monotonic_clock=monotonic_clock,
        )
        second_capture = L1C.capture_v2_2_l1_receipts!(
            root; sources=(source,), http_get=fake_get,
            utc_clock=utc_clock, monotonic_clock=monotonic_clock,
        )

        @test only(first_capture).sequence == 1
        @test only(second_capture).sequence == 2
        @test only(first_capture).body_sha256 == only(second_capture).body_sha256
        @test only(second_capture).previous_record_sha256 ==
              only(first_capture).record_sha256
        @test only(second_capture).previous_record_relative_path ==
              only(first_capture).record_relative_path
        @test only(first_capture).http_status == 200
        @test only(first_capture).capture_outcome == "http_response"
        @test isempty(only(first_capture).transport_error_type)
        @test isempty(only(first_capture).transport_error_message)
        @test only(first_capture).http_date ==
              "Sat, 31 Dec 2022 23:59:00 GMT"
        @test only(first_capture).http_etag == "\"frozen\""
        @test only(first_capture).http_last_modified ==
              "Sat, 31 Dec 2022 23:58:30 GMT"
        @test only(first_capture).json_valid
        @test only(first_capture).array_valid
        @test only(first_capture).row_count == 1
        @test only(first_capture).minimum_time_tag ==
              "2022-12-31T23:58:00Z"
        @test only(first_capture).receipt_completed_utc ==
              "2022-12-31T23:59:00.140Z"
        @test basename(only(first_capture).record_relative_path) ==
              "00000000000000000001.json"
        @test basename(only(second_capture).record_relative_path) ==
              "00000000000000000002.json"

        raw_files = String[]
        for (directory, _, files) in walkdir(joinpath(root, "raw"))
            append!(raw_files, joinpath.(directory, files))
        end
        @test length(raw_files) == 1
        @test read(only(raw_files), String) == body

        verification = L1C.verify_v2_2_l1_receipts(root; sources=(source,))
        @test only(verification).records == 2
        @test only(verification).latest_record_sha256 ==
              only(second_capture).record_sha256

        open(only(raw_files), "a") do io
            write(io, UInt8('x'))
        end
        @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
            root; sources=(source,),
        )
    end

    @testset "non-success and non-JSON responses remain immutable evidence" begin
        mktempdir() do root
            response_503 = _v22_l1_fake_response(503, "upstream unavailable")
            record = only(L1C.capture_v2_2_l1_receipts!(
                root;
                sources=(source,),
                http_get=(url; kwargs...) -> response_503,
                utc_clock=_v22_l1_sequence_clock([
                    DateTime(2022, 1, 1), DateTime(2022, 1, 1, 0, 0, 1),
                ]),
                monotonic_clock=_v22_l1_sequence_clock([10, 20]),
            ))
            @test record.http_status == 503
            @test !record.json_valid
            @test !record.array_valid
            @test record.row_count == 0
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end
    end

    @testset "unsafe archive identities and clocks fail closed" begin
        mktempdir() do root
            response_ok = _v22_l1_fake_response(200, "[]")
            @test_throws ArgumentError L1C.capture_v2_2_l1_receipts!(
                root;
                sources=((name="../escape", url=source.url),),
                http_get=(url; kwargs...) -> response_ok,
            )
            @test_throws ArgumentError L1C.capture_v2_2_l1_receipts!(
                root;
                sources=((name="duplicate", url=source.url),
                         (name="duplicate", url=source.url)),
                http_get=(url; kwargs...) -> response_ok,
            )
            @test_throws ArgumentError L1C.capture_v2_2_l1_receipts!(
                root;
                sources=((name="CaseName", url=source.url),
                         (name="casename", url=source.url)),
                http_get=(url; kwargs...) -> response_ok,
            )
            @test_throws ArgumentError L1C.capture_v2_2_l1_receipts!(
                root;
                sources=((name="insecure", url="http://example.invalid"),),
                http_get=(url; kwargs...) -> response_ok,
            )
            @test_throws ArgumentError L1C._v22_l1_install_record!(
                realpath(root), source, response_ok,
                DateTime(2022, 1, 1, 0, 0, 1), DateTime(2022, 1, 1),
                10, 20,
            )
            @test_throws ArgumentError L1C._v22_l1_install_record!(
                realpath(root), source, response_ok,
                DateTime(2022, 1, 1), DateTime(2022, 1, 1, 0, 0, 1),
                20, 10,
            )
        end
        if !Sys.iswindows()
            mktempdir() do parent
                real_root = joinpath(parent, "real")
                mkpath(real_root)
                linked_root = joinpath(parent, "linked")
                symlink(real_root, linked_root)
                @test_throws ArgumentError L1C.verify_v2_2_l1_receipts(
                    linked_root; sources=(source,),
                )
            end
        end
    end

    @testset "intermediate symbolic links cannot escape the root" begin
        if !Sys.iswindows()
            mktempdir() do parent
                root = joinpath(parent, "root")
                outside = joinpath(parent, "outside")
                mkpath(joinpath(root, "raw"))
                mkpath(outside)
                checksum = L1C._v22_l1_sha256(response.body)
                symlink(outside, joinpath(root, "raw", checksum[1:2]))
                @test_throws ArgumentError _v22_l1_capture_at(
                    root, source, response, DateTime(2022, 2, 1), 100,
                )
                # A missing component check would write the response outside root.
                @test isempty(readdir(outside))
            end

            mktempdir() do parent
                root = joinpath(parent, "root")
                outside = joinpath(parent, "outside")
                mkpath(root)
                mkpath(outside)
                symlink(outside, joinpath(root, "records"))
                @test_throws ArgumentError _v22_l1_capture_at(
                    root, source, response, DateTime(2022, 2, 1), 200,
                )
                @test isempty(readdir(outside))
            end
        end
    end

    @testset "head rollback, orphan records, and broken predecessors fail" begin
        mktempdir() do root
            first_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 1), 100,
            )
            second_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 1, 0, 1), 200,
            )
            _v22_l1_test_latest!(
                root, source, first_record.sequence,
                first_record.record_relative_path, first_record.record_sha256,
            )
            # A verifier that follows only `latest` misses the still-present suffix.
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
            @test isfile(joinpath(root, second_record.record_relative_path))
            @test_throws ErrorException _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 1, 0, 2), 250,
            )
        end

        mktempdir() do root
            first_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 2), 300,
            )
            rm(joinpath(root, first_record.record_relative_path))
            @test_throws ErrorException _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 2, 0, 1), 400,
            )
            latest = JSON3.read(read(
                joinpath(root, "latest", source.name * ".json"), String,
            ))
            # Append must not advance a syntactically valid pointer with no record.
            @test Int(latest.sequence) == 1
        end

        mktempdir() do root
            first_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 2, 1), 410,
            )
            second_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 2, 1, 1), 420,
            )
            rm(joinpath(root, first_record.record_relative_path))
            # A valid head is insufficient: startup validation must reject a
            # missing older predecessor before it appends sequence three.
            @test_throws ErrorException _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 2, 1, 2), 430,
            )
            @test !ispath(joinpath(
                root, L1C._v22_l1_record_relative(source.name, 3),
            ))
            @test isfile(joinpath(root, second_record.record_relative_path))
        end

        mktempdir() do root
            _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 3), 500,
            )
            orphan_relative = joinpath(
                "records", source.name,
                "00000000000000000002.json",
            )
            write(joinpath(root, orphan_relative), "{}")
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
        end

        mktempdir() do root
            _v22_l1_capture_at(
                root, source, response, DateTime(2022, 3, 4), 600,
            )
            orphan_sha = repeat("b", 64)
            orphan_directory = joinpath(root, "raw", orphan_sha[1:2])
            mkpath(orphan_directory)
            write(joinpath(orphan_directory, orphan_sha * ".raw"), UInt8[0x00])
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
        end
    end

    @testset "source URL and receipt chronology remain immutable" begin
        mktempdir() do root
            old_source = (name=source.name, url="https://example.invalid/old")
            new_source = (name=source.name, url="https://example.invalid/new")
            old_record = _v22_l1_capture_at(
                root, old_source, response, DateTime(2022, 4, 1), 100,
            )
            @test_throws ErrorException _v22_l1_capture_at(
                root, new_source, response, DateTime(2022, 4, 1, 0, 1), 200,
            )
            verified = only(L1C.verify_v2_2_l1_receipts(
                root; sources=(old_source,),
            ))
            @test verified.records == 1
            @test verified.latest_record_sha256 == old_record.record_sha256
        end

        mktempdir() do root
            first_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 4, 3), 300,
            )
            @test_throws ArgumentError _v22_l1_capture_at(
                root, source, response, DateTime(2022, 4, 2), 400,
            )
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).latest_record_sha256 == first_record.record_sha256
        end

        mktempdir() do root
            first_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 4, 4), 500,
            )
            second_record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 4, 5), 600,
            )
            second_path = joinpath(root, second_record.record_relative_path)
            second_json = JSON3.read(read(second_path, String))
            mutated_payload = merge(
                L1C._v22_l1_record_payload(second_json),
                (
                    request_started_utc="2022-04-03T00:00:00.000Z",
                    receipt_completed_utc="2022-04-03T00:00:00.001Z",
                ),
            )
            mutated_sha = L1C._v22_l1_record_sha256(mutated_payload)
            mutated_relative = L1C._v22_l1_record_relative(
                source.name, 2,
            )
            rm(second_path)
            write(
                joinpath(root, mutated_relative),
                JSON3.write(merge(mutated_payload, (record_sha256=mutated_sha,))),
            )
            _v22_l1_test_latest!(root, source, 2, mutated_relative, mutated_sha)
            # Hash/path recomputation ensures chronology, not checksum mismatch, rejects it.
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
            @test isfile(joinpath(root, first_record.record_relative_path))
        end
    end

    @testset "verification is read-only and rejects empty evidence" begin
        mktempdir() do parent
            missing = joinpath(parent, "missing")
            @test_throws ArgumentError L1C.verify_v2_2_l1_receipts(
                missing; sources=(source,),
            )
            @test !ispath(missing)
        end
        mktempdir() do root
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
            @test_throws ArgumentError L1C.verify_v2_2_l1_receipts(
                root; sources=(),
            )
        end
        mktempdir() do root
            _v22_l1_capture_at(
                root, source, response, DateTime(2022, 5, 1), 100,
            )
            @test !ispath(joinpath(root, ".collector.lock"))
            before = sort([
                relpath(joinpath(directory, name), root)
                for (directory, _, names) in walkdir(root) for name in names
            ])
            L1C.verify_v2_2_l1_receipts(root; sources=(source,))
            # Verification must not create its own lock or any other archive entry.
            @test !ispath(joinpath(root, ".collector.lock"))
            after = sort([
                relpath(joinpath(directory, name), root)
                for (directory, _, names) in walkdir(root) for name in names
            ])
            @test after == before
        end
    end

    @testset "transport exceptions are durable and do not stop sibling sources" begin
        mktempdir() do root
            failing_source = (
                name="test_transport_failure",
                url="https://example.invalid/failure",
            )
            healthy_source = (
                name="test_transport_success",
                url="https://example.invalid/success",
            )
            utc_clock = _v22_l1_sequence_clock([
                DateTime(2022, 6, 1),
                DateTime(2022, 6, 1) + Millisecond(1),
                DateTime(2022, 6, 1) + Second(1),
                DateTime(2022, 6, 1) + Second(1) + Millisecond(1),
            ])
            monotonic_clock = _v22_l1_sequence_clock([10, 11, 20, 21])
            calls = String[]
            records = L1C.capture_v2_2_l1_receipts!(
                root;
                sources=(failing_source, healthy_source),
                http_get=(url; kwargs...) -> begin
                    push!(calls, url)
                    url == failing_source.url && error("synthetic transport outage")
                    response
                end,
                utc_clock=utc_clock,
                monotonic_clock=monotonic_clock,
            )
            @test calls == [failing_source.url, healthy_source.url]
            @test length(records) == 2
            @test records[1].capture_outcome == "transport_error"
            @test records[1].http_status == 0
            @test occursin("ErrorException", records[1].transport_error_type)
            @test occursin("synthetic transport outage", records[1].transport_error_message)
            @test isempty(records[1].raw_relative_path)
            @test records[2].capture_outcome == "http_response"
            verification = L1C.verify_v2_2_l1_receipts(
                root; sources=(failing_source, healthy_source),
            )
            @test getproperty.(verification, :records) == [1, 1]

            recovered = _v22_l1_capture_at(
                root, failing_source, response,
                DateTime(2022, 6, 1, 0, 1), 30,
            )
            @test recovered.sequence == 2
            @test recovered.previous_record_sha256 == records[1].record_sha256
            @test getproperty.(L1C.verify_v2_2_l1_receipts(
                root; sources=(failing_source, healthy_source),
            ), :records) == [2, 1]
        end
    end

    @testset "strict JSON diagnostics preserve invalid raw bytes" begin
        for (index, invalid_body) in enumerate(("[NaN]", "[Infinity]"))
            mktempdir() do root
                invalid_response = _v22_l1_fake_response(200, invalid_body)
                record = _v22_l1_capture_at(
                    root, source, invalid_response,
                    DateTime(2022, 7, index), 100 * index,
                )
                @test !record.json_valid
                @test !record.array_valid
                @test record.row_count == 0
                @test read(joinpath(root, record.raw_relative_path), String) == invalid_body
                @test only(L1C.verify_v2_2_l1_receipts(
                    root; sources=(source,),
                )).records == 1
            end
        end
    end
end
