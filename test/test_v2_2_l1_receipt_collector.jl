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

function _v22_l1_rehash_metadata!(root, source, record, changes)
    path = joinpath(root, record.record_relative_path)
    stored = JSON3.read(read(path, String))
    payload = L1C._v22_l1_record_payload(stored)
    mutated_metadata = merge(payload.metadata_provenance, changes)
    mutated_payload = merge(
        payload, (metadata_provenance=mutated_metadata,),
    )
    mutated_sha = L1C._v22_l1_record_sha256(mutated_payload)
    write(path, JSON3.write(merge(
        mutated_payload, (record_sha256=mutated_sha,),
    )))
    _v22_l1_test_latest!(
        root, source, record.sequence, record.record_relative_path, mutated_sha,
    )
    return mutated_sha
end

@testset "V2.2 prospective L1 receipt collector" begin
    source = (
        name="test_swpc_mag",
        url=L1C.V22_L1_RECEIPT_SOURCES[1].url,
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

        metadata = only(first_capture).metadata_provenance
        @test metadata.metadata_contract_version ==
              L1C.V22_L1_METADATA_CONTRACT_VERSION
        @test metadata.identity_authority_url ==
              L1C.V22_L1_NOAA_METADATA_AUTHORITY_URL
        @test metadata.source_field_semantics ==
              L1C.V22_L1_SOURCE_FIELD_SEMANTICS
        @test metadata.source_tokens == ["DSCOVR"]
        @test metadata.source_rows == 1
        @test metadata.identity_status == "bound_noaa_source_field"
        @test metadata.active_field_semantics ==
              L1C.V22_L1_ACTIVE_FIELD_SEMANTICS
        @test metadata.active_source_tokens == ["DSCOVR"]
        @test metadata.active_boolean_rows == 1
        @test metadata.active_status == "bound_noaa_active_field"
        @test metadata.quality_binding_status ==
              "missing_noaa_overall_quality"
        @test metadata.quality_semantics == L1C.V22_L1_QUALITY_SEMANTICS
        @test metadata.archive_quality_semantics ==
              L1C.V22_L1_ARCHIVE_QUALITY_SEMANTICS
        @test metadata.archive_quality_transfer_status ==
              "not_bound_to_swpc_rows"
        @test metadata.ephemeris_binding_status ==
              "missing_bound_ephemeris_record"
        @test metadata.ephemeris_record_sha256 ==
              L1C.V22_L1_RECEIPT_ZERO_SHA256
        @test isempty(metadata.ephemeris_record_json)
        @test !metadata.rows_admissible
        @test metadata.admissibility_blockers == [
            "missing_or_non_normal_row_quality",
            "missing_bound_ephemeris_record",
        ]

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
            equivalent_url = replace(
                source.url,
                "services.swpc.noaa.gov" => "SERVICES.SWPC.NOAA.GOV:443",
            )
            @test_throws ArgumentError L1C.capture_v2_2_l1_receipts!(
                root;
                sources=((name="first_name", url=source.url),
                         (name="second_name", url=equivalent_url)),
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

    @testset "verification rejects an existing duplicate-URL archive" begin
        mktempdir() do root
            first_source = (name="first_name", url=source.url)
            second_source = (name="second_name", url=source.url)
            _v22_l1_capture_at(
                root, first_source, response, DateTime(2022, 1, 2), 100,
            )
            @test_throws ArgumentError _v22_l1_capture_at(
                root, second_source, response, DateTime(2022, 1, 2), 200,
            )
            # Bypass the public gate to model an archive created by the prior
            # collector implementation; public verification must still reject it.
            L1C._v22_l1_install_record!(
                realpath(root), second_source, response,
                DateTime(2022, 1, 2),
                DateTime(2022, 1, 2) + Millisecond(1),
                200, 201,
            )
            @test_throws ArgumentError L1C.verify_v2_2_l1_receipts(
                root; sources=(first_source, second_source),
            )
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
            @test old_record.metadata_provenance.identity_status ==
                  "untrusted_non_noaa_rtsw_endpoint"
            @test !old_record.metadata_provenance.rows_admissible
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

    @testset "authoritative row metadata stays fail closed" begin
        mktempdir() do root
            mixed_body = """[
                {"time_tag":"2022-05-02T00:00:00Z","source":"DSCOVR","active":true},
                {"time_tag":"2022-05-02T00:01:00Z","source":"ACE","active":false}
            ]"""
            record = _v22_l1_capture_at(
                root, source, _v22_l1_fake_response(200, mixed_body),
                DateTime(2022, 5, 2), 200,
            )
            metadata = record.metadata_provenance
            # Exact sorted tokens independently follow the two raw `source` values.
            @test metadata.source_tokens == ["ACE", "DSCOVR"]
            @test metadata.source_rows == 2
            @test metadata.identity_status == "bound_noaa_source_field"
            # Only the literal Boolean-true DSCOVR row has the NOAA active designation.
            @test metadata.active_source_tokens == ["DSCOVR"]
            @test metadata.active_boolean_rows == 2
            @test metadata.active_status == "bound_noaa_active_field"
            @test !metadata.rows_admissible
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end

        mktempdir() do root
            unbound_body = """[
                {"time_tag":"2022-05-03T00:00:00Z","source":"DSCOVR",
                 "active":"true","quality":0,"position_x":1.0},
                {"time_tag":"2022-05-03T00:01:00Z","active":false}
            ]"""
            record = _v22_l1_capture_at(
                root, source, _v22_l1_fake_response(200, unbound_body),
                DateTime(2022, 5, 3), 300,
            )
            metadata = record.metadata_provenance
            # One absent source makes identity partial; no token is invented.
            @test metadata.source_tokens == ["DSCOVR"]
            @test metadata.source_rows == 1
            @test metadata.identity_status == "partial_noaa_source_field"
            @test "spacecraft_identity_not_fully_bound" in
                  metadata.admissibility_blockers
            # A string lookalike must not be promoted to NOAA's documented Boolean field.
            @test metadata.active_boolean_rows == 1
            @test isempty(metadata.active_source_tokens)
            @test metadata.active_status == "partial_noaa_active_field"
            @test "active_designation_not_fully_bound" in
                  metadata.admissibility_blockers
            # Undocumented lookalike quality/position fields confer no provenance.
            @test metadata.quality_binding_status ==
                  "missing_documented_per_row_quality"
            @test metadata.ephemeris_binding_status ==
                  "missing_bound_ephemeris_record"
            @test isempty(metadata.ephemeris_record_json)
            @test !metadata.rows_admissible
        end

        mktempdir() do root
            record = _v22_l1_capture_at(
                root, source, response, DateTime(2022, 5, 4), 400,
            )
            path = joinpath(root, record.record_relative_path)
            stored = JSON3.read(read(path, String))
            payload = L1C._v22_l1_record_payload(stored)
            forged_metadata = merge(payload.metadata_provenance, (
                quality_binding_status="bound",
                ephemeris_binding_status="bound",
                ephemeris_record_sha256=repeat("a", 64),
                ephemeris_record_json="{}",
                rows_admissible=true,
                admissibility_blockers=String[],
            ))
            forged_payload = merge(
                payload, (metadata_provenance=forged_metadata,),
            )
            forged_sha = L1C._v22_l1_record_sha256(forged_payload)
            write(path, JSON3.write(merge(
                forged_payload, (record_sha256=forged_sha,),
            )))
            _v22_l1_test_latest!(
                root, source, 1, record.record_relative_path, forged_sha,
            )
            # Rehashing cannot make quality, position, or admissibility appear.
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
        end
    end

    @testset "issue-causal GSE ephemeris is archived and independently replayed" begin
        measurement_body = """[
            {"time_tag":"2022-05-05T00:29:00Z","source":"DSCOVR","active":true,
             "overall_quality":0,"bx_gsm":1.0,"by_gsm":-2.0,"bz_gsm":-4.0},
            {"time_tag":"2022-05-05T00:30:00Z","source":"DSCOVR","active":true,
             "overall_quality":0,"bx_gsm":2.0,"by_gsm":-3.0,"bz_gsm":-5.0}
        ]"""
        measurement_response = _v22_l1_fake_response(200, measurement_body)
        ephemeris_body = """[
            {"time_tag":"2022-05-05T00:00:00Z","source":"DSCOVR","active":true,
             "x_gse":10.0,"y_gse":20.0,"z_gse":30.0},
            {"time_tag":"2022-05-05T01:00:00Z","source":"DSCOVR","active":true,
             "x_gse":20.0,"y_gse":40.0,"z_gse":60.0}
        ]"""
        ephemeris_response = _v22_l1_fake_response(
            200, ephemeris_body;
            headers=[
                "Date" => "Thu, 05 May 2022 00:30:30 GMT",
                "ETag" => "\"ephemeris-frozen\"",
                "Last-Modified" => "Thu, 05 May 2022 00:30:00 GMT",
            ],
        )

        function capture_bound_ephemeris(root; ephemeris=ephemeris_response,
                                         measurement=measurement_response,
                                         capture_source=source)
            base = DateTime(2022, 5, 5, 0, 30, 30)
            return only(L1C.capture_v2_2_l1_receipts!(
                root;
                sources=(capture_source,),
                http_get=(url; kwargs...) -> begin
                    @test url == capture_source.url
                    measurement
                end,
                ephemeris_http_get=(url; kwargs...) -> begin
                    @test url == L1C.V22_L1_NOAA_EPHEMERIS_URL
                    ephemeris
                end,
                utc_clock=_v22_l1_sequence_clock([
                    base,
                    base + Millisecond(10),
                    base + Second(1),
                    base + Second(1) + Millisecond(10),
                ]),
                monotonic_clock=_v22_l1_sequence_clock([100, 110, 120, 130]),
            ))
        end

        mktempdir() do root
            record = capture_bound_ephemeris(root)
            metadata = record.metadata_provenance
            @test metadata.ephemeris_capture_outcome == "http_response"
            @test metadata.ephemeris_http_status == 200
            @test metadata.ephemeris_http_etag == "\"ephemeris-frozen\""
            @test metadata.ephemeris_receipt_completed_utc ==
                  "2022-05-05T00:30:30.010Z"
            @test metadata.ephemeris_source_available_before_issue
            @test metadata.ephemeris_binding_status ==
                  "bound_issue_causal_swpc_ephemeris"
            @test metadata.ephemeris_position_timestamp_utc ==
                  "2022-05-05T00:30:00.000Z"
            @test metadata.ephemeris_position_frame == "GSE"
            @test metadata.ephemeris_position_units == "km"
            @test metadata.ephemeris_interpolation_rule ==
                  L1C.V22_L1_EPHEMERIS_INTERPOLATION_RULE
            @test metadata.ephemeris_source_object_sha256 ==
                  L1C._v22_l1_sha256(codeunits(ephemeris_body))
            @test isfile(joinpath(
                root, metadata.ephemeris_source_object_raw_relative_path,
            ))
            position = JSON3.read(metadata.ephemeris_record_json)
            @test position.source == "DSCOVR"
            @test position.method == "linear"
            @test position.interpolation_fraction == 0.5
            @test position.x_gse == 15.0
            @test position.y_gse == 30.0
            @test position.z_gse == 45.0
            @test metadata.ephemeris_record_sha256 ==
                  L1C._v22_l1_sha256(codeunits(metadata.ephemeris_record_json))
            @test metadata.quality_authority_url ==
                  L1C.V22_L1_NOAA_MAG_QUALITY_SCHEMA_URL
            @test metadata.quality_source_product == "dscovr_m1m"
            @test metadata.quality_row_timestamp_utc ==
                  "2022-05-05T00:30:00.000Z"
            @test metadata.quality_row_source == "DSCOVR"
            @test metadata.quality_value == 0
            @test metadata.quality_binding_status ==
                  "bound_noaa_dscovr_overall_quality"
            @test metadata.quality_required_fields_status ==
                  "bound_required_bx_by_bz_gsm"
            @test metadata.quality_decision ==
                  "accept_normal_overall_quality"
            @test metadata.rows_admissible
            @test isempty(metadata.admissibility_blockers)
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1

            ephemeris_path = joinpath(
                root, metadata.ephemeris_source_object_raw_relative_path,
            )
            open(ephemeris_path, "a") do io
                write(io, UInt8('x'))
            end
            @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )
        end

        exact_measurement = _v22_l1_fake_response(
            200,
            """[{"time_tag":"2022-05-05T00:00:00Z",\
                   "source":"DSCOVR","active":true,\
                   "overall_quality":0,"bx_gsm":1.0,"by_gsm":-2.0,
                   "bz_gsm":-5.0}]""",
        )
        mktempdir() do root
            metadata = capture_bound_ephemeris(
                root; measurement=exact_measurement,
            ).metadata_provenance
            position = JSON3.read(metadata.ephemeris_record_json)
            @test position.method == "exact"
            @test position.lower_time_utc == position.upper_time_utc ==
                  "2022-05-05T00:00:00.000Z"
            @test position.x_gse == 10.0
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end

        @testset "NOAA DSCOVR row-quality decisions fail closed" begin
            rejected = (
                (
                    body="""[{"time_tag":"2022-05-05T00:30:00Z",\
                               "source":"DSCOVR","active":true,\
                               "overall_quality":1,"bz_gsm":-5.0}]""",
                    status="bound_noaa_dscovr_overall_quality",
                    decision="reject_suspect_overall_quality",
                ),
                (
                    body="""[{"time_tag":"2022-05-05T00:30:00Z",\
                               "source":"DSCOVR","active":true,\
                               "overall_quality":2,"bz_gsm":-5.0}]""",
                    status="bound_noaa_dscovr_overall_quality",
                    decision="reject_error_overall_quality",
                ),
                (
                    body="""[{"time_tag":"2022-05-05T00:30:00Z",\
                               "source":"DSCOVR","active":true,\
                               "bz_gsm":-5.0}]""",
                    status="missing_noaa_overall_quality",
                    decision="reject_missing_noaa_overall_quality",
                ),
                (
                    body="""[{"time_tag":"2022-05-05T00:30:00Z",\
                               "source":"DSCOVR","active":true,\
                               "overall_quality":0,"by_gsm":-2.0,
                               "bz_gsm":-5.0}]""",
                    status="bound_noaa_dscovr_overall_quality",
                    decision="reject_missing_or_invalid_required_forecast_fields",
                ),
                (
                    body="""[{"time_tag":"2022-05-05T00:30:00Z",\
                               "source":"ACE","active":true,\
                               "overall_quality":0,"bz_gsm":-5.0}]""",
                    status="unverified_non_dscovr_quality_semantics",
                    decision="reject_unverified_source_quality_semantics",
                ),
            )
            for case in rejected
                mktempdir() do root
                    response = _v22_l1_fake_response(200, case.body)
                    metadata = capture_bound_ephemeris(
                        root; measurement=response,
                    ).metadata_provenance
                    @test metadata.quality_binding_status == case.status
                    @test metadata.quality_decision == case.decision
                    @test !metadata.rows_admissible
                    @test "missing_or_non_normal_row_quality" in
                          metadata.admissibility_blockers
                    @test only(L1C.verify_v2_2_l1_receipts(
                        root; sources=(source,),
                    )).records == 1
                end
            end

            invalid_quality_values = ("true", "0.5", "3", "1e300", "\"0\"")
            for value in invalid_quality_values
                mktempdir() do root
                    body = """[{"time_tag":"2022-05-05T00:30:00Z",\
                                  "source":"DSCOVR","active":true,\
                                  "overall_quality":$value,"bz_gsm":-5.0}]"""
                    response = _v22_l1_fake_response(200, body)
                    record = capture_bound_ephemeris(
                        root; measurement=response,
                    )
                    metadata = record.metadata_provenance
                    @test metadata.quality_binding_status ==
                          "invalid_noaa_overall_quality"
                    @test metadata.quality_decision ==
                          "reject_invalid_noaa_overall_quality"
                    @test !metadata.rows_admissible
                    @test read(joinpath(root, record.raw_relative_path), String) == body
                    @test only(L1C.verify_v2_2_l1_receipts(
                        root; sources=(source,),
                    )).records == 1
                end
            end

            wind_source = (
                name="test_swpc_wind",
                url=L1C.V22_L1_RECEIPT_SOURCES[2].url,
            )
            wind_response = _v22_l1_fake_response(
                200,
                """[{"time_tag":"2022-05-05T00:30:00Z",\
                       "source":"DSCOVR","active":true,\
                       "overall_quality":0.0,"proton_speed":400.0,\
                       "proton_density":5.0,"proton_vx_gse":-395.0}]""",
            )
            mktempdir() do root
                metadata = capture_bound_ephemeris(
                    root; measurement=wind_response,
                    capture_source=wind_source,
                ).metadata_provenance
                @test metadata.quality_authority_url ==
                      L1C.V22_L1_NOAA_WIND_QUALITY_SCHEMA_URL
                @test metadata.quality_source_product == "dscovr_f1m"
                @test metadata.quality_value == 0
                @test metadata.quality_required_fields_status ==
                      "bound_required_speed_density_vx_gse"
                @test metadata.quality_decision ==
                      "accept_normal_overall_quality"
                @test metadata.rows_admissible
                @test isempty(metadata.admissibility_blockers)
                @test only(L1C.verify_v2_2_l1_receipts(
                    root; sources=(wind_source,),
                )).records == 1
            end

            invalid_wind_fields = (
                """[{"time_tag":"2022-05-05T00:30:00Z",\
                       "source":"DSCOVR","active":true,\
                       "overall_quality":0,"proton_speed":400.0,\
                       "proton_vx_gse":-395.0}]""",
                """[{"time_tag":"2022-05-05T00:30:00Z",\
                       "source":"DSCOVR","active":true,\
                       "overall_quality":0,"proton_speed":400.0,\
                       "proton_density":5.0}]""",
            )
            for body in invalid_wind_fields
                mktempdir() do root
                    metadata = capture_bound_ephemeris(
                        root; measurement=_v22_l1_fake_response(200, body),
                        capture_source=wind_source,
                    ).metadata_provenance
                    @test metadata.quality_binding_status ==
                          "bound_noaa_dscovr_overall_quality"
                    @test metadata.quality_required_fields_status ==
                          "missing_or_invalid_required_speed_density_vx_gse"
                    @test metadata.quality_decision ==
                          "reject_missing_or_invalid_required_forecast_fields"
                    @test !metadata.rows_admissible
                    @test only(L1C.verify_v2_2_l1_receipts(
                        root; sources=(wind_source,),
                    )).records == 1
                end
            end
        end

        wide_gap_response = _v22_l1_fake_response(
            200,
            """[
                {"time_tag":"2022-05-05T00:00:00Z","source":"DSCOVR",\
                 "active":true,"x_gse":10.0,"y_gse":20.0,"z_gse":30.0},
                {"time_tag":"2022-05-05T02:00:00Z","source":"DSCOVR",\
                 "active":true,"x_gse":20.0,"y_gse":40.0,"z_gse":60.0}
            ]""",
        )
        mktempdir() do root
            metadata = capture_bound_ephemeris(
                root; ephemeris=wide_gap_response,
            ).metadata_provenance
            @test metadata.ephemeris_binding_status ==
                  "ephemeris_bracket_exceeds_one_hour"
            @test isempty(metadata.ephemeris_record_json)
            @test !metadata.rows_admissible
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end

        sentinel_response = _v22_l1_fake_response(
            200,
            """[
                {"time_tag":"2022-05-05T00:00:00Z","source":"DSCOVR",\
                 "active":true,"x_gse":-99999.0,"y_gse":20.0,"z_gse":30.0},
                {"time_tag":"2022-05-05T01:00:00Z","source":"DSCOVR",\
                 "active":true,"x_gse":20.0,"y_gse":40.0,"z_gse":60.0}
            ]""",
        )
        mktempdir() do root
            metadata = capture_bound_ephemeris(
                root; ephemeris=sentinel_response,
            ).metadata_provenance
            @test metadata.ephemeris_binding_status ==
                  "ephemeris_extrapolation_required"
            @test isempty(metadata.ephemeris_record_json)
            @test metadata.ephemeris_source_available_before_issue
            @test !metadata.rows_admissible
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end

        mktempdir() do root
            base = DateTime(2022, 5, 5, 0, 30, 30)
            record = only(L1C.capture_v2_2_l1_receipts!(
                root;
                sources=(source,),
                http_get=(url; kwargs...) -> measurement_response,
                ephemeris_http_get=(url; kwargs...) ->
                    error("synthetic ephemeris transport outage"),
                utc_clock=_v22_l1_sequence_clock([
                    base,
                    base + Millisecond(10),
                    base + Second(1),
                    base + Second(1) + Millisecond(10),
                ]),
                monotonic_clock=_v22_l1_sequence_clock([100, 110, 120, 130]),
            ))
            metadata = record.metadata_provenance
            @test metadata.ephemeris_capture_outcome == "transport_error"
            @test occursin(
                "synthetic ephemeris transport outage",
                metadata.ephemeris_transport_error_message,
            )
            @test metadata.ephemeris_binding_status ==
                  "ephemeris_transport_error"
            @test metadata.ephemeris_source_object_sha256 ==
                  L1C.V22_L1_RECEIPT_ZERO_SHA256
            @test isempty(metadata.ephemeris_source_object_raw_relative_path)
            @test !metadata.rows_admissible
            @test only(L1C.verify_v2_2_l1_receipts(
                root; sources=(source,),
            )).records == 1
        end

        mutations = (
            (ephemeris_position_units="m",),
            (ephemeris_interpolation_rule="nearest neighbor",),
            (quality_decision="accept_active_and_finite",),
            (quality_authority_url=L1C.V22_L1_NOAA_QUALITY_AUTHORITY_URL,),
            (quality_row_timestamp_utc="2022-05-05T00:29:00.000Z",),
            (quality_row_source="ACE",),
            (quality_value=1,),
            (quality_required_fields_status="unverified",),
            (ephemeris_receipt_completed_utc="2022-05-05T00:30:31.005Z",),
        )
        for mutation in mutations
            mktempdir() do root
                record = capture_bound_ephemeris(root)
                _v22_l1_rehash_metadata!(root, source, record, mutation)
                @test_throws ErrorException L1C.verify_v2_2_l1_receipts(
                    root; sources=(source,),
                )
            end
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
            @test records[1].metadata_provenance.identity_status ==
                  "unavailable_no_http_response"
            @test records[1].metadata_provenance.active_status ==
                  "unavailable_no_http_response"
            @test !records[1].metadata_provenance.rows_admissible
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
                @test record.metadata_provenance.identity_status ==
                      "missing_noaa_source_field"
                @test !record.metadata_provenance.rows_admissible
                @test read(joinpath(root, record.raw_relative_path), String) == invalid_body
                @test only(L1C.verify_v2_2_l1_receipts(
                    root; sources=(source,),
                )).records == 1
            end
        end
    end
end
