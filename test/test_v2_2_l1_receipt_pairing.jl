using Test
using Dates
using JSON3
using SHA

include(joinpath(@__DIR__, "..", "examples", "v2_2_l1_receipt_pairing.jl"))
const L1P = V22L1ReceiptPairing

_v22_pair_response(body::AbstractString) = (
    status=200,
    headers=Pair{String,String}[],
    body=Vector{UInt8}(codeunits(body)),
)

function _v22_pair_clock(values)
    remaining = collect(values)
    return () -> begin
        isempty(remaining) && error("pairing test clock exhausted")
        popfirst!(remaining)
    end
end

function _v22_pair_mag(time_tag::AbstractString; bx=1.0, by=2.0, bz=-5.0,
                       quality=0, source="DSCOVR", active=true)
    return _v22_pair_response(JSON3.write([(
        time_tag=String(time_tag), source=String(source), active=active,
        overall_quality=quality, bx_gsm=bx, by_gsm=by, bz_gsm=bz,
    )]))
end

function _v22_pair_wind(time_tag::AbstractString; speed=400.0, density=5.0,
                        vx=-395.0, quality=0, source="DSCOVR", active=true)
    return _v22_pair_response(JSON3.write([(
        time_tag=String(time_tag), source=String(source), active=active,
        overall_quality=quality, proton_speed=speed, proton_density=density,
        proton_vx_gse=vx,
    )]))
end

function _v22_pair_ephemeris(; lower=(10.0, 20.0, 30.0),
                             upper=(20.0, 40.0, 60.0))
    return _v22_pair_response(JSON3.write([
        (
            time_tag="2022-05-05T00:00:00Z", source="DSCOVR", active=true,
            x_gse=lower[1], y_gse=lower[2], z_gse=lower[3],
        ),
        (
            time_tag="2022-05-05T01:00:00Z", source="DSCOVR", active=true,
            x_gse=upper[1], y_gse=upper[2], z_gse=upper[3],
        ),
    ]))
end

function _v22_capture_pair!(root, base::DateTime, mag, wind;
                            ephemeris=_v22_pair_ephemeris())
    sources = L1P.V22_L1_RECEIPT_SOURCES
    utc = [
        base,
        base + Millisecond(1),
        base + Millisecond(10),
        base + Millisecond(11),
        base + Millisecond(20),
        base + Millisecond(21),
    ]
    monotonic = [100, 101, 110, 111, 120, 121] .+
                Dates.value(base - DateTime(2022, 5, 5))
    return L1P.capture_v2_2_l1_receipts!(
        root;
        sources=sources,
        http_get=(url; kwargs...) -> begin
            url == sources[1].url && return mag
            url == sources[2].url && return wind
            error("unexpected synthetic measurement URL")
        end,
        ephemeris_http_get=(url; kwargs...) -> begin
            @test url == L1P.V22_L1_NOAA_EPHEMERIS_URL
            ephemeris
        end,
        utc_clock=_v22_pair_clock(utc),
        monotonic_clock=_v22_pair_clock(monotonic),
    )
end

function _v22_capture_one!(root, base::DateTime, source, response, ephemeris)
    return only(L1P.capture_v2_2_l1_receipts!(
        root;
        sources=(source,),
        http_get=(url; kwargs...) -> begin
            @test url == source.url
            response
        end,
        ephemeris_http_get=(url; kwargs...) -> ephemeris,
        utc_clock=_v22_pair_clock([
            base, base + Millisecond(1),
            base + Millisecond(10), base + Millisecond(11),
        ]),
        monotonic_clock=_v22_pair_clock([100, 101, 110, 111] .+
                                       Dates.value(base - DateTime(2022, 5, 5))),
    ))
end

function _v22_pair_tree_hashes(root)
    hashes = Dict{String,String}()
    for (directory, _, names) in walkdir(root)
        for name in names
            path = joinpath(directory, name)
            isfile(path) || continue
            hashes[relpath(path, root)] = bytes2hex(sha256(read(path)))
        end
    end
    return hashes
end

function _v22_pair_rehash_latest!(root, source; record_changes=NamedTuple(),
                                  metadata_changes=NamedTuple())
    latest_path = joinpath(root, "latest", source.name * ".json")
    latest = JSON3.read(read(latest_path, String))
    record_path = joinpath(root, String(latest.record_relative_path))
    record = JSON3.read(read(record_path, String))
    payload = L1P._v22_l1_record_payload(record)
    metadata = merge(payload.metadata_provenance, metadata_changes)
    mutated = merge(payload, record_changes, (metadata_provenance=metadata,))
    checksum = L1P._v22_l1_record_sha256(mutated)
    write(record_path, JSON3.write(merge(mutated, (record_sha256=checksum,))))
    write(latest_path, JSON3.write((
        schema_version=L1P.V22_L1_RECEIPT_SCHEMA_VERSION,
        source_name=String(source.name),
        sequence=Int(latest.sequence),
        record_relative_path=String(latest.record_relative_path),
        record_sha256=checksum,
    )))
    return nothing
end

@testset verbose=true "V2.2 offline L1 issue pairing" begin
    sources = L1P.V22_L1_RECEIPT_SOURCES
    measurement_time = "2022-05-05T00:30:00Z"

    @testset "hand-derived exact-time pair and immutable provenance" begin
        mktempdir() do root
            mag_response = _v22_pair_mag(
                measurement_time; bx=1.5, by=-2.5, bz=-7.5,
            )
            wind_response = _v22_pair_wind(
                measurement_time; speed=420.0, density=6.0, vx=-415.0,
            )
            records = _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                mag_response, wind_response,
            )
            before = _v22_pair_tree_hashes(root)
            pair = L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 30, 11),
            )
            @test pair.schema_version == L1P.V22_L1_ISSUE_PAIR_SCHEMA_VERSION
            @test pair.issue_time_utc == "2022-05-05T00:30:11.000Z"
            @test pair.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test pair.source == "DSCOVR"
            # Independent raw-row values catch component swaps and sign changes.
            @test pair.bx_gsm == 1.5
            @test pair.by_gsm == -2.5
            @test pair.bz_gsm == -7.5
            @test pair.proton_speed == 420.0
            @test pair.proton_density == 6.0
            @test pair.proton_vx_gse == -415.0
            # Halfway interpolation between the two hand-written orbit rows.
            @test pair.position_x_gse == 15.0
            @test pair.position_y_gse == 30.0
            @test pair.position_z_gse == 45.0
            @test pair.position_frame == "GSE"
            @test pair.position_units == "km"
            @test pair.position_method == "linear"
            @test pair.position_interpolation_fraction == 0.5
            @test pair.mag_sequence == pair.wind_sequence == 1
            @test pair.mag_record_sha256 == records[1].record_sha256
            @test pair.wind_record_sha256 == records[2].record_sha256
            @test pair.mag_raw_sha256 == bytes2hex(sha256(mag_response.body))
            @test pair.wind_raw_sha256 == bytes2hex(sha256(wind_response.body))
            @test pair.ephemeris_record_sha256 ==
                  records[1].metadata_provenance.ephemeris_record_sha256 ==
                  records[2].metadata_provenance.ephemeris_record_sha256
            @test _v22_pair_tree_hashes(root) == before
        end
    end

    @testset "issue cutoff and latest exact common minute" begin
        mktempdir() do root
            first = _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time; bz=-5.0),
                _v22_pair_wind(measurement_time; speed=400.0),
            )
            later_mag = _v22_pair_response(JSON3.write([
                (
                    time_tag=measurement_time, source="DSCOVR", active=true,
                    overall_quality=0, bx_gsm=1.0, by_gsm=2.0, bz_gsm=-99.0,
                ),
                (
                    time_tag="2022-05-05T00:31:00Z", source="DSCOVR",
                    active=true, overall_quality=0, bx_gsm=1.0, by_gsm=2.0,
                    bz_gsm=-8.0,
                ),
            ]))
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 31, 10),
                later_mag,
                _v22_pair_wind("2022-05-05T00:31:00Z"; speed=500.0),
            )
            early = L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 30, 20),
            )
            @test early.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test early.mag_record_sha256 == first[1].record_sha256
            @test early.wind_record_sha256 == first[2].record_sha256
            late = L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 20),
            )
            @test late.measurement_time_utc == "2022-05-05T00:31:00.000Z"
            @test late.bz_gsm == -8.0
            @test late.proton_speed == 500.0
            @test late.mag_sequence == late.wind_sequence == 2
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time),
            )
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 32, 10),
                _v22_pair_mag("2022-05-05T00:31:00Z"),
                _v22_pair_wind("2022-05-05T00:32:00Z"),
            )
            pair = L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 32, 20),
            )
            # No interpolation or nearest-minute join: the older exact pair wins.
            @test pair.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test pair.mag_sequence == pair.wind_sequence == 1
        end
    end

    @testset "missing, unadmitted, and revised rows fail closed" begin
        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind("2022-05-05T00:31:00Z"),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time; quality=1),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            missing_vx = _v22_pair_response(JSON3.write([(
                time_tag=measurement_time, source="DSCOVR", active=true,
                overall_quality=0, proton_speed=400.0, proton_density=5.0,
            )]))
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time), missing_vx,
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            # Repeated identical rows are not revisions; the latest receipt wins.
            for second in (10, 20)
                _v22_capture_pair!(
                    root, DateTime(2022, 5, 5, 0, 30, second),
                    _v22_pair_mag(measurement_time),
                    _v22_pair_wind(measurement_time),
                )
            end
            pair = L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
            @test pair.mag_sequence == 2
            @test pair.wind_sequence == 2
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time; bz=-5.0),
                _v22_pair_wind(measurement_time),
            )
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 20),
                _v22_pair_mag(measurement_time; bz=-9.0),
                _v22_pair_wind(measurement_time),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time; quality=0),
            )
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 20),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time; quality=1),
            )
            # A later suspect revision revokes the earlier normal row.
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time; bz=-5.0),
                _v22_pair_wind(measurement_time),
            )
            revised_history = _v22_pair_response(JSON3.write([
                (
                    time_tag=measurement_time, source="DSCOVR", active=true,
                    overall_quality=0, bx_gsm=1.0, by_gsm=2.0, bz_gsm=-9.0,
                ),
                (
                    time_tag="2022-05-05T00:31:00Z", source="DSCOVR",
                    active=true, overall_quality=0, bx_gsm=1.0, by_gsm=2.0,
                    bz_gsm=-6.0,
                ),
            ]))
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 32, 10), revised_history,
                _v22_pair_wind("2022-05-05T00:32:00Z"),
            )
            # A revision cannot hide below a newer latest row in a later receipt.
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 33, 0),
            )
        end
    end

    @testset "receipt, metadata, raw, and orbit mutations fail closed" begin
        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time),
            )
            _v22_pair_rehash_latest!(
                root, sources[1];
                record_changes=(
                    receipt_completed_utc="2022-05-05T00:31:00.000Z",
                ),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 30, 30),
            )
        end

        mktempdir() do root
            _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time),
            )
            _v22_pair_rehash_latest!(
                root, sources[1];
                metadata_changes=(
                    quality_row_timestamp_utc="2022-05-05T00:29:00.000Z",
                ),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            records = _v22_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _v22_pair_mag(measurement_time),
                _v22_pair_wind(measurement_time),
            )
            open(joinpath(root, records[1].raw_relative_path), "a") do io
                write(io, UInt8('x'))
            end
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _v22_capture_one!(
                root, DateTime(2022, 5, 5, 0, 30, 10), sources[1],
                _v22_pair_mag(measurement_time),
                _v22_pair_ephemeris(),
            )
            _v22_capture_one!(
                root, DateTime(2022, 5, 5, 0, 30, 20), sources[2],
                _v22_pair_wind(measurement_time),
                _v22_pair_ephemeris(
                    lower=(11.0, 21.0, 31.0), upper=(21.0, 41.0, 61.0),
                ),
            )
            @test_throws ErrorException L1P.select_v2_2_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end
    end
end
