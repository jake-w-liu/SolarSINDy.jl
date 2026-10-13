using Test
using Dates
using JSON3
using SHA

include(joinpath(@__DIR__, "..", "examples", "l1_receipt_pairing.jl"))
const L1P = StaticStackL1ReceiptPairing

_static_stack_pair_response(body::AbstractString) = (
    status=200,
    headers=Pair{String,String}[],
    body=Vector{UInt8}(codeunits(body)),
)

function _static_stack_pair_clock(values)
    remaining = collect(values)
    return () -> begin
        isempty(remaining) && error("pairing test clock exhausted")
        popfirst!(remaining)
    end
end

function _static_stack_pair_mag(time_tag::AbstractString; bx=1.0, by=2.0, bz=-5.0,
                       quality=0, source="DSCOVR", active=true)
    return _static_stack_pair_response(JSON3.write([(
        time_tag=String(time_tag), source=String(source), active=active,
        overall_quality=quality, bx_gsm=bx, by_gsm=by, bz_gsm=bz,
    )]))
end

function _static_stack_pair_wind(time_tag::AbstractString; speed=400.0, density=5.0,
                        vx=-395.0, quality=0, source="DSCOVR", active=true)
    return _static_stack_pair_response(JSON3.write([(
        time_tag=String(time_tag), source=String(source), active=active,
        overall_quality=quality, proton_speed=speed, proton_density=density,
        proton_vx_gse=vx,
    )]))
end

function _static_stack_pair_ephemeris(; lower=(10.0, 20.0, 30.0),
                             upper=(20.0, 40.0, 60.0))
    return _static_stack_pair_response(JSON3.write([
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

function _static_stack_capture_pair!(root, base::DateTime, mag, wind;
                            ephemeris=_static_stack_pair_ephemeris())
    sources = L1P.L1_RECEIPT_SOURCES
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
    return L1P.capture_l1_receipts!(
        root;
        sources=sources,
        http_get=(url; kwargs...) -> begin
            url == sources[1].url && return mag
            url == sources[2].url && return wind
            error("unexpected synthetic measurement URL")
        end,
        ephemeris_http_get=(url; kwargs...) -> begin
            @test url == L1P.L1_NOAA_EPHEMERIS_URL
            ephemeris
        end,
        utc_clock=_static_stack_pair_clock(utc),
        monotonic_clock=_static_stack_pair_clock(monotonic),
    )
end

function _static_stack_capture_one!(root, base::DateTime, source, response, ephemeris)
    return only(L1P.capture_l1_receipts!(
        root;
        sources=(source,),
        http_get=(url; kwargs...) -> begin
            @test url == source.url
            response
        end,
        ephemeris_http_get=(url; kwargs...) -> ephemeris,
        utc_clock=_static_stack_pair_clock([
            base, base + Millisecond(1),
            base + Millisecond(10), base + Millisecond(11),
        ]),
        monotonic_clock=_static_stack_pair_clock([100, 101, 110, 111] .+
                                       Dates.value(base - DateTime(2022, 5, 5))),
    ))
end

function _static_stack_pair_tree_hashes(root)
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

function _static_stack_pair_contract_sha(pair)
    names = Tuple(filter(!=(:pair_contract_sha256), propertynames(pair)))
    payload = NamedTuple{names}(Tuple(getproperty(pair, name) for name in names))
    return bytes2hex(sha256(codeunits(JSON3.write(payload))))
end

function _static_stack_pair_rehash_latest!(root, source; record_changes=NamedTuple(),
                                  metadata_changes=NamedTuple())
    latest_path = joinpath(root, "latest", source.name * ".json")
    latest = JSON3.read(read(latest_path, String))
    record_path = joinpath(root, String(latest.record_relative_path))
    record = JSON3.read(read(record_path, String))
    payload = L1P._l1_record_payload(record)
    metadata = merge(payload.metadata_provenance, metadata_changes)
    mutated = merge(payload, record_changes, (metadata_provenance=metadata,))
    checksum = L1P._l1_record_sha256(mutated)
    write(record_path, JSON3.write(merge(mutated, (record_sha256=checksum,))))
    write(latest_path, JSON3.write((
        schema_version=L1P.L1_RECEIPT_SCHEMA_VERSION,
        source_name=String(source.name),
        sequence=Int(latest.sequence),
        record_relative_path=String(latest.record_relative_path),
        record_sha256=checksum,
    )))
    return nothing
end

@testset verbose=true "static stack offline L1 issue pairing" begin
    sources = L1P.L1_RECEIPT_SOURCES
    measurement_time = "2022-05-05T00:30:00Z"

    @testset "hand-derived exact-time pair and immutable provenance" begin
        mktempdir() do root
            mag_response = _static_stack_pair_mag(
                measurement_time; bx=1.5, by=-2.5, bz=-7.5,
            )
            wind_response = _static_stack_pair_wind(
                measurement_time; speed=420.0, density=6.0, vx=-415.0,
            )
            records = _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                mag_response, wind_response,
            )
            issue = DateTime(2022, 5, 5, 0, 30, 11)
            cutoff = L1P.capture_l1_issue_cutoff!(root, issue)
            before = _static_stack_pair_tree_hashes(root)
            pair = L1P.select_l1_issue_pair(
                root, issue, cutoff.cutoff_relative_path,
            )
            @test pair.schema_version == L1P.L1_ISSUE_PAIR_SCHEMA_VERSION
            @test pair.issue_time_utc == "2022-05-05T00:30:11.000Z"
            @test pair.first_eligible_issue_time_utc ==
                  "2022-05-05T01:00:00.000Z"
            @test pair.issue_cutoff_relative_path == cutoff.cutoff_relative_path
            @test pair.issue_cutoff_sha256 == cutoff.cutoff_sha256
            @test pair.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test pair.source == "DSCOVR"
            @test pair.mag_source_product_id == "swpc_rtsw_mag_1m"
            @test pair.wind_source_product_id == "swpc_rtsw_wind_1m"
            @test pair.magnetic_component_frame == "GSM"
            @test pair.magnetic_component_units == "nT"
            @test pair.proton_speed_units == "km/s"
            @test pair.proton_density_units == "cm^-3"
            @test pair.proton_vx_frame == "GSE"
            @test pair.proton_vx_units == "km/s"
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
            @test pair.mag_quality_source_product == "dscovr_m1m"
            @test pair.wind_quality_source_product == "dscovr_f1m"
            @test pair.mag_quality_value == pair.wind_quality_value == 0
            @test pair.mag_quality_binding_status ==
                  pair.wind_quality_binding_status ==
                  "bound_noaa_rtsw_overall_quality"
            @test pair.mag_quality_decision == pair.wind_quality_decision ==
                  "accept_normal_overall_quality"
            @test pair.mag_quality_required_fields_status ==
                  "bound_required_bx_by_bz_gsm"
            @test pair.wind_quality_required_fields_status ==
                  "bound_required_speed_density_vx_gse"
            @test pair.pair_contract_sha256 == _static_stack_pair_contract_sha(pair)
            @test !ispath(joinpath(root, ".collector.lock"))
            @test _static_stack_pair_tree_hashes(root) == before
        end
    end

    @testset "live multi-source SWPC rows pair the active stream (G11-01/02/04)" begin
        # Rows copied from the 2026-10-12 live RTSW feeds: no trailing Z,
        # inactive ACE in the same minute, newer inactive IMAP. Active source
        # renamed SOLAR1 -> DSCOVR (DSCOVR-only contract) and shifted from
        # 09:24 to 08:24 so the published hourly ephemeris brackets it.
        live_mag = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T08:26:01", "active": false, "source": "IMAP",
             "bx_gsm": 5.9, "by_gsm": -0.59, "bz_gsm": -0.05, "overall_quality": 0},
            {"time_tag": "2026-10-12T08:24:01", "active": false, "source": "IMAP",
             "bx_gsm": 5.86, "by_gsm": -0.82, "bz_gsm": -0.2, "overall_quality": 0},
            {"time_tag": "2026-10-12T08:24:00", "active": true, "source": "DSCOVR",
             "bx_gsm": 6.34, "by_gsm": -0.72, "bz_gsm": 0.28, "overall_quality": 0},
            {"time_tag": "2026-10-12T08:24:00", "active": false, "source": "ACE",
             "bx_gsm": 5.75, "by_gsm": -0.42, "bz_gsm": 0.01, "overall_quality": 0}
        ]""")
        live_wind = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T08:24:00", "active": true, "source": "DSCOVR",
             "proton_speed": 458.4, "proton_density": 0.41,
             "proton_vx_gse": -457.1, "overall_quality": 0},
            {"time_tag": "2026-10-12T08:24:00", "active": false, "source": "ACE",
             "proton_speed": 460.88, "proton_density": 0.0,
             "proton_vx_gse": null, "overall_quality": 0}
        ]""")
        # Includes the real duplicated active 05:00 hour, unrelated to the bracket.
        live_ephemeris = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T09:00:00", "active": false, "source": "ACE",
             "x_gse": 1415916, "y_gse": 57402, "z_gse": 51024},
            {"time_tag": "2026-10-12T09:00:00", "active": true, "source": "DSCOVR",
             "x_gse": 1576421, "y_gse": -113113, "z_gse": 88675},
            {"time_tag": "2026-10-12T08:00:00", "active": true, "source": "DSCOVR",
             "x_gse": 1576509, "y_gse": -112727, "z_gse": 88879},
            {"time_tag": "2026-10-12T05:00:00", "active": true, "source": "DSCOVR",
             "x_gse": 1576770, "y_gse": -111565, "z_gse": 89492},
            {"time_tag": "2026-10-12T05:00:00", "active": true, "source": "DSCOVR",
             "x_gse": 1576770, "y_gse": -111564, "z_gse": 89492}
        ]""")
        mktempdir() do root
            records = _static_stack_capture_pair!(
                root, DateTime(2026, 10, 12, 8, 30, 10),
                live_mag, live_wind; ephemeris=live_ephemeris,
            )
            for record in records
                @test record.metadata_provenance.rows_admissible
                @test record.metadata_provenance.quality_row_timestamp_utc ==
                      "2026-10-12T08:24:00.000Z"
            end
            issue = DateTime(2026, 10, 12, 8, 30, 11)
            cutoff = L1P.capture_l1_issue_cutoff!(root, issue)
            pair = L1P.select_l1_issue_pair(
                root, issue, cutoff.cutoff_relative_path,
            )
            @test pair.measurement_time_utc == "2026-10-12T08:24:00.000Z"
            @test pair.source == "DSCOVR"
            @test pair.bz_gsm == 0.28
            @test pair.proton_speed == 458.4
            @test pair.proton_vx_gse == -457.1
            @test pair.position_method == "linear"
            @test pair.position_x_gse ≈ 1576509 + 0.4 * (1576421 - 1576509)
        end

        observations = L1P._l1_row_observations(
            JSON3.read(String(copy(live_mag.body))),
        )
        @test all(item -> item.status == "unique_measurement_row", observations)
        dscovr = filter(item -> item.source == "DSCOVR", observations)
        @test only(dscovr).timestamp == DateTime(2026, 10, 12, 8, 24)
        # Two rows from the same source in one minute remain ambiguous.
        doubled = JSON3.read("""[
            {"time_tag": "2026-10-12T08:24:00", "active": true, "source": "DSCOVR", "bz_gsm": 0.28},
            {"time_tag": "2026-10-12T08:24:00", "active": true, "source": "DSCOVR", "bz_gsm": 0.29},
            {"time_tag": "2026-10-12T08:24:00", "active": false, "source": "ACE", "bz_gsm": 0.01}
        ]""")
        doubled_observations = L1P._l1_row_observations(doubled)
        @test only(filter(item -> item.source == "DSCOVR", doubled_observations)).status ==
              "ambiguous_measurement_time"
        @test only(filter(item -> item.source == "ACE", doubled_observations)).status ==
              "unique_measurement_row"
    end

    @testset "public select_l1_issue_pair carries its docstring (G11-07)" begin
        @test occursin(
            "latest common", string(@doc L1P.select_l1_issue_pair),
        )
        @test !occursin(
            "latest common", string(@doc L1P._l1_pair_floor_30),
        )
    end

    @testset "collector lock makes the pairing snapshot quiescent" begin
        mktempdir() do root
            mag_response = _static_stack_pair_mag(measurement_time)
            wind_response = _static_stack_pair_wind(measurement_time)
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                mag_response, wind_response,
            )
            storage = realpath(root)
            L1P._l1_with_lock(storage; timeout_sec=1.0) do
                @test_throws ErrorException L1P.select_l1_issue_pair(
                    storage, DateTime(2022, 5, 5, 0, 30, 11);
                    lock_timeout_sec=0.0,
                )
            end
            @test !ispath(joinpath(storage, ".collector.lock"))
            @test L1P.select_l1_issue_pair(
                storage, DateTime(2022, 5, 5, 0, 30, 11),
            ).measurement_time_utc == "2022-05-05T00:30:00.000Z"
        end
    end

    @testset "SOLAR1 same-source pairs, bounded hold, and lookback (protocol amendment)" begin
        # Unmodified active SOLAR1 rows from the 2026-10-12T09:31Z live
        # snapshot; the hourly ephemeris ends at 09:00 (hold of 24 min).
        solar1_mag = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T09:24:01", "active": false, "source": "IMAP",
             "bx_gsm": 5.86, "by_gsm": -0.82, "bz_gsm": -0.2, "overall_quality": 0},
            {"time_tag": "2026-10-12T09:24:00", "active": true, "source": "SOLAR1",
             "bx_gsm": 6.34, "by_gsm": -0.72, "bz_gsm": 0.28,
             "max_data_flag": -9999, "overall_quality": 0}
        ]""")
        solar1_wind = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T09:24:00", "active": true, "source": "SOLAR1",
             "proton_speed": 458.4, "proton_density": 0.41,
             "proton_vx_gse": -457.1, "max_data_flag": 0, "overall_quality": 0}
        ]""")
        dscovr_wind = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T09:24:00", "active": true, "source": "DSCOVR",
             "proton_speed": 458.4, "proton_density": 0.41,
             "proton_vx_gse": -457.1, "overall_quality": 0}
        ]""")
        hourly = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T09:00:00", "active": true, "source": "SOLAR1",
             "x_gse": 1576421, "y_gse": -113113, "z_gse": 88675},
            {"time_tag": "2026-10-12T08:00:00", "active": true, "source": "SOLAR1",
             "x_gse": 1576509, "y_gse": -112727, "z_gse": 88879},
            {"time_tag": "2026-10-12T09:00:00", "active": true, "source": "DSCOVR",
             "x_gse": 1500000, "y_gse": 1000, "z_gse": 2000}
        ]""")
        stale = _static_stack_pair_response("""[
            {"time_tag": "2026-10-12T07:00:00", "active": true, "source": "SOLAR1",
             "x_gse": 1576597, "y_gse": -112339, "z_gse": 89083}
        ]""")
        base = DateTime(2026, 10, 12, 9, 31, 10)
        unavailable(f) = try
            f()
            ""
        catch error
            sprint(showerror, error)
        end

        mktempdir() do root
            records = _static_stack_capture_pair!(
                root, base, solar1_mag, solar1_wind; ephemeris=hourly,
            )
            for record in records
                metadata = record.metadata_provenance
                @test metadata.rows_admissible
                @test metadata.quality_row_source == "SOLAR1"
                @test JSON3.read(metadata.ephemeris_record_json).hold_age_seconds ==
                      1440.0
            end
            issue = base + Second(1)
            cutoff = L1P.capture_l1_issue_cutoff!(root, issue)
            pair = L1P.select_l1_issue_pair(
                root, issue, cutoff.cutoff_relative_path,
            )
            @test pair.source == "SOLAR1"
            @test pair.measurement_time_utc == "2026-10-12T09:24:00.000Z"
            @test pair.bz_gsm == 0.28
            @test pair.proton_vx_gse == -457.1
            @test pair.position_method == "hold"
            @test pair.position_lower_time_utc == pair.position_upper_time_utc ==
                  "2026-10-12T09:00:00.000Z"
            @test pair.position_interpolation_fraction == 0.0
            @test (pair.position_x_gse, pair.position_y_gse, pair.position_z_gse) ==
                  (1576421.0, -113113.0, 88675.0)
            @test pair.mag_quality_binding_status ==
                  pair.wind_quality_binding_status ==
                  "bound_noaa_rtsw_overall_quality"
            @test pair.pair_contract_sha256 == _static_stack_pair_contract_sha(pair)
            # Documented 2 h lookback for latest-only selection: an issue
            # 1 h 56 min after the 09:24 minute still pairs it; one whose
            # lookback starts after 09:24 reports no admitted pair.
            inside = DateTime(2026, 10, 12, 11, 20)
            @test L1P.select_l1_issue_pair(root, inside).source == "SOLAR1"
            outside = DateTime(2026, 10, 12, 11, 30)
            @test occursin(
                "no exact admitted mag/wind timestamp",
                unavailable(() -> L1P.select_l1_issue_pair(root, outside)),
            )
            # The windowed overload is bounded by its explicit window instead.
            @test length(L1P.select_l1_issue_pairs(
                root, outside; measurement_start_utc=DateTime(2026, 10, 12, 9),
            )) == 1
        end

        # Magnetometer and wind admitted from different sources never pair.
        mktempdir() do root
            records = _static_stack_capture_pair!(
                root, base, solar1_mag, dscovr_wind; ephemeris=hourly,
            )
            @test all(record -> record.metadata_provenance.rows_admissible, records)
            @test records[2].metadata_provenance.quality_row_source == "DSCOVR"
            @test occursin(
                "no exact admitted mag/wind timestamp",
                unavailable(() -> L1P.select_l1_issue_pair(
                    root, base + Second(1),
                )),
            )
        end

        # An ephemeris older than 90 minutes is stale and admits nothing.
        mktempdir() do root
            records = _static_stack_capture_pair!(
                root, base, solar1_mag, solar1_wind; ephemeris=stale,
            )
            for record in records
                @test record.metadata_provenance.ephemeris_binding_status ==
                      "ephemeris_stale"
                @test !record.metadata_provenance.rows_admissible
            end
            @test occursin(
                "no exact admitted mag/wind timestamp",
                unavailable(() -> L1P.select_l1_issue_pair(
                    root, base + Second(1),
                )),
            )
        end

        # A forged hold (age beyond 90 min, or a wrong recorded age) fails
        # closed in the pair reader even when re-hashed.
        held = (
            source="SOLAR1", position_time_utc="2026-10-12T09:24:00.000Z",
            position_frame="GSE", position_units="km", method="hold",
            lower_time_utc="2026-10-12T09:00:00.000Z",
            upper_time_utc="2026-10-12T09:00:00.000Z",
            interpolation_fraction=0.0, x_gse=1.0, y_gse=2.0, z_gse=3.0,
            hold_age_seconds=1440.0,
        )
        reader(record) = L1P._l1_pair_position(
            (ephemeris_record_json=JSON3.write(record),),
        )
        @test reader(held).method == "hold"
        for changes in (
                (lower_time_utc="2026-10-12T07:53:59.999Z",
                 upper_time_utc="2026-10-12T07:53:59.999Z",
                 hold_age_seconds=5400.001),
                (hold_age_seconds=1500.0,),
                (interpolation_fraction=0.5,),
                (upper_time_utc="2026-10-12T09:01:00.000Z",),
                (lower_time_utc="2026-10-12T09:24:00.000Z",
                 upper_time_utc="2026-10-12T09:24:00.000Z",
                 hold_age_seconds=0.0),
                (source="ACE",),
            )
            @test_throws ErrorException reader(merge(held, changes))
        end
        @test_throws ErrorException reader(Base.structdiff(held, (hold_age_seconds=0,)))
    end

    @testset "per-issue verification work is independent of archive length (G11-06/G11-08)" begin
        # Hourly ephemeris covering the whole synthetic day; one capture cycle
        # per 20 minutes, so the 2 h lookback spans about six records.
        day = DateTime(2022, 5, 5)
        ephemeris = _static_stack_pair_response(JSON3.write([
            (time_tag=L1P._l1_utc(day + Hour(h)), source="DSCOVR", active=true,
             x_gse=1.5e6 + h, y_gse=1000.0, z_gse=2000.0)
            for h in 0:24
        ]))
        cycles = 14
        function run_cohort(root)
            raw_reads = Int[]
            record_reads = Int[]
            for k in 0:(cycles - 1)
                minute = day + Minute(20 * k)
                stamp = L1P._l1_utc(minute)
                _static_stack_capture_pair!(
                    root, minute + Second(10),
                    _static_stack_pair_mag(stamp; bz=-1.0 - k),
                    _static_stack_pair_wind(stamp; speed=400.0 + k),
                    ephemeris=ephemeris,
                )
                L1P._l1_reset_read_counts!()
                pair = L1P.select_l1_issue_pair(root, minute + Second(11))
                @test pair.measurement_time_utc == stamp
                push!(raw_reads, L1P._l1_read_count(:raw) +
                                 L1P._l1_read_count(:ephemeris_raw))
                push!(record_reads, L1P._l1_read_count(:record))
            end
            return (raw=raw_reads, record=record_reads)
        end
        previous_age = L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[]
        try
            @test previous_age == 24.0 * 3600.0
            L1P._l1_clear_verify_memo!()
            memo = mktempdir() do root
                run_cohort(root)
            end
            # Once the lookback window is full (> 6 cycles), every issue reads
            # the same number of raw objects and records however long the
            # archive already is.
            @test length(unique(memo.raw[9:end])) == 1
            @test length(unique(memo.record[9:end])) == 1
            @test memo.raw[end] < 2 * cycles

            # Control: with the memo disabled (maximum age < 0) every issue
            # re-verifies the whole archive, so the work grows linearly.
            L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[] = -1.0
            full = mktempdir() do root
                run_cohort(root)
            end
            @test issorted(full.raw; lt=<=)
            @test full.raw[end] - full.raw[9] >= 2 * (cycles - 9)
            @test full.raw[end] > memo.raw[end]
        finally
            L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[] = previous_age
        end

        # Tamper evidence: a rewritten memoized head forces full verification
        # and fails; an older prefix rewrite is caught as soon as the memo
        # expires (configurable maximum age).
        mktempdir() do root
            storage = realpath(root)
            for k in 0:9
                minute = day + Minute(20 * k)
                stamp = L1P._l1_utc(minute)
                _static_stack_capture_pair!(
                    root, minute + Second(10), _static_stack_pair_mag(stamp),
                    _static_stack_pair_wind(stamp), ephemeris=ephemeris,
                )
            end
            issue = day + Minute(180) + Second(11)
            @test L1P.select_l1_issue_pair(root, issue).source == "DSCOVR"
            # Prefix record 2 lies outside the 2 h lookback: with a fresh memo
            # a later issue does not re-read it.
            old = joinpath(storage, L1P._l1_record_relative(sources[1].name, 2))
            original = read(old)
            write(old, replace(String(copy(original)), "\"http_status\":200" =>
                                                       "\"http_status\":201"))
            @test read(old) != original
            later = issue + Second(1)
            @test L1P.select_l1_issue_pair(root, later).source == "DSCOVR"
            previous_age = L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[]
            try
                L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[] = 0.0
                @test_throws ErrorException L1P.select_l1_issue_pair(
                    root, later + Second(1),
                )
            finally
                L1P.L1_VERIFY_MEMO_MAX_AGE_SEC[] = previous_age
            end
            # The full public verifier never uses the memo.
            @test_throws ErrorException L1P.verify_l1_receipts(root)
            write(old, original)
            @test L1P.select_l1_issue_pair(root, later + Second(2)).source ==
                  "DSCOVR"
            # Rewriting the memoized head record itself is detected immediately.
            head = joinpath(storage, L1P._l1_record_relative(sources[1].name, 10))
            head_original = read(head)
            write(head, replace(String(copy(head_original)), "\"http_status\":200" =>
                                                             "\"http_status\":201"))
            @test read(head) != head_original
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, later + Second(3),
            )
            write(head, head_original)
        end
    end

    @testset "issue cutoff and latest exact common minute" begin
        mktempdir() do root
            first = _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time; bz=-5.0),
                _static_stack_pair_wind(measurement_time; speed=400.0),
            )
            early_issue = DateTime(2022, 5, 5, 0, 30, 20)
            early_cutoff = L1P.capture_l1_issue_cutoff!(
                root, early_issue,
            )
            later_mag = _static_stack_pair_response(JSON3.write([
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
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 31, 10),
                later_mag,
                _static_stack_pair_wind("2022-05-05T00:31:00Z"; speed=500.0),
            )
            early = L1P.select_l1_issue_pair(
                root, early_issue, early_cutoff.cutoff_relative_path,
            )
            @test early.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test early.mag_record_sha256 == first[1].record_sha256
            @test early.wind_record_sha256 == first[2].record_sha256
            @test_throws ArgumentError L1P.select_l1_issue_pair(
                root, early_issue,
            )
            late = L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 20),
            )
            @test late.measurement_time_utc == "2022-05-05T00:31:00.000Z"
            @test late.bz_gsm == -8.0
            @test late.proton_speed == 500.0
            @test late.mag_sequence == late.wind_sequence == 2
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time),
            )
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 32, 10),
                _static_stack_pair_mag("2022-05-05T00:31:00Z"),
                _static_stack_pair_wind("2022-05-05T00:32:00Z"),
            )
            pair = L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 32, 20),
            )
            # No interpolation or nearest-minute join: the older exact pair wins.
            @test pair.measurement_time_utc == "2022-05-05T00:30:00.000Z"
            @test pair.mag_sequence == pair.wind_sequence == 1
        end
    end

    @testset "saved cutoff isolates replay from every later object" begin
        mktempdir() do root
            early_issue = DateTime(2022, 5, 5, 0, 30, 20)
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time; bz=-5.0),
                _static_stack_pair_wind(measurement_time; speed=400.0),
            )
            cutoff = L1P.capture_l1_issue_cutoff!(root, early_issue)
            baseline = L1P.select_l1_issue_pair(
                root, early_issue, cutoff.cutoff_relative_path,
            )
            verified = L1P.verify_l1_issue_cutoff(
                root, early_issue, cutoff.cutoff_relative_path,
            )
            @test verified.cutoff_sha256 == cutoff.cutoff_sha256

            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 31, 10),
                _static_stack_pair_mag("2022-05-05T00:31:00Z"; bz=-9.0),
                _static_stack_pair_wind("2022-05-05T00:31:00Z"; speed=500.0),
            )
            @test_throws ArgumentError L1P.select_l1_issue_pair(
                root, early_issue,
            )

            later_record_path = joinpath(
                root, "records", sources[1].name,
                "00000000000000000002.json",
            )
            later_record = JSON3.read(read(later_record_path, String))
            later_raw_path = joinpath(
                root, String(later_record.raw_relative_path),
            )
            open(later_raw_path, "a") do io
                write(io, UInt8('x'))
            end
            write(later_record_path, "{")
            write(joinpath(root, "latest", sources[1].name * ".json"), "{")

            # The saved head verifies exactly its prefixes and never reads the
            # corrupted later record, raw body, or latest pointer.
            @test L1P.verify_l1_issue_cutoff(
                root, early_issue, cutoff.cutoff_relative_path,
            ).cutoff_sha256 == cutoff.cutoff_sha256
            @test L1P.select_l1_issue_pair(
                root, early_issue, cutoff.cutoff_relative_path,
            ) == baseline
        end

        mktempdir() do root
            issue = DateTime(2022, 5, 5, 0, 30, 20)
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time),
            )
            cutoff = L1P.capture_l1_issue_cutoff!(root, issue)
            cutoff_path = joinpath(root, cutoff.cutoff_relative_path)
            original = read(cutoff_path, String)
            document = JSON3.read(original)
            mutated = replace(
                original,
                String(document.cutoff_sha256) => repeat("0", 64),
            )
            write(cutoff_path, mutated)
            @test_throws ArgumentError L1P.verify_l1_issue_cutoff(
                root, issue, cutoff.cutoff_relative_path,
            )
            @test_throws ArgumentError L1P.select_l1_issue_pair(
                root, issue, cutoff.cutoff_relative_path,
            )
        end
    end

    @testset "one-pass causal-window pairing returns every stable minute" begin
        mktempdir() do root
            for minute in 28:30
                timestamp = "2022-05-05T00:$(minute):00Z"
                _static_stack_capture_pair!(
                    root, DateTime(2022, 5, 5, 0, minute, 10),
                    _static_stack_pair_mag(timestamp; bz=-minute),
                    _static_stack_pair_wind(timestamp; speed=400.0 + minute),
                )
            end
            issue = DateTime(2022, 5, 5, 0, 31)
            cutoff = L1P.capture_l1_issue_cutoff!(root, issue)
            before = _static_stack_pair_tree_hashes(root)
            pairs = L1P.select_l1_issue_pairs(
                root, issue, cutoff.cutoff_relative_path;
                measurement_start_utc=DateTime(2022, 5, 5, 0, 29),
            )
            @test length(pairs) == 2
            @test getproperty.(pairs, :measurement_time_utc) == (
                "2022-05-05T00:29:00.000Z",
                "2022-05-05T00:30:00.000Z",
            )
            @test getproperty.(pairs, :bz_gsm) == (-29.0, -30.0)
            @test all(pair -> pair.issue_time_utc ==
                      "2022-05-05T00:31:00.000Z", pairs)
            @test last(pairs) == L1P.select_l1_issue_pair(
                root, issue, cutoff.cutoff_relative_path,
            )
            @test _static_stack_pair_tree_hashes(root) == before
            @test_throws ArgumentError L1P.select_l1_issue_pairs(
                root, issue, cutoff.cutoff_relative_path;
                measurement_start_utc=DateTime(2022, 5, 5, 0, 32),
            )
        end
    end

    @testset "missing, unadmitted, and revised rows fail closed" begin
        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind("2022-05-05T00:31:00Z"),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time; quality=1),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            missing_vx = _static_stack_pair_response(JSON3.write([(
                time_tag=measurement_time, source="DSCOVR", active=true,
                overall_quality=0, proton_speed=400.0, proton_density=5.0,
            )]))
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time), missing_vx,
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            # Repeated identical rows are not revisions; the latest receipt wins.
            for second in (10, 20)
                _static_stack_capture_pair!(
                    root, DateTime(2022, 5, 5, 0, 30, second),
                    _static_stack_pair_mag(measurement_time),
                    _static_stack_pair_wind(measurement_time),
                )
            end
            pair = L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
            @test pair.mag_sequence == 2
            @test pair.wind_sequence == 2
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time; bz=-5.0),
                _static_stack_pair_wind(measurement_time),
            )
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 20),
                _static_stack_pair_mag(measurement_time; bz=-9.0),
                _static_stack_pair_wind(measurement_time),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time; quality=0),
            )
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 20),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time; quality=1),
            )
            # A later suspect revision revokes the earlier normal row.
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time; bz=-5.0),
                _static_stack_pair_wind(measurement_time),
            )
            revised_history = _static_stack_pair_response(JSON3.write([
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
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 32, 10), revised_history,
                _static_stack_pair_wind("2022-05-05T00:32:00Z"),
            )
            # A revision cannot hide below a newer latest row in a later receipt.
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 33, 0),
            )
        end
    end

    @testset "receipt, metadata, raw, and orbit mutations fail closed" begin
        mktempdir() do root
            future_measurement = "2022-05-05T00:31:00Z"
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(future_measurement),
                _static_stack_pair_wind(future_measurement),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 32),
            )
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time),
            )
            _static_stack_pair_rehash_latest!(
                root, sources[1];
                record_changes=(
                    receipt_completed_utc="2022-05-05T00:31:00.000Z",
                ),
            )
            @test_throws ArgumentError L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 30, 30),
            )
        end

        mktempdir() do root
            _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time),
            )
            _static_stack_pair_rehash_latest!(
                root, sources[1];
                metadata_changes=(
                    quality_row_timestamp_utc="2022-05-05T00:29:00.000Z",
                ),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            records = _static_stack_capture_pair!(
                root, DateTime(2022, 5, 5, 0, 30, 10),
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_wind(measurement_time),
            )
            open(joinpath(root, records[1].raw_relative_path), "a") do io
                write(io, UInt8('x'))
            end
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end

        mktempdir() do root
            _static_stack_capture_one!(
                root, DateTime(2022, 5, 5, 0, 30, 10), sources[1],
                _static_stack_pair_mag(measurement_time),
                _static_stack_pair_ephemeris(),
            )
            _static_stack_capture_one!(
                root, DateTime(2022, 5, 5, 0, 30, 20), sources[2],
                _static_stack_pair_wind(measurement_time),
                _static_stack_pair_ephemeris(
                    lower=(11.0, 21.0, 31.0), upper=(21.0, 41.0, 61.0),
                ),
            )
            @test_throws ErrorException L1P.select_l1_issue_pair(
                root, DateTime(2022, 5, 5, 0, 31, 0),
            )
        end
    end
end
