using Test
using Dates
using LinearAlgebra
using SHA
using JSON3
using SolarSINDy

_v22a_utc(time::DateTime) =
    Dates.format(time, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"
_v22a_sha(parts...) = bytes2hex(sha256(codeunits(join(string.(parts), "|"))))

function _v22a_floor_30(time::DateTime)
    minute = Dates.minute(time) < 30 ? 0 : 30
    return DateTime(
        Dates.year(time), Dates.month(time), Dates.day(time), Dates.hour(time),
        minute,
    )
end

function _v22a_ceil_30(time::DateTime)
    floored = _v22a_floor_30(time)
    return floored == time ? floored : floored + Minute(30)
end

function _v22a_pair_payload(pair)
    names = Tuple(filter(!=(:pair_contract_sha256), propertynames(pair)))
    return NamedTuple{names}(Tuple(getproperty(pair, name) for name in names))
end

function _v22a_rehash_pair(pair, changes::NamedTuple=NamedTuple())
    mutated = merge(pair, changes)
    payload = _v22a_pair_payload(mutated)
    return merge(payload, (
        pair_contract_sha256=bytes2hex(sha256(codeunits(JSON3.write(payload)))),
    ))
end

function _v22a_pair(
        measurement::DateTime;
        issue::DateTime,
        delay_minutes::Real=20.0,
        vx::Real=-500.0,
        bx::Real=1.0,
        by::Real=2.0,
        bz::Real=-3.0,
        speed::Real=400.0,
        density::Real=5.0,
        sequence::Integer=1,
        label::AbstractString=string(Dates.value(measurement)),
        receipt::DateTime=measurement + Second(5))
    x = -Float64(vx) * 60.0 * Float64(delay_minutes)
    ephemeris_hash = _v22a_sha("ephemeris", measurement, x, label)
    wind_receipt = receipt + Millisecond(1)
    cutoff_token = Dates.format(issue, dateformat"yyyymmddTHHMMSSsss") * "Z"
    payload = (
        schema_version=OPERATIONAL_V22_ARRIVAL_PAIR_SCHEMA_VERSION,
        issue_time_utc=_v22a_utc(issue),
        first_eligible_issue_time_utc=_v22a_utc(
            _v22a_ceil_30(max(receipt, wind_receipt)),
        ),
        issue_cutoff_relative_path=joinpath(
            "issue_cutoffs", cutoff_token * ".json",
        ),
        issue_cutoff_sha256=_v22a_sha("cutoff", issue),
        measurement_time_utc=_v22a_utc(measurement),
        source="DSCOVR",
        mag_source_product_id="swpc_rtsw_mag_1m",
        wind_source_product_id="swpc_rtsw_wind_1m",
        magnetic_component_frame="GSM",
        magnetic_component_units="nT",
        proton_speed_units="km/s",
        proton_density_units="cm^-3",
        proton_vx_frame="GSE",
        proton_vx_units="km/s",
        bx_gsm=bx,
        by_gsm=by,
        bz_gsm=bz,
        proton_speed=speed,
        proton_density=density,
        proton_vx_gse=vx,
        position_x_gse=x,
        position_y_gse=1000.0 + sequence,
        position_z_gse=-500.0 - sequence,
        position_frame="GSE",
        position_units="km",
        position_method="exact",
        position_lower_time_utc=_v22a_utc(measurement),
        position_upper_time_utc=_v22a_utc(measurement),
        position_interpolation_fraction=0.0,
        ephemeris_record_sha256=ephemeris_hash,
        mag_quality_source_product="dscovr_m1m",
        mag_quality_value=0,
        mag_quality_binding_status="bound_noaa_dscovr_overall_quality",
        mag_quality_decision="accept_normal_overall_quality",
        mag_quality_required_fields_status="bound_required_bx_by_bz_gsm",
        mag_sequence=sequence,
        mag_receipt_completed_utc=_v22a_utc(receipt),
        mag_record_sha256=_v22a_sha("mag-record", label),
        mag_raw_sha256=_v22a_sha("mag-raw", label),
        mag_ephemeris_source_object_sha256=_v22a_sha("mag-eph", label),
        wind_quality_source_product="dscovr_f1m",
        wind_quality_value=0,
        wind_quality_binding_status="bound_noaa_dscovr_overall_quality",
        wind_quality_decision="accept_normal_overall_quality",
        wind_quality_required_fields_status=
            "bound_required_speed_density_vx_gse",
        wind_sequence=sequence,
        wind_receipt_completed_utc=_v22a_utc(wind_receipt),
        wind_record_sha256=_v22a_sha("wind-record", label),
        wind_raw_sha256=_v22a_sha("wind-raw", label),
        wind_ephemeris_source_object_sha256=_v22a_sha("wind-eph", label),
    )
    return merge(payload, (
        pair_contract_sha256=bytes2hex(sha256(codeunits(JSON3.write(payload)))),
    ))
end

function _v22a_pairs_for_bins(starts, issue::DateTime)
    return [
        _v22a_pair(
            start - Minute(15);
            issue=issue,
            delay_minutes=20.0,
            sequence=index,
            bx=Float64(index),
            by=-0.5 * index,
            bz=-2.0 - 0.1 * index,
            speed=350.0 + index,
            density=4.0 + 0.1 * index,
            label="history-$index",
        )
        for (index, start) in enumerate(starts)
    ]
end

function _v22a_ready_pairs(; future_count::Integer=0,
                           missing_starts=Set{DateTime}())
    base = DateTime(2022, 5, 5)
    issue = base + Hour(12) + Minute(30)
    starts = [
        base + Minute(30 * index) for index in 0:24
        if !(base + Minute(30 * index) in missing_starts)
    ]
    result = _v22a_pairs_for_bins(starts, issue)
    if future_count >= 1
        push!(result, _v22a_pair(
            base + Hour(12) + Minute(5);
            issue=issue, delay_minutes=25.0, sequence=101,
            bx=91.0, by=-19.0, bz=-9.1, speed=491.0, density=9.1,
            label="future-1",
        ))
    end
    if future_count >= 2
        push!(result, _v22a_pair(
            base + Hour(12) + Minute(10);
            issue=issue, delay_minutes=50.0, sequence=102,
            bx=92.0, by=-29.0, bz=-9.2, speed=492.0, density=9.2,
            label="future-2",
        ))
    end
    return result, issue
end

function _v22a_artifact(factor::Real=0.5; intercept=zeros(5), label="arrival")
    coefficients = zeros(5, 5, 6)
    coefficients[:, :, 1] .= Float64(factor) .* Matrix{Float64}(I, 5, 5)
    return OperationalV22DriverArtifact(
        coefficients;
        center=zeros(5), scale=ones(5), intercept=intercept,
        fit_rows=100, label=label,
    )
end

@testset verbose=true "Operational V2.2 receipt transport and arrival queue" begin
    @testset "frozen units, boundary, window, and hand transport" begin
        @test OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES == 30
        @test OPERATIONAL_V22_ARRIVAL_TRAILING_MINUTES == 15
        @test OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS == 25
        @test OPERATIONAL_V22_ARRIVAL_PATH_STEPS == 14
        @test OPERATIONAL_V22_ARRIVAL_MIN_DELAY_MINUTES == 20
        @test OPERATIONAL_V22_ARRIVAL_MAX_DELAY_MINUTES == 120
        @test OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM == 0.0
        @test OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM == 1.5e6

        issue = DateTime(2022, 5, 5, 2)
        pairs = [
            _v22a_pair(DateTime(2022, 5, 5, 0); issue=issue,
                       vx=-400.0, delay_minutes=50.0, sequence=1),
            _v22a_pair(DateTime(2022, 5, 5, 0, 1); issue=issue,
                       vx=-500.0, delay_minutes=50.0, sequence=2),
            _v22a_pair(DateTime(2022, 5, 5, 0, 15); issue=issue,
                       vx=-600.0, delay_minutes=55.0, sequence=3),
        ]
        # Set the last x so the independently expected median Vx=-550 km/s
        # produces exactly a 60-minute Earth-boundary delay.
        pairs[3] = _v22a_rehash_pair(
            pairs[3], (position_x_gse=550.0 * 3600.0,),
        )
        queue = build_operational_v22_arrival_queue(pairs, issue)
        last_pair = last(queue.transported_pairs)
        @test last_pair.trailing_vx_gse == -550.0
        @test last_pair.delay_seconds == 3600.0
        @test last_pair.arrival_time_utc == DateTime(2022, 5, 5, 1, 15)
        @test last_pair.v21_compatibility_delay_seconds == 1.5e6 / 550.0
        @test last_pair.delay_seconds != last_pair.v21_compatibility_delay_seconds

        boundary = _v22a_pair(
            DateTime(2022, 5, 5, 0, 10);
            issue=issue, delay_minutes=50.0, sequence=4,
        )
        boundary_queue = build_operational_v22_arrival_queue((boundary,), issue)
        @test only(boundary_queue.transported_pairs).arrival_time_utc ==
              DateTime(2022, 5, 5, 1)
        @test only(boundary_queue.arrival_bins).start_utc ==
              DateTime(2022, 5, 5, 1)
        @test only(boundary_queue.arrival_bins).end_utc ==
              DateTime(2022, 5, 5, 1, 30)
    end

    @testset "S9: the linear ephemeris-position branch is verified, not only accepted" begin
        # Every fixture pair uses `position_method = "exact"`, so the linear branch of the position
        # contract — the one a real interpolated ephemeris row takes — was never entered: its
        # bracket rule and its interpolation-fraction check were unexecuted code.
        issue = DateTime(2022, 5, 5, 2)
        measurement = DateTime(2022, 5, 5, 0, 10)
        lower = measurement - Minute(2)
        upper = measurement + Minute(6)
        exact_fraction = Dates.value(measurement - lower) / Dates.value(upper - lower)
        @test exact_fraction == 0.25

        linear = _v22a_rehash_pair(
            _v22a_pair(measurement; issue=issue, delay_minutes=50.0, sequence=7),
            (position_method="linear",
             position_lower_time_utc=_v22a_utc(lower),
             position_upper_time_utc=_v22a_utc(upper),
             position_interpolation_fraction=exact_fraction),
        )
        queue = build_operational_v22_arrival_queue((linear,), issue)
        @test only(queue.transported_pairs).arrival_time_utc == DateTime(2022, 5, 5, 1)

        # A fraction that does not match the bracket is refused.
        for bad_fraction in (0.0, 0.5, exact_fraction + 1e-6)
            bad = _v22a_rehash_pair(linear, (position_interpolation_fraction=bad_fraction,))
            @test_throws ArgumentError build_operational_v22_arrival_queue((bad,), issue)
        end
        # The linear branch requires a STRICT bracket on BOTH sides. Each half-open case below
        # carries the fraction the loosened rule would compute, so only the strictness of the
        # bracket test can refuse it: a `<` weakened to `<=` on either side lets it through.
        touching_lower = _v22a_rehash_pair(
            linear,
            (position_lower_time_utc=_v22a_utc(measurement),
             position_upper_time_utc=_v22a_utc(upper),
             position_interpolation_fraction=0.0),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue((touching_lower,), issue)
        touching_upper = _v22a_rehash_pair(
            linear,
            (position_lower_time_utc=_v22a_utc(lower),
             position_upper_time_utc=_v22a_utc(measurement),
             position_interpolation_fraction=1.0),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue((touching_upper,), issue)
        # A fully degenerate bracket is refused too, even though the exact branch accepts that shape.
        degenerate = _v22a_rehash_pair(
            linear,
            (position_lower_time_utc=_v22a_utc(measurement),
             position_upper_time_utc=_v22a_utc(measurement),
             position_interpolation_fraction=0.0),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue((degenerate,), issue)
        # A bracket longer than one hour is refused on either branch.
        too_wide = _v22a_rehash_pair(
            linear,
            (position_lower_time_utc=_v22a_utc(measurement - Minute(31)),
             position_upper_time_utc=_v22a_utc(measurement + Minute(31)),
             position_interpolation_fraction=0.5),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue((too_wide,), issue)
        # And an unknown method is refused outright, so the two branches are the whole contract.
        unknown = _v22a_rehash_pair(linear, (position_method="spline",))
        @test_throws ArgumentError build_operational_v22_arrival_queue((unknown,), issue)
    end

    @testset "bin medians and exact physical fallbacks" begin
        issue = DateTime(2022, 5, 5, 2)
        pairs = [
            _v22a_pair(DateTime(2022, 5, 5, 0, index);
                       issue=issue, delay_minutes=20.0, sequence=index + 1,
                       bx=bx, by=2bx, bz=-bx, speed=speed, density=density)
            for (index, bx, speed, density) in
                ((0, 1.0, 300.0, 3.0),
                 (1, 9.0, 500.0, 9.0),
                 (2, 5.0, 400.0, 5.0))
        ]
        queue = build_operational_v22_arrival_queue(pairs, issue)
        bin = only(queue.arrival_bins)
        @test (bin.bx_gsm, bin.by_gsm, bin.bz_gsm) == (5.0, 10.0, -5.0)
        @test (bin.proton_speed, bin.proton_density) == (400.0, 5.0)
        @test length(bin.contributing_pair_sha256) == 3

        at_min = build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue,
                       delay_minutes=20.0),
        ), issue)
        at_max = build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue,
                       delay_minutes=120.0),
        ), issue)
        @test only(at_min.transported_pairs).delay_seconds == 1200.0
        @test only(at_max.transported_pairs).delay_seconds == 7200.0
        @test build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue,
                       delay_minutes=19.999),
        ), issue).fallback_reason == :delay_out_of_bounds
        @test build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue,
                       delay_minutes=120.001),
        ), issue).fallback_reason == :delay_out_of_bounds
        @test build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue, vx=1.0),
        ), issue).fallback_reason == :invalid_vx
        @test build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue, speed=0.0),
        ), issue).fallback_reason == :invalid_plasma
        @test build_operational_v22_arrival_queue((
            _v22a_pair(DateTime(2022, 5, 5); issue=issue, density=-1.0),
        ), issue).fallback_reason == :invalid_plasma
    end

    @testset "history completeness, fills, and freshness boundaries" begin
        pairs, issue = _v22a_ready_pairs()
        queue = build_operational_v22_arrival_queue(pairs, issue)
        @test queue.status == :ready
        @test queue.fallback_reason == :none
        @test verify_operational_v22_arrival_queue(queue)
        @test size(operational_v22_arrival_history(queue)) == (25, 5)
        @test all(bin -> bin.observed, queue.history_bins)
        @test last(queue.history_bins).end_utc == issue

        base = DateTime(2022, 5, 5)
        missing = Set((base + Hour(5),))
        one_gap_pairs, _ = _v22a_ready_pairs(missing_starts=missing)
        one_gap = build_operational_v22_arrival_queue(one_gap_pairs, issue)
        @test one_gap.status == :ready
        filled = only(filter(bin -> !bin.observed, one_gap.history_bins))
        @test filled.start_utc == base + Hour(5)
        @test filled.filled_from_start_utc == base + Hour(4) + Minute(30)
        previous = one_gap.history_bins[10]
        @test (filled.bx_gsm, filled.by_gsm, filled.bz_gsm,
               filled.proton_speed, filled.proton_density) ==
              (previous.bx_gsm, previous.by_gsm, previous.bz_gsm,
               previous.proton_speed, previous.proton_density)

        two_missing = Set((base + Hour(5), base + Hour(7)))
        two_gap_pairs, _ = _v22a_ready_pairs(missing_starts=two_missing)
        @test build_operational_v22_arrival_queue(
            two_gap_pairs, issue,
        ).fallback_reason == :incomplete_history
        leading_pairs, _ = _v22a_ready_pairs(missing_starts=Set((base,)))
        @test build_operational_v22_arrival_queue(
            leading_pairs, issue,
        ).fallback_reason == :incomplete_history

        latest_start = issue - Minute(120)
        starts = [latest_start - Minute(30 * index) for index in 24:-1:0]
        freshness_pairs = _v22a_pairs_for_bins(starts, issue)
        at_boundary = build_operational_v22_arrival_queue(
            freshness_pairs, issue,
        )
        @test at_boundary.status == :ready
        @test issue - last(at_boundary.history_bins).end_utc == Minute(90)
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            freshness_pairs, issue + Millisecond(1),
        )
        stale_bin = OperationalV22ArrivalBin(
            issue - Minute(120) - Millisecond(1),
            issue - Minute(90) - Millisecond(1),
            1.0, 2.0, -3.0, 400.0, 5.0, true, nothing, (),
        )
        @test SolarSINDy._operational_v22_arrival_history(
            (stale_bin,), issue,
        ).reason == :stale_history
    end

    @testset "overtaking threshold is strict and crossings aggregate" begin
        issue = DateTime(2022, 5, 5, 3)
        first_pair = _v22a_pair(
            DateTime(2022, 5, 5); issue=issue,
            delay_minutes=120.0, sequence=1,
        )
        exact_pair = _v22a_pair(
            DateTime(2022, 5, 5, 0, 1); issue=issue,
            delay_minutes=89.0, sequence=2,
        )
        exact = build_operational_v22_arrival_queue(
            (first_pair, exact_pair), issue,
        )
        @test exact.fallback_reason != :overtaking_exceeds_one_bin
        @test last(exact.transported_pairs).arrival_time_utc ==
              DateTime(2022, 5, 5, 1, 30)

        excessive_pair = _v22a_pair(
            DateTime(2022, 5, 5, 0, 1); issue=issue,
            delay_minutes=88.0 + 59.999 / 60.0, sequence=3,
        )
        excessive = build_operational_v22_arrival_queue(
            (first_pair, excessive_pair), issue,
        )
        @test length(excessive.transported_pairs) == 1
        @test only(excessive.transported_pairs).arrival_time_utc ==
              DateTime(2022, 5, 5, 2)
        @test excessive.fallback_reason == :overtaking_exceeds_one_bin
    end

    @testset "receipt causality, revision stability, and mutation identity" begin
        pairs, issue = _v22a_ready_pairs()
        baseline = build_operational_v22_arrival_queue(pairs, issue)
        future = _v22a_pair(
            issue + Minute(1);
            issue=issue + Minute(2), receipt=issue + Minute(1) + Second(5),
            sequence=999, label="post-issue",
        )
        future_mutated = (issue_time_utc=future.issue_time_utc,)
        with_future = build_operational_v22_arrival_queue(
            vcat(pairs, [future]), issue,
        )
        with_mutated_future = build_operational_v22_arrival_queue(
            vcat(pairs, [future_mutated]), issue,
        )
        @test with_future.composite_sha256 == baseline.composite_sha256
        @test with_mutated_future.composite_sha256 == baseline.composite_sha256
        @test with_mutated_future.receipt_pairs == baseline.receipt_pairs

        changed_raw = copy(pairs)
        changed_raw[1] = _v22a_rehash_pair(changed_raw[1], (
            mag_raw_sha256=_v22a_sha("changed-preissue-raw"),
        ))
        raw_queue = build_operational_v22_arrival_queue(changed_raw, issue)
        @test raw_queue.composite_sha256 != baseline.composite_sha256
        @test operational_v22_arrival_history(raw_queue) ==
              operational_v22_arrival_history(baseline)

        duplicate = pairs[1]
        deduplicated = build_operational_v22_arrival_queue(
            vcat(pairs, [duplicate]), issue,
        )
        @test length(deduplicated.receipt_pairs) == length(baseline.receipt_pairs)
        revised_provenance = _v22a_rehash_pair(pairs[1], (
            mag_sequence=1001,
            wind_sequence=1001,
            mag_record_sha256=_v22a_sha("duplicate-mag-record"),
            mag_raw_sha256=_v22a_sha("duplicate-mag-raw"),
            wind_record_sha256=_v22a_sha("duplicate-wind-record"),
            wind_raw_sha256=_v22a_sha("duplicate-wind-raw"),
        ))
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            vcat(pairs, [revised_provenance]), issue,
        )
        conflicting = _v22a_rehash_pair(
            pairs[1], (bz_gsm=pairs[1].bz_gsm - 1.0,),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            vcat(pairs, [conflicting]), issue,
        )

        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [_v22a_rehash_pair(pairs[1], (position_frame="GSM",))], issue,
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [_v22a_rehash_pair(pairs[1], (position_units="m",))], issue,
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [_v22a_rehash_pair(pairs[1], (
                mag_receipt_completed_utc=_v22a_utc(issue + Second(1)),
            ))], issue,
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [_v22a_rehash_pair(pairs[1], (
                mag_quality_value=1,
            ))], issue,
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [_v22a_rehash_pair(pairs[1], (
                proton_vx_frame="GSM",
            ))], issue,
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            [merge(pairs[1], (mag_source_product_id="mutated",))], issue,
        )
        mixed_cutoff = copy(pairs)
        mixed_cutoff[2] = _v22a_rehash_pair(mixed_cutoff[2], (
            issue_cutoff_sha256=_v22a_sha("different-cutoff"),
        ))
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            mixed_cutoff, issue,
        )

        tampered = OperationalV22ArrivalQueue(
            baseline.schema_version,
            baseline.issue_time_utc,
            baseline.status,
            baseline.fallback_reason,
            baseline.x_ref_gse_km,
            baseline.v21_compatibility_distance_km,
            baseline.receipt_pairs,
            baseline.transported_pairs,
            baseline.arrival_bins,
            baseline.history_bins,
            baseline.future_bins,
            repeat("0", 64),
        )
        @test_throws ArgumentError verify_operational_v22_arrival_queue(tampered)

        changed_transport = collect(baseline.transported_pairs)
        changed_transport[1] = merge(changed_transport[1], (
            delay_seconds=changed_transport[1].delay_seconds + 1.0,
        ))
        provisional = OperationalV22ArrivalQueue(
            baseline.schema_version,
            baseline.issue_time_utc,
            baseline.status,
            baseline.fallback_reason,
            baseline.x_ref_gse_km,
            baseline.v21_compatibility_distance_km,
            baseline.receipt_pairs,
            Tuple(changed_transport),
            baseline.arrival_bins,
            baseline.history_bins,
            baseline.future_bins,
            repeat("0", 64),
        )
        rehashed = OperationalV22ArrivalQueue(
            provisional.schema_version,
            provisional.issue_time_utc,
            provisional.status,
            provisional.fallback_reason,
            provisional.x_ref_gse_km,
            provisional.v21_compatibility_distance_km,
            provisional.receipt_pairs,
            provisional.transported_pairs,
            provisional.arrival_bins,
            provisional.history_bins,
            provisional.future_bins,
            operational_v22_arrival_sha256(provisional),
        )
        @test_throws ArgumentError verify_operational_v22_arrival_queue(rehashed)
    end

    @testset "future queue and issue grid are exact" begin
        pairs, issue = _v22a_ready_pairs(future_count=2)
        aligned = build_operational_v22_arrival_queue(pairs, issue)
        @test getproperty.(aligned.future_bins, :start_utc) == (
            issue, issue + Minute(30),
        )

        @test_throws ArgumentError build_operational_v22_arrival_queue(
            pairs, issue + Minute(1),
        )
        @test_throws ArgumentError build_operational_v22_arrival_queue(
            pairs, issue + Millisecond(1),
        )
    end

    @testset "queue plus sparse path has exact prefix and state order" begin
        pairs, issue = _v22a_ready_pairs(future_count=2)
        queue = build_operational_v22_arrival_queue(pairs, issue)
        artifact = _v22a_artifact(0.5; label="half-map")
        path = build_operational_v22_arrival_path(queue, artifact)
        @test path.status == :ready
        @test path.gate_status == :ungated_candidate ==
              OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS
        @test path.fallback_reason == :none
        @test length(path.steps) == 14
        @test verify_operational_v22_arrival_path(path)
        @test path.queue_sha256 == operational_v22_arrival_sha256(queue)
        @test path.artifact_sha256 == operational_v22_driver_sha256(artifact)
        @test verify_operational_v22_arrival_path(path, queue, artifact)
        @test getproperty.(path.steps, :origin)[1:2] == (:queued, :queued)
        @test all(==(:sparse), getproperty.(path.steps, :origin)[3:end])
        for index in 1:2
            bin = queue.future_bins[index]
            step = path.steps[index]
            @test (step.start_utc, step.bx_gsm, step.by_gsm, step.bz_gsm,
                   step.proton_speed, step.proton_density,
                   step.contributing_pair_sha256) ==
                  (bin.start_utc, bin.bx_gsm, bin.by_gsm, bin.bz_gsm,
                   bin.proton_speed, bin.proton_density,
                   bin.contributing_pair_sha256)
        end
        expected_sparse = 0.5 .* collect((
            path.steps[2].bx_gsm,
            path.steps[2].by_gsm,
            path.steps[2].bz_gsm,
            log(path.steps[2].proton_speed),
            log(path.steps[2].proton_density),
        ))
        @test collect((
            path.steps[3].bx_gsm,
            path.steps[3].by_gsm,
            path.steps[3].bz_gsm,
            log(path.steps[3].proton_speed),
            log(path.steps[3].proton_density),
        )) ≈ expected_sparse atol=1e-13 rtol=1e-13
        matrix = operational_v22_arrival_path_matrix(path)
        @test size(matrix) == (14, 5)
        @test matrix[1, :] == [
            path.steps[1].bx_gsm,
            path.steps[1].by_gsm,
            path.steps[1].bz_gsm,
            log(path.steps[1].proton_speed),
            log(path.steps[1].proton_density),
        ]
        @test length(operational_v22_hourly_drivers(matrix)) == 7
        library = build_solar_wind_library()
        coefficients = zeros(length(library))
        active = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13)
        coefficients[collect(active)] .= collect(1.0e-5:1.0e-5:11.0e-5)
        core = OperationalCore(
            OperationalCoreArtifacts(
                OPERATIONAL_V2_1_MODEL_VERSION,
                "arrival-test-coefficients.csv",
                "arrival-test-ensemble.csv",
                "arrival-test-draws.csv",
                20,
                11,
            ),
            library,
            coefficients,
        )
        @test_throws ArgumentError operational_v22_core_path_forecast(
            core, -25.0, path,
        )
        pinned_core_sha = operational_v22_core_sha256(core)
        @test_throws ArgumentError operational_v22_core_path_forecast(
            core, -25.0, path, queue, artifact, pinned_core_sha,
        )
        @test_throws ArgumentError operational_v22_core_path_forecast(
            core, -25.0, path, queue, artifact, repeat("0", 64),
        )
        @test_throws ArgumentError operational_v22_core_path_forecast(
            core, -25.0, path, queue, _v22a_artifact(0.4), pinned_core_sha,
        )
        @test operational_v22_core_path_forecast(
            core, -25.0, matrix,
        ).execution_scope == :low_level_research_only

        changed_steps = collect(path.steps)
        first_step = changed_steps[1]
        changed_steps[1] = OperationalV22ArrivalPathStep(
            first_step.start_utc,
            first_step.end_utc,
            first_step.bx_gsm + 1.0,
            first_step.by_gsm,
            first_step.bz_gsm,
            first_step.proton_speed,
            first_step.proton_density,
            first_step.origin,
            first_step.contributing_pair_sha256,
        )
        provisional_path = OperationalV22ArrivalPath(
            path.schema_version,
            path.gate_status,
            path.issue_time_utc,
            path.status,
            path.fallback_reason,
            path.queue_sha256,
            path.artifact_sha256,
            Tuple(changed_steps),
            repeat("0", 64),
        )
        rehashed_path = OperationalV22ArrivalPath(
            provisional_path.schema_version,
            provisional_path.gate_status,
            provisional_path.issue_time_utc,
            provisional_path.status,
            provisional_path.fallback_reason,
            provisional_path.queue_sha256,
            provisional_path.artifact_sha256,
            provisional_path.steps,
            operational_v22_arrival_path_sha256(provisional_path),
        )
        @test verify_operational_v22_arrival_path(rehashed_path)
        @test_throws ArgumentError verify_operational_v22_arrival_path(
            rehashed_path, queue, artifact,
        )
    end

    @testset "post-issue, queued-prefix, and sparse-tail mutations separate" begin
        pairs, issue = _v22a_ready_pairs(future_count=2)
        artifact = _v22a_artifact(0.5; label="prefix-map")
        queue = build_operational_v22_arrival_queue(pairs, issue)
        baseline = build_operational_v22_arrival_path(queue, artifact)

        postissue = _v22a_pair(
            issue + Minute(1);
            issue=issue + Minute(2), receipt=issue + Minute(1) + Second(5),
            sequence=999, label="ignored-path-pair", bx=9999.0,
        )
        post_queue = build_operational_v22_arrival_queue(
            vcat(pairs, [postissue]), issue,
        )
        post_path = build_operational_v22_arrival_path(post_queue, artifact)
        @test post_queue.composite_sha256 == queue.composite_sha256
        @test post_path.composite_sha256 == baseline.composite_sha256

        changed_pairs = copy(pairs)
        future_index = findfirst(pair ->
            pair.measurement_time_utc == _v22a_utc(
                DateTime(2022, 5, 5, 12, 10),
            ), changed_pairs,
        )
        changed_pairs[future_index] = _v22a_rehash_pair(
            changed_pairs[future_index], (
            bx_gsm=-92.0,
            mag_raw_sha256=_v22a_sha("mutated-future-two"),
        ))
        changed_queue = build_operational_v22_arrival_queue(changed_pairs, issue)
        changed_path = build_operational_v22_arrival_path(
            changed_queue, artifact,
        )
        @test changed_path.steps[1] == baseline.steps[1]
        @test changed_path.steps[2].bx_gsm == -92.0
        @test changed_path.steps[2] != baseline.steps[2]
        @test changed_path.steps[3] != baseline.steps[3]

        changed_artifact = _v22a_artifact(0.4; label="changed-tail-map")
        changed_tail = build_operational_v22_arrival_path(
            queue, changed_artifact,
        )
        @test changed_tail.steps[1:2] == baseline.steps[1:2]
        @test changed_tail.steps[3:end] != baseline.steps[3:end]
        @test changed_tail.queue_sha256 == baseline.queue_sha256
        @test changed_tail.artifact_sha256 != baseline.artifact_sha256
        @test changed_tail.composite_sha256 != baseline.composite_sha256
    end

    @testset "transported persistence is explicit on every path failure" begin
        pairs, issue = _v22a_ready_pairs(future_count=2)
        queue = build_operational_v22_arrival_queue(pairs, issue)
        overflow_artifact = _v22a_artifact(
            0.0; intercept=fill(1000.0, 5), label="overflow-tail",
        )
        fallback = build_operational_v22_arrival_path(
            queue, overflow_artifact,
        )
        @test fallback.status == :fallback
        @test fallback.fallback_reason == :sparse_tail_out_of_domain
        @test length(fallback.steps) == 14
        @test getproperty.(fallback.steps, :origin)[1:2] == (:queued, :queued)
        @test all(==(:persistence), getproperty.(fallback.steps, :origin)[3:end])
        @test all(step ->
            (step.bx_gsm, step.by_gsm, step.bz_gsm,
             step.proton_speed, step.proton_density) ==
            (fallback.steps[2].bx_gsm, fallback.steps[2].by_gsm,
             fallback.steps[2].bz_gsm, fallback.steps[2].proton_speed,
             fallback.steps[2].proton_density), fallback.steps[3:end])
        @test size(operational_v22_arrival_path_matrix(fallback)) == (14, 5)

        causal_pairs, _ = _v22a_ready_pairs()
        physical_mutations = (
            (
                reason=:invalid_plasma,
                pairs=let changed=copy(causal_pairs)
                    changed[10] = _v22a_rehash_pair(
                        changed[10], (proton_density=0.0,),
                    )
                    changed
                end,
            ),
            (
                reason=:invalid_vx,
                pairs=let changed=copy(causal_pairs)
                    changed[10] = _v22a_rehash_pair(
                        changed[10], (proton_vx_gse=1.0,),
                    )
                    changed
                end,
            ),
            (
                reason=:delay_out_of_bounds,
                pairs=let changed=copy(causal_pairs)
                    changed[10] = _v22a_rehash_pair(
                        changed[10], (position_x_gse=500.0 * 60.0 * 19.0,),
                    )
                    changed
                end,
            ),
            (
                reason=:overtaking_exceeds_one_bin,
                pairs=let changed=copy(causal_pairs)
                    changed[19] = _v22a_rehash_pair(
                        changed[19], (position_x_gse=500.0 * 60.0 * 120.0,),
                    )
                    changed
                end,
            ),
        )
        for case in physical_mutations
            partial_queue = build_operational_v22_arrival_queue(case.pairs, issue)
            @test partial_queue.status == :fallback
            @test partial_queue.fallback_reason == case.reason
            @test !isempty(partial_queue.transported_pairs)
            @test !isempty(partial_queue.arrival_bins)
            partial_path = build_operational_v22_arrival_path(
                partial_queue, _v22a_artifact(),
            )
            @test partial_path.status == :fallback
            @test partial_path.fallback_reason == case.reason
            @test length(partial_path.steps) == 14
            @test all(==(:persistence), getproperty.(partial_path.steps, :origin))
        end

        base = DateTime(2022, 5, 5)
        gap_pairs, _ = _v22a_ready_pairs(
            future_count=2,
            missing_starts=Set((base + Hour(5), base + Hour(7))),
        )
        gap_queue = build_operational_v22_arrival_queue(gap_pairs, issue)
        gap_path = build_operational_v22_arrival_path(
            gap_queue, _v22a_artifact(),
        )
        @test gap_queue.fallback_reason == :incomplete_history
        @test gap_path.status == :fallback
        @test gap_path.fallback_reason == :incomplete_history
        @test length(gap_path.steps) == 14

        only_second_pairs, _ = _v22a_ready_pairs(future_count=2)
        filter!(pair -> pair.measurement_time_utc !=
            _v22a_utc(DateTime(2022, 5, 5, 12, 5)), only_second_pairs)
        nonprefix_queue = build_operational_v22_arrival_queue(
            only_second_pairs, issue,
        )
        nonprefix = build_operational_v22_arrival_path(
            nonprefix_queue, _v22a_artifact(),
        )
        @test nonprefix_queue.status == :ready
        @test nonprefix.fallback_reason == :future_queue_not_contiguous
        @test all(==(:persistence), getproperty.(nonprefix.steps, :origin))

        no_pairs = build_operational_v22_arrival_queue((), issue)
        unavailable = build_operational_v22_arrival_path(
            no_pairs, _v22a_artifact(),
        )
        @test unavailable.fallback_reason == :no_receipt_eligible_pairs
        @test isempty(unavailable.steps)
        @test_throws ArgumentError operational_v22_arrival_path_matrix(unavailable)

        bad_checksum = OperationalV22ArrivalPath(
            fallback.schema_version,
            fallback.gate_status,
            fallback.issue_time_utc,
            fallback.status,
            fallback.fallback_reason,
            fallback.queue_sha256,
            fallback.artifact_sha256,
            fallback.steps,
            repeat("f", 64),
        )
        @test_throws ArgumentError verify_operational_v22_arrival_path(bad_checksum)
    end
end
