module ArchiveDstTimingTests
using Test, Dates, DataFrames, CSV
using SolarSINDy

@testset "Archive interval algebra and explicitly assumed or measured chronology" begin
    origin = DateTime(2019, 12, 31, 23)
    for step in (1, 2, 3, 4, 6, 7), delay in (0, 1, 2)
        target = origin + Hour(step)
        kwargs = (;archive_basis=:omni_hour_start_legacy_origin, extra_dst_delay_hours=delay)
        if step <= delay
            @test_throws ArgumentError archive_dst_forecast_timing(origin, target; kwargs...)
        end
        timing = archive_dst_forecast_timing(origin, target; kwargs..., require_positive_completion_lead=false)
        @test timing.raw_anchor_tag_utc == origin
        @test timing.raw_target_tag_utc == target
        @test timing.anchor_hour_complete_utc == DateTime(2020)
        @test timing.reconstructed_issue_time_utc == DateTime(2020) + Hour(delay)
        @test timing.target_hour_start_utc == target
        @test timing.target_hour_complete_utc == target + Hour(1)
        @test timing.completion_lead_hours == step - delay
        @test timing.archive_model_step_hours == step
        @test timing.is_positive_completion_lead == (step > delay)
        @test timing.has_measured_issue_and_anchor_receipt == false
        @test ismissing(timing.anchor_received_at_utc) && ismissing(timing.target_received_at_utc)
        @test timing.legacy_origin_year == 2019 && timing.reconstructed_issue_year == 2020
    end
    basis = (;archive_basis=:omni_hour_start_legacy_origin)
    received = DateTime(2020, 1, 1, 0, 13, 14, 15)
    actual = DateTime(2020, 1, 1, 0, 15, 16, 17)
    target_receipt = DateTime(2020, 1, 1, 2, 7, 8, 9)
    measured = archive_dst_forecast_timing(origin, origin + Hour(2); basis...,
        anchor_received_at=received, actual_issue=actual, target_received_at=target_receipt)
    @test measured.anchor_received_at_utc == received
    @test measured.reconstructed_issue_time_utc == actual
    @test measured.target_received_at_utc == target_receipt
    @test measured.has_measured_issue_and_anchor_receipt
    @test ismissing(measured.assumed_extra_dst_delay_hours)
    @test measured.completion_lead_hours == Dates.value(DateTime(2020, 1, 1, 2) - actual) / 3_600_000
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); archive_basis=:live_noaa)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin; basis...)
    @test_throws ArgumentError archive_dst_forecast_timing(origin + Minute(1), origin + Hour(1); basis...)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis..., extra_dst_delay_hours=-1)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis..., extra_dst_delay_hours=true)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis..., anchor_received_at=received)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis..., actual_issue=actual)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        anchor_received_at=origin, actual_issue=actual)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        anchor_received_at=received, actual_issue=DateTime(2020))
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        anchor_received_at=received, actual_issue=actual, extra_dst_delay_hours=1)
    @test_throws ArgumentError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        target_received_at=origin + Hour(1))
    huge = Int(1) << 59
    exact_issue_ms = BigInt(Dates.value(origin)) + (BigInt(huge) + 1) * 3_600_000
    @test exact_issue_ms > typemax(Int64)
    @test mod(BigInt(huge) * 3_600_000, BigInt(2)^64) == 0
    @test_throws OverflowError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        extra_dst_delay_hours=huge)
    @test_throws OverflowError archive_dst_provenance(DataFrame(issue_time_utc=[origin], model_step_hours=[huge+1]); basis...)
    last_hour_ms = fld(typemax(Int64), 3_600_000) * 3_600_000
    last_hour = DateTime(Dates.UTInstant{Millisecond}(Millisecond(last_hour_ms)))
    @test mod(BigInt(Dates.value(last_hour)), 3_600_000) == 0
    @test BigInt(last_hour_ms) + 3_600_000 > typemax(Int64)
    @test_throws OverflowError archive_dst_forecast_timing(origin, last_hour; basis...)
    @test_throws OverflowError archive_dst_forecast_timing(origin, origin + Hour(1); basis...,
        extra_dst_delay_hours=fld(typemax(Int64), 3_600_000))
end

@testset "Legacy export preserves keys, predictions and measured timestamps" begin
    origin = DateTime(2019, 12, 31, 23)
    table = DataFrame(issue_time_utc=string.([origin, origin + Hour(1)]), model_step_hours=[1, 2],
        forecast_dst_nt=[-81.25, -40.5], observation_dst_nt=[-90.0, -35.0], legacy_fold=[2019, 2020])
    original = deepcopy(table)
    basis = (;archive_basis=:omni_hour_start_legacy_origin)
    export_table = archive_dst_provenance(table; basis...)
    @test table == original
    @test export_table[:, names(original)] == original
    @test export_table.reconstructed_issue_time_utc == [DateTime(2020), DateTime(2020, 1, 1, 1)]
    @test export_table.target_hour_complete_utc == [DateTime(2020, 1, 1, 1), DateTime(2020, 1, 1, 3)]
    @test export_table.completion_lead_hours == [1., 2.]
    @test export_table.legacy_fold == [2019, 2020]
    @test export_table.reconstructed_issue_year == [2020, 2020]
    @test_throws ArgumentError archive_dst_provenance(export_table; basis...)
    @test_throws ArgumentError archive_dst_provenance(table; archive_basis=:live_noaa)
    @test_throws ArgumentError archive_dst_provenance(table; basis..., extra_dst_delay_hours=1)
    audited = archive_dst_provenance(table; basis..., extra_dst_delay_hours=1, require_positive_completion_lead=false)
    @test audited.is_positive_completion_lead == [false, true]
    malformed = deepcopy(table)
    malformed.model_step_hours = [1.5, 2.0]
    @test_throws ArgumentError archive_dst_provenance(malformed; basis...)
    measured = deepcopy(table)
    measured.anchor_receipt = [DateTime(2020, 1, 1, 0, 10), DateTime(2020, 1, 1, 1, 10)]
    measured.actual_issue = measured.anchor_receipt .+ Minute(2)
    measured.target_receipt = [DateTime(2020, 1, 1, 1, 15), DateTime(2020, 1, 1, 3, 15)]
    with_receipts = archive_dst_provenance(measured; basis..., anchor_receipt_column=:anchor_receipt,
        actual_issue_column=:actual_issue, target_receipt_column=:target_receipt)
    @test with_receipts[:, names(measured)] == measured
    @test with_receipts.anchor_received_at_utc == measured.anchor_receipt
    @test with_receipts.reconstructed_issue_time_utc == measured.actual_issue
    @test with_receipts.target_received_at_utc == measured.target_receipt
    mktempdir() do directory
        path = joinpath(directory, "legacy_export.csv")
        CSV.write(path, export_table)
        restored = CSV.read(path, DataFrame)
        @test restored.forecast_dst_nt == original.forecast_dst_nt
        @test restored.issue_time_utc == DateTime.(original.issue_time_utc)
        @test restored.model_step_hours == original.model_step_hours
        @test restored.completion_lead_hours == [1., 2.]
    end
end

include(joinpath(@__DIR__, "..", "validation", "operational", "export_archive_dst_provenance.jl"))

# Independent publication injection after the preflight but before publication.
# The real writer remains unchanged; only this module's read-only digest seam
# schedules an independent competing publisher at an exact reproducible boundary.
const _ARCHIVE_HASH_HOOK = Ref{Function}(path -> nothing)
function _archive_export_sha(path::String)
    _ARCHIVE_HASH_HOOK[](path)
    return open(io -> bytes2hex(sha256(io)), path)
end

@testset "Actual archive exporter preserves source bytes and refuses evidence replacement" begin
    mktempdir() do directory
        source = joinpath(directory, "legacy.csv")
        output = joinpath(directory, "annotated.csv")
        original = DataFrame(issue_time_utc=[DateTime(2015, 3, 17)], model_step_hours=[1],
            prediction_dst_nt=[-89.125], observation_dst_nt=[-92.0])
        CSV.write(source, original)
        source_bytes = read(source)
        basis = (;archive_basis=:omni_hour_start_legacy_origin)
        result = export_archive_dst_provenance(source, output; basis...)
        @test result.rows == 1
        @test read(source) == source_bytes
        exported = CSV.read(output, DataFrame)
        @test exported[:, names(original)] == original
        @test exported.reconstructed_issue_time_utc == [DateTime(2015, 3, 17, 1)]
        @test exported.target_hour_complete_utc == [DateTime(2015, 3, 17, 2)]
        receipts = CSV.read(result.receipt_path, DataFrame)
        @test receipts.value[receipts.entry_type .== "input_sha256"] == [bytes2hex(sha256(source_bytes))]
        @test receipts.value[receipts.entry_type .== "output_sha256"] == [bytes2hex(sha256(read(output)))]
        @test_throws ArgumentError export_archive_dst_provenance(source, source; basis...)
        @test_throws ArgumentError export_archive_dst_provenance(source, output; basis...)
        @test read(source) == source_bytes
        @test_throws ArgumentError _archive_export_main([source, joinpath(directory, "new.csv"), "--live"])
        @test_throws ArgumentError _archive_export_main([source, joinpath(directory, "new.csv"),
            "--archive-basis=omni-hour-start-legacy-origin", "--unknown"])
        @test !isfile(joinpath(directory, "new.csv"))
        custom = rename(original, :issue_time_utc=>:issue_utc, :model_step_hours=>:lead)
        custom_source = joinpath(directory, "custom.csv")
        custom_output = joinpath(directory, "custom-annotated.csv")
        CSV.write(custom_source, custom)
        _archive_export_main([custom_source, custom_output,
            "--archive-basis=omni-hour-start-legacy-origin", "--origin-column=issue_utc", "--step-column=lead"])
        custom_export = CSV.read(custom_output, DataFrame)
        @test custom_export[:, names(custom)] == custom
        @test custom_export.reconstructed_issue_time_utc == [DateTime(2015, 3, 17, 1)]
    end
end

@testset "Concurrent output and receipt publication cannot replace competing bytes" begin
    mktempdir() do directory
        source = joinpath(directory, "source.csv")
        CSV.write(source, DataFrame(issue_time_utc=[DateTime(2015, 3, 17)], model_step_hours=[1], prediction=[-20.0]))
        source_bytes = read(source)
        for collision in (:output, :receipt)
            output = joinpath(directory, "$(collision).csv")
            receipt = output * ".provenance.csv"
            hash_calls = Ref(0)
            _ARCHIVE_HASH_HOOK[] = function(path)
                if collision == :output && path == source
                    hash_calls[] += 1
                    hash_calls[] == 2 && write(output, "independently published output\n")
                elseif collision == :receipt && path == output
                    write(receipt, "independently published receipt\n")
                end
            end
            try
                @test_throws Base.IOError export_archive_dst_provenance(source, output;
                    archive_basis=:omni_hour_start_legacy_origin)
            finally
                _ARCHIVE_HASH_HOOK[] = path -> nothing
            end
            if collision == :output
                @test read(output, String) == "independently published output\n"
                @test !isfile(receipt)
            else
                @test read(receipt, String) == "independently published receipt\n"
            end
            @test read(source) == source_bytes
        end
    end
end
end
