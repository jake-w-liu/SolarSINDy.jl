import SHA
import SolarSINDy

@testset "Completed-source package namespace and local reference boundary" begin
    @test CompletedSourceDBDT.CMR === SolarSINDy.CompletedMinuteReplay
    # This additive namespace exposes corrected admission, never the raw-tag builder.
    @test !isdefined(CompletedSourceDBDT, :ground_cases)
    @test !(:ground_cases in names(CompletedSourceDBDT))
    @test !(:DBDTCausalForecaster in names(CompletedSourceDBDT))
    expected = Set([
        :PreparedIssue, :prepare_issue, :prepare_schedule, :label_audit, :case_audit,
        :split_eligible, :bind_predictions, :CompletedCalibration, :static_bounds,
        :replay_phase, :replay_fifo, :CompletedMaximumController, :issue!, :receive!,
        :advance!, :certificate, :materialize_receipts, :replay_receipts,
    ])
    @test Set(filter(!=(:CompletedSourceDBDT), names(CompletedSourceDBDT))) == expected
    # Raw-byte provenance is independent of executable reference calculations.
    reference = joinpath(@__DIR__, "..", "..", "src", "completed_source_dbdt",
                         "reference", "dbdt_causal_forecaster.jl")
    @test isfile(reference)
    @test bytes2hex(SHA.sha256(read(reference))) ==
          "b2c46d28551443dc88c031b0e9cd3bf6b0e24bbba05f81d515e355afdb424922"
    @test realpath(CompletedSourceDBDT.REFERENCE) == realpath(reference)
end
