import SolarSINDy

@testset "Completed-source package namespace and local reference boundary" begin
    @test CompletedSourceDBDT.CMR === SolarSINDy.CompletedMinuteReplay
    # This additive namespace exposes corrected admission, never the raw-tag builder.
    @test !isdefined(CompletedSourceDBDT, :ground_cases)
    @test !(:ground_cases in names(CompletedSourceDBDT))
    @test !(:DBDTCausalForecaster in names(CompletedSourceDBDT))
    # The local reference remains distinct from the public completed-source API.
    reference = joinpath(@__DIR__, "..", "..", "src", "completed_source_dbdt",
                         "reference", "dbdt_causal_forecaster.jl")
    @test isfile(reference)
    @test realpath(CompletedSourceDBDT.REFERENCE) == realpath(reference)
end
