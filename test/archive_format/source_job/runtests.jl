module ArchiveCatalogSourceJobContract
include("contract_body.jl")
@testset "Pinned archive source reconstruction" begin
    fixture = joinpath(@__DIR__,"fixtures","FROZEN_GENERATED_FIXTURES_AND_ORACLES_V1.json")
    for role in ("primary","independent"), layout in ("ground-only","mixed")
        mktempdir() do directory
            @testset "$role / $layout" begin
                result = source_job_contract(realpath(directory),role,layout,fixture)
                @test result["complete"]
                @test result["actual_fits"]==0
            end
        end
    end
end
end
