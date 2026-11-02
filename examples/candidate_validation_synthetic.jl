module CandidateValidationSynthetic
using SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","candidate_validation.jl"))
include(joinpath(@__DIR__,"..","test","candidate_validation","generated_fixture.jl"))
function run_example()
    mktempdir() do temporary
        result=CandidateValidationGeneratedFixture.run_chain(SourceBoundCandidateValidation,realpath(temporary))
        @assert length(result.stations)==2
        @assert all(s->length(s.verified)==16 && length(s.refits)==3 && !s.decision.actual,result.stations)
    end
    println("Generated fixed-grid candidate fits, independent score reconstruction and three-family refits completed; actual archival fits=0.")
    nothing
end
end
