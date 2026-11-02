module CandidateCampaignPackageControls
using Test,SolarSINDy
# This nested bridge includes every frozen descriptor literal, exact23-key/
# canonical-digest assertion and genuine generated no-fit Q.VerifiedCases parity.
include(joinpath(@__DIR__,"..","examples","candidate_campaign_metadata_synthetic.jl"))
@testset "Fixed17 campaign Python controls in the unfiltered package entry" begin
    result=CandidateCampaignMetadataSynthetic.run_package_controls()
    @test result.complete===true && result.methods==17 && result.skipped==0
    @test result.native.native_exit==0 && result.native.timed_out===false
    @test result.Julia_executed===false && result.actual_data_access===false && result.actual_fits==0
    @test isempty(result.changed_controls) && isempty(result.control_read_errors)
    @test_throws ArgumentError CandidateCampaignMetadataSynthetic.run_fixed(:arbitrary_script)
end
end
