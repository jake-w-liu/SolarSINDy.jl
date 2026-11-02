"""Run literal untrusted descriptor previews in-process, without native launch."""
module CandidateDescriptorPreviewSynthetic
include(joinpath(@__DIR__,"..","test","candidate_campaign","descriptor_preview_tests.jl"))
function run_example()
    result=CandidateCampaignDescriptorTests.run_literal_fixture()
    @assert result.descriptor_only && result.metadata_unverified && !result.actual_source_bodies_read && result.actual_fits==0 && !result.authority_granted
    println("UNTRUSTED_DESCRIPTOR_PREVIEW_ONLY;ACTUAL_FITS=0;SOURCE_BODY_READS=0;AUTHORITY=false")
    result
end
end
