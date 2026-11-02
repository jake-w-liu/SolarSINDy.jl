"""Execute the fixed Julia dispatcher in-process on its literal synthetic chain."""
module CandidateStageDispatchSynthetic
using SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","run_candidate_stage.jl"))
function run_example()
    mktempdir() do temporary
        root=realpath(temporary)
        request=Dict("operation"=>"generated-chain","evidence_mode"=>"generated","execution_directory"=>root,"inputs"=>Dict{String,Any}())
        result=CandidateNativeStage.dispatch(request)
        @assert result["generated_candidate_fits"]==32 && result["generated_selected_refits"]==6
        @assert result["actual_source_access"]===false && result["actual_fits"]==0
        return (;generated_candidate_fits=result["generated_candidate_fits"],generated_selected_refits=result["generated_selected_refits"],
            actual_source_access=false,actual_fits=0,recursive_Julia_processes=0,native_actual_qualification=false)
    end
end
end
