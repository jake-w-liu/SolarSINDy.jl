module CandidateStageDriverPackageControls
using Test,SolarSINDy,JSON3
include(joinpath(@__DIR__,"completed_refit_artifacts","run_ci_native_chain.jl"))
include(joinpath(@__DIR__,"..","examples","candidate_stage_dispatch_synthetic.jl"))
@testset "All17 fixed Python native driver controls under the bounded owner" begin
    python=CompletedRefitNativeDriver.choose_python()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    parent=realpath(mktempdir(base;prefix="solarsindy-candidate-controls-",cleanup=false))
    output=joinpath(parent,"controls");entry=joinpath(@__DIR__,"candidate_stage_driver","run_controls.py")
    command=addenv(Cmd([python,"-B","-X","utf8",entry,"--output",output]),"PYTHONDONTWRITEBYTECODE"=>"1")
    CompletedRefitNativeDriver._run_support_command(command,parent)
    terminal=JSON3.read(read(joinpath(output,"TERMINAL.json"),String))
    @test terminal.complete===true && terminal.methods==17
    @test terminal.native.native_exit==0 && terminal.native.timed_out===false
    @test terminal.Julia_executed===false && terminal.actual_data_access===false && isempty(terminal.changed_controls)
end
@testset "Real fixed dispatcher generated32fit/6refit chain without recursive Julia" begin
    result=CandidateStageDispatchSynthetic.run_example()
    @test result.generated_candidate_fits==32 && result.generated_selected_refits==6
    @test !result.actual_source_access && result.actual_fits==0 && result.recursive_Julia_processes==0
    D=CandidateStageDispatchSynthetic.CandidateNativeStage
    mktempdir() do temporary
        root=realpath(temporary)
        inspect=Dict("operation"=>"inspect-runtime","evidence_mode"=>"generated","execution_directory"=>root,"inputs"=>Dict{String,Any}())
        @test D.dispatch(inspect)==Dict("actual_source_access"=>false,"actual_fits"=>0)
        @test_throws ArgumentError D.dispatch(merge(inspect,Dict("inputs"=>Dict("caller_data"=>"forbidden"))))
        @test_throws ArgumentError D.dispatch(merge(inspect,Dict("operation"=>"arbitrary-operation")))
    end
end
end
