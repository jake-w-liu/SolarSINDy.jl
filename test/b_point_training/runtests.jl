module BPointTrainingTests
using Test,SolarSINDy,Dates,SHA,JSON3
include(joinpath(@__DIR__,"..","..","examples","support","b_point_generated_fixture.jl"))
const F=BPointSyntheticFixture
const T=F.T
const P=F.P
const B=F.B
const S=F.S
const I=B.BInputs
include("json_depth_checks.jl")
include("core_checks.jl")
include("nonzero_history_checks.jl")
include("protocol_checks.jl")
include("private_context_checks.jl")
include("codec_checks.jl")
include("fit_header_grammar_checks.jl")
include("consistency_checks.jl")
include("basis_integer_checks.jl")
include("reference_checks.jl")
include("selector_checks.jl")
include("qualification_checks.jl")
include("tree_checks.jl")
include("fixture_runtime_checks.jl")
F.with_single_blas() do
@testset "B point training, source replay and separate codecs" begin
    @test fixture_runtime_checks()
    println("B_POINT_STAGE core");flush(stdout)
    @test core_checks()
    println("B_POINT_STAGE nnls_literals");flush(stdout)
    @test nnls_literal_checks()
    println("B_POINT_STAGE nonzero_history_literals");flush(stdout)
    @test nonzero_history_checks()
    println("B_POINT_STAGE nine_task_protocol");flush(stdout)
    @test protocol_checks()
    mktempdir() do root
        println("B_POINT_STAGE tree_inventory");flush(stdout)
        @test tree_checks(realpath(root))
        println("B_POINT_STAGE basis_integer_grammar");flush(stdout)
        @test basis_integer_checks(realpath(root))
        println("B_POINT_STAGE all36_support_failure");flush(stdout)
        @test support_failure_checks(realpath(root))
    end
    mktempdir() do root
        root=realpath(root)
        println("B_POINT_STAGE reference_admission");flush(stdout)
        @test reference_checks(root)
        println("B_POINT_STAGE qualification_grammar");flush(stdout)
        @test qualification_checks(root)
        println("B_POINT_STAGE joint_selector_controls");flush(stdout)
        @test selector_checks(root)
    end
    mktempdir() do root
        root=realpath(root)
        println("B_POINT_STAGE full36_stage");flush(stdout)
        result=F.run_stage(root)
        @test result.stage.gain_fit_count==36&&result.verified.certificate_count==36
        println("B_POINT_STAGE all_provenance_and_codec");flush(stdout)
        @test codec_checks(root,result)
        println("B_POINT_STAGE fit_header_grammar");flush(stdout)
        @test fit_header_grammar_checks(root,result)
        println("B_POINT_STAGE completion_metadata");flush(stdout)
        @test completion_checks(root,result)
        println("B_POINT_STAGE support_journal_point_consistency");flush(stdout)
        @test consistency_checks(root,result)
        @test result.verified.selected_index===nothing
    end
end
end
end
