module FullScaleGeneratedControls
using Test
include(joinpath(@__DIR__,"..","examples","point_input_export_synthetic.jl"))
@testset "Standalone entry points load attached and detached header preludes" begin
    for relative in ("examples/point_input_export_resource_synthetic.jl","examples/archive_case_full_scale_agreement.jl",
                     "validation/operational/static_stack_served_replay.jl")
        path=joinpath(@__DIR__,"..",split(relative,'/')...)
        parsed=Meta.parseall(read(path,String);filename=path)
        expressions=filter(x->!(x isa LineNumberNode),parsed.args)
        @test !isempty(expressions)
        # A blank line leaves a header as a valid standalone string. Evaluate
        # that string and the following first code expression in the same scope.
        first_code=findfirst(x->!(x isa AbstractString),expressions)
        @test first_code!==nothing
        scope=Module(gensym(:NativeEntryPrelude))
        for expression in expressions[1:first_code]
            @test_nowarn Core.eval(scope,expression)
        end
    end
end
@testset "Full-size admission, calendar counts and corruption rejection" begin
    python=PointInputExportSynthetic._python()
    for name in ("test_full_scale_generated.py","test_harness_evidence.py","test_canonical_harness_artifact.py")
        script=joinpath(@__DIR__,"completed_refit_artifacts",name)
        @test success(Cmd([python,"-B","-X","utf8",script]))
    end
end
end
