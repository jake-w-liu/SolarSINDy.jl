module FullScaleGeneratedControls
using Test
include(joinpath(@__DIR__,"..","examples","point_input_export_synthetic.jl"))
@testset "Standalone native resource entry points load their preludes" begin
    for name in ("point_input_export_resource_synthetic.jl","archive_case_full_scale_agreement.jl")
        path=joinpath(@__DIR__,"..","examples",name)
        parsed=Meta.parseall(read(path,String);filename=path)
        expressions=filter(x->!(x isa LineNumberNode),parsed.args)
        @test !isempty(expressions)
        @test Core.eval(Module(gensym(:NativeResourcePrelude)),first(expressions))===nothing
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
