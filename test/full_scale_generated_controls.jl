module FullScaleGeneratedControls
using Test
include(joinpath(@__DIR__,"..","examples","point_input_export_synthetic.jl"))
@testset "Full-size admission, calendar counts and corruption rejection" begin
    python=PointInputExportSynthetic._python()
    for name in ("test_full_scale_generated.py","test_harness_evidence.py","test_canonical_harness_artifact.py")
        script=joinpath(@__DIR__,"completed_refit_artifacts",name)
        @test success(Cmd([python,"-B","-X","utf8",script]))
    end
end
end
