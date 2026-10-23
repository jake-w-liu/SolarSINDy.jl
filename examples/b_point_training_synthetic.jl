# Entirely generated1900-start inputs. No archive, fitted actual model, or skill claim.
using SolarSINDy
include(joinpath(@__DIR__,"support","b_point_generated_fixture.jl"))
function run_b_point_training_synthetic()
    mktempdir() do root
        result=BPointSyntheticFixture.run_stage(realpath(root))
        println("Synthetic B duration point stage: 2 laws, 36 gain fits, independent readback of all 36 models; no primary selection.")
        return result.verified.complete
    end
end
run_b_point_training_synthetic()
