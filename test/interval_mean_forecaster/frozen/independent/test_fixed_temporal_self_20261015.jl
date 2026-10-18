using Test, LinearAlgebra, Dates
if !isdefined(Main, :IndependentGaussianOracle)
    include("IndependentGaussianOracle.jl")
end
using .IndependentGaussianOracle
include("FixedTemporalGaussianFixture.jl")
using .FixedTemporalGaussianFixture

@testset "Dense temporal Gaussian oracle and affine gain self-checks" begin
    f = fixture()
    @test f.selected[6] == ["revision-bin1", "late-bin2", "initial-bin4", "boundary-bin6"]
    for boundary in 1:8
        zero = expected_posterior(f, boundary; gains=[0.0, 0.0])
        first_basis = expected_posterior(f, boundary; gains=[1.0, 0.0])
        second_basis = expected_posterior(f, boundary; gains=[0.0, 1.0])
        actual = expected_posterior(f, boundary)
        affine = zero.mean + f.gains[1] * (first_basis.mean - zero.mean) +
                 f.gains[2] * (second_basis.mean - zero.mean)
        @test actual.mean ≈ affine atol=5e-11 rtol=5e-11
        @test actual.covariance == zero.covariance == first_basis.covariance == second_basis.covariance
        @test minimum(eigvals(Symmetric(actual.covariance))) >= -5e-11
        for horizon in (1, 4, 6)
            target0 = expected_target(f, boundary, horizon; gains=[0.0, 0.0])
            target1 = expected_target(f, boundary, horizon; gains=[1.0, 0.0])
            target2 = expected_target(f, boundary, horizon; gains=[0.0, 1.0])
            target = expected_target(f, boundary, horizon)
            @test target.mean ≈ target0.mean + f.gains[1] * (target1.mean - target0.mean) +
                  f.gains[2] * (target2.mean - target0.mean) atol=5e-11 rtol=5e-11
            @test target.variance == target0.variance == target1.variance == target2.variance
            @test target.variance >= 0
        end
    end
    # Different-dimensional augmented systems check endpoint/integral semigroup.
    for lambdas in ([0.0, 0.0], [1e-14, 1e-9], [log(2.0), 0.125], [100.0, 10000.0])
        delta1, delta2 = 0.375, 2.125
        direct = interval_joint(f.mean, f.covariance, lambdas, f.sigmas, [-3.0, -5.0], delta1+delta2)
        split = timeline_joint(f.mean, f.covariance, lambdas, f.sigmas,
                               [[-3.0, -5.0], [-3.0, -5.0]], [delta1, delta2])
        Tdirect = [1.0 0 0 0; 0 1 0 0; 0 0 1 1]
        Tsplit = [1.0 0 0 0; 0 1 0 0; 0 0 delta1/(delta1+delta2) delta2/(delta1+delta2)]
        @test Tdirect * direct.mean ≈ Tsplit * split.mean atol=5e-11 rtol=5e-11
        @test Tdirect * direct.covariance * Tdirect' ≈ Tsplit * split.covariance * Tsplit' atol=5e-11 rtol=5e-11
    end
end
