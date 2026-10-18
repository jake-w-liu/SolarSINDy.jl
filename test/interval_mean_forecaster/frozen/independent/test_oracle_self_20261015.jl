using Test, LinearAlgebra
include("IndependentGaussianOracle.jl")
using .IndependentGaussianOracle

@testset "Independent matrix exponential oracle self-checks" begin
    for delta in (0.125, 1.0, 3.25), sigma in (0.0, 1.0, 3.0), f in (-7.5, 0.0, 4.0)
        result = interval_joint([-2.0], zeros(1, 1), [0.0], [sigma], [f], delta)
        @test result.mean ≈ [-2 + f * delta, -2 + f * delta/2] atol=1e-12 rtol=1e-12
        @test result.covariance ≈ sigma^2 * delta * [1 1/2; 1/2 1/3] atol=1e-12 rtol=1e-12
    end
    fixed = interval_joint([0.0], zeros(1, 1), [log(2.0)], [1.0], [0.0], 1.0)
    @test fixed.covariance[1, 1] ≈ 0.5410106403333613 atol=1e-14 rtol=1e-13
    @test fixed.covariance[1, 2] ≈ 0.26017112262570097 atol=1e-14 rtol=1e-13
    @test fixed.covariance[2, 2] ≈ 0.20463103903254207 atol=1e-14 rtol=1e-13
    # Hand-solved nonzero measurement-noise conditioning of a Brownian interval.
    # Prior x0=2, Var(x0)=3, sigma=2, delta=1, f=-6:
    # [x1,mean] mean[-4,-1], covariance[7 5;5 13/3]. R=2 => V=19/3.
    brownian = interval_joint([2.0], fill(3.0, 1, 1), [0.0], [2.0], [-6.0], 1.0)
    @test brownian.mean ≈ [-4.0, -1.0] atol=1e-13
    @test brownian.covariance ≈ [7.0 5.0; 5.0 13/3] atol=1e-13
    posterior = gaussian_condition(brownian, [1], [2], [8.0], fill(2.0, 1, 1))
    @test posterior.mean ≈ [-4 + 135/19] atol=1e-12
    @test posterior.covariance ≈ fill(58/19, 1, 1) atol=1e-12
    # One global Brownian state with two integrals. Their covariance 1/2 follows
    # directly by integrating min(s,t) over [0,1]x[1,2], not a filter recurrence.
    timeline = timeline_joint([0.0], zeros(1, 1), [0.0], [1.0], [[0.0], [0.0]], [1.0, 1.0])
    @test timeline.covariance ≈ [2.0 0.5 1.5; 0.5 1/3 0.5; 1.5 0.5 4/3] atol=1e-12
end
