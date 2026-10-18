using Test, LinearAlgebra

@testset "Independent dense Gaussian endpoint conditioning" begin
    for lambdas in ([0.0, 0.0], [1e-14, 1e-9], [0.25, log(2.0)], [100.0, 10000.0]),
        sigmas in ([0.0, 0.0], [1.0, 3.0]), R in (0.125, 1.0, 20.0), observed in (-90.0, 0.0, 70.0)
        mean, P, forcing, delta = [-12.0, 5.0], [9.0 -3.0; -3.0 16.0], [-3.0, 4.0], 1.0
        full = interval_joint(mean, P, lambdas, sigmas, forcing, delta)
        total = [1.0 0 0 0; 0 1 0 0; 0 0 1 1]
        expected_joint = (mean=total*full.mean, covariance=total*full.covariance*total')
        expected = gaussian_condition(expected_joint, [1, 2], [3], [observed], fill(R, 1, 1))
        joint = IMPL.interval_moments(mean, P, lambdas, sigmas, forcing, delta)
        actual = IMPL.condition_interval(joint, observed, R)
        @test compare(actual.mean, expected.mean)
        @test compare(actual.covariance, expected.covariance)
        @test compare(actual.gain, expected_joint.covariance[1:2, 3] ./ (expected_joint.covariance[3,3] + R))
        @test compare(actual.innovation, observed - expected_joint.mean[3])
        @test compare(actual.innovation_variance, expected_joint.covariance[3,3] + R)
        @test compare(actual.covariance, actual.covariance')
        @test minimum(eigvals(Symmetric(actual.covariance))) >= -PEER_ATOL * max(1, opnorm(actual.covariance))
    end
    hand = IMPL.interval_moments([2.0], fill(3.0, 1, 1), [0.0], [2.0], [-6.0], 1.0)
    actual = IMPL.condition_interval(hand, 8.0, 2.0)
    @test actual.mean ≈ [-4 + 135/19] atol=1e-12 rtol=1e-12
    @test actual.covariance ≈ fill(58/19, 1, 1) atol=1e-12 rtol=1e-12
    @test actual.gain ≈ [15/19] atol=1e-13 rtol=1e-13
    signed = IMPL.interval_moments([0.0, 0.0], [1.0 -2.0; -2.0 9.0],
                                  [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], 1.0)
    posterior = IMPL.condition_interval(signed, 1.0, 1.0)
    @test posterior.gain[2] < 0
    @test posterior.covariance[1, 2] < 0
end

@testset "Finite narrowing, invalid covariance and unchanged input boundaries" begin
    for args in ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 0.0),
                 (1.0, 1.0, -1.0), (Inf, 1.0, 1.0), (NaN, 1.0, 1.0),
                 (true, 1.0, 1.0), (1.0, true, 1.0), (1.0, 1.0, true))
        @test_throws ArgumentError IMPL.interval_kernel(args...)
    end
    @test_throws ArgumentError IMPL.interval_kernel(parse(BigFloat, "1e1000"), 1, 1)
    @test_throws ArgumentError IMPL.interval_kernel(1, 1, parse(BigFloat, "1e-1000"))
    m, P, rates, sigmas, forcing = [-12.0, 5.0], [9.0 -3.0; -3.0 16.0], [0.25, 0.125], [1.0, 3.0], [-3.0, 4.0]
    original = deepcopy((m, P, rates, sigmas, forcing))
    moment = IMPL.interval_moments(m, P, rates, sigmas, forcing, 1.0)
    IMPL.condition_interval(moment, -30.0, 1.0)
    @test (m, P, rates, sigmas, forcing) == original
    @test_throws ArgumentError IMPL.interval_moments(m, [1.0 2.0; 2.0 1.0], rates, sigmas, forcing, 1.0)
    @test_throws ArgumentError IMPL.interval_moments(m, [-1.0 0.0; 0.0 1.0], rates, sigmas, forcing, 1.0)
    @test_throws ArgumentError IMPL.interval_moments(m, [1.0 0.0; 0.1 1.0], rates, sigmas, forcing, 1.0)
    @test_throws DimensionMismatch IMPL.interval_moments(m, ones(1,1), rates, sigmas, forcing, 1.0)
    @test_throws DimensionMismatch IMPL.interval_moments(m, P, [1.0], sigmas, forcing, 1.0)
    @test_throws ArgumentError IMPL.condition_interval(moment, -30.0, 0.0)
    @test_throws ArgumentError IMPL.condition_interval(moment, -30.0, parse(BigFloat,"1e-1000"))
    @test_throws ArgumentError IMPL.condition_interval(moment, Inf, 1.0)
    @test_throws ArgumentError IMPL.condition_interval(moment, -30.0, Inf)
    @test_throws ArgumentError IMPL.condition_interval(merge(moment, (interval_variance=-1.0,)), -30.0, 1.0)
    @test_throws ArgumentError IMPL.condition_interval(merge(moment, (endpoint_interval_covariance=[1e3, 1e3],)), -30.0, 1.0)
end
