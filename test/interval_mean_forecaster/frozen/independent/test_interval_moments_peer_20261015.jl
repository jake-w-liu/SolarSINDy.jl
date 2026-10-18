using Test, LinearAlgebra
BLAS.set_num_threads(1)
if !isdefined(Main, :IndependentGaussianOracle)
    include("IndependentGaussianOracle.jl")
end
using .IndependentGaussianOracle

# This file is included only after the driver binds a held source SHA and loads
# the separate implementation as DstIntervalObserver. No implementation internals
# or scalar OU formulas enter the matrix-exponential expected values.
const IMPL = DstIntervalObserver
const PEER_ATOL = 5e-11
const PEER_RTOL = 5e-11
compare(a, b) = isapprox(a, b; atol=PEER_ATOL, rtol=PEER_RTOL)

@testset "Independent interval state matrix exponential" begin
    for n in (1, 2), rate in (0.0, 1e-14, 1e-9, log(2.0), 0.25, 7.0, 100.0, 10000.0),
        duration in (0.125, 1.0, 3.25), noise in (0.0, 1.0, 3.0)
        m = [-12.0, 5.0][1:n]
        P = [9.0 -3.0; -3.0 16.0][1:n, 1:n]
        lambdas = n == 1 ? [rate] : [rate, rate == 0 ? 0.0 : rate/7]
        sigmas = n == 1 ? [noise] : [noise, noise/2]
        forcing = [-3.0, 4.0][1:n]
        expected = interval_joint(m, P, lambdas, sigmas, forcing, duration)
        actual = IMPL.interval_moments(m, P, lambdas, sigmas, forcing, duration)
        total = zeros(n+1, 2n)
        total[1:n, 1:n] .= Matrix{Float64}(I, n, n)
        total[n+1, n+1:2n] .= 1.0
        expected_mean = total * expected.mean
        expected_cov = total * expected.covariance * total'
        @test compare(actual.endpoint_mean, expected_mean[1:n])
        @test compare(actual.endpoint_covariance, expected_cov[1:n, 1:n])
        @test compare(actual.interval_mean, expected_mean[n+1])
        @test compare(actual.interval_variance, expected_cov[n+1, n+1])
        @test compare(actual.endpoint_interval_covariance, expected_cov[1:n, n+1])
        actual_cov = [actual.endpoint_covariance actual.endpoint_interval_covariance;
                      actual.endpoint_interval_covariance' actual.interval_variance]
        @test compare(actual_cov, actual_cov')
        @test minimum(eigvals(Symmetric(actual_cov))) >= -PEER_ATOL * max(1, opnorm(actual_cov))
    end
    # Valid negative covariance entries must survive; elementwise clipping,
    # absolute values or treating endpoint/integral noise as independent fail.
    signed = IMPL.interval_moments([0.0, 0.0], [1.0 -2.0; -2.0 9.0],
                                  [0.0, 10.0], [0.0, 0.0], [0.0, 0.0], 1.0)
    @test signed.endpoint_interval_covariance[2] < 0
    @test signed.endpoint_covariance[1, 2] < 0
    @test signed.interval_variance > 0
end

@testset "Candidate semigroup including integrated uncertainty" begin
    m, P = [-12.0, 5.0], [9.0 -3.0; -3.0 16.0]
    for rates in ([0.0, 0.0], [1e-14, 1e-9], [log(2.0), 0.125], [100.0, 10000.0]),
        sigmas in ([0.0, 0.0], [1.0, 3.0]), forcing in ([-3.0, 5.0], [0.0, 0.0])
        d1, d2 = 0.375, 2.125
        whole = IMPL.interval_moments(m, P, rates, sigmas, forcing, d1+d2)
        first = IMPL.interval_moments(m, P, rates, sigmas, forcing, d1)
        second = IMPL.interval_moments(first.endpoint_mean, first.endpoint_covariance, rates, sigmas, forcing, d2)
        # Independent deterministic augmented-state responses supply Phi and C.
        basis = [interval_joint(collect(Matrix{Float64}(I, 2, 2)[:,j]), zeros(2, 2),
                                rates, zeros(2), zeros(2), d2).mean for j in 1:2]
        phi = hcat([v[1:2] for v in basis]...)
        C = [sum(v[3:4]) for v in basis]
        cross_means = dot(C, first.endpoint_interval_covariance)
        combined_mean = (d1*first.interval_mean + d2*second.interval_mean)/(d1+d2)
        combined_variance = (d1^2*first.interval_variance + d2^2*second.interval_variance +
                             2d1*d2*cross_means)/(d1+d2)^2
        combined_cross = (d1*phi*first.endpoint_interval_covariance +
                          d2*second.endpoint_interval_covariance)/(d1+d2)
        @test compare(whole.endpoint_mean, second.endpoint_mean)
        @test compare(whole.endpoint_covariance, second.endpoint_covariance)
        @test compare(whole.interval_mean, combined_mean)
        @test compare(whole.interval_variance, combined_variance)
        @test compare(whole.endpoint_interval_covariance, combined_cross)
    end
end
