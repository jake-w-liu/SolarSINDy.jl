using Test, LinearAlgebra, SHA
include("DstIntervalObserver.jl")
using .DstIntervalObserver

@testset "independent scalar and finite-domain checks" begin
    k = interval_kernel(0,2,3)
    @test k.phi == k.mean_coefficient == 1
    @test k.forcing_endpoint == 3
    @test k.forcing_mean == 1.5
    @test k.endpoint_process_variance == 12
    @test k.endpoint_mean_process_covariance == 6
    @test k.mean_process_variance == 4
    k = interval_kernel(log(2),1,1)
    @test k.endpoint_process_variance ≈ 0.5410106403333613 atol=2e-16
    @test k.endpoint_mean_process_covariance ≈ 0.26017112262570097 atol=2e-16
    @test k.mean_process_variance ≈ 0.20463103903254207 atol=2e-16
    for l in (0.,1e-300,1e-12,0.00999,0.01,1.,1000.)
        k = interval_kernel(l,1,1)
        @test all(isfinite,values(k))
        @test eigmin(Symmetric([k.endpoint_process_variance k.endpoint_mean_process_covariance; k.endpoint_mean_process_covariance k.mean_process_variance])) >= -1e-15
    end
    for args in ((-1,1,1),(1,-1,1),(1,1,0),(1,1,-1),(NaN,1,1),(1,Inf,1),(true,1,1))
        @test_throws ArgumentError interval_kernel(args...)
    end
    @test_throws ArgumentError interval_kernel(big"1e-1000",1,1)
    @test_throws ArgumentError interval_kernel(1,1e308,1)
    j = interval_moments([2.],reshape([3.],1,1),[0.],[2.],[4.],3.)
    @test j.endpoint_mean == [14.]
    @test j.interval_mean == 8
    @test j.endpoint_covariance == reshape([15.],1,1)
    @test j.interval_variance == 7
    @test j.endpoint_interval_covariance == [9.]
    c = condition_interval(j,10,2)
    @test c.mean == [16.]
    @test c.covariance == reshape([6.],1,1)
    @test c.gain == [1.]
    for p in ([-1.;;],[1. 2.;2. 1.],[1e-300 2e-300;2e-300 1e-300],[1. 0.2;0.3 1.],[1e308 -1e308;1e308 1e308])
        n = size(p,1)
        @test_throws ArgumentError interval_moments(zeros(n),p,ones(n),ones(n),zeros(n),1)
    end
    j = interval_moments([1.,-2.],[2. -0.5;-0.5 1.],[0.,1.],[0.,0.],[0.,0.],1.)
    @test j.endpoint_covariance[1,2] < 0
    @test j.endpoint_interval_covariance[2] > 0
    @test_throws ArgumentError condition_interval(j,1,0)
    @test_throws DimensionMismatch interval_moments([0.],zeros(2,2),[0.],[0.],[0.],1)
end
println("SOURCE_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalObserver.jl")))))
