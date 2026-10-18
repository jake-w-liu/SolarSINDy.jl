using Dates, LinearAlgebra, Test, SHA
include("DstIntervalAdapter.jl")
using .DstIntervalAdapter
@testset "clock andjoint mask" begin
    issue=DateTime(2020)
    for l in (0,1),h in (4,6)
        c=forecast_clock(issue,h,l)
        @test c.driver_tag==DateTime(2019,12,31,23)
        @test c.anchor_tag==issue-Hour(1+l)
        @test c.anchor_receipt==issue
        @test c.target_tag==issue+Hour(h-1)
        @test c.target_complete==issue+Hour(h)
        @test c.target_receipt==issue+Hour(h+l)
        for boundary in (DateTime(2020),DateTime(2023))
            last=boundary-Hour(168+h+l)
            @test fit_eligible(last,last+Hour(h),last+Hour(h+l),boundary)
            @test !fit_eligible(last,last+Hour(h),last+Hour(h+l)+Millisecond(1),boundary)
            @test !fit_eligible(last+Hour(1),last+Hour(h+1),last+Hour(h+l+1),boundary)
        end
    end
    issues=[DateTime(2019,12,24,17),DateTime(2019,12,24,18)]
    @test joint_fit_mask(issues,issues.+Hour(4),issues.+Hour(5),issues.+Hour(6),issues.+Hour(7),DateTime(2020))==[true,false]
    @test_throws ArgumentError forecast_clock(issue,3,0)
    @test_throws ArgumentError forecast_clock(issue,4,2)
    @test_throws ArgumentError forecast_clock(issue+Millisecond(1),4,0)
    @test_throws ArgumentError fit_eligible(issue,issue-Hour(1),issue,DateTime(2021))
    @test length(settings())==18
    @test settings()[1]==(index=1,taus=(2.,12.),sigma=1.,eta=0.)
    @test settings()[18]==(index=18,taus=(8.,48.),sigma=10.,eta=.25)
end
@testset "independent analyticNNLS" begin
    cases=[([1. 0.;0. 1.],[2.,3.],(2.,3.),0.),
        ([1. 0.;0. 1.],[-2.,3.],(0.,3.),4.),
        ([1. 1.;1. -1.],[-1.,-3.],(0.,1.),8.),
        ([-1. 0.;0. -1.],[-2.,-3.],(2.,3.),0.),
        ([1. 1.;1. 1.],[4.,4.],(2.,2.),0.),
        ([1. 0.;1. 0.],[3.,3.],(3.,0.),0.),
        (zeros(2,2),[1.,2.],(0.,0.),5.)]
    for (a,b,g,sse) in cases
        result=nnls2(a,b)
        @test collect(result.gains)≈collect(g) atol=1e-12
        @test result.weighted_sse≈sse atol=1e-25 rtol=1e-12
        @test result.candidates[result.selected_candidate].kkt
    end
    result=nnls2(Matrix{Float64}(I,2,2),[1e-8,1.])
    @test result.gains[1]≈1e-8 atol=1e-20
    @test !result.candidates[3].kkt
    @test result.candidates[result.selected_candidate].kkt
    a=[1. 0.;0. 1.;1. 1.];b=[1.,2.,0.]
    weighted=nnls2(a,b;weights=[7/12,1/3,1/3])
    @test collect(weighted.gains)≈[1/3,5/6] atol=1e-12
    @test weighted.weighted_sse≈7/6 atol=1e-14
    for scale in (1e-100,1e100)
        result=nnls2(a.*scale,b.*scale;weights=[7/12,1/3,1/3])
        @test collect(result.gains)≈collect(weighted.gains) atol=1e-12
        @test result.weighted_sse≈(7/6)*scale^2 rtol=1e-12
    end
    for scale in (1e-200,1e200)
        @test_throws ArgumentError nnls2(a.*scale,b.*scale)
    end
    @test nnls2(zeros(2,2),zeros(2)).gains==(0.,0.)
    @test_throws ArgumentError nnls2([1e308 1e-308;0. 1.],[1.,2.])
    @test_throws ArgumentError nnls2(a,b;weights=[1.,-1.,1.])
    @test_throws ArgumentError nnls2([NaN 1.],[1.])
    @test_throws DimensionMismatch nnls2(ones(3,3),ones(3))
end
println("ADAPTER_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalAdapter.jl")))))
