using Test, Dates, SHA
include("DstIntervalReplay.jl")
using .DstIntervalReplay
t=DateTime(2020,1,1)
@testset "immutable inputs, empty chain and transactional failures" begin
    mean=[1.,-2.]
    cov=[2. -.25;-.25 1.]
    c=Checkpoint(t,mean,cov)
    mean[1]=1000
    cov[1,1]=1000
    @test c.mean==(1.,-2.)
    @test c.covariance[1][1]==2
    forcing=[1.,2.]
    b=ForcingInterval("b",t,t+Hour(1),forcing,t+Hour(1))
    forcing[1]=999
    @test b.forcing==(1.,2.)
    l,s=[.1,.5],[1.,2.]
    cache=ReplayCache(c,l,s)
    l[1]=10
    s[1]=10
    @test cache.lambdas==(.1,.5)
    @test cache.sigmas==(1.,2.)
    empty=replay!(cache,ForcingInterval[],MeanObservation[],t)
    @test empty.state===c
    @test length(empty.history)==1
    @test isempty(empty.gains)
    @test empty.replayed_steps==empty.reused_steps==0
    obs=MeanObservation("one","b",t,b.stop,3.,1.,b.stop,b.stop)
    current=replay!(cache,[b],[obs],b.stop;asof=b.stop+Millisecond(1))
    actual=rollout_targets(current.state,cache.lambdas,cache.sigmas,[ScenarioInterval(b.stop,b.stop+Hour(1),[1.,2.])])
    @test length(actual)==1
    for bad in (
        MeanObservation("bad-receipt","b",t,b.stop,3.,1.,b.stop-Millisecond(1),b.stop),
        MeanObservation("bad-verification","b",t,b.stop,3.,1.,b.stop,b.stop-Millisecond(1)),
        MeanObservation("unknown-bin","other",t,b.stop,3.,1.,b.stop,b.stop),
        MeanObservation("wrong-bin","b",t+Millisecond(1),b.stop,3.,1.,b.stop,b.stop))
        @test_throws ArgumentError replay!(cache,[b],[bad],b.stop;asof=b.stop+Millisecond(1))
        @test cache.snapshot===current
    end
    @test_throws ArgumentError replay!(cache,[b],[obs],b.stop;asof=b.stop+Millisecond(1),max_records=0)
    @test_throws ArgumentError replay!(cache,[b],[obs],b.stop;asof=b.stop+Millisecond(1),max_bins=4097)
    @test cache.snapshot===current
    @test_throws ArgumentError rollout_targets(current.state,cache.lambdas,cache.sigmas,[ScenarioInterval(b.stop+Millisecond(1),b.stop+Hour(1),[1.,2.])])
    @test_throws DimensionMismatch rollout_targets(current.state,cache.lambdas,cache.sigmas,[ScenarioInterval(b.stop,b.stop+Hour(1),[1.])])
    # Retrospective replay is explicitly permitted. It does not rewrite issued
    # records or pretend its historical checkpoints were earlier predictions.
    issued=record_forecast!(cache,"fixed",current,[ScenarioInterval(b.stop,b.stop+Hour(1),[1.,2.])])
    back=replay!(cache,ForcingInterval[],MeanObservation[],t)
    @test back.state===c
    @test back.reused_steps==0
    @test cache.issued["fixed"]==issued
    @test_throws ArgumentError record_forecast!(cache,"fixed",back,[ScenarioInterval(t,t+Hour(1),[1.,2.])])
    @test_throws ArgumentError MeanObservation("bad","b",t,b.stop,1.,0.,b.stop,b.stop)
    @test_throws ArgumentError ForcingInterval("",t,b.stop,[1.],b.stop)
    @test_throws ArgumentError ForcingInterval("bad",b.stop,t,[1.],b.stop)
    @test_throws ArgumentError Checkpoint(t,[Inf],reshape([1.],1,1))
end
println("REPLAY_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalReplay.jl")))))
