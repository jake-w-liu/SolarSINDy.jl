using Test, Dates, LinearAlgebra, Random, SHA
include("DstIntervalReplay.jl")
using .DstIntervalReplay
const O=DstIntervalReplay.DstIntervalObserver
t=DateTime(2020,1,1)
mat(x)=[x[i][j] for i in eachindex(x),j in eachindex(x)]

@testset "receipt timing, replacement and immutable issuance" begin
    c=Checkpoint(t,[2.],reshape([3.],1,1))
    bins=[ForcingInterval("b$i",t+Hour(i-1),t+Hour(i),[4.],t+Hour(i)) for i in 1:6]
    observations=[MeanObservation("o$i","b$i",t+Hour(i-1),t+Hour(i),8.,2.,t+Hour(i),t+Hour(i)) for i in 1:6]
    @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,observations,t+Hour(6))
    r=replay_asof(c,[0.],[2.],bins,observations,t+Hour(6);asof=t+Hour(6)+Millisecond(1))
    @test length(r.history)==7
    @test length(r.selected)==6
    @test all(x->x!==nothing,r.selected)
    @test r.replayed_steps==6
    inclusive=replay_asof(c,[0.],[2.],bins,observations,t+Hour(6);receipt_boundary=:hypothetical_inclusive)
    @test inclusive.state.mean==r.state.mean
    @test inclusive.state.covariance==r.state.covariance
    cache=ReplayCache(c,[0.],[2.])
    a=replay!(cache,bins,observations,t+Hour(6);asof=t+Hour(6)+Millisecond(1))
    @test a.state.mean==r.state.mean
    @test a.state.covariance==r.state.covariance
    scenarios=[ScenarioInterval(t+Hour(i),t+Hour(i+1),[4.]) for i in 6:11]
    issued=record_forecast!(cache,"issued-1",a,scenarios)
    @test length(issued.targets)==6
    @test record_forecast!(cache,"issued-1",a,scenarios)==issued
    again=replay!(cache,bins,observations,t+Hour(6);asof=t+Hour(6)+Second(1))
    @test again.reused_steps==6
    @test again.replayed_steps==0
    revision=MeanObservation("revision","b2",t+Hour(1),t+Hour(2),-20.,2.,t+Hour(6)+Second(1),t+Hour(6)+Second(2))
    revised=vcat(observations,[revision])
    revised_result=replay!(cache,bins,revised,t+Hour(6);asof=t+Hour(6)+Second(3))
    fresh=replay_asof(c,[0.],[2.],bins,revised,t+Hour(6);asof=t+Hour(6)+Second(3))
    @test revised_result.state.mean==fresh.state.mean
    @test revised_result.state.covariance==fresh.state.covariance
    @test revised_result.reused_steps==1
    @test revised_result.replayed_steps==5
    @test cache.issued["issued-1"]==issued
    @test_throws ArgumentError record_forecast!(cache,"issued-1",revised_result,scenarios)
    @test_throws ArgumentError record_forecast!(cache,"old-result",a,scenarios)
    fake_state=Checkpoint(revised_result.state.boundary,[12345.],mat(revised_result.state.covariance))
    fake=merge(revised_result,(state=fake_state,))
    @test_throws ArgumentError record_forecast!(cache,"forged-result",fake,scenarios)
    # An older capture verified later does not replace the newest capture.
    oldlate=MeanObservation("old-late","b2",t+Hour(1),t+Hour(2),999.,2.,t+Hour(4),t+Hour(6)+Second(2))
    x=replay_asof(c,[0.],[2.],bins,vcat(revised,[oldlate]),t+Hour(6);asof=t+Hour(6)+Second(3))
    @test x.state.mean==fresh.state.mean
    @test x.selected[2].id=="revision"
    duplicate=MeanObservation("equivalent","b2",revision.start,revision.stop,revision.value,revision.variance,revision.received,revision.verified)
    @test replay_asof(c,[0.],[2.],bins,vcat(revised,[duplicate]),t+Hour(6);asof=t+Hour(6)+Second(3)).state.mean==fresh.state.mean
    conflict=MeanObservation("conflict","b2",revision.start,revision.stop,123.,revision.variance,revision.received,revision.verified)
    @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,vcat(revised,[conflict]),t+Hour(6);asof=t+Hour(6)+Second(3))
    # Equal-clock conflict must reject regardless of a later capture or order.
    oldconflict=MeanObservation("old-conflict","b2",oldlate.start,oldlate.stop,1000.,2.,oldlate.received,oldlate.verified)
    for obs in (vcat(revised,[oldlate,oldconflict]),vcat([oldconflict],revised,[oldlate]))
        @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,obs,t+Hour(6);asof=t+Hour(6)+Second(3))
    end
    badid=MeanObservation(revision.id,revision.bin_id,revision.start,revision.stop,revision.value+1,revision.variance,revision.received,revision.verified)
    @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,vcat(revised,[badid]),t+Hour(6);asof=t+Hour(6)+Second(3))
    prior_snapshot=cache.snapshot
    @test_throws ArgumentError replay!(cache,bins[2:end],revised,t+Hour(6);asof=t+Hour(6)+Second(3))
    @test cache.snapshot===prior_snapshot
    @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,observations,t+Hour(6);asof=t+Hour(5))
    @test_throws ArgumentError replay_asof(c,[0.],[2.],bins,observations,t+Hour(6);asof=t+Hour(7),max_bins=5)
    future=ForcingInterval("future",t+Hour(6),t+Hour(7),[-1e10],t+Hour(7))
    futureobs=MeanObservation("future-obs","future",future.start,future.stop,1e10,9.,t+Hour(7),t+Hour(7))
    for i in 1:6
        boundary=t+Hour(i)
        asof=boundary+Millisecond(1)
        p=replay_asof(c,[0.],[2.],bins[1:i],observations[1:i],boundary;asof)
        full=replay_asof(c,[0.],[2.],vcat(bins,[future]),vcat(observations,[futureobs]),boundary;asof)
        @test p.state.mean==full.state.mean
        @test p.state.covariance==full.state.covariance
        @test p.fingerprint==full.fingerprint
    end
    head=MeanObservation("head-revision","b1",t,t+Hour(1),-123.,2.,t+Hour(6)+Second(3),t+Hour(6)+Second(3))
    head_records=vcat(revised,[head])
    head_result=replay!(cache,bins,head_records,t+Hour(6);asof=t+Hour(6)+Second(4))
    head_fresh=replay_asof(c,[0.],[2.],bins,head_records,t+Hour(6);asof=t+Hour(6)+Second(4))
    @test head_result.reused_steps==0
    @test head_result.replayed_steps==6
    @test head_result.state.mean==head_fresh.state.mean
    @test head_result.state.covariance==head_fresh.state.covariance
    @test cache.issued["issued-1"]==issued
end

@testset "gain basis and future expectation identity" begin
    rng=MersenneTwister(123)
    c=Checkpoint(t,[2.,-5.],[3. -.4;-.4 2.])
    l,s=[0.,log(2)],[1.,2.]
    bins=[ForcingInterval("b$i",t+Hour(i-1),t+Hour(i),randn(rng,2),t+Hour(i)) for i in 1:6]
    obs=[MeanObservation("o$i","b$i",t+Hour(i-1),t+Hour(i),randn(rng)*10,2.,t+Hour(i),t+Hour(i)) for i in (1,3,5)]
    bases=Dict(x.id=>randn(rng,2,2) for x in bins)
    initialbasis=randn(rng,2,2)
    basis=affine_gain_replay(c,l,s,bins,obs,t+Hour(6),bases,initialbasis;asof=t+Hour(6)+Millisecond(1))
    for k in 1:100
        theta=randn(rng,2)*10
        changed=Checkpoint(t,collect(c.mean)+initialbasis*theta,mat(c.covariance))
        altered=[ForcingInterval(b.id,b.start,b.stop,collect(b.forcing)+bases[b.id]*theta,b.received) for b in bins]
        direct=replay_asof(changed,l,s,altered,obs,t+Hour(6);asof=t+Hour(6)+Millisecond(1))
        affine=evaluate_affine(basis,theta)
        @test collect(affine.mean)≈collect(direct.state.mean) atol=5e-12 rtol=5e-12
        @test affine.covariance==direct.state.covariance
    end
    # Arbitrary correlated future forcing paths: averaging target means equals
    # a rollout of componentwise mean forcing at every horizon.
    paths=[[randn(rng,2)*20 for _ in 1:6] for _ in 1:9]
    outcomes=[rollout_targets(basis.base,l,s,[ScenarioInterval(t+Hour(5+i),t+Hour(6+i),path[i]) for i in 1:6]) for path in paths]
    meanpath=[sum(path[i] for path in paths)/length(paths) for i in 1:6]
    average=rollout_targets(basis.base,l,s,[ScenarioInterval(t+Hour(5+i),t+Hour(6+i),meanpath[i]) for i in 1:6])
    for i in 1:6
        @test average[i].mean≈sum(y[i].mean for y in outcomes)/length(outcomes) atol=1e-12
        @test all(y->y[i].variance==average[i].variance,outcomes)
    end
end
println("REPLAY_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalReplay.jl")))))
