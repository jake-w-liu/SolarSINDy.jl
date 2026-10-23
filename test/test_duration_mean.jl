module DurationMeanTests
using Test, Dates, SHA, JSON3, LinearAlgebra
using SolarSINDy
const D=isdefined(Main,:DURATION_MEAN_TEST_OVERRIDE) ? Main.DURATION_MEAN_TEST_OVERRIDE : SolarSINDy.DurationConditionedForcing
const A=SolarSINDy.IntervalMeanForecaster
const R=A.DstIntervalAdapter.Replay
const O_PATH=joinpath(@__DIR__,"duration_forcing","mean_precode_oracles_v1.json")
@test bytes2hex(sha256(read(O_PATH)))=="ab5726e0bc4df1843e14f0a58bb272096ea71c61aec2f6e324f26544908886ca"
const O=JSON3.read(read(O_PATH))
const ATOL=5e-11;const RTOL=5e-14 # Frozen prior B contract, not selected from observations.
fraction(x)=begin a,b=split(String(x),'/');parse(BigInt,a)//parse(BigInt,b) end
const BOUNDARY=DateTime(2024,2)
const ISSUE=DateTime(2024,3)
function model(beta)
    cells(x)=ntuple(i->ntuple(j->Int(x[i][j]),5),49)
    D._model(beta,BOUNDARY,Tuple(Float64.(fraction.(O.edges))),cells(O.risks),cells(O.flips),
      Tuple(Int.(O.global_risks)),Tuple(Int.(O.global_flips)),Tuple(Float64.(fraction.(O.marks))),
      Tuple(Int.(O.mark_counts)),repeat("a",64),repeat("b",64))
end
function current(state=:N,age=2,q=0.;issue=ISSUE,status=:available,reason=:available)
    d=(;status,reason,issue,state,age,q,input_prefix_sha256=repeat("c",64),availability_kind=:assumed_completed_hour)
    merge(d,(state_id=D._hash(d),))
end
function forecast(m,c;horizon=6,lambdas=(0.,0.),sigmas=(1.,1.),basis=[1. -2.;3. 4.],
      mean=[-2.,3.],covariance=[2. .5;.5 4.],pressure=0.,noise=0.,full=false)
    cp=R.Checkpoint(c.issue,mean,covariance)
    f=full ? D.affine_rollout : D.mean_affine_rollout
    f(m,c,cp,basis;lambdas,sigmas,pressure,target_observation_variance=noise,horizon)
end

@testset "Pre-code exact full-tree/Brownian mean oracles" begin
    for case in O.cases
        m=model(Int(case.beta));c=current(Symbol(case.state),Int(case.age),Float64(fraction(case.q)))
        result=forecast(m,c);evaluated=D.evaluate_mean_rollout(result,(2.,3.))
        @test result.computation_scope==:point_mean_only && !result.claim_ready && !result.measured_receipts_available
        @test !any(k->occursin("variance",String(k))||k in (:schedule,:paths,:components),keys(evaluated))
        for hour in case.hours
            h=Int(hour.depth)
            expected=Tuple((;state=Symbol(x.state),age=Int(x.age),mark_id=Int(x.mark_id),probability=fraction(x.mass)) for x in hour.states)
            @test result.state_distributions[h]==expected
            @test sum(x.probability for x in result.state_distributions[h])==1
            @test result.qmeans[h]==fraction(hour.qmean)
            @test result.offsets[h]==Float64(fraction(hour.offset))
            @test all(isapprox(result.coefficients[h][k],Float64(fraction(hour.coefficients[k]));atol=ATOL,rtol=RTOL) for k in 1:2)
            @test all(isapprox(result.endpoint_bases[h][j][k],Float64(fraction(hour.endpoint_basis[j][k]));atol=ATOL,rtol=RTOL) for j in 1:2,k in 1:2)
            @test isapprox(evaluated.summaries[h].mean,Float64(fraction(hour.mean));atol=ATOL,rtol=RTOL)
        end
        @test result.peak_states<=31 && result.expanded_states<=112
    end
end

@testset "Independent exhaustive B equivalence: all9 corners, horizons and starts" begin
    for beta in (5,20),state in (:N,:S),age in (2,48,49),taus in ((2.,12.),(4.,24.),(8.,48.)),sigma in (1.,3.,10.)
        m=model(beta);c=current(state,age,state==:S ? 3. : 0.)
        for h in 1:6
            args=(;horizon=h,lambdas=1 ./ taus,sigmas=(sigma,sigma),mean=[-1e4,1e4],
              covariance=[1e4 500.;500. 1e4],basis=[-1e4 1e4;1e4 1e4],pressure=4.,noise=1.)
            meanonly=forecast(m,c;args...);full=forecast(m,c;args...,full=true)
            fullpaths=collect(full.paths) # Same ordered independent branches; avoid tuple-length specialization in the oracle.
            evaluated=D.evaluate_mean_rollout(meanonly,(100.,30.));reference=D.evaluate_rollout(full,(100.,30.))
            for t in 1:h
                @test meanonly.qmeans[t]==sum(p.probability*Rational{BigInt}(p.injections[t]) for p in fullpaths)
                for k in 1:2
                    @test isapprox(meanonly.coefficients[t][k],reference.summaries[t].coefficients[k];atol=ATOL,rtol=RTOL)
                end
                @test meanonly.offsets[t]==full.schedule.offsets[t]
                @test isapprox(evaluated.summaries[t].mean,reference.summaries[t].mean;atol=ATOL,rtol=RTOL)
                for j in 1:2,k in 1:2
                    expected=setprecision(BigFloat,256) do
                        Float64(sum(BigFloat(p.probability)*BigFloat(p.endpoint_bases[t][j][k]) for p in fullpaths))
                    end
                    @test isapprox(meanonly.endpoint_bases[t][j][k],expected;atol=ATOL,rtol=RTOL)
                end
            end
        end
    end
end

@testset "Duplicate centroids and current-q identity" begin
    m=model(5)
    for q in (0.,2.,3.,100.)
        c=current(:S,49,q);out=forecast(m,c)
        @test any(x->x.mark_id==5,out.state_distributions[6])
        @test any(x->x.mark_id==1,out.state_distributions[6])
        @test any(x->x.mark_id==2,out.state_distributions[6])
        tree=forecast(m,c;full=true)
        branches=collect(tree.paths)
        @test out.qmeans==ntuple(h->sum(p.probability*Rational{BigInt}(p.injections[h]) for p in branches),6)
    end
    # The duplicate zero centroids have unequal1:2 masses; IDs must not collapse.
    out=forecast(m,current())
    firstmass=only(x.probability for x in out.state_distributions[1] if x.mark_id==1)
    secondmass=only(x.probability for x in out.state_distributions[1] if x.mark_id==2)
    @test firstmass==2secondmass
end

@testset "Support, source/boundary and numeric failure parity" begin
    m=model(5);c=current();cp=R.Checkpoint(c.issue,[-2.,3.],zeros(2,2))
    call(m,c,cp=cp,basis=zeros(2,2);kw...)=D.mean_affine_rollout(m,c,cp,basis;
      lambdas=(.5,1/12),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.,kw...)
    for status in (:unknown_initial_age,:latest_driver_missing,:latest_driver_unavailable)
        bad=current(;status=:unavailable,reason=status)
        @test call(m,bad).reason==status
    end
    @test_throws ArgumentError call(m,merge(c,(age=3,)))
    @test_throws ArgumentError call(m,current(:N,true,0.))
    @test_throws ArgumentError call(m,current(:N,0,0.))
    @test_throws ArgumentError call(m,current(:N,1,1.))
    # Form the invalid state without hashing NaN in the fixture constructor.
    # Both public paths reject it while JSON3 recomputes the state identity.
    nonfinite=merge(current(:S,1,0.),(q=NaN,))
    @test_throws ErrorException call(m,nonfinite)
    @test_throws ErrorException D.affine_rollout(m,nonfinite,cp,zeros(2,2);
      lambdas=(.5,1/12),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.)
    @test_throws ArgumentError call(m,current(;issue=BOUNDARY-Hour(1)))
    @test_throws ArgumentError call(m,c,R.Checkpoint(ISSUE+Hour(1),[0.,0.],zeros(2,2)))
    @test_throws ArgumentError call(m,c,R.Checkpoint(ISSUE,[0.],zeros(1,1)))
    @test_throws ArgumentError call(m,c,cp,zeros(2,3))
    @test_throws ArgumentError call(m,c,cp,[NaN 0.;0. 0.])
    for h in (0,7,true)
        @test_throws ArgumentError call(m,c;horizon=h)
    end
    for (l,s,p,r) in [((-1.,1.),(1.,1.),0.,0.),((1.,1.),(-1.,1.),0.,0.),
                      ((true,1.),(1.,1.),0.,0.),((1.,1.),(1.,1.),Inf,0.),((1.,1.),(1.,1.),0.,-1.)]
        @test_throws ArgumentError D.mean_affine_rollout(m,c,cp,zeros(2,2);lambdas=l,sigmas=s,pressure=p,target_observation_variance=r)
    end
    zero=ntuple(_->ntuple(_->0,5),49)
    noN=D._model(5,BOUNDARY,(.5,2.,4.),ntuple(i->(0,m.risks[i][2:5]...),49),
      ntuple(i->(0,m.flips[i][2:5]...),49),(0,28),(0,12),nothing,(0,0,0,0),repeat("a",64),repeat("b",64))
    @test call(noN,c).reason==:global_state_support_missing
    @test call(noN,current(:S,1,3.);horizon=1).status==:available
    @test call(noN,current(:S,1,3.);horizon=2).reason==:global_state_support_missing
    nrisks=ntuple(i->(m.risks[i][1],0,0,0,0),49);nflips=ntuple(i->(m.flips[i][1],0,0,0,0),49)
    noS=D._model(5,BOUNDARY,nothing,nrisks,nflips,(17,0),(5,0),(0.,0.,2.,4.),(2,1,1,1),repeat("a",64),repeat("b",64))
    @test call(noS,c;horizon=1).status==:available
    @test call(noS,c;horizon=2).reason==:global_state_support_missing
    insufficient=D._model(5,BOUNDARY,nothing,ntuple(i->(i==1 ? 1 : 0,0,0,0,0),49),
      ntuple(i->(i==1 ? 1 : 0,0,0,0,0),49),(1,0),(1,0),nothing,(1,0,0,0),repeat("a",64),repeat("b",64))
    @test call(insufficient,c).reason==:first_S_mark_support_missing
    good=call(m,c)
    for g in ((-1.,1.),(NaN,1.),(true,1.),(1.,))
        @test_throws ArgumentError D.evaluate_mean_rollout(good,g)
    end
    @test D.evaluate_mean_rollout(call(noN,c),(1.,1.)).reason==:global_state_support_missing
    # Valid finite point means may outlive the full probabilistic covariance domain.
    huge=R.Checkpoint(ISSUE,[0.,0.],[1e308 0.;0. 1e308])
    point=D.mean_affine_rollout(m,c,huge,zeros(2,2);lambdas=(0.,0.),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.)
    @test point.status==:available
    @test_throws ArgumentError D.affine_rollout(m,c,huge,zeros(2,2);lambdas=(0.,0.),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.)
end
end
