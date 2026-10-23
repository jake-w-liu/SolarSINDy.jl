module DurationForcingTests
using Test, Dates, JSON3, SHA, LinearAlgebra
using SolarSINDy
const DurationTestModule = isdefined(Main,:DURATION_TEST_OVERRIDE) ? Main.DURATION_TEST_OVERRIDE : SolarSINDy.DurationConditionedForcing
const DT = DurationTestModule
const DA = SolarSINDy.IntervalMeanForecaster
const DK = DA.DstIntervalAdapter.Kernel
const DR = DA.DstIntervalAdapter.Replay
const DURATION_ORACLE_PATH = joinpath(@__DIR__,"duration_forcing","precode_oracles_v1.json")
@test bytes2hex(sha256(read(DURATION_ORACLE_PATH)))=="5d0f7c930b59e7e19d2597bdc4187e34434478ecb3f2d8684e96d0f0a22de344"
const DURATION_ORACLE = JSON3.read(read(DURATION_ORACLE_PATH))
const DURATION_SOURCE = repeat("a",64)
const DURATION_ATOL = 5e-11
const DURATION_RTOL = 5e-14

duration_fraction(s) = begin
    a,b=split(String(s),'/');parse(BigInt,a)//parse(BigInt,b)
end
function duration_rows(data;shift=Day(0))
    [DT.WindHour(DateTime(String(r.tag))+shift,DateTime(String(r.completed))+shift,
        DateTime(String(r.received))+shift,r.bz,r.q) for r in data]
end
function duration_sequence(states,qs;start=DateTime(2024))
    [DT.WindHour(start+Hour(i-1),start+Hour(i),start+Hour(i),states[i]==:S ? -1. : 1.,qs[i]) for i in eachindex(states)]
end
function duration_example(model,rows;horizon=6,lambdas=(.5,1/12),sigmas=(1.,1.),
        mean=[0.,0.],covariance=zeros(2,2),basis=zeros(2,2),pressure=0.,observation_variance=0.)
    current=DT.current_state(rows,last(rows).completed)
    checkpoint=DR.Checkpoint(current.issue,mean,covariance)
    DT.affine_rollout(model,current,checkpoint,basis;lambdas,sigmas,pressure,
        target_observation_variance=observation_variance,horizon)
end

@testset "Duration forcing independent risk and mark counts" begin
    fixture=DURATION_ORACLE.duration_fixture;rows=duration_rows(fixture.rows)
    boundary=DateTime(String(fixture.boundary))
    for beta in (5,20)
        m=DT.fit_duration(rows,boundary;beta,source_sha256=DURATION_SOURCE)
        @test m.global_risks==Tuple(Int.(fixture.global_risks_N_S))
        @test m.global_flips==Tuple(Int.(fixture.global_flips_N_S))
        @test m.edges==(2.,2.,2.) && m.marks==(2.,2.,2.,2.) && m.mark_counts==(1,1,1,1)
        @test [m.risks[a][2] for a in 1:4]==Int.(fixture.south_risks_age1_4)
        @test [m.flips[a][2] for a in 1:4]==Int.(fixture.south_flips_age1_4)
        expected=getproperty(fixture.hazards_beta5_20_age1_4_empty,Symbol(string(beta)))
        for (j,age) in enumerate((1,2,3,4,20))
            @test DT.hazard(m,:S,age,2.).probability==duration_fraction(expected[j])
        end
        @test DT.hazard(m,:S,20,100.).probability==4//11 # empty q/age cell shrinks globally
        unshrunk=BigInt(1)//BigInt(1)
        for (j,age) in enumerate(2:4)
            unshrunk*=1-BigInt(m.flips[age][2])//BigInt(m.risks[age][2])
            @test unshrunk==duration_fraction(fixture.unshrunk_remaining_S_from_age2[j])
        end
    end
    for c in DURATION_ORACLE.quantile_cases
        edges=DT._edges(Float64.(c.values));@test edges==Tuple(Float64.(c.edges))
        @test [DT._bin(Float64(q),edges) for q in c.queries]==Int.(c.bins)
    end
    for c in DURATION_ORACLE.rank_marks_cases
        marks,counts=DT._marks(Float64.(c.values))
        @test marks==Tuple(Float64.(c.centroids)) && counts==Tuple(Int.(c.counts))
    end
    @test DT._marks([1.,2.,3.])[1]===nothing
    @test_throws ArgumentError DT.fit_duration(rows,boundary;beta=0,source_sha256=DURATION_SOURCE)
    @test_throws ArgumentError DT.fit_duration(rows,boundary;beta=5,source_sha256="not-a-digest")
end

@testset "Duration source censoring, age and cutoff" begin
    boundary=DateTime(2024,2);cutoff=boundary-Hour(168)
    censored=duration_sequence(vcat(fill(:N,49),fill(:S,3)),vcat(zeros(49),fill(2.,3)))
    m=DT.fit_duration(censored,boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test m.global_risks==(1,2) && m.global_flips==(1,0)
    @test m.risks[1][2]==1 && m.risks[2][2]==1 && m.risks[3][2]==0
    @test m.marks===nothing && sum(m.mark_counts)==1
    for n in (48,49,50)
        same=duration_sequence(fill(:S,n),ones(n))
        model=DT.fit_duration(same,boundary;beta=5,source_sha256=DURATION_SOURCE)
        @test model.global_risks==(0,n==50 ? 1 : 0)
        @test model.risks[49][2]==(n==50 ? 1 : 0)
        state=DT.current_state(same,last(same).completed)
        @test state.status==(n<49 ? :unavailable : :available)
        n>=49 && @test state.age==49
    end
    # Only the final South pair crosses the exact mature-receipt boundary.
    rows=duration_sequence(vcat(fill(:N,49),fill(:S,2)),vcat(zeros(49),fill(2.,2));start=cutoff-Hour(51))
    equal=DT.fit_duration(rows,boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test equal.global_risks==(1,1)
    changed=copy(rows);lastrow=last(rows)
    changed[end]=DT.WindHour(lastrow.tag,lastrow.completed,lastrow.received+Millisecond(1),lastrow.bz,lastrow.q)
    late=DT.fit_duration(changed,boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test late.global_risks==(1,0)
    future=DT.WindHour(cutoff,cutoff+Hour(1),cutoff+Hour(1),-1.,100.)
    future_changed=DT.WindHour(cutoff,cutoff+Hour(1),cutoff+Hour(20),1.,999.)
    first=DT.fit_duration(vcat(rows,[future]),boundary;beta=5,source_sha256=DURATION_SOURCE)
    second=DT.fit_duration(vcat(rows,[future_changed]),boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test first.model_id==second.model_id==equal.model_id
    @test first.training_prefix_sha256==second.training_prefix_sha256
    # A missing hour and an absent tag both break the run; they are not known stays.
    broken=copy(censored);r=broken[51]
    broken[51]=DT.WindHour(r.tag,r.completed,r.received,NaN,NaN)
    gap=DT.fit_duration(broken,boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test gap.global_risks==(1,0)
    deleted=DT.fit_duration(vcat(censored[1:50],censored[52:end]),boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test deleted.global_risks==(1,0)
    @test DT.current_state(broken,broken[51].completed).reason==:latest_driver_missing
    @test DT.current_state(censored,last(censored).completed+Hour(1)).reason==:latest_driver_unavailable
    @test_throws ArgumentError DT.fit_duration(reverse(rows),boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test_throws ArgumentError DT.fit_duration(vcat(rows,[last(rows)]),boundary;beta=5,source_sha256=DURATION_SOURCE)
    @test_throws ArgumentError DT.WindHour(DateTime(2024),DateTime(2024)+Hour(1),DateTime(2024),1.,0.)
end

@testset "Exact forcing paths and independent mixture moments" begin
    fixture=DURATION_ORACLE.alternating_fixture
    training=duration_rows(fixture.rows);drivers=duration_rows(fixture.rows;shift=Day(40))
    model=DT.fit_duration(training,DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    @test model.global_risks==(4,4) && model.global_flips==(4,4)
    @test model.marks==(0.,0.,2.,2.) && model.mark_counts==(1,1,1,1)
    raw=duration_example(model,drivers;horizon=2,lambdas=(0.,0.),sigmas=(1.,0.))
    result=DT.evaluate_rollout(raw,(1.,0.));oracle=DURATION_ORACLE.brownian_mixture
    @test result.status==:available && length(result.paths)==4
    @test all(p.probability==1//4 for p in result.paths)
    @test length(unique(p.id for p in result.paths))==4
    @test sort([p.injections for p in result.paths])==[(0.,0.),(0.,0.),(2.,0.),(2.,0.)]
    for h in 1:2
        s=result.summaries[h]
        @test isapprox(s.mean,Float64(duration_fraction(oracle.means_h1_h2[h]));atol=DURATION_ATOL,rtol=DURATION_RTOL)
        @test isapprox(s.conditional_variance,Float64(duration_fraction(oracle.conditional_variance_h1_h2[h]));atol=DURATION_ATOL,rtol=DURATION_RTOL)
        @test isapprox(s.between_forcing_variance,Float64(duration_fraction(oracle.between_variance_h1_h2[h]));atol=DURATION_ATOL,rtol=DURATION_RTOL)
        @test isapprox(s.total_variance,Float64(duration_fraction(oracle.total_variance_h1_h2[h]));atol=DURATION_ATOL,rtol=DURATION_RTOL)
    end
    noisy=DT.evaluate_rollout(duration_example(model,drivers;horizon=2,lambdas=(0.,0.),sigmas=(1.,0.),observation_variance=1.),(1.,0.))
    @test isapprox(noisy.summaries[2].total_variance,10/3;atol=DURATION_ATOL,rtol=DURATION_RTOL)
    @test sum(p.probability for p in result.paths)==1
    @test result.claim_ready==false && result.measured_receipts_available==false
    @test result.variance_scope==:fixed_fitted_parameters_fixed_pressure_plus_explicit_target_noise
    # q must be thresholded before averaging raw drivers; linear A propagation is a separate issue.
    rawdrivers=[0//1,2//1];threshold=1//2
    @test sum(max(x-threshold,0) for x in rawdrivers)/2==3//4
    @test max(sum(rawdrivers)/2-threshold,0)==1//2
    # Every supported stay/flip branch is retained, including duplicate-centroid marks.
    fulltraining=duration_rows(DURATION_ORACLE.duration_fixture.rows)
    full=DT.fit_duration(fulltraining,DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    Ndrivers=duration_rows(DURATION_ORACLE.duration_fixture.rows;shift=Day(40))
    Nresult=duration_example(full,Ndrivers)
    Sresult=duration_example(full,Ndrivers[1:end-1])
    @test length(Nresult.paths)==1093 && Nresult.expanded_nodes==1639
    @test length(Sresult.paths)==547 && Sresult.expanded_nodes==820
    @test sum(p.probability for p in Nresult.paths)==sum(p.probability for p in Sresult.paths)==1
    @test length(unique(p.id for p in Nresult.paths))==1093
    @test all(p.probability>0 for p in Nresult.paths)
    @test DT.hazard(full,:N,1,0.).probability>0 && DT.hazard(full,:N,1,0.).probability<1
    # Unsupported global states and required first-S marks do not become defaults.
    partialtraining=duration_sequence(vcat(fill(:N,49),fill(:S,3),[:N]),vcat(zeros(49),fill(2.,3),[0.]))
    partial=DT.fit_duration(partialtraining,DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    partialdrivers=[DT.WindHour(r.tag+Day(40),r.completed+Day(40),r.received+Day(40),r.bz,r.q) for r in partialtraining]
    @test duration_example(partial,partialdrivers).reason==:first_S_mark_support_missing
    untrained=DT.fit_duration(DT.WindHour[],DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    @test DT.hazard(untrained,:N,1,0.).status==:unavailable
    @test duration_example(untrained,Ndrivers).reason==:global_state_support_missing
    # Unequal rank-block frequencies must survive forecast branching as unequal masses.
    five_states=vcat(fill(:N,49),repeat([:S,:N],5))
    five_q=vcat(zeros(49),[0.,0.,2.,0.,4.,0.,6.,0.,8.,0.])
    five_training=duration_sequence(five_states,five_q)
    five=DT.fit_duration(five_training,DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    five_drivers=duration_sequence(five_states,five_q;start=DateTime(2024)+Day(40))
    five_result=DT.evaluate_rollout(duration_example(five,five_drivers;horizon=1,lambdas=(0.,0.),sigmas=(0.,0.)),(1.,0.))
    @test five.marks==(1.,4.,6.,8.) && five.mark_counts==(2,1,1,1)
    @test [p.probability for p in five_result.paths]==[2//5,1//5,1//5,1//5]
    @test five_result.summaries[1].mean==-13.
    # Later driver values/receipts cannot change an already available current state.
    issue=last(drivers).completed;original=DT.current_state(drivers,issue)
    extra1=DT.WindHour(issue,issue+Hour(1),issue+Hour(1),-1.,100.)
    extra2=DT.WindHour(issue,issue+Hour(1),issue+Hour(10),1.,999.)
    @test DT.current_state(vcat(drivers,[extra1]),issue)==DT.current_state(vcat(drivers,[extra2]),issue)==original
    forged=merge(original,(q=999.,))
    @test_throws ArgumentError DT.affine_rollout(model,forged,DR.Checkpoint(issue,[0.,0.],zeros(2,2)),zeros(2,2);
        lambdas=(.5,1/12),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.)
end
@testset "B affine paths versus full A per-branch Gaussian rollouts" begin
    fixture=DURATION_ORACLE.alternating_fixture;training=duration_rows(fixture.rows)
    drivers=duration_rows(fixture.rows;shift=Day(40))
    model=DT.fit_duration(training,DateTime(2024,2);beta=20,source_sha256=DURATION_SOURCE)
    g=(2.,3.);basis=[.25 -.5;.75 1.25];initial_mean=[-75.,12.]
    initial_covariance=[9. 2.;2. 4.]
    for taus in ((2.,12.),(4.,24.),(8.,48.)),sigma in (1.,3.,10.)
        lambdas=1 ./ collect(taus);sigmas=[sigma,sigma]
        result=DT.evaluate_rollout(duration_example(model,drivers;lambdas,sigmas,basis,
            mean=initial_mean,covariance=initial_covariance,pressure=4.,observation_variance=1.),g)
        @test result.status==:available && length(result.paths)==64
        for component in result.components
            mean=initial_mean+basis*collect(g);covariance=copy(initial_covariance)
            for h in 1:6
                q=component.injections[h]
                joint=DK.interval_moments(mean,covariance,lambdas,sigmas,[-g[1]*q,-g[2]*q],1.)
                target=joint.interval_mean+7.26sqrt(4.)-11.
                @test isapprox(component.interval_means[h],target;atol=DURATION_ATOL,rtol=DURATION_RTOL)
                @test isapprox(component.conditional_variances[h],joint.interval_variance+1.;atol=DURATION_ATOL,rtol=DURATION_RTOL)
                @test all(isapprox(component.endpoint_means[h][j],joint.endpoint_mean[j];atol=DURATION_ATOL,rtol=DURATION_RTOL) for j in 1:2)
                @test all(isapprox(component.endpoint_covariances[h][j][k],joint.endpoint_covariance[j,k];atol=DURATION_ATOL,rtol=DURATION_RTOL) for j in 1:2,k in 1:2)
                mean=joint.endpoint_mean;covariance=joint.endpoint_covariance
            end
        end
    end
    # Independent 100-digit Decimal convolution coefficients and targets, not B-generated labels.
    raw=duration_example(model,drivers;lambdas=(.5,1/12),sigmas=(0.,0.))
    evaluated=DT.evaluate_rollout(raw,g);design=zeros(2,2);residual=zeros(2)
    for (slot,h) in enumerate((4,6))
        expected=getproperty(DURATION_ORACLE.independent_affine_h4_h6,Symbol(string(h)))
        z=Tuple(parse(Float64,String(v)) for v in expected.coefficients_decimal)
        @test all(isapprox(evaluated.summaries[h].coefficients[k],z[k];atol=DURATION_ATOL,rtol=DURATION_RTOL) for k in 1:2)
        target=parse(Float64,String(expected.target_for_gains_2_3_and_P0))
        @test isapprox(evaluated.summaries[h].mean,target;atol=DURATION_ATOL,rtol=DURATION_RTOL)
        design[slot,:].=evaluated.summaries[h].coefficients;residual[slot]=target+11.
    end
    fitted=DA.nnls2(design,residual)
    @test isapprox(fitted.gains[1],2.;atol=DURATION_ATOL,rtol=DURATION_RTOL)
    @test isapprox(fitted.gains[2],3.;atol=DURATION_ATOL,rtol=DURATION_RTOL)
    # A model whose nominal fit boundary is after issue cannot forecast retrospectively.
    olddrivers=training;current=DT.current_state(olddrivers,last(olddrivers).completed)
    @test_throws ArgumentError DT.affine_rollout(model,current,DR.Checkpoint(current.issue,[0.,0.],zeros(2,2)),zeros(2,2);
        lambdas=(.5,1/12),sigmas=(1.,1.),pressure=0.,target_observation_variance=0.)
end

@testset "All1093 forcing leaves versus full A and hold-q delta operator" begin
    fixture=DURATION_ORACLE.duration_fixture
    model=DT.fit_duration(duration_rows(fixture.rows),DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    drivers=duration_rows(fixture.rows;shift=Day(40));current=DT.current_state(drivers,last(drivers).completed)
    g=(2.,3.);lambdas=[.5,1/12];sigmas=[3.,3.];basis=[.25 -.5;.75 1.25]
    initial_mean=[-75.,12.];initial_covariance=[9. 2.;2. 4.]
    result=DT.evaluate_rollout(duration_example(model,drivers;lambdas,sigmas,basis,
        mean=initial_mean,covariance=initial_covariance,pressure=4.,observation_variance=1.),g)
    @test length(result.components)==1093 && result.expanded_nodes==1639
    @test sum(c.probability for c in result.components)==1
    hold=Float64[];mean=initial_mean+basis*collect(g);covariance=copy(initial_covariance)
    for _ in 1:6
        joint=DK.interval_moments(mean,covariance,lambdas,sigmas,[-g[1]*current.q,-g[2]*current.q],1.)
        push!(hold,joint.interval_mean+7.26sqrt(4.)-11.)
        mean=joint.endpoint_mean;covariance=joint.endpoint_covariance
    end
    kernels=[DK.interval_kernel(lambdas[k],sigmas[k],1.) for k in 1:2]
    for component in result.components
        mean=initial_mean+basis*collect(g);covariance=copy(initial_covariance)
        for h in 1:6
            q=component.injections[h]
            joint=DK.interval_moments(mean,covariance,lambdas,sigmas,[-g[1]*q,-g[2]*q],1.)
            @test isapprox(component.interval_means[h],joint.interval_mean+7.26sqrt(4.)-11.;atol=DURATION_ATOL,rtol=DURATION_RTOL)
            @test isapprox(component.conditional_variances[h],joint.interval_variance+1.;atol=DURATION_ATOL,rtol=DURATION_RTOL)
            @test all(isapprox(component.endpoint_means[h][k],joint.endpoint_mean[k];atol=DURATION_ATOL,rtol=DURATION_RTOL) for k in 1:2)
            @test all(isapprox(component.endpoint_covariances[h][i][j],joint.endpoint_covariance[i,j];atol=DURATION_ATOL,rtol=DURATION_RTOL) for i in 1:2,j in 1:2)
            adjustment=sum(g[k]*sum((j==h ? -kernels[k].forcing_mean :
                -kernels[k].mean_coefficient*kernels[k].phi^(h-j-1)*kernels[k].forcing_endpoint)*
                (component.injections[j]-current.q) for j in 1:h) for k in 1:2)
            @test isapprox(component.interval_means[h],hold[h]+adjustment;atol=DURATION_ATOL,rtol=DURATION_RTOL)
            mean=joint.endpoint_mean;covariance=joint.endpoint_covariance
        end
    end
end

@testset "Duration artifact roundtrip and fail-closed guards" begin
    training=duration_rows(DURATION_ORACLE.duration_fixture.rows)
    model=DT.fit_duration(training,DateTime(2024,2);beta=5,source_sha256=DURATION_SOURCE)
    drivers=duration_rows(DURATION_ORACLE.duration_fixture.rows;shift=Day(40))
    mktempdir() do temporary
        artifact=DT.save_model(joinpath(temporary,"model"),model)
        loaded=DT.load_model(artifact.directory;expected_manifest_sha256=artifact.manifest_sha256)
        @test loaded.model_id==model.model_id && DT._data(loaded)==DT._data(model)
        before=DT.evaluate_rollout(duration_example(model,drivers),(2.,3.))
        after=DT.evaluate_rollout(duration_example(loaded,drivers),(2.,3.))
        @test before.summaries==after.summaries
        @test [p.probability for p in before.paths]==[p.probability for p in after.paths]
        @test_throws ArgumentError DT.save_model(artifact.directory,model)
        @test_throws ArgumentError DT.load_model(artifact.directory;expected_manifest_sha256=repeat("0",64))
        relocated=joinpath(temporary,"relocated");cp(artifact.directory,relocated)
        @test DT.load_model(relocated;expected_manifest_sha256=artifact.manifest_sha256).model_id==model.model_id
        open(joinpath(relocated,"extra"),"w") do io;write(io,"unexpected");end
        @test_throws ArgumentError DT.load_model(relocated;expected_manifest_sha256=artifact.manifest_sha256)
        corrupted=joinpath(temporary,"corrupted");cp(artifact.directory,corrupted)
        open(joinpath(corrupted,"model.json"),"a") do io;write(io," ");end
        @test_throws ArgumentError DT.load_model(corrupted;expected_manifest_sha256=artifact.manifest_sha256)
        incomplete=joinpath(temporary,"incomplete");cp(artifact.directory,incomplete)
        rm(joinpath(incomplete,"COMMITTED.json"))
        @test_throws ArgumentError DT.load_model(incomplete;expected_manifest_sha256=artifact.manifest_sha256)
        semantic=joinpath(temporary,"semantic");cp(artifact.directory,semantic)
        data=JSON3.read(read(joinpath(semantic,"model.json")),Dict{String,Any})
        data["global_risks"][1]+=1
        body=Vector{UInt8}(codeunits(JSON3.write(data)));write(joinpath(semantic,"model.json"),body)
        ledger=Vector{UInt8}(codeunits(JSON3.write((;schema="duration_model_manifest_v1",model_sha256=bytes2hex(sha256(body))))))
        write(joinpath(semantic,"MANIFEST.json"),ledger);pin=bytes2hex(sha256(ledger))
        write(joinpath(semantic,"COMMITTED.json"),JSON3.write((;manifest_sha256=pin)))
        @test_throws ArgumentError DT.load_model(semantic;expected_manifest_sha256=pin)
        failed=joinpath(temporary,"failed-publication")
        @test_throws EOFError DT._save_model(failed,model,()->throw(EOFError()))
        @test !isfile(joinpath(failed,"COMMITTED.json")) && isfile(joinpath(failed,"COMMITTED.json.incomplete"))
        @test !isdir(failed*".claim")
        @test_throws ArgumentError DT.load_model(failed;expected_manifest_sha256=artifact.manifest_sha256)
        source_copy=joinpath(temporary,"duration_source_copy.jl")
        cp(joinpath(@__DIR__,"..","src","duration_conditioned_forcing.jl"),source_copy)
        host=Module(:DurationSourceGuardFixture)
        Core.eval(host,:(const IntervalMeanForecaster = $DA))
        Base.include(host,source_copy);other=Base.invokelatest(()->getfield(host,:DurationConditionedForcing))
        @test last(Base.invokelatest(()->getfield(other,:source_identity)())).second==bytes2hex(sha256(read(source_copy)))
        open(source_copy,"a") do io;write(io,"\n# owned post-load mutation witness\n");end
        @test_throws ArgumentError Base.invokelatest(()->getfield(other,:source_identity)())
    end
end
@testset "Bounded known-next source counts in model and artifact" begin
    oraclepath=joinpath(@__DIR__,"duration_forcing","artifact_transition_bound_v1.json")
    @test bytes2hex(sha256(read(oraclepath)))=="ef64f58ac9a2b4ba4f7975d7e0a2d6ca43be78f68cec33b629c2612cfddeee33"
    oracle=JSON3.read(read(oraclepath))
    @test DT.MAX_HOURS==Int(oracle.max_source_hours)
    @test Int(oracle.max_combined_known_next_risks)==DT.MAX_HOURS-2
    zero_cells=ntuple(_->ntuple(_->0,5),49)
    boundary=DateTime(2024,2)
    # A valid universal-boundary count pattern: initial S, then N until right censoring.
    boundary_risks=ntuple(age->(age<49 ? 1 : DT.MAX_HOURS-50,0,0,0,0),49)
    allowed=DT._model(5,boundary,nothing,boundary_risks,zero_cells,(DT.MAX_HOURS-2,0),(0,0),
        nothing,(0,0,0,0),DURATION_SOURCE,DURATION_SOURCE)
    @test sum(allowed.global_risks)==499998
    beyond=ntuple(age->(age<49 ? 1 : DT.MAX_HOURS-49,0,0,0,0),49)
    @test_throws ArgumentError DT._model(5,boundary,nothing,beyond,zero_cells,(DT.MAX_HOURS-1,0),(0,0),
        nothing,(0,0,0,0),DURATION_SOURCE,DURATION_SOURCE)
    impossible_risks=ntuple(age->ntuple(slot->age==49&&slot in (1,2) ? 300000 : 0,5),49)
    fields=(5,boundary,boundary-Hour(168),(1.,1.,1.),impossible_risks,zero_cells,(300000,300000),(0,0),
        nothing,(0,0,0,0),DURATION_SOURCE,DURATION_SOURCE,DT._source_signature())
    provisional=DT.DurationModel(fields...,repeat("0",64))
    impossible=DT.DurationModel(fields...,DT._hash(DT._data(provisional)))
    @test_throws ArgumentError DT._validate(impossible)
    @test_throws ArgumentError DT._model(5,boundary,(1.,1.,1.),impossible_risks,zero_cells,(300000,300000),(0,0),
        nothing,(0,0,0,0),DURATION_SOURCE,DURATION_SOURCE)
    mktempdir() do temporary
        @test_throws ArgumentError DT.save_model(joinpath(temporary,"impossible"),impossible)
        @test !ispath(joinpath(temporary,"impossible")) && !ispath(joinpath(temporary,"impossible.claim"))
        published=DT.save_model(joinpath(temporary,"rehashed"),allowed)
        # Even self-consistent byte hashes and a caller-supplied forged pin cannot bypass semantics.
        body=Vector{UInt8}(codeunits(JSON3.write(merge(DT._data(impossible),(model_id=impossible.model_id,)))))
        write(joinpath(published.directory,"model.json"),body)
        manifest=Vector{UInt8}(codeunits(JSON3.write((;schema="duration_model_manifest_v1",model_sha256=bytes2hex(sha256(body))))))
        write(joinpath(published.directory,"MANIFEST.json"),manifest);pin=bytes2hex(sha256(manifest))
        write(joinpath(published.directory,"COMMITTED.json"),JSON3.write((;manifest_sha256=pin)))
        @test_throws ArgumentError DT.load_model(published.directory;expected_manifest_sha256=pin)
    end
end
end # module
