module PublisherUPProjectionTests
using Dates,Test,SolarSINDy,SHA
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","publisher_up_projection.jl"))
const U=PublisherUPProjection
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const I=DateTime(2020)
function calibration()
    D.CompletedCalibration(zeros(9),fill(I-Millisecond(1),9);model_schema=D.POINT_SCHEMA,
        point_model_sha256="1"^64,purpose=:synthetic)
end
function source(;missing=false,asof=I+Minute(96))
    mag=Dict(I+Minute(k)=>(3.0k,4.0k) for k in -40:100)
    prepared=D.prepare_schedule(mag,I:Minute(1):I+Minute(65))
    centers=fill(log1p(5.)-1,66);scales=ones(66)
    batches=missing ? NamedTuple[] : D.materialize_receipts(mag,I+Minute(1):Minute(1):I+Minute(95);asof,scheduled=false).batches
    D.CompletedLabelEvents.replay_shared(prepared,centers,scales,batches,calibration();
        subscribers=(E.SubscriberSpec(:source_static,:static;nominal=.5),),stream_id="generated-up",protocol_id="generated-up31-v1",asof)
end
function run_project(s;warmup=(),kind=:upper_positive,phase=:policy,asof=I+Minute(96))
    U.project_shared(s,calibration(),warmup;start=I,asof,nominal=.5,score_kind=kind,phase)
end
@testset "Shared complete-source log and independent rational phase oracle" begin
    s=source();before=deepcopy(s.events);r=run_project(s)
    @test r.phase_period_minutes==31 && r.attempted==r.issued_bounds==r.binary_resolved==66
    @test r.unavailable==r.nonfinite_bounds==r.target_unknown==r.binary_unknown==0
    @test r.own_numeric_study_pass && !r.scientific_coverage_qualified && !r.source_bound_provenance_verified
    @test isequal(s.events,before)
    counts=zeros(Int,31);misses=zeros(Int,31);wealth=fill(big(1)//big(1),31);expected_misses=0
    for (row,update) in zip(r.issued,r.updates)
        # Use the literal UTC-minute remainder, independently of the
        # production phase-assignment function.
        slot=mod(Int(Dates.value(row.issue-DateTime(1970))÷60000),31)+1
        n=counts[slot];m=misses[slot];weight=(2m+1)//(2(n+1))
        radius=max(big(0)//big(1),wealth[slot]*(weight-1//2)/(1//4))
        @test row.slot==slot && isapprox(row.radius,Float64(radius);rtol=4e-14,atol=1e-14)
        error=1>radius
        @test update.issue==row.issue && update.completed_at==row.issue+Minute(30)+Second(45)
        @test update.signed_score==1. && update.score_miss===error
        wealth[slot]*=error ? 2weight : 2(1-weight)
        counts[slot]+=1;misses[slot]+=error;expected_misses+=error
    end
    @test r.score_misses==expected_misses
    @test [state.resolved for state in r.states]==counts && [state.misses for state in r.states]==misses
    @test r.max_pending<=32 && r.full_logs_grow_with_attempts
    @test r.all_attempted_score_enclosure.issued==66 && r.all_attempted_score_enclosure.unknown==0
end
@testset "Unknown labels, terminal-safe sensitivity and all-attempted accounting" begin
    s=source(;missing=true);r=run_project(s)
    @test r.attempted==66 && r.binary_resolved==0 && r.binary_unknown==66 && r.target_unknown==66
    @test r.completed_source_targets==0 && r.censored_source_targets==65 && r.pending_source_targets==1
    @test s.source.censored_targets==65 && s.source.pending_targets==1
    # Strict expiry is one millisecond after the last whole-minute deadline.
    tail=I+Minute(96)+Millisecond(1)
    flushed=run_project(source(;missing=true,asof=tail);asof=tail)
    @test flushed.censored_source_targets==66 && flushed.pending_source_targets==0
    @test flushed.binary_unknown==66 && isequal(flushed.issued,r.issued)
    println("STRICT_EXPIRY_ORACLE cutoff_censored=",r.censored_source_targets," cutoff_pending=",r.pending_source_targets,
        " plus1ms_censored=",flushed.censored_source_targets," plus1ms_pending=",flushed.pending_source_targets)
    @test r.unavailable>0 && r.issued_bounds+r.unavailable==66 && !r.own_numeric_study_pass
    @test r.issued_bound_binary_unknown==r.issued_bounds && r.max_pending<=32
    safe=run_project(s;phase=:safe)
    @test safe.phase_period_minutes==32 && safe.unavailable==0 && safe.issued_bounds==66
    @test safe.target_unknown==66 && safe.binary_unknown==66
    # Diagnostic processing after the last issue changes no issued value.
    prefix=run_project(source(;missing=true,asof=I+Minute(65));asof=I+Minute(65))
    @test isequal(prefix.issued,r.issued)
end
@testset "Strict warmup clocks, variants and unopened future target fields" begin
    s=source();cal=calibration();period=31
    row(id,issue,complete,value)=(;id,issue,label_ready=issue+Minute(30)+Second(45),completed_at=complete,
        phase_period_minutes=period,point_model_sha256=cal.point_model_sha256,policy=cal.policy,mode=cal.mode,signed_score=value)
    first=I-Minute(62);second=I-Minute(31)
    warm=(row(1,first,first+Minute(30)+Second(45),-1.),row(2,second,second+Minute(30)+Second(45),-1.),
        row(3,I-Minute(30),I+Second(45),nothing),row(4,first,I,nothing),row(5,I-Minute(93),I-Millisecond(1),.25))
    upper=run_project(s;warmup=warm);symmetric=run_project(s;warmup=warm,kind=:symmetric_absolute)
    @test upper.warmup_ids==symmetric.warmup_ids==[1,2,5]
    @test any(upper.states[i].misses!=symmetric.states[i].misses for i in 1:31)
    @test_throws ArgumentError run_project(s;warmup=(warm[1],warm[1]))
    bad=merge(warm[1],(;point_model_sha256="0"^64));@test_throws ArgumentError run_project(s;warmup=(bad,))
    bad=merge(warm[1],(;signed_score=NaN));@test_throws ArgumentError run_project(s;warmup=(bad,))
    corrupted=deepcopy(s.events)
    for i in eachindex(corrupted)
        e=corrupted[i]
        e.time>I+Minute(40) && e.kind===:completed_target && (corrupted[i]=merge(e,(;target="UNOPENED_FUTURE_VALUE")))
    end
    future=merge(s,(;events=corrupted))
    @test isequal(run_project(s;asof=I+Minute(40)).issued,run_project(future;asof=I+Minute(40)).issued)
    @test_throws ArgumentError run_project(future)
    @test_throws ArgumentError run_project(s;kind=:unknown)
    @test_throws ArgumentError run_project(s;phase=:unknown)
    missing=source(;missing=true);changed=deepcopy(missing.events)
    where=findfirst(e->e.kind===:censored_target,changed)
    @test where!==nothing
    event=changed[where];changed[where]=merge(event,(;time=event.expires))
    @test_throws ArgumentError run_project(merge(missing,(;events=changed)))
    duplicated=deepcopy(s.events);at=findfirst(e->e.kind===:completed_target,duplicated)
    insert!(duplicated,at+1,duplicated[at])
    @test_throws ArgumentError run_project(merge(s,(;events=duplicated)))
end
@test Sys.maxrss()<=8*1024^3
println("PUBLISHER_UP_SHARED_PROJECTION_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
