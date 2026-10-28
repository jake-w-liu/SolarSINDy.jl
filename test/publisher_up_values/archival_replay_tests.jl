module ArchivalReplayTests
using Dates,SHA,Test,JSON3,SolarSINDy
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_archival_static.jl"))
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_archival_replay.jl"))
const S=SourceBoundSharedReplay
const AR=SourceBoundArchivalReplay
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const E=SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","archive_source_bound_fit","case_fixture.jl"))
module BeforeArchival
using SHA,SolarSINDy
const FIXTURE=joinpath(@__DIR__,"shared_replay_before_archival_20261028.jl")
const BYTES=read(FIXTURE)
bytes2hex(sha256(BYTES))=="a3a13cd6f2f3cc50e907c6d9931c7b7edd3e0342193af3a9e5d5c413c3e762f8" || error("original generated replay fixture changed")
const ORIGINAL_PATH=joinpath(pkgdir(SolarSINDy),"research","dbdt_completed_source","source_bound_shared_replay.jl")
# include_string's filename supplies source locations; SOURCE_PATH separately
# gives the retained worker its original unchanged dependency directory.
task_local_storage(:SOURCE_PATH,ORIGINAL_PATH) do
    Base.include_string(@__MODULE__,String(copy(BYTES)),ORIGINAL_PATH)
end
end
const OLD=BeforeArchival.SourceBoundSharedReplay
specs()=(E.SubscriberSpec(:static,:static;nominal=.1),E.SubscriberSpec(:phase,:phase;nominal=.1,gamma=.01),
    E.SubscriberSpec(:fifo,:fifo;nominal=.1,gamma=.01/31),E.SubscriberSpec(:early,:early;nominal=.1,gamma=.01/31),
    E.SubscriberSpec(:maturity,:maturity;nominal=.1,gamma=.01/31))
function pointfit(value)
    recipe=C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
    start=START+Minute(40);stop=START+Minute(90);asof=START+Minute(211)
    expected=F.prepare_expected(value,1,recipe;start,stop,asof)
    F.fit_generated(value,1,recipe,expected;start,stop,asof).capsule
end
function calibrated(worker,value,point)
    start=START+Minute(90);stop=START+Minute(135);asof=START+Minute(211);first_issue=START+Minute(150)
    expected=worker.K.prepare_expected(value,1,point;start,stop,asof,first_issue)
    calibration=worker.K.calibrate_generated(value,1,point,expected;start,stop,asof,first_issue)
    static=worker.B.calibrate_generated(value,1,point,calibration,expected)
    (;calibration,expected,static)
end
function replay(worker,value,point,fitted;warmup=true)
    worker.replay_generated(value,1,point,fitted.calibration,fitted.expected;
        start=START+Minute(150),stop=START+Minute(166),asof=START+Minute(211),
        subscribers=specs(),nominal=.1,warmup,static_calibration=fitted.static)
end
function feature_rows(value,point,fitted)
    input=S._inputs(value,1,point,fitted.calibration,fitted.expected,START+Minute(150),START+Minute(166),START+Minute(211))
    [(;row_id=input.row_ids[i],issue=row.issue,accepted=row.accepted,reason=row.reason,
        center=input.centers[i],scale=input.scales[i]) for (i,row) in enumerate(input.prepared)]
end
@testset "Independent typed trace frames retain nonfinite words, symbols and clocks" begin
    oracle=JSON3.read(read(joinpath(@__DIR__,"archival_replay_trace_oracle.json"),String),Dict{String,Any})
    @test oracle["fixture_only"]===true && oracle["actual_source_authority_granted"]===false
    rows=[(;id=1,issue=START,upper=Inf,scale=2.,available=true,reason=:issued),
          (;id=2,issue=START+Minute(1),upper=nothing,scale=-0.,available=false,reason=:prior_label_pending)]
    frames=[R.canonical_bytes(Dict("schema"=>"archival_shared_replay_numeric_trace_v1","request_sha256"=>repeat("0",64))),
        R.canonical_bytes(Dict("section"=>"fixture.rows","records"=>2)),[R.canonical_bytes(AR._trace_value(row)) for row in rows]...]
    @test bytes2hex.(frames)==oracle["frame_hex"]
    @test bytes2hex(codeunits(AR.TRACE_DOMAIN))==oracle["domain_hex"]
    context=AR._trace_start(repeat("0",64));AR._trace_section!(context,"fixture.rows",rows)
    @test bytes2hex(SHA.digest!(context))==oracle["trace_sha256"]=="0117e921a7e64b3a4a9c0f2fbf54ea6b2262773ff91a20a79b53560a173953f5"
    @test digest(hex2bytes(oracle["full_frame_hex"]))==oracle["trace_sha256"]
    for changed in (reverse(rows),[rows[1],merge(rows[2],(;scale=0.))])
        context=AR._trace_start(repeat("0",64));AR._trace_section!(context,"fixture.rows",changed)
        @test bytes2hex(SHA.digest!(context))!=oracle["trace_sha256"]
    end
    @test_throws ArgumentError AR._trace_value(Ref(1))
    @test_throws ArgumentError AR._trace_value(zeros(8193))
    @test_throws ArgumentError AR._trace_start("invalid")
    @test_throws ArgumentError AR._trace_value(Dict("same"=>1,:same=>2))
    unsigned=JSON3.read(read(joinpath(@__DIR__,"archival_replay_unsigned_oracle.json"),String),Dict{String,Any})
    @test unsigned["fixture_only"]===true && unsigned["actual_source_authority_granted"]===false
    for row in unsigned["rows"]
        @test bytes2hex(R.canonical_bytes(AR._trace_value(parse(UInt64,row["value"]))))==row["frame_hex"]
    end
end
@testset "Mandatory static/phase/FIFO inventory is fixed before archival source access" begin
    encoded=AR._specifications(specs(),.1)
    @test length(encoded.specs)==length(encoded.encoded)==5
    @test encoded.encoded[2]["phase_period"]=="policy" && encoded.encoded[2]["mandatory"]===true
    @test encoded.encoded[2]["gamma_bits"]==F.bits(.01)
    @test_throws ArgumentError AR._specifications((E.SubscriberSpec(:phase,:phase),),.1)
    @test_throws ArgumentError AR._specifications((E.SubscriberSpec(:phase,:phase),E.SubscriberSpec(:same,:static)),.1)
    @test_throws ArgumentError AR._specifications((E.SubscriberSpec(:phase,:phase;mandatory=false),E.SubscriberSpec(:fifo,:fifo)),.1)
    @test_throws ArgumentError AR._specifications((E.SubscriberSpec(:same,:phase),E.SubscriberSpec(:same,:fifo)),.1)
    @test_throws ArgumentError AR._specifications(specs(),.2)
end
@testset "Refactored generated replay matches the complete frozen original worker" begin
    mktempdir() do root
        for role in ("primary","independent"),layout in ("ground-only","mixed")
            directory=joinpath(root,role*"-"*layout);mkdir(directory)
            value=make_cases(realpath(directory),role,layout);point=pointfit(value)
            fitted=calibrated(S,value,point);old_fitted=calibrated(OLD,value,point)
            @test fitted.expected.context_sha256==old_fitted.expected.context_sha256
            @test fitted.static.capsule.context==old_fitted.static.capsule.context && fitted.static.capsule.numeric==old_fitted.static.capsule.numeric
            features=feature_rows(value,point,fitted);request_pin=repeat("0",64)
            for warmup in (true,false)
                result=replay(S,value,point,fitted;warmup);reference=replay(OLD,value,point,old_fitted;warmup)
                @test isequal(result.shared.events,reference.shared.events)
                @test isequal(result.shared.subscriber_events,reference.shared.subscriber_events)
                @test isequal(result.shared.issue_records,reference.shared.issue_records)
                @test isequal(result.shared.source,reference.shared.source) && isequal(result.shared.subscribers,reference.shared.subscribers)
                @test result.row_ids==reference.row_ids && result.accepted_rows==reference.accepted_rows
                @test isequal(result.centers,reference.centers) && isequal(result.scales,reference.scales)
                @test isequal(result.static_bounds.bounds,reference.static_bounds.bounds)
                @test result.shared.source.completed_targets==1 && result.shared.source.censored_targets==15 && result.shared.source.pending_targets==0
                for name in (:publisher_upper,:publisher_absolute)
                    portfolio=getproperty(result,name);old=getproperty(reference,name)
                    @test isequal(portfolio.issued,old.issued) && isequal(portfolio.updates,old.updates)
                    @test only(portfolio.updates).completed_at==START+Minute(180)+Second(45)
                end
                fingerprint=AR.result_digest(result,features;request_sha256=request_pin)
                @test fingerprint==AR.result_digest(reference,features;request_sha256=request_pin)
                @test fingerprint!=AR.result_digest(result,features;request_sha256=repeat("1",64))
                before=result.centers[1];result.centers[1]+=1
                @test fingerprint!=AR.result_digest(result,features;request_sha256=request_pin);result.centers[1]=before
                before=result.static_bounds.bounds.raw_split[1];result.static_bounds.bounds.raw_split[1]=before==0. ? 1. : 0.
                @test fingerprint!=AR.result_digest(result,features;request_sha256=request_pin);result.static_bounds.bounds.raw_split[1]=before
                before=result.publisher_upper.issued[1];result.publisher_upper.issued[1]=merge(before,(;upper=before.upper==0. ? 1. : 0.))
                @test fingerprint!=AR.result_digest(result,features;request_sha256=request_pin);result.publisher_upper.issued[1]=before
                snapshot=result.shared.owner.snapshot
                result.shared.owner.snapshot=E.replace_fields(snapshot;attempted=snapshot.attempted+1)
                @test fingerprint!=AR.result_digest(result,features;request_sha256=request_pin);result.shared.owner.snapshot=snapshot
                @test fingerprint==AR.result_digest(result,features;request_sha256=request_pin)
                if role=="primary" && layout=="ground-only"
                    input=S._inputs(value,1,point,fitted.calibration,fitted.expected,START+Minute(150),START+Minute(166),START+Minute(211))
                    identity="generated-source-control/"*repeat("0",64)
                    named=S._execute_common(input;start=START+Minute(150),stop=START+Minute(166),asof=START+Minute(211),
                        specs=specs(),alpha=.1,grace_ms=0,warmup,fixed=result.static_bounds,audit=result.audit,
                        protocol_id="generated-source-bound-shared31-v1",source_id=identity)
                    source_events=filter(event->haskey(event,:raw_identities),named.shared.events)
                    @test !isempty(source_events) && all(event->all(item->item.source_id==identity,event.raw_identities),source_events)
                    without_identities=event->haskey(event,:raw_identities) ? merge(event,(;raw_identities=())) : event
                    @test isequal(without_identities.(named.shared.events),without_identities.(result.shared.events))
                    @test isequal(named.shared.subscriber_events,result.shared.subscriber_events)
                    @test isequal(named.publisher_upper.issued,result.publisher_upper.issued)
                    @test isequal(named.publisher_absolute.updates,result.publisher_absolute.updates)
                    @test isequal(named.shared.source,result.shared.source)
                    clocks=Tuple((;asof=label.asof) for label in value.labels)
                    instrumented=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,clocks,
                        value.request_sha256,value.manifest_sha256,value.complete_sha256,false,false)
                    repeated=replay(S,instrumented,point,fitted;warmup)
                    @test AR.result_digest(repeated,features;request_sha256=request_pin)==fingerprint
                end
            end
            invalid=AR.B.ReverifiedCases(value,"{}\n",AR.B._ReadToken())
            @test_throws ArgumentError AR._read_admission(invalid)
            options=(;start=START+Minute(150),stop=START+Minute(166),asof=START+Minute(211),subscribers=specs(),nominal=.1)
            @test_throws MethodError AR.prepare_request(value,1,value,1,point,fitted.calibration,fitted.expected,fitted.static;options...)
            @test_throws ArgumentError AR.prepare_request(invalid,1,invalid,1,point,fitted.calibration,fitted.expected,fitted.static;options...)
            @test_throws ArgumentError AR.replay_archival(invalid,1,invalid,1,point,fitted.calibration,fitted.expected,fitted.static;options...)
            @test !ispath(joinpath(directory,"REPLAY_ATTEMPT.json"))
        end
        directory=joinpath(root,"missing");mkdir(directory)
        value=make_cases(realpath(directory),"primary","ground-only";profile=:pulse_eval_missing)
        point=pointfit(value);fitted=calibrated(S,value,point);old_fitted=calibrated(OLD,value,point)
        result=replay(S,value,point,fitted);reference=replay(OLD,value,point,old_fitted);features=feature_rows(value,point,fitted)
        @test result.status===:no_feature_eligible && result.audit.features_rejected==16
        @test AR.result_digest(result,features;request_sha256=repeat("0",64))==AR.result_digest(reference,features;request_sha256=repeat("0",64))
    end
end
end
