module SourceBoundSharedReplayTests
using Dates,Test,SHA,SolarSINDy
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
const S=SourceBoundSharedReplay
const K=S.K
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","archive_source_bound_fit","case_fixture.jl"))
include("static_tests.jl")
function fitted(value)
    recipe=C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
    train_start=START+Minute(40);train_stop=START+Minute(90);asof=START+Minute(211)
    pe=F.prepare_expected(value,1,recipe;start=train_start,stop=train_stop,asof)
    point=F.fit_generated(value,1,recipe,pe;start=train_start,stop=train_stop,asof).capsule
    start=START+Minute(90);stop=START+Minute(135);first_issue=START+Minute(150)
    ce=K.prepare_expected(value,1,point;start,stop,asof,first_issue)
    calibration=K.calibrate_generated(value,1,point,ce;start,stop,asof,first_issue)
    static=S.B.calibrate_generated(value,1,point,calibration,ce)
    point,calibration,ce,static
end
specs()=(E.SubscriberSpec(:static,:static;nominal=.1),E.SubscriberSpec(:phase,:phase;nominal=.1,gamma=.01),
    E.SubscriberSpec(:fifo,:fifo;nominal=.1,gamma=.01/31),E.SubscriberSpec(:early,:early;nominal=.1,gamma=.01/31),
    E.SubscriberSpec(:maturity,:maturity;nominal=.1,gamma=.01/31))
function replay(value,point,calibration,expected;index=1,start=START+Minute(150),warmup=true,static_calibration=nothing)
    S.replay_generated(value,index,point,calibration,expected;start,stop=START+Minute(166),asof=START+Minute(211),
        subscribers=specs(),nominal=.1,warmup,static_calibration)
end
@testset "Verified source/point/calibration bridge with independent integer completion counts" begin
    mktempdir() do root
        for role in ("primary","independent"),layout in ("ground-only","mixed")
            directory=joinpath(root,role*"-"*layout);mkdir(directory)
            value=make_cases(realpath(directory),role,layout)
            point,calibration,expected,static=fitted(value);before=F.model_bytes(point.model)
            # The external pins identify the retained nonempty bytes, not a
            # vector emptied by conversion to String during construction.
            @test static.expected.context_sha256==digest(codeunits(static.capsule.context))
            @test static.expected.numeric_sha256==digest(codeunits(static.capsule.numeric))
            @test static.expected.context_sha256!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            @test static.expected.numeric_sha256!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            result=replay(value,point,calibration,expected;static_calibration=static)
            @test result.status===:replayed && result.audit.scheduled==result.audit.features_accepted==16
            @test result.audit.features_rejected==0 && isempty(result.audit.rejection_reasons)
            @test result.audit.mandatory_static_inventory_complete && result.static_bounds.methods==SB.METHODS
            @test all(c->c.issued==16,values(result.static_bounds.counts))
            @test !result.static_bounds.actual_source_authority_granted && !result.static_bounds.scientific_coverage_qualified
            @test all(==(log(6.)),result.centers) && all(==(1+log(11.)),result.scales)
            # The literal fixture ends at minute180. Only issue150 has all
            # target centers151:180; issues151:165 lack at least one source.
            @test result.shared.source.completed_targets==1 && result.shared.source.censored_targets==15
            @test result.shared.source.pending_targets==0 && result.shared.source.target_unknown==15
            @test length(result.shared.subscribers)==5 && all(c->c.attempted==16,result.shared.subscribers)
            for portfolio in (result.publisher_upper,result.publisher_absolute)
                @test portfolio.phase_period_minutes==31 && portfolio.attempted==portfolio.issued_bounds==16
                @test portfolio.binary_resolved==1 && portfolio.binary_unknown==15 && portfolio.target_unknown==15
                @test portfolio.warmup_ids==collect(1:15) && sum(s.resolved for s in portfolio.states)==16
                @test only(portfolio.updates).target==10. && only(portfolio.updates).completed_at==START+Minute(180)+Second(45)
                @test !portfolio.scientific_coverage_qualified && !portfolio.actual_source_authority_granted
            end
            @test result.accepted_rows==collect(151:166) && result.source_reads<=4*45
            # Test instrumentation supplies only label clock metadata. Any
            # attempt to use cached final targets for feedback must fail.
            clocks=Tuple((;asof=label.asof) for label in value.labels)
            instrumented=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,clocks,
                value.request_sha256,value.manifest_sha256,value.complete_sha256,false,false)
            repeated=replay(instrumented,point,calibration,expected;static_calibration=static)
            @test isequal(repeated.shared.events,result.shared.events)
            @test repeated.centers==result.centers && F.model_bytes(point.model)==before
            @test repeated.static_bounds.bounds==result.static_bounds.bounds
            repeated.static_bounds.bounds.raw_split[1]=999.
            @test SB.check_calibration(static.capsule,point,calibration,expected;expected_static=static.expected).numeric.raw!=999.
            no_static=replay(instrumented,point,calibration,expected)
            @test !no_static.audit.mandatory_static_inventory_complete && !no_static.study_pass && no_static.static_bounds===nothing
            badpins=SB.ExpectedStatic(repeat("0",64),static.expected.numeric_sha256)
            @test_throws ArgumentError replay(instrumented,point,calibration,expected;static_calibration=(capsule=static.capsule,expected=badpins))
            result.centers[1]=999.
            @test repeated.centers[1]==log(6.) && F.model_bytes(point.model)==before
            @test_throws ArgumentError replay(value,point,calibration,expected;index=2)
            @test_throws ArgumentError replay(value,point,calibration,expected;start=START+Minute(149))
            @test_throws ArgumentError replay(value,point,calibration,expected;warmup=1)
            cold=replay(value,point,calibration,expected;warmup=false)
            @test isempty(cold.publisher_upper.warmup_ids) && cold.shared.source.completed_targets==1
            ground=getproperty(value.source.catalogs,Symbol("ground-CMO"))
            view=S._GroundView(ground,START+Minute(150),START+Minute(195),START+Minute(151),D.CMR.ReplayPolicy(),0)
            @test_throws ArgumentError haskey(view,START+Minute(151))
            @test view.reads==0
        end
    end
end
@testset "All scheduled feature exclusions remain explicit" begin
    mktempdir() do root
        value=make_cases(realpath(root),"primary","ground-only";profile=:pulse_eval_missing)
        point,calibration,expected,static=fitted(value);result=replay(value,point,calibration,expected;static_calibration=static)
        @test result.status===:no_feature_eligible && result.audit.scheduled==result.audit.features_rejected==16
        @test result.audit.features_accepted==0 && !result.study_pass && result.shared===nothing
        @test result.publisher_upper===nothing && result.publisher_absolute===nothing
        @test result.audit.mandatory_static_inventory_complete && result.static_bounds===nothing
        @test sum(values(result.audit.rejection_reasons))==16 && result.audit.actual_fits==0
    end
end
@test Sys.maxrss()<=8*1024^3
println("SOURCE_BOUND_SHARED_REPLAY_GENERATED_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
println("SOURCE_BOUND_STATIC_INVENTORY_PASSED;METHODS=4;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0")
end
