module SourceBoundCalibrationTests
using Test,Dates,SHA,SolarSINDy
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_calibration.jl"))
const K=SourceBoundCalibration
const F=SolarSINDy.ArchiveSourceBoundPointFit
const B=SolarSINDy.ArchiveCatalogCaseReader
const Q=SolarSINDy.ArchiveCatalogCaseJob
const C=SolarSINDy.ArchiveClassicPointModels
const D=SolarSINDy.CompletedSourceDBDT
const R=D.CompletedRefitArtifacts
const J=SolarSINDy.ArchiveCatalogSourceJob
const START=DateTime(2020)
digest(x)=bytes2hex(sha256(x))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","archive_source_bound_fit","case_fixture.jl"))
include("authority_tests.jl")
function recipes()
    (F.BoostRecipe(depth=1,rounds=2,bins=2,seed=17,eta=.5,min_weight=1.,lambda=0.,L2=0.),
     C.Recipe(kind=:ridge,penalty=1.,bins=2),C.Recipe(kind=:boost,depth=1,trees=2,bins=2,rate=.5,minleaf=1),
     C.Recipe(kind=:forest,depth=1,trees=2,bins=2,minleaf=1,sample_rows=20,seed=17))
end
function run_scope(root)
    outputs=Any[]
    @testset "Independent zero-training and nonzero signed calibration oracle" begin
        for role in ("primary","independent"),layout in ("ground-only","mixed")
            case_root=joinpath(root,role*"-"*layout);mkdir(case_root)
            value=make_cases(realpath(case_root),role,layout;profile=:zero_then_10_then_20)
            cases_path=joinpath(case_root,"cases");request_path=joinpath(case_root,"case-request.json")
            rereverified=B.read_reverified(request_path,cases_path;request_sha256=value.request_sha256,
                manifest_sha256=value.manifest_sha256,complete_sha256=value.complete_sha256)
            train_start=START+Minute(40);train_stop=START+Minute(90);asof=START+Minute(211)
            start=START+Minute(100);stop=START+Minute(175);first_issue=START+Minute(180)
            train=Q.split_arrays(value,1;start=train_start,stop=train_stop,asof)
            @test train.issues==collect(START+Minute(40):Minute(1):START+Minute(59))
            @test train.targets==zeros(20)
            cal=Q.split_arrays(value,1;start,stop,asof)
            @test cal.issues==collect(START+Minute(100):Minute(1):START+Minute(144))
            @test cal.targets==vcat(fill(10.,10),fill(20.,35))
            @test cal.scales==vcat(fill(1+log(11.),41),fill(1+log(21.),4))
            # Integer(6,8) and(12,16) increments yield exact rates10 and20.
            # The maximum in the future changes at issue110; the last known
            # ground maximum changes at issue141 under the 45-second clock.
            expected_scores=vcat(fill(log(11.)/(1+log(11.)),10),fill(log(21.)/(1+log(11.)),31),fill(log(21.)/(1+log(21.)),4))
            expected_rank=vcat(collect(1:10),collect(42:45),collect(11:41))
            for (i,recipe) in enumerate(recipes())
                pe=F.prepare_expected(value,1,recipe;start=train_start,stop=train_stop,asof)
                point=F.fit_generated(value,1,recipe,pe;start=train_start,stop=train_stop,asof).capsule
                @test F.predict_log(point,cal.features)==zeros(45)
                expected=K.prepare_expected(value,1,point;start,stop,asof,first_issue)
                capsule=K.calibrate_generated(value,1,point,expected;start,stop,asof,first_issue)
                raw_rank=Int.(ltoh.(reinterpret(UInt32,Vector{UInt8}(codeunits(capsule.ranks)))))
                @test raw_rank==expected_rank
                raw_rank==expected_rank || continue
                checked=K._checked(capsule,point,expected)
                @test checked.arrays.scores==expected_scores
                @test checked.order==expected_rank
                @test checked.arrays.ready==[START+Minute(t+30)+Second(45) for t in 100:144]
                projection=K.completed_calibration(capsule,point,expected;first_issue)
                @test projection.chronological_scores==expected_scores && projection.sorted_scores==expected_scores[expected_rank]
                @test projection.point_model_sha256==digest(F.model_bytes(point.model)) && projection.purpose===:synthetic
                @test projection.mode===(layout=="ground-only" ? :ground11 : :mixed8)
                @test D.CMR.issue_timing(first_issue,projection.policy).sufficient_phase_period_minutes==31
                @test_throws ArgumentError K.completed_calibration(capsule,point,expected;first_issue=first_issue-Millisecond(1))
                directory=joinpath(case_root,"calibration-$i")
                pins=K.publish_calibration(capsule,point,expected,directory)
                reread=K.read_calibration(directory,point,expected;pins...)
                @test reread.records==capsule.records && reread.ranks==capsule.ranks && reread.receipt==capsule.receipt
                @test_throws ArgumentError K.publish_calibration(capsule,point,expected,directory)
                wrong=K.ExpectedCalibration(repeat("0",64),expected.input_sha256)
                @test_throws ArgumentError K.calibrate_generated(value,1,point,wrong;start,stop,asof,first_issue)
                @test_throws ArgumentError K.read_calibration(directory,point,wrong;pins...)
                @test_throws ArgumentError K.prepare_expected(value,1,point;start=train_stop-Millisecond(1),stop,asof,first_issue)
                @test_throws ArgumentError K.prepare_expected(value,1,point;start,stop,asof,first_issue=stop-Millisecond(1))
                @test_throws MethodError K.calibrate_archival(value,1,point,expected;start,stop,asof,first_issue)
                @test_throws ArgumentError K.calibrate_archival(rereverified,1,point,expected;start,stop,asof,first_issue)
                missing=joinpath(case_root,"does-not-exist");pin=repeat("a",64)
                authority=K.CalibrationAuthority(missing,pin,missing,pin,missing,pin)
                producer=F.FitAuthority(missing,pin,missing,pin,missing,pin)
                @test_throws ArgumentError K.calibrate_archival(rereverified,1,point,expected;start,stop,asof,first_issue,authority,producer)
                @test_throws ArgumentError K._producer_evidence(point,nothing)
                @test_throws ArgumentError K._producer_evidence(point,producer)
                @test !ispath(missing) && !ispath(joinpath(case_root,"CALIBRATION_ATTEMPT.json"))
                if role=="primary" && layout=="ground-only" && i==1
                    run_invalid_calibration_authority_tests(value,rereverified,point,expected,case_root)
                end
                # A record with a changed score is rejected even with freshly
                # recomputed surrounding data hashes. Keep the external context
                # exact to distinguish a formula/rank check from a hash check.
                for kind in (:score,:rank,:readiness,:nonfinite,:flag)
                    context=deepcopy(checked.context);receipt=deepcopy(checked.receipt)
                    records=copy(checked.records);ranks=copy(checked.ranks)
                    if kind===:score
                        # Increasing the final member of the first equal-score
                        # group preserves the rank projection. The formula
                        # check must detect this corruption independently.
                        records[553:560]=reinterpret(UInt8,[htol(reinterpret(UInt64,nextfloat(expected_scores[10])))])
                    elseif kind===:rank
                        ranks[1:4]=reinterpret(UInt8,[htol(UInt32(2))])
                    elseif kind===:readiness
                        records[17:24]=reinterpret(UInt8,[htol(reinterpret(UInt64,Int64(Dates.value(stop))))])
                    elseif kind===:nonfinite
                        records[41:48]=reinterpret(UInt8,[htol(reinterpret(UInt64,Inf))])
                    else
                        receipt["actual_model_fit_performed"]=true
                    end
                    context["records_sha256"]=digest(records);context["ranks_sha256"]=digest(ranks)
                    cb=R.canonical_bytes(context;limit=131072);ep=K.ExpectedCalibration(digest(cb),expected.input_sha256)
                    receipt["context_sha256"]=ep.context_sha256;receipt["records_sha256"]=digest(records);receipt["ranks_sha256"]=digest(ranks)
                    bad=K.CalibrationCapsule(String(copy(cb)),records,ranks,String(R.canonical_bytes(receipt;limit=65536)),K._Token())
                    @test_throws ArgumentError K.completed_calibration(bad,point,ep;first_issue)
                end
                for (field,bad_value) in (("point_model_sha256","0"^64),("point_receipt_sha256","0"^64),
                    ("point_context_sha256","0"^64),("point_recipe_sha256","0"^64),("point_source_sha256","0"^64),
                    ("job_sha256","0"^64),("table_index",true),("rows",46),("runtime","wrong"),
                    ("measured_receipts_available",true),("score_definition","absolute"),
                    ("rank_definition","unstable"),("point_training_stop",clock(start+Minute(1))),
                    ("first_evaluation_issue",clock(stop-Millisecond(1))))
                    context=deepcopy(checked.context);context[field]=bad_value
                    cb=R.canonical_bytes(context;limit=131072);ep=K.ExpectedCalibration(digest(cb),expected.input_sha256)
                    receipt=deepcopy(checked.receipt);receipt["context_sha256"]=ep.context_sha256
                    bad=K.CalibrationCapsule(String(copy(cb)),copy(checked.records),copy(checked.ranks),String(R.canonical_bytes(receipt;limit=65536)),K._Token())
                    @test_throws ArgumentError K.completed_calibration(bad,point,ep;first_issue)
                end
                extra=joinpath(directory,"extra");write(extra,"x")
                @test_throws ArgumentError K.read_calibration(directory,point,expected;pins...)
                rm(extra)
                incomplete=joinpath(case_root,"incomplete-$i");mkdir(incomplete)
                @test_throws Exception K.read_calibration(incomplete,point,expected;pins...)
                # Exact readiness equality is excluded; 1ms later admits the row.
                boundary=START+Minute(130)+Second(45)
                @test_throws ArgumentError K.prepare_expected(value,1,point;start,stop=boundary,asof,first_issue)
                one=Q.split_arrays(value,1;start,stop=boundary+Millisecond(1),asof)
                @test one.issues==[start]
                two_stop=boundary+Minute(1)+Millisecond(1)
                ep=K.prepare_expected(value,1,point;start,stop=two_stop,asof,first_issue)
                two=K.calibrate_generated(value,1,point,ep;start,stop=two_stop,asof,first_issue)
                @test K._checked(two,point,ep).arrays.issues==[start,start+Minute(1)]
                push!(outputs,Dict("role"=>role,"layout"=>layout,"family"=>F.family(recipe),"directory"=>directory,
                    "point_directory"=>joinpath(case_root,"point-$i"),"expected_context_sha256"=>expected.context_sha256,
                    "expected_input_sha256"=>expected.input_sha256,"manifest_sha256"=>pins.manifest_sha256,
                    "receipt_sha256"=>pins.receipt_sha256,"complete_sha256"=>pins.complete_sha256))
                F.publish_fit(point,joinpath(case_root,"point-$i"))
            end
        end
    end
    @testset "Signed scores retain negative residuals" begin
        path=joinpath(root,"negative-score-control");mkdir(path)
        value=make_cases(realpath(path),"primary","ground-only";profile=:five_then_zero)
        # A single depth-zero tree with one bootstrap sample predicts its sole
        # sampled target exactly. All training targets are log(6), so this
        # supplies a literal negative-score oracle without a linear solve.
        recipe=C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
        train_start=START+Minute(40);train_stop=START+Minute(90);asof=START+Minute(211)
        start=START+Minute(100);stop=START+Minute(175);first_issue=START+Minute(180)
        pe=F.prepare_expected(value,1,recipe;start=train_start,stop=train_stop,asof)
        point=F.fit_generated(value,1,recipe,pe;start=train_start,stop=train_stop,asof).capsule
        ep=K.prepare_expected(value,1,point;start,stop,asof,first_issue)
        capsule=K.calibrate_generated(value,1,point,ep;start,stop,asof,first_issue)
        raw=Vector{UInt8}(codeunits(capsule.records))
        signed=[reinterpret(Float64,ltoh(only(reinterpret(UInt64,raw[(i-1)*56+49:i*56])))) for i in 1:45]
        @test all(<(0.),signed)
        if all(<(0.),signed)
            checked=K._checked(capsule,point,ep)
            @test checked.arrays.targets==zeros(45)
            @test all(<(0.),checked.arrays.scores)
            @test isapprox(checked.arrays.predictions,fill(log(6.),45);atol=2e-15,rtol=0)
            @test isapprox(checked.arrays.scores,-log(6.)./checked.arrays.scales;atol=2e-15,rtol=0)
            pins=K.publish_calibration(capsule,point,ep,joinpath(path,"calibration"))
            readback=K.read_calibration(joinpath(path,"calibration"),point,ep;pins...)
            projected=K.completed_calibration(readback,point,ep;first_issue)
            @test all(<(0.),projected.sorted_scores)
        end
    end
    @test length(outputs)==16
    R._exclusive_write(joinpath(root,"CALIBRATION_TEST_REPORT.json"),R.canonical_bytes(Dict("complete"=>true,"runtime"=>string(VERSION),
        "artifacts"=>outputs,"job_sha256"=>K.job_identity(),"actual_source_access"=>false,"actual_fits"=>0);limit=131072))
end
requested=get(ENV,"SOLARSINDY_CALIBRATION_TEST_OUTPUT","")
if isempty(requested)
    mktempdir(run_scope)
else
    J._ordinary(dirname(abspath(requested));directory=true);@test !ispath(requested)&&!islink(requested)
    mkdir(requested);run_scope(realpath(requested))
end
@test Sys.maxrss()<=8*1024^3
println("SOURCE_BOUND_CALIBRATION_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
