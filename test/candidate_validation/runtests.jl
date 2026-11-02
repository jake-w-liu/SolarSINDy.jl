module CandidateValidationTests
using Test,Dates,SHA,SolarSINDy,Statistics
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","candidate_validation.jl"))
const N=SourceBoundCandidateValidation
include("generated_fixture.jl")
const G=CandidateValidationGeneratedFixture
const R=N.R
const F=N.F
const C=N.C
include("authority_tests.jl")
@testset "Independent raw metric and nonconstant input frame" begin
    @test reinterpret(UInt64,N.log_mse([0.,0.],[-2.,1.]))==0x4004000000000000
    @test N._independent_loss([0.,0.],[-2.,1.])==2.5
    @test N.log_mse([0.,0.],[0.,0.])==0.
    @test N.log_mse([1.,1.],[0.,0.])==mean(abs2,fill(log(2.),2))
    # Frozen rational80-term log2 enclosure independently fixes these bits.
    @test reinterpret(UInt64,N.log_mse([1.,1.],[1.,1.]))==0x3fb81ac80e677748
    @test N.log_mse([1.,1.],[-1.,1.])!=N.log_mse([1.,1.],[1.,1.])
    for (ys,p) in (([0.,0.],[Inf,1.]),([-1.,0.],[0.,0.]),([0.,0.],[-1e308,1e308]),([0.],[0.]))
        @test_throws ArgumentError N.log_mse(ys,p)
    end
    X=[Float64(100*(i-1)+j) for i in 1:2,j in 1:11]
    a=(;rows=[11,13],issues=[DateTime(Dates.UTInstant{Millisecond}(Millisecond(t))) for t in (63634551000000,63634551120000)],
        targets=[0.,0.],raw_now=[1.,101.],scales=[1.,100.],features=X)
    oracle="413fd2f802af14bd25bd2272610c81af605f80227afdd68c113eb58b933bf54b"
    @test N._independent_input_hash(a)==oracle
    @test first(F._data_sha(a))==oracle
    @test N._independent_input_hash(merge(a,(features=reverse(X;dims=2),)))!=oracle
end
@testset "Real frozen16 fits, K scores, independent numerical reconstruction and3 refits" begin
    mktempdir() do temporary
        root=realpath(temporary);result=G.run_chain(N,root)
        case_root=joinpath(root,"source-case")
        @test N._rebind_case_files(result.cases,joinpath(case_root,"case-request.json"),joinpath(case_root,"cases"))===nothing
        extra_case_member=joinpath(case_root,"cases","extra");write(extra_case_member,"data-only tamper fixture")
        @test_throws ArgumentError N._rebind_case_files(result.cases,joinpath(case_root,"case-request.json"),joinpath(case_root,"cases"))
        rm(extra_case_member)
        @test length(result.stations)==2
        for (index,station) in enumerate(result.stations)
            @test length(station.verified)==16
            @test [row.candidate_id for row in station.verified]==[spec.id for spec in N.S.POINT_GRID]
            @test all(c->c.n>128 && !c.actual && isfinite(c.loss),station.verified)
            @test length(unique(c.cohort for c in station.verified))==1
            @test [r.family for r in station.refits]==[:ridge,:boost,:forest]
            @test all(r->r.selection===station.decision && !F._check_capsule(r.fit.point).actual,station.refits)
            @test N.choose_verified_R2(reverse(station.verified)).decision.decision_sha256==station.decision.decision.decision_sha256
            @test_throws ArgumentError N.choose_verified_R2(station.verified[1:15])
            @test_throws ArgumentError N.choose_verified_R2(vcat(station.verified[1:15],[station.verified[1]]))
            first=station.verified[1];fit=station.fits[1];score=station.validations[1]
            arrays=N.Q.split_arrays(result.cases,index;start=result.validation.start,stop=result.validation.stop,asof=result.validation.asof)
            @test length(unique(arrays.targets))>1
            @test first.loss==Statistics.mean(abs2,log1p.(arrays.targets).-F.predict_log(fit.point,arrays.features))
            @test F.predict_log(fit.point,arrays.features)==C.predict(fit.point.model,arrays.features)
            completion=station.completed[1]
            fixture_owner=N.decode(R.bounded_read(completion.completion.receipt_path;limit=16384))
            @test fixture_owner["native_execution_evidence"]===false && fixture_owner["actual_authority"]===false
            for (key,bad_value) in (("job_sha256","0"^64),("output_witness_sha256","1"^64),("timed_out",true),
                ("output_complete_sha256","2"^64),("declared_fixture_native_exit",false))
                damaged=copy(fixture_owner);damaged[key]=bad_value;bytes=N.canonical(damaged)
                path=joinpath(root,"bad-native-shape-$index-"*key*".json");R._exclusive_write(path,bytes)
                wrong=N.NativeEvidence(path,N.digest(bytes),completion.completion.raw_path,completion.completion.raw_sha256)
                @test_throws ArgumentError N.read_completed_generated_candidate(result.cases,index,"ridge-0.01",result.validation,
                    fit,score.artifact.directory,score.pins,completion.witness_directory,completion.witness_pins;completion=wrong)
            end
            if index==1
                # Rehash every affected enclosing artifact after changing one
                # prediction, score, rank and scalar loss consistently. Strict
                # data readback succeeds; source/model reconstruction must fail.
                context=N.decode(Vector{UInt8}(codeunits(score.artifact.context_json)))
                records=Vector{UInt8}(codeunits(score.artifact.records));decoded=N._independent_records(records,first.n)
                changed=decoded.predictions[1]+1.
                records[33:40]=reinterpret(UInt8,[htol(reinterpret(UInt64,changed))])
                changed_score=(log1p(decoded.targets[1])-changed)/decoded.scales[1]
                records[49:56]=reinterpret(UInt8,[htol(reinterpret(UInt64,changed_score))])
                parsed=N._independent_records(records,first.n)
                ranks=Vector{UInt8}(reinterpret(UInt8,htol.(UInt32.(sortperm(parsed.scores;alg=MergeSort)))))
                metric=N.decode(Vector{UInt8}(codeunits(score.artifact.metric_json));limit=16384)
                metric["loss_bits"]=C.bits(N.log_mse(parsed.targets,parsed.predictions))
                for key in ("records_sha256","ranks_sha256")
                    hash=key=="records_sha256" ? N.digest(records) : N.digest(ranks)
                    context[key]=hash;context["calibration_numeric_context"][key]=hash
                end
                context["metric_sha256"]=N.digest(N.canonical(metric;limit=16384))
                forged=N._publish_validation(context,records,ranks,metric,nothing,joinpath(root,"self-consistent-forgery"))
                @test N.read_validation(forged.artifact.directory,forged.pins).records==String(copy(records))
                @test_throws ArgumentError N.verify_candidate_validation_generated(result.cases,index,"ridge-0.01",result.validation,
                    fit,forged.artifact.directory,forged.pins)
                report=N.decode(N._completed_witness(first,nothing))
                report["validation_context_sha256"]=forged.pins.context_sha256
                report["validation_complete_sha256"]=forged.pins.complete_sha256
                report["loss_bits"]=metric["loss_bits"]
                report["witness_sha256"]=N.digest(N.canonical(Dict("descriptor_sha256"=>N.digest(N.canonical(context["descriptor"])),
                    "validation_context_sha256"=>forged.pins.context_sha256,"validation_complete_sha256"=>forged.pins.complete_sha256,
                    "point_complete_sha256"=>fit.pins.complete_sha256,"cohort_sha256"=>first.cohort,"loss_bits"=>metric["loss_bits"],"job_sha256"=>N.job_identity())))
                forged_witness=joinpath(root,"self-consistent-forged-witness")
                wp=N._publish_payloads(forged_witness,[("request.json",N.canonical(context["descriptor"])),("receipt.json",N.canonical(report))],
                    "candidate_independent_numeric";limit=N.MAX_INPUT_BYTES,closecheck=()->nothing)
                @test_throws ArgumentError N.read_completed_generated_candidate(result.cases,index,"ridge-0.01",result.validation,
                    fit,forged.artifact.directory,forged.pins,forged_witness,wp)
            end
            @test_throws ArgumentError N.verify_candidate_validation_generated(result.cases,index,"ridge-0.1",result.validation,fit,score.artifact.directory,score.pins)
            @test_throws ArgumentError N.validate_candidate_generated(result.cases,index,"ridge-0.01",result.validation,fit.point,score.artifact.directory)
            extra=joinpath(score.artifact.directory,"unexpected");write(extra,"fixture corruption")
            @test_throws ArgumentError N.read_validation(score.artifact.directory,score.pins)
            rm(extra)
        end
        # A source change strictly after570 cannot enter the selected530stop.
        altered_root=joinpath(root,"future-altered");mkdir(altered_root);altered=G.make_cases(realpath(altered_root);future_delta=700)
        for index in 1:2
            a=N._independent_source(result.cases,index,result.validation);b=N._independent_source(altered,index,result.validation)
            @test a.rows==b.rows && a.features==b.features && a.targets==b.targets && a.scales==b.scales
            @test N._independent_input_hash(a)==N._independent_input_hash(b)
        end
        run_authority_controls(N,result,root)
    end
end
end
