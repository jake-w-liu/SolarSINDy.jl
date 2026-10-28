if !isdefined(@__MODULE__,:SourceBoundSharedReplayTests)
    include("source_bound_tests.jl")
end
module SeparateEvaluationTests
using Dates,SHA,Test,SolarSINDy
const T=getfield(parentmodule(@__MODULE__),:SourceBoundSharedReplayTests)
const S=T.S
const F=T.F
const Q=T.Q
const R=T.R
digest(bytes)=bytes2hex(sha256(bytes))
module FutureFixture
using Dates,SHA,SolarSINDy
const START=DateTime(2021)
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","archive_source_bound_fit","case_fixture.jl"))
end

function save_fixture(directory,evaluation,point,calibration,binding)
    mkdir(directory)
    bodies=Dict("point-context.json"=>Vector{UInt8}(codeunits(point.context)),
        "point-model.bin"=>F.model_bytes(point.model),"point-recipe.json"=>Vector{UInt8}(codeunits(point.recipe)),
        "point-receipt.json"=>F.receipt_bytes(point),"calibration-context.json"=>Vector{UInt8}(codeunits(calibration.context)),
        "case-request.json"=>read(joinpath(evaluation,"case-request.json")),
        "case-manifest.json"=>read(joinpath(evaluation,"cases","manifest.json")),
        "case-complete.json"=>read(joinpath(evaluation,"cases","COMPLETE.json")),
        "evaluation-binding.json"=>Vector{UInt8}(codeunits(binding.context)))
    for (name,body) in bodies;R._exclusive_write(joinpath(directory,name),body);end
    Dict("directory"=>realpath(directory),"pins"=>Dict(name=>digest(body) for (name,body) in bodies),
        "table_index"=>1,"start"=>FutureFixture.clock(FutureFixture.START+Minute(150)),
        "stop"=>FutureFixture.clock(FutureFixture.START+Minute(166)),"asof"=>FutureFixture.clock(FutureFixture.START+Minute(211)))
end

@testset "Frozen 2020 models evaluate separately sealed 2021 sources" begin
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    parent=realpath(mktempdir(base;prefix="solarsindy-separate-evaluation-",cleanup=false))
    entries=Any[]
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        root=joinpath(parent,role*"-"*layout);mkdir(root)
        training=joinpath(root,"training");mkdir(training)
        value=T.make_cases(realpath(training),role,layout)
        # Fit every model and score every calibration before future data exist.
        point,calibration,expected,static=T.fitted(value)
        before=F.model_bytes(point.model)
        future=joinpath(root,"future");mkdir(future)
        evaluation=FutureFixture.make_cases(realpath(future),role,layout)
        start=FutureFixture.START+Minute(150);stop=FutureFixture.START+Minute(166);asof=FutureFixture.START+Minute(211)
        @test evaluation.request_sha256!=value.request_sha256
        binding=S.prepare_evaluation_binding(evaluation,1,point,calibration,expected;start,stop,asof)
        @test digest(codeunits(binding.context))==binding.expected.context_sha256
        @test binding.expected.context_sha256!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        call(v,b;index=1,begin_at=start)=S.replay_generated(v,index,point,calibration,expected;
            start=begin_at,stop,asof,subscribers=T.specs(),nominal=.1,static_calibration=static,evaluation_binding=b)
        @test_throws ArgumentError call(evaluation,nothing)
        @test_throws ArgumentError call(evaluation,S.ExpectedEvaluation(repeat("0",64)))
        @test_throws ArgumentError call(evaluation,binding.expected;index=2)
        @test_throws ArgumentError call(evaluation,binding.expected;begin_at=start+Minute(1))
        @test_throws ArgumentError S.prepare_evaluation_binding(evaluation,true,point,calibration,expected;start,stop,asof)
        @test_throws ArgumentError S.prepare_evaluation_binding(evaluation,1,point,calibration,expected;start=start+Millisecond(1),stop,asof)
        result=call(evaluation,binding.expected)
        @test result.status===:replayed && result.audit.explicit_evaluation_binding_matched
        @test result.evaluation_binding_sha256==binding.expected.context_sha256
        @test result.audit.scheduled==result.audit.features_accepted==16 && result.audit.features_rejected==0
        @test result.shared.source.completed_targets==1 && result.shared.source.censored_targets==15 && result.shared.source.pending_targets==0
        @test result.static_bounds.methods==S.B.METHODS && all(c->c.issued==16,values(result.static_bounds.counts))
        @test all(==(log(6.)),result.centers) && result.accepted_rows==collect(151:166)
        @test length(result.shared.subscribers)==5 && all(c->c.attempted==16,result.shared.subscribers)
        for portfolio in (result.publisher_upper,result.publisher_absolute)
            @test portfolio.phase_period_minutes==31 && portfolio.attempted==portfolio.issued_bounds==16
            @test portfolio.binary_resolved==1 && portfolio.target_unknown==15
            @test only(portfolio.updates).target==10. && only(portfolio.updates).completed_at==FutureFixture.START+Minute(180)+Second(45)
            @test !portfolio.actual_source_authority_granted && !portfolio.scientific_coverage_qualified
        end
        # Only asof survives: source feedback cannot read final cached targets.
        clocks=Tuple((;asof=label.asof) for label in evaluation.labels)
        trapped=Q.VerifiedCases(Q._Checked(),evaluation.request_json,evaluation.source,evaluation.features,clocks,
            evaluation.request_sha256,evaluation.manifest_sha256,evaluation.complete_sha256,false,false)
        repeated=call(trapped,binding.expected)
        @test isequal(repeated.shared.events,result.shared.events) && repeated.static_bounds.bounds==result.static_bounds.bounds
        @test repeated.centers==result.centers && F.model_bytes(point.model)==before
        @test !result.audit.actual_source_authority_granted && result.audit.actual_fits==0 && !result.audit.scientific_coverage_qualified
        entry=save_fixture(joinpath(root,"oracle"),future,point,calibration,binding)
        entry["role"]=role;entry["layout"]=layout;push!(entries,entry)
        for (other_role,other_layout) in ((role=="primary" ? "independent" : "primary",layout),
                                        (role,layout=="ground-only" ? "mixed" : "ground-only"))
            wrong=joinpath(root,"wrong-"*other_role*"-"*other_layout);mkdir(wrong)
            mismatch=FutureFixture.make_cases(realpath(wrong),other_role,other_layout)
            @test_throws ArgumentError S.prepare_evaluation_binding(mismatch,1,point,calibration,expected;start,stop,asof)
        end
    end
    report=Dict("schema"=>"generated_separate_evaluation_contexts_v1","runtime"=>string(VERSION),"entries"=>entries,
        "actual_source_access"=>false,"actual_fits"=>0)
    path=joinpath(parent,"report.json");bytes=R.canonical_bytes(report;limit=131072);R._exclusive_write(path,bytes)
    python=get(ENV,"SOLARSINDY_RESEARCH_PYTHON",Sys.iswindows() ? "python" : "python3")
    output=joinpath(parent,"independent-checks")
    script=joinpath(@__DIR__,"..","..","tools","point_input_export","check_evaluation_bindings.py")
    @test success(`$python -B -X utf8 $script --report $path --report-sha256 $(digest(bytes)) --output $output`)
    terminal=R.decode_canonical(read(joinpath(output,"terminal.json"));limit=131072)
    @test terminal["complete"]===true && terminal["contexts"]==4 && terminal["actual_fits"]==0
    @test Sys.maxrss()<=8*1024^3
end
println("SEPARATE_EVALUATION_GENERATED_PASSED;CONTEXTS=4;FIXED_POINT_FITS;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
