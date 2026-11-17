# Test-only source types. Any accidental source access raises ErrorException,
# while intended pre-admission failures are ArgumentError; no real label bytes
# or fitting/prediction callbacks can be supplied by these objects.
struct PoisonCandidateFeatures
    reads::Base.RefValue{Int}
end
struct PoisonCandidateLabels
    reads::Base.RefValue{Int}
end
struct CountingCandidateLabels
    original::N.V.CatalogLabelAudit
    feature_schedule_sha256::String
    asof::DateTime
    calls::Vector{Int}
end
Base.length(v::CountingCandidateLabels)=length(v.original)
function N.V.label_row(v::CountingCandidateLabels,i)
    push!(v.calls,i);N.V.label_row(v.original,i)
end
Base.length(::PoisonCandidateFeatures)=651
function N.V.feature_row(value::PoisonCandidateFeatures,index)
    value.reads[]+=1;error("UNAUTHORIZED FEATURE READ")
end
function N.V.label_row(value::PoisonCandidateLabels,index)
    value.reads[]+=1;error("UNAUTHORIZED TARGET READ")
end
function poison_cases(N,value;index=1,role=nothing,request_pin=nothing,manifest_pin=nothing,complete_pin=nothing,actual=false)
    request=N.decode(Vector{UInt8}(codeunits(value.request_json)))
    role===nothing || (request["role"]=role)
    if actual;request["schema"]="archive_catalog_case_request_v1";request["evidence_mode"]="actual_archival_cache";end
    raw=String(N.canonical(request));f=Ref(0);l=Ref(0)
    cases=N.Q.VerifiedCases(N.Q._Checked(),raw,value.source,
        (PoisonCandidateFeatures(f),PoisonCandidateFeatures(f)),(PoisonCandidateLabels(l),PoisonCandidateLabels(l)),
        request_pin===nothing ? N.digest(codeunits(raw)) : request_pin,
        manifest_pin===nothing ? value.manifest_sha256 : manifest_pin,
        complete_pin===nothing ? value.complete_sha256 : complete_pin,actual,false)
    (;cases,feature_reads=f,target_reads=l)
end
function run_authority_controls(N,result,root)
    @testset "Prework freshness and station-specific R2 refit lineage" begin
        station=result.stations[1];selection=station.decision;family=:ridge
        winner=only(filter(w->w.family===family,selection.decision.family_winners)).candidate_id
        input=N.prepare_candidate_fit_input_generated(result.cases,1,winner,result.refit,joinpath(root,"lineage-input"))
        for (name,kwargs,index) in (("cross-station",(;),2),("manifest",(;manifest_pin="b"^64),1),
            ("complete",(;complete_pin="c"^64),1),("role",(;role="independent"),1),("request",(;request_pin="d"^64),1))
            p=poison_cases(N,result.cases;kwargs...);destination=joinpath(root,"denied-"*name)
            @test_throws ArgumentError N.refit_selected_generated(selection,p.cases,index,family,result.refit,input.input;directory=destination)
            @test p.feature_reads[]==p.target_reads[]==0
            @test !ispath(destination)
        end
        p=poison_cases(N,result.cases);existing=joinpath(root,"already-exists");mkdir(existing)
        @test_throws ArgumentError N.prepare_candidate_fit_input_generated(p.cases,1,winner,result.train,existing)
        @test_throws ArgumentError N.validate_candidate_generated(p.cases,1,winner,result.validation,station.fits[1].point,existing)
        @test_throws ArgumentError N.fit_candidate_generated(p.cases,1,winner,result.train,input.input;directory=existing)
        @test p.feature_reads[]==p.target_reads[]==0 && isempty(readdir(existing))
    end
    @testset "Admitted generated numerical failure retains spent attempt" begin
        original=result.cases;calls=Int[];label=original.labels[1]
        wrapper=CountingCandidateLabels(label,label.feature_schedule_sha256,label.asof,calls)
        cases=N.Q.VerifiedCases(N.Q._Checked(),original.request_json,original.source,original.features,(wrapper,original.labels[2]),
            original.request_sha256,original.manifest_sha256,original.complete_sha256,false,false)
        fit=result.stations[1].fits[1];point=fit.point;model=point.model
        coefficients=zeros(12);coefficients[2]=1e308
        bad_model=C.NumericModel(model.recipe,:ground11,model.cuts,();center=model.center,scale=model.scale,coefficients)
        receipt=N.decode(F.receipt_bytes(point);limit=65536);receipt["model_sha256"]=N.digest(C.encode_model(bad_model))
        bad_point=F.BoundFit(bad_model,point.expected,point.context,point.recipe,String(N.canonical(receipt;limit=65536)),F._FitToken())
        split=N.CandidateSplit(result.validation.start,result.validation.start+Minute(32),result.validation.asof)
        destination=joinpath(root,"admitted-failed-score")
        @test_throws ArgumentError N.validate_candidate_generated(cases,1,"ridge-0.01",split,bad_point,destination)
        marker=destination*".CANDIDATE_VALIDATION_ATTEMPT.json"
        @test isfile(marker) && length(calls)==2 && !ispath(joinpath(destination,"COMPLETE.json"))
        empty!(calls)
        @test_throws Exception N.validate_candidate_generated(cases,1,"ridge-0.01",split,bad_point,destination)
        @test isempty(calls) && isfile(marker)
    end
end

struct BoundaryFeatureSchedule
    mode::Symbol
    schedule_sha256::String
    rows::Tuple
end
struct BoundaryLabelSchedule
    feature_schedule_sha256::String
    rows::Tuple
    accesses::Vector{Int}
end
Base.length(v::BoundaryFeatureSchedule)=length(v.rows)
Base.length(v::BoundaryLabelSchedule)=length(v.rows)
N.V.feature_row(v::BoundaryFeatureSchedule,i)=v.rows[i]
function N.V.label_row(v::BoundaryLabelSchedule,i)
    push!(v.accesses,i);i>=3 && error("OUT-OF-SPLIT TARGET OPENED");v.rows[i]
end
@testset "Independent readiness gate precedes target decoding" begin
    stop=DateTime(2017,7,1);issues=(stop-Minute(32),stop-Minute(31),stop-Minute(30))
    features=Tuple((;issue=t,label_ready=t+Minute(30)+Second(45),accepted=true,
        features=ntuple(_->1.,11),raw_now=1.,scale=1.) for t in issues)
    labels=Tuple((;issue=f.issue,label_ready=f.label_ready,accepted=true,target=1.) for f in features)
    calls=Int[];value=(features=(BoundaryFeatureSchedule(:ground11,"same",features),),labels=(BoundaryLabelSchedule("same",labels,calls),))
    output=N._independent_source(value,1,N.CandidateSplit(DateTime(2017,6,30),stop,stop))
    @test output.rows==[1,2] && calls==[1,2]
    @test output.counts["not_ready_stop"]==1 && output.counts["accepted"]==2
end
