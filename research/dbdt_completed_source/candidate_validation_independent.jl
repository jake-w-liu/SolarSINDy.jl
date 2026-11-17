"""Independent source/cohort/record reconstruction; no producer selection loop is reused."""
struct _VerifiedToken end
struct VerifiedCandidate
    station::String
    candidate_id::String
    family::Symbol
    cohort::String
    n::Int
    loss::Float64
    descriptor_json::String
    artifact::ValidationArtifact
    fit::FitPublication
    actual::Bool
    witness_sha256::String
    verification_authorization::Union{Nothing,String}
    function VerifiedCandidate(station,id,family,cohort,n,loss,d,a,f,actual,w,authorization,::_VerifiedToken)
        new(station,id,family,cohort,n,loss,String(d),a,f,actual,w,authorization)
    end
end
struct _SelectionToken end
struct CheckedR2Selection
    candidates::Tuple{Vararg{VerifiedCandidate}}
    decision::NamedTuple
    actual::Bool
    evidence_sha256::String
    CheckedR2Selection(c,d,a,e,::_SelectionToken)=new(c,d,a,e)
end
struct SelectedRefit
    fit::FitPublication
    selection::CheckedR2Selection
    family::Symbol
    candidate_id::String
    binding_json::String
end
function _independent_source(value,index,split)
    f=value.features[index];l=value.labels[index]
    need(f.mode===:ground11 && f.schedule_sha256==l.feature_schedule_sha256 && length(f)==length(l),"independent case join")
    rows=Int[];issues=DateTime[];ready=DateTime[];ys=Float64[];now_values=Float64[];scales=Float64[];features=NTuple{11,Float64}[]
    counts=Dict(k=>0 for k in ("scheduled","outside_issue","not_ready_stop","not_ready_asof","feature_rejected","eligible_features"))
    for row in 1:length(f)
        feature=V.feature_row(f,row);counts["scheduled"]+=1
        if feature.issue<split.start || feature.issue>=split.stop;counts["outside_issue"]+=1;continue;end
        if feature.label_ready>=split.stop;counts["not_ready_stop"]+=1;continue;end
        if feature.label_ready>split.asof;counts["not_ready_asof"]+=1;continue;end
        if !feature.accepted;counts["feature_rejected"]+=1;continue;end
        counts["eligible_features"]+=1;need(counts["eligible_features"]<=MAX_ROWS,"independent target query cap")
        label=V.label_row(l,row)
        need(label.issue==feature.issue && label.label_ready==feature.label_ready,"independent row/clock join")
        label.accepted || continue
        push!(rows,row);push!(issues,feature.issue);push!(ready,feature.label_ready);push!(ys,label.target)
        push!(now_values,feature.raw_now);push!(scales,feature.scale);push!(features,feature.features)
    end
    n=length(rows);need(2<=n<=MAX_ROWS,"independent accepted rows")
    X=Matrix{Float64}(undef,n,11)
    for j in 1:11,i in 1:n;X[i,j]=features[i][j];end
    need(all(isfinite,X) && all(x->isfinite(x)&&x>=0,ys) && all(isfinite,now_values) && all(x->isfinite(x)&&x>0,scales),"independent source finite domain")
    counts["accepted"]=n;counts["label_rejected"]=counts["eligible_features"]-n
    (;rows,issues,ready,targets=ys,raw_now=now_values,scales,features=X,counts)
end
function _independent_input_hash(a)
    # Deliberately use an IO frame rather than production F's streaming hasher.
    io=IOBuffer();write(io,"SOURCE_BOUND_POINT_INPUT_V1\n",htol(UInt64(length(a.rows))),htol(UInt64(11)))
    for row in a.rows;write(io,htol(UInt64(row)));end
    for issue in a.issues;write(io,htol(reinterpret(UInt64,Int64(Dates.value(issue)))));end
    for j in 1:11,i in eachindex(a.rows);write(io,htol(reinterpret(UInt64,a.features[i,j])));end
    transformed=map(log1p,a.targets)
    for values in (a.targets,transformed,a.raw_now,a.scales),v in values;write(io,htol(reinterpret(UInt64,v)));end
    digest(take!(io))
end
function _independent_records(raw,n)
    need(length(raw)==56n,"independent record length");io=IOBuffer(raw)
    rows=UInt64[];issues=Int64[];ready=Int64[];targets=Float64[];predictions=Float64[];scales=Float64[];scores=Float64[]
    for _ in 1:n
        push!(rows,ltoh(read(io,UInt64)));push!(issues,reinterpret(Int64,ltoh(read(io,UInt64))))
        push!(ready,reinterpret(Int64,ltoh(read(io,UInt64))))
        for v in (targets,predictions,scales,scores);push!(v,reinterpret(Float64,ltoh(read(io,UInt64))));end
    end
    need(eof(io),"independent trailing record bytes");(;rows,issues,ready,targets,predictions,scales,scores)
end
function _independent_loss(ys,predictions)
    residuals=Vector{Float64}(undef,length(ys))
    for i in eachindex(ys)
        need(isfinite(ys[i]) && ys[i]>=0 && isfinite(predictions[i]),"independent metric finite domain")
        residuals[i]=log1p(ys[i])-predictions[i]
        need(isfinite(residuals[i]) && isfinite(residuals[i]*residuals[i]),"independent metric overflow")
    end
    result=Statistics.mean(abs2,residuals)
    need(isfinite(result) && result>=0,"independent metric reduction");result
end
samebits(a,b)=length(a)==length(b) && all(reinterpret(UInt64,x)==reinterpret(UInt64,y) for (x,y) in zip(a,b))
function _verify_candidate(value,index,id,split,fit::FitPublication,directory,pins::ValidationPins;actual=false,admission=nothing,verification_authorization=nothing)
    point=F.read_published(fit.directory;expected=fit.point.expected,model_sha256=fit.pins.model_sha256,
        receipt_sha256=fit.pins.receipt_sha256,complete_sha256=fit.pins.complete_sha256)
    artifact=read_validation(directory,pins);context=decode(Vector{UInt8}(codeunits(artifact.context_json)))
    d=descriptor(value,index,id,split;operation="fixed_candidate_2017_validation_only",point,actual,admission)
    need(canonical(d)==canonical(context["descriptor"]),"independent validation descriptor mismatch")
    source=_independent_source(value,index,split);n=length(source.rows)
    actual && need(n>=1000,"actual independent candidate minimum1000 rows")
    # C's strict numeric reader/model is used directly, bypassing F.predict_log
    # and K._prepare; preprocessing comes entirely from the published fit.
    need(point.model isa C.NumericModel && point.model.mode===:ground11,"independent candidate model family/layout")
    predictions=C.predict(point.model,source.features)
    need(length(predictions)==n && all(isfinite,predictions),"independent model predictions")
    rows=_independent_records(Vector{UInt8}(codeunits(artifact.records)),n)
    need(rows.rows==UInt64.(source.rows) && rows.issues==Int64[Dates.value(t) for t in source.issues] &&
        rows.ready==Int64[Dates.value(t) for t in source.ready],"independent source row/clock reconstruction")
    need(samebits(rows.targets,source.targets) && samebits(rows.predictions,predictions) && samebits(rows.scales,source.scales),"independent source/point numerical reconstruction")
    scores=[(log1p(source.targets[i])-predictions[i])/source.scales[i] for i in 1:n]
    need(all(isfinite,scores) && samebits(scores,rows.scores),"independent signed score reconstruction")
    ranks=Int.(ltoh.(reinterpret(UInt32,Vector{UInt8}(codeunits(artifact.ranks)))))
    need(ranks==sortperm(scores;alg=MergeSort),"independent rank reconstruction")
    pc=context["calibration_numeric_context"];cohort=cohort_sha(source.rows,source.issues,source.ready)
    need(pc["input_sha256"]==_independent_input_hash(source) && context["cohort_sha256"]==cohort &&
        context["counts"]==source.counts && pc["rows"]==n && pc["point_model_sha256"]==digest(F.model_bytes(point.model)),"independent input/cohort/count reconstruction")
    metric=decode(Vector{UInt8}(codeunits(artifact.metric_json));limit=16384);loss=_independent_loss(source.targets,predictions)
    need(C.bits(loss)==metric["loss_bits"],"independent raw log-MSE reconstruction")
    # Enforce the entire numeric K context independently, including its loaded
    # kernel/source/split conventions, without invoking its preparation again.
    checked=F._check_capsule(point);p=point_descriptor(point)
    expected=Dict{String,Any}("schema"=>"source_bound_calibration_context_v1","evidence_mode"=>(actual ? "actual_archival_cache" : "generated"),
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
        "table_index"=>Int(index),"role"=>d["role"],"layout"=>"ground-only","table"=>checked.context["table"],
        "point_context_sha256"=>p["context_sha256"],"point_recipe_sha256"=>p["recipe_sha256"],"point_model_sha256"=>p["model_sha256"],
        "point_receipt_sha256"=>p["receipt_sha256"],"point_source_sha256"=>p["source_sha256"],"job_sha256"=>K.job_identity(),"runtime"=>string(VERSION),
        "start_inclusive"=>F.clock(split.start),"stop_exclusive"=>F.clock(split.stop),"target_asof"=>F.clock(split.asof),
        "first_evaluation_issue"=>F.clock(split.first_issue),"point_training_stop"=>checked.context["stop_exclusive"],"rows"=>n,
        "input_sha256"=>_independent_input_hash(source),"records_sha256"=>digest(codeunits(artifact.records)),"ranks_sha256"=>digest(codeunits(artifact.ranks)),
        "score_definition"=>"(log1p(target)-point_log_prediction)/case_scale","rank_definition"=>"stable_MergeSort_isless_Float64_chronological_indices",
        "availability_kind"=>"assumed_completed_source_intervals","measured_receipts_available"=>false)
    actual && (expected["case_read_admission_sha256"]=admission)
    need(canonical(expected)==canonical(pc),"independent full numeric context differs")
    witness=digest(canonical(Dict("descriptor_sha256"=>digest(canonical(d)),"validation_context_sha256"=>pins.context_sha256,
        "validation_complete_sha256"=>pins.complete_sha256,"point_complete_sha256"=>fit.pins.complete_sha256,
        "cohort_sha256"=>cohort,"loss_bits"=>C.bits(loss),"job_sha256"=>job_identity())))
    VerifiedCandidate(d["station"],String(id),recipe_for(id).kind,cohort,n,loss,String(canonical(d)),artifact,fit,actual,witness,
        verification_authorization===nothing ? nothing : String(canonical(verification_authorization)),_VerifiedToken())
end
verify_candidate_validation_generated(value::Q.VerifiedCases,index,id,split::CandidateSplit,fit::FitPublication,directory,pins::ValidationPins)=
    _verify_candidate(value,index,id,split,fit,directory,pins)
function verify_candidate_validation_archival(value::B.ReverifiedCases,index,id,split::CandidateSplit,fit::FitPublication,directory,pins::ValidationPins;
        authority=nothing,producer=nothing,native=nothing,validation_producer=nothing,validation_native=nothing,witness_directory=nothing)
    need(all(isnothing,(authority,producer,native,validation_producer,validation_native)),"execution certificates are not accepted; supply scientific artifact pins")
    _locked_split(split,:validation);F._archival_request(value,index,split.start,split.stop,split.asof)
    d=descriptor(value.cases,index,id,split;operation="fixed_candidate_2017_validation_only",point=fit.point,
        actual=true,admission=digest(B.admission_bytes(value)))
    auth=Dict("schema"=>"candidate_operation_binding_v2","operation"=>"fixed_candidate_2017_independent_read_only",
        "input_descriptor_sha256"=>digest(canonical(d)),"case_read_admission_sha256"=>d["case_read_admission_sha256"],"native_fits"=>0)
    result=_verify_candidate(value.cases,index,id,split,fit,directory,pins;actual=true,
        admission=d["case_read_admission_sha256"],verification_authorization=auth)
    if witness_directory!==nothing
        _publish_payloads(witness_directory,[("request.json",canonical(d)),("receipt.json",_completed_witness(result,auth))],
            "candidate_independent_numeric";limit=MAX_INPUT_BYTES,closecheck=()->job_identity())
    end
    result
end
function _completed_witness(candidate,authorization)
    canonical(Dict("schema"=>"candidate_independent_numeric_witness_v1","witness_sha256"=>candidate.witness_sha256,
        "authorization"=>authorization,"validation_context_sha256"=>candidate.artifact.pins.context_sha256,
        "validation_complete_sha256"=>candidate.artifact.pins.complete_sha256,"fit_complete_sha256"=>candidate.fit.pins.complete_sha256,
        "cohort_sha256"=>candidate.cohort,"rows"=>candidate.n,"loss_bits"=>C.bits(candidate.loss)))
end
function publish_generated_verification(candidate::VerifiedCandidate,directory)
    need(!candidate.actual && candidate.verification_authorization===nothing,"generated witness cannot contain actual admission")
    _publish_payloads(directory,[("request.json",Vector{UInt8}(codeunits(candidate.descriptor_json))),
        ("receipt.json",_completed_witness(candidate,nothing))],"candidate_independent_numeric";limit=MAX_INPUT_BYTES,closecheck=()->job_identity())
end
function _rebind_case_files(value,request_path,directory)
    request_bytes=R.bounded_read(request_path;limit=131072)
    need(digest(request_bytes)==value.request_sha256 && request_bytes==Vector{UInt8}(codeunits(value.request_json)),"completed witness frozen case request differs")
    J._ordinary(directory;directory=true);names=readdir(directory)
    manifest_text,mi=FS._read(joinpath(directory,"manifest.json"),1048576);complete_text,ci=FS._read(joinpath(directory,"COMPLETE.json"),4096)
    need(digest(codeunits(manifest_text))==value.manifest_sha256 && digest(codeunits(complete_text))==value.complete_sha256,"completed witness frozen case artifact differs")
    manifest=decode(Vector{UInt8}(codeunits(manifest_text));limit=1048576);complete=decode(Vector{UInt8}(codeunits(complete_text));limit=4096)
    entries=manifest["members"];need(entries isa AbstractVector && 1<=length(entries)<=1022,"completed case member bound")
    allowed=Set(("manifest.json","COMPLETE.json"));total=ncodeunits(manifest_text);identities=Dict{String,Any}();bodies=Dict{String,String}()
    for entry in entries
        J._object(entry,Set(("name","bytes","sha256")));name=entry["name"]
        need(name isa String && occursin(r"\Atable-[0-9]{2}-(features|labels)-[0-9]{4}\.bin\z",name) && !(name in allowed) &&
            F.isint(entry["bytes"]) && 0<entry["bytes"]<=8388608,"completed case member descriptor")
        body,identity=FS._read(joinpath(directory,name),8388608)
        need(ncodeunits(body)==entry["bytes"] && digest(codeunits(body))==J._sha(entry["sha256"]),"completed case member bytes differ")
        total+=ncodeunits(body);need(total<=1073741824,"completed case total bound")
        push!(allowed,name);identities[name]=identity;bodies[name]=body
    end
    request=decode(request_bytes)
    need(Set(names)==allowed && complete["manifest_sha256"]==value.manifest_sha256 && complete["members"]==length(names) &&
        complete["snapshot_bytes_without_marker"]==total && manifest["request_sha256"]==value.request_sha256 &&
        length(manifest["tables"])==length(value.features),"completed case manifest binding")
    for (index,table) in enumerate(manifest["tables"])
        f=value.features[index];l=value.labels[index]
        need(Q._metadata(request["tables"][index],f,l,table["feature_shards"],table["label_shards"])==table,"completed case immutable table differs")
        for (field,payload,width) in (("feature_shards",f.payload,160),("label_shards",l.payload,64))
            for (ordinal,name) in enumerate(table[field])
                first=(ordinal-1)*Q.ROWS_PER_SHARD*width+1;last=min(ordinal*Q.ROWS_PER_SHARD*width,ncodeunits(payload))
                need(codeunits(bodies[name])==codeunits(payload)[first:last],"completed case member differs from reverified value")
            end
        end
    end
    need(readdir(directory)==names && J._identity(joinpath(directory,"manifest.json"))==mi && J._identity(joinpath(directory,"COMPLETE.json"))==ci &&
        all(J._identity(joinpath(directory,n))==id for (n,id) in identities),"completed case changed during rebind")
    need(R.bounded_read(request_path;limit=131072)==request_bytes,"completed case request changed during rebind")
    nothing
end
function _read_completed(value,index,id,split,fit,directory,pins,witness_directory,witness_pins;actual,admission=nothing)
    point=F.read_published(fit.directory;expected=fit.point.expected,model_sha256=fit.pins.model_sha256,
        receipt_sha256=fit.pins.receipt_sha256,complete_sha256=fit.pins.complete_sha256)
    artifact=read_validation(directory,pins);context=decode(Vector{UInt8}(codeunits(artifact.context_json)))
    d=descriptor(value,index,id,split;operation="fixed_candidate_2017_validation_only",point,actual,admission)
    need(canonical(d)==canonical(context["descriptor"]),"completed witness source/model descriptor differs")
    raw=_read_payloads(witness_directory,"candidate_independent_numeric",witness_pins.manifest_sha256,witness_pins.complete_sha256;
        maximum=6,limit=MAX_INPUT_BYTES)
    need(Set(keys(raw.bodies))==Set(("request.json","receipt.json")) && raw.bodies["request.json"]==canonical(d),"completed witness request differs")
    report=decode(raw.bodies["receipt.json"]);J._object(report,Set(("schema","witness_sha256","authorization","validation_context_sha256",
        "validation_complete_sha256","fit_complete_sha256","cohort_sha256","rows","loss_bits")))
    n=context["calibration_numeric_context"]["rows"];rows=_independent_records(Vector{UInt8}(codeunits(artifact.records)),n)
    actual && need(n>=1000,"actual completed candidate minimum1000 rows")
    loss=_independent_loss(rows.targets,rows.predictions)
    need(report["schema"]=="candidate_independent_numeric_witness_v1" && report["validation_context_sha256"]==pins.context_sha256 &&
        report["validation_complete_sha256"]==pins.complete_sha256 && report["fit_complete_sha256"]==fit.pins.complete_sha256 &&
        report["cohort_sha256"]==context["cohort_sha256"] && F.isint(report["rows"]) && report["rows"]==n &&
        report["loss_bits"]==C.bits(loss) && (report["authorization"]!==nothing)===actual,"completed witness result bindings differ")
    witness=digest(canonical(Dict("descriptor_sha256"=>digest(canonical(d)),"validation_context_sha256"=>pins.context_sha256,
        "validation_complete_sha256"=>pins.complete_sha256,"point_complete_sha256"=>fit.pins.complete_sha256,
        "cohort_sha256"=>context["cohort_sha256"],"loss_bits"=>C.bits(loss),"job_sha256"=>job_identity())))
    need(witness==report["witness_sha256"],"completed witness numerical frame differs");raw.unchanged()
    VerifiedCandidate(d["station"],String(id),recipe_for(id).kind,context["cohort_sha256"],n,loss,String(canonical(d)),artifact,
        fit,actual,witness,actual ? String(canonical(report["authorization"])) : nothing,_VerifiedToken())
end
function read_completed_generated_candidate(value::Q.VerifiedCases,index,id,split,fit,directory,pins,witness_directory,witness_pins)
    candidate=_read_completed(value,index,id,split,fit,directory,pins,witness_directory,witness_pins;actual=false)
    independently_reconstructed=_verify_candidate(value,index,id,split,fit,directory,pins)
    need(independently_reconstructed.witness_sha256==candidate.witness_sha256,"generated completed witness source reconstruction differs")
    independently_reconstructed
end
function read_completed_candidate(value::B.ReverifiedCases,index,id,split,fit,directory,pins,witness_directory,witness_pins;
                                  request_path,case_directory,verification_authority=nothing,verification_native=nothing,
                                  producer=nothing,native=nothing,validation_producer=nothing,validation_native=nothing)
    need(all(isnothing,(verification_authority,verification_native,producer,native,validation_producer,validation_native)),
        "execution certificates are not accepted; supply scientific artifact pins")
    F._archival_request(value,index,split.start,split.stop,split.asof);_locked_split(split,:validation)
    _rebind_case_files(value.cases,request_path,case_directory)
    candidate=_read_completed(value.cases,index,id,split,fit,directory,pins,witness_directory,witness_pins;
        actual=true,admission=digest(B.admission_bytes(value)))
    rebuilt=verify_candidate_validation_archival(value,index,id,split,fit,directory,pins)
    need(rebuilt.witness_sha256==candidate.witness_sha256,"completed witness independent source reconstruction differs")
    _check_operation_binding(decode(Vector{UInt8}(codeunits(candidate.verification_authorization))),
        decode(Vector{UInt8}(codeunits(candidate.descriptor_json))))
    rebuilt
end

function choose_verified_R2(candidates::AbstractVector{VerifiedCandidate};independent_native=nothing)
    need(length(candidates)==16,"verified full16 candidates required")
    first_candidate=first(candidates);metadata=decode(Vector{UInt8}(codeunits(first_candidate.descriptor_json)))
    need(independent_native===nothing,"native certificates are not selection prerequisites")
    if first_candidate.actual
        for candidate in candidates
            need(candidate.verification_authorization!==nothing,"actual R2 requires independent numerical reconstruction")
            _check_operation_binding(decode(Vector{UInt8}(codeunits(candidate.verification_authorization))),
                decode(Vector{UInt8}(codeunits(candidate.descriptor_json))))
        end
    end
    for candidate in candidates
        need(candidate.actual===first_candidate.actual && candidate.station==first_candidate.station && candidate.cohort==first_candidate.cohort &&
            candidate.n==first_candidate.n,"verified candidate cohort/mode differs")
        d=decode(Vector{UInt8}(codeunits(candidate.descriptor_json)))
        need(all(d[k]==metadata[k] for k in ("case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index","role","split","policy_sha256","job_sha256")) &&
            d["job_sha256"]==job_identity(),"verified candidate input lineage differs")
    end
    metrics=[S.PointMetrics(;station=c.station,candidate_id=c.candidate_id,family=c.family,cohort=c.cohort,n=c.n,
        selection_log_mse=c.loss,phase_period_minutes=31) for c in candidates]
    decision=S.choose_R2(metrics,S.PointExpectation(;station=first_candidate.station,cohort=first_candidate.cohort,n=first_candidate.n,phase_period_minutes=31))
    ordered=Tuple(only(filter(c->c.candidate_id==spec.id,candidates)) for spec in S.POINT_GRID)
    evidence=digest(canonical(Dict("schema"=>"producer_checked_R2_evidence_v1","decision_sha256"=>decision.decision_sha256,
        "candidates"=>[Dict("id"=>c.candidate_id,"witness_sha256"=>c.witness_sha256) for c in ordered]);limit=16384))
    CheckedR2Selection(ordered,decision,first_candidate.actual,evidence,_SelectionToken())
end
function _selected_id(selection::CheckedR2Selection,family,split;actual)
    need(selection.actual===actual && family in (:ridge,:boost,:forest),"selected refit evidence mode/family")
    id=only(filter(w->w.family===family,selection.decision.family_winners)).candidate_id
    original=only(filter(c->c.candidate_id==id,selection.candidates))
    point=F._check_capsule(original.fit.point).context
    d=decode(Vector{UInt8}(codeunits(original.descriptor_json)))
    need(split.start==F._fit_clock(point["start_inclusive"]) && split.stop==F._fit_clock(d["split"]["stop_exclusive"]),"selected refit does not span training and selection")
    actual && _locked_split(split,:refit)
    id
end
function _selection_target(selection::CheckedR2Selection,value::Q.VerifiedCases,index;actual)
    need(selection.actual===actual,"R2 selection target evidence mode differs")
    req=case_metadata(value,index;actual)
    fields=Dict("case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,
        "case_complete_sha256"=>value.complete_sha256,"table_index"=>Int(index),"role"=>req["role"],
        "station"=>req["tables"][index]["station"],"layout"=>"ground11")
    need(all(begin
        d=decode(Vector{UInt8}(codeunits(candidate.descriptor_json)))
        candidate.actual===actual && candidate.station==fields["station"] && all(d[k]==v for (k,v) in fields)
    end for candidate in selection.candidates),"station-specific R2 selection does not belong to refit target")
    nothing
end
function refit_selected_generated(selection::CheckedR2Selection,value::Q.VerifiedCases,index,family,split::CandidateSplit,input;directory)
    _selection_target(selection,value,index;actual=false);_assert_fresh_destination(directory)
    id=_selected_id(selection,family,split;actual=false)
    fit=fit_candidate_generated(value,index,id,split,input;directory)
    binding=canonical(Dict("schema"=>"producer_checked_generated_R2_refit_v1","selection_evidence_sha256"=>selection.evidence_sha256,
        "decision_sha256"=>selection.decision.decision_sha256,"family"=>String(family),"candidate_id"=>id,
        "point"=>point_descriptor(fit.point),"actual_source_access"=>false,"software_qualification"=>false))
    SelectedRefit(fit,selection,family,id,String(binding))
end
function prepare_selected_refit_input_archival(selection::CheckedR2Selection,value::B.ReverifiedCases,index,family,split::CandidateSplit,directory;
                                               authority=nothing)
    _selection_target(selection,value.cases,index;actual=true)
    id=_selected_id(selection,family,split;actual=true)
    auth,d,attempt=_start_operation(value,index,id,split,directory,authority,:refitinput;selection)
    _prepare_input(value.cases,index,id,split,directory;authorization=auth,actual=true,admission=d["case_read_admission_sha256"],
        operation=d["operation"],selection_sha256=selection.evidence_sha256,attempt_bytes=attempt.bytes,
        closecheck=()->_close_operation(value,index,id,split,directory,authority,:refitinput,d,attempt;selection))
end
function refit_selected_archival(selection::CheckedR2Selection,value::B.ReverifiedCases,index,family,split::CandidateSplit,input;
                                 directory,authority=nothing,fit_authority=nothing)
    need(authority===nothing && fit_authority===nothing,"execution certificates are not accepted; supply checked selection and input pins")
    _selection_target(selection,value.cases,index;actual=true);_assert_fresh_destination(directory)
    id=_selected_id(selection,family,split;actual=true);admission=digest(B.admission_bytes(value))
    need(input.descriptor["operation"]=="selected_R2_2017_refit_input_only" &&
        input.descriptor["selection_sha256"]==selection.evidence_sha256,"selected input does not retain checked decision")
    expected=_match_input(value.cases,index,id,split,input;actual=true,admission)
    result=F.fit_archival(value,index,recipe_for(id),expected;start=split.start,stop=split.stop,asof=split.asof)
    p=F.publish_fit(result.capsule,directory)
    point=F.read_published(directory;expected,model_sha256=p.model_sha256,receipt_sha256=p.receipt_sha256,complete_sha256=p.complete_sha256)
    fit=FitPublication(point,String(directory),p)
    binding=canonical(Dict("schema"=>"producer_checked_archival_R2_refit_v1","selection_evidence_sha256"=>selection.evidence_sha256,
        "decision_sha256"=>selection.decision.decision_sha256,"family"=>String(family),"candidate_id"=>id,"point"=>point_descriptor(point),
        "actual_source_access"=>true,"software_qualification"=>false))
    SelectedRefit(fit,selection,family,id,String(binding))
end
