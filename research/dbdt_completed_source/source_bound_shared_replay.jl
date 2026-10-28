"""Generated source-bound shared replay; actual-data replay is separately admitted.

Every comparison uses the same fixed point predictions and source event log.
No empirical coverage, forecast skill or actual-source authority is asserted.
"""
module SourceBoundSharedReplay
using Dates,SHA,SolarSINDy
include("source_bound_calibration.jl")
include("source_bound_static_bounds.jl")
include("publisher_up_projection.jl")
const K=SourceBoundCalibration
const B=SourceBoundStaticBounds
const U=PublisherUPProjection
const F=SolarSINDy.ArchiveSourceBoundPointFit
const Q=SolarSINDy.ArchiveCatalogCaseJob
const C=SolarSINDy.ArchiveCatalogCases
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const R=D.CompletedRefitArtifacts
export replay_generated,ExpectedEvaluation,prepare_evaluation_binding
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
include("source_bound_evaluation_binding.jl")
mutable struct _GroundView{T} <: AbstractDict{DateTime,NTuple{2,Float64}}
    catalog::T
    first::DateTime
    last::DateTime
    asof::DateTime
    policy::D.CMR.ReplayPolicy
    reads::Int
end
function _read(view::_GroundView,tag)
    need(tag isa DateTime && D.whole(tag) && view.first<=tag<=view.last,"replay rate query outside declared window")
    need(D.feedback_ready(tag,view.policy)<=view.asof,"replay rate source precedes readiness")
    need(view.reads<4*(C.MAX_ROWS+30),"replay rate source read limit")
    view.reads+=1;row=C._locate(view.catalog,tag)
    row!==nothing && row.valid_mask==UInt8(3) ? row : nothing
end
Base.haskey(view::_GroundView,tag)=_read(view,tag)!==nothing
function Base.getindex(view::_GroundView,tag)
    row=_read(view,tag);row===nothing && throw(KeyError(tag));(row.values[1],row.values[2])
end
Base.length(::_GroundView)=throw(ArgumentError("replay rate source scans are forbidden"))
Base.iterate(::_GroundView)=throw(ArgumentError("replay rate source scans are forbidden"))
function _inputs(value::Q.VerifiedCases,index,point,calibration,expected,start,stop,asof;evaluation_binding=nothing)
    need(value.actual_source_authority_granted===false && value.actual_fit_performed===false,"generated replay cannot consume actual source authority")
    need(F.isint(index)&&1<=index<=length(value.features),"replay table index")
    need(all(t->t isa DateTime && D.whole(t),(start,stop,asof)) && start<stop<=asof,"replay split clocks")
    pc=F._check_capsule(point);need(!pc.actual,"generated replay cannot consume actual point fits")
    kc=K._checked(calibration,point,expected);need(!kc.actual,"generated replay cannot consume actual calibration")
    request=R.decode_canonical(Vector{UInt8}(codeunits(value.request_json));limit=131072)
    need(digest(codeunits(value.request_json))==value.request_sha256 && request["evidence_mode"]=="generated" &&
        request["runtime"]==string(VERSION),"replay source request differs")
    table=request["tables"][index];features=value.features[index]
    binding_pin=if evaluation_binding===nothing
        need(pc.context["case_request_sha256"]==value.request_sha256 && pc.context["case_manifest_sha256"]==value.manifest_sha256 &&
            pc.context["case_complete_sha256"]==value.complete_sha256,"separate evaluation case requires explicit binding")
        nothing
    else
        _verify_evaluation_binding(evaluation_binding,value,index,point,calibration,expected,start,stop,asof,pc,kc)
    end
    _common_inputs(value,index,point,calibration,expected,start,stop,asof,pc,kc,request,table,binding_pin)
end
"""Evaluate matching metadata and source rows after the caller admits its evidence mode."""
function _common_inputs(value,index,point,calibration,expected,start,stop,asof,pc,kc,request,table,binding_pin)
    features=value.features[index]
    need(pc.context["role"]==request["role"] &&
        pc.context["layout"]==request["layout"] && pc.context["table"]["station"]==table["station"]==features.station,
        "replay point/case/station identity differs")
    need(F._fit_clock(kc.context["stop_exclusive"])<=start && kc.first_issue<=start &&
        SolarSINDy.ArchiveCatalogSourceJob._clock(table["start_inclusive"])<=start<stop<=SolarSINDy.ArchiveCatalogSourceJob._clock(table["stop_exclusive"]) &&
        asof<=value.labels[index].asof,"replay overlaps calibration or exceeds sealed table")
    cal=K.completed_calibration(calibration,point,expected;first_issue=start)
    need(D.policy_equal(features.policy,cal.policy) && features.mode===cal.mode,"replay feature policy/layout differs")
    ground=getproperty(value.source.catalogs,Symbol("ground-"*features.station))
    wind=features.mode===:mixed8 ? getproperty(value.source.catalogs,Symbol("wind-"*features.station)) : nothing
    need(C._pins(ground,wind,features.mode)==features.source_pins,"replay sealed source pins differ")
    prepared=D.PreparedIssue[];row_ids=Int[];accepted=Int[];scales=Float64[];peaks=Float64[];reasons=Dict{String,Int}()
    for row_id in 1:length(features)
        row=C.feature_row(features,row_id);start<=row.issue<stop || continue
        checked=C._prepare(ground,row.issue;policy=features.policy,mode=features.mode,wind)
        need(checked.accepted===row.accepted && checked.reason===row.reason,"replay feature admission differs")
        if row.accepted
            need(checked.features==row.features && row.scale==1+checked.ground11[3],"replay feature values or scale differ")
            push!(prepared,D.PreparedIssue(row.issue,features.policy,features.mode,checked.timing,true,:complete,nothing,
                collect(checked.ground11),collect(checked.features)))
            push!(accepted,length(prepared));push!(scales,row.scale)
            peak=B._raw_peak(ground,row.issue,features.policy)
            need(log1p(peak)==checked.ground11[3],"replay raw peak differs from source-bound feature")
            push!(peaks,peak)
        else
            push!(prepared,D.PreparedIssue(row.issue,features.policy,features.mode,checked.timing,false,row.reason,nothing,nothing,nothing))
            reasons[String(row.reason)]=get(reasons,String(row.reason),0)+1;push!(scales,1.)
        end
        push!(row_ids,row_id)
    end
    need(length(prepared)==Dates.value(stop-start)÷60000,"replay lost scheduled feature rows")
    centers=zeros(length(prepared));width=features.mode===:ground11 ? 11 : 8
    matrix=Matrix{Float64}(undef,length(accepted),width)
    if !isempty(accepted)
        for (i,row) in enumerate(accepted),j in 1:width;matrix[i,j]=prepared[row].features[j];end
        predicted=F.model_predict(point.model,matrix)
        need(length(predicted)==length(accepted) && all(isfinite,predicted),"replay fixed point prediction domain")
        centers[accepted]=predicted
    end
    (;pc,kc,cal,ground,prepared,row_ids,accepted,centers,scales,peaks,matrix,reasons,features,table,binding_pin)
end
"""Run fixed consumers and both publisher variants on generated verified sources.

The final label cache does not determine issuance or provide replay feedback.
Missing-source targets remain unknown in the common event engine.
"""
function replay_generated(value::Q.VerifiedCases,index,point,calibration,expected;
        start,stop,asof,subscribers,nominal,grace_ms=0,warmup=true,static_calibration=nothing,evaluation_binding=nothing)
    need(warmup isa Bool,"replay warmup selector must be Boolean")
    input=_inputs(value,index,point,calibration,expected,start,stop,asof;evaluation_binding)
    alpha=D.nominal_value(nominal);need(0<alpha<1,"publisher comparison requires interior nominal")
    if static_calibration!==nothing
        need(static_calibration isa NamedTuple && keys(static_calibration)==(:capsule,:expected),"static calibration and external pins required")
        checked=B.check_calibration(static_calibration.capsule,point,calibration,expected;expected_static=static_calibration.expected)
        need(checked.numeric.alpha==alpha,"static/adaptive nominal differs")
    end
    specs=Tuple(subscribers);need(!isempty(specs) && all(s->s isa E.SubscriberSpec && s.nominal==alpha,specs),"replay subscriber nominal differs")
    audit=(;scheduled=length(input.prepared),features_accepted=length(input.accepted),features_rejected=length(input.prepared)-length(input.accepted),
        rejection_reasons=copy(input.reasons),source_bound_provenance_verified=false,scientific_coverage_qualified=false,
        measured_receipts_available=false,actual_source_authority_granted=false,actual_fits=0,
        mandatory_static_inventory_complete=static_calibration!==nothing,explicit_evaluation_binding_matched=input.binding_pin!==nothing)
    isempty(input.accepted) && return (;status=:no_feature_eligible,audit,shared=nothing,publisher_upper=nothing,publisher_absolute=nothing,static_bounds=nothing,study_pass=false,evaluation_binding_sha256=input.binding_pin)
    fixed=static_calibration===nothing ? nothing : B.predict_upper(static_calibration.capsule,input.matrix,input.centers[input.accepted],
        input.scales[input.accepted],input.peaks;point,calibration,expected,expected_static=static_calibration.expected,start)
    result=_execute_common(input;start,stop,asof,specs,alpha,grace_ms,warmup,fixed,audit,protocol_id="generated-source-bound-shared31-v1")
    need(F._check_capsule(point).context==input.pc.context && K._checked(calibration,point,expected).context==input.kc.context,"replay fitted inputs changed")
    if evaluation_binding!==nothing
        _verify_evaluation_binding(evaluation_binding,value,index,point,calibration,expected,start,stop,asof,input.pc,input.kc)
    end
    result
end
"""Run the unchanged numeric event and publisher engines on admitted prepared inputs."""
function _execute_common(input;start,stop,asof,specs,alpha,grace_ms,warmup,fixed,audit,protocol_id,source_id=nothing)
    need(source_id===nothing || (source_id isa String && 0<ncodeunits(source_id)<=512),"replay receipt source identity")
    last_center=input.prepared[end].issue+Minute(30)
    source=_GroundView(input.ground,start,last_center,asof,input.features.policy,0)
    receipts=D.materialize_receipts(source,start+Minute(1):Minute(1):last_center;policy=input.features.policy,asof,scheduled=false)
    batches=source_id===nothing ? receipts.batches : ((;received=batch.received,
        observations=[(;row...,source_id) for row in batch.observations]) for batch in receipts.batches)
    shared=E.replay_shared(input.prepared,input.centers,input.scales,batches,input.cal;
        subscribers=specs,stream_id=input.table["id"],protocol_id,asof,grace_ms)
    period=E.base_period(start,input.cal.policy)
    records=warmup ? [(;id=i,issue=input.kc.arrays.issues[i],label_ready=input.kc.arrays.ready[i],completed_at=input.kc.arrays.ready[i],
        phase_period_minutes=period,point_model_sha256=input.cal.point_model_sha256,policy=input.cal.policy,mode=input.cal.mode,
        signed_score=input.kc.arrays.scores[i]) for i in eachindex(input.kc.arrays.scores)] : NamedTuple[]
    upper=U.project_shared(shared,input.cal,records;start,asof,nominal=alpha,score_kind=:upper_positive,grace_ms)
    absolute=U.project_shared(shared,input.cal,records;start,asof,nominal=alpha,score_kind=:symmetric_absolute,grace_ms)
    need(shared.source.attempted==upper.attempted==absolute.attempted==length(input.accepted),"replay comparison denominator differs")
    (;status=:replayed,audit,shared,evaluation_binding_sha256=input.binding_pin,publisher_upper=upper,publisher_absolute=absolute,static_bounds=fixed,
      source_reads=source.reads,receipt_rejections=receipts.rejections,receipt_pending=receipts.pending,
      study_pass=fixed!==nothing && fixed.all_bounds_finite_nonempty && shared.study_pass && upper.own_numeric_study_pass && absolute.own_numeric_study_pass,
      row_ids=copy(input.row_ids),accepted_rows=input.row_ids[input.accepted],centers=copy(input.centers),scales=copy(input.scales))
end
end
