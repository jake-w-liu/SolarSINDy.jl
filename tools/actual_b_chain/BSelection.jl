module BSelection
using SolarSINDy,Dates,SHA,Statistics
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
export BReferencePanel,BMatchedReference,BL0Panels,BJointPanels,provisional_l0,select_joint_setting
struct BReferenceAudit
    path::String
    sha256::String
    grid_path::String
    grid_sha256::String
end
struct BReferencePanel
    metadata::NamedTuple
    rows::Vector{T.BPointPredictionRow}
    admission_id::String
    audit::Union{Nothing,BReferenceAudit}
end
struct BMatchedReference
    delays::NTuple{2,BReferencePanel}
end
struct BPointFilePanel
    path::String
    sha256::String
    reference::BReferencePanel
end
struct BL0Panels
    rows::Vector{Union{Vector{T.BPointPredictionRow},BPointFilePanel}}
end
BL0Panels(rows::Vector{Vector{T.BPointPredictionRow}})=BL0Panels(Union{Vector{T.BPointPredictionRow},BPointFilePanel}[x for x in rows])
struct BJointPanels
    delays::NTuple{2,BL0Panels}
end
struct BSelectionResult
    data::NamedTuple
end
function _row_identity(rows)
    T._id([(;issue=x.issue,anchor=isfinite(x.anchor) ? x.anchor : nothing,target=Tuple(isfinite(y) ? y : nothing for y in x.target),source_sha256=x.source_sha256,delay_hours=x.delay_hours) for x in rows])
end
function _reference(metadata,rows::Vector{T.BPointPredictionRow};audit=nothing)
    fields=Tuple(k for k in P.REFERENCE_PANEL_FIELDS if k!=:rows)
    T._fields(metadata,fields)
    T._need(metadata.schema=="b_point_reference_panel_v1"&&metadata.evidence_mode in (:synthetic_only,:archival_assumed_completion_research)&&metadata.reference_kind in (:original_reference_diagnostic_only,:matched_information_gbm_initial_v1),"B reference schema/role differs")
    T._int(metadata.delay_hours,0,1)
    T._need(metadata.training_boundary<=metadata.validation_start<metadata.validation_end&&metadata.max_fit_target_complete<=T.Adapter._shift(metadata.training_boundary,-168)&&metadata.max_fit_label_receipt<=T.Adapter._shift(metadata.training_boundary,-168),"B reference fitting/validation clocks differ")
    T._need(metadata.raw_row_identity_sha256==_row_identity(rows),"B reference raw row identity differs")
    for field in (:source_sha256,:feature_contract_sha256,:fixed_grid_manifest_sha256,:tuning_loss_contract_sha256,:tuning_proof_completion_sha256,:prediction_panel_sha256,:raw_row_identity_sha256,:reference_completion_sha256,:audit_completion_sha256)
        T._need(T._hex(getproperty(metadata,field)),"B reference proof digest differs")
    end
    T._need(length(rows)<=T.MAX_ROWS&&issorted(getproperty.(rows,:issue))&&allunique(getproperty.(rows,:issue)),"B reference row keys differ")
    for row in rows;T._need(row.source_sha256==metadata.source_sha256&&row.delay_hours==metadata.delay_hours&&all(isfinite,row.means),"B reference row source/prediction differs");end
    audit===nothing||T._need(audit isa BReferenceAudit,"typed source-bound B reference audit required")
    identity=T._id((;metadata,raw_rows=metadata.raw_row_identity_sha256,audit_sha256=audit===nothing ? nothing : audit.sha256))
    BReferencePanel(metadata,copy(rows),identity,audit)
end
const MATCHED_AUDIT_FIELDS=(:schema,:complete,:evidence_mode,:reference_kind,:source_sha256,:training_boundary,:validation_start,:validation_end,:training_history_start,:delay_hours,:feature_contract_sha256,:fixed_grid_manifest_sha256,:prediction_panel_sha256,:raw_row_identity_sha256,:reference_completion_sha256,:latest_observation_age,:missing_flags,:run_age,:channel_count,:same_history,:same_receipt_prefix,:fixed_grid_count,:initial_only,:audit_id)
function _check_matched_audit(ref)
    audit=ref.audit;T._need(audit isa BReferenceAudit,"matched reference needs its independently pinned audit and fixed-grid members")
    data,body=P._json(audit.path;fields=MATCHED_AUDIT_FIELDS);m=ref.metadata
    T._need(T._sha(body)==audit.sha256==m.audit_completion_sha256&&data.schema=="matched_information_gbm_audit_v1"&&data.complete===true&&data.reference_kind=="matched_information_gbm_initial_v1"&&data.evidence_mode==String(m.evidence_mode),"B matched audit schema/role/evidence differs")
    for field in (:source_sha256,:feature_contract_sha256,:fixed_grid_manifest_sha256,:prediction_panel_sha256,:raw_row_identity_sha256,:reference_completion_sha256)
        T._need(getproperty(data,field)==getproperty(m,field),"B matched audit source/panel differs")
    end
    for field in (:training_boundary,:validation_start,:validation_end,:training_history_start);T._need(T._clock(getproperty(data,field))==getproperty(m,field),"B matched audit history/split differs");end
    T._need(T._int(data.delay_hours,0,1)==m.delay_hours&&T._int(data.channel_count,17,17)==17&&T._int(data.fixed_grid_count,8,8)==8,"B matched audit information/grid differs")
    for field in (:latest_observation_age,:missing_flags,:run_age,:same_history,:same_receipt_prefix,:initial_only);T._need(getproperty(data,field)===true,"B matched information proof is incomplete");end
    fields=Tuple(k for k in MATCHED_AUDIT_FIELDS if k!=:audit_id)
    T._need(T._id(NamedTuple{fields}(Tuple(getproperty(data,k) for k in fields)))==data.audit_id,"B matched audit identity differs")
    grid,gridbody=P._json(audit.grid_path;fields=(:schema,:complete,:evidence_mode,:feature_contract_sha256,:settings,:grid_id))
    T._need(T._sha(gridbody)==audit.grid_sha256==m.fixed_grid_manifest_sha256&&grid.schema=="fixed_eight_gbm_settings_v1"&&grid.complete===true&&grid.evidence_mode==String(m.evidence_mode)&&grid.feature_contract_sha256==m.feature_contract_sha256&&length(grid.settings)==8,"B matched fixed-eight-grid proof differs")
    T._need([T._int(x.index,1,8) for x in grid.settings]==collect(1:8),"B matched grid IDs differ")
    identity=(;schema=grid.schema,complete=grid.complete,evidence_mode=grid.evidence_mode,feature_contract_sha256=grid.feature_contract_sha256,settings=grid.settings)
    T._need(T._id(identity)==grid.grid_id,"B matched grid identity differs");true
end
function _check_reference(ref::BReferencePanel;matched=false)
    checked=_reference(ref.metadata,ref.rows;audit=ref.audit);T._need(checked.admission_id==ref.admission_id,"B reference mutated")
    if matched
        T._need(ref.metadata.reference_kind==:matched_information_gbm_initial_v1,"matched-information GBM reference required")
        # The eight-setting tuning proof is externally pinned, never inferred
        # from an old GBM29 role string. Its decoder/admission belongs to BInputs.
        T._need(length(ref.metadata.selected_setting_ids)==2&&all(x->x isa Integer&&!(x isa Bool)&&1<=x<=8,ref.metadata.selected_setting_ids),"B matched fixed-eight-setting proof differs")
        T._need(ref.metadata.feature_contract_sha256!=repeat("0",64)&&ref.metadata.fixed_grid_manifest_sha256!=repeat("0",64)&&ref.metadata.tuning_proof_completion_sha256!=repeat("0",64),"B matched-information proof absent")
        _check_matched_audit(ref)
    end
    true
end
function _cohorts(ref,start,stop)
    _check_reference(ref);m=ref.metadata
    T._need((start,stop)==(m.validation_start,m.validation_end),"B reference validation range differs")
    indices=(Int[],Int[])
    for (i,row) in enumerate(ref.rows),lead in 1:2
        clock=T.Adapter.forecast_clock(row.issue,(4,6)[lead],m.delay_hours)
        if start<=row.issue<stop&&isfinite(row.anchor)&&isfinite(row.target[lead])&&T.Adapter.fit_eligible(row.issue,clock.target_complete,clock.target_receipt,stop)
            push!(indices[lead],i)
        end
    end
    union_indices=sort!(union(indices...))
    (;indices,union_indices,keys=[ref.rows[i].issue for i in union_indices])
end
function _candidate(rows,ref,cohort)
    T._need(length(rows)==length(cohort.keys)&&getproperty.(rows,:issue)==cohort.keys,"B candidate must contain exact complete per-lead union")
    for (row,index) in zip(rows,cohort.union_indices)
        reference=ref.rows[index]
        T._need(isequal(row.anchor,reference.anchor)&&isequal(row.target,reference.target)&&row.source_sha256==reference.source_sha256&&row.delay_hours==reference.delay_hours&&all(isfinite,row.means),"B candidate/reference raw identity differs")
    end
    Dict(row.issue=>row for row in rows)
end
_panel_rows(rows::Vector{T.BPointPredictionRow})=rows
function _panel_rows(panel::BPointFilePanel)
    ref=panel.reference;cohort=_cohorts(ref,ref.metadata.validation_start,ref.metadata.validation_end)
    hash,bytes=P._file_sha(panel.path);T._need(hash==panel.sha256&&bytes==16+16length(cohort.keys),"B point-panel bytes differ")
    rows=T.BPointPredictionRow[]
    open(panel.path,"r") do io
        P._check_header(io,"SSBPPTS1",16)
        for index in cohort.union_indices
            raw=ref.rows[index];means=(P._get_float(io),P._get_float(io))
            push!(rows,T.BPointPredictionRow(raw.issue,means,raw.anchor,raw.target,false,:point_artifact_only,raw.source_sha256,raw.delay_hours))
        end
        T._need(eof(io),"B point-panel extra bytes")
    end
    T._need(P._file_sha(panel.path)==(hash,bytes),"B point-panel changed during read");rows
end
function _metric(rows,lead)
    isempty(rows)&&throw(ArgumentError("empty B evaluation cohort"))
    T.K._precise() do
        pooled=Float64(sqrt(sum((BigFloat(x.means[lead])-BigFloat(x.target[lead]))^2 for x in rows)/length(rows)))
        storms=[x for x in rows if x.target[lead]<=-50]
        storm=isempty(storms) ? nothing : Float64(sqrt(sum((BigFloat(x.means[lead])-BigFloat(x.target[lead]))^2 for x in storms)/length(storms)))
        T._need(isfinite(pooled)&&(storm===nothing||isfinite(storm)),"B metric range unsupported")
        (;pooled,storm,count=length(rows),storm_count=length(storms))
    end
end
function _selection(status,selected,provisional,order,kind,completion,cohort_hash,reference_metrics,diagnostics,sources)
    data=(;schema="b_joint_delay_point_selection_v1",status,selected_index=selected,provisional_index=provisional,cell_order=order,reference_kind=kind,reference_completion_sha256=completion,cohort_sha256=cohort_hash,reference_metrics,settings=diagnostics,source_hashes=T._hashdata(sources))
    BSelectionResult(merge(data,(selection_id=T._id(data),)))
end
function _select(panels,refs,start,stop;primary)
    sources=T._hashmap(P.source_identity());order=primary ? (:L0h4,:L0h6,:L1h4,:L1h6) : (:L0h4,:L0h6)
    T._need(length(panels)==length(refs)&&(primary ? length(refs)==2 : length(refs)==1),"B delay cell count differs")
    for delay in eachindex(refs)
        T._need(refs[delay].metadata.delay_hours==delay-1&&length(panels[delay].rows)==36,"B full36 setting/delay panels required")
        _check_reference(refs[delay])
    end
    if primary&&any(r->r.metadata.reference_kind!=:matched_information_gbm_initial_v1||r.audit===nothing,refs)
        return _selection(:blocked_reference,nothing,nothing,order,:missing_matched_information_reference,refs[1].metadata.reference_completion_sha256,repeat("0",64),NamedTuple[],NamedTuple[],sources)
    end
    if primary
        foreach(r->_check_reference(r;matched=true),refs)
        a,b=refs[1].metadata,refs[2].metadata
        T._need((a.source_sha256,a.training_boundary,a.training_history_start,a.validation_start,a.validation_end)==(b.source_sha256,b.training_boundary,b.training_history_start,b.validation_start,b.validation_end),"B matched references do not share fold/history/source")
    end
    cohorts=[_cohorts(r,start,stop) for r in refs]
    reference_metrics=[_metric([refs[d].rows[i] for i in cohorts[d].indices[k]],k) for d in eachindex(refs) for k in 1:2]
    diagnostics=NamedTuple[];best=nothing;objective=Inf
    blocked=any(x->x.storm===nothing,reference_metrics)
    for index in 1:36
        metrics=NamedTuple[]
        for delay in eachindex(refs)
            map=_candidate(_panel_rows(panels[delay].rows[index]),refs[delay],cohorts[delay])
            for lead in 1:2
                push!(metrics,_metric([map[refs[delay].rows[i].issue] for i in cohorts[delay].indices[lead]],lead))
            end
        end
        qualified=all(metrics[i].pooled<=reference_metrics[i].pooled for i in eachindex(metrics))
        ratios=blocked ? Float64[] : [reference_metrics[i].storm==0 ? (metrics[i].storm==0 ? 1. : Inf) : metrics[i].storm/reference_metrics[i].storm for i in eachindex(metrics)]
        score=blocked ? Inf : sum(ratios)/length(ratios)
        push!(diagnostics,(;index,qualified,metrics,storm_ratios=[isfinite(x) ? x : "Inf" for x in ratios],objective=isfinite(score) ? score : "Inf"))
        if qualified&&isfinite(score)&&score<objective;best=index;objective=score;end
    end
    cohort_hash=T._id([(;delay=d-1,h4=[refs[d].rows[i].issue for i in cohorts[d].indices[1]],h6=[refs[d].rows[i].issue for i in cohorts[d].indices[2]],union=cohorts[d].keys) for d in eachindex(refs)])
    status=blocked ? :blocked_storm_reference : primary ? (best===nothing ? :none : :selected) : :provisional_l0
    result=_selection(status,primary&&!blocked ? best : nothing,!primary&&!blocked ? best : nothing,order,refs[1].metadata.reference_kind,refs[1].metadata.reference_completion_sha256,cohort_hash,reference_metrics,diagnostics,sources)
    foreach(_check_reference,refs);T._need(T._hashmap(P.source_identity())==sources,"B selector sources changed");result
end
provisional_l0(panels::BL0Panels,ref::BReferencePanel;validation_start,validation_end)=_select((panels,),(ref,),validation_start,validation_end;primary=false)
select_joint_setting(panels::BJointPanels,ref::BMatchedReference;validation_start,validation_end)=_select(panels.delays,ref.delays,validation_start,validation_end;primary=true)
end
