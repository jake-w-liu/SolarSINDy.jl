# BInputs private reader. Every actual reference comes from externally pinned
# ordinary files; no array-only construction supplies matched/actual authority.
const BReferencePins=Selection.BReferencePins
export BReferencePins,admit_reference
const REFERENCE_COMMON=(:complete,:evidence_mode,:reference_kind,:source_sha256,:input_prefix_sha256,:input_view_id,:delay_hours,:training_boundary,:validation_start,:validation_end,:training_history_start,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256,:baseline_sources)
const REFERENCE_COMPLETION_FIELDS=(:schema,REFERENCE_COMMON...,:audit_contract_sha256,:model_ids,:selected_setting_ids,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:audit_sha256,:members,:completion_id)
const REFERENCE_AUDIT_FIELDS=(:schema,REFERENCE_COMMON...,:audit_contract_sha256,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:all_eight_settings_checked,:full_per_lead_cohorts_checked,:feature_receipt_history_checked,:selected_prediction_columns_checked,:initial_only,:audit_id)
const REFERENCE_TUNING_FIELDS=(:schema,REFERENCE_COMMON...,:grid_sha256,:lead_order,:leads,:tuning_id)
const REFERENCE_LEAD_FIELDS=(:horizon,:training_mask_sha256,:validation_cohort_sha256,:validation_count,:max_fit_target_complete,:max_fit_label_receipt,:max_selection_target_complete,:max_selection_label_receipt,:candidates,:selected_index,:selected_model_id)
const REFERENCE_CANDIDATE_FIELDS=(:index,:max_depth,:nrounds,:nbins,:model_id,:fit_input_sha256,:prediction_member,:prediction_bytes,:prediction_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const REFERENCE_GRID=Tuple((max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255))
function _reference_id(data,field)
    names=Tuple(Symbol(k) for k in keys(data) if Symbol(k)!=field)
    T._need(T._id(NamedTuple{names}(Tuple(getproperty(data,k) for k in names)))==getproperty(data,field),"B reference metadata identity differs")
end
function _reference_qualification(path,pins,evidence_mode)
    body=P._bound(path);T._need(T._sha(body)==pins.baseline_qualification_sha256,"B reference component qualification pin differs")
    mode=evidence_mode==:synthetic_only ? "synthetic_only" : "software_qualified"
    P._software_qualification(body;schema="b_reference_component_software_qualification_v1",evidence_mode=mode,head=pins.baseline_qualified_head,sources=pins.baseline_sources)
    body
end
function _reference_geometry(v,boundary,start,stop,delay)
    masks=T._masks(v,boundary,start,stop,delay);indices=findall(masks.forecast)
    training=(Int[],Int[])
    for i in 1:v.source_hours,k in 1:2
        issue=T.Adapter._shift(v.input.time[i],1);h=(4,6)[k]
        anchor=i-delay>=1 ? T._clean(v,15,i-delay) : NaN
        if i+h<=v.source_hours&&isfinite(anchor)&&isfinite(T._clean(v,15,i+h))
            c=T.Adapter.forecast_clock(issue,h,delay)
            T.Adapter.fit_eligible(issue,c.target_complete,c.target_receipt,boundary)&&push!(training[k],i)
        end
    end
    leads=ntuple(k->[i for i in indices if masks.leads[k][i]],2)
    T._need(all(x->!isempty(x),training)&&all(x->!isempty(x),leads),"B reference full training/validation cohorts required")
    (;indices,leads,training)
end
# This canonical mask binds raw source/view, role, lead, split, delay and ALL
# ordered issue/completion/receipt triples. Candidate predictions cannot alter it.
function _reference_mask(v,indices,h,role,boundary,start,stop,delay)
    T._id((;schema="b_reference_per_lead_mask_v1",source_sha256=v.source_sha256,input_view_id=v.view_id,role,horizon=h,delay_hours=delay,training_boundary=boundary,validation_start=start,validation_end=stop,clocks=[begin issue=T.Adapter._shift(v.input.time[i],1);c=T.Adapter.forecast_clock(issue,h,delay);(;issue,target_complete=c.target_complete,target_receipt=c.target_receipt) end for i in indices]))
end
function _reference_metric(values,labels)
    T._need(!isempty(values)&&length(values)==length(labels),"B reference metric full cohort differs")
    errors=values.-labels
    result=(;n=length(errors),rmse_nt=sqrt(sum(abs2,errors)/length(errors)),mae_nt=sum(abs,errors)/length(errors),bias_nt=sum(errors)/length(errors))
    T._need(all(isfinite,(result.rmse_nt,result.mae_nt,result.bias_nt)),"B reference metric overflow")
    result
end
function _reference_tree(root,matched)
    names=String["completion.json","audit.json","predictions.bin","qualification.json"]
    if matched
        append!(names,["grid.json","tuning.json"])
        append!(names,["candidates/h$h/setting-$s.bin" for h in (4,6) for s in 1:8])
    end
    observed=String[];directories=String[];total=0
    for (parent,dirs,files) in walkdir(root;follow_symlinks=false)
        P._ordinary(parent)
        for name in dirs;P._ordinary(joinpath(parent,name));push!(directories,replace(relpath(joinpath(parent,name),root),'\\'=>'/'));end
        for name in files
            path=joinpath(parent,name);hash,bytes=P._file_sha(path);total+=bytes
            push!(observed,replace(relpath(path,root),'\\'=>'/'))
        end
    end
    T._need(sort(observed)==sort(names)&&Set(directories)==(matched ? Set(("candidates","candidates/h4","candidates/h6")) : Set{String}())&&total<=128*1024^2,"B reference exact bounded member tree differs")
    [(;name,bytes=P._file_sha(P._member(root,name))[2],sha256=P._file_sha(P._member(root,name))[1]) for name in sort(names) if name!="completion.json"]
end
function _check_reference_common(data,base,pins,v)
    T._need(data.complete===true,"B reference incomplete")
    for field in REFERENCE_COMMON;T._need(T._bytes(getproperty(data,field))==T._bytes(getproperty(base,field)),"B reference shared field differs: $field");end
    T._need(data.evidence_mode==String(v.evidence_mode)&&data.source_sha256==v.source_sha256&&data.input_prefix_sha256==v.input_prefix_sha256&&data.input_view_id==v.view_id,"B reference admitted view/source differs")
    T._int(data.delay_hours,0,1)
    for (field,value) in ((:feature_contract_sha256,pins.feature_contract_sha256),(:tuning_loss_contract_sha256,pins.tuning_loss_contract_sha256),(:baseline_qualification_sha256,pins.baseline_qualification_sha256));T._need(getproperty(data,field)==value,"B reference external contract/qualification pin differs");end
    T._need(P._sources(data.baseline_sources)==pins.baseline_sources,"B reference external component source map differs")
end
function _decode_reference(root,pins::BReferencePins,v,expected_sources)
    sources=P._source(expected_sources);T._recheck(v)
    T._need(pins.pins_id==T._id(Selection._pins_data(pins)),"B reference trusted pins mutated")
    c,cb=P._json(P._member(root,"completion.json");fields=REFERENCE_COMPLETION_FIELDS)
    T._need(T._sha(cb)==pins.completion_sha256,"B reference completion external pin differs");_reference_id(c,:completion_id)
    kind=Symbol(c.reference_kind);matched=kind==:matched_information_gbm_initial_v1
    T._need(kind in (:matched_information_gbm_initial_v1,:original_reference_diagnostic_only),"B reference role differs")
    T._need(c.schema==(matched ? "b_matched_reference_completion_v2" : "b_original_reference_completion_v1"),"B reference completion schema differs")
    members=_reference_tree(root,matched)
    T._need(length(c.members)==length(members),"B reference completion ledger count differs")
    for (entry,expected) in zip(c.members,members)
        T._fields(entry,(:name,:bytes,:sha256));T._int(entry.bytes,1,T.MAX_FILE)
        T._need(T._bytes(entry)==T._bytes(expected),"B reference completion ledger differs")
    end
    _check_reference_common(c,c,pins,v)
    boundary=T._clock(c.training_boundary);start=T._clock(c.validation_start);stop=T._clock(c.validation_end);delay=T._int(c.delay_hours,0,1)
    T._need(T._clock(c.training_history_start)==first(v.input.time)&&first(v.input.time)<boundary<=start<stop<=T.Adapter._shift(last(v.input.time),1),"B reference exact history/split differs")
    T._need(matched||delay==0,"original diagnostic adapter is L0 only")
    T._need(c.audit_sha256==pins.audit_sha256&&c.grid_sha256==pins.grid_sha256&&c.tuning_sha256==pins.tuning_sha256&&c.prediction_panel_sha256==pins.prediction_sha256&&T._hex(c.audit_contract_sha256),"B reference external audit/tuning/prediction pins differ")
    qb=_reference_qualification(P._member(root,"qualification.json"),pins,v.evidence_mode)
    audit,ab=P._json(P._member(root,"audit.json");fields=REFERENCE_AUDIT_FIELDS)
    T._need(T._sha(ab)==pins.audit_sha256&&audit.schema==(matched ? "b_matched_reference_audit_v2" : "b_original_reference_audit_v1"),"B reference independent audit pin/schema differs");_reference_id(audit,:audit_id)
    _check_reference_common(audit,c,pins,v)
    for field in (:audit_contract_sha256,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256);T._need(getproperty(audit,field)==getproperty(c,field),"B reference audit repeated pin differs");end
    for field in (:full_per_lead_cohorts_checked,:feature_receipt_history_checked,:selected_prediction_columns_checked,:initial_only);T._need(getproperty(audit,field)===true,"B reference independent audit incomplete");end
    T._need(audit.all_eight_settings_checked===(matched ? true : false),"B reference audit grid scope differs")
    geometry=_reference_geometry(v,boundary,start,stop,delay);indices=geometry.indices
    selected=P._member(root,"predictions.bin");ph,pbytes=P._file_sha(selected)
    T._need(ph==pins.prediction_sha256&&pbytes==16+16length(indices),"B reference selected prediction bytes differ")
    rows=T.BPointPredictionRow[]
    open(selected,"r") do io
        P._check_header(io,"SSBPPTS1",16)
        for i in indices
            means=(P._get_float(io),P._get_float(io));issue=T.Adapter._shift(v.input.time[i],1)
            push!(rows,T.BPointPredictionRow(issue,means,T._clean(v,15,i-delay),ntuple(k->T._clean(v,15,i+(4,6)[k]),2),true,:admitted_reference_prediction,v.source_sha256,delay))
        end
        T._need(eof(io),"B reference selected prediction extra bytes")
    end
    T._need(Selection._row_identity(rows)==c.raw_row_identity_sha256,"B reference raw union identity differs")
    T._need(length(c.model_ids)==length(c.selected_setting_ids)==2&&all(T._hex,c.model_ids),"B reference model/selection ID shape differs")
    maxima=Tuple{DateTime,DateTime}[]
    if matched
        grid,gb=P._json(P._member(root,"grid.json");fields=(:schema,:complete,:evidence_mode,:feature_contract_sha256,:settings,:grid_id))
        T._need(T._sha(gb)==pins.grid_sha256&&grid.schema=="fixed_eight_gbm_settings_v2"&&grid.complete===true&&grid.evidence_mode==String(v.evidence_mode)&&grid.feature_contract_sha256==pins.feature_contract_sha256&&length(grid.settings)==8,"B reference fixed grid differs");_reference_id(grid,:grid_id)
        for (i,item) in enumerate(grid.settings)
            T._fields(item,(:index,:max_depth,:nrounds,:nbins))
            T._need(T._int(item.index,1,8)==i&&all(T._int(getproperty(item,k),1,10000)==getproperty(REFERENCE_GRID[i],k) for k in keys(REFERENCE_GRID[i])),"B exact eight ordered configurations required")
        end
        tuning,tb=P._json(P._member(root,"tuning.json");fields=REFERENCE_TUNING_FIELDS)
        T._need(T._sha(tb)==pins.tuning_sha256&&tuning.schema=="b_matched_reference_tuning_v2"&&tuning.grid_sha256==pins.grid_sha256&&length(tuning.leads)==2&&collect(tuning.lead_order)==[4,6]&&all(x->x isa Integer&&!(x isa Bool),tuning.lead_order),"B reference tuning proof differs");_reference_id(tuning,:tuning_id)
        _check_reference_common(tuning,c,pins,v)
        for k in 1:2
            h=(4,6)[k];lead=tuning.leads[k];T._fields(lead,REFERENCE_LEAD_FIELDS)
            train=geometry.training[k];evaluation=geometry.leads[k]
            T._need(T._int(lead.horizon,4,6)==h&&T._int(lead.validation_count,1,T.MAX_ROWS)==length(evaluation)&&lead.training_mask_sha256==_reference_mask(v,train,h,:training,boundary,start,stop,delay)&&lead.validation_cohort_sha256==_reference_mask(v,evaluation,h,:validation,boundary,start,stop,delay),"B reference full per-lead masks differ")
            lastfit=T.Adapter.forecast_clock(T.Adapter._shift(v.input.time[last(train)],1),h,delay);lastselection=T.Adapter.forecast_clock(T.Adapter._shift(v.input.time[last(evaluation)],1),h,delay)
            for (field,value) in ((:max_fit_target_complete,lastfit.target_complete),(:max_fit_label_receipt,lastfit.target_receipt),(:max_selection_target_complete,lastselection.target_complete),(:max_selection_label_receipt,lastselection.target_receipt));T._need(T._clock(getproperty(lead,field))==value,"B reference fitting/selection clock differs");end
            push!(maxima,(lastfit.target_complete,lastfit.target_receipt))
            T._need(length(lead.candidates)==8,"B all eight tuning candidates required")
            stored_best=0;computed_best=0;stored_min=Inf;computed_min=Inf
            selected_index=T._int(lead.selected_index,1,8)
            T._need(T._int(c.selected_setting_ids[k],1,8)==selected_index&&lead.selected_model_id==c.model_ids[k],"B reference selected IDs disagree")
            evaluation_set=Set(evaluation);union_positions=findall(i->i in evaluation_set,indices)
            labels=[T._clean(v,15,i+h) for i in evaluation]
            for s in 1:8
                item=lead.candidates[s];T._fields(item,REFERENCE_CANDIDATE_FIELDS)
                T._need(T._int(item.index,1,8)==s&&all(T._int(getproperty(item,f),1,10000)==getproperty(REFERENCE_GRID[s],f) for f in keys(REFERENCE_GRID[s]))&&T._hex(item.model_id)&&T._hex(item.fit_input_sha256),"B reference candidate exact configuration/model differs")
                name="candidates/h$h/setting-$s.bin";path=P._member(root,name);hash,bytes=P._file_sha(path)
                T._need(item.prediction_member==name&&item.prediction_sha256==hash&&T._int(item.prediction_bytes,1,T.MAX_FILE)==bytes==16+8length(indices)&&T._int(item.n,1,T.MAX_ROWS)==length(evaluation),"B reference candidate prediction/cohort differs")
                values=open(path,"r") do io
                    P._check_header(io,"SSBRFPR1",8);x=[P._get_float(io) for _ in indices];T._need(eof(io),"B candidate extra bytes");x
                end
                metric=_reference_metric(values[union_positions],labels)
                stored=ntuple(j->T._float(getproperty(item,(:rmse_nt,:mae_nt,:bias_nt)[j])),3)
                for (actual,expected) in zip(stored,(metric.rmse_nt,metric.mae_nt,metric.bias_nt));T._need(abs(actual-expected)<=5e-12*max(1.,abs(expected)),"B reference maintained metric parity differs");end
                if stored[1]<stored_min;stored_best=s;stored_min=stored[1];end
                if metric.rmse_nt<computed_min;computed_best=s;computed_min=metric.rmse_nt;end
                if s==selected_index
                    T._need(item.model_id==lead.selected_model_id&&all(reinterpret(UInt64,values[j])==reinterpret(UInt64,rows[j].means[k]) for j in eachindex(rows)),"B selected prediction column differs bitwise from chosen candidate")
                end
                T._need(P._file_sha(path)==(hash,bytes),"B candidate changed during metric decode")
            end
            T._need(stored_best==computed_best==selected_index,"B stored/recomputed strict first minimum differs; epsilon ties are unsupported")
        end
    else
        for k in 1:2
            T._int(c.selected_setting_ids[k],1,10000)
            clock=T.Adapter.forecast_clock(T.Adapter._shift(v.input.time[last(geometry.training[k])],1),(4,6)[k],delay)
            push!(maxima,(clock.target_complete,clock.target_receipt))
        end
    end
    metadata=(;schema="b_point_reference_panel_v1",evidence_mode=v.evidence_mode,reference_kind=kind,source_sha256=v.source_sha256,delay_hours=delay,training_boundary=boundary,validation_start=start,validation_end=stop,training_history_start=first(v.input.time),model_ids=String.(c.model_ids),max_fit_target_complete=maximum(first,maxima),max_fit_label_receipt=maximum(last,maxima),feature_contract_sha256=pins.feature_contract_sha256,fixed_grid_manifest_sha256=pins.grid_sha256,selected_setting_ids=Int.(c.selected_setting_ids),tuning_loss_contract_sha256=pins.tuning_loss_contract_sha256,tuning_proof_completion_sha256=pins.tuning_sha256,prediction_panel_sha256=pins.prediction_sha256,raw_row_identity_sha256=String(c.raw_row_identity_sha256),reference_completion_sha256=pins.completion_sha256,audit_completion_sha256=pins.audit_sha256)
    provisional=Selection.BReferenceAudit(root,pins.audit_sha256,pins,v,copy(sources),repeat("0",64));witness=Selection.BReferenceAudit(root,pins.audit_sha256,pins,v,copy(sources),T._id(Selection._audit_data(provisional)))
    ref=Selection._reference(metadata,rows;audit=witness)
    T._need(P._bound(P._member(root,"completion.json"))==cb&&P._bound(P._member(root,"audit.json"))==ab&&P._bound(P._member(root,"qualification.json"))==qb&&P._file_sha(selected)==(ph,pbytes)&&T._bytes(_reference_tree(root,matched))==T._bytes(members),"B reference members changed during admission")
    T._recheck(v);T._need(P._source(expected_sources)==sources,"B reference admission sources changed");ref
end
function admit_reference(completion_path;trusted_pins::BReferencePins,roots,view::T.PointInputView,expected_sources)
    map=_roots(roots);path=P._ordinary(completion_path);root=dirname(path)
    T._need(basename(path)=="completion.json"&&root in values(map),"B reference completion must be in an independently supplied ordinary root")
    _decode_reference(root,trusted_pins,view,expected_sources)
end
function _check_reference_admission(ref)
    a=ref.audit;T._need(a isa Selection.BReferenceAudit&&a.admission_id==T._id(Selection._audit_data(a)),"B reference prediction-file witness differs")
    checked=_decode_reference(a.root,a.pins,a.view,a.expected_sources)
    T._need(checked.admission_id==ref.admission_id,"B reference metadata/prediction values differ from pinned files");true
end
