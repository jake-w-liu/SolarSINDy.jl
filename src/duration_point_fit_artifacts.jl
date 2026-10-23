struct BExpectedProvenance
    stage::Symbol
    evidence_mode::Symbol
    source_sha256::String
    input_view_id::String
    input_admission_sha256::String
    input_prefix_sha256::String
    training_boundary::DateTime
    validation_start::DateTime
    validation_end::DateTime
    reference_completion_sha256::String
    independent_audit_completion_sha256::String
    actual_execution_permit_sha256::Union{Nothing,String}
    delay_hours::Int
    scope_id::String
end
function _scope_data(s::BExpectedProvenance)
    (;stage=s.stage,evidence_mode=s.evidence_mode,source_sha256=s.source_sha256,input_view_id=s.input_view_id,input_admission_sha256=s.input_admission_sha256,input_prefix_sha256=s.input_prefix_sha256,training_boundary=s.training_boundary,validation_start=s.validation_start,validation_end=s.validation_end,reference_completion_sha256=s.reference_completion_sha256,independent_audit_completion_sha256=s.independent_audit_completion_sha256,actual_execution_permit_sha256=s.actual_execution_permit_sha256,delay_hours=s.delay_hours)
end
function _expected_scope(;stage,evidence_mode,source_sha256,input_view_id,input_admission_sha256,input_prefix_sha256,training_boundary,validation_start,validation_end,reference_completion_sha256,independent_audit_completion_sha256,actual_execution_permit_sha256,delay_hours)
    delay=T._int(delay_hours,0,0)
    _need(stage===:initial_l0&&evidence_mode in (:synthetic_only,:archival_assumed_completion_research),"B publication scope differs")
    _need(training_boundary<=validation_start<validation_end,"B expected publication split differs")
    for hash in (source_sha256,input_view_id,input_admission_sha256,input_prefix_sha256,reference_completion_sha256,independent_audit_completion_sha256);_need(T._hex(hash),"B expected authority digest differs");end
    _need(evidence_mode==:synthetic_only ? actual_execution_permit_sha256===nothing : T._hex(actual_execution_permit_sha256),"B expected permit/mode differs")
    fields=(stage,evidence_mode,String(source_sha256),String(input_view_id),String(input_admission_sha256),String(input_prefix_sha256),training_boundary,validation_start,validation_end,String(reference_completion_sha256),String(independent_audit_completion_sha256),actual_execution_permit_sha256===nothing ? nothing : String(actual_execution_permit_sha256),delay)
    provisional=BExpectedProvenance(fields...,repeat("0",64))
    BExpectedProvenance(fields...,_id(_scope_data(provisional)))
end
struct BFitExportContext
    binding::T.PointTrainingBinding
    view::T.PointInputView
    basis::BVerifiedBasisPanel
    expected::Union{Nothing,BExpectedProvenance}
end
BFitExportContext(binding,view,basis)=BFitExportContext(binding,view,basis,nothing)
function _expected_provenance(model::T.BPointModel,context::BFitExportContext)
    s=context.expected;_need(s isa BExpectedProvenance,"B bundle needs independently admitted expected provenance")
    _need(s.scope_id==_id(_scope_data(s))&&s.stage===:initial_l0&&s.delay_hours==0,"B expected publication scope mutated")
    b=context.binding;v=context.view;p=context.basis
    _check_panel(p,v)
    _need(model.evidence_mode==v.evidence_mode==s.evidence_mode&&model.source_sha256==b.source_sha256==v.source_sha256==s.source_sha256&&model.input_view_id==b.input_view_id==v.view_id==s.input_view_id&&model.input_admission_sha256==b.input_admission_sha256==v.input_admission_sha256==s.input_admission_sha256&&model.input_prefix_sha256==v.input_prefix_sha256==s.input_prefix_sha256,"B expected provenance source authority differs")
    _need(model.boundary==b.boundary==s.training_boundary&&model.delay_hours==b.delay_hours==s.delay_hours,"B expected provenance model clock differs")
    manifest,body=_json(_member(p.replay.directory,"manifest.json");fields=BASIS_MANIFEST_FIELDS)
    _need(_sha(body)==p.replay.manifest_sha256&&T._clock(manifest.boundary)==s.training_boundary&&T._clock(manifest.validation_start)==s.validation_start&&T._clock(manifest.validation_end)==s.validation_end&&T._int(manifest.delay_hours,0,1)==s.delay_hours,"B expected validation authority differs from independently replayed basis")
    (;schema="b_point_fit_provenance_v1",stage=s.stage,evidence_mode=s.evidence_mode,source_sha256=s.source_sha256,input_view_id=s.input_view_id,input_admission_sha256=s.input_admission_sha256,input_prefix_sha256=s.input_prefix_sha256,reference_completion_sha256=s.reference_completion_sha256,independent_audit_completion_sha256=s.independent_audit_completion_sha256,training_boundary=s.training_boundary,validation_start=s.validation_start,validation_end=s.validation_end,basis_sha256=p.basis_sha256,fit_input_manifest_sha256=model.fit_input_manifest_sha256,initial_hazard_fit_count=2,initial_gain_fit_count=36,final_fit_count=0,calibration_fit_count=0,exposed_development_scoring=false,scientific_verdict=false,serving_promotion=false,actual_execution_permit_sha256=s.actual_execution_permit_sha256)
end
function _check_provenance(data,model,context)
    _fields(data,PROVENANCE_FIELDS)
    for (field,value) in ((:initial_hazard_fit_count,2),(:initial_gain_fit_count,36),(:final_fit_count,0),(:calibration_fit_count,0));_need(T._int(getproperty(data,field),0,36)==value,"B provenance count differs");end
    _need(_bytes(data)==_bytes(_expected_provenance(model,context)),"B provenance differs from independently admitted authority")
    true
end
function _check_panel(p::BVerifiedBasisPanel,v)
    _check_verified(p.replay,v);index=2*(p.physical_schedule-1)+p.beta_index
    _need(p.basis_sha256==p.replay.data_sha256[index]&&p.training.basis_sha256==p.basis_sha256,"B verified panel identity differs")
    path=_member(p.replay.directory,"basis-$index.bin")
    _need(_file_sha(path)[1]==p.basis_sha256,"B verified panel bytes changed")
    _need(p.training.binding.context_id==p.replay.bindings[p.beta_index].context_id,"B verified panel binding differs")
    # Re-encode both concrete role partitions against the already replayed bytes.
    open(path,"r") do io
        _check_header(io,"SSBPBAS1",248)
        train=1;forecast=1
        for i in 1:v.source_hours
            row=if p.replay.training_mask[i]
                x=p.training.rows[train];train+=1;x
            elseif p.replay.forecast_mask[i]
                x=p.forecast.rows[forecast];forecast+=1;x
            else continue
            end
            _need(row.design.values.index==i&&read(io,248)==_basis_bytes(row),"B replayed panel alias/value changed")
        end
        _need(eof(io)&&train==length(p.training.rows)+1&&forecast==length(p.forecast.rows)+1,"B replayed role partitions differ")
    end
    true
end
function export_fit_input(produce::Function,directory::AbstractString,context::BFitExportContext,s::T.BSetting;max_source_rows=500000,max_chunk_rows=4096)
    sources=source_identity();T._setting(s);v=context.view;b=context.binding;p=context.basis
    maxrows=T._int(max_source_rows,1,T.MAX_ROWS);chunk=T._int(max_chunk_rows,1,T.MAX_CHUNK)
    _check_panel(p,v);_need(v.source_hours<=maxrows&&b.context_id==p.training.binding.context_id&&s.physical_schedule==p.physical_schedule&&s.beta==p.training.law.beta,"B export source/setting differs")
    root=_root_new(directory);data_io=nothing;mask_io=nothing;completion=nothing;errors=Any[]
    count=0;informative=0;storm=0;previous=nothing;maxcomplete=nothing;maxreceipt=nothing
    try
        flags=Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL
        data_io=Base.Filesystem.open(_member(root,"joint-fit.bin"),flags,0o600);mask_io=Base.Filesystem.open(_member(root,"fit-mask.ascii"),flags,0o600)
        _header(data_io,"SSBPFIT1",64)
        write(mask_io,"B-point-joint-mask-v1|$(b.fold_id)|$(Dates.value(b.boundary))|$(b.delay_hours)\n")
        consume=rows->begin
            _need(rows isa AbstractVector&&axes(rows)==(Base.OneTo(length(rows)),)&&1<=length(rows)<=chunk,"B export chunk bound/axes differ")
            for row in rows
                _need(row isa T.BTrainingRow,"B export accepts only typed training rows")
                count+=1;_need(count<=maxrows&&count<=length(p.training.rows),"B export row count exceeded")
                _need(_basis_bytes(row)==_basis_bytes(p.training.rows[count]),"B export is not the verified full training panel")
                d=row.design.values;issue=d.issue
                _need(previous===nothing||previous<issue,"B export issues reordered")
                a=T.Adapter.forecast_clock(issue,4,b.delay_hours);c=T.Adapter.forecast_clock(issue,6,b.delay_hours)
                _need(all(cl->T.Adapter.fit_eligible(issue,cl.target_complete,cl.target_receipt,b.boundary),(a,c)),"B export maturity differs")
                for k in 1:2
                    for x in (d.coefficients[k]...,row.target[k]-d.offsets[k],row.target[k])
                        _need(isfinite(x),"B export contains nonfinite fit scalar");write(data_io,htol(reinterpret(UInt64,x)))
                    end
                    storm+=row.target[k]<=-50
                end
                write(mask_io,join(Dates.value.((issue,a.anchor_tag,a.target_tag,a.target_complete,a.target_receipt,c.target_tag,c.target_complete,c.target_receipt)),"|")*"\n")
                informative+=T._informative(row);previous=issue;maxcomplete=c.target_complete;maxreceipt=c.target_receipt
            end
        end
        produce(consume)
        close(data_io);data_io=nothing;close(mask_io);mask_io=nothing
        _need(count==length(p.training.rows)&&count>0,"B export omitted complete joint-training rows")
        dh,db=_file_sha(_member(root,"joint-fit.bin"));mh,mb=_file_sha(_member(root,"fit-mask.ascii"))
        _need(db==16+64count&&mh==b.training_mask_sha256,"B export byte/mask geometry differs")
        manifest=(;schema="b_point_fit_input_v1",complete=true,evidence_mode=v.evidence_mode,role=:in_sample_two_stage_gain_design,contract_sha256=T.CONTRACT,fold_id=b.fold_id,context_id=b.context_id,boundary_ms=b.boundary,cutoff_ms=b.cutoff,delay_hours=b.delay_hours,source_sha256=b.source_sha256,input_prefix_sha256=b.input_prefix_sha256,input_view_id=b.input_view_id,input_admission_sha256=b.input_admission_sha256,source_hashes=T._hashdata(b.source_hashes),hazard_model_id=b.hazard_model_id,hazard_training_prefix_sha256=b.hazard_training_prefix_sha256,setting_index=s.index,a_setting_index=s.a_index,beta=s.beta,taus_bits=s.taus,sigma_bits=s.sigma,eta_bits=s.eta,max_source_rows=maxrows,max_chunk_rows=chunk,total_source_rows=v.source_hours,joint_origins=count,informative_origins=informative,fit_scalar_rows=2count,storm_scalar_rows=storm,data_member="joint-fit.bin",data_bytes=db,data_sha256=dh,mask_member="fit-mask.ascii",mask_bytes=mb,fit_mask_sha256=mh,fit_data_sha256=dh,basis_member_sha256=p.basis_sha256,max_fit_issue_ms=previous,max_target_complete_ms=maxcomplete,max_label_receipt_ms=maxreceipt)
        _fields(manifest,FIT_INPUT_FIELDS);completion=_bytes(manifest);_exclusive(_member(root,"manifest.json"),completion)
        _check_panel(p,v);_need(source_identity()==sources,"B fit-export sources changed")
    catch e;push!(errors,e)
    finally
        for io in (data_io,mask_io);io===nothing||try close(io) catch e;push!(errors,e);end;end
        try T._recheck(v);_need(source_identity()==sources,"B fit-export exit sources differ") catch e;push!(errors,e);end
        if !isempty(errors);try _invalidate(root,"manifest.json",completion) catch e;push!(errors,e);end;end
    end
    T._raise(errors);(;directory=root,manifest_sha256=_sha(completion))
end
function fit_point_gains_from_input(directory::AbstractString;expected_manifest_sha256,expected_sources,expected_context::T.PointTrainingBinding,view::T.PointInputView,basis::BVerifiedBasisPanel)
    sources=_source(expected_sources);_check_panel(basis,view);root=_ordinary(directory)
    _need(isdir(root)&&sort(readdir(root))==["fit-mask.ascii","joint-fit.bin","manifest.json"],"B fit-input membership differs")
    d,body=_json(_member(root,"manifest.json");fields=FIT_INPUT_FIELDS)
    _need(T._hex(expected_manifest_sha256)&&_sha(body)==expected_manifest_sha256&&d.schema=="b_point_fit_input_v1"&&d.complete===true&&d.role=="in_sample_two_stage_gain_design"&&d.contract_sha256==T.CONTRACT,"B fit-input schema/completion differs")
    b=expected_context;s=T.settings()[T._int(d.setting_index,1,36)]
    _need(b.context_id==basis.training.binding.context_id&&b.context_id==d.context_id&&T._id(T._binding_data(b))==b.context_id,"B expected fit context differs")
    _need(d.evidence_mode==String(view.evidence_mode)&&d.source_sha256==b.source_sha256&&d.input_view_id==view.view_id&&d.input_prefix_sha256==b.input_prefix_sha256&&d.input_admission_sha256==b.input_admission_sha256&&_sources(d.source_hashes)==b.source_hashes,"B fit source/evidence differs")
    _need(d.fold_id==b.fold_id&&T._clock(d.boundary_ms)==b.boundary&&T._clock(d.cutoff_ms)==b.cutoff&&T._int(d.delay_hours,0,1)==b.delay_hours&&d.hazard_model_id==b.hazard_model_id&&d.hazard_training_prefix_sha256==b.hazard_training_prefix_sha256,"B fit law/fold/clock differs")
    _need(T._int(d.a_setting_index,1,18)==s.a_index&&d.beta==s.beta&&Tuple(T._float.(d.taus_bits))==s.taus&&T._float(d.sigma_bits)==s.sigma&&T._float(d.eta_bits)==s.eta,"B fit setting grammar differs")
    _need(d.data_member=="joint-fit.bin"&&d.mask_member=="fit-mask.ascii"&&d.basis_member_sha256==basis.basis_sha256,"B fit member identity differs")
    T._int(d.max_source_rows,1,T.MAX_ROWS);T._int(d.max_chunk_rows,1,T.MAX_CHUNK)
    n=T._int(d.joint_origins,1,T.MAX_ROWS)
    _need(d.total_source_rows==view.source_hours&&view.source_hours<=d.max_source_rows&&d.fit_scalar_rows==2n&&d.informative_origins==count(T._informative,basis.training.rows)&&d.storm_scalar_rows==sum(x<=-50 for row in basis.training.rows for x in row.target),"B fit counts differ")
    data_path=_member(root,"joint-fit.bin");mask_path=_member(root,"fit-mask.ascii")
    data=_bound(data_path,T.MAX_FILE);mask=_bound(mask_path,T.MAX_FILE)
    _need(length(data)==d.data_bytes==16+64n&&_sha(data)==d.data_sha256==d.fit_data_sha256&&data==T._fit_data_bytes(basis.training),"B fit data does not match replayed design")
    _need(length(mask)==d.mask_bytes&&_sha(mask)==d.fit_mask_sha256==b.training_mask_sha256&&mask==T._fit_mask_bytes(basis.training),"B fit ledger does not match replayed mature mask")
    lastissue=last(basis.training.rows).design.values.issue;clock=T.Adapter.forecast_clock(lastissue,6,b.delay_hours)
    _need(T._clock(d.max_fit_issue_ms)==lastissue&&T._clock(d.max_target_complete_ms)==clock.target_complete&&T._clock(d.max_label_receipt_ms)==clock.target_receipt,"B fit maximum maturity differs")
    model=T._fit_point_gains(basis.training,s;fit_input_manifest_sha256=String(expected_manifest_sha256))
    _need(_bound(data_path,T.MAX_FILE)==data&&_bound(mask_path,T.MAX_FILE)==mask&&_bound(_member(root,"manifest.json"))==body,"B fit input changed during solve")
    _check_panel(basis,view);_need(source_identity()==Tuple(sources)||T._hashmap(source_identity())==sources,"B fit reader exit sources differ")
    _register_certificate!(basis,model;independently_checked=true)
    model
end
function _register_certificate!(basis::BVerifiedBasisPanel,model::T.BPointModel;independently_checked=false)
    ledger=basis.replay.certificates;index=model.setting_index
    identity=(;model_id=model.model_id,context_id=model.context_id,fit_data_sha256=model.fit_data_sha256,fit_mask_sha256=model.fit_mask_sha256,basis_sha256=basis.basis_sha256,solver_sha256=_id(model.solver),source_sha256=model.source_sha256,input_view_id=model.input_view_id)
    if haskey(ledger,index)
        _need(ledger[index]==identity,"B setting certificate/model changed within replay scope")
        return
    end
    _need(length(ledger)<36,"B certificate ledger exceeded fixed36 settings")
    if !independently_checked
        problem=T._fit_problem(basis.training,T.settings()[index])
        T._verify_solver(model.solver,problem.Z,problem.rhs,problem.weights)
    end
    ledger[index]=identity
end
