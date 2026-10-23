function _preflight_duration_artifact(root)
    # The caller owns a quiescent, source-bound B stage tree. This adds only
    # B's declared metadata resource guard; D.load_model keeps all its checks.
    for name in ("COMMITTED.json","MANIFEST.json","model.json")
        _json_depth_preflight(_bound(_member(root,name)))
    end
    true
end
function _embedded_law(root,law::D.DurationModel)
    D._validate(law)
    body=D._json(merge(D._data(law),(model_id=law.model_id,)))
    manifest=D._json((;schema="duration_model_manifest_v1",model_sha256=_sha(body)))
    _exclusive(_member(root,"model.json"),body);_exclusive(_member(root,"manifest.json"),manifest)
end
function _load_embedded_law(root)
    _need(sort(readdir(root))==["manifest.json","model.json"],"embedded B law membership differs")
    manifest=_bound(_member(root,"manifest.json"));body=_bound(_member(root,"model.json"))
    m=_bounded_json_parse(manifest);_fields(m,(:schema,:model_sha256))
    _need(m.schema=="duration_model_manifest_v1"&&m.model_sha256==_sha(body),"embedded duration manifest differs")
    d=_bounded_json_parse(body)
    fields=(:schema,:availability_kind,:beta,:boundary,:cutoff,:edges,:risks,:flips,:global_risks,:global_flips,:marks,:mark_counts,:training_prefix_sha256,:source_sha256,:source_signature,:quantile_rule,:mark_rule,:model_id)
    _fields(d,fields)
    _need(d.schema=="duration_forcing_research_v1"&&d.availability_kind=="assumed_completed_hour"&&d.quantile_rule=="nearest_rank_lower_bin_v1"&&d.mark_rule=="four_ceil_rank_blocks_v1","embedded duration schema differs")
    integer(x)=T._int(x,0,D.MAX_HOURS)
    cells(x)=(_need(length(x)==49&&all(r->length(r)==5,x),"embedded duration cell shape differs");ntuple(i->ntuple(j->integer(x[i][j]),5),49))
    integers(x,n)=(_need(length(x)==n,"embedded duration vector shape differs");Tuple(integer(v) for v in x))
    floats(x,n)=x===nothing ? nothing : (_need(length(x)==n,"embedded duration float shape differs");Tuple(D.K._number(v,"embedded duration float") for v in x))
    law=D.DurationModel(integer(d.beta),DateTime(String(d.boundary)),DateTime(String(d.cutoff)),floats(d.edges,3),cells(d.risks),cells(d.flips),integers(d.global_risks,2),integers(d.global_flips,2),floats(d.marks,4),integers(d.mark_counts,4),String(d.training_prefix_sha256),String(d.source_sha256),String(d.source_signature),String(d.model_id))
    D._validate(law)
    _need(_bound(_member(root,"manifest.json"))==manifest&&_bound(_member(root,"model.json"))==body,"embedded law changed during load")
    law
end
function _decode_solver(d)
    _fields(d,(:gains,:weighted_sse,:normalized_sse,:normalization_scale,:rank,:rank_cutoff,:selected_candidate,:candidates))
    pair(x)=(_need(length(x)==2,"B solver pair shape differs");Tuple(T._float.(x)))
    _need(length(d.candidates)==4,"B solver must retain four candidates")
    candidates=ntuple(4) do i
        c=d.candidates[i];_fields(c,(:gains,:feasible,:kkt,:normalized_sse,:gradient,:kkt_bound))
        _need(c.feasible isa Bool&&c.kkt isa Bool&&c.normalized_sse isa AbstractString&&c.kkt_bound isa AbstractString&&length(c.gradient)==2&&all(x->x isa AbstractString,c.gradient),"B solver candidate grammar differs")
        (;gains=pair(c.gains),feasible=c.feasible,kkt=c.kkt,normalized_sse=String(c.normalized_sse),gradient=Tuple(String.(c.gradient)),kkt_bound=String(c.kkt_bound))
    end
    (;gains=pair(d.gains),weighted_sse=T._float(d.weighted_sse),normalized_sse=T._float(d.normalized_sse),normalization_scale=T._float(d.normalization_scale),rank=T._int(d.rank,0,2),rank_cutoff=T._float(d.rank_cutoff),selected_candidate=T._int(d.selected_candidate,1,4),candidates)
end
function _decode_model(d,law)
    _fields(d,MODEL_FIELDS);s=T.settings()[T._int(d.setting_index,1,36)]
    _need(length(d.taus)==length(d.gains)==2,"B model pair shape differs")
    str(x)=(_need(x isa AbstractString,"B model string required");String(x))
    model=T.BPointModel(str(d.schema),Symbol(str(d.evidence_mode)),Symbol(str(d.training_role)),str(d.contract_sha256),str(d.fold_id),s.index,T._int(d.a_setting_index,1,18),T._int(d.beta,1,20),Tuple(T._float.(d.taus)),T._float(d.sigma),T._float(d.eta),T._int(d.delay_hours,0,1),T._clock(d.boundary),T._clock(d.cutoff),Tuple(T._float.(d.gains)),str(d.hazard_model_id),str(d.hazard_training_prefix_sha256),str(d.source_sha256),str(d.input_prefix_sha256),str(d.input_view_id),str(d.input_admission_sha256),str(d.context_id),_sources(d.source_hashes),T._int(d.fit_count,1,T.MAX_ROWS),T._int(d.informative_count,100,T.MAX_ROWS),T._int(d.storm_scalar_count,0,2T.MAX_ROWS),str(d.fit_mask_sha256),str(d.fit_data_sha256),str(d.fit_input_manifest_sha256),T._clock(d.max_fit_issue),T._clock(d.max_target_complete),T._clock(d.max_label_receipt),_decode_solver(d.solver),str(d.model_id),law)
    T._validate_model(model);model
end
function _decode_a(d)
    _fields(d,(:last_index,:last_issue,:source_sha256,:prefix_sha256,:delay_hours,:taus,:sigma,:sealed,:basis,:pending,:active,:consecutive_valid,:reset_tag))
    _fields(d.sealed,(:boundary,:mean,:covariance));_need(length(d.sealed.mean)==2&&length(d.sealed.covariance)==2&&all(x->length(x)==2,d.sealed.covariance)&&length(d.basis)==2&&all(x->length(x)==2,d.basis),"B A state shape differs")
    sealed=D.R.Checkpoint(T._clock(d.sealed.boundary),T._float.(d.sealed.mean),[T._float(d.sealed.covariance[i][j]) for i in 1:2,j in 1:2])
    pending=if d.pending===nothing;nothing
    else
        p=d.pending;_fields(p,(:index,:start,:stop,:q,:pressure));Stream.PendingDriver(T._int(p.index,1,T.MAX_ROWS),T._clock(p.start),T._clock(p.stop),T._float(p.q),T._float(p.pressure))
    end
    _need(d.active isa Bool&&length(d.taus)==2,"B A checkpoint grammar differs")
    Stream.StreamCheckpoint(T._int(d.last_index,0,T.MAX_ROWS),T._clock(d.last_issue),String(d.source_sha256),String(d.prefix_sha256),T._int(d.delay_hours,0,1),Tuple(T._float.(d.taus)),T._float(d.sigma),sealed,ntuple(i->ntuple(j->T._float(d.basis[i][j]),2),2),pending,d.active,T._int(d.consecutive_valid,0,T.MAX_ROWS),T._clock(d.reset_tag))
end
function _decode_wind(d)
    fields=(:schema,:source_sha256,:wind_sequence_sha256,:source_row_count,:implementation_hashes,:last_issue,:consumed_count,:input_prefix_sha256,:previous_tag,:previous_state,:age,:run,:checkpoint_id)
    _fields(d,fields)
    _sources(d.implementation_hashes)
    hashes=Tuple(String(x.name)=>String(x.sha256) for x in d.implementation_hashes)
    _need(hashes==C.source_identity(),"embedded wind implementation order/source differs")
    last=d.last_issue===nothing ? nothing : T._clock(d.last_issue);previous=d.previous_tag===nothing ? nothing : T._clock(d.previous_tag)
    cp=C.CurrentStateCheckpoint(String(d.schema),String(d.source_sha256),String(d.wind_sequence_sha256),T._int(d.source_row_count,0,T.MAX_ROWS),hashes,last,T._int(d.consumed_count,0,T.MAX_ROWS),String(d.input_prefix_sha256),previous,Symbol(d.previous_state),T._int(d.age,0,49),T._int(d.run,0,T.MAX_ROWS),String(d.checkpoint_id))
    _need(D._hash(C._checkpoint_data(cp))==cp.checkpoint_id,"embedded wind checkpoint identity differs");cp
end
function _check_closures(model,a,wind,context::BFitExportContext)
    _check_panel(context.basis,context.view);T._validate_model(model)
    proof=context.basis.replay;physical=context.basis.physical_schedule
    _need(model.context_id==context.binding.context_id&&model.hazard_model_id==context.basis.training.law.model_id&&T.settings()[model.setting_index].physical_schedule==physical,"B bundle replay/model binding differs")
    _need(_bytes(_a_data(a))==_bytes(_a_data(proof.a_checkpoints[physical]))&&_bytes(merge(C._checkpoint_data(wind),(checkpoint_id=wind.checkpoint_id,)))==_bytes(merge(C._checkpoint_data(proof.wind_checkpoint),(checkpoint_id=proof.wind_checkpoint.checkpoint_id,))),"B state closure differs from exact raw-source replay")
    _need(a.source_sha256==wind.source_sha256==model.source_sha256&&a.last_issue==wind.last_issue&&a.delay_hours==model.delay_hours,"B closure clock/source/delay differs")
    _need(model.gains==model.solver.gains&&model.fit_count==length(context.basis.training.rows)&&model.informative_count==count(T._informative,context.basis.training.rows)&&model.storm_scalar_count==sum(x<=-50 for row in context.basis.training.rows for x in row.target)&&model.fit_data_sha256==_sha(T._fit_data_bytes(context.basis.training))&&model.fit_mask_sha256==_sha(T._fit_mask_bytes(context.basis.training)),"B bundle fitted certificate/data differs")
    _register_certificate!(context.basis,model)
    true
end
function _closure_data(model,a,wind,sources)
    data=(;schema="b_point_state_closure_v1",role=:source_replayed_A_and_wind_state,model_id=model.model_id,source_sha256=model.source_sha256,input_admission_sha256=model.input_admission_sha256,a_checkpoint_type="A.StreamCheckpoint",a_checkpoint=_a_data(a),wind_checkpoint_type="DurationCurrentStateStream.CurrentStateCheckpoint",wind_checkpoint=merge(C._checkpoint_data(wind),(checkpoint_id=wind.checkpoint_id,)),closure_issue=a.last_issue,a_prefix_sha256=a.prefix_sha256,wind_prefix_sha256=wind.input_prefix_sha256,source_hashes=_sources_data(sources))
    merge(data,(state_closure_id=_id(data),))
end
function save_point_bundle(directory,model::T.BPointModel,a::Stream.StreamCheckpoint,wind::C.CurrentStateCheckpoint;provenance)
    sources=source_identity();context=provenance.verification
    _need(context isa BFitExportContext,"B bundle requires independently replayed state proof")
    _check_closures(model,a,wind,context);_check_provenance(provenance.data,model,context)
    root=_root_new(directory);completion=nothing;errors=Any[]
    try
        mkdir(_member(root,"hazard"));_embedded_law(_member(root,"hazard"),model.law)
        _exclusive(_member(root,"model.json"),_bytes(T._model_data(model;identity=true)))
        _exclusive(_member(root,"checkpoints.json"),_bytes(_closure_data(model,a,wind,sources)))
        _exclusive(_member(root,"provenance.json"),_bytes(provenance.data))
        members=[begin hash,bytes=_file_sha(_member(root,n));(;name=n,bytes,sha256=hash) end for n in ("model.json","checkpoints.json","provenance.json","hazard/model.json","hazard/manifest.json")]
        manifest=(;schema="b_duration_point_bundle_v1",complete=true,evidence_mode=model.evidence_mode,role=:two_stage_batch_fit_research_point_model,contract_sha256=T.CONTRACT,source_sha256=model.source_sha256,source_hashes=_sources_data(sources),model_id=model.model_id,setting_index=model.setting_index,delay_hours=model.delay_hours,hazard_model_id=model.hazard_model_id,clock_contract="assumed_completed_hour_h4_h6_v1",members)
        _fields(manifest,BUNDLE_FIELDS);completion=_bytes(manifest);_exclusive(_member(root,"manifest.json"),completion)
        _check_closures(model,a,wind,context);_need(source_identity()==sources,"B bundle sources changed")
    catch e;push!(errors,e)
    finally
        try T._recheck(context.view);_need(source_identity()==sources,"B bundle exit sources differ") catch e;push!(errors,e);end
        if !isempty(errors);try _invalidate(root,"manifest.json",completion) catch e;push!(errors,e);end;end
    end
    T._raise(errors);(;directory=root,manifest_sha256=_sha(completion))
end
function load_point_bundle(directory;expected_manifest_sha256,expected_sources,expected_binding::BFitExportContext)
    sources=_source(expected_sources);root=_ordinary(directory)
    _need(isdir(root)&&sort(readdir(root))==["checkpoints.json","hazard","manifest.json","model.json","provenance.json"],"B bundle membership differs")
    manifest,body=_json(_member(root,"manifest.json");fields=BUNDLE_FIELDS)
    _need(_sha(body)==expected_manifest_sha256&&manifest.schema=="b_duration_point_bundle_v1"&&manifest.complete===true&&manifest.contract_sha256==T.CONTRACT&&manifest.role=="two_stage_batch_fit_research_point_model"&&_sources(manifest.source_hashes)==sources,"B bundle completion/schema/source differs")
    names=("model.json","checkpoints.json","provenance.json","hazard/model.json","hazard/manifest.json")
    _need(length(manifest.members)==5,"B bundle ledger count differs")
    for (entry,name) in zip(manifest.members,names)
        _fields(entry,(:name,:bytes,:sha256));T._int(entry.bytes,1,T.MAX_META)
        hash,bytes=_file_sha(_member(root,name));_need(entry.name==name&&entry.bytes==bytes&&entry.sha256==hash,"B bundle member bytes differ")
    end
    law=_load_embedded_law(_member(root,"hazard"));data,modelbody=_json(_member(root,"model.json");fields=MODEL_FIELDS);model=_decode_model(data,law)
    _need(manifest.model_id==model.model_id&&manifest.source_sha256==model.source_sha256&&T._int(manifest.setting_index,1,36)==model.setting_index&&T._int(manifest.delay_hours,0,1)==model.delay_hours&&manifest.hazard_model_id==law.model_id&&manifest.evidence_mode==String(model.evidence_mode)&&manifest.clock_contract=="assumed_completed_hour_h4_h6_v1","B bundle/model identity differs")
    c,closurebody=_json(_member(root,"checkpoints.json");fields=STATE_CLOSURE_FIELDS)
    _need(c.schema=="b_point_state_closure_v1"&&c.a_checkpoint_type=="A.StreamCheckpoint"&&c.wind_checkpoint_type=="DurationCurrentStateStream.CurrentStateCheckpoint","B state type differs")
    a=_decode_a(c.a_checkpoint);wind=_decode_wind(c.wind_checkpoint)
    _need(_bytes(_closure_data(model,a,wind,sources))==closurebody,"B closure metadata/identity differs")
    provenance,provenancebody=_json(_member(root,"provenance.json");fields=PROVENANCE_FIELDS)
    _check_provenance(provenance,model,expected_binding)
    _check_closures(model,a,wind,expected_binding)
    for (name,saved) in (("manifest.json",body),("model.json",modelbody),("checkpoints.json",closurebody),("provenance.json",provenancebody));_need(_bound(_member(root,name))==saved,"B bundle changed during readback");end
    _need(T._hashmap(source_identity())==sources,"B bundle readback source differs")
    (;model,a_checkpoint=a,wind_checkpoint=wind,provenance)
end
