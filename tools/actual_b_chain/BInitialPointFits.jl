module BInitialPointFits
using SolarSINDy,Dates,SHA,JSON3,LinearAlgebra
using ..BInputs,..BSelection
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const D=SolarSINDy.DurationConditionedForcing
const REPO=normpath(joinpath(@__DIR__,"..",".."))
const OWNER_SHA="1c593d90793f6f70e05622772aec83b53919aa2c22575ab738a4ee1968f116f0"
export InitialPointExecutionPermit,run_initial_l0,verify_initial_l0
struct InitialPointExecutionPermit
    path::String
    sha256::String
end
function _permission(a,permit)
    if a.view.evidence_mode==:synthetic_only
        T._need(permit===nothing,"synthetic B execution rejects actual permits");return nothing
    end
    T._need(permit isa InitialPointExecutionPermit,"new qualified B initial permit required")
    d,body=P._json(permit.path;fields=P.PERMIT_FIELDS)
    T._need(T._sha(body)==permit.sha256&&d.schema=="qualified_actual_initial_b_l0_point_permit_v1"&&d.complete===true&&d.stage=="initial_l0"&&d.training_semantics=="option_T_in_sample_two_stage_batch_v1"&&d.contract_sha256==T.CONTRACT,"B actual permit schema/semantics differs")
    for (name,value) in ((:initial_hazard_fit_count,2),(:initial_gain_fit_count,36),(:state_schedule_count,9),(:beta_design_count,18),(:final_fit_count,0),(:calibration_fit_count,0),(:delay_hours,0),(:embargo_hours,168))
        T._need(T._int(getproperty(d,name),0,500000)==value,"B actual permit count/clock differs")
    end
    for name in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion);T._need(getproperty(d,name)===false,"B actual permit exceeds initial scope");end
    T._need(collect(d.beta_order)==[5,20]&&all(x->x isa Integer&&!(x isa Bool),d.beta_order)&&collect(d.b_setting_indices)==collect(1:36)&&all(x->x isa Integer&&!(x isa Bool),d.b_setting_indices),"B actual fixed-grid order differs")
    T._need(d.source_sha256==a.view.source_sha256&&d.input_prefix_sha256==a.view.input_prefix_sha256&&d.input_view_id==a.view.view_id&&d.input_envelope_sha256==a.envelope_sha256&&d.admitted_input_seal==a.admission_id&&d.reference_completion_sha256==a.metadata.reference_completion_sha256&&d.reference_audit_sha256==a.metadata.reference_audit_sha256,"B actual permit admission/reference differs")
    for (key,value) in ((:training_boundary,a.metadata.training_boundary),(:validation_start,a.metadata.validation_start),(:validation_end,a.metadata.validation_end));T._need(T._clock(getproperty(d,key))==value,"B actual permit split differs");end
    own=T._hashmap(P.source_identity());T._need(P._sources(d.implementation_sources)==own,"B actual permit source map differs")
    T._fields(d.limits,keys(P.LIMITS))
    T._need(all(T._int(getproperty(d.limits,k),1,getproperty(P.LIMITS,k))==getproperty(P.LIMITS,k) for k in keys(P.LIMITS)),"B actual permit limits differ")
    T._need(d.qualification_file isa AbstractString&&isabspath(d.qualification_file)&&T._hex(d.qualification_sha256),"B qualification path/pin grammar differs")
    T._need(Threads.nthreads()==1&&BLAS.get_num_threads()==1&&d.julia_version==string(VERSION)&&d.julia_binary_sha256==P._file_sha(joinpath(Sys.BINDIR,Base.julia_exename()))[1],"B actual runtime differs")
    T._need(d.native_owner_helper_sha256==OWNER_SHA==get(ENV,"SOLARSINDY_B_NATIVE_OWNER_SHA256",""),"B actual owned-native controller binding absent")
    head=readchomp(Cmd(["git","-C",REPO,"rev-parse","HEAD"]))
    T._need(d.qualified_head==head&&isempty(readchomp(Cmd(["git","-C",REPO,"status","--porcelain"]))),"B actual qualified clean head differs")
    qualification=P._bound(d.qualification_file);T._need(T._sha(qualification)==d.qualification_sha256,"B qualification pin differs")
    P._software_qualification(qualification;schema="b_point_execution_software_qualification_v1",evidence_mode="software_qualified",head,sources=own)
    (;path=String(permit.path),sha256=String(permit.sha256),qualification_path=String(d.qualification_file),qualification_sha256=String(d.qualification_sha256),qualified_head=head)
end
function _permission_recheck(permission)
    permission===nothing&&return
    T._need(P._file_sha(permission.path)[1]==permission.sha256&&P._file_sha(permission.qualification_path)[1]==permission.qualification_sha256,"B actual authority changed during stage")
end
function _reference(a,reference)
    BSelection._check_reference(reference);m=reference.metadata
    T._need(m.evidence_mode==a.view.evidence_mode&&m.delay_hours==0&&m.source_sha256==a.view.source_sha256&&m.training_boundary==a.metadata.training_boundary&&m.validation_start==a.metadata.validation_start&&m.validation_end==a.metadata.validation_end&&m.reference_kind==a.metadata.reference_kind&&m.reference_completion_sha256==a.metadata.reference_completion_sha256&&m.audit_completion_sha256==a.metadata.reference_audit_sha256,"B initial reference/admission differs")
    masks=T._masks(a.view,m.training_boundary,m.validation_start,m.validation_end,0)
    cohort=BSelection._cohorts(reference,m.validation_start,m.validation_end)
    keys=[T.Adapter._shift(a.view.input.time[i],1) for i in findall(masks.forecast)]
    T._need(cohort.keys==keys,"B reference union does not match admitted raw source")
    for (index,row) in zip(findall(masks.forecast),[reference.rows[i] for i in cohort.union_indices])
        T._need(isequal(row.anchor,T._clean(a.view,15,index))&&isequal(row.target,ntuple(k->T._clean(a.view,15,index+(4,6)[k]),2)),"B reference raw labels/anchor differ from admitted source")
    end
    true
end
function _outside_inputs(a,directory)
    output=lowercase(replace(abspath(directory),'\\'=>'/'))
    for root in vcat(collect(values(a.roots)),[REPO])
        input=lowercase(replace(realpath(root),'\\'=>'/'))
        T._need(output!=input&&!startswith(output,input*"/")&&!startswith(input,output*"/"),"B output overlaps input/repository root")
    end
end
function _tree(root;exclude_completion=true)
    members=NamedTuple[];directories=0;bytes=0;directory_names=String[]
    for (parent,dirs,files) in walkdir(root;follow_symlinks=false)
        P._ordinary(parent);directories+=length(dirs)
        for name in dirs
            P._ordinary(joinpath(parent,name));push!(directory_names,replace(relpath(joinpath(parent,name),root),'\\'=>'/'))
        end
        for name in files
            relative=replace(relpath(joinpath(parent,name),root),'\\'=>'/')
            exclude_completion&&relative=="completion.json"&&continue
            hash,size=P._file_sha(joinpath(parent,name));bytes+=size;push!(members,(;name=relative,bytes=size,sha256=hash))
        end
    end
    T._need(bytes<=16*1024^3&&length(members)<=2048&&directories<=256,"B stage output budget exceeded")
    expected=String["support.json","provisional-selection.json","basis/manifest.json"]
    append!(expected,["basis/basis-$i.bin" for i in 1:18])
    for beta in (5,20),name in ("model.json","MANIFEST.json","COMMITTED.json");push!(expected,"laws/beta-$beta/$name");end
    for i in 1:36
        for name in ("joint-fit.bin","fit-mask.ascii","manifest.json");push!(expected,"fits/setting-$i/$name");end
        for name in ("model.json","checkpoints.json","provenance.json","manifest.json","hazard/model.json","hazard/manifest.json");push!(expected,"models/setting-$i/$name");end
        push!(expected,"points/setting-$i.bin");push!(expected,"gain-journal/setting-$i.json")
    end
    exclude_completion||push!(expected,"completion.json")
    T._need(sort(getproperty.(members,:name))==sort(expected),"B stage exact file membership differs")
    expected_dirs=Set{String}()
    for name in expected
        parent=dirname(name)
        while !isempty(parent)&&parent!="."
            push!(expected_dirs,replace(parent,'\\'=>'/'));parent=dirname(parent)
        end
    end
    T._need(Set(directory_names)==expected_dirs,"B stage exact directory membership differs")
    sort!(members;by=x->x.name);(;members,bytes,files=length(members),directories,sha256=T._id(members))
end
function _point_file(path,rows)
    io=IOBuffer();P._header(io,"SSBPPTS1",16)
    for row in rows,x in row.means;T._need(isfinite(x),"finite B point required");P._put_float(io,x);end
    P._exclusive(path,take!(io));P._file_sha(path)
end
function _provenance(a,model,panel,permission)
    (;schema="b_point_fit_provenance_v1",stage=:initial_l0,evidence_mode=a.view.evidence_mode,source_sha256=a.view.source_sha256,input_view_id=a.view.view_id,input_admission_sha256=a.view.input_admission_sha256,input_prefix_sha256=a.view.input_prefix_sha256,reference_completion_sha256=a.metadata.reference_completion_sha256,independent_audit_completion_sha256=a.metadata.audit_completion_sha256,training_boundary=a.metadata.training_boundary,validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end,basis_sha256=panel.basis_sha256,fit_input_manifest_sha256=model.fit_input_manifest_sha256,initial_hazard_fit_count=2,initial_gain_fit_count=36,final_fit_count=0,calibration_fit_count=0,exposed_development_scoring=false,scientific_verdict=false,serving_promotion=false,actual_execution_permit_sha256=permission===nothing ? nothing : permission.sha256)
end
function _export_context(a,panel,permission)
    scope=P._expected_scope(;stage=:initial_l0,evidence_mode=a.view.evidence_mode,source_sha256=a.view.source_sha256,input_view_id=a.view.view_id,input_admission_sha256=a.view.input_admission_sha256,input_prefix_sha256=a.view.input_prefix_sha256,training_boundary=a.metadata.training_boundary,validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end,reference_completion_sha256=a.metadata.reference_completion_sha256,independent_audit_completion_sha256=a.metadata.audit_completion_sha256,actual_execution_permit_sha256=permission===nothing ? nothing : permission.sha256,delay_hours=a.metadata.delay_hours)
    P.BFitExportContext(panel.training.binding,a.view,panel,scope)
end
function run_initial_l0(a::BInputs.BPrefixAdmission,reference::BSelection.BReferencePanel,directory;execution_permit=nothing,chunk_hours=24)
    sources=P.source_identity();BInputs.recheck(a);_reference(a,reference);permission=_permission(a,execution_permit)
    T._need(a.metadata.delay_hours==0&&Threads.nthreads()==1&&BLAS.get_num_threads()==1,"B1 only single-threaded initial L0 supported")
    chunk=T._int(chunk_hours,1,T.MAX_CHUNK);_outside_inputs(a,directory);root=P._root_new(directory);completion=nothing;errors=Any[];answer=nothing
    try
        for name in ("laws","fits","models","points","gain-journal");mkdir(P._member(root,name));end
        laws=ntuple(bi->D.fit_duration(a.view.wind,a.metadata.training_boundary;beta=(5,20)[bi],source_sha256=a.view.source_sha256),2)
        lawpins=[D.save_model(P._member(root,"laws/beta-$(law.beta)"),law) for law in laws]
        produced=P._export_basis(a.view,laws,P._member(root,"basis");delay_hours=0,chunk_hours=chunk,validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end)
        proof=P._verify_basis(a.view,laws,produced.directory;expected_manifest_sha256=produced.manifest_sha256,expected_sources=P.source_identity(),chunk_hours=chunk==1 ? 2 : 1)
        support=NamedTuple[]
        for physical in 1:9,bi in 1:2
            panel=P._read_verified_panel(proof,a.view,physical,bi)
            for eta_index in 1:2
                setting=T.settings()[4*(physical-1)+2*(eta_index-1)+bi]
                informative=count(T._informative,panel.training.rows)
                T._need(T._sha(T._fit_mask_bytes(panel.training))==panel.training.binding.training_mask_sha256,"B preflight full joint mask differs")
                push!(support,(;setting_index=setting.index,joint_origins=length(panel.training.rows),informative_origins=informative,supported=informative>=100))
            end
        end
        sort!(support;by=x->x.setting_index)
        P._exclusive(P._member(root,"support.json"),T._bytes((;schema="b_all36_support_preflight_v1",all36_checked=length(support)==36,accepted=all(x->x.supported,support),selected_index=nothing,gain_fit_count=0,settings=support)))
        T._need(length(support)==36&&all(x->x.supported,support),"B all36 support preflight failed; no gain solve permitted")
        models=Vector{T.BPointModel}(undef,36);fitpins=Vector{NamedTuple}(undef,36);pointpins=Vector{NamedTuple}(undef,36)
        gain_count=0
        for physical in 1:9
            panels=ntuple(bi->P._read_verified_panel(proof,a.view,physical,bi),2)
            for eta_index in 1:2,bi in 1:2
                index=4*(physical-1)+2*(eta_index-1)+bi;s=T.settings()[index];panel=panels[bi]
                context=_export_context(a,panel,permission)
                fit=P.export_fit_input(P._member(root,"fits/setting-$index"),context,s;max_chunk_rows=chunk) do consume
                    for first in 1:chunk:length(panel.training.rows);consume(panel.training.rows[first:min(first+chunk-1,end)]);end
                end
                model=P.fit_point_gains_from_input(fit.directory;expected_manifest_sha256=fit.manifest_sha256,expected_sources=P.source_identity(),expected_context=context.binding,view=a.view,basis=panel)
                gain_count+=1;T._need(gain_count==index,"B gain-fit order changed")
                models[index]=model;fitpins[index]=fit
                P._exclusive(P._member(root,"gain-journal/setting-$index.json"),T._bytes(T._model_data(model;identity=true)))
                rows=T.predict_point_rows(model,panel.forecast);name="points/setting-$index.bin";hash,bytes=_point_file(P._member(root,name),rows)
                pointpins[index]=(;name,sha256=hash,bytes,model_id=model.model_id)
                _permission_recheck(permission);BInputs.recheck(a)
            end
        end
        T._need(gain_count==36,"B gain solve count incomplete")
        modelpins=Vector{NamedTuple}(undef,36)
        for physical in 1:9
            panels=ntuple(bi->P._read_verified_panel(proof,a.view,physical,bi),2)
            for eta_index in 1:2,bi in 1:2
                index=4*(physical-1)+2*(eta_index-1)+bi;panel=panels[bi];model=models[index]
                context=_export_context(a,panel,permission)
                pin=P.save_point_bundle(P._member(root,"models/setting-$index"),model,proof.a_checkpoints[physical],proof.wind_checkpoint;provenance=(;data=_provenance(a,model,panel,permission),verification=context))
                loaded=P.load_point_bundle(pin.directory;expected_manifest_sha256=pin.manifest_sha256,expected_sources=P.source_identity(),expected_binding=context)
                T._need(loaded.model.model_id==model.model_id,"B immediate bundle readback model differs")
                modelpins[index]=(;setting_index=index,directory="models/setting-$index",manifest_sha256=pin.manifest_sha256,model_id=model.model_id,fit_directory="fits/setting-$index",fit_manifest_sha256=fitpins[index].manifest_sha256,points=pointpins[index])
            end
        end
        descriptors=Union{Vector{T.BPointPredictionRow},BSelection.BPointFilePanel}[BSelection.BPointFilePanel(P._member(root,x.name),x.sha256,reference) for x in pointpins]
        selection=BSelection.provisional_l0(BSelection.BL0Panels(descriptors),reference;validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end)
        T._need(selection.data.selected_index===nothing,"B1 cannot finalize primary selection")
        P._exclusive(P._member(root,"provisional-selection.json"),T._bytes(selection.data))
        tree=_tree(root)
        manifest=(;schema="source_bound_b_initial_l0_point_stage_v1",complete=true,stage=:initial_l0,evidence_mode=a.view.evidence_mode,qualified_head=permission===nothing ? nothing : permission.qualified_head,implementation_sources=T._hashdata(T._hashmap(sources)),contract_sha256=T.CONTRACT,source_sha256=a.view.source_sha256,input_view_id=a.view.view_id,input_envelope_sha256=a.envelope_sha256,admitted_input_seal=a.admission_id,reference_completion_sha256=a.metadata.reference_completion_sha256,reference_audit_sha256=a.metadata.reference_audit_sha256,input_prefix_sha256=a.view.input_prefix_sha256,source_rows=a.view.source_hours,retained_prefix_rows=a.view.source_hours,training_boundary=a.metadata.training_boundary,validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end,embargo_hours=168,delay_hours=0,training_semantics="option_T_in_sample_two_stage_batch_v1",initial_hazard_fit_count=2,initial_gain_fit_count=gain_count,state_schedule_count=9,beta_design_count=18,final_fit_count=0,calibration_fit_count=0,exposed_development_scoring=false,joint_primary_selection=false,scientific_verdict=false,serving_promotion=false,actual_execution_permit_sha256=permission===nothing ? nothing : permission.sha256,hazards=[(;beta=laws[i].beta,model_id=laws[i].model_id,directory="laws/beta-$(laws[i].beta)",manifest_sha256=lawpins[i].manifest_sha256) for i in 1:2],models=modelpins,streams=(;basis_manifest_sha256=produced.manifest_sha256,producer_counts=produced.result.counts,replay_A_streams=9,replay_gain_solves=0,terminated_producer_tasks=produced.result.value.terminated_tasks),provisional_selection=selection.data,output_sha256=tree.sha256,output_bytes=tree.bytes,output_file_count=tree.files,output_directory_count=tree.directories)
        T._fields(manifest,P.COMPLETION_FIELDS);completion=T._bytes(manifest);P._exclusive(P._member(root,"completion.json"),completion)
        BInputs.recheck(a);_permission_recheck(permission);T._need(P.source_identity()==sources,"B initial stage exit sources differ")
        answer=(;directory=root,completion_sha256=T._sha(completion),gain_fit_count=gain_count)
    catch e;push!(errors,e)
    finally
        try BInputs.recheck(a);_permission_recheck(permission);T._need(P.source_identity()==sources,"B initial finally sources differ") catch e;push!(errors,e);end
        if !isempty(errors);try P._invalidate(root,"completion.json",completion) catch e;push!(errors,e);end;end
    end
    T._raise(errors);answer
end
function _verify_fit_input_readback(fitroot,model,panel,view,setting)
    fitmeta,fitbody=P._json(P._member(fitroot,"manifest.json");fields=P.FIT_INPUT_FIELDS)
    P._fit_input_integer_grammar(fitmeta)
    T._need(view.source_hours<=fitmeta.max_source_rows,"B fit resource declaration cannot hold admitted source")
    data_hash,data_bytes=P._file_sha(P._member(fitroot,"joint-fit.bin"));mask_hash,mask_bytes=P._file_sha(P._member(fitroot,"fit-mask.ascii"))
    b=panel.training.binding;s=setting;n=length(panel.training.rows)
    expected_fit=(;schema="b_point_fit_input_v1",complete=true,evidence_mode=view.evidence_mode,role=:in_sample_two_stage_gain_design,
        contract_sha256=T.CONTRACT,fold_id=b.fold_id,context_id=b.context_id,boundary_ms=b.boundary,cutoff_ms=b.cutoff,delay_hours=b.delay_hours,
        source_sha256=b.source_sha256,input_prefix_sha256=b.input_prefix_sha256,input_view_id=b.input_view_id,input_admission_sha256=b.input_admission_sha256,
        source_hashes=T._hashdata(b.source_hashes),hazard_model_id=b.hazard_model_id,hazard_training_prefix_sha256=b.hazard_training_prefix_sha256,
        setting_index=s.index,a_setting_index=s.a_index,beta=s.beta,taus_bits=s.taus,sigma_bits=s.sigma,eta_bits=s.eta,
        max_source_rows=Int(fitmeta.max_source_rows),max_chunk_rows=Int(fitmeta.max_chunk_rows),total_source_rows=view.source_hours,
        joint_origins=n,informative_origins=model.informative_count,fit_scalar_rows=2n,storm_scalar_rows=model.storm_scalar_count,
        data_member="joint-fit.bin",data_bytes,data_sha256=data_hash,mask_member="fit-mask.ascii",mask_bytes,fit_mask_sha256=mask_hash,fit_data_sha256=data_hash,
        basis_member_sha256=panel.basis_sha256,max_fit_issue_ms=model.max_fit_issue,max_target_complete_ms=model.max_target_complete,max_label_receipt_ms=model.max_label_receipt)
    T._need(T._bytes(expected_fit)==fitbody,"B fit header differs from independent model/source replay")
    T._need(T._sha(fitbody)==model.fit_input_manifest_sha256&&data_hash==model.fit_data_sha256&&mask_hash==model.fit_mask_sha256,"B serialized fit readback differs")
    nothing
end
function verify_initial_l0(directory,a::BInputs.BPrefixAdmission,reference::BSelection.BReferencePanel;expected_completion_sha256,expected_sources,execution_permit=nothing,chunk_hours=7)
    sources=P._source(expected_sources);BInputs.recheck(a);_reference(a,reference);permission=_permission(a,execution_permit);root=P._ordinary(directory)
    d,body=P._json(P._member(root,"completion.json");fields=P.COMPLETION_FIELDS)
    T._need(T._sha(body)==expected_completion_sha256&&d.schema=="source_bound_b_initial_l0_point_stage_v1"&&d.complete===true&&d.stage=="initial_l0"&&d.contract_sha256==T.CONTRACT&&d.input_view_id==a.view.view_id&&d.admitted_input_seal==a.admission_id&&P._sources(d.implementation_sources)==sources,"B stage completion/source differs")
    for (name,value) in ((:initial_hazard_fit_count,2),(:initial_gain_fit_count,36),(:state_schedule_count,9),(:beta_design_count,18),(:final_fit_count,0),(:calibration_fit_count,0),(:delay_hours,0),(:embargo_hours,168));T._need(T._int(getproperty(d,name),0,500000)==value,"B readback scope count differs");end
    for name in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion);T._need(getproperty(d,name)===false,"B readback exceeds stage scope");end
    expected=(;evidence_mode=a.view.evidence_mode,qualified_head=permission===nothing ? nothing : permission.qualified_head,source_sha256=a.view.source_sha256,input_envelope_sha256=a.envelope_sha256,reference_completion_sha256=a.metadata.reference_completion_sha256,reference_audit_sha256=a.metadata.reference_audit_sha256,input_prefix_sha256=a.view.input_prefix_sha256,training_boundary=a.metadata.training_boundary,validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end,training_semantics="option_T_in_sample_two_stage_batch_v1",actual_execution_permit_sha256=permission===nothing ? nothing : permission.sha256)
    for field in keys(expected);T._need(T._bytes(getproperty(d,field))==T._bytes(getproperty(expected,field)),"B completion admitted metadata differs: $field");end
    for field in (:source_rows,:retained_prefix_rows);T._need(T._int(getproperty(d,field),1,T.MAX_ROWS)==a.view.source_hours,"B completion source row count differs");end
    for field in (:output_bytes,:output_file_count,:output_directory_count);T._int(getproperty(d,field),0,16*1024^3);end
    T._fields(d.streams,(:basis_manifest_sha256,:producer_counts,:replay_A_streams,:replay_gain_solves,:terminated_producer_tasks))
    T._need(T._int(d.streams.replay_A_streams,9,9)==9&&T._int(d.streams.replay_gain_solves,0,0)==0&&T._int(d.streams.terminated_producer_tasks,9,9)==9,"B completion stream scope differs")
    for field in keys(d.streams.producer_counts);T._int(getproperty(d.streams.producer_counts,field),0,20*T.MAX_ROWS);end
    tree=_tree(root);T._need(tree.sha256==d.output_sha256&&tree.bytes==d.output_bytes&&tree.files==d.output_file_count&&tree.directories==d.output_directory_count,"B stage full output inventory differs")
    support,support_body=P._json(P._member(root,"support.json");fields=(:schema,:all36_checked,:accepted,:selected_index,:gain_fit_count,:settings))
    T._need(support.schema=="b_all36_support_preflight_v1"&&support.all36_checked===true&&support.accepted===true&&support.selected_index===nothing&&T._int(support.gain_fit_count,0,0)==0&&length(support.settings)==36,"B completed stage support scope differs")
    for (index,entry) in enumerate(support.settings)
        T._fields(entry,(:setting_index,:joint_origins,:informative_origins,:supported))
        T._need(T._int(entry.setting_index,1,36)==index&&T._int(entry.joint_origins,100,T.MAX_ROWS)>=T._int(entry.informative_origins,100,T.MAX_ROWS)&&entry.supported===true,"B completed stage support entry differs")
    end
    T._need(length(d.hazards)==2&&length(d.models)==36,"B stage law/model count differs")
    laws=ntuple(2) do bi
        entry=d.hazards[bi];T._fields(entry,(:beta,:model_id,:directory,:manifest_sha256));T._need(T._int(entry.beta,1,20)==(5,20)[bi]&&entry.directory=="laws/beta-$((5,20)[bi])","B law order differs")
        lawroot=P._member(root,String(entry.directory));P._preflight_duration_artifact(lawroot)
        law=D.load_model(lawroot;expected_manifest_sha256=String(entry.manifest_sha256))
        T._need(law.model_id==entry.model_id&&law.source_sha256==a.view.source_sha256&&law.boundary==a.metadata.training_boundary,"B loaded law source/boundary differs");law
    end
    basis_pin=String(d.streams.basis_manifest_sha256)
    proof=P._verify_basis(a.view,laws,P._member(root,"basis");expected_manifest_sha256=basis_pin,expected_sources=sources,chunk_hours)
    T._need(T._bytes(d.streams.producer_counts)==T._bytes(proof.producer_counts),"B completion producer counters differ from independent replay")
    pointpins=BSelection.BPointFilePanel[];expected_support=NamedTuple[]
    for physical in 1:9
        panels=ntuple(bi->P._read_verified_panel(proof,a.view,physical,bi),2)
        for eta_index in 1:2,bi in 1:2
            index=4*(physical-1)+2*(eta_index-1)+bi;entry=d.models[index];panel=panels[bi]
            T._fields(entry,(:setting_index,:directory,:manifest_sha256,:model_id,:fit_directory,:fit_manifest_sha256,:points));T._int(entry.setting_index,1,36)
            T._fields(entry.points,(:name,:sha256,:bytes,:model_id));T._int(entry.points.bytes,1,T.MAX_FILE)
            T._need(entry.setting_index==index&&entry.directory=="models/setting-$index"&&entry.fit_directory=="fits/setting-$index","B model/fit order differs")
            context=_export_context(a,panel,permission)
            loaded=P.load_point_bundle(P._member(root,String(entry.directory));expected_manifest_sha256=String(entry.manifest_sha256),expected_sources=sources,expected_binding=context)
            model=loaded.model;T._need(model.setting_index==index,"B loaded model differs from fixed setting slot")
            T._need(model.model_id==entry.model_id&&model.fit_input_manifest_sha256==entry.fit_manifest_sha256,"B independently loaded fit/model differs")
            T._need(entry.points.model_id==model.model_id,"B point sidecar model identity differs from verified model")
            T._need(P._bound(P._member(root,"gain-journal/setting-$index.json"))==T._bytes(T._model_data(model;identity=true)),"B gain journal differs from verified model")
            # load_point_bundle has just recomputed and checked this informative
            # count against the independent raw-replayed panel. Reuse that
            # checked scalar rather than scanning the N-row panel a second time.
            push!(expected_support,(;setting_index=index,joint_origins=length(panel.training.rows),informative_origins=model.informative_count,supported=model.informative_count>=100))
            fitroot=P._member(root,String(entry.fit_directory))
            _verify_fit_input_readback(fitroot,model,panel,a.view,T.settings()[index])
            predictions=T.predict_point_rows(model,panel.forecast);io=IOBuffer();P._header(io,"SSBPPTS1",16);for row in predictions,x in row.means;P._put_float(io,x);end;expected=take!(io)
            point=entry.points;T._need(point.name=="points/setting-$index.bin"&&P._bound(P._member(root,String(point.name)),T.MAX_FILE)==expected&&T._sha(expected)==point.sha256&&length(expected)==point.bytes,"B points differ from independent model/source replay")
            push!(pointpins,BSelection.BPointFilePanel(P._member(root,String(point.name)),String(point.sha256),reference))
        end
    end
    T._need(Set(keys(proof.certificates))==Set(1:36),"B readback requires all36 distinct fixed-setting certificates")
    expected_support_record=(;schema="b_all36_support_preflight_v1",all36_checked=true,accepted=true,selected_index=nothing,gain_fit_count=0,settings=expected_support)
    T._need(T._bytes(expected_support_record)==support_body&&P._bound(P._member(root,"support.json"))==support_body,"B all36 support differs from independently replayed panel counts")
    descriptors=Union{Vector{T.BPointPredictionRow},BSelection.BPointFilePanel}[x for x in pointpins]
    selection=BSelection.provisional_l0(BSelection.BL0Panels(descriptors),reference;validation_start=a.metadata.validation_start,validation_end=a.metadata.validation_end)
    T._need(T._bytes(selection.data)==P._bound(P._member(root,"provisional-selection.json"))&&T._bytes(selection.data)==T._bytes(d.provisional_selection),"B provisional selection replay differs")
    BInputs.recheck(a);_permission_recheck(permission);T._need(P._bound(P._member(root,"completion.json"))==body&&T._hashmap(P.source_identity())==sources,"B readback source/completion changed")
    (;complete=true,model_count=36,hazard_fit_calls=0,gain_fit_calls=0,raw_replay_A_streams=9,certificate_count=length(proof.certificates),selected_index=nothing)
end
end
