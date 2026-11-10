"""Initial32 matched-reference fitting and data-only publication; no work on import."""
module BMatchedInitialProducer
using Dates, SHA, JSON3
const Parent=parentmodule(@__MODULE__)
const BI=Parent.BI
const T=Parent.T
const P=Parent.P
const M=Parent.M
const FC=Parent.BMatched342Core
const Audit=Parent.MatchedReferenceAudit342
const Selection=Parent.ActualBChain.BSelection
const CELLS=((0,4),(0,6),(1,4),(1,6))
const MODEL_BYTES=4*1024^2
const STORE_BYTES=128*1024^2
const RUNTIME_FIELDS=(:julia_version,:julia_binary_sha256,:evotrees_version,:evotrees_source_sha256,:threads,:blas_threads,:omp_threads)
const COMPONENT_FIELDS=(:schema,:complete,:evidence_mode,:initial_only,:pair_id,:source_bindings,:training_boundary,:validation_start,:validation_end,:feature_contract_sha256,:feature_schema_sha256,:tuning_loss_contract_sha256,:evotrees_source_sha256,:baseline_qualified_head,:baseline_qualification_sha256,:baseline_sources,:runtime,:execution_authority_sha256,:cells,:drafts,:members,:component_id)
const CELL_FIELDS=(:delay_hours,:horizon,:context_id,:matched_input_id,:training_count,:validation_count,:union_count,:validation_persistence_count,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256,:max_fit_target_complete,:max_fit_label_receipt,:max_validation_target_complete,:max_validation_label_receipt,:encoder_member,:models)
const MODEL_FIELDS=(:setting_index,:model_id,:model_member,:fit_receipt_member,:candidate_member,:native_increment_sha256,:reloaded_increment_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const FIT_FIELDS=(:schema,:complete,:evidence_mode,:pair_id,:context_id,:delay_hours,:horizon,:setting_index,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256,:runtime_sha256,:recipe,:model_id,:model_sha256,:encoder_sha256,:native_increment_sha256,:reloaded_increment_sha256,:union_count,:fit_elapsed_ns,:execution_authority_sha256,:receipt_id)
const DRAFT_FIELDS=(:delay_hours,:root,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:routing_member,:model_count,:persistence_count,:model_ids,:selected_setting_ids,:members)
const SOURCE_BINDING_FIELDS=(:delay_hours,:source_proof_sha256,:source_admission_id,:input_view_id,:input_prefix_sha256,:raw_prefix_sha256,:source_sha256,:matched_input_id,:context_id)
const _HEAD=r"\A[0-9a-f]{40}\z"
export InitialMatchedDraft342,produce_initial_generated,produce_initial_actual,read_draft,recheck_draft,draft_metadata,publish_initial

_need(ok,message)=T._need(ok,message)
_same(a,b)=T._bytes(a)==T._bytes(b)
_cellname(d,h)="L$d-h$h"
_modelname(d,h,s)="$(_cellname(d,h))/setting-$s.model.json"
_fitname(d,h,s)="$(_cellname(d,h))/setting-$s.fit.json"
_encodername(d,h)="$(_cellname(d,h))/encoder.json"
_candidate(h,s)="candidates/h$h/setting-$s.bin"
_routename(d)="routing/L$d.bin"
function _sources(expected)
    observed=T._hashmap(Parent.source_identity());chosen=T._hashmap(expected)
    _need(observed==chosen,"matched342 producer component closure differs");Tuple(chosen)
end
_source_rows(sources)=Tuple((;name=first(x),sha256=last(x)) for x in sources)
function _path(path;directory=false)
    _need(path isa AbstractString&&isabspath(path)&&all(x->x!="."&&x!="..",split(replace(path,'\\'=>'/'),'/')),"absolute nonalias matched342 path required")
    full=P._ordinary(String(path));_need(directory ? isdir(full) : isfile(full),"ordinary matched342 path kind differs")
    _need(normpath(full)==realpath(full),"matched342 path case/alias differs");full
end
function _ref(ref;limit=T.MAX_META,contract=false)
    T._fields(ref,contract ? (:path,:sha256,:member) : (:path,:sha256))
    _need(T._hex(ref.sha256),"matched342 external file pin grammar differs")
    path=_path(ref.path);body=P._bound(path,limit)
    _need(T._sha(body)==ref.sha256,"matched342 external file pin changed");(;path,sha256=String(ref.sha256),body)
end
function _identified(data,field)
    merge(data,NamedTuple{(field,)}((T._id(data),)))
end
function _identity(data,field)
    names=Tuple(Symbol(k) for k in keys(data) if Symbol(k)!=field)
    _need(T._hex(getproperty(data,field))&&T._id(NamedTuple{names}(Tuple(getproperty(data,k) for k in names)))==getproperty(data,field),"matched342 canonical identity differs")
end
function _member(root,name;limit=T.MAX_FILE)
    _need(name isa AbstractString&&ncodeunits(name)<=256,"matched342 member name bound")
    path=P._member(root,name);_path(path);hash,bytes=P._file_sha(path)
    _need(0<bytes<=limit,"matched342 member byte bound");(;name=String(name),bytes,sha256=hash)
end
function _member_fields(row)
    T._fields(row,(:name,:bytes,:sha256));_need(row.name isa AbstractString&&T._hex(row.sha256),"matched342 member fields differ")
    T._int(row.bytes,1,T.MAX_FILE);(;name=String(row.name),bytes=Int(row.bytes),sha256=String(row.sha256))
end
function _component_names()
    names=String["qualification.json","feature-contract","loss-contract",_routename(0),_routename(1)]
    for (d,h) in CELLS
        push!(names,_encodername(d,h))
        for s in 1:8;push!(names,_modelname(d,h,s),_fitname(d,h,s));end
    end
    sort!(names)
end
function _draft_names()
    sort!(vcat(String["grid.json","tuning.json","predictions.bin","qualification.json"],[_candidate(h,s) for h in (4,6) for s in 1:8]))
end
function _inventory(root,files,dirs;later=false)
    observed=String[];directories=String[]
    for (parent,children,leaves) in walkdir(root;follow_symlinks=false)
        _path(parent;directory=true)
        _need(length(directories)+length(children)<=16&&length(observed)+length(leaves)<=96,"matched342 directory inventory bound")
        for name in children
            path=joinpath(parent,name);_path(path;directory=true);push!(directories,replace(relpath(path,root),'\\'=>'/'))
        end
        for name in leaves
            path=joinpath(parent,name);_path(path);push!(observed,replace(relpath(path,root),'\\'=>'/'))
        end
    end
    allowed=later ? union(Set(files),Set(("audit.json","completion.json"))) : Set(files)
    _need(Set(directories)==Set(dirs)&&issubset(Set(observed),allowed)&&issubset(Set(files),Set(observed)),"matched342 exact file/directory inventory differs")
    !later&&_need(Set(observed)==Set(files),"matched342 unexpected member")
    true
end
function _component_members(root;completion=false)
    names=_component_names();files=completion ? vcat(names,["completion.json"]) : names
    _inventory(root,files,vcat([_cellname(d,h) for (d,h) in CELLS],["routing"]))
    rows=NamedTuple[];total=completion ? filesize(joinpath(root,"completion.json")) : 0
    for name in names
        limit=endswith(name,".model.json")||endswith(name,"/encoder.json") ? MODEL_BYTES : T.MAX_META
        row=_member(root,name;limit);total+=row.bytes;_need(total<=STORE_BYTES,"matched342 component total exceeds128MiB");push!(rows,row)
    end
    Tuple(rows)
end
function _draft_members(root;later=false)
    names=_draft_names();_inventory(root,names,("candidates","candidates/h4","candidates/h6");later)
    rows=Tuple(_member(root,name;limit=endswith(name,".json") ? T.MAX_META : T.MAX_FILE) for name in names)
    total=sum(x.bytes for x in rows)
    for name in ("audit.json","completion.json");isfile(joinpath(root,name))&&(total+=filesize(joinpath(root,name)));end
    _need(total<=STORE_BYTES,"matched342 B22 byte bound exceeded");rows
end
function _ledger(rows,expected)
    _need(rows isa AbstractVector&&length(rows)==length(expected),"matched342 exact ledger count differs")
    for (row,value) in zip(rows,expected);_need(_same(_member_fields(row),value),"matched342 member ledger changed");end
    true
end
function _runtime(runtime)
    T._fields(runtime,RUNTIME_FIELDS)
    _need(runtime.julia_version isa AbstractString&&!isempty(runtime.julia_version)&&runtime.evotrees_version=="0.18.7"&&T._hex(runtime.julia_binary_sha256)&&runtime.evotrees_source_sha256==M.EVOTREES_SOURCE_SHA256,"matched342 runtime/library identity differs")
    for k in (:threads,:blas_threads,:omp_threads);T._int(getproperty(runtime,k),1,1);end
    NamedTuple{RUNTIME_FIELDS}(Tuple(k in (:threads,:blas_threads,:omp_threads) ? Int(getproperty(runtime,k)) : String(getproperty(runtime,k)) for k in RUNTIME_FIELDS))
end
_runtime_sha(runtime)=T._id((;schema="matched342_native_runtime_v1",runtime))
function _qualification(body,mode,head,sources)
    _need(head isa AbstractString&&occursin(_HEAD,head),"matched342 qualified head grammar differs")
    P._software_qualification(body;schema="b_reference_component_software_qualification_v1",evidence_mode=mode=="synthetic_only" ? "synthetic_only" : "software_qualified",head=String(head),sources)
end
function _controls(pair,qualification,feature,loss,head,sources)
    meta=FC.source_pair_metadata(pair);q=_ref(qualification);f=_ref(feature;contract=true);l=_ref(loss;contract=true)
    _need(feature.member==Parent.FEATURE_CONTRACT_MEMBER&&feature.sha256==Parent.FEATURE_CONTRACT_SHA256&&loss.member==Parent.LOSS_CONTRACT_MEMBER&&loss.sha256==Parent.LOSS_CONTRACT_SHA256,"matched342 registered feature/loss contracts differ")
    mapping=Dict(sources)
    _need(get(mapping,String(feature.member),nothing)==f.sha256&&get(mapping,String(loss.member),nothing)==l.sha256&&meta.tuning_loss_contract_sha256==l.sha256,"matched342 contract absent from qualified closure")
    _qualification(q.body,String(meta.evidence_mode),head,sources)
    (;qualification=q,feature=f,loss=l,head=String(head))
end
function _overlap(a,b)
    clean(p)=begin s=replace(normpath(p),'\\'=>'/');Sys.iswindows() ? lowercase(s) : s end
    x=clean(a);y=clean(b);x==y||startswith(x,y*"/")||startswith(y,x*"/")
end
function _fresh_stage(output,pair,controls)
    _need(output isa AbstractString&&isabspath(output)&&all(x->x!="."&&x!="..",split(replace(output,'\\'=>'/'),'/')),"fresh absolute matched342 output required")
    full=P._ordinary(String(output));_need(!ispath(full)&&!islink(full),"matched342 output already exists")
    _need(!Sys.iswindows()||(!endswith(full,".")&&!endswith(full," ")),"matched342 output has a Windows path alias")
    _path(dirname(full);directory=true);_need(!_overlap(full,Parent.PACKAGE_ROOT),"matched342 output overlaps source checkout")
    for source in FC.source_admissions(pair),root in values(source.roots);_need(!_overlap(full,root),"matched342 output overlaps admitted source");end
    for ref in (controls.qualification,controls.feature,controls.loss);_need(!_overlap(full,ref.path),"matched342 output overlaps input control");end
    root=P._root_new(full);component=P._root_new(joinpath(root,"component"));P._root_new(joinpath(component,"routing"))
    for (d,h) in CELLS;P._root_new(joinpath(component,_cellname(d,h)));end
    references=ntuple(i->begin
        dir=P._root_new(joinpath(root,"reference-L$(i-1)"));P._root_new(joinpath(dir,"candidates"))
        for h in (4,6);P._root_new(joinpath(dir,"candidates","h$h"));end;dir
    end,2)
    (;root,component,references)
end
function _geometry(pair,d)
    source=FC.source_admissions(pair)[d+1];m=source.metadata;v=source.view
    geometry=BI._reference_geometry(v,m.training_boundary,m.validation_start,m.validation_end,d)
    (;source,view=v,indices=geometry.indices,training=geometry.training,leads=geometry.leads,
      boundary=m.training_boundary,start=m.validation_start,stop=m.validation_end)
end
function _routing(g)
    UInt8[all(isfinite,(T._clean(g.view,8,i),T._clean(g.view,6,i),T._clean(g.view,12,i)))&&T._clean(g.view,8,i)>=0 ? 0 : 1 for i in g.indices]
end
function _write_routing(path,route)
    io=IOBuffer();write(io,"SSBMRTE1");write(io,htol(UInt64(1)));write(io,route);P._exclusive(path,take!(io))
end
function _read_routing(path,n;expected_sha256)
    _need(filesize(path)==16+n,"matched342 routing size differs")
    body=P._bound(path,16+n);_need(T._hex(expected_sha256)&&T._sha(body)==expected_sha256,"matched342 routing content pin differs")
    io=IOBuffer(body)
    try
        _need(read(io,8)==collect(codeunits("SSBMRTE1"))&&ltoh(read(io,UInt64))==1,"matched342 routing header/version differs")
        route=read(io,n);_need(length(route)==n&&eof(io)&&all(x->x==0||x==1,route),"matched342 routing byte differs");route
    finally close(io) end
end
function _column(path,n;selected=false,expected_sha256)
    _need(filesize(path)==16+(selected ? 16 : 8)*n,"matched342 prediction member geometry differs")
    body=P._bound(path,16+(selected ? 16 : 8)*n)
    _need(T._hex(expected_sha256)&&T._sha(body)==expected_sha256,"matched342 prediction content pin differs")
    io=IOBuffer(body)
    try
        P._check_header(io,selected ? "SSBPPTS1" : "SSBRFPR1",selected ? 16 : 8)
        values=if selected
            matrix=Matrix{Float64}(undef,2,n)
            for j in 1:n,k in 1:2;matrix[k,j]=P._get_float(io);end
            matrix
        else
            [P._get_float(io) for _ in 1:n]
        end
        _need(eof(io),"matched342 prediction extra bytes");values
    finally close(io) end
end
function _write_column(path,values;selected=false)
    io=IOBuffer();P._header(io,selected ? "SSBPPTS1" : "SSBRFPR1",selected ? 16 : 8)
    if selected
        for i in eachindex(values[1]),k in 1:2;P._put_float(io,values[k][i]);end
    else
        for value in values;P._put_float(io,value);end
    end
    P._exclusive(path,take!(io))
end
_bits_equal(a,b)=length(a)==length(b)&&all(reinterpret(UInt64,a[i])==reinterpret(UInt64,b[i]) for i in eachindex(a,b))
function _increment_sha(values)
    _need(all(isfinite,values),"matched342 increment overflow");T._sha(reinterpret(UInt8,htol.(reinterpret(UInt64,Float64.(values)))))
end
function _raw_row_id(g,d)
    T._id([(;issue=T.Adapter._shift(g.view.input.time[i],1),anchor=T._clean(g.view,15,i-d),target=ntuple(k->begin x=T._clean(g.view,15,i+(4,6)[k]);isfinite(x) ? x : nothing end,2),source_sha256=g.view.source_sha256,delay_hours=d) for i in g.indices])
end
function _common(meta,d,qsha,head,sources)
    s=meta.source_bindings[d+1]
    (;complete=true,evidence_mode=String(meta.evidence_mode),reference_kind="matched_information_gbm_initial_v1",source_sha256=s.source_sha256,input_prefix_sha256=s.input_prefix_sha256,input_view_id=s.input_view_id,delay_hours=d,training_boundary=meta.training_boundary,validation_start=meta.validation_start,validation_end=meta.validation_end,training_history_start=meta.training_history_start,feature_contract_sha256=Parent.FEATURE_CONTRACT_SHA256,tuning_loss_contract_sha256=Parent.LOSS_CONTRACT_SHA256,baseline_qualification_sha256=qsha,baseline_sources=_source_rows(sources))
end
function _common_check(data,common)
    for k in BI.REFERENCE_COMMON;_need(_same(getproperty(data,k),getproperty(common,k)),"matched342 repeated B22 fact differs: $k");end
end
function _model_mask(binding,g,d,h,indices)
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("matched342-training-mask-v1\n$(binding.matched_input_id)\n"))
    SHA.update!(ctx,reinterpret(UInt8,htol.(Int64[d,h,Dates.value(g.boundary),Dates.value(g.start),Dates.value(g.stop),length(indices)])))
    record=Vector{Int64}(undef,8)
    for i in indices
        issue=T.Adapter._shift(g.view.input.time[i],1);anchor=T.Adapter._shift(issue,-1-d);target=T.Adapter._shift(issue,h-1)
        record.=(i,Dates.value(issue),Dates.value(anchor),Dates.value(T.Adapter._shift(anchor,1)),Dates.value(T.Adapter._shift(anchor,1+d)),Dates.value(target),Dates.value(T.Adapter._shift(target,1)),Dates.value(T.Adapter._shift(target,1+d)))
        record.=htol.(record);SHA.update!(ctx,reinterpret(UInt8,record))
    end
    bytes2hex(SHA.digest!(ctx))
end
function _clockmax(g,d,h,indices)
    T.Adapter.forecast_clock(T.Adapter._shift(g.view.input.time[last(indices)],1),h,d)
end
function _header_fields(cell,binding,d,h,s)
    grid=M.GRID342[s]
    Dict{String,Any}("schema"=>"matched_gbm_scalar342_model_v1","feature_count"=>342,"feature_schema_sha256"=>M.FEATURE_SCHEMA_SHA256,"evotrees_version"=>"0.18.7","evotrees_source_sha256"=>M.EVOTREES_SOURCE_SHA256,"cell_delay"=>d,"cell_horizon"=>h,"setting_index"=>s,"max_depth"=>grid.max_depth,"nrounds"=>grid.nrounds,"nbins"=>grid.nbins,"seed"=>22022026,"training_mask_sha256"=>String(cell.model_training_mask_sha256),"preprocessing_sha256"=>String(cell.preprocessing_sha256),"input_admission_sha256"=>String(binding.context_id),"feature_contract_sha256"=>Parent.FEATURE_CONTRACT_SHA256,"loss_contract_sha256"=>Parent.LOSS_CONTRACT_SHA256)
end
function _metric_check(record,metric)
    _need(T._int(record.n,1,T.MAX_ROWS)==metric.n,"matched342 per-lead metric count differs")
    for key in (:rmse_nt,:mae_nt,:bias_nt)
        stored=T._float(getproperty(record,key));value=getproperty(metric,key)
        _need(abs(stored-value)<=5e-12*max(1.,abs(value)),"matched342 maintained metric parity differs")
    end
end
function _fit_receipt(path,c,cell,model,d,h,s,modelsha,encodersha;expected_sha256)
    fit,body=P._json(path;fields=FIT_FIELDS);_need(T._sha(body)==expected_sha256,"matched342 fit receipt content pin differs");_identity(fit,:receipt_id)
    _need(fit.schema=="matched342_initial_fit_receipt_v1"&&fit.complete===true&&fit.evidence_mode==c.evidence_mode&&fit.pair_id==c.pair_id&&fit.context_id==cell.context_id,"matched342 fit receipt context differs")
    _need(T._int(fit.delay_hours,0,1)==d&&T._int(fit.horizon,4,6)==h&&T._int(fit.setting_index,1,8)==s,"matched342 fit receipt cell differs")
    for k in (:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256)
        _need(T._hex(getproperty(fit,k))&&getproperty(fit,k)==getproperty(cell,k),"matched342 fit receipt data pin differs")
    end
    _need(fit.runtime_sha256==_runtime_sha(c.runtime)&&_same(fit.recipe,M._recipe(s))&&fit.model_id==model.model_id&&fit.model_sha256==modelsha&&fit.encoder_sha256==encodersha,"matched342 fit receipt recipe/runtime/model differs")
    _need(fit.native_increment_sha256==fit.reloaded_increment_sha256==model.native_increment_sha256==model.reloaded_increment_sha256&&T._hex(fit.native_increment_sha256),"matched342 retained native/saved increment evidence differs")
    _need(T._int(fit.union_count,1,T.MAX_ROWS)==cell.union_count&&_same(fit.execution_authority_sha256,c.execution_authority_sha256),"matched342 fit receipt union/authority differs")
    T._int(fit.fit_elapsed_ns,0,typemax(Int64));fit
end
function _ref_matches(row,expected,name)
    value=_member_fields(row);_need(value.name==name&&_same(value,expected),"matched342 linked member differs");value
end
function _read_info(pair,path,expected_sha,expected_sources)
    sources=_sources(expected_sources);FC.recheck_pair(pair);meta=FC.source_pair_metadata(pair)
    full=_path(path);component=dirname(full);stage=dirname(component)
    _need(basename(full)=="completion.json"&&basename(component)=="component"&&T._hex(expected_sha),"matched342 fixed component completion path differs")
    c,body=P._json(full;fields=COMPONENT_FIELDS);_need(T._sha(body)==expected_sha,"matched342 component outside SHA differs");_identity(c,:component_id)
    _need(c.schema=="matched342_initial_component_completion_v1"&&c.complete===true&&c.initial_only===true&&c.evidence_mode==String(meta.evidence_mode)&&c.pair_id==meta.pair_id,"matched342 component role/context differs")
    _need(_same(c.source_bindings,meta.source_bindings)&&_same(c.baseline_sources,_source_rows(sources)),"matched342 component source bindings differ")
    for (k,value) in ((:training_boundary,meta.training_boundary),(:validation_start,meta.validation_start),(:validation_end,meta.validation_end));_need(T._clock(getproperty(c,k))==value,"matched342 component split differs");end
    _need(c.feature_contract_sha256==Parent.FEATURE_CONTRACT_SHA256&&c.feature_schema_sha256==M.FEATURE_SCHEMA_SHA256&&c.tuning_loss_contract_sha256==Parent.LOSS_CONTRACT_SHA256&&c.evotrees_source_sha256==M.EVOTREES_SOURCE_SHA256,"matched342 component contracts/library differ")
    runtime=_runtime(c.runtime);_need(c.baseline_qualified_head isa AbstractString&&occursin(_HEAD,c.baseline_qualified_head)&&T._hex(c.baseline_qualification_sha256),"matched342 component qualification grammar differs")
    _need(c.evidence_mode=="synthetic_only" ? c.execution_authority_sha256===nothing : T._hex(c.execution_authority_sha256),"matched342 generated/actual authority domain differs")
    members=_component_members(component;completion=true);_ledger(c.members,members);membermap=Dict(x.name=>x for x in members)
    _need(membermap["qualification.json"].sha256==c.baseline_qualification_sha256&&membermap["feature-contract"].sha256==Parent.FEATURE_CONTRACT_SHA256&&membermap["loss-contract"].sha256==Parent.LOSS_CONTRACT_SHA256,"matched342 component contract copy differs")
    qualification_body=P._bound(joinpath(component,"qualification.json"))
    _need(T._sha(qualification_body)==c.baseline_qualification_sha256,"matched342 qualification content pin differs")
    _qualification(qualification_body,c.evidence_mode,c.baseline_qualified_head,sources)
    _need(c.cells isa AbstractVector&&length(c.cells)==4&&c.drafts isa AbstractVector&&length(c.drafts)==2,"matched342 four cells/two drafts required")
    geometries=ntuple(i->_geometry(pair,i-1),2);routes=ntuple(i->_routing(geometries[i]),2)
    for (ci,(d,h)) in enumerate(CELLS)
        cell=c.cells[ci];T._fields(cell,CELL_FIELDS);g=geometries[d+1];k=h==4 ? 1 : 2;binding=meta.source_bindings[d+1]
        _need(T._int(cell.delay_hours,0,1)==d&&T._int(cell.horizon,4,6)==h&&cell.context_id==binding.context_id&&cell.matched_input_id==binding.matched_input_id,"matched342 cell order/context differs")
        _need(T._int(cell.training_count,1,T.MAX_ROWS)==length(g.training[k])&&T._int(cell.validation_count,1,T.MAX_ROWS)==length(g.leads[k])&&T._int(cell.union_count,1,T.MAX_ROWS)==length(g.indices),"matched342 full per-lead cohort differs")
        positions=Dict(i=>j for (j,i) in enumerate(g.indices));persistence=count(i->routes[d+1][positions[i]]==1,g.leads[k])
        _need(T._int(cell.validation_persistence_count,0,T.MAX_ROWS)==persistence,"matched342 per-lead persistence count differs")
        _need(cell.model_training_mask_sha256==_model_mask(binding,g,d,h,g.training[k])&&cell.reference_training_mask_sha256==BI._reference_mask(g.view,g.training[k],h,:training,g.boundary,g.start,g.stop,d)&&cell.reference_validation_cohort_sha256==BI._reference_mask(g.view,g.leads[k],h,:validation,g.boundary,g.start,g.stop,d),"matched342 separate mask domains differ")
        for key in (:fit_input_sha256,:preprocessing_sha256,:training_design_sha256);_need(T._hex(getproperty(cell,key)),"matched342 cell design pin grammar differs");end
        fitclock=_clockmax(g,d,h,g.training[k]);valclock=_clockmax(g,d,h,g.leads[k])
        for (key,value) in ((:max_fit_target_complete,fitclock.target_complete),(:max_fit_label_receipt,fitclock.target_receipt),(:max_validation_target_complete,valclock.target_complete),(:max_validation_label_receipt,valclock.target_receipt));_need(T._clock(getproperty(cell,key))==value,"matched342 per-lead maximum clock differs");end
        enc=_ref_matches(cell.encoder_member,membermap[_encodername(d,h)],_encodername(d,h))
        encoder=M.load_encoder(joinpath(component,enc.name);expected_sha256=enc.sha256,expected_training_design_sha256=cell.training_design_sha256)
        _need(collect(encoder.training_rows)==g.training[k]&&M.preprocessing_sha256(encoder)==cell.preprocessing_sha256==enc.sha256,"matched342 saved encoder training/identity differs")
        _need(cell.models isa AbstractVector&&length(cell.models)==8,"matched342 all eight model states required")
        for s in 1:8
            model=cell.models[s];T._fields(model,MODEL_FIELDS);_need(T._int(model.setting_index,1,8)==s&&T._hex(model.model_id),"matched342 model order/ID differs")
            mr=_ref_matches(model.model_member,membermap[_modelname(d,h,s)],_modelname(d,h,s));fr=_ref_matches(model.fit_receipt_member,membermap[_fitname(d,h,s)],_fitname(d,h,s))
            object=M._read_json(joinpath(component,mr.name),mr.sha256)
            M._decode_model_data(object,_header_fields(cell,binding,d,h,s),M.GRID342[s])
            _need(object["model_id"]==model.model_id,"matched342 saved model ID differs")
            _fit_receipt(joinpath(component,fr.name),c,cell,model,d,h,s,mr.sha256,enc.sha256;expected_sha256=fr.sha256)
        end
    end
    roots=ntuple(i->_path(joinpath(stage,"reference-L$(i-1)");directory=true),2)
    for d in 0:1
        draft=c.drafts[d+1];T._fields(draft,DRAFT_FIELDS);g=geometries[d+1];route=routes[d+1];n=length(g.indices);root=roots[d+1]
        _need(T._int(draft.delay_hours,0,1)==d&&draft.root=="reference-L$d","matched342 draft delay/root differs")
        members20=_draft_members(root;later=true);_ledger(draft.members,members20);map20=Dict(x.name=>x for x in members20)
        _ref_matches(draft.routing_member,membermap[_routename(d)],_routename(d))
        _need(_read_routing(joinpath(component,_routename(d)),n;expected_sha256=membermap[_routename(d)].sha256)==route&&T._int(draft.persistence_count,0,T.MAX_ROWS)==count(==(UInt8(1)),route)&&T._int(draft.model_count,0,T.MAX_ROWS)==count(==(UInt8(0)),route),"matched342 routing bytes/counts differ")
        _need(draft.raw_row_identity_sha256==_raw_row_id(g,d)&&draft.prediction_panel_sha256==map20["predictions.bin"].sha256&&draft.grid_sha256==map20["grid.json"].sha256&&draft.tuning_sha256==map20["tuning.json"].sha256&&map20["qualification.json"].sha256==c.baseline_qualification_sha256,"matched342 draft identity links differ")
        common=_common(meta,d,c.baseline_qualification_sha256,c.baseline_qualified_head,sources)
        grid,grid_body=P._json(joinpath(root,"grid.json");fields=(:schema,:complete,:evidence_mode,:feature_contract_sha256,:settings,:grid_id));_need(T._sha(grid_body)==draft.grid_sha256,"matched342 grid content pin differs");_identity(grid,:grid_id)
        _need(grid.schema=="fixed_eight_gbm_settings_v2"&&grid.complete===true&&grid.evidence_mode==c.evidence_mode&&grid.feature_contract_sha256==Parent.FEATURE_CONTRACT_SHA256&&length(grid.settings)==8,"matched342 fixed grid header differs")
        for s in 1:8;item=grid.settings[s];T._fields(item,(:index,:max_depth,:nrounds,:nbins));_need(T._int(item.index,1,8)==s&&all(T._int(getproperty(item,k),1,10000)==getproperty(M.GRID342[s],k) for k in keys(M.GRID342[s])),"matched342 exact grid recipe differs");end
        tuning,tuning_body=P._json(joinpath(root,"tuning.json");fields=BI.REFERENCE_TUNING_FIELDS);_need(T._sha(tuning_body)==draft.tuning_sha256,"matched342 tuning content pin differs");_identity(tuning,:tuning_id);_common_check(tuning,common)
        _need(tuning.schema=="b_matched_reference_tuning_v2"&&tuning.grid_sha256==draft.grid_sha256&&collect(tuning.lead_order)==[4,6]&&all(x->x isa Integer&&!(x isa Bool),tuning.lead_order)&&length(tuning.leads)==2,"matched342 tuning header differs")
        _need(length(draft.model_ids)==length(draft.selected_setting_ids)==2&&all(T._hex,draft.model_ids),"matched342 selected model shape differs")
        selected=_column(joinpath(root,"predictions.bin"),n;selected=true,expected_sha256=draft.prediction_panel_sha256)
        for k in 1:2
            h=(4,6)[k];cell=c.cells[2d+k];lead=tuning.leads[k];T._fields(lead,BI.REFERENCE_LEAD_FIELDS)
            _need(T._int(lead.horizon,4,6)==h&&lead.training_mask_sha256==cell.reference_training_mask_sha256&&lead.validation_cohort_sha256==cell.reference_validation_cohort_sha256&&T._int(lead.validation_count,1,T.MAX_ROWS)==cell.validation_count,"matched342 tuning masks/counts differ")
            for (key,sourcekey) in ((:max_fit_target_complete,:max_fit_target_complete),(:max_fit_label_receipt,:max_fit_label_receipt),(:max_selection_target_complete,:max_validation_target_complete),(:max_selection_label_receipt,:max_validation_label_receipt));_need(getproperty(lead,key)==getproperty(cell,sourcekey),"matched342 tuning clock differs");end
            choice=T._int(lead.selected_index,1,8);_need(T._int(draft.selected_setting_ids[k],1,8)==choice&&lead.selected_model_id==draft.model_ids[k]==cell.models[choice].model_id&&length(lead.candidates)==8,"matched342 selected choice/model differs")
            positions=Dict(i=>j for (j,i) in enumerate(g.indices));leadpos=[positions[i] for i in g.leads[k]];labels=[T._clean(g.view,15,i+h) for i in g.leads[k]]
            best=0;stored_best=0;best_rmse=Inf;stored_rmse=Inf
            for s in 1:8
                item=lead.candidates[s];T._fields(item,BI.REFERENCE_CANDIDATE_FIELDS);model=cell.models[s];record=_member_fields(model.candidate_member)
                _need(record.name==_candidate(h,s)&&_same(record,map20[record.name])&&item.prediction_member==record.name&&item.prediction_sha256==record.sha256&&T._int(item.prediction_bytes,1,T.MAX_FILE)==record.bytes==16+8n,"matched342 candidate member links differ")
                _need(T._int(item.index,1,8)==s&&all(T._int(getproperty(item,key),1,10000)==getproperty(M.GRID342[s],key) for key in keys(M.GRID342[s]))&&item.model_id==model.model_id&&item.fit_input_sha256==cell.fit_input_sha256,"matched342 tuning recipe/model/data differs")
                values=_column(joinpath(root,record.name),n;expected_sha256=record.sha256);metric=BI._reference_metric(values[leadpos],labels);_metric_check(item,metric);_metric_check(model,metric)
                _need(all(route[j]==0||reinterpret(UInt64,values[j])==reinterpret(UInt64,T._clean(g.view,15,g.indices[j]-d)) for j in eachindex(route)),"matched342 persistence value differs")
                stored=T._float(item.rmse_nt);stored<stored_rmse&&(stored_best=s;stored_rmse=stored)
                metric.rmse_nt<best_rmse&&(best=s;best_rmse=metric.rmse_nt)
                s==choice&&_need(_bits_equal(values,vec(selected[k,:])),"matched342 selected column bit identity differs")
            end
            _need(best==stored_best==choice,"matched342 strict first minimum differs")
        end
    end
    FC.recheck_pair(pair);_sources(sources)
    _need(_same(_component_members(component;completion=true),members),"matched342 component members changed during read")
    for d in 0:1;_ledger(c.drafts[d+1].members,_draft_members(roots[d+1];later=true));end
    _need(P._bound(full)==body,"matched342 component changed during read")
    (;pair,metadata=meta,data=c,path=full,sha256=String(expected_sha),component,roots,sources,runtime,members)
end

struct _DraftToken end
struct InitialMatchedDraft342{C}
    pair::C
    completion_path::String
    completion_sha256::String
    expected_sources::Tuple
    metadata::NamedTuple
    seal::String
    function InitialMatchedDraft342(pair,path,sha,sources,metadata,seal,::_DraftToken)
        new{typeof(pair)}(pair,path,sha,sources,metadata,seal)
    end
end
_draft_seal(pair,path,sha,sources,metadata)=T._id((;schema="checked_matched342_initial_draft_v1",pair_id=FC.source_pair_metadata(pair).pair_id,completion_path=path,completion_sha256=sha,baseline_sources=_source_rows(sources),metadata))
function _draft_projection(info)
    (;evidence_mode=String(info.data.evidence_mode),pair_id=String(info.data.pair_id),component_root=info.component,component_completion_path=info.path,component_completion_sha256=info.sha256,reference_roots=info.roots,qualified_head=String(info.data.baseline_qualified_head),baseline_sources=_source_rows(info.sources))
end
function read_draft(pair,completion_path;expected_component_sha256,expected_sources)
    info=_read_info(pair,completion_path,expected_component_sha256,expected_sources);metadata=_draft_projection(info)
    InitialMatchedDraft342(pair,info.path,info.sha256,info.sources,metadata,_draft_seal(pair,info.path,info.sha256,info.sources,metadata),_DraftToken())
end
function recheck_draft(draft::InitialMatchedDraft342)
    _need(draft.seal==_draft_seal(draft.pair,draft.completion_path,draft.completion_sha256,draft.expected_sources,draft.metadata),"matched342 draft handle changed")
    info=_read_info(draft.pair,draft.completion_path,draft.completion_sha256,draft.expected_sources)
    _need(_same(_draft_projection(info),draft.metadata),"matched342 draft projection changed");true
end
function draft_metadata(draft::InitialMatchedDraft342)
    recheck_draft(draft);draft.metadata
end

function _run(pair,output;qualification,feature_contract,loss_contract,qualified_head,execution_authority,expected_sources)
    sources=_sources(expected_sources);meta=FC.source_pair_metadata(pair);mode=String(meta.evidence_mode)
    controls=_controls(pair,qualification,feature_contract,loss_contract,qualified_head,sources)
    runtime_info=FC.runtime_identity();runtime=_runtime(runtime_info.runtime);runtime_sha=_runtime_sha(runtime)
    _need(runtime_info.runtime_sha256==runtime_sha,"matched342 runtime digest domain differs")
    authority_sha=nothing
    if mode=="archival_assumed_completion_research"
        FC.recheck_execution_authority(execution_authority,pair);authority=FC.execution_authority_metadata(execution_authority)
        _need(authority.qualified_head==controls.head&&authority.component_qualification_sha256==controls.qualification.sha256&&authority.runtime_sha256==runtime_sha&&_same(authority.runtime,runtime)&&_same(authority.implementation_sources,_source_rows(sources))&&_same(authority.source_bindings,meta.source_bindings),"matched342 actual component/permit facts differ")
        _need(normpath(String(output))==normpath(authority.execution_directory)&&authority.contracts.feature_contract_sha256==controls.feature.sha256&&authority.contracts.loss_contract_sha256==controls.loss.sha256,"matched342 actual output/contracts differ")
        authority_sha=String(authority.permit_sha256)
    else
        _need(mode=="synthetic_only"&&execution_authority===nothing,"generated producer cannot accept actual authority")
    end
    stage=_fresh_stage(output,pair,controls)
    for (name,ref) in (("qualification.json",controls.qualification),("feature-contract",controls.feature),("loss-contract",controls.loss));P._exclusive(joinpath(stage.component,name),ref.body);end
    for root in stage.references;P._exclusive(joinpath(root,"qualification.json"),controls.qualification.body);end
    geometries=ntuple(i->_geometry(pair,i-1),2);routes=ntuple(i->_routing(geometries[i]),2)
    for d in 0:1;_write_routing(joinpath(stage.component,_routename(d)),routes[d+1]);end
    cell_records=NamedTuple[]
    for (d,h) in CELLS
        cell=FC.prepare_cell(pair,d,h);cm=FC.cell_metadata(cell);g=geometries[d+1];k=h==4 ? 1 : 2
        _need(cm.training_rows==g.training[k]&&cm.lead_validation_rows==g.leads[k]&&cm.validation_union_rows==g.indices,"matched342 core/reference full cohorts differ")
        enc=FC.save_encoder(joinpath(stage.component,_encodername(d,h)),cell)
        encoder=M.load_encoder(enc.path;expected_sha256=enc.sha256,expected_training_design_sha256=cm.training_design_sha256)
        positions=Dict(i=>j for (j,i) in enumerate(g.indices));leadpos=[positions[i] for i in g.leads[k]];labels=[T._clean(g.view,15,i+h) for i in g.leads[k]]
        models=NamedTuple[]
        for s in 1:8
            FC.recheck_pair(pair);FC.recheck_cell(cell);_sources(sources)
            now=FC.runtime_identity();_need(now.runtime_sha256==runtime_sha&&_same(now.runtime,runtime),"matched342 runtime changed before fit")
            execution_authority===nothing||FC.recheck_execution_authority(execution_authority,pair)
            expected=FC.expected_bindings(cell,s);started=time_ns()
            fitted=mode=="synthetic_only" ? FC.fit_generated_cell(cell,expected) : FC.fit_actual_cell(cell,expected;execution_authority)
            elapsed=time_ns()-started;_need(elapsed<=typemax(Int64),"matched342 measured fit clock overflow")
            native_values=FC.predict_cell(cell,fitted.capsule,expected;native=fitted.native)
            model=FC.save_model(joinpath(stage.component,_modelname(d,h,s)),fitted.capsule;expected_bindings=expected)
            loaded=FC.load_model(model.path;expected_sha256=model.sha256,expected_bindings=expected)
            replay=FC.replay_cell(cell,loaded,encoder,expected;expected_model_sha256=model.sha256,expected_encoder_sha256=enc.sha256)
            _need(_bits_equal(native_values,replay)&&length(replay)==length(g.indices),"matched342 native/saved full-union predictions differ")
            digest=_increment_sha(replay);object=M._read_json(model.path,model.sha256);model_id=String(object["model_id"])
            values=Float64[routes[d+1][j]==1 ? cm.union_anchors[j] : cm.union_anchors[j]+replay[j] for j in eachindex(replay)]
            _need(all(isfinite,values),"matched342 absolute prediction overflow")
            candidate=_candidate(h,s);_write_column(joinpath(stage.references[d+1],candidate),values)
            metric=BI._reference_metric(values[leadpos],labels)
            receipt=_identified((;schema="matched342_initial_fit_receipt_v1",complete=true,evidence_mode=mode,pair_id=meta.pair_id,context_id=cm.context_id,delay_hours=d,horizon=h,setting_index=s,model_training_mask_sha256=cm.model_training_mask_sha256,reference_training_mask_sha256=cm.reference_training_mask_sha256,reference_validation_cohort_sha256=cm.reference_validation_cohort_sha256,fit_input_sha256=cm.fit_input_sha256,preprocessing_sha256=cm.preprocessing_sha256,training_design_sha256=cm.training_design_sha256,runtime_sha256=runtime_sha,recipe=M._recipe(s),model_id,model_sha256=model.sha256,encoder_sha256=enc.sha256,native_increment_sha256=digest,reloaded_increment_sha256=digest,union_count=length(g.indices),fit_elapsed_ns=Int(elapsed),execution_authority_sha256=authority_sha),:receipt_id)
            P._exclusive(joinpath(stage.component,_fitname(d,h,s)),T._bytes(receipt))
            push!(models,(;setting_index=s,model_id,model_member=_member(stage.component,_modelname(d,h,s);limit=MODEL_BYTES),fit_receipt_member=_member(stage.component,_fitname(d,h,s);limit=T.MAX_META),candidate_member=_member(stage.references[d+1],candidate),native_increment_sha256=digest,reloaded_increment_sha256=digest,n=metric.n,rmse_nt=metric.rmse_nt,mae_nt=metric.mae_nt,bias_nt=metric.bias_nt))
            FC.recheck_cell(cell);_sources(sources);after=FC.runtime_identity()
            _need(after.runtime_sha256==runtime_sha&&_same(after.runtime,runtime),"matched342 runtime changed during fit")
            execution_authority===nothing||FC.recheck_execution_authority(execution_authority,pair)
            _need(P._file_sha(model.path)==(model.sha256,model.bytes)&&P._file_sha(enc.path)==(enc.sha256,enc.bytes),"matched342 saved state changed")
            fitted=nothing;loaded=nothing;expected=nothing;native_values=nothing;replay=nothing;values=nothing;object=nothing
        end
        persistence=count(i->routes[d+1][positions[i]]==1,g.leads[k])
        push!(cell_records,(;delay_hours=d,horizon=h,context_id=cm.context_id,matched_input_id=meta.source_bindings[d+1].matched_input_id,training_count=length(cm.training_rows),validation_count=length(cm.lead_validation_rows),union_count=length(cm.validation_union_rows),validation_persistence_count=persistence,model_training_mask_sha256=cm.model_training_mask_sha256,reference_training_mask_sha256=cm.reference_training_mask_sha256,reference_validation_cohort_sha256=cm.reference_validation_cohort_sha256,fit_input_sha256=cm.fit_input_sha256,preprocessing_sha256=cm.preprocessing_sha256,training_design_sha256=cm.training_design_sha256,max_fit_target_complete=cm.max_fit_target_complete,max_fit_label_receipt=cm.max_fit_label_receipt,max_validation_target_complete=cm.max_validation_target_complete,max_validation_label_receipt=cm.max_validation_label_receipt,encoder_member=_member(stage.component,_encodername(d,h);limit=MODEL_BYTES),models=Tuple(models)))
        # Drop the previous cell before the next prepare call allocates its design.
        cell=nothing;cm=nothing;encoder=nothing
    end
    drafts=NamedTuple[]
    for d in 0:1
        g=geometries[d+1];root=stage.references[d+1];common=_common(meta,d,controls.qualification.sha256,controls.head,sources)
        settings=Tuple(merge((;index=s),M.GRID342[s]) for s in 1:8)
        grid=_identified((;schema="fixed_eight_gbm_settings_v2",complete=true,evidence_mode=mode,feature_contract_sha256=Parent.FEATURE_CONTRACT_SHA256,settings),:grid_id)
        P._exclusive(joinpath(root,"grid.json"),T._bytes(grid));gridsha=_member(root,"grid.json";limit=T.MAX_META).sha256
        leads=NamedTuple[];chosen=Vector{Float64}[];selected=Int[];modelids=String[]
        for k in 1:2
            h=(4,6)[k];cell=cell_records[2d+k];best=1
            for s in 2:8;cell.models[s].rmse_nt<cell.models[best].rmse_nt&&(best=s);end
            push!(selected,best);push!(modelids,cell.models[best].model_id);push!(chosen,_column(joinpath(root,_candidate(h,best)),length(g.indices);expected_sha256=cell.models[best].candidate_member.sha256))
            candidates=Tuple(merge((;index=s),M.GRID342[s],(;model_id=cell.models[s].model_id,fit_input_sha256=cell.fit_input_sha256,prediction_member=cell.models[s].candidate_member.name,prediction_bytes=cell.models[s].candidate_member.bytes,prediction_sha256=cell.models[s].candidate_member.sha256,n=cell.models[s].n,rmse_nt=cell.models[s].rmse_nt,mae_nt=cell.models[s].mae_nt,bias_nt=cell.models[s].bias_nt)) for s in 1:8)
            push!(leads,(;horizon=h,training_mask_sha256=cell.reference_training_mask_sha256,validation_cohort_sha256=cell.reference_validation_cohort_sha256,validation_count=cell.validation_count,max_fit_target_complete=cell.max_fit_target_complete,max_fit_label_receipt=cell.max_fit_label_receipt,max_selection_target_complete=cell.max_validation_target_complete,max_selection_label_receipt=cell.max_validation_label_receipt,candidates,selected_index=best,selected_model_id=cell.models[best].model_id))
        end
        _write_column(joinpath(root,"predictions.bin"),chosen;selected=true)
        tuning=_identified(merge((;schema="b_matched_reference_tuning_v2"),common,(;grid_sha256=gridsha,lead_order=(4,6),leads=Tuple(leads))),:tuning_id)
        P._exclusive(joinpath(root,"tuning.json"),T._bytes(tuning))
        push!(drafts,(;delay_hours=d,root="reference-L$d",raw_row_identity_sha256=_raw_row_id(g,d),prediction_panel_sha256=_member(root,"predictions.bin").sha256,grid_sha256=gridsha,tuning_sha256=_member(root,"tuning.json";limit=T.MAX_META).sha256,routing_member=_member(stage.component,_routename(d)),model_count=count(==(UInt8(0)),routes[d+1]),persistence_count=count(==(UInt8(1)),routes[d+1]),model_ids=Tuple(modelids),selected_setting_ids=Tuple(selected),members=_draft_members(root)))
    end
    FC.recheck_pair(pair);_sources(sources);execution_authority===nothing||FC.recheck_execution_authority(execution_authority,pair)
    for ref in (controls.qualification,controls.feature,controls.loss);_need(P._bound(ref.path)==ref.body,"matched342 input control changed during fitting");end
    component=_identified((;schema="matched342_initial_component_completion_v1",complete=true,evidence_mode=mode,initial_only=true,pair_id=meta.pair_id,source_bindings=meta.source_bindings,training_boundary=meta.training_boundary,validation_start=meta.validation_start,validation_end=meta.validation_end,feature_contract_sha256=Parent.FEATURE_CONTRACT_SHA256,feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,tuning_loss_contract_sha256=Parent.LOSS_CONTRACT_SHA256,evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,baseline_qualified_head=controls.head,baseline_qualification_sha256=controls.qualification.sha256,baseline_sources=_source_rows(sources),runtime,execution_authority_sha256=authority_sha,cells=Tuple(cell_records),drafts=Tuple(drafts),members=_component_members(stage.component)),:component_id)
    body=T._bytes(component)
    _need(length(body)<=T.MAX_META&&sum(x.bytes for x in component.members)+length(body)<=STORE_BYTES,"matched342 component completion/total byte bound")
    path=joinpath(stage.component,"completion.json");P._exclusive(path,body)
    try
        result=read_draft(pair,path;expected_component_sha256=T._sha(body),expected_sources=sources)
        execution_authority===nothing||FC.recheck_execution_authority(execution_authority,pair);result
    catch
        P._invalidate(stage.component,"completion.json",body);rethrow()
    end
end
function produce_initial_generated(s0::BI.BSourcePrefixAdmissionV2,s1::BI.BSourcePrefixAdmissionV2,output;qualification,feature_contract,loss_contract,qualified_head,expected_sources)
    pair=FC.bind_source_pair(s0,s1;loss_contract_sha256=Parent.LOSS_CONTRACT_SHA256,expected_sources)
    _need(FC.source_pair_metadata(pair).evidence_mode===:synthetic_only,"generated matched342 producer requires synthetic admissions")
    _run(pair,output;qualification,feature_contract,loss_contract,qualified_head,execution_authority=nothing,expected_sources)
end
function produce_initial_actual(s0::BI.BSourcePrefixAdmissionV2,s1::BI.BSourcePrefixAdmissionV2,output;qualification,feature_contract,loss_contract,qualified_head,execution_authority,expected_sources)
    pair=FC.bind_source_pair(s0,s1;loss_contract_sha256=Parent.LOSS_CONTRACT_SHA256,expected_sources)
    _need(FC.source_pair_metadata(pair).evidence_mode===:archival_assumed_completion_research,"actual matched342 producer requires archival admissions")
    _run(pair,output;qualification,feature_contract,loss_contract,qualified_head,execution_authority,expected_sources)
end

function publish_initial(draft::InitialMatchedDraft342,checked_audit::Audit.CheckedComponentAudit342;expected_audit_completion_sha256)
    recheck_draft(draft);Audit.recheck_component_audit(checked_audit);audit=Audit.component_audit_metadata(checked_audit)
    info=_read_info(draft.pair,draft.completion_path,draft.completion_sha256,draft.expected_sources);c=info.data;meta=info.metadata
    _need(T._hex(expected_audit_completion_sha256)&&audit.audit_completion_sha256==expected_audit_completion_sha256&&audit.component_completion_sha256==info.sha256&&audit.component_root==info.component&&audit.pair_id==meta.pair_id&&audit.evidence_mode==c.evidence_mode,"matched342 outside audit/component identity differs")
    _need(audit.qualified_head==c.baseline_qualified_head&&audit.component_qualification.sha256==c.baseline_qualification_sha256&&_same(audit.implementation_sources,c.baseline_sources)&&_same(audit.source_bindings,c.source_bindings)&&_same(audit.runtime,c.runtime),"matched342 audit qualification/source/runtime differs")
    _need(audit.contracts.feature_contract_sha256==c.feature_contract_sha256&&audit.contracts.feature_schema_sha256==c.feature_schema_sha256&&audit.contracts.loss_contract_sha256==c.tuning_loss_contract_sha256&&audit.contracts.evotrees_source_sha256==c.evotrees_source_sha256&&T._hex(audit.contracts.audit_contract_sha256)&&length(audit.references)==2,"matched342 audit contract differs")
    _need(!_overlap(audit.audit_root,dirname(info.component)),"matched342 audit store overlaps producer stage")
    for root in info.roots;_need(!ispath(joinpath(root,"audit.json"))&&!ispath(joinpath(root,"completion.json")),"matched342 reference publication already started");end
    owned=NamedTuple[];results=NamedTuple[]
    try
        for d in 0:1
            reference=audit.references[d+1];declared=c.drafts[d+1];root=info.roots[d+1]
            _need(reference.delay_hours==d&&reference.draft_root==root&&_same(reference.source_binding,c.source_bindings[d+1])&&_same(reference.draft_members,declared.members)&&_same(reference.routing_member,declared.routing_member)&&reference.model_count==declared.model_count&&reference.persistence_count==declared.persistence_count,"matched342 audit draft/routing pair differs")
            for (key,value) in ((:training_history_start,meta.training_history_start),(:training_boundary,meta.training_boundary),(:validation_start,meta.validation_start),(:validation_end,meta.validation_end))
                _need(T._clock(getproperty(reference,key))==value,"matched342 audited reference clock differs")
            end
            for key in (:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:model_ids,:selected_setting_ids);_need(_same(getproperty(reference,key),getproperty(declared,key)),"matched342 audited reference fact differs");end
            common=_common(meta,d,c.baseline_qualification_sha256,c.baseline_qualified_head,info.sources)
            ra=_ref(reference.audit_ref);data,_=P._json(ra.path;fields=BI.REFERENCE_AUDIT_FIELDS);_identity(data,:audit_id);_common_check(data,common)
            _need(data.schema=="b_matched_reference_audit_v2"&&data.audit_contract_sha256==audit.contracts.audit_contract_sha256,"matched342 independent reference audit contract differs")
            for key in (:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256);_need(getproperty(data,key)==getproperty(declared,key),"matched342 independent reference audit payload differs");end
            for key in (:all_eight_settings_checked,:full_per_lead_cohorts_checked,:feature_receipt_history_checked,:selected_prediction_columns_checked,:initial_only);_need(getproperty(data,key)===true,"matched342 independent audit incomplete");end
            Audit.recheck_component_audit(checked_audit);recheck_draft(draft)
            P._exclusive(joinpath(root,"audit.json"),ra.body)
            members=sort!(vcat(collect(_draft_members(root;later=true)),[_member(root,"audit.json";limit=T.MAX_META)]);by=x->x.name)
            completion=_identified(merge((;schema="b_matched_reference_completion_v2"),common,(;audit_contract_sha256=data.audit_contract_sha256,model_ids=Tuple(String.(reference.model_ids)),selected_setting_ids=Tuple(Int.(reference.selected_setting_ids)),raw_row_identity_sha256=declared.raw_row_identity_sha256,prediction_panel_sha256=declared.prediction_panel_sha256,grid_sha256=declared.grid_sha256,tuning_sha256=declared.tuning_sha256,audit_sha256=ra.sha256,members=Tuple(members))),:completion_id)
            body=T._bytes(completion)
            _need(length(body)<=T.MAX_META&&sum(x.bytes for x in members)+length(body)<=STORE_BYTES,"matched342 final reference byte bound")
            path=joinpath(root,"completion.json");P._exclusive(path,body);push!(owned,(;root,body))
            pins=BI.BReferencePins(;completion_sha256=T._sha(body),audit_sha256=ra.sha256,grid_sha256=declared.grid_sha256,tuning_sha256=declared.tuning_sha256,prediction_sha256=declared.prediction_panel_sha256,feature_contract_sha256=Parent.FEATURE_CONTRACT_SHA256,tuning_loss_contract_sha256=Parent.LOSS_CONTRACT_SHA256,baseline_qualification_sha256=c.baseline_qualification_sha256,baseline_sources=collect(info.sources),baseline_qualified_head=c.baseline_qualified_head,feature_contract_member=Parent.FEATURE_CONTRACT_MEMBER,tuning_loss_contract_member=Parent.LOSS_CONTRACT_MEMBER)
            panel=BI.admit_reference(path;trusted_pins=pins,roots=Dict("reference"=>root),view=FC.source_admissions(draft.pair)[d+1].view,expected_sources=P.source_identity())
            pin_data=Selection._pins_data(pins)
            push!(results,(;delay_hours=d,completion_path=path,completion_sha256=T._sha(body),audit_sha256=ra.sha256,reference_pins=pin_data,readback_admission_id=panel.admission_id))
            Audit.recheck_component_audit(checked_audit);recheck_draft(draft)
        end
        (;component_completion_sha256=info.sha256,audit_completion_sha256=String(expected_audit_completion_sha256),references=Tuple(results))
    catch error
        cleanup=Any[]
        for item in owned
            try P._invalidate(item.root,"completion.json",item.body) catch failure;push!(cleanup,failure);end
        end
        isempty(cleanup)&&rethrow();throw(CompositeException(vcat(Any[error],cleanup)))
    end
end
end
