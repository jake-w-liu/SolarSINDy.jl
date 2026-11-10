"""Independent source-to-saved-state audit of the initial matched342 reference.

This module never fits or deserializes a native model. Numerical reconstruction
does not call the producer, core preparation, or maintained feature/encoder code.
"""
module MatchedReferenceAudit342
using Dates, SHA, Statistics, JSON3
const Parent=parentmodule(@__MODULE__)
const BI=Parent.BI
const T=Parent.T
const P=Parent.P
const FC=Parent.BMatched342Core
const Replay=Parent.M.MatchedScalarTreeReplay342
export CheckedComponentAudit342,audit_component,read_component_audit,
       recheck_component_audit,component_audit_metadata

const META=2*1024^2
const PAYLOAD=4*1024^2
const TOTAL=128*1024^2
const MAXROWS=500000
const WIDTH=342
const BASE=165
const HOURMS=Int64(3600000)
const PROTOCOL="7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404"
const FEATURE="2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683"
const SCHEMA="109464af1630d16ac766b4f8791ad9f065a528d0aabac201a11fb68a4156249d"
const LOSS="f24c0dd5b9ef4baf58f0595056d5b5a30ab21ff5c278126b5ebefaa9435bc73c"
const EVO="f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f"
const OP="independent_saved_state_audit_initial32"
const CELLS=((0,4),(0,6),(1,4),(1,6))
const GRID=Tuple((;max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255))
const FILLS=(999.,999.,999.,999.,999.,999.,999.,9999.,999.,9999999.,999.,99.99,999.,99.,99999.,9999.,9.)
const RAWLAGS=(0,1,2,3,4,8,12,24)
const DSTLAGS=(1,2,3,4,5,6,12,24)
const SOURCE8=(:delay_hours,:source_proof_sha256,:source_admission_id,:input_view_id,:input_prefix_sha256,:raw_prefix_sha256,:source_sha256,:matched_input_id)
const SOURCE9=(SOURCE8...,:context_id)
const CONTRACTS=(:protocol_sha256,:feature_contract_sha256,:feature_schema_sha256,:loss_contract_sha256,:audit_contract_sha256,:evotrees_source_sha256)
const RUNTIME=(:julia_version,:julia_binary_sha256,:evotrees_version,:evotrees_source_sha256,:threads,:blas_threads,:omp_threads)
const REQUEST=(:schema,:evidence_mode,:operation,:component,:component_qualification,:source_proof,:source_bindings,:contracts,:runtime,:execution_authority,:drafts,:output_directory,:request_id)
const COMPONENT=(:schema,:complete,:evidence_mode,:initial_only,:pair_id,:source_bindings,:training_boundary,:validation_start,:validation_end,:feature_contract_sha256,:feature_schema_sha256,:tuning_loss_contract_sha256,:evotrees_source_sha256,:baseline_qualified_head,:baseline_qualification_sha256,:baseline_sources,:runtime,:execution_authority_sha256,:cells,:drafts,:members,:component_id)
const CELL=(:delay_hours,:horizon,:context_id,:matched_input_id,:training_count,:validation_count,:union_count,:validation_persistence_count,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256,:max_fit_target_complete,:max_fit_label_receipt,:max_validation_target_complete,:max_validation_label_receipt,:encoder_member,:models)
const MODELREC=(:setting_index,:model_id,:model_member,:fit_receipt_member,:candidate_member,:native_increment_sha256,:reloaded_increment_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const FITREC=(:schema,:complete,:evidence_mode,:pair_id,:context_id,:delay_hours,:horizon,:setting_index,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256,:runtime_sha256,:recipe,:model_id,:model_sha256,:encoder_sha256,:native_increment_sha256,:reloaded_increment_sha256,:union_count,:fit_elapsed_ns,:execution_authority_sha256,:receipt_id)
const DRAFT=(:delay_hours,:root,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:routing_member,:model_count,:persistence_count,:model_ids,:selected_setting_ids,:members)
const MODELKEYS=(:schema,:feature_count,:feature_schema_sha256,:evotrees_version,:evotrees_source_sha256,:cell_delay,:cell_horizon,:setting_index,:max_depth,:nrounds,:nbins,:seed,:training_mask_sha256,:preprocessing_sha256,:input_admission_sha256,:feature_contract_sha256,:loss_contract_sha256,:numeric_feature_types,:edges_f64_bits,:trees,:model_id)
const ENCODERKEYS=(:schema,:feature_count,:feature_schema_sha256,:feature_contract_sha256,:training_rows,:finite_counts,:imputation_f64_bits,:center_f64_bits,:scale_f64_bits,:training_design_sha256)
const REF_COMMON=(:complete,:evidence_mode,:reference_kind,:source_sha256,:input_prefix_sha256,:input_view_id,:delay_hours,:training_boundary,:validation_start,:validation_end,:training_history_start,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256,:baseline_sources)
const TUNING=(:schema,REF_COMMON...,:grid_sha256,:lead_order,:leads,:tuning_id)
const LEAD=(:horizon,:training_mask_sha256,:validation_cohort_sha256,:validation_count,:max_fit_target_complete,:max_fit_label_receipt,:max_selection_target_complete,:max_selection_label_receipt,:candidates,:selected_index,:selected_model_id)
const CANDIDATE=(:index,:max_depth,:nrounds,:nbins,:model_id,:fit_input_sha256,:prediction_member,:prediction_bytes,:prediction_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const REFERENCE=(:delay_hours,:source_binding,:draft_root,:draft_members,:routing_member,:model_count,:persistence_count,:audit_ref,:training_history_start,:training_boundary,:validation_start,:validation_end,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:model_ids,:selected_setting_ids)
const CELLCHECK=(:delay_hours,:horizon,:context_id,:training_count,:validation_count,:union_count,:validation_persistence_count,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:training_design_sha256,:preprocessing_sha256,:fit_input_sha256,:encoder_sha256,:max_fit_target_complete,:max_fit_label_receipt,:max_selection_target_complete,:max_selection_label_receipt,:models)
const MODELCHECK=(:setting_index,:model_id,:model_sha256,:fit_receipt_sha256,:native_increment_sha256,:reloaded_increment_sha256,:audited_increment_sha256,:candidate_prediction_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const REPORT=(:schema,:complete,:evidence_mode,:pair_id,:component_completion_sha256,:source_bindings,:cells,:references,:fitting_performed,:native_model_deserialization,:report_id)
const AUDIT_NAMES=("audit-completion.json","audit-report.json","reference-audit-L0.json","reference-audit-L1.json")
const CA=(:schema,:complete,:evidence_mode,:operation,:qualified_head,:implementation_sources,:pair_id,:source_proof_sha256,:source_proof,:source_bindings,:contracts,:runtime,:component_root,:component_completion_sha256,:component_qualification,:execution_authority_sha256,:execution_authority,:references,:report_sha256,:members,:audit_id)
const PERMIT=(:schema,:complete,:evidence_mode,:operation,:qualified_head,:implementation_sources,:contracts,:sources,:source_proof,:root_qualification,:component_qualification,:resource_qualification,:runtime,:execution,:scope,:limits,:permit_id)
const SCOPE=(:training_boundary_ms,:validation_start_ms,:validation_end_ms,:embargo_hours,:cell_order,:setting_order,:fit_count,:encoder_count,:final_fit_count,:calibration_fit_count,:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion)

need(x,msg)=x||throw(ArgumentError(msg))
is_sequence(x)=x isa AbstractVector||x isa Tuple
int(x,lo,hi)=T._int(x,lo,hi)
is_digest(x)=x isa AbstractString&&occursin(r"\A[0-9a-f]{64}\z",x)
bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
sha(x)=bytes2hex(sha256(x))
same(a,b)=T._bytes(a)==T._bytes(b)
clock(x::DateTime)=string(Dates.value(x))
shift(t::DateTime,h::Integer)=DateTime(Dates.UTInstant{Millisecond}(Millisecond(Base.Checked.checked_add(Dates.value(t),Base.Checked.checked_mul(Int64(h),HOURMS)))))
freeze(x::NamedTuple)=NamedTuple{keys(x)}(map(freeze,values(x)))
freeze(x::AbstractDict)=begin names=Tuple(Symbol(k) for k in sort!(String.(collect(keys(x)))));NamedTuple{names}(Tuple(freeze(x[String(k)]) for k in names)) end
freeze(x::AbstractVector)=Tuple(freeze.(x))
freeze(x::Tuple)=map(freeze,x)
freeze(x::AbstractString)=String(x)
freeze(x)=x
function fields(x,names)
    need(x isa NamedTuple||x isa AbstractDict,"audit object required")
    T._fields(x,names);x
end
function check_identity(x,key)
    names=Tuple(Symbol(k) for k in keys(x) if Symbol(k)!=key)
    need(getproperty(x,key)==T._id(NamedTuple{names}(Tuple(getproperty(x,k) for k in names))),"audit canonical content identity differs")
    x
end
function withid(x,key)
    merge(x,NamedTuple{(key,)}((T._id(x),)))
end
function json_depth(body)
    stack=UInt8[];quoted=false;escaped=false;scalar=false;values=0;containers=0
    for b in body
        if quoted
            if escaped;escaped=false
            elseif b==0x5c;escaped=true
            elseif b==0x22;quoted=false
            end
        elseif b==0x22
            need(length(stack)<=12,"audit JSON nesting bound");quoted=true;scalar=false;values+=1
        elseif b in (0x7b,0x5b)
            need(length(stack)<=12,"audit JSON nesting bound");push!(stack,b);scalar=false;values+=1;containers+=1
        elseif b in (0x7d,0x5d)
            need(!isempty(stack)&&pop!(stack)==(b==0x7d ? 0x7b : 0x5b),"audit JSON container mismatch");scalar=false
        elseif b in (0x20,0x09,0x0a,0x0d,0x2c,0x3a);scalar=false
        else
            need(length(stack)<=12,"audit JSON nesting bound")
            if !scalar;values+=1;scalar=true;end
        end
        # Counts include keys as well as values: every valid bounded model and
        # 500000-row encoder is inside these pre-backend structural budgets.
        need(values<=600000&&containers<=4096,"audit JSON structural budget exceeded")
    end
    need(!quoted&&!escaped&&isempty(stack),"unfinished audit JSON")
end
function readjson(path;expected=nothing,names=nothing,limit=META)
    body=P._bound(path,limit);need(!isempty(body),"empty audit JSON")
    expected===nothing||need(is_digest(expected)&&sha(body)==expected,"audit file content pin differs")
    json_depth(body)
    object=try JSON3.read(body) catch e
        (e isa InterruptException||e isa OutOfMemoryError||e isa StackOverflowError)&&rethrow()
        throw(ArgumentError("invalid audit JSON"))
    end
    need(T._bytes(object)==body,"audit JSON must be canonical without duplicate keys or numeric float tokens")
    names===nothing||fields(object,names)
    object,body
end
function fileref(x;limit=META)
    fields(x,(:path,:sha256));need(x.path isa AbstractString&&isabspath(x.path)&&is_digest(x.sha256),"audit external file reference differs")
    path=P._ordinary(String(x.path));need(path==String(x.path)&&isfile(path)&&1<=filesize(path)<=limit,"audit external ordinary canonical file differs")
    need(P._file_sha(path)==(String(x.sha256),filesize(path)),"audit external file SHA differs")
    (;path,sha256=String(x.sha256))
end
function source_map(x)
    if x isa AbstractDict
        return Tuple(T._hashmap(pairs(x)))
    end
    pairs=Pair{String,String}[]
    for entry in x
        if entry isa Pair;push!(pairs,String(first(entry))=>String(last(entry)))
        else;fields(entry,(:name,:sha256));push!(pairs,String(entry.name)=>String(entry.sha256))
        end
    end
    Tuple(T._hashmap(pairs))
end
source_records(x)=Tuple((;name=first(p),sha256=last(p)) for p in source_map(x))
function current_sources(expected)
    actual=source_map(Parent.source_identity());need(source_map(expected)==actual,"audit current new-component closure differs");source_records(actual)
end
function filemember(root,x;name=nothing,limit=META)
    fields(x,(:name,:bytes,:sha256));need(x.name isa AbstractString&&is_digest(x.sha256),"audit member grammar differs")
    name===nothing||need(x.name==name,"audit fixed member name differs")
    n=int(x.bytes,1,limit);path=P._member(root,String(x.name))
    need(P._file_sha(path)==(String(x.sha256),n),"audit member bytes/SHA differ")
    (;name=String(x.name),bytes=n,sha256=String(x.sha256))
end
function fixed_members(root,entries,names;payload=false)
    need(entries isa AbstractVector||entries isa Tuple,"audit member list required")
    expected=sort!(collect(names));need(length(entries)==length(expected),"audit exact member count differs")
    Tuple(filemember(root,e;name=n,limit=payload&&(endswith(n,".model.json")||endswith(n,"encoder.json")) ? PAYLOAD : n in ("routing/L0.bin","routing/L1.bin") ? MAXROWS+16 : startswith(n,"candidates/") ? 16+8MAXROWS : n=="predictions.bin" ? 16+16MAXROWS : META) for (e,n) in zip(entries,expected))
end
function component_names()
    names=String["qualification.json","feature-contract","loss-contract","routing/L0.bin","routing/L1.bin"]
    for (L,h) in CELLS
        push!(names,"L$L-h$h/encoder.json")
        for s in 1:8;push!(names,"L$L-h$h/setting-$s.model.json","L$L-h$h/setting-$s.fit.json");end
    end
    sort!(names)
end
draft_names()=sort!(vcat(["grid.json","tuning.json","predictions.bin","qualification.json"],["candidates/h$h/setting-$s.bin" for h in (4,6) for s in 1:8]))
function exact_tree(root,names,dirs;cap=TOTAL,optional=())
    need(isdir(P._ordinary(root)),"audit ordinary root required")
    observed=String[];directories=String[];total=0
    for (parent,sub,files) in walkdir(root;follow_symlinks=false)
        P._ordinary(parent)
        for n in sub;path=P._ordinary(joinpath(parent,n));push!(directories,replace(relpath(path,root),'\\'=>'/'));end
        for n in files
            path=P._ordinary(joinpath(parent,n));push!(observed,replace(relpath(path,root),'\\'=>'/'))
            total+=filesize(path);need(total<=cap,"audit store byte bound exceeded")
        end
    end
    need(Set(directories)==Set(dirs)&&all(n->n in names||n in optional,observed)&&all(n->n in observed,names),"audit exact ordinary tree differs")
    nothing
end
function pin!(pins,path;limit=T.MAX_FILE,expected=nothing,expected_bytes=nothing)
    path=P._ordinary(path);digest,n=P._file_sha(path);need(1<=n<=limit,"audit pin size differs")
    expected===nothing||need(digest==expected,"audit pin changed from declared content")
    expected_bytes===nothing||need(n==expected_bytes,"audit pin changed from declared size")
    p=(;path,sha256=digest,bytes=n)
    old=findfirst(x->x.path==path,pins)
    if old===nothing;push!(pins,p)
    else;need(pins[old]==p,"audit input changed during scope")
    end
    p
end
function recheck_pins(pins)
    for p in pins;need(P._file_sha(p.path)==(p.sha256,p.bytes),"audit retained input changed");end
    nothing
end
function disjoint(a,b)
    aa=replace(normpath(abspath(a)),'\\'=>'/');bb=replace(normpath(abspath(b)),'\\'=>'/')
    Sys.iswindows()&&(aa=lowercase(aa);bb=lowercase(bb))
    !(aa==bb||startswith(aa,bb*"/")||startswith(bb,aa*"/"))
end
function f64(x)
    T._float(x)
end
function f32(x)
    need(x isa AbstractString&&occursin(r"\A[0-9a-f]{8}\z",x),"audit finite Float32 bits required")
    y=reinterpret(Float32,parse(UInt32,x;base=16));need(isfinite(y),"audit nonfinite Float32 leaf");y
end

# Independent raw/source identities. Default completion clocks are constructed
# here, not taken from a prepared-cell or model declaration.
function independent_pair(pair,expected_sources)
    FC.recheck_pair(pair);sources=current_sources(expected_sources)
    admissions=FC.source_admissions(pair);meta=FC.source_pair_metadata(pair)
    need(length(admissions)==2,"audit two genuine source admissions required")
    raw=admissions[1].view.input;times=copy(raw.time);n=length(times)
    need(1<=n<=MAXROWS&&size(raw.values)==(17,n)&&all(diff(times).==Hour(1))&&all(t->floor(t,Hour)==t,times),"audit fixed contiguous raw geometry differs")
    mode=String(admissions[1].metadata.evidence_mode)
    need(mode in ("synthetic_only","archival_assumed_completion_research"),"audit source evidence mode differs")
    clean=Matrix{Float64}(undef,17,n)
    for i in 1:n,j in 1:17
        x=raw.values[j,i];need(x isa Real&&!(x isa Bool)&&(isfinite(x)||isnan(x)),"audit invalid raw scalar")
        y=Float64(x);need(isfinite(y)||isnan(y),"audit raw conversion overflow")
        clean[j,i]=(!isfinite(y)||y>=FILLS[j]) ? NaN : y
    end
    upstream=ntuple(2) do k
        s=admissions[k];m=s.metadata;v=s.view;L=k-1
        BI.recheck_source_v2(s)
        need(m.delay_hours==L&&String(m.evidence_mode)==mode&&v.input.time==times&&isequal(v.input.values,raw.values),"audit independently admitted delay/raw pair differs")
        for key in (:source_sha256,:raw_prefix_sha256,:training_boundary,:validation_start,:validation_end,:prefix_start,:prefix_stop,:prefix_rows)
            need(getproperty(m,key)==getproperty(admissions[1].metadata,key),"audit shared source/split differs")
        end
        need(m.prefix_rows==n&&m.prefix_start==first(times)&&m.prefix_stop==shift(last(times),1)&&v.input.source_sha256==m.source_sha256,"audit source prefix bounds differ")
        ctx=SHA.SHA2_256_CTX();buffer=IOBuffer();write(buffer,"matched342-input-v1|$(m.source_sha256)|$L|$n\n")
        for i in 1:n
            write(buffer,htol(Int64(Dates.value(times[i]))))
            for j in 1:17
                write(buffer,htol(reinterpret(UInt64,clean[j,i])));write(buffer,htol(Int64(Dates.value(shift(times[i],1+(j==15 ? L : 0))))))
            end
            position(buffer)>=65536&&SHA.update!(ctx,take!(buffer))
        end
        position(buffer)>0&&SHA.update!(ctx,take!(buffer));input_id=bytes2hex(SHA.digest!(ctx))
        (;delay_hours=L,source_proof_sha256=s.source_proof_sha256,source_admission_id=s.source_admission_id,input_view_id=v.view_id,input_prefix_sha256=v.input_prefix_sha256,raw_prefix_sha256=m.raw_prefix_sha256,source_sha256=m.source_sha256,matched_input_id=input_id)
    end
    m=admissions[1].metadata
    pairdata=(;schema="matched342_source_pair_context_v1",evidence_mode=mode,source_sha256=m.source_sha256,raw_prefix_sha256=m.raw_prefix_sha256,input_prefix_sha256=upstream[1].input_prefix_sha256,training_history_start_ms=clock(first(times)),prefix_stop_ms=clock(m.prefix_stop),source_rows=n,training_boundary_ms=clock(m.training_boundary),validation_start_ms=clock(m.validation_start),validation_end_ms=clock(m.validation_end),source_bindings=upstream,feature_contract_sha256=FEATURE,feature_schema_sha256=SCHEMA,tuning_loss_contract_sha256=LOSS,baseline_sources=sources)
    pair_id=T._id(pairdata)
    bindings=ntuple(k->merge(upstream[k],(;context_id=T._id((;schema="matched342_source_context_v1",evidence_mode=mode,pair_id,source_binding=upstream[k])))),2)
    need(meta.pair_id==pair_id&&same(meta.source_bindings,bindings)&&same(meta.baseline_sources,sources),"audit independently reconstructed pair/context differs")
    FC.recheck_pair(pair)
    (;pair,admissions,mode,times,clean,n,sources,pair_id,bindings,boundary=m.training_boundary,start=m.validation_start,stop=m.validation_end)
end

# Analytical direct29, raw136, missing165 and trace12 construction. Wind rows
# are completed at issue; virtual Dst rows are shifted by the source delay.
function direct29!(out,raw,i,L)
    ai=i-L
    1<=ai<=size(raw,2)&&isfinite(raw[15,ai])&&ai>1&&isfinite(raw[15,ai-1])||return
    for lag in 0:6
        r=i-lag;r>=1||return
        all(j->isfinite(raw[j,r]),(5,6,8,9,12))&&raw[9,r]>0.||return
    end
    all(lag->ai-lag>=1&&isfinite(raw[15,ai-lag]),DSTLAGS)||return
    bsum=0.;perpsum=0.;lnsum=0.
    for lag in 0:5;bsum+=raw[6,i-lag];perpsum+=hypot(raw[5,i-lag],raw[6,i-lag]);lnsum+=log(raw[9,i-lag]);end
    bm=bsum/6;ss=0.;vs=0.
    for lag in 0:5;ss+=abs2(raw[6,i-lag]-bm);end
    for lag in 0:2;vs+=raw[8,i-lag]*max(0.,-raw[6,i-lag])/1000.;end
    south=0
    for lag in 0:11;r=i-lag;r>=1&&isfinite(raw[6,r])&&raw[6,r]<0.||break;south+=1;end
    out[1]=raw[6,i];out[2]=raw[6,i-1];out[3]=raw[6,i-2];out[4]=bm;out[5]=sqrt(ss/6)
    out[6]=raw[5,i];out[7]=hypot(raw[5,i],raw[6,i]);out[8]=perpsum/6
    out[9]=raw[8,i];out[10]=raw[8,i]-raw[8,i-6];out[11]=log(raw[9,i]);out[12]=lnsum/6
    out[13]=raw[12,i];out[14]=raw[8,i]*max(0.,-raw[6,i])/1000.;out[15]=vs/3;out[16]=south
    out[17]=raw[15,ai];out[18]=raw[15,ai]-raw[15,ai-1]
    for (j,lag) in enumerate(DSTLAGS);out[18+j]=raw[15,ai-lag];end
    for lag in 1:3;out[26+lag]=raw[8,i-lag]*max(0.,-raw[6,i-lag])/1000.;end
    need(all(isfinite,@view(out[1:29])),"audit direct29 arithmetic overflow")
end
function design342(raw::Matrix{Float64},L)
    n=size(raw,2);X=Matrix{Float64}(undef,n,WIDTH);age=0;run=0;censored=false;previous=0
    for i in 1:n
        row=@view X[i,:];fill!(row,NaN);direct29!(row,raw,i,L)
        k=29
        for j in 1:17,lag in RAWLAGS
            k+=1;r=i-lag
            row[k]=r>=1&&(j!=15||r+L<=i) ? raw[j,r] : NaN
        end
        need(k==BASE&&all(x->isfinite(x)||isnan(x),@view(row[1:BASE])),"audit raw feature width/overflow differs")
        for j in 1:BASE;row[BASE+j]=Float64(isnan(row[j]));end
        bz=raw[6,i];v=raw[8,i];q=isfinite(bz)&&isfinite(v)&&v>=0 ? max(v*max(-bz,0.)/1000-.49,0.) : NaN
        need(isfinite(q)||isnan(q),"audit trace injection overflow")
        known=isfinite(bz)&&isfinite(q);state=known ? (bz<0. ? 2 : 1) : 0
        if !known;age=0;run=0;censored=false
        elseif previous==0;age=0;run=1;censored=true
        elseif state!=previous;age=1;run=1;censored=false
        else;run+=1;age=age>0 ? min(age+1,49) : run>=49 ? 49 : 0
        end
        row[331]=L;row[332]=0.;row[333]=state==1;row[334]=state==2
        row[335]=known&&age>0;row[336]=age;row[337]=min(run,49);row[338]=censored
        row[339]=age==49;row[340]=known&&age==0;row[341]=0.;row[342]=!known
        previous=state
    end
    X
end
function cohorts(a,L,h)
    train=Int[];validation=Int[];fitcut=shift(a.boundary,-168);valcut=shift(a.stop,-168)
    for i in 1:a.n
        ai=i-L;ti=i+h;1<=ai<=a.n&&ti<=a.n||continue
        isfinite(a.clean[15,ai])&&isfinite(a.clean[15,ti])||continue
        issue=shift(a.times[i],1);complete=shift(issue,h);receipt=shift(complete,L)
        shift(a.times[ai],1+L)<=issue||continue
        issue<a.boundary&&complete<=fitcut&&receipt<=fitcut&&push!(train,i)
        a.start<=issue<a.stop&&complete<=valcut&&receipt<=valcut&&push!(validation,i)
    end
    need(!isempty(train)&&!isempty(validation),"audit full nonempty per-lead cohorts required")
    (;train,validation)
end
function modelmask(a,L,h,train)
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("matched342-training-mask-v1\n$(a.bindings[L+1].matched_input_id)\n"))
    header=Int64[L,h,Dates.value(a.boundary),Dates.value(a.start),Dates.value(a.stop),length(train)]
    SHA.update!(ctx,reinterpret(UInt8,htol.(header)))
    for i in train
        issue=shift(a.times[i],1);anchor=shift(issue,-1-L);target=shift(issue,h-1)
        r=Int64[i,Dates.value(issue),Dates.value(anchor),Dates.value(shift(anchor,1)),Dates.value(shift(anchor,1+L)),Dates.value(target),Dates.value(shift(target,1)),Dates.value(shift(target,1+L))]
        SHA.update!(ctx,reinterpret(UInt8,htol.(r)))
    end
    bytes2hex(SHA.digest!(ctx))
end
function refmask(a,L,h,rows,role)
    T._id((;schema="b_reference_per_lead_mask_v1",source_sha256=a.bindings[L+1].source_sha256,input_view_id=a.bindings[L+1].input_view_id,role,horizon=h,delay_hours=L,training_boundary=a.boundary,validation_start=a.start,validation_end=a.stop,clocks=[begin issue=shift(a.times[i],1);(;issue,target_complete=shift(issue,h),target_receipt=shift(issue,h+L)) end for i in rows]))
end
function training_design_sha(X,rows)
    ctx=SHA.SHA2_256_CTX();buffer=IOBuffer();write(buffer,"matched342-training-design-v1\n")
    for i in rows
        write(buffer,htol(Int64(i)))
        for j in 1:WIDTH;x=X[i,j];write(buffer,htol(isnan(x) ? UInt64(0x7ff8000000000000) : reinterpret(UInt64,x)));end
        position(buffer)>=65536&&SHA.update!(ctx,take!(buffer))
    end
    position(buffer)>0&&SHA.update!(ctx,take!(buffer));bytes2hex(SHA.digest!(ctx))
end
function independently_encoded(X,rows)
    need(!isempty(rows)&&issorted(rows)&&allunique(rows),"audit ordered training rows differ")
    counts=Int[];impute=Float64[]
    for j in 1:BASE
        finite=Float64[X[i,j] for i in rows if isfinite(X[i,j])]
        push!(counts,length(finite));push!(impute,isempty(finite) ? 0. : mean(finite))
    end
    selected=Matrix{Float64}(X[rows,:])
    for r in axes(selected,1),j in 1:BASE;isnan(selected[r,j])&&(selected[r,j]=impute[j]);end
    center=vec(mean(selected;dims=1));scale=length(rows)==1 ? ones(WIDTH) : vec(std(selected;dims=1,corrected=true))
    scale[scale.==0.].=1.;need(all(isfinite,impute)&&all(isfinite,center)&&all(x->isfinite(x)&&x>0,scale),"audit training moments overflow")
    object=(;schema="matched342_encoder_v1",feature_count=WIDTH,feature_schema_sha256=SCHEMA,feature_contract_sha256=FEATURE,training_rows=rows,finite_counts=counts,imputation_f64_bits=bits.(impute),center_f64_bits=bits.(center),scale_f64_bits=bits.(scale),training_design_sha256=training_design_sha(X,rows))
    (;object,impute=Tuple(impute),center=Tuple(center),scale=Tuple(scale),sha256=sha(T._bytes(object)))
end
function normalize_row!(dest,raw,encoder)
    for j in 1:WIDTH
        x=raw[j];isnan(x)&&(x=encoder.impute[j])
        dest[j]=(x-encoder.center[j])/encoder.scale[j];need(isfinite(dest[j]),"audit normalized feature overflow")
    end
    dest
end
function fitinput(a,L,h,rows,X,e,mask)
    header=(;schema="matched342_initial_fit_input_header_v1",evidence_mode=a.mode,context_id=a.bindings[L+1].context_id,delay_hours=L,horizon=h,model_training_mask_sha256=mask,preprocessing_sha256=e.sha256,training_design_sha256=e.object.training_design_sha256,training_rows_count=length(rows),feature_count=WIDTH)
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("matched342-initial-fit-input-v1\n"));SHA.update!(ctx,T._bytes(header));buffer=IOBuffer();values=Vector{Float64}(undef,WIDTH)
    for i in rows
        normalize_row!(values,@view(X[i,:]),e);write(buffer,htol(Int64(i)))
        for x in values;write(buffer,htol(reinterpret(UInt64,x)));end
        y=a.clean[15,i+h]-a.clean[15,i-L];need(isfinite(y),"audit response increment overflow");write(buffer,htol(reinterpret(UInt64,y)))
        position(buffer)>=65536&&SHA.update!(ctx,take!(buffer))
    end
    position(buffer)>0&&SHA.update!(ctx,take!(buffer));bytes2hex(SHA.digest!(ctx))
end
function decode_model(path,digest,expected,s)
    o,_=readjson(path;expected=digest,names=MODELKEYS,limit=PAYLOAD);check_identity(o,:model_id);g=GRID[s]
    for (key,value) in pairs(expected);need(same(getproperty(o,key),value),"audit saved model context differs: $key");end
    need(o.schema=="matched_gbm_scalar342_model_v1"&&int(o.feature_count,WIDTH,WIDTH)==WIDTH&&o.feature_schema_sha256==SCHEMA&&o.evotrees_version=="0.18.7"&&o.evotrees_source_sha256==EVO,"audit saved model schema/library differs")
    need(int(o.max_depth,1,6)==g.max_depth&&int(o.nrounds,0,400)==g.nrounds&&int(o.nbins,2,255)==g.nbins&&int(o.seed,22022026,22022026)==22022026,"audit fixed model recipe differs")
    need(is_sequence(o.numeric_feature_types)&&length(o.numeric_feature_types)==WIDTH&&all(x->x===true,o.numeric_feature_types)&&is_sequence(o.edges_f64_bits)&&length(o.edges_f64_bits)==WIDTH&&is_sequence(o.trees)&&length(o.trees)==g.nrounds+1,"audit saved numeric model geometry differs")
    edges=Tuple(begin need(is_sequence(x)&&length(x)==g.nbins-1,"audit saved bin edge count differs");Tuple(f64.(x)) end for x in o.edges_f64_bits)
    trees=Tuple(begin
        fields(t,(:feature,:condition,:prediction_f32_bits,:split));n=k==1 ? 1 : 2^g.max_depth-1
        need(all(is_sequence(getproperty(t,f))&&length(getproperty(t,f))==n for f in keys(t)),"audit saved four-field heap differs")
        (;feature=Tuple(int(x,0,WIDTH) for x in t.feature),condition=Tuple(int(x,0,g.nbins-1) for x in t.condition),prediction=Tuple(f32.(t.prediction_f32_bits)),split=Tuple(begin need(x isa Bool,"audit saved split Boolean required");x end for x in t.split))
    end for (k,t) in enumerate(o.trees))
    model=Replay.Model(edges,trees;max_depth=g.max_depth,nrounds=g.nrounds)
    need(model.nbins==g.nbins,"audit saved common bin geometry differs")
    model,String(o.model_id)
end

function check_contracts(x,auditsha)
    fields(x,CONTRACTS);need(all(is_digest,getproperty.(Ref(x),CONTRACTS)),"audit contract digest grammar differs")
    need(x.protocol_sha256==PROTOCOL&&x.feature_contract_sha256==FEATURE&&x.feature_schema_sha256==SCHEMA&&x.loss_contract_sha256==LOSS&&x.evotrees_source_sha256==EVO&&x.audit_contract_sha256==auditsha,"audit external frozen contracts differ")
    NamedTuple{CONTRACTS}(Tuple(String(getproperty(x,k)) for k in CONTRACTS))
end
function check_runtime(x)
    fields(x,RUNTIME)
    need(x.julia_version isa AbstractString&&!isempty(x.julia_version)&&is_digest(x.julia_binary_sha256)&&x.evotrees_version=="0.18.7"&&x.evotrees_source_sha256==EVO,"audit fitted runtime identity differs")
    need(all(int(getproperty(x,k),1,1)==1 for k in (:threads,:blas_threads,:omp_threads)),"audit single-thread native runtime required")
    NamedTuple{RUNTIME}((String(x.julia_version),String(x.julia_binary_sha256),String(x.evotrees_version),String(x.evotrees_source_sha256),Int(x.threads),Int(x.blas_threads),Int(x.omp_threads)))
end
function check_bindings(x,expected)
    need(is_sequence(x)&&length(x)==2,"audit exactly two source bindings required")
    for (i,b) in enumerate(x)
        fields(b,SOURCE9);need(int(b.delay_hours,0,1)==i-1&&all(is_digest,getproperty.(Ref(b),SOURCE9[2:end])),"audit ordered source-binding grammar differs")
    end
    need(same(x,expected),"audit independent source bindings differ")
    freeze(x)
end
function check_common(x,a,L,qsha)
    base=(;complete=true,evidence_mode=a.mode,reference_kind="matched_information_gbm_initial_v1",source_sha256=a.bindings[L+1].source_sha256,input_prefix_sha256=a.bindings[L+1].input_prefix_sha256,input_view_id=a.bindings[L+1].input_view_id,delay_hours=L,training_boundary=clock(a.boundary),validation_start=clock(a.start),validation_end=clock(a.stop),training_history_start=clock(first(a.times)),feature_contract_sha256=FEATURE,tuning_loss_contract_sha256=LOSS,baseline_qualification_sha256=qsha,baseline_sources=a.sources)
    for k in REF_COMMON;need(same(getproperty(x,k),getproperty(base,k)),"audit B reference common fact differs: $k");end
    base
end
function member_headers(entries,names;payload=false)
    need(entries isa AbstractVector||entries isa Tuple,"audit closed member list required")
    sorted=sort!(collect(names));need(length(entries)==length(sorted),"audit closed member count differs")
    for (e,n) in zip(entries,sorted)
        fields(e,(:name,:bytes,:sha256));need(e.name==n&&is_digest(e.sha256),"audit closed member name/SHA differs")
        int(e.bytes,1,payload&&(endswith(n,".model.json")||endswith(n,"encoder.json")) ? PAYLOAD : n in ("routing/L0.bin","routing/L1.bin") ? MAXROWS+16 : startswith(n,"candidates/")||n=="predictions.bin" ? 16+16MAXROWS : META)
    end
    nothing
end
function component_headers(c,a)
    fields(c,COMPONENT);check_identity(c,:component_id)
    need(c.schema=="matched342_initial_component_completion_v1"&&c.complete===true&&c.initial_only===true&&c.evidence_mode==a.mode&&c.pair_id==a.pair_id,"audit completed initial component header differs")
    check_bindings(c.source_bindings,a.bindings);check_runtime(c.runtime)
    need(c.training_boundary==clock(a.boundary)&&c.validation_start==clock(a.start)&&c.validation_end==clock(a.stop)&&c.feature_contract_sha256==FEATURE&&c.feature_schema_sha256==SCHEMA&&c.tuning_loss_contract_sha256==LOSS&&c.evotrees_source_sha256==EVO,"audit component split/contracts differ")
    need(c.baseline_qualified_head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",c.baseline_qualified_head)&&is_digest(c.baseline_qualification_sha256)&&same(c.baseline_sources,a.sources),"audit component externally qualified source/head differs")
    need(is_sequence(c.cells)&&length(c.cells)==4&&is_sequence(c.drafts)&&length(c.drafts)==2,"audit complete four cells/two drafts required")
    member_headers(c.members,component_names();payload=true)
    for (cell,(L,h)) in zip(c.cells,CELLS)
        fields(cell,CELL);need(int(cell.delay_hours,0,1)==L&&int(cell.horizon,4,6)==h&&cell.context_id==a.bindings[L+1].context_id&&cell.matched_input_id==a.bindings[L+1].matched_input_id,"audit component ordered cell binding differs")
        for k in (:training_count,:validation_count,:union_count);int(getproperty(cell,k),1,MAXROWS);end
        int(cell.validation_persistence_count,0,MAXROWS)
        for k in (:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,:preprocessing_sha256,:training_design_sha256);need(is_digest(getproperty(cell,k)),"audit component cell digest differs");end
        for k in (:max_fit_target_complete,:max_fit_label_receipt,:max_validation_target_complete,:max_validation_label_receipt);T._clock(getproperty(cell,k));end
        fields(cell.encoder_member,(:name,:bytes,:sha256));need(cell.encoder_member.name=="L$L-h$h/encoder.json"&&is_digest(cell.encoder_member.sha256),"audit fixed cell encoder member differs");int(cell.encoder_member.bytes,1,PAYLOAD)
        need(is_sequence(cell.models)&&length(cell.models)==8,"audit all eight cell states required")
        for (s,m) in enumerate(cell.models)
            fields(m,MODELREC);need(int(m.setting_index,1,8)==s&&is_digest(m.model_id)&&is_digest(m.native_increment_sha256)&&is_digest(m.reloaded_increment_sha256),"audit ordered cell model record differs")
            for (k,n,cap) in ((:model_member,"L$L-h$h/setting-$s.model.json",PAYLOAD),(:fit_receipt_member,"L$L-h$h/setting-$s.fit.json",META),(:candidate_member,"candidates/h$h/setting-$s.bin",16+8MAXROWS))
                e=getproperty(m,k);fields(e,(:name,:bytes,:sha256));need(e.name==n&&is_digest(e.sha256),"audit fixed model/receipt/candidate member differs");int(e.bytes,1,cap)
            end
            int(m.n,1,MAXROWS);for k in (:rmse_nt,:mae_nt,:bias_nt);f64(getproperty(m,k));end
        end
    end
    for (L,d) in enumerate(c.drafts)
        L-=1;fields(d,DRAFT);need(int(d.delay_hours,0,1)==L&&d.root=="reference-L$L","audit fixed delay draft differs")
        for k in (:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256);need(is_digest(getproperty(d,k)),"audit draft digest differs");end
        fields(d.routing_member,(:name,:bytes,:sha256));need(d.routing_member.name=="routing/L$L.bin"&&is_digest(d.routing_member.sha256),"audit draft route member differs");int(d.routing_member.bytes,17,MAXROWS+16)
        int(d.model_count,0,MAXROWS);int(d.persistence_count,0,MAXROWS)
        need(is_sequence(d.model_ids)&&is_sequence(d.selected_setting_ids)&&length(d.model_ids)==length(d.selected_setting_ids)==2&&all(is_digest,d.model_ids),"audit selected model shape differs")
        for x in d.selected_setting_ids;int(x,1,8);end
        member_headers(d.members,draft_names())
    end
    nothing
end
function qualification(ref,a,head,proofsha,pins)
    qref=fileref(ref);body=P._bound(qref.path);need(sha(body)==qref.sha256,"audit qualification changed before parse");pin!(pins,qref.path;limit=META,expected=qref.sha256)
    q=P._software_qualification(body;schema="b_reference_component_software_qualification_v1",evidence_mode=a.mode=="synthetic_only" ? "synthetic_only" : "software_qualified",head,sources=collect(source_map(a.sources)))
    need(q.source_proof_sha256==proofsha,"audit component qualification whole-source proof differs")
    qref,q
end
function historical_permit(ref,a,c,qref,proofref,contracts,runtime,pins)
    if a.mode=="synthetic_only"
        need(ref===nothing&&c.execution_authority_sha256===nothing,"audit generated component cannot carry actual authority")
        return nothing
    end
    pref=fileref(ref);need(pref.sha256==c.execution_authority_sha256,"audit historical permit SHA differs")
    p,_=readjson(pref.path;expected=pref.sha256,names=PERMIT);check_identity(p,:permit_id);pin!(pins,pref.path;limit=META,expected=pref.sha256)
    need(p.schema=="qualified_actual_initial_matched342_permit_v1"&&p.complete===true&&p.evidence_mode==a.mode&&p.operation=="initial_matched342_32fits"&&p.qualified_head==c.baseline_qualified_head&&same(p.implementation_sources,a.sources),"audit historical permit completed code/scope differs")
    check_bindings(p.sources,a.bindings);need(same(p.runtime,runtime),"audit historical fitted runtime differs")
    names=Tuple(k for k in CONTRACTS if k!=:audit_contract_sha256);fields(p.contracts,names)
    for k in names;need(getproperty(p.contracts,k)==getproperty(contracts,k),"audit historical permit contract differs");end
    need(same(fileref(p.source_proof),proofref)&&same(fileref(p.component_qualification),qref),"audit historical source/component qualification differs")
    fields(p.scope,SCOPE);s=p.scope
    need(T._clock(s.training_boundary_ms)==a.boundary&&T._clock(s.validation_start_ms)==a.start&&T._clock(s.validation_end_ms)==a.stop&&int(s.embargo_hours,168,168)==168&&same(s.cell_order,[[0,4],[0,6],[1,4],[1,6]])&&same(s.setting_order,1:8)&&int(s.fit_count,32,32)==32&&int(s.encoder_count,4,4)==4&&int(s.final_fit_count,0,0)==0&&int(s.calibration_fit_count,0,0)==0,"audit historical initial-only fit scope differs")
    for k in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion);need(getproperty(s,k)===false,"audit historical permit expands scope");end
    fields(p.execution,(:directory,:native_owner_sha256,:preserved_native_owner_sha256,:intent))
    need(p.execution.directory isa AbstractString&&isabspath(p.execution.directory)&&p.execution.native_owner_sha256=="c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00"&&p.execution.preserved_native_owner_sha256=="30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c","audit historical fixed owner provenance differs")
    intentref=fileref(p.execution.intent);intent,_=readjson(intentref.path;expected=intentref.sha256,names=(:schema,:evidence_mode,:operation,:qualified_head,:implementation_sources,:contracts,:sources,:runtime,:execution_directory,:scope,:limits));pin!(pins,intentref.path;limit=META,expected=intentref.sha256)
    need(intent.schema=="matched342_initial_execution_intent_v1"&&intent.evidence_mode==p.evidence_mode&&intent.operation==p.operation&&intent.qualified_head==p.qualified_head&&intent.execution_directory==p.execution.directory,"audit historical execution intent differs")
    for k in (:implementation_sources,:contracts,:sources,:runtime,:scope,:limits);need(same(getproperty(intent,k),getproperty(p,k)),"audit historical execution intent repeated fact differs");end
    # These are retained outside evidence, never new fitting permission. Their
    # immutable bytes are reopened on every restart; no active clock is reset.
    for k in (:root_qualification,:resource_qualification)
        r=fileref(getproperty(p,k));pin!(pins,r.path;limit=META,expected=r.sha256)
    end
    fields(p.limits,(:source_rows,:model_member_bytes,:total_model_member_bytes,:required_available_bytes,:required_disk_bytes,:stage_wall_seconds,:fit_wall_seconds))
    need(int(p.limits.source_rows,MAXROWS,MAXROWS)==MAXROWS&&int(p.limits.model_member_bytes,PAYLOAD,PAYLOAD)==PAYLOAD&&int(p.limits.total_model_member_bytes,TOTAL,TOTAL)==TOTAL&&int(p.limits.required_available_bytes,8*1024^3,typemax(Int64))>=8*1024^3,"audit historical native limits differ")
    for k in (:required_disk_bytes,:stage_wall_seconds,:fit_wall_seconds);int(getproperty(p.limits,k),1,typemax(Int64));end
    pref
end
function candidate_values(path,n)
    need(filesize(path)==16+8n,"audit candidate full-union length differs")
    open(path,"r") do io
        P._check_header(io,"SSBRFPR1",8);values=Float64[P._get_float(io) for _ in 1:n]
        need(eof(io),"audit candidate trailing bytes");values
    end
end
function selected_values(path,n)
    need(filesize(path)==16+16n,"audit selected full-union length differs")
    open(path,"r") do io
        P._check_header(io,"SSBPPTS1",16);values=Matrix{Float64}(undef,n,2)
        for i in 1:n,k in 1:2;values[i,k]=P._get_float(io);end
        need(eof(io),"audit selected trailing bytes");values
    end
end
function metric(pred,labels)
    need(!isempty(pred)&&length(pred)==length(labels),"audit full per-lead metric rows differ")
    errors=pred.-labels;n=length(errors)
    result=(;n,rmse_nt=sqrt(sum(abs2,errors)/n),mae_nt=sum(abs,errors)/n,bias_nt=sum(errors)/n)
    need(all(isfinite,(result.rmse_nt,result.mae_nt,result.bias_nt)),"audit metric overflow");result
end
function metric_equal(x,y)
    need(int(x.n,1,MAXROWS)==y.n,"audit per-lead metric count differs")
    for k in (:rmse_nt,:mae_nt,:bias_nt)
        got=f64(getproperty(x,k));expected=getproperty(y,k)
        need(abs(got-expected)<=5e-12*max(1.,abs(expected)),"audit maintained metric parity differs")
    end
end
function increment_sha(values)
    ctx=SHA.SHA2_256_CTX();buffer=IOBuffer()
    for x in values;need(isfinite(x),"audit nonfinite saved increment");write(buffer,htol(reinterpret(UInt64,x)));position(buffer)>=65536&&SHA.update!(ctx,take!(buffer));end
    position(buffer)>0&&SHA.update!(ctx,take!(buffer));bytes2hex(SHA.digest!(ctx))
end
function raw_row_sha(a,L,rows)
    T._id([begin
        anchor=a.clean[15,i-L]
        target=ntuple(k->begin ti=i+(4,6)[k];ti<=a.n&&isfinite(a.clean[15,ti]) ? a.clean[15,ti] : nothing end,2)
        (;issue=shift(a.times[i],1),anchor=isfinite(anchor) ? anchor : nothing,target,source_sha256=a.bindings[L+1].source_sha256,delay_hours=L)
    end for i in rows])
end
function route_values(root,member,a,L,rows)
    m=filemember(root,member;name="routing/L$L.bin",limit=MAXROWS+16);need(m.bytes==16+length(rows),"audit routing length differs")
    route=open(P._member(root,m.name),"r") do io
        need(read(io,8)==collect(codeunits("SSBMRTE1"))&&ltoh(read(io,UInt64))==1,"audit routing header differs")
        r=read(io,length(rows));need(length(r)==length(rows)&&eof(io)&&all(x->x in (0x00,0x01),r),"audit routing enum/size differs");r
    end
    for (j,i) in enumerate(rows)
        required=all(isfinite,(a.clean[8,i],a.clean[6,i],a.clean[12,i]))&&a.clean[8,i]>=0.
        need(route[j]==(required ? 0x00 : 0x01),"audit common required-driver routing differs")
    end
    m,route
end

function recipe(s)
    merge(GRID[s],(;seed=22022026,eta=.05,min_weight=64.,lambda=1.,L2=1.,gamma=0.,rowsample=1.,colsample=1.))
end
function maxima(a,L,h,rows)
    need(!isempty(rows),"audit empty clock maxima")
    complete=maximum(shift(a.times[i],h+1) for i in rows)
    receipt=maximum(shift(a.times[i],h+1+L) for i in rows)
    (;complete,receipt)
end
function grid_and_tuning(root,draft,a,L,qsha,pins)
    grid,gb=readjson(P._member(root,"grid.json");expected=draft.grid_sha256,names=(:schema,:complete,:evidence_mode,:feature_contract_sha256,:settings,:grid_id));check_identity(grid,:grid_id)
    need(grid.schema=="fixed_eight_gbm_settings_v2"&&grid.complete===true&&grid.evidence_mode==a.mode&&grid.feature_contract_sha256==FEATURE&&is_sequence(grid.settings)&&length(grid.settings)==8,"audit exact eight-setting grid differs")
    for (s,g) in enumerate(grid.settings)
        fields(g,(:index,:max_depth,:nrounds,:nbins))
        need(int(g.index,1,8)==s&&all(int(getproperty(g,k),1,10000)==getproperty(GRID[s],k) for k in keys(GRID[s])),"audit fixed grid order differs")
    end
    tuning,tb=readjson(P._member(root,"tuning.json");expected=draft.tuning_sha256,names=TUNING);check_identity(tuning,:tuning_id)
    need(tuning.schema=="b_matched_reference_tuning_v2"&&tuning.grid_sha256==draft.grid_sha256&&same(tuning.lead_order,(4,6))&&is_sequence(tuning.leads)&&length(tuning.leads)==2,"audit B tuning header differs")
    check_common(tuning,a,L,qsha)
    for (k,h) in enumerate((4,6))
        lead=tuning.leads[k];fields(lead,LEAD);need(int(lead.horizon,4,6)==h&&is_sequence(lead.candidates)&&length(lead.candidates)==8&&is_digest(lead.training_mask_sha256)&&is_digest(lead.validation_cohort_sha256)&&is_digest(lead.selected_model_id),"audit B per-lead tuning header differs")
        int(lead.validation_count,1,MAXROWS);int(lead.selected_index,1,8)
        for f in (:max_fit_target_complete,:max_fit_label_receipt,:max_selection_target_complete,:max_selection_label_receipt);T._clock(getproperty(lead,f));end
        for (s,t) in enumerate(lead.candidates)
            fields(t,CANDIDATE)
            need(int(t.index,1,8)==s&&all(int(getproperty(t,k),1,10000)==getproperty(GRID[s],k) for k in keys(GRID[s]))&&is_digest(t.model_id)&&is_digest(t.fit_input_sha256)&&t.prediction_member=="candidates/h$h/setting-$s.bin"&&is_digest(t.prediction_sha256),"audit B candidate header differs")
            int(t.n,1,MAXROWS);int(t.prediction_bytes,17,16+8MAXROWS)
            for f in (:rmse_nt,:mae_nt,:bias_nt);f64(getproperty(t,f));end
        end
    end
    pin!(pins,P._member(root,"grid.json");limit=META,expected=draft.grid_sha256);pin!(pins,P._member(root,"tuning.json");limit=META,expected=draft.tuning_sha256)
    grid,tuning
end
function fit_receipt(root,record,cell,a,L,h,s,encoder,fitsha,mask,trainmask,valmask,increment,authoritysha,runtimesha,pins)
    member=filemember(root,record.fit_receipt_member;name="L$L-h$h/setting-$s.fit.json")
    path=P._member(root,member.name);f,_=readjson(path;expected=member.sha256,names=FITREC);check_identity(f,:receipt_id)
    need(f.schema=="matched342_initial_fit_receipt_v1"&&f.complete===true&&f.evidence_mode==a.mode&&f.pair_id==a.pair_id&&f.context_id==a.bindings[L+1].context_id&&int(f.delay_hours,0,1)==L&&int(f.horizon,4,6)==h&&int(f.setting_index,1,8)==s,"audit fit receipt header/cell differs")
    expected=(;model_training_mask_sha256=mask,reference_training_mask_sha256=trainmask,reference_validation_cohort_sha256=valmask,fit_input_sha256=fitsha,preprocessing_sha256=encoder.sha256,training_design_sha256=encoder.object.training_design_sha256,runtime_sha256=runtimesha,recipe=recipe(s),model_id=record.model_id,model_sha256=record.model_member.sha256,encoder_sha256=cell.encoder_member.sha256,native_increment_sha256=increment,reloaded_increment_sha256=increment,union_count=cell.union_count,execution_authority_sha256=authoritysha)
    for k in keys(expected);need(same(getproperty(f,k),getproperty(expected,k)),"audit fit receipt independently derived fact differs: $k");end
    int(f.fit_elapsed_ns,0,typemax(Int64));pin!(pins,path;limit=META,expected=member.sha256,expected_bytes=member.bytes)
    member
end
function audit_cell(root,draftroot,c,cell,lead,a,L,h,X,rows,route,selected,runtimesha,pins)
    cohort=cohorts(a,L,h);training=cohort.train;validation=cohort.validation
    need(int(cell.training_count,1,MAXROWS)==length(training)&&int(cell.validation_count,1,MAXROWS)==length(validation)&&int(cell.union_count,1,MAXROWS)==length(rows),"audit independent full per-lead counts differ")
    positions=Dict(i=>k for (k,i) in enumerate(rows));vp=Int[positions[i] for i in validation]
    fallback=count(==(0x01),route[vp]);need(cell.validation_persistence_count==fallback,"audit per-lead persistence count differs")
    trainmask=refmask(a,L,h,training,:training);valmask=refmask(a,L,h,validation,:validation);mask=modelmask(a,L,h,training)
    encoder=independently_encoded(X,training);em=filemember(root,cell.encoder_member;name="L$L-h$h/encoder.json",limit=PAYLOAD)
    saved,_=readjson(P._member(root,em.name);expected=em.sha256,names=ENCODERKEYS,limit=PAYLOAD)
    need(same(saved,encoder.object)&&em.sha256==encoder.sha256,"audit independent encoder rows/means/sample SD/bits differ")
    pin!(pins,P._member(root,em.name);limit=PAYLOAD,expected=em.sha256,expected_bytes=em.bytes)
    # Replay uses the verified STORED bits, rather than replacing them by newly
    # rounded moments. Equality above includes every index and all ten fields.
    replay_encoder=(;impute=Tuple(f64.(saved.imputation_f64_bits)),center=Tuple(f64.(saved.center_f64_bits)),scale=Tuple(f64.(saved.scale_f64_bits)))
    fitsha=fitinput(a,L,h,training,X,replay_encoder_with_identity(replay_encoder,encoder),mask)
    fitmax=maxima(a,L,h,training);valmax=maxima(a,L,h,validation)
    expected=(;model_training_mask_sha256=mask,reference_training_mask_sha256=trainmask,reference_validation_cohort_sha256=valmask,fit_input_sha256=fitsha,preprocessing_sha256=encoder.sha256,training_design_sha256=encoder.object.training_design_sha256,max_fit_target_complete=clock(fitmax.complete),max_fit_label_receipt=clock(fitmax.receipt),max_validation_target_complete=clock(valmax.complete),max_validation_label_receipt=clock(valmax.receipt))
    for k in keys(expected);need(same(getproperty(cell,k),getproperty(expected,k)),"audit independent cell fact differs: $k");end
    need(lead.training_mask_sha256==trainmask&&lead.validation_cohort_sha256==valmask&&lead.validation_count==length(validation),"audit B full per-lead masks/count differ")
    for (k,v) in ((:max_fit_target_complete,fitmax.complete),(:max_fit_label_receipt,fitmax.receipt),(:max_selection_target_complete,valmax.complete),(:max_selection_label_receipt,valmax.receipt));need(T._clock(getproperty(lead,k))==v,"audit B tuning target clocks differ");end
    Z=Matrix{Float64}(undef,length(rows),WIDTH)
    for (k,i) in enumerate(rows);normalize_row!(@view(Z[k,:]),@view(X[i,:]),replay_encoder);end
    labels=Float64[a.clean[15,i+h] for i in validation];anchors=Float64[a.clean[15,i-L] for i in rows]
    checks=NamedTuple[];stored_best=0;computed_best=0;stored_min=Inf;computed_min=Inf
    for s in 1:8
        record=cell.models[s];item=lead.candidates[s]
        mm=filemember(root,record.model_member;name="L$L-h$h/setting-$s.model.json",limit=PAYLOAD)
        expected_model=(;cell_delay=L,cell_horizon=h,setting_index=s,training_mask_sha256=mask,preprocessing_sha256=encoder.sha256,input_admission_sha256=a.bindings[L+1].context_id,feature_contract_sha256=FEATURE,loss_contract_sha256=LOSS)
        model,modelid=decode_model(P._member(root,mm.name),mm.sha256,expected_model,s)
        need(modelid==record.model_id==item.model_id,"audit C/B/model scalar identity differs")
        increments=Replay.predict_deltas(model,Z);increment=increment_sha(increments)
        need(increment==record.native_increment_sha256==record.reloaded_increment_sha256,"audit native/reloaded/independent scalar increment SHA differs")
        absolute=Vector{Float64}(undef,length(rows))
        for j in eachindex(rows)
            absolute[j]=route[j]==0x01 ? anchors[j] : anchors[j]+increments[j]
            need(isfinite(absolute[j]),"audit absolute prediction overflow")
        end
        cm=filemember(draftroot,record.candidate_member;name="candidates/h$h/setting-$s.bin",limit=16+8MAXROWS)
        need(cm.bytes==16+8length(rows)&&item.prediction_member==cm.name&&item.prediction_bytes==cm.bytes&&item.prediction_sha256==cm.sha256&&item.fit_input_sha256==fitsha,"audit B candidate/file/fit-input links differ")
        observed=candidate_values(P._member(draftroot,cm.name),length(rows))
        need(all(reinterpret(UInt64,x)==reinterpret(UInt64,y) for (x,y) in zip(observed,absolute)),"audit full candidate independent saved-state bits differ")
        m=metric(absolute[vp],labels);metric_equal(record,m);metric_equal(item,m)
        # Exact first minima are checked independently for retained metrics and
        # recomputed metrics. The reporting parity tolerance is never a tie rule.
        stored=f64(item.rmse_nt)
        if stored<stored_min;stored_min=stored;stored_best=s;end
        if m.rmse_nt<computed_min;computed_min=m.rmse_nt;computed_best=s;end
        if s==lead.selected_index
            need(lead.selected_model_id==modelid&&all(reinterpret(UInt64,absolute[j])==reinterpret(UInt64,selected[j]) for j in eachindex(rows)),"audit selected candidate column differs bitwise")
        end
        fm=fit_receipt(root,record,cell,a,L,h,s,encoder,fitsha,mask,trainmask,valmask,increment,c.execution_authority_sha256,runtimesha,pins)
        pin!(pins,P._member(root,mm.name);limit=PAYLOAD,expected=mm.sha256,expected_bytes=mm.bytes);pin!(pins,P._member(draftroot,cm.name);limit=16+8MAXROWS,expected=cm.sha256,expected_bytes=cm.bytes)
        check=(;setting_index=s,model_id=modelid,model_sha256=mm.sha256,fit_receipt_sha256=fm.sha256,native_increment_sha256=increment,reloaded_increment_sha256=increment,audited_increment_sha256=increment,candidate_prediction_sha256=cm.sha256,n=m.n,rmse_nt=bits(m.rmse_nt),mae_nt=bits(m.mae_nt),bias_nt=bits(m.bias_nt))
        fields(check,MODELCHECK);push!(checks,check)
    end
    need(stored_best==computed_best==lead.selected_index,"audit exact first selected minimum differs")
    check=(;delay_hours=L,horizon=h,context_id=a.bindings[L+1].context_id,training_count=length(training),validation_count=length(validation),union_count=length(rows),validation_persistence_count=fallback,model_training_mask_sha256=mask,reference_training_mask_sha256=trainmask,reference_validation_cohort_sha256=valmask,training_design_sha256=encoder.object.training_design_sha256,preprocessing_sha256=encoder.sha256,fit_input_sha256=fitsha,encoder_sha256=em.sha256,max_fit_target_complete=clock(fitmax.complete),max_fit_label_receipt=clock(fitmax.receipt),max_selection_target_complete=clock(valmax.complete),max_selection_label_receipt=clock(valmax.receipt),models=Tuple(checks))
    fields(check,CELLCHECK);check
end
replay_encoder_with_identity(e,derived)=merge(e,(;sha256=derived.sha256,object=derived.object))

function calculate(pair,cref,qref,proofref,contracts,runtime,authorityref,drafts,auditroot,expected_sources,auditsha;allow_published=false)
    a=independent_pair(pair,expected_sources);pins=NamedTuple[]
    cref=fileref(cref);need(basename(cref.path)=="completion.json"&&basename(dirname(cref.path))=="component","audit fixed component completion path differs")
    root=dirname(cref.path);c,_=readjson(cref.path;expected=cref.sha256,names=COMPONENT);component_headers(c,a)
    contracts=check_contracts(contracts,auditsha);runtime=check_runtime(runtime);need(same(c.runtime,runtime),"audit external expected fitted runtime differs")
    proofref=fileref(proofref);pin!(pins,proofref.path;limit=META,expected=proofref.sha256)
    FC.recheck_source_proof(proofref;expected_head=String(c.baseline_qualified_head),expected_sources=source_map(a.sources))
    qref,q=qualification(qref,a,c.baseline_qualified_head,proofref.sha256,pins)
    need(c.baseline_qualification_sha256==qref.sha256,"audit component qualification external SHA differs")
    authorityref=historical_permit(authorityref,a,c,qref,proofref,contracts,runtime,pins)
    need(is_sequence(drafts)&&length(drafts)==2,"audit two fixed draft requests required")
    cnames=component_names();dirs=vcat(["L$L-h$h" for (L,h) in CELLS],["routing"])
    exact_tree(root,vcat(cnames,["completion.json"]),dirs)
    members=fixed_members(root,c.members,cnames;payload=true)
    for m in members;pin!(pins,P._member(root,m.name);limit=m.bytes,expected=m.sha256,expected_bytes=m.bytes);end
    pin!(pins,cref.path;limit=META,expected=cref.sha256)
    need(P._bound(P._member(root,"qualification.json"))==P._bound(qref.path)&&sha(P._bound(P._member(root,"feature-contract")))==FEATURE&&sha(P._bound(P._member(root,"loss-contract")))==LOSS,"audit exact qualification/contract copies differ")
    checks=NamedTuple[];references=NamedTuple[];ras=NamedTuple[]
    runtimesha=T._id((;schema="matched342_native_runtime_v1",runtime))
    for L in 0:1
        request=drafts[L+1];fields(request,(:delay_hours,:root,:members));d=c.drafts[L+1]
        draftroot=P._ordinary(String(request.root));need(int(request.delay_hours,0,1)==L&&isabspath(request.root)&&draftroot==request.root&&draftroot==joinpath(dirname(root),"reference-L$L"),"audit externally pinned fixed sibling draft root differs")
        member_headers(request.members,draft_names());need(same(request.members,d.members),"audit request/C draft inventory differs")
        exact_tree(draftroot,draft_names(),("candidates","candidates/h4","candidates/h6");optional=allow_published ? ("audit.json","completion.json") : ())
        dm=fixed_members(draftroot,d.members,draft_names())
        for m in dm;pin!(pins,P._member(draftroot,m.name);limit=m.bytes,expected=m.sha256,expected_bytes=m.bytes);end
        need(P._bound(P._member(draftroot,"qualification.json"))==P._bound(qref.path),"audit B22 qualification copy differs")
        cs=(cohorts(a,L,4),cohorts(a,L,6));rows=sort!(union(cs[1].validation,cs[2].validation))
        need(d.model_count+d.persistence_count==length(rows)&&raw_row_sha(a,L,rows)==d.raw_row_identity_sha256,"audit full union/raw row identity differs")
        routing,route=route_values(root,d.routing_member,a,L,rows)
        need(count(==(0x00),route)==d.model_count&&count(==(0x01),route)==d.persistence_count,"audit saved union routing counts differ")
        grid,tuning=grid_and_tuning(draftroot,d,a,L,qref.sha256,pins)
        selectedpath=P._member(draftroot,"predictions.bin");need(P._file_sha(selectedpath)[1]==d.prediction_panel_sha256,"audit selected panel SHA differs")
        selected=selected_values(selectedpath,length(rows));X=design342(a.clean,L)
        for (k,h) in enumerate((4,6))
            cell=c.cells[2L+k];lead=tuning.leads[k]
            need(d.model_ids[k]==lead.selected_model_id&&d.selected_setting_ids[k]==lead.selected_index,"audit component/B selected IDs differ")
            push!(checks,audit_cell(root,draftroot,c,cell,lead,a,L,h,X,rows,route,@view(selected[:,k]),runtimesha,pins))
        end
        X=nothing
        common=check_common(tuning,a,L,qref.sha256)
        ra=withid((;schema="b_matched_reference_audit_v2",common...,audit_contract_sha256=auditsha,raw_row_identity_sha256=String(d.raw_row_identity_sha256),prediction_panel_sha256=String(d.prediction_panel_sha256),grid_sha256=String(d.grid_sha256),tuning_sha256=String(d.tuning_sha256),all_eight_settings_checked=true,full_per_lead_cohorts_checked=true,feature_receipt_history_checked=true,selected_prediction_columns_checked=true,initial_only=true),:audit_id)
        need(length(T._bytes(ra))<=META,"audit reference receipt size bound");push!(ras,ra)
        ref=(;delay_hours=L,source_binding=a.bindings[L+1],draft_root=draftroot,draft_members=dm,routing_member=routing,model_count=Int(d.model_count),persistence_count=Int(d.persistence_count),audit_ref=(;path=joinpath(auditroot,"reference-audit-L$L.json"),sha256=sha(T._bytes(ra))),training_history_start=clock(first(a.times)),training_boundary=clock(a.boundary),validation_start=clock(a.start),validation_end=clock(a.stop),raw_row_identity_sha256=String(d.raw_row_identity_sha256),prediction_panel_sha256=String(d.prediction_panel_sha256),grid_sha256=String(d.grid_sha256),tuning_sha256=String(d.tuning_sha256),model_ids=Tuple(String.(d.model_ids)),selected_setting_ids=Tuple(Int.(d.selected_setting_ids)))
        fields(ref,REFERENCE);push!(references,ref)
    end
    report=withid((;schema="matched342_initial_component_audit_report_v1",complete=true,evidence_mode=a.mode,pair_id=a.pair_id,component_completion_sha256=cref.sha256,source_bindings=a.bindings,cells=Tuple(checks),references=Tuple(references),fitting_performed=false,native_model_deserialization=false),:report_id)
    fields(report,REPORT);need(length(T._bytes(report))<=META,"audit report size bound")
    recheck_pins(pins);FC.recheck_pair(pair);current_sources(expected_sources)
    FC.recheck_source_proof(proofref;expected_head=String(c.baseline_qualified_head),expected_sources=source_map(a.sources))
    (;a,c,cref,qref,proofref,authorityref,contracts,runtime,references=Tuple(references),ras=Tuple(ras),report,pins)
end

function audit_root(path;fresh=false)
    need(path isa AbstractString&&isabspath(path)&&abspath(path)==path,"audit canonical absolute output root required")
    full=P._ordinary(String(path));need(isdir(dirname(full)),"audit output parent must already be ordinary")
    if fresh;need(!ispath(full)&&!islink(full),"audit output must be fresh")
    else;need(isdir(full),"audit completed output directory required")
    end
    full
end
function check_output_disjoint(root,a,component,qref,proofref,authorityref,requestpath=nothing)
    controls=String[Parent.PACKAGE_ROOT,dirname(dirname(component.path)),qref.path,proofref.path]
    authorityref===nothing||push!(controls,authorityref.path)
    requestpath===nothing||push!(controls,requestpath)
    for s in a.admissions
        push!(controls,s.source_proof_path)
        append!(controls,values(s.roots))
        append!(controls,String[m.absolute for m in s.members])
    end
    append!(controls,[joinpath(dirname(dirname(component.path)),"reference-L$L") for L in 0:1])
    need(all(p->disjoint(root,p),controls),"audit output overlaps source, controls, checkout, component or reference roots")
    nothing
end
function ca_header(ca,pair,expected_sources,expected_component_sha256,auditsha)
    fields(ca,CA);check_identity(ca,:audit_id)
    need(ca.schema=="matched342_initial_component_audit_completion_v1"&&ca.complete===true&&ca.operation==OP&&ca.evidence_mode in ("synthetic_only","archival_assumed_completion_research"),"audit completed receipt header differs")
    need(is_digest(expected_component_sha256)&&ca.component_completion_sha256==expected_component_sha256&&is_digest(ca.pair_id)&&is_digest(ca.source_proof_sha256)&&is_digest(ca.report_sha256)&&ca.qualified_head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",ca.qualified_head),"audit outside completion/code/proof pins differ")
    need(same(ca.implementation_sources,current_sources(expected_sources)),"audit receipt source closure differs")
    FC.recheck_pair(pair);metadata=FC.source_pair_metadata(pair)
    need(ca.evidence_mode==String(metadata.evidence_mode)&&ca.pair_id==metadata.pair_id,"audit completed nominal source pair differs")
    check_bindings(ca.source_bindings,metadata.source_bindings);check_contracts(ca.contracts,auditsha);check_runtime(ca.runtime)
    proof=fileref(ca.source_proof);need(proof.sha256==ca.source_proof_sha256,"audit restart whole-proof reference differs")
    fileref(ca.component_qualification)
    if ca.evidence_mode=="synthetic_only";need(ca.execution_authority===nothing&&ca.execution_authority_sha256===nothing,"audit generated receipt carries actual authority")
    else;authority=fileref(ca.execution_authority);need(authority.sha256==ca.execution_authority_sha256,"audit restart historical permit reference differs")
    end
    need(ca.component_root isa AbstractString&&isabspath(ca.component_root)&&P._ordinary(ca.component_root)==ca.component_root&&basename(ca.component_root)=="component","audit completed component root differs")
    need(is_sequence(ca.references)&&length(ca.references)==2,"audit completed two-delay reference projection required")
    for (k,r) in enumerate(ca.references)
        fields(r,REFERENCE);L=k-1;need(int(r.delay_hours,0,1)==L&&same(r.source_binding,ca.source_bindings[k]),"audit completed ordered reference source differs")
        need(r.draft_root isa AbstractString&&isabspath(r.draft_root)&&P._ordinary(r.draft_root)==r.draft_root&&r.draft_root==joinpath(dirname(ca.component_root),"reference-L$L"),"audit completed fixed reference root differs")
        member_headers(r.draft_members,draft_names());fields(r.routing_member,(:name,:bytes,:sha256))
        need(r.routing_member.name=="routing/L$L.bin"&&is_digest(r.routing_member.sha256),"audit completed route member differs");int(r.routing_member.bytes,17,MAXROWS+16)
        int(r.model_count,0,MAXROWS);int(r.persistence_count,0,MAXROWS);fields(r.audit_ref,(:path,:sha256));need(is_digest(r.audit_ref.sha256)&&r.audit_ref.path isa AbstractString&&isabspath(r.audit_ref.path),"audit completed reference receipt pin differs")
        for f in (:training_history_start,:training_boundary,:validation_start,:validation_end);T._clock(getproperty(r,f));end
        for f in (:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256);need(is_digest(getproperty(r,f)),"audit completed reference digest differs");end
        need(is_sequence(r.model_ids)&&is_sequence(r.selected_setting_ids)&&length(r.model_ids)==length(r.selected_setting_ids)==2&&all(is_digest,r.model_ids),"audit completed selected model identities differ")
        for x in r.selected_setting_ids;int(x,1,8);end
    end
    member_headers(ca.members,("audit-report.json","reference-audit-L0.json","reference-audit-L1.json"))
    nothing
end
function publication_state(pair,metadata)
    admissions=FC.source_admissions(pair)
    for ref in metadata.references
        root=ref.draft_root
        exact_tree(root,draft_names(),("candidates","candidates/h4","candidates/h6");optional=("audit.json","completion.json"))
        if isfile(joinpath(root,"audit.json"))
            need(P._file_sha(joinpath(root,"audit.json"))[1]==ref.audit_ref.sha256,"audit later reference audit copy differs")
        end
        if isfile(joinpath(root,"completion.json"))
            need(isfile(joinpath(root,"audit.json")),"audit later completion lacks copied reference audit")
            cp=joinpath(root,"completion.json");ch=P._file_sha(cp)[1]
            pins=BI.BReferencePins(;completion_sha256=ch,audit_sha256=ref.audit_ref.sha256,grid_sha256=ref.grid_sha256,tuning_sha256=ref.tuning_sha256,prediction_sha256=ref.prediction_panel_sha256,feature_contract_sha256=FEATURE,tuning_loss_contract_sha256=LOSS,baseline_qualification_sha256=metadata.component_qualification.sha256,baseline_sources=source_map(metadata.implementation_sources),baseline_qualified_head=metadata.qualified_head,feature_contract_member=Parent.FEATURE_CONTRACT_MEMBER,tuning_loss_contract_member=Parent.LOSS_CONTRACT_MEMBER)
            BI.admit_reference(cp;trusted_pins=pins,roots=Dict("reference"=>root),view=admissions[ref.delay_hours+1].view,expected_sources=P.source_identity())
        end
    end
    nothing
end
struct _AuditSeal342 end
"Immutable source/file-pinned audit result. No public unsealed constructor."
struct CheckedComponentAudit342
    pair::Any
    completion_path::String
    completion_sha256::String
    expected_sources::Tuple{Vararg{Pair{String,String}}}
    expected_audit_contract_sha256::String
    pins::Tuple{Vararg{NamedTuple}}
    metadata::NamedTuple
    seal_sha256::String
    function CheckedComponentAudit342(pair,path,digest,sources,auditsha,pins,metadata,seal,::_AuditSeal342)
        new(pair,path,digest,sources,auditsha,pins,metadata,seal)
    end
end
function handle_data(pair,path,digest,sources,auditsha,pins,metadata)
    (;pair_id=FC.source_pair_metadata(pair).pair_id,completion_path=path,completion_sha256=digest,expected_sources=source_records(sources),expected_audit_contract_sha256=auditsha,pins,metadata)
end
function make_handle(pair,path,digest,expected_sources,auditsha,pins,metadata)
    sources=source_map(expected_sources);retained=Tuple(sort!(collect(pins);by=x->x.path));projection=freeze(metadata)
    seal=T._id(handle_data(pair,path,digest,sources,auditsha,retained,projection))
    CheckedComponentAudit342(pair,String(path),String(digest),sources,String(auditsha),retained,projection,seal,_AuditSeal342())
end
function publisher_metadata(result,root,digest)
    a=result.a
    (;schema="checked_matched342_component_audit_v1",evidence_mode=a.mode,pair_id=a.pair_id,qualified_head=String(result.c.baseline_qualified_head),implementation_sources=a.sources,source_proof_sha256=result.proofref.sha256,source_bindings=a.bindings,contracts=result.contracts,runtime=result.runtime,component_root=dirname(result.cref.path),component_completion_sha256=result.cref.sha256,component_qualification=result.qref,audit_root=root,audit_completion_sha256=String(digest),references=result.references)
end
"Recheck all source/qualification/control/member pins; never restart fitting."
function recheck_component_audit(h::CheckedComponentAudit342)
    need(h.seal_sha256==T._id(handle_data(h.pair,h.completion_path,h.completion_sha256,h.expected_sources,h.expected_audit_contract_sha256,h.pins,h.metadata)),"audit sealed handle differs")
    FC.recheck_pair(h.pair);current_sources(h.expected_sources);recheck_pins(h.pins)
    ca,_=readjson(h.completion_path;expected=h.completion_sha256,names=CA)
    ca_header(ca,h.pair,h.expected_sources,h.metadata.component_completion_sha256,h.expected_audit_contract_sha256)
    FC.recheck_source_proof(fileref(ca.source_proof);expected_head=String(ca.qualified_head),expected_sources=h.expected_sources)
    exact_tree(dirname(h.completion_path),AUDIT_NAMES,();cap=4META)
    exact_tree(h.metadata.component_root,vcat(component_names(),["completion.json"]),vcat(["L$L-h$h" for (L,h) in CELLS],["routing"]))
    publication_state(h.pair,h.metadata);recheck_pins(h.pins);nothing
end
"Return only the frozen publisher projection after source/file rechecks."
function component_audit_metadata(h::CheckedComponentAudit342)
    recheck_component_audit(h);h.metadata
end
"Reconstruct a pinned completed audit in a new process, using data only."
function read_component_audit(pair,completion_path,completion_sha256;expected_component_sha256,expected_sources,expected_audit_contract_sha256)
    need(is_digest(expected_audit_contract_sha256)&&is_digest(completion_sha256),"audit externally supplied completion/contract pins required")
    path=P._ordinary(String(completion_path));need(isabspath(completion_path)&&path==completion_path&&basename(path)=="audit-completion.json","audit fixed completed receipt path differs")
    root=audit_root(dirname(path));ca,_=readjson(path;expected=completion_sha256,names=CA)
    ca_header(ca,pair,expected_sources,expected_component_sha256,expected_audit_contract_sha256)
    exact_tree(root,AUDIT_NAMES,();cap=4META)
    members=fixed_members(root,ca.members,("audit-report.json","reference-audit-L0.json","reference-audit-L1.json"))
    drafts=Tuple((;delay_hours=r.delay_hours,root=r.draft_root,members=r.draft_members) for r in ca.references)
    result=calculate(pair,(;path=joinpath(ca.component_root,"completion.json"),sha256=String(ca.component_completion_sha256)),ca.component_qualification,ca.source_proof,ca.contracts,ca.runtime,ca.execution_authority,drafts,root,expected_sources,expected_audit_contract_sha256;allow_published=true)
    a=result.a;check_output_disjoint(root,a,result.cref,result.qref,result.proofref,result.authorityref)
    need(ca.qualified_head==result.c.baseline_qualified_head&&ca.source_proof_sha256==result.proofref.sha256&&same(ca.source_bindings,a.bindings)&&same(ca.references,result.references)&&same(ca.runtime,result.runtime)&&same(ca.contracts,result.contracts),"audit completed independently reconstructed facts differ")
    report,rb=readjson(joinpath(root,"audit-report.json");expected=ca.report_sha256,names=REPORT);check_identity(report,:report_id)
    need(rb==T._bytes(result.report),"audit report does not match independent reconstruction")
    for L in 0:1
        r=result.references[L+1];need(r.audit_ref.path==joinpath(root,"reference-audit-L$L.json"),"audit RA is outside fixed audit store")
        ra,body=readjson(r.audit_ref.path;expected=r.audit_ref.sha256,names=BI.REFERENCE_AUDIT_FIELDS);check_identity(ra,:audit_id)
        need(body==T._bytes(result.ras[L+1]),"audit unchanged B reference receipt differs independently")
    end
    need(only(filter(m->m.name=="audit-report.json",members)).sha256==ca.report_sha256,"audit report ledger pin differs")
    for m in members;pin!(result.pins,joinpath(root,m.name);limit=META,expected=m.sha256,expected_bytes=m.bytes);end
    pin!(result.pins,path;limit=META,expected=completion_sha256)
    metadata=publisher_metadata(result,root,completion_sha256)
    h=make_handle(pair,path,completion_sha256,expected_sources,expected_audit_contract_sha256,result.pins,metadata)
    recheck_component_audit(h);h
end
"Audit all32 saved states and4 encoders and write completion exclusively last."
function audit_component(pair,request_path;request_sha256,expected_sources,expected_audit_contract_sha256)
    need(is_digest(request_sha256)&&is_digest(expected_audit_contract_sha256),"audit outside request/contract pins required")
    requestpath=P._ordinary(String(request_path));request,requestbody=readjson(requestpath;expected=request_sha256,names=REQUEST);check_identity(request,:request_id)
    need(request.schema=="matched342_initial_component_audit_request_v1"&&request.operation==OP&&request.evidence_mode in ("synthetic_only","archival_assumed_completion_research"),"audit closed request header differs")
    contracts=check_contracts(request.contracts,expected_audit_contract_sha256);runtime=check_runtime(request.runtime)
    metadata=FC.source_pair_metadata(pair);need(request.evidence_mode==String(metadata.evidence_mode),"audit request nominal source mode differs")
    check_bindings(request.source_bindings,metadata.source_bindings)
    cref=fileref(request.component);qref=fileref(request.component_qualification);proofref=fileref(request.source_proof)
    authorityref=request.evidence_mode=="synthetic_only" ? (need(request.execution_authority===nothing,"audit generated request carries actual authority");nothing) : fileref(request.execution_authority)
    need(is_sequence(request.drafts)&&length(request.drafts)==2,"audit request two fixed drafts required")
    for (k,d) in enumerate(request.drafts)
        fields(d,(:delay_hours,:root,:members));need(int(d.delay_hours,0,1)==k-1&&d.root isa AbstractString&&isabspath(d.root),"audit request ordered draft root differs")
        member_headers(d.members,draft_names())
    end
    root=audit_root(String(request.output_directory);fresh=true)
    # Validate disjointness before large body loading and repeat with independently
    # reconstructed source records after the numerical audit.
    a=independent_pair(pair,expected_sources);check_output_disjoint(root,a,cref,qref,proofref,authorityref,requestpath)
    result=calculate(pair,cref,qref,proofref,contracts,runtime,authorityref,request.drafts,root,expected_sources,expected_audit_contract_sha256)
    check_output_disjoint(root,result.a,cref,qref,proofref,authorityref,requestpath)
    need(P._bound(requestpath)==requestbody,"audit request changed during independent audit")
    completionbody=nothing
    mkdir(root)
    try
        for L in 0:1;P._exclusive(joinpath(root,"reference-audit-L$L.json"),T._bytes(result.ras[L+1]));end
        reportbody=T._bytes(result.report);P._exclusive(joinpath(root,"audit-report.json"),reportbody)
        names=("audit-report.json","reference-audit-L0.json","reference-audit-L1.json")
        members=Tuple(begin p=joinpath(root,n);digest,bytes=P._file_sha(p);(;name=n,bytes,sha256=digest) end for n in names)
        ca=withid((;schema="matched342_initial_component_audit_completion_v1",complete=true,evidence_mode=result.a.mode,operation=OP,qualified_head=String(result.c.baseline_qualified_head),implementation_sources=result.a.sources,pair_id=result.a.pair_id,source_proof_sha256=result.proofref.sha256,source_proof=result.proofref,source_bindings=result.a.bindings,contracts=result.contracts,runtime=result.runtime,component_root=dirname(result.cref.path),component_completion_sha256=result.cref.sha256,component_qualification=result.qref,execution_authority_sha256=result.authorityref===nothing ? nothing : result.authorityref.sha256,execution_authority=result.authorityref,references=result.references,report_sha256=sha(reportbody),members),:audit_id)
        fields(ca,CA);completionbody=T._bytes(ca);need(length(completionbody)<=META,"audit completion metadata bound")
        recheck_pins(result.pins);FC.recheck_pair(pair);current_sources(expected_sources)
        need(P._bound(requestpath)==requestbody,"audit request changed before completion")
        P._exclusive(joinpath(root,"audit-completion.json"),completionbody)
        # Reopen and compare the exact completed bytes from this independent
        # numerical pass. A separate-process reader performs its own numerical
        # reconstruction; this writer does not repeat all32 replays needlessly.
        capath=joinpath(root,"audit-completion.json");digest=sha(completionbody)
        checked,body=readjson(capath;expected=digest,names=CA)
        need(body==completionbody,"audit completion changed after exclusive write")
        ca_header(checked,pair,expected_sources,cref.sha256,expected_audit_contract_sha256)
        for L in 0:1;need(P._bound(joinpath(root,"reference-audit-L$L.json"))==T._bytes(result.ras[L+1]),"audit reference receipt changed after write");end
        need(P._bound(joinpath(root,"audit-report.json"))==reportbody,"audit report changed after write")
        for member in members;pin!(result.pins,joinpath(root,member.name);limit=META,expected=member.sha256,expected_bytes=member.bytes);end
        pin!(result.pins,capath;limit=META,expected=digest,expected_bytes=length(completionbody))
        pin!(result.pins,requestpath;limit=META,expected=request_sha256,expected_bytes=length(requestbody))
        h=make_handle(pair,capath,digest,expected_sources,expected_audit_contract_sha256,result.pins,publisher_metadata(result,root,digest))
        recheck_component_audit(h)
        need(P._bound(requestpath)==requestbody,"audit request changed after completion")
        h
    catch
        completionbody===nothing||P._invalidate(root,"audit-completion.json",completionbody)
        rethrow()
    end
end

end # module
