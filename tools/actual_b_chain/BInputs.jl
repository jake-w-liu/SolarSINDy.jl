module BInputs
using SolarSINDy,Dates,SHA,JSON3
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const D=SolarSINDy.DurationConditionedForcing
const Selection=parentmodule(@__MODULE__).BSelection
const ACTUAL_SOURCE="593354e93e8f6c1486dc46de3ae96d4281d1129f9d5d39933347c2702122d248"
export BPrefixAdmission,admit_prefix,recheck
struct BPrefixAdmission
    envelope_path::String
    envelope_sha256::String
    roots::Dict{String,String}
    metadata::NamedTuple
    members::Vector{NamedTuple}
    view::T.PointInputView
    independent_admission_sha256::String
    admission_id::String
end
const PROOF_FIELDS=(:schema,:complete,:evidence_mode,:source_sha256,:raw_prefix_sha256,:prefix_start_ms,:prefix_stop_ms,:prefix_rows,:training_boundary_ms,:validation_start_ms,:validation_end_ms,:receipt_delay_hours,:reference_completion_sha256,:reference_audit_sha256,:audit_completion_sha256,:implementation_sources)
function _roots(roots)
    T._need(roots isa AbstractDict&&1<=length(roots)<=32,"B ordinary named roots required")
    result=Dict{String,String}()
    for (name,path) in roots
        key=String(name);T._need(occursin(r"\A[A-Za-z0-9_-]+\z",key),"B root name differs")
        full=P._ordinary(path);T._need(isdir(full),"B root must exist");result[key]=realpath(full)
    end
    canonical=sort!(collect(values(result)))
    folded=lowercase.(replace.(canonical,'\\'=>'/'))
    for i in eachindex(folded),j in eachindex(folded)
        i==j&&continue
        T._need(folded[i]!=folded[j]&&!startswith(folded[i],folded[j]*"/"),"B roots overlap or alias")
    end
    result
end
function _resolve(roots,root,path)
    T._need(haskey(roots,root),"B member root not supplied")
    full=P._member(roots[root],path);parent=roots[root]
    for part in split(path,'/')
        T._need(part in readdir(parent),"B member case/name alias rejected")
        parent=joinpath(parent,part)
    end
    full
end
function _admission_data(a::BPrefixAdmission)
    if hasproperty(a.metadata,:input_contract_version)
        T._need(a.metadata.input_contract_version===2,"B input contract version differs")
        return (;schema="b_prefix_admission_v2",envelope_sha256=a.envelope_sha256,metadata=a.metadata,members=[(;root=x.root,path=x.path,bytes=x.bytes,sha256=x.sha256,role=x.role) for x in a.members],input_view_id=a.view.view_id,independent_admission_sha256=a.independent_admission_sha256)
    end
    (;schema="b_prefix_admission_v1",envelope_sha256=a.envelope_sha256,metadata=a.metadata,members=[(;root=x.root,path=x.path,bytes=x.bytes,sha256=x.sha256,role=x.role) for x in a.members],input_view_id=a.view.view_id,independent_admission_sha256=a.independent_admission_sha256)
end
function _proof(path,pin,envelope,mode)
    T._need(T._hex(pin)&&P._file_sha(path)[1]==pin,"B independent admission proof pin differs")
    d,_=P._json(path;fields=PROOF_FIELDS)
    T._need(d.schema=="independently_audited_b_prefix_v1"&&d.complete===true&&d.evidence_mode==String(mode),"new independent B admission proof required")
    T._int(d.prefix_rows,1,T.MAX_ROWS);T._int(d.receipt_delay_hours,0,1)
    for key in PROOF_FIELDS
        key in (:schema,:complete,:evidence_mode)||T._need(isequal(getproperty(d,key),getproperty(envelope,key)),"B independent admission/envelope field differs: $key")
    end
    d
end
function admit_prefix(envelope_path;expected_envelope_sha256,roots,expected_sources,independent_admission)
    sources=P._source(expected_sources);rootmap=_roots(roots);path=P._ordinary(envelope_path)
    d,body=P._json(path;fields=P.INPUT_FIELDS)
    T._need(T._hex(expected_envelope_sha256)&&T._sha(body)==expected_envelope_sha256&&d.schema=="b_point_input_envelope_v1"&&d.complete===true&&d.availability_kind=="assumed_completed_hour","B input envelope schema/completion differs")
    mode=Symbol(d.evidence_mode);T._need(mode in (:synthetic_only,:archival_assumed_completion_research),"B input evidence mode differs")
    n=T._int(d.prefix_rows,1,T.MAX_ROWS);T._need(T._int(d.channel_count,17,17)==17,"B source channel count differs")
    start=T._clock(d.prefix_start_ms);stop=T._clock(d.prefix_stop_ms);boundary=T._clock(d.training_boundary_ms);first=T._clock(d.validation_start_ms);last=T._clock(d.validation_end_ms);delay=T._int(d.receipt_delay_hours,0,1)
    T._need(floor(start,Hour)==start&&T.Adapter._shift(start,n)==stop&&start<boundary<=first<last<=stop,"B half-open prefix/split clocks differ")
    if mode==:archival_assumed_completion_research
        T._need(d.source_sha256==ACTUAL_SOURCE&&start==DateTime(1976,11,1)&&stop<=DateTime(2023)&&boundary==DateTime(2020)&&first==DateTime(2020)&&last==DateTime(2023),"B initial archival source/split differs")
    end
    T._need(P._sources(d.implementation_sources)==sources,"B input implementation source map differs")
    T._fields(d.limits,keys(P.LIMITS))
    for key in keys(P.LIMITS)
        value=T._int(getproperty(d.limits,key),1,getproperty(P.LIMITS,key))
        T._need(value==getproperty(P.LIMITS,key),"B fixed input limits differ")
    end
    T._need(n<=d.limits.source_rows&&136n<=d.limits.ordinary_file_bytes,"B raw prefix exceeds declared limits")
    T._need(d.reference_kind in ("original_reference_diagnostic_only","matched_information_gbm_initial_v1"),"B reference role differs")
    T._need(1<=length(d.members)<=2048,"B input member count differs")
    members=NamedTuple[];identities=Set{String}();roles=Set{String}()
    for item in d.members
        T._fields(item,(:root,:path,:bytes,:sha256,:role))
        key=String(item.root);name=String(item.path);role=String(item.role)
        T._need(ncodeunits(name)<=256&&role in ("raw_prefix","reference_completion","reference_audit","source_admission","independent_audit","raw_provenance"),"B input member grammar differs")
        identity=lowercase(key*"/"*name);T._need(!(identity in identities),"B input member aliases another member");push!(identities,identity);push!(roles,role)
        file=_resolve(rootmap,key,name);bytes=T._int(item.bytes,1,T.MAX_FILE);hash,observed=P._file_sha(file)
        T._need(hash==item.sha256&&observed==bytes,"B input member bytes differ")
        push!(members,(;root=key,path=name,bytes,sha256=hash,role,absolute=file))
    end
    T._need(roles==Set(("raw_prefix","reference_completion","reference_audit","source_admission","independent_audit","raw_provenance")),"B complete input/proof roles required")
    for (role,field) in (("reference_completion",:reference_completion_sha256),("reference_audit",:reference_audit_sha256),("source_admission",:independent_admission_completion_sha256),("independent_audit",:audit_completion_sha256))
        entries=filter(x->x.role==role,members)
        T._need(length(entries)==1&&only(entries).sha256==getproperty(d,field),"B input role/proof digest differs")
    end
    rawmembers=filter(x->x.role=="raw_prefix",members);T._need(length(rawmembers)==1,"B exactly one raw prefix required")
    raw=only(rawmembers)
    T._need(d.raw_prefix_member==raw.root*"/"*raw.path&&d.raw_prefix_bytes==raw.bytes==136n&&d.raw_prefix_sha256==raw.sha256,"B raw-prefix geometry/identity differs")
    T._need(independent_admission isa NamedTuple&&keys(independent_admission)==(:path,:sha256),"B independently pinned new admission required")
    T._need(independent_admission.sha256==d.independent_admission_completion_sha256,"B independent admission role/pin differs")
    _proof(independent_admission.path,independent_admission.sha256,d,mode)
    bytes=P._bound(raw.absolute,T.MAX_FILE);T._need(length(bytes)==136n,"B raw-prefix length changed")
    words=reinterpret(UInt64,bytes);values=Matrix{Float64}(undef,17,n)
    for i in eachindex(values)
        word=ltoh(words[i]);value=reinterpret(Float64,word)
        T._need(isfinite(value)||word==UInt64(0x7ff8000000000000),"B raw scalar must be finite or canonical missing")
        values[i]=value
    end
    times=[T.Adapter._shift(start,i) for i in 0:n-1]
    input=T.A.HourlyInputs(times,values;source_sha256=String(d.source_sha256))
    clean(channel,i)=begin x=values[channel,i];(!isfinite(x)||x>=T.Adapter.Features.FILL_LIMITS[channel]) ? NaN : x end
    wind=D.WindHour[]
    for i in 1:n
        bz=clean(6,i);speed=clean(8,i);q=isfinite(bz)&&isfinite(speed)&&speed>=0 ? max(speed*max(-bz,0.)/1000-.49,0.) : NaN
        push!(wind,D.WindHour(times[i],T.Adapter._shift(times[i],1),T.Adapter._shift(times[i],1),bz,q))
    end
    view=T._point_input_view(input,wind,(;evidence_mode=mode,input_admission_sha256=String(independent_admission.sha256),contract_sha256=T.CONTRACT))
    limits=NamedTuple{keys(P.LIMITS)}(Tuple(Int(getproperty(d.limits,k)) for k in keys(P.LIMITS)))
    meta=(;evidence_mode=mode,source_sha256=String(d.source_sha256),training_boundary=boundary,validation_start=first,validation_end=last,delay_hours=delay,reference_kind=Symbol(d.reference_kind),reference_completion_sha256=String(d.reference_completion_sha256),reference_audit_sha256=String(d.reference_audit_sha256),audit_completion_sha256=String(d.audit_completion_sha256),prefix_start=start,prefix_stop=stop,prefix_rows=n,limits)
    provisional=BPrefixAdmission(path,String(expected_envelope_sha256),rootmap,meta,members,view,String(independent_admission.sha256),repeat("0",64))
    admission=BPrefixAdmission(path,String(expected_envelope_sha256),rootmap,meta,members,view,String(independent_admission.sha256),T._id(_admission_data(provisional)))
    recheck(admission);admission
end
function recheck(a::BPrefixAdmission)
    T._need(T._id(_admission_data(a))==a.admission_id&&T._sha(P._bound(a.envelope_path))==a.envelope_sha256,"B input admission/envelope changed")
    hasproperty(a.metadata,:input_contract_version)&&_recheck_prefix_v2(a)
    for member in a.members
        path=_resolve(a.roots,member.root,member.path)
        T._need(path==member.absolute&&P._file_sha(path)==(member.sha256,member.bytes),"B admitted input member changed")
    end
    T._recheck(a.view);true
end

# Upstream source facts never depend on a future reference, join or envelope.
const SOURCE_V2_COMMON=(:complete,:evidence_mode,:availability_kind,:source_sha256,:raw_prefix_sha256,:prefix_start_ms,:prefix_stop_ms,:prefix_rows,:channel_count,:training_boundary_ms,:validation_start_ms,:validation_end_ms,:receipt_delay_hours,:implementation_sources)
const PROVENANCE_V2_FIELDS=(:schema,SOURCE_V2_COMMON...,:source_format_sha256,:parser_source_sha256,:raw_layout)
const SOURCE_AUDIT_V2_FIELDS=(:schema,SOURCE_V2_COMMON...,:raw_provenance_sha256)
const SOURCE_PROOF_V2_FIELDS=(:schema,SOURCE_V2_COMMON...,:raw_prefix_member,:raw_prefix_bytes,:raw_provenance_sha256,:source_audit_sha256,:limits,:members)
const JOIN_AUDIT_V2_FIELDS=(:schema,SOURCE_V2_COMMON...,:source_admission_sha256,:input_view_id,:input_prefix_sha256,:reference_kind,:reference_completion_sha256,:reference_audit_sha256,:source_audit_sha256,:raw_provenance_sha256)
const ENVELOPE_V2_FIELDS=(P.INPUT_FIELDS...,:reference_join_audit_sha256)
const SOURCE_V2_ROLES=("raw_prefix","raw_provenance","independent_audit")
const JOIN_V2_ROLES=("raw_prefix","raw_provenance","independent_audit","source_admission","reference_completion","reference_audit","reference_join_audit")
const SOURCE_FORMAT_V2="ab7cc12ea18d178ed3b529d603b867c9e948df419275a4cea56a8da08042feee"
const SOURCE_PARSER_V2="src/interval_mean_forecaster/sandu_fullchannel_features.jl"
const REFERENCE_PINS_V2_FIELDS=(:completion_sha256,:audit_sha256,:grid_sha256,:tuning_sha256,:prediction_sha256,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256,:baseline_sources,:baseline_qualified_head,:feature_contract_member,:tuning_loss_contract_member)
export BSourcePrefixAdmissionV2,admit_source_prefix_v2,recheck_source_v2,admit_prefix_v2

struct BSourcePrefixAdmissionV2
    source_proof_path::String
    source_proof_sha256::String
    roots::Dict{String,String}
    metadata::NamedTuple
    members::Vector{NamedTuple}
    view::T.PointInputView
    source_admission_id::String
end
function _source_common_v2(d,sources)
    T._need(d.complete===true&&d.availability_kind=="assumed_completed_hour","B v2 complete assumed source required")
    T._need(d.evidence_mode in ("synthetic_only","archival_assumed_completion_research")&&T._hex(d.source_sha256)&&T._hex(d.raw_prefix_sha256),"B v2 source mode/hash differs")
    n=T._int(d.prefix_rows,1,T.MAX_ROWS);T._int(d.channel_count,17,17);delay=T._int(d.receipt_delay_hours,0,1)
    start=T._clock(d.prefix_start_ms);stop=T._clock(d.prefix_stop_ms);boundary=T._clock(d.training_boundary_ms);first=T._clock(d.validation_start_ms);last=T._clock(d.validation_end_ms)
    T._need(floor(start,Hour)==start&&all(t->floor(t,Hour)==t,(stop,boundary,first,last))&&T.Adapter._shift(start,n)==stop&&start<boundary<=first<last<=stop,"B v2 contiguous source/split differs")
    if d.evidence_mode=="archival_assumed_completion_research"
        T._need(d.source_sha256==ACTUAL_SOURCE&&start==DateTime(1976,11,1)&&stop==DateTime(2023)&&boundary==DateTime(2020)&&first==DateTime(2020)&&last==DateTime(2023),"B v2 actual source/split differs")
    end
    T._need(P._sources(d.implementation_sources)==sources,"B v2 implementation map differs")
    (;evidence_mode=Symbol(d.evidence_mode),source_sha256=String(d.source_sha256),raw_prefix_sha256=String(d.raw_prefix_sha256),training_boundary=boundary,validation_start=first,validation_end=last,delay_hours=delay,prefix_start=start,prefix_stop=stop,prefix_rows=n)
end
function _common_equal_v2(d,base)
    for name in SOURCE_V2_COMMON
        T._need(T._bytes(getproperty(d,name))==T._bytes(getproperty(base,name)),"B v2 repeated source fact differs: $name")
    end
end
function _limits_v2(d)
    T._fields(d,keys(P.LIMITS))
    for name in keys(P.LIMITS);T._need(T._int(getproperty(d,name),1,getproperty(P.LIMITS,name))==getproperty(P.LIMITS,name),"B v2 fixed limit differs");end
    NamedTuple{keys(P.LIMITS)}(Tuple(Int(getproperty(d,k)) for k in keys(P.LIMITS)))
end
function _ordinary_input_v2(path)
    T._need(path isa AbstractString&&isabspath(path)&&all(x->x!="."&&x!="..",split(replace(path,'\\'=>'/'),'/')),"B v2 absolute nonalias path required")
    P._ordinary(path)
end
function _roots_v2(roots)
    T._need(roots isa AbstractDict&&1<=length(roots)<=32,"B v2 bounded named roots required")
    for path in values(roots);_ordinary_input_v2(path);end
    _roots(roots)
end
function _members_v2(rows,roots,roles)
    T._need(rows isa AbstractVector&&length(rows)==length(roles),"B v2 exact member count differs")
    members=NamedTuple[];seen=Set{String}();seen_roles=Set{String}()
    for item in rows
        T._fields(item,(:root,:path,:bytes,:sha256,:role))
        T._need(all(x->x isa AbstractString,(item.root,item.path,item.role))&&ncodeunits(item.path)<=256&&T._hex(item.sha256),"B v2 member grammar differs")
        key=String(item.root);name=String(item.path);role=String(item.role);identity=lowercase(key*"/"*name)
        T._need(role in roles&&!(role in seen_roles)&&!(identity in seen),"B v2 duplicate/unsupported member")
        push!(seen,identity);push!(seen_roles,role);path=_resolve(roots,key,name);bytes=T._int(item.bytes,1,T.MAX_FILE)
        T._need(isfile(path)&&filesize(path)==bytes,"B v2 member size/kind differs")
        role=="raw_prefix"||T._need(bytes<=T.MAX_META&&P._file_sha(path)==(String(item.sha256),bytes),"B v2 bounded metadata member differs")
        push!(members,(;root=key,path=name,bytes,sha256=String(item.sha256),role,absolute=path))
    end
    T._need(seen_roles==Set(roles),"B v2 complete member roles required");members
end
_role_v2(members,role)=only(filter(x->x.role==role,members))
function _source_metadata_v2(path;expected_source_proof_sha256,roots,expected_sources)
    sources=P._source(expected_sources);map=_roots_v2(roots);full=_ordinary_input_v2(path)
    d,body=P._json(full;fields=SOURCE_PROOF_V2_FIELDS)
    T._need(T._hex(expected_source_proof_sha256)&&T._sha(body)==expected_source_proof_sha256&&d.schema=="independently_audited_b_source_prefix_v2","B v2 source proof pin/schema differs")
    metadata=_source_common_v2(d,sources);limits=_limits_v2(d.limits);members=_members_v2(d.members,map,SOURCE_V2_ROLES)
    raw=_role_v2(members,"raw_prefix");provenance=_role_v2(members,"raw_provenance");audit=_role_v2(members,"independent_audit")
    T._need(d.raw_prefix_member==raw.root*"/"*raw.path&&T._int(d.raw_prefix_bytes,1,T.MAX_FILE)==raw.bytes==136metadata.prefix_rows&&d.raw_prefix_sha256==raw.sha256,"B v2 raw source geometry differs")
    T._need(d.raw_provenance_sha256==provenance.sha256&&d.source_audit_sha256==audit.sha256,"B v2 upstream source pins differ")
    p,pb=P._json(provenance.absolute;fields=PROVENANCE_V2_FIELDS);sa,sab=P._json(audit.absolute;fields=SOURCE_AUDIT_V2_FIELDS)
    T._need(T._sha(pb)==provenance.sha256&&T._sha(sab)==audit.sha256&&p.schema=="b_raw_prefix_provenance_v2"&&sa.schema=="independently_audited_b_raw_source_v2","B v2 upstream schema/pin differs")
    _common_equal_v2(p,d);_common_equal_v2(sa,d)
    T._need(sa.raw_provenance_sha256==provenance.sha256&&T._hex(p.source_format_sha256)&&T._hex(p.parser_source_sha256)&&p.parser_source_sha256==get(Dict(sources),SOURCE_PARSER_V2,nothing)&&p.raw_layout=="source17_le_f64_column_major_v1","B v2 parser/provenance facts differ")
    metadata.evidence_mode==:synthetic_only||T._need(p.source_format_sha256==SOURCE_FORMAT_V2,"B v2 actual source format differs")
    used=Set(x.root for x in members)
    proof_roots=String[];folded=lowercase(replace(full,'\\'=>'/'))
    for (key,root) in map
        folded_root=lowercase(replace(root,'\\'=>'/'))
        if startswith(folded,folded_root*"/")
            relative=replace(relpath(full,root),'\\'=>'/')
            T._need(_resolve(map,key,relative)==full,"B v2 source proof alias differs")
            push!(used,key);push!(proof_roots,key)
        end
    end
    T._need(!isempty(proof_roots),"B v2 source proof outside named source roots")
    selected=Dict(key=>map[key] for key in used)
    (;path=full,sha256=String(expected_source_proof_sha256),data=d,metadata=merge(metadata,(;limits)),members,roots=selected,provenance=p,source_audit=sa)
end
function _source_admission_data_v2(a::BSourcePrefixAdmissionV2)
    (;schema="b_source_prefix_admission_v2",source_proof_sha256=a.source_proof_sha256,metadata=a.metadata,members=[(;root=x.root,path=x.path,bytes=x.bytes,sha256=x.sha256,role=x.role) for x in a.members],input_view_id=a.view.view_id)
end
function admit_source_prefix_v2(path;expected_source_proof_sha256,roots,expected_sources)
    info=_source_metadata_v2(path;expected_source_proof_sha256,roots,expected_sources);m=info.metadata;raw=_role_v2(info.members,"raw_prefix")
    bytes=P._bound(raw.absolute,T.MAX_FILE);T._need(length(bytes)==raw.bytes&&T._sha(bytes)==raw.sha256,"B v2 raw source changed")
    words=reinterpret(UInt64,bytes);values=Matrix{Float64}(undef,17,m.prefix_rows)
    for i in eachindex(values)
        word=ltoh(words[i]);value=reinterpret(Float64,word)
        T._need(isfinite(value)||word==UInt64(0x7ff8000000000000),"B v2 raw scalar must be finite or canonical missing");values[i]=value
    end
    times=[T.Adapter._shift(m.prefix_start,i) for i in 0:m.prefix_rows-1];input=T.A.HourlyInputs(times,values;source_sha256=m.source_sha256)
    clean(channel,i)=begin x=values[channel,i];(!isfinite(x)||x>=T.Adapter.Features.FILL_LIMITS[channel]) ? NaN : x end
    wind=D.WindHour[]
    for i in 1:m.prefix_rows
        bz=clean(6,i);speed=clean(8,i);q=isfinite(bz)&&isfinite(speed)&&speed>=0 ? max(speed*max(-bz,0.)/1000-.49,0.) : NaN
        push!(wind,D.WindHour(times[i],T.Adapter._shift(times[i],1),T.Adapter._shift(times[i],1),bz,q))
    end
    view=T._point_input_view(input,wind,(;evidence_mode=m.evidence_mode,input_admission_sha256=info.sha256,contract_sha256=T.CONTRACT))
    provisional=BSourcePrefixAdmissionV2(info.path,info.sha256,info.roots,m,info.members,view,repeat("0",64))
    admission=BSourcePrefixAdmissionV2(info.path,info.sha256,info.roots,m,info.members,view,T._id(_source_admission_data_v2(provisional)))
    recheck_source_v2(admission);admission
end
function recheck_source_v2(a::BSourcePrefixAdmissionV2)
    T._need(T._id(_source_admission_data_v2(a))==a.source_admission_id&&a.view.input_admission_sha256==a.source_proof_sha256,"B v2 source admission mutated")
    info=_source_metadata_v2(a.source_proof_path;expected_source_proof_sha256=a.source_proof_sha256,roots=a.roots,expected_sources=P.source_identity())
    T._need(T._bytes(info.metadata)==T._bytes(a.metadata)&&info.members==a.members&&info.roots==a.roots,"B v2 source admission metadata changed")
    for member in a.members;T._need(P._file_sha(member.absolute)==(member.sha256,member.bytes),"B v2 source member changed");end
    T._recheck(a.view);true
end
function _joined_metadata_v2(path;expected_envelope_sha256,roots,expected_sources,independent_admission,expected_view_id=nothing,expected_input_prefix_sha256=nothing,reference_pins=nothing)
    sources=P._source(expected_sources);map=_roots_v2(roots);full=_ordinary_input_v2(path);d,body=P._json(full;fields=ENVELOPE_V2_FIELDS)
    T._need(T._hex(expected_envelope_sha256)&&T._sha(body)==expected_envelope_sha256&&d.schema=="b_point_input_envelope_v2","B v2 joined envelope pin/schema differs")
    metadata=_source_common_v2(d,sources);limits=_limits_v2(d.limits)
    T._need(independent_admission isa NamedTuple&&keys(independent_admission)==(:path,:sha256)&&T._hex(independent_admission.sha256)&&independent_admission.sha256==d.independent_admission_completion_sha256,"B v2 external source proof differs")
    source=_source_metadata_v2(independent_admission.path;expected_source_proof_sha256=independent_admission.sha256,roots=map,expected_sources=sources);_common_equal_v2(d,source.data)
    members=_members_v2(d.members,map,JOIN_V2_ROLES)
    for upstream in source.members
        joined=_role_v2(members,upstream.role)
        T._need(joined==upstream,"B v2 joined source member differs")
    end
    sp=_role_v2(members,"source_admission");jmember=_role_v2(members,"reference_join_audit");r=_role_v2(members,"reference_completion");ra=_role_v2(members,"reference_audit")
    T._need(sp.absolute==source.path&&sp.sha256==source.sha256&&d.raw_prefix_member==source.data.raw_prefix_member&&d.raw_prefix_bytes==source.data.raw_prefix_bytes,"B v2 joined source role differs")
    T._need(d.audit_completion_sha256==source.data.source_audit_sha256&&d.reference_join_audit_sha256==jmember.sha256&&d.reference_completion_sha256==r.sha256&&d.reference_audit_sha256==ra.sha256,"B v2 joined member pins differ")
    T._need(d.reference_kind in ("original_reference_diagnostic_only","matched_information_gbm_initial_v1")&&(d.reference_kind!="original_reference_diagnostic_only"||metadata.delay_hours==0),"B v2 reference role/delay differs")
    j,jb=P._json(jmember.absolute;fields=JOIN_AUDIT_V2_FIELDS);T._need(j.schema=="independently_audited_b_reference_join_v2"&&T._sha(jb)==jmember.sha256,"B v2 independent join audit differs");_common_equal_v2(j,source.data)
    for (field,value) in ((:source_admission_sha256,source.sha256),(:reference_kind,d.reference_kind),(:reference_completion_sha256,r.sha256),(:reference_audit_sha256,ra.sha256),(:source_audit_sha256,source.data.source_audit_sha256),(:raw_provenance_sha256,source.data.raw_provenance_sha256))
        T._need(getproperty(j,field)==value,"B v2 join audit upstream pin differs")
    end
    T._need(T._hex(j.input_view_id)&&T._hex(j.input_prefix_sha256),"B v2 join view/prefix grammar differs")
    expected_view_id===nothing||T._need(T._hex(expected_view_id)&&j.input_view_id==expected_view_id,"B v2 outside expected view differs")
    expected_input_prefix_sha256===nothing||T._need(T._hex(expected_input_prefix_sha256)&&j.input_prefix_sha256==expected_input_prefix_sha256,"B v2 outside expected prefix differs")
    info=(;path=full,sha256=String(expected_envelope_sha256),data=d,metadata=merge(metadata,(;limits)),source,members,roots=map,join=j)
    _reference_headers_v2(info;reference_pins);info
end
function _reference_pins_metadata_v2(ref)
    T._fields(ref,(:path,:sha256));T._need(T._hex(ref.sha256),"B v2 reference pins SHA grammar differs")
    d,body=P._json(_ordinary_input_v2(ref.path);fields=REFERENCE_PINS_V2_FIELDS)
    T._need(T._sha(body)==ref.sha256,"B v2 reference pins changed")
    for field in (:completion_sha256,:audit_sha256,:grid_sha256,:tuning_sha256,:prediction_sha256,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256)
        T._need(T._hex(getproperty(d,field)),"B v2 reference pin grammar differs")
    end
    T._need(d.baseline_qualified_head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",d.baseline_qualified_head),"B v2 baseline head grammar differs")
    sources=Dict(P._sources(d.baseline_sources))
    T._need(d.feature_contract_member isa AbstractString&&d.tuning_loss_contract_member isa AbstractString&&get(sources,d.feature_contract_member,nothing)==d.feature_contract_sha256&&get(sources,d.tuning_loss_contract_member,nothing)==d.tuning_loss_contract_sha256,"B v2 baseline contract members differ")
    d
end
function _reference_headers_v2(info;reference_pins=nothing)
    member=_role_v2(info.members,"reference_completion");audit_member=_role_v2(info.members,"reference_audit")
    T._need(basename(member.absolute)=="completion.json"&&audit_member.absolute==P._member(dirname(member.absolute),"audit.json"),"B v2 reference header paths differ")
    c,cb=P._json(member.absolute;fields=REFERENCE_COMPLETION_FIELDS);audit,ab=P._json(audit_member.absolute;fields=REFERENCE_AUDIT_FIELDS)
    T._need(T._sha(cb)==member.sha256&&T._sha(ab)==audit_member.sha256,"B v2 reference header changed")
    matched=info.data.reference_kind=="matched_information_gbm_initial_v1"
    T._need(c.schema==(matched ? "b_matched_reference_completion_v2" : "b_original_reference_completion_v1")&&audit.schema==(matched ? "b_matched_reference_audit_v2" : "b_original_reference_audit_v1")&&c.complete===true&&audit.complete===true,"B v2 reference header schema/completion differs")
    _reference_id(c,:completion_id);_reference_id(audit,:audit_id)
    for field in REFERENCE_COMMON;T._need(T._bytes(getproperty(c,field))==T._bytes(getproperty(audit,field)),"B v2 reference/audit common field differs");end
    for (field,value) in ((:evidence_mode,info.data.evidence_mode),(:reference_kind,info.data.reference_kind),(:source_sha256,info.data.source_sha256),(:input_view_id,info.join.input_view_id),(:input_prefix_sha256,info.join.input_prefix_sha256),(:training_boundary,info.data.training_boundary_ms),(:validation_start,info.data.validation_start_ms),(:validation_end,info.data.validation_end_ms),(:training_history_start,info.data.prefix_start_ms))
        T._need(getproperty(c,field)==value,"B v2 reference header source/clock/view differs")
    end
    T._need(T._int(c.delay_hours,0,1)==info.metadata.delay_hours,"B v2 reference header delay differs")
    for field in (:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256,:audit_contract_sha256,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256,:audit_sha256)
        T._need(T._hex(getproperty(c,field)),"B v2 reference header hash grammar differs")
    end
    T._need(!isempty(P._sources(c.baseline_sources)),"B v2 reference component sources are empty")
    for field in (:audit_contract_sha256,:raw_row_identity_sha256,:prediction_panel_sha256,:grid_sha256,:tuning_sha256)
        T._need(getproperty(audit,field)==getproperty(c,field),"B v2 repeated reference audit hash differs")
    end
    for field in (:full_per_lead_cohorts_checked,:feature_receipt_history_checked,:selected_prediction_columns_checked,:initial_only)
        T._need(getproperty(audit,field)===true,"B v2 reference audit header incomplete")
    end
    T._need(audit.all_eight_settings_checked===matched&&c.audit_sha256==audit_member.sha256,"B v2 reference audit scope/pin differs")
    T._need(c.model_ids isa AbstractVector&&c.selected_setting_ids isa AbstractVector&&length(c.model_ids)==length(c.selected_setting_ids)==2&&all(T._hex,c.model_ids),"B v2 reference model ID shape differs")
    for id in c.selected_setting_ids;T._int(id,1,matched ? 8 : 10000);end
    names=String["audit.json","predictions.bin","qualification.json"]
    if matched;append!(names,["grid.json","tuning.json"]);append!(names,["candidates/h$h/setting-$s.bin" for h in (4,6) for s in 1:8]);end
    sort!(names);T._need(c.members isa AbstractVector&&length(c.members)==length(names),"B v2 reference member ledger count differs")
    ledger=Dict{String,NamedTuple}();total=length(cb);directory=dirname(member.absolute)
    for (row,name) in zip(c.members,names)
        T._fields(row,(:name,:bytes,:sha256));T._need(row.name==name&&T._hex(row.sha256),"B v2 reference member ledger differs")
        bytes=T._int(row.bytes,1,T.MAX_FILE);endswith(name,".json")&&T._need(bytes<=T.MAX_META,"B v2 reference metadata bound differs")
        path=P._member(directory,name);T._need(isfile(path)&&filesize(path)==bytes,"B v2 reference member size/kind differs")
        total+=bytes;T._need(total<=128*1024^2,"B v2 reference total byte bound exceeded")
        ledger[name]=(;bytes,sha256=String(row.sha256))
    end
    pred=ledger["predictions.bin"];T._need(pred.bytes>=32&&(pred.bytes-16)%16==0&&1<=(pred.bytes-16)÷16<=info.metadata.prefix_rows,"B v2 prediction header geometry differs")
    T._need(ledger["audit.json"].sha256==audit_member.sha256&&pred.sha256==c.prediction_panel_sha256&&ledger["qualification.json"].sha256==c.baseline_qualification_sha256,"B v2 reference member/header pins differ")
    if matched
        T._need(ledger["grid.json"].sha256==c.grid_sha256&&ledger["tuning.json"].sha256==c.tuning_sha256,"B v2 grid/tuning header pins differ")
        count=(pred.bytes-16)÷16
        for h in (4,6),s in 1:8;T._need(ledger["candidates/h$h/setting-$s.bin"].bytes==16+8count,"B v2 candidate header geometry differs");end
    end
    if reference_pins!==nothing
        pins=_reference_pins_metadata_v2(reference_pins)
        T._need(pins.completion_sha256==member.sha256&&pins.audit_sha256==audit_member.sha256,"B v2 outside reference header pins differ")
        for field in (:grid_sha256,:tuning_sha256,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256)
            T._need(getproperty(c,field)==getproperty(pins,field),"B v2 outside reference component pin differs")
        end
        T._need(c.prediction_panel_sha256==pins.prediction_sha256&&P._sources(c.baseline_sources)==P._sources(pins.baseline_sources),"B v2 outside reference sources/predictions differ")
    end
    true
end
function _joined_view_v2(info,view)
    T._need(view.input_admission_sha256==info.source.sha256&&view.view_id==info.join.input_view_id&&view.input_prefix_sha256==info.join.input_prefix_sha256,"B v2 stable source view/join differs")
end
function admit_prefix_v2(path;expected_envelope_sha256,roots,expected_sources,independent_admission,expected_view_id=nothing,expected_input_prefix_sha256=nothing,reference_pins=nothing)
    info=_joined_metadata_v2(path;expected_envelope_sha256,roots,expected_sources,independent_admission,expected_view_id,expected_input_prefix_sha256,reference_pins)
    source=admit_source_prefix_v2(info.source.path;expected_source_proof_sha256=info.source.sha256,roots=info.source.roots,expected_sources)
    _joined_view_v2(info,source.view);m=info.metadata;d=info.data
    meta=(;evidence_mode=m.evidence_mode,source_sha256=m.source_sha256,training_boundary=m.training_boundary,validation_start=m.validation_start,validation_end=m.validation_end,delay_hours=m.delay_hours,reference_kind=Symbol(d.reference_kind),reference_completion_sha256=String(d.reference_completion_sha256),reference_audit_sha256=String(d.reference_audit_sha256),audit_completion_sha256=String(d.audit_completion_sha256),prefix_start=m.prefix_start,prefix_stop=m.prefix_stop,prefix_rows=m.prefix_rows,limits=m.limits,input_contract_version=2,source_admission_sha256=source.source_proof_sha256,reference_join_audit_sha256=String(d.reference_join_audit_sha256))
    provisional=BPrefixAdmission(info.path,info.sha256,info.roots,meta,info.members,source.view,source.source_proof_sha256,repeat("0",64))
    admission=BPrefixAdmission(info.path,info.sha256,info.roots,meta,info.members,source.view,source.source_proof_sha256,T._id(_admission_data(provisional)))
    recheck(admission);admission
end
function _recheck_prefix_v2(a::BPrefixAdmission)
    T._need(a.metadata.input_contract_version===2&&a.metadata.source_admission_sha256==a.independent_admission_sha256,"B v2 joined contract/source mutated")
    sp=_role_v2(a.members,"source_admission")
    info=_joined_metadata_v2(a.envelope_path;expected_envelope_sha256=a.envelope_sha256,roots=a.roots,expected_sources=P.source_identity(),independent_admission=(;path=sp.absolute,sha256=a.independent_admission_sha256),expected_view_id=a.view.view_id,expected_input_prefix_sha256=a.view.input_prefix_sha256)
    T._need(info.members==a.members&&info.data.reference_join_audit_sha256==a.metadata.reference_join_audit_sha256,"B v2 joined admission changed");_joined_view_v2(info,a.view);true
end
include("BReferenceAdmission.jl")
end
