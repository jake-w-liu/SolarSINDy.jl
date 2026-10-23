module BInputs
using SolarSINDy,Dates,SHA
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
    (;schema="b_prefix_admission_v1",envelope_sha256=a.envelope_sha256,metadata=a.metadata,members=[(;root=x.root,path=x.path,bytes=x.bytes,sha256=x.sha256,role=x.role) for x in a.members],input_view_id=a.view.view_id,independent_admission_sha256=a.independent_admission_sha256)
end
function _proof(path,pin,envelope,mode)
    T._need(T._hex(pin)&&P._file_sha(path)[1]==pin,"B independent admission proof pin differs")
    d,_=P._json(path;fields=PROOF_FIELDS)
    T._need(d.schema=="independently_audited_b_prefix_v1"&&d.complete===true&&d.evidence_mode==String(mode),"new independent B admission proof required")
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
        T._need(d.source_sha256==ACTUAL_SOURCE&&start==DateTime(1976)&&stop<=DateTime(2023)&&boundary==DateTime(2020)&&first==DateTime(2020)&&last==DateTime(2023),"B initial archival source/split differs")
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
    for member in a.members
        path=_resolve(a.roots,member.root,member.path)
        T._need(path==member.absolute&&P._file_sha(path)==(member.sha256,member.bytes),"B admitted input member changed")
    end
    T._recheck(a.view);true
end
end
