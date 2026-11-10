"""Generated full500000-row/32-fit resource experiment; grants no archival authority."""
module Matched342FullShapeProfile
using Dates,SHA,JSON3,LinearAlgebra
include("BMatchedReference342.jl")
include("fixtures/FullShapeFixture342.jl")
const Entry=BMatchedReference342
const FC=Entry.BMatched342Core
const BI=Entry.BI
const P=Entry.P
const T=Entry.T
const M=Entry.M
const Producer=Entry.BMatchedInitialProducer
const Audit=Entry.MatchedReferenceAudit342
const G=FullShapeFixture342
const REPO=Entry.PACKAGE_ROOT
const MEMBER="tools/actual_b_chain/run_matched342_full_shape_profile.jl"
const SPEC_MEMBER="tools/actual_b_chain/fixtures/full_shape342_spec_v1.json"
const AUDIT_SHA="60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a"
const PROFILE_FIELDS=(:schema,:evidence_mode,:operation,:qualified_head,:implementation_sources,:source_proof,
    :component_qualification,:contracts,:runtime,:fixture_spec,:limits,:budgets,:output_directory,:owner_directory,:request_id)
const RESULT_FIELDS=(:schema,:complete,:evidence_mode,:head,:source_proof_sha256,:fixture_spec_sha256,:fixture_data,
    :sources,:component,:component_audit,:measurement,:runtime,:runtime_sha256,:publication,:profile_id)
need(ok,message)=T._need(ok,message)
fields(x,names)=FC._fields(x,names)
same(a,b)=T._bytes(a)==T._bytes(b)
fileref(x)=FC._ref(x)
function absolute_grammar(path)
    need(path isa AbstractString&&isabspath(path)&&normpath(String(path))==path&&
        all(x->x!="."&&x!="..",split(replace(path,'\\'=>'/'),'/')),"canonical absolute path required")
    String(path)
end
function ref_grammar(ref)
    fields(ref,(:path,:sha256));absolute_grammar(ref.path);need(T._hex(ref.sha256),"file SHA grammar differs")
    (;path=String(ref.path),sha256=String(ref.sha256))
end
idcheck(x,key)=FC._check_id(x,key)
function identified(data,key)
    merge(data,NamedTuple{(key,)}((T._id(data),)))
end
function write_json(path,data)
    body=T._bytes(data);need(length(body)<=T.MAX_META,"profile metadata exceeds2MiB")
    P._exclusive(String(path),body);(;path=String(path),sha256=T._sha(body))
end
function read_json(ref,names;controls=nothing)
    FC._json_ref(ref,controls;fields=names)
end
function validate_budgets(data)
    fields(data,(:peak_rss_bytes,:julia_gc_allocated_bytes,:descriptor_growth))
    (;peak_rss_bytes=T._int(data.peak_rss_bytes,1,8*1024^3),
      julia_gc_allocated_bytes=T._int(data.julia_gc_allocated_bytes,1,typemax(Int64)),
      descriptor_growth=T._int(data.descriptor_growth,0,typemax(Int64)))
end
function validate_limits(data)
    FC._check_limits(data)
    T._int(data.required_available_bytes,8*1024^3,typemax(Int64))
    T._int(data.required_disk_bytes,16*1024^3,typemax(Int64));data
end
function validate_profile_metadata(data)
    fields(data,PROFILE_FIELDS);idcheck(data,:request_id)
    need(data.schema=="matched342_full_shape_profile_request_v1"&&data.evidence_mode=="synthetic_only"&&
        data.operation=="initial32_full_shape_profile"&&data.qualified_head isa AbstractString&&
        occursin(r"\A[0-9a-f]{40}\z",data.qualified_head),"generated full-shape request header differs")
    P._sources(data.implementation_sources);ref_grammar(data.source_proof);ref_grammar(data.component_qualification)
    fields(data.fixture_spec,(:member,:path,:sha256));need(data.fixture_spec.member==SPEC_MEMBER,"fixed full-shape spec member required")
    ref_grammar((;path=data.fixture_spec.path,sha256=data.fixture_spec.sha256))
    Audit.check_contracts(data.contracts,AUDIT_SHA);Audit.check_runtime(data.runtime)
    need(data.runtime.julia_version in ("1.12.6","1.13.1"),"qualified Julia runtime version differs")
    validate_limits(data.limits);validate_budgets(data.budgets)
    absolute_grammar(data.output_directory);absolute_grammar(data.owner_directory);data
end
function disjoint(a,b)
    x=lowercase(replace(normpath(abspath(a)),'\\'=>'/'));y=lowercase(replace(normpath(abspath(b)),'\\'=>'/'))
    x!=y&&!startswith(x,y*"/")&&!startswith(y,x*"/")
end
function fresh_directory(path,forbidden)
    full=FC._absolute(path);need(isdir(dirname(full))&&!ispath(full),"fresh output under ordinary existing parent required")
    need(all(other->disjoint(full,other),forbidden),"profile output overlaps source/control/input")
    P._root_new(full)
end
function read_profile_request(path;expected_sha256)
    ref=fileref((;path,sha256=expected_sha256));controls=Dict{String,String}();data=read_json(ref,PROFILE_FIELDS;controls)
    validate_profile_metadata(data);sources=FC._sources(P._sources(data.implementation_sources))
    proof=FC.recheck_source_proof(data.source_proof;expected_head=String(data.qualified_head),expected_sources=sources)
    FC._pinned(data.source_proof,controls)
    FC._source_member(MEMBER,proof.source_sha256[MEMBER],proof,controls)
    runtime=FC._check_runtime(data.runtime)
    spec=FC._resource_spec((;fixture_spec=data.fixture_spec),proof,controls)
    qref=FC._pinned(data.component_qualification,controls);q=P._software_qualification(P._bound(qref.path);
        schema="b_reference_component_software_qualification_v1",evidence_mode="synthetic_only",head=String(data.qualified_head),sources)
    need(q.source_proof_sha256==data.source_proof.sha256,"profile genuine component qualification proof differs")
    (;data,ref,controls,sources,proof,runtime,spec)
end
function validate_bridge_metadata(data,request_sha256)
    fields(data,(:schema,:request_sha256,:prelaunch_admission))
    need(data.schema=="matched342_native_on_launch_bridge_v1"&&T._hex(request_sha256)&&data.request_sha256==request_sha256,
        "on-launch request bridge differs")
    ref_grammar(data.prelaunch_admission)
end
function read_bridge(path,request_ref,data;mode,permit_sha=nothing)
    expected=joinpath(String(data.owner_directory),"ON_LAUNCH.json")
    need(FC._absolute(path)==expected&&isdir(data.owner_directory),"fixed ordinary owner bridge path differs")
    deadline=time_ns()+UInt64(min(30,T._int(data.limits.stage_wall_seconds,1,div(typemax(Int64),10^9)))*10^9)
    while !isfile(expected)
        need(time_ns()<deadline,"native owner on-launch bridge unavailable");sleep(.01)
    end
    controls=Dict{String,String}();bridge,_=P._json(expected;fields=(:schema,:request_sha256,:prelaunch_admission))
    pref=validate_bridge_metadata(bridge,request_ref.sha256)
    need(pref.path==joinpath(String(data.owner_directory),"PRELAUNCH.json"),"prelaunch ref outside fixed owner directory")
    runtime=FC._check_runtime(data.runtime)
    FC._prelaunch(pref,String(data.qualified_head),String(data.source_proof.sha256),String(data.fixture_spec.sha256),
        runtime.runtime_sha256,data.limits,controls;mode,permit_sha,pid=getpid())
    pref
end
function member(root,name,role)
    path=P._member(root,name);sha,bytes=P._file_sha(path)
    (;root="source",path=name,bytes,sha256=sha,role)
end
function scan_generated_raw(ref)
    raw=FC._pinned(ref,nothing;limit=68_000_000);need(filesize(raw.path)==68_000_000,"exact68000000 fixture bytes required")
    buffer=Vector{UInt8}(undef,136*256);offset=0;bz=0.;speed=0.
    open(raw.path,"r") do io
        while !eof(io)
            n=readbytes!(io,buffer,length(buffer));need(n>0&&n%136==0,"generated raw record boundary differs")
            for word in reinterpret(UInt64,@view buffer[1:n])
                channel=mod1(offset+1,17);value=reinterpret(Float64,ltoh(word))
                need(isfinite(value)&&value<M._FILL_LIMITS[channel],"full resource raw fixture has missing/fill value")
                channel==6&&(bz=value);channel==8&&(speed=value)
                channel==12&&need(speed>=0&&isfinite(max(speed*max(-bz,0.)/1000-.49,0.)),"invalid full resource required driver")
                offset+=1
            end
        end
    end
    need(offset==17*500000&&P._file_sha(raw.path)[1]==raw.sha256,"generated raw changed during independent scan");raw
end
function build_source_pair(root,rawref;spec)
    need(!ispath(root),"fresh generated source metadata root required");P._root_new(root)
    need(spec.source_rows==500000,"full source builder cannot admit a small proxy")
    raw=scan_generated_raw(rawref);source_map=T._hashdata(T._hashmap(P.source_identity()))
    parser=Dict(P.source_identity())[BI.SOURCE_PARSER_V2];sources=NamedTuple[];admissions=BI.BSourcePrefixAdmissionV2[]
    # A shared raw root is declared explicitly; source-only IDs bind only the
    # listed upstream files, never the later reference or directory inventory.
    rawroot=dirname(raw.path)
    for delay in 0:1
        directory=joinpath(root,"L$delay");P._root_new(directory)
        common=(;complete=true,evidence_mode="synthetic_only",availability_kind="assumed_completed_hour",
            source_sha256=raw.sha256,raw_prefix_sha256=raw.sha256,prefix_start_ms=String(spec.prefix_start_ms),
            prefix_stop_ms=string(Dates.value(DateTime(1976,11,1)+Hour(500000))),prefix_rows=500000,channel_count=17,
            training_boundary_ms=String(spec.training_boundary_ms),validation_start_ms=String(spec.validation_start_ms),
            validation_end_ms=String(spec.validation_end_ms),receipt_delay_hours=delay,implementation_sources=source_map)
        provenance=merge((;schema="b_raw_prefix_provenance_v2"),common,(;source_format_sha256=T._id((;schema="generated_full_shape342_raw_format_v1",generator=spec.generator)),parser_source_sha256=parser,raw_layout="source17_le_f64_column_major_v1"))
        p=write_json(joinpath(directory,"provenance.json"),provenance)
        sa=write_json(joinpath(directory,"source-audit.json"),merge((;schema="independently_audited_b_raw_source_v2"),common,(;raw_provenance_sha256=p.sha256)))
        roots=Dict("source"=>directory,"raw"=>rawroot)
        members=[(;root="raw",path=basename(raw.path),bytes=68_000_000,sha256=raw.sha256,role="raw_prefix"),
            member(directory,"provenance.json","raw_provenance"),member(directory,"source-audit.json","independent_audit")]
        proof=merge((;schema="independently_audited_b_source_prefix_v2"),common,(;raw_prefix_member="raw/"*basename(raw.path),
            raw_prefix_bytes=68_000_000,raw_provenance_sha256=p.sha256,source_audit_sha256=sa.sha256,limits=P.LIMITS,members))
        saved=write_json(joinpath(directory,"source-proof.json"),proof)
        push!(sources,(;delay_hours=delay,source_proof=saved,roots))
        push!(admissions,BI.admit_source_prefix_v2(saved.path;expected_source_proof_sha256=saved.sha256,roots,expected_sources=P.source_identity()))
    end
    pair=FC.bind_source_pair(admissions...;loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())
    (;sources=Tuple(sources),admissions=Tuple(admissions),pair)
end
function admit_sources(specs;mode)
    need((specs isa Tuple||specs isa AbstractVector)&&length(specs)==2,"two independent source specifications required")
    admissions=map(enumerate(specs)) do (index,spec)
        fields(spec,(:delay_hours,:source_proof,:roots));T._int(spec.delay_hours,index-1,index-1);ref=fileref(spec.source_proof)
        roots=Dict(String(k)=>String(v) for (k,v) in pairs(spec.roots))
        result=BI.admit_source_prefix_v2(ref.path;expected_source_proof_sha256=ref.sha256,roots,expected_sources=P.source_identity())
        need(result.metadata.evidence_mode===mode,"source admission evidence mode differs");result
    end
    pair=FC.bind_source_pair(admissions...;loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())
    (;admissions=Tuple(admissions),pair)
end
function audit_and_publish(pair,draft,root,data;authority_ref=nothing)
    metadata=Producer.draft_metadata(draft);component=read_json((;path=metadata.component_completion_path,sha256=metadata.component_completion_sha256),Producer.COMPONENT_FIELDS)
    request=identified((;schema="matched342_initial_component_audit_request_v1",evidence_mode=String(FC.source_pair_metadata(pair).evidence_mode),
        operation="independent_saved_state_audit_initial32",component=(;path=metadata.component_completion_path,sha256=metadata.component_completion_sha256),
        component_qualification=data.component_qualification,source_proof=data.source_proof,source_bindings=component.source_bindings,
        contracts=data.contracts,runtime=data.runtime,execution_authority=authority_ref,
        drafts=Tuple((;delay_hours=k-1,root=metadata.reference_roots[k],members=component.drafts[k].members) for k in 1:2),
        output_directory=joinpath(root,"independent-audit")),:request_id)
    q=write_json(joinpath(root,"audit-request.json"),request)
    checked=Audit.audit_component(pair,q.path;request_sha256=q.sha256,expected_sources=Entry.source_identity(),expected_audit_contract_sha256=AUDIT_SHA)
    audit=Audit.component_audit_metadata(checked)
    publication=Producer.publish_initial(draft,checked;expected_audit_completion_sha256=audit.audit_completion_sha256)
    (;component=request.component,component_audit=(;path=joinpath(audit.audit_root,"audit-completion.json"),sha256=audit.audit_completion_sha256),publication)
end
function main(request_path,request_sha,bridge_path)
    BLAS.set_num_threads(1);info=read_profile_request(request_path;expected_sha256=request_sha);d=info.data
    prelaunch=read_bridge(bridge_path,info.ref,d;mode=:synthetic_only)
    forbidden=[REPO,info.ref.path,String(d.owner_directory),String(d.source_proof.path),String(d.component_qualification.path)]
    root=fresh_directory(String(d.output_directory),forbidden);budgets=validate_budgets(d.budgets)
    before=FC._descriptor_observation();started=time_ns()
    timed=Base.@timed begin
        rawdir=P._root_new(joinpath(root,"fixture"));raw=G.write_raw(joinpath(rawdir,"source17.bin"))
        source=build_source_pair(joinpath(root,"sources"),(;path=raw.path,sha256=raw.sha256);spec=info.spec)
        need(!ispath(joinpath(root,"initial32")),"reference files must not predate source-only admission")
        feature=(;member=Entry.FEATURE_CONTRACT_MEMBER,path=joinpath(REPO,Entry.FEATURE_CONTRACT_MEMBER),sha256=Entry.FEATURE_CONTRACT_SHA256)
        loss=(;member=Entry.LOSS_CONTRACT_MEMBER,path=joinpath(REPO,Entry.LOSS_CONTRACT_MEMBER),sha256=Entry.LOSS_CONTRACT_SHA256)
        draft=Producer.produce_initial_generated(source.admissions...,joinpath(root,"initial32");qualification=d.component_qualification,
            feature_contract=feature,loss_contract=loss,qualified_head=String(d.qualified_head),expected_sources=info.sources)
        published=audit_and_publish(source.pair,draft,root,d)
        FC.recheck_pair(source.pair);FC._check_runtime(d.runtime);FC._check_controls(info.controls)
        (;raw,sources=source.sources,published)
    end
    elapsed=time_ns()-started;after=FC._descriptor_observation();peak=Sys.maxrss()
    need(before.kind==after.kind&&before.method==after.method&&after.count-before.count<=budgets.descriptor_growth,"profile descriptor observation/budget differs")
    need(0<peak<=budgets.peak_rss_bytes&&0<timed.bytes<=budgets.julia_gc_allocated_bytes&&
        0<elapsed<UInt64(d.limits.stage_wall_seconds)*UInt64(10^9),"profile measured RSS/allocation/time budget exceeded")
    value=timed.value;runtime=FC._check_runtime(d.runtime)
    measure=identified((;schema="matched342_julia_process_measurement_v1",complete=true,evidence_mode="synthetic_only",head=String(d.qualified_head),
        source_proof_sha256=String(d.source_proof.sha256),fixture_spec_sha256=String(d.fixture_spec.sha256),fixture_sha256=value.raw.sha256,
        runtime_sha256=runtime.runtime_sha256,julia_pid=getpid(),scope="initial32_producer_and_independent_saved_state_audit",observed=true,
        peak_rss_bytes=Int(peak),peak_rss_method="Julia Sys.maxrss bytes; OS process-lifetime maximum",julia_gc_allocated_bytes=Int(timed.bytes),
        allocation_method="Julia Base.@timed.bytes; GC_Diff.allocd managed cumulative bytes",elapsed_ns=Int(elapsed),
        elapsed_method="Julia time_ns difference around complete timed body",descriptors=(;kind=before.kind,method=before.method,before=before.count,after=after.count,supported=true,observed=true)),:measurement_id)
    measurement=write_json(joinpath(root,"measurement.json"),measure)
    result=identified((;schema="matched342_full_shape_profile_result_v1",complete=true,evidence_mode="synthetic_only",head=String(d.qualified_head),
        source_proof_sha256=String(d.source_proof.sha256),fixture_spec_sha256=String(d.fixture_spec.sha256),fixture_data=(;path=value.raw.path,sha256=value.raw.sha256),
        sources=value.sources,component=value.published.component,component_audit=value.published.component_audit,measurement,
        runtime=runtime.runtime,runtime_sha256=runtime.runtime_sha256,publication=value.published.publication),:profile_id)
    saved=write_json(joinpath(root,"PROFILE_RESULT.json"),result)
    FC.recheck_source_proof(d.source_proof;expected_head=String(d.qualified_head),expected_sources=info.sources)
    need(P._file_sha(info.ref.path)[1]==info.ref.sha256,"profile request changed")
    println("MATCHED342_PROFILE_COMPLETE ",JSON3.write((;profile_request=info.ref,julia_pid=getpid(),profile_result_sha256=saved.sha256,measurement_sha256=measurement.sha256,
        fixture_sha256=value.raw.sha256,runtime_sha256=runtime.runtime_sha256,source_proof_sha256=String(d.source_proof.sha256))));flush(stdout)
    nothing
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==3||error("profile requires request path, request SHA and fixed on-launch bridge path")
    Matched342FullShapeProfile.main(ARGS...)
end
