# Test-only file composition. No production constructor receives an unchecked
# array reference, qualification, permit or predeclared future output hash.
b2_plain(x::AbstractDict)=Dict{String,Any}(String(k)=>b2_plain(v) for (k,v) in pairs(x))
b2_plain(x::NamedTuple)=Dict{String,Any}(String(k)=>b2_plain(v) for (k,v) in pairs(x))
b2_plain(x::AbstractVector)=[b2_plain(v) for v in x]
b2_plain(x::Tuple)=[b2_plain(v) for v in x]
b2_plain(x)=x
b2_sha(body)=bytes2hex(sha256(body))
b2_file(path)=open(io->bytes2hex(sha256(io)),path)
b2_json(path)=b2_plain(JSON3.read(read(path)))

function b2_emit(io,x)
    if x isa AbstractDict || x isa NamedTuple
        data=b2_plain(x);names=sort!(collect(keys(data)));write(io,'{')
        for (i,name) in enumerate(names)
            i>1&&write(io,',');JSON3.write(io,name);write(io,':');b2_emit(io,data[name])
        end
        write(io,'}')
    elseif x isa Tuple || x isa AbstractVector
        write(io,'[')
        for (i,value) in enumerate(x);i>1&&write(io,',');b2_emit(io,value);end
        write(io,']')
    elseif x isa DateTime
        JSON3.write(io,string(Dates.value(x)))
    elseif x isa Symbol
        JSON3.write(io,String(x))
    else
        JSON3.write(io,x)
    end
end
b2_bytes(x)=begin io=IOBuffer();b2_emit(io,x);write(io,'\n');take!(io) end
b2_id(x)=b2_sha(b2_bytes(x))
function b2_write(path,value)
    @assert !ispath(path)
    open(path,"w") do io;write(io,b2_bytes(value));end
    b2_file(path)
end
function b2_replace(path,value)
    open(path,"w") do io;write(io,b2_bytes(value));end
    b2_file(path)
end

function b2_raw_matrix()
    values=ones(17,1024)
    for i in 1:1024
        hour=i-1
        values[6,i]=mod(hour,10)<5 ? 1. : (-1.,-2.5,-5.,-7.5)[mod(div(hour,10),4)+1]
        values[8,i]=400.;values[12,i]=2.;values[15,i]=mod(hour,192)<96 ? -60. : -30.
    end
    values
end
function b2_raw_bytes(values)
    io=IOBuffer()
    for x in values;write(io,htol(reinterpret(UInt64,x)));end
    take!(io)
end
b2_source_rows()=[Dict("name"=>String(first(p)),"sha256"=>String(last(p))) for p in sort!(collect(P.source_identity());by=first)]
b2_member(root,path,directory,role)=Dict{String,Any}("root"=>root,"path"=>path,"bytes"=>filesize(joinpath(directory,path)),"sha256"=>b2_file(joinpath(directory,path)),"role"=>role)

function b2_source_fixture(base;delay=0,common_changes=Dict{String,Any}(),provenance_changes=Dict{String,Any}(),audit_changes=Dict{String,Any}(),source_changes=Dict{String,Any}(),raw_change=identity,source_edit=identity)
    mkdir(base);directory=joinpath(base,"source");mkdir(directory)
    values=b2_raw_matrix();raw_change(values);raw=b2_raw_bytes(values)
    rawpath=joinpath(directory,"raw.bin");open(io->write(io,raw),rawpath,"w")
    start=DateTime(1900);stop=start+Hour(1024);boundary=start+Hour(512)
    common=Dict{String,Any}("complete"=>true,"evidence_mode"=>"synthetic_only","availability_kind"=>"assumed_completed_hour",
        "source_sha256"=>b2_sha(raw),"raw_prefix_sha256"=>b2_sha(raw),"prefix_start_ms"=>string(Dates.value(start)),
        "prefix_stop_ms"=>string(Dates.value(stop)),"prefix_rows"=>1024,"channel_count"=>17,
        "training_boundary_ms"=>string(Dates.value(boundary)),"validation_start_ms"=>string(Dates.value(boundary)),
        "validation_end_ms"=>string(Dates.value(stop)),"receipt_delay_hours"=>delay,"implementation_sources"=>b2_source_rows())
    merge!(common,common_changes)
    parser=Dict(P.source_identity())[ORACLE["parser_member"]]
    provenance=merge(copy(common),Dict{String,Any}("schema"=>"b_raw_prefix_provenance_v2",
        "source_format_sha256"=>b2_sha(codeunits("explicit generated source17 fixture format")),
        "parser_source_sha256"=>parser,"raw_layout"=>ORACLE["raw_layout"]),provenance_changes)
    provenance_path=joinpath(directory,"provenance.json");provenance_sha=b2_write(provenance_path,provenance)
    audit=merge(copy(common),Dict{String,Any}("schema"=>"independently_audited_b_raw_source_v2","raw_provenance_sha256"=>provenance_sha),audit_changes)
    audit_path=joinpath(directory,"source-audit.json");audit_sha=b2_write(audit_path,audit)
    members=[b2_member("source",name,directory,role) for (name,role) in
        (("raw.bin","raw_prefix"),("provenance.json","raw_provenance"),("source-audit.json","independent_audit"))]
    proof=merge(copy(common),Dict{String,Any}("schema"=>"independently_audited_b_source_prefix_v2",
        "raw_prefix_member"=>"source/raw.bin","raw_prefix_bytes"=>length(raw),"raw_provenance_sha256"=>provenance_sha,
        "source_audit_sha256"=>audit_sha,"limits"=>deepcopy(ORACLE["limits"]),"members"=>members),source_changes)
    source_edit(proof)
    path=joinpath(directory,"source-proof.json");proof_sha=b2_write(path,proof)
    (;base,directory,values,common,provenance,audit,proof,path,proof_sha,roots=Dict("source"=>directory))
end
b2_admit_source(f)=I.admit_source_prefix_v2(f.path;expected_source_proof_sha256=f.proof_sha,roots=f.roots,expected_sources=P.source_identity())

function b2_expected_view(f)
    # Independent byte/clock calculation: no production _input_digest,
    # _prefix_digest, _view_data, _point_input_view, _bytes or _id is used.
    start=DateTime(1900);times=IOBuffer();wind=IOBuffer()
    for i in 1:1024
        tag=start+Hour(i-1);completed=tag+Hour(1)
        write(times,htol(Int64(Dates.value(tag))))
        bz=f.values[6,i];speed=f.values[8,i]
        q=max(speed*max(-bz,0.)/1000-.49,0.)
        bitword(x)=isnan(x) ? "missing" : string(reinterpret(UInt64,x))
        write(wind,join((Dates.value(tag),Dates.value(completed),Dates.value(completed),bitword(bz),bitword(q)),"|")*"\n")
    end
    prefix=b2_id(Dict("schema"=>"b_point_input_prefix_v1","source_sha256"=>f.common["source_sha256"],
        "time_sha256"=>b2_sha(take!(times)),"raw_sha256"=>b2_sha(b2_raw_bytes(f.values)),"source_rows"=>1024))
    wind_sha=b2_sha(take!(wind))
    viewdata=Dict("schema"=>"b_point_input_view_v1","evidence_mode"=>"synthetic_only","source_sha256"=>f.common["source_sha256"],
        "input_prefix_sha256"=>prefix,"wind_sequence_sha256"=>wind_sha,"input_admission_sha256"=>f.proof_sha,
        "source_hours"=>1024,"source_hashes"=>b2_source_rows(),"contract_sha256"=>ORACLE["contract_sha256"])
    (;view_id=b2_id(viewdata),input_prefix_sha256=prefix,wind_sequence_sha256=wind_sha)
end

function b2_pin_record(pins)
    data=Dict{String,Any}()
    for field in (:completion_sha256,:audit_sha256,:grid_sha256,:tuning_sha256,:prediction_sha256,:feature_contract_sha256,
            :tuning_loss_contract_sha256,:baseline_qualification_sha256,:baseline_qualified_head,:feature_contract_member,:tuning_loss_contract_member)
        data[String(field)]=getproperty(pins,field)
    end
    data["baseline_sources"]=[Dict("name"=>first(p),"sha256"=>last(p)) for p in pins.baseline_sources]
    data
end

function b2_join_fixture(f,source;tag="matched",kind=:matched_information_gbm_initial_v1,variant=:good)
    reference_directory=joinpath(f.base,"reference-"*tag)
    joined_directory=joinpath(f.base,"joined-"*tag)
    witness=write_reference(reference_directory,source.view;delay=f.common["receipt_delay_hours"],kind,variant)
    reference=I.admit_reference(joinpath(reference_directory,"completion.json");trusted_pins=witness.pins,
        roots=Dict("reference"=>reference_directory),view=source.view,expected_sources=P.source_identity())
    mkdir(joined_directory)
    pins_data=b2_pin_record(witness.pins);pins_path=joinpath(joined_directory,"reference-pins.json");pins_sha=b2_write(pins_path,pins_data)
    join_data=merge(copy(f.common),Dict{String,Any}("schema"=>"independently_audited_b_reference_join_v2",
        "source_admission_sha256"=>f.proof_sha,"input_view_id"=>source.view.view_id,"input_prefix_sha256"=>source.view.input_prefix_sha256,
        "reference_kind"=>String(kind),"reference_completion_sha256"=>witness.pins.completion_sha256,
        "reference_audit_sha256"=>witness.pins.audit_sha256,"source_audit_sha256"=>f.proof["source_audit_sha256"],
        "raw_provenance_sha256"=>f.proof["raw_provenance_sha256"]))
    join_path=joinpath(joined_directory,"reference-join.json");join_sha=b2_write(join_path,join_data)
    members=deepcopy(f.proof["members"])
    append!(members,[b2_member("source","source-proof.json",f.directory,"source_admission"),
        b2_member("reference","completion.json",reference_directory,"reference_completion"),
        b2_member("reference","audit.json",reference_directory,"reference_audit"),
        b2_member("join","reference-join.json",joined_directory,"reference_join_audit")])
    envelope=merge(copy(f.common),Dict{String,Any}("schema"=>"b_point_input_envelope_v2","raw_prefix_member"=>"source/raw.bin",
        "raw_prefix_bytes"=>f.proof["raw_prefix_bytes"],"independent_admission_completion_sha256"=>f.proof_sha,
        "audit_completion_sha256"=>f.proof["source_audit_sha256"],"reference_kind"=>String(kind),
        "reference_completion_sha256"=>witness.pins.completion_sha256,"reference_audit_sha256"=>witness.pins.audit_sha256,
        "reference_join_audit_sha256"=>join_sha,"limits"=>deepcopy(ORACLE["limits"]),"members"=>members))
    envelope_path=joinpath(joined_directory,"envelope.json");envelope_sha=b2_write(envelope_path,envelope)
    roots=Dict("source"=>f.directory,"reference"=>reference_directory,"join"=>joined_directory)
    independent=(;path=f.path,sha256=f.proof_sha)
    admission=I.admit_prefix_v2(envelope_path;expected_envelope_sha256=envelope_sha,roots,expected_sources=P.source_identity(),independent_admission=independent,
        expected_view_id=source.view.view_id,expected_input_prefix_sha256=source.view.input_prefix_sha256,reference_pins=(;path=pins_path,sha256=pins_sha))
    input_spec=(;envelope=(;path=envelope_path,sha256=envelope_sha),roots,independent_admission=independent,
        expected_view_id=source.view.view_id,expected_admitted_seal=admission.admission_id)
    reference_spec=(;completion_path=joinpath(reference_directory,"completion.json"),roots=Dict("reference"=>reference_directory),pins=(;path=pins_path,sha256=pins_sha))
    (;source_fixture=f,source,witness,reference,admission,reference_directory,joined_directory,join_data,join_path,
        envelope,envelope_path,envelope_sha,roots,input_spec,reference_spec)
end
function b2_admit_join(f;metadata_only=false,reference_pins=f.reference_spec.pins)
    checked=metadata_only ? I._joined_metadata_v2 : I.admit_prefix_v2
    checked(f.envelope_path;expected_envelope_sha256=b2_file(f.envelope_path),roots=f.roots,
        expected_sources=P.source_identity(),independent_admission=(;path=f.source_fixture.path,sha256=f.source_fixture.proof_sha),
        expected_view_id=f.source.view.view_id,expected_input_prefix_sha256=f.source.view.input_prefix_sha256,reference_pins)
end

function b2_reseal_join!(f;join_change=identity,envelope_change=identity)
    join_data=deepcopy(f.join_data);join_change(join_data);join_sha=b2_replace(f.join_path,join_data)
    envelope=deepcopy(f.envelope);envelope["reference_join_audit_sha256"]=join_sha
    member=only(filter(row->row["role"]=="reference_join_audit",envelope["members"]))
    member["sha256"]=join_sha;member["bytes"]=filesize(f.join_path)
    envelope_change(envelope);b2_replace(f.envelope_path,envelope)
    nothing
end

function b2_reseal_reference_headers!(f;completion_change=identity,audit_change=identity)
    auditpath=joinpath(f.reference_directory,"audit.json");audit=b2_json(auditpath)
    audit_change(audit);delete!(audit,"audit_id");audit["audit_id"]=b2_id(audit)
    audit_sha=b2_replace(auditpath,audit)
    completionpath=joinpath(f.reference_directory,"completion.json");completion=b2_json(completionpath)
    completion_change(completion);completion["audit_sha256"]=audit_sha
    entry=only(filter(row->row["name"]=="audit.json",completion["members"]))
    entry["sha256"]=audit_sha;entry["bytes"]=filesize(auditpath)
    delete!(completion,"completion_id");completion["completion_id"]=b2_id(completion)
    completion_sha=b2_replace(completionpath,completion)
    pinpath=f.reference_spec.pins.path;outside=b2_json(pinpath)
    outside["completion_sha256"]=completion_sha;outside["audit_sha256"]=audit_sha
    pinsha=b2_replace(pinpath,outside)
    changed_join=j->begin
        j["reference_completion_sha256"]=completion_sha;j["reference_audit_sha256"]=audit_sha
    end
    changed_envelope=e->begin
        e["reference_completion_sha256"]=completion_sha;e["reference_audit_sha256"]=audit_sha
        for (role,path,sha) in (("reference_completion",completionpath,completion_sha),("reference_audit",auditpath,audit_sha))
            member=only(filter(row->row["role"]==role,e["members"]))
            member["sha256"]=sha;member["bytes"]=filesize(path)
        end
    end
    b2_reseal_join!(f;join_change=changed_join,envelope_change=changed_envelope)
    input_spec=merge(f.input_spec,(;envelope=(;path=f.envelope_path,sha256=b2_file(f.envelope_path))))
    reference_spec=merge(f.reference_spec,(;pins=(;path=pinpath,sha256=pinsha)))
    (;input_spec,reference_spec)
end

function b2_expected_admission(f)
    c=f.source_fixture.common;s=f.source_fixture.proof
    metadata=Dict{String,Any}("evidence_mode"=>c["evidence_mode"],"source_sha256"=>c["source_sha256"],
        "training_boundary"=>c["training_boundary_ms"],"validation_start"=>c["validation_start_ms"],"validation_end"=>c["validation_end_ms"],
        "delay_hours"=>c["receipt_delay_hours"],"reference_kind"=>f.envelope["reference_kind"],
        "reference_completion_sha256"=>f.envelope["reference_completion_sha256"],"reference_audit_sha256"=>f.envelope["reference_audit_sha256"],
        "audit_completion_sha256"=>s["source_audit_sha256"],"prefix_start"=>c["prefix_start_ms"],"prefix_stop"=>c["prefix_stop_ms"],
        "prefix_rows"=>1024,"limits"=>deepcopy(ORACLE["limits"]),"input_contract_version"=>2,
        "source_admission_sha256"=>f.source_fixture.proof_sha,"reference_join_audit_sha256"=>f.envelope["reference_join_audit_sha256"])
    b2_id(Dict("schema"=>"b_prefix_admission_v2","envelope_sha256"=>f.envelope_sha,"metadata"=>metadata,
        "members"=>f.envelope["members"],"input_view_id"=>f.source.view.view_id,"independent_admission_sha256"=>f.source_fixture.proof_sha))
end
