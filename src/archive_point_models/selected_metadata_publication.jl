const _METADATA_FILES=("request.json","payload.json","receipt.json","COMPLETE.json")
const _METADATA_LIMITS=(MAX_REQUEST,MAX_DECISION,65536,4096)
const _PUBLICATION_LIMIT=462848
struct MetadataPins
    request_sha256::String
    payload_sha256::String
    receipt_sha256::String
    complete_sha256::String
    MetadataPins(request,payload,receipt,complete)=new(sha(request),sha(payload),sha(receipt),sha(complete))
end
_metadata_kind(::SelectionRequest)=:request
_metadata_kind(::ParsedSelectionDecision)=:decision
_metadata_kind(::SelectedPointBinding)=:binding
_payload(value::SelectionRequest)=encode_request(value)
_payload(value::ParsedSelectionDecision)=encode_decision(value)
_payload(value::SelectedPointBinding)=encode_binding(value)
_request_bytes(value::SelectionRequest)=encode_request(value)
_request_bytes(value::ParsedSelectionDecision)=bytes(decoded(encode_decision(value);limit=MAX_DECISION)["request"];limit=MAX_REQUEST)
function _request_bytes(value::SelectedPointBinding)
    d=decoded(encode_decision(value.rates.decision);limit=MAX_DECISION)
    point_request=value.point_choice===nothing ? nothing :
        decoded(encode_decision(value.point_choice);limit=MAX_DECISION)["request_sha256"]
    bytes(Dict("schema"=>"selected_point_binding_request_data_v1","point_context_sha256"=>value.point.expected.context_sha256,
        "primary_gamma_request_sha256"=>d["request_sha256"],"primary_gamma_decision_sha256"=>value.rates.decision.sha256,
        "point_choice_request_sha256"=>point_request,
        "point_choice_decision_sha256"=>(value.point_choice===nothing ? nothing : value.point_choice.sha256));limit=MAX_REQUEST)
end
"""Publish parsed metadata only to a fresh bounded ordinary four-file directory."""
function publish_metadata(value::Union{SelectionRequest,ParsedSelectionDecision,SelectedPointBinding},directory)
    request=_request_bytes(value);payload=_payload(value);kind=String(_metadata_kind(value))
    receipt=bytes(Dict("schema"=>"selected_metadata_receipt_data_v1","kind"=>kind,"complete_as_metadata"=>true,
        "request_sha256"=>digest(request),"payload_sha256"=>digest(payload),"producer_evidence_verified"=>false,
        "actual_authority_granted"=>false,"new_model_fits"=>0);limit=65536)
    marker=bytes(Dict("schema"=>"selected_metadata_complete_data_v1","kind"=>kind,
        "request_sha256"=>digest(request),"payload_sha256"=>digest(payload),"receipt_sha256"=>digest(receipt),
        "members"=>4,"bytes_without_marker"=>length(request)+length(payload)+length(receipt),"parsed_metadata_only"=>true);limit=4096)
    payloads=(request,payload,receipt,marker)
    need(all(length(raw)<=limit for (raw,limit) in zip(payloads,_METADATA_LIMITS)) && sum(length,payloads)<=_PUBLICATION_LIMIT,
        "selected metadata publication bound")
    path=abspath(text(directory,4096,"selected metadata output path"));J._ordinary(dirname(path);directory=true)
    need(!ispath(path)&&!islink(path),"selected metadata publication must be fresh")
    mkdir(path);J._ordinary(path;directory=true)
    for (name,raw) in zip(_METADATA_FILES[1:3],payloads[1:3]);R._exclusive_write(joinpath(path,name),raw);end
    need(_request_bytes(value)==request && _payload(value)==payload,"selected metadata changed during publication")
    R._exclusive_write(joinpath(path,"COMPLETE.json"),marker)
    MetadataPins(digest(request),digest(payload),digest(receipt),digest(marker))
end
function _read_metadata(directory,expected::MetadataPins,kind::Symbol)
    need(kind in (:request,:decision,:binding),"selected metadata publication kind")
    J._ordinary(directory;directory=true);names=readdir(directory)
    need(length(names)==4 && Set(names)==Set(_METADATA_FILES),"selected metadata publication fileset")
    payloads=Vector{UInt8}[];identities=Any[]
    for (name,limit) in zip(_METADATA_FILES,_METADATA_LIMITS)
        value,identity=FS._read(joinpath(directory,name),limit)
        push!(payloads,Vector{UInt8}(codeunits(value)));push!(identities,identity)
    end
    need(sum(length,payloads)<=_PUBLICATION_LIMIT,"selected metadata total byte bound")
    pins=(expected.request_sha256,expected.payload_sha256,expected.receipt_sha256,expected.complete_sha256)
    need(all(digest(raw)==pin for (raw,pin) in zip(payloads,pins)),"selected metadata external pins differ")
    receipt=decoded(payloads[3];limit=65536);marker=decoded(payloads[4];limit=4096)
    object(receipt,("schema","kind","complete_as_metadata","request_sha256","payload_sha256","producer_evidence_verified",
        "actual_authority_granted","new_model_fits"),"selected metadata receipt fields")
    need(receipt["schema"]=="selected_metadata_receipt_data_v1" && receipt["kind"]==String(kind) && receipt["complete_as_metadata"]===true &&
        receipt["request_sha256"]==pins[1] && receipt["payload_sha256"]==pins[2] && receipt["producer_evidence_verified"]===false &&
        receipt["actual_authority_granted"]===false && count(receipt["new_model_fits"],0,0)==0,"selected metadata receipt differs")
    object(marker,("schema","kind","request_sha256","payload_sha256","receipt_sha256","members","bytes_without_marker","parsed_metadata_only"),"selected metadata completion fields")
    need(marker["schema"]=="selected_metadata_complete_data_v1" && marker["kind"]==String(kind) && marker["parsed_metadata_only"]===true &&
        count(marker["members"],4,4)==4 && count(marker["bytes_without_marker"],0,_PUBLICATION_LIMIT)==sum(length,payloads[1:3]) &&
        all(marker[key]==pin for (key,pin) in zip(("request_sha256","payload_sha256","receipt_sha256"),pins[1:3])),"selected metadata completion differs")
    (;payloads,names,identities)
end
function _unchanged(directory,checked)
    need(readdir(directory)==checked.names && all(J._identity(joinpath(directory,name))==identity
        for (name,identity) in zip(_METADATA_FILES,checked.identities)),"selected metadata publication changed during read")
end
function read_request(directory;expected::MetadataPins)
    checked=_read_metadata(directory,expected,:request)
    value=decode_request(checked.payloads[2];expected_sha256=expected.payload_sha256)
    need(_request_bytes(value)==checked.payloads[1],"selected request sidecar differs");_unchanged(directory,checked);value
end
function read_decision(directory;expected::MetadataPins)
    checked=_read_metadata(directory,expected,:decision)
    value=decode_decision(checked.payloads[2];expected_sha256=expected.payload_sha256)
    need(_request_bytes(value)==checked.payloads[1],"selected decision request sidecar differs");_unchanged(directory,checked);value
end
function read_binding(directory;expected::MetadataPins,point::F.BoundFit,rates::PrimaryRateChoice,point_choice=nothing)
    checked=_read_metadata(directory,expected,:binding)
    value=decode_binding(checked.payloads[2];expected_sha256=expected.payload_sha256,point,rates,point_choice)
    need(_request_bytes(value)==checked.payloads[1],"selected binding request sidecar differs");_unchanged(directory,checked);value
end
