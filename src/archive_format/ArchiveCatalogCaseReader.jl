"""Reverify immutable producer artifacts with separately qualified reader code.

Original producer grants and artifacts stay unchanged. A distinct read-only
grant binds the current fully qualified reader before source bodies are read.
"""
module ArchiveCatalogCaseReader
using SHA
import ..ArchiveCatalogCaseAuthority, ..ArchiveCatalogCaseJob, ..ArchiveCatalogSourceJob, ..CompletedSourceDBDT
const A=ArchiveCatalogCaseAuthority
const Q=ArchiveCatalogCaseJob
const J=ArchiveCatalogSourceJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
export ReaderAuthority,ReverifiedCases,read_reverified,admission_bytes
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))

"""Pins for an independent existing-artifact read grant and qualified reader source."""
struct ReaderAuthority
    grant_path::String
    grant_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end
struct _ReadToken end
"""Immutable verified cases and a receipt binding both producer and reader; no fit permission."""
struct ReverifiedCases
    cases::Q.VerifiedCases
    admission_json::String
    function ReverifiedCases(cases,receipt,::_ReadToken)
        new(cases,receipt)
    end
end

function _reader_binding(request,request_pin,manifest_pin,complete_pin,producer,reader)
    need(producer isa A.Authority && reader isa ReaderAuthority,"existing archival cases require distinct producer and reader authorities")
    manifest_pin=J._sha(manifest_pin);complete_pin=J._sha(complete_pin)
    # Validate the new read permit before loading any producer/source evidence.
    grant=A._pinned(reader.grant_path,reader.grant_sha256;canonical=true,limit=131072)
    J._object(grant,Set(("schema","operation","request_sha256","artifact_manifest_sha256","artifact_complete_sha256",
        "producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256",
        "reader_source_proof_sha256","reader_qualification_sha256","head",
        "actual_source_access_authorized","actual_model_fit_authorized","new_case_generation_authorized",
        "dataset_discovery","provider_receipts_available")))
    need(grant["schema"]=="archive_catalog_existing_case_read_grant_v1" && grant["operation"]=="verify_existing_cases_only" &&
        grant["request_sha256"]==J._sha(request_pin) && grant["artifact_manifest_sha256"]==manifest_pin &&
        grant["artifact_complete_sha256"]==complete_pin && grant["producer_grant_sha256"]==J._sha(producer.grant_sha256) &&
        grant["producer_source_proof_sha256"]==J._sha(producer.source_proof_sha256) &&
        grant["producer_qualification_sha256"]==J._sha(producer.qualification_sha256) &&
        grant["reader_source_proof_sha256"]==J._sha(reader.source_proof_sha256) &&
        grant["reader_qualification_sha256"]==J._sha(reader.qualification_sha256) &&
        grant["actual_source_access_authorized"]===true && grant["actual_model_fit_authorized"]===false &&
        grant["new_case_generation_authorized"]===false && grant["dataset_discovery"]===false &&
        grant["provider_receipts_available"]===false,"existing-case read grant differs")
    # The producer evidence is retained provenance, not a claim that old code is
    # loaded in this process. The independent reader qualification below binds
    # every current source file and is mandatory before numerical source access.
    old=A._evidence(request,request_pin,producer)
    proof=A._pinned(reader.source_proof_path,reader.source_proof_sha256;limit=1048576)
    qualification=A._pinned(reader.qualification_path,reader.qualification_sha256;limit=1048576)
    A._qualification(proof,qualification,reader,grant)
    root=realpath(pkgdir(parentmodule(@__MODULE__)))
    name="src/archive_format/ArchiveCatalogCaseReader.jl"
    need(haskey(proof["source_sha256"],name) && J._same(realpath(@__FILE__),joinpath(root,split(name,'/')...)),
        "existing-case reader missing from loaded qualification")
    receipt=Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"actual_archival_cache",
        "request_sha256"=>request_pin,"artifact_manifest_sha256"=>manifest_pin,"artifact_complete_sha256"=>complete_pin,
        "producer_head"=>old.proof["head"],"producer_grant_sha256"=>producer.grant_sha256,
        "producer_source_proof_sha256"=>producer.source_proof_sha256,"producer_qualification_sha256"=>producer.qualification_sha256,
        "reader_head"=>proof["head"],"reader_grant_sha256"=>reader.grant_sha256,
        "reader_source_proof_sha256"=>reader.source_proof_sha256,"reader_qualification_sha256"=>reader.qualification_sha256,
        "all_artifact_bytes_recomputed"=>true,"new_case_generation_authorized"=>false,"actual_model_fit_authorized"=>false,
        "provider_receipts_available"=>false,"actual_fits"=>0)
    old.binding,receipt
end

"""Recompute an existing case artifact under a separate qualified read grant.

Generated cases accept no actual authorities. Archival cases require both their
original producer authority and a new existing-case-only reader authority. The
original bounded reader recomputes every byte; any drift fails closed. No new
case artifact, model fit, provider receipt or forecast claim is authorized.
"""
function read_reverified(request_path,directory;request_sha256,manifest_sha256,complete_sha256,producer=nothing,reader=nothing)
    request,request_bytes,request_identity=Q._request(request_path,request_sha256)
    if request["evidence_mode"]=="generated"
        need(producer===nothing && reader===nothing,"generated case rereads cannot consume actual authorities")
        binding=nothing
        receipt=Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"generated",
            "request_sha256"=>request_sha256,"artifact_manifest_sha256"=>J._sha(manifest_sha256),
            "artifact_complete_sha256"=>J._sha(complete_sha256),"all_artifact_bytes_recomputed"=>true,
            "actual_model_fit_authorized"=>false,"new_case_generation_authorized"=>false,
            "provider_receipts_available"=>false,"actual_fits"=>0)
    else
        binding,receipt=_reader_binding(request,request_sha256,manifest_sha256,complete_sha256,producer,reader)
    end
    cases=Q._read_verified_bound(request_path,directory,request_sha256,manifest_sha256,complete_sha256,
        request,request_bytes,request_identity,binding)
    need(cases.actual_fit_performed===false,"case reread fit flag differs")
    bytes=R.canonical_bytes(receipt;limit=131072)
    ReverifiedCases(cases,String(copy(bytes)),_ReadToken())
end
"""Return owned canonical admission bytes; this receipt grants no model-fit permission."""
admission_bytes(value::ReverifiedCases)=Vector{UInt8}(codeunits(value.admission_json))
end
