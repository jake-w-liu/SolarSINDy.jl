"""Recompute existing scientific artifacts under their explicit outside pins."""
module ArchiveCatalogCaseReader
using SHA
import ..ArchiveCatalogCaseAuthority, ..ArchiveCatalogCaseJob, ..ArchiveCatalogSourceJob, ..CompletedSourceDBDT
const A=ArchiveCatalogCaseAuthority
const Q=ArchiveCatalogCaseJob
const J=ArchiveCatalogSourceJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
export ReverifiedCases,read_reverified,admission_bytes
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
struct _ReadToken end
"""Immutable fully recomputed cases and their data identity."""
struct ReverifiedCases
    cases::Q.VerifiedCases
    admission_json::String
    function ReverifiedCases(cases,receipt,::_ReadToken)
        new(cases,receipt)
    end
end
"""
    read_reverified(request_path, directory; request_sha256, manifest_sha256, complete_sha256)

Recompute an existing case artifact from its pinned request and preserved source
snapshot. Check every feature and label byte and return `ReverifiedCases` with a
canonical data-identity receipt. This operation performs no model fitting.
"""
function read_reverified(request_path,directory;request_sha256,manifest_sha256,complete_sha256,producer=nothing,reader=nothing)
    need(producer===nothing && reader===nothing,"execution certificates are not accepted; supply artifact pins")
    request,request_bytes,request_identity=Q._request(request_path,request_sha256)
    binding=A.authorize(request,request_sha256)
    cases=Q._read_verified_bound(request_path,directory,request_sha256,manifest_sha256,complete_sha256,
        request,request_bytes,request_identity,binding)
    need(cases.actual_fit_performed===false,"case reread fit flag differs")
    actual=request["evidence_mode"]=="actual_archival_cache"
    receipt=Dict("schema"=>(actual ? "archive_catalog_case_read_admission_v2" : "archive_catalog_case_read_admission_v1"),
        "evidence_mode"=>request["evidence_mode"],"request_sha256"=>J._sha(request_sha256),
        "artifact_manifest_sha256"=>J._sha(manifest_sha256),"artifact_complete_sha256"=>J._sha(complete_sha256),
        "all_artifact_bytes_recomputed"=>true,"provider_receipts_available"=>false,"actual_fits"=>0)
    if !actual
        receipt["actual_model_fit_authorized"]=false
        receipt["new_case_generation_authorized"]=false
    end
    ReverifiedCases(cases,String(R.canonical_bytes(receipt;limit=131072)),_ReadToken())
end
"""
    admission_bytes(value::ReverifiedCases)

Return a caller-owned `Vector{UInt8}` containing the canonical case-read receipt.
"""
admission_bytes(value::ReverifiedCases)=Vector{UInt8}(codeunits(value.admission_json))
end
