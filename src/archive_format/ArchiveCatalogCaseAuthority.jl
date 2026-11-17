"""Ordinary source/request identity for archival feature and label artifacts."""
module ArchiveCatalogCaseAuthority
import ..ArchiveCatalogSourceJob
const J=ArchiveCatalogSourceJob
need(ok,message)=ok || throw(ArgumentError(message))
_pinned(path,pin;kwargs...)=J._pinned(path,pin;kwargs...)
function _plan_clock(value)
    text=J._text(value;limit=24)
    if occursin(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",text)
        text=text[1:end-1]*".000Z"
    end
    J._clock(text)
    text
end
function _project_table(table)
    need(table isa AbstractDict,"case feature plan table differs")
    Dict("id"=>table["id"],"station"=>table["station"],
        "start_inclusive"=>_plan_clock(table["start_inclusive"]),
        "stop_exclusive"=>_plan_clock(table["stop_exclusive"]),
        "label_asof"=>_plan_clock(table["label_asof"]))
end

"""Bind explicit actual source pins; no software certificate grants execution."""
function authorize(request,request_sha256,authority=nothing)
    need(authority===nothing,"execution certificates are not accepted; supply the pinned case request")
    J._sha(request_sha256)
    request["evidence_mode"]=="generated" && return nothing
    need(request["evidence_mode"]=="actual_archival_cache" && request["schema"]=="archive_catalog_case_request_v1",
        "actual case request differs")
    source=request["source"];J._object(source,Set(("directory","manifest_sha256","complete_sha256")))
    Dict("schema"=>"archive_case_source_binding_v2","operation"=>"sealed_features_labels_only",
        "request_sha256"=>request_sha256,"source_manifest_sha256"=>J._sha(source["manifest_sha256"]),
        "source_complete_sha256"=>J._sha(source["complete_sha256"]))
end
end
