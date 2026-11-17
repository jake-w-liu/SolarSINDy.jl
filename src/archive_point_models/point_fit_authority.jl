"""Check actual data lineage and strict training clocks before fitting."""
function _archival_request(value::B.ReverifiedCases,index,start,stop,asof)
    cases=value.cases
    need(cases.actual_source_authority_granted===true && cases.actual_fit_performed===false,"archival fit requires reverified archival cases")
    need(isint(index) && 1<=index<=length(cases.features),"archival fit table index")
    need(all(t->t isa DateTime,(start,stop,asof)) && DateTime(2017)<=start<stop<=DateTime(2018) && stop<=asof,
        "archival fitting requires a strict 2017 training split")
    request=R.decode_canonical(Vector{UInt8}(codeunits(cases.request_json));limit=131072)
    need(digest(codeunits(cases.request_json))==cases.request_sha256 &&
        request["schema"]=="archive_catalog_case_request_v1" && request["evidence_mode"]=="actual_archival_cache",
        "archival case request differs")
    admission=R.decode_canonical(B.admission_bytes(value);limit=131072)
    common=Set(("schema","evidence_mode","request_sha256","artifact_manifest_sha256","artifact_complete_sha256",
        "all_artifact_bytes_recomputed","provider_receipts_available","actual_fits"))
    if get(admission,"schema",nothing)=="archive_catalog_case_read_admission_v2"
        J._object(admission,common)
    else
        need(get(admission,"schema",nothing)=="archive_catalog_case_read_admission_v1","archival case read schema differs")
        # Legacy certificate fields are provenance, never execution authority.
        need(issubset(common,Set(keys(admission))),"archival case read fields differ")
    end
    need(admission["evidence_mode"]=="actual_archival_cache" && admission["all_artifact_bytes_recomputed"]===true &&
        admission["provider_receipts_available"]===false && isint(admission["actual_fits"]) && admission["actual_fits"]==0 &&
        admission["request_sha256"]==cases.request_sha256 && admission["artifact_manifest_sha256"]==cases.manifest_sha256 &&
        admission["artifact_complete_sha256"]==cases.complete_sha256,"archival case read admission differs")
    for key in ("request_sha256","artifact_manifest_sha256","artifact_complete_sha256");J._sha(admission[key]);end
    admission
end
function _data_binding(value,index,recipe,start,stop,asof)
    _archival_request(value,index,start,stop,asof)
    arrays=Q.split_arrays(value.cases,index;start,stop,asof)
    input,_=_data_sha(arrays)
    _ArchivalBinding("{}",digest(B.admission_bytes(value)),input)
end
function prepare_expected(value::B.ReverifiedCases,index,recipe::PointRecipe;start,stop,asof)
    binding=_data_binding(value,index,recipe,start,stop,asof)
    p=_prepare(value.cases,index,recipe,start,stop,asof,binding)
    ExpectedFit(p.context_sha256,recipe.sha256,p.source,LIBRARY_SHA)
end
function _admit_archival(value::B.ReverifiedCases,index,recipe::PointRecipe,expected::ExpectedFit,start,stop,asof,authority=nothing)
    need(authority===nothing,"execution certificates are not accepted; supply expected fit and input pins")
    binding=_data_binding(value,index,recipe,start,stop,asof)
    prepared=_prepare(value.cases,index,recipe,start,stop,asof,binding);_matches(expected,prepared,recipe)
    need(BLAS.get_num_threads()==1,"archival point fit requires one BLAS thread")
    receipt=Dict("schema"=>"archive_point_fit_binding_v2","case_read_admission_sha256"=>binding.admission_sha256,
        "input_sha256"=>binding.input_sha256,"context_sha256"=>expected.context_sha256,"recipe_sha256"=>recipe.sha256,
        "source_sha256"=>expected.source_sha256,"library_sha256"=>LIBRARY_SHA)
    _ArchivalBinding(String(R.canonical_bytes(receipt;limit=16384)),binding.admission_sha256,binding.input_sha256)
end
"""Fit one independently pinned recipe on reverified actual cases and a strict2017 split."""
function fit_archival(value::B.ReverifiedCases,index,recipe::PointRecipe,expected::ExpectedFit;start,stop,asof,authority=nothing)
    binding=_admit_archival(value,index,recipe,expected,start,stop,asof,authority)
    result=_fit_prepared(value.cases,index,recipe,expected,start,stop,asof,_native_fit,binding)
    after=_admit_archival(value,index,recipe,expected,start,stop,asof,authority)
    need(after.json==binding.json && after.admission_sha256==binding.admission_sha256 &&
        after.input_sha256==binding.input_sha256,"archival fit inputs changed during fitting")
    result
end
