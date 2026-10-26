"""Pins for a separately approved, fixed-recipe archival point-model fit."""
struct FitAuthority
    grant_path::String
    grant_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end
const _FIT_GRANT_FIELDS=Set(("schema","operation","family","case_read_admission_sha256",
    "case_request_sha256","case_manifest_sha256","case_complete_sha256","context_sha256",
    "recipe_sha256","input_sha256","source_sha256","source_proof_sha256","qualification_sha256",
    "library_sha256","table_index","start_inclusive","stop_exclusive","target_asof","head",
    "actual_source_access_authorized","actual_model_fit_authorized","new_case_generation_authorized",
    "dataset_discovery","provider_receipts_available","expected_model_fits","execution_directory"))
function _archival_request(value::B.ReverifiedCases,index,start,stop,asof)
    need(value.cases.actual_source_authority_granted===true && value.cases.actual_fit_performed===false,"archival fit requires reverified archival cases")
    need(isint(index) && 1<=index<=length(value.cases.features),"archival fit table index")
    need(all(t->t isa DateTime,(start,stop,asof)) && DateTime(2017)<=start<stop<=DateTime(2018) && stop<=asof,
        "archival fitting requires a strict 2017 training split")
    admission=R.decode_canonical(B.admission_bytes(value);limit=131072)
    expected=Set(("schema","evidence_mode","request_sha256","artifact_manifest_sha256","artifact_complete_sha256",
        "producer_head","producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256",
        "reader_head","reader_grant_sha256","reader_source_proof_sha256","reader_qualification_sha256",
        "all_artifact_bytes_recomputed","new_case_generation_authorized","actual_model_fit_authorized",
        "provider_receipts_available","actual_fits"))
    J._object(admission,expected)
    need(admission["schema"]=="archive_catalog_case_read_admission_v1" && admission["evidence_mode"]=="actual_archival_cache" &&
        admission["all_artifact_bytes_recomputed"]===true && admission["new_case_generation_authorized"]===false &&
        admission["actual_model_fit_authorized"]===false && admission["provider_receipts_available"]===false &&
        isint(admission["actual_fits"]) && admission["actual_fits"]==0 &&
        admission["request_sha256"]==value.cases.request_sha256 &&
        admission["artifact_manifest_sha256"]==value.cases.manifest_sha256 &&
        admission["artifact_complete_sha256"]==value.cases.complete_sha256,"archival case read admission differs")
    for key in ("producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256",
                "reader_grant_sha256","reader_source_proof_sha256","reader_qualification_sha256");J._sha(admission[key]);end
    for key in ("producer_head","reader_head")
        need(admission[key] isa String && occursin(r"\A[0-9a-f]{40}\z",admission[key]),"archival read source head")
    end
    admission
end
function _admit_archival(value::B.ReverifiedCases,index,recipe::PointRecipe,expected::ExpectedFit,start,stop,asof,authority)
    need(authority isa FitAuthority,"archival point fit requires a distinct fit authority")
    admission=_archival_request(value,index,start,stop,asof)
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    J._object(grant,_FIT_GRANT_FIELDS)
    execution=J._text(grant["execution_directory"];limit=4096)
    need(isabspath(execution),"archival fit execution directory must be absolute")
    J._ordinary(execution;directory=true)
    need(J._same(realpath(execution),realpath(dirname(authority.grant_path))),"archival fit grant outside its execution directory")
    admission_pin=digest(B.admission_bytes(value))
    need(grant["schema"]=="archive_source_bound_point_fit_grant_v1" && grant["operation"]=="fixed_recipe_2017_point_fit_only" &&
        grant["family"]==family(recipe) && grant["case_read_admission_sha256"]==admission_pin &&
        grant["case_request_sha256"]==value.cases.request_sha256 && grant["case_manifest_sha256"]==value.cases.manifest_sha256 &&
        grant["case_complete_sha256"]==value.cases.complete_sha256 && grant["context_sha256"]==expected.context_sha256 &&
        grant["recipe_sha256"]==recipe.sha256==expected.recipe_sha256 && grant["source_sha256"]==expected.source_sha256 &&
        grant["source_proof_sha256"]==J._sha(authority.source_proof_sha256)==admission["reader_source_proof_sha256"] &&
        grant["qualification_sha256"]==J._sha(authority.qualification_sha256)==admission["reader_qualification_sha256"] &&
        grant["library_sha256"]==expected.library_sha256==LIBRARY_SHA &&
        isint(grant["table_index"]) && grant["table_index"]==index && grant["start_inclusive"]==clock(start) &&
        grant["stop_exclusive"]==clock(stop) && grant["target_asof"]==clock(asof) &&
        grant["actual_source_access_authorized"]===true && grant["actual_model_fit_authorized"]===true &&
        grant["new_case_generation_authorized"]===false && grant["dataset_discovery"]===false &&
        grant["provider_receipts_available"]===false && isint(grant["expected_model_fits"]) && grant["expected_model_fits"]==1,
        "archival point fit grant differs")
    input_pin=J._sha(grant["input_sha256"])
    proof=A._pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qualification=A._pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    A._qualification(proof,qualification,authority,grant)
    need(proof["head"]==admission["reader_head"],"archival fit reader qualification differs")
    root=realpath(pkgdir(ROOT))
    for name in ("src/archive_point_models/ArchiveSourceBoundPointFit.jl",
                 "src/archive_point_models/point_fit_authority.jl","src/archive_point_models/point_fit_publication.jl",
                 "src/archive_point_models/ArchiveClassicPointModels.jl","src/archive_point_models/classic_training.jl",
                 "src/archive_point_models/ArchivePointNumericTrees.jl","src/archive_format/ArchiveCatalogCaseReader.jl")
        need(haskey(proof["source_sha256"],name),"archival point fit worker missing from qualification")
    end
    need(J._same(realpath(@__FILE__),joinpath(root,"src","archive_point_models","point_fit_authority.jl")),"archival fit authority outside loaded package")
    need(source_identity()==expected.source_sha256,"archival fit current source differs")
    need(BLAS.get_num_threads()==1,"archival point fit requires one BLAS thread")
    receipt=Dict("schema"=>"archive_point_fit_admission_v1","operation"=>grant["operation"],"head"=>proof["head"],
        "grant_sha256"=>J._sha(authority.grant_sha256),"source_proof_sha256"=>authority.source_proof_sha256,
        "qualification_sha256"=>authority.qualification_sha256,"case_read_admission_sha256"=>admission_pin,
        "input_sha256"=>input_pin,"actual_model_fit_authorized"=>true,"expected_model_fits"=>1,
        "dataset_discovery"=>false,"provider_receipts_available"=>false)
    _ArchivalBinding(String(R.canonical_bytes(receipt;limit=16384)),admission_pin,input_pin)
end
"""Fit one independently pinned recipe on reverified archival cases.

A distinct fit permit and a genuine qualification of the complete loaded source
are mandatory. Training issues and target readiness stay inside a strict 2017
split. The original case read grants authorize no fit. No source discovery,
provider receipt claim, model selection or external evaluation occurs here.
"""
function fit_archival(value::B.ReverifiedCases,index,recipe::PointRecipe,expected::ExpectedFit;
                      start,stop,asof,authority=nothing)
    binding=_admit_archival(value,index,recipe,expected,start,stop,asof,authority)
    prepared=_prepare(value.cases,index,recipe,start,stop,asof,binding);_matches(expected,prepared,recipe)
    # One fixed path inside the grant-pinned execution directory consumes the
    # permit atomically. A failed or repeated attempt remains spent and retained.
    attempt_path=joinpath(realpath(dirname(authority.grant_path)),"POINT_FIT_ATTEMPT.json")
    attempt=R.canonical_bytes(Dict("schema"=>"archive_point_fit_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "context_sha256"=>expected.context_sha256,"recipe_sha256"=>recipe.sha256,"source_sha256"=>expected.source_sha256,
        "library_sha256"=>expected.library_sha256,"started_utc"=>clock(now(UTC)),"expected_model_fits"=>1);limit=16384)
    R._exclusive_write(attempt_path,attempt)
    result=_fit_prepared(value.cases,index,recipe,expected,start,stop,asof,_native_fit,binding;attempt_sha256=digest(attempt))
    after=_admit_archival(value,index,recipe,expected,start,stop,asof,authority)
    need(after.json==binding.json && after.admission_sha256==binding.admission_sha256 &&
        after.input_sha256==binding.input_sha256,"archival fit admission changed during fitting")
    need(R.bounded_read(attempt_path;limit=16384)==attempt,"archival fit attempt marker changed")
    result
end
