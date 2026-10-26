"""Caller-pinned admission for sealed archival feature and label artifacts.

Source reconstruction, feature admission and model fitting are separate grants.
This namespace never opens a source snapshot or authorizes a model fit.
"""
module ArchiveCatalogCaseAuthority
import ..ArchiveCatalogSourceJob
import ..ArchiveCatalogCaseComparison
const J=ArchiveCatalogSourceJob
need(ok,message)=ok ? nothing : throw(ArgumentError(message))

"""Pins for a separately reviewed feature-only grant and its supporting evidence."""
struct Authority
    grant_path::String
    grant_sha256::String
    plan_path::String
    plan_sha256::String
    agreement_path::String
    agreement_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
    floating_policy_path::String
    floating_policy_sha256::String
end

function _pinned(path,pin;limit=262144,canonical=false)
    J._pinned(path,J._sha(pin);limit,canonical)
end

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

function _qualification(proof,qualification,authority,grant)
    need(get(qualification,"complete",false)===true && get(proof,"complete",false)===true &&
        qualification["head"]==proof["head"]==grant["head"] &&
        qualification["source_proof"]["sha256"]==authority.source_proof_sha256 &&
        qualification["software_qualification_only"]===true &&
        qualification["unfiltered_package_and_full_experiments_executed"]===true &&
        qualification["strict_docs_both_complete"]===true &&
        qualification["all_eight_latest_ci_jobs_passed"]===true &&
        qualification["all_eight_raw_logs_retained"]===true &&
        qualification["harness_counts"]["FAIL"] isa Integer &&
        !(qualification["harness_counts"]["FAIL"] isa Bool) && qualification["harness_counts"]["FAIL"]==0,
        "case feature software qualification differs")
    need(proof["head"] isa String && occursin(r"^[0-9a-f]{40}$",proof["head"]),"case feature source head")
    expected=Set(["Julia $v / $s-latest" for v in ("1.12.6","1.13.1") for s in ("ubuntu","macos","windows")])
    union!(expected,Set(("Documentation build","Julia 1.12 deployment images")))
    jobs=qualification["ci_jobs"]
    need(jobs isa AbstractVector && length(jobs)==8 && length(Set(job["job_id"] for job in jobs))==8 &&
        Set(job["name"] for job in jobs)==expected,"case feature eight CI identities differ")
    for job in jobs
        need(job["job_id"] isa Integer && !(job["job_id"] isa Bool) && job["job_id"]>0,"case feature CI job id")
        for key in ("log","receipt","original_log","original_receipt");J._sha(job[key]["sha256"]);end
    end
    # The root grant pins a genuine qualification. The loaded package must also
    # contain this worker and its complete checked dependency inventory.
    root=realpath(pkgdir(parentmodule(@__MODULE__)))
    worker="src/archive_format/ArchiveCatalogCaseAuthority.jl"
    need(J._same(realpath(@__FILE__),joinpath(root,split(worker,'/')...)),"case feature authority outside loaded package")
    for name in (worker,"src/archive_format/ArchiveCatalogCaseJob.jl",
                 "src/archive_format/ArchiveCatalogCaseComparison.jl",
                 "src/archive_format/ArchiveCatalogSnapshotFilesystem.jl",
                 "src/archive_format/ArchiveCatalogSnapshotValues.jl")
        need(haskey(proof["source_sha256"],name),"case feature worker missing from qualification")
    end
    J._source_inventory(proof)
    nothing
end

function _declarations(request,plan,agreement,grant)
    need(plan["schema"]=="completed_catalog_case_artifact_precode_v1" &&
        plan["status"]=="frozen_before_worker_implementation_and_before_any_actual_feature_access" &&
        plan["fits_or_predictions_in_this_job"]===false,"case feature frozen plan differs")
    schedule=plan["schedule_contract"]
    need(schedule["layouts"]==["ground-only","mixed"] &&
        schedule["policy"]=="unchanged default completed-source replay policy" &&
        schedule["all_scheduled_rows_preserved"]===true,"case feature schedule policy differs")
    tables=schedule["tables_per_layout"]
    need(tables isa AbstractVector && length(tables)==8 &&
        request["tables"]==[_project_table(table) for table in tables],
        "case feature locked tables differ")
    need(get(agreement,"complete",false)===true &&
        agreement["all_eight_fresh_native_reconstructions_verified"]===true &&
        agreement["payloads_and_catalog_metadata_equal_across_four_roles_per_layout"]===true &&
        agreement["input_files_mixed"] isa Integer && !(agreement["input_files_mixed"] isa Bool) && agreement["input_files_mixed"]==49 &&
        agreement["input_files_ground_only"] isa Integer && !(agreement["input_files_ground_only"] isa Bool) && agreement["input_files_ground_only"]==32 &&
        agreement["source_agreement_only"]===true &&
        agreement["historical_provider_receipts_available"]===false &&
        agreement["features_labels_eligible_cases_or_models_evaluated"]===false &&
        agreement["actual_fits"] isa Integer && !(agreement["actual_fits"] isa Bool) && agreement["actual_fits"]==0 &&
        agreement["forecast_skill_or_novelty_qualified"]===false,
        "case feature R0 agreement differs")
    need(J._sha(agreement["original_input_plan_sha256"])==grant["source_input_plan_sha256"],"case feature original input plan differs")
    cases=agreement["cases"]
    expected=Set("$runtime-$role-$layout" for runtime in ("1.12.6","1.13.1")
                 for role in ("primary","independent") for layout in ("ground-only","mixed"))
    need(cases isa AbstractDict && Set(keys(cases))==expected,"case feature R0 contexts differ")
    for value in values(cases);J._sha(value["manifest_sha256"]);end
    key=request["runtime"]*"-"*request["role"]*"-"*request["layout"]
    need(cases[key]["manifest_sha256"]==request["source"]["manifest_sha256"],"case feature R0 source manifest differs")
    key
end

"""Validate authority before the case worker reads any source member bodies."""
function authorize(request,request_sha256,authority)
    if request["evidence_mode"]=="generated"
        need(authority===nothing,"generated case jobs cannot consume actual feature authority")
        return nothing
    end
    need(authority isa Authority,"actual case job requires caller-pinned feature authority")
    grant=_pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    J._object(grant,Set(("schema","operation","request_sha256","plan_sha256","agreement_sha256",
        "source_proof_sha256","qualification_sha256","floating_policy_sha256","source_input_plan_sha256","head",
        "actual_source_access_authorized","actual_model_fit_authorized","dataset_discovery","provider_receipts_available")))
    need(grant["schema"]=="archive_catalog_case_feature_grant_v1" && grant["operation"]=="sealed_features_labels_only" &&
        grant["request_sha256"]==J._sha(request_sha256) && grant["plan_sha256"]==authority.plan_sha256 &&
        grant["agreement_sha256"]==authority.agreement_sha256 &&
        grant["source_proof_sha256"]==authority.source_proof_sha256 &&
        grant["qualification_sha256"]==authority.qualification_sha256 &&
        grant["floating_policy_sha256"]==authority.floating_policy_sha256 &&
        grant["actual_source_access_authorized"]===true && grant["actual_model_fit_authorized"]===false &&
        grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false,
        "case feature grant differs")
    J._sha(grant["source_input_plan_sha256"])
    plan=_pinned(authority.plan_path,authority.plan_sha256)
    agreement=_pinned(authority.agreement_path,authority.agreement_sha256)
    key=_declarations(request,plan,agreement,grant)
    floating=_pinned(authority.floating_policy_path,authority.floating_policy_sha256)
    need(get(floating,"complete",false)===true && floating["schema"]=="catalog_case_float_comparison_policy_v1" &&
        floating["artifact_plan_sha256"]==authority.plan_sha256 &&
        floating["R0_agreement_sha256"]==authority.agreement_sha256 &&
        floating["frozen_before_actual_feature_access"]===true &&
        floating["generated_native_oracles_passed"]===true &&
        floating["supported_native_runtimes"]==["1.12.6","1.13.1"] &&
        floating["comparison_implementation_schema"]==ArchiveCatalogCaseComparison.POLICY_SCHEMA &&
        floating["comparison_policy_sha256"]==ArchiveCatalogCaseComparison.policy_digest() &&
        floating["existing_generated_absolute_tolerance"]==1e-13 &&
        floating["existing_generated_relative_tolerance"] isa Integer &&
        !(floating["existing_generated_relative_tolerance"] isa Bool) && floating["existing_generated_relative_tolerance"]==0,
        "case feature pre-actual floating policy differs")
    proof=_pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qualification=_pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    _qualification(proof,qualification,authority,grant)
    Dict("operation"=>grant["operation"],"request_sha256"=>request_sha256,"grant_sha256"=>authority.grant_sha256,
        "plan_sha256"=>authority.plan_sha256,"agreement_sha256"=>authority.agreement_sha256,"R0_context"=>key,
        "source_proof_sha256"=>authority.source_proof_sha256,"qualification_sha256"=>authority.qualification_sha256,
        "floating_policy_sha256"=>authority.floating_policy_sha256,"head"=>proof["head"],"actual_model_fit_authorized"=>false)
end
end
