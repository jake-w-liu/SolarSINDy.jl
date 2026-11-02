"""Pure known-metadata formatting shared by production and untrusted previews.

No CaseReader, model codec, feature/label body, ExpectedFit or authority token
is accessed here. A descriptor is an intent binding, never an admission.
"""
const DESCRIPTOR_FACT_KEYS=("operation","case_request_sha256","case_manifest_sha256","case_complete_sha256",
    "table_index","station","role","candidate_id","recipe_sha256","grid_sha256","split","runtime",
    "policy_sha256","job_sha256","point_source_sha256","actual_source_access","case_read_admission_sha256",
    "selection_sha256","point")
function descriptor_from_metadata(facts)
    need(facts isa AbstractDict && Set(keys(facts))==Set(DESCRIPTOR_FACT_KEYS),"descriptor known metadata fields")
    need(facts["operation"] in ("fixed_candidate_2017_fit_input_only","fixed_candidate_2017_validation_only",
        "selected_R2_2017_refit_input_only"),"descriptor fixed operation")
    need(F.isint(facts["table_index"]) && 1<=facts["table_index"]<=16 && facts["station"] in ("CMO","FRD") &&
        facts["role"] in ("primary","independent"),"descriptor table/station/role")
    need(facts["actual_source_access"] isa Bool && facts["runtime"] in ("1.12.6","1.13.1"),"descriptor mode/runtime")
    for name in ("case_request_sha256","case_manifest_sha256","case_complete_sha256","recipe_sha256",
                 "grid_sha256","policy_sha256","job_sha256","point_source_sha256")
        J._sha(facts[name])
    end
    recipe=recipe_for(facts["candidate_id"])
    need(recipe.sha256==facts["recipe_sha256"] && facts["grid_sha256"]==M.grid_identity(:R2),"descriptor frozen candidate/recipe/grid")
    for name in ("case_read_admission_sha256","selection_sha256")
        facts[name]===nothing || J._sha(facts[name])
    end
    split=facts["split"];J._object(split,Set(("start_inclusive","stop_exclusive","target_asof","first_evaluation_issue")))
    start,stop,asof,first_issue=(F._fit_clock(split[k]) for k in ("start_inclusive","stop_exclusive","target_asof","first_evaluation_issue"))
    need(start<stop<=asof && stop<=first_issue,"descriptor split clocks")
    point=facts["point"]
    if point!==nothing
        J._object(point,Set(("context_sha256","recipe_sha256","model_sha256","receipt_sha256","source_sha256")))
        for value in values(point);J._sha(value);end
        need(point["recipe_sha256"]==recipe.sha256,"descriptor point recipe")
    end
    # Keep the original descriptor's exact field values and fixed budgets.
    result=Dict{String,Any}("schema"=>"candidate_known_input_descriptor_v1","layout"=>"ground11",
        "phase_period_minutes"=>31,"work_budget"=>work_budget())
    for key in DESCRIPTOR_FACT_KEYS;result[key]=deepcopy(facts[key]);end
    result
end
