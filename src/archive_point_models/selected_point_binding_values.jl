function _point_descriptor(point::F.BoundFit)
    checked=F._check_capsule(point);context=checked.context
    family=point.model isa L.NumericModel ? L.FAMILY : point.model isa C.NumericModel ? String(point.model.recipe.kind) : nothing
    need(family!==nothing,"selected metadata supports locked L8 or retained R2 point values")
    Dict("station"=>context["table"]["station"],"layout"=>(context["layout"]=="mixed" ? "mixed8" : "ground11"),
        "family"=>family,"context_sha256"=>point.expected.context_sha256,"recipe_sha256"=>point.expected.recipe_sha256,
        "model_sha256"=>digest(checked.model_bytes),"receipt_sha256"=>digest(checked.receipt_bytes),"source_sha256"=>point.expected.source_sha256)
end
function _rate_value(rates::PrimaryRateChoice)
    d=decoded(encode_decision(rates.decision);limit=MAX_DECISION);p=d["pure_decisions"];request=d["request"]
    Dict("kind"=>"primary_L8_joint_CMO_FRD","phase_grid"=>p["phase"]["selected_grid"],"phase_gamma_bits"=>p["phase"]["gamma_bits"],
        "fifo_grid"=>p["fifo"]["selected_grid"],"fifo_gamma_bits"=>p["fifo"]["gamma_bits"],"decision_sha256"=>rates.decision.sha256,
        "policy_sha256"=>request["policy_sha256"],"phase_period_minutes"=>31)
end
function _origin_values(rates::PrimaryRateChoice)
    d=decoded(encode_decision(rates.decision);limit=MAX_DECISION)
    [Dict("point"=>point,"calibration_context_sha256"=>cal["context_sha256"],"calibration_input_sha256"=>cal["input_sha256"],
        "calibration_receipt_sha256"=>cal["receipt_sha256"]) for (point,cal) in zip(d["origin_points"],d["origin_calibrations"])]
end
function _validate_binding(value)
    object(value,("schema","point_choice","primary_rate_choice","gamma_origin","deployment_point","transfer_policy",
        "producer_evidence_verified","actual_authority_granted","binding_performed_new_fit"),"selected binding fields")
    need(value["schema"]==BINDING_SCHEMA && value["producer_evidence_verified"]===false && value["actual_authority_granted"]===false &&
        value["binding_performed_new_fit"]===false,"selected binding grants no work or verified evidence")
    rate=value["primary_rate_choice"]
    object(rate,("kind","phase_grid","phase_gamma_bits","fifo_grid","fifo_gamma_bits","decision_sha256","policy_sha256","phase_period_minutes"),"selected primary rate fields")
    need(rate["kind"]=="primary_L8_joint_CMO_FRD" && count(rate["phase_period_minutes"],31,31)==31,"selected primary rate type/policy")
    sha(rate["decision_sha256"]);sha(rate["policy_sha256"])
    need(rate["phase_grid"]=="phase_declared" && C.unbits(rate["phase_gamma_bits"]) in S.GAMMAS &&
        rate["fifo_grid"] in S.grid_names(:fifo) && C.unbits(rate["fifo_gamma_bits"]) in S.grid_values(rate["fifo_grid"]),"selected primary gamma grid")
    origins=array(value["gamma_origin"],2,"selected gamma origin inventory")
    for (station,origin) in zip(S.STATIONS,origins)
        object(origin,("point","calibration_context_sha256","calibration_input_sha256","calibration_receipt_sha256"),"selected origin fields")
        p=_point(origin["point"]);need(p["station"]==station && p["layout"]=="mixed8" && p["family"]==L.FAMILY,"selected L8 origin descriptor")
        for field in ("calibration_context_sha256","calibration_input_sha256","calibration_receipt_sha256");sha(origin[field]);end
    end
    deployment=_point(value["deployment_point"]);choice=value["point_choice"]
    object(choice,("kind","candidate_id","decision_sha256"),"selected point choice fields")
    if value["transfer_policy"]=="locked_mixed8_identity_v1"
        need(choice["kind"]=="fixed_locked_mixed8" && choice["candidate_id"]=="locked-mixed8-penalty1" && choice["decision_sha256"]===nothing,"selected fixed point choice")
        origin=origins[deployment["station"]=="CMO" ? 1 : 2]["point"]
        need(deployment==origin,"selected identity deployment differs from gamma-origin point")
    else
        need(value["transfer_policy"]=="primary_mixed8_to_ground11_v1" && deployment["layout"]=="ground11" &&
            deployment["family"] in ("ridge","boost","forest"),"selected primary transfer policy")
        sha(choice["decision_sha256"])
        need(choice["kind"]=="selected_R2_ground11","selected R2 point choice type")
        index=findfirst(spec->spec.id==choice["candidate_id"],S.POINT_GRID)
        need(index!==nothing && String(S.POINT_GRID[index].family)==deployment["family"],"selected R2 point choice family")
    end
    bytes(value;limit=MAX_BINDING);value
end
function _binding_value(point::F.BoundFit,rates::PrimaryRateChoice,point_choice)
    checked=F._check_capsule(point);deployment=_point_descriptor(point)
    rate_value=_rate_value(rates)
    choice=if point_choice===nothing
        need(point.model isa L.NumericModel && point.model.recipe.json==L.Recipe().json,"fixed selected binding requires locked mixed8 ridge")
        Dict("kind"=>"fixed_locked_mixed8","candidate_id"=>"locked-mixed8-penalty1","decision_sha256"=>nothing)
    else
        need(point_choice isa ParsedSelectionDecision && point_choice.kind===:R2 && !checked.actual &&
            point.model isa C.NumericModel && point.model.mode===:ground11,"generated R2 binding requires generated ground11 point and R2 metadata")
        d=decoded(encode_decision(point_choice);limit=MAX_DECISION);req=d["request"]
        need(only(req["stations"])==deployment["station"] && req["policy_sha256"]==rate_value["policy_sha256"],"selected R2 decision station/policy differs")
        c=only(req["source_case_pins"]);pc=checked.context
        need(c["request_sha256"]==pc["case_request_sha256"] && c["manifest_sha256"]==pc["case_manifest_sha256"] &&
            c["complete_sha256"]==pc["case_complete_sha256"] && c["table_index"]==pc["table_index"] && c["role"]==pc["role"],"selected generated R2 case binding differs")
        family=deployment["family"];winners=d["pure_decisions"]["R2"]["family_winners"]
        winner=only(filter(w->w["family"]==family,winners));candidate=winner["candidate_id"]
        need(point.model.recipe.json==recipe_for_candidate(candidate).json,"selected generated point recipe differs from retained family winner")
        Dict("kind"=>"selected_R2_ground11","candidate_id"=>candidate,"decision_sha256"=>point_choice.sha256)
    end
    _validate_binding(Dict("schema"=>BINDING_SCHEMA,"point_choice"=>choice,"primary_rate_choice"=>rate_value,
        "gamma_origin"=>_origin_values(rates),"deployment_point"=>deployment,
        "transfer_policy"=>(point_choice===nothing ? "locked_mixed8_identity_v1" : "primary_mixed8_to_ground11_v1"),
        "producer_evidence_verified"=>false,"actual_authority_granted"=>false,"binding_performed_new_fit"=>false))
end
struct _BindingToken end
"""Untrusted immutable derived metadata; never an actual selected-operation permit."""
struct SelectedPointBinding
    point::F.BoundFit
    rates::PrimaryRateChoice
    point_choice::Union{Nothing,ParsedSelectionDecision}
    json::String
    sha256::String
    function SelectedPointBinding(point,rates,point_choice,value,::_BindingToken)
        raw=bytes(value;limit=MAX_BINDING);new(point,rates,point_choice,String(copy(raw)),digest(raw))
    end
end
"""Bind the exact L8 origin point without targets, predictions or another fit."""
function bind_fixed_point(point::F.BoundFit,rates::PrimaryRateChoice)
    value=_binding_value(point,rates,nothing);SelectedPointBinding(point,rates,nothing,value,_BindingToken())
end
"""Bind generated R2 values under the declared primary L8 rate transfer.

This does not perform a selected refit or verify candidate producers. Actual
selected-refit/calibration/replay entry points remain unavailable.
"""
function bind_generated_selected_point(point::F.BoundFit,rates::PrimaryRateChoice,choice::ParsedSelectionDecision)
    value=_binding_value(point,rates,choice);SelectedPointBinding(point,rates,choice,value,_BindingToken())
end
function encode_binding(value::SelectedPointBinding)
    raw=bytes(_binding_value(value.point,value.rates,value.point_choice);limit=MAX_BINDING)
    need(raw==Vector{UInt8}(codeunits(value.json)) && digest(raw)==value.sha256,"selected binding changed");raw
end
function decode_binding(raw;expected_sha256,point::F.BoundFit,rates::PrimaryRateChoice,point_choice=nothing)
    need(raw isa AbstractVector{UInt8} && length(raw)<=MAX_BINDING && digest(raw)==sha(expected_sha256),"selected binding outside pin/bound")
    _validate_binding(decoded(copy(raw);limit=MAX_BINDING))
    value=point_choice===nothing ? bind_fixed_point(point,rates) : bind_generated_selected_point(point,rates,point_choice)
    need(encode_binding(value)==raw,"selected binding outside model/decision reconstruction differs");value
end
