"""Strict untrusted selection metadata and zero-fit point bindings.

These values never verify candidate producers or grant actual selected fitting,
calibration or replay. No public actual-operation API is supplied here.
"""
module ArchiveSelectedPointBindings
using Dates,SHA
import ..ArchiveSourceBoundPointFit, ..ArchiveClassicPointModels, ..ArchiveLockedMixed8Ridge
import ..ArchiveCatalogSourceJob, ..ArchiveCatalogSnapshotFilesystem, ..CompletedSourceDBDT
import ..CompletedSelectionPrimitives
const F=ArchiveSourceBoundPointFit
const C=ArchiveClassicPointModels
const L=ArchiveLockedMixed8Ridge
const J=ArchiveCatalogSourceJob
const FS=ArchiveCatalogSnapshotFilesystem
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const S=CompletedSelectionPrimitives
export SelectionRequest,ParsedSelectionDecision,PrimaryRateChoice,SensitivityRateChoice,
       SelectedPointBinding,MetadataPins,encode_request,decode_request,encode_decision,
       decode_decision,encode_binding,decode_binding,bind_fixed_point,
       bind_generated_selected_point,publish_metadata,read_request,read_decision,read_binding,
       grid_identity,recipe_for_candidate
const REQUEST_SCHEMA="selected_source_bound_selection_request_data_v1"
const DECISION_SCHEMA="selected_source_bound_selection_decision_data_v1"
const BINDING_SCHEMA="selected_point_binding_data_v1"
const MAX_REQUEST=131072
const MAX_DECISION=262144
const MAX_BINDING=16384
const SHAS=("context_sha256","recipe_sha256","model_sha256","receipt_sha256","source_sha256")
const EVIDENCE_SHAS=("case_request_sha256","case_manifest_sha256","case_complete_sha256","cohort_sha256",
    "input_sha256","point_context_sha256","point_recipe_sha256","point_model_sha256","point_receipt_sha256",
    "point_source_sha256","fit_grant_sha256","fit_attempt_sha256","fit_source_proof_sha256","fit_qualification_sha256",
    "validation_payload_sha256","validation_native_receipt_sha256","validation_job_sha256",
    "validation_source_proof_sha256","validation_qualification_sha256")
const GAMMA_SHAS=("calibration_context_sha256","calibration_input_sha256","calibration_receipt_sha256",
    "calibration_grant_sha256","calibration_attempt_sha256","event_source_sha256","subscriber_output_sha256")
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
function object(value,fields,message)
    need(value isa AbstractDict && length(value)==length(fields) && Set(keys(value))==Set(fields),message);value
end
function array(value,count,message)
    need(value isa AbstractVector && length(value)==count,message);value
end
function text(value,limit,message)
    need(value isa AbstractString && 0<ncodeunits(value)<=limit,message);String(value)
end
function sha(value)
    need(value isa AbstractString && ncodeunits(value)==64 && occursin(r"\A[0-9a-f]{64}\z",value),"selection SHA256 syntax");String(value)
end
function count(value,minimum=0,maximum=600000)
    need(value isa Integer && !(value isa Bool) && minimum<=value<=maximum,"selection integer bound");Int(value)
end
bytes(value;limit)=R.canonical_bytes(value;limit)
decoded(value;limit)=R.decode_canonical(value;limit)
function _kind(value)
    need(value isa AbstractString && value in ("gamma_primary","fifo_clock31_sensitivity","R2"),"selection kind");Symbol(value)
end
function recipe_for_candidate(id)
    index=findfirst(spec->spec.id==id,S.POINT_GRID);need(index!==nothing,"unknown selected candidate")
    spec=S.POINT_GRID[index]
    C.Recipe(;kind=spec.family,depth=spec.depth,trees=spec.trees,penalty=spec.penalty)
end
function _grid(kind)
    kind===:R2 && return [Dict("id"=>x.id,"family"=>String(x.family),"depth"=>x.depth,
        "trees"=>x.trees,"penalty_bits"=>C.bits(x.penalty),"recipe_sha256"=>recipe_for_candidate(x.id).sha256) for x in S.POINT_GRID]
    schedules=kind===:gamma_primary ? (:phase,:fifo) : (:fifo_clock31_sensitivity,)
    [Dict("schedule"=>String(schedule),"grid"=>grid,"gamma_bits"=>C.bits(gamma))
        for schedule in schedules for grid in S.grid_names(schedule) for gamma in S.grid_values(grid)]
end
grid_identity(kind::Symbol)=begin need(kind in (:gamma_primary,:fifo_clock31_sensitivity,:R2),"selection grid kind");digest(bytes(_grid(kind);limit=MAX_REQUEST)) end
function _splits(kind)
    ranges=kind===:R2 ? (("fit","2017-01-01","2017-07-01"),("selection","2017-07-01","2017-10-01"),
        ("refit","2017-01-01","2017-10-01"),("final_calibration","2017-10-01","2018-01-01")) :
        (("fit","2017-01-01","2017-10-01"),("score","2017-10-01","2017-11-01"),("validation","2017-11-01","2018-01-01"))
    Dict(name=>Dict("start_inclusive"=>start*"T00:00:00.000Z","stop_exclusive"=>stop*"T00:00:00.000Z",
        "label_ready_strictly_before"=>stop*"T00:00:00.000Z") for (name,start,stop) in ranges)
end
function _keys(kind,stations)
    kind===:R2 && return [spec.id for spec in S.POINT_GRID]
    [row["schedule"]*"/"*station*"/"*row["grid"]*"/"*row["gamma_bits"] for row in _grid(kind) for station in stations]
end
function _request(value)
    object(value,("schema","kind","protocol","policy_sha256","phase_period_minutes","nominal_bits","stations","layout",
        "splits","grid_sha256","cohort_expectations","source_case_pins","producer_expectations"),"selection request fields")
    need(value["schema"]==REQUEST_SCHEMA,"selection request schema");kind=_kind(value["kind"])
    protocol=kind===:R2 ? "completed_source_R2_2017_v1" : kind===:gamma_primary ? "completed_source_gamma_primary_2017_v1" : "completed_source_FIFO31_sensitivity_2017_v1"
    need(value["protocol"]==protocol && count(value["phase_period_minutes"],31,31)==31 &&
        value["nominal_bits"]==C.bits(.1),"selection protocol or nominal")
    sha(value["policy_sha256"]);need(value["grid_sha256"]==grid_identity(kind),"selection fixed grid differs")
    need(bytes(value["splits"];limit=4096)==bytes(_splits(kind);limit=4096),"selection strict2017 splits differ")
    stations=value["stations"];array(stations,kind===:R2 ? 1 : 2,"selection station inventory")
    need(kind===:R2 ? stations[1] in S.STATIONS : Tuple(stations)==S.STATIONS,"selection station order")
    need(value["layout"]==(kind===:R2 ? "ground11" : "mixed8"),"selection feature layout")
    cohorts=array(value["cohort_expectations"],length(stations),"selection cohort inventory")
    cases=array(value["source_case_pins"],length(stations),"selection case inventory")
    for (station,cohort,case) in zip(stations,cohorts,cases)
        object(cohort,("station","cohort_sha256","n","normalization_bits"),"selection cohort fields")
        need(cohort["station"]==station,"selection cohort station");sha(cohort["cohort_sha256"]);count(cohort["n"],1)
        kind===:R2 ? need(cohort["normalization_bits"]===nothing,"R2 has no pinball normalizer") :
            need(C.unbits(cohort["normalization_bits"])>0,"selection positive normalizer")
        object(case,("station","role","table_index","request_sha256","manifest_sha256","complete_sha256"),"selection case fields")
        need(case["station"]==station && case["role"] in ("primary","independent"),"selection case station or role")
        count(case["table_index"],1,16);for name in ("request_sha256","manifest_sha256","complete_sha256");sha(case[name]);end
    end
    need(all(case->case["role"]==cases[1]["role"],cases),"selection roles differ")
    keys=_keys(kind,stations);evidence=array(value["producer_expectations"],length(keys),"selection producer inventory")
    for (key,entry) in zip(keys,evidence)
        fields=kind===:R2 ? ("candidate_key",EVIDENCE_SHAS...) : ("candidate_key","origin_station",EVIDENCE_SHAS...,GAMMA_SHAS...)
        object(entry,fields,"selection producer fields");need(entry["candidate_key"]==key,"selection producer order/key")
        for field in EVIDENCE_SHAS;sha(entry[field]);end
        station=kind===:R2 ? stations[1] : split(key,'/')[2]
        index=findfirst(==(station),stations);case=cases[index];cohort=cohorts[index]
        need(entry["case_request_sha256"]==case["request_sha256"] && entry["case_manifest_sha256"]==case["manifest_sha256"] &&
            entry["case_complete_sha256"]==case["complete_sha256"] && entry["cohort_sha256"]==cohort["cohort_sha256"],"selection producer case/cohort")
        if kind===:R2
            need(entry["point_recipe_sha256"]==recipe_for_candidate(key).sha256,"selection candidate recipe")
        else
            need(entry["origin_station"]==station && entry["point_recipe_sha256"]==L.Recipe().sha256,"selection L8 origin recipe")
            for field in GAMMA_SHAS;sha(entry[field]);end
        end
    end
    bytes(value;limit=MAX_REQUEST);kind
end
struct SelectionRequest
    kind::Symbol
    json::String
    sha256::String
    function SelectionRequest(value)
        kind=_request(value);raw=bytes(value;limit=MAX_REQUEST);new(kind,String(copy(raw)),digest(raw))
    end
end
encode_request(value::SelectionRequest)=Vector{UInt8}(codeunits(value.json))
function decode_request(raw;expected_sha256)
    need(raw isa AbstractVector{UInt8} && length(raw)<=MAX_REQUEST && digest(raw)==sha(expected_sha256),"selection request outside pin/bound")
    value=SelectionRequest(decoded(copy(raw);limit=MAX_REQUEST));need(encode_request(value)==raw,"selection request reconstruction");value
end
function _point(value)
    object(value,("station","layout","family",SHAS...),"selection point descriptor fields")
    need(value["station"] in S.STATIONS && value["layout"] in ("mixed8","ground11") &&
        value["family"] in (L.FAMILY,"ridge","boost","forest"),"selection point descriptor domain")
    for key in SHAS;sha(value[key]);end;value
end
function _report(value,kind,request)
    if kind===:R2
        object(value,("candidate_key","station","candidate_id","family","cohort_sha256","n","selection_log_mse_bits"),"selection R2 report fields")
        need(value["candidate_key"]==value["candidate_id"],"selection R2 report key")
        need(value["family"] in ("ridge","boost","forest"),"selection family")
        return S.PointMetrics(;station=value["station"],candidate_id=value["candidate_id"],family=Symbol(value["family"]),
            cohort=sha(value["cohort_sha256"]),n=value["n"],selection_log_mse=C.unbits(value["selection_log_mse_bits"]),phase_period_minutes=31)
    end
    object(value,("candidate_key","station","schedule","grid","gamma_bits","cohort_sha256","n","normalization_bits",
        "normalized_pinball_bits","full_count","empty_count","positive_inf_count","negative_inf_count","finite_negative_count","nan_count"),"selection gamma report fields")
    need(value["schedule"] in ("phase","fifo","fifo_clock31_sensitivity"),"selection schedule")
    schedule=Symbol(value["schedule"]);gamma=C.unbits(value["gamma_bits"])
    need(value["candidate_key"]==String(schedule)*"/"*String(value["station"])*"/"*String(value["grid"])*"/"*String(value["gamma_bits"]),"selection gamma report key")
    loss=value["normalized_pinball_bits"]===nothing ? nothing : C.unbits(value["normalized_pinball_bits"])
    S.GammaMetrics(;station=value["station"],schedule,grid=value["grid"],gamma,cohort=sha(value["cohort_sha256"]),n=value["n"],
        normalization=C.unbits(value["normalization_bits"]),normalized_pinball=loss,full_count=value["full_count"],empty_count=value["empty_count"],
        positive_inf_count=value["positive_inf_count"],negative_inf_count=value["negative_inf_count"],finite_negative_count=value["finite_negative_count"],nan_count=value["nan_count"],phase_period_minutes=31)
end
function _pure(request,reports)
    kind=_kind(request["kind"]);cohorts=request["cohort_expectations"]
    if kind===:R2
        c=only(cohorts);expected=S.PointExpectation(;station=c["station"],cohort=c["cohort_sha256"],n=c["n"],phase_period_minutes=31)
        result=S.choose_R2([_report(row,kind,request) for row in reports],expected)
        return Dict("R2"=>Dict("family_winners"=>[Dict("family"=>String(row.family),"candidate_id"=>row.candidate_id,"selection_log_mse_bits"=>C.bits(row.selection_log_mse)) for row in result.family_winners],
            "primary_family"=>String(result.primary_family),"primary_candidate_id"=>result.primary_candidate_id,"input_sha256"=>result.input_sha256,"decision_sha256"=>result.decision_sha256))
    end
    rows=[_report(row,kind,request) for row in reports];out=Dict{String,Any}()
    schedules=kind===:gamma_primary ? (:phase,:fifo) : (:fifo_clock31_sensitivity,)
    for schedule in schedules
        expected=S.GammaExpectation(schedule;cohorts=Tuple(c["cohort_sha256"] for c in cohorts),counts=Tuple(c["n"] for c in cohorts),
            normalizations=Tuple(C.unbits(c["normalization_bits"]) for c in cohorts),phase_period_minutes=31)
        result=S.choose_gamma(filter(row->row.schedule===schedule,rows),expected)
        out[String(schedule)]=Dict("selected_grid"=>result.selected_grid,"gamma_bits"=>C.bits(result.gamma),"objective_bits"=>C.bits(result.objective),
            "input_sha256"=>result.input_sha256,"decision_sha256"=>result.decision_sha256,"primary_selection"=>result.primary_selection,"sensitivity_only"=>result.sensitivity_only)
    end;out
end
function _decision(value)
    object(value,("schema","request","request_sha256","candidate_reports","candidate_evidence","origin_points","origin_calibrations","pure_decisions",
        "producer_evidence_verified","actual_authority_granted"),"selection decision fields")
    need(value["schema"]==DECISION_SCHEMA && value["producer_evidence_verified"]===false && value["actual_authority_granted"]===false,"selection parsed status")
    request=SelectionRequest(value["request"]);need(value["request_sha256"]==request.sha256,"selection decision request pin")
    req=value["request"];keys=_keys(request.kind,req["stations"])
    reports=array(value["candidate_reports"],length(keys),"selection report inventory")
    for row in reports;_report(row,request.kind,req);end
    need(length(unique(row["candidate_key"] for row in reports))==length(keys) && Set(row["candidate_key"] for row in reports)==Set(keys),"selection report keys")
    need(bytes(value["candidate_evidence"];limit=MAX_REQUEST)==bytes(req["producer_expectations"];limit=MAX_REQUEST),"selection candidate evidence differs")
    if request.kind===:R2
        array(value["origin_points"],0,"R2 decision has no gamma origin");array(value["origin_calibrations"],0,"R2 decision has no gamma scores")
    else
        points=array(value["origin_points"],2,"selection L8 origin inventory");calibrations=array(value["origin_calibrations"],2,"selection October origin inventory")
        for (station,p,c) in zip(S.STATIONS,points,calibrations)
            _point(p);need(p["station"]==station && p["layout"]=="mixed8" && p["family"]==L.FAMILY && p["recipe_sha256"]==L.Recipe().sha256,"selection L8 origin")
            object(c,("station","context_sha256","input_sha256","receipt_sha256","point_context_sha256","point_model_sha256"),"selection origin calibration fields")
            need(c["station"]==station && c["point_context_sha256"]==p["context_sha256"] && c["point_model_sha256"]==p["model_sha256"],"selection origin calibration point")
            for name in ("context_sha256","input_sha256","receipt_sha256");sha(c[name]);end
            for e in req["producer_expectations"]
                e["origin_station"]==station || continue
                need(all(e["point_"*field]==p[field] for field in SHAS) && e["calibration_context_sha256"]==c["context_sha256"] &&
                    e["calibration_input_sha256"]==c["input_sha256"] && e["calibration_receipt_sha256"]==c["receipt_sha256"],"selection origin evidence differs")
            end
        end
    end
    need(bytes(value["pure_decisions"];limit=16384)==bytes(_pure(req,reports);limit=16384),"selection pure decision reconstruction differs")
    bytes(value;limit=MAX_DECISION);request.kind
end
struct ParsedSelectionDecision
    kind::Symbol
    json::String
    sha256::String
    function ParsedSelectionDecision(value)
        kind=_decision(value);raw=bytes(value;limit=MAX_DECISION);new(kind,String(copy(raw)),digest(raw))
    end
end
function ParsedSelectionDecision(request::SelectionRequest,reports;origin_points=Any[],origin_calibrations=Any[])
    req=decoded(encode_request(request);limit=MAX_REQUEST)
    ParsedSelectionDecision(Dict("schema"=>DECISION_SCHEMA,"request"=>req,"request_sha256"=>request.sha256,"candidate_reports"=>reports,
        "candidate_evidence"=>req["producer_expectations"],"origin_points"=>origin_points,"origin_calibrations"=>origin_calibrations,
        "pure_decisions"=>_pure(req,reports),"producer_evidence_verified"=>false,"actual_authority_granted"=>false))
end
encode_decision(value::ParsedSelectionDecision)=Vector{UInt8}(codeunits(value.json))
function decode_decision(raw;expected_sha256)
    need(raw isa AbstractVector{UInt8} && length(raw)<=MAX_DECISION && digest(raw)==sha(expected_sha256),"selection decision outside pin/bound")
    value=ParsedSelectionDecision(decoded(copy(raw);limit=MAX_DECISION));need(encode_decision(value)==raw,"selection decision reconstruction");value
end
struct PrimaryRateChoice
    decision::ParsedSelectionDecision
    function PrimaryRateChoice(decision::ParsedSelectionDecision)
        need(decision.kind===:gamma_primary,"primary rate choice requires primary gamma decision");new(decision)
    end
end
struct SensitivityRateChoice
    decision::ParsedSelectionDecision
    function SensitivityRateChoice(decision::ParsedSelectionDecision)
        need(decision.kind===:fifo_clock31_sensitivity,"sensitivity rate choice requires separate FIFO31 decision");new(decision)
    end
end
include("selected_point_binding_values.jl")
include("selected_metadata_publication.jl")
end
