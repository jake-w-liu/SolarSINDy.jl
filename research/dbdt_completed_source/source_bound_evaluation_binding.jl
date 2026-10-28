"""Explicit generated-only connection from frozen training to new evaluation data."""
const EVALUATION_BINDING_NAME="research/dbdt_completed_source/source_bound_evaluation_binding.jl"
const EVALUATION_BINDING_FILES=Tuple(sort([
    "research/dbdt_completed_source/source_bound_shared_replay.jl",EVALUATION_BINDING_NAME,
    "research/dbdt_completed_source/source_bound_calibration.jl",
    "research/dbdt_completed_source/source_bound_calibration_codec.jl",
    "research/dbdt_completed_source/source_bound_static_bounds.jl",
    "research/dbdt_completed_source/publisher_up_projection.jl",
    "research/dbdt_completed_source/publisher_up_values.jl",
    "research/dbdt_completed_source/retained_publisher_up.jl"]))
const EVALUATION_BINDING_PINS=Tuple((name,digest(read(joinpath(pkgdir(SolarSINDy),split(name,'/')...)))) for name in EVALUATION_BINDING_FILES)

"""Externally pinned metadata connection; grants no actual-source authority."""
struct ExpectedEvaluation
    context_sha256::String
    ExpectedEvaluation(value)=new(SolarSINDy.ArchiveCatalogSourceJob._sha(value))
end
function _evaluation_job_identity()
    J=SolarSINDy.ArchiveCatalogSourceJob;root=realpath(pkgdir(SolarSINDy))
    need(J._same(realpath(@__FILE__),joinpath(root,split(EVALUATION_BINDING_NAME,'/')...)),"evaluation binding outside loaded package")
    files=Any[]
    for (name,pin) in EVALUATION_BINDING_PINS
        need(digest(read(joinpath(root,split(name,'/')...)))==pin,"evaluation binding code changed")
        push!(files,Dict("path"=>name,"sha256"=>pin))
    end
    digest(R.canonical_bytes(Dict("schema"=>"generated_replay_evaluation_job_closure_v1","files"=>files);limit=32768))
end
function _evaluation_context(value,index,point,calibration,expected,start,stop,asof,pc,kc)
    J=SolarSINDy.ArchiveCatalogSourceJob
    need(value.actual_source_authority_granted===false && value.actual_fit_performed===false && !pc.actual && !kc.actual,
        "generated evaluation binding rejects actual source and fitted authority")
    need(F.isint(index) && 1<=index<=length(value.features),"evaluation binding table index")
    need(all(t->t isa DateTime && D.whole(t),(start,stop,asof)) && start<stop<=asof,"evaluation binding whole-minute split")
    request=R.decode_canonical(Vector{UInt8}(codeunits(value.request_json));limit=131072)
    need(digest(codeunits(value.request_json))==value.request_sha256 && request["schema"]=="generated_catalog_case_request_v1" &&
        request["evidence_mode"]=="generated" && request["runtime"]==string(VERSION),"evaluation binding case request differs")
    table=request["tables"][index];features=value.features[index]
    need(pc.context["role"]==request["role"] && pc.context["layout"]==request["layout"] &&
        pc.context["table"]["station"]==table["station"]==features.station,"evaluation binding role/layout/station differs")
    need(F._fit_clock(kc.context["stop_exclusive"])<=start && kc.first_issue<=start &&
        J._clock(table["start_inclusive"])<=start<stop<=J._clock(table["stop_exclusive"]) &&
        asof<=value.labels[index].asof,"evaluation binding overlaps calibration or exceeds sealed table")
    mode=pc.context["layout"]=="ground-only" ? :ground11 : :mixed8
    need(features.mode===mode && D.policy_equal(features.policy,D.CMR.ReplayPolicy()),"evaluation binding feature policy/layout differs")
    context=Dict("schema"=>"generated_source_bound_evaluation_binding_v1","evidence_mode"=>"generated","runtime"=>string(VERSION),
        "role"=>request["role"],"layout"=>request["layout"],"job_sha256"=>_evaluation_job_identity(),
        "point"=>Dict("context_sha256"=>point.expected.context_sha256,"model_sha256"=>digest(pc.model_bytes),
            "receipt_sha256"=>digest(pc.receipt_bytes),"recipe_sha256"=>point.expected.recipe_sha256,
            "source_sha256"=>point.expected.source_sha256,"library_sha256"=>point.expected.library_sha256),
        "calibration"=>Dict("context_sha256"=>expected.context_sha256,"input_sha256"=>expected.input_sha256,
            "stop_exclusive"=>kc.context["stop_exclusive"],"first_evaluation_issue"=>kc.context["first_evaluation_issue"]),
        "training_case"=>Dict("request_sha256"=>pc.context["case_request_sha256"],"manifest_sha256"=>pc.context["case_manifest_sha256"],
            "complete_sha256"=>pc.context["case_complete_sha256"],"table_index"=>pc.context["table_index"],"table"=>pc.context["table"]),
        "evaluation_case"=>Dict("request_sha256"=>value.request_sha256,"manifest_sha256"=>value.manifest_sha256,
            "complete_sha256"=>value.complete_sha256,"table_index"=>Int(index),"table"=>table,
            "feature_payload_sha256"=>features.payload_sha256,"source"=>request["source"]),
        "start_inclusive"=>F.clock(start),"stop_exclusive"=>F.clock(stop),"target_asof"=>F.clock(asof),
        "actual_source_authority_granted"=>false,"actual_model_fit_authorized"=>false,
        "measured_receipts_available"=>false,"scientific_coverage_qualified"=>false)
    R.canonical_bytes(context;limit=131072)
end

"""Plan a metadata-only generated evaluation binding without reading target values.

The returned hash is an explicit caller pin, not independent scientific
qualification. Data-only context reconstruction and native tests are separate.
"""
function prepare_evaluation_binding(value::Q.VerifiedCases,index,point,calibration,expected;start,stop,asof)
    pc=F._check_capsule(point);kc=K._checked(calibration,point,expected)
    bytes=_evaluation_context(value,index,point,calibration,expected,start,stop,asof,pc,kc)
    (;context=String(bytes),expected=ExpectedEvaluation(digest(bytes)))
end
function _verify_evaluation_binding(binding,value,index,point,calibration,expected,start,stop,asof,pc,kc)
    need(binding isa ExpectedEvaluation,"separate evaluation requires immutable external context pin")
    bytes=_evaluation_context(value,index,point,calibration,expected,start,stop,asof,pc,kc)
    need(digest(bytes)==binding.context_sha256,"separate evaluation external binding differs")
    binding.context_sha256
end
