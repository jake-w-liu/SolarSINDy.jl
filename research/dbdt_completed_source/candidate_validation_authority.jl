function _spend_generated(directory,d,kind)
    path=directory*(kind===:input ? ".CANDIDATE_FIT_INPUT_ATTEMPT.json" : ".CANDIDATE_VALIDATION_ATTEMPT.json")
    bytes=canonical(Dict("schema"=>"generated_candidate_operation_attempt_v1","descriptor_sha256"=>digest(canonical(d)),
        "operation"=>d["operation"],"started_utc"=>F.clock(now(UTC)),"actual_source_authority"=>false,"native_fits"=>0);limit=16384)
    R._exclusive_write(path,bytes);(;path,bytes,started=time_ns())
end
function _close_generated(attempt)
    need(R.bounded_read(attempt.path;limit=16384)==attempt.bytes && (time_ns()-attempt.started)/1e9<=7200,"generated candidate spent attempt changed")
    job_identity();nothing
end
function _locked_split(split,kind)
    expected=kind===:input ? (DateTime(2017),DateTime(2017,7,1)) : kind===:refit ?
        (DateTime(2017),DateTime(2017,10,1)) : (DateTime(2017,7,1),DateTime(2017,10,1))
    need((split.start,split.stop)==expected && split.stop<=split.asof<=DateTime(2018),"candidate locked2017 split differs")
    kind===:validation && need(split.first_issue==DateTime(2017,10,1),"candidate validation first issue differs")
end
function _operation_binding(value,index,id,split,directory,authority,kind;point=nothing,producer=nothing,native=nothing,selection=nothing)
    need(authority===nothing && producer===nothing && native===nothing,"execution certificates are not accepted; supply scientific artifact pins")
    _assert_fresh_destination(directory)
    _locked_split(split,kind===:read ? :validation : kind===:refitinput ? :refit : kind)
    F._archival_request(value,index,split.start,split.stop,split.asof)
    operation=kind===:refitinput ? "selected_R2_2017_refit_input_only" : kind===:input ? "fixed_candidate_2017_fit_input_only" :
        kind===:read ? "fixed_candidate_2017_independent_read_only" : "fixed_candidate_2017_validation_only"
    if kind===:refitinput
        need(selection isa CheckedR2Selection && selection.actual &&
            id==_selected_id(selection,recipe_for(id).kind,split;actual=true),"refit input requires independently checked selected recipe")
        _selection_target(selection,value.cases,index;actual=true)
    else
        need(selection===nothing,"candidate operation cannot consume selected-refit evidence")
    end
    d=descriptor(value.cases,index,id,split;operation=kind===:read ? "fixed_candidate_2017_validation_only" : operation,
        point,actual=true,admission=digest(B.admission_bytes(value)),selection_sha256=selection===nothing ? nothing : selection.evidence_sha256)
    auth=Dict("schema"=>"candidate_operation_binding_v2","operation"=>operation,
        "input_descriptor_sha256"=>digest(canonical(d)),"case_read_admission_sha256"=>d["case_read_admission_sha256"],
        "native_fits"=>0)
    (;auth,d)
end
function _start_operation(value,index,id,split,directory,authority,kind;kwargs...)
    a=_operation_binding(value,index,id,split,directory,authority,kind;kwargs...)
    a.auth,a.d,(bytes=UInt8[],started=time_ns())
end
function _close_operation(value,index,id,split,directory,authority,kind,d,observation;kwargs...)
    need(authority===nothing,"execution certificates are not accepted")
    F._archival_request(value,index,split.start,split.stop,split.asof)
    point=get(kwargs,:point,nothing)
    current=descriptor(value.cases,index,id,split;operation=d["operation"],point,actual=true,
        admission=digest(B.admission_bytes(value)),selection_sha256=d["selection_sha256"])
    need(canonical(current)==canonical(d),"candidate scientific inputs changed during work")
    job_identity();nothing
end
function _check_operation_binding(auth,d)
    J._object(auth,Set(("schema","operation","input_descriptor_sha256","case_read_admission_sha256","native_fits")))
    need(auth["schema"]=="candidate_operation_binding_v2" && auth["operation"] in
        (d["operation"],"fixed_candidate_2017_independent_read_only") &&
        auth["input_descriptor_sha256"]==digest(canonical(d)) &&
        auth["case_read_admission_sha256"]==J._sha(d["case_read_admission_sha256"]) && auth["native_fits"]===0,
        "candidate scientific input binding differs")
end
