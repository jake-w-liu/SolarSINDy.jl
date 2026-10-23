_payload_digest(chunks)=length(chunks)==1 ? digest(only(chunks)) : _frame_digest("DBDTPAYLOAD1",chunks)
function _payload_chunks(metadata,rows)
    parts=_pack_rows(rows);isempty(parts) && (parts=("[]\n",))
    Tuple(_jsontext(Dict{String,Any}("schema"=>"dbdt_completed_native_payload_v1","ordinal"=>i,
        "chunks"=>length(parts),"metadata"=>metadata,"rows"=>decode_canonical(part))) for (i,part) in enumerate(parts))
end
function _job_payload(job::_Bundle)
    require(job.kind=="native_job","payload requires native job")
    names=_chunk_catalog([first(p) for p in job.members],"payload";single="payload.json")
    chunks=Tuple(String(_member(job,name)) for name in names)
    firstvalue=decode_canonical(first(chunks))
    if !(firstvalue isa AbstractDict && haskey(firstvalue,"ordinal"))
        require(length(chunks)==1,"checker payload cannot be chunked")
        return firstvalue,nothing
    end
    rows=Any[];metadata=nothing
    for (i,text) in enumerate(chunks)
        value=decode_canonical(text);_schema(value,"native_payload_chunk")
        require(value["schema"]=="dbdt_completed_native_payload_v1" && checked_integer(value["ordinal"],1,MAX_FILES)==i &&
            checked_integer(value["chunks"],1,MAX_FILES)==length(chunks),"native payload ordinal/schema differs")
        i==1 ? (metadata=value["metadata"]) : require(value["metadata"]==metadata,"native payload metadata differs across chunks")
        part=_array(value["rows"]);require(i==length(chunks)||length(part)==CHUNK_ROWS,"nonfinal native payload chunk size differs")
        require(length(rows)+length(part)<=MAX_ROWS,"native payload row count exceeds bound")
        append!(rows,part)
    end
    metadata,rows
end
function _draft_payload(draft::_Draft)
    metadata=decode_canonical(draft.metadata)
    if draft.kind=="refit_point"
        rows=Any[_draft_member(draft,name) for name in ("recipe.json","preprocessing.json","model.json","fit-input-reference.json")]
    else
        rows=_unpack_rows(draft.chunks)
        if draft.kind=="fit_input"
            excluded=[last(p) for p in draft.members if startswith(first(p),"excluded-")]
            append!(rows,_unpack_rows(excluded))
        end
    end
    _payload_chunks(metadata,rows)
end

function _feature_request(pin,sourcepin,pins)
    path=_resolve(pins,pin);chunks=String[];names=String[]
    if isfile(path)
        push!(chunks,String(_guard_file(path,pin)))
    else
        names=readdir(path);require(0<length(names)<=MAX_FILES && length(unique(lowercase.(names)))==length(names),"invalid feature request directory size/case")
        expected=[_row_name(i,"request") for i in 1:length(names)]
        require(sort(names)==expected,"feature request members differ")
        total=Int64(0)
        for name in expected
            file=joinpath(path,name);require(isfile(file)&&!islink(file),"feature request members must be ordinary")
            bytes=bounded_read(file);total=Base.Checked.checked_add(total,length(bytes));require(total<=MAX_BUNDLE_BYTES,"feature request total bytes exceeded")
            push!(chunks,String(bytes))
        end
        require(_payload_digest(chunks)==pin,"feature request framed digest differs")
    end
    issues=DateTime[]
    for (i,text) in enumerate(chunks)
        value=decode_canonical(text);_schema(value,"native_payload_chunk");_schema(value["metadata"],"feature_request_metadata")
        require(value["schema"]=="dbdt_completed_feature_request_v1" && value["metadata"]["source_manifest_sha256"]==sourcepin,"feature request source/schema differs")
        require(checked_integer(value["ordinal"],1,MAX_FILES)==i && checked_integer(value["chunks"],1,MAX_FILES)==length(chunks),"feature request ordinal differs")
        rows=_array(value["rows"];nonempty=true)
        require(i==length(chunks)||length(rows)==CHUNK_ROWS,"nonfinal feature request chunk length differs")
        require(length(issues)+length(rows)<=MAX_ROWS,"feature request issue bound exceeded")
        append!(issues,read_time_text.(rows))
    end
    require(!isempty(issues) && all(t->mod(Dates.value(t),60_000)==0,issues) &&
        all(issues[i]<issues[i+1] for i in 1:length(issues)-1),"feature request issues are irregular/duplicate/backward")
    if isfile(path)
        require(String(_guard_file(path,pin))==only(chunks),"feature request changed during read")
    else
        require(Set(readdir(path))==Set(names),"feature request members changed")
        for (i,text) in enumerate(chunks);require(String(bounded_read(joinpath(path,_row_name(i,"request"))))==text,"feature request bytes changed");end
    end
    issues
end
function _split_request(pin,schedulepin,pins)
    bytes=_guard_file(_resolve(pins,pin;directory=false),pin);value=decode_canonical(bytes);_schema(value,"split_request")
    require(value["schema"]=="dbdt_completed_split_request_v1" && value["schedule_manifest_sha256"]==schedulepin,"split request identity differs")
    identifier(value["split"])
end

function _job_common(request,protocol,pins,runtime)
    op=request["operation"];input=request["input_manifest_pin_or_null"]
    if op in ("source_audit","source_independent_audit")
        return _common_context(protocol,op,runtime)
    end
    b=_read_bundle(_resolve(pins,input;directory=true),protocol,pins;expected_manifest=input)
    c=decode_canonical(b.context)
    if op=="feature_schedule"
        require(b.kind in ("source_audit","native_job"),"feature operation input kind differs")
        source=b.kind=="source_audit" ? input : c["source_manifest_sha256_or_null"]
        return _common_context(protocol,op,runtime;source)
    end
    schedule=op=="input_audit" ? input : c["schedule_manifest_sha256_or_null"]
    _common_context(protocol,op,runtime;source=c["source_manifest_sha256_or_null"],schedule)
end

function _check_result(payload,name,protocol)
    _schema(payload,name)
    require(payload["protocol_sha256"]==protocol.sha256 && payload["result"]=="pass","independent payload did not qualify")
    payload
end

function _execute_request(request,protocol,pins,runtime)
    _schema(request,"job_request");op=request["operation"]
    require(request["schema"]=="dbdt_completed_job_request_v1" && op in OPERATIONS &&
        request["evidence_mode"]=="synthetic_only" && request["protocol_pin"]==protocol.sha256,"unsupported native request")
    require(request["required_runtime"]==runtime,"native inspected runtime differs from request")
    work=bounded_string(request["work_directory"];nonempty=true)
    require(isabspath(work)&&isdir(work)&&!islink(work),"native work directory must be caller owned and ordinary")
    _validate_runtime(runtime);_deps(request)
    common=_job_common(request,protocol,pins,runtime);input=request["input_manifest_pin_or_null"]
    datapin=sha_string(request["input_data_pin"]);deps=_deps(request)
    require(Set(keys(deps))==(op in ("fit_independent_check",) ? Set(["producer"]) :
        op in ("calibrate",) ? Set(["point"]) : op=="cal_independent_check" ? Set(["point","producer"]) : Set{String}()),"native dependency roles differ")
    if op=="source_audit"
        require(input===nothing && request["recipe_pin_or_null"]===nothing,"source producer cannot have input/recipe")
        permission=digest(canonical_bytes(_spec(protocol)["source_permissions"];limit=MAX_METADATA_BYTES))
        require(datapin==permission==request["source_permission_pin_or_null"],"source permission/data pin differs")
        return _draft_payload(_audit_source_bodies(protocol))
    elseif op=="source_independent_audit"
        producer=_load_native_job(_resolve(pins,input;directory=true);protocol,pins,expected_operation="source_audit")
        metadata,_=_job_payload(producer);rebuilt=_audit_source_bodies(protocol;independent=true);m=decode_canonical(rebuilt.metadata)
        require(datapin==metadata["source_audit_data_sha256"]==m["source_audit_data_sha256"],"independent synthetic source data differs")
        payload=Dict{String,Any}("schema"=>"dbdt_completed_source_independent_check_v1","protocol_sha256"=>protocol.sha256,
            "source_permission_sha256"=>m["source_permission_sha256"],"producer_source_data_sha256"=>datapin,"rebuilt_source_data_sha256"=>m["source_audit_data_sha256"],
            "ground_rows"=>m["ground_rows"],"wind_rows"=>m["wind_rows"],"read_body_ids"=>m["read_body_ids"],
            "code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"result"=>"pass")
        return (_jsontext(payload),)
    elseif op=="feature_schedule"
        b=_read_bundle(_resolve(pins,input;directory=true),protocol,pins;expected_manifest=input)
        if b.kind=="source_audit"
            source=open_audited_sources(b.directory;protocol,pins);issues=_feature_request(datapin,input,pins)
            return _draft_payload(seal_feature_schedule(source,issues;protocol))
        end
        producer=_load_native_job(b.directory;protocol,pins,expected_operation="feature_schedule")
        metadata,rows=_job_payload(producer);_schema(metadata,"schedule_payload_metadata")
        source=open_audited_sources(_resolve(pins,metadata["source_audit_pin"];directory=true);protocol,pins)
        require(_member_json(producer,"job.json")["input_manifest_pin_or_null"]==_pin(source),"schedule checker requires a producer role")
        issues=_feature_request(_member_json(producer,"job.json")["input_data_pin"],_pin(source),pins)
        draft=_seal_schedule(source,issues,protocol;independent=true);m=decode_canonical(draft.metadata)
        require(datapin==metadata["schedule_data_sha256"]==m["schedule_data_sha256"] && rows==_unpack_rows(draft.chunks),"independent schedule differs")
        payload=Dict{String,Any}("schema"=>"dbdt_completed_schedule_independent_check_v1","protocol_sha256"=>protocol.sha256,
            "source_audit_pin"=>_pin(source),"producer_schedule_data_sha256"=>datapin,"rebuilt_schedule_data_sha256"=>m["schedule_data_sha256"],
            "rows"=>m["scheduled_rows"],"accepted"=>m["accepted_features"],"rejected"=>m["rejected_features"],
            "code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"result"=>"pass")
        return (_jsontext(payload),)
    elseif op in ("input_audit","input_independent_audit")
        producer=op=="input_independent_audit" ? _load_native_job(_resolve(pins,input;directory=true);protocol,pins,expected_operation="input_audit") : nothing
        request0=producer===nothing ? request : _member_json(producer,"job.json")
        schedulepin=request0["input_manifest_pin_or_null"]
        sourcepin=common["source_manifest_sha256_or_null"]
        source=open_audited_sources(_resolve(pins,sourcepin;directory=true);protocol,pins)
        schedule=_load_schedule(_resolve(pins,schedulepin;directory=true);protocol,source,pins)
        split=_split_request(request0["input_data_pin"],schedulepin,pins)
        draft=_build_input(schedule,source,split,protocol;independent=producer!==nothing)
        producer===nothing && return _draft_payload(draft)
        m=decode_canonical(draft.metadata);original,rows=_job_payload(producer)
        require(datapin==original["input_data_sha256"]==m["input_data_sha256"],"independent supervised partition differs")
        require(_draft_payload(draft)==_payload_chunks(m,rows),"independent input rows/metadata differ")
        checks=decode_canonical(last(only(filter(p->first(p)=="readiness-checks",draft.members))))
        payload=Dict{String,Any}("schema"=>"dbdt_completed_input_independent_check_v1","protocol_sha256"=>protocol.sha256,
            "source_audit_pin"=>sourcepin,"schedule_manifest_sha256"=>schedulepin,"producer_input_data_sha256"=>datapin,
            "rebuilt_input_data_sha256"=>m["input_data_sha256"],"row_id_sha256"=>m["row_id_sha256"],"rows"=>m["supervised_rows"],
            "excluded_rows"=>m["excluded_rows"],"readiness_gate_checks"=>checks,"code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"result"=>"pass")
        return (_jsontext(payload),)
    elseif op in ("fit_ridge","fit_independent_check")
        fittedinput=_load_input(_resolve(pins,input;directory=true);protocol,pins)
        require(datapin==_context(fittedinput)["kind_context"]["input_data_sha256"],"native fit data pin differs")
        if op=="fit_ridge"
            recipes=[r for r in _spec(protocol)["model_recipes"] if digest(_jsontext(r))==request["recipe_pin_or_null"]]
            require(length(recipes)==1,"native recipe pin is absent/ambiguous")
            r=only(recipes);recipe=RidgeRecipe(protocol;role=r["role"],penalty=read_float_bits(r["penalty"]))
            return _draft_payload(fit_point(fittedinput,recipe))
        end
        producer=_load_native_job(_resolve(pins,deps["producer"];directory=true);protocol,pins,expected_operation="fit_ridge")
        draft=_draft_from_job(producer,pins;reconstruct_statistics=true);diagnostics=check_ridge_fit(draft,fittedinput;protocol)
        receipt=_draft_member(draft,"fit-receipt.json")
        payload=Dict{String,Any}("schema"=>"dbdt_completed_fit_independent_check_v1","protocol_sha256"=>protocol.sha256,
            "input_manifest_sha256"=>input,"input_data_sha256"=>datapin,"producer_fit_job_manifest_sha256"=>producer.manifest_sha256,
            "fit_receipt_sha256"=>digest(_jsontext(receipt)),"model_payload_sha256"=>receipt["model_payload_sha256"],"preprocessing_sha256"=>receipt["preprocessing_sha256"],
            "method"=>"augmented_qr_directed_inverse_residual_v1","precision_bits"=>512,
            "backward_eta"=>diagnostics["backward_eta"],"condition_estimate"=>diagnostics["condition_estimate"],"forward_enclosure"=>diagnostics["forward_enclosure"],"normal_residual"=>diagnostics["normal_residual"],
            "toy_oracle_sha256"=>_toy_ridge_check(),"code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"result"=>"pass")
        return (_jsontext(payload),)
    elseif op in ("calibrate","cal_independent_check","load_cal_check")
        if op=="load_cal_check"
            b=_read_bundle(_resolve(pins,input;directory=true),protocol,pins;expected_kind="calibration",expected_manifest=input)
            k=decode_canonical(b.context)["kind_context"]
            point=_load_point(_resolve(pins,k["point_manifest_sha256"];directory=true);pins,expected_role=nothing)
            calibration=load_calibration(b.directory;point,pins,first_issue=read_time_text(k["first_allowed_issue"]))
            fitinput=_load_input(_resolve(pins,k["input_manifest_sha256"];directory=true);protocol,pins)
            return (_jsontext(_cal_check_payload(_data_rows(calibration),_data_rows(calibration,"sorted"),fitinput,point,read_time_text(k["first_allowed_issue"]),common)),)
        end
        fitinput=_load_input(_resolve(pins,input;directory=true);protocol,pins)
        require(datapin==_context(fitinput)["kind_context"]["input_data_sha256"],"native cal data pin differs")
        point=_load_point(_resolve(pins,deps["point"];directory=true);pins,expected_role=nothing)
        split=_context(fitinput)["kind_context"]["split"];next_issue=read_time_text(split["label_ready_before"])
        purpose=read_time_text(split["issue_stop_exclusive"])==DateTime(2017,11,1) ? "october_development" : "octdec_final"
        draft=calibrate_point(point,fitinput;purpose,next_issue,protocol)
        op=="calibrate" && return _draft_payload(draft)
        producer=_load_native_job(_resolve(pins,deps["producer"];directory=true);protocol,pins,expected_operation="calibrate")
        actual=_draft_from_job(producer,pins)
        require(actual.chunks==draft.chunks && actual.members==draft.members,"independent cal computation differs")
        rows=_unpack_rows(actual.chunks);sorted=_sorted_projection(rows)
        return (_jsontext(_cal_check_payload(rows,sorted,fitinput,point,next_issue,common)),)
    end
    fail("unsupported fixed native operation")
end

function _cal_check_payload(rows,sorted,input,point,first_issue,common)
    _check_cal_rows(rows,sorted,input,point,first_issue)
    Dict{String,Any}("schema"=>"dbdt_completed_cal_independent_check_v1","protocol_sha256"=>_bundle(input).protocol.sha256,
        "point_manifest_sha256"=>_pin(point),"input_manifest_sha256"=>_pin(input),"input_data_sha256"=>_context(input)["kind_context"]["input_data_sha256"],
        "chronological_data_sha256"=>_frame_digest("DBDTCALROWS1",_pack_rows(rows)),"sorted_projection_sha256"=>_frame_digest("DBDTCALSORTED1",_pack_rows(sorted)),
        "reproduced_predictions_sha256"=>_frame_digest("DBDTCALPREDICTIONS1",_pack_rows([row["prediction"] for row in rows])),
        "reproduced_scores_sha256"=>_frame_digest("DBDTCALSCORES1",_pack_rows([row["score"] for row in rows])),
        "max_label_ready"=>time_text(maximum(read_time_text(row["label_ready"]) for row in rows)),"first_allowed_issue"=>time_text(first_issue),
        "method"=>"independent_full_chronology_v1","code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"result"=>"pass")
end
