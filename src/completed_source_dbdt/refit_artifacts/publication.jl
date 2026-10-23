function _load_native_job(directory;protocol::RefitProtocol,pins,expected_operation=nothing)
    p=_pins(pins);b=_read_bundle(directory,protocol,p;expected_kind="native_job")
    _native_receipt_check(b,p)
    operation=decode_canonical(b.context)["kind_context"]["operation"]
    require(expected_operation===nothing || operation==expected_operation,"native operation differs from required role")
    b
end
function _as_job(value,protocol,pins,operation)
    pin=value isa _Bundle ? value.manifest_sha256 : sha_string(value)
    _load_native_job(_resolve(pins,pin;directory=true);protocol,pins,expected_operation=operation)
end
function _native_reference(job::_Bundle)
    k=decode_canonical(job.context)["kind_context"]
    Dict{String,Any}("schema"=>"dbdt_native_receipt_reference_v1","native_job_manifest_sha256"=>job.manifest_sha256,
        "native_receipt_sha256"=>k["native_receipt_sha256"],"payload_sha256"=>k["payload_sha256"],"operation"=>k["operation"])
end
function _independent_reference(job::_Bundle)
    k=decode_canonical(job.context)["kind_context"];payload,rows=_job_payload(job)
    require(rows===nothing && payload["result"]=="pass","independent reference requires a computational checker payload")
    context=decode_canonical(job.context)
    require(payload["code_hashes"]==context["code_hashes"] && payload["runtime_id"]==context["runtime_id"],"independent payload code/runtime differs")
    Dict{String,Any}("schema"=>"dbdt_independent_check_reference_v1","native_job_manifest_sha256"=>job.manifest_sha256,
        "check_payload_sha256"=>k["payload_sha256"],"operation"=>k["operation"],"result"=>"pass")
end
function _same_common(context,producer::_Bundle)
    current=deepcopy(context);delete!(current,"kind_context")
    native=decode_canonical(producer.context);delete!(native,"kind_context")
    require(current==native,"published scientific common context differs from native producer")
end

function _validate_schedule_projection(rows,issues,protocol)
    require(length(rows)==length(issues)>0,"sealed schedule/request issue count differs")
    p=length(_spec(protocol)["feature_names"]);mixed=_spec(protocol)["layout"]=="mixed8"
    reasons=("missing_ground_history","invalid_ground_history","missing_omni_history","invalid_omni_history","invalid_feature_summary")
    for (row,issue) in zip(rows,issues)
        _schema(row,"schedule_row");accepted=_bool(row["accepted"]);timing=CMR.issue_timing(issue,_policy(protocol))
        require(row["row_id"]==_row_id(protocol,issue) && row["issue"]==time_text(issue) &&
            row["ground_anchor"]==time_text(timing.latest_ground_center) && row["label_ready"]==time_text(timing.label_assumed_ready) &&
            row["first_feedback"]==time_text(timing.first_scheduled_feedback) &&
            row["omni_anchor_or_null"]==(mixed ? time_text(timing.latest_omni_arrival_start) : nothing),"sealed prospective row identity/clock differs")
        sha_string(row["source_support_sha256"])
        if accepted
            require(row["reason"]=="complete" && length(_array(row["features_or_null"]))==p,"accepted feature projection differs")
            foreach(read_float_bits,row["features_or_null"])
        else
            require(row["reason"] in reasons && row["features_or_null"]===nothing,"rejected feature projection differs")
        end
        unhashed=deepcopy(row);delete!(unhashed,"feature_row_sha256")
        require(row["feature_row_sha256"]==digest(canonical_bytes(unhashed)),"sealed feature row digest differs")
    end
    nothing
end

function _validate_input_projection(protocol,k,accepted,excluded,schedule::ScheduleHandle,checker::_Bundle)
    split=k["split"]
    require(split==_split(protocol,split["name"]),"input split is not the exact declared protocol split")
    start=read_time_text(split["issue_start_inclusive"]);stop=read_time_text(split["issue_stop_exclusive"])
    boundary=read_time_text(split["label_ready_before"]);asof=read_time_text(split["target_admit_asof"])
    scheduled=[row for row in _data_rows(schedule) if start<=read_time_text(row["issue"])<stop]
    require(length(accepted)>=split["min_supervised_rows"] && length(accepted)+length(excluded)==length(scheduled),"input minima/complete cohort partition differs")
    maps=Dict{String,Tuple{Bool,Any}}()
    for (isaccepted,rows) in ((true,accepted),(false,excluded))
        previous=nothing
        for row in rows
            _schema(row,isaccepted ? "fit_row" : "excluded_row")
            id=sha_string(row["row_id"]);issue=read_time_text(row["issue"])
            require(!haskey(maps,id) && (previous===nothing || previous<issue),"input partition duplicate/backward row")
            previous=issue;maps[id]=(isaccepted,row)
        end
    end
    eligible=0;notready=0
    for prospective in scheduled
        id=prospective["row_id"];require(haskey(maps,id),"scheduled source window was dropped from input")
        admitted,row=maps[id];require(row["issue"]==prospective["issue"],"input row issue differs from sealed window")
        feature=prospective["accepted"]===true;ready=read_time_text(prospective["label_ready"])
        can_query=feature && ready<boundary
        feature && !can_query && (notready+=1)
        can_query && (eligible+=1)
        if admitted
            require(can_query && ready<=asof && row["label_ready"]==prospective["label_ready"] && row["target_admit_asof"]==split["target_admit_asof"] &&
                row["feature_row_sha256"]==prospective["feature_row_sha256"] && row["features"]==prospective["features_or_null"],"supervised row violates strict source/feature/readiness identity")
            sha_string(row["target_support_sha256"])
            require(read_float_bits(row["target"])>=0 && read_float_bits(row["scale"])>0,"invalid supervised target/scale")
            read_float_bits(row["log_target"])
        elseif !feature
            require(row["feature_status"]=="rejected" && row["label_status"]=="not_queried" && row["reason"]==prospective["reason"],"feature rejection partition differs")
        elseif !can_query
            require(row["feature_status"]=="accepted" && row["label_status"]=="not_queried" && row["reason"]=="split_not_strictly_ready","strict-readiness exclusion differs")
        else
            require(row["feature_status"]=="accepted" && row["label_status"]=="rejected" && row["reason"] in ("missing_ground_target","invalid_ground_target"),"target exclusion partition differs")
        end
    end
    require(Set(keys(maps))==Set(row["row_id"] for row in scheduled),"input contains an unscheduled/out-of-split row")
    rowpin=_ordered_digest([row["row_id"] for row in accepted])
    datapin=_frame_digest("DBDTFITINPUT1",(protocol.sha256,_pin(schedule),_jsontext(split),
        _frame_digest("DBDTACCEPTED1",_pack_rows(accepted)),_frame_digest("DBDTEXCLUDED1",_pack_rows(excluded))))
    require(k["row_id_sha256"]==rowpin && k["input_data_sha256"]==datapin,"full supervised/excluded stream digest differs")
    payload,_=_job_payload(checker);checks=payload["readiness_gate_checks"]
    require(checks["checked_rows"]==length(scheduled) && checks["eligible_rows"]==length(accepted) && checks["excluded_readiness_rows"]==notready &&
        checks["post_gate_target_rows"]==eligible && checks["pre_gate_target_lookups"]==0 && checks["failures"]==0,"independent full-cohort readiness counters differ")
    nothing
end
function _validate_input_partition(input::FitInput,schedule::ScheduleHandle,checker::_Bundle)
    _validate_input_projection(_bundle(input).protocol,_context(input)["kind_context"],_data_rows(input),_data_rows(input,"excluded"),schedule,checker)
end
function _referenced_job(reference,protocol,pins;independent=false,operation)
    _schema(reference,independent ? "independent_check_reference" : "native_receipt_reference")
    job=_as_job(reference["native_job_manifest_sha256"],protocol,pins,operation)
    require(reference==(independent ? _independent_reference(job) : _native_reference(job)),"native/check reference fields differ")
    job
end

function _publish_native_job(directory,request,payload_chunks,internal,external,log;pins,protocol::RefitProtocol)
    _schema(request,"job_request");_schema(internal,"job_internal_receipt");_schema(external,"native_receipt")
    chunks=Tuple(bounded_string(c;limit=MAX_MEMBER_BYTES,nonempty=true) for c in payload_chunks)
    require(0<length(chunks)<=MAX_FILES-6,"native payload chunk count differs")
    foreach(decode_canonical,chunks)
    require(ncodeunits(log)<=MAX_MEMBER_BYTES,"native log bound exceeded")
    p=_pins(pins);op=request["operation"];runtime=external["runtime"]
    common=_job_common(request,protocol,p,runtime)
    require(internal["loaded_code_hashes"]==common["code_hashes"] && internal["runtime_id"]==common["runtime_id"],"native publication code/runtime differs")
    kind=Dict{String,Any}("operation"=>op,"payload_type"=>op,"input_manifest_sha256_or_null"=>request["input_manifest_pin_or_null"],
        "input_data_sha256"=>request["input_data_pin"],"payload_sha256"=>_payload_digest(chunks),
        "internal_receipt_sha256"=>digest(_jsontext(internal)),"native_receipt_sha256"=>digest(_jsontext(external)))
    members=Pair{String,String}["job.json"=>_jsontext(request)]
    append!(members,[(length(chunks)==1 ? "payload.json" : _row_name(i,"payload"))=>c for (i,c) in enumerate(chunks)])
    append!(members,["internal-receipt.json"=>_jsontext(internal),"native-receipt.json"=>_jsontext(external),"native.log"=>String(log)])
    # Validate the external trust anchor before creating any destination. The
    # same full validator runs on the structural snapshot before publication.
    context=_with_kind(common,kind)
    temporary=_Bundle(_VERIFIED,abspath(directory),repeat("0",64),repeat("0",64),"native_job",protocol,
        _jsontext(context;limit=MAX_METADATA_BYTES),Tuple(vcat(["context.json"=>_jsontext(context;limit=MAX_METADATA_BYTES)],members)),0)
    _native_receipt_check(temporary,p)
    _publish_bundle(directory,protocol,"native_job",context,members,0;pins=p)
end

function _draft_from_job(job::_Bundle,pins;reconstruct_statistics=false)
    metadata,rows=_job_payload(job);require(rows!==nothing,"producer job requires bounded row payload")
    request=_member_json(job,"job.json");op=request["operation"];common=decode_canonical(job.context);delete!(common,"kind_context")
    protocol=job.protocol;members=Pair{String,String}[]
    if op=="source_audit"
        _schema(metadata,"source_payload_metadata");kind="source_audit";chunks=_pack_rows(rows)
    elseif op=="feature_schedule"
        _schema(metadata,"schedule_payload_metadata");kind="schedule";chunks=_pack_rows(rows)
    elseif op=="input_audit"
        _schema(metadata,"input_payload_metadata");kind="fit_input"
        accepted=Any[];excluded=Any[];seen_excluded=false
        for row in rows
            if haskey(row,"features")
                require(!seen_excluded,"accepted input rows follow exclusions");_schema(row,"fit_row");push!(accepted,row)
            else
                seen_excluded=true;_schema(row,"excluded_row");push!(excluded,row)
            end
        end
        require(length(accepted)==metadata["supervised_rows"] && length(excluded)==metadata["excluded_rows"],"native input partition count differs")
        chunks=_pack_rows(accepted);append!(members,[_row_name(i,"excluded")=>c for (i,c) in enumerate(_pack_rows(excluded))]);rows=accepted
    elseif op=="fit_ridge"
        kind="refit_point";_schema(metadata,"fit_receipt");require(length(rows)==4,"native fit payload object count differs")
        for (name,row) in zip(("recipe.json","preprocessing.json","model.json","fit-input-reference.json"),rows)
            push!(members,name=>_jsontext(row))
        end
        push!(members,"fit-receipt.json"=>_jsontext(metadata))
        input=_load_input(_resolve(pins,request["input_manifest_pin_or_null"];directory=true);protocol,pins)
        if reconstruct_statistics
            values=_input_values(input);Z=_independent_design(values.X,rows[1],rows[2]);A,b,_=_normal_statistics(Z,values.y,rows[1])
            statistics=_native_statistics_bytes(A,b,_spec(protocol)["layout"])
            require(digest(statistics)==metadata["diagnostic_payload_sha256"],"native as-solved statistics digest does not reconstruct")
            chunks=(statistics,)
        else
            chunks=()
        end
        rows=_data_rows(input)
    elseif op=="calibrate"
        kind="calibration";_schema(metadata,"input_reference");chunks=_pack_rows(rows)
        pointpin=_deps(request)["point"];point=_load_point(_resolve(pins,pointpin;directory=true);pins,expected_role=nothing)
        receipt=_member_json(point,"fit-receipt.json");recipe=_member_json(point,"recipe.json")
        reference=Dict{String,Any}("manifest_sha256"=>pointpin,"model_payload_sha256"=>receipt["model_payload_sha256"],
            "recipe_sha256"=>receipt["recipe_sha256"],"protocol_sha256"=>protocol.sha256,"role"=>recipe["role"])
        append!(members,["point-reference.json"=>_jsontext(reference),"input-reference.json"=>_jsontext(metadata)])
        append!(members,[_row_name(i,"sorted")=>c for (i,c) in enumerate(_pack_rows(_sorted_projection(rows)))])
    else
        fail("checker job cannot be converted into a producer draft")
    end
    if kind in ("source_audit","schedule","fit_input")
        require(metadata["scientific_context"]==common,"producer scientific common context differs")
    elseif kind=="refit_point"
        require(metadata["scientific_context_sha256"]==digest(_jsontext(common;limit=MAX_METADATA_BYTES)),"fit scientific context digest differs")
    end
    _Draft(_VERIFIED,kind,protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(metadata),chunks,Tuple(members),Int64(length(rows)))
end
function _producer_matches(draft,job,pins)
    original=_draft_from_job(job,pins)
    require(_draft_payload(draft)==_draft_payload(original),"draft differs from genuine native producer payload")
    draft.kind=="calibration" && require(draft.members==original.members,"cal point reference/stable projection differs from native producer")
    nothing
end

function _source_pair(protocol,pins,primary,checker)
    producer=_as_job(primary,protocol,pins,"source_audit");independent=_as_job(checker,protocol,pins,"source_independent_audit")
    require(_member_json(independent,"job.json")["input_manifest_pin_or_null"]==producer.manifest_sha256,"source checker producer pin differs")
    metadata,rows=_job_payload(producer);payload,_=_job_payload(independent)
    _schema(metadata,"source_payload_metadata");_check_result(payload,"source_independent_check_payload",protocol)
    permission=digest(canonical_bytes(_spec(protocol)["source_permissions"];limit=MAX_METADATA_BYTES))
    data=_frame_digest("DBDTSOURCE1",_pack_rows(rows))
    require(data==metadata["source_audit_data_sha256"]==payload["producer_source_data_sha256"]==payload["rebuilt_source_data_sha256"] &&
        permission==metadata["source_permission_sha256"]==payload["source_permission_sha256"],"source parser digest proof differs")
    for key in ("ground_rows","wind_rows","read_body_ids");require(metadata[key]==payload[key],"source parser counts/body proof differs");end
    for key in ("ground_rows","wind_rows")
        checked_integer(metadata[key],0,MAX_ROWS);checked_integer(payload[key],0,MAX_ROWS)
    end
    approved=[item for item in _spec(protocol)["source_permissions"] if item["disposition"]=="approved"]
    source_os=_runtime_os(_member_json(producer,"native-receipt.json")["runtime"])
    require(all(item->_absolute_for_os(item["path_or_null"],source_os),approved),"original source metadata path syntax differs from its producer OS")
    require(metadata["read_body_ids"]==[item["body_id"] for item in approved],"source audit opened-body list differs from approved descriptors")
    lastkey=nothing;ground=0;wind=0
    for row in rows
        _schema(row,"source_audit_record");kind=row["kind"];require(kind in ("ground","wind"),"invalid normalized source kind")
        t=read_time_text(row["time"]);key=(t,kind);require(lastkey===nothing || lastkey<key,"normalized source rows are not unique/ordered");lastkey=key
        matches=[item for item in approved if item["body_id"]==row["body_id"]]
        require(length(matches)==1,"normalized source body is not approved");permission=only(matches)
        require(permission["sha256_or_null"]==row["body_sha256"] && read_time_text(permission["period_start"])<=t<read_time_text(permission["period_stop"]),"normalized source body/time proof differs")
        require(permission["format"]==(kind=="ground" ? "dbdt_synthetic_ground_vectors_v1" : "dbdt_synthetic_arrival_wind_v1"),"normalized source kind/body format differs")
        values=read_float_bits.(_array(row["values"]));require(length(values)==(kind=="ground" ? 2 : 4),"normalized vector dimension differs")
        require(mod(Dates.value(t),60_000)==0 && (kind!="wind" || (_spec(protocol)["layout"]=="mixed8" && values[1]>=0 && values[4]>=0)),"normalized source value/time invalid")
        kind=="ground" ? (ground+=1) : (wind+=1)
    end
    require(ground==metadata["ground_rows"] && wind==metadata["wind_rows"],"normalized source row counts differ")
    producer,independent,metadata,rows
end
function _publish_source_audit(directory,draft::_Draft,primary,independent;pins)
    p=_pins(pins);protocol=draft.protocol;producer,checker,metadata,rows=_source_pair(protocol,p,primary,independent)
    require(draft.kind=="source_audit","wrong source draft kind");_producer_matches(draft,producer,p)
    kind=Dict{String,Any}("source_permission_sha256"=>metadata["source_permission_sha256"],"source_audit_data_sha256"=>metadata["source_audit_data_sha256"],
        "primary_native_job_manifest_sha256"=>producer.manifest_sha256,"independent_native_job_manifest_sha256"=>checker.manifest_sha256,
        "raw_body_count"=>length(metadata["read_body_ids"]),"neighbor_policy"=>_spec(protocol)["neighbor_policy"])
    members=Pair{String,String}["source-permissions.json"=>_jsontext(_spec(protocol)["source_permissions"];limit=MAX_METADATA_BYTES)]
    append!(members,[(length(draft.chunks)==1 ? "source-audit-data.json" : _row_name(i,"source-audit"))=>c for (i,c) in enumerate(draft.chunks)])
    append!(members,["native-audit.json"=>_jsontext(_native_reference(producer)),"independent-parser-check.json"=>_jsontext(_independent_reference(checker))])
    _publish_bundle(directory,protocol,"source_audit",_with_kind(decode_canonical(draft.scientific_context),kind),members,length(rows);pins=p)
end
"""
    open_audited_sources(directory; pins, protocol)

Verify a committed synthetic source audit and both genuine owned native parser
receipts without opening original source descriptor paths. Actual source
formats and actual-source provenance remain unsupported.
"""
function open_audited_sources(directory;pins,protocol::RefitProtocol)
    p=_pins(pins);b=_read_bundle(directory,protocol,p;expected_kind="source_audit");context=decode_canonical(b.context);k=context["kind_context"]
    _context_codes(context,"source_audit")
    primary=_referenced_job(_member_json(b,"native-audit.json"),protocol,p;operation="source_audit")
    checker=_referenced_job(_member_json(b,"independent-parser-check.json"),protocol,p;independent=true,operation="source_independent_audit")
    producer,independent,m,rows=_source_pair(protocol,p,primary,checker)
    _same_common(context,producer)
    require(k["primary_native_job_manifest_sha256"]==producer.manifest_sha256 && k["independent_native_job_manifest_sha256"]==independent.manifest_sha256 &&
        k["source_permission_sha256"]==m["source_permission_sha256"] && k["source_audit_data_sha256"]==m["source_audit_data_sha256"] &&
        k["raw_body_count"]==length(m["read_body_ids"]) && k["neighbor_policy"]==_spec(protocol)["neighbor_policy"],"source audit context proof differs")
    require(_member_json(b,"source-permissions.json")==_spec(protocol)["source_permissions"],"source permissions differ")
    source=AuditedSources(_VERIFIED,b)
    require(_source_records(source)==rows && b.rows==length(rows),"source audit normalized bytes differ from native proof")
    source
end
_load_source_audit(directory;protocol,pins)=open_audited_sources(directory;protocol,pins)

function _schedule_pair(protocol,pins,primary,independent)
    producer=_as_job(primary,protocol,pins,"feature_schedule");checker=_as_job(independent,protocol,pins,"feature_schedule")
    require(_member_json(checker,"job.json")["input_manifest_pin_or_null"]==producer.manifest_sha256,"schedule checker producer pin differs")
    m,rows=_job_payload(producer);payload,_=_job_payload(checker)
    _schema(m,"schedule_payload_metadata");_check_result(payload,"independent_schedule_check_payload",protocol)
    for key in ("scheduled_rows","accepted_features","rejected_features");checked_integer(m[key],0,MAX_ROWS);end
    for key in ("rows","accepted","rejected");checked_integer(payload[key],0,MAX_ROWS);end
    require(_member_json(producer,"job.json")["input_manifest_pin_or_null"]==m["source_audit_pin"] &&
        decode_canonical(producer.context)["source_manifest_sha256_or_null"]==m["source_audit_pin"],"schedule producer source identity differs")
    data=_frame_digest("DBDTSCHEDULE1",(protocol.sha256,_pack_rows(rows)...))
    require(data==m["schedule_data_sha256"]==payload["producer_schedule_data_sha256"]==payload["rebuilt_schedule_data_sha256"] &&
        m["source_audit_pin"]==payload["source_audit_pin"],"independent schedule digest/source differs")
    require(length(rows)==m["scheduled_rows"]==payload["rows"] && count(row->row["accepted"]===true,rows)==m["accepted_features"]==payload["accepted"] &&
        length(rows)-m["accepted_features"]==m["rejected_features"]==payload["rejected"],"schedule partition counts differ")
    producer,checker,m,rows
end
function _publish_schedule(directory,draft::_Draft,primary,independent;pins)
    p=_pins(pins);producer,checker,m,rows=_schedule_pair(draft.protocol,p,primary,independent)
    require(draft.kind=="schedule","wrong schedule draft kind");_producer_matches(draft,producer,p)
    issues=_feature_request(_member_json(producer,"job.json")["input_data_pin"],m["source_audit_pin"],p)
    _validate_schedule_projection(rows,issues,draft.protocol)
    kind=Dict{String,Any}("issue_start_inclusive"=>first(rows)["issue"],"issue_stop_exclusive"=>time_text(CMR.shift(read_time_text(last(rows)["issue"]),60_000)),
        "scheduled_rows"=>length(rows),"accepted_features"=>m["accepted_features"],"rejected_features"=>m["rejected_features"])
    members=Pair{String,String}[_row_name(i)=>c for (i,c) in enumerate(draft.chunks)]
    append!(members,["native-audit.json"=>_jsontext(_native_reference(producer)),"independent-schedule-check.json"=>_jsontext(_independent_reference(checker))])
    _publish_bundle(directory,draft.protocol,"schedule",_with_kind(decode_canonical(draft.scientific_context),kind),members,length(rows);pins=p)
end
function _load_schedule(directory;protocol::RefitProtocol,source::AuditedSources,pins)
    p=_pins(pins);b=_read_bundle(directory,protocol,p;expected_kind="schedule");context=decode_canonical(b.context);k=context["kind_context"]
    _context_codes(context,"feature_schedule");require(context["source_manifest_sha256_or_null"]==_pin(source),"schedule source differs")
    producer=_referenced_job(_member_json(b,"native-audit.json"),protocol,p;operation="feature_schedule")
    checker=_referenced_job(_member_json(b,"independent-schedule-check.json"),protocol,p;independent=true,operation="feature_schedule")
    _,_,m,rows=_schedule_pair(protocol,p,producer,checker)
    _same_common(context,producer)
    handle=ScheduleHandle(_VERIFIED,b);require(_data_rows(handle)==rows && b.rows==length(rows),"schedule row bytes differ from native proof")
    issues=_feature_request(_member_json(producer,"job.json")["input_data_pin"],_pin(source),p)
    _validate_schedule_projection(rows,issues,protocol)
    require(k["scheduled_rows"]==length(rows) && k["accepted_features"]==m["accepted_features"] && k["rejected_features"]==m["rejected_features"] &&
        k["issue_start_inclusive"]==first(rows)["issue"] && k["issue_stop_exclusive"]==time_text(CMR.shift(read_time_text(last(rows)["issue"]),60_000)),"schedule context differs")
    handle
end

function _input_pair(protocol,pins,primary,independent)
    producer=_as_job(primary,protocol,pins,"input_audit");checker=_as_job(independent,protocol,pins,"input_independent_audit")
    require(_member_json(checker,"job.json")["input_manifest_pin_or_null"]==producer.manifest_sha256,"input checker producer differs")
    m,allrows=_job_payload(producer);payload,_=_job_payload(checker)
    _schema(m,"input_payload_metadata");_check_result(payload,"independent_input_check_payload",protocol)
    for key in ("supervised_rows","excluded_rows");checked_integer(m[key],0,MAX_ROWS);end
    for key in ("rows","excluded_rows");checked_integer(payload[key],0,MAX_ROWS);end
    require(m["input_data_sha256"]==payload["producer_input_data_sha256"]==payload["rebuilt_input_data_sha256"] &&
        m["row_id_sha256"]==payload["row_id_sha256"] && m["source_audit_pin"]==payload["source_audit_pin"] &&
        m["schedule_manifest_sha256"]==payload["schedule_manifest_sha256"] && m["supervised_rows"]==payload["rows"] && m["excluded_rows"]==payload["excluded_rows"],"independent input identity/counts differ")
    checks=payload["readiness_gate_checks"];_schema(checks,"readiness_gate_checks")
    for key in PRIVATE_FIELDS["readiness_gate_checks"];checked_integer(checks[key],0,MAX_ROWS);end
    require(checks["pre_gate_target_lookups"]==0 && checks["failures"]==0 && checks["eligible_rows"]==payload["rows"],"independent strict readiness check failed")
    producer,checker,m,allrows
end
function _publish_input(directory,draft::_Draft,primary,independent;pins)
    p=_pins(pins);producer,checker,m,_=_input_pair(draft.protocol,p,primary,independent)
    require(draft.kind=="fit_input","wrong input draft kind");_producer_matches(draft,producer,p)
    rows=_unpack_rows(draft.chunks);ready=[read_time_text(r["label_ready"]) for r in rows]
    kind=Dict{String,Any}("split"=>m["split"],"input_data_sha256"=>m["input_data_sha256"],"row_id_sha256"=>m["row_id_sha256"],
        "scheduled_split_rows"=>m["supervised_rows"]+m["excluded_rows"],"supervised_rows"=>m["supervised_rows"],"excluded_rows"=>m["excluded_rows"],
        "min_label_ready_or_null"=>(isempty(ready) ? nothing : time_text(minimum(ready))),"max_label_ready_or_null"=>(isempty(ready) ? nothing : time_text(maximum(ready))))
    source=open_audited_sources(_resolve(p,m["source_audit_pin"];directory=true);protocol=draft.protocol,pins=p)
    schedule=_load_schedule(_resolve(p,m["schedule_manifest_sha256"];directory=true);protocol=draft.protocol,source,pins=p)
    exclusions=_unpack_rows([last(pair) for pair in draft.members if startswith(first(pair),"excluded-")])
    _validate_input_projection(draft.protocol,kind,rows,exclusions,schedule,checker)
    members=Pair{String,String}[_row_name(i)=>c for (i,c) in enumerate(draft.chunks)]
    append!(members,[p for p in draft.members if startswith(first(p),"excluded-")])
    append!(members,["native-input-audit.json"=>_jsontext(_native_reference(producer)),"independent-input-check.json"=>_jsontext(_independent_reference(checker))])
    _publish_bundle(directory,draft.protocol,"fit_input",_with_kind(decode_canonical(draft.scientific_context),kind),members,length(rows);pins=p)
end
function _load_input(directory;protocol::RefitProtocol,pins)
    p=_pins(pins);b=_read_bundle(directory,protocol,p;expected_kind="fit_input");context=decode_canonical(b.context);k=context["kind_context"]
    _context_codes(context,"input_audit")
    producer=_referenced_job(_member_json(b,"native-input-audit.json"),protocol,p;operation="input_audit")
    checker=_referenced_job(_member_json(b,"independent-input-check.json"),protocol,p;independent=true,operation="input_independent_audit")
    _,_,m,_=_input_pair(protocol,p,producer,checker)
    _same_common(context,producer)
    require(k["input_data_sha256"]==m["input_data_sha256"] && k["row_id_sha256"]==m["row_id_sha256"] && k["split"]==m["split"],"input context differs")
    source=open_audited_sources(_resolve(p,context["source_manifest_sha256_or_null"];directory=true);protocol,pins=p)
    schedule=_load_schedule(_resolve(p,context["schedule_manifest_sha256_or_null"];directory=true);protocol,source,pins=p)
    handle=FitInput(_VERIFIED,b)
    original=_draft_from_job(producer,p)
    require(_data_rows(handle)==_unpack_rows(original.chunks) && _data_rows(handle,"excluded")==_unpack_rows([last(pair) for pair in original.members if startswith(first(pair),"excluded-")]),"input partition bytes differ from original native producer")
    _validate_input_partition(handle,schedule,checker)
    require(k["supervised_rows"]==length(_data_rows(handle))==b.rows && k["excluded_rows"]==length(_data_rows(handle,"excluded")) &&
        k["scheduled_split_rows"]==k["supervised_rows"]+k["excluded_rows"],"input context counts differ")
    for key in ("supervised_rows","excluded_rows","scheduled_split_rows");checked_integer(k[key],0,MAX_ROWS);end
    ready=[read_time_text(row["label_ready"]) for row in _data_rows(handle)]
    require(k["min_label_ready_or_null"]==(isempty(ready) ? nothing : time_text(minimum(ready))) &&
        k["max_label_ready_or_null"]==(isempty(ready) ? nothing : time_text(maximum(ready))),"input readiness extrema differ")
    _input_values(handle);handle
end

function _protocol_for(directory,pins)
    commit=decode_canonical(bounded_read(joinpath(directory,"COMMIT.json");limit=MAX_METADATA_BYTES));_schema(commit,"commit")
    pin=sha_string(commit["protocol_sha256"])
    bytes=_guard_file(_resolve(pins,pin;directory=false),pin;limit=MAX_METADATA_BYTES)
    RefitProtocol(bytes;expected_spec_sha256=pin)
end

function _point_proof(draft,primary,independent,pins)
    protocol=draft.protocol;producer=_as_job(primary,protocol,pins,"fit_ridge");checker=_as_job(independent,protocol,pins,"fit_independent_check")
    _producer_matches(draft,producer,pins);receipt=_draft_member(draft,"fit-receipt.json")
    payload,_=_job_payload(checker);_check_result(payload,"independent_fit_check_payload",protocol)
    request=_member_json(checker,"job.json")
    require(_deps(request)["producer"]==producer.manifest_sha256 && request["input_manifest_pin_or_null"]==receipt["input_manifest_sha256"] &&
        payload["producer_fit_job_manifest_sha256"]==producer.manifest_sha256 && payload["input_manifest_sha256"]==receipt["input_manifest_sha256"] &&
        payload["input_data_sha256"]==receipt["input_data_sha256"] && payload["fit_receipt_sha256"]==digest(_jsontext(receipt)) &&
        payload["model_payload_sha256"]==receipt["model_payload_sha256"] && payload["preprocessing_sha256"]==receipt["preprocessing_sha256"],"native independent fit graph differs")
    require(payload["method"]=="augmented_qr_directed_inverse_residual_v1" && payload["precision_bits"]==512 && payload["toy_oracle_sha256"]==_TOY_ORACLE_SHA,"independent fit method/oracle differs")
    _fit_diagnostic_sanity(payload,draft.rows,length(_spec(protocol)["feature_names"]))
    producer,checker,payload
end

"""
    publish_point(directory, draft; native_receipt, independent_input_receipt,
                  independent_fit_receipt, pins)

Publish a new immutable synthetic point only after its actual owned fit and
independent input/numerical receipt graph has been verified. Existing outputs
cannot be overwritten, resumed or upgraded by metadata.
"""
function publish_point(directory,draft::_Draft;native_receipt,independent_input_receipt,independent_fit_receipt,pins)
    p=_pins(pins);require(draft.kind=="refit_point","wrong point draft kind")
    producer,checker,_=_point_proof(draft,native_receipt,independent_fit_receipt,p)
    receipt=_draft_member(draft,"fit-receipt.json");recipe=_draft_member(draft,"recipe.json")
    input=_load_input(_resolve(p,receipt["input_manifest_sha256"];directory=true);protocol=draft.protocol,pins=p)
    require(independent_input_receipt==_member_json(input,"independent-input-check.json"),"point independent input reference differs")
    kind=Dict{String,Any}("family"=>"ridge","role"=>recipe["role"],"recipe_sha256"=>receipt["recipe_sha256"],
        "input_manifest_sha256"=>_pin(input),"input_data_sha256"=>receipt["input_data_sha256"],"model_payload_sha256"=>receipt["model_payload_sha256"],
        "preprocessing_sha256"=>receipt["preprocessing_sha256"],"fit_receipt_sha256"=>digest(_jsontext(receipt)),
        "native_receipt_sha256"=>_native_reference(producer)["native_receipt_sha256"],"independent_input_sha256"=>digest(_jsontext(independent_input_receipt)),
        "independent_fit_sha256"=>digest(_jsontext(_independent_reference(checker))),"selection_decision_pin_or_null"=>nothing)
    members=collect(draft.members)
    append!(members,["native-fit-receipt.json"=>_jsontext(_native_reference(producer)),"independent-fit-check.json"=>_jsontext(_independent_reference(checker))])
    _publish_bundle(directory,draft.protocol,"refit_point",_with_kind(decode_canonical(draft.scientific_context),kind),members,draft.rows;pins=p)
end

"""
    load_point(directory; pins, expected_role)

Load an immutable point after checking exact member bytes, caller-pinned
protocol/source/input and genuine native/independent fit proofs. No ridge fit
is performed by this loader. Actual-source qualification is not implemented.
"""
function load_point(directory;pins,expected_role)
    role=expected_role isa Symbol ? String(expected_role) : bounded_string(expected_role)
    require(role in ("candidate","fixed_mixed8"),"unsupported expected point role")
    _load_point(directory;pins,expected_role=role)
end
function _load_point(directory;pins,expected_role=nothing)
    p=_pins(pins);protocol=_protocol_for(directory,p);b=_read_bundle(directory,protocol,p;expected_kind="refit_point")
    context=decode_canonical(b.context);k=context["kind_context"];_context_codes(context,"refit_point")
    require(expected_role===nothing || k["role"]==(expected_role isa Symbol ? String(expected_role) : expected_role),"loaded point role differs")
    primary=_referenced_job(_member_json(b,"native-fit-receipt.json"),protocol,p;operation="fit_ridge")
    checker=_referenced_job(_member_json(b,"independent-fit-check.json"),protocol,p;independent=true,operation="fit_independent_check")
    draft=_draft_from_job(primary,p);_,_,payload=_point_proof(draft,primary,checker,p)
    _same_common(context,primary)
    for (name,bytes) in draft.members;require(_member(b,name)==_jsonbytes(bytes),"published point differs from actual fit numerical bytes");end
    receipt=_member_json(b,"fit-receipt.json");recipe=_member_json(b,"recipe.json")
    input=_load_input(_resolve(p,k["input_manifest_sha256"];directory=true);protocol,pins=p)
    require(k["family"]=="ridge" && k["role"]==recipe["role"] && k["recipe_sha256"]==receipt["recipe_sha256"] &&
        k["input_data_sha256"]==receipt["input_data_sha256"] && k["model_payload_sha256"]==receipt["model_payload_sha256"] &&
        k["preprocessing_sha256"]==receipt["preprocessing_sha256"] && k["fit_receipt_sha256"]==digest(_member(b,"fit-receipt.json")) &&
        k["native_receipt_sha256"]==_native_reference(primary)["native_receipt_sha256"] &&
        k["independent_input_sha256"]==digest(_member(input,"independent-input-check.json")) &&
        k["independent_fit_sha256"]==digest(_member(b,"independent-fit-check.json")) && k["selection_decision_pin_or_null"]===nothing,"point context differs from required proof graph")
    require(b.rows==_bundle(input).rows && _member_json(b,"fit-input-reference.json")==_input_reference(input),"point input row reference differs")
    point=PointHandle(_VERIFIED,b);_ridge_parts(point);point
end

function _cal_proof(draft,primary,independent,pins)
    protocol=draft.protocol;producer=_as_job(primary,protocol,pins,"calibrate");checker=_as_job(independent,protocol,pins,"cal_independent_check")
    _producer_matches(draft,producer,pins);payload,_=_job_payload(checker);_check_result(payload,"independent_cal_check_payload",protocol)
    inputref=_draft_member(draft,"input-reference.json");pointref=_draft_member(draft,"point-reference.json")
    rows=_unpack_rows(draft.chunks);sorted=_sorted_projection(rows);request=_member_json(checker,"job.json")
    require(_deps(request)["producer"]==producer.manifest_sha256 && _deps(request)["point"]==pointref["manifest_sha256"] &&
        request["input_manifest_pin_or_null"]==inputref["manifest_sha256"] && payload["input_manifest_sha256"]==inputref["manifest_sha256"] &&
        payload["point_manifest_sha256"]==pointref["manifest_sha256"] && payload["chronological_data_sha256"]==_frame_digest("DBDTCALROWS1",draft.chunks) &&
        payload["sorted_projection_sha256"]==_frame_digest("DBDTCALSORTED1",_pack_rows(sorted)),"cal independent graph/digest differs")
    producer,checker,payload
end
"""
    publish_calibration(directory, draft; native_receipt, independent_cal_receipt, pins)

Publish the complete chronological synthetic calibration and stable projection
after genuine owned producer and independent calibration checks.
"""
function publish_calibration(directory,draft::_Draft;native_receipt,independent_cal_receipt,pins)
    p=_pins(pins);require(draft.kind=="calibration","wrong calibration draft kind")
    producer,checker,payload=_cal_proof(draft,native_receipt,independent_cal_receipt,p)
    inputref=_draft_member(draft,"input-reference.json");pointref=_draft_member(draft,"point-reference.json")
    input=_load_input(_resolve(p,inputref["manifest_sha256"];directory=true);protocol=draft.protocol,pins=p)
    split=_context(input)["kind_context"]["split"];first_issue=read_time_text(payload["first_allowed_issue"])
    purpose=read_time_text(split["issue_stop_exclusive"])==DateTime(2017,11,1) ? "october_development" : "octdec_final"
    _cal_info(input,purpose,first_issue);rows=_unpack_rows(draft.chunks)
    kind=Dict{String,Any}("purpose"=>purpose,"first_allowed_issue"=>time_text(first_issue),"point_manifest_sha256"=>pointref["manifest_sha256"],
        "input_manifest_sha256"=>inputref["manifest_sha256"],"input_data_sha256"=>_context(input)["kind_context"]["input_data_sha256"],
        "chronological_data_sha256"=>payload["chronological_data_sha256"],"sorted_projection_sha256"=>payload["sorted_projection_sha256"],
        "native_receipt_sha256"=>_native_reference(producer)["native_receipt_sha256"],"independent_cal_sha256"=>digest(_jsontext(_independent_reference(checker))),
        "min_label_ready"=>time_text(minimum(read_time_text(row["label_ready"]) for row in rows)),"max_label_ready"=>payload["max_label_ready"])
    members=Pair{String,String}[_row_name(i)=>c for (i,c) in enumerate(draft.chunks)];append!(members,collect(draft.members))
    append!(members,["native-cal-receipt.json"=>_jsontext(_native_reference(producer)),"independent-cal-check.json"=>_jsontext(_independent_reference(checker))])
    _publish_bundle(directory,draft.protocol,"calibration",_with_kind(decode_canonical(draft.scientific_context),kind),members,length(rows);pins=p)
end
"""
    load_calibration(directory; point, pins, first_issue)

Validate complete saved chronology, exact stable projection, pinned point/input
and owned calibration proofs. Same-runtime loads reproduce all score bits;
foreign numerical-runtime checks remain diagnostics and never replace scores.
"""
function load_calibration(directory;point::PointHandle,pins,first_issue)
    p=_pins(pins);protocol=_bundle(point).protocol;b=_read_bundle(directory,protocol,p;expected_kind="calibration")
    context=decode_canonical(b.context);k=context["kind_context"];_context_codes(context,"calibration")
    require(k["point_manifest_sha256"]==_pin(point) && first_issue isa DateTime && first_issue>=read_time_text(k["first_allowed_issue"]),"cal point/use boundary differs")
    primary=_referenced_job(_member_json(b,"native-cal-receipt.json"),protocol,p;operation="calibrate")
    checker=_referenced_job(_member_json(b,"independent-cal-check.json"),protocol,p;independent=true,operation="cal_independent_check")
    draft=_draft_from_job(primary,p);_,_,payload=_cal_proof(draft,primary,checker,p)
    _same_common(context,primary)
    input=_load_input(_resolve(p,k["input_manifest_sha256"];directory=true);protocol,pins=p)
    _cal_info(input,k["purpose"],read_time_text(k["first_allowed_issue"]))
    handle=CalHandle(_VERIFIED,b);rows=_data_rows(handle);sorted=_data_rows(handle,"sorted")
    require(rows==_unpack_rows(draft.chunks) && b.rows==length(rows) && _member_json(b,"point-reference.json")==_draft_member(draft,"point-reference.json") &&
        _member_json(b,"input-reference.json")==_input_reference(input),"published calibration differs from native producer")
    same_runtime=_same_numeric_runtime(context)
    _check_cal_rows(rows,sorted,input,point,first_issue;reproduce=same_runtime)
    require(k["chronological_data_sha256"]==payload["chronological_data_sha256"]==_frame_digest("DBDTCALROWS1",draft.chunks) &&
        k["sorted_projection_sha256"]==payload["sorted_projection_sha256"]==_frame_digest("DBDTCALSORTED1",_pack_rows(sorted)) &&
        k["input_data_sha256"]==_context(input)["kind_context"]["input_data_sha256"] &&
        k["native_receipt_sha256"]==_native_reference(primary)["native_receipt_sha256"] &&
        k["independent_cal_sha256"]==digest(_member(b,"independent-cal-check.json")) &&
        k["min_label_ready"]==time_text(minimum(read_time_text(row["label_ready"]) for row in rows)) && k["max_label_ready"]==payload["max_label_ready"],"calibration context proof differs")
    handle
end
