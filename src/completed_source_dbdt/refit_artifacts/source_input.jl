function _source_records(source::AuditedSources)
    b=_bundle(source);names=[first(p) for p in b.members]
    catalog=_chunk_catalog(names,"source-audit";single="source-audit-data.json")
    _unpack_rows(Tuple(String(_member(b,name)) for name in catalog))
end
function _source_index(source::AuditedSources)
    ground=Dict{DateTime,Any}();wind=Dict{DateTime,Any}()
    for record in _source_records(source)
        _schema(record,"source_audit_record")
        kind=record["kind"];require(kind in ("ground","wind"),"unknown normalized source kind")
        time=read_time_text(record["time"]);require(mod(Dates.value(time),60_000)==0,"irregular source timestamp")
        values=_array(record["values"])
        require(length(values)==(kind=="ground" ? 2 : 4),"normalized source vector shape differs")
        # Numeric payloads were validated by the committed source audit. Index
        # construction itself does not project future target values.
        sha_string(record["body_sha256"]);identifier(record["body_id"])
        target=kind=="ground" ? ground : wind
        require(!haskey(target,time),"duplicate normalized source time")
        target[time]=record
    end
    (;ground,wind)
end

struct _ReadWindow{N} <: AbstractDict{DateTime,NTuple{N,Float64}}
    index::Dict{DateTime,Any}
    first::DateTime
    last::DateTime
    queries::Vector{DateTime}
end
function _query(window::_ReadWindow,time)
    require(time isa DateTime && window.first<=time<=window.last,"source value queried outside its predeclared window")
    push!(window.queries,time)
    time
end
Base.haskey(window::_ReadWindow,time)=haskey(window.index,_query(window,time))
function Base.getindex(window::_ReadWindow{N},time) where N
    record=window.index[_query(window,time)]
    Tuple(read_float_bits(v) for v in record["values"])::NTuple{N,Float64}
end
Base.iterate(::_ReadWindow)=fail("source windows do not permit scans")
Base.length(window::_ReadWindow)=length(window.index)

function _support_digest(index,ground_queries,wind_queries=DateTime[])
    rows=Any[]
    for (kind,source,queries) in (("ground",index.ground,ground_queries),("wind",index.wind,wind_queries))
        for time in sort!(unique(queries))
            push!(rows,Dict{String,Any}("kind"=>kind,"time"=>time_text(time),
                "present"=>haskey(source,time),"record"=>get(source,time,nothing)))
        end
    end
    _frame_digest("DBDTSUPPORT1",_pack_rows(rows))
end

function _feature_row(source::AuditedSources,index,issue::DateTime;independent=false)
    protocol=_bundle(source).protocol;spec=_spec(protocol);policy=_policy(protocol);mode=spec["layout"]
    timing=CMR.issue_timing(issue,policy)
    CMR.shift(timing.latest_ground_center,-30*60_000)
    ground=_ReadWindow{2}(index.ground,timing.latest_ground_center-Minute(30),timing.latest_ground_center,DateTime[])
    wind=_ReadWindow{4}(index.wind,timing.latest_omni_arrival_start,timing.latest_omni_arrival_start,DateTime[])
    accepted=false;reason="missing_ground_history";features=nothing;ground11=nothing
    if !independent
        prepared=mode=="ground11" ? D.prepare_issue(ground,issue;policy,mode=:ground11) :
            D.prepare_issue(ground,issue;policy,mode=:mixed8,omni=wind)
        accepted=prepared.accepted;reason=String(prepared.reason)
        if accepted
            features=prepared.features;ground11=prepared.ground11
        end
    else
        # Independent reconstruction of the admission/support path. The frozen
        # reference feature primitive is shared; its numerical equations have
        # separate pre-code literal oracles, not producer-generated expectations.
        rates=Float64[];complete=true
        for k in 29:-1:0
            t=timing.latest_ground_center-Minute(k)
            if !haskey(ground,t) || !haskey(ground,t-Minute(1))
                complete=false;break
            end
            before=ground[t-Minute(1)];after=ground[t]
            rate=hypot(after[1]-before[1],after[2]-before[2])
            if !isfinite(rate);complete=false;reason="invalid_ground_history";break;end
            push!(rates,rate)
        end
        if complete
            # Match the declared admission order independently: wind presence/
            # validity precedes summary computation for mixed8. Product overflow
            # is a feature-summary failure, not an invalid finite wind vector.
            wind_values=nothing;wind_ready=true
            if mode=="mixed8"
                if !haskey(wind,timing.latest_omni_arrival_start)
                    wind_ready=false;reason="missing_omni_history"
                else
                    wind_values=wind[timing.latest_omni_arrival_start]
                    v,bz,by,n=wind_values
                    (v<0 || n<0) && (wind_ready=false;reason="invalid_omni_history")
                end
            end
            if wind_ready
                average=mean(rates);deviation=std(rates)
                if !all(isfinite,(average,deviation))
                    reason="invalid_feature_summary"
                else
                    peak=maximum(rates)
                    if mode=="ground11"
                        ground11=vec(CF.feature_matrix([issue],[last(rates)],[average],[peak],[deviation]))
                        features=ground11;accepted=true;reason="complete"
                    else
                        v,bz,by,n=wind_values;south=max(-bz,0.0);coupling=v*south
                        if !isfinite(coupling)
                            reason="invalid_feature_summary"
                        else
                            ground11=vec(CF.feature_matrix([issue],[last(rates)],[average],[peak],[deviation]))
                            features=[last(rates),average,peak,deviation,v,bz,south,coupling]
                            accepted=true;reason="complete"
                        end
                    end
                end
            end
        end
    end
    row=Dict{String,Any}("row_id"=>_row_id(protocol,issue),"issue"=>time_text(issue),"accepted"=>accepted,
        "reason"=>reason,"ground_anchor"=>time_text(timing.latest_ground_center),
        "omni_anchor_or_null"=>(mode=="mixed8" ? time_text(timing.latest_omni_arrival_start) : nothing),
        "label_ready"=>time_text(timing.label_assumed_ready),"first_feedback"=>time_text(timing.first_scheduled_feedback),
        "source_support_sha256"=>_support_digest(index,ground.queries,wind.queries),
        "features_or_null"=>(accepted ? float_bits.(features) : nothing))
    row["feature_row_sha256"]=digest(canonical_bytes(row))
    row,ground11
end

"""
    seal_feature_schedule(source, issues; protocol)

Reconstruct every explicitly scheduled synthetic issue from its completed-source
feature window. Retain rejections, query no target projection, and return an
immutable uncommitted draft. Native producer/check receipts are required to
publish/load a committed schedule.
"""
function seal_feature_schedule(source::AuditedSources,issues;protocol::RefitProtocol)
    _seal_schedule(source,issues,protocol;independent=false)
end
function _seal_schedule(source::AuditedSources,issues,protocol;independent)
    require(_bundle(source).protocol.sha256==protocol.sha256,"source/schedule protocol differs")
    require(issues isa AbstractVector || issues isa Tuple,"schedule requires a bounded issue array")
    issues isa AbstractVector && Base.require_one_based_indexing(issues)
    require(0<length(issues)<=MAX_ROWS && all(t->t isa DateTime && mod(Dates.value(t),60_000)==0,issues),"invalid whole-minute issue array")
    require(all(issues[i]<issues[i+1] for i in 1:length(issues)-1),"schedule issues must increase strictly")
    index=_source_index(source);rows=Any[]
    for issue in issues
        row,_=_feature_row(source,index,issue;independent)
        push!(rows,row)
    end
    chunks=_pack_rows(rows);scheduled_count=length(rows);accepted=count(row->row["accepted"],rows)
    common=_draft_common(protocol,"feature_schedule";source=_pin(source))
    data_sha=_frame_digest("DBDTSCHEDULE1",(protocol.sha256,chunks...))
    metadata=Dict{String,Any}("scientific_context"=>common,"source_audit_pin"=>_pin(source),"schedule_data_sha256"=>data_sha,
        "scheduled_rows"=>scheduled_count,"accepted_features"=>accepted,"rejected_features"=>scheduled_count-accepted)
    _Draft(_VERIFIED,"schedule",protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(metadata;limit=MAX_METADATA_BYTES),chunks,(),Int64(scheduled_count))
end

function _target_row(source,index,issue,asof;independent=false)
    timing=CMR.issue_timing(issue,_policy(_bundle(source).protocol))
    require(timing.label_assumed_ready<=asof,"target lookup before declared maturity")
    window=_ReadWindow{2}(index.ground,issue,issue+Minute(30),DateTime[])
    if !independent
        target=CMR.admit_target(window,issue,asof;policy=_policy(_bundle(source).protocol))
        return target.accepted, target.accepted ? target.value.target_max_dbdt : nothing,
            String(target.reason),_support_digest(index,window.queries)
    end
    rates=Float64[]
    for k in 1:30
        t=issue+Minute(k)
        if !haskey(window,t) || !haskey(window,t-Minute(1))
            return false,nothing,"missing_ground_target",_support_digest(index,window.queries)
        end
        before=window[t-Minute(1)];after=window[t]
        value=hypot(after[1]-before[1],after[2]-before[2])
        isfinite(value) || return false,nothing,"invalid_ground_target",_support_digest(index,window.queries)
        push!(rates,value)
    end
    true,maximum(rates),"complete",_support_digest(index,window.queries)
end

"""
    build_fit_input(schedule, source; split, protocol)

Reconstruct the complete requested supervised/excluded partition. Strict label
readiness is checked before target lookup. This returns an uncommitted input
draft; it does not create a qualified input from arbitrary features/targets.
"""
function build_fit_input(schedule::ScheduleHandle,source::AuditedSources;split,protocol::RefitProtocol)
    _build_input(schedule,source,split,protocol;independent=false)
end
function _build_input(schedule,source,split,protocol;independent)
    require(_bundle(schedule).protocol.sha256==protocol.sha256==_bundle(source).protocol.sha256,"input protocol differs")
    require(_context(schedule)["source_manifest_sha256_or_null"]==_pin(source),"input source differs from sealed schedule")
    spec=_split(protocol,split);first=read_time_text(spec["issue_start_inclusive"]);last=read_time_text(spec["issue_stop_exclusive"])
    boundary=read_time_text(spec["label_ready_before"]);asof=read_time_text(spec["target_admit_asof"])
    saved=_data_rows(schedule);index=_source_index(source)
    rows=Any[];excluded=Any[];scheduled=0;ready_excluded=0;target_rows=0
    for row in saved
        _schema(row,"schedule_row");issue=read_time_text(row["issue"])
        first<=issue<last || continue
        scheduled+=1
        if row["accepted"]!==true
            push!(excluded,Dict{String,Any}("row_id"=>row["row_id"],"issue"=>row["issue"],"feature_status"=>"rejected","label_status"=>"not_queried","reason"=>row["reason"]))
            continue
        end
        ready=read_time_text(row["label_ready"])
        # Strict '<' here is independent of CMR's within-stream '<=' admission.
        if !(ready<boundary)
            ready_excluded+=1
            push!(excluded,Dict{String,Any}("row_id"=>row["row_id"],"issue"=>row["issue"],"feature_status"=>"accepted","label_status"=>"not_queried","reason"=>"split_not_strictly_ready"))
            continue
        end
        rebuilt,ground11=_feature_row(source,index,issue;independent)
        require(rebuilt==row,"sealed source/feature row does not reconstruct")
        target_rows+=1
        accepted,target,reason,support=_target_row(source,index,issue,asof;independent)
        if !accepted
            push!(excluded,Dict{String,Any}("row_id"=>row["row_id"],"issue"=>row["issue"],"feature_status"=>"accepted","label_status"=>"rejected","reason"=>reason))
            continue
        end
        scale=1+ground11[3];require(isfinite(scale) && scale>0,"invalid normalized score scale")
        push!(rows,Dict{String,Any}("row_id"=>row["row_id"],"issue"=>row["issue"],"label_ready"=>row["label_ready"],
            "target_admit_asof"=>time_text(asof),"target_support_sha256"=>support,"feature_row_sha256"=>row["feature_row_sha256"],
            "features"=>row["features_or_null"],"target"=>float_bits(target),"log_target"=>float_bits(log1p(target)),"scale"=>float_bits(scale)))
    end
    require(length(rows)>=spec["min_supervised_rows"],"supervised cohort fails its predeclared minimum")
    chunks=_pack_rows(rows);excluded_chunks=_pack_rows(excluded)
    rowpin=_ordered_digest([row["row_id"] for row in rows])
    inputpin=_frame_digest("DBDTFITINPUT1",(protocol.sha256,_pin(schedule),_jsontext(spec),
        _frame_digest("DBDTACCEPTED1",chunks),_frame_digest("DBDTEXCLUDED1",excluded_chunks)))
    common=_draft_common(protocol,"input_audit";source=_pin(source),schedule=_pin(schedule))
    metadata=Dict{String,Any}("scientific_context"=>common,"source_audit_pin"=>_pin(source),"schedule_manifest_sha256"=>_pin(schedule),
        "split"=>spec,"input_data_sha256"=>inputpin,"row_id_sha256"=>rowpin,"supervised_rows"=>length(rows),"excluded_rows"=>length(excluded))
    members=Pair{String,String}[_row_name(i,"excluded")=>chunk for (i,chunk) in enumerate(excluded_chunks)]
    checks=Dict{String,Any}("checked_rows"=>scheduled,"eligible_rows"=>length(rows),"excluded_readiness_rows"=>ready_excluded,
        "pre_gate_target_lookups"=>0,"post_gate_target_rows"=>target_rows,"failures"=>0)
    push!(members,"readiness-checks"=>_jsontext(checks))
    _Draft(_VERIFIED,"fit_input",protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(metadata;limit=MAX_METADATA_BYTES),chunks,Tuple(members),Int64(length(rows)))
end

function _audit_source_bodies(protocol::RefitProtocol;independent=false)
    spec=_spec(protocol);records=Any[];read_ids=String[];controls=Dict{String,String}()
    for permission in spec["source_permissions"]
        permission["disposition"]=="approved" || continue
        format=permission["format"]
        require(format in ("dbdt_synthetic_ground_vectors_v1","dbdt_synthetic_arrival_wind_v1"),"actual parser is unimplemented")
        wind=format=="dbdt_synthetic_arrival_wind_v1"
        require(!wind || spec["layout"]=="mixed8","ground11 source audit cannot open wind")
        path=permission["path_or_null"];pin=permission["sha256_or_null"]
        require(isabspath(path) && !islink(path) && isfile(path),"active source audit requires an ordinary current-host absolute approved path")
        bytes=_guard_file(path,pin)
        require(length(bytes)==permission["bytes_or_null"],"source descriptor byte count differs")
        controls[path]=pin;body=decode_canonical(bytes)
        _schema(body,wind ? "synthetic_wind_body" : "synthetic_ground_body")
        require(body["schema"]==format && body["station"]==spec["station"] && body["product"]==permission["product"],"synthetic source identity differs")
        require(body["units"]==(wind ? Any["km/s","nT","nT","cm^-3"] : "nT"),"synthetic source units differ")
        start=read_time_text(permission["period_start"]);stop=read_time_text(permission["period_stop"])
        previous=nothing
        for row in _array(body["rows"];nonempty=true)
            _schema(row,wind ? "synthetic_wind_row" : "synthetic_ground_row")
            t=read_time_text(row["time"])
            require(start<=t<stop && mod(Dates.value(t),60_000)==0 && (previous===nothing || previous<t),"source times are outside period/irregular/unordered")
            previous=t;values=_array(row[wind ? "values" : "xy"])
            require(length(values)==(wind ? 4 : 2),"source vector dimension differs")
            decoded=read_float_bits.(values)
            wind && require(decoded[1]>=0 && decoded[4]>=0,"invalid synthetic wind V/density")
            # Independent path constructs the same canonical record by explicit
            # field placement rather than reusing a producer row transform.
            record=independent ? Dict{String,Any}("time"=>time_text(t),"body_sha256"=>pin,"kind"=>(wind ? "wind" : "ground"),
                "body_id"=>permission["body_id"],"values"=>[float_bits(v) for v in decoded]) :
                Dict{String,Any}("kind"=>(wind ? "wind" : "ground"),"time"=>row["time"],"values"=>copy(values),"body_id"=>permission["body_id"],"body_sha256"=>pin)
            push!(records,record)
        end
        require(_guard_file(path,pin)==bytes,"source body changed during audit")
        push!(read_ids,permission["body_id"])
    end
    require(!isempty(records) && length(records)<=MAX_ROWS,"empty/oversized synthetic source audit")
    sort!(records;by=row->(read_time_text(row["time"]),row["kind"]))
    require(length(unique((row["kind"],row["time"]) for row in records))==length(records),"duplicate source kind/time")
    for (path,pin) in controls;_guard_file(path,pin);end
    chunks=_pack_rows(records);data_sha=_frame_digest("DBDTSOURCE1",chunks)
    common=_draft_common(protocol,independent ? "source_independent_audit" : "source_audit")
    metadata=Dict{String,Any}("scientific_context"=>common,
        "source_permission_sha256"=>digest(canonical_bytes(spec["source_permissions"];limit=MAX_METADATA_BYTES)),
        "source_audit_data_sha256"=>data_sha,"read_body_ids"=>read_ids,
        "ground_rows"=>count(row->row["kind"]=="ground",records),"wind_rows"=>count(row->row["kind"]=="wind",records))
    _Draft(_VERIFIED,"source_audit",protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(metadata;limit=MAX_METADATA_BYTES),chunks,(),Int64(length(records)))
end
