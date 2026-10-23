function _cal_info(input::FitInput,purpose,next_issue)
    k=_context(input)["kind_context"];split=k["split"]
    first=read_time_text(split["issue_start_inclusive"]);last=read_time_text(split["issue_stop_exclusive"])
    expected=first==DateTime(2017,10,1) && last==DateTime(2017,11,1) ? "october_development" :
        first==DateTime(2017,10,1) && last==DateTime(2018) ? "octdec_final" : fail("input is not a declared calibration split")
    name=purpose isa Symbol ? String(purpose) : bounded_string(purpose)
    require(name==expected && next_issue isa DateTime && next_issue==read_time_text(split["label_ready_before"]),"calibration purpose/boundary differs from fixed split")
    expected
end
function _sorted_projection(rows)
    scores=[read_float_bits(row["score"]) for row in rows]
    permutation=sortperm(eachindex(rows);by=i->(scores[i],i))
    Any[Dict{String,Any}("rank"=>rank,"chronological_ordinal"=>ordinal,"score"=>rows[ordinal]["score"]) for (rank,ordinal) in enumerate(permutation)]
end
function _cal_rows(point::PointHandle,input::FitInput,next_issue)
    parts=_ridge_parts(point);rows=_data_rows(input);result=Any[]
    for (ordinal,row) in enumerate(rows)
        _schema(row,"fit_row")
        require(read_time_text(row["label_ready"])<next_issue,"calibration label is not strictly ready before use")
        target=read_float_bits(row["target"]);y=log1p(target)
        require(float_bits(y)==row["log_target"],"calibration log target does not reconstruct under this native runtime")
        prediction=_predict_encoded(parts,row["features"]);scale=read_float_bits(row["scale"])
        difference=y-prediction;require(isfinite(difference)&&isfinite(scale)&&scale>0,"invalid calibration residual/scale")
        score=difference/scale;require(isfinite(score),"nonfinite signed calibration score")
        push!(result,Dict{String,Any}("ordinal"=>ordinal,"row_id"=>row["row_id"],"issue"=>row["issue"],
            "label_ready"=>row["label_ready"],"target_support_sha256"=>row["target_support_sha256"],
            "feature_row_sha256"=>row["feature_row_sha256"],"target"=>row["target"],"log_target"=>float_bits(y),
            "prediction"=>float_bits(prediction),"scale"=>row["scale"],"score"=>float_bits(score)))
    end
    result
end

"""
    calibrate_point(point, input; purpose, next_issue, protocol)

Compute every chronological signed score from the loaded immutable point and
complete calibration input. Strict readiness precedes prediction. The draft
retains all rows and the full stable rank projection; it cannot publish itself.
"""
function calibrate_point(point::PointHandle,input::FitInput;purpose,next_issue,protocol::RefitProtocol)
    require(_bundle(point).protocol.sha256==protocol.sha256==_bundle(input).protocol.sha256,"calibration point/input protocol differs")
    pc=_context(point);ic=_context(input)
    require(pc["source_manifest_sha256_or_null"]==ic["source_manifest_sha256_or_null"] &&
        pc["schedule_manifest_sha256_or_null"]==ic["schedule_manifest_sha256_or_null"],"calibration point/input source or schedule differs")
    _cal_info(input,purpose,next_issue)
    before=_frame_digest("DBDTCALINPUTBUFFER1",_pack_rows(_data_rows(input)))
    rows=_cal_rows(point,input,next_issue);sorted=_sorted_projection(rows)
    require(before==_frame_digest("DBDTCALINPUTBUFFER1",_pack_rows(_data_rows(input))),"calibration input changed")
    common=_draft_common(protocol,"calibrate";source=ic["source_manifest_sha256_or_null"],schedule=ic["schedule_manifest_sha256_or_null"])
    receipt=_member_json(point,"fit-receipt.json");recipe=_member_json(point,"recipe.json")
    reference=Dict{String,Any}("manifest_sha256"=>_pin(point),"model_payload_sha256"=>receipt["model_payload_sha256"],
        "recipe_sha256"=>receipt["recipe_sha256"],"protocol_sha256"=>protocol.sha256,"role"=>recipe["role"])
    members=Pair{String,String}["point-reference.json"=>_jsontext(reference),"input-reference.json"=>_jsontext(_input_reference(input))]
    append!(members,[_row_name(i,"sorted")=>chunk for (i,chunk) in enumerate(_pack_rows(sorted))])
    _Draft(_VERIFIED,"calibration",protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(_input_reference(input)),
        _pack_rows(rows),Tuple(members),Int64(length(rows)))
end

function _check_cal_rows(rows,sorted,input::FitInput,point::PointHandle,first_issue;reproduce=true)
    require(length(rows)==length(_data_rows(input))>0,"calibration chronology count differs")
    previous=nothing;ids=String[];inputrows=_data_rows(input)
    for (i,row) in enumerate(rows)
        _schema(row,"cal_row");require(checked_integer(row["ordinal"],1,MAX_ROWS)==i,"calibration ordinal differs")
        issue=read_time_text(row["issue"]);ready=read_time_text(row["label_ready"])
        require((previous===nothing || previous<issue) && ready<first_issue,"calibration chronology/readiness differs")
        previous=issue;push!(ids,sha_string(row["row_id"]))
        source=inputrows[i]
        for key in ("row_id","issue","label_ready","target_support_sha256","feature_row_sha256","target","log_target","scale")
            require(row[key]==source[key],"calibration/input row projection differs")
        end
        y=read_float_bits(row["log_target"]);prediction=read_float_bits(row["prediction"]);scale=read_float_bits(row["scale"])
        require(scale>0 && isfinite(y-prediction) && float_bits((y-prediction)/scale)==row["score"],"saved signed score arithmetic differs")
    end
    require(length(unique(ids))==length(ids),"duplicate calibration row id")
    for (i,row) in enumerate(sorted)
        _schema(row,"sorted_row")
        require(checked_integer(row["rank"],1,MAX_ROWS)==i,"sorted rank differs")
        checked_integer(row["chronological_ordinal"],1,length(rows));read_float_bits(row["score"])
    end
    require(sorted==_sorted_projection(rows),"stable sorted projection/permutation differs")
    reproduce && require(rows==_cal_rows(point,input,first_issue),"native full calibration reproduction differs")
    nothing
end

"""
    completed_calibration(calibration; first_issue)

Return a fresh legacy completed calibration and a separate immutable artifact
pin. Synthetic source provenance remains false; a legacy purpose label is not
evidence that actual fitting occurred.
"""
function completed_calibration(calibration::CalHandle;first_issue)
    context=_context(calibration);k=context["kind_context"]
    require(first_issue isa DateTime && first_issue>=read_time_text(k["first_allowed_issue"]),"calibration use precedes sealed boundary")
    rows=_data_rows(calibration)
    require(all(row->read_time_text(row["label_ready"])<first_issue,rows),"calibration labels are not strictly ready")
    scores=[read_float_bits(row["score"]) for row in rows];ready=[read_time_text(row["label_ready"]) for row in rows]
    legacy=D.CompletedCalibration(scores,ready;model_schema=D.POINT_SCHEMA,point_model_sha256=k["point_manifest_sha256"],
        purpose=:synthetic,policy=_policy(_bundle(calibration).protocol),mode=Symbol(_spec(_bundle(calibration).protocol)["layout"]))
    (calibration=legacy,artifact_pin=_pin(calibration),point_manifest_sha256=k["point_manifest_sha256"],source_bound_provenance_verified=false)
end
