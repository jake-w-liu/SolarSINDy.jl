function _input_values(input::FitInput;recompute_logs=false)
    rows=_data_rows(input);require(length(rows)==_bundle(input).rows && !isempty(rows),"fit input row count differs")
    p=length(_spec(_bundle(input).protocol)["feature_names"])
    X=Matrix{Float64}(undef,length(rows),p);y=Vector{Float64}(undef,length(rows))
    previous=nothing;ids=String[]
    for (i,row) in enumerate(rows)
        _schema(row,"fit_row");issue=read_time_text(row["issue"])
        require(previous===nothing || previous<issue,"supervised rows are not chronological")
        previous=issue;push!(ids,sha_string(row["row_id"]))
        values=_array(row["features"]);require(length(values)==p,"supervised feature dimension differs")
        for j in 1:p;X[i,j]=read_float_bits(values[j]);end
        target=read_float_bits(row["target"]);require(target>=0,"negative target")
        y[i]=read_float_bits(row["log_target"])
        recompute_logs && require(float_bits(log1p(target))==row["log_target"],"supervised log target does not reconstruct")
        require(read_float_bits(row["scale"])>0,"nonpositive score scale")
    end
    require(length(unique(ids))==length(ids),"duplicate supervised row id")
    (;rows,X,y)
end

function _ridge_preprocess(X,recipe)
    Base.require_one_based_indexing(X)
    require(X isa Matrix{Float64} && size(X,1)>=2 && all(isfinite,X),"invalid owned ridge design")
    center=vec(mean(X;dims=1));rawscale=vec(std(X;dims=1))
    constants=findall(j->all(==(X[1,j]),view(X,:,j)),1:size(X,2))
    scale=copy(rawscale)
    if recipe["std_rule"]=="sample_std_plus_1e_minus9"
        scale .+= 1e-9
    else
        scale[.!isfinite.(scale) .| (scale .== 0)] .= 1
    end
    require(all(isfinite,center) && all(s->isfinite(s)&&s>0,scale),"invalid native preprocessing")
    Z=hcat(ones(size(X,1)),(X.-center')./scale')
    require(all(isfinite,Z),"nonfinite native normalization")
    preprocessing=Dict{String,Any}("convention"=>recipe["std_rule"],"mean"=>float_bits.(center),
        "scale"=>float_bits.(scale),"constant_columns"=>constants)
    preprocessing,Z
end

function _normal_statistics(Z,y,recipe)
    penalty=read_float_bits(recipe["penalty"]);p=size(Z,2)
    weights=vcat(recipe["intercept_penalty"] ? penalty : 0.0,fill(penalty,p-1))
    A=Z'*Z+Diagonal(weights);b=Z'*y
    require(all(isfinite,A) && all(isfinite,b),"nonfinite native sufficient statistics")
    A,b,weights
end

function _native_statistics_bytes(A,b,layout)
    m=layout=="mixed8" ? 9 : layout=="ground11" ? 12 : fail("unsupported statistics layout")
    require(A isa AbstractMatrix && b isa AbstractVector,"invalid native statistics containers")
    Base.require_one_based_indexing(A,b)
    require(size(A)==(m,m) && length(b)==m,"native statistics dimension differs")
    _jsontext(Any[float_bits.(vec(permutedims(copy(A)))),float_bits.(copy(b))])
end
function _read_native_statistics(bytes,layout)
    m=layout=="mixed8" ? 9 : layout=="ground11" ? 12 : fail("unsupported statistics layout")
    values=_array(decode_canonical(bytes));require(length(values)==2,"native statistic array count differs")
    A=_array(values[1]);b=_array(values[2]);require(length(A)==m*m && length(b)==m,"native statistic vector length differs")
    read_float_bits.(A),read_float_bits.(b)
end

function _input_reference(input::FitInput)
    b=_bundle(input);k=_context(input)["kind_context"];rows=_data_rows(input);split=k["split"]
    reference=_member_json(input,"independent-input-check.json")
    Dict{String,Any}("manifest_sha256"=>b.manifest_sha256,"source_audit_pin"=>_context(input)["source_manifest_sha256_or_null"],
        "independent_input_pin"=>reference["check_payload_sha256"],"row_id_sha256"=>k["row_id_sha256"],
        "first_issue"=>first(rows)["issue"],"last_issue"=>last(rows)["issue"],
        "min_label_ready"=>time_text(minimum(read_time_text(row["label_ready"]) for row in rows)),
        "max_label_ready"=>time_text(maximum(read_time_text(row["label_ready"]) for row in rows)),
        "issue_start_inclusive"=>split["issue_start_inclusive"],"issue_stop_exclusive"=>split["issue_stop_exclusive"],
        "label_ready_before"=>split["label_ready_before"],"target_admit_asof"=>split["target_admit_asof"],"rows"=>length(rows))
end

"""
    fit_point(input, recipe)

Perform the prescribed ridge solve on owned, sealed synthetic input buffers.
The immutable draft requires genuine producer and independent native receipts
before publication. It does not establish actual-source fit provenance.
"""
function fit_point(input::FitInput,recipe::RidgeRecipe)
    protocol=_bundle(input).protocol;values=_input_values(input);r=_recipe(recipe)
    require(r["split"]==_context(input)["kind_context"]["split"]["name"],"recipe/input split differs")
    require(any(candidate->candidate==r,_spec(protocol)["model_recipes"]),"recipe is not in input protocol")
    before=_frame_digest("DBDTFITBUFFER1",_pack_rows(values.rows));X=copy(values.X);y=copy(values.y)
    preprocessing,Z=_ridge_preprocess(X,r)
    A,b,_=_normal_statistics(Z,y,r)
    statistic_bytes=_native_statistics_bytes(A,b,_spec(protocol)["layout"])
    coefficients=A\b
    require(all(isfinite,coefficients),"nonfinite native ridge solution")
    require(_frame_digest("DBDTFITBUFFER1",_pack_rows(_data_rows(input)))==before && X==values.X && y==values.y,"fit input changed during solve")
    model=Dict{String,Any}("kind"=>"ridge","coefficients"=>float_bits.(coefficients))
    context=_context(input);common=_draft_common(protocol,"fit_ridge";
        source=context["source_manifest_sha256_or_null"],schedule=context["schedule_manifest_sha256_or_null"])
    preprocessing_text=_jsontext(preprocessing);model_text=_jsontext(model)
    receipt=Dict{String,Any}("schema"=>"dbdt_completed_fit_receipt_v1","protocol_sha256"=>protocol.sha256,
        "scientific_context_sha256"=>digest(_jsontext(common;limit=MAX_METADATA_BYTES)),"input_manifest_sha256"=>_pin(input),
        "input_data_sha256"=>context["kind_context"]["input_data_sha256"],"row_id_sha256"=>context["kind_context"]["row_id_sha256"],
        "recipe_sha256"=>recipe.sha256,"preprocessing_sha256"=>digest(preprocessing_text),"model_payload_sha256"=>digest(model_text),
        "loaded_code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"operation"=>"fit_ridge","completed"=>true,
        "diagnostic_payload_sha256"=>digest(statistic_bytes))
    members=("recipe.json"=>recipe._bytes,"preprocessing.json"=>preprocessing_text,"model.json"=>model_text,
        "fit-input-reference.json"=>_jsontext(_input_reference(input)),"fit-receipt.json"=>_jsontext(receipt))
    _Draft(_VERIFIED,"refit_point",protocol,_jsontext(common;limit=MAX_METADATA_BYTES),_jsontext(receipt),(statistic_bytes,),members,Int64(length(y)))
end

function _ridge_parts(point::PointHandle)
    preprocessing=_member_json(point,"preprocessing.json");model=_member_json(point,"model.json")
    _schema(preprocessing,"ridge_preprocessing");_schema(model,"ridge_model")
    p=length(_spec(_bundle(point).protocol)["feature_names"])
    center=read_float_bits.(_array(preprocessing["mean"]));scale=read_float_bits.(_array(preprocessing["scale"]))
    beta=read_float_bits.(_array(model["coefficients"]))
    require(model["kind"]=="ridge" && length(center)==length(scale)==p && length(beta)==p+1 && all(>(0),scale),"invalid loaded ridge dimensions/scales")
    (;center,scale,beta)
end

function _prediction_inputs(parts,features)
    x=read_float_bits.(_array(features));p=length(parts.center)
    require(length(x)==p,"prediction feature dimension differs")
    differences=x.-parts.center;require(all(isfinite,differences),"prediction subtraction overflow")
    z=differences./parts.scale;require(all(isfinite,z),"prediction normalization overflow")
    products=z.*parts.beta[2:end];require(all(isfinite,products),"prediction product overflow")
    x,z
end
function _predict_encoded(parts,features)
    x,z=_prediction_inputs(parts,features)
    prediction=dot(vcat(1.0,z),parts.beta)
    require(isfinite(prediction),"prediction reduction overflow")
    _check_prediction_bound(x,parts.center,parts.scale,parts.beta,z,prediction)
    prediction
end

"""
    predict_log(point, schedule; protocol)

Predict accepted sealed feature rows from the loaded immutable point, without
targets. Return every attempted row, including feature rejections.
"""
function predict_log(point::PointHandle,schedule::ScheduleHandle;protocol::RefitProtocol)
    require(_bundle(point).protocol.sha256==protocol.sha256==_bundle(schedule).protocol.sha256,"point/schedule protocol differs")
    require(_context(point)["source_manifest_sha256_or_null"]==_context(schedule)["source_manifest_sha256_or_null"],"point/schedule source differs")
    parts=_ridge_parts(point);rows=_data_rows(schedule);result=NamedTuple[]
    for row in rows
        if row["accepted"]===true
            prediction=_predict_encoded(parts,row["features_or_null"])
            peak=read_float_bits(row["features_or_null"][3])
            scale=1+(_spec(protocol)["layout"]=="ground11" ? peak : log1p(peak))
            require(isfinite(scale)&&scale>0,"invalid predicted-row scale")
            push!(result,(row_id=row["row_id"],feature_row_sha256=row["feature_row_sha256"],accepted=true,
                prediction=prediction,scale=scale,point_manifest_sha256=_pin(point)))
        else
            push!(result,(row_id=row["row_id"],feature_row_sha256=row["feature_row_sha256"],accepted=false,
                prediction=nothing,scale=nothing,point_manifest_sha256=_pin(point)))
        end
    end
    result
end
