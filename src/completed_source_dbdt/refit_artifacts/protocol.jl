const MIXED_FEATURES = ("dbdt_now","dbdt_mean30","dbdt_max30","dbdt_std30","V","Bz","Bs","VBs")
const GROUND_FEATURES = Tuple(CF.FEATURE_NAMES)
const PROTOCOL_SCHEMA = "dbdt_completed_refit_protocol_v1"
const ARTIFACT_KINDS = ("schedule","fit_input","refit_point","calibration","native_job","source_audit")
const OPERATIONS = ("source_audit","source_independent_audit","feature_schedule","input_audit",
    "input_independent_audit","fit_ridge","fit_independent_check","calibrate","cal_independent_check","load_cal_check")
const PRIVATE_FIELDS = Dict{String,Tuple}(
    "synthetic_ground_body"=>("schema","station","product","units","rows"),
    "synthetic_wind_body"=>("schema","station","product","units","rows"),
    "synthetic_ground_row"=>("time","xy"),
    "synthetic_wind_row"=>("time","values"),
    "source_audit_record"=>("kind","time","values","body_id","body_sha256"),
    "native_payload_chunk"=>("schema","ordinal","chunks","metadata","rows"),
    "source_payload_metadata"=>("scientific_context","source_permission_sha256","source_audit_data_sha256","read_body_ids","ground_rows","wind_rows"),
    "source_independent_check_payload"=>("schema","protocol_sha256","source_permission_sha256","producer_source_data_sha256","rebuilt_source_data_sha256","ground_rows","wind_rows","read_body_ids","code_hashes","runtime_id","result"),
    "schedule_payload_metadata"=>("scientific_context","source_audit_pin","schedule_data_sha256","scheduled_rows","accepted_features","rejected_features"),
    "independent_schedule_check_payload"=>("schema","protocol_sha256","source_audit_pin","producer_schedule_data_sha256","rebuilt_schedule_data_sha256","rows","accepted","rejected","code_hashes","runtime_id","result"),
    "input_payload_metadata"=>("scientific_context","source_audit_pin","schedule_manifest_sha256","split","input_data_sha256","row_id_sha256","supervised_rows","excluded_rows"),
    "native_control_entry"=>("name","sha256"),
    "dependency_pin_entry"=>("role","manifest_sha256"),
    "readiness_gate_checks"=>("checked_rows","eligible_rows","excluded_readiness_rows","pre_gate_target_lookups","post_gate_target_rows","failures"),
    "feature_request_metadata"=>("source_manifest_sha256",),
    "split_request"=>("schema","schedule_manifest_sha256","split")
)

function _schema(value, name::String)
    haskey(PRIVATE_FIELDS,name) || return schema_fields(value,name)
    require(value isa AbstractDict && all(k->k isa String,keys(value)),"expected private schema object")
    require(length(value)==length(PRIVATE_FIELDS[name]) && Set(keys(value))==Set(PRIVATE_FIELDS[name]),"unknown or missing private schema field")
    value
end
_jsontext(value;limit=MAX_MEMBER_BYTES) = String(canonical_bytes(value;limit))
_jsonbytes(value) = Vector{UInt8}(codeunits(value))
_bool(value) = (require(value isa Bool,"expected Bool"); value)
_nullable_sha(value) = value===nothing ? nothing : sha_string(value)
_gitsha(value) = (text=bounded_string(value;limit=40);require(occursin(r"^[0-9a-f]{40}$",text),"invalid Git identity");text)
function _absolute_for_os(path,os)
    path isa AbstractString || return false
    all(c->!iscntrl(c),path) || return false
    os=="NT" ? occursin(r"^[A-Za-z]:[\\/]",path) :
        os in ("Linux","Darwin") ? (startswith(path,"/") && !startswith(path,"//")) : false
end
_portable_absolute(path)=_absolute_for_os(path,"NT") || _absolute_for_os(path,"Linux")

function _array(value;maximum=MAX_ARRAY,nonempty=false)
    require(value isa AbstractVector,"expected array")
    Base.require_one_based_indexing(value)
    require((!nonempty || !isempty(value)) && length(value)<=maximum,"array cardinality outside bounds")
    value
end

function _permissions(value, station, ground_product, wind_product)
    rows=_array(value;maximum=32,nonempty=true)
    ids=String[]
    for item in rows
        _schema(item,"source_permission")
        push!(ids,identifier(item["body_id"]))
        require(identifier(item["station"])==station,"source station differs")
        identifier(item["product"])
        require(item["neighbor_role"] in ("primary","left","right"),"unsupported neighbor role")
        require(item["disposition"] in ("approved","absent","unapproved"),"unsupported source disposition")
        first=read_time_text(item["period_start"]);last=read_time_text(item["period_stop"])
        require(first<last,"source period must increase")
        sha_string(item["approval_sha256"])
        identifier(item["format"])
        if item["disposition"]=="approved"
            path=bounded_string(item["path_or_null"];nonempty=true)
            require(_portable_absolute(path),"approved source metadata path must have supported absolute syntax")
            checked_integer(item["bytes_or_null"],1,Int64(16)*1024^3)
            sha_string(item["sha256_or_null"])
            require(item["format"] in ("dbdt_synthetic_ground_vectors_v1","dbdt_synthetic_arrival_wind_v1"),"actual source formats are unimplemented")
            require(item["product"]==(item["format"]=="dbdt_synthetic_ground_vectors_v1" ? ground_product : wind_product),"source format/product differs")
            require(item["format"]!="dbdt_synthetic_arrival_wind_v1" || wind_product!==nothing,"ground11 cannot approve a wind body")
        else
            require(all(item[k]===nothing for k in ("path_or_null","bytes_or_null","sha256_or_null")),"absent/unapproved bodies must have null paths/sizes/hashes")
        end
    end
    require(issorted(ids) && length(unique(ids))==length(ids),"source body ids must be unique and ordered")
    rows
end

function _split_spec(value)
    _schema(value,"split")
    identifier(value["name"])
    first=read_time_text(value["issue_start_inclusive"]);last=read_time_text(value["issue_stop_exclusive"])
    boundary=read_time_text(value["label_ready_before"]);asof=read_time_text(value["target_admit_asof"])
    require(first<last && first>=DateTime(2017) && last<=DateTime(2018),"split issue bounds must be within2017")
    require(mod(Dates.value(first),60_000)==0 && mod(Dates.value(last),60_000)==0,"split bounds must be whole minutes")
    require(boundary>=last && asof>=boundary,"split readiness/asof precedes issue stop")
    minimum=checked_integer(value["min_supervised_rows"],1,MAX_ROWS)
    iscal=first==DateTime(2017,10,1) && last in (DateTime(2017,11,1),DateTime(2018))
    require(minimum==(iscal ? 9 : 3),"synthetic minima are fixed3fit/9cal")
    value
end

function _validate_recipe(value, layout, splits)
    _schema(value,"ridge_recipe")
    require(value["family"]=="ridge","only ridge is implemented")
    require(value["split"] in splits,"recipe split is absent from protocol")
    require(value["selection_decision_pin"]===nothing,"R2 selected-refit verification is unimplemented")
    penalty=read_float_bits(value["penalty"])
    if layout=="mixed8"
        require(value["role"]=="fixed_mixed8" && penalty==1 && value["intercept_penalty"]===true && value["std_rule"]=="sample_std_plus_1e_minus9","mixed8 ridge convention differs")
    else
        require(value["role"]=="candidate" && penalty in (0.01,0.1,1.0,10.0,100.0) && value["intercept_penalty"]===false && value["std_rule"]=="sample_std_zero_nonfinite_to_one","ground11 ridge convention differs")
    end
    value
end

"""
    RefitProtocol(spec; expected_spec_sha256)

Validate the exact canonical, caller-pinned synthetic protocol. Actual source
formats/execution are not implemented. All names, splits, recipes and numerical
policies are immutable bytes; they are not proof that a fit was executed.
"""
struct RefitProtocol
    _bytes::String
    sha256::String
    function RefitProtocol(spec;expected_spec_sha256)
        expected=sha_string(expected_spec_sha256)
        bytes = if spec isa AbstractDict
            canonical_bytes(spec;limit=MAX_METADATA_BYTES)
        elseif spec isa AbstractString || spec isa AbstractVector{UInt8}
            spec isa AbstractVector && Base.require_one_based_indexing(spec)
            amount=spec isa AbstractString ? ncodeunits(spec) : length(spec)
            require(0<amount<=MAX_METADATA_BYTES,"protocol byte bound exceeded before copy")
            spec isa AbstractString ? _jsonbytes(spec) : Vector{UInt8}(spec)
        else
            fail("protocol must be a canonical object or canonical bytes")
        end
        require(length(bytes)<=MAX_METADATA_BYTES && digest(bytes)==expected,"protocol pin differs")
        value=decode_canonical(bytes;limit=MAX_METADATA_BYTES)
        _schema(value,"protocol")
        require(value["schema"]==PROTOCOL_SCHEMA && value["evidence_mode"]=="synthetic_only","only the synthetic protocol is implemented")
        for k in ("protocol_id","station","ground_product");identifier(value[k]);end
        require(value["layout"] in ("mixed8","ground11"),"unsupported feature layout")
        layout=value["layout"]
        expected_names=layout=="ground11" ? GROUND_FEATURES : MIXED_FEATURES
        require(Tuple(_array(value["feature_names"]))==expected_names,"feature order differs")
        require(value["units"]==(layout=="ground11" ? "ground_nT_rate_nT_per_min" : "ground_nT_rate_nT_per_min_wind_km_s_nT_cm3"),"units differ")
        require(value["cohort_kind"] in ("natural","intersection"),"unsupported cohort kind")
        if layout=="ground11"
            require(value["wind_product_or_null"]===nothing,"ground11 cannot approve wind")
        else
            identifier(value["wind_product_or_null"])
        end
        _schema(value["policy"],"policy")
        for k in FIELD_KEYS["policy"];checked_integer(value["policy"][k],0,typemax(Int64));end
        policy=value["policy"]
        CMR.ReplayPolicy(;ground_extra_ms=policy["ground_extra_ms"],omni_extra_ms=policy["omni_extra_ms"],label_extra_ms=policy["label_extra_ms"])
        require(value["neighbor_policy"]=="explicit_approved_only","neighbor policy differs")
        _permissions(value["source_permissions"],value["station"],value["ground_product"],value["wind_product_or_null"])
        names=String[]
        for split in _array(value["splits"];maximum=16,nonempty=true)
            _split_spec(split);push!(names,split["name"])
        end
        require(length(unique(names))==length(names),"duplicate split name")
        require(value["minima_mode"]=="synthetic_oracle","actual minima/execution are unimplemented")
        recipes=_array(value["model_recipes"];maximum=16,nonempty=true)
        for recipe in recipes;_validate_recipe(recipe,layout,names);end
        require(value["selection_rule"]=="frozen_2017_mse_first_exact_tie_ridge_boost_forest_v1","selection policy differs")
        require(value["numeric_contract"]=="directed_ridge_fit_v1_portable_prediction_v2","numeric contract differs")
        require(value["runtime_contract"]=="owned_julia_1_12_6_or_1_13_1_single_threaded_v1","synthetic runtime contract differs")
        require(value["availability_kind"]=="assumed_completed_source_intervals" && value["measured_receipts_available"]===false,"availability declaration differs")
        new(String(bytes),expected)
    end
end

_spec(protocol::RefitProtocol) = decode_canonical(protocol._bytes;limit=MAX_METADATA_BYTES)
function _policy(protocol::RefitProtocol)
    v=_spec(protocol)["policy"]
    CMR.ReplayPolicy(;ground_extra_ms=v["ground_extra_ms"],omni_extra_ms=v["omni_extra_ms"],label_extra_ms=v["label_extra_ms"])
end
function _split(protocol::RefitProtocol,name)
    text=identifier(name)
    for row in _spec(protocol)["splits"];row["name"]==text && return row;end
    fail("requested split is not in protocol")
end

"""
    RidgeRecipe(protocol; role, penalty, decision_pin=nothing)

Select one exact predeclared ridge recipe. Mixed8 uses penalty1 including bias
and sample standard deviation plus1e-9. Ground11 candidates preserve the frozen
five penalties and unpenalized bias. Selected refits require the later R2 chain.
"""
struct RidgeRecipe
    _bytes::String
    sha256::String
    function RidgeRecipe(protocol::RefitProtocol;role,penalty,decision_pin=nothing)
        require(decision_pin===nothing,"R2 selected-refit verification is unimplemented")
        name=role isa Symbol ? (role===:candidate ? "candidate" : role===:fixed_mixed8 ? "fixed_mixed8" : fail("unsupported ridge role")) : identifier(role)
        value=float_value(penalty)
        matches=[recipe for recipe in _spec(protocol)["model_recipes"] if recipe["role"]==name && read_float_bits(recipe["penalty"])==value]
        require(length(matches)==1,"ridge recipe requires exactly one predeclared role/penalty match")
        text=_jsontext(only(matches));new(text,digest(text))
    end
end
_recipe(value::RidgeRecipe) = decode_canonical(value._bytes)
