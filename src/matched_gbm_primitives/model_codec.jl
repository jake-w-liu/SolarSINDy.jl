const _MODEL_KEYS=("schema","feature_count","feature_schema_sha256","evotrees_version","evotrees_source_sha256",
    "cell_delay","cell_horizon","setting_index","max_depth","nrounds","nbins","seed","training_mask_sha256",
    "preprocessing_sha256","input_admission_sha256","feature_contract_sha256","loss_contract_sha256",
    "numeric_feature_types","edges_f64_bits","trees","model_id")
const _TREE_KEYS=("feature","condition","prediction_f32_bits","split")
_bits32(x::Float32)=string(reinterpret(UInt32,x);base=16,pad=8)
function _float32(text)
    _need(text isa AbstractString&&occursin(r"\A[0-9a-f]{8}\z",text),"Float32 bit string differs")
    value=reinterpret(Float32,parse(UInt32,text;base=16));_need(isfinite(value),"nonfinite Float32 parameter");value
end
function _expected_fields(expected::ExpectedModelBindings342)
    grid=GRID342[expected.setting_index]
    Dict{String,Any}("schema"=>"matched_gbm_scalar342_model_v1","feature_count"=>WIDTH,
        "feature_schema_sha256"=>FEATURE_SCHEMA_SHA256,"evotrees_version"=>"0.18.7",
        "evotrees_source_sha256"=>expected.evotrees_source_sha256,"cell_delay"=>expected.delay_hours,
        "cell_horizon"=>expected.horizon,"setting_index"=>expected.setting_index,
        "max_depth"=>grid.max_depth,"nrounds"=>grid.nrounds,"nbins"=>grid.nbins,"seed"=>22022026,
        "training_mask_sha256"=>expected.training_mask_sha256,"preprocessing_sha256"=>expected.preprocessing_sha256,
        "input_admission_sha256"=>expected.admission.admission_id,"feature_contract_sha256"=>FEATURE_CONTRACT_SHA256,
        "loss_contract_sha256"=>expected.loss_contract_sha256)
end
function _model_object(model::MatchedScalarTreeReplay342.Model,expected::ExpectedModelBindings342)
    _check_bindings(expected);grid=GRID342[expected.setting_index]
    _need((model.max_depth,model.nrounds,model.nbins)==(grid.max_depth,grid.nrounds,grid.nbins),"model is outside the exact selected fixed grid")
    object=_expected_fields(expected)
    object["numeric_feature_types"]=fill(true,WIDTH)
    object["edges_f64_bits"]=[_bits.(collect(edge)) for edge in model.edges]
    object["trees"]=[Dict{String,Any}("feature"=>collect(tree.feature),"condition"=>collect(tree.condition),
        "prediction_f32_bits"=>_bits32.(collect(tree.prediction)),"split"=>collect(tree.split)) for tree in model.trees]
    object["model_id"]=bytes2hex(sha256(_canonical(object)));object
end
"Exclusively write a sealed controlled-fit capsule's canonical21-field model under independent expected_bindings."
function save_model(path,capsule::FittedCell342;expected_bindings::ExpectedModelBindings342)
    _check_capsule(capsule,expected_bindings)
    result=_write_json(path,_model_object(capsule.model,expected_bindings));_check_capsule(capsule,expected_bindings);result
end
function _decode_model(object,expected::ExpectedModelBindings342)
    _check_bindings(expected);_exact_keys(object,_MODEL_KEYS)
    integers=("feature_count","cell_delay","cell_horizon","setting_index","max_depth","nrounds","nbins","seed")
    for key in integers;_int(object[key],0,22022026);end
    fields=_expected_fields(expected)
    _need(all(object[key]==value for (key,value) in fields),"model external context differs")
    _need(_hex(object["model_id"]),"model ID domain differs")
    without_id=Dict(key=>value for (key,value) in object if key!="model_id")
    _need(bytes2hex(sha256(_canonical(without_id)))==object["model_id"],"model self ID differs")
    flags=object["numeric_feature_types"]
    _need(flags isa AbstractVector&&length(flags)==WIDTH&&all(x->x===true,flags),"numeric feature type vector differs")
    grid=GRID342[expected.setting_index];raw_edges=object["edges_f64_bits"]
    _need(raw_edges isa AbstractVector&&length(raw_edges)==WIDTH,"model edge width differs")
    edges=Vector{Float64}[]
    for raw in raw_edges
        _need(raw isa AbstractVector&&length(raw)==grid.nbins-1,"model edge count differs")
        push!(edges,_float.(raw))
    end
    raw_trees=object["trees"];_need(raw_trees isa AbstractVector&&length(raw_trees)==grid.nrounds+1,"model tree count differs")
    trees=NamedTuple[]
    for (i,tree) in enumerate(raw_trees)
        _exact_keys(tree,_TREE_KEYS);nodes=i==1 ? 1 : 2^grid.max_depth-1
        _need(all(tree[key] isa AbstractVector&&length(tree[key])==nodes for key in _TREE_KEYS),"model tree geometry differs")
        _need(all(x->x isa Bool,tree["split"]),"model split flag types differ")
        push!(trees,(;feature=Int[_int(x,0,WIDTH) for x in tree["feature"]],
            condition=Int[_int(x,0,grid.nbins-1) for x in tree["condition"]],
            prediction=_float32.(tree["prediction_f32_bits"]),split=Bool.(tree["split"])))
    end
    model=MatchedScalarTreeReplay342.Model(edges,trees;max_depth=grid.max_depth,nrounds=grid.nrounds)
    _need(_model_object(model,expected)==object,"decoded model data differs");model
end
"Load immutable scalar model data under external content/context pins; performs no native loading, fitting, execution or qualification."
function load_model(path;expected_sha256,expected_bindings::ExpectedModelBindings342)
    source_identity();model=_decode_model(_read_json(path,expected_sha256),expected_bindings);source_identity();model
end
