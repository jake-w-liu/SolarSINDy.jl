# Fixed-grid generated-data fitting with immutable data-only export authority.
"Fixed8 depth(4,6)-rounds(200,400)-bins(64,255) settings, depth outermost and bins innermost."
const GRID342=Tuple((;max_depth=depth,nrounds=rounds,nbins=bins)
    for depth in (4,6) for rounds in (200,400) for bins in (64,255))
const EVOTREES_SOURCE_SHA256="f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f"
function _trusted_evo_sha()
    library=Parent.EvoTrees
    _need(string(Base.pkgversion(library))=="0.18.7","installed EvoTrees version differs")
    root=pkgdir(library);_ordinary(root;directory=true);paths=String["Project.toml"]
    for name in ("src","ext")
        directory=joinpath(root,name);_ordinary(directory;directory=true)
        for (current,dirs,files) in walkdir(directory;follow_symlinks=false)
            _ordinary(current;directory=true)
            for dir in dirs;_ordinary(joinpath(current,dir);directory=true);end
            for file in files
                endswith(file,".jl")||continue
                path=_ordinary(joinpath(current,file));push!(paths,replace(relpath(path,root),'\\'=>'/'))
                _need(length(paths)<=1024,"library source inventory bound")
            end
        end
    end
    files=[Dict("path"=>name,"sha256"=>bytes2hex(open(sha256,joinpath(root,name)))) for name in sort!(paths)]
    actual=bytes2hex(sha256(_canonical(Dict("schema"=>"evotrees_cpu_source_closure_v1","version"=>"0.18.7","files"=>files))))
    _need(actual==EVOTREES_SOURCE_SHA256,"installed EvoTrees source closure differs");actual
end
function _admission_object(input_id,fixture,audit)
    Dict("schema"=>"matched342_synthetic_admission_v1","evidence_mode"=>"synthetic_only",
        "input_id"=>input_id,"fixture_sha256"=>fixture,"external_audit_witness_sha256"=>audit)
end
"Immutable synthetic-only input/fixture/external-witness binding; grants no actual-source fit or release permission."
struct SyntheticAdmission342
    input_id::String
    fixture_sha256::String
    external_audit_witness_sha256::String
    admission_id::String
    function SyntheticAdmission342(input::HourlyInputs342;fixture_sha256,external_audit_witness_sha256)
        source_identity();raw=_snapshot(input)
        _need(_hex(fixture_sha256)&&_hex(external_audit_witness_sha256),"external synthetic fixture/audit pins required")
        fixture=String(fixture_sha256);audit=String(external_audit_witness_sha256)
        id=bytes2hex(sha256(_canonical(_admission_object(raw.input_id,fixture,audit))))
        new(raw.input_id,fixture,audit,id)
    end
end
function _check_admission(admission::SyntheticAdmission342)
    _need(all(_hex,(admission.input_id,admission.fixture_sha256,admission.external_audit_witness_sha256,admission.admission_id)),"synthetic admission domain differs")
    _need(bytes2hex(sha256(_canonical(_admission_object(admission.input_id,admission.fixture_sha256,
        admission.external_audit_witness_sha256))))==admission.admission_id,"synthetic admission ID differs");nothing
end
"Independent expected synthetic admission, cell/grid, cohort/preprocessing, library-source and loss-contract pins."
struct ExpectedModelBindings342
    admission::SyntheticAdmission342
    delay_hours::Int
    horizon::Int
    setting_index::Int
    training_mask_sha256::String
    preprocessing_sha256::String
    evotrees_source_sha256::String
    loss_contract_sha256::String
    function ExpectedModelBindings342(admission::SyntheticAdmission342,delay,horizon,setting;
            training_mask_sha256,preprocessing_sha256,evotrees_source_sha256,loss_contract_sha256)
        _check_admission(admission);d=_int(delay,0,1);h=_int(horizon,4,6);s=_int(setting,1,8)
        _need(h in (4,6),"cell horizon differs")
        _need(all(_hex,(training_mask_sha256,preprocessing_sha256,evotrees_source_sha256,loss_contract_sha256)),"external model context pins required")
        _need(evotrees_source_sha256==EVOTREES_SOURCE_SHA256,"trusted library source pin differs")
        new(admission,d,h,s,String(training_mask_sha256),String(preprocessing_sha256),String(evotrees_source_sha256),String(loss_contract_sha256))
    end
end
function _check_bindings(expected::ExpectedModelBindings342)
    _check_admission(expected.admission)
    _need(expected.delay_hours in (0,1)&&expected.horizon in (4,6)&&1<=expected.setting_index<=8&&
        all(_hex,(expected.training_mask_sha256,expected.preprocessing_sha256,expected.loss_contract_sha256))&&
        expected.evotrees_source_sha256==EVOTREES_SOURCE_SHA256,"model expected context domain differs")
    _trusted_evo_sha();nothing
end
function _recipe(setting)
    merge(GRID342[setting],(;seed=22022026,eta=0.05,min_weight=64.0,lambda=1.0,L2=1.0,gamma=0.0,rowsample=1.0,colsample=1.0))
end
function _training_mask_sha(raw,cohort,boundary,validation_start,validation_end)
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("matched342-training-mask-v1\n$(raw.input_id)\n"))
    header=Int64[raw.delay_hours,cohort.horizon,Dates.value(boundary),Dates.value(validation_start),Dates.value(validation_end),length(cohort.train)]
    SHA.update!(ctx,reinterpret(UInt8,htol.(header)));record=Vector{Int64}(undef,8)
    for i in cohort.train
        issue=cohort.issues[i];anchor=_shift(issue,-1-raw.delay_hours);target=_shift(issue,cohort.horizon-1)
        ai=_index(raw,anchor);ti=_index(raw,target)
        record.=(i,Dates.value(issue),Dates.value(anchor),Dates.value(_shift(anchor,1)),Dates.value(raw.received[15,ai]),
            Dates.value(target),Dates.value(_shift(target,1)),Dates.value(raw.received[15,ti]))
        record.=htol.(record);SHA.update!(ctx,reinterpret(UInt8,record))
    end
    bytes2hex(SHA.digest!(ctx))
end
function _prepare_synthetic(input,admission,horizon;training_boundary,validation_start,validation_end)
    source_identity();_check_admission(admission);_trusted_evo_sha()
    _need(all(t->t isa DateTime&&floor(t,Hour)==t,(training_boundary,validation_start,validation_end)),"exact-hour split boundaries required")
    raw=_snapshot(input);_need(raw.input_id==admission.input_id,"synthetic admission input differs")
    cohort=cohorts(raw,horizon;training_boundary,validation_start,validation_end)
    _need(!isempty(cohort.train)&&!isempty(cohort.validation),"nonempty train and perlead evaluation cohorts required")
    design=build_features(raw,cohort.issues).matrix;encoder=fit_encoder(design,cohort.train)
    targets=Float64[raw.values[15,i+cohort.horizon]-raw.values[15,i-raw.delay_hours] for i in cohort.train]
    _need(all(isfinite,targets),"target increment overflow")
    mask=_training_mask_sha(raw,cohort,training_boundary,validation_start,validation_end)
    (;raw,cohort,design,encoder,targets,mask)
end
"Derive pre-fit synthetic expected bindings from raw data and exact-hour split boundaries, without invoking the native fitter."
function prepare_synthetic_bindings(input::HourlyInputs342,admission::SyntheticAdmission342,horizon,setting;
        training_boundary::DateTime,validation_start::DateTime,validation_end::DateTime,loss_contract_sha256)
    s=_int(setting,1,8);prepared=_prepare_synthetic(input,admission,horizon;training_boundary,validation_start,validation_end)
    ExpectedModelBindings342(admission,input.delay_hours,prepared.cohort.horizon,s;
        training_mask_sha256=prepared.mask,preprocessing_sha256=preprocessing_sha256(prepared.encoder),
        evotrees_source_sha256=EVOTREES_SOURCE_SHA256,loss_contract_sha256)
end
function _snapshot_native(native,recipe)
    E=Parent.EvoTrees;R=MatchedScalarTreeReplay342
    _need(native isa E.EvoTree{E.MSE,1}&&native.K==1,"native scalar MSE model differs")
    info=native.info
    _need(all(key->haskey(info,key),(:feature_names,:edges,:feattypes,:featbins,:nrounds)),"native numeric metadata missing")
    _need(String.(info[:feature_names])==collect(FEATURE_NAMES)&&length(info[:feattypes])==WIDTH&&all(x->x===true,info[:feattypes]),"native ordered numeric feature schema differs")
    _need(info[:nrounds] isa Integer&&!(info[:nrounds] isa Bool)&&info[:nrounds]==recipe.nrounds&&
        length(info[:featbins])==WIDTH&&all(x->x==recipe.nbins,info[:featbins]),"native round/bin metadata differs")
    trees=NamedTuple[]
    for tree in native.trees
        _need(size(tree.pred)==(1,length(tree.feat)),"native scalar prediction geometry differs")
        push!(trees,(;feature=copy(tree.feat),condition=Int.(tree.cond_bin),prediction=vec(copy(tree.pred)),split=copy(tree.split)))
    end
    model=R.Model(info[:edges],trees;max_depth=recipe.max_depth,nrounds=recipe.nrounds)
    _need(model.nbins==recipe.nbins,"native edge count differs");model
end
_component_sha()=bytes2hex(sha256(_canonical(Dict(String(name)=>pin for (name,pin) in source_identity()))))
struct _ControlledFitToken342 end
"Immutable controlled-fit capsule containing copied replay data/encoder/context/recipe; mutable native arrays are excluded."
struct FittedCell342
    model::MatchedScalarTreeReplay342.Model
    encoder::Encoder342
    expected::ExpectedModelBindings342
    recipe::NamedTuple
    component_sha256::String
    seal_sha256::String
    function FittedCell342(model,encoder,expected,recipe,component,seal,::_ControlledFitToken342)
        new(model,encoder,expected,recipe,component,seal)
    end
end
function _capsule_seal(model,encoder,expected,recipe,component)
    # Recipe floats are encoded as exact bit strings, never JSON float tokens.
    recipe_data=Dict{String,Any}(String(key)=>(value isa AbstractFloat ? _bits(value) : value) for (key,value) in pairs(recipe))
    bytes2hex(sha256(_canonical(Dict("model_id"=>_model_object(model,expected)["model_id"],
        "preprocessing_sha256"=>preprocessing_sha256(encoder),"component_sha256"=>component,"recipe"=>recipe_data))))
end
function _same_expected(a,b)
    _check_bindings(a);_check_bindings(b);_expected_fields(a)==_expected_fields(b)
end
function _check_capsule(capsule::FittedCell342,expected)
    _need(expected isa ExpectedModelBindings342&&_same_expected(capsule.expected,expected),"independent expected capsule context differs")
    _need(capsule.component_sha256==_component_sha()&&capsule.recipe==_recipe(expected.setting_index)&&
        preprocessing_sha256(capsule.encoder)==expected.preprocessing_sha256,"capsule source/recipe/encoder differs")
    _need(_capsule_seal(capsule.model,capsule.encoder,expected,capsule.recipe,capsule.component_sha256)==capsule.seal_sha256,"capsule seal differs");nothing
end
"Fit the selected exact fixed recipe on generated data; return sealed capsule, separate native model, and train/evaluation counts. No actual-source mode exists."
function fit_synthetic_cell(input::HourlyInputs342,expected::ExpectedModelBindings342;
        training_boundary::DateTime,validation_start::DateTime,validation_end::DateTime)
    _check_bindings(expected);_need(input.delay_hours==expected.delay_hours,"fit cell delay differs")
    prepared=_prepare_synthetic(input,expected.admission,expected.horizon;training_boundary,validation_start,validation_end)
    _need(prepared.mask==expected.training_mask_sha256&&preprocessing_sha256(prepared.encoder)==expected.preprocessing_sha256,"fit input cohort/preprocessing differs from external expected pins")
    recipe=_recipe(expected.setting_index);component=_component_sha()
    standardized=transform(prepared.encoder,prepared.design)
    native=Parent.gbm_fit(standardized[prepared.cohort.train,:],prepared.targets;recipe...,feature_names=collect(FEATURE_NAMES))
    model=_snapshot_native(native,recipe)
    _need(_snapshot(input).input_id==prepared.raw.input_id&&_component_sha()==component,"fit input/source changed")
    _check_bindings(expected)
    seal=_capsule_seal(model,prepared.encoder,expected,recipe,component)
    capsule=FittedCell342(model,prepared.encoder,expected,recipe,component,seal,_ControlledFitToken342())
    _check_capsule(capsule,expected)
    (;capsule,native,training_rows=length(prepared.cohort.train),validation_rows=length(prepared.cohort.validation))
end
