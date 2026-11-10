"""Source-bound matched342 preparation/fitting and data-only state codecs.

Generated contexts cannot confer archival execution authority. Actual native
fitting additionally requires the externally supplied qualified authority.
"""
module BMatched342Core
using SolarSINDy, Dates, SHA, LinearAlgebra, JSON3
const R=parentmodule(@__MODULE__)
const BI=R.BI
const T=R.T
const P=R.P
const M=R.M
const REPO=R.PACKAGE_ROOT
const ACTUAL=:archival_assumed_completion_research
const GENERATED=:synthetic_only
const CELLS=((0,4),(0,6),(1,4),(1,6))
const PROTOCOL_SHA="7b6d6d6dca4b667c890d6cb763016871be0912c3417c23e20110ad2d3d6fe404"
const OWNER_SHA="3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1"
const LEGACY_OWNER_SHA="8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836"
const RSS_LIMIT=8*1024^3
const SOURCE_FACT_FIELDS=(:delay_hours,:source_proof_sha256,:source_admission_id,:input_view_id,
    :input_prefix_sha256,:raw_prefix_sha256,:source_sha256,:matched_input_id)
const RUNTIME_FIELDS=(:julia_version,:julia_binary_sha256,:evotrees_version,:evotrees_source_sha256,
    :threads,:blas_threads,:omp_threads)
struct _Token342 end
_need(ok,message)=T._need(ok,message)
function _fields(data,fields)
    _need(data isa NamedTuple||data isa AbstractDict||data isa JSON3.Object,"matched342 metadata object required")
    T._fields(data,fields)
end
_record(data,fields)=NamedTuple{fields}(Tuple(getproperty(data,k) for k in fields))
_source_rows(sources)=Tuple((;name=String(first(p)),sha256=String(last(p))) for p in sources)
function _sources(expected)
    current=T._hashmap(R.source_identity())
    _need(T._hashmap(expected)==current,"matched342 new component source map differs")
    Tuple(current)
end
function _clock_string(t::DateTime)
    string(Dates.value(t))
end
function _contracts()
    (;protocol_sha256=PROTOCOL_SHA,feature_contract_sha256=M.FEATURE_CONTRACT_SHA256,
      feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,loss_contract_sha256=R.LOSS_CONTRACT_SHA256,
      evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256)
end
function _binary_hash(domain,values)
    context=SHA.SHA2_256_CTX();SHA.update!(context,codeunits(domain));buffer=IOBuffer()
    for value in values
        word=value isa Integer ? reinterpret(UInt64,Int64(value)) :
            isnan(value) ? UInt64(0x7ff8000000000000) : reinterpret(UInt64,Float64(value))
        write(buffer,htol(word));position(buffer)>=65536&&SHA.update!(context,take!(buffer))
    end
    position(buffer)>0&&SHA.update!(context,take!(buffer));bytes2hex(SHA.digest!(context))
end
function _source_fact(a,input)
    (;delay_hours=a.metadata.delay_hours,source_proof_sha256=a.source_proof_sha256,
      source_admission_id=a.source_admission_id,input_view_id=a.view.view_id,
      input_prefix_sha256=a.view.input_prefix_sha256,raw_prefix_sha256=a.metadata.raw_prefix_sha256,
      source_sha256=a.view.source_sha256,matched_input_id=input.input_id)
end
function _pair_data(mode,admissions,inputs,sources)
    a=admissions[1];m=a.metadata
    (;schema="matched342_source_pair_context_v1",evidence_mode=String(mode),source_sha256=m.source_sha256,
      raw_prefix_sha256=m.raw_prefix_sha256,input_prefix_sha256=a.view.input_prefix_sha256,
      training_history_start_ms=_clock_string(m.prefix_start),prefix_stop_ms=_clock_string(m.prefix_stop),
      source_rows=m.prefix_rows,training_boundary_ms=_clock_string(m.training_boundary),
      validation_start_ms=_clock_string(m.validation_start),validation_end_ms=_clock_string(m.validation_end),
      source_bindings=ntuple(k->_source_fact(admissions[k],inputs[k]),2),
      feature_contract_sha256=M.FEATURE_CONTRACT_SHA256,feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,
      tuning_loss_contract_sha256=R.LOSS_CONTRACT_SHA256,baseline_sources=_source_rows(sources))
end
_context_id(mode,pair_id,fact)=T._id((;schema="matched342_source_context_v1",
    evidence_mode=String(mode),pair_id,source_binding=fact))
struct SourcePair342{Mode}
    admissions::NTuple{2,BI.BSourcePrefixAdmissionV2}
    inputs::NTuple{2,M.HourlyInputs342}
    sources::Tuple
    pair_id::String
    bindings::Tuple
    function SourcePair342{Mode}(admissions,inputs,sources,id,bindings,::_Token342) where Mode
        new{Mode}(admissions,inputs,sources,id,bindings)
    end
end
function _same_source_pair(admissions)
    a,b=admissions
    _need((a.metadata.delay_hours,b.metadata.delay_hours)==(0,1),"matched342 requires independently admitted L0 and L1")
    for key in (:evidence_mode,:source_sha256,:raw_prefix_sha256,:training_boundary,:validation_start,
                :validation_end,:prefix_start,:prefix_stop,:prefix_rows,:limits)
        _need(isequal(getproperty(a.metadata,key),getproperty(b.metadata,key)),"matched342 paired source/split fact differs: $key")
    end
    _need(a.view.evidence_mode==b.view.evidence_mode==a.metadata.evidence_mode&&
        a.view.source_sha256==b.view.source_sha256==a.metadata.source_sha256&&
        a.view.input_prefix_sha256==b.view.input_prefix_sha256&&a.view.source_hours==b.view.source_hours==a.metadata.prefix_rows,
        "matched342 paired physical source geometry differs")
    _need(a.metadata.evidence_mode!=GENERATED||a.metadata.source_sha256!=BI.ACTUAL_SOURCE,
        "matched342 generated source role cannot relabel the pinned actual archive")
    _need(a.source_proof_sha256!=b.source_proof_sha256&&a.view.view_id!=b.view.view_id,
        "matched342 source delay authorities must remain distinct")
end
"Bind two genuine independent B source admissions; this grants no native fitting permission."
function bind_source_pair(s0::BI.BSourcePrefixAdmissionV2,s1::BI.BSourcePrefixAdmissionV2;
        loss_contract_sha256,expected_sources)
    sources=_sources(expected_sources);admissions=(s0,s1)
    foreach(BI.recheck_source_v2,admissions);_same_source_pair(admissions)
    _need(loss_contract_sha256==R.LOSS_CONTRACT_SHA256,"matched342 frozen initial loss contract differs")
    mode=s0.metadata.evidence_mode;_need(mode in (GENERATED,ACTUAL),"matched342 evidence mode differs")
    inputs=ntuple(k->begin
        a=admissions[k]
        M.HourlyInputs342(a.view.input.time,a.view.input.values;delay_hours=k-1,source_sha256=a.view.source_sha256)
    end,2)
    data=_pair_data(mode,admissions,inputs,sources);id=T._id(data)
    bindings=ntuple(k->merge(data.source_bindings[k],(;context_id=_context_id(mode,id,data.source_bindings[k]))),2)
    pair=SourcePair342{mode}(admissions,inputs,sources,id,bindings,_Token342())
    recheck_pair(pair);pair
end
function recheck_pair(pair::SourcePair342{Mode}) where Mode
    _need(Mode in (GENERATED,ACTUAL),"matched342 context mode differs")
    _sources(pair.sources);foreach(BI.recheck_source_v2,pair.admissions);_same_source_pair(pair.admissions)
    _need(pair.admissions[1].metadata.evidence_mode==Mode,"matched342 source context mode changed")
    foreach(M._snapshot,pair.inputs)
    data=_pair_data(Mode,pair.admissions,pair.inputs,pair.sources)
    _need(T._id(data)==pair.pair_id&&pair.bindings==ntuple(k->merge(data.source_bindings[k],
        (;context_id=_context_id(Mode,pair.pair_id,data.source_bindings[k]))),2),"matched342 source context changed")
    true
end
function source_admissions(pair::SourcePair342)
    recheck_pair(pair);pair.admissions
end
function source_pair_metadata(pair::SourcePair342{Mode}) where Mode
    recheck_pair(pair);m=pair.admissions[1].metadata
    (;evidence_mode=Mode,pair_id=pair.pair_id,source_bindings=pair.bindings,
      source_sha256=m.source_sha256,raw_prefix_sha256=m.raw_prefix_sha256,
      input_prefix_sha256=pair.admissions[1].view.input_prefix_sha256,
      training_history_start=m.prefix_start,prefix_stop=m.prefix_stop,source_rows=m.prefix_rows,
      training_boundary=m.training_boundary,validation_start=m.validation_start,validation_end=m.validation_end,
      feature_contract_sha256=M.FEATURE_CONTRACT_SHA256,feature_schema_sha256=M.FEATURE_SCHEMA_SHA256,
      tuning_loss_contract_sha256=R.LOSS_CONTRACT_SHA256,baseline_sources=_source_rows(pair.sources))
end
"Observed runtime facts; metadata only, never an actual execution permit."
function runtime_identity()
    M.source_identity();evo=M._trusted_evo_sha()
    runtime=(;julia_version=string(VERSION),julia_binary_sha256=P._file_sha(joinpath(Sys.BINDIR,Base.julia_exename()))[1],
      evotrees_version=string(Base.pkgversion(SolarSINDy.EvoTrees)),evotrees_source_sha256=evo,
      threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),omp_threads=tryparse(Int,get(ENV,"OMP_NUM_THREADS","")))
    _need(runtime.threads==runtime.blas_threads==runtime.omp_threads==1,"matched342 controlled native runtime requires single threads")
    (;runtime,runtime_sha256=T._id((;schema="matched342_native_runtime_v1",runtime)))
end
struct PreparedCell342{Mode}
    pair::SourcePair342{Mode}
    delay::Int
    horizon::Int
    cohort::NamedTuple
    union::Vector{Int}
    design::Matrix{Float64}
    standardized::Matrix{Float64}
    targets::Vector{Float64}
    encoder::M.Encoder342
    facts::NamedTuple
    seal::String
    function PreparedCell342{Mode}(pair,delay,horizon,cohort,union,design,standardized,targets,encoder,facts,seal,::_Token342) where Mode
        new{Mode}(pair,delay,horizon,cohort,union,design,standardized,targets,encoder,facts,seal)
    end
end
function _fit_input_sha(mode,context,delay,horizon,cohort,X,targets,mask,encoder)
    header=(;schema="matched342_initial_fit_input_header_v1",evidence_mode=String(mode),context_id=context,
      delay_hours=delay,horizon,model_training_mask_sha256=mask,preprocessing_sha256=M.preprocessing_sha256(encoder),
      training_design_sha256=encoder.training_design_sha256,training_rows_count=length(cohort.train),feature_count=342)
    _need(length(targets)==length(cohort.train),"matched342 fit response shape differs")
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("matched342-initial-fit-input-v1\n"));SHA.update!(ctx,T._bytes(header));io=IOBuffer()
    for (k,i) in enumerate(cohort.train)
        write(io,htol(Int64(i)))
        for j in 1:342
            x=X[i,j];_need(isfinite(x),"matched342 nonfinite normalized training value")
            write(io,htol(reinterpret(UInt64,x)))
        end
        y=targets[k];_need(isfinite(y),"matched342 nonfinite increment response");write(io,htol(reinterpret(UInt64,y)))
        position(io)>=65536&&SHA.update!(ctx,take!(io))
    end
    position(io)>0&&SHA.update!(ctx,take!(io));bytes2hex(SHA.digest!(ctx))
end
function _cell_seal(cell)
    T._id((;schema="matched342_prepared_cell_seal_v1",pair_id=cell.pair.pair_id,delay_hours=cell.delay,horizon=cell.horizon,
      facts=cell.facts,design_sha256=_binary_hash("matched342-raw-design-cache-v1\n",cell.design),
      normalized_sha256=_binary_hash("matched342-normalized-design-cache-v1\n",cell.standardized),
      response_sha256=_binary_hash("matched342-response-cache-v1\n",cell.targets),
      issue_times_sha256=_binary_hash("matched342-issue-times-cache-v1\n",(Dates.value(t) for t in cell.cohort.issues)),
      training_rows_sha256=_binary_hash("matched342-training-rows-cache-v1\n",cell.cohort.train),
      validation_rows_sha256=_binary_hash("matched342-validation-rows-cache-v1\n",cell.cohort.validation),
      union_rows_sha256=_binary_hash("matched342-union-rows-cache-v1\n",cell.union)))
end
function prepare_cell(pair::SourcePair342{Mode},delay,horizon) where Mode
    recheck_pair(pair);d=T._int(delay,0,1);h=T._int(horizon,4,6);_need(h in (4,6),"matched342 h4/h6 only")
    input=pair.inputs[d+1];m=pair.admissions[d+1].metadata
    options=(;training_boundary=m.training_boundary,validation_start=m.validation_start,validation_end=m.validation_end)
    cohort=M.cohorts(input,h;options...)
    other=M.cohorts(input,h==4 ? 6 : 4;options...)
    union=sort!(unique!(vcat(cohort.validation,other.validation)))
    geometry=BI._reference_geometry(pair.admissions[d+1].view,m.training_boundary,m.validation_start,m.validation_end,d)
    k=h==4 ? 1 : 2
    _need(cohort.train==geometry.training[k]&&cohort.validation==geometry.leads[k]&&union==geometry.indices,
        "matched342 per-lead and B reference clock geometry disagree")
    _need(!isempty(cohort.train)&&!isempty(cohort.validation),"matched342 nonempty full per-lead cohorts required")
    design=M.build_features(input,cohort.issues).matrix
    encoder=M.fit_encoder(design,cohort.train);standardized=M.transform(encoder,design)
    targets=Float64[input.values[15,i+h]-input.values[15,i-d] for i in cohort.train]
    mask=M._training_mask_sha(input,cohort,m.training_boundary,m.validation_start,m.validation_end)
    context=pair.bindings[d+1].context_id
    fit_input=_fit_input_sha(Mode,context,d,h,cohort,standardized,targets,mask,encoder)
    lastfit=T.Adapter.forecast_clock(cohort.issues[last(cohort.train)],h,d)
    lastvalidation=T.Adapter.forecast_clock(cohort.issues[last(cohort.validation)],h,d)
    facts=(;context_id=context,model_training_mask_sha256=mask,
      reference_training_mask_sha256=BI._reference_mask(pair.admissions[d+1].view,cohort.train,h,:training,m.training_boundary,m.validation_start,m.validation_end,d),
      reference_validation_cohort_sha256=BI._reference_mask(pair.admissions[d+1].view,cohort.validation,h,:validation,m.training_boundary,m.validation_start,m.validation_end,d),
      fit_input_sha256=fit_input,preprocessing_sha256=M.preprocessing_sha256(encoder),training_design_sha256=encoder.training_design_sha256,
      max_fit_target_complete=lastfit.target_complete,max_fit_label_receipt=lastfit.target_receipt,
      max_validation_target_complete=lastvalidation.target_complete,max_validation_label_receipt=lastvalidation.target_receipt)
    provisional=PreparedCell342{Mode}(pair,d,h,cohort,union,design,standardized,targets,encoder,facts,"",_Token342())
    cell=PreparedCell342{Mode}(pair,d,h,cohort,union,design,standardized,targets,encoder,facts,_cell_seal(provisional),_Token342())
    recheck_cell(cell);cell
end
function recheck_cell(cell::PreparedCell342{Mode}) where Mode
    recheck_pair(cell.pair);M._check_encoder(cell.encoder)
    _need(cell.horizon in (4,6)&&cell.delay in (0,1)&&cell.pair.bindings[cell.delay+1].context_id==cell.facts.context_id,
        "matched342 prepared cell context differs")
    _need(size(cell.design)==size(cell.standardized)==(length(cell.cohort.issues),342)&&
        length(cell.targets)==length(cell.cohort.train)&&all(isfinite,cell.standardized)&&all(isfinite,cell.targets),
        "matched342 prepared buffers differ")
    _need(collect(cell.encoder.training_rows)==cell.cohort.train&&M.preprocessing_sha256(cell.encoder)==cell.facts.preprocessing_sha256&&
        cell.encoder.training_design_sha256==cell.facts.training_design_sha256&&_cell_seal(cell)==cell.seal,
        "matched342 prepared design/encoder/cache seal differs")
    true
end
function cell_metadata(cell::PreparedCell342{Mode}) where Mode
    recheck_cell(cell);pair=cell.pair;input=pair.inputs[cell.delay+1];m=pair.admissions[cell.delay+1].metadata
    selected=Set(cell.cohort.validation)
    (;evidence_mode=Mode,pair_id=pair.pair_id,source_binding=pair.bindings[cell.delay+1],
      delay_hours=cell.delay,horizon=cell.horizon,training_boundary=m.training_boundary,validation_start=m.validation_start,
      validation_end=m.validation_end,issues=copy(cell.cohort.issues),training_rows=copy(cell.cohort.train),
      lead_validation_rows=copy(cell.cohort.validation),validation_union_rows=copy(cell.union),
      union_issues=cell.cohort.issues[cell.union],union_anchors=Float64[input.values[15,i-cell.delay] for i in cell.union],
      union_targets=Union{Nothing,Float64}[i in selected ? input.values[15,i+cell.horizon] : nothing for i in cell.union],
      cell.facts...)
end
struct ExpectedBoundModel342{Mode}
    pair::SourcePair342{Mode}
    context_id::String
    delay_hours::Int
    horizon::Int
    setting_index::Int
    training_mask_sha256::String
    preprocessing_sha256::String
    training_design_sha256::String
    fit_input_sha256::String
    seal::String
    function ExpectedBoundModel342{Mode}(pair,context,delay,horizon,setting,mask,preprocess,design,input,seal,::_Token342) where Mode
        new{Mode}(pair,context,delay,horizon,setting,mask,preprocess,design,input,seal)
    end
end
function _expected_data(expected::ExpectedBoundModel342{Mode}) where Mode
    (;schema="matched342_bound_expected_model_v1",evidence_mode=String(Mode),pair_id=expected.pair.pair_id,
      context_id=expected.context_id,delay_hours=expected.delay_hours,horizon=expected.horizon,setting_index=expected.setting_index,
      training_mask_sha256=expected.training_mask_sha256,preprocessing_sha256=expected.preprocessing_sha256,
      training_design_sha256=expected.training_design_sha256,fit_input_sha256=expected.fit_input_sha256,
      evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,loss_contract_sha256=R.LOSS_CONTRACT_SHA256)
end
function expected_bindings(cell::PreparedCell342{Mode},setting) where Mode
    recheck_cell(cell);s=T._int(setting,1,8);f=cell.facts
    provisional=ExpectedBoundModel342{Mode}(cell.pair,f.context_id,cell.delay,cell.horizon,s,f.model_training_mask_sha256,
        f.preprocessing_sha256,f.training_design_sha256,f.fit_input_sha256,"",_Token342())
    ExpectedBoundModel342{Mode}(cell.pair,f.context_id,cell.delay,cell.horizon,s,f.model_training_mask_sha256,
        f.preprocessing_sha256,f.training_design_sha256,f.fit_input_sha256,T._id(_expected_data(provisional)),_Token342())
end
function _check_expected(expected::ExpectedBoundModel342{Mode}) where Mode
    recheck_pair(expected.pair);M._trusted_evo_sha()
    _need(expected.delay_hours in (0,1)&&expected.horizon in (4,6)&&1<=expected.setting_index<=8&&
        expected.context_id==expected.pair.bindings[expected.delay_hours+1].context_id&&
        all(T._hex,(expected.training_mask_sha256,expected.preprocessing_sha256,expected.training_design_sha256,expected.fit_input_sha256))&&
        T._id(_expected_data(expected))==expected.seal,"matched342 expected bindings differ")
    true
end
function _expected_fields(expected::ExpectedBoundModel342)
    _check_expected(expected);g=M.GRID342[expected.setting_index]
    Dict{String,Any}("schema"=>"matched_gbm_scalar342_model_v1","feature_count"=>342,
      "feature_schema_sha256"=>M.FEATURE_SCHEMA_SHA256,"evotrees_version"=>"0.18.7",
      "evotrees_source_sha256"=>M.EVOTREES_SOURCE_SHA256,"cell_delay"=>expected.delay_hours,"cell_horizon"=>expected.horizon,
      "setting_index"=>expected.setting_index,"max_depth"=>g.max_depth,"nrounds"=>g.nrounds,"nbins"=>g.nbins,"seed"=>22022026,
      "training_mask_sha256"=>expected.training_mask_sha256,"preprocessing_sha256"=>expected.preprocessing_sha256,
      "input_admission_sha256"=>expected.context_id,"feature_contract_sha256"=>M.FEATURE_CONTRACT_SHA256,
      "loss_contract_sha256"=>R.LOSS_CONTRACT_SHA256)
end
function _cell_expected(cell,expected)
    recheck_cell(cell);_check_expected(expected)
    _need(cell.pair.pair_id==expected.pair.pair_id&&cell.delay==expected.delay_hours&&cell.horizon==expected.horizon&&
        cell.facts.context_id==expected.context_id&&cell.facts.model_training_mask_sha256==expected.training_mask_sha256&&
        cell.facts.preprocessing_sha256==expected.preprocessing_sha256&&cell.facts.training_design_sha256==expected.training_design_sha256&&
        cell.facts.fit_input_sha256==expected.fit_input_sha256,"matched342 expected cell pins differ")
    true
end
struct ActualExecutionAuthority342
    permit::NamedTuple
    data::NamedTuple
    sources::Tuple
    controls::Tuple
    resource_facts::NamedTuple
    prelaunch::NamedTuple
    began_ns::UInt64
    seal::String
    function ActualExecutionAuthority342(permit,data,sources,controls,resource_facts,prelaunch,began,seal,::_Token342)
        new(permit,data,sources,controls,resource_facts,prelaunch,began,seal)
    end
end
struct BoundFittedCell342{Mode}
    model::M.MatchedScalarTreeReplay342.Model
    encoder::M.Encoder342
    expected::ExpectedBoundModel342{Mode}
    recipe::NamedTuple
    component_sha256::String
    authority::Union{Nothing,ActualExecutionAuthority342}
    seal::String
    function BoundFittedCell342{Mode}(model,encoder,expected,recipe,component,authority,seal,::_Token342) where Mode
        new{Mode}(model,encoder,expected,recipe,component,authority,seal)
    end
end
_component_sha(sources)=T._id((;schema="matched342_bound_component_sources_v1",sources=_source_rows(_sources(sources))))
function _capsule_seal(capsule::BoundFittedCell342{Mode}) where Mode
    object=M._model_object_data(capsule.model,_expected_fields(capsule.expected),M.GRID342[capsule.expected.setting_index])
    T._id((;schema="matched342_bound_fit_capsule_v1",evidence_mode=String(Mode),model_id=object["model_id"],
      preprocessing_sha256=M.preprocessing_sha256(capsule.encoder),expected_sha256=capsule.expected.seal,
      component_sha256=capsule.component_sha256,recipe=capsule.recipe,
      execution_authority_sha256=capsule.authority===nothing ? nothing : capsule.authority.permit.sha256))
end
function _check_capsule(capsule::BoundFittedCell342{Mode},expected::ExpectedBoundModel342{Mode}) where Mode
    _check_expected(expected)
    _need(capsule.expected.seal==expected.seal&&capsule.recipe==M._recipe(expected.setting_index)&&
        capsule.component_sha256==_component_sha(expected.pair.sources)&&
        M.preprocessing_sha256(capsule.encoder)==expected.preprocessing_sha256&&_capsule_seal(capsule)==capsule.seal,
        "matched342 bound capsule source/encoder/recipe/seal differs")
    if Mode==ACTUAL
        _need(capsule.authority isa ActualExecutionAuthority342,"actual capsule lacks actual supplied authority")
        recheck_execution_authority(capsule.authority,expected.pair)
    else
        _need(capsule.authority===nothing,"generated capsule rejects actual authority")
    end
    true
end
function _fit_prepared(cell::PreparedCell342{Mode},expected::ExpectedBoundModel342{Mode},authority) where Mode
    if Mode==ACTUAL
        _need(authority isa ActualExecutionAuthority342,"actual matched342 fit requires supplied qualified authority before native work")
        recheck_execution_authority(authority,cell.pair)
    elseif Mode==GENERATED
        _need(authority===nothing,"generated matched342 fit rejects actual authority before native work")
    else
        throw(ArgumentError("matched342 fitting evidence mode differs"))
    end
    _cell_expected(cell,expected);identity=runtime_identity();recipe=M._recipe(expected.setting_index)
    component=_component_sha(cell.pair.sources)
    authority===nothing||_reserve_fit!(authority,cell,expected)
    started=time_ns()
    native=SolarSINDy.gbm_fit(cell.standardized[cell.cohort.train,:],cell.targets;recipe...,feature_names=collect(M.FEATURE_NAMES))
    model=M._snapshot_native(native,recipe)
    if authority!==nothing
        elapsed=time_ns()-started
        _need(elapsed<UInt64(authority.data.limits.fit_wall_seconds)*UInt64(10^9),"matched342 actual fit deadline exceeded")
    end
    _cell_expected(cell,expected);_need(runtime_identity()==identity&&_component_sha(cell.pair.sources)==component,
        "matched342 fit source/runtime changed")
    provisional=BoundFittedCell342{Mode}(model,cell.encoder,expected,recipe,component,authority,"",_Token342())
    capsule=BoundFittedCell342{Mode}(model,cell.encoder,expected,recipe,component,authority,_capsule_seal(provisional),_Token342())
    _check_capsule(capsule,expected)
    (;capsule,native,training_rows=length(cell.cohort.train),validation_rows=length(cell.cohort.validation))
end
function fit_generated_cell(cell::PreparedCell342{GENERATED},expected::ExpectedBoundModel342{GENERATED})
    _fit_prepared(cell,expected,nothing)
end
function fit_actual_cell(cell::PreparedCell342{ACTUAL},expected::ExpectedBoundModel342{ACTUAL};execution_authority::ActualExecutionAuthority342)
    _fit_prepared(cell,expected,execution_authority)
end
function predict_cell(cell::PreparedCell342{Mode},capsule::BoundFittedCell342{Mode},expected::ExpectedBoundModel342{Mode};native) where Mode
    _cell_expected(cell,expected);_check_capsule(capsule,expected)
    X=cell.standardized[cell.union,:]
    observed=SolarSINDy.gbm_predict(native,X);replayed=M.MatchedScalarTreeReplay342.predict_deltas(capsule.model,X)
    _need(length(observed)==length(replayed)==length(cell.union)&&reinterpret(UInt64,observed)==reinterpret(UInt64,replayed),
        "matched342 native versus immutable scalar union predictions differ")
    _cell_expected(cell,expected);_check_capsule(capsule,expected);replayed
end
function save_encoder(path,cell::PreparedCell342)
    recheck_cell(cell);result=M.save_encoder(path,cell.encoder);recheck_cell(cell);result
end
function save_model(path,capsule::BoundFittedCell342{Mode};expected_bindings::ExpectedBoundModel342{Mode}) where Mode
    _check_capsule(capsule,expected_bindings)
    result=M._write_json(path,M._model_object_data(capsule.model,_expected_fields(expected_bindings),M.GRID342[expected_bindings.setting_index]))
    _check_capsule(capsule,expected_bindings);result
end
function load_model(path;expected_sha256,expected_bindings::ExpectedBoundModel342)
    _check_expected(expected_bindings);object=M._read_json(path,expected_sha256)
    fields=_expected_fields(expected_bindings);grid=M.GRID342[expected_bindings.setting_index]
    model=M._decode_model_data(object,fields,grid)
    _check_expected(expected_bindings)
    _need(M._model_object_data(model,_expected_fields(expected_bindings),grid)==object,"matched342 loaded model context changed")
    model
end
function replay_cell(cell::PreparedCell342{Mode},model::M.MatchedScalarTreeReplay342.Model,encoder::M.Encoder342,
        expected::ExpectedBoundModel342{Mode};expected_model_sha256,expected_encoder_sha256) where Mode
    _cell_expected(cell,expected);M._check_encoder(encoder)
    _need(T._hex(expected_model_sha256)&&T._hex(expected_encoder_sha256)&&
        expected_encoder_sha256==expected.preprocessing_sha256==M.preprocessing_sha256(encoder)&&
        encoder.training_design_sha256==expected.training_design_sha256&&collect(encoder.training_rows)==cell.cohort.train,
        "matched342 reloaded encoder context differs")
    object=M._model_object_data(model,_expected_fields(expected),M.GRID342[expected.setting_index])
    _need(bytes2hex(sha256(M._canonical(object)))==expected_model_sha256,"matched342 outside saved model pin differs")
    X=M.transform(encoder,cell.design[cell.union,:]);result=M.MatchedScalarTreeReplay342.predict_deltas(model,X)
    _cell_expected(cell,expected);result
end

# Supplied source/qualification/resource/actual-authority readers.
function _freeze(x)
    if x isa NamedTuple||x isa JSON3.Object
        names=Tuple(Symbol(k) for k in keys(x));NamedTuple{names}(Tuple(_freeze(getproperty(x,k)) for k in names))
    elseif x isa AbstractDict
        names=Tuple(Symbol(k) for k in keys(x))
        NamedTuple{names}(Tuple(_freeze(haskey(x,k) ? x[k] : x[String(k)]) for k in names))
    elseif x isa AbstractVector||x isa Tuple
        Tuple(_freeze(v) for v in x)
    else
        x
    end
end
function _absolute(path)
    _need(path isa AbstractString&&isabspath(path)&&!any(x->x in (".",".."),split(replace(path,'\\'=>'/'),'/')),
        "matched342 absolute nonalias path required")
    P._ordinary(String(path))
end
function _ref(ref)
    _fields(ref,(:path,:sha256));_need(T._hex(ref.sha256),"matched342 external file pin required")
    (;path=_absolute(ref.path),sha256=String(ref.sha256))
end
function _pinned(ref,controls;limit=T.MAX_META,allow_empty=false)
    checked=_ref(ref);minimum=allow_empty ? 0 : 1
    _need(isfile(checked.path)&&minimum<=filesize(checked.path)<=limit,"matched342 bounded external file required")
    _need(P._file_sha(checked.path)[1]==checked.sha256,"matched342 external file pin differs")
    if controls!==nothing
        _need(!haskey(controls,checked.path)||controls[checked.path]==checked.sha256,"matched342 inconsistent control pin")
        controls[checked.path]=checked.sha256
    end
    checked
end
function _json_ref(ref,controls;fields=nothing)
    checked=_pinned(ref,controls);data,body=P._json(checked.path;fields)
    _need(T._sha(body)==checked.sha256,"matched342 canonical external metadata changed");data
end
function _check_id(data,key)
    T._need(T._hex(getproperty(data,key)),"matched342 canonical identity domain differs")
    names=Tuple(Symbol(k) for k in keys(data) if Symbol(k)!=key)
    _need(T._id(NamedTuple{names}(Tuple(getproperty(data,k) for k in names)))==getproperty(data,key),
        "matched342 canonical identity differs")
end
function _check_controls(controls)
    buffer=Vector{UInt8}(undef,65536)
    for (path,pin) in controls
        P._ordinary(path);declared=filesize(path);_need(declared<=T.MAX_FILE,"matched342 control byte bound")
        context=SHA.SHA2_256_CTX();total=0
        open(path,"r") do io
            while !eof(io)
                n=readbytes!(io,buffer,65536);_need(n>0,"matched342 control read made no progress")
                total+=n;_need(total<=declared,"matched342 control grew during read");SHA.update!(context,buffer,n)
            end
        end
        _need(total==declared==filesize(path)&&bytes2hex(SHA.digest!(context))==pin,
            "matched342 authority/control evidence changed: "*path)
    end
    true
end
function _git_environment()
    allowed=Dict("GIT_PAGER"=>"cat","GIT_TERMINAL_PROMPT"=>"0")
    for (name,value) in ENV
        startswith(name,"GIT_")&&_need(get(allowed,name,nothing)==value,"matched342 inherited Git override refused")
    end
    result=Dict(k=>v for (k,v) in ENV if !startswith(k,"GIT_"));result["GIT_TERMINAL_PROMPT"]="0";result
end
function _drain(pipe,limit)
    data=UInt8[];failure=nothing
    try
        while true
            block=read(pipe,min(65536,limit+1-length(data)))
            isempty(block)&&break;append!(data,block)
            if length(data)>limit;failure="live Git pipe byte bound exceeded";break;end
        end
    catch error
        error isa InterruptException&&rethrow();failure=sprint(showerror,error)
    finally
        close(pipe)
    end
    (;data,failure)
end
function _git(arguments...;limit=4*1024^2,stdin_path=nothing,deadline_ns=nothing)
    out=Pipe();err=Pipe();input=stdin_path===nothing ? devnull : open(_absolute(stdin_path),"r")
    process=nothing;tasks=Task[];started=time_ns();budget=UInt64(60*10^9)
    try
        command=setenv(Cmd(String["git","-C",REPO,arguments...]),_git_environment())
        process=run(pipeline(command;stdin=input,stdout=out,stderr=err);wait=false)
        close(out.in);close(err.in)
        push!(tasks,@async _drain(out,limit));push!(tasks,@async _drain(err,65536))
        while true
            elapsed=time_ns()-started
            expired=elapsed>=budget||(deadline_ns!==nothing&&time_ns()>=deadline_ns)
            broken=any(task->istaskdone(task)&&fetch(task).failure!==nothing,tasks)
            if expired||broken
                process_running(process)&&kill(process)
                _need(false,expired ? "matched342 bounded Git deadline exhausted" : "matched342 Git capture failed")
            end
            !process_running(process)&&all(istaskdone,tasks)&&break
            sleep(.005)
        end
        wait(process);stdout=fetch(tasks[1]);stderr=fetch(tasks[2])
        _need(success(process)&&stdout.failure===nothing&&stderr.failure===nothing,"matched342 read-only Git failed: "*String(stderr.data))
        stdout.data
    finally
        stdin_path===nothing||close(input)
        if process!==nothing&&process_running(process);kill(process);end
        for pipe in (out,err)
            try close(pipe) catch end
        end
        if process!==nothing
            try wait(process) catch end
        end
    end
end
function _index_entries(raw)
    entries=Dict{String,Tuple{String,String}}()
    for row in split(String(raw),'\0';keepempty=false)
        fields=split(row,'\t';limit=2);_need(length(fields)==2,"matched342 Git index row differs")
        head=split(fields[1]);_need(length(head)==3&&head[3]=="0"&&head[1] in ("100644","100755")&&
            occursin(r"\A[0-9a-f]{40}\z",head[2])&&!haskey(entries,fields[2]),"matched342 regular stage0 index required")
        entries[fields[2]]=(head[1],head[2])
    end
    entries
end
const WHOLE_PROOF_FIELDS=(:complete,:head,:tree,:source_files,:source_sha256,:commit_object_ids,:all_clean_and_checkout_filters_match)
function _proof_ref(ref,controls)
    checked=_pinned(ref,controls);body=P._bound(checked.path)
    # The original whole-source proof is externally pinned native JSON, not a
    # canonical B artifact. Preserve its bytes and check its integer token and
    # duplicate-key grammar before JSON3 can normalize integral float literals.
    count=P._qualification_integer(body,("source_files",));data=P._bounded_json_parse(body)
    _fields(data,WHOLE_PROOF_FIELDS);_need(data.source_files==count,"matched342 whole-proof integer token differs");data
end
"Recheck whole-source authority only; performs no numerical preparation or fitting."
function recheck_source_proof(ref;expected_head,expected_sources)
    sources=_sources(expected_sources);proof=_proof_ref(ref,nothing)
    _need(proof.complete===true&&proof.all_clean_and_checkout_filters_match===true&&proof.head==expected_head&&
        expected_head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",expected_head)&&
        proof.tree isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",proof.tree),"matched342 complete current source proof required")
    count=T._int(proof.source_files,1,8192)
    _need((proof.source_sha256 isa JSON3.Object||proof.source_sha256 isa AbstractDict)&&
        (proof.commit_object_ids isa JSON3.Object||proof.commit_object_ids isa AbstractDict),"matched342 proof maps required")
    hashes=Dict(String(k)=>begin _need(T._hex(v),"matched342 source SHA grammar differs");String(v) end for (k,v) in pairs(proof.source_sha256))
    objects=Dict(String(k)=>begin _need(v isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",v),"matched342 source OID grammar differs");String(v) end for (k,v) in pairs(proof.commit_object_ids))
    _need(length(hashes)==length(objects)==count&&Set(keys(hashes))==Set(keys(objects)),"matched342 whole-source inventories differ")
    for (name,pin) in sources;_need(get(hashes,name,nothing)==pin,"matched342 component member absent from whole proof");end
    deadline=time_ns()+UInt64(900*10^9)
    git(args...;kwargs...)=_git(args...;deadline_ns=deadline,kwargs...)
    _need(strip(String(git("rev-parse","HEAD")))==expected_head&&strip(String(git("rev-parse","HEAD^{tree}")))==proof.tree&&
        isempty(git("status","--porcelain=v1","--untracked-files=all")),"matched342 current source checkout is not clean/exacthead")
    entries=_index_entries(git("ls-files","--stage","-z"))
    _need(Set(keys(entries))==Set(keys(hashes)),"matched342 tracked whole-source member set differs")
    tree=Dict{String,Tuple{String,String}}()
    for row in split(String(git("ls-tree","-r","-z","HEAD")),'\0';keepempty=false)
        fields=split(row,'\t';limit=2);header=split(fields[1])
        _need(length(fields)==2&&length(header)==3&&header[2]=="blob"&&header[1] in ("100644","100755")&&
            !haskey(tree,fields[2]),"matched342 committed regular source tree differs")
        tree[fields[2]]=(header[1],header[3])
    end
    _need(tree==entries,"matched342 index/committed tree differs")
    for name in sort!(collect(keys(hashes)))
        pin=hashes[name];oid=objects[name];path=P._member(REPO,name)
        _need(T._hex(pin)&&occursin(r"\A[0-9a-f]{40}\z",oid)&&entries[name][2]==oid&&
            P._file_sha(path)[1]==pin,"matched342 source member/OID differs: "*name)
        cleaned=strip(String(git("hash-object","--path="*name,"--stdin";limit=128,stdin_path=path)))
        _need(cleaned==oid,"matched342 clean-filter source object differs")
        size_text=strip(String(git("cat-file","-s",oid;limit=64)))
        _need(occursin(r"\A(?:0|[1-9][0-9]*)\z",size_text),"matched342 raw Git blob size grammar differs")
        rawsize=tryparse(Int,size_text);_need(rawsize!==nothing&&0<=rawsize<=T.MAX_FILE,"matched342 raw Git blob byte bound")
        raw=git("cat-file","blob",oid;limit=max(1,rawsize))
        _need(length(raw)==rawsize&&bytes2hex(sha1(vcat(codeunits("blob $rawsize\0"),raw)))==oid,"matched342 raw Git object integrity differs")
        size=filesize(path);filtered=git("cat-file","--filters","--path="*name,oid;limit=max(1,size))
        _need(length(filtered)==size&&T._sha(filtered)==pin&&P._file_sha(path)[1]==pin,"matched342 checkout-filter source bytes differ")
    end
    _need(strip(String(git("rev-parse","HEAD")))==expected_head&&isempty(git("status","--porcelain=v1","--untracked-files=all")),
        "matched342 source changed during whole-proof validation")
    _proof_ref(ref,nothing);_sources(sources);proof
end

const PERMIT_FIELDS=(:schema,:complete,:evidence_mode,:operation,:qualified_head,:implementation_sources,:contracts,:sources,
    :source_proof,:root_qualification,:component_qualification,:resource_qualification,:runtime,:execution,:scope,:limits,:permit_id)
const CONTRACT_FIELDS=(:protocol_sha256,:feature_contract_sha256,:feature_schema_sha256,:loss_contract_sha256,:evotrees_source_sha256)
const INTENT_FIELDS=(:schema,:evidence_mode,:operation,:qualified_head,:implementation_sources,:contracts,:sources,:runtime,:execution_directory,:scope,:limits)
const SCOPE_FIELDS=(:training_boundary_ms,:validation_start_ms,:validation_end_ms,:embargo_hours,:cell_order,:setting_order,
    :fit_count,:encoder_count,:final_fit_count,:calibration_fit_count,:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion)
const LIMIT_FIELDS=(:source_rows,:model_member_bytes,:total_model_member_bytes,:required_available_bytes,:required_disk_bytes,:stage_wall_seconds,:fit_wall_seconds)
function _check_runtime(data)
    _fields(data,RUNTIME_FIELDS);actual=runtime_identity()
    for key in RUNTIME_FIELDS
        _need(getproperty(data,key)==getproperty(actual.runtime,key),"matched342 supplied/observed runtime differs: $key")
    end
    for key in (:threads,:blas_threads,:omp_threads);T._int(getproperty(data,key),1,1);end
    _need(data.julia_version in ("1.12.6","1.13.1"),"matched342 qualified runtime version differs")
    actual
end
function _check_contracts(data)
    _fields(data,CONTRACT_FIELDS)
    _need(T._bytes(data)==T._bytes(_contracts()),"matched342 supplied fixed contracts differ");true
end
function _check_scope(scope)
    _fields(scope,SCOPE_FIELDS)
    for (key,time) in ((:training_boundary_ms,DateTime(2020)),(:validation_start_ms,DateTime(2020)),(:validation_end_ms,DateTime(2023)))
        _need(T._clock(getproperty(scope,key))==time,"matched342 actual initial split differs")
    end
    for (key,value) in ((:embargo_hours,168),(:fit_count,32),(:encoder_count,4),(:final_fit_count,0),(:calibration_fit_count,0))
        T._int(getproperty(scope,key),value,value)
    end
    _need(Tuple(Tuple(T._int(x,0,6) for x in row) for row in scope.cell_order)==CELLS&&
        Tuple(T._int(x,1,8) for x in scope.setting_order)==Tuple(1:8),"matched342 exact cell/grid scope differs")
    for key in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion)
        _need(getproperty(scope,key)===false,"matched342 actual permission exceeds initial32 scope")
    end
    true
end
function _check_limits(limits)
    _fields(limits,LIMIT_FIELDS)
    for (key,value) in ((:source_rows,500000),(:model_member_bytes,4*1024^2),(:total_model_member_bytes,128*1024^2))
        T._int(getproperty(limits,key),value,value)
    end
    T._int(limits.required_available_bytes,RSS_LIMIT,typemax(Int64));T._int(limits.required_disk_bytes,1,typemax(Int64))
    stage=T._int(limits.stage_wall_seconds,1,div(typemax(Int64),10^9));fit=T._int(limits.fit_wall_seconds,1,stage)
    (;stage,fit)
end
function _binding_grammar(rows)
    _need((rows isa AbstractVector||rows isa Tuple)&&length(rows)==2,"matched342 actual authority needs two source bindings")
    for (k,row) in enumerate(rows)
        _fields(row,(SOURCE_FACT_FIELDS...,:context_id));T._int(row.delay_hours,k-1,k-1)
        _need(all(T._hex(getproperty(row,key)) for key in (SOURCE_FACT_FIELDS[2:end]...,:context_id)),"matched342 source binding pin grammar differs")
    end
    true
end
function _component_qualification(ref,head,sources,controls;root_sha,proof_sha,mode="software_qualified")
    checked=_pinned(ref,controls);body=P._bound(checked.path)
    q=P._software_qualification(body;schema="b_reference_component_software_qualification_v1",evidence_mode=mode,head,sources)
    _need(q.root_qualification_sha256==root_sha&&q.source_proof_sha256==proof_sha,"matched342 component qualification ancestry differs")
    q
end
function _root_qualification(ref,head,proof_sha,controls)
    checked=_pinned(ref,controls);data=P._bounded_json_parse(P._bound(checked.path));e=data.candidate_pipeline_full18
    _need(data.complete===true&&data.head==head&&data.source_proof.sha256==proof_sha&&
        data.strict_docs_both_complete===true&&data.unfiltered_package_and_full_experiments_executed===true&&
        data.software_qualification_only===true&&e.complete===true&&e.head==head&&
        e.both_actual_strict_docs_passed===true&&e.unfiltered_Pkg_and_complete_examples_passed===true&&
        e.eight_full_resources_verified===true&&e.agreement48_verified===true,"matched342 current full18 source qualification absent")
    expected=Set(vcat(["Julia $v / $os-latest" for v in ("1.12.6","1.13.1") for os in ("ubuntu","windows","macos")],
      ["Documentation build","Julia 1.12 deployment images","Canonical development harness","Full-scale cross-runtime agreement"],
      ["Full-scale $v / $role / $layout" for v in ("1.12.6","1.13.1") for role in ("primary","independent") for layout in ("ground-only","mixed")]))
    _need(length(e.jobs)==18&&Set(String(j.name) for j in e.jobs)==expected&&allunique([j.job_id for j in e.jobs]),
        "matched342 exact original18 native job inventory differs")
    for job in e.jobs
        T._int(job.native_exit,0,0);T._int(job.job_id,1,typemax(Int64))
        _need(job.timed_out===false&&job.head==head&&job.conclusion=="success","matched342 original native job not positive")
        _pinned((;path=String(job.raw_path),sha256=String(job.raw_sha256)),controls;limit=T.MAX_FILE)
    end
    _need(T._hex(e.harness_raw_sha256),"matched342 harness raw pin missing");data
end
const OWNER_RECEIPT_FIELDS=(:native,:start_ns,:end_ns,:deadline_ns,:worker_raw_sha256,:individual_descendant_OS_reaping_observed)
const WIN_NATIVE_FIELDS=(:pid,:native_exit,:timed_out,:leader_reaped,:ownership,:descendant_cleanup,:ownership_boundary_closed,
    :all_descendants_os_reaped,:wall_seconds,:wrapper_observed_wall_seconds,:descendant_evidence)
const POSIX_NATIVE_FIELDS=(:pid,:native_exit,:timed_out,:leader_reaped,:ownership,:descendant_cleanup,:ownership_boundary_closed,
    :all_descendants_os_reaped,:wall_seconds,:observation_method,:descendant_evidence)
function _native_receipt(ref,rawref,limits,controls)
    checked=_pinned(ref,controls);raw=_pinned(rawref,controls;limit=T.MAX_FILE,allow_empty=true);body=P._bound(checked.path)
    exit=P._qualification_integer(body,("native","native_exit"));pid=P._qualification_integer(body,("native","pid"))
    start=P._qualification_integer(body,("start_ns",));stop=P._qualification_integer(body,("end_ns",));deadline=P._qualification_integer(body,("deadline_ns",))
    data=P._bounded_json_parse(body);_fields(data,OWNER_RECEIPT_FIELDS);native=data.native
    if native.ownership=="CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread"
        _fields(native,WIN_NATIVE_FIELDS)
    else
        _need(native.ownership=="start_new_session=True","matched342 fixed native ownership method differs")
        _fields(native,POSIX_NATIVE_FIELDS)
    end
    stage=T._int(limits.stage_wall_seconds,1,div(typemax(Int64),10^9));wall=native.wall_seconds
    _need(exit==0&&pid>0&&native.native_exit==exit&&native.pid==pid&&native.timed_out===false&&native.leader_reaped===true&&
        native.ownership_boundary_closed===true&&native.all_descendants_os_reaped===false&&
        data.individual_descendant_OS_reaping_observed===false&&wall isa Real&&!(wall isa Bool)&&isfinite(wall)&&0<=wall<stage&&
        start<stop<deadline&&deadline-start==stage*10^9&&(stop-start)/1e9<stage&&data.worker_raw_sha256==raw.sha256,
        "matched342 genuine bounded native resource evidence differs")
    if hasproperty(native,:wrapper_observed_wall_seconds)
        observed=native.wrapper_observed_wall_seconds
        _need(observed isa Real&&!(observed isa Bool)&&isfinite(observed)&&0<=observed<stage,"matched342 owner observed wall differs")
    end
    (;data,native,pid,raw)
end
function _source_member(member,pin,proof,controls)
    _need(member isa AbstractString&&T._hex(pin),"matched342 source-controlled member pin required")
    path=P._member(REPO,member)
    _need(get(Dict(String(k)=>String(v) for (k,v) in pairs(proof.source_sha256)),String(member),nothing)==pin,
        "matched342 qualification generator/spec/profiler absent from whole source proof")
    _pinned((;path,sha256=String(pin)),controls;limit=T.MAX_FILE);path
end
function _source_ref(ref,proof,controls)
    _fields(ref,(:member,:path,:sha256));path=_source_member(ref.member,ref.sha256,proof,controls)
    _need(_absolute(ref.path)==path,"matched342 source member absolute path differs")
    (;path,sha256=String(ref.sha256))
end
const PRELAUNCH_FIELDS=(:schema,:complete,:evidence_mode,:head,:source_proof_sha256,:fixture_spec_sha256,:runtime_sha256,
    :owner_pid,:julia_pid,:execution_authority_sha256,:available_physical_bytes,:memory_method,:free_disk_bytes,
    :required_available_bytes,:required_disk_bytes,:native_owner_sha256,:available_observed_before_julia_start,:admission_id)
function _prelaunch(ref,head,proof_sha,spec_sha,runtime_sha,limits,controls;mode,permit_sha=nothing,pid=nothing)
    data=_json_ref(ref,controls;fields=PRELAUNCH_FIELDS);_check_id(data,:admission_id)
    _need(data.schema=="matched342_native_prelaunch_admission_v1"&&data.complete===true&&data.evidence_mode==String(mode)&&
        data.head==head&&data.source_proof_sha256==proof_sha&&data.fixture_spec_sha256==spec_sha&&data.runtime_sha256==runtime_sha&&
        data.execution_authority_sha256==permit_sha&&data.native_owner_sha256==OWNER_SHA&&
        data.available_observed_before_julia_start===true,"matched342 supplied prelaunch witness differs")
    owner=T._int(data.owner_pid,1,typemax(Int64));julia=T._int(data.julia_pid,1,typemax(Int64))
    _need(owner!=julia&&(pid===nothing||pid==julia),"matched342 prelaunch Julia/owner PID differs")
    required=T._int(data.required_available_bytes,RSS_LIMIT,typemax(Int64));disk=T._int(data.required_disk_bytes,1,typemax(Int64))
    _need(required==limits.required_available_bytes&&disk==limits.required_disk_bytes&&
        T._int(data.available_physical_bytes,required,typemax(Int64))>=required&&
        T._int(data.free_disk_bytes,disk,typemax(Int64))>=disk&&data.memory_method isa AbstractString&&!isempty(data.memory_method),
        "matched342 actual pre-LAUNCH admission was not observed")
    _freeze(data)
end
const PROCESS_RESOURCE_FIELDS=(:schema,:complete,:evidence_mode,:head,:source_proof_sha256,:fixture_spec_sha256,:fixture_sha256,
    :runtime_sha256,:julia_pid,:julia_binary_sha256,:profiler,:prelaunch_admission,:native,:raw,:measurement,:budgets,:record_id)
const MEASUREMENT_FIELDS=(:schema,:complete,:evidence_mode,:head,:source_proof_sha256,:fixture_spec_sha256,:fixture_sha256,
    :runtime_sha256,:julia_pid,:scope,:observed,:peak_rss_bytes,:peak_rss_method,:julia_gc_allocated_bytes,:allocation_method,
    :elapsed_ns,:elapsed_method,:descriptors,:measurement_id)
function _descriptor_observation()
    if Sys.iswindows()
        value=Ref{UInt32}(0)
        process=ccall((:GetCurrentProcess,"kernel32"),stdcall,Ptr{Cvoid},())
        _need(ccall((:GetProcessHandleCount,"kernel32"),stdcall,Int32,(Ptr{Cvoid},Ref{UInt32}),process,value)!=0,
            "matched342 Julia process handle count unavailable")
        (;kind="windows_handles",method="Windows GetProcessHandleCount on Julia current process",count=Int(value[]))
    elseif Sys.islinux()
        (;kind="posix_file_descriptors",method="Linux /proc/self/fd count on Julia process",count=length(readdir("/proc/self/fd")))
    elseif Sys.isapple()
        (;kind="posix_file_descriptors",method="macOS /dev/fd count on Julia process",count=length(readdir("/dev/fd")))
    else
        throw(ArgumentError("matched342 observed process descriptors unsupported on this platform"))
    end
end
function _process_resources(ref,q,readback,proof,controls)
    data=_json_ref(ref,controls;fields=PROCESS_RESOURCE_FIELDS);_check_id(data,:record_id)
    _need(data.schema=="matched342_measured_julia_process_resources_v1"&&data.complete===true&&data.evidence_mode=="synthetic_only"&&
        data.head==q.head&&data.source_proof_sha256==q.source_proof_sha256&&data.fixture_spec_sha256==q.fixture_spec.sha256&&
        data.fixture_sha256==q.fixture_sha256&&data.runtime_sha256==readback.runtime_sha256&&
        data.julia_binary_sha256==readback.runtime.julia_binary_sha256&&_ref(data.native)==_ref(q.native)&&_ref(data.raw)==_ref(q.raw),
        "matched342 process-resource provenance differs")
    _fields(data.profiler,(:member,:sha256));_need(endswith(data.profiler.member,".jl"),"matched342 profiler must be source-controlled Julia")
    _source_member(data.profiler.member,data.profiler.sha256,proof,controls)
    native=_native_receipt(data.native,data.raw,q.limits,controls);pid=T._int(data.julia_pid,1,typemax(Int64))
    _need(pid==native.pid,"matched342 RSS evidence measured controller instead of actual Julia child")
    _prelaunch(data.prelaunch_admission,q.head,q.source_proof_sha256,q.fixture_spec.sha256,data.runtime_sha256,q.limits,controls;mode=GENERATED,pid)
    measure=_json_ref(data.measurement,controls;fields=MEASUREMENT_FIELDS);_check_id(measure,:measurement_id)
    _need(measure.schema=="matched342_julia_process_measurement_v1"&&measure.complete===true&&measure.observed===true&&
        measure.evidence_mode=="synthetic_only"&&measure.head==q.head&&measure.source_proof_sha256==q.source_proof_sha256&&
        measure.fixture_spec_sha256==q.fixture_spec.sha256&&measure.fixture_sha256==q.fixture_sha256&&
        measure.runtime_sha256==data.runtime_sha256&&T._int(measure.julia_pid,pid,pid)==pid&&
        measure.scope=="initial32_producer_and_independent_saved_state_audit"&&
        measure.peak_rss_method=="Julia Sys.maxrss bytes; OS process-lifetime maximum"&&
        measure.allocation_method=="Julia Base.@timed.bytes; GC_Diff.allocd managed cumulative bytes"&&
        measure.elapsed_method=="Julia time_ns difference around complete timed body","matched342 resource measurement scope/method/PID differs")
    _fields(data.budgets,(:peak_rss_bytes,:julia_gc_allocated_bytes,:descriptor_growth))
    peak=T._int(data.budgets.peak_rss_bytes,1,RSS_LIMIT);allocated=T._int(data.budgets.julia_gc_allocated_bytes,1,typemax(Int64))
    growth=T._int(data.budgets.descriptor_growth,0,typemax(Int64))
    T._int(measure.peak_rss_bytes,1,peak);T._int(measure.julia_gc_allocated_bytes,1,allocated)
    T._int(measure.elapsed_ns,1,q.limits.stage_wall_seconds*10^9-1)
    descriptors=measure.descriptors;_fields(descriptors,(:kind,:method,:before,:after,:supported,:observed))
    observed=_descriptor_observation()
    _need(descriptors.supported===true&&descriptors.observed===true&&descriptors.kind==observed.kind&&descriptors.method==observed.method,
        "matched342 mandatory process descriptor observation unavailable/wrong platform")
    before=T._int(descriptors.before,0,typemax(Int64));after=T._int(descriptors.after,0,typemax(Int64))
    _need(after-before<=growth,"matched342 measured descriptor growth exceeds frozen budget")
    (;peak_rss_budget_bytes=peak,julia_gc_allocation_budget_bytes=allocated,descriptor_growth_budget=growth,
      descriptor_kind=observed.kind,descriptor_method=observed.method,measurement_sha256=data.measurement.sha256)
end
const RESOURCE_FIELDS=(:schema,:complete,:evidence_mode,:head,:implementation_sources,:source_proof_sha256,:fixture_sha256,
    :fixture_spec,:source_rows,:feature_count,:cell_order,:cells,:native_fit_count,:encoder_count,:limits,:native,:raw,:readback,:process_resources)
const SPEC_FIELDS=(:schema,:evidence_mode,:source_rows,:prefix_start_ms,:training_boundary_ms,:validation_start_ms,:validation_end_ms,
    :receipt_semantics,:all_17_channels_finite,:required_drivers_valid,:full_eligible_cohorts,:variable_training_increments,
    :feature_contract_sha256,:tuning_loss_contract_sha256,:generator,:cell_order,:spec_id)
const RESOURCE_CELL_FIELDS=(:delay_hours,:horizon,:training_count,:validation_count,:union_count)
const READBACK_FIELDS=(:schema,:complete,:evidence_mode,:head,:implementation_sources,:source_proof_sha256,:fixture_spec_sha256,
    :fixture_sha256,:fixture_data,:component,:component_audit,:runtime,:runtime_sha256,:cells,:models,:encoder_count,:fit_count,:native,:raw,:process_resources,:readback_id)
const READBACK_CELL_FIELDS=(:delay_hours,:horizon,:context_id,:training_count,:validation_count,:union_count,:response_min_f64_bits,
    :response_max_f64_bits,:encoder_member,:fit_input_sha256,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256)
const READBACK_MODEL_FIELDS=(:delay_hours,:horizon,:setting_index,:model_id,:model_member,:fit_receipt_member,:encoder_sha256,
    :training_count,:validation_count,:union_count,:fit_input_sha256,:nonbias_split_count,:native_increment_sha256,
    :reloaded_increment_sha256,:audited_increment_sha256)
const COMPONENT_FIELDS=(:schema,:complete,:evidence_mode,:initial_only,:pair_id,:source_bindings,:training_boundary,:validation_start,
    :validation_end,:feature_contract_sha256,:feature_schema_sha256,:tuning_loss_contract_sha256,:evotrees_source_sha256,
    :baseline_qualified_head,:baseline_qualification_sha256,:baseline_sources,:runtime,:execution_authority_sha256,:cells,:drafts,:members,:component_id)
const COMPONENT_CELL_FIELDS=(:delay_hours,:horizon,:context_id,:matched_input_id,:training_count,:validation_count,:union_count,
    :validation_persistence_count,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,
    :fit_input_sha256,:preprocessing_sha256,:training_design_sha256,:max_fit_target_complete,:max_fit_label_receipt,
    :max_validation_target_complete,:max_validation_label_receipt,:encoder_member,:models)
const COMPONENT_MODEL_FIELDS=(:setting_index,:model_id,:model_member,:fit_receipt_member,:candidate_member,:native_increment_sha256,
    :reloaded_increment_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
const FIT_RECEIPT_FIELDS=(:schema,:complete,:evidence_mode,:pair_id,:context_id,:delay_hours,:horizon,:setting_index,
    :model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:fit_input_sha256,
    :preprocessing_sha256,:training_design_sha256,:recipe,:model_id,:model_sha256,:encoder_sha256,:runtime_sha256,
    :native_increment_sha256,:reloaded_increment_sha256,:union_count,:fit_elapsed_ns,:execution_authority_sha256,:receipt_id)
const CA_FIELDS=(:schema,:complete,:evidence_mode,:operation,:qualified_head,:implementation_sources,:pair_id,:source_proof_sha256,
    :source_proof,:source_bindings,:contracts,:runtime,:component_root,:component_completion_sha256,:component_qualification,
    :execution_authority_sha256,:execution_authority,:references,:report_sha256,:members,:audit_id)
const REPORT_FIELDS=(:schema,:complete,:evidence_mode,:pair_id,:component_completion_sha256,:source_bindings,:cells,:references,
    :fitting_performed,:native_model_deserialization,:report_id)
const AUDIT_CELL_FIELDS=(:delay_hours,:horizon,:context_id,:training_count,:validation_count,:union_count,:validation_persistence_count,
    :model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,:training_design_sha256,
    :preprocessing_sha256,:fit_input_sha256,:encoder_sha256,:max_fit_target_complete,:max_fit_label_receipt,
    :max_selection_target_complete,:max_selection_label_receipt,:models)
const AUDIT_MODEL_FIELDS=(:setting_index,:model_id,:model_sha256,:fit_receipt_sha256,:native_increment_sha256,
    :reloaded_increment_sha256,:audited_increment_sha256,:candidate_prediction_sha256,:n,:rmse_nt,:mae_nt,:bias_nt)
function _member_record(root,row,controls;limit=T.MAX_META)
    _fields(row,(:name,:bytes,:sha256));_need(T._hex(row.sha256),"matched342 member pin grammar differs")
    path=P._member(root,row.name);size=T._int(row.bytes,1,limit)
    _need(filesize(path)==size,"matched342 declared member bytes differ")
    _pinned((;path,sha256=String(row.sha256)),controls;limit);path
end
function _inventory(root,rows,n,controls;total_limit=128*1024^2,expected_names=nothing)
    _need(length(rows)==n,"matched342 exact evidence inventory size differs")
    names=String[];total=0
    for row in rows
        _fields(row,(:name,:bytes,:sha256));_need(row.name isa AbstractString&&T._hex(row.sha256),"matched342 evidence member grammar differs")
        push!(names,String(row.name));total+=T._int(row.bytes,1,4*1024^2)
        _need(total<=total_limit,"matched342 component evidence exceeds total byte bound")
    end
    _need(issorted(names)&&allunique(names),"matched342 sorted unique member inventory required")
    expected_names===nothing||_need(Set(names)==Set(expected_names),"matched342 exact declared evidence member names differ before bodies")
    for row in rows
        _member_record(root,row,controls;limit=endswith(row.name,".model.json")||endswith(row.name,"encoder.json") ? 4*1024^2 : T.MAX_META)
    end
    Dict(String(row.name)=>row for row in rows)
end
function _full_shape_geometry(delay,horizon)
    d=T._int(delay,0,1);h=T._int(horizon,4,6);_need(h in (4,6),"matched342 full shape horizon differs")
    start=DateTime(1976,11,1);b=div(Dates.value(DateTime(2020))-Dates.value(start),3_600_000)
    v=div(Dates.value(DateTime(2023))-Dates.value(start),3_600_000)
    training=(1+d):(b-168-h-d);validation=b:(v-168-h-d);union=b:(v-168-4-d)
    _need(!isempty(training)&&!isempty(validation)&&last(union)+6<=500000,"matched342 full calendar exceeds frozen source bound")
    (;training,validation,union,training_count=length(training),validation_count=length(validation),union_count=length(union))
end
function _resource_spec(q,proof,controls)
    ref=_source_ref(q.fixture_spec,proof,controls);spec=_json_ref(ref,controls;fields=SPEC_FIELDS);_check_id(spec,:spec_id)
    _need(spec.schema=="matched342_initial_full_shape_fixture_spec_v1"&&spec.evidence_mode=="synthetic_only"&&
        T._int(spec.source_rows,500000,500000)==500000&&T._clock(spec.prefix_start_ms)==DateTime(1976,11,1)&&
        T._clock(spec.training_boundary_ms)==T._clock(spec.validation_start_ms)==DateTime(2020)&&T._clock(spec.validation_end_ms)==DateTime(2023)&&
        spec.receipt_semantics=="drivers_completion_dst_completion_plus_delay_v1"&&
        spec.all_17_channels_finite===true&&spec.required_drivers_valid===true&&spec.full_eligible_cohorts===true&&
        spec.variable_training_increments===true&&spec.feature_contract_sha256==M.FEATURE_CONTRACT_SHA256&&
        spec.tuning_loss_contract_sha256==R.LOSS_CONTRACT_SHA256&&Tuple(Tuple(row) for row in spec.cell_order)==CELLS,
        "matched342 meaningful full-shape fixture specification differs")
    _fields(spec.generator,(:member,:sha256));_need(endswith(spec.generator.member,".jl"),"matched342 fixture generator must be Julia")
    _source_member(spec.generator.member,spec.generator.sha256,proof,controls);spec
end
function _fixture_values(ref,expected,controls)
    checked=_pinned(ref,controls;limit=68_000_000)
    _need(checked.sha256==expected&&filesize(checked.path)==68_000_000,"matched342 full500000x17 raw fixture required")
    body=P._bound(checked.path,68_000_000);words=reinterpret(UInt64,body);values=Matrix{Float64}(undef,17,500000)
    for i in eachindex(values)
        x=reinterpret(Float64,ltoh(words[i]));channel=mod1(i,17)
        _need(isfinite(x)&&x<M._FILL_LIMITS[channel],"matched342 full-shape fixture must retain every finite raw channel")
        values[i]=x
    end
    for i in axes(values,2)
        _need(values[8,i]>=0&&isfinite(max(values[8,i]*max(-values[6,i],0.)/1000-.49,0.)),
            "matched342 full-shape required driver invalid")
    end
    values
end
function _same_sources(rows,sources)
    _need(P._sources(rows)==T._hashmap(sources),"matched342 qualified new source closure differs");true
end
function _check_counts(row,geometry)
    T._int(row.training_count,geometry.training_count,geometry.training_count)
    T._int(row.validation_count,geometry.validation_count,geometry.validation_count)
    T._int(row.union_count,geometry.union_count,geometry.union_count);true
end
function _tree_members(root,allowed;directories=nothing)
    observed=String[];dirs_seen=String[]
    for (parent,dirs,files) in walkdir(root;follow_symlinks=false)
        P._ordinary(parent)
        for dir in dirs
            path=P._ordinary(joinpath(parent,dir));push!(dirs_seen,replace(relpath(path,root),'\\'=>'/'))
        end
        for name in files
            path=P._ordinary(joinpath(parent,name));push!(observed,replace(relpath(path,root),'\\'=>'/'))
        end
    end
    _need(Set(observed)<=Set(allowed)&&allunique(observed),"matched342 unknown evidence tree member")
    directories===nothing||_need(Set(dirs_seen)==Set(directories),"matched342 evidence directory set differs")
    Set(observed)
end
function _component_names()
    names=String["qualification.json","feature-contract","loss-contract","routing/L0.bin","routing/L1.bin"]
    for (d,h) in CELLS
        push!(names,"L$d-h$h/encoder.json")
        for s in 1:8
            push!(names,"L$d-h$h/setting-$s.model.json");push!(names,"L$d-h$h/setting-$s.fit.json")
        end
    end
    Set(names)
end
function _resource_model_fields(cell,setting)
    g=M.GRID342[setting]
    Dict{String,Any}("schema"=>"matched_gbm_scalar342_model_v1","feature_count"=>342,
      "feature_schema_sha256"=>M.FEATURE_SCHEMA_SHA256,"evotrees_version"=>"0.18.7",
      "evotrees_source_sha256"=>M.EVOTREES_SOURCE_SHA256,"cell_delay"=>Int(cell.delay_hours),
      "cell_horizon"=>Int(cell.horizon),"setting_index"=>setting,"max_depth"=>g.max_depth,
      "nrounds"=>g.nrounds,"nbins"=>g.nbins,"seed"=>22022026,
      "training_mask_sha256"=>String(cell.model_training_mask_sha256),"preprocessing_sha256"=>String(cell.preprocessing_sha256),
      "input_admission_sha256"=>String(cell.context_id),"feature_contract_sha256"=>M.FEATURE_CONTRACT_SHA256,
      "loss_contract_sha256"=>R.LOSS_CONTRACT_SHA256)
end
function _resource_qualification(ref,head,sources,proof,proof_sha,root_sha,limits,controls)
    q=_json_ref(ref,controls;fields=RESOURCE_FIELDS)
    _need(q.schema=="matched342_initial_full_shape_resource_qualification_v1"&&q.complete===true&&q.evidence_mode=="synthetic_only"&&
        q.head==head&&q.source_proof_sha256==proof_sha&&T._hex(q.fixture_sha256),
        "matched342 current full-shape qualification header differs")
    _same_sources(q.implementation_sources,sources);_check_limits(q.limits)
    _need(T._bytes(q.limits)==T._bytes(limits),"matched342 actual/resource frozen limits differ")
    T._int(q.source_rows,500000,500000);T._int(q.feature_count,342,342);T._int(q.native_fit_count,32,32);T._int(q.encoder_count,4,4)
    _need(Tuple(Tuple(T._int(x,0,6) for x in row) for row in q.cell_order)==CELLS&&length(q.cells)==4,
        "matched342 full-shape cells differ")
    _resource_spec(q,proof,controls)
    readback=_json_ref(q.readback,controls;fields=READBACK_FIELDS);_check_id(readback,:readback_id)
    _need(readback.schema=="matched342_initial_full_shape_readback_v1"&&readback.complete===true&&readback.evidence_mode=="synthetic_only"&&
        readback.head==head&&readback.source_proof_sha256==q.source_proof_sha256&&readback.fixture_spec_sha256==q.fixture_spec.sha256&&
        readback.fixture_sha256==q.fixture_sha256&&_ref(readback.native)==_ref(q.native)&&_ref(readback.raw)==_ref(q.raw)&&
        _ref(readback.process_resources)==_ref(q.process_resources),"matched342 full-shape readback header/provenance differs")
    _same_sources(readback.implementation_sources,sources);runtime=_check_runtime(readback.runtime)
    _need(readback.runtime_sha256==runtime.runtime_sha256&&length(readback.cells)==4&&length(readback.models)==32,
        "matched342 full-shape readback runtime/counts differ")
    T._int(readback.encoder_count,4,4);T._int(readback.fit_count,32,32)
    values=_fixture_values(readback.fixture_data,q.fixture_sha256,controls)
    component=_json_ref(readback.component,controls;fields=COMPONENT_FIELDS);_check_id(component,:component_id)
    component_ref=_ref(readback.component);component_root=dirname(component_ref.path)
    _need(basename(component_ref.path)=="completion.json"&&basename(component_root)=="component"&&
        component.schema=="matched342_initial_component_completion_v1"&&component.complete===true&&component.initial_only===true&&
        component.evidence_mode=="synthetic_only"&&component.baseline_qualified_head==head&&
        component.execution_authority_sha256===nothing&&T._bytes(component.runtime)==T._bytes(readback.runtime)&&
        component.feature_contract_sha256==M.FEATURE_CONTRACT_SHA256&&component.feature_schema_sha256==M.FEATURE_SCHEMA_SHA256&&
        component.tuning_loss_contract_sha256==R.LOSS_CONTRACT_SHA256&&component.evotrees_source_sha256==M.EVOTREES_SOURCE_SHA256,
        "matched342 generated component scope/runtime differs")
    _same_sources(component.baseline_sources,sources);_binding_grammar(component.source_bindings)
    _need(all(row.raw_prefix_sha256==q.fixture_sha256 for row in component.source_bindings)&&length(component.cells)==4&&length(component.drafts)==2&&
        T._clock(component.training_boundary)==T._clock(component.validation_start)==DateTime(2020)&&T._clock(component.validation_end)==DateTime(2023),
        "matched342 component raw fixture/full calendar differs")
    members=_inventory(component_root,component.members,73,controls;expected_names=_component_names())
    _need(sum(Int(row.bytes) for row in component.members)+filesize(component_ref.path)<=128*1024^2,
        "matched342 component including completion exceeds128MiB")
    _need(Set(keys(members))==_component_names(),"matched342 exact73 generated component members required")
    observed=_tree_members(component_root,union(Set(keys(members)),Set(("completion.json",)));
        directories=("L0-h4","L0-h6","L1-h4","L1-h6","routing"))
    _need(observed==union(Set(keys(members)),Set(("completion.json",))),"matched342 component member missing")
    _need(members["feature-contract"].sha256==M.FEATURE_CONTRACT_SHA256&&members["loss-contract"].sha256==R.LOSS_CONTRACT_SHA256&&
        members["qualification.json"].sha256==component.baseline_qualification_sha256,"matched342 component contract/qualification copy differs")
    audit=_json_ref(readback.component_audit,controls;fields=CA_FIELDS);_check_id(audit,:audit_id)
    audit_ref=_ref(readback.component_audit);audit_root=dirname(audit_ref.path)
    _need(audit.schema=="matched342_initial_component_audit_completion_v1"&&audit.complete===true&&audit.evidence_mode=="synthetic_only"&&
        audit.operation=="independent_saved_state_audit_initial32"&&audit.qualified_head==head&&audit.pair_id==component.pair_id&&
        audit.source_proof_sha256==q.source_proof_sha256&&_absolute(audit.component_root)==component_root&&
        audit.component_completion_sha256==component_ref.sha256&&audit.execution_authority_sha256===nothing&&audit.execution_authority===nothing&&
        T._bytes(audit.runtime)==T._bytes(component.runtime)&&T._bytes(audit.source_bindings)==T._bytes(component.source_bindings),
        "matched342 independent generated CA provenance differs")
    _same_sources(audit.implementation_sources,sources)
    _need(_ref(audit.source_proof).sha256==q.source_proof_sha256,"matched342 CA upstream source proof pin differs")
    _proof_ref(audit.source_proof,controls)
    _fields(audit.contracts,(CONTRACT_FIELDS...,:audit_contract_sha256))
    for key in CONTRACT_FIELDS;_need(getproperty(audit.contracts,key)==getproperty(_contracts(),key),"matched342 CA contract differs");end
    _need(T._hex(audit.contracts.audit_contract_sha256)&&audit.contracts.audit_contract_sha256!=repeat("0",64),"matched342 independent audit contract missing")
    _need(audit.component_qualification.sha256==component.baseline_qualification_sha256,"matched342 CA component qualification differs")
    _component_qualification(audit.component_qualification,head,sources,controls;root_sha,proof_sha=q.source_proof_sha256,mode="synthetic_only")
    audit_members=_inventory(audit_root,audit.members,3,controls;total_limit=8*1024^2,
        expected_names=("reference-audit-L0.json","reference-audit-L1.json","audit-report.json"))
    _need(sum(Int(row.bytes) for row in audit.members)+filesize(audit_ref.path)<=8*1024^2,
        "matched342 audit including completion exceeds8MiB")
    _need(basename(audit_ref.path)=="audit-completion.json"&&
        Set(keys(audit_members))==Set(("reference-audit-L0.json","reference-audit-L1.json","audit-report.json")),
        "matched342 exact audit store members differ")
    _need(_tree_members(audit_root,union(Set(keys(audit_members)),Set((basename(audit_ref.path),)));
        directories=())==union(Set(keys(audit_members)),Set((basename(audit_ref.path),))),"matched342 complete audit store differs")
    report_path=P._member(audit_root,"audit-report.json");report=_json_ref((;path=report_path,sha256=String(audit.report_sha256)),controls;fields=REPORT_FIELDS)
    _check_id(report,:report_id)
    _need(report.schema=="matched342_initial_component_audit_report_v1"&&report.complete===true&&report.evidence_mode=="synthetic_only"&&
        report.pair_id==component.pair_id&&report.component_completion_sha256==component_ref.sha256&&
        report.fitting_performed===false&&report.native_model_deserialization===false&&length(report.cells)==4&&
        T._bytes(report.source_bindings)==T._bytes(component.source_bindings)&&T._bytes(report.references)==T._bytes(audit.references)&&
        audit_members["audit-report.json"].sha256==audit.report_sha256,"matched342 completed independent audit report differs")
    for k in 1:2
        d=k-1;draft=component.drafts[k];g=_full_shape_geometry(d,4)
        _need(draft.delay_hours==d&&draft.root=="reference-L$d"&&T._int(draft.model_count,g.union_count,g.union_count)==g.union_count&&
            T._int(draft.persistence_count,0,0)==0,"matched342 meaningful fixture must use full known-driver routing")
        route_path=_member_record(component_root,draft.routing_member,controls;limit=500016)
        _need(draft.routing_member.name=="routing/L$d.bin"&&T._bytes(draft.routing_member)==T._bytes(members["routing/L$d.bin"]),
            "matched342 resource routing member differs")
        route=P._bound(route_path,500016)
        _need(length(route)==16+g.union_count&&String(route[1:8])=="SSBMRTE1"&&ltoh(reinterpret(UInt64,route[9:16])[1])==1&&all(iszero,@view(route[17:end])),
            "matched342 all-valid full-shape routing bytes differ")
        names=Set(vcat(["grid.json","tuning.json","predictions.bin","qualification.json"],
            ["candidates/h$h/setting-$s.bin" for h in (4,6) for s in 1:8]))
        draft_root=P._member(dirname(component_root),draft.root);draft_members=_inventory(draft_root,draft.members,20,controls;expected_names=names)
        _need(Set(keys(draft_members))==names,"matched342 full-shape reference draft members differ")
        _tree_members(draft_root,union(names,Set(("audit.json","completion.json")));
            directories=("candidates","candidates/h4","candidates/h6"))
    end
    model_ids=String[]
    for (cell_index,(d,h)) in enumerate(CELLS)
        g=_full_shape_geometry(d,h);qcell=q.cells[cell_index];cell=component.cells[cell_index]
        rc=readback.cells[cell_index];ac=report.cells[cell_index]
        _fields(qcell,RESOURCE_CELL_FIELDS);_fields(cell,COMPONENT_CELL_FIELDS);_fields(rc,READBACK_CELL_FIELDS);_fields(ac,AUDIT_CELL_FIELDS)
        for row in (qcell,cell,rc,ac)
            T._int(row.delay_hours,d,d);T._int(row.horizon,h,h);_check_counts(row,g)
        end
        _need(cell.context_id==rc.context_id==ac.context_id==component.source_bindings[d+1].context_id&&
            cell.matched_input_id==component.source_bindings[d+1].matched_input_id&&length(cell.models)==length(ac.models)==8,
            "matched342 full-shape cell context/models differ")
        T._int(cell.validation_persistence_count,0,0);T._int(ac.validation_persistence_count,0,0)
        for key in (:fit_input_sha256,:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256)
            _need(T._hex(getproperty(cell,key))&&getproperty(cell,key)==getproperty(rc,key)==getproperty(ac,key),"matched342 full-shape cell identity differs")
        end
        _need(cell.preprocessing_sha256==ac.preprocessing_sha256&&cell.training_design_sha256==ac.training_design_sha256&&
            T._bytes(rc.encoder_member)==T._bytes(cell.encoder_member)&&cell.encoder_member.sha256==ac.encoder_sha256,
            "matched342 full-shape encoder/design differs")
        encoder_name="L$d-h$h/encoder.json"
        _need(cell.encoder_member.name==encoder_name&&T._bytes(cell.encoder_member)==T._bytes(members[encoder_name]),"matched342 encoder member linkage differs")
        encoder_path=_member_record(component_root,cell.encoder_member,controls;limit=4*1024^2)
        encoder=M.load_encoder(encoder_path;expected_sha256=String(cell.encoder_member.sha256),expected_training_design_sha256=String(cell.training_design_sha256))
        _need(M.preprocessing_sha256(encoder)==cell.preprocessing_sha256&&collect(encoder.training_rows)==collect(g.training),
            "matched342 padded/narrow full-shape encoder training rows rejected")
        response=Float64[values[15,i+h]-values[15,i-d] for i in g.training]
        low,high=extrema(response)
        _need(all(isfinite,response)&&low<high&&M._bits(low)==rc.response_min_f64_bits&&M._bits(high)==rc.response_max_f64_bits,
            "matched342 constant/incorrect full training increment response rejected")
        for (key,lastrow) in ((:max_fit_target_complete,last(g.training)),(:max_fit_label_receipt,last(g.training)),
                             (:max_validation_target_complete,last(g.validation)),(:max_validation_label_receipt,last(g.validation)))
            extra=key in (:max_fit_label_receipt,:max_validation_label_receipt) ? d : 0
            expected_clock=T.Adapter._shift(DateTime(1976,11,1),lastrow+h+extra)
            _need(T._clock(getproperty(cell,key))==expected_clock,"matched342 full-shape observed label clock differs")
            audit_key=key==:max_validation_target_complete ? :max_selection_target_complete :
                key==:max_validation_label_receipt ? :max_selection_label_receipt : key
            _need(T._clock(getproperty(ac,audit_key))==expected_clock,"matched342 independent full-shape label clock differs")
        end
        for s in 1:8
            cm=cell.models[s];am=ac.models[s];rm=readback.models[8*(cell_index-1)+s]
            _fields(cm,COMPONENT_MODEL_FIELDS);_fields(am,AUDIT_MODEL_FIELDS);_fields(rm,READBACK_MODEL_FIELDS)
            T._int(cm.setting_index,s,s);T._int(am.setting_index,s,s);T._int(rm.setting_index,s,s)
            T._int(rm.delay_hours,d,d);T._int(rm.horizon,h,h);_check_counts(rm,g)
            model_name="L$d-h$h/setting-$s.model.json";receipt_name="L$d-h$h/setting-$s.fit.json"
            _need(cm.model_member.name==model_name&&cm.fit_receipt_member.name==receipt_name&&
                T._bytes(cm.model_member)==T._bytes(members[model_name])==T._bytes(rm.model_member)&&
                T._bytes(cm.fit_receipt_member)==T._bytes(members[receipt_name])==T._bytes(rm.fit_receipt_member),
                "matched342 full32 model/fit member linkage differs")
            model_path=_member_record(component_root,cm.model_member,controls;limit=4*1024^2)
            object=M._read_json(model_path,String(cm.model_member.sha256));expected_fields=_resource_model_fields(cell,s)
            state=M._decode_model_data(object,expected_fields,M.GRID342[s])
            _need(object["model_id"]==cm.model_id==am.model_id==rm.model_id&&cm.model_member.sha256==am.model_sha256&&
                cm.fit_receipt_member.sha256==am.fit_receipt_sha256,"matched342 readback saved model/audit identity differs")
            splits=sum(count(identity,tree.split) for tree in state.trees[2:end])
            _need(splits>0&&T._int(rm.nonbias_split_count,splits,splits)==splits,"matched342 full-shape model has no reachable nonbias split evidence")
            receipt=_json_ref((;path=P._member(component_root,receipt_name),sha256=String(cm.fit_receipt_member.sha256)),controls;fields=FIT_RECEIPT_FIELDS)
            _check_id(receipt,:receipt_id)
            _need(receipt.schema=="matched342_initial_fit_receipt_v1"&&receipt.complete===true&&receipt.evidence_mode=="synthetic_only"&&
                receipt.pair_id==component.pair_id&&receipt.context_id==cell.context_id&&receipt.delay_hours==d&&receipt.horizon==h&&
                receipt.setting_index==s&&receipt.model_id==cm.model_id&&receipt.model_sha256==cm.model_member.sha256&&
                receipt.encoder_sha256==cell.encoder_member.sha256==rm.encoder_sha256&&receipt.runtime_sha256==readback.runtime_sha256&&
                receipt.execution_authority_sha256===nothing&&T._int(receipt.union_count,g.union_count,g.union_count)==g.union_count&&
                T._bytes(receipt.recipe)==T._bytes(M._recipe(s)),"matched342 actual32 native fit receipt differs")
            T._int(receipt.fit_elapsed_ns,1,q.limits.fit_wall_seconds*10^9-1)
            for key in (:model_training_mask_sha256,:reference_training_mask_sha256,:reference_validation_cohort_sha256,
                        :fit_input_sha256,:preprocessing_sha256,:training_design_sha256)
                _need(getproperty(receipt,key)==getproperty(cell,key),"matched342 native receipt full-cell input linkage differs")
            end
            _need(rm.fit_input_sha256==cell.fit_input_sha256&&
                cm.native_increment_sha256==cm.reloaded_increment_sha256==receipt.native_increment_sha256==receipt.reloaded_increment_sha256==
                am.native_increment_sha256==am.reloaded_increment_sha256==am.audited_increment_sha256==
                rm.native_increment_sha256==rm.reloaded_increment_sha256==rm.audited_increment_sha256&&
                T._hex(rm.native_increment_sha256),"matched342 native/save/reload/independent replay bits differ")
            T._int(cm.n,g.validation_count,g.validation_count);T._int(am.n,g.validation_count,g.validation_count)
            for key in (:rmse_nt,:mae_nt,:bias_nt)
                stored=T._float(getproperty(cm,key));audited=T._float(getproperty(am,key))
                _need(abs(stored-audited)<=5e-12*max(1.,abs(audited)),"matched342 independent full-cell metric differs")
            end
            push!(model_ids,String(cm.model_id))
        end
    end
    _need(length(model_ids)==32&&allunique(model_ids),"matched342 all32 distinct fixed-context states required")
    process_facts=_process_resources(q.process_resources,q,readback,proof,controls)
    _check_controls(controls)
    merge(process_facts,(;fixture_spec_sha256=String(q.fixture_spec.sha256),resource_qualification_sha256=String(ref.sha256),
        runtime_sha256=runtime.runtime_sha256,full_shape_cells=CELLS,actual_native_fit_count=32,encoder_count=4))
end
const _EXECUTION_STATES=Dict{String,Any}()
const _EVER_ADMITTED_PERMITS=Set{String}()
function _fold_path(path)
    lowercase(replace(normpath(abspath(path)),'\\'=>'/'))
end
function _within(path,root)
    a=_fold_path(path);b=_fold_path(root);a==b||startswith(a,b*"/")
end
function _execution_directory(path;fresh=false)
    directory=_absolute(path)
    _need(isdir(dirname(directory))&&!_within(directory,REPO)&&!_within(REPO,directory),
        "matched342 execution stage must be disjoint from checkout under an ordinary existing parent")
    if fresh
        _need(!ispath(directory),"matched342 actual execution stage must be a fresh leaf")
    else
        _need(!ispath(directory)||isdir(directory),"matched342 actual stage kind changed")
    end
    directory
end
function _authority_seal(authority::ActualExecutionAuthority342)
    T._id((;schema="matched342_supplied_actual_authority_seal_v1",permit=authority.permit,data=authority.data,
      sources=_source_rows(authority.sources),controls=Tuple((;path=first(p),sha256=last(p)) for p in authority.controls),
      resource_facts=authority.resource_facts,prelaunch=authority.prelaunch,began_ns=authority.began_ns))
end
function _authority_files(authority::ActualExecutionAuthority342;active=true)
    _need(_authority_seal(authority)==authority.seal,"matched342 supplied authority seal differs")
    _sources(authority.sources);_check_controls(authority.controls)
    data=authority.data;_check_contracts(data.contracts);_check_scope(data.scope);_check_limits(data.limits)
    observed=_check_runtime(data.runtime)
    _need(observed.runtime_sha256==authority.resource_facts.runtime_sha256&&
        authority.prelaunch.julia_pid==getpid()&&authority.prelaunch.execution_authority_sha256==authority.permit.sha256,
        "matched342 actual running Julia/prelaunch witness differs")
    _execution_directory(data.execution.directory)
    _need(strip(String(_git("rev-parse","HEAD")))==data.qualified_head&&
        isempty(_git("status","--porcelain=v1","--untracked-files=all")),"matched342 actual qualified checkout changed")
    _need(Sys.maxrss()>0&&Sys.maxrss()<=authority.resource_facts.peak_rss_budget_bytes,
        "matched342 actual Julia process resident peak exceeds frozen budget/unobserved")
    if active
        elapsed=time_ns()-authority.began_ns
        _need(elapsed<UInt64(data.limits.stage_wall_seconds)*UInt64(10^9),"matched342 actual initial stage deadline exhausted")
        _need(haskey(_EXECUTION_STATES,authority.permit.sha256),"matched342 active execution state absent")
        state=_EXECUTION_STATES[authority.permit.sha256]
        _need(state.authority_seal==authority.seal,"matched342 active execution state differs")
        expected=Tuple((d,h,s) for (d,h) in CELLS for s in 1:8)
        _need(length(state.attempts)<=32&&all(state.attempts[i]==expected[i] for i in eachindex(state.attempts)),
            "matched342 actual ordered32 attempt history differs")
        allocated=Base.GC_Diff(Base.gc_num(),state.gc_start).allocd
        _need(allocated>=0&&allocated<=authority.resource_facts.julia_gc_allocation_budget_bytes,
            "matched342 actual managed allocation budget exceeded")
        descriptors=_descriptor_observation()
        _need(descriptors.kind==authority.resource_facts.descriptor_kind&&descriptors.method==authority.resource_facts.descriptor_method&&
            descriptors.count-state.descriptor_before<=authority.resource_facts.descriptor_growth_budget,
            "matched342 actual process descriptor budget/method differs")
    end
    true
end
"Read supplied current qualifications/permission; never writes or mints authority files."
function admit_execution_authority(permit_ref;expected_sources,prelaunch_admission=nothing)
    began=time_ns();sources=_sources(expected_sources);controls=Dict{String,String}()
    checked=_pinned(permit_ref,controls);data=_json_ref(checked,controls;fields=PERMIT_FIELDS);_check_id(data,:permit_id)
    _need(data.schema=="qualified_actual_initial_matched342_permit_v1"&&data.complete===true&&
        data.evidence_mode==String(ACTUAL)&&data.operation=="initial_matched342_32fits"&&
        data.qualified_head isa AbstractString&&occursin(r"\A[0-9a-f]{40}\z",data.qualified_head),
        "matched342 explicit actual initial32 permit required")
    _need(!(checked.sha256 in _EVER_ADMITTED_PERMITS),"matched342 supplied actual permit already admitted in this Julia process")
    _same_sources(data.implementation_sources,sources);_check_contracts(data.contracts);_binding_grammar(data.sources)
    _check_scope(data.scope);_check_limits(data.limits);runtime=_check_runtime(data.runtime)
    proof_ref=_pinned(data.source_proof,controls);proof=recheck_source_proof(proof_ref;expected_head=String(data.qualified_head),expected_sources=sources)
    for (name,pin) in pairs(proof.source_sha256)
        controls[P._member(REPO,String(name))]=String(pin)
    end
    config=joinpath(REPO,".git","config")
    isfile(config)&&(controls[P._ordinary(config)]=P._file_sha(config)[1])
    root_ref=_pinned(data.root_qualification,controls)
    root=_root_qualification(root_ref,String(data.qualified_head),proof_ref.sha256,controls)
    q=_component_qualification(data.component_qualification,String(data.qualified_head),sources,controls;
        root_sha=root_ref.sha256,proof_sha=proof_ref.sha256)
    expected_pass=count(pair->endswith(String(first(pair)),".jl"),pairs(proof.source_sha256))+12
    T._int(q.harness.counts.PASS,expected_pass,expected_pass)
    _need(q.canonical_harness_sha256==root.candidate_pipeline_full18.harness_raw_sha256,
        "matched342 component/root harness raw linkage differs")
    resource=_resource_qualification(data.resource_qualification,String(data.qualified_head),sources,proof,proof_ref.sha256,
        root_ref.sha256,data.limits,controls)
    _need(resource.runtime_sha256==runtime.runtime_sha256,"matched342 profile/actual native runtime differs")
    execution=data.execution;_fields(execution,(:directory,:native_owner_sha256,:preserved_native_owner_sha256,:intent))
    directory=_execution_directory(execution.directory;fresh=true)
    _need(execution.native_owner_sha256==OWNER_SHA&&execution.preserved_native_owner_sha256==LEGACY_OWNER_SHA,
        "matched342 supplied fixed owner identities differ")
    intent=_json_ref(execution.intent,controls;fields=INTENT_FIELDS)
    _need(intent.schema=="matched342_initial_execution_intent_v1"&&intent.evidence_mode==String(ACTUAL)&&
        intent.operation==data.operation&&intent.qualified_head==data.qualified_head&&_absolute(intent.execution_directory)==directory,
        "matched342 actual prior intent differs")
    for key in (:implementation_sources,:contracts,:sources,:runtime,:scope,:limits)
        _need(T._bytes(getproperty(intent,key))==T._bytes(getproperty(data,key)),"matched342 prior intent/permit fact differs: $key")
    end
    _need(prelaunch_admission!==nothing,"matched342 real parent prelaunch admission is required; not unused post-allocation RAM")
    prelaunch=_prelaunch(prelaunch_admission,String(data.qualified_head),proof_ref.sha256,resource.fixture_spec_sha256,
        runtime.runtime_sha256,data.limits,controls;mode=ACTUAL,permit_sha=checked.sha256,pid=getpid())
    for path in keys(controls)
        _need(!_within(path,directory),"matched342 supplied control/evidence cannot be inside the fresh output stage")
    end
    _check_controls(controls);immutable_controls=Tuple(sort!(collect(controls);by=first))
    _need(time_ns()-began<UInt64(data.limits.stage_wall_seconds)*UInt64(10^9),"matched342 authority admission exhausted stage budget")
    provisional=ActualExecutionAuthority342(checked,_freeze(data),sources,immutable_controls,resource,prelaunch,began,"",_Token342())
    authority=ActualExecutionAuthority342(checked,_freeze(data),sources,immutable_controls,resource,prelaunch,began,_authority_seal(provisional),_Token342())
    descriptors=_descriptor_observation()
    state=(;attempts=NTuple{3,Int}[],gc_start=Base.gc_num(),descriptor_before=descriptors.count,authority_seal=authority.seal)
    _EXECUTION_STATES[checked.sha256]=state;push!(_EVER_ADMITTED_PERMITS,checked.sha256)
    _authority_files(authority);authority
end
"Active actual fitting check; post-fit independent readers use historical file pins instead."
function recheck_execution_authority(authority::ActualExecutionAuthority342,pair::SourcePair342{ACTUAL})
    recheck_pair(pair);_authority_files(authority)
    _need(T._bytes(authority.data.sources)==T._bytes(pair.bindings),"matched342 actual authority/source-pair bindings differ")
    for admission in pair.admissions
        _need(admission.metadata.source_sha256==BI.ACTUAL_SOURCE&&admission.metadata.prefix_start==DateTime(1976,11,1)&&
            admission.metadata.prefix_stop==DateTime(2023)&&admission.metadata.training_boundary==DateTime(2020)&&
            admission.metadata.validation_start==DateTime(2020)&&admission.metadata.validation_end==DateTime(2023),
            "matched342 actual admitted source/splits exceed initial scope")
        for path in (admission.source_proof_path,(member.absolute for member in admission.members)...)
            _need(!_within(path,authority.data.execution.directory),"matched342 actual input cannot overlap output stage")
        end
    end
    true
end
function recheck_execution_authority(::ActualExecutionAuthority342,::SourcePair342)
    throw(ArgumentError("matched342 actual authority rejects generated source contexts"))
end
function execution_authority_metadata(authority::ActualExecutionAuthority342)
    _authority_files(authority);data=authority.data
    (;permit_sha256=authority.permit.sha256,qualified_head=String(data.qualified_head),
      component_qualification_sha256=String(data.component_qualification.sha256),contracts=_freeze(data.contracts),
      runtime=_freeze(data.runtime),runtime_sha256=authority.resource_facts.runtime_sha256,
      execution_directory=String(data.execution.directory),scope=_freeze(data.scope),limits=_freeze(data.limits),
      implementation_sources=_source_rows(authority.sources),source_bindings=_freeze(data.sources))
end
function _reserve_fit!(authority::ActualExecutionAuthority342,cell::PreparedCell342{ACTUAL},expected::ExpectedBoundModel342{ACTUAL})
    recheck_execution_authority(authority,cell.pair);_cell_expected(cell,expected)
    state=_EXECUTION_STATES[authority.permit.sha256];index=length(state.attempts)+1
    _need(index<=32,"matched342 supplied actual authority has spent all32 fit attempts")
    sequence=Tuple((d,h,s) for (d,h) in CELLS for s in 1:8)
    requested=(cell.delay,cell.horizon,expected.setting_index)
    _need(requested==sequence[index],"matched342 actual fit order/retry differs from fixed32 scope")
    # An attempted native invocation remains spent even when its native fit,
    # snapshot, resource gate or subsequent saved-state verification fails.
    push!(state.attempts,requested);nothing
end

export SourcePair342,PreparedCell342,ExpectedBoundModel342,BoundFittedCell342,ActualExecutionAuthority342,
    bind_source_pair,recheck_pair,source_admissions,source_pair_metadata,runtime_identity,
    prepare_cell,recheck_cell,cell_metadata,expected_bindings,fit_generated_cell,fit_actual_cell,
    predict_cell,save_encoder,save_model,load_model,replay_cell,
    recheck_source_proof,admit_execution_authority,recheck_execution_authority,execution_authority_metadata
end
