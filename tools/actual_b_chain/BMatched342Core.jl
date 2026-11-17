"""Source-bound matched342 preparation/fitting and data-only state codecs.

Generated and archival source contexts retain distinct evidence modes.
Native fitting uses their admitted source, split and prepared-design bindings.
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
"Bind independently checked L0/L1 source files and their causal split identities."
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
struct BoundFittedCell342{Mode}
    model::M.MatchedScalarTreeReplay342.Model
    encoder::M.Encoder342
    expected::ExpectedBoundModel342{Mode}
    recipe::NamedTuple
    component_sha256::String
    seal::String
    function BoundFittedCell342{Mode}(model,encoder,expected,recipe,component,seal,::_Token342) where Mode
        new{Mode}(model,encoder,expected,recipe,component,seal)
    end
end
_component_sha(sources)=T._id((;schema="matched342_bound_component_sources_v1",sources=_source_rows(_sources(sources))))
function _capsule_seal(capsule::BoundFittedCell342{Mode}) where Mode
    object=M._model_object_data(capsule.model,_expected_fields(capsule.expected),M.GRID342[capsule.expected.setting_index])
    T._id((;schema="matched342_bound_fit_capsule_v1",evidence_mode=String(Mode),model_id=object["model_id"],
      preprocessing_sha256=M.preprocessing_sha256(capsule.encoder),expected_sha256=capsule.expected.seal,
      component_sha256=capsule.component_sha256,recipe=capsule.recipe))
end
function _check_capsule(capsule::BoundFittedCell342{Mode},expected::ExpectedBoundModel342{Mode}) where Mode
    _check_expected(expected)
    _need(capsule.expected.seal==expected.seal&&capsule.recipe==M._recipe(expected.setting_index)&&
        capsule.component_sha256==_component_sha(expected.pair.sources)&&
        M.preprocessing_sha256(capsule.encoder)==expected.preprocessing_sha256&&_capsule_seal(capsule)==capsule.seal,
        "matched342 bound capsule source/encoder/recipe/seal differs")
    true
end
function _fit_prepared(cell::PreparedCell342{Mode},expected::ExpectedBoundModel342{Mode}) where Mode
    _need(Mode in (ACTUAL,GENERATED),"matched342 fitting evidence mode differs")
    _cell_expected(cell,expected);identity=runtime_identity();recipe=M._recipe(expected.setting_index)
    component=_component_sha(cell.pair.sources)
    started=time_ns();gc_before=Base.gc_num()
    native=SolarSINDy.gbm_fit(cell.standardized[cell.cohort.train,:],cell.targets;recipe...,feature_names=collect(M.FEATURE_NAMES))
    model=M._snapshot_native(native,recipe)
    observations=(;fit_elapsed_ns=Int(time_ns()-started),allocated_bytes=Base.GC_Diff(Base.gc_num(),gc_before).allocd,
        peak_rss_bytes=Sys.maxrss(),threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads())
    _cell_expected(cell,expected);_need(runtime_identity()==identity&&_component_sha(cell.pair.sources)==component,
        "matched342 fit source/runtime changed")
    provisional=BoundFittedCell342{Mode}(model,cell.encoder,expected,recipe,component,"",_Token342())
    capsule=BoundFittedCell342{Mode}(model,cell.encoder,expected,recipe,component,_capsule_seal(provisional),_Token342())
    _check_capsule(capsule,expected)
    (;capsule,native,training_rows=length(cell.cohort.train),validation_rows=length(cell.cohort.validation),observations)
end
function fit_generated_cell(cell::PreparedCell342{GENERATED},expected::ExpectedBoundModel342{GENERATED})
    _fit_prepared(cell,expected)
end
function fit_actual_cell(cell::PreparedCell342{ACTUAL},expected::ExpectedBoundModel342{ACTUAL})
    _fit_prepared(cell,expected)
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
            "matched342 data control changed: "*path)
    end
    true
end
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
export SourcePair342,PreparedCell342,ExpectedBoundModel342,BoundFittedCell342,
    bind_source_pair,recheck_pair,source_admissions,source_pair_metadata,runtime_identity,
    prepare_cell,recheck_cell,cell_metadata,expected_bindings,fit_generated_cell,fit_actual_cell,
    predict_cell,save_encoder,save_model,load_model,replay_cell
end
