module DstIntervalTraining

using Dates, SHA
using ..DstIntervalAdapter
using ..DstIntervalStream
const A=DstIntervalAdapter
const S=DstIntervalStream
const K=A.Kernel
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
const CONTRACT_SHA256="18bdd26ae44267f7992378b82663558af3a5af3b64d45800366c107af6201658"
S.verify_source_identity()

export fit_gains, predict_rows, select_setting

_hash(x)=bytes2hex(sha256(repr(x)))
_valid_hash(x)=x isa AbstractString && occursin(r"^[0-9a-f]{64}$",x)

function verify_source_identity()
    dependencies=S.verify_source_identity()
    bytes2hex(sha256(read(@__FILE__)))==LOADED_SOURCE_SHA256 ||
        throw(ArgumentError("loadedsource/disk mismatch: DstIntervalTraining.jl"))
    (dependencies...,"DstIntervalTraining.jl"=>LOADED_SOURCE_SHA256)
end

function _check_setting(setting)
    setting in A.settings() || throw(ArgumentError("setting differs from frozen18grid"))
    setting
end

function _check_rows(rows,delay,source;setting=nothing)
    isempty(rows) && throw(ArgumentError("emptybasis panel"))
    issorted(getproperty.(rows,:issue)) && allunique(getproperty.(rows,:issue)) || throw(ArgumentError("issues must beordered andunique"))
    _valid_hash(source) || throw(ArgumentError("invalidsourceidentity"))
    for row in rows
        row.dst_delay_hours==delay && row.source_sha256==source || throw(ArgumentError("mixedsource/latency panel"))
        for (slot,h) in enumerate((4,6))
            clock=A.forecast_clock(row.issue,h,delay)
            (row.driver_tag,row.driver_receipt,row.anchor_tag,row.anchor_complete,row.anchor_receipt,
             row.target_tag[slot],row.target_complete[slot],row.target_receipt[slot])==
            (clock.driver_tag,row.issue,clock.anchor_tag,clock.anchor_complete,clock.anchor_receipt,
             clock.target_tag,clock.target_complete,clock.target_receipt) || throw(ArgumentError("rowclock contract mismatch"))
            (isfinite(row.target[slot]) || isnan(row.target[slot])) || throw(ArgumentError("invalidtargetvalue"))
            row.scoreable[slot]==(isfinite(row.anchor)&&isfinite(row.target[slot])) || throw(ArgumentError("scoreablecohort flag mismatch"))
        end
        if setting!==nothing
            row.taus==setting.taus && row.sigma==setting.sigma || throw(ArgumentError("mixedA dynamics panel"))
        end
        if setting!==nothing && row.model_available
            isfinite(row.anchor) && all(isfinite,row.offset) && all(isfinite,Iterators.flatten(row.coefficients)) ||
                throw(ArgumentError("nonfiniteavailableAdesign"))
        elseif setting!==nothing && isfinite(row.anchor)
            row.offset==(row.anchor,row.anchor) && row.coefficients==S._zero_basis() ||
                throw(ArgumentError("fallbackmust beexactcommonpersistence"))
        end
    end
    nothing
end

"""Fit two gains on a supplied synthetic/research affine panel. This performs
no raw-data I/O and does not qualify a real experiment or serving model.
"""
function fit_gains(rows,boundary::DateTime,setting;delay_hours::Integer,source_sha256::AbstractString)
    hashes=verify_source_identity()
    selected=_check_setting(setting)
    delay_hours in (0,1) && !(delay_hours isa Bool) || throw(ArgumentError("unknownlatency"))
    _check_rows(rows,delay_hours,source_sha256;setting=selected)
    admitted=Int[]
    for (index,row) in enumerate(rows)
        mature=all(A.fit_eligible(row.issue,row.target_complete[j],row.target_receipt[j],boundary) for j in 1:2)
        mature && all(row.scoreable) && push!(admitted,index)
    end
    informative=count(i->rows[i].model_available && any(!iszero,Iterators.flatten(rows[i].coefficients)),admitted)
    informative>=100 || throw(ArgumentError("fewerthan100informative jointorigins"))
    design=Matrix{Float64}(undef,2length(admitted),2)
    residual=Vector{Float64}(undef,2length(admitted))
    labels=similar(residual)
    mask_hash=_hash(("A-joint-fit-mask-v1",boundary,delay_hours))
    data_hash=_hash(("A-joint-fit-data-v1",boundary,delay_hours))
    cursor=0
    for index in admitted
        row=rows[index]
        mask_hash=_hash((mask_hash,(row.issue,row.anchor_tag,row.target_tag,row.target_complete,row.target_receipt)))
        data_hash=_hash((data_hash,(row.issue,row.anchor,row.offset,row.coefficients,row.target)))
        for slot in 1:2
            cursor+=1
            design[cursor,:].=row.coefficients[slot]
            labels[cursor]=row.target[slot]
            residual[cursor]=K._number(row.target[slot]-row.offset[slot],"trainingresidual")
        end
    end
    count_all=length(residual);count_storm=count(<=( -50.),labels)
    weights=[1/count_all+(count_storm>0 && labels[i]<=-50 ? selected.eta/count_storm : 0.) for i in eachindex(labels)]
    solved=A.nnls2(design,residual;weights)
    maximum_complete=maximum(maximum(rows[i].target_complete) for i in admitted)
    maximum_receipt=maximum(maximum(rows[i].target_receipt) for i in admitted)
    verify_source_identity()==hashes || throw(ArgumentError("sourceschanged duringgainfit"))
    base=(schema="candidate_a_intervalmean_research_v1",evidence_mode="synthetic_or_unqualified_research",contract_sha256=CONTRACT_SHA256,
        gains=solved.gains,setting_index=selected.index,taus=selected.taus,sigma=selected.sigma,eta=selected.eta,
        delay_hours=Int(delay_hours),boundary=boundary,source_sha256=String(source_sha256),source_hashes=hashes,
        fit_count=length(admitted),informative_count=informative,storm_count=count_storm,
        fit_data_sha256=data_hash,fit_mask_sha256=mask_hash,
        max_target_complete=maximum_complete,max_label_receipt=maximum_receipt,
        max_fit_issue=maximum(rows[i].issue for i in admitted),solver=solved)
    merge(base,(model_id=_hash(base),))
end

function predict_rows(model,rows)
    model.source_hashes==verify_source_identity() || throw(ArgumentError("model/loadedsource mismatch"))
    setting=A.settings()[model.setting_index]
    (model.taus,model.sigma,model.eta)==(setting.taus,setting.sigma,setting.eta) || throw(ArgumentError("modelsettings mismatch"))
    all(isfinite,model.gains) && all(>=(0),model.gains) || throw(ArgumentError("invalidmodelgains"))
    _check_rows(rows,model.delay_hours,model.source_sha256;setting)
    result=[begin
        prediction=Tuple(isfinite(row.anchor) ? (row.model_available ?
            row.offset[j]+sum(row.coefficients[j][k]*model.gains[k] for k in 1:2) : row.anchor) : NaN for j in 1:2)
        isfinite(row.anchor) && !all(isfinite,prediction) && throw(ArgumentError("nonfinitepointprediction"))
        merge(row,(prediction=prediction,setting_index=model.setting_index,eta=model.eta,model_id=model.model_id))
    end for row in rows]
    verify_source_identity()
    result
end

function _metrics(rows,indices,slot)
    isempty(indices) && return (n=0,rmse=NaN)
    value=K._precise() do
        total=zero(BigFloat)
        for index in indices
            value=rows[index].prediction[slot]
            isfinite(value) || throw(ArgumentError("predictionmissing onscoreable row"))
            total+=(BigFloat(value)-BigFloat(rows[index].target[slot]))^2
        end
        sqrt(total/length(indices))
    end
    (n=length(indices),rmse=A._nonzero_result(value,"panelRMSE"))
end

_identity(row)=(row.issue,row.driver_tag,row.anchor_tag,row.anchor,row.target_tag,row.target_complete,
    row.target_receipt,row.target,row.scoreable,row.source_sha256,row.dst_delay_hours)

function _selection_result(result,hashes)
    verify_source_identity()==hashes || throw(ArgumentError("sourceschanged duringselection"))
    result
end

"""Initial-reference-constrained A selection on supplied rows. Synthetic
reference metadata exercises this gate; realGBM lineage still requires its
separate source/model/prediction replay qualification before actual use.
"""
function select_setting(panels::AbstractDict,reference;validation_start::DateTime,
        validation_end::DateTime,delay_hours::Integer)
    hashes=verify_source_identity()
    reference===nothing && return _selection_result((status=:blocked_reference_absent,selected_index=nothing,panels=(),reference=nothing,cohort_sha256=()),hashes)
    validation_start<validation_end || throw(ArgumentError("unorderedvalidationwindow"))
    sort(collect(keys(panels)))==collect(1:18) || throw(ArgumentError("all18candidatepanels required"))
    delay_hours in (0,1) && !(delay_hours isa Bool) || throw(ArgumentError("unknownlatency"))
    reference.evidence_kind==:synthetic_initial_reference || throw(ArgumentError("unqualifiedreferencekind"))
    reference.training_boundary==validation_start && reference.delay_hours==delay_hours || throw(ArgumentError("referenceisnotinitialprevalidationmodel"))
    cutoff=A._shift(validation_start,-168)
    reference.max_fit_target_complete isa NTuple{2,DateTime} && reference.max_fit_label_receipt isa NTuple{2,DateTime} ||
        throw(ArgumentError("referencefitclocks require exactlytwo DateTime values"))
    all(j->reference.max_fit_target_complete[j]<=reference.max_fit_label_receipt[j]<=cutoff &&
        reference.max_fit_label_receipt[j]==A._shift(reference.max_fit_target_complete[j],delay_hours),1:2) ||
        throw(ArgumentError("referencefitlabelmaturity violation"))
    _valid_hash(reference.source_sha256) || throw(ArgumentError("invalidreferencesourceidentity"))
    length(reference.model_ids)==2 && all(_valid_hash,reference.model_ids) || throw(ArgumentError("missingreferencemodelidentities"))
    baseline=reference.rows
    _check_rows(baseline,delay_hours,reference.source_sha256)
    selection_rows=[i for i in eachindex(baseline) if validation_start<=baseline[i].issue<validation_end]
    isempty(selection_rows) && throw(ArgumentError("emptyvalidationreference"))
    reference_keys=getproperty.(baseline[selection_rows],:issue)
    admitted=Tuple([i for i in selection_rows if baseline[i].scoreable[j] &&
        A.fit_eligible(baseline[i].issue,baseline[i].target_complete[j],baseline[i].target_receipt[j],validation_end)] for j in 1:2)
    all(!isempty,admitted) || throw(ArgumentError("emptyperleadvalidationcohort"))
    storm=Tuple([i for i in admitted[j] if baseline[i].target[j]<=-50] for j in 1:2)
    base_metrics=Tuple((pooled=_metrics(baseline,admitted[j],j),storm=_metrics(baseline,storm[j],j)) for j in 1:2)
    cohort=Tuple(begin
        digest=_hash(("A-validation-cohort-v1",j,delay_hours,validation_start,validation_end))
        for i in admitted[j]
            digest=_hash((digest,(baseline[i].issue,baseline[i].target_tag[j])))
        end
        digest
    end for j in 1:2)
    results=NamedTuple[]
    for index in 1:18
        candidate=panels[index];setting=A.settings()[index]
        _check_rows(candidate,delay_hours,reference.source_sha256;setting)
        all(x->x.setting_index==index && x.eta==setting.eta,candidate) || throw(ArgumentError("candidateindex/eta mismatch"))
        locations=Dict(row.issue=>i for (i,row) in enumerate(candidate))
        keys=[row.issue for row in candidate if validation_start<=row.issue<validation_end]
        keys==reference_keys || throw(ArgumentError("candidatevalidationkeyset differs"))
        for i in selection_rows
            isequal(_identity(candidate[locations[baseline[i].issue]]),_identity(baseline[i])) || throw(ArgumentError("candidate/reference rowidentity differs"))
        end
        pooled=Tuple(_metrics(candidate,[locations[baseline[i].issue] for i in admitted[j]],j) for j in 1:2)
        activity=Tuple(_metrics(candidate,[locations[baseline[i].issue] for i in storm[j]],j) for j in 1:2)
        qualifies=all(pooled[j].rmse<=base_metrics[j].pooled.rmse for j in 1:2)
        normalized=any(isempty,storm) ? NaN : sum(base_metrics[j].storm.rmse==0 ?
            (activity[j].rmse==0 ? 1. : Inf) : activity[j].rmse/base_metrics[j].storm.rmse for j in 1:2)/2
        push!(results,(index=index,pooled=pooled,storm=activity,qualifies_pooled=qualifies,normalized_storm_rmse=normalized))
    end
    any(isempty,storm) && return _selection_result((status=:blocked_storm_reference,selected_index=nothing,
        panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
    qualifiers=[x for x in results if x.qualifies_pooled && isfinite(x.normalized_storm_rmse)]
    isempty(qualifiers) && return _selection_result((status=:none,selected_index=nothing,panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
    best=first(qualifiers)
    for row in qualifiers[2:end]
        row.normalized_storm_rmse<best.normalized_storm_rmse && (best=row)
    end
    _selection_result((status=:selected,selected_index=best.index,panels=Tuple(results),reference=base_metrics,cohort_sha256=cohort),hashes)
end

end # module
