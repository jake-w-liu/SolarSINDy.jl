module DstInitialGBMReference

using Dates, SHA
const LOADED_SHA256=bytes2hex(sha256(read(@__FILE__)))
const SOURCE_METRIC_TOLERANCE=5e-12
const GRID=Tuple((max_depth=d,nrounds=r,nbins=b) for d in (4,6) for r in (200,400) for b in (64,255))
const REFERENCE_CONTRACT_SHA256="b3b081bc567b07e2b0ee091b8ccef38e89616cf7d9bcc9396998ec0ede70a6d9"
export RawLabels, clock, eligible, cohorts, first_grid_choice, verify_choice, verify_metric, metric,
    validation_rows, join_for_a, validate_receipts, verify_source_identity

_require(condition,message)=condition || throw(ArgumentError(message))
_hex(x)=x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)
function _get(x,k)
    if x isa AbstractDict
        _require(haskey(x,String(k)),"missing required receipt/row field: $k")
        return x[String(k)]
    end
    _require(hasproperty(x,Symbol(k)),"missing required receipt/row field: $k")
    getproperty(x,Symbol(k))
end
_hash(x)=bytes2hex(sha256(repr(x)))
function verify_source_identity()
    _require(bytes2hex(sha256(read(@__FILE__)))==LOADED_SHA256,"loadedsource/disk mismatch: reference adapter")
    ("DstInitialGBMReference.jl"=>LOADED_SHA256,)
end
function _shift(t::DateTime,h::Integer)
    _require(!(h isa Bool),"Boolean clock offset")
    value=try
        Base.Checked.checked_add(Dates.value(t),Base.Checked.checked_mul(Int64(h),Int64(3_600_000)))
    catch
        throw(ArgumentError("clock arithmetic overflow"))
    end
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(value)))
end
function clock(issue::DateTime,horizon::Integer;delay_hours::Integer=0)
    _require(!(horizon isa Bool) && horizon in (4,6) && !(delay_hours isa Bool) && delay_hours==0,
        "only initial L0 h4/h6 reference is qualified by this interface")
    _require(floor(issue,Hour)==issue,"issue must be an exact hour")
    tag=_shift(issue,-1);complete=_shift(issue,horizon)
    (issue=issue,origin_tag=tag,driver_tag=tag,anchor_tag=tag,anchor_complete=issue,anchor_receipt=issue,
        target_tag=_shift(complete,-1),target_complete=complete,target_receipt=complete,horizon=Int(horizon),delay_hours=0)
end
function eligible(issue::DateTime,complete::DateTime,receipt::DateTime,boundary::DateTime)
    _require(issue<complete<=receipt,"invalid label chronology")
    cutoff=_shift(boundary,-168)
    issue<boundary && complete<=cutoff && receipt<=cutoff
end

struct RawLabels
    time::Vector{DateTime}
    dst::Vector{Float64}
    source_sha256::String
    function RawLabels(time,dst;source_sha256,delay_hours::Integer=0)
        _require(delay_hours==0 && !(delay_hours isa Bool),"L1 requires a separate delay-aware baseline")
        times=DateTime.(time)
        _require(!isempty(times) && length(times)==length(dst) && allunique(times) && issorted(times),"invalid raw label grid")
        _require(all(t->floor(t,Hour)==t,times) && all(diff(times).==Hour(1)),"raw grid must be contiguous exact hours")
        _require(all(x->x isa Real && !(x isa Bool),dst),"raw labels must be real")
        values=Float64.(dst)
        _require(all(x->isfinite(x)||isnan(x),values) && _hex(source_sha256),"nonfinite label or invalid raw source digest")
        new(times,values,String(source_sha256))
    end
end

function cohorts(raw::RawLabels,horizon::Integer;validation_start::DateTime,validation_end::DateTime)
    verify_source_identity()
    _require(validation_start<validation_end && all(t->floor(t,Hour)==t,(validation_start,validation_end)),"invalid validation boundaries")
    n=length(raw.time);clocks=[clock(_shift(tag,1),horizon) for tag in raw.time]
    target=[i+horizon<=n ? raw.dst[i+horizon] : NaN for i in 1:n]
    valid=isfinite.(raw.dst).&isfinite.(target)
    train=[i for i in 1:n if valid[i] && eligible(clocks[i].issue,clocks[i].target_complete,clocks[i].target_receipt,validation_start)]
    validation=[i for i in 1:n if valid[i] && validation_start<=clocks[i].issue<validation_end &&
        eligible(clocks[i].issue,clocks[i].target_complete,clocks[i].target_receipt,validation_end)]
    (;horizon=Int(horizon),clocks,target,valid,train,validation)
end

function metric(prediction,observed)
    _require(length(prediction)==length(observed)>0 && all(isfinite,prediction) && all(isfinite,observed),"empty/nonfinite metric cohort")
    errors=Float64.(prediction).-Float64.(observed)
    result=(n=length(errors),rmse_nt=sqrt(sum(abs2,errors)/length(errors)),mae_nt=sum(abs,errors)/length(errors),
        bias_nt=sum(errors)/length(errors))
    _require(all(isfinite,(result.rmse_nt,result.mae_nt,result.bias_nt)),"metric overflow")
    result
end
function verify_metric(actual,expected)
    _require(_get(actual,:n)==_get(expected,:n),"original grid metric row count differs")
    for field in (:rmse_nt,:mae_nt,:bias_nt)
        a,e=_get(actual,field),_get(expected,field)
        _require(a isa Real && e isa Real && !(a isa Bool) && !(e isa Bool) && isfinite(a) && isfinite(e) &&
            abs(a-e)<=SOURCE_METRIC_TOLERANCE*max(1.,abs(e)),"original grid metric parity failed: $field")
    end
    true
end
function _configuration(x)
    values=Tuple(_get(x,k) for k in (:max_depth,:nrounds,:nbins))
    _require(all(v->v isa Integer && !(v isa Bool),values),"GBM grid settings must be integers")
    (max_depth=Int(values[1]),nrounds=Int(values[2]),nbins=Int(values[3]))
end

function first_grid_choice(candidates,stored_choice;expected_count::Integer)
    _require(!(expected_count isa Bool) && expected_count>0 && length(candidates)==8 &&
        Tuple(_configuration(_get(c,:candidate)) for c in candidates)==GRID,"original GBM grid differs")
    best=1;minimum=Inf
    for (i,candidate) in enumerate(candidates)
        score=_get(candidate,:score);n=_get(score,:n)
        _require(n isa Integer && !(n isa Bool) && n==expected_count,"grid metric count differs")
        values=Tuple(_get(score,k) for k in (:rmse_nt,:mae_nt,:bias_nt))
        _require(all(x->x isa Real && !(x isa Bool) && isfinite(x),values) && values[1]>=0 && values[2]>=0,"invalid stored grid metric")
        if values[1]<minimum;minimum=values[1];best=i;end
    end
    selected=_configuration(stored_choice)
    _require(selected==GRID[best],"stored choice differs from deterministic first-grid minimum")
    (selected_index=best,configuration=selected,expected_metric=_get(candidates[best],:score))
end

function verify_choice(record,raw::RawLabels,design_available;horizon,validation_start,validation_end)
    hashes=verify_source_identity()
    _require(_get(record,:horizon)==horizon && length(design_available)==length(raw.time) && eltype(design_available)<:Bool,
        "selection horizon or feature availability differs")
    frame=cohorts(raw,horizon;validation_start,validation_end)
    fit=[i for i in frame.train if design_available[i]]
    _require(collect(_get(record,:train_rows))==fit,"original per-lead GBM fit mask differs")
    _require(collect(_get(record,:validation_common_rows))==frame.validation,"original grid-choice validation cohort differs")
    choice=first_grid_choice(_get(record,:candidates),_get(record,:selected);expected_count=length(frame.validation))
    _require(!isempty(fit) && !isempty(frame.validation),"empty original reference cohort")
    fit_complete=maximum(frame.clocks[i].target_complete for i in fit)
    fit_receipt=maximum(frame.clocks[i].target_receipt for i in fit)
    selection_complete=maximum(frame.clocks[i].target_complete for i in frame.validation)
    selection_receipt=maximum(frame.clocks[i].target_receipt for i in frame.validation)
    _require(fit_complete<=_shift(validation_start,-168) && fit_receipt<=_shift(validation_start,-168) &&
        selection_complete<=_shift(validation_end,-168) && selection_receipt<=_shift(validation_end,-168),"reference fit/selection label clocks violate embargo")
    verify_source_identity()==hashes || throw(ArgumentError("sourcechanged duringchoicevalidation"))
    (selected_index=choice.selected_index,configuration=choice.configuration,fit_rows=fit,validation_rows=frame.validation,
        expected_metric=choice.expected_metric,max_fit_target_complete=fit_complete,max_fit_label_receipt=fit_receipt,
        max_selection_target_complete=selection_complete,max_selection_label_receipt=selection_receipt)
end

function validation_rows(raw::RawLabels,horizon,indices,prediction,available)
    _require(length(indices)==length(prediction) && length(available)==length(raw.time) && eltype(available)<:Bool,
        "reference export dimensions or availability differ")
    _require(issorted(indices) && allunique(indices),"reference export indices must be ordered and unique")
    [begin
        _require(i isa Integer && !(i isa Bool) && 1<=i<=length(raw.time) && i+horizon<=length(raw.time),"reference export index outside source")
        c=clock(_shift(raw.time[i],1),horizon)
        _require(isfinite(raw.dst[i]) && isfinite(raw.dst[i+horizon]) && isfinite(prediction[k]),"nonfinite reference export")
        _require(available[i] || prediction[k]==raw.dst[i],"unavailable reference must retain persistence fallback")
        merge(c,(row=Int(i),source_sha256=raw.source_sha256,anchor=raw.dst[i],target=raw.dst[i+horizon],
            prediction=Float64(prediction[k]),status=available[i] ? "model" : "persistence_feature_unavailable"))
    end for (k,i) in enumerate(indices)]
end
_key(row)=Tuple(_get(row,k) for k in (:source_sha256,:issue,:origin_tag,:driver_tag,:anchor_tag,:anchor_complete,
    :anchor_receipt,:target_tag,:target_complete,:target_receipt,:horizon,:delay_hours))

"""Join the complete independently reconstructed per-lead A cohort, including
fallbacks. Passing only the joint h4/h6 gain-fit subset fails its exact keyset.
"""
function join_for_a(reference_rows,a_rows,raw::RawLabels;horizon,validation_start,validation_end)
    hashes=verify_source_identity()
    frame=cohorts(raw,horizon;validation_start,validation_end)
    expected=validation_rows(raw,horizon,frame.validation,raw.dst[frame.validation],falses(length(raw.time)))
    keys_expected=_key.(expected)
    _require(_key.(a_rows)==keys_expected,"A selection rows differ from full per-lead mature raw cohort")
    reference=Dict{Any,Any}()
    for row in reference_rows
        key=_key(row)
        _require(!haskey(reference,key),"duplicate/conflicting reference key")
        reference[key]=row
    end
    joined=NamedTuple[]
    for (i,key) in enumerate(keys_expected)
        _require(haskey(reference,key),"missing reference key for eligible A row")
        row=reference[key];a=a_rows[i];truth=expected[i]
        _require(_get(row,:anchor)==_get(a,:anchor)==truth.anchor && _get(row,:target)==_get(a,:target)==truth.target,
            "joined source anchor/target value differs")
        p=_get(row,:prediction);status=_get(row,:status)
        _require(p isa Real && !(p isa Bool) && isfinite(p) && status in ("model","persistence_feature_unavailable"),"invalid reference point/status")
        _require(status!="persistence_feature_unavailable" || p==truth.anchor,"reference fallback differs from raw anchor")
        push!(joined,(key=key,row=truth.row,issue=truth.issue,target=truth.target,anchor=truth.anchor,
            prediction=Float64(p),status=status))
    end
    score=metric(getproperty.(joined,:prediction),getproperty.(joined,:target))
    verify_source_identity()==hashes || throw(ArgumentError("sourcechanged duringAjoin"))
    (rows=joined,metric=score,row_keys_sha256=_hash(keys_expected),
        max_target_complete=maximum(r.target_complete for r in expected),
        max_label_receipt=maximum(r.target_receipt for r in expected))
end

"""Validate receipt CONTENT against separately pinned expected identities. The
I/O wrapper must also hash actual receipt/files and the full immutable source
and package trees before loading/executing, and again at completion.
"""
function validate_receipts(terminal,completion,audit,expected)
    hashes=verify_source_identity()
    for field in (:driver_sha256,:runtime_gate_sha256,:synthetic_gate_sha256,:input_sha256,:format_sha256,
            :native_library_sha256,:julia_binary_sha256,:raw_hourly_matrix_sha256,:completion_sha256,:checker_sha256)
        _require(_hex(_get(expected,field)),"invalid expected source identity: $field")
    end
    _require(_get(expected,:package_commit) isa AbstractString && occursin(r"\A[0-9a-f]{40}\z",_get(expected,:package_commit)),"invalid package commit")
    _require(_get(terminal,:success)===true && _get(terminal,:all13_runner_files_unchanged)===true &&
        _get(terminal,:all388_package_files_unchanged)===true,"source run did not finish cleanly")
    runs=_get(terminal,:runs)
    _require(length(runs)==2 && [_get(r,:name) for r in runs]==["real_fit","independent_reader"],"source run steps incomplete or reordered")
    _require(all(r->_get(r,:native_exit) isa Integer && !(_get(r,:native_exit) isa Bool) && _get(r,:native_exit)==0 &&
        _get(r,:timed_out)===false && _hex(_get(r,:log_sha256)),runs),"failed/timed-out source step")
    for field in (:driver_sha256,:runtime_gate_sha256,:synthetic_gate_sha256,:source_sha256,:package_files,:package_commit,
            :input_sha256,:format_sha256,:native_library_sha256,:julia_binary_sha256)
        _require(_get(terminal,field)==_get(expected,field),"source execution identity differs: $field")
    end
    _require(length(_get(terminal,:source_sha256))==13 && length(_get(terminal,:package_files))==388,"source/package scope is incomplete")
    for field in (:source_sha256,:package_files,:completion_source_sha256)
        entries=_get(expected,field)
        _require(entries isa AbstractDict && all(p->first(p) isa AbstractString && !isempty(first(p)) && _hex(last(p)),entries),
            "invalid expected file identity map: $field")
    end
    _require(length(_get(expected,:completion_source_sha256))==18,"completion source scope differs")
    _require(_get(terminal,:independent_confirmation)===false && _get(terminal,:novelty_or_operational_superiority)===false,
        "source execution claim scope differs")
    _require(_get(completion,:complete)===true && _get(completion,:input_sha256)==_get(expected,:input_sha256) &&
        _get(completion,:raw_hourly_matrix_sha256)==_get(expected,:raw_hourly_matrix_sha256) &&
        _get(completion,:source_sha256)==_get(expected,:completion_source_sha256),"source completion identities differ")
    _require(_get(completion,:configuration)==_get(expected,:configuration),"source frozen configuration differs")
    config=_get(expected,:configuration)
    _require(collect(_get(config,:horizons))==[4,6] && _get(config,:mode) in ("synthetic_fixture","exposed_development") &&
        _get(config,:embargo_hours) isa Integer && !(_get(config,:embargo_hours) isa Bool) && _get(config,:embargo_hours)==168 &&
        all(k->_get(config,k) isa Integer && !(_get(config,k) isa Bool) && _get(config,k)==0,
            (:extra_dst_delay_hours,:extra_driver_delay_hours)) &&
        _get(config,:target_imputation)===false && _get(config,:novel_or_operational_superiority_claim)===false,
        "source clock/method configuration differs")
    _require(_get(completion,:no_original_model_parity)===true && _get(completion,:no_live_serving_changes)===true,"source claim flags differ")
    live=_get(_get(completion,:native),:live_objects)
    _require(_get(_get(completion,:native),:sha256)==_get(expected,:native_library_sha256) &&
        collect(_get(_get(completion,:native),:version))==[3,2,0] && live isa Integer && !(live isa Bool) && live==0,
        "source native identity/state differs")
    _require(_get(audit,:complete)===true && _get(audit,:completion_sha256)==_get(expected,:completion_sha256) &&
        _get(audit,:checker_sha256)==_get(expected,:checker_sha256),"independent source audit binding differs")
    _require(_get(audit,:source_mode)==_get(_get(expected,:configuration),:mode) && collect(_get(audit,:horizons))==[4,6] &&
        _get(audit,:checks) isa Integer && !(_get(audit,:checks) isa Bool) && _get(audit,:checks)>0 &&
        _get(audit,:new_model_fitting)===false && _get(audit,:author_or_ml_library_imported)===false &&
        _get(audit,:scientific_verdict)===false,"independent source audit scope differs")
    outputs=_get(completion,:output_sha256)
    _require(outputs isa AbstractDict && all(name->haskey(outputs,name),
        ("started.json","initial_imputation.json","initial_imputed_counts.json","gbm29_selection_h4.json",
         "gbm29_selection_h6.json","input_hourly_channels.f64","input_hourly_channels.json")),"source initial artifact map is incomplete")
    for (name,digest) in outputs
        _require(name isa AbstractString && _hex(digest),"invalid source output digest map")
    end
    verify_source_identity()==hashes || throw(ArgumentError("sourcechanged duringreceiptvalidation"))
    true
end

end # module
