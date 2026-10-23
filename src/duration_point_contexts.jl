_clean(v::PointInputView,channel,i)=begin x=v.input.values[channel,i];(!isfinite(x)||x>=Adapter.Features.FILL_LIMITS[channel]) ? NaN : x end
function _masks(v::PointInputView,boundary,start,stop,delay)
    _int(delay,0,1);_need(start>=boundary&&stop>start,"B forecast range differs")
    train=falses(v.source_hours);forecast=falses(v.source_hours)
    leads=(falses(v.source_hours),falses(v.source_hours))
    for i in 1:v.source_hours
        issue=Adapter._shift(v.input.time[i],1);anchor=i-delay>=1 ? _clean(v,15,i-delay) : NaN
        clocks=ntuple(k->Adapter.forecast_clock(issue,(4,6)[k],delay),2)
        finite=ntuple(k->i+(4,6)[k]<=v.source_hours&&isfinite(anchor)&&isfinite(_clean(v,15,i+(4,6)[k])),2)
        train[i]=all(finite)&&all(c->Adapter.fit_eligible(issue,c.target_complete,c.target_receipt,boundary),clocks)
        for k in 1:2
            leads[k][i]=start<=issue<stop&&finite[k]&&Adapter.fit_eligible(issue,clocks[k].target_complete,clocks[k].target_receipt,stop)
        end
        forecast[i]=leads[1][i]||leads[2][i]
    end
    (;training=train,forecast,leads)
end
function _mask_bytes(v,mask,fold,boundary,delay)
    _need(!isempty(fold)&&isascii(fold)&&!occursin(r"[|\r\n]",fold),"canonical fold ID required")
    io=IOBuffer();write(io,"B-point-joint-mask-v1|$fold|$(Dates.value(boundary))|$delay\n")
    for i in eachindex(mask)
        mask[i]||continue
        issue=Adapter._shift(v.input.time[i],1);a=Adapter.forecast_clock(issue,4,delay);b=Adapter.forecast_clock(issue,6,delay)
        write(io,join(Dates.value.((issue,a.anchor_tag,a.target_tag,a.target_complete,a.target_receipt,b.target_tag,b.target_complete,b.target_receipt)),"|")*"\n")
    end
    take!(io)
end
function _make_binding(v,law,mask,fold,boundary,delay)
    D._validate(law)
    _need(law.boundary==boundary&&law.cutoff==Adapter._shift(boundary,-168)&&law.source_sha256==v.source_sha256,"B training law/source/boundary differs")
    _need(law.training_prefix_sha256==D._prefix_digest([x for x in v.wind if x.completed<=law.cutoff&&x.received<=law.cutoff]),"B law training prefix differs")
    fields=("b_point_training_binding_v1",:in_sample_two_stage_training,String(fold),boundary,law.cutoff,Int(delay),v.source_sha256,v.input_admission_sha256,v.input_prefix_sha256,v.view_id,_sha(_mask_bytes(v,mask,fold,boundary,delay)),v.wind_sequence_sha256,law.model_id,law.training_prefix_sha256,copy(v.source_hashes),CONTRACT)
    b=PointTrainingBinding(fields...,repeat("0",64));PointTrainingBinding(fields...,_id(_binding_data(b)))
end
mutable struct _ScopeState
    owner::Task
    active::Bool
    busy::Bool
    epoch::Int
    pair_id::String
    schedules::Vector{NamedTuple}
    counts::Dict{Symbol,Int}
end
_new_scope()=_ScopeState(current_task(),true,false,0,"",NamedTuple[],Dict{Symbol,Int}())
_count!(s,name,n=1)=(s.counts[name]=get(s.counts,name,0)+n)
function _scope(s)
    _need(s.active&&s.owner===current_task(),"B context expired or belongs to another Task")
end
struct _BatchTrainingContext
    binding::PointTrainingBinding
    view::PointInputView
    law::D.DurationModel
    mask::BitVector
    scope::_ScopeState
    beta_index::Int
end
struct _ForecastDesignContext
    binding::PointTrainingBinding
    view::PointInputView
    law::D.DurationModel
    mask::BitVector
    scope::_ScopeState
    beta_index::Int
    expected_gain_boundary::DateTime
    context_id::String
end
struct SharedForcingPair
    scope_token::_ScopeState
    owner_task::Task
    epoch_index::Int
    issue::DateTime
    horizon::Int
    role::Symbol
    source_sha256::String
    input_admission_sha256::String
    wind_state_id::String
    wind_prefix_sha256::String
    law_ids::NTuple{2,String}
    status::NTuple{2,Symbol}
    reason::NTuple{2,Symbol}
    qmeans::NTuple{2,Union{Nothing,NTuple{6,Rational{BigInt}}}}
    pivot_schedule_index::Int
    pivot_point_results::NTuple{2,Union{Nothing,NamedTuple}}
    pair_id::String
end
function _raise(errors)
    isempty(errors)&&return
    length(errors)==1 ? throw(only(errors)) : throw(CompositeException(errors))
end
function _with_batch_training_context(f::Function,v::PointInputView,law::D.DurationModel;fold_id,boundary::DateTime,delay_hours::Integer,training_mask,contract_sha256)
    _recheck(v);_need(contract_sha256==CONTRACT,"B contract differs")
    delay=_int(delay_hours,0,1);_need(training_mask isa AbstractVector{Bool}&&axes(training_mask)==(Base.OneTo(v.source_hours),),"B training mask axes differ")
    expected=_masks(v,boundary,boundary,Adapter._shift(boundary,1),delay).training
    _need(training_mask==expected,"B joint mature training mask differs")
    state=_new_scope();binding=_make_binding(v,law,copy(expected),fold_id,boundary,delay)
    context=_BatchTrainingContext(binding,v,law,copy(expected),state,law.beta==5 ? 1 : 2)
    errors=Any[];result=nothing
    try result=f(context) catch e;push!(errors,e)
    finally
        state.active=false
        try _recheck(v) catch e;push!(errors,e);end
    end
    _raise(errors);result
end
const ISSUE_KEYS=(:index,:issue,:driver_tag,:driver_receipt,:delay_hours,:taus,:sigma,:availability_kind,:active,:consecutive_valid,:reset_tag,:anchor_index,:anchor_tag,:anchor_complete,:anchor_receipt,:anchor,:a_model_available,:a_reason,:current_driver,:issue_checkpoint,:issue_basis,:resume_checkpoint,:source_sha256,:input_prefix_sha256,:implementation_hashes)
function _payload(context,p,physical)
    _scope(context.scope);v=context.view;b=context.binding;i=p.index
    _need(keys(p)==ISSUE_KEYS&&i isa Int&&1<=i<=v.source_hours,"B A payload keys/index differ")
    _need(context.mask[i]&&p.issue==Adapter._shift(v.input.time[i],1)&&p.driver_tag==v.input.time[i]&&p.driver_receipt==p.issue,"B origin outside bound mask or clock")
    recipe=Adapter.settings()[2*physical-1]
    _need(p.delay_hours==b.delay_hours&&p.taus==recipe.taus&&p.sigma==recipe.sigma&&p.source_sha256==v.source_sha256&&p.availability_kind==:assumed_completed_hour,"B A recipe/source differs")
    _need(p.implementation_hashes==LOADED_STREAM,"B A payload implementation differs")
    _need(p.resume_checkpoint.last_index==i&&p.resume_checkpoint.last_issue==p.issue&&p.resume_checkpoint.prefix_sha256==p.input_prefix_sha256,"B detached A resume identity differs")
    if p.a_model_available
        _need(p.issue_checkpoint isa D.R.Checkpoint&&p.issue_checkpoint.boundary==p.issue&&size(p.issue_basis)==(2,2)&&axes(p.issue_basis)==(Base.OneTo(2),Base.OneTo(2)),"B past-only A state differs")
    end
    _need(b.context_id==_id(_binding_data(b))&&b.input_view_id==v.view_id&&b.hazard_model_id==context.law.model_id,"B context identity differs")
    p
end
function _issue_state_id(p)
    p.issue_checkpoint===nothing&&return repeat("0",64)
    cp=p.resume_checkpoint
    pending=cp.pending===nothing ? nothing : (;index=cp.pending.index,start=cp.pending.start,stop=cp.pending.stop,q=cp.pending.q,pressure=cp.pending.pressure)
    data=(;schema="b_a_issue_state_v1",source_sha256=p.source_sha256,taus=p.taus,sigma=p.sigma,delay_hours=p.delay_hours,index=p.index,issue=p.issue,input_prefix_sha256=p.input_prefix_sha256,implementation_hashes=_hashdata(_hashmap(p.implementation_hashes)),mean=p.issue_checkpoint.mean,covariance=p.issue_checkpoint.covariance,basis=Tuple(Tuple(p.issue_basis[j,k] for k in 1:2) for j in 1:2),current_driver=p.current_driver===nothing ? nothing : (;index=p.current_driver.index,start=p.current_driver.start,stop=p.current_driver.stop,q=p.current_driver.q,pressure=p.current_driver.pressure),anchor=isfinite(p.anchor) ? p.anchor : nothing,reset_tag=p.reset_tag,pending)
    _id(data)
end
function _design(context,p,prepared::SharedForcingPair,physical;forecast=false)
    _payload(context,p,physical);s=context.scope;bi=context.beta_index;b=context.binding
    _need(prepared.scope_token===s&&prepared.owner_task===current_task()&&s.epoch==p.index==prepared.epoch_index&&s.pair_id==prepared.pair_id&&prepared.issue==p.issue,"B prepared pair expired or wrong epoch/Task")
    _need(prepared.source_sha256==b.source_sha256&&prepared.input_admission_sha256==b.input_admission_sha256&&prepared.law_ids[bi]==context.law.model_id,"B prepared law/source differs")
    _need(prepared.role==(forecast ? :validation_affine_design : :in_sample_two_stage_training),"B design role differs")
    if forecast
        _need(p.issue>=context.law.boundary&&p.issue>=context.expected_gain_boundary,"B actual forecast boundaries differ")
    end
    available=p.a_model_available&&prepared.status[bi]==:available
    offset=(p.anchor,p.anchor);coeff=((0.,0.),(0.,0.))
    if available
        schedule=s.schedules[physical]
        result=forecast&&physical==prepared.pivot_schedule_index ? prepared.pivot_point_results[bi] : D._mean_affine_coefficients(schedule,D._basis(map(x->K._number(x,"historical basis"),p.issue_basis)),prepared.qmeans[bi])
        if !(forecast&&physical==prepared.pivot_schedule_index);_count!(s,:coefficient_evaluations);end
        offset=(schedule.offsets[4],schedule.offsets[6]);coeff=(result.coefficients[4],result.coefficients[6])
    end
    reason=prepared.reason[bi]
    d=DesignValues(p.index,p.issue,p.anchor,offset,coeff,p.a_model_available,available,Symbol(p.a_reason),reason,p.input_prefix_sha256,prepared.wind_prefix_sha256,prepared.wind_state_id,context.law.model_id,_issue_state_id(p),b.context_id)
    forecast ? BForecastDesignRow(d) : BTrainingDesignRow(d)
end
_training_design(c::_BatchTrainingContext,p,prepared::SharedForcingPair)=_design(c,p,prepared,_physical_index(p))
forecast_design(c::_ForecastDesignContext,p,prepared::SharedForcingPair)=_design(c,p,prepared,_physical_index(p);forecast=true)
function _physical_index(p)
    ti=findfirst(==(p.taus),Adapter.TAUS);si=findfirst(==(p.sigma),Adapter.SIGMAS)
    _need(ti!==nothing&&si!==nothing,"unknown B physical recipe");(ti-1)*3+si
end
