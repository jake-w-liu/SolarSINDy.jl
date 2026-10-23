mutable struct MaximumTicket
    id::Int
    issue::DateTime
    target_end::DateTime
    label_ready::DateTime
    feedback_due::DateTime
    expires::DateTime
    center::Float64
    scale::Float64
    upper::Float64
    rates::Vector{Float64}
    feedback_applied::Bool
end

"""
    CompletedMaximumController(calibration; start, gamma, nominal=0.1,
                               early_feedback=true, grace_ms=0, rolling_capacity=0)

Bounded assumed-receipt controller holding30-rate target windows. Early mode
can certify a miss from a received exceeding rate; a completed residual always
requires the whole valid window and label readiness. Expiry/censoring retains
unknown outcomes. This controller is independent of the static-map replay path.
"""
mutable struct CompletedMaximumController
    calibration::CompletedCalibration
    nominal::Float64
    gamma::Float64
    alpha::Float64
    early_feedback::Bool
    grace_ms::Int64
    retention_ms::Int64
    capacity::Int
    rolling_capacity::Int
    score_head::Int
    scores::Vector{Float64}
    clock::DateTime
    last_issue::Union{Nothing,DateTime}
    issued::Int
    resolved::Int
    misses::Int
    censored_labels::Int
    mature_scores::Int
    pending::Dict{Int,MaximumTicket}
    seen::Dict{DateTime,Float64}
    max_pending::Int
    max_seen::Int
end

function CompletedMaximumController(cal::CompletedCalibration;start::DateTime,
        gamma,nominal=0.1,early_feedback=true,grace_ms=0,rolling_capacity=0)
    whole(start) || throw(ArgumentError("invalid controller start clock"))
    nominal=nominal_value(nominal);gamma=gain_value(gamma)
    early_feedback isa Bool || throw(ArgumentError("early_feedback must be Boolean"))
    grace_ms isa Integer && !(grace_ms isa Bool) && 0<=grace_ms<=86_400_000 || throw(ArgumentError("invalid receipt grace"))
    rolling_capacity isa Integer && !(rolling_capacity isa Bool) && rolling_capacity>=0 || throw(ArgumentError("invalid score capacity"))
    check_calibration(cal,start,cal.policy,cal.mode)
    period=CMR.issue_timing(start,cal.policy).sufficient_phase_period_minutes
    retention=(period+1)*MINUTE_MS+Int64(grace_ms)
    capacity=period+cld(grace_ms,MINUTE_MS)+1
    scores=rolling_capacity==0 ? copy(cal.sorted_scores) : copy(cal.chronological_scores[max(1,end-rolling_capacity+1):end])
    CompletedMaximumController(cal,Float64(nominal),Float64(gamma),Float64(nominal),early_feedback,
        Int64(grace_ms),retention,capacity,Int(rolling_capacity),1,scores,start,nothing,0,0,0,0,0,
        Dict{Int,MaximumTicket}(),Dict{DateTime,Float64}(),0,0)
end

function feedback!(state,ticket,miss,time,events)
    ticket.feedback_applied && return
    ticket.feedback_applied=true
    state.resolved+=1;state.misses+=miss
    state.alpha=state.nominal+state.gamma*(state.nominal*state.resolved-state.misses)
    push!(events,(;kind=:feedback,id=ticket.id,time,error=miss))
end

function insert_score!(state,score)
    state.mature_scores+=1
    state.rolling_capacity==0 && return
    if length(state.scores)<state.rolling_capacity
        push!(state.scores,score)
    else
        state.scores[state.score_head]=score
        state.score_head=mod1(state.score_head+1,state.rolling_capacity)
    end
end

function finish!(state,ticket,time,events)
    time>=ticket.label_ready && all(isfinite,ticket.rates) || return
    target=maximum(ticket.rates)
    score=(log1p(target)-ticket.center)/ticket.scale
    isfinite(score) || throw(ArgumentError("completed score is not finite"))
    feedback!(state,ticket,target>ticket.upper,time,events)
    insert_score!(state,score)
    push!(events,(;kind=:score,id=ticket.id,time,target,score))
    delete!(state.pending,ticket.id)
end

function prune_seen!(state,time)
    cutoff=shifted(time,-state.retention_ms)
    for observed in collect(keys(state.seen))
        observed<cutoff && delete!(state.seen,observed)
    end
end

function advance_internal!(state,time,events)
    state.clock=time
    for id in sort!(collect(keys(state.pending)))
        ticket=state.pending[id]
        finish!(state,ticket,time,events)
        haskey(state.pending,id) || continue
        if time>ticket.expires
            state.censored_labels+=1
            push!(events,(;kind=:censored,id,time,binary_known=ticket.feedback_applied))
            delete!(state.pending,id)
        end
    end
    prune_seen!(state,time)
end

"""
    advance!(state::CompletedMaximumController, time)

Advance a nondecreasing receipt clock, complete ready windows and censor expired
windows. Return emitted events. A window expires strictly after its expiry time.
"""
function advance!(state::CompletedMaximumController,time::DateTime)
    time>=state.clock || throw(ArgumentError("clock cannot move backward"))
    # Check representability before any mutation.
    shifted(time,-state.retention_ms)
    events=NamedTuple[];advance_internal!(state,time,events);events
end

"""
    issue!(state::CompletedMaximumController, prepared, center, scale)

Issue a bound after processing pending events at the issue clock. Issue times
increase on whole UTC minutes; process all receipts at a clock before issuing.
Return the id, fixed upper bound, alpha and pending-window events.
"""
function issue!(state::CompletedMaximumController,prepared::PreparedIssue,center::Real,scale::Real)
    prepared.accepted && prepared.mode===state.calibration.mode && policy_equal(prepared.policy,state.calibration.policy) ||
        throw(ArgumentError("issue has incompatible or rejected features"))
    time=prepared.issue
    whole(time) && time>=state.clock && (state.last_issue===nothing || time>state.last_issue) ||
        throw(ArgumentError("issue clocks must increase on whole UTC minutes"))
    valid_number(center) && positive_number(scale) || throw(ArgumentError("invalid point/scale"))
    expires=shifted(prepared.timing.first_scheduled_feedback,state.grace_ms)
    shifted(time,-state.retention_ms)
    events=NamedTuple[];advance_internal!(state,time,events)
    length(state.pending)<state.capacity || error("bounded pending capacity exceeded")
    sorted=state.rolling_capacity==0 ? state.scores : sort(state.scores)
    upper=upper_bound(state.alpha,Float64(center),Float64(scale),sorted)
    state.issued+=1;state.last_issue=time
    ticket=MaximumTicket(state.issued,time,prepared.timing.last_target_center,
        prepared.timing.label_assumed_ready,prepared.timing.first_scheduled_feedback,expires,
        Float64(center),Float64(scale),upper,fill(NaN,30),false)
    state.pending[ticket.id]=ticket
    state.max_pending=max(state.max_pending,length(state.pending))
    (;id=ticket.id,upper,alpha=state.alpha,events)
end

feedback_ready(center,policy)=shifted(center,45_000+policy.ground_extra_ms+policy.label_extra_ms)

"""Receive a complete same-time batch, validating conflicts before state mutation."""
function receive!(state::CompletedMaximumController,observations;received::DateTime)
    received>=state.clock || throw(ArgumentError("receipt clock cannot move backward"))
    cutoff=shifted(received,-state.retention_ms)
    unique=Dict{DateTime,Float64}()
    for row in observations
        center=row.center;rate=row.rate
        center isa DateTime && whole(center) && valid_number(rate) && rate>=0 || throw(ArgumentError("invalid ground-rate observation"))
        received>=feedback_ready(center,state.calibration.policy) || throw(ArgumentError("rate used before assumed feedback readiness"))
        value=Float64(rate)
        haskey(unique,center) && unique[center]!=value && throw(ArgumentError("conflicting duplicate in source batch"))
        center>=cutoff && haskey(state.seen,center) && state.seen[center]!=value &&
            throw(ArgumentError("conflicting retained source duplicate"))
        unique[center]=value
    end
    # A finite-score failure must also leave the batch atomic.
    for ticket in values(state.pending)
        received<=ticket.expires || continue
        proposed=copy(ticket.rates)
        for (center,rate) in unique
            ticket.issue<center<=ticket.target_end || continue
            slot=Int(Dates.value(center-ticket.issue)÷MINUTE_MS)
            isfinite(proposed[slot]) && proposed[slot]!=rate && throw(ArgumentError("conflicting pending source duplicate"))
            proposed[slot]=rate
        end
        if received>=ticket.label_ready && all(isfinite,proposed)
            isfinite((log1p(maximum(proposed))-ticket.center)/ticket.scale) || throw(ArgumentError("completed score is not finite"))
        end
    end
    events=NamedTuple[];advance_internal!(state,received,events)
    for center in sort!(collect(keys(unique)))
        if center<cutoff
            push!(events,(;kind=:ignored_stale,center,time=received))
        else
            state.seen[center]=unique[center]
        end
    end
    for id in sort!(collect(keys(state.pending)))
        ticket=state.pending[id]
        received<=ticket.expires || continue
        for (center,rate) in unique
            ticket.issue<center<=ticket.target_end || continue
            slot=Int(Dates.value(center-ticket.issue)÷MINUTE_MS)
            ticket.rates[slot]=rate
        end
        state.early_feedback && any(value->isfinite(value) && value>ticket.upper,ticket.rates) &&
            feedback!(state,ticket,true,received,events)
        finish!(state,ticket,received,events)
    end
    state.max_seen=max(state.max_seen,length(state.seen))
    events
end

"""
    certificate(state::CompletedMaximumController)

Return issued/resolved/miss/unknown counts and outward-rounded gap endpoints,
plus censoring, complete-score and bounded-state diagnostics. The accounting
residual is a floating-point diagnostic, not an additional scientific guarantee.
"""
function certificate(state::CompletedMaximumController)
    proof=gap_certificate(state.issued,state.resolved,state.misses,state.nominal)
    residual=state.misses-state.resolved*state.nominal+(state.alpha-state.nominal)/state.gamma
    merge(proof,(;accounting_residual=residual,censored_labels=state.censored_labels,
        completed_scores=state.mature_scores,pending_labels=length(state.pending),
        max_pending=state.max_pending,max_retained_source=state.max_seen,
        availability_kind=AVAILABILITY,measured_receipts_available=false))
end

"""Materialize assumed receipt batches only when their explicit asof permits access."""
function materialize_receipts(mag::AbstractDict,centers;policy=CMR.ReplayPolicy(),asof::DateTime,scheduled=true)
    scheduled isa Bool || throw(ArgumentError("scheduled must be Boolean"))
    times=collect(centers)
    all(time->time isa DateTime && whole(time),times) && length(unique(times))==length(times) ||
        throw(ArgumentError("source centers must be distinct whole UTC minutes"))
    batches=NamedTuple[];rejections=NamedTuple[];pending=DateTime[]
    for center in sort(times)
        ready=feedback_ready(center,policy);received=scheduled ? ceil_minute(ready) : ready
        if received>asof
            push!(pending,center);continue
        end
        previous=shifted(center,-MINUTE_MS)
        if !(haskey(mag,center) && haskey(mag,previous))
            push!(rejections,(;center,reason=:missing_ground_rate));continue
        end
        before,after=mag[previous],mag[center]
        if !(before isa Union{Tuple,AbstractVector} && after isa Union{Tuple,AbstractVector} &&
             length(before)==length(after)==2 && all(valid_number,before) && all(valid_number,after))
            push!(rejections,(;center,reason=:invalid_ground_rate));continue
        end
        rate=hypot(Float64(first(after))-Float64(first(before)),Float64(last(after))-Float64(last(before)))
        if !isfinite(rate)
            push!(rejections,(;center,reason=:invalid_ground_rate));continue
        end
        push!(batches,(;received,observations=[(;center,rate)]))
    end
    (;batches,rejections,pending,availability_kind=AVAILABILITY,measured_receipts_available=false)
end

"""Merge same-time receipts before issues; post-stream processing changes no bound."""
function replay_receipts(prepared,centers,scales,batches,cal::CompletedCalibration;
        gamma,nominal=0.1,early_feedback=true,grace_ms=0,rolling_capacity=0,
        asof::DateTime,flush=true)
    stream=checked_stream(prepared,centers,scales,cal,asof)
    first_issue=prepared[first(stream.indices)].issue
    stop=flush ? asof : last(prepared).issue
    state=CompletedMaximumController(cal;start=first_issue,gamma,nominal,early_feedback,grace_ms,rolling_capacity)
    bytime=Dict{DateTime,Vector{NamedTuple}}()
    skipped_prestart=0
    for batch in batches
        if batch.received<first_issue
            skipped_prestart+=1;continue
        end
        batch.received<=stop || continue
        append!(get!(bytime,batch.received,NamedTuple[]),batch.observations)
    end
    issues=Dict(prepared[i].issue=>i for i in stream.indices)
    times=sort!(unique(vcat(collect(keys(bytime)),collect(keys(issues)))))
    upper=fill(NaN,length(prepared));alpha_issue=copy(upper);events=NamedTuple[]
    for time in times
        haskey(bytime,time) && append!(events,receive!(state,bytime[time];received=time))
        if haskey(issues,time)
            index=issues[time]
            answer=issue!(state,prepared[index],centers[index],scales[index])
            upper[index]=answer.upper;alpha_issue[index]=answer.alpha
            append!(events,answer.events)
        end
    end
    flush && append!(events,advance!(state,asof))
    selected_upper=upper[stream.indices]
    (;indices=stream.indices,upper,alpha_issue,events,state,certificate=certificate(state),
      skipped_prestart,diagnostic_flush=flush,availability_kind=AVAILABILITY,
      full_count=count(==(Inf),selected_upper),empty_count=count(<(0.0),selected_upper),
      finite_negative_count=count(value->isfinite(value) && value<0,selected_upper),
      asof,mode=stream.mode,policy=stream.policy,gamma=state.gamma,nominal=state.nominal,
      schedule=early_feedback ? :early_receipt : :maturity_receipt,
      point_model_sha256=cal.point_model_sha256,calibration_schema=CALIBRATION_SCHEMA,
      measured_receipts_available=false,requires_source_bound_refits=true)
end
