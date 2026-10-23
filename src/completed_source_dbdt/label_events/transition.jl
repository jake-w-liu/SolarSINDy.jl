function normalized_row(row,time,policy)
    row isa NamedTuple && (Set(keys(row))==Set((:center,:rate)) || Set(keys(row))==Set((:center,:rate,:source_id))) ||
        throw(ArgumentError("source row must contain center/rate and optional source_id"))
    center=row.center;raw=row.rate
    center isa DateTime && D.whole(center) && D.valid_number(raw) && raw>=0 || throw(ArgumentError("invalid rate observation"))
    time>=D.feedback_ready(center,policy) || throw(ArgumentError("rate precedes assumed feedback readiness"))
    source_id=haskey(row,:source_id) ? row.source_id : "synthetic"
    source_id isa AbstractString && 0<ncodeunits(source_id)<=512 || throw(ArgumentError("invalid source identity"))
    converted=Float64(raw)
    (;center,rate=converted==0 ? 0.0 : converted,raw_bits=reinterpret(UInt64,converted),source_id=String(source_id))
end
function phase_slot(issue,period)
    elapsed=Base.Checked.checked_sub(Dates.value(issue),Dates.value(D.EPOCH))
    mod(div(elapsed,MINUTE_MS),period)+1
end
function updated_feedback(state::SubscriberState,slot,error)
    counts=collect(state.resolved);misses=collect(state.misses);alphas=collect(state.alpha)
    counts[slot]=increment(counts[slot]);misses[slot]=Base.Checked.checked_add(misses[slot],Int(error))
    if state.spec.kind!==:static
        alphas[slot]=state.spec.nominal+state.spec.gamma*(counts[slot]*state.spec.nominal-misses[slot])
    end
    replace_fields(state;resolved=Tuple(counts),misses=Tuple(misses),alpha=Tuple(alphas))
end
function updated_score(state::SubscriberState,score)
    count=increment(state.completed_scores);capacity=state.spec.rolling_capacity
    capacity==0 && return replace_fields(state;completed_scores=count)
    values=collect(state.scores);head=state.score_head
    if length(values)<capacity
        push!(values,score)
    else
        values[head]=score;head=mod1(head+1,capacity)
    end
    replace_fields(state;scores=Tuple(values),score_head=head,completed_scores=count)
end
function complete_score(window,bound)
    target=maximum(window.rates)
    score=(log1p(target)-bound.center)/bound.scale
    isfinite(score) || throw(ArgumentError("completed subscriber score is not finite"))
    score
end

"""
    prepare_time(owner, time; observations=(), issues=(), transaction_id)

Prepare one complete same-time batch without owner mutation. Receipts and labels
precede issues. Fixed consumers validate every full residual before commit.
Malformed rows, conflicting retained duplicates, stale clocks and score overflow
reject atomically. An expected pending phase instead records an unavailable bound
and fails a mandatory-comparator study while retaining the attempted source row.
"""
function prepare_time(owner::EventReplay,time::DateTime;observations=(),issues=(),transaction_id)
    current=owner.snapshot
    transaction_id isa AbstractString && 0<ncodeunits(transaction_id)<=256 || throw(ArgumentError("invalid transaction identity"))
    tx=String(transaction_id)
    time>=current.clock || throw(ArgumentError("clock cannot move backward"))
    same_clock=current.closed && time==current.clock
    same_clock && tx!=current.last_transaction_id && throw(ArgumentError("time watermark already closed"))
    cutoff=D.shifted(time,-current.retention_ms)
    rows=Tuple(normalized_row(row,time,current.calibration.policy) for row in observations)
    requests=Tuple(issues)
    length(requests)<=1 && all(row->row isa IssueRequest,requests) || throw(ArgumentError("at most one fixed issue request per clock"))
    for request in requests
        request.issue==time && D.whole(time) || throw(ArgumentError("issue does not match whole-minute transaction"))
        D.policy_equal(request.policy,current.calibration.policy) && request.mode===current.calibration.mode || throw(ArgumentError("issue policy/layout differs"))
    end
    canonical_rows=sort!(unique([(Dates.value(r.center),reinterpret(UInt64,r.rate),r.raw_bits,r.source_id) for r in rows]))
    request_key=Tuple((Dates.value(r.issue),r.feature_row_id,reinterpret(UInt64,r.center),reinterpret(UInt64,r.scale)) for r in requests)
    context_key=(current.context.stream_id,current.context.protocol_id,current.context.mode,current.context.point_model_sha256,
        current.calibration.policy.ground_extra_ms,current.calibration.policy.omni_extra_ms,current.calibration.policy.label_extra_ms)
    fingerprint=bytes2hex(sha256(codeunits(repr((context_key,Dates.value(time),Tuple(canonical_rows),request_key)))))
    if same_clock
        fingerprint==current.last_fingerprint || throw(ArgumentError("transaction identity reused with different payload"))
        result=TimeResult(time,(),(),(),true)
        return PreparedTime(owner.token,current.revision,tx,fingerprint,current,result)
    end
    tx==current.last_transaction_id && current.closed && throw(ArgumentError("transaction identity reused at another clock"))
    revision=increment(current.revision)
    grouped=Dict{DateTime,Vector{NamedTuple}}()
    for row in rows
        bucket=get!(grouped,row.center,NamedTuple[])
        !isempty(bucket) && first(bucket).rate!=row.rate && throw(ArgumentError("conflicting duplicate in source batch"))
        push!(bucket,row)
    end
    retained=Dict(row.center=>row.rate for row in current.seen if row.center>=cutoff)
    for center in keys(grouped)
        center>=cutoff && haskey(retained,center) && retained[center]!=first(grouped[center]).rate &&
            throw(ArgumentError("conflicting retained source duplicate"))
    end
    pending=Dict(w.id=>w for w in current.pending)
    subscribers=collect(current.subscribers)
    events=NamedTuple[];local_events=NamedTuple[];issue_records=NamedTuple[]
    novel=Dict{DateTime,Float64}()
    for center in sort!(collect(keys(grouped)))
        group=grouped[center];rate=first(group).rate
        identities=Tuple(sort!(unique([(source_id=r.source_id,raw_bits=r.raw_bits) for r in group]);by=r->(r.source_id,r.raw_bits)))
        kind=center<cutoff ? :ignored_stale : haskey(retained,center) ? :duplicate_source : :source_receipt
        push!(events,(;kind,center,rate,time,raw_identities=identities))
        if kind===:source_receipt
            novel[center]=rate;retained[center]=rate
        end
    end
    # Candidate windows are immutable; all projections finish before owner commit.
    for id in sort!(collect(keys(pending)))
        window=pending[id]
        if time<=window.expires
            rates=collect(window.rates)
            for (center,rate) in novel
                window.issue<center<=window.target_end || continue
                slot=Int(div(Dates.value(center-window.issue),MINUTE_MS))
                isfinite(rates[slot]) && rates[slot]!=rate && throw(ArgumentError("conflicting pending source duplicate"))
                rates[slot]=rate
            end
            window=replace_fields(window;rates=Tuple(rates));pending[id]=window
        end
        if time>=window.label_ready && all(isfinite,window.rates)
            for bound in window.bounds
                bound.available && complete_score(window,bound)
            end
        end
    end
    # Partial early misses are deterministic by ticket id, after whole-batch validation.
    for id in sort!(collect(keys(pending)))
        window=pending[id];time<=window.expires || continue
        applied=collect(window.feedback_applied)
        for j in eachindex(subscribers)
            state=subscribers[j];bound=window.bounds[j]
            if state.spec.kind===:early && bound.available && !applied[j] &&
                    any(value->isfinite(value) && value>bound.upper,window.rates)
                subscribers[j]=updated_feedback(state,bound.slot,true);applied[j]=true
                push!(local_events,(;kind=:feedback,name=state.spec.name,id,time,error=true,early=true))
            end
        end
        pending[id]=replace_fields(window;feedback_applied=Tuple(applied))
    end
    completed=current.completed;censored=current.censored
    complete_ids=[id for id in keys(pending) if time>=pending[id].label_ready && all(isfinite,pending[id].rates)]
    sort!(complete_ids;by=id->(pending[id].issue,id))
    for id in complete_ids
        window=pending[id];target=maximum(window.rates)
        push!(events,(;kind=:completed_target,id,issue=window.issue,time,completed_at=time,target,
            target_end=window.target_end,label_ready=window.label_ready,feedback_due=window.feedback_due,
            expires=window.expires,feature_row_id=window.feature_row_id,rates=window.rates))
        for j in eachindex(subscribers)
            state=subscribers[j];bound=window.bounds[j];bound.available || continue
            if !window.feedback_applied[j]
                error=target>bound.upper;state=updated_feedback(state,bound.slot,error)
                push!(local_events,(;kind=:feedback,name=state.spec.name,id,time,error,early=false))
            end
            score=complete_score(window,bound);state=updated_score(state,score);subscribers[j]=state
            push!(local_events,(;kind=:score,name=state.spec.name,id,time,target,score))
        end
        completed=increment(completed);delete!(pending,id)
    end
    expired_ids=sort!([id for id in keys(pending) if time>pending[id].expires];by=id->(pending[id].issue,id))
    for id in expired_ids
        window=pending[id]
        missing=Tuple(D.shifted(window.issue,k*MINUTE_MS) for k in 1:30 if !isfinite(window.rates[k]))
        push!(events,(;kind=:censored_target,id,issue=window.issue,time,reason=:incomplete_support,
            missing_centers=missing,target_end=window.target_end,label_ready=window.label_ready,
            feedback_due=window.feedback_due,expires=window.expires,feature_row_id=window.feature_row_id))
        censored=increment(censored);delete!(pending,id)
    end
    last_issue=current.last_issue;last_id=current.last_id;attempted=current.attempted;study_pass=current.study_pass
    for request in requests
        last_issue===nothing || request.issue>last_issue || throw(ArgumentError("issue clocks must increase"))
        last_id=increment(last_id)
        expires=D.shifted(request.timing.first_scheduled_feedback,current.grace_ms)
        expiry_timer=D.shifted(expires,1)
        length(pending)<current.capacity || throw(ArgumentError("pending capacity exceeded"))
        bounds=NamedTuple[]
        for j in eachindex(subscribers)
            state=subscribers[j];spec=state.spec
            slot=spec.kind===:phase ? phase_slot(time,state.period) : 1
            previous=state.previous[slot]
            available=spec.kind!==:phase || previous==0 || !haskey(pending,previous)
            upper=available ? D.upper_bound(state.alpha[slot],request.center,request.scale,
                spec.rolling_capacity==0 ? state.scores : sort!(collect(state.scores))) : nothing
            bound=(;name=spec.name,kind=spec.kind,available,reason=available ? :issued : :prior_label_pending,
                upper,alpha=state.alpha[slot],center=request.center,scale=request.scale,slot)
            push!(bounds,bound)
            prior=collect(state.previous)
            available && spec.kind===:phase && (prior[slot]=last_id)
            state=replace_fields(state;attempted=increment(state.attempted),previous=Tuple(prior),
                issued_bounds=Base.Checked.checked_add(state.issued_bounds,Int(available)),
                unavailable=Base.Checked.checked_add(state.unavailable,Int(!available)))
            subscribers[j]=state
            if !available
                study_pass &= !spec.mandatory
                push!(local_events,(;kind=:unavailable_bound,name=spec.name,id=last_id,time,reason=:prior_label_pending))
            end
        end
        window=WindowState(last_id,time,request.timing.last_target_center,request.timing.label_assumed_ready,
            request.timing.first_scheduled_feedback,expires,expiry_timer,request.feature_row_id,ntuple(_->NaN,30),
            Tuple(bounds),ntuple(_->false,length(subscribers)))
        pending[last_id]=window;last_issue=time;attempted=increment(attempted)
        record=(;kind=:issue,id=last_id,issue=time,time,feature_row_id=request.feature_row_id,bounds=Tuple(bounds))
        push!(events,record);push!(issue_records,record)
    end
    candidate=Snapshot(current.calibration,current.context,current.grace_ms,current.retention_ms,current.capacity,time,true,last_issue,last_id,
        revision,tx,fingerprint,Tuple(pending[id] for id in sort!(collect(keys(pending)))),
        Tuple((;center,rate=retained[center]) for center in sort!(collect(keys(retained)))),Tuple(subscribers),attempted,
        completed,censored,max(current.max_pending,length(pending)),max(current.max_seen,length(retained)),study_pass)
    contextual(values)=Tuple(merge(value,(;context=current.context)) for value in values)
    PreparedTime(owner.token,current.revision,tx,fingerprint,candidate,TimeResult(time,contextual(events),contextual(local_events),contextual(issue_records),false))
end

"""
    commit_time!(owner, plan)

Commit a prepared all-consumer transaction by one immutable snapshot assignment.
Plans bind owner identity/revision. An identical last transaction replay emits
no additional events; stale or cross-owner plans are rejected before mutation.
"""
function commit_time!(owner::EventReplay,plan::PreparedTime)
    plan.token===owner.token || throw(ArgumentError("plan belongs to another owner"))
    current=owner.snapshot
    isequal(plan.candidate.context,current.context) || throw(ArgumentError("prepared context differs"))
    if current.closed && current.last_transaction_id==plan.transaction_id && current.last_fingerprint==plan.fingerprint && current.clock==plan.result.time
        return TimeResult(current.clock,(),(),(),true)
    end
    plan.base_revision==current.revision || throw(ArgumentError("stale prepared revision"))
    plan.candidate.revision==increment(current.revision) || throw(ArgumentError("invalid prepared revision"))
    owner.snapshot=plan.candidate
    plan.result
end
