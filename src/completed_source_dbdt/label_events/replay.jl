"""
    source_certificate(owner)

Count all registered feature attempts, completed/censored/pending targets and
bounded live state. Unknown full targets include censored and pending windows,
even if an early consumer already knows a binary miss. No receipt-vintage claim.
"""
function source_certificate(owner::EventReplay)
    state=owner.snapshot
    (;context=state.context,attempted=state.attempted,completed_targets=state.completed,censored_targets=state.censored,
      pending_targets=length(state.pending),target_unknown=state.attempted-state.completed,
      max_pending=state.max_pending,max_retained_source=state.max_seen,capacity=state.capacity,
      retained_source=length(state.seen),study_pass=state.study_pass,
      availability_kind=D.AVAILABILITY,measured_receipts_available=false)
end

"""
    subscriber_certificate(owner, name)

Separate attempted rows, bounds actually issued, unavailable attempts and known
binary outcomes. The main enclosure uses all attempted rows, retaining unavailable
bounds as unknown. A separately labeled issued-bound enclosure cannot rescue
study failure; no risk/coverage claim applies when bounds were unavailable.
"""
function subscriber_certificate(owner::EventReplay,name::Symbol)
    selected=findfirst(s->s.spec.name===name,owner.snapshot.subscribers)
    selected===nothing && throw(ArgumentError("unknown subscriber"))
    state=owner.snapshot.subscribers[selected];resolved=sum(state.resolved);misses=sum(state.misses)
    all_attempted=D.gap_certificate(state.attempted,resolved,misses,state.spec.nominal)
    actual_bounds=D.gap_certificate(state.issued_bounds,resolved,misses,state.spec.nominal)
    residual=state.spec.kind===:static ? nothing : misses-resolved*state.spec.nominal+
        (sum(state.alpha)-length(state.alpha)*state.spec.nominal)/state.spec.gamma
    (;context=owner.snapshot.context,name,kind=state.spec.kind,attempted=state.attempted,issued_bounds=state.issued_bounds,unavailable=state.unavailable,
      binary_resolved=resolved,binary_misses=misses,binary_unknown=state.attempted-resolved,
      issued_bound_binary_unknown=state.issued_bounds-resolved,completed_scores=state.completed_scores,
      all_attempted_enclosure=all_attempted,actual_bound_enclosure=actual_bounds,
      actual_bound_denominator=:bounds_actually_issued,accounting_residual=residual,
      phase_period_minutes=state.period,phase_selector=state.spec.phase_period,
      all_attempted_bounds_defined=state.unavailable==0,scientific_coverage_qualified=false,study_pass=owner.snapshot.study_pass,
      rolling_score_count=length(state.scores),availability_kind=D.AVAILABILITY,measured_receipts_available=false)
end

"""
    warmup_membership(records; start, calibration, phase_period_minutes)

Select labels whose declared readiness and actual completion are both strictly
before start. Each record binds id/issue/readiness/completion/phase period plus
point SHA, policy and layout. Incompatible identities/periods throw. This supplies
membership only; publisher-UP numerical warmup/update state is not implemented.
"""
function warmup_membership(records;start::DateTime,calibration::D.CompletedCalibration,phase_period_minutes)
    phase_period_minutes isa Integer && !(phase_period_minutes isa Bool) && phase_period_minutes>0 ||
        throw(ArgumentError("invalid warmup period"))
    phase_period_minutes>=base_period(start,calibration.policy) || throw(ArgumentError("warmup period precedes policy period"))
    selected=NamedTuple[];ids=Set{Int}()
    expected=Set((:id,:issue,:label_ready,:completed_at,:phase_period_minutes,:point_model_sha256,:policy,:mode))
    for row in records
        row isa NamedTuple && Set(keys(row))==expected || throw(ArgumentError("invalid warmup record"))
        row.id isa Integer && !(row.id isa Bool) && 0<row.id<=typemax(Int) && !(row.id in ids) || throw(ArgumentError("invalid warmup id"))
        push!(ids,Int(row.id))
        row.issue isa DateTime && D.whole(row.issue) && row.label_ready isa DateTime && row.completed_at isa DateTime &&
            row.completed_at>=row.label_ready || throw(ArgumentError("invalid warmup chronology"))
        row.phase_period_minutes isa Integer && !(row.phase_period_minutes isa Bool) &&
            row.phase_period_minutes==phase_period_minutes && row.mode===calibration.mode &&
            row.point_model_sha256==calibration.point_model_sha256 && row.policy isa CMR.ReplayPolicy &&
            D.policy_equal(row.policy,calibration.policy) || throw(ArgumentError("incompatible warmup identity/period"))
        row.label_ready==CMR.issue_timing(row.issue,row.policy).label_assumed_ready || throw(ArgumentError("warmup readiness differs from policy"))
        row.label_ready<start && row.completed_at<start && push!(selected,row)
    end
    Tuple(row.id for row in sort!(selected;by=r->(r.completed_at,r.issue,r.id)))
end

"""
    replay_shared(prepared, centers, scales, batches, calibration;
                  subscribers, stream_id, protocol_id, asof, grace_ms=0, flush=true)

Run all fixed consumers on one ordered source/label log. Complete same-time
batches precede issues; explicit readiness/strict-expiry timers are processed.
Future receipt payloads remain unopened. A diagnostic flush never changes an
issued record. Complete logs are returned separately from bounded live state.
"""
function replay_shared(prepared,centers,scales,batches,calibration::D.CompletedCalibration;
                       subscribers,stream_id,protocol_id,asof::DateTime,grace_ms=0,flush=true)
    flush isa Bool || throw(ArgumentError("flush must be Boolean"))
    Base.require_one_based_indexing(prepared,centers,scales)
    stream=D.checked_stream(prepared,centers,scales,calibration,asof)
    start=prepared[first(stream.indices)].issue;stop=flush ? asof : last(prepared).issue
    owner=EventReplay(calibration;start,subscribers,stream_id,protocol_id,grace_ms)
    bytime=Dict{DateTime,Vector{Any}}();skipped_prestart=0;skipped_future=0
    for batch in batches
        time=batch.received
        time isa DateTime || throw(ArgumentError("invalid receipt clock metadata"))
        if time<start
            skipped_prestart+=1;continue
        elseif time>stop
            skipped_future+=1;continue
        end
        push!(get!(bytime,time,Any[]),batch)
    end
    issue_map=Dict(prepared[i].issue=>i for i in stream.indices)
    input_times=sort!(unique(vcat(collect(keys(bytime)),collect(keys(issue_map)))))
    cursor=1;events=NamedTuple[];subscriber_events=NamedTuple[];issue_records=NamedTuple[];transaction=0
    while true
        candidates=DateTime[]
        cursor<=length(input_times) && push!(candidates,input_times[cursor])
        for window in owner.snapshot.pending
            window.label_ready>owner.snapshot.clock && push!(candidates,window.label_ready)
            window.expiry_timer>owner.snapshot.clock && push!(candidates,window.expiry_timer)
        end
        isempty(candidates) && break
        time=minimum(candidates);time>stop && break
        observations=Any[]
        if haskey(bytime,time)
            for batch in bytime[time]
                append!(observations,batch.observations)
            end
        end
        requests=IssueRequest[]
        if haskey(issue_map,time)
            i=issue_map[time]
            push!(requests,IssueRequest(prepared[i],centers[i],scales[i];feature_row_id=string(i)))
        end
        transaction=increment(transaction)
        result=commit_time!(owner,prepare_time(owner,time;observations,issues=requests,transaction_id="replay/"*string(transaction)))
        append!(events,result.events);append!(subscriber_events,result.subscriber_events);append!(issue_records,result.issues)
        cursor<=length(input_times) && input_times[cursor]==time && (cursor+=1)
    end
    if stop>owner.snapshot.clock
        transaction=increment(transaction)
        result=commit_time!(owner,prepare_time(owner,stop;transaction_id="replay/"*string(transaction)))
        append!(events,result.events);append!(subscriber_events,result.subscriber_events)
    end
    certificates=Tuple(subscriber_certificate(owner,state.spec.name) for state in owner.snapshot.subscribers)
    (;context=owner.snapshot.context,indices=stream.indices,owner,events,subscriber_events,issue_records,source=source_certificate(owner),subscribers=certificates,
      study_pass=owner.snapshot.study_pass,diagnostic_flush=flush,asof,skipped_prestart,skipped_future,
      point_model_sha256=calibration.point_model_sha256,calibration_purpose=calibration.purpose,
      requires_source_bound_refits=true,publisher_UP_numeric_implemented=false,
      availability_kind=D.AVAILABILITY,measured_receipts_available=false)
end
