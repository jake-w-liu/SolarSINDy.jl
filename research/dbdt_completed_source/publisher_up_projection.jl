"""Pure publisher numerical projection of a completed-source event log.

Source provenance and actual-data authority belong to a separately qualified
caller. The shared engine owns completion, censoring and same-time ordering.
"""
module PublisherUPProjection
using Dates,SolarSINDy
include("publisher_up_values.jl")
const P=PublisherUPValues
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const M=D.CMR
export project_shared
need(ok,message)=ok || throw(ArgumentError(message))
score(value,kind)=kind===:symmetric_absolute ? abs(value) : max(0.,value)
function warm_states(records,calibration,start,period,nominal,kind)
    need((records isa AbstractVector || records isa Tuple) && length(records)<=600000,"publisher warmup bound")
    owned=collect(records)
    metadata=NamedTuple[];mapping=Dict{Int,Any}()
    fields=Set((:id,:issue,:label_ready,:completed_at,:phase_period_minutes,:point_model_sha256,:policy,:mode,:signed_score))
    for row in owned
        need(row isa NamedTuple && Set(keys(row))==fields,"publisher warmup fields")
        push!(metadata,(;row.id,row.issue,row.label_ready,row.completed_at,row.phase_period_minutes,
            row.point_model_sha256,row.policy,row.mode))
        need(row.id isa Integer && !(row.id isa Bool) && !haskey(mapping,row.id),"publisher warmup identity")
        mapping[row.id]=row
    end
    selected=E.warmup_membership(metadata;start,calibration,phase_period_minutes=period)
    states=[P.State(;nominal) for _ in 1:period]
    for id in selected
        row=mapping[id];value=row.signed_score
        need(D.valid_number(value),"publisher selected warmup score")
        transformed=score(Float64(value),kind);slot=E.phase_slot(row.issue,period)
        states[slot]=P.advance(states[slot],transformed).state
    end
    states,selected
end
"""Project the exact publisher rule with the shared log's pending-label policy.

The default phase period comes from the completion policy (31 minutes under
the default clock). A phase with a still-pending prior issued bound records an
unavailable attempt. Complete/censored labels follow the shared source log;
future target fields are never examined. Numeric results assert no provenance
or scientific coverage qualification.
"""
function project_shared(source,calibration::D.CompletedCalibration,warmup;
                        start::DateTime,asof::DateTime,nominal=.1,score_kind=:upper_positive,phase=:policy,grace_ms=0)
    need(D.whole(start) && start<=asof,"publisher replay clocks")
    need(score_kind in (:upper_positive,:symmetric_absolute) && phase in (:policy,:safe),"publisher variant")
    alpha=D.nominal_value(nominal);need(0<alpha<1,"publisher requires interior nominal")
    grace=E.checked_grace(grace_ms);D.check_calibration(calibration,start,calibration.policy,calibration.mode)
    context=source.context
    need(D.policy_equal(context.policy,calibration.policy) && context.mode===calibration.mode &&
        context.point_model_sha256==calibration.point_model_sha256 && source.point_model_sha256==calibration.point_model_sha256 &&
        source.calibration_purpose===calibration.purpose && context.availability_kind===D.AVAILABILITY &&
        context.measured_receipts_available===false,"publisher shared context differs")
    period=phase===:policy ? E.base_period(start,calibration.policy) : E.safe_terminal_period(start,calibration.policy;grace_ms=grace)
    states,selected=warm_states(warmup,calibration,start,period,alpha,score_kind)
    need((source.events isa AbstractVector || source.events isa Tuple) && length(source.events)<=2_640_000,"publisher source log bound")
    events=collect(source.events)
    previous=zeros(Int,period);source_pending=Dict{Int,Any}();tickets=Dict{Int,Any}()
    issued=NamedTuple[];updates=NamedTuple[];attempted=0;resolved=0;misses=0;upper_misses=0;interval_misses=0
    unavailable=0;nonfinite=0;mismatches=0;completed=0;censored=0;max_pending=0;last_time=start;last_issue=nothing;last_id=0
    for event in events
        need(event isa NamedTuple && event.time isa DateTime && event.time>=last_time,"publisher event metadata chronology")
        event.time>asof && break
        last_time=event.time
        need(isequal(event.context,context),"publisher event context differs")
        need(event.kind in (:issue,:completed_target,:censored_target,:source_receipt,:duplicate_source,:ignored_stale),"publisher source event kind")
        event.kind in (:source_receipt,:duplicate_source,:ignored_stale) && continue
        if event.kind===:issue
            need(event.id isa Integer && !(event.id isa Bool) && event.id==last_id+1 && event.issue==event.time &&
                D.whole(event.issue) && (last_issue===nothing || event.issue>last_issue),"publisher issue identity/order")
            need(attempted<1_320_000,"publisher attempt bound")
            bounds=Tuple(event.bounds);need(!isempty(bounds),"publisher source issue lacks point/scale")
            center=first(bounds).center;scale=first(bounds).scale
            need(D.valid_number(center) && D.positive_number(scale) && all(b->b.center==center && b.scale==scale,bounds),"publisher frozen point/scale differs")
            timing=M.issue_timing(event.issue,calibration.policy);expires=D.shifted(timing.first_scheduled_feedback,grace)
            slot=E.phase_slot(event.issue,period);available=previous[slot]==0 || !haskey(tickets,previous[slot])
            attempted+=1;last_id=Int(event.id);last_issue=event.issue
            source_pending[last_id]=(;issue=event.issue,ready=timing.label_assumed_ready,expires)
            if available
                state=states[slot];radius=P.radius(state);upper=expm1(Float64(center)+Float64(scale)*radius)
                lower=score_kind===:symmetric_absolute ? max(0.,expm1(Float64(center)-Float64(scale)*radius)) : 0.
                need(!isnan(radius) && radius>=0 && !isnan(upper) && !isnan(lower),"publisher invalid numeric bound")
                finite=isfinite(upper)&&isfinite(lower);nonfinite+=!finite
                ticket=(;id=last_id,issue=event.issue,slot,center=Float64(center),scale=Float64(scale),radius,upper,lower,state)
                tickets[last_id]=ticket;previous[slot]=last_id
                push!(issued,(;id=last_id,issue=event.issue,available=true,reason=:issued,slot,radius,upper,lower,finite))
            else
                unavailable+=1
                push!(issued,(;id=last_id,issue=event.issue,available=false,reason=:prior_label_pending,slot,
                    radius=nothing,upper=nothing,lower=nothing,finite=false))
            end
            max_pending=max(max_pending,length(source_pending))
            need(length(source_pending)<=source.source.capacity,"publisher source pending capacity differs")
        else
            need(event.id isa Integer && !(event.id isa Bool) && haskey(source_pending,event.id),"publisher terminal event has no pending source issue")
            known=source_pending[event.id]
            need(event.issue==known.issue && event.label_ready==known.ready && event.expires==known.expires,"publisher terminal source clocks differ")
            if event.kind===:completed_target
                need(event.time>=known.ready && event.completed_at==event.time,"publisher completed label is premature")
                target=event.target;need(D.valid_number(target)&&target>=0,"publisher completed target domain")
                completed+=1
                if haskey(tickets,event.id)
                    ticket=tickets[event.id];state=states[ticket.slot]
                    need(P.encode_state(state)==P.encode_state(ticket.state),"publisher phase changed before prior label")
                    signed=(log1p(Float64(target))-ticket.center)/ticket.scale
                    need(isfinite(signed),"publisher signed residual overflow")
                    result=P.advance(state,score(signed,score_kind))
                    need(result.miss===(score(signed,score_kind)>ticket.radius),"publisher update differs from issued radius")
                    states[ticket.slot]=result.state;resolved+=1;misses+=result.miss
                    upper_error=target>ticket.upper;interval_error=target<ticket.lower || upper_error
                    upper_misses+=upper_error;interval_misses+=interval_error
                    raw_error=score_kind===:symmetric_absolute ? interval_error : upper_error
                    mismatches+=raw_error!=result.miss
                    push!(updates,(;id=event.id,issue=event.issue,completed_at=event.time,slot=ticket.slot,
                        target=Float64(target),signed_score=signed,score_miss=result.miss,upper_miss=upper_error,interval_miss=interval_error))
                end
            else
                need(event.time>known.expires,"publisher source censoring is premature")
                censored+=1
            end
            delete!(source_pending,event.id);delete!(tickets,event.id)
        end
    end
    count=length(issued)-unavailable
    (;context,variant=:publisher_uncorrected_algorithm1,score_kind,nominal=alpha,phase_selector=phase,phase_period_minutes=period,
      warmup_ids=collect(selected),states=Tuple(states),issued,updates,attempted,issued_bounds=count,unavailable,
      nonfinite_bounds=nonfinite,completed_source_targets=completed,censored_source_targets=censored,pending_source_targets=length(source_pending),
      target_unknown=censored+length(source_pending),binary_resolved=resolved,binary_unknown=attempted-resolved,
      issued_bound_binary_unknown=count-resolved,score_misses=misses,raw_upper_misses=upper_misses,raw_interval_misses=interval_misses,
      score_raw_membership_mismatches=mismatches,max_pending,all_attempted_score_enclosure=D.gap_certificate(attempted,resolved,misses,alpha),
      issued_bound_score_enclosure=D.gap_certificate(count,resolved,misses,alpha),all_attempted_upper_enclosure=D.gap_certificate(attempted,resolved,upper_misses,alpha),
      own_numeric_study_pass=unavailable==0 && nonfinite==0,shared_study_pass=source.study_pass,publisher_UP_numeric_implemented=true,
      source_bound_provenance_verified=false,scientific_coverage_qualified=false,availability_kind=D.AVAILABILITY,
      measured_receipts_available=false,full_logs_grow_with_attempts=true,actual_source_authority_granted=false,actual_fits=0)
end
end
