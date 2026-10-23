"""
    SubscriberSpec(name, kind; nominal=0.1, gamma=0.1, phase_period=:policy,
                   rolling_capacity=0, mandatory=true)

Declare one fixed `:static`, `:phase`, `:fifo`, `:early` or `:maturity` consumer.
Only early/maturity support rolling scores. `:safe` fixes the terminal-safe phase
period from declared grace before replay; it must never be selected from outcomes.
No callable consumer or user callback is supported.
"""
struct SubscriberSpec
    name::Symbol
    kind::Symbol
    nominal::Float64
    gamma::Float64
    phase_period::Symbol
    rolling_capacity::Int
    mandatory::Bool
    function SubscriberSpec(name,kind;nominal=0.1,gamma=0.1,phase_period=:policy,
                            rolling_capacity=0,mandatory=true)
        name isa Symbol && 0<ncodeunits(String(name))<=128 || throw(ArgumentError("invalid subscriber name"))
        kind in (:static,:phase,:fifo,:early,:maturity) || throw(ArgumentError("unsupported fixed subscriber"))
        phase_period in (:policy,:safe) || throw(ArgumentError("invalid phase-period policy"))
        rolling_capacity isa Integer && !(rolling_capacity isa Bool) && 0<=rolling_capacity<=typemax(Int) ||
            throw(ArgumentError("invalid rolling capacity"))
        kind in (:early,:maturity) || rolling_capacity==0 || throw(ArgumentError("this subscriber uses fixed calibration"))
        kind===:phase || phase_period===:policy || throw(ArgumentError("phase selector requires phase subscriber"))
        mandatory isa Bool || throw(ArgumentError("mandatory must be Boolean"))
        new(name,kind,D.nominal_value(nominal),D.gain_value(gamma),phase_period,Int(rolling_capacity),mandatory)
    end
end

"""
    IssueRequest(prepared, center, scale; feature_row_id=string(prepared.issue))

Capture a feature-eligible issue and frozen point/scale for every fixed consumer.
Targets are not read. The adapter independently rechecks the policy timing.
"""
struct IssueRequest
    issue::DateTime
    timing::NamedTuple
    policy::CMR.ReplayPolicy
    mode::Symbol
    center::Float64
    scale::Float64
    feature_row_id::String
    function IssueRequest(prepared::D.PreparedIssue,center,scale;feature_row_id=string(prepared.issue))
        prepared.accepted || throw(ArgumentError("rejected features cannot issue"))
        D.valid_number(center) && D.positive_number(scale) || throw(ArgumentError("invalid frozen point/scale"))
        feature_row_id isa AbstractString && 0<ncodeunits(feature_row_id)<=512 || throw(ArgumentError("invalid feature row id"))
        timing=CMR.issue_timing(prepared.issue,prepared.policy)
        timing==prepared.timing || throw(ArgumentError("issue timing differs from policy"))
        new(prepared.issue,timing,prepared.policy,prepared.mode,Float64(center),Float64(scale),String(feature_row_id))
    end
end

struct SubscriberState
    spec::SubscriberSpec
    period::Int
    alpha::Tuple
    resolved::Tuple
    misses::Tuple
    previous::Tuple
    scores::Tuple
    score_head::Int
    attempted::Int
    issued_bounds::Int
    unavailable::Int
    completed_scores::Int
end
struct WindowState
    id::Int
    issue::DateTime
    target_end::DateTime
    label_ready::DateTime
    feedback_due::DateTime
    expires::DateTime
    expiry_timer::DateTime
    feature_row_id::String
    rates::NTuple{30,Float64}
    bounds::Tuple
    feedback_applied::Tuple
end
struct Snapshot
    calibration::NamedTuple
    context::NamedTuple
    grace_ms::Int64
    retention_ms::Int64
    capacity::Int
    clock::DateTime
    closed::Bool
    last_issue::Union{Nothing,DateTime}
    last_id::Int
    revision::Int
    last_transaction_id::String
    last_fingerprint::String
    pending::Tuple
    seen::Tuple
    subscribers::Tuple
    attempted::Int
    completed::Int
    censored::Int
    max_pending::Int
    max_seen::Int
    study_pass::Bool
end

replace_fields(value::T;kwargs...) where T = T((haskey(kwargs,n) ? kwargs[n] : getfield(value,n) for n in fieldnames(T))...)
increment(value::Int)=Base.Checked.checked_add(value,1)
function checked_grace(grace)
    grace isa Integer && !(grace isa Bool) && 0<=grace<=86_400_000 || throw(ArgumentError("invalid receipt grace"))
    Int64(grace)
end
function base_period(start,policy)
    CMR.issue_timing(start,policy).sufficient_phase_period_minutes
end

"""
    safe_terminal_period(start, policy=CMR.ReplayPolicy(); grace_ms=0)

Return policy period + floor(grace/60000) + 1, the earliest whole-minute reuse
strictly after expiry. Fix this selector before replay. Terminalization may be
censoring and does not prove coverage for unknown outcomes.
"""
function safe_terminal_period(start::DateTime,policy=CMR.ReplayPolicy();grace_ms=0)
    grace=checked_grace(grace_ms)
    Base.Checked.checked_add(base_period(start,policy),Base.Checked.checked_add(div(grace,MINUTE_MS),1))
end

"""
    EventReplay(calibration; start, subscribers, stream_id, protocol_id, grace_ms=0)

Create a single-owner synthetic event state. Calibration is copied into immutable
tuples. Live windows/source memory are policy/grace bounded; returned full replay
logs grow separately with the stream. Concurrent mutation is unsupported.
"""
mutable struct EventReplay
    snapshot::Snapshot
    token::Base.RefValue{Nothing}
    function EventReplay(calibration::D.CompletedCalibration;start::DateTime,subscribers,stream_id,protocol_id,grace_ms=0)
        D.whole(start) || throw(ArgumentError("start must be a whole UTC minute"))
        D.check_calibration(calibration,start,calibration.policy,calibration.mode)
        length(calibration.chronological_scores)==length(calibration.sorted_scores)==length(calibration.label_ready)>0 &&
            all(D.valid_number,calibration.chronological_scores) &&
            issorted(calibration.label_ready) && isequal(sort(calibration.chronological_scores),calibration.sorted_scores) ||
            throw(ArgumentError("calibration projection or chronology was mutated"))
        for label in (stream_id,protocol_id)
            label isa AbstractString && !isempty(strip(label)) && ncodeunits(label)<=512 || throw(ArgumentError("explicit bounded stream/protocol labels required"))
        end
        specs=Tuple(subscribers)
        !isempty(specs) && all(s->s isa SubscriberSpec,specs) || throw(ArgumentError("fixed subscriber specifications required"))
        length(unique(s.name for s in specs))==length(specs) || throw(ArgumentError("duplicate subscriber name"))
        grace=checked_grace(grace_ms);period=base_period(start,calibration.policy)
        retention=Base.Checked.checked_add(Base.Checked.checked_mul(Base.Checked.checked_add(period,1),MINUTE_MS),grace)
        capacity=Base.Checked.checked_add(period,Base.Checked.checked_add(cld(grace,MINUTE_MS),1))
        D.shifted(start,-retention)
        cal=(;chronological_scores=Tuple(calibration.chronological_scores),sorted_scores=Tuple(calibration.sorted_scores),
             policy=calibration.policy,mode=calibration.mode,point_model_sha256=calibration.point_model_sha256,purpose=calibration.purpose)
        context=(;stream_id=String(stream_id),protocol_id=String(protocol_id),policy=cal.policy,mode=cal.mode,
            point_model_sha256=cal.point_model_sha256,availability_kind=D.AVAILABILITY,
            measured_receipts_available=false,source_bound_provenance_verified=false)
        states=Tuple(begin
            slots=spec.kind===:phase ? (spec.phase_period===:safe ? safe_terminal_period(start,cal.policy;grace_ms=grace) : period) : 1
            scores=spec.rolling_capacity==0 ? cal.sorted_scores : Tuple(cal.chronological_scores[max(1,length(cal.chronological_scores)-spec.rolling_capacity+1):end])
            SubscriberState(spec,slots,ntuple(_->spec.nominal,slots),ntuple(_->0,slots),ntuple(_->0,slots),
                ntuple(_->0,slots),scores,1,0,0,0,0)
        end for spec in specs)
        new(Snapshot(cal,context,grace,retention,capacity,start,false,nothing,0,0,"","",(),(),states,0,0,0,0,0,true),Ref(nothing))
    end
end

struct TimeResult
    time::DateTime
    events::Tuple
    subscriber_events::Tuple
    issues::Tuple
    repeated::Bool
end
struct PreparedTime
    token::Base.RefValue{Nothing}
    base_revision::Int
    transaction_id::String
    fingerprint::String
    candidate::Snapshot
    result::TimeResult
end
