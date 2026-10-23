"""
Completed-source clocks, prefix-only features and readiness-gated labels for an
explicitly assumed retrospective 30-minute ground dB/dt replay. Ground tags,
OMNI arrival times and actual receipts remain distinct. No historical receipt
is inferred and default monitoring does not invoke this research namespace.
"""
module CompletedMinuteReplay

using Dates
export ReplayPolicy, issue_timing, minute_phase, training_eligible

const EPOCH = DateTime(1970)
const MINUTE_MS = Int64(60_000)
const GROUND_SUPPORT_MS = Int64(45_000)
const TARGET_MINUTES = 30

"""
    ReplayPolicy(; ground_extra_ms=0, omni_extra_ms=0, label_extra_ms=0)

Assumed analysis delays in integer milliseconds, each between zero and one day.
The fixed ground filter extends 45 seconds after its center; OMNI arrival
intervals span one minute. Ground delay also applies to future target labels.
`label_extra_ms` adds processing after that ground source becomes available.
These parameters do not reconstruct historical publication or receipt times.
"""
struct ReplayPolicy
    ground_extra_ms::Int64
    omni_extra_ms::Int64
    # Additional label processing after the same ground source is available.
    label_extra_ms::Int64
    function ReplayPolicy(; ground_extra_ms=0, omni_extra_ms=0, label_extra_ms=0)
        values = (ground_extra_ms, omni_extra_ms, label_extra_ms)
        all(x -> x isa Integer && !(x isa Bool) && 0 <= x <= 86_400_000, values) ||
            throw(ArgumentError("extra delays must be integer milliseconds in [0,86400000]"))
        new(Int64.(values)...)
    end
end

function shift(t::DateTime, milliseconds::Int64)
    expected = Base.Checked.checked_add(Dates.value(t), milliseconds)
    result = t + Millisecond(milliseconds)
    Dates.value(result) == expected || throw(OverflowError("clock shift overflow"))
    result
end

floor_minute(t::DateTime) = shift(t, -mod(Dates.value(t), MINUTE_MS))
function ceil_minute(t::DateTime)
    remainder = mod(Dates.value(t), MINUTE_MS)
    shift(t, remainder==0 ? Int64(0) : MINUTE_MS-remainder)
end

function whole_minute(t::DateTime)
    mod(Dates.value(t), MINUTE_MS) == 0 ||
        throw(ArgumentError("scheduled issue must lie on a whole UTC minute"))
    t
end

"""
    issue_timing(issue::DateTime, policy=ReplayPolicy())

Return source anchors, assumed readiness, 30 future target centers and the
first whole-minute feedback. Reject clock overflow. UTC is the input convention.
Results retain `measured_receipts_available=false`. Filtered-vector targets do
not establish an unfiltered physical warning interval.
"""
function issue_timing(issue::DateTime, policy::ReplayPolicy=ReplayPolicy())
    whole_minute(issue)
    ground = floor_minute(shift(issue, -GROUND_SUPPORT_MS-policy.ground_extra_ms))
    omni = floor_minute(shift(issue, -MINUTE_MS-policy.omni_extra_ms))
    first_target = shift(issue, MINUTE_MS)
    last_target = shift(issue, TARGET_MINUTES*MINUTE_MS)
    label_ready = shift(last_target, GROUND_SUPPORT_MS+policy.ground_extra_ms+policy.label_extra_ms)
    feedback = ceil_minute(label_ready)
    period = cld(TARGET_MINUTES*MINUTE_MS+GROUND_SUPPORT_MS+policy.ground_extra_ms+policy.label_extra_ms, MINUTE_MS)
    (; issue, latest_ground_center=ground,
       latest_ground_assumed_ready=shift(ground, GROUND_SUPPORT_MS+policy.ground_extra_ms),
       latest_omni_arrival_start=omni,
       latest_omni_assumed_ready=shift(omni, MINUTE_MS+policy.omni_extra_ms),
       first_target_center=first_target, last_target_center=last_target,
       label_assumed_ready=label_ready, first_scheduled_feedback=feedback,
       sufficient_phase_period_minutes=period,
       legacy_30_minute_feedback_ready=last_target >= label_ready,
       availability_kind=:assumed_completed_source_intervals,
       measured_receipts_available=false)
end

"""
    minute_phase(issue::DateTime, policy=ReplayPolicy())

One-based absolute-minute phase for this bounded-delay analysis policy.
The zero-extra-delay period is 31 minutes. Unknown real receipts require
separate handling before this schedule can support an operational claim.
"""
function minute_phase(issue::DateTime, policy::ReplayPolicy=ReplayPolicy())
    whole_minute(issue)
    elapsed = Base.Checked.checked_sub(Dates.value(issue), Dates.value(EPOCH))
    period = issue_timing(issue, policy).sufficient_phase_period_minutes
    mod(div(elapsed, MINUTE_MS), period)+1
end

"""
    training_eligible(label_ready::DateTime, next_split_issue::DateTime)

Require label readiness strictly before the next temporal split. Equality is
excluded even when a declared within-stream update resolves at exact readiness.
"""
training_eligible(label_ready::DateTime, next_split_issue::DateTime) =
    label_ready < next_split_issue


using Statistics
export admit_features, admit_target, replay_case

rejected(reason) = (; accepted=false, reason, value=nothing)
accepted(value) = (; accepted=true, reason=:complete, value)
finite_scalar(x) = x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x))

function rate_at(mag::AbstractDict, center::DateTime)
    previous = center-Minute(1)
    haskey(mag,center) && haskey(mag,previous) || return rejected(:missing)
    before, after = mag[previous], mag[center]
    before isa Union{Tuple,AbstractVector} && after isa Union{Tuple,AbstractVector} || return rejected(:invalid)
    length(before)==length(after)==2 || return rejected(:invalid)
    all(finite_scalar,before) && all(finite_scalar,after) || return rejected(:invalid)
    rate = hypot(Float64(first(after))-Float64(first(before)), Float64(last(after))-Float64(last(before)))
    isfinite(rate) || return rejected(:invalid)
    accepted(rate)
end

"""
    admit_features(mag, omni, issue; policy=ReplayPolicy())

Admit 30 ground rates and one wind row using prefix source keys. Ground vectors
are `(X,Y)` in nT; wind rows are `(V,Bz,By,n)` in km/s, nT, nT and cm^-3.
Keys are UTC ground-center and OMNI arrival-start tags. Return `accepted`,
`reason` and `value`, preserving missing or invalid input as explicit rejections.
"""
function admit_features(mag::AbstractDict, omni::AbstractDict, issue::DateTime;
                        policy::ReplayPolicy=ReplayPolicy())
    timing = issue_timing(issue,policy)
    # Thirty rates require thirty-one vectors; reject before any wrapped key.
    shift(timing.latest_ground_center, -TARGET_MINUTES*MINUTE_MS)
    history = Float64[]
    for k in 29:-1:0
        rate = rate_at(mag,timing.latest_ground_center-Minute(k))
        rate.accepted || return rejected(rate.reason===:missing ? :missing_ground_history : :invalid_ground_history)
        push!(history,rate.value)
    end
    haskey(omni,timing.latest_omni_arrival_start) || return rejected(:missing_omni_history)
    wind = omni[timing.latest_omni_arrival_start]
    wind isa Union{Tuple,AbstractVector} && length(wind)==4 && all(finite_scalar,wind) ||
        return rejected(:invalid_omni_history)
    wind_values = Vector{Float64}(undef,4)
    for (index,value) in enumerate(wind)
        wind_values[index] = Float64(value)
    end
    velocity,bz,by,density = wind_values
    velocity>=0 && density>=0 || return rejected(:invalid_omni_history)
    south = max(-bz,0.0); coupling = velocity*south
    avg=mean(history); deviation=std(history)
    all(isfinite,(avg,deviation,coupling)) || return rejected(:invalid_feature_summary)
    row=(; datetime=issue,ground_anchor=timing.latest_ground_center,
          omni_anchor=timing.latest_omni_arrival_start,
          dbdt_now=last(history),dbdt_mean30=avg,dbdt_max30=maximum(history),dbdt_std30=deviation,
          V=velocity,Bz=bz,By=by,n=density,Bs=south,VBs=coupling,
          availability_kind=timing.availability_kind,measured_receipts_available=false)
    accepted((;row,timing))
end

"""
    admit_target(mag, issue, asof; policy=ReplayPolicy())

Read the 30-center future maximum only after assumed label readiness. An early
`asof` returns `:label_not_ready` before accessing future vectors. Actual source
receipt and data vintage remain outside this counterfactual analysis policy.
"""
function admit_target(mag::AbstractDict, issue::DateTime, asof::DateTime;
                      policy::ReplayPolicy=ReplayPolicy())
    timing = issue_timing(issue,policy)
    # Readiness is checked before accessing any future vector.
    timing.label_assumed_ready<=asof || return rejected(:label_not_ready)
    future = Float64[]
    for k in 1:30
        rate = rate_at(mag,issue+Minute(k))
        rate.accepted || return rejected(rate.reason===:missing ? :missing_ground_target : :invalid_ground_target)
        push!(future,rate.value)
    end
    accepted((;target_max_dbdt=maximum(future),label_ready=timing.label_assumed_ready,
               first_target_center=timing.first_target_center,last_target_center=timing.last_target_center))
end

"""
    replay_case(mag, omni, issue, asof; policy=ReplayPolicy())

Join independently admitted features and a readiness-gated retrospective label.
Return an explicit rejection for incomplete inputs. This constructs a research
case; it performs no model fitting, calibration or serving operation.
"""
function replay_case(mag::AbstractDict, omni::AbstractDict, issue::DateTime, asof::DateTime;
                     policy::ReplayPolicy=ReplayPolicy())
    features=admit_features(mag,omni,issue;policy)
    features.accepted || return features
    target=admit_target(mag,issue,asof;policy)
    target.accepted || return target
    accepted((;row=merge(features.value.row,target.value),timing=features.value.timing))
end

end
