"""
    CompletedSourceDBDT

Research adapters and synthetic replay controllers for completed-source dB/dt
windows. Uses declared retrospective availability, never historical receipt proof.
The internal hash-pinned legacy reference supplies feature/model primitives;
legacy raw-tag case construction is not part of this namespace's public API.
Actual source-bound refitting and common completion-event comparison remain
separate qualification requirements.
"""
module CompletedSourceDBDT
using Dates, Statistics, SHA
import ..CompletedMinuteReplay
const CMR=CompletedMinuteReplay
const REFERENCE=joinpath(@__DIR__,"reference","dbdt_causal_forecaster.jl")
open(io->bytes2hex(sha256(io)),REFERENCE)=="b2c46d28551443dc88c031b0e9cd3bf6b0e24bbba05f81d515e355afdb424922" ||
    error("locked feature reference differs")
include(REFERENCE)
const CF=DBDTCausalForecaster
const MINUTE_MS=Int64(60_000)
const EPOCH=DateTime(1970)
const POINT_SCHEMA="dbdt_completed_source_point_v1"
const CALIBRATION_SCHEMA="dbdt_completed_source_calibration_v1"
const AVAILABILITY=:assumed_completed_source_intervals
export PreparedIssue, prepare_issue, prepare_schedule, label_audit, case_audit,
       split_eligible, bind_predictions, CompletedCalibration, static_bounds,
       replay_phase, replay_fifo, CompletedMaximumController, issue!, receive!,
       advance!, certificate, materialize_receipts, replay_receipts

function shifted(time::DateTime,milliseconds::Integer)
    delta=Int64(milliseconds)
    expected=Base.Checked.checked_add(Dates.value(time),delta)
    answer=time+Millisecond(delta)
    Dates.value(answer)==expected || throw(OverflowError("replay clock shift"))
    answer
end
whole(time)=mod(Dates.value(time),MINUTE_MS)==0
function ceil_minute(time)
    remainder=mod(Dates.value(time),MINUTE_MS)
    shifted(time,remainder==0 ? 0 : MINUTE_MS-remainder)
end
function valid_number(value)
    value isa Real && !(value isa Bool) && isfinite(value) || return false
    converted=try
        Float64(value)
    catch
        return false
    end
    isfinite(converted) && (value==0 || converted!=0)
end
function nominal_value(value)
    valid_number(value) || throw(ArgumentError("invalid nominal miscoverage"))
    converted=Float64(value)
    0<converted<1 || throw(ArgumentError("nominal miscoverage is not representable in (0,1)"))
    converted
end
function gain_value(value)
    valid_number(value) || throw(ArgumentError("invalid controller gain"))
    converted=Float64(value)
    0<converted<=1 || throw(ArgumentError("gain is not representable in (0,1]"))
    converted
end
positive_number(value)=valid_number(value) && Float64(value)>0
policy_equal(a,b)=a.ground_extra_ms==b.ground_extra_ms && a.omni_extra_ms==b.omni_extra_ms && a.label_extra_ms==b.label_extra_ms
check_mode(mode)=mode in (:ground11,:mixed8) || throw(ArgumentError("mode must be ground11 or mixed8"))

include("adapter.jl")
include("controllers.jl")
include("maximum.jl")
include("label_events/CompletedLabelEvents.jl")
end
