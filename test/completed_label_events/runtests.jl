module CompletedLabelEventTests
using Test, Dates, JSON3, SHA
using SolarSINDy
using SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const CMR=D.CMR
const I=DateTime(2024,1,1)
const STREAM="synthetic_station/ground11/policy0"
const PROTOCOL="synthetic-completed-events-v1"
const ORACLES=JSON3.read(read(joinpath(@__DIR__,"frozen","independent_oracles_v1.json")),Dict{String,Any})
const FIXTURES=Dict(row["id"]=>row for row in ORACLES["fixtures"])
offset(ms)=I+Millisecond(ms)
function calibration(;policy=CMR.ReplayPolicy(),start=I)
    D.CompletedCalibration(zeros(9),fill(start-Millisecond(1),9);policy,
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
end
function request(time=I;center=0.0,scale=1.0,policy=CMR.ReplayPolicy())
    mag=Dict(time+Minute(k)=>(0.0,0.0) for k in -40:-1)
    IssueRequest(D.prepare_issue(mag,time;policy),center,scale)
end
function specs(;phase=:policy,rolling=0)
    (SubscriberSpec(:static,:static),SubscriberSpec(:phase,:phase;phase_period=phase),
     SubscriberSpec(:fifo,:fifo),SubscriberSpec(:early,:early;rolling_capacity=rolling),
     SubscriberSpec(:maturity,:maturity;rolling_capacity=rolling))
end
function state(;grace=0,subscribers=specs(),policy=CMR.ReplayPolicy(),start=I)
    EventReplay(calibration(;policy,start);start,subscribers,grace_ms=grace,stream_id=STREAM,protocol_id=PROTOCOL)
end
function transact!(owner,time;rows=(),issues=(),tx=string(Dates.value(time)))
    commit_time!(owner,prepare_time(owner,time;observations=rows,issues,transaction_id=tx))
end
function rows(values)
    [(;center=offset(row["center_offset_ms"]),rate=haskey(row,"rate_bits_hex") ?
        reinterpret(Float64,parse(UInt64,row["rate_bits_hex"];base=16)) : row["rate"]) for row in values]
end
function rate_rows(first,last;positive_at=0,value=2.0)
    [(;center=I+Minute(k),rate=k==positive_at ? value : 0.0) for k in first:last]
end
function subscriber(owner,name)
    only(filter(s->s.spec.name===name,owner.snapshot.subscribers))
end
function literal_alpha(value)
    pieces=split(value,"/");Float64(parse(BigInt,pieces[1])//parse(BigInt,pieces[2]))
end
include("test_events.jl")
include("test_errors.jl")
include("test_replay.jl")
include("test_context.jl")
include("test_axes.jl")
end
