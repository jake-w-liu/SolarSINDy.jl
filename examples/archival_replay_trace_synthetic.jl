module ArchivalReplayTraceSynthetic
using Dates,SHA,JSON3,SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_archival_static.jl"))
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_archival_replay.jl"))
const AR=SourceBoundArchivalReplay
"""Check the independent opaque trace frame, without archival access or replay."""
function run_example()
    oracle=JSON3.read(read(joinpath(@__DIR__,"..","test","publisher_up_values","archival_replay_trace_oracle.json"),String),Dict{String,Any})
    @assert oracle["fixture_only"]===true && oracle["actual_source_authority_granted"]===false
    rows=[(;id=1,issue=DateTime(2020),upper=Inf,scale=2.,available=true,reason=:issued),
          (;id=2,issue=DateTime(2020)+Minute(1),upper=nothing,scale=-0.,available=false,reason=:prior_label_pending)]
    context=AR._trace_start(repeat("0",64));AR._trace_section!(context,"fixture.rows",rows)
    @assert bytes2hex(SHA.digest!(context))==oracle["trace_sha256"]=="0117e921a7e64b3a4a9c0f2fbf54ea6b2262773ff91a20a79b53560a173953f5"
    (;fixture_only=true,numeric_trace_only=true,actual_source_access=false,actual_fits=0,scientific_qualification=false)
end
end
