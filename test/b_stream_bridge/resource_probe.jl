module BridgeResourceProbe
using SolarSINDy, Dates, SHA, JSON3, Test
const D=SolarSINDy.DurationConditionedForcing
const C=SolarSINDy.DurationCurrentStateStream
const A=SolarSINDy.IntervalMeanForecaster
const SOURCE=bytes2hex(sha256("generated-only-b-bridge-resource-v1"))
function generated_wind(n)
    first=DateTime(1900)
    rows=Vector{D.WindHour}(undef,n)
    for i in 1:n
        hour=i-1+div(i-1,101);tag=first+Hour(hour)
        south=isodd(div(i,17));missing=mod(i,127)==0
        bz=missing ? NaN : south ? -1. : 1.
        q=missing ? NaN : south ? Float64(mod(i,7)) : 0.
        receipt=first+Hour(4*cld(hour+1,4))
        rows[i]=D.WindHour(tag,tag+Hour(1),receipt,bz,q)
    end
    rows
end
function consume(rows)
    tracker=Ref{Any}()
    result=C.with_current_state_tracker(rows;source_sha256=SOURCE) do (advance,_)
        tracker[]=getfield(advance,:tracker) # Private resource diagnostic only.
        latest=nothing
        for row in rows;latest=advance(row.completed);end
        latest=advance(last(rows).received)
        latest
    end
    result,C._statistics(tracker[])
end
consume(generated_wind(128)) # Compile the fixed-length public API shapes.
records=Any[]
@testset "Generated one-pass resource counters and cap" begin
    for n in (4096,32768,131072,500000)
        rows=generated_wind(n)
        measured=@timed consume(rows)
        result,stats=measured.value
        @test result.value==D.current_state(rows,last(rows).received)
        # Independent row byte construction, not production C._line.
        bit_token(x)=isnan(x) ? "missing" : string(reinterpret(UInt64,x))
        expected_bytes=sum(ncodeunits(string(Dates.value(r.tag),"|",Dates.value(r.completed),"|",Dates.value(r.received),"|",bit_token(r.bz),"|",bit_token(r.q),"\n")) for r in rows)
        @test stats.source_rows==stats.consumed_rows==n
        @test stats.admission_hash_bytes==stats.eligible_hash_bytes==expected_bytes
        @test stats.advance_calls==stats.digest_snapshots==n+1
        @test stats.source_checks==2
        @test Base.summarysize(result.checkpoint)<8192
        push!(records,(;scope=:archival_tracker,n,issue_count=n+1,elapsed_seconds=measured.time,
            allocated_bytes=measured.bytes,gc_seconds=measured.gctime,source_payload_bytes=40n,
            checkpoint_bytes=Base.summarysize(result.checkpoint),stats,
            concurrency_qualified=true,quiet_benchmark=false))
    end
end
function generated_a(n)
    raw=ones(17,n);raw[8,:].=400.;raw[6,:].=-2.;raw[12,:].=2.;raw[15,:].=-30.
    for i in 257:257:n;raw[6,i]=NaN;end
    for i in 499:499:n;raw[15,i]=NaN;end
    A.HourlyInputs([DateTime(2000)+Hour(i-1) for i in 1:n],raw;source_sha256=SOURCE)
end
@testset "Generated A full physical grid/chunk resource bounds" begin
    input=generated_a(4096)
    for tau in A.DstIntervalAdapter.TAUS,sigma in A.DstIntervalAdapter.SIGMAS,delay in (0,1),chunk in (1,7,24,4096)
        seen=Ref(0);issues=Ref(0)
        measured=@timed A.stream_basis(input;delay_hours=delay,taus=tau,sigma,chunk_hours=chunk,
            sink=rows->(seen[]+=length(rows)),issue_sink=_->(issues[]+=1))
        result=measured.value
        @test seen[]==issues[]==result.row_count==4096
        @test result.max_buffered_rows<=chunk
        @test Base.summarysize(result.checkpoint)<4096
        push!(records,(;scope=:detached_A_stream,n=4096,taus=tau,sigma,delay,chunk,
            elapsed_seconds=measured.time,allocated_bytes=measured.bytes,gc_seconds=measured.gctime,
            cleaned_matrix_payload_bytes=17*8*4096,max_buffered_rows=result.max_buffered_rows,
            checkpoint_bytes=Base.summarysize(result.checkpoint),retained_rows=0,
            concurrency_qualified=true,quiet_benchmark=false))
    end
end
path=joinpath(only(ARGS),"resource_results.json");ispath(path)&&error("refusing to replace resource result")
write(path,JSON3.write((;schema="b_stream_bridge_generated_resources_v1",complete=true,
    julia_version=string(VERSION),actual_data=false,quiet_benchmark=false,
    generated_tracker_cap=500000,A_grid_probe_hours=4096,records)))
println("B_STREAM_BRIDGE_RESOURCES_COMPLETE records=",length(records))
end
