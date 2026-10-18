using Dates, Test, SHA
include("DstIntervalAdapter.jl")
include("DstIntervalStream.jl")
using .DstIntervalStream
function fixture(n;gap=nothing,dst_gap=nothing)
    times=[DateTime(2018)+Hour(i-1) for i in 1:n]
    values=ones(17,n)
    values[8,:].=400.
    values[6,:]=[-5.0+3sin(i/43) for i in 1:n]
    values[12,:]=[2.0+.5cos(i/17) for i in 1:n]
    values[15,:]=[-60.0+40sin(i/61) for i in 1:n]
    gap===nothing || (values[6,gap]=NaN)
    dst_gap===nothing || (values[15,dst_gap]=NaN)
    HourlyInputs(times,values;source_sha256=bytes2hex(sha256("synthetic-stream:$n:$gap:$dst_gap")))
end
function run(input,l;chunk=24,checkpoint=nothing,stop_after_index=length(input.time))
    rows=NamedTuple[]
    result=stream_basis(input;delay_hours=l,taus=(4.,24.),sigma=3.,chunk_hours=chunk,
        sink=chunk->append!(rows,chunk),checkpoint,stop_after_index)
    rows,result
end
samepoint(a,b)=isequal((a.offset,a.coefficients,a.variance,a.reason),(b.offset,b.coefficients,b.variance,b.reason))
@testset "completeprefix andchunkedexacttargets" begin
    input=fixture(600)
    for l in (0,1)
        rows,summary=run(input,l;chunk=1)
        @test length(rows)==summary.row_count==600
        @test summary.max_buffered_rows==1
        @test rows[1].issue==first(input.time)+Hour(1)
        @test rows[end].issue==last(input.time)+Hour(1)
        @test rows[167].reason=="warmup"
        @test rows[168].model_available
        @test rows[300].anchor==input.values[15,300-l]
        @test rows[300].driver_tag==input.time[300]
        @test rows[300].driver_receipt==rows[300].issue
        @test rows[300].anchor_complete==rows[300].issue-Hour(l)
        @test rows[300].anchor_receipt==rows[300].issue
        @test (rows[300].taus,rows[300].sigma,rows[300].source_sha256)==((4.,24.),3.,input.source_sha256)
        @test rows[300].target==(input.values[15,304],input.values[15,306])
        @test rows[300].target_tag==(input.time[304],input.time[306])
        for chunk in (7,24,511,4096)
            other,result=run(input,l;chunk)
            @test all(samepoint.(rows,other))
            @test result.checkpoint.prefix_sha256==summary.checkpoint.prefix_sha256
            @test result.max_buffered_rows<=chunk
        end
        prefix=HourlyInputs(input.time[1:300],input.values[:,1:300];source_sha256=input.source_sha256)
        short,_=run(prefix,l)
        early,cp=run(input,l;stop_after_index=300)
        @test all(samepoint.(rows[1:300],early))
        @test all(samepoint.(short,early))
        resumed,resume=run(input,l;checkpoint=cp.checkpoint)
        @test length(resumed)==300
        @test all(samepoint.(rows[301:end],resumed))
        @test resume.checkpoint.prefix_sha256==summary.checkpoint.prefix_sha256
        none,nochange=run(input,l;checkpoint=cp.checkpoint,stop_after_index=300)
        @test isempty(none) && nochange.checkpoint===cp.checkpoint
        @test_throws ArgumentError run(input,l;checkpoint=cp.checkpoint,stop_after_index=299)
        @test_throws ArgumentError run(input,l;stop_after_index=601)
        @test_throws ArgumentError run(input,l;stop_after_index=true)
        damaged=copy(input.values);damaged[6,200]=-20.
        changed=HourlyInputs(input.time,damaged;source_sha256=input.source_sha256)
        @test_throws ArgumentError run(changed,l;checkpoint=cp.checkpoint)
        future=copy(input.values);future[:,301:end].*=2
        changed_future=HourlyInputs(input.time,future;source_sha256=input.source_sha256)
        future_rows,_=run(changed_future,l)
        @test all(samepoint.(rows[1:300],future_rows[1:300]))
        if l==1
            revealed_later=copy(input.values);revealed_later[15,300]=-150.
            revised=HourlyInputs(input.time,revealed_later;source_sha256=input.source_sha256)
            later,_=run(revised,l;checkpoint=cp.checkpoint)
            @test !samepoint(rows[301],later[1])
        end
    end
end
@testset "missingdriver reset andmissingDstskip" begin
    for l in (0,1)
        input=fixture(600;gap=300,dst_gap=500)
        rows,_=run(input,l)
        @test rows[300].reason=="known_driver_missing"
        @test rows[301].consecutive_valid==1
        @test rows[467].reason=="warmup"
        @test rows[468].model_available
        @test rows[500].consecutive_valid==200
        @test rows[500+l].reason=="anchor_missing"
        @test rows[501+l].model_available
    end
    @test_throws ArgumentError HourlyInputs([DateTime(2018),DateTime(2018)+Hour(2)],ones(17,2))
    input=fixture(200)
    values=copy(input.values);values[12,190]=-1.
    negative=HourlyInputs(input.time,values;source_sha256=input.source_sha256)
    rows,result=run(negative,0)
    @test result.negative_pressure_hours==1
    @test rows[190].model_available
    @test rows[190].latest_pressure==-1.
end
println("STREAM_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalStream.jl")))))
