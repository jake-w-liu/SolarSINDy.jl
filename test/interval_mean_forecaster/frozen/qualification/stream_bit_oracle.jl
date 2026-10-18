# Frozen before optimization: unchanged public streams and checkpoints by bits.
using Dates, SHA, TOML, Test
length(ARGS)==3 || error("source-root output-root record-or-reference-root")
const SOURCE=abspath(ARGS[1]); const OUT=abspath(ARGS[2]); mkpath(OUT)
const RECORD=ARGS[3]=="record"; const REFERENCE=RECORD ? OUT : abspath(ARGS[3])
include(joinpath(SOURCE,"DstIntervalAdapter.jl"))
include(joinpath(SOURCE,"DstIntervalStream.jl"))
using .DstIntervalAdapter,.DstIntervalStream
const BEFORE=Dict(DstIntervalStream.verify_source_identity())
function token(io,x)
    if x isa Float64
        write(io,UInt8('F'),htol(reinterpret(UInt64,x)))
    elseif x isa DateTime
        write(io,UInt8('D'),htol(Int64(Dates.value(x))))
    elseif x isa Bool
        write(io,UInt8('B'),UInt8(x))
    elseif x isa Integer
        write(io,UInt8('I'),htol(Int64(x)))
    elseif x isa AbstractString || x isa Symbol
        value=String(x); write(io,UInt8('S'),htol(UInt64(ncodeunits(value))),codeunits(value))
    elseif x===nothing
        write(io,UInt8('0'))
    elseif x isa NamedTuple
        write(io,UInt8('N'),htol(UInt64(length(x))))
        for (name,value) in pairs(x);token(io,String(name));token(io,value);end
    elseif x isa Tuple || x isa AbstractArray
        write(io,UInt8('T'),htol(UInt64(length(x))))
        x isa AbstractArray && token(io,size(x))
        for value in x;token(io,value);end
    elseif x isa AbstractDict
        write(io,UInt8('M'),htol(UInt64(length(x))))
        for name in sort(collect(keys(x));by=string);token(io,name);token(io,x[name]);end
    elseif isstructtype(typeof(x))
        token(io,String(nameof(typeof(x))))
        for name in fieldnames(typeof(x));token(io,name);token(io,getfield(x,name));end
    else
        error("unsupported bit token $(typeof(x))")
    end
end
payload(x)=begin io=IOBuffer();token(io,x);take!(io) end
function fixture(;gap=false)
    n=600; values=ones(17,n); values[8,:].=400.
    values[6,:]=[-5+3sin(i/43) for i in 1:n]
    values[12,:]=[2+.5cos(i/17) for i in 1:n]
    values[15,:]=[-60+40sin(i/61) for i in 1:n]
    if gap;values[6,300]=NaN;values[15,500]=NaN;values[12,550]=-1.;end
    HourlyInputs([DateTime(2018)+Hour(i-1) for i in 1:n],values;
        source_sha256=bytes2hex(sha256("frozen-cache-stream-matrix-v1:$gap")))
end
function stream(input,l,taus,sigma;chunk=24,checkpoint=nothing,stop=length(input.time))
    rows=NamedTuple[]
    result=stream_basis(input;delay_hours=l,taus,sigma,chunk_hours=chunk,checkpoint,
        stop_after_index=stop,sink=part->append!(rows,part))
    rows,result
end
const RECORDS=Dict{String,Any}[]
function bind(name,value)
    bytes=payload(value);path=joinpath(REFERENCE,name*".bin")
    if RECORD
        ispath(path) && error("preserve frozen bit oracle")
        write(path,bytes)
    else
        @test bytes==read(path)
    end
    push!(RECORDS,Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>bytes2hex(sha256(bytes))))
end
function exception(f)
    try f(); return ("SUCCESS","") catch e; return (string(nameof(typeof(e))),sprint(showerror,e)) end
end
@testset "nine recipes both latencies public bits and boundary behavior" begin
    input=fixture(); gaps=fixture(;gap=true)
    for (ti,taus) in enumerate(DstIntervalAdapter.TAUS), (si,sigma) in enumerate(DstIntervalAdapter.SIGMAS), l in (0,1)
        tag="tau$(ti)-sigma$(si)-L$(l)"
        rows,result=stream(input,l,taus,sigma;chunk=1)
        @test length(rows)==600 && rows[167].reason=="warmup" && rows[168].model_available
        bind(tag*"-full",(rows,result))
        for chunk in (7,24,511,4096)
            other,summary=stream(input,l,taus,sigma;chunk)
            @test payload(other)==payload(rows)
            @test payload(summary.checkpoint)==payload(result.checkpoint)
            bind(tag*"-chunk$chunk",(other,summary))
        end
        early,partial=stream(input,l,taus,sigma;stop=300)
        tail,resumed=stream(input,l,taus,sigma;checkpoint=partial.checkpoint)
        @test payload(vcat(early,tail))==payload(rows)
        @test payload(resumed.checkpoint)==payload(result.checkpoint)
        bind(tag*"-partial",(early,partial));bind(tag*"-resumed",(tail,resumed))
        none,noop=stream(input,l,taus,sigma;checkpoint=partial.checkpoint,stop=300)
        @test isempty(none) && noop.checkpoint===partial.checkpoint
        bind(tag*"-noop",(none,noop))
        for (name,change) in (("past",200),("future",301))
            values=copy(input.values);values[:,change:end].*=2
            revised=HourlyInputs(input.time,values;source_sha256=input.source_sha256)
            if name=="past"
                caught=exception(()->stream(revised,l,taus,sigma;checkpoint=partial.checkpoint))
                @test caught[1]=="ArgumentError";bind(tag*"-revision-rejected",caught)
            else
                future,future_result=stream(revised,l,taus,sigma)
                @test payload([r.offset for r in future[1:300]])==payload([r.offset for r in rows[1:300]])
                @test payload([r.coefficients for r in future[1:300]])==payload([r.coefficients for r in rows[1:300]])
                bind(tag*"-future-revision",(future,future_result))
            end
        end
        gaprows,gapresult=stream(gaps,l,taus,sigma)
        @test gaprows[300].reason=="known_driver_missing" && gaprows[301].consecutive_valid==1
        @test gaprows[468].model_available && gaprows[500+l].reason=="anchor_missing"
        @test gapresult.negative_pressure_hours==1
        bind(tag*"-gaps",(gaprows,gapresult))
        if l==1
            values=copy(input.values);values[15,300]=-150.
            late=HourlyInputs(input.time,values;source_sha256=input.source_sha256)
            late_rows,late_summary=stream(late,l,taus,sigma;checkpoint=partial.checkpoint)
            @test !isequal(late_rows[1].offset,rows[301].offset)
            bind(tag*"-late-receipt",(late_rows,late_summary))
        end
    end
    @test Dict(DstIntervalStream.verify_source_identity())==BEFORE
end
open(joinpath(OUT,"receipt.toml"),"w") do io
    TOML.print(io,Dict("complete"=>true,"mode"=>RECORD ? "frozen-baseline" : "exact-bit-comparison",
        "julia_version"=>string(VERSION),"source_sha256"=>BEFORE,"records"=>RECORDS,
        "driver_sha256"=>bytes2hex(sha256(read(@__FILE__)))),sorted=true)
end
println("STREAM_BIT_ORACLE_COMPLETE records=",length(RECORDS))
