# Exact fixed10k public emitted-row bit payloads were frozen before implementation.
using Dates,SHA,TOML,Test,InteractiveUtils
const BASE=abspath(ARGS[1]);const OUT=abspath(ARGS[2]);mkpath(OUT)
include(joinpath(BASE,"DstIntervalAdapter.jl"));include(joinpath(BASE,"DstIntervalStream.jl"))
using .DstIntervalAdapter,.DstIntervalStream
const K=DstIntervalAdapter.Kernel
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
    elseif x isa AbstractString
        write(io,UInt8('S'),htol(UInt64(ncodeunits(x))),codeunits(x))
    elseif x isa NamedTuple
        write(io,UInt8('N'),htol(UInt64(length(x))))
        for (name,value) in pairs(x);token(io,String(name));token(io,value);end
    elseif x isa Tuple
        write(io,UInt8('T'),htol(UInt64(length(x))))
        for value in x;token(io,value);end
    else
        error("unsupported diagnostic token: $(typeof(x))")
    end
end
function bit_payload(rows)
    io=IOBuffer()
    for row in rows;token(io,row);end
    take!(io)
end
const N=10_000;const T=DateTime(2018);const VALUES=ones(17,N)
VALUES[8,:].=400.;VALUES[6,:]=[-6+2sin(i/43) for i in 1:N]
VALUES[12,:].=2.;VALUES[15,:]=[-60+40sin(i/61) for i in 1:N]
const INPUT=HourlyInputs([T+Hour(i-1) for i in 1:N],VALUES;
    source_sha256=bytes2hex(sha256("owned10k-profile-synthetic-v1")))
const RECORDS=Dict{String,Any}[]
@testset "entire unchanged fixed10k L0/L1 public row bits" begin
    for delay in (0,1)
        rows=NamedTuple[]
        result=stream_basis(INPUT;delay_hours=delay,taus=(4.,24.),sigma=3.,chunk_hours=24,
            sink=chunk->append!(rows,chunk))
        @test result.row_count==N && length(rows)==N && result.max_buffered_rows==24
        payload=bit_payload(rows)
        reference=read(joinpath(abspath(ARGS[3]),"unmodified-L$delay-row-bits.bin"))
        @test payload==reference
        @test count(row->row.model_available,rows)==9833
        push!(RECORDS,Dict("delay"=>delay,"rows"=>N,"bytes"=>length(payload),"sha256"=>bytes2hex(sha256(payload))))
    end
    @test (@inferred K.interval_kernel(.25,3.,1.)) isa NamedTuple
    joint=@inferred K.interval_moments([1.,2.],[3. .1;.1 4.],[.25,1/24],[3.,3.],[-.5,-.25],1.)
    @test joint isa NamedTuple
    @test (@inferred K.condition_interval(joint,-2.,1.)) isa NamedTuple
    open(joinpath(OUT,"code_warntype.txt"),"w") do io
        code_warntype(io,K.interval_kernel,(Float64,Float64,Float64))
        code_warntype(io,K.interval_moments,(Vector{Float64},Matrix{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Float64))
        code_warntype(io,K.condition_interval,(typeof(joint),Float64,Float64))
    end
    @test Dict(DstIntervalStream.verify_source_identity())==BEFORE
end
open(joinpath(OUT,"receipt.toml"),"w") do io
    TOML.print(io,Dict("complete"=>true,"source_sha256"=>BEFORE,"julia_version"=>string(VERSION),
        "records"=>RECORDS,"driver_sha256"=>bytes2hex(sha256(read(@__FILE__)))),sorted=true)
end
