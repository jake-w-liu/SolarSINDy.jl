using Test, Dates, SHA, JSON3, SolarSINDy
include(joinpath(@__DIR__,"..","..","tools","actual_a_chain","ActualAChain.jl"))
const C=ActualAChain
const P=C.Inputs
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
length(ARGS)==3 || error("EXPORT_ROOT EXPORT_SHA256 AUDIT_SHA256")
const ORACLE=joinpath(@__DIR__,"frozen","initial_stage_cohorts_v1.json")
@test bytes2hex(sha256(read(ORACLE)))=="fa0c46a40e8d1dd56178cdb068aed9db390a74588cefd7c410fcf92a76ac7b34"
const O=JSON3.read(read(ORACLE),Dict{String,Any})
reference=C.load_reference(ARGS[1],ARGS[2];audit_completion_sha256=ARGS[3])
token=C.load_prefix(reference)

@testset "Audited generated raw prefix remains distinct from actual parsing" begin
    @test P.recheck(token)
    @test token.evidence_mode=="synthetic_only"
    @test token.parser_rows==900
    @test token.parser_source_bytes==17*900*8
    @test token.source_rows_from_audit>=900
    @test size(token.input.values)==(17,900)
    @test first(token.input.time)==DateTime(2019,1,1)
    @test last(token.input.time)==DateTime(2019,2,7,11)
    @test token.prefix_matrix_sha256==O["prefix_raw_sha256"]
    @test token.input.source_sha256==C.ReferenceImport.verified_metadata(reference).meta["source_sha256"]
    @test token.input.availability_kind==:assumed_completed_hour
end

@testset "Pre-code full joint mask retains warmup and fallback origins" begin
    raw=A.Features.censor_fills(token.input.values)
    start=first(token.input.time);boundary=start+Hour(500)
    expected=Int[];available=Int[];consecutive=0
    for i in 1:900
        speed,bz,pressure=raw[8,i],raw[6,i],raw[12,i]
        valid=all(isfinite,(speed,bz,pressure)) && speed>=0
        consecutive=valid ? consecutive+1 : 0
        issue=start+Hour(i)
        if i+6<=900 && all(isfinite,raw[15,[i,i+4,i+6]]) &&
                all(A.fit_eligible(issue,issue+Hour(h),issue+Hour(h),boundary) for h in (4,6))
            push!(expected,i)
            consecutive>=168 && push!(available,i)
        end
    end
    @test expected==O["joint_fit_origin_rows"]
    @test length(expected)==326
    @test available==O["model_available_joint_origin_rows"]
    @test length(available)==159
    @test length(expected)>length(available)
end

function forged(token;mode=token.evidence_mode,parser_bytes=token.parser_source_bytes,input=token.input)
    digest=P.input_digest(input)
    seal=P.fingerprint(token.reference,input,mode,token.source_rows_from_audit,token.parser_rows,
        parser_bytes,token.parsed_matrix_sha256,token.prefix_matrix_sha256,digest,token.own)
    P.PrefixInput(token.reference,input,mode,token.source_rows_from_audit,token.parser_rows,parser_bytes,
        token.parsed_matrix_sha256,token.prefix_matrix_sha256,digest,copy(token.own),seal)
end

@testset "Raw prefix mutation and resealed provenance corruption reject" begin
    saved=token.input.values[1,1]
    try
        token.input.values[1,1]=saved+1
        @test_throws ArgumentError P.recheck(token)
    finally;token.input.values[1,1]=saved;end
    @test P.recheck(token)
    saved=token.input.time[10]
    try
        token.input.time[10]=saved+Millisecond(1)
        @test_throws ArgumentError P.recheck(token)
    finally;token.input.time[10]=saved;end
    @test P.recheck(token)
    @test_throws ArgumentError P.recheck(forged(token;mode="exposed_development_reconstruction"))
    @test_throws ArgumentError P.recheck(forged(token;parser_bytes=-1))
    @test_throws ArgumentError P.recheck(forged(token;parser_bytes=token.parser_source_bytes+1))
    wrong=F.DstIntervalStream.HourlyInputs(token.input.time,token.input.values;source_sha256=repeat("b",64))
    @test_throws ArgumentError P.recheck(forged(token;input=wrong))
    key="tools/actual_a_chain/InitialAInputs.jl";saved=token.own[key]
    try
        token.own[key]=repeat("0",64)
        @test_throws ArgumentError P.recheck(token)
    finally;token.own[key]=saved;end
    @test P.recheck(token)
end
