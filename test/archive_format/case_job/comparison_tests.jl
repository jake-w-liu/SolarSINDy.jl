module ArchiveCatalogCaseComparisonTests
using Dates,JSON3,Test,SolarSINDy
const A=SolarSINDy.ArchiveCatalogCaseComparison
const B=JSON3.read(read(joinpath(pkgdir(SolarSINDy),"test","archive_format","cases","FROZEN_BINARY_CONTRACT_AND_ORACLES_V1.json")),Any)
const START=DateTime(2020)+Minute(32)
const FEATURE=String(hex2bytes(B["mixed8_issue32_feature_hex"]))
function word!(bytes,offset,value::UInt64)
    for index in 0:7;bytes[offset+index+1]=UInt8((value>>(8index))&0xff);end
    bytes
end
function number_at(bytes,offset)
    word=UInt64(0)
    for index in 0:7;word|=UInt64(codeunit(bytes,offset+index+1))<<(8index);end
    reinterpret(Float64,word)
end
function changed(bytes,offset,value::Float64)
    copy=Vector{UInt8}(codeunits(bytes));word!(copy,offset,reinterpret(UInt64,value));String(copy)
end
function byte_changed(bytes,offset,value::UInt8)
    copy=Vector{UInt8}(codeunits(bytes));copy[offset+1]=value;String(copy)
end
function label()
    bytes=zeros(UInt8,64)
    # The literal independent feature frame supplies the known issue/readiness
    # bytes. The target and support below are a separate generated binary oracle.
    bytes[1:16]=codeunits(FEATURE)[1:16];bytes[17]=1
    word!(bytes,24,reinterpret(UInt64,5.0));bytes[33:64].=0xa5
    String(bytes)
end
function failure(action,expected)
    message=nothing
    try;action();catch error;error isa ArgumentError || rethrow();message=error.msg;end
    @test message==expected
end
@testset "Independent encoded case agreement and corruption controls" begin
    l=label()
    compare(f=FEATURE,target=l;left=FEATURE,left_target=l,kwargs...)=A.compare_case_bytes(left,left_target,f,target;mode=:mixed8,start=START,stop=START+Minute(1),kwargs...)
    good=compare()
    @test good.complete && good.scheduled==good.features_accepted==good.labels_accepted==good.supervised==1
    @test good.features_rejected==good.labels_rejected==good.changed_feature_numbers==good.changed_label_numbers==0
    @test good.maximum_fraction_of_bound==0
    @test !good.source_provenance_qualified && !good.actual_source_authority_granted && !good.actual_fit_performed
    @test good.policy_sha256==A.policy_digest() && occursin(r"^[0-9a-f]{64}$",good.policy_sha256)
    @test good.policy_sha256=="666f9e8112c4ee7747e2121e4cb3b5ed38cf081484468e4cc2399bea271b8c94"
    @test A.comparison_policy().unit_roundoff_bits=="3ca0000000000000"
    @test A.comparison_policy().admitted_ground_component_limit===90000
    @test A.comparison_policy().direct_wind_fields_bit_exact
    @test A.comparison_policy() isa NamedTuple
    rounded=changed(FEATURE,48,nextfloat(number_at(FEATURE,48)))
    result=compare(rounded)
    @test result.changed_feature_numbers==1 && result.changed_label_numbers==0 && 0<result.maximum_fraction_of_bound<=1
    result=compare(FEATURE,changed(l,24,nextfloat(5.0)))
    @test result.changed_label_numbers==1 && 0<result.maximum_fraction_of_bound<=1
    failure(()->compare(changed(FEATURE,48,number_at(FEATURE,48)+0.01)),"case comparison numerical disagreement")
    failure(()->compare(FEATURE,changed(l,24,5.01)),"case comparison numerical disagreement")
    failure(()->compare(changed(FEATURE,64,NaN)),"case comparison invalid accepted summary")
    failure(()->compare(changed(FEATURE,32,0.5)),"case comparison invalid accepted summary")
    failure(()->compare(FEATURE,changed(l,24,Inf)),"case comparison invalid target")
    for offset in (0,8)
        bytes=Vector{UInt8}(codeunits(FEATURE));bytes[offset+1]⊻=0x01
        failure(()->compare(String(bytes)),"case comparison issue/readiness differs")
        bytes=Vector{UInt8}(codeunits(l));bytes[offset+1]⊻=0x01
        failure(()->compare(FEATURE,String(bytes)),"case comparison issue/readiness differs")
    end
    for (offset,value) in ((16,0x02),(17,0x06))
        failure(()->compare(byte_changed(FEATURE,offset,value)),"case comparison status/reason differs")
    end
    failure(()->compare(FEATURE,byte_changed(l,17,0x05)),"case comparison status/reason differs")
    for offset in (18,23)
        bad=byte_changed(FEATURE,offset,0x01)
        failure(()->compare(bad;left=bad),"case comparison padding differs")
        bad=byte_changed(l,offset,0x01)
        failure(()->compare(FEATURE,bad;left_target=bad),"case comparison padding differs")
    end
    failure(()->compare(byte_changed(FEATURE,128,codeunit(FEATURE,129)⊻0x01)),"case comparison source support differs")
    failure(()->compare(FEATURE,byte_changed(l,32,0xa4)),"case comparison source support differs")
    for offset in (104,127)
        failure(()->compare(byte_changed(FEATURE,offset,0x01)),"case comparison unused feature columns differ")
    end
    wind=changed(FEATURE,72,nextfloat(number_at(FEATURE,72)))
    wind=changed(wind,96,number_at(wind,72)*number_at(wind,88))
    failure(()->compare(wind),"case comparison direct wind bits differ")
    excluded=Vector{UInt8}(codeunits(FEATURE));excluded[17]=0;excluded[18]=1;excluded[25:128].=0
    missing=String(copy(excluded));result=compare(missing;left=missing)
    @test result.features_accepted==0 && result.features_rejected==1 && result.labels_accepted==1 && result.supervised==0
    other=byte_changed(missing,17,0x02)
    failure(()->compare(other;left=missing),"case comparison exclusion partition differs")
    excluded[25]=1;invalid=String(excluded)
    failure(()->compare(invalid;left=invalid),"case comparison rejected feature payload differs")
    excluded_label=Vector{UInt8}(codeunits(l));excluded_label[17]=0;excluded_label[18]=3;excluded_label[25:32].=0
    missing_label=String(copy(excluded_label));result=compare(FEATURE,missing_label;left_target=missing_label)
    @test result.features_accepted==1 && result.labels_accepted==0 && result.labels_rejected==1 && result.supervised==0
    excluded_label[25]=1;invalid_label=String(excluded_label)
    failure(()->compare(FEATURE,invalid_label;left_target=invalid_label),"case comparison rejected target differs")
    failure(()->A.compare_case_bytes(FEATURE,l,FEATURE,l;mode="mixed8",start=START,stop=START+Minute(1)),"case comparison layout differs")
    failure(()->A.compare_case_bytes(FEATURE,l,FEATURE,l;mode=:mixed8,start=Date(2020),stop=START+Minute(1)),"case comparison whole-minute domain required")
    failure(()->A.compare_case_bytes(FEATURE,l,FEATURE,l;mode=:mixed8,start=START,stop=START),"case comparison domain bound")
    failure(()->compare(String(copy(codeunits(FEATURE)[1:end-1]))),"case comparison payload width differs")
    failure(()->compare(Vector{UInt8}(codeunits(FEATURE))),"case comparison bounded binary string required")
    failure(()->A.compare_case_bytes(FEATURE*FEATURE,l*l,FEATURE*FEATURE,l*l;mode=:mixed8,start=START,stop=START+Minute(2)),"case comparison issue/readiness differs")
end
end
