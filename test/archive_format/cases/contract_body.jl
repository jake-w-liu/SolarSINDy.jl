using Dates, JSON3, Test, SHA, SolarSINDy
using SolarSINDy.ArchiveCatalogCases
const FIXTURE_PATH = joinpath(@__DIR__,"..","windows","fixtures","FROZEN_INPUTS_AND_ORACLES_V1.json")
const FIX = JSON3.read(read(FIXTURE_PATH),Any)
const ORACLE = JSON3.read(read(joinpath(@__DIR__,"FROZEN_BINARY_CONTRACT_AND_ORACLES_V1.json")),Any)
const EXTRA = JSON3.read(read(joinpath(@__DIR__,"ADDITIONAL_LITERAL_ORACLES_V1.json")),Any)
const P = SolarSINDy.ArchiveFormatParsers
const A = SolarSINDy.ArchiveFormatAudit
const C = SolarSINDy.ArchiveSourceCatalog
const I = SolarSINDy.ArchiveCatalogAudit
const M = SolarSINDy.CompletedMinuteReplay
const START = DateTime(2020)
const ISSUE = START+Minute(32)

function catalog(text,kind)
    kw=(;station="CMO",product=kind===:ground ? "quasi-definitive" : "OMNI_HRO_1MIN",start=START,stop=START+Minute(90))
    if ROLE=="primary"
        body=kind===:ground ? P.parse_ground(text;kw...) : P.parse_wind(text;kw...)
        C.build_catalog([body])
    else
        body=kind===:ground ? A.audit_ground(text;kw...) : A.audit_wind(text;kw...)
        I.audit_catalog([body])
    end
end
ROLE in ("primary","audit") || error("unsupported role")
const G = catalog(FIX["ground"],:ground)
const WIND = catalog(FIX["wind"],:wind)
const READY = ISSUE+Minute(30)+Second(45)
bits(value::Float64)=string(reinterpret(UInt64,value);base=16,pad=16)
const MIXED = seal_catalog_features(G,[ISSUE];mode=:mixed8,wind=WIND)
const GROUND = seal_catalog_features(G,[ISSUE];wind=:forbidden)

@testset "Independent literal binary record and numerical oracles" begin
    @test ncodeunits(MIXED.payload)==160
    @test bytes2hex(codeunits(MIXED.payload))==ORACLE["mixed8_issue32_feature_hex"]
    @test MIXED.payload_sha256==ORACLE["mixed8_issue32_feature_sha256"]
    @test feature_row(MIXED,1).support_sha256==ORACLE["mixed8_issue32_support_sha256"]
    @test feature_row(GROUND,1).support_sha256==ORACLE["ground11_issue32_support_sha256"]
    m=feature_row(MIXED,1);g=feature_row(GROUND,1)
    @test m.accepted && m.reason===:complete && m.issue==ISSUE && m.label_ready==READY
    @test bits.(m.features)==Tuple(FIX["mixed8_expected_bits"])
    @test m.raw_now==g.raw_now==5.0
    @test m.scale==g.scale==1+FIX["ground11_expected"][3]
    for (actual,expected) in zip(g.features,FIX["ground11_expected"])
        @test isapprox(actual,expected;atol=1e-13,rtol=0)
    end
    @test isimmutable(MIXED) && isimmutable(m) && isimmutable(m.features)
    @test !MIXED.source_pins.measured_receipts_available && !MIXED.source_pins.source_bound_provenance_verified
    @test GROUND.source_pins.wind_selected_sha256===nothing
    @test GROUND.source_pins.wind_raw_sha256===nothing
end

@testset "Immutable schedule membership and complete exclusion accounting" begin
    issues=collect(START:Minute(1):START+Minute(90))
    schedule=seal_catalog_features(G,issues;mode=:mixed8,wind=WIND)
    saved=schedule.schedule_sha256;issues[1]=ISSUE
    @test length(schedule)==91 && feature_row(schedule,1).issue==START
    @test schedule.schedule_sha256==saved
    @test !feature_row(schedule,1).accepted
    @test feature_row(schedule,1).features===nothing
    @test feature_row(schedule,1).raw_now===nothing
    @test feature_row(schedule,1).scale===nothing
    @test feature_row(schedule,1).reason===:missing_ground_history
    @test feature_row(schedule,32).accepted
    labels=audit_catalog_labels(schedule,G;asof=START+Minute(121))
    count=case_counts(schedule,labels)
    @test count.scheduled==91 && count.features_accepted==60
    @test count.labels_accepted==61 && count.supervised==30
    @test count.features_rejected==31 && count.labels_rejected==30
    arrays=case_arrays(schedule,labels)
    @test arrays.issues==collect(START+Minute(31):Minute(1):START+Minute(60))
    @test arrays.rows==collect(32:61)
    @test size(arrays.features)==(30,8)
    @test length(arrays.targets)==length(arrays.raw_now)==length(arrays.scales)==30
    @test !arrays.actual_fit_performed && !arrays.source_bound_provenance_verified
    @test label_row(labels,1).accepted  # diagnostic label for a feature-rejected issue
    @test !label_row(labels,62).accepted && label_row(labels,62).reason===:missing_ground_target
    @test label_row(labels,62).target===nothing
    arrays.features[1,1]=-99
    @test feature_row(schedule,32).features[1]!=-99
end

@testset "Readiness and strict split checks precede target queries" begin
    early=audit_catalog_labels(MIXED,G;asof=READY-Millisecond(1))
    e=label_row(early,1)
    @test !e.accepted && e.reason===:label_not_ready && e.target===nothing
    @test e.support_sha256==EXTRA["empty_support_sha256"]
    equal=audit_catalog_labels(MIXED,G;asof=READY,label_ready_before=READY)
    @test !label_row(equal,1).accepted && label_row(equal,1).reason===:split_not_strictly_ready
    @test label_row(equal,1).support_sha256==EXTRA["empty_support_sha256"]
    admitted=audit_catalog_labels(MIXED,G;asof=READY,label_ready_before=READY+Millisecond(1))
    @test label_row(admitted,1).accepted
    @test bits(label_row(admitted,1).target)==FIX["target_expected_bits"]
    @test label_row(admitted,1).support_sha256!=EXTRA["empty_support_sha256"]
    @test case_counts(MIXED,equal).supervised==0
    @test size(case_arrays(MIXED,equal).features)==(0,8)
end

@testset "Future source values do not determine numerical features" begin
    plain=catalog(FIX["ground_no_future_spike"],:ground)
    changed=seal_catalog_features(plain,[ISSUE];mode=:mixed8,wind=WIND)
    @test feature_row(changed,1).features==feature_row(MIXED,1).features
    @test feature_row(changed,1).raw_now==feature_row(MIXED,1).raw_now
    @test feature_row(changed,1).scale==feature_row(MIXED,1).scale
    changed_labels=audit_catalog_labels(changed,plain;asof=READY)
    @test bits(label_row(changed_labels,1).target)==FIX["target_without_spike_bits"]
    @test_throws ArgumentError audit_catalog_labels(MIXED,plain;asof=READY)
    @test_throws ArgumentError case_arrays(changed,audit_catalog_labels(MIXED,G;asof=READY))
end

@testset "Explicit original-vector and wind rejections" begin
    for key in ("ground_missing_history","ground_partial_history")
        g=catalog(FIX[key],:ground)
        value=seal_catalog_features(g,[ISSUE];mode=:mixed8,wind=WIND)
        @test !feature_row(value,1).accepted && feature_row(value,1).reason===:missing_ground_history
    end
    g=catalog(FIX["ground_missing_target"],:ground)
    value=seal_catalog_features(g,[ISSUE])
    @test feature_row(value,1).accepted
    result=audit_catalog_labels(value,g;asof=READY)
    @test !label_row(result,1).accepted && label_row(result,1).reason===:missing_ground_target
    w=catalog(FIX["wind_partial_anchor"],:wind)
    value=seal_catalog_features(G,[ISSUE];mode=:mixed8,wind=w)
    @test !feature_row(value,1).accepted && feature_row(value,1).reason===:missing_omni_history
    @test feature_row(seal_catalog_features(G,[ISSUE];wind=w),1).accepted
    negative=catalog(replace(FIX["wind"],",400,"=>",-400,"),:wind)
    negative_row=ROLE=="primary" ? C.locate(negative,ISSUE-Minute(1)) : I.audit_locate(negative,ISSUE-Minute(1))
    @test negative_row.valid_mask==UInt8(14) && negative_row.reason_bits==UInt8(3)
    @test feature_row(seal_catalog_features(G,[ISSUE];mode=:mixed8,wind=negative),1).reason===:missing_omni_history
    overflow=catalog(replace(FIX["wind"],",-2,400,"=>",-1e308,1e308,"),:wind)
    overflow_row=ROLE=="primary" ? C.locate(overflow,ISSUE-Minute(1)) : I.audit_locate(overflow,ISSUE-Minute(1))
    @test overflow_row.valid_mask==UInt8(12) && overflow_row.reason_bits==UInt8(15)
    @test feature_row(seal_catalog_features(G,[ISSUE];mode=:mixed8,wind=overflow),1).reason===:missing_omni_history
end

@testset "Input boundary and context errors fail before artifacts" begin
    @test_throws ArgumentError seal_catalog_features(G,DateTime[])
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE,ISSUE])
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE,ISSUE-Minute(1)])
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE+Millisecond(1)])
    @test_throws ArgumentError seal_catalog_features(G,[true])
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE];policy=true)
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE];mode=:unknown)
    @test_throws ArgumentError seal_catalog_features(G,[ISSUE];mode=:mixed8,wind=:forbidden)
    @test_throws ArgumentError seal_catalog_features(WIND,[ISSUE])
    @test_throws ArgumentError seal_catalog_features(G,fill(ISSUE,600001))
    @test_throws ArgumentError feature_row(MIXED,true)
    @test_throws ArgumentError feature_row(MIXED,0)
    @test_throws ArgumentError feature_row(MIXED,2)
    @test_throws ArgumentError audit_catalog_labels(MIXED,G;asof=true)
    @test_throws ArgumentError audit_catalog_labels(MIXED,G;asof=READY,label_ready_before=true)
    labels=audit_catalog_labels(MIXED,G;asof=READY)
    @test_throws ArgumentError label_row(labels,true)
    @test_throws ArgumentError label_row(labels,0)
end

include(joinpath(@__DIR__,"test_catalog_cases_numeric_oracles.jl"))

mutable struct ChangingIssues <: AbstractVector{DateTime}
    initial::Vector{DateTime}
    reads::Int
    change_after::Int
end
Base.size(value::ChangingIssues)=(length(value.initial),)
Base.IndexStyle(::Type{ChangingIssues})=IndexLinear()
function Base.getindex(value::ChangingIssues,index::Int)
    value.reads+=1
    value.reads>value.change_after ? first(value.initial) : value.initial[index]
end
@testset "Own caller schedule before validating and reading source" begin
    for threshold in (3,7)
        input=ChangingIssues([ISSUE,ISSUE+Minute(1),ISSUE+Minute(2)],0,threshold)
        rejected=false;times=DateTime[]
        try
            schedule=seal_catalog_features(G,input)
            times=[feature_row(schedule,index).issue for index in 1:length(schedule)]
        catch error
            error isa ArgumentError || rethrow()
            rejected=true
        end
        @test rejected || length(unique(times))==3
        @test rejected || all(times[index]<times[index+1] for index in 1:length(times)-1)
        @test rejected || times==[ISSUE,ISSUE+Minute(1),ISSUE+Minute(2)]
    end
end

