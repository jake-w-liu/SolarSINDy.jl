module ArchivalStaticTests
using Dates,SHA,Test,JSON3,SolarSINDy
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","source_bound_archival_static.jl"))
const AS=SourceBoundArchivalStatic
const S=SourceBoundSharedReplay
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const J=SolarSINDy.ArchiveCatalogSourceJob
const Q=SolarSINDy.ArchiveCatalogCaseJob
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","archive_source_bound_fit","case_fixture.jl"))
fixture()=JSON3.read(read(joinpath(@__DIR__,"archival_static_codec_oracle.json"),String),Dict{String,Any})
function payloads()
    oracle=fixture()
    context=hex2bytes(oracle["context_hex"]);numeric=hex2bytes(oracle["numeric_hex"]);receipt=hex2bytes(oracle["receipt_hex"])
    expected=AS.ExpectedStatic(oracle["context_sha256"],oracle["numeric_sha256"])
    (;oracle,context,numeric,receipt,expected)
end
function prepare(root,name="data")
    f=payloads();p=AS._prepare_publication(joinpath(root,name),f.context,f.numeric,f.receipt,f.expected)
    (;f...,prepared=p)
end
function reread(f,pins)
    AS._read_publication(f.prepared.directory,f.expected;pins...)
end
function repin_metadata(f,pins,change)
    directory=f.prepared.directory
    manifest=R.decode_canonical(read(joinpath(directory,"manifest.json"));limit=16384)
    marker=R.decode_canonical(read(joinpath(directory,"COMPLETE.json"));limit=4096)
    change(manifest,marker)
    mb=R.canonical_bytes(manifest;limit=16384);marker["manifest_sha256"]=digest(mb)
    cb=R.canonical_bytes(marker;limit=4096)
    write(joinpath(directory,"manifest.json"),mb);write(joinpath(directory,"COMPLETE.json"),cb)
    merge(pins,(manifest_sha256=digest(mb),complete_sha256=digest(cb)))
end
@testset "Independent literal static numeric codec, without archival authority" begin
    f=payloads()
    @test f.oracle["fixture_only"]===true && f.oracle["actual_source_authority_granted"]===false
    @test length(f.numeric)==186 && digest(f.numeric)=="1449b3e36a0d75a728654e0b5229a351ba2c88ae280b3c5a49bbaa10ed0e65aa"
    @test digest(f.context)==f.expected.context_sha256 && digest(f.receipt)==f.oracle["receipt_sha256"]
    # Literal IEEE754 dyadic values were frozen independently of the Julia codec.
    v=S.B._decode(f.numeric)
    @test v.means==[0.,0.] && v.scales==[1.,1.] && v.weights==[2.,0.,0.]
    @test v.raw==5. && v.normalized==2.5 && v.edges==[0.,1.,2.,3.,4.]
    @test v.quantiles==fill(5.,4) && v.alpha==.5 && v.normal==0.
    @test S.B._encode(v)==f.numeric
    result=S.B._predict(v,zeros(2,2),[0.,1.],[2.,3.],[.5,3.])
    @test result.bounds.het_gaussian==expm1.([0.,1.])
    @test result.bounds.raw_split==expm1.([5.,6.])
    @test result.bounds.normalized_split==expm1.([5.,8.5])
    @test result.bounds.regime_split==expm1.([5.,6.])
    capsule=AS.ArchivalStaticCapsule(f.context,f.numeric,f.receipt,AS._Token())
    f.context[1]=0;f.numeric[1]=0;f.receipt[1]=0
    @test digest(codeunits(capsule.context))==f.expected.context_sha256
    @test digest(codeunits(capsule.numeric))==f.expected.numeric_sha256
    @test digest(codeunits(capsule.receipt))==f.oracle["receipt_sha256"]
    @test capsule isa AS.ArchivalStaticCapsule && !(capsule isa S.B.StaticCalibration)
end
@testset "Static data publication is fresh, bounded, complete last and externally pinned" begin
    mktempdir() do root
        f=prepare(root);directory=f.prepared.directory
        @test Set(readdir(directory))==Set(("context.json","numeric.bin","receipt.json","manifest.json"))
        @test !ispath(joinpath(directory,"COMPLETE.json"))
        @test_throws ArgumentError AS._read_publication(directory,f.expected;manifest_sha256=f.prepared.manifest_sha256,
            receipt_sha256=f.prepared.receipt_sha256,complete_sha256=f.prepared.complete_sha256)
        # The publication owns its input. Mutation cannot alter the prepared files.
        f.context[1]=0;f.numeric[1]=0;f.receipt[1]=0
        pins=AS._complete_publication(f.prepared);readback=reread(f,pins);original=payloads()
        @test readback.context==original.context && readback.numeric==original.numeric && readback.receipt==original.receipt
        @test readback.codec_only===true && length(readdir(directory))==5
        @test pins.receipt_sha256==original.oracle["receipt_sha256"]
        @test_throws Exception AS._complete_publication(f.prepared)
        before=Dict(name=>read(joinpath(directory,name)) for name in readdir(directory))
        @test_throws ArgumentError AS._prepare_publication(directory,original.context,original.numeric,original.receipt,original.expected)
        @test before==Dict(name=>read(joinpath(directory,name)) for name in readdir(directory))
        for key in (:manifest_sha256,:receipt_sha256,:complete_sha256)
            changed=merge(pins,NamedTuple{(key,)}((repeat("0",64),)))
            @test_throws ArgumentError reread(f,changed)
        end
        wrong=AS.ExpectedStatic(repeat("0",64),original.expected.numeric_sha256)
        @test_throws ArgumentError AS._read_publication(directory,wrong;pins...)
    end
    for (index,name) in enumerate(("context.json","numeric.bin","receipt.json","manifest.json"))
        mktempdir() do root
            f=prepare(root);file=joinpath(f.prepared.directory,name);body=read(file);body[1]⊻=0x01;write(file,body)
            @test_throws ArgumentError AS._complete_publication(f.prepared)
            @test !ispath(joinpath(f.prepared.directory,"COMPLETE.json"))
        end
    end
    for name in ("context.json","numeric.bin","receipt.json","manifest.json","COMPLETE.json")
        mktempdir() do root
            f=prepare(root);pins=AS._complete_publication(f.prepared);file=joinpath(f.prepared.directory,name)
            body=read(file);write(file,body[1:end-1])
            @test_throws ArgumentError reread(f,pins)
        end
    end
    mktempdir() do root
        f=prepare(root);write(joinpath(f.prepared.directory,"extra.txt"),"retained")
        @test_throws ArgumentError AS._complete_publication(f.prepared)
        @test !ispath(joinpath(f.prepared.directory,"COMPLETE.json"))
        @test read(joinpath(f.prepared.directory,"extra.txt"),String)=="retained"
    end
    mktempdir() do root
        f=prepare(root);pins=AS._complete_publication(f.prepared);write(joinpath(f.prepared.directory,"extra.txt"),"retained")
        @test_throws ArgumentError reread(f,pins)
    end
    mktempdir() do root
        f=payloads()
        @test_throws ArgumentError AS._prepare_publication(joinpath(root,"oversize"),zeros(UInt8,32769),f.numeric,f.receipt,f.expected)
        @test !ispath(joinpath(root,"oversize"))
        @test_throws ArgumentError AS._prepare_publication(joinpath(root,"empty"),UInt8[],f.numeric,f.receipt,f.expected)
        @test !ispath(joinpath(root,"empty"))
        @test_throws ArgumentError AS._prepare_publication(joinpath(root,"wrong"),f.context,f.numeric,f.receipt,AS.ExpectedStatic(repeat("0",64),f.expected.numeric_sha256))
        @test !ispath(joinpath(root,"wrong"))
    end
    # Re-pin malformed metadata so these checks exercise the structural contract,
    # rather than merely failing the outer byte hash.
    for change in ((m,c)->(m["members"][1]["name"]="../context.json"),
                   (m,c)->(m["members"][2]["bytes"]=true),
                   (m,c)->(m["members"][3]["sha256"]=repeat("0",64)),
                   (m,c)->(c["data_only"]=false),
                   (m,c)->(c["members"]=true),
                   (m,c)->push!(m["members"],copy(m["members"][1])),
                   (m,c)->(m["unknown"]=false))
        mktempdir() do root
            f=prepare(root);pins=AS._complete_publication(f.prepared);changed=repin_metadata(f,pins,change)
            @test_throws ArgumentError reread(f,changed)
        end
    end
end
@testset "Opaque fixture and missing evidence cannot become an actual calibration" begin
    f=payloads();opaque=AS.ArchivalStaticCapsule(f.context,f.numeric,f.receipt,AS._Token())
    @test_throws MethodError AS.prepare_request(nothing,1,nothing,nothing,nothing)
    @test_throws MethodError AS.calibrate_archival(nothing,1,nothing,nothing,nothing)
    @test_throws MethodError AS.check_calibration(opaque,nothing,nothing,nothing;expected_static=f.expected)
    # A structural roundtrip alone cannot publish or read an archival capsule.
    mktempdir() do root
        f=prepare(root);pins=AS._complete_publication(f.prepared)
        @test_throws MethodError AS.read_static(f.prepared.directory,nothing,nothing,nothing;expected_static=f.expected,pins...)
        @test_throws MethodError AS.publish_static(opaque,nothing,nothing,nothing,joinpath(root,"not-authorized");expected_static=f.expected)
        @test !ispath(joinpath(root,"not-authorized"))
    end
    @test length(AS.job_identity())==64
end
@testset "Generated source/model capsules cannot enter archival uncertainty fitting" begin
    mktempdir() do root
        value=make_cases(realpath(root),"primary","ground-only")
        recipe=C.Recipe(kind=:forest,depth=0,trees=1,bins=2,minleaf=1,sample_rows=1,seed=17)
        train_start=START+Minute(40);train_stop=START+Minute(90);asof=START+Minute(211)
        pe=F.prepare_expected(value,1,recipe;start=train_start,stop=train_stop,asof)
        point=F.fit_generated(value,1,recipe,pe;start=train_start,stop=train_stop,asof).capsule
        start=START+Minute(90);stop=START+Minute(135);first_issue=START+Minute(150)
        ce=S.K.prepare_expected(value,1,point;start,stop,asof,first_issue)
        calibration=S.K.calibrate_generated(value,1,point,ce;start,stop,asof,first_issue)
        @test_throws MethodError AS.prepare_request(value,1,point,calibration,ce)
        @test_throws MethodError AS.calibrate_archival(value,1,point,calibration,ce)
        # This deliberately invalid wrapper retains false generated authority.
        # No positive archival admission or qualified source is fabricated.
        invalid=AS.B.ReverifiedCases(value,"{}\n",AS.B._ReadToken())
        @test_throws ArgumentError AS.prepare_request(invalid,1,point,calibration,ce)
        @test_throws ArgumentError AS.calibrate_archival(invalid,1,point,calibration,ce)
        missing=AS.StaticFitAuthority(joinpath(root,"missing.json"),repeat("0",64),
            joinpath(root,"missing-proof.json"),repeat("0",64),joinpath(root,"missing-qualification.json"),repeat("0",64))
        @test_throws ArgumentError AS.calibrate_archival(invalid,1,point,calibration,ce;authority=missing)
        @test !ispath(joinpath(root,"STATIC_FIT_ATTEMPT.json"))
    end
end
end
