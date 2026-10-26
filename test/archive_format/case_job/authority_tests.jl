module ArchiveCatalogCaseAuthorityTests
using Dates,SHA,Test,SolarSINDy
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
digest(bytes)=bytes2hex(sha256(bytes))
function failure(action,expected)
    message=nothing
    try;action();catch error;error isa ArgumentError || rethrow();message=error.msg;end
    @test message==expected
end
@testset "Feature authority separation before source access" begin
    mktempdir() do temporary
        directory=realpath(temporary);absent=joinpath(directory,"never-created")
        pin="0"^64
        feature=A.Authority(absent,pin,absent,pin,absent,pin,absent,pin,absent,pin,absent,pin)
        source=J.Authority(absent,pin,absent,pin,absent,pin,absent,pin)
        @test Q.Authority===A.Authority
        @test A.authorize(Dict("evidence_mode"=>"generated"),pin,nothing)===nothing
        for supplied in (feature,source,Dict(),true)
            failure(()->A.authorize(Dict("evidence_mode"=>"generated"),pin,supplied),
                "generated case jobs cannot consume actual feature authority")
        end
        for supplied in (nothing,source,Dict(),true)
            failure(()->A.authorize(Dict("evidence_mode"=>"actual_archival_cache"),pin,supplied),
                "actual case job requires caller-pinned feature authority")
        end
        request=Dict("schema"=>"archive_catalog_case_request_v1","evidence_mode"=>"actual_archival_cache",
            "layout"=>"ground-only","role"=>"primary","runtime"=>string(VERSION),
            "source"=>Dict("directory"=>absent,"manifest_sha256"=>pin,"complete_sha256"=>pin),
            "tables"=>[Dict("id"=>"CMO-declaration-only","station"=>"CMO",
                "start_inclusive"=>"2017-01-01T00:00:00.000Z","stop_exclusive"=>"2017-01-01T00:01:00.000Z",
                "label_asof"=>"2017-01-01T00:32:00.000Z")])
        path=joinpath(directory,"request.json");bytes=R.canonical_bytes(request;limit=131072)
        R._exclusive_write(path,bytes);request_pin=digest(bytes)
        for supplied in (nothing,source)
            failure(()->Q.publish(path,joinpath(directory,"must-not-publish");request_sha256=request_pin,authority=supplied),
                "actual case job requires caller-pinned feature authority")
            failure(()->Q.read_verified(path,joinpath(directory,"must-not-read");request_sha256=request_pin,
                manifest_sha256=pin,complete_sha256=pin,authority=supplied),
                "actual case job requires caller-pinned feature authority")
        end
        @test !ispath(absent) && !ispath(joinpath(directory,"must-not-publish")) && !ispath(joinpath(directory,"must-not-read"))
        @test Set(readdir(directory))==Set(["request.json"])
    end
end
@testset "Frozen plan clocks have one canonical request representation" begin
    @test A._plan_clock("2017-01-01T00:00:00Z")=="2017-01-01T00:00:00.000Z"
    @test A._plan_clock("2018-01-01T00:31:00.000Z")=="2018-01-01T00:31:00.000Z"
    @test A._plan_clock("2024-02-29T23:59:00Z")=="2024-02-29T23:59:00.000Z"
    for bad in ("2017-01-01T00:00:01Z","2017-01-01T00:00:00.001Z","2017-01-01T00:00:00+00:00",
                "2017-01-01 00:00:00Z","2017-02-29T00:00:00Z",true,0)
        @test_throws ArgumentError A._plan_clock(bad)
    end
    input=Dict("id"=>"CMO-example","station"=>"CMO","start_inclusive"=>"2017-01-01T00:00:00Z",
        "stop_exclusive"=>"2018-01-01T00:00:00Z","label_asof"=>"2018-01-01T00:31:00.000Z",
        "scheduled_minutes"=>525600,"evidence_use"=>"declaration-only generated test")
    expected=Dict("id"=>"CMO-example","station"=>"CMO","start_inclusive"=>"2017-01-01T00:00:00.000Z",
        "stop_exclusive"=>"2018-01-01T00:00:00.000Z","label_asof"=>"2018-01-01T00:31:00.000Z")
    @test A._project_table(input)==expected
    @test input["start_inclusive"]=="2017-01-01T00:00:00Z"
end
end
