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
