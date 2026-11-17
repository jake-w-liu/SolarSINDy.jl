module PointInputExportTests
using Dates,SHA,Test,SolarSINDy,JSON3
include(joinpath(@__DIR__,"..","..","examples","point_input_export_synthetic.jl"))
const S=PointInputExportSynthetic
const P=S.P
const R=S.R
const B=SolarSINDy.ArchiveCatalogCaseReader
@testset "Generated exports and independent full-record reader" begin
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    parent=realpath(mktempdir(base;prefix="solarsindy-point-input-test-",cleanup=false))
    result=S.run_example(;destination=joinpath(parent,"generated"))
    @test result.exports==8 && result.input_rows==400 && !result.actual_source_access && result.actual_fits==0
    request=joinpath(result.directory,"primary-ground-only","case-request.json");directory=joinpath(result.directory,"primary-ground-only","cases")
    value=S.Q.read_verified(request,directory;request_sha256=bytes2hex(sha256(read(request))),
        manifest_sha256=bytes2hex(sha256(read(joinpath(directory,"manifest.json")))),complete_sha256=bytes2hex(sha256(read(joinpath(directory,"COMPLETE.json")))))
    start=S.START+Minute(40);stop=S.START+Minute(120);asof=S.START+Minute(211)
    @test_throws ArgumentError P.planning_request(value,true;start,stop,asof)
    @test_throws ArgumentError P.planning_request(value,1;start=start+Millisecond(1),stop,asof)
    @test_throws ArgumentError P.export_generated(value,1,joinpath(result.directory,"primary-ground-only","input-CMO");start,stop,asof)
    reread=B.read_reverified(request,directory;request_sha256=bytes2hex(sha256(read(request))),
        manifest_sha256=bytes2hex(sha256(read(joinpath(directory,"manifest.json")))),complete_sha256=bytes2hex(sha256(read(joinpath(directory,"COMPLETE.json")))))
    denied=joinpath(parent,"actual-denied")
    @test_throws ArgumentError P.export_archival(reread,1,denied;start,stop,asof)
    @test !ispath(denied)
end
println("PORTABLE_POINT_INPUT_PATH_AND_DENIALS_PASSED;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
