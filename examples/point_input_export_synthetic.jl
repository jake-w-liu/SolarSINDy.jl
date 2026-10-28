module PointInputExportSynthetic
using Dates,SHA,SolarSINDy,JSON3
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","point_input_export.jl"))
const P=PointInputExport
include(joinpath(@__DIR__,"..","test","archive_source_bound_fit","case_fixture.jl"))

function _python()
    if haskey(ENV,"SOLARSINDY_RESEARCH_PYTHON")
        candidate=ENV["SOLARSINDY_RESEARCH_PYTHON"]
        isabspath(candidate)&&isfile(candidate)&&!islink(candidate) || error("explicit research Python must be an absolute ordinary executable")
        return realpath(candidate)
    end
    for name in (Sys.iswindows() ? ("python","python3") : ("python3","python"))
        candidate=Sys.which(name);candidate===nothing || return realpath(candidate)
    end
    error("Python>=3.11 is required for independent point-input verification")
end
function run_example(;destination=nothing)
    parent=if destination===nothing
        base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
        realpath(mktempdir(base;prefix="solarsindy-point-input-",cleanup=false))
    else
        isabspath(destination)&&!ispath(destination)&&!islink(destination) || error("generated input destination must be a fresh absolute path")
        mkdir(destination);realpath(destination)
    end
    exports=Any[];start=START+Minute(40);stop=START+Minute(120);asof=START+Minute(211)
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        directory=joinpath(parent,role*"-"*layout);mkdir(directory)
        value=make_cases(realpath(directory),role,layout)
        for (index,station) in enumerate(("CMO","FRD"))
            output=joinpath(directory,"input-"*station);request=P.planning_request(value,index;start,stop,asof)
            result=P.export_generated(value,index,output;start,stop,asof)
            @assert result.rows==50 && result.columns==(layout=="ground-only" ? 11 : 8) && result.chunks==1
            @assert !result.actual_source_access && result.actual_fits==0
            push!(exports,Dict("directory"=>output,"case_directory"=>joinpath(directory,"cases"),"case_request_path"=>joinpath(directory,"case-request.json"),
                "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
                "table_index"=>index,"start"=>clock(start),"stop"=>clock(stop),"asof"=>clock(asof),
                "entry_sha256"=>request["entry_sha256"],"numeric_source_sha256"=>request["numeric_source_sha256"],"library_sha256"=>request["library_sha256"],
                "context_sha256"=>result.context_sha256,"manifest_sha256"=>result.manifest_sha256,"complete_sha256"=>result.complete_sha256,
                "receipt_sha256"=>result.receipt_sha256,"input_sha256"=>result.input_sha256,"role"=>role,"layout"=>layout,"station"=>station))
        end
    end
    report=joinpath(parent,"REPORT.json")
    R._exclusive_write(report,R.canonical_bytes(Dict("schema"=>"generated_portable_point_input_exports_v1","complete"=>true,
        "runtime"=>string(VERSION),"exports"=>exports,"actual_source_access"=>false,"actual_fits"=>0);limit=1048576))
    entry=joinpath(@__DIR__,"..","tools","point_input_export","run_generated_checks.py");output=joinpath(parent,"independent")
    logpath=joinpath(parent,"independent-controller.log")
    try
        open(logpath,"w") do log
            run(pipeline(Cmd([_python(),"-B",entry,"--report",report,"--report-sha256",digest(read(report)),"--output",output]);stdout=log,stderr=log))
        end
    catch
        println(stderr,"POINT_INPUT_GENERATED_FAILURE_RAW=",logpath);rethrow()
    end
    terminal=JSON3.read(read(joinpath(output,"TERMINAL.json"),String))
    @assert terminal.complete===true && terminal.independently_verified_exports==8 && terminal.total_verified_input_rows==400 && isempty(terminal.changed_controls)
    println("PORTABLE_POINT_INPUT_GENERATED_PASSED;EXPORTS=8;INPUT_ROWS=400;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0")
    (;directory=parent,exports=8,input_rows=400,actual_source_access=false,actual_fits=0)
end
end
