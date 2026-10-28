module CompletedRefitNativeDriver
using SolarSINDy, SHA, JSON3
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const PACKAGE=normpath(joinpath(@__DIR__,"..",".."))
const OUTPUT_ENV="SOLARSINDY_SYNTHETIC_REFIT_TEST_ROOT"
file_sha(path)=open(io->bytes2hex(sha256(io)),path)
read_control(path)=R.decode_canonical(R.bounded_read(path;limit=R.MAX_MEMBER_BYTES))

function choose_python()
    if haskey(ENV,"SOLARSINDY_RESEARCH_PYTHON")
        path=ENV["SOLARSINDY_RESEARCH_PYTHON"]
        isabspath(path)&&isfile(path)&&!islink(path) || error("explicit Python must be an absolute ordinary executable")
        return realpath(path)
    end
    for name in (Sys.iswindows() ? ("python","python3") : ("python3","python"))
        path=Sys.which(name)
        if path!==nothing
            resolved=realpath(path)
            isabspath(resolved)&&isfile(resolved)&&!islink(resolved) || error("Python executable is not ordinary")
            return resolved
        end
    end
    error("Python>=3.11 is required for the fixed synthetic native tests")
end

function verify_root(directory)
    isabspath(directory)&&isdir(directory)&&!islink(directory) || error("synthetic test root must be an absolute ordinary directory")
    terminal=read_control(joinpath(directory,"TERMINAL.json"))
    terminal["schema"]=="explicit_synthetic_test_descendant_routes_v1" && terminal["complete"]===true || error("synthetic routes incomplete")
    projection_path=joinpath(directory,"SOURCE_PROJECTION.json")
    file_sha(projection_path)==terminal["source_projection_sha256"] || error("source projection pin differs")
    projection=read_control(projection_path)
    projection["original_checkout_unchanged"]===true && projection["child_checkpoint_truthful"]===true || error("source projection is incomplete")
    projection["child_head"]==terminal["actual_native_child_head"] || error("native child identity differs")
    file_sha(joinpath(directory,"Manifest.diff"))==projection["manifest_diff_sha256"] || error("exact manifest diff pin differs")
    rows=projection["source_before"]["files_sha256"]
    mapping=Dict{String,String}(pair[1]=>pair[2] for pair in rows)
    length(mapping)==length(rows) || error("duplicate source projection path")
    required=unique(vcat(R._code_files("source_audit"),R._code_files("refit_point"),["Project.toml","Manifest.toml"]))
    all(name->get(mapping,name,nothing)==file_sha(joinpath(PACKAGE,name)),required) || error("test proof differs from current source/dependency bytes")
    for layout in ("ground11","mixed8")
        route=terminal["routes"][layout]
        path=joinpath(directory,layout,"native","TERMINAL.json")
        file_sha(path)==route["terminal_sha256"] || error("native terminal pin differs")
        value=read_control(path)
        value["complete"]===true && value["source_head"]==projection["child_head"] && value["layout"]==layout || error("native route identity differs")
    end
    directory
end

function ensure_root()
    haskey(ENV,OUTPUT_ENV) && return verify_root(ENV[OUTPUT_ENV])
    parent=if haskey(ENV,"RUNNER_TEMP")
        realpath(mktempdir(realpath(ENV["RUNNER_TEMP"]);prefix="solarsindy-r1-native-",cleanup=false))
    else
        realpath(mktempdir(;prefix="solarsindy-r1-native-",cleanup=false))
    end
    destination=joinpath(parent,"proof")
    python=choose_python()
    julia=realpath(joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia"))
    entry=joinpath(PACKAGE,"research","dbdt_completed_source","synthetic_checkpoint.py")
    command=Cmd([python,"-B",entry,"--julia",julia,"--destination",destination])
    logpath=joinpath(parent,"native-controller.log")
    open(logpath,"w") do log
        run(pipeline(command;stdout=log,stderr=log))
    end
    verify_root(destination)
    ENV[OUTPUT_ENV]=destination
    destination
end

function _run_support_command(command,parent)
    logpath=joinpath(parent,"native.log")
    try
        open(logpath,"w") do log
            run(pipeline(command;stdout=log,stderr=log))
        end
    catch
        println(stderr,"SUPPORT_TEST_FAILURE_RAW=",logpath)
        try
            open(logpath,"r") do log
                seekend(log);bytes=position(log);seek(log,max(0,bytes-65536))
                write(stderr,read(log,65536));println(stderr)
            end
        catch diagnostic_error
            println(stderr,"Support diagnostic read failed: ",typeof(diagnostic_error))
        end
        rethrow()
    end
    nothing
end
function run_support_tests()
    python=choose_python()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    isdir(base)&&!islink(base) || error("support diagnostic parent must be ordinary")
    for name in ("test_portable_support.py","test_hosted_harness.py")
        script=joinpath(@__DIR__,name)
        parent=realpath(mktempdir(base;prefix="solarsindy-r1-support-log-",cleanup=false))
        command=addenv(Cmd([python,"-B",script]),"TMP"=>parent,"TEMP"=>parent,"TMPDIR"=>parent)
        _run_support_command(command,parent)
    end
end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    CompletedRefitNativeDriver.run_support_tests()
    directory=CompletedRefitNativeDriver.ensure_root()
    if get(ENV,"GITHUB_ACTIONS","")=="true"
        path=ENV["GITHUB_ENV"]
        isabspath(path)&&isfile(path)&&!islink(path) || error("GitHub environment file must be ordinary")
        !occursin('\n',directory)&&!occursin('\r',directory) || error("invalid output directory")
        open(path,"a") do stream
            println(stream,CompletedRefitNativeDriver.OUTPUT_ENV,"=",directory)
        end
    end
    println("Generated synthetic native source/fit/calibration routes verified: ",directory)
end
