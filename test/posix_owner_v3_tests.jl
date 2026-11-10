module PosixOwnerV3PackageTests
using Test,SHA,JSON3,SolarSINDy
include(joinpath(@__DIR__,"completed_refit_artifacts","run_ci_native_chain.jl"))
const SUPPORT=CompletedRefitNativeDriver
const CONTROL=joinpath(@__DIR__,"run_posix_owner_v3_controls.py")
const CONTROL_SHA="5f5c19b711b7ca3fa3ff8f500f308c0397ba9f38866cd99178267a9134ce3738"
file_sha(path)=open(io->bytes2hex(sha256(io)),path)

function run_tests()
    @testset "Current POSIX owner V3 mandatory independent controls" begin
        file_sha(CONTROL)==CONTROL_SHA||error("reviewed V3 controller pin changed")
        parent=realpath(get(ENV,"RUNNER_TEMP",tempdir()))
        root=realpath(mktempdir(parent;prefix="solarsindy-posix-v3-",cleanup=false))
        output=joinpath(root,"evidence")
        logpath=joinpath(root,"controller.log")
        python=SUPPORT.choose_python()
        command=addenv(Cmd([python,"-B","-X","utf8",CONTROL,"--output-directory",output]),
            "PYTHONDONTWRITEBYTECODE"=>"1")
        process=nothing
        started=time_ns()
        try
            open(logpath,"w") do log
                process=run(pipeline(command;stdout=log,stderr=log);wait=false)
                while process_running(process)
                    time_ns()-started<UInt64(690*10^9)||error("V3 controller watchdog expired")
                    sleep(.05)
                end
                wait(process)
                time_ns()-started<UInt64(690*10^9)||error("V3 controller completed after watchdog deadline")
                success(process)||error("V3 controls failed; see "*logpath)
            end
            path=joinpath(output,"TERMINAL.json")
            isfile(path)&&!islink(path)||error("ordinary V3 terminal required")
            0<filesize(path)<=2*1024^2||error("V3 terminal exceeds metadata bound")
            bytes=read(path)
            0<length(bytes)<=2*1024^2||error("V3 terminal exceeds metadata bound")
            result=JSON3.read(bytes,Dict{String,Any})
            @test result["schema"]=="owned_posix_v3_registered_controls_v1"
            @test result["complete"]===true&&result["source_unchanged"]===true
            @test result["state_methods"]==5&&result["simulated_rows"]==85
            applicable=Sys.islinux()||Sys.isapple()
            @test result["native_POSIX_applicable"]===applicable
            @test result["actual_current_native_cases"]==(applicable ? 21 : 0)
            @test result["historical_negative_cases"]==(Sys.isapple() ? 3 : 0)
            @test result["Julia_launched"]===false&&result["actual_archive_exposed"]===false
            @test result["source_qualified"]===false&&result["scientific_qualified"]===false
            @test file_sha(CONTROL)==CONTROL_SHA
            println("POSIX_OWNER_V3_EVIDENCE=",output," terminal_sha256=",bytes2hex(sha256(bytes)))
            time_ns()-started<UInt64(690*10^9)||error("V3 verification/output completed after watchdog deadline")
        catch
            println(stderr,"POSIX_OWNER_V3_FAILURE_RAW=",logpath)
            if isfile(logpath)
                open(logpath,"r") do io
                    seekend(io);seek(io,max(0,position(io)-65536))
                    write(stderr,read(io,65536));println(stderr)
                end
            end
            rethrow()
        finally
            if process!==nothing&&process_running(process)
                kill(process)
                cleanup=time_ns();escalated=false
                while process_running(process)
                    if !escalated&&time_ns()-cleanup>=UInt64(35*10^9)
                        kill(process,Base.SIGKILL);escalated=true
                    end
                    time_ns()-cleanup<UInt64(40*10^9)||error("V3 controller cleanup not observed within40s")
                    sleep(.05)
                end
                wait(process)
            end
        end
    end
end
end
PosixOwnerV3PackageTests.run_tests()
