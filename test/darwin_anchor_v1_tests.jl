module DarwinAnchorV1PackageTests
using Test,SHA,JSON3,SolarSINDy
include(joinpath(@__DIR__,"completed_refit_artifacts","run_ci_native_chain.jl"))
const SUPPORT=CompletedRefitNativeDriver
const CONTROL=joinpath(@__DIR__,"run_darwin_anchor_v1_controls.py")
const CONTROL_SHA="54e16f5531f5605e2c0fc8434520c8bc7ca01a4d4e2c16ed96279faf57af7eff"
file_sha(path)=open(io->bytes2hex(sha256(io)),path)

function run_tests()
    @testset "Mandatory independent Darwin launch-anchor controls" begin
        file_sha(CONTROL)==CONTROL_SHA||error("reviewed Darwin-anchor controller changed")
        parent=realpath(get(ENV,"RUNNER_TEMP",tempdir()))
        root=realpath(mktempdir(parent;prefix="solarsindy-darwin-anchor-",cleanup=false))
        output=joinpath(root,"evidence")
        logpath=joinpath(root,"controller.log")
        python=SUPPORT.choose_python()
        command=addenv(Cmd([python,"-I","-B","-X","utf8",CONTROL,"--output-directory",output]),
            "PYTHONDONTWRITEBYTECODE"=>"1")
        process=nothing
        started=time_ns()
        try
            open(logpath,"w") do log
                process=run(pipeline(command;stdout=log,stderr=log);wait=false)
                while process_running(process)
                    time_ns()-started<UInt64(690*10^9)||error("Darwin-anchor controller watchdog expired")
                    sleep(.05)
                end
                wait(process)
                time_ns()-started<UInt64(690*10^9)||error("Darwin-anchor controller completed after watchdog deadline")
                success(process)||error("Darwin-anchor controls failed; see "*logpath)
            end
            path=joinpath(output,"TERMINAL.json")
            isfile(path)&&!islink(path)||error("ordinary Darwin-anchor terminal required")
            0<filesize(path)<=2*1024^2||error("Darwin-anchor terminal exceeds metadata bound")
            bytes=open(io->read(io,2*1024^2+1),path)
            0<length(bytes)<=2*1024^2||error("Darwin-anchor terminal grew beyond metadata bound")
            time_ns()-started<UInt64(690*10^9)||error("Darwin-anchor result read exceeded watchdog deadline")
            result=JSON3.read(bytes,Dict{String,Any})
            @test result["schema"]=="independent_darwin_anchor_registered_controls_v1"
            @test result["complete"]===true&&result["source_unchanged"]===true
            @test result["state_methods"]==3&&result["literal_rows"]==72
            applicable=Sys.isapple()
            @test result["native_Darwin_applicable"]===applicable
            @test result["actual_current_native_cases"]==(applicable ? 12 : 0)
            @test result["historical_negative_cases"]==(applicable ? 3 : 0)
            @test result["failure_cleanup_negative_cases"]==(applicable ? 1 : 0)
            @test result["actual_native_cases"]==(applicable ? 16 : 0)
            @test result["Julia_launched"]===false&&result["actual_data_exposed"]===false
            @test result["native_Darwin_qualified"]===false
            @test result["source_qualified"]===false&&result["scientific_qualified"]===false
            @test file_sha(CONTROL)==CONTROL_SHA
            time_ns()-started<UInt64(690*10^9)||error("Darwin-anchor verification exceeded watchdog deadline")
            println("DARWIN_ANCHOR_V1_EVIDENCE=",output," terminal_sha256=",bytes2hex(sha256(bytes)))
            time_ns()-started<UInt64(690*10^9)||error("Darwin-anchor verification/output completed after watchdog deadline")
        catch
            println(stderr,"DARWIN_ANCHOR_V1_FAILURE_RAW=",logpath)
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
                    time_ns()-cleanup<UInt64(40*10^9)||error("Darwin-anchor controller cleanup not observed within40s")
                    sleep(.05)
                end
                wait(process)
            end
        end
    end
end
end
DarwinAnchorV1PackageTests.run_tests()
