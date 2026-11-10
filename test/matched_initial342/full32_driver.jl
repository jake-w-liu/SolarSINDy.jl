# A resolver may legitimately update only Project/Manifest in the source tree.
# Full32 still requires a genuinely clean source proof, in an owned projection
# when necessary; no production reader or numerical assertion is relaxed.
function full32_python()
    explicit=get(ENV,"SOLARSINDY_RESEARCH_PYTHON",nothing)
    choices=explicit===nothing ? (Sys.iswindows() ? ("python","python3") : ("python3","python")) : (explicit,)
    for name in choices
        path=explicit===nothing ? Sys.which(name) : name
        path===nothing&&continue
        isabspath(path)&&isfile(path)&&!islink(path)||error("Full32 Python must be an ordinary absolute executable")
        return realpath(path)
    end
    error("Python>=3.11 is required for bounded full32 source projection")
end
function run_owned_full32_tests()
    parent=realpath(get(ENV,"RUNNER_TEMP",tempdir()))
    owned_case(parent,"full32-controller") do root
        destination=joinpath(root,"evidence")
        logpath=joinpath(root,"controller.log")
        active=Base.active_project()
        active!==nothing&&isfile(active)||error("Full32 requires its actual resolved active environment")
        selected=Base.project_file_manifest_path(String(active))
        selected!==nothing||error("Full32 active environment has no selected manifest")
        P._ordinary(selected)
        realpath(selected)==realpath(joinpath(dirname(active),"Manifest.toml"))||
            error("Full32 supports only the actual selected same-directory Manifest.toml")
        command=Cmd([full32_python(),"-I","-B",joinpath(@__DIR__,"full32_projection.py"),
            "--package-root",realpath(pkgdir(SolarSINDy)),"--destination",destination,
            "--julia",realpath(joinpath(Sys.BINDIR,Base.julia_exename())),"--active-project",realpath(active),
            "--selected-manifest",realpath(selected)])
        command=addenv(command,"JULIA_DEPOT_PATH"=>join(DEPOT_PATH,Sys.iswindows() ? ';' : ':'),
            "JULIA_NUM_THREADS"=>"1","OMP_NUM_THREADS"=>"1","OPENBLAS_NUM_THREADS"=>"1")
        process=nothing;started=time_ns()
        try
            open(logpath,"w") do log
                process=run(pipeline(command;stdout=log,stderr=log);wait=false)
                # The wrapper owns a 7200s whole-work deadline and <=20s native
                # cleanup. This independent parent watchdog also bounds its wait.
                while process_running(process)
                    time_ns()-started<UInt64(7230*10^9)||error("Full32 controller watchdog expired")
                    sleep(.05)
                end
                wait(process)
                time_ns()-started<UInt64(7230*10^9)||error("Full32 controller completed after watchdog deadline")
                success(process)||error("Full32 owned worker failed; see "*logpath)
            end
            path=joinpath(destination,"TERMINAL.json")
            filesize(path)<=2*1024^2||error("Full32 terminal exceeds metadata bound")
            result=P._bounded_json_parse(P._bound(path,2*1024^2))
            @test result.schema=="matched342_full32_owned_test_terminal_v1"
            @test result.complete===true&&result.original_checkout_unchanged===true
            @test result.native.native_exit==0&&result.native.timed_out===false
            @test result.native.ownership_boundary_closed===true&&result.native.all_descendants_os_reaped===false
            @test result.worker.cases==["nonlinear","flat"]&&result.worker.fit_count==64
            @test result.worker.nonlinear_complete===true&&result.worker.flat_complete===true
            @test result.worker.julia_pid==result.julia_pid&&result.owned_leader_pid==result.native.pid
            @test result.available_physical_bytes>=8*1024^3&&result.free_disk_bytes>=16*1024^3
            @test result.wall_seconds<7200
            println("MATCHED342_FULL32_OWNED_DONE projection=",result.projected," julia_pid=",result.julia_pid,
                " terminal_sha256=",bytes2hex(sha256(P._bound(path,2*1024^2))));flush(stdout)
            time_ns()-started<UInt64(7230*10^9)||error("Full32 verification/output completed after watchdog deadline")
        catch
            println(stderr,"MATCHED342_FULL32_CONTROLLER_RAW ",logpath)
            if isfile(logpath)
                open(logpath,"r") do io
                    seekend(io);seek(io,max(0,position(io)-65536));write(stderr,read(io,65536));println(stderr)
                end
            end
            rethrow()
        finally
            if process!==nothing&&process_running(process)
                kill(process)
                cleanup_started=time_ns()
                escalated=false
                while process_running(process)
                    if !escalated&&time_ns()-cleanup_started>=UInt64(35*10^9)
                        kill(process,Base.SIGKILL);escalated=true
                    end
                    time_ns()-cleanup_started<UInt64(40*10^9)||error("Full32 controller cleanup was not observed within40s")
                    sleep(.05)
                end
                wait(process)
            end
        end
    end
end
