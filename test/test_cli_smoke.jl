# Short, isolated launcher checks. No monitor or dashboard is started.
isdefined(@__MODULE__, :solarsindy_test_bash) || include(joinpath(@__DIR__, "bash_runtime.jl"))

@testset "bin/solarsindy CLI essentials" begin
    bash = solarsindy_test_bash()
    cli = replace(normpath(joinpath(@__DIR__, "..", "bin", "solarsindy")), '\\' => '/')
    @test isfile(cli)

    mktempdir() do dir
        config = joinpath(dir, "solarsindy.env")
        write(config, "")
        environment = merge(ENV, Dict(
            "SOLARSINDY_IGNORE_SERVICE" => "1",
            "SOLARSINDY_MONITOR_DIR" => replace(dir, '\\' => '/'),
            "SOLARSINDY_ENV_FILE" => replace(config, '\\' => '/'),
            "SOLARSINDY_NO_OPEN" => "1",
            "SWM_HOST" => "127.0.0.1",
            "SWM_PORT" => "1",
            "NO_PROXY" => "127.0.0.1,localhost",
            "JULIA" => replace(first(Base.julia_cmd().exec), '\\' => '/'),
        ))
        pop!(environment, "BASH_ENV", nothing)
        pop!(environment, "SWM_WEBHOOK_URL", nothing)

        function capture_cli(arguments...; trace=false, env=environment)
            argv = trace ? [bash, "-x", cli, arguments...] : [bash, cli, arguments...]
            output, errors = IOBuffer(), IOBuffer()
            process = run(pipeline(setenv(ignorestatus(Cmd(argv)), env);
                                   stdout=output, stderr=errors))
            return (; code=process.exitcode, out=String(take!(output)), err=String(take!(errors)))
        end

        @test success(`$bash -n $cli`)
        help = capture_cli("help")
        @test help.code == 0
        @test all(word -> occursin(word, help.out), ("start", "stop", "status", "help"))
        @test capture_cli("unknown-command").code != 0

        status = capture_cli("status")
        @test status.code == 2
        @test occursin("monitor:   stopped", status.out)
        @test occursin("dashboard: stopped", status.out)

        # A broken optional setting must not prevent recovery commands.
        write(config, "SWM_WEBHOOK_URL='unterminated\nSWM_PORT=65200\n")
        configured = copy(environment)
        delete!(configured, "SWM_PORT")
        broken = capture_cli("status"; env=configured)
        @test broken.code == 2
        @test occursin("malformed", broken.err)
        @test capture_cli("help"; env=configured).code == 0
        @test capture_cli("stop", "monitor"; env=configured).code == 0

        # Inspect the resolved runtime URL, including the end of the port.
        write(config, "SWM_PORT=65201\r\n")
        crlf = capture_cli("help"; trace=true, env=configured)
        @test crlf.code == 0
        url_lines = filter(line -> startswith(line, "+ URL="), split(crlf.err, '\n'))
        @test url_lines == ["+ URL=http://127.0.0.1:65201"]
        @test capture_cli("status"; env=configured).code == 2

        # Only an owned harmless child is used to check the Unix stale-PID guard.
        if Sys.isunix()
            decoy = run(ignorestatus(`$bash -c 'exec sleep 20'`); wait=false)
            try
                @test process_running(decoy)
                pidfile = joinpath(dir, "run", "monitor.pid")
                mkpath(dirname(pidfile))
                write(pidfile, string(getpid(decoy)), "\n")
                stopped = capture_cli("stop", "monitor")
                @test stopped.code == 0
                @test !isfile(pidfile)
                @test process_running(decoy)
            finally
                process_running(decoy) && kill(decoy)
                wait(decoy)
            end
        end
    end
end
