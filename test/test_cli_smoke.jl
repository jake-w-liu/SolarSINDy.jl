# bin/solarsindy CLI smoke tests: black-box, network-free, bounded.
#
# The full daemon lifecycle (start → health → stop for both daemons, PID-reuse decoy,
# port-busy refusal) spawns real Julia daemons and — for the monitor — live NOAA SWPC
# fetches, so it is exercised out-of-band as a functional check. This suite pins the
# contract that must not rot silently inside Pkg.test():
#   * the script parses and `help` documents the full command surface,
#   * unknown commands fail loudly,
#   * `status` on an isolated empty instance reports both components stopped (exit 2),
#   * a pidfile naming a live process that is NOT our daemon is treated as stale —
#     removed WITHOUT signaling the process (the PID-reuse safety guard),
#   * a malformed solarsindy.env line is skipped with a warning and NEVER bricks the CLI
#     (recovery commands like `stop` must keep working with a broken config),
#   * CRLF config files cannot smuggle \r into ports/paths,
#   * a live start lock makes a concurrent `start` refuse instead of double-spawning.
@testset "bin/solarsindy CLI smoke" begin
    cli = normpath(joinpath(@__DIR__, "..", "bin", "solarsindy"))
    @test isfile(cli)
    bash = Sys.isunix() ? Sys.which("bash") : nothing
    if bash === nothing
        @warn "bin/solarsindy smoke tests skipped: POSIX bash required" cli
    else
        # Isolated instance: nothing listens on the port, state dir is empty, the
        # installed-service guard is bypassed, and the developer's real solarsindy.env
        # (if any) is masked so results do not depend on the host.
        dir = mktempdir()
        env = copy(ENV)
        env["SOLARSINDY_IGNORE_SERVICE"] = "1"
        env["SOLARSINDY_MONITOR_DIR"] = dir
        env["SWM_PORT"] = "65123"
        env["SOLARSINDY_ENV_FILE"] = "/dev/null"

        @test success(`$bash -n $cli`)                       # parses under bash

        help_out = read(setenv(`$bash $cli help`, env), String)
        for word in ("setup", "start", "stop", "restart", "status", "once", "logs",
                     "open", "install-service", "uninstall-service")
            @test occursin(word, help_out)
        end

        bad = run(setenv(ignorestatus(`$bash $cli frobnicate`), env);
                  wait=true)
        @test bad.exitcode != 0                              # unknown command fails loudly

        st = run(setenv(ignorestatus(`$bash $cli status`), env); wait=true)
        @test st.exitcode == 2                               # both components stopped

        lg = run(setenv(ignorestatus(`$bash $cli logs monitor`), env); wait=true)
        @test lg.exitcode != 0                               # no log yet -> error, not success

        lg2 = run(setenv(ignorestatus(`$bash $cli logs monitor bogus`), env); wait=true)
        @test lg2.exitcode != 0                              # unknown extra argument rejected

        # PID-reuse guard: a pidfile pointing at a live, unrelated process must be
        # cleaned as stale and the process must NOT be signaled.
        decoy = run(`sleep 15`; wait=false)
        decoy_pid = getpid(decoy)                            # capture while alive: getpid throws after exit
        rundir = joinpath(dir, "run"); mkpath(rundir)
        pidfile = joinpath(rundir, "monitor.pid")
        write(pidfile, string(decoy_pid))
        st2 = run(setenv(ignorestatus(`$bash $cli stop monitor`), env); wait=true)
        @test st2.exitcode == 0                              # "not running" is a clean stop
        @test !isfile(pidfile)                               # stale pidfile removed
        @test process_running(decoy)                         # decoy untouched by the CLI

        # Start lock: while a live process owns the start lock, `start` refuses fast
        # instead of racing to a second spawn. (Uses the decoy as the lock owner and a
        # no-op julia shim so nothing real is ever spawned.)
        shim_dir = mktempdir()
        shim = joinpath(shim_dir, "julia")
        write(shim, "#!/bin/sh\nexit 0\n"); chmod(shim, 0o755)
        envlock = copy(env); envlock["JULIA"] = shim
        lockfile = joinpath(rundir, ".monitor.start.lock")
        write(lockfile, string(decoy_pid))
        lock_io = IOBuffer()
        locked = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envlock);
                              stdout=devnull, stderr=lock_io); wait=true)
        @test locked.exitcode != 0                           # live lock owner -> refuse
        # Pin the refusal PATH, not just a non-zero exit: deleting the lock entirely still
        # exits non-zero (the shim dies during startup), so assert the refusal was announced.
        @test occursin("another start is already in progress", String(take!(lock_io)))
        @test read(lockfile, String) == string(decoy_pid)    # foreign lock left in place
        kill(decoy); wait(decoy)

        # Stale lock (owner dead): reclaimed, start proceeds to spawn the shim, which
        # exits immediately -> death-during-startup is reported, and neither the pidfile
        # nor the lock survives.
        stale = run(setenv(ignorestatus(`$bash $cli start monitor`), envlock); wait=true)
        @test stale.exitcode != 0                            # shim died during startup
        @test !isfile(pidfile)
        @test !isfile(lockfile)

        # Dead-PID pidfile: cleaned up the same way (the decoy has now exited, so its PID
        # is guaranteed dead — modulo an unlikely immediate reuse, in which case the
        # command-match guard still classifies it as stale).
        write(pidfile, string(decoy_pid))
        st3 = run(setenv(ignorestatus(`$bash $cli status`), env); wait=true)
        @test st3.exitcode == 2
        @test !isfile(pidfile)

        # Broken config must not brick the CLI: the malformed line is skipped with a
        # warning, the good line still applies, and every command keeps working.
        cfg = joinpath(mktempdir(), "solarsindy.env")
        write(cfg, "SWM_WEBHOOK_URL=https://hooks.example.com/it's\nSWM_PORT=65200\n")
        envcfg = copy(env); envcfg["SOLARSINDY_ENV_FILE"] = cfg
        delete!(envcfg, "SWM_PORT")                          # let the file's good line win
        delete!(envcfg, "SWM_WEBHOOK_URL")                   # file line must be the one parsed
        cfg_io = IOBuffer()
        stc = run(pipeline(setenv(ignorestatus(`$bash $cli status`), envcfg);
                           stdout=devnull, stderr=cfg_io); wait=true)
        @test stc.exitcode == 2                              # still a working "stopped" status
        @test occursin("malformed", String(take!(cfg_io)))   # and the skip was announced
        hlp = run(setenv(ignorestatus(`$bash $cli help`), envcfg); wait=true)
        @test hlp.exitcode == 0                              # help unaffected by broken config

        # CRLF config: the \r must be stripped, not smuggled into the resolved URL/port
        # (a CR-suffixed port silently disables the port-busy and readiness checks).
        write(cfg, "SWM_PORT=65201\r\n")
        trace_io = IOBuffer()
        run(pipeline(setenv(ignorestatus(`$bash -x $cli help`), envcfg);
                     stdout=devnull, stderr=trace_io); wait=true)
        trace = String(take!(trace_io))
        url_lines = filter(l -> occursin("URL=", l), split(trace, '\n'))
        @test !isempty(url_lines)
        @test any(l -> occursin("http://127.0.0.1:65201", l), url_lines)
        @test all(l -> !occursin('\r', l), url_lines)
        # `bash -x` escapes a stray CR to the two literal characters backslash-r (verified via
        # od -c), so a raw-CR check alone can never fail under the exact bug it pins (dropping
        # the \r strip). Pin the escaped form too: a non-stripped port shows `...:65201\r` here.
        @test all(l -> !occursin("\\r", l), url_lines)
    end
end
