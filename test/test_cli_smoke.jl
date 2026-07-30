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
#   * the start lock refuses a concurrent start only for a live *CLI* owner; a dead or
#     PID-recycled owner is reclaimed (never wedging start after a SIGKILL).
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

        # Start lock, identity semantics: the lock is honored only when its recorded owner
        # PID is a live *solarsindy CLI* process (same discipline as pid_of). A live but
        # UNRELATED owner — the SIGKILLed-start-then-PID-recycled case — is stale: the lock
        # is reclaimed without signaling that process, start proceeds (the no-op shim dies
        # at once), and no in-progress refusal is announced.
        shim_dir = mktempdir()
        shim = joinpath(shim_dir, "julia")
        write(shim, "#!/bin/sh\nexit 0\n"); chmod(shim, 0o755)
        envlock = copy(env); envlock["JULIA"] = shim
        lockfile = joinpath(rundir, ".monitor.start.lock")
        write(lockfile, string(decoy_pid))                   # decoy is `sleep`: live, not the CLI
        rec_io = IOBuffer()
        rec = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envlock);
                           stdout=devnull, stderr=rec_io); wait=true)
        rec_err = String(take!(rec_io))
        @test rec.exitcode != 0                              # shim died during startup
        @test !occursin("another start is already in progress", rec_err)
        @test occursin("exited during startup", rec_err)     # i.e. the reclaim path ran
        @test !isfile(lockfile)                              # recycled-PID lock reclaimed
        @test process_running(decoy)                         # reclaim never signals the owner
        kill(decoy); wait(decoy)

        # Genuine live CLI owner: a start sitting in its readiness loop holds the lock, and
        # a concurrent start must refuse with the in-progress message and leave the lock.
        # The shim must answer need_julia's version probe instantly (it runs BEFORE the
        # lock is acquired) and sleep only when invoked as the daemon itself.
        slow = joinpath(shim_dir, "julia_slow")
        write(slow, "#!/bin/sh\ncase \"\$*\" in *live_monitor.jl*) sleep 30 ;; *) exit 0 ;; esac\n")
        chmod(slow, 0o755)
        envslow = copy(env); envslow["JULIA"] = slow
        p1 = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envslow);
                          stdout=devnull, stderr=devnull); wait=false)
        t0 = time()
        while !isfile(lockfile) && time() - t0 < 15 && process_running(p1)
            sleep(0.2)
        end
        @test isfile(lockfile)
        ref_io = IOBuffer()
        ref = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envlock);
                           stdout=devnull, stderr=ref_io); wait=true)
        @test ref.exitcode != 0
        @test occursin("another start is already in progress", String(take!(ref_io)))
        @test isfile(lockfile)                               # genuine owner's lock untouched
        # Tear down: SIGKILL the owner (EXIT trap cannot fire), kill its shim daemon, and
        # clear the leftovers; the dead-owner reclaim itself is pinned right below.
        kill(p1, 9)
        shim_daemon = try parse(Int, strip(readline(pidfile))) catch; 0 end
        shim_daemon > 0 && (try run(ignorestatus(`kill -9 $shim_daemon`)) catch end)
        rm(pidfile; force=true); rm(lockfile; force=true)
        wait(p1)
        decoy_dead = run(`sleep 1`; wait=false)              # short-lived: guaranteed-dead PID
        dead_pid = getpid(decoy_dead)
        wait(decoy_dead)

        # Stale lock (owner dead): reclaimed, start proceeds to spawn the shim, which
        # exits immediately -> death-during-startup is reported, and neither the pidfile
        # nor the lock survives.
        write(lockfile, string(dead_pid))                    # dead owner: guaranteed stale
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

        # Multi-line pidfile is stale by definition: never "repaired" into its first line,
        # and the live process named there is never signaled.
        decoy2 = run(`sleep 15`; wait=false)
        write(pidfile, string(getpid(decoy2), "\n999999\n"))
        stm = run(setenv(ignorestatus(`$bash $cli status`), env); wait=true)
        @test stm.exitcode == 2
        @test !isfile(pidfile)
        @test process_running(decoy2)
        kill(decoy2); wait(decoy2)

        # Binding all interfaces must not yield an unusable probe/browse URL.
        z_io = IOBuffer()
        envz = copy(env); envz["SWM_HOST"] = "0.0.0.0"
        run(pipeline(setenv(ignorestatus(`$bash -x $cli help`), envz);
                     stdout=devnull, stderr=z_io); wait=true)
        zurls = filter(l -> occursin("URL=", l), split(String(take!(z_io)), '\n'))
        @test any(l -> occursin("http://127.0.0.1:65123", l), zurls)

        # Broken config must not brick the CLI: the malformed line is skipped with a
        # warning, the good lines still apply, every command keeps working, and the two
        # foot-gun classes (unquoted whitespace value, parser-reserved variable name) are
        # each announced with their own warning.
        cfg = joinpath(mktempdir(), "solarsindy.env")
        write(cfg, "SWM_WEBHOOK_URL=https://hooks.example.com/it's\nSWM_PORT=65200\n" *
                   "X_SPACED=a b\n__solarsindy_cfg_line=evil\n")
        envcfg = copy(env); envcfg["SOLARSINDY_ENV_FILE"] = cfg
        delete!(envcfg, "SWM_PORT")                          # let the file's good line win
        delete!(envcfg, "SWM_WEBHOOK_URL")                   # file line must be the one parsed
        cfg_io = IOBuffer()
        stc = run(pipeline(setenv(ignorestatus(`$bash $cli status`), envcfg);
                           stdout=devnull, stderr=cfg_io); wait=true)
        cfg_err = String(take!(cfg_io))
        @test stc.exitcode == 2                              # still a working "stopped" status
        @test occursin("malformed", cfg_err)                 # and the skip was announced
        @test occursin("unquoted whitespace", cfg_err)
        @test occursin("reserved variable name", cfg_err)
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
