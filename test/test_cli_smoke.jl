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
#   * the start lock preserves any positively live owner (including aliases and a
#     recycled/unrelated live PID); only confirmed absence permits dead-owner recovery.
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

        # The CLI and supervised service both use Julia's normal compilation for concurrent requests.
        script = read(cli, String)
        @test occursin("--startup-file=no --project=\"\$ROOT/app\"", script)
        @test !occursin("--compile=", script)

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

        # GNU stat may emit partial stdout when BSD flags fail. Age reporting
        # must use only the successful fallback, including under set -u.
        mktempdir() do stat_dir
            stat_bin = joinpath(stat_dir, "bin"); mkpath(stat_bin)
            stat_shim = joinpath(stat_bin, "stat")
            write(stat_shim, raw"""#!/bin/bash
if [ "$1" = "-f" ]; then
  printf 'partial filesystem output\n'
  exit 1
fi
printf '%s\n' "$CLI_TEST_MTIME"
""")
            chmod(stat_shim, 0o755)
            write(joinpath(stat_dir, "live_forecast_log.csv"), "issue_time_utc\n2026-09-19T12:00\n")
            stat_env = merge(env, Dict(
                "PATH" => string(stat_bin, ":", get(ENV, "PATH", "")),
                "SOLARSINDY_MONITOR_DIR" => stat_dir,
                "CLI_TEST_MTIME" => string(floor(Int, time()) - 600),
            ))
            stat_out, stat_err = IOBuffer(), IOBuffer()
            stat_result = run(pipeline(setenv(ignorestatus(`$bash $cli status`), stat_env);
                                      stdout=stat_out, stderr=stat_err))
            @test stat_result.exitcode == 2
            @test occursin("forecasts: log updated 10 min ago", String(take!(stat_out)))
            @test isempty(String(take!(stat_err)))
            write(stat_shim, "#!/bin/bash\nprintf 'failed stat output\\n'\nexit 1\n")
            stat_result = run(pipeline(setenv(ignorestatus(`$bash $cli status`), stat_env);
                                      stdout=stat_out, stderr=stat_err))
            @test stat_result.exitcode == 2
            @test occursin("? min", String(take!(stat_out)))
            @test isempty(String(take!(stat_err)))
        end

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

        # Start-lock ownership is deliberately conservative: flattened argv cannot
        # distinguish physical/alias/relative launches. A positively live unrelated
        # or recycled owner is held without signaling or spawning a replacement.
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
        @test rec.exitcode != 0
        @test occursin("another start is already in progress", rec_err)
        @test !occursin("exited during startup", rec_err)
        @test read(lockfile, String) == string(decoy_pid)
        @test !isfile(pidfile)                               # no replacement spawned
        @test process_running(decoy)                         # held owner never signaled
        kill(decoy); wait(decoy)

        # Genuine live CLI owner: a start sitting in its readiness loop holds the lock, and
        # a concurrent start must refuse with the in-progress message and leave the lock.
        # The shim must answer need_julia's version probe instantly (it runs BEFORE the
        # lock is acquired) and sleep only when invoked as the daemon itself.
        slow = joinpath(shim_dir, "julia_slow")
        slow_started = joinpath(shim_dir, "monitor-main.pid")
        write(slow, "#!/bin/sh\ncase \"\$*\" in *live_monitor.jl*) printf '%s\\n' \"\$\$\" > \"\$CLI_SMOKE_MAIN_STARTED\"; sleep 30 ;; *) exit 0 ;; esac\n")
        chmod(slow, 0o755)
        envslow = copy(env); envslow["JULIA"] = slow; envslow["CLI_SMOKE_MAIN_STARTED"] = slow_started
        p1 = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envslow);
                          stdout=devnull, stderr=devnull); wait=false)
        admission = lockfile * ".admission"
        function smoke_numeric_pid(path)
            try
                record = read(path, String)
                ncodeunits(record) <= 128 && occursin(r"\A[1-9][0-9]*\n?\z", record) || return nothing
                return tryparse(Int, chomp(record))
            catch
                return nothing
            end
        end
        function smoke_start_phase_ready()
            process_running(p1) && isfile(lockfile) && !ispath(admission) || return false
            committed = smoke_numeric_pid(pidfile)
            return committed !== nothing && committed == smoke_numeric_pid(slow_started)
        end
        # Numeric lock visibility precedes admission release. Killing at that earlier
        # boundary deliberately leaves UNKNOWN admission, not a recoverable dead lock.
        # Actual shim main entry plus its matching PID proves the intended phase.
        t0 = time()
        while !smoke_start_phase_ready() && time() - t0 < 15 && process_running(p1)
            sleep(0.2)
        end
        @test smoke_start_phase_ready()                      # bounded post-admission main-phase precondition
        @test isfile(lockfile)
        ref_io = IOBuffer()
        ref = run(pipeline(setenv(ignorestatus(`$bash $cli start monitor`), envlock);
                           stdout=devnull, stderr=ref_io); wait=true)
        @test ref.exitcode != 0
        @test occursin("another start is already in progress", String(take!(ref_io)))
        @test isfile(lockfile)                               # genuine owner's lock untouched
        @test smoke_start_phase_ready()                      # contender released its admission before SIGKILL
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

        # Installed-service port: once a service exists, its rendered plist is the authority for
        # the port it binds. Editing the config file without reinstalling used to make `status`
        # probe the config's port and report a healthy dashboard unreachable.
        if Sys.isapple() && Sys.which("plutil") !== nothing
            svc_root = mktempdir()
            svc_home = joinpath(svc_root, "home"); mkpath(svc_home)
            fake_bin = joinpath(svc_root, "bin"); mkpath(fake_bin)
            fake_launchctl = joinpath(fake_bin, "launchctl")
            # Reports only the dashboard as loaded, so the monitor stays "stopped" and the two
            # branches are exercised independently.
            write(fake_launchctl, """#!/bin/sh
case "\$1" in
  list) echo "1234\t0\tcom.empire.solarsindy.dashboard" ;;
esac
exit 0
""")
            chmod(fake_launchctl, 0o755)
            installer = normpath(joinpath(@__DIR__, "..", "deploy", "install_launchd.sh"))
            install_env = copy(ENV)
            install_env["HOME"] = svc_home
            install_env["PATH"] = string(fake_bin, ":", get(ENV, "PATH", ""))
            install_env["SOLARSINDY_LOAD"] = "0"
            install_env["SOLARSINDY_ORG"] = "empire"
            install_env["SOLARSINDY_JULIA"] = shim
            install_env["SOLARSINDY_MONITOR_DIR"] = joinpath(svc_root, "monitor")
            install_env["SWM_PORT"] = "9137"
            delete!(install_env, "SWM_HOST")
            delete!(install_env, "SWM_WEBHOOK_URL")
            @test success(pipeline(setenv(`$bash $installer dashboard`, install_env);
                                   stdout=devnull, stderr=devnull))
            plist = joinpath(svc_home, "Library", "LaunchAgents",
                             "com.empire.solarsindy.dashboard.plist")
            @test isfile(plist)

            svc_env = copy(env)
            delete!(svc_env, "SOLARSINDY_IGNORE_SERVICE")   # let the service guard see the fake
            svc_env["HOME"] = svc_home
            svc_env["PATH"] = string(fake_bin, ":", get(ENV, "PATH", ""))
            svc_env["SOLARSINDY_MONITOR_DIR"] = joinpath(svc_root, "state")
            svc_io = IOBuffer()
            svc = run(pipeline(setenv(ignorestatus(`$bash $cli status`), svc_env);
                               stdout=svc_io, stderr=devnull); wait=true)
            svc_out = String(take!(svc_io))
            @test svc.exitcode == 1                          # dashboard up, monitor stopped
            @test occursin("installed dashboard service binds port 9137", svc_out)
            @test occursin("9137", svc_out)
            @test !occursin("http://127.0.0.1:65123", svc_out)
        end

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

# G10-01: the pid_of identity mark embeds ROOT, so ROOT must be the physical clone path. A
# daemon started via the real path must still be recognized (and its pidfile kept) when the CLI
# is invoked through a symlink to the same clone. `ps` is stubbed so no real daemon is needed.
@testset "CLI recognizes its daemon through a symlinked clone path" begin
    cli_bash = Sys.isunix() ? Sys.which("bash") : nothing
    if cli_bash !== nothing
        mktempdir() do root
            repo = normpath(joinpath(@__DIR__, ".."))
            link = joinpath(root, "clone_link")
            symlink(repo, link)
            physical = readchomp(`$cli_bash -c 'cd "$1" && pwd -P' _ $repo`)
            stub_bin = joinpath(root, "bin"); mkpath(stub_bin)
            write(joinpath(stub_bin, "ps"), """#!/bin/sh
case "\$*" in
  *etime*) echo 01:00 ;;
  *) echo "julia --startup-file=no --project=$physical $physical/examples/live_monitor.jl" ;;
esac
""")
            chmod(joinpath(stub_bin, "ps"), 0o755)
            state = joinpath(root, "state"); mkpath(joinpath(state, "run"))
            pidfile = joinpath(state, "run", "monitor.pid")
            write(pidfile, "4242\n")
            env = merge(ENV, Dict("SOLARSINDY_IGNORE_SERVICE" => "1",
                "SOLARSINDY_MONITOR_DIR" => state, "SWM_PORT" => "65124",
                "SOLARSINDY_ENV_FILE" => "/dev/null",
                "PATH" => string(stub_bin, ":", get(ENV, "PATH", ""))))
            out = IOBuffer()
            st = run(pipeline(setenv(ignorestatus(`$cli_bash $(joinpath(link, "bin", "solarsindy")) status`), env);
                              stdout=out, stderr=devnull))
            @test st.exitcode == 1                           # monitor up, dashboard stopped
            @test occursin("monitor:   RUNNING (pid 4242", String(take!(out)))
            @test isfile(pidfile)                            # live daemon's pidfile kept
        end
    end
end

# G12-02: an IPv6 SWM_HOST must be bracketed in every probe/browse URL; unbracketed
# http://::1:8723 is rejected by curl, so the watchdog would report a permanent false outage.
@testset "IPv6 dashboard host is bracketed in probe URLs" begin
    cli_bash = Sys.isunix() ? Sys.which("bash") : nothing
    if cli_bash !== nothing
        mktempdir() do root
            repo = normpath(joinpath(@__DIR__, ".."))
            cli = joinpath(repo, "bin", "solarsindy")
            env = merge(ENV, Dict("SOLARSINDY_IGNORE_SERVICE" => "1",
                "SOLARSINDY_MONITOR_DIR" => joinpath(root, "state"), "SWM_PORT" => "65125",
                "SOLARSINDY_ENV_FILE" => "/dev/null", "SWM_HOST" => "::1"))
            trace = IOBuffer()
            run(pipeline(setenv(ignorestatus(`$cli_bash -x $cli help`), env);
                         stdout=devnull, stderr=trace))
            urls = filter(l -> occursin("URL=", l), split(String(take!(trace)), '\n'))
            @test any(l -> occursin("http://[::1]:65125", l), urls)
            @test !any(l -> occursin("http://::1", l), urls)

            # Installer: macOS-only tools stubbed, nothing loaded.
            stub_bin = joinpath(root, "bin"); mkpath(stub_bin)
            for tool in ("plutil", "launchctl", "julia")
                write(joinpath(stub_bin, tool), "#!/bin/sh\nexit 0\n")
                chmod(joinpath(stub_bin, tool), 0o755)
            end
            home = joinpath(root, "home"); mkpath(home)
            ienv = merge(ENV, Dict("HOME" => home, "SOLARSINDY_LOAD" => "0",
                "PATH" => string(stub_bin, ":", get(ENV, "PATH", "")),
                "SOLARSINDY_JULIA" => joinpath(stub_bin, "julia"),
                "SOLARSINDY_MONITOR_DIR" => joinpath(root, "monitor"),
                "SWM_HOST" => "::1", "SWM_PORT" => "8723"))
            delete!(ienv, "SWM_WEBHOOK_URL"); delete!(ienv, "SOLARSINDY_WATCHDOG_DASH_URL")
            installer = joinpath(repo, "deploy", "install_launchd.sh")
            @test success(pipeline(setenv(`$cli_bash $installer watchdog`, ienv);
                                   stdout=devnull, stderr=devnull))
            plist = read(joinpath(home, "Library", "LaunchAgents",
                                  "com.empire.solarsindy.watchdog.plist"), String)
            @test occursin("http://[::1]:8723/api/health", plist)
        end
    end
end

# G12-03: a watchdog-authored OUTAGE.md is refreshed when the problem set changes (the dashboard
# serves its Summary), but a daemon-authored sentinel is never clobbered.
@testset "watchdog refreshes its own outage sentinel" begin
    cli_bash = Sys.isunix() ? Sys.which("bash") : nothing
    if cli_bash !== nothing && Sys.which("curl") !== nothing
        mktempdir() do mon
            wd = normpath(joinpath(@__DIR__, "..", "deploy", "watchdog.sh"))
            env = merge(ENV, Dict("SOLARSINDY_MONITOR_DIR" => mon,
                "SOLARSINDY_WATCHDOG_DASH_URL" => "http://127.0.0.1:1/api/health"))
            delete!(env, "SWM_WEBHOOK_URL")
            probe() = run(pipeline(setenv(ignorestatus(`$cli_bash $wd`), env);
                                   stdout=devnull, stderr=devnull))
            sentinel = joinpath(mon, "OUTAGE.md")
            probe()                                          # no log + dashboard down
            @test occursin("forecast log missing", read(sentinel, String))
            write(joinpath(mon, "live_forecast_log.csv"), "x\n")
            probe()                                          # only dashboard down now
            s2 = read(sentinel, String)
            @test occursin("Source: external watchdog", s2)
            @test occursin("dashboard health endpoint unreachable", s2)
            @test !occursin("forecast log missing", s2)
            @test !isfile(sentinel * ".tmp")
            daemon = "# OUTAGE\nSource: live monitor\nSummary: issuance failing\n"
            write(sentinel, daemon)
            rm(joinpath(mon, "live_forecast_log.csv"))      # problem set changes again
            probe()
            @test read(sentinel, String) == daemon           # daemon sentinel never clobbered
        end
    end
end

# G12-04 / G12-05: release activation must install services with the operator's solarsindy.env
# (env vars win, like bin/solarsindy), and a relative RELEASES_ROOT is resolved against the
# caller's cwd into an absolute path so `current` never dangles.
@testset "Release activation loads operator config; relative releases root" begin
    cli_bash = Sys.isunix() ? Sys.which("bash") : nothing
    if cli_bash !== nothing && Sys.which("git") !== nothing
        mktempdir() do root
            clone = joinpath(root, "clone"); mkpath(joinpath(clone, "deploy"))
            cp(joinpath(@__DIR__, "../deploy/deploy_release.sh"),
               joinpath(clone, "deploy/deploy_release.sh"))
            stub = joinpath(clone, "deploy/install_launchd.sh")
            write(stub, raw"""#!/bin/bash
echo "STUB webhook=${SWM_WEBHOOK_URL-<unset>} port=${SWM_PORT-<unset>} mon=${SOLARSINDY_MONITOR_DIR-<unset>}"
""")
            chmod(stub, 0o755)
            write(joinpath(clone, ".gitignore"), "solarsindy.env\n")
            run(`git -C $clone init --quiet`)
            run(`git -C $clone add .`)
            run(`git -C $clone -c user.name=Test -c user.email=test@example.invalid commit -qm fixture`)
            filemon = joinpath(root, "filemon")
            write(joinpath(clone, "solarsindy.env"),
                  "SWM_WEBHOOK_URL=https://hooks.example/x\nSWM_PORT=9100\nSOLARSINDY_MONITOR_DIR='$filemon'\n")
            script = joinpath(clone, "deploy/deploy_release.sh")
            base = copy(ENV)
            for k in ("SWM_WEBHOOK_URL", "SWM_PORT", "SOLARSINDY_MONITOR_DIR", "SOLARSINDY_ENV_FILE")
                delete!(base, k)
            end
            base["HOME"] = joinpath(root, "home")
            base["SOLARSINDY_INSTANTIATE"] = "0"
            deploy(args, extra) = begin
                out = IOBuffer()
                r = run(pipeline(Cmd(setenv(ignorestatus(`$cli_bash $script HEAD $args`),
                                            merge(base, extra)); dir=root);
                                 stdout=out, stderr=devnull))
                (r.exitcode, String(take!(out)))
            end
            rc, out = deploy(joinpath(root, "releases"), Dict("SOLARSINDY_ACTIVATE" => "1"))
            @test rc == 0
            @test occursin("STUB webhook=https://hooks.example/x port=9100 mon=$filemon", out)
            rc, out = deploy(joinpath(root, "releases"),
                             Dict("SOLARSINDY_ACTIVATE" => "1", "SWM_PORT" => "7777"))
            @test rc == 0
            @test occursin("port=7777", out)                 # env wins over the file

            rc, _ = deploy("relreleases", Dict("SOLARSINDY_ACTIVATE" => "0"))
            @test rc == 0
            @test isdir(joinpath(root, "relreleases"))      # caller-relative, not clone-relative
            @test !ispath(joinpath(clone, "relreleases"))
            current = joinpath(root, "relreleases", "current")
            @test ispath(current) && isfile(joinpath(realpath(current), "RELEASE.json"))
        end
    end
end

# G12-07: the static stack receipt collector's launchd streams live outside $MONITOR_DIR/logs; the watchdog
# must size-bound them too, and the installer must render their directory into the watchdog job.
@testset "watchdog bounds the static stack receipt-collector streams" begin
    cli_bash = Sys.isunix() ? Sys.which("bash") : nothing
    if cli_bash !== nothing
        mktempdir() do root
            repo = normpath(joinpath(@__DIR__, ".."))
            mon = joinpath(root, "mon"); rlogs = joinpath(root, "rlogs"); mkpath(rlogs)
            write(joinpath(rlogs, "launchd.out"), repeat("x", 5000))
            write(joinpath(rlogs, "launchd.err"), repeat("x", 500))
            env = merge(ENV, Dict("SOLARSINDY_MONITOR_DIR" => mon,
                "SOLARSINDY_L1_RECEIPT_LOG_DIR" => rlogs,
                "SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES" => "1000",
                "SOLARSINDY_WATCHDOG_DASH_URL" => "http://127.0.0.1:1/api/health"))
            delete!(env, "SWM_WEBHOOK_URL")
            run(pipeline(setenv(ignorestatus(`$cli_bash $(joinpath(repo, "deploy", "watchdog.sh"))`), env);
                         stdout=devnull, stderr=devnull))
            @test filesize(joinpath(rlogs, "launchd.out")) == 0          # oversized: truncated
            @test filesize(joinpath(rlogs, "launchd.out.1")) == 5000     # previous copy kept
            @test filesize(joinpath(rlogs, "launchd.err")) == 500        # under the bound

            stub_bin = joinpath(root, "bin"); mkpath(stub_bin)
            for tool in ("plutil", "launchctl", "julia")
                write(joinpath(stub_bin, tool), "#!/bin/sh\nexit 0\n")
                chmod(joinpath(stub_bin, tool), 0o755)
            end
            home = joinpath(root, "home"); mkpath(home)
            ienv = merge(ENV, Dict("HOME" => home, "SOLARSINDY_LOAD" => "0",
                "PATH" => string(stub_bin, ":", get(ENV, "PATH", "")),
                "SOLARSINDY_JULIA" => joinpath(stub_bin, "julia"),
                "SOLARSINDY_MONITOR_DIR" => mon, "SOLARSINDY_L1_RECEIPT_LOG_DIR" => rlogs))
            delete!(ienv, "SWM_WEBHOOK_URL")
            installer = joinpath(repo, "deploy", "install_launchd.sh")
            @test success(pipeline(setenv(`$cli_bash $installer watchdog`, ienv);
                                   stdout=devnull, stderr=devnull))
            plist = read(joinpath(home, "Library", "LaunchAgents",
                                  "com.empire.solarsindy.watchdog.plist"), String)
            @test occursin("<key>SOLARSINDY_L1_RECEIPT_LOG_DIR</key>\n    <string>$rlogs</string>", plist)
        end
    end
end

@testset "Pinned release rejects failed environments" begin
    if Sys.isunix()
        mktempdir() do root
            clone = joinpath(root, "clone")
            mkpath(joinpath(clone, "deploy")); mkpath(joinpath(clone, "app"))
            cp(joinpath(@__DIR__, "../deploy/deploy_release.sh"),
               joinpath(clone, "deploy/deploy_release.sh"))
            write(joinpath(clone, "Project.toml"), "[deps]\n")
            write(joinpath(clone, "app/Project.toml"), "[deps]\n")
            run(`git -C $clone init --quiet`)
            run(`git -C $clone add .`)
            run(`git -C $clone -c user.name=Test -c user.email=test@example.invalid commit -qm fixture`)
            shim = joinpath(root, "julia")
            write(shim, raw"""#!/bin/bash
set -euo pipefail
case "$*" in
  *VERSION*) exit "${TEST_VERSION_EXIT:-0}" ;;
esac
for arg in "$@"; do
  case "$arg" in --project=*/app) exit "${TEST_APP_EXIT:-0}" ;; esac
done
exit 0
""")
            chmod(shim, 0o755)
            script = joinpath(clone, "deploy/deploy_release.sh")
            for (name, version_exit, app_exit, expected) in
                    (("version", 7, 0, 1), ("app", 0, 42, 42),
                     ("missing", 0, 0, 1), ("valid", 0, 0, 0))
                releases = joinpath(root, name)
                previous = joinpath(releases, "previous"); mkpath(previous)
                current = joinpath(releases, "current"); symlink(previous, current)
                cmd = addenv(`bash $script HEAD $releases`,
                    "SOLARSINDY_JULIA"=>(name == "missing" ? joinpath(root,"missing-julia") : shim),
                    "SOLARSINDY_MONITOR_DIR"=>joinpath(root,"state"),
                    "SOLARSINDY_INSTANTIATE"=>"1", "SOLARSINDY_ACTIVATE"=>"0",
                    "TEST_VERSION_EXIT"=>string(version_exit), "TEST_APP_EXIT"=>string(app_exit))
                result = run(pipeline(ignorestatus(cmd); stdout=devnull, stderr=devnull))
                @test result.exitcode == expected
                @test (readlink(current) == previous) == (expected != 0)
                expected == 0 && @test isfile(joinpath(realpath(current), "RELEASE.json"))
            end
        end
    end
end

# G12-06: under plain docker compose `restart: always` never acts on an unhealthy HEALTHCHECK; the
# `autoheal` sidecar runs deploy/autoheal.sh against the Docker socket. Exercised here against a
# stub `docker` that emulates `ps -q --filter` (AND of filters) over a fake container table:
# only labelled, unhealthy containers of this compose project are restarted (`-t 30`), a failed
# restart or a failed `docker ps` never stops the loop, and AUTOHEAL_ONCE=1 runs a single pass.
@testset "autoheal sidecar restarts only labelled unhealthy containers" begin
    ah_sh = Sys.isunix() ? Sys.which("sh") : nothing
    if ah_sh !== nothing
        script = normpath(joinpath(@__DIR__, "..", "deploy", "autoheal.sh"))
        stub = raw"""#!/bin/sh
echo "$*" >> "$STUB_DIR/calls"
cmd=$1; shift
case "$cmd" in
  ps)
    if [ -f "$STUB_DIR/ps_fail" ]; then rm -f "$STUB_DIR/ps_fail"; echo "Cannot connect to the Docker daemon" >&2; exit 1; fi
    lab=""; proj=""; health=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --filter) shift
          case "$1" in
            label=com.docker.compose.project=*) proj=${1#label=com.docker.compose.project=} ;;
            label=*) lab=${1#label=} ;;
            health=*) health=${1#health=} ;;
            *) echo "stub: unknown filter $1" >&2; exit 2 ;;
          esac ;;
      esac
      shift
    done
    while read -r id h p l; do
      [ -n "$health" ] && [ "$h" != "$health" ] && continue
      [ -n "$proj" ] && [ "$p" != "$proj" ] && continue
      [ -n "$lab" ] && [ "$l" != "$lab" ] && continue
      echo "$id"
    done < "$STUB_DIR/containers"
    ;;
  restart)
    for a in "$@"; do id=$a; done
    echo "$*" >> "$STUB_DIR/restarts"
    grep -qx "$id" "$STUB_DIR/restart_fail" 2>/dev/null && { echo "stub: restart failed" >&2; exit 1; }
    echo "$id"
    ;;
  inspect)
    for a in "$@"; do id=$a; done
    case "$*" in
      *com.docker.compose.project*)
        row=$(grep "^$id " "$STUB_DIR/containers") || { echo "No such container" >&2; exit 1; }
        set -- $row; echo "$3" ;;
      *) echo "/c-$id" ;;
    esac
    ;;
  *) exit 2 ;;
esac
"""
        table = """
            5e1f00 none proj -
            f0f666 unhealthy proj com.solarsindy.autoheal=true
            aaa111 unhealthy proj com.solarsindy.autoheal=true
            bbb222 healthy proj com.solarsindy.autoheal=true
            ccc333 unhealthy proj -
            ddd444 unhealthy other com.solarsindy.autoheal=true
            eee555 starting proj com.solarsindy.autoheal=true
            """
        function stub_env(dir; extra...)
            mkpath(joinpath(dir, "bin"))
            write(joinpath(dir, "bin", "docker"), stub)
            chmod(joinpath(dir, "bin", "docker"), 0o755)
            write(joinpath(dir, "containers"), join(strip.(split(strip(table), '\n')), '\n') * "\n")
            write(joinpath(dir, "restart_fail"), "f0f666\n")
            env = merge(ENV, Dict("STUB_DIR" => dir,
                "PATH" => joinpath(dir, "bin") * (Sys.iswindows() ? ";" : ":") * get(ENV, "PATH", "")))
            for k in ("AUTOHEAL_ONCE", "AUTOHEAL_PROJECT", "AUTOHEAL_LABEL", "AUTOHEAL_INTERVAL_SEC")
                delete!(env, k)
            end
            for (k, v) in extra
                env[string(k)] = string(v)
            end
            return env
        end
        restarts(dir) = isfile(joinpath(dir, "restarts")) ?
            readlines(joinpath(dir, "restarts")) : String[]
        # AUTOHEAL_ONCE=1 must exit by itself; bounded so a regression fails instead of hanging.
        function run_once(env, logfile)
            proc = open(logfile, "w") do io
                run(pipeline(setenv(ignorestatus(`$ah_sh $script`), env); stdout = io, stderr = io);
                    wait = false)
            end
            deadline = time() + 30
            while time() < deadline && process_running(proc)
                sleep(0.1)
            end
            exited = process_exited(proc)
            exited || (kill(proc); wait(proc))
            return exited && proc.exitcode == 0
        end

        # Single scoped pass: own project resolved from this container's label (HOSTNAME = id).
        mktempdir() do dir
            env = stub_env(dir; HOSTNAME = "5e1f00", AUTOHEAL_ONCE = 1)
            @test run_once(env, joinpath(dir, "log"))
            ahlog = read(joinpath(dir, "log"), String)
            @test restarts(dir) == ["-t 30 f0f666", "-t 30 aaa111"]
            @test occursin("restart of f0f666 (/c-f0f666) failed", ahlog)
            @test occursin("restarted aaa111 (/c-aaa111)", ahlog)
            @test any(l -> occursin("label=com.docker.compose.project=proj", l),
                      readlines(joinpath(dir, "calls")))
        end
        # No resolvable project: the label alone scopes the restarts.
        mktempdir() do dir
            env = stub_env(dir; HOSTNAME = "", AUTOHEAL_ONCE = 1)
            @test run_once(env, joinpath(dir, "log"))
            @test sort(last.(split.(restarts(dir)))) == ["aaa111", "ddd444", "f0f666"]
        end
        # A failing `docker ps` is logged and the loop keeps running.
        mktempdir() do dir
            env = stub_env(dir; HOSTNAME = "5e1f00", AUTOHEAL_INTERVAL_SEC = 1)
            touch(joinpath(dir, "ps_fail"))
            logfile = joinpath(dir, "log")
            proc = open(logfile, "w") do io
                run(pipeline(setenv(`$ah_sh $script`, env); stdout = io, stderr = io); wait = false)
            end
            deadline = time() + 30
            while time() < deadline && !("-t 30 aaa111" in restarts(dir))
                sleep(0.2)
            end
            @test process_running(proc)
            kill(proc)
            wait(proc)
            @test occursin("docker ps failed", read(logfile, String))
            @test "-t 30 aaa111" in restarts(dir)
        end
    end
end
