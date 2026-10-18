# Mandatory inert lifecycle coverage, including Windows (legacy Unix smoke stays intact).
using TOML
using Sockets

isdefined(@__MODULE__, :solarsindy_test_bash) || include(joinpath(@__DIR__, "bash_runtime.jl"))

@testset "CLI inert process identity and lifecycle" begin
    bash = solarsindy_test_bash()
    source_root = normpath(joinpath(@__DIR__, ".."))
    logs = get(ENV, "SOLARSINDY_CLI_TEST_LOG_DIR", "")
    if isempty(logs)
        logs = mktempdir(; prefix="solarsindy-cli-process-", cleanup=false)
    else
        ispath(logs) && !isempty(readdir(logs)) && error("CLI test log directory must be empty")
        mkpath(logs)
    end
    @info "inert CLI process transcripts" logs bash
    clone = joinpath(logs, "clone with spaces")
    for dir in ("bin", "examples", joinpath("app", "src"))
        mkpath(joinpath(clone, dir))
    end
    for file in ("solarsindy", "cli_process_identity.jl")
        cp(joinpath(source_root, "bin", file), joinpath(clone, "bin", file))
        @test read(joinpath(clone, "bin", file)) == read(joinpath(source_root, "bin", file))
    end
    # This is an inert bounded main, not a numerical or live-network proxy.
    monitor_source = raw"""
open(joinpath(ENV["SOLARSINDY_MONITOR_DIR"], "monitor-main.txt"), "a") do io
    println(io, getpid())
end
sleep(parse(Float64, get(ENV, "SOLARSINDY_INERT_BANNER_DELAY_SEC", "0")))
gate = get(ENV, "SOLARSINDY_INERT_BANNER_GATE", "")
deadline = time() + 30
while !isempty(gate) && !isfile(gate) && time() < deadline
    sleep(0.05)
end
isempty(gate) || isfile(gate) || error("inert banner gate exceeded 30 s")
write(joinpath(ENV["SOLARSINDY_MONITOR_DIR"], "monitor-banner.txt"), "written\n")
println("interval=600 INERT CLI TEST"); flush(stdout)
sleep(300)
"""
    dashboard_source = raw"""
using Sockets
open(joinpath(dirname(ENV["SOLARSINDY_LOG"]), "dashboard-main.txt"), "a") do io
    println(io, getpid())
end
listener = listen(ip"127.0.0.1", parse(Int, ENV["SWM_PORT"]))
@async begin
    sleep(300)
    exit()
end
while true
    client = accept(listener)
    try
        readuntil(client, "\r\n\r\n")
        write(client, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
    finally
        close(client)
    end
end
"""
    write(joinpath(clone, "examples", "live_monitor.jl"), monitor_source)
    write(joinpath(clone, "app", "src", "server.jl"), dashboard_source)
    state = joinpath(logs, "state with spaces"); mkpath(state)
    cli = joinpath(clone, "bin", "solarsindy")
    helper = joinpath(clone, "bin", "cli_process_identity.jl")
    mon = joinpath(clone, "examples", "live_monitor.jl")
    pidfile = joinpath(state, "run", "monitor.pid")
    lockfile = joinpath(state, "run", ".monitor.start.lock")
    listener = listen(ip"127.0.0.1", 0)
    port = getsockname(listener)[2]; close(listener)
    env = copy(ENV)
    merge!(env, Dict("SOLARSINDY_MONITOR_DIR"=>state, "SOLARSINDY_IGNORE_SERVICE"=>"1",
        "SOLARSINDY_ENV_FILE"=>"/dev/null", "SOLARSINDY_NO_OPEN"=>"1", "SWM_PORT"=>string(port),
        "JULIA"=>joinpath(Sys.BINDIR, Base.julia_exename())))
    rows = Dict{String,Any}[]
    helper_rows = Dict{String,Any}[]
    function launch(args; extra=Dict{String,String}(), timeout=75, script=cli, cwd=nothing)
        index = length(rows) + 1
        out = joinpath(logs, "$(index).stdout.log")
        err = joinpath(logs, "$(index).stderr.log")
        started = time()
        command = setenv(ignorestatus(`$bash $script $args`), merge(env, extra))
        cwd === nothing || (command = Cmd(command; dir=cwd))
        io_out, io_err = open(out, "w"), open(err, "w")
        process = run(pipeline(command; stdout=io_out, stderr=io_err); wait=false)
        row = Dict{String,Any}("args"=>args, "script"=>script, "stdout"=>basename(out), "stderr"=>basename(err),
                               "native_pid"=>getpid(process))
        push!(rows, row)
        return (; process, row, started, io_out, io_err, timeout, out, err)
    end
    function finish(job)
        while process_running(job.process) && time() - job.started < job.timeout
            sleep(0.05)
        end
        if process_running(job.process)
            kill(job.process); wait(job.process)
            close(job.io_out); close(job.io_err)
            error("owned inert CLI exceeded $(job.timeout) s; logs retained at $logs")
        end
        wait(job.process); close(job.io_out); close(job.io_err)
        job.row["owned_process_wait_confirmed"] = process_exited(job.process)
        @test process_exited(job.process)
        job.row["exit"] = job.process.exitcode
        job.row["seconds"] = time() - job.started
        return (; code=job.process.exitcode, out=read(job.out, String), err=read(job.err, String))
    end
    call(args; kwargs...) = finish(launch(args; kwargs...))
    function helper_call(action; path=pidfile, root=clone, entry=mon)
        out, err = IOBuffer(), IOBuffer()
        started = time()
        p = run(pipeline(ignorestatus(`$(env["JULIA"]) --startup-file=no $helper $action $path $root $entry`);
                         stdout=out, stderr=err); wait=false)
        native_pid = getpid(p)
        while process_running(p) && time() - started < 20
            sleep(0.05)
        end
        if process_running(p)
            kill(p); wait(p)
            error("owned identity helper exceeded 20 s")
        end
        wait(p)
        @test process_exited(p)
        output, errors = String(take!(out)), String(take!(err))
        index = length(helper_rows) + 1
        write(joinpath(logs, "helper-$index.stdout.log"), output)
        write(joinpath(logs, "helper-$index.stderr.log"), errors)
        push!(helper_rows, Dict("command"=>action, "exit"=>p.exitcode, "seconds"=>time()-started,
                               "native_pid"=>native_pid, "owned_process_wait_confirmed"=>process_exited(p)))
        return (; code=p.exitcode, out=output, err=errors)
    end
    main_count() = isfile(joinpath(state, "monitor-main.txt")) ? length(readlines(joinpath(state, "monitor-main.txt"))) : 0
    try
        @test success(`$bash -n $cli`)
        # Controlled exact-function tests exercise read/query failures in the Unix
        # branch on every Bash host. This is shell control-flow coverage, not a
        # claim that a Windows host has executed a native Unix process query.
        script_text = read(cli, String)
        function source_section(start_marker, end_marker)
            a = Base.first(findfirst(start_marker, script_text))
            b = Base.first(findnext(end_marker, script_text, a))
            script_text[a:prevind(script_text, b)]
        end
        predicate = source_section("unix_path_matches_fd() {", "\nunix_reservation_matches()")
        for kind in ("valid", "wrong-device", "wrong-inode", "wrong-type", "garbage", "partial-error",
                     "missing-separators", "extra-newline", "query-error", "unknown-platform")
            probe_dir = joinpath(logs, "controlled-fd-$kind"); mkpath(probe_dir)
            probe = joinpath(probe_dir, "probe.sh")
            prefix = "#!/bin/bash\nset -euo pipefail\nRUN_DIR=\"\$1\"\nMODE=\"\$2\"\nUNIX_FD_PLATFORM=Darwin\n"
            # Change only the native metadata primitive in this controlled copy.
            # Native Darwin stdin-fstat versus independent fstat is a separate CI oracle.
            controlled = replace(predicate, "/usr/bin/stat"=>"cli_probe_stat")
            seam = raw"""
cli_probe_stat() {
  printf 'queried\n' >> "$RUN_DIR/queries.txt"
  case "$MODE" in
    query-error) return 77 ;;
    partial-error) printf '1:2:100644\n'; return 77 ;;
    garbage) printf 'not-metadata\n'; return 0 ;;
    missing-separators) printf '100644\n'; return 0 ;;
    extra-newline) printf '1:2:100644\n\n'; return 0 ;;
    wrong-type) printf '1:2:40700\n'; return 0 ;;
  esac
  if [ "$#" -eq 2 ]; then printf '1:2:100644\n'; else
    case "$MODE" in wrong-device) printf '9:2:100644\n' ;; wrong-inode) printf '1:9:100644\n' ;; *) printf '1:2:100644\n' ;; esac
  fi
}
"""
            body = raw"""
[ "$MODE" != unknown-platform ] || UNIX_FD_PLATFORM=UNKNOWN
inspect() {
  printf 'preserve-owned-bytes\n' >&9
  rc=0; unix_path_matches_fd "$RUN_DIR/owned.txt" 9 || rc=$?
  printf 'IDENTITY_RETURN=%s\n' "$rc"
}
inspect 9> "$RUN_DIR/owned.txt"
"""
            write(probe, prefix * controlled * "\n" * seam * body)
            result = call([probe_dir, kind]; script=probe, timeout=10)
            @test result.code == 0
            @test occursin("IDENTITY_RETURN=$(kind == "valid" ? 0 : 1)", result.out)
            @test read(joinpath(probe_dir, "owned.txt"), String) == "preserve-owned-bytes\n"
            @test isfile(joinpath(probe_dir, "queries.txt")) == (kind != "unknown-platform")
        end
        for kind in ("closed", "different-file")
            probe_dir = joinpath(logs, "native-fd-$kind"); mkpath(probe_dir)
            probe = joinpath(probe_dir, "probe.sh")
            prefix = "#!/bin/bash\nset -euo pipefail\nRUN_DIR=\"\$1\"\nMODE=\"\$2\"\nUNIX_FD_PLATFORM=\"\$(uname -s)\"\n"
            body = raw"""
inspect() {
  printf 'first-owned-inode\n' >&9
  printf 'second-owned-inode\n' > "$RUN_DIR/other.txt"
  target="$RUN_DIR/other.txt"
  if [ "$MODE" = closed ]; then exec 9>&-; target="$RUN_DIR/owned.txt"; fi
  rc=0; unix_path_matches_fd "$target" 9 || rc=$?
  printf 'IDENTITY_RETURN=%s\n' "$rc"
}
inspect 9> "$RUN_DIR/owned.txt"
"""
            write(probe, prefix * predicate * "\n" * body)
            result = call([probe_dir, kind]; script=probe, timeout=10)
            @test result.code == 0
            @test occursin("IDENTITY_RETURN=1", result.out)
            @test read(joinpath(probe_dir, "owned.txt"), String) == "first-owned-inode\n"
            @test read(joinpath(probe_dir, "other.txt"), String) == "second-owned-inode\n"
        end
        functions = source_section("unix_pid_absent() {", "\n# pid_of FILE") * "\n" *
                    source_section("pid_of() {", "\n# Exclusive per-component") * "\n" *
                    source_section("acquire_start_lock() {", "\nrelease_start_lock()")
        public_pid_functions = source_section("unix_pid_absent() {", "\n# pid_of FILE") * "\n" *
                               source_section("pid_of() {", "\n# Exclusive per-component")
        public_fixtures = (("valid", "1234\n", 0), ("valid-no-lf", "1234", 0),
            ("boundary-valid-lf", repeat("1", 127) * "\n", 0),
            ("boundary-valid-no-lf", repeat("1", 128), 0),
            ("boundary-oversize-lf", repeat("1", 128) * "\n", 1),
            ("embedded-space", "12 34\n", 1), ("leading-space", " 1234\n", 1),
            ("trailing-space", "1234 \n", 1), ("tab", "12\t34\n", 1), ("cr", "1234\r", 1),
            ("crlf", "1234\r\n", 1), ("empty", "", 1), ("newline", "\n", 1),
            ("extra-newline", "1234\n\n", 1), ("second-line", "1234\n5678\n", 1),
            ("alphabetic", "1234x\n", 1), ("sign", "+1234\n", 1), ("zero", "0\n", 1),
            ("leading-zero", "01234\n", 1), ("nul", "12\0\x33\x34\n", 1),
            ("oversize", repeat("1", 129), 1), ("large", repeat("1", 129), 1),
            ("read1", "1234\n", 2), ("read2", "1234\n", 2),
            ("cleanup-error", "12 34\n", 2))
        for (kind, bytes, expected) in public_fixtures
            probe_dir = joinpath(logs, "public-pid-$kind"); mkpath(probe_dir)
            ownerfile = joinpath(probe_dir, "owned.pid"); write(ownerfile, bytes)
            if kind == "large"
                open(ownerfile, "r+") do io; truncate(io, 64 * 1024 * 1024); end
                @test filesize(ownerfile) == 64 * 1024 * 1024
            end
            probe = joinpath(probe_dir, "probe.sh")
            prefix = raw"""
#!/bin/bash
set -euo pipefail
WINDOWS_PROCESS=0; RUN_DIR="$1"; CLI_PROBE_KIND="$2"
err() { printf '%s\n' "$*" >&2; }
ps() {
  printf '%s\n' "$*" >> "$RUN_DIR/queries.txt"
  printf 'owned /owned/entrypoint\n'
}
head() {
  local count=0
  [ "$#" -eq 3 ] && [ "$1" = -c ] && [ "$2" = 129 ] || return 90
  [ ! -f "$RUN_DIR/heads.txt" ] || count="$(wc -l < "$RUN_DIR/heads.txt")"
  printf '%s\n' bounded129 >> "$RUN_DIR/heads.txt"
  if { [ "$CLI_PROBE_KIND" = read1 ] && [ "$count" -eq 0 ]; } ||
     { [ "$CLI_PROBE_KIND" = read2 ] && [ "$count" -eq 1 ]; }; then return 13; fi
  command head "$@"
}
rm() { [ "$CLI_PROBE_KIND" != cleanup-error ] || return 73; command rm "$@"; }
"""
            body = raw"""
rc=0; admitted="$(pid_of "$RUN_DIR/owned.pid" /owned/entrypoint)" || rc=$?
printf 'PID_RETURN=%s\nADMITTED=%s\n' "$rc" "$admitted"
"""
            write(probe, prefix * public_pid_functions * "\n" * body)
            result = call([probe_dir, kind]; script=probe, timeout=15)
            @test result.code == 0
            @test occursin("PID_RETURN=$expected", result.out)
            admitted = endswith(bytes, "\n") ? chop(bytes) : bytes
            @test occursin(expected == 0 ? "ADMITTED=$admitted\n" : "ADMITTED=\n", result.out)
            @test isfile(joinpath(probe_dir, "queries.txt")) == (expected == 0)
            if expected == 0
                @test length(readlines(joinpath(probe_dir, "queries.txt"))) == 1
            end
            @test isfile(ownerfile) == (expected != 1)
            if expected != 1
                @test read(ownerfile, String) == bytes
            end
            reads = readlines(joinpath(probe_dir, "heads.txt"))
            @test 1 <= length(reads) <= 2
            @test all(==("bounded129"), reads)
        end
        if Sys.isunix()
            native_uid = ccall(:geteuid, Cuint, ())
            @info "native Unix PID-parent permission coverage" native_uid
            permission_cases = native_uid == 0 ? ("missing", "nested-missing") :
                               ("direct", "nested", "missing", "nested-missing")
            for kind in permission_cases
                probe_dir = joinpath(logs, "public-pid-parent-$kind"); mkpath(probe_dir)
                blocked = joinpath(probe_dir, "locked")
                denied = kind in ("direct", "nested")
                ownerfile = kind in ("nested", "nested-missing") ? joinpath(blocked, "inner", "owned.pid") :
                            joinpath(blocked, "owned.pid")
                saved_mode = 0
                if denied
                    mkpath(dirname(ownerfile)); write(ownerfile, "1234\n")
                    saved_mode = stat(blocked).mode & 0o7777
                end
                probe = joinpath(probe_dir, "probe.sh")
                prefix = raw"""
#!/bin/bash
set -euo pipefail
WINDOWS_PROCESS=0; CLI_PROBE_STATE="$1"; CLI_PROBE_PIDFILE="$2"
err() { printf '%s\n' "$*" >&2; }
ps() { printf 'queried\n' >> "$CLI_PROBE_STATE/queries.txt"; printf 'owned /owned/entrypoint\n'; }
"""
                body = raw"""
rc=0; admitted="$(pid_of "$CLI_PROBE_PIDFILE" /owned/entrypoint)" || rc=$?
printf 'PID_RETURN=%s\nADMITTED=%s\n' "$rc" "$admitted"
"""
                write(probe, prefix * public_pid_functions * "\n" * body)
                try
                    denied && chmod(blocked, 0o000)
                    result = call([probe_dir, ownerfile]; script=probe, timeout=15)
                    @test result.code == 0
                    @test occursin("PID_RETURN=$(denied ? 2 : 1)", result.out)
                    @test occursin("ADMITTED=\n", result.out)
                    @test !isfile(joinpath(probe_dir, "queries.txt"))
                finally
                    denied && chmod(blocked, saved_mode)
                end
                @test isfile(ownerfile) == denied
                if denied
                    @test read(ownerfile, String) == "1234\n"
                    @test (stat(blocked).mode & 0o7777) == saved_mode
                end
            end
        end
        for failure in ("read", "query")
            probe_dir = joinpath(logs, "controlled-$failure"); mkpath(probe_dir)
            probe = joinpath(probe_dir, "probe.sh")
            prefix = "#!/bin/bash\nset -euo pipefail\nWINDOWS_PROCESS=0\nROOT=/owned/clone\nRUN_DIR=\"\$1\"\nSTART_LOCK=\"\"\nerr() { printf '%s\\n' \"\$*\" >&2; }\n"
            seam = failure == "read" ? "head() { return 13; }\n" : "ps() { return 77; }\n"
            body = raw"""
printf '1234\n' > "$RUN_DIR/owned.pid"
printf '1234\n' > "$RUN_DIR/.monitor.start.lock"
if result="$(pid_of "$RUN_DIR/owned.pid" /owned/entrypoint)"; then rc=0; else rc=$?; fi
printf 'PID_RETURN=%s\n' "$rc"
[ -f "$RUN_DIR/owned.pid" ] && echo PRESERVED=1 || echo PRESERVED=0
if acquire_start_lock monitor; then rc=0; else rc=$?; fi
printf 'LOCK_RETURN=%s\n' "$rc"
printf 'OWNER='; cat "$RUN_DIR/.monitor.start.lock"
"""
            write(probe, prefix * seam * functions * "\n" * body)
            result = call([probe_dir]; script=probe, timeout=10)
            @test result.code == 0
            output = result.out
            write(joinpath(probe_dir, "stdout.log"), output)
            @test occursin("PID_RETURN=2", output)
            @test occursin("PRESERVED=1", output)
            @test occursin("LOCK_RETURN=2", output)
            @test occursin("OWNER=1234", output)
        end
        # Exact Unix admission functions with controlled process queries. All
        # live argv spellings hold the lock, even unrelated/recycled live PIDs;
        # only a successful independent absence query permits dead recovery.
        for kind in ("physical", "alias", "relative", "unrelated", "empty", "dead",
                     "multiline", "space", "oversize", "zero", "nul", "extra-newline",
                     "admission-empty", "admission-foreign", "rmdir-error")
            probe_dir = joinpath(logs, "controlled-lock-$kind"); mkpath(probe_dir)
            probe = joinpath(probe_dir, "probe.sh")
            prefix = "#!/bin/bash\nset -euo pipefail\nWINDOWS_PROCESS=0\nROOT=/owned/physical\nRUN_DIR=\"\$1\"\nSTART_LOCK=\"\"\nerr() { printf '%s\\n' \"\$*\" >&2; }\n"
            argv = Dict("physical"=>"bash /owned/physical/bin/solarsindy start monitor",
                        "alias"=>"bash /owned/alias/bin/solarsindy start monitor",
                        "relative"=>"bash bin/solarsindy start monitor", "unrelated"=>"sleep 60")
            seam = if haskey(argv, kind)
                "ps() { printf '%s\\n' '$(argv[kind])'; }\n"
            elseif kind == "dead"
                "ps() { if [ \"\$1\" = -e ]; then printf '9999\\n'; else return 1; fi; }\n"
            else
                "ps() { return 0; }\n"
            end
            bytes = kind == "multiline" ? "1234\n9999\n" : kind == "space" ? "12 34\n" :
                    kind == "oversize" ? repeat("1", 129) : kind == "zero" ? "0\n" :
                    kind == "nul" ? "12\0\x33\x34\n" : kind == "extra-newline" ? "1234\n\n" : "1234\n"
            ownerfile = joinpath(probe_dir, ".monitor.start.lock"); write(ownerfile, bytes)
            admission = ownerfile * ".admission"
            if startswith(kind, "admission-")
                mkpath(admission)
                kind == "admission-foreign" && write(joinpath(admission, "foreign.txt"), "preserve foreign content\n")
            elseif kind == "rmdir-error"
                seam *= "rmdir() { return 75; }\n"
            end
            body = raw"""
if acquire_start_lock monitor; then rc=0; else rc=$?; fi
printf 'LOCK_RETURN=%s\n' "$rc"
"""
            write(probe, prefix * seam * functions * "\n" * body)
            result = call([probe_dir]; script=probe, timeout=10)
            @test result.code == 0
            expected = haskey(argv, kind) ? 1 : kind == "dead" ? 0 : 2
            @test occursin("LOCK_RETURN=$expected", result.out)
            @test (read(ownerfile, String) == bytes) == (kind != "dead")
            @test isdir(admission) == (startswith(kind, "admission-") || kind == "rmdir-error")
            if kind == "admission-foreign"
                @test read(joinpath(admission, "foreign.txt"), String) == "preserve foreign content\n"
            end
        end
        # Existing nonregular/symlink lock state is unknown before either open.
        # A failure-path FIFO reader releases an old blocking writer so the
        # regression itself cannot orphan it; the fixed path never needs a reader.
        for kind in (Sys.isunix() ? ("directory", "fifo", "symlink") : ("directory",))
            probe_dir = joinpath(logs, "controlled-lock-type-$kind"); mkpath(probe_dir)
            ownerfile = joinpath(probe_dir, ".monitor.start.lock")
            targetfile = joinpath(probe_dir, "foreign-target.txt")
            write(targetfile, "1234\n")
            if kind == "directory"
                mkpath(ownerfile); write(joinpath(ownerfile, "foreign.txt"), "foreign-content\n")
            elseif kind == "symlink"
                symlink(targetfile, ownerfile)
            else
                creator = joinpath(probe_dir, "create-fifo.sh")
                write(creator, "#!/bin/bash\nset -euo pipefail\nmkfifo \"\$1\"\n[ -p \"\$1\" ]\n")
                @test call([ownerfile]; script=creator, timeout=10).code == 0
            end
            probe = joinpath(probe_dir, "probe.sh")
            prefix = "#!/bin/bash\nset -euo pipefail\nWINDOWS_PROCESS=0\nRUN_DIR=\"\$1\"\nSTART_LOCK=\"\"\nerr() { printf '%s\\n' \"\$*\" >&2; }\nps() { echo PS_MUST_NOT_RUN >&2; return 99; }\n"
            body = raw"""
if acquire_start_lock monitor; then rc=0; else rc=$?; fi
printf 'LOCK_RETURN=%s\n' "$rc"
[ ! -p "$RUN_DIR/.monitor.start.lock" ] || echo FIFO_PRESERVED=1
"""
            write(probe, prefix * functions * "\n" * body)
            caller = launch([probe_dir]; script=probe, timeout=10)
            reader = nothing
            reader_io = nothing
            reader_started = false
            reader_native_pid = 0
            try
                deadline = time() + 5
                while process_running(caller.process) && time() < deadline
                    sleep(0.05)
                end
                if kind == "fifo" && process_running(caller.process)
                    cat = Sys.which("cat"); cat === nothing && error("FIFO regression cleanup requires POSIX cat")
                    reader_io = open(joinpath(probe_dir, "failure-reader.log"), "w")
                    reader = run(pipeline(ignorestatus(`$cat $ownerfile`); stdout=reader_io, stderr=reader_io); wait=false)
                    reader_native_pid = getpid(reader)
                    reader_started = true
                end
                result = finish(caller)
                @test result.code == 0
                @test occursin("LOCK_RETURN=2", result.out)
                @test occursin("not a regular file", result.err)
                @test !occursin("PS_MUST_NOT_RUN", result.err)
                @test !reader_started
                @test read(targetfile, String) == "1234\n"
                @test !isdir(ownerfile * ".admission")
                if kind == "directory"
                    @test read(joinpath(ownerfile, "foreign.txt"), String) == "foreign-content\n"
                elseif kind == "symlink"
                    @test islink(ownerfile)
                else
                    @test occursin("FIFO_PRESERVED=1", result.out)
                end
            finally
                if !haskey(caller.row, "exit")
                    finish(caller)
                end
                if reader !== nothing
                    deadline = time() + 3
                    while process_running(reader) && time() < deadline; sleep(0.05); end
                    process_running(reader) && kill(reader)
                    wait(reader)
                    @test process_exited(reader)
                    push!(helper_rows, Dict("command"=>"failure-only-owned-fifo-reader", "exit"=>reader.exitcode,
                        "native_pid"=>reader_native_pid, "owned_process_wait_confirmed"=>process_exited(reader)))
                end
                reader_io === nothing || close(reader_io)
            end
        end
        # Keep two actual owned Bash processes alive while their start-lock
        # results are inspected. On Unix these use real ps; Windows supplies an
        # explicit /proc-based query seam for this Unix control-flow test.
        for race in ("stale", "fresh")
            probe_dir = joinpath(logs, "admission-race-$race"); mkpath(probe_dir)
            probe = joinpath(probe_dir, "probe.sh")
            prefix = raw"""
#!/bin/bash
set -euo pipefail
WINDOWS_PROCESS=0
ROOT=/owned/physical
RUN_DIR="$1"
CLI_PROBE_LABEL="$2"
START_LOCK=""
err() { printf '%s\n' "$*" >&2; }
rm() {
  printf 'before-remove\n' > "$RUN_DIR/$CLI_PROBE_LABEL.at-remove"
  deadline=$((SECONDS + 10))
  until [ -f "$RUN_DIR/$CLI_PROBE_LABEL.remove" ]; do
    [ "$SECONDS" -lt "$deadline" ] || return 76
    sleep 0.02
  done
  command rm "$@"
}
"""
            seam = Sys.iswindows() ? raw"""
# Controlled Unix query on Windows Bash; native Unix uses its real ps below.
ps() {
  if [ "$1" = -e ]; then printf '%s\n' "$$"; return 0; fi
  [ -r "/proc/$4/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$4/cmdline"
}
""" : ""
            body = raw"""
if acquire_start_lock monitor; then rc=0; else rc=$?; fi
printf 'RETURN=%s\nSELF=%s\n' "$rc" "$$"
printf '%s\n' "$$" > "$RUN_DIR/$CLI_PROBE_LABEL.pid"
printf '%s\n' "$rc" > "$RUN_DIR/$CLI_PROBE_LABEL.done"
if [ "$rc" -eq 0 ]; then
  deadline=$((SECONDS + 10))
  until [ -f "$RUN_DIR/$CLI_PROBE_LABEL.exit" ]; do
    [ "$SECONDS" -lt "$deadline" ] || exit 77
    sleep 0.02
  done
fi
"""
            write(probe, prefix * seam * functions * "\n" * body)
            numeric = joinpath(probe_dir, ".monitor.start.lock")
            if race == "stale"
                dead = launch(["exit 0"]; script="-c", timeout=10)
                dead_pid = getpid(dead.process)
                @test finish(dead).code == 0
                write(numeric, string(Sys.iswindows() ? 8675309 : dead_pid, "\n"))
            end
            function await_file(file)
                deadline = time() + 7
                while !isfile(file) && time() < deadline
                    sleep(0.01)
                end
                isfile(file) || error("owned admission barrier absent: $file")
            end
            callers = Any[]
            try
                push!(callers, launch([probe_dir, "A"]; script=probe, timeout=25))
                race == "stale" && await_file(joinpath(probe_dir, "A.at-remove"))
                push!(callers, launch([probe_dir, "B"]; script=probe, timeout=25))
                if race == "stale"
                    await_file(joinpath(probe_dir, "B.done"))
                    @test strip(read(joinpath(probe_dir, "B.done"), String)) == "2"
                    @test !isfile(joinpath(probe_dir, "B.at-remove"))
                    @test read(numeric, String) == string(Sys.iswindows() ? 8675309 : dead_pid, "\n")
                    write(joinpath(probe_dir, "A.remove"), "release\n")
                end
                for label in ("A", "B"); await_file(joinpath(probe_dir, "$label.done")); end
                returns = [parse(Int, strip(read(joinpath(probe_dir, "$label.done"), String))) for label in ("A", "B")]
                @test count(==(0), returns) == 1
                @test all(rc -> rc in (0, 1, 2), returns)
                winner = ("A", "B")[only(findall(==(0), returns))]
                @test read(numeric) == read(joinpath(probe_dir, "$winner.pid"))
                @test process_running(callers[winner == "A" ? 1 : 2].process)
                @test !isdir(numeric * ".admission")
            finally
                for label in ("A", "B"), action in ("remove", "exit")
                    write(joinpath(probe_dir, "$label.$action"), "release\n")
                end
                for caller in callers
                    @test finish(caller).code == 0
                end
            end
        end
        # Execute the exact Unix gated-spawn functions on every real Bash host.
        # Windows uses inert Bash as the command oracle; native Unix additionally
        # verifies actual Julia exec-PID continuity in the lifecycle below.
        gate_functions = source_section("UNIX_NEW_PID=\"\"", "\nhealth_json()")
        gate_functions = replace(gate_functions,
            "unix_clean_start_gate() {"=>"original_unix_clean_start_gate() {",
            "unix_write_pid() {"=>"original_unix_write_pid() {",
            "unix_spawn_in_gate() {"=>"original_unix_spawn_in_gate() {")
        for kind in ("success", "write-failure", "replaced-pid", "mktemp-failure",
                     "ack-before-directory", "ack-before-file", "ack-replaced-directory",
                     "ack-replaced-file", "ack-readonly", "foreign-marker", "foreign-directory",
                     "cleanup-failure", "cancel-failure", "descriptor-preflight", "token-write-failure",
                     "helper-extraction-error", "helper-extraction-empty")
            probe_dir = joinpath(logs, "controlled-spawn-$kind"); mkpath(joinpath(probe_dir, "run"))
            probe = joinpath(probe_dir, "probe.sh")
            inert = joinpath(probe_dir, "inert main.sh")
            shim = joinpath(probe_dir, "inert command.sh")
            write(inert, raw"""
#!/bin/bash
printf '%s\n' "$$" > "$CLI_PROBE_STATE/main.pid"
for descriptor in 5 6 7 8 9; do
  [ ! -e "/dev/fd/$descriptor" ] || exit 87
done
printf 'completed\n' > "$CLI_PROBE_STATE/main.completed"
""")
            write(shim, "#!/bin/bash\nshift 2\nexec \"\$BASH\" \"\$@\"\n")
            chmod(shim, 0o755)
            prefix = raw"""
#!/bin/bash
set -euo pipefail
export CLI_PROBE_STATE="$1"
RUN_DIR="$1/run"; JULIA="$3"
CLI_PROBE_PIDFILE="$RUN_DIR/component.pid"
err() { printf '%s\n' "$*" >&2; }
"""
            seam = raw"""
unix_spawn_in_gate() { original_unix_spawn_in_gate "$@"; }
unix_clean_start_gate() { original_unix_clean_start_gate "$@"; }
# Retain this owned child in the probe's job table so its exact exit can be waited.
# Production disowns it; launcher-exited native lifecycle is tested separately.
disown() { :; }
wait() {
  local status=0
  builtin wait "$@" || status=$?
  printf '%s:%s\n' "$1" "$status" > "$CLI_PROBE_STATE/owned-wait.txt"
  return "$status"
}
unix_write_pid() {
  printf '%s\n' "$1" > "$CLI_PROBE_STATE/child.pid"
  original_unix_write_pid "$@"
}
"""
            if kind in ("write-failure", "replaced-pid", "cancel-failure")
                seam *= "unix_write_pid() {\n  printf '%s\\n' \"\$1\" > \"\$CLI_PROBE_STATE/child.pid\"\n"
                if kind == "replaced-pid"
                    seam *= "  rm -f \"\$CLI_PROBE_PIDFILE\"\n  mkdir \"\$CLI_PROBE_PIDFILE\"\n  printf 'foreign-content\\n' > \"\$CLI_PROBE_PIDFILE/foreign.txt\"\n"
                end
                seam *= "  return 73\n}\n"
                kind == "cancel-failure" && (seam *= "mv() { return 74; }\n")
            elseif kind == "mktemp-failure"
                seam *= "mktemp() { return 74; }\n"
            elseif startswith(kind, "ack-before-")
                seam *= "mktemp() { local gate; gate=\"\$(command mktemp \"\$@\")\" || return \$?\n"
                seam *= endswith(kind, "directory") ? "mkdir \"\$gate/acknowledged\"\n" :
                        "printf 'foreign-content\\n' > \"\$gate/acknowledged\"\n"
                seam *= "printf '%s\\n' \"\$gate\"\n}\n"
            elseif startswith(kind, "ack-replaced-")
                seam *= raw"""
unix_write_pid() {
  printf '%s\n' "$1" > "$CLI_PROBE_STATE/child.pid"
  original_unix_write_pid "$@" || return $?
  local gates=("$RUN_DIR"/.daemon-start.*)
  rm -f "${gates[0]}/acknowledged"
"""
                seam *= endswith(kind, "directory") ? "mkdir \"\${gates[0]}/acknowledged\"\n" :
                        "printf 'foreign-content\\n' > \"\${gates[0]}/acknowledged\"\n"
                seam *= "}\n"
            elseif kind in ("ack-readonly", "token-write-failure")
                descriptor = kind == "ack-readonly" ? "6" : "7"
                marker = kind == "ack-readonly" ? "acknowledged" : "ready-token"
                seam *= "unix_spawn_in_gate() { original_unix_spawn_in_gate \"\$@\" $descriptor< \"\$5/$marker\"; }\n"
            elseif kind in ("foreign-marker", "foreign-directory")
                seam *= "unix_clean_start_gate() {\n CLI_PROBE_FOREIGN_PATH=\"\$1\"\n"
                if kind == "foreign-directory"
                    seam *= "if mv \"\$1\" \"\$1.saved\"; then mkdir \"\$1\" || return 74; else echo DIRECTORY_REPLACEMENT_BLOCKED=1; return 2; fi\n"
                else
                    seam *= "rm -f \"\$1/allow\" || return 74\n"
                end
                seam *= "printf 'foreign-content\\n' > \"\$1/allow\" || return 74\n original_unix_clean_start_gate \"\$1\"\n}\n"
            elseif kind == "cleanup-failure"
                seam *= "unix_clean_start_gate() { return 74; }\n"
            elseif kind == "descriptor-preflight"
                seam *= "unix_reservation_matches() { return 2; }\n"
            elseif kind == "helper-extraction-error"
                seam *= "declare() { return 77; }\n"
            elseif kind == "helper-extraction-empty"
                seam *= "declare() { return 0; }\n"
            end
            body = raw"""
if unix_spawn "$CLI_PROBE_PIDFILE" "$2" "$CLI_PROBE_STATE" "$CLI_PROBE_STATE/main.log"; then rc=0; else rc=$?; fi
printf 'SPAWN_RETURN=%s\n' "$rc"
for descriptor in 5 6 7 8 9; do
  [ ! -e "/dev/fd/$descriptor" ] || exit 88
done
echo DESCRIPTORS_CLOSED=1
if [ -f "$CLI_PROBE_STATE/child.pid" ] && [ ! -f "$CLI_PROBE_STATE/owned-wait.txt" ]; then
  child="$(cat "$CLI_PROBE_STATE/child.pid")"
  wait "$child" || true
fi
if [ -n "${CLI_PROBE_FOREIGN_PATH:-}" ] && [ -f "$CLI_PROBE_FOREIGN_PATH/allow" ]; then echo FOREIGN_PRESERVED=1; fi
"""
            write(probe, prefix * gate_functions * "\n" * seam * body)
            result = call([probe_dir, inert, shim]; script=probe, timeout=45)
            @test result.code == 0
            @test occursin("SPAWN_RETURN=$(kind == "success" ? 0 : 2)", result.out)
            @test occursin("DESCRIPTORS_CLOSED=1", result.out)
            reached_main = kind in ("success", "foreign-marker", "foreign-directory", "cleanup-failure")
            @test isfile(joinpath(probe_dir, "main.completed")) == reached_main
            record = joinpath(probe_dir, "run", "component.pid")
            if reached_main
                @test read(record) == read(joinpath(probe_dir, "main.pid"))
            elseif kind in ("write-failure", "mktemp-failure", "ack-before-directory", "ack-before-file", "token-write-failure",
                            "helper-extraction-error", "helper-extraction-empty")
                @test !ispath(record)
            elseif kind == "replaced-pid"
                @test read(joinpath(record, "foreign.txt"), String) == "foreign-content\n"
            elseif kind in ("ack-replaced-directory", "ack-replaced-file", "ack-readonly")
                @test read(record) == read(joinpath(probe_dir, "child.pid"))
            end
            if isfile(joinpath(probe_dir, "child.pid"))
                expected_exit = kind in ("write-failure", "replaced-pid") ? 75 : kind == "cancel-failure" ? 77 :
                                startswith(kind, "ack-") ? 76 : 0
                @test strip(read(joinpath(probe_dir, "owned-wait.txt"), String)) ==
                      strip(read(joinpath(probe_dir, "child.pid"), String)) * ":$expected_exit"
            end
            if kind == "foreign-marker" || (kind == "foreign-directory" && Sys.isunix())
                @test occursin("FOREIGN_PRESERVED=1", result.out)
            elseif kind == "foreign-directory"
                @test occursin("DIRECTORY_REPLACEMENT_BLOCKED=1", result.out) || occursin("FOREIGN_PRESERVED=1", result.out)
            end
        end
        # A colliding PID directory must never authorize readiness or orphan a main.
        for target in ("monitor", "dashboard")
            collision_state = joinpath(logs, "$target-pid-directory state")
            collision = joinpath(collision_state, "run", "$target.pid"); mkpath(collision)
            write(joinpath(collision, "foreign.txt"), "preserve foreign directory\n")
            result = call(["start", target]; extra=Dict("SOLARSINDY_MONITOR_DIR"=>collision_state))
            @test result.code != 0
            @test !occursin("RUNNING", result.out)
            @test read(joinpath(collision, "foreign.txt"), String) == "preserve foreign directory\n"
            @test !isfile(joinpath(collision_state, "$target-main.txt"))
        end
        if Sys.isunix()
            # Native filesystem oracle: stale parsing must not unlink a symlink.
            # The malformed regular-file cleanup remains a separate legacy case.
            for target in ("monitor", "dashboard")
                link_state = joinpath(logs, "$target-pid-symlink state"); mkpath(joinpath(link_state, "run"))
                destination = joinpath(link_state, "foreign-state.txt")
                write(destination, "malformed foreign state\n")
                link = joinpath(link_state, "run", "$target.pid"); symlink(destination, link)
                refused = call(["start", target]; extra=Dict("SOLARSINDY_MONITOR_DIR"=>link_state))
                @test refused.code != 0
                @test islink(link)
                @test read(destination, String) == "malformed foreign state\n"
                @test !isfile(joinpath(link_state, "$target-main.txt"))
            end
        end
        # Inject only shell utility failure, through Bash's supported startup
        # file. Execute the real CLI on each OS; no daemon/pending record may be
        # prepared and the original appendable stale-banner log must survive.
        for target in ("monitor", "dashboard"), fault in ("rotation", "mkdir", "lock-mkdir", "rmdir", "foreign-admission")
            fault_state = joinpath(logs, "$target-$fault state"); mkpath(joinpath(fault_state, "logs"))
            old_log = joinpath(fault_state, "logs", "cli-$target.log")
            original_log = "interval=STALE-OLD-BANNER\n"; write(old_log, original_log)
            seam = joinpath(logs, "$target-$fault.bash-env")
            if fault == "rotation"
                write(seam, "mv() { printf 'INJECTED_MV_ERROR=73\\n' >&2; return 73; }\n")
            elseif fault == "rmdir"
                # Unix admission cleanup only. Windows OS dispatch is explicitly
                # controlled here; this does not exercise the native Windows helper.
                write(seam, (Sys.iswindows() ? "uname() { printf 'Linux\\n'; }\n" : "") *
                            "rmdir() { printf 'INJECTED_RMDIR_ERROR=75\\n' >&2; return 75; }\n")
            elseif fault == "foreign-admission"
                admission = joinpath(fault_state, "run", ".$target.start.lock.admission"); mkpath(admission)
                write(joinpath(admission, "foreign.txt"), "preserve foreign content\n")
                write(seam, Sys.iswindows() ? "uname() { printf 'Linux\\n'; }\n" : "# native Unix admission\n")
            else
                suffix = fault == "mkdir" ? "" : "/run"
                write(seam, "mkdir() { local arg; for arg in \"\$@\"; do if [ \"\$arg\" = \"\$SOLARSINDY_MONITOR_DIR$suffix\" ]; then printf 'INJECTED_MKDIR_ERROR=74\\n' >&2; return 74; fi; done; command mkdir \"\$@\"; }\n")
            end
            result = call(["start", target]; extra=Dict("SOLARSINDY_MONITOR_DIR"=>fault_state, "BASH_ENV"=>seam))
            @test result.code != 0
            @test occursin(fault == "rotation" ? "INJECTED_MV_ERROR=73" : fault == "rmdir" ? "INJECTED_RMDIR_ERROR=75" :
                           fault == "foreign-admission" ? "admission is held or unknown" : "INJECTED_MKDIR_ERROR=74", result.err)
            @test read(old_log, String) == original_log
            @test !isfile(old_log * ".old")
            @test !isfile(joinpath(fault_state, "run", "$target.pid"))
            @test !isfile(joinpath(fault_state, "$target-main.txt"))
            @test !isfile(joinpath(fault_state, "run", ".$target.start.lock"))
            @test isdir(joinpath(fault_state, "run", ".$target.start.lock.admission")) == (fault in ("rmdir", "foreign-admission"))
            if fault == "foreign-admission"
                @test read(joinpath(fault_state, "run", ".$target.start.lock.admission", "foreign.txt"), String) == "preserve foreign content\n"
            end
        end
        @test call(["status"]).code == 2
        # Concurrent starts admit exactly one main. Both spawning Bash processes exit
        # before the independent fresh status/stop invocations below.
        first = launch(["start", "monitor"]; extra=Dict("SOLARSINDY_INERT_BANNER_DELAY_SEC"=>"6"))
        second = launch(["start", "monitor"]; extra=Dict("SOLARSINDY_INERT_BANNER_DELAY_SEC"=>"6"))
        deadline = time() + 20
        while !isfile(lockfile) && time() < deadline
            sleep(0.05)
        end
        @test isfile(lockfile)
        old_lock = read(lockfile)
        a, b = finish(first), finish(second)
        @test process_exited(first.process)
        @test process_exited(second.process)
        @test count(==(0), [a.code, b.code]) >= 1
        @test main_count() == 1
        @test isfile(joinpath(state, "monitor-banner.txt")) # new delayed banner, not stale log
        @test !isfile(lockfile)
        live = call(["status"])
        @test live.code == 1
        @test occursin("monitor:   RUNNING", live.out)
        @test !occursin("unknown option", live.err)
        # The admitted native/Unix owner is now dead (finish waited for its Bash).
        # Reclaim exactly that old generation, not a live caller or foreign record.
        write(lockfile, old_lock)
        @test call(["start", "monitor"]).code == 0
        @test !isfile(lockfile)
        @test main_count() == 1
        original = read(pidfile)
        write(joinpath(logs, "admitted-monitor-state.bin"), original)
        if Sys.isunix()
            @test parse(Int, strip(String(copy(original)))) ==
                  parse(Int, only(readlines(joinpath(state, "monitor-main.txt"))))
        end
        if Sys.iswindows()
            record = TOML.parse(String(copy(original)))
            @test record["pid"] == parse(Int, only(readlines(joinpath(state, "monitor-main.txt"))))
            @test record["phase"] == "ready"
            @test length(record["nonce"]) == 64
            @test occursin(r"up [0-9]+ s", live.out)
            # A query-tool failure cannot remove state, report stopped, spawn or signal.
            bad = Dict("JULIA"=>joinpath(logs, "missing query executable"))
            failed = call(["status"]; extra=bad)
            @test failed.code == 3
            @test occursin("UNKNOWN", failed.out)
            @test call(["stop", "monitor"]; extra=bad).code != 0
            @test call(["start", "monitor"]; extra=bad).code != 0
            @test read(pidfile) == original
            @test main_count() == 1
            # Wrong clone, legacy native/cyg numeric state, malformed/oversize, and
            # pending startup are preserved and refuse adoption/signal/new start.
            wrong = copy(record); wrong["root"] = dirname(clone)
            pending = copy(record); pending["phase"] = "pending"
            pending["pid"] = 0; pending["creation"] = ""; pending["executable"] = ""
            variants = [sprint(io -> TOML.print(io, wrong)), string(record["pid"], "\n"),
                "pid = true\n", repeat("x", 16385), sprint(io -> TOML.print(io, pending))]
            for bytes in variants
                write(pidfile, bytes)
                @test helper_call("query").code == 2
                @test read(pidfile, String) == bytes
            end
            write(pidfile, variants[1])
            @test call(["status"]).code == 3
            @test call(["start", "monitor"]).code != 0
            @test call(["stop", "monitor"]).code != 0
            @test read(pidfile, String) == variants[1]
            @test main_count() == 1
            write(pidfile, original)
            # Native creation stamp distinguishes PID reuse. Never stop that PID's
            # live replacement; the old-generation record may be removed as stale.
            stale = copy(record); stale["creation"] = string(parse(UInt64, record["creation"]) + 1)
            open(io -> TOML.print(io, stale), pidfile, "w")
            @test helper_call("query").code == 1
            @test helper_call("stop").code == 1
            @test !isfile(pidfile)
            write(pidfile, original)
            @test helper_call("query").code == 0
            # A malformed/foreign lock is unknown, never a stale-lock permission.
            write(lockfile, "1234\n")
            lockbytes = read(lockfile)
            @test call(["start", "monitor"]).code != 0
            @test read(lockfile) == lockbytes
            @test main_count() == 1
            rm(lockfile)
            foreign_lock = TOML.parse(String(copy(old_lock)))
            foreign_lock["root"] = dirname(clone)
            open(io -> TOML.print(io, foreign_lock), lockfile, "w")
            lockbytes = read(lockfile)
            @test call(["start", "monitor"]).code != 0
            @test read(lockfile) == lockbytes
            rm(lockfile)
            # Native slash spelling variants must retain the same clone binding.
            @test helper_call("query"; root=replace(clone, '\\'=>'/'), entry=replace(mon, '\\'=>'/')).code == 0
            # Controlled API-error classification oracle: PID0 really fails
            # OpenProcess; substituting ACCESS_DENIED for its error code must be
            # unknown, not the INVALID_PARAMETER/dead branch. No denied process
            # or security setting is touched by this seam.
            sandbox = Module(:CLIProcessQueryFault)
            Base.include(sandbox, helper)
            adapter = Base.invokelatest(getfield, sandbox, :SolarSINDyCLIProcess)
            Core.eval(adapter, :(last_error() = UInt32(5)))
            error_type = Base.invokelatest(getfield, adapter, :UnknownIdentity)
            query = Base.invokelatest(getfield, adapter, :open_process)
            @test_throws error_type Base.invokelatest(query, UInt32(0))
            write(joinpath(logs, "api-fault-control.txt"), "controlled OpenProcess error5 => UnknownIdentity; no security settings changed\n")
            decoy = run(`$(env["JULIA"]) --startup-file=no -e 'sleep(60)'`; wait=false)
            try
                write(pidfile, string(getpid(decoy), "\n"))
                @test call(["stop", "monitor"]).code != 0
                @test process_running(decoy) # never adopt julia.exe by image name
                @test read(pidfile, String) == string(getpid(decoy), "\n")
            finally
                write(pidfile, original)
                process_running(decoy) && kill(decoy)
                wait(decoy)
                @test process_exited(decoy)
            end
        end
        @test call(["start", "dashboard"]).code == 0
        write(joinpath(logs, "admitted-dashboard-state.bin"), read(joinpath(state, "run", "dashboard.pid")))
        both = call(["status"])
        @test both.code == 0
        @test occursin("dashboard: RUNNING", both.out)
        @test call(["stop", "dashboard"]).code == 0
        @test call(["stop", "monitor"]).code == 0
        @test !isfile(pidfile)
        @test call(["status"]).code == 2
        if Sys.isunix()
            # These use actual native ps and a real symlink/relative launch, not
            # the controlled query seams above. The main waits for our bounded
            # gate so the second caller necessarily observes its live launcher.
            alias = joinpath(logs, "clone alias with spaces"); symlink(clone, alias)
            for spelling in ("alias", "relative")
                alias_state = joinpath(logs, "$spelling state"); mkpath(alias_state)
                gate = joinpath(alias_state, "release-banner")
                options = Dict("SOLARSINDY_MONITOR_DIR"=>alias_state, "SOLARSINDY_INERT_BANNER_GATE"=>gate)
                script = spelling == "alias" ? joinpath(alias, "bin", "solarsindy") : joinpath("bin", "solarsindy")
                held = launch(["start", "monitor"]; extra=options, script=script, cwd=clone)
                alias_lock = joinpath(alias_state, "run", ".monitor.start.lock")
                try
                    deadline = time() + 15
                    while !isfile(alias_lock) && time() < deadline && process_running(held.process)
                        sleep(0.05)
                    end
                    @test isfile(alias_lock)
                    owner = read(alias_lock)
                    refused = call(["start", "monitor"]; extra=options)
                    @test refused.code != 0
                    @test occursin("another start is already in progress", refused.err)
                    @test read(alias_lock) == owner
                    write(gate, "release\n")
                    @test finish(held).code == 0
                    @test length(readlines(joinpath(alias_state, "monitor-main.txt"))) == 1
                    @test isfile(joinpath(alias_state, "monitor-banner.txt"))
                    @test !isfile(alias_lock)
                    @test call(["stop", "monitor"]; extra=options).code == 0
                    @test call(["status"]; extra=options).code == 2
                finally
                    write(gate, "release\n")
                    if !haskey(held.row, "exit")
                        finish(held)
                    end
                    @test call(["stop", "monitor"]; extra=options).code == 0
                end
            end
        end
    finally
        # Only these owned, self-expiring inert fixture mains are eligible for cleanup.
        for target in ("monitor", "dashboard")
            try call(["stop", target]) catch e; @warn "owned fixture cleanup failed" target exception=e end
        end
        open(joinpath(logs, "execution.toml"), "w") do io
            TOML.print(io, Dict("julia"=>string(VERSION), "platform"=>string(Sys.KERNEL),
                               "commands"=>rows, "helper_commands"=>helper_rows); sorted=true)
        end
    end
end
