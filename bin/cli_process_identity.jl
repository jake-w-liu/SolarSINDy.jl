# Stdlib-only Windows adapter for bin/solarsindy. See CLI_PROCESS_CONTRACT.md.
# The bootstrap is CLI-owned startup attestation, not an OS argv query.
module SolarSINDyCLIProcess
using TOML, Random, SHA

const RECORD_LIMIT = 16384
const RECORD_KEYS = Set(["schema", "kind", "phase", "root", "entry", "nonce",
                         "pid", "creation", "executable"])
const WAIT_OBJECT_0 = UInt32(0)
const WAIT_TIMEOUT = UInt32(258)
const WAIT_ABANDONED = UInt32(128)
const QUERY_RIGHTS = UInt32(0x1000 | 0x100000)
struct UnknownIdentity <: Exception
    message::String
end
unknown(s) = throw(UnknownIdentity(s))
wide(s) = transcode(UInt16, String(s) * '\0')
last_error() = ccall((:GetLastError, "kernel32"), UInt32, ())
close_handle(h) = ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), h)
path_key(p) = lowercase(replace(normpath(String(p)), '\\' => '/'))

function bindings(root, entry; lock=false)
    r, e = realpath(root), realpath(entry)
    allowed = lock ? [joinpath(r, "bin", "solarsindy")] :
                    [joinpath(r, "examples", "live_monitor.jl"), joinpath(r, "app", "src", "server.jl")]
    any(p -> path_key(p) == path_key(e), allowed) || unknown("entry point is outside this clone")
    return r, e
end

function with_mutex(f, path)
    # Resolve aliases of the containing directory so spelling variants share a lock.
    canonical = joinpath(realpath(dirname(path)), basename(path))
    name = wide("Local\\SolarSINDyCLI-" * bytes2hex(sha256(path_key(canonical))))
    h = ccall((:CreateMutexW, "kernel32"), Ptr{Cvoid},
              (Ptr{Cvoid}, Cint, Ptr{UInt16}), C_NULL, 0, name)
    h == C_NULL && unknown("cannot create state mutex ($(last_error()))")
    acquired = false
    try
        result = ccall((:WaitForSingleObject, "kernel32"), UInt32,
                       (Ptr{Cvoid}, UInt32), h, 3000)
        result in (WAIT_OBJECT_0, WAIT_ABANDONED) || unknown("state mutex unavailable ($result)")
        acquired = true
        return f()
    finally
        acquired && ccall((:ReleaseMutex, "kernel32"), Cint, (Ptr{Cvoid},), h)
        close_handle(h)
    end
end

function read_record(path, root, entry, kind)
    info = lstat(path)
    Int(info.ioerrno) == Base.UV_ENOENT && return nothing
    info.ioerrno == 0 || unknown("state lstat failed ($(info.ioerrno))")
    isfile(info) && !islink(info) || unknown("state is not a regular non-symlink file")
    bytes = open(path, "r") do io
        read(io, RECORD_LIMIT + 1)
    end
    length(bytes) <= RECORD_LIMIT || unknown("state exceeds byte limit")
    d = TOML.parse(String(bytes))
    Set(keys(d)) == RECORD_KEYS || unknown("state keys are invalid (legacy numeric state is ambiguous on Windows)")
    d["schema"] isa Int && !(d["schema"] isa Bool) && d["schema"] == 1 || unknown("state schema is invalid")
    all(k -> d[k] isa String && ncodeunits(d[k]) <= 4096,
        ["kind", "phase", "root", "entry", "nonce", "creation", "executable"]) || unknown("invalid state strings")
    d["kind"] == kind || unknown("state kind mismatch")
    path_key(d["root"]) == path_key(root) && path_key(d["entry"]) == path_key(entry) || unknown("state belongs to another clone or entry point")
    occursin(r"^[a-f0-9]{64}$", d["nonce"]) || unknown("invalid state nonce")
    d["pid"] isa Int && !(d["pid"] isa Bool) || unknown("invalid native PID")
    if d["phase"] == "pending" && kind == "component"
        d["pid"] == 0 && d["creation"] == "" && d["executable"] == "" || unknown("invalid pending state")
    elseif d["phase"] == "ready"
        4 < d["pid"] <= typemax(UInt32) || unknown("invalid native PID range")
        occursin(r"^[0-9]{1,20}$", d["creation"]) && tryparse(UInt64, d["creation"]) !== nothing || unknown("invalid creation stamp")
        isabspath(d["executable"]) || unknown("invalid executable path")
    else
        unknown("invalid state phase")
    end
    return d
end

function publish(path, d; replace=false)
    mkpath(dirname(path))
    temporary, io = mktemp(dirname(path); cleanup=false)
    try
        TOML.print(io, d; sorted=true)
        close(io)
        filesize(temporary) <= RECORD_LIMIT || unknown("generated state exceeds byte limit")
        if replace
            # All readers/writers use the same native mutex. Only an admitted record
            # with the expected nonce is replaced; initial publication never replaces.
            Base.Filesystem.rename(temporary, path)
        else
            hardlink(temporary, path)
        end
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary)
    end
end

function open_process(pid; terminate=false)
    rights = QUERY_RIGHTS | (terminate ? UInt32(1) : UInt32(0))
    h = ccall((:OpenProcess, "kernel32"), Ptr{Cvoid}, (UInt32, Cint, UInt32), rights, 0, pid)
    if h == C_NULL
        code = last_error()
        code == 87 && return nothing # invalid process identifier, not access denied
        unknown("OpenProcess failed ($code); identity is unknown")
    end
    return h
end

function process_identity(h)
    c, e, k, u = Ref{UInt64}(0), Ref{UInt64}(0), Ref{UInt64}(0), Ref{UInt64}(0)
    ccall((:GetProcessTimes, "kernel32"), Cint,
          (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}), h, c, e, k, u) != 0 || unknown("GetProcessTimes failed ($(last_error()))")
    buffer = zeros(UInt16, 32768)
    count = Ref{UInt32}(length(buffer))
    ccall((:QueryFullProcessImageNameW, "kernel32"), Cint,
          (Ptr{Cvoid}, UInt32, Ptr{UInt16}, Ref{UInt32}), h, 0, buffer, count) != 0 || unknown("executable query failed ($(last_error()))")
    wait = ccall((:WaitForSingleObject, "kernel32"), UInt32, (Ptr{Cvoid}, UInt32), h, 0)
    wait in (WAIT_OBJECT_0, WAIT_TIMEOUT) || unknown("process wait query failed ($wait)")
    return (; creation=string(c[]), executable=transcode(String, buffer[1:count[]]), live=wait == WAIT_TIMEOUT)
end

function check_record(f, d; terminate=false)
    d["phase"] == "ready" || unknown("startup receipt is pending; preserving state")
    h = open_process(UInt32(d["pid"]); terminate)
    h === nothing && return 1
    try
        identity = process_identity(h)
        identity.creation == d["creation"] || return 1 # old generation is gone; never signal replacement
        path_key(identity.executable) == path_key(d["executable"]) || unknown("executable identity mismatch")
        identity.live || return 1
        return f(h, identity)
    finally
        close_handle(h)
    end
end

function fresh_record(kind, phase, root, entry, nonce; pid=0, creation="", executable="")
    Dict{String,Any}("schema"=>1, "kind"=>kind, "phase"=>phase, "root"=>root,
        "entry"=>entry, "nonce"=>nonce, "pid"=>pid, "creation"=>creation, "executable"=>executable)
end
nonce() = bytes2hex(rand(RandomDevice(), UInt8, 32))

function component(command, path, root, entry, token="")
    r, e = bindings(root, entry)
    # Only ENOENT establishes absence. Permission/query failures and dangling
    # symlinks must reach the guarded error path, never authorize another spawn.
    if command in ("query", "uptime", "describe", "stop")
        info = lstat(path)
        Int(info.ioerrno) == Base.UV_ENOENT && return 1
        info.ioerrno == 0 || unknown("state lstat failed ($(info.ioerrno))")
    end
    return with_mutex(path) do
        d = read_record(path, r, e, "component")
        if command == "prepare"
            if d !== nothing
                state = check_record((h, identity) -> 0, d)
                state == 1 || unknown("component is already running")
                rm(path) # confirmed dead generation, under the path mutex
            end
            n = nonce()
            publish(path, fresh_record("component", "pending", r, e, n))
            println(n)
            return 0
        elseif command == "bootstrap"
            d !== nothing && d["phase"] == "pending" && d["nonce"] == token || unknown("startup nonce/state mismatch")
            path_key(realpath(PROGRAM_FILE)) == path_key(e) || unknown("actual PROGRAM_FILE does not match the admitted entry point")
            h = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
            identity = process_identity(h)
            ready = fresh_record("component", "ready", r, e, token; pid=getpid(),
                                 creation=identity.creation, executable=identity.executable)
            publish(path, ready; replace=true)
            return 0
        end
        d === nothing && return 1
        result = check_record(d; terminate=command == "stop") do h, identity
            if command == "query"
                println(d["pid"])
            elseif command in ("uptime", "describe")
                now = Ref{UInt64}(0)
                ccall((:GetSystemTimeAsFileTime, "kernel32"), Cvoid, (Ref{UInt64},), now)
                seconds = max(Int128(0), Int128(now[]) - parse(UInt64, d["creation"])) ÷ 10_000_000
                command == "describe" && print(d["pid"], " ")
                println(seconds, " s")
            elseif command == "stop"
                ccall((:TerminateProcess, "kernel32"), Cint, (Ptr{Cvoid}, UInt32), h, 0) != 0 || unknown("TerminateProcess failed ($(last_error()))")
                ccall((:WaitForSingleObject, "kernel32"), UInt32, (Ptr{Cvoid}, UInt32), h, 5000) == WAIT_OBJECT_0 || unknown("termination not confirmed within 5 s")
            else
                unknown("unknown process command")
            end
            return 0
        end
        # A stop removes only an admitted record after its generation is confirmed dead.
        command == "stop" && result in (0, 1) && rm(path)
        return result
    end
end

function start_lock(command, path, root, entry, owner, token="")
    r, e = bindings(root, entry; lock=true)
    4 < owner <= typemax(UInt32) || unknown("invalid native lock owner PID")
    return with_mutex(path) do
        d = read_record(path, r, e, "lock")
        if command == "lock-release"
            d !== nothing && d["pid"] == owner && d["nonce"] == token || unknown("lock release identity mismatch")
            check_record((h, identity) -> 0, d) == 0 || unknown("lock owner generation is no longer live")
            rm(path)
            return 0
        end
        if d !== nothing
            check_record((h, identity) -> 0, d) == 0 && return 1
            rm(path) # only a confirmed old/dead generation can be reclaimed
        end
        h = open_process(UInt32(owner))
        h === nothing && unknown("lock owner is absent")
        try
            identity = process_identity(h)
            identity.live || unknown("lock owner has exited")
            lowercase(basename(identity.executable)) == "bash.exe" || unknown("native lock owner is not the admitted Bash process")
            n = nonce()
            publish(path, fresh_record("lock", "ready", r, e, n; pid=Int(owner),
                    creation=identity.creation, executable=identity.executable))
            println(n)
            return 0
        finally
            close_handle(h)
        end
    end
end

function main(args)
    Sys.iswindows() || unknown("Windows adapter invoked on another platform")
    length(args) >= 1 || unknown("missing command")
    command = args[1]
    if command == "await" && length(args) == 4
        # Only the admitted startup handshake may wait through a pending receipt.
        # Other readers report unknown immediately. Do not hold the path mutex
        # while sleeping: the bootstrap must be able to publish its ready record.
        deadline = time_ns() + UInt64(30_000_000_000)
        while true
            r, e = bindings(args[3], args[4])
            phase = with_mutex(args[2]) do
                d = read_record(args[2], r, e, "component")
                d === nothing && unknown("startup state disappeared")
                d["phase"]
            end
            phase == "ready" && return component("query", args[2:4]...)
            time_ns() < deadline || unknown("startup identity remained pending after 30 s")
            sleep(0.1)
        end
    elseif command in ("query", "uptime", "describe", "prepare", "stop") && length(args) == 4
        return component(command, args[2:4]...)
    elseif command in ("lock-acquire", "lock-release") && length(args) == (command == "lock-acquire" ? 5 : 6)
        owner = tryparse(UInt32, args[5])
        owner === nothing && unknown("invalid native lock owner PID")
        return start_lock(command, args[2:4]..., owner, length(args) == 6 ? args[6] : "")
    end
    unknown("invalid helper arguments")
end

function guarded(f)
    try
        return f()
    catch exception
        println(stderr, "process identity unknown: ", sprint(showerror, exception))
        return 2
    end
end
end # module

if get(ENV, "SOLARSINDY_CLI_BOOTSTRAP", "") == "1"
    result = SolarSINDyCLIProcess.guarded() do
        SolarSINDyCLIProcess.component("bootstrap", ENV["SOLARSINDY_CLI_STATE"],
            ENV["SOLARSINDY_CLI_ROOT"], ENV["SOLARSINDY_CLI_ENTRY"], ENV["SOLARSINDY_CLI_NONCE"])
    end
    for key in ("SOLARSINDY_CLI_BOOTSTRAP", "SOLARSINDY_CLI_STATE", "SOLARSINDY_CLI_ROOT", "SOLARSINDY_CLI_ENTRY", "SOLARSINDY_CLI_NONCE")
        delete!(ENV, key)
    end
    result == 0 || exit(result)
elseif abspath(PROGRAM_FILE) == @__FILE__
    exit(SolarSINDyCLIProcess.guarded(() -> SolarSINDyCLIProcess.main(ARGS)))
end
