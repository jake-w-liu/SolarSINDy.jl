"""Standard-library syntax controls and tracked-source parsing; no candidate evaluation."""
module JuliaSourceSyntaxAudit
using SHA, TOML

const ROOT = realpath(joinpath(@__DIR__, "..", ".."))
const HELPER = joinpath(ROOT, "tools", "verify-julia-syntax.jl")
const HELPER_SHA = "f1c342450194b97fe5033e2ac11b98218b229af0db89599abc1c8ade0c6f54fc"
const TEST = joinpath(@__DIR__, "test_strict_julia_syntax.jl")
const TEST_SHA = "102dd999f5cf9fc093217d27665e55cbb7c8ec6af34e35f4fead8f55797369d7"
const ORACLE = joinpath(@__DIR__, "ORACLES_V1.json")
const ORACLE_SHA = "0589f789614d8ac9aaf7e7d59663037ff6976a1ece8ef7a2e8d7bc6f2d09a1bd"

need(ok, message) = ok || error(message)
file_sha(path) = open(io -> bytes2hex(sha256(io)), path)

function inputs(runtime)
    need(runtime in ("1.12.6", "1.13.1") && string(VERSION) == runtime,
         "Exact requested syntax runtime required")
    need(Threads.nthreads() == 1, "One Julia thread required")
    expected = Dict(HELPER => HELPER_SHA, TEST => TEST_SHA, ORACLE => ORACLE_SHA)
    for (path, digest) in expected
        need(isfile(path) && !islink(path) && file_sha(path) == digest,
             "Pinned ordinary syntax source changed: " * path)
    end
    return expected
end

function recheck(expected)
    for (path, digest) in expected
        need(isfile(path) && !islink(path) && file_sha(path) == digest,
             "Syntax source changed during verification: " * path)
    end
    return nothing
end

function run_oracles(output, runtime)
    expected = inputs(runtime)
    need(isabspath(output) && !ispath(output), "Fresh absolute oracle output required")
    scope = Module(gensym(:IndependentStrictJuliaSyntax))
    Base.include(scope, TEST)
    code = Base.invokelatest(() -> getfield(scope, :test_main)(
        ["--helper", HELPER, "--expected-helper-sha256", HELPER_SHA,
         "--output-directory", output]))
    recheck(expected)
    need(typeof(code) === Int && code == 0, "Independent syntax controls failed")
    return 0
end

function run_inventory(inventory, output, runtime)
    started = time_ns()
    deadline = started + UInt64(280) * 10^9
    clock_check() = need(time_ns() < deadline, "Tracked-source syntax deadline")
    expected = inputs(runtime)
    need(isabspath(output) && !ispath(output) && isdir(dirname(output)),
         "Fresh absolute inventory output required")
    need(isfile(inventory) && !islink(inventory) && filesize(inventory) <= 1024^2,
         "Bounded ordinary NUL-delimited Git source inventory required")
    raw = open(io -> read(io, 1024^2 + 1), inventory)
    need(!isempty(raw) && length(raw) <= 1024^2 && raw[end] == 0,
         "Complete nonempty NUL-delimited inventory required")
    paths = split(String(raw), '\0'; keepempty=true)
    pop!(paths)
    need(!isempty(paths) && length(Set(paths)) == length(paths), "Unique tracked Julia source paths required")
    # Include only the helper bytes whose digest was just verified. Candidate
    # files below are supplied to check_text as data and are never included.
    body = read(HELPER)
    need(bytes2hex(sha256(body)) == HELPER_SHA, "Syntax helper changed before load")
    scope = Module(gensym(:TrackedSourceSyntax))
    Base.include_string(scope, String(body), HELPER)
    sources = Dict{String,String}()
    for relative in paths
        clock_check()
        parts = split(relative, '/'; keepempty=true)
        need(!isabspath(relative) && endswith(relative, ".jl") &&
             !occursin('\\', relative) && !occursin(':', relative) &&
             all(part -> !(part in ("", ".", "..")), parts), "Unsafe source inventory path")
        path = joinpath(ROOT, parts...)
        need(isfile(path) && !islink(path) && realpath(path) == normpath(path),
             "Ordinary checkout source required: " * relative)
        bytes = read(path)
        digest = bytes2hex(sha256(bytes))
        Base.invokelatest(() -> getfield(scope, :StrictJuliaSyntax).check_text(String(bytes); filename=path))
        need(file_sha(path) == digest, "Source changed during parse: " * relative)
        sources[String(relative)] = digest
    end
    recheck(expected)
    for (relative, digest) in sources
        clock_check()
        need(file_sha(joinpath(ROOT, split(relative, '/')...)) == digest,
             "Source changed after parse: " * relative)
    end
    clock_check()
    mkdir(output)
    open(joinpath(output, "SOURCE_INVENTORY.toml"), "w") do io
        TOML.print(io, Dict("julia_version" => runtime, "source_files" => length(sources),
            "helper_sha256" => HELPER_SHA, "inventory_sha256" => file_sha(inventory),
            "source_sha256" => sources, "started_ns" => started, "finished_ns" => time_ns(),
            "candidate_evaluated" => false, "scientific_qualified" => false))
    end
    println("STRICT_JULIA_SOURCE_FILES=", length(sources), " candidate_evaluated=false")
    flush(stdout)
    clock_check()
    return 0
end

function main(args)
    try
        need(length(args) in (6, 8) && args[1] == "--mode" && args[3] == "--runtime" &&
             args[5] == "--output-directory", "Explicit syntax mode/runtime/output required")
        mode, runtime, output = args[2], args[4], args[6]
        if mode == "oracles"
            need(length(args) == 6, "Fixed oracle arguments required")
            return run_oracles(output, runtime)
        end
        need(mode == "inventory" && length(args) == 8 && args[7] == "--inventory-file",
             "Fixed inventory arguments required")
        return run_inventory(args[8], output, runtime)
    catch exception
        exception isa InterruptException && rethrow()
        showerror(stderr, exception)
        println(stderr)
        return 1
    end
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(JuliaSourceSyntaxAudit.main(ARGS))
end
