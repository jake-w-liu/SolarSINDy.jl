# Independent frozen contracts; candidate text is never evaluated by this driver.
const READER_STARTED_NS = Int(time_ns())
const READER_DEADLINE_NS = READER_STARTED_NS + 180 * 10^9
using Test, SHA, TOML

const ORACLE_SHA = "0589f789614d8ac9aaf7e7d59663037ff6976a1ece8ef7a2e8d7bc6f2d09a1bd"
const VALID = (
    ("P01", ""),
    ("P02", "# comment only\n#= block comment =#\n"),
    ("P03", "module Ω\nα = 1\nf(β) = α + β\nend\n"),
    ("P04", "function f(x)\n    x > 0.0 || return nothing\n    return x\nend\n"),
    ("P05", "function g(x)\n    for i in 1:1\n        x < 0.0 || break\n    end\n    nothing\nend\n"),
)
const INVALID = (
    ("N01", "x = )\n"),
    ("N02", "function unfinished(\n"),
    ("N03", "value = \"unterminated"),
    ("N04", "x = 1\nx = )\n"),
)
const OLD_AMBIGUOUS = (
    ("N05", "function old_return(raw,r)\n    raw[9,r]>0.||return\nend\n"),
    ("N06", "function old_break(raw,r)\n    for i in 1:1\n        raw[6,r]<0.||break\n    end\nend\n"),
)
need(ok, why) = ok || error(why)
clock_ns() = Int(time_ns())
function clock_check()
    need(clock_ns() < READER_DEADLINE_NS, "Independent syntax tests180s deadline")
end
function file_sha(path)
    need(isfile(path) && !islink(path), "ordinary pinned source required")
    open(io -> bytes2hex(sha256(io)), path)
end
function threw(body)
    try
        body()
        return false
    catch error
        error isa InterruptException && rethrow()
        return true
    end
end
deep(tree) = foldl((value, _) -> Expr(:block, value), 1:64; init=tree)

function test_main(args)
    need(length(args) == 6 && args[1] == "--helper" &&
         args[3] == "--expected-helper-sha256" && args[5] == "--output-directory",
         "Explicit helper/hash/fresh output arguments required")
    helper, expected, requested_output = args[2], args[4], args[6]
    need(isabspath(helper) && isabspath(requested_output) &&
         occursin(r"^[0-9a-f]{64}$", expected), "Absolute fixed inputs required")
    need(VERSION in (v"1.12.6", v"1.13.1"), "Version-bound oracles require1.12.6 or1.13.1")
    need(!ispath(requested_output), "Fresh test-owned output required")
    output = joinpath(realpath(dirname(requested_output)), basename(requested_output))
    oracle = joinpath(@__DIR__, "ORACLES_V1.json")
    source_before = Dict(abspath(@__FILE__) => file_sha(abspath(@__FILE__)),
                         helper => file_sha(helper), oracle => file_sha(oracle))
    need(source_before[helper] == expected && source_before[oracle] == ORACLE_SHA,
         "Frozen helper/oracle source changed")
    clock_check()
    mkdir(output)
    need(!isdefined(Main, :StrictJuliaSyntax), "Fresh helper module required")
    # This loads only the explicitly pinned implementation; never a specimen.
    Base.include(Main, helper)
    # Newly included methods and module bindings are resolved in the latest world.
    return Base.invokelatest(function()
    H = getfield(Main, :StrictJuliaSyntax)
    need(file_sha(helper) == expected, "Helper changed during include")
    rows = Dict{String,Any}[]
    probes = Dict{String,Any}[]
    classifications = Dict{String,Any}[]
    paths = Dict{String,String}()
    marker = joinpath(output, "CANDIDATE_WAS_EVALUATED")
    function specimen(id, text)
        path = joinpath(output, id * ".jl")
        open(path, "w") do io
            write(io, text)
        end
        paths[id] = path
        return path
    end
    for (id, text) in (VALID..., INVALID..., OLD_AMBIGUOUS...)
        specimen(id, text)
    end
    side_effect = "open(" * repr(marker) * ", \"w\") do io\n    write(io, \"must not be evaluated\")\nend\n"
    specimen("P06", side_effect)
    function record_case(body, id)
        clock_check()
        passed = false
        @testset "$id" begin
            passed = body() === true
            @test passed  # Each body's independent equality/exception oracle must hold.
        end
        push!(rows, Dict("id" => id, "asserted" => true, "passed" => passed))
    end
    function probe(label, tail)
        clock_check()
        started = clock_ns()
        deadline = min(started + 30 * 10^9, READER_DEADLINE_NS - 10 * 10^9)
        need(deadline > started, "No original budget remains for CLI probe")
        argv = [Base.julia_cmd().exec...; "--startup-file=no"; "--history-file=no"; tail...]
        logpath = joinpath(output, label * ".log")
        process = nothing
        timed_out = false
        child_pid = 0
        open(logpath, "w") do log
            try
                process = run(pipeline(ignorestatus(Cmd(argv)); stdout=log, stderr=log); wait=false)
                child_pid = getpid(process)
                while process_running(process)
                    if clock_ns() >= deadline
                        timed_out = true
                        break
                    end
                    sleep(.01)
                end
            finally
                if process !== nothing && process_running(process)
                    kill(process)
                    cleanup_deadline = min(clock_ns() + 5 * 10^9, READER_DEADLINE_NS)
                    escalated = false
                    while process_running(process)
                        if !escalated && cleanup_deadline - clock_ns() <= 2 * 10^9
                            kill(process, Base.SIGKILL)
                            escalated = true
                        end
                        need(clock_ns() < cleanup_deadline, "Owned direct CLI child cleanup deadline")
                        sleep(.01)
                    end
                end
                process !== nothing && wait(process)
            end
        end
        finished = clock_ns()
        need(process !== nothing && child_pid > 0 && filesize(logpath) <= 65536,
             "Actual bounded CLI child/raw result required")
        row = Dict{String,Any}("label" => label, "argv" => argv, "pid" => child_pid,
            "native_exit" => process.exitcode, "timed_out" => timed_out,
            "direct_child_wait_observed" => true, "all_descendants_OS_reaped" => false,
            "started_ns" => started, "deadline_ns" => deadline, "finished_ns" => finished,
            "raw_path" => logpath, "raw_bytes" => filesize(logpath), "raw_sha256" => file_sha(logpath))
        push!(probes, row)
        clock_check()
        return row
    end
    function main_and_cli(id, files, expected_exit)
        value = H.main(files)
        result = probe(id, [helper; files])
        return typeof(value) === Int && value == expected_exit &&
               result["native_exit"] == expected_exit && result["timed_out"] === false &&
               result["finished_ns"] < result["deadline_ns"]
    end
    suite_error = ""
    try
        @testset verbose=true "StrictJuliaSyntax independent frozen oracles" begin
            @testset verbose=true "Independent AST structure" begin
                record_case("T01") do
                    H.has_parse_error(:error) === false &&
                    H.has_parse_error(LineNumberNode(1, :independent)) === false
                end
                record_case("T02") do
                    H.has_parse_error(deep(Expr(:call, :+, 1, 2))) === false
                end
                record_case("T03") do
                    H.has_parse_error(Expr(:error, "independent specimen")) === true
                end
                record_case("T04") do
                    H.has_parse_error(Expr(:incomplete, "independent specimen")) === true
                end
                record_case("T05") do
                    H.has_parse_error(Expr(:toplevel, Expr(Symbol("="), :valid, 1),
                                           deep(Expr(:error, "independent specimen")))) === true
                end
                record_case("T06") do
                    H.has_parse_error(Expr(:toplevel, Expr(Symbol("="), :valid, 1),
                                           deep(Expr(:incomplete, "independent specimen")))) === true
                end
            end
            @testset verbose=true "Independent valid text and non-evaluation" begin
                for (id, text) in VALID
                    record_case(id) do
                        H.check_text(text; filename=id * ".jl") === nothing &&
                        H.check_file(paths[id]) === nothing
                    end
                end
                record_case("P06") do
                    H.check_text(side_effect; filename="P06.jl") === nothing &&
                    H.check_file(paths["P06"]) === nothing && !ispath(marker)
                end
            end
            @testset verbose=true "Malformed/incomplete and version-bound old forms" begin
                for (id, text) in INVALID
                    record_case(id) do
                        threw(() -> H.check_text(text; filename=id * ".jl")) &&
                        threw(() -> H.check_file(paths[id]))
                    end
                end
                for (id, text) in OLD_AMBIGUOUS
                    item = Dict{String,Any}("id" => id, "julia_version" => string(VERSION))
                    try
                        tree = Meta.parseall(text; filename=id * ".jl")
                        item["parser_returned"] = true
                        item["raw_parser_tree_repr"] = repr(tree)
                    catch error
                        error isa InterruptException && rethrow()
                        item["parser_returned"] = false
                        item["parser_error"] = sprint(showerror, error)
                    end
                    item["checker_threw"] = threw(() -> H.check_text(text; filename=id * ".jl"))
                    push!(classifications, item)
                    if VERSION == v"1.13.1"
                        record_case(id) do
                            item["checker_threw"] === true && threw(() -> H.check_file(paths[id]))
                        end
                    else
                        # This is a measurement request, not an invented1.12 acceptance oracle.
                        push!(rows, Dict("id" => id, "asserted" => false,
                                         "outcome" => "recorded_unasserted_runtime_classification"))
                    end
                end
            end
            @testset verbose=true "Real CLI and original false-positive witness" begin
                record_case("C01") do
                    main_and_cli("C01", [paths["P01"], paths["P03"], paths["P04"], paths["P05"]], 0)
                end
                record_case("C02") do
                    main_and_cli("C02", [paths["P03"], paths["N04"]], 1)
                end
                record_case("C03") do
                    main_and_cli("C03", String[], 1)
                end
                record_case("C04") do
                    missing = joinpath(output, "does-not-exist.jl")
                    threw(() -> H.check_file(missing)) && main_and_cli("C04", [missing], 1)
                end
                record_case("C05") do
                    # The fixed expression parses the file; it does not evaluate its contents.
                    before = probe("C05-old-unexamined",
                        ["-e", "Meta.parseall(read(ARGS[1], String))", paths["N01"]])
                    after = probe("C05-strict", [helper, paths["N01"]])
                    before["native_exit"] == 0 && before["timed_out"] === false &&
                    before["finished_ns"] < before["deadline_ns"] &&
                    after["native_exit"] == 1 && after["timed_out"] === false &&
                    after["finished_ns"] < after["deadline_ns"]
                end
            end
        end
    catch error
        error isa InterruptException && rethrow()
        suite_error = sprint(showerror, error)
    end
    source_after = Dict(path => file_sha(path) for path in keys(source_before))
    expected_ids = [("T" * lpad(string(i), 2, '0')) for i in 1:6]
    append!(expected_ids, [("P" * lpad(string(i), 2, '0')) for i in 1:6])
    append!(expected_ids, [("N" * lpad(string(i), 2, '0')) for i in 1:6])
    append!(expected_ids, [("C" * lpad(string(i), 2, '0')) for i in 1:5])
    asserted = [row for row in rows if row["asserted"] === true]
    unasserted = [row for row in rows if row["asserted"] === false]
    complete = isempty(suite_error) && [row["id"] for row in rows] == expected_ids &&
               length(asserted) == (VERSION == v"1.13.1" ? 23 : 21) &&
               length(unasserted) == (VERSION == v"1.13.1" ? 0 : 2) &&
               all(row["passed"] === true for row in asserted) &&
               source_before == source_after && !ispath(marker)
    clock_check()
    result = Dict{String,Any}("schema" => "strict_julia_syntax_independent_test_result_v1",
        "complete" => complete, "julia_version" => string(VERSION), "case_ids" => expected_ids,
        "asserted_cases" => length(asserted), "unasserted_version_classifications" => length(unasserted),
        "rows" => rows, "old_ambiguous_native_classifications" => classifications, "CLI_probes" => probes,
        "source_before" => source_before, "source_after" => source_after,
        "source_unchanged" => source_before == source_after, "candidate_marker_absent" => !ispath(marker),
        "suite_error" => suite_error, "started_ns" => READER_STARTED_NS,
        "finished_ns" => clock_ns(), "deadline_ns" => READER_DEADLINE_NS,
        "source_qualified" => false, "software_qualified" => false, "scientific_qualified" => false)
    println("STRICT_JULIA_SYNTAX_TESTS complete=", complete,
            " asserted=", length(asserted), " unasserted_classifications=", length(unasserted))
    flush(stdout)
    clock_check()
    open(joinpath(output, "RESULT.toml"), "w") do io
        TOML.print(io, result)
    end
    clock_check()
    return complete ? 0 : 1
    end)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(test_main(ARGS))
end
