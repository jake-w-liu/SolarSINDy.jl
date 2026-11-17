module StrictJuliaSyntaxPackageTests
using Test, SHA, TOML

const HELPER = normpath(joinpath(@__DIR__, "..", "..", "tools", "verify-julia-syntax.jl"))
const HELPER_SHA = "f1c342450194b97fe5033e2ac11b98218b229af0db89599abc1c8ade0c6f54fc"
const PORTABLE_ORACLE = joinpath(@__DIR__, "PORTABLE_ORACLES_V1.json")
const PORTABLE_ORACLE_SHA = "bfb25ffdb7fb09a63878f1061f967a0c4a74597e5640c53e8224d67faca10508"
const REGISTRATION = abspath(@__FILE__)
need(ok, why) = ok || error(why)
file_sha(path) = open(io -> bytes2hex(sha256(io)), path)
need(isfile(HELPER) && !islink(HELPER) && isfile(PORTABLE_ORACLE) && !islink(PORTABLE_ORACLE),
     "Ordinary portable syntax sources required")
const HELPER_BODY = read(HELPER)
need(bytes2hex(sha256(HELPER_BODY)) == HELPER_SHA && file_sha(PORTABLE_ORACLE) == PORTABLE_ORACLE_SHA,
     "Pinned portable syntax helper/oracle changed")
# Load only the verified implementation at module top level. Candidate text is
# data, and the frozen version-specific suite retains its separate Main helper.
Base.include_string(@__MODULE__, String(HELPER_BODY), HELPER)
include("run_syntax_audit.jl")

deep(tree) = foldl((value, _) -> Expr(:block, value), 1:64; init=tree)
function threw(body)
    try
        body()
        return false
    catch exception
        exception isa InterruptException && rethrow()
        return true
    end
end

function portable_contracts(evidence)
    started = time_ns()
    deadline = started + UInt64(30) * 10^9
    clock_check() = need(time_ns() < deadline, "Portable syntax contracts30s deadline")
    output = joinpath(evidence, "portable")
    mkdir(output)
    marker = joinpath(output, "CANDIDATE_WAS_EVALUATED")
    sources_before = Dict(HELPER => file_sha(HELPER), PORTABLE_ORACLE => file_sha(PORTABLE_ORACLE),
                          REGISTRATION => file_sha(REGISTRATION))
    rows = Dict{String,Any}[]
    paths = Dict{String,String}()
    valid = (("P01", ""), ("P02", "# comment only\n#= block comment =#\n"),
        ("P03", "module Ω\nα = 1\nf(β) = α + β\nend\n"),
        ("P04", "function f(x)\n    x > 0.0 || return nothing\n    return x\nend\n"),
        ("P05", "function g(x)\n    for i in 1:1\n        x < 0.0 || break\n    end\n    nothing\nend\n"))
    invalid = (("N01", "x = )\n"), ("N02", "function unfinished(\n"),
               ("N03", "value = \"unterminated"), ("N04", "x = 1\nx = )\n"))
    for (id, text) in (valid..., invalid...)
        path = joinpath(output, id * ".jl")
        write(path, text)
        paths[id] = path
    end
    side_effect = "open(" * repr(marker) * ", \"w\") do io\n    write(io, \"must not be evaluated\")\nend\n"
    paths["P06"] = joinpath(output, "P06.jl")
    write(paths["P06"], side_effect)
    function record_case(body, id)
        clock_check()
        passed = false
        @testset "$id" begin
            passed = body() === true
            @test passed
        end
        push!(rows, Dict("id" => id, "passed" => passed))
    end
    record_case("T01") do
        StrictJuliaSyntax.has_parse_error(:error) === false &&
        StrictJuliaSyntax.has_parse_error(LineNumberNode(1, :portable)) === false
    end
    record_case("T02") do
        StrictJuliaSyntax.has_parse_error(deep(Expr(:call, :+, 1, 2))) === false
    end
    record_case("T03") do
        StrictJuliaSyntax.has_parse_error(Expr(:error, "independent specimen")) === true
    end
    record_case("T04") do
        StrictJuliaSyntax.has_parse_error(Expr(:incomplete, "independent specimen")) === true
    end
    record_case("T05") do
        StrictJuliaSyntax.has_parse_error(Expr(:toplevel, Expr(Symbol("="), :prefix, 1),
            deep(Expr(:error, "independent specimen")))) === true
    end
    record_case("T06") do
        StrictJuliaSyntax.has_parse_error(Expr(:toplevel, Expr(Symbol("="), :prefix, 1),
            deep(Expr(:incomplete, "independent specimen")))) === true
    end
    for (id, text) in valid
        record_case(id) do
            StrictJuliaSyntax.check_text(text; filename=id * ".jl") === nothing &&
            StrictJuliaSyntax.check_file(paths[id]) === nothing
        end
    end
    record_case("P06") do
        StrictJuliaSyntax.check_text(side_effect; filename="P06.jl") === nothing &&
        StrictJuliaSyntax.check_file(paths["P06"]) === nothing && !ispath(marker)
    end
    for (id, text) in invalid
        record_case(id) do
            threw(() -> StrictJuliaSyntax.check_text(text; filename=id * ".jl")) &&
            threw(() -> StrictJuliaSyntax.check_file(paths[id]))
        end
    end
    record_case("C01") do
        value = StrictJuliaSyntax.main([paths["P01"], paths["P03"], paths["P04"], paths["P05"]])
        typeof(value) === Int && value == 0
    end
    record_case("C02") do
        value = StrictJuliaSyntax.main([paths["P03"], paths["N04"]])
        typeof(value) === Int && value == 1
    end
    record_case("C03") do
        value = StrictJuliaSyntax.main(String[])
        typeof(value) === Int && value == 1
    end
    record_case("C04") do
        missing = joinpath(output, "does-not-exist.jl")
        rejected = threw(() -> StrictJuliaSyntax.check_file(missing))
        value = StrictJuliaSyntax.main([missing])
        rejected && typeof(value) === Int && value == 1
    end
    record_case("C05") do
        # A compatible parser may either throw or return an error-bearing tree.
        parser_threw = false
        tree = try
            Meta.parseall("x = )\n"; filename="portable-old-parser.jl")
        catch exception
            exception isa InterruptException && rethrow()
            parser_threw = true
            nothing
        end
        (parser_threw || StrictJuliaSyntax.has_parse_error(tree) === true) &&
        threw(() -> StrictJuliaSyntax.check_text("x = )\n")) && !ispath(marker)
    end
    expected_ids = ["T01", "T02", "T03", "T04", "T05", "T06",
        "P01", "P02", "P03", "P04", "P05", "P06", "N01", "N02", "N03", "N04",
        "C01", "C02", "C03", "C04", "C05"]
    sources_after = Dict(path => file_sha(path) for path in keys(sources_before))
    need([row["id"] for row in rows] == expected_ids && length(rows) == 21 &&
         all(row["passed"] === true for row in rows), "All21 portable syntax contracts required")
    need(sources_before == sources_after && sources_after[HELPER] == HELPER_SHA &&
         sources_after[PORTABLE_ORACLE] == PORTABLE_ORACLE_SHA && !ispath(marker),
         "Portable syntax source/marker guard failed")
    clock_check()
    open(joinpath(output, "RESULT.toml"), "w") do io
        TOML.print(io, Dict("julia_version" => string(VERSION), "portable_cases" => 21,
            "case_ids" => expected_ids, "rows" => rows, "candidate_marker_absent" => !ispath(marker),
            "source_before" => sources_before, "source_after" => sources_after,
            "source_unchanged" => sources_before == sources_after, "started_ns" => started,
            "finished_ns" => time_ns(), "deadline_ns" => deadline,
            "source_qualified" => false, "software_qualified" => false, "scientific_qualified" => false))
    end
    println("PORTABLE_JULIA_SYNTAX_CASES=21 candidate_marker_absent=true evidence=", output)
    flush(stdout)
    clock_check()
    return nothing
end

@testset "Independent strict Julia syntax contracts" begin
    parent = realpath(get(ENV, "RUNNER_TEMP", tempdir()))
    evidence = mktempdir(parent; prefix="matched342-source-syntax-", cleanup=false)
    output = joinpath(evidence, "oracles")
    try
        @testset "Always-on portable parser contracts" begin
            portable_contracts(evidence)
        end
        if VERSION in (v"1.12.6", v"1.13.1")
            @test JuliaSourceSyntaxAudit.run_oracles(output, string(VERSION)) === 0
        end
        println("STRICT_JULIA_SYNTAX_EVIDENCE=", evidence)
    catch
        println(stderr, "STRICT_JULIA_SYNTAX_FAILURE_EVIDENCE=", evidence)
        rethrow()
    end
end
end
