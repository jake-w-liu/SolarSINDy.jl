module StrictJuliaSyntaxPackageTests
using Test
include("run_syntax_audit.jl")

@testset "Independent strict Julia syntax contracts" begin
    parent = realpath(get(ENV, "RUNNER_TEMP", tempdir()))
    evidence = mktempdir(parent; prefix="matched342-source-syntax-", cleanup=false)
    output = joinpath(evidence, "oracles")
    try
        @test JuliaSourceSyntaxAudit.run_oracles(output, string(VERSION)) === 0
        println("STRICT_JULIA_SYNTAX_EVIDENCE=", evidence)
    catch
        println(stderr, "STRICT_JULIA_SYNTAX_FAILURE_EVIDENCE=", evidence)
        rethrow()
    end
end
end
