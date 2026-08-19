module CompatContractTests

# The declared environment contract: what this package claims to need, and on which Julia it claims
# to run.
#
# Two failures live here and neither is visible to any other test, because both are properties of
# `Project.toml` rather than of running code:
#
#   * a `[compat]` bound on a standard library pins the package to the Julia release that shipped
#     that stdlib version, so the package stops resolving on its own declared minimum — the state the
#     package shipped in, with `Logging = "1.11.0"` beside `julia = "1.10"`;
#   * a `[deps]` entry that nothing in the repository uses is dead weight every consumer installs.
#
# The dependency check distinguishes what `src/` uses from what the package's own shipped scripts and
# test suite use under this same project environment. Both are legitimate reasons to declare a
# dependency; neither of them being true is not.

using Test
using TOML

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT = TOML.parsefile(joinpath(PACKAGE_ROOT, "Project.toml"))
const DEPS = PROJECT["deps"]
const COMPAT = get(PROJECT, "compat", Dict{String,Any}())

"""
Standard libraries this package depends on. Their versions track the Julia release, so a `[compat]`
bound on one is a Julia pin in disguise.
"""
const STDLIB_DEPS = ["Dates", "Downloads", "FileWatching", "LinearAlgebra", "Logging", "Printf",
                     "Random", "SHA", "Statistics"]

"""
Dependencies that `src/` does not load, but that the package's own shipped code needs under this
project environment: the figure-generation and study scripts under `validation/`, the operational
scripts under `examples/`, and the test files that include them. Each entry names the consumer that
keeps it.
"""
const ENVIRONMENT_ONLY_DEPS = Dict(
    "PlotlySupply" => "validation/canonical_figure_generation.jl (loaded by the test suite) and the " *
                      "figure scripts",
    "Printf" => "examples/external_dst_snapshot_collector.jl and the validation reports",
    "Logging" => "validation/operational/v2_3_common.jl",
)

"Names loaded by `using`/`import` anywhere under `directory`."
function _loaded_modules(directory::AbstractString)
    found = Set{String}()
    for (root, _, files) in walkdir(directory), file in files
        endswith(file, ".jl") || continue
        for line in eachline(joinpath(root, file))
            m = match(r"^\s*(?:using|import)\s+([A-Za-z][A-Za-z0-9_]*)", line)
            m === nothing || push!(found, m.captures[1])
        end
    end
    return found
end

@testset "Declared environment contract" begin
    @testset "no standard library carries a version bound" begin
        # `Logging = \"1.11.0\"` with `julia = \"1.10\"` made the package unresolvable on its own
        # declared minimum: Julia 1.10 ships Logging without a version at all.
        for dep in STDLIB_DEPS
            @test haskey(DEPS, dep)
            @test !haskey(COMPAT, dep)
        end
        # Every other bound must name a package that is actually a dependency.
        for name in keys(COMPAT)
            name == "julia" && continue
            @test haskey(DEPS, name)
        end
    end

    @testset "the declared minimum Julia is the one the documentation promises" begin
        @test haskey(COMPAT, "julia")
        declared = String(COMPAT["julia"])
        @test declared == "1.10"
        readme = read(joinpath(PACKAGE_ROOT, "README.md"), String)
        @test occursin("Requires Julia $(declared)+", readme)
    end

    @testset "every dependency has a consumer" begin
        src_modules = _loaded_modules(joinpath(PACKAGE_ROOT, "src"))
        repo_modules = union(src_modules,
                             _loaded_modules(joinpath(PACKAGE_ROOT, "validation")),
                             _loaded_modules(joinpath(PACKAGE_ROOT, "examples")),
                             _loaded_modules(joinpath(PACKAGE_ROOT, "test")))
        for dep in keys(DEPS)
            # A dependency is justified either by `src/` loading it, or by being one of the
            # declared environment-only dependencies whose named consumer still loads it.
            @test (dep in src_modules) ||
                  (haskey(ENVIRONMENT_ONLY_DEPS, dep) && dep in repo_modules)
            dep in src_modules || @test haskey(ENVIRONMENT_ONLY_DEPS, dep)
        end
        # The declared environment-only set may not outlive its consumers either.
        for dep in keys(ENVIRONMENT_ONLY_DEPS)
            @test haskey(DEPS, dep)
            @test !(dep in src_modules)
            @test dep in repo_modules
        end
    end
end

end # module CompatContractTests
