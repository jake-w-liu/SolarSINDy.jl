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
using Pkg
using SolarSINDy

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT = TOML.parsefile(joinpath(PACKAGE_ROOT, "Project.toml"))
const DEPS = PROJECT["deps"]
const COMPAT = get(PROJECT, "compat", Dict{String,Any}())

# Derive the required image inputs from the package's actual load-time source
# contract. This is deliberately limited to this image's literal COPY layout.
function _monitor_copies_point_sources_before_precompile(source)
    copies=Tuple{String,String}[];precompile_seen=false
    for raw in split(source,'\n')
        line=strip(raw)
        if startswith(line,"RUN ") && occursin("Pkg.precompile()",line)
            precompile_seen=true;break
        end
        startswith(line,"COPY ") || continue
        words=split(line);length(words)>=3 || return false
        any(w->startswith(w,"--"),words[2:end]) && return false
        destination=replace(normpath(words[end]),'\\'=>'/')
        for original in words[2:end-1]
            push!(copies,(replace(normpath(original),'\\'=>'/'),destination))
        end
    end
    function supplies(name,original,destination)
        if isdir(joinpath(PACKAGE_ROOT,original))
            if original=="."
                relative=name
            elseif startswith(name,original*"/")
                relative=name[length(original)+2:end]
            else
                return false
            end
            target=joinpath(destination,relative)
        else
            name==original || return false
            target=joinpath(destination,basename(original))
        end
        replace(normpath(target),'\\'=>'/')==name
    end
    precompile_seen && all(name->any(pair->supplies(name,pair...),copies),
        SolarSINDy.DurationPointTraining.SOURCE_FILES)
end

"""
Standard libraries this package depends on. Their versions track the Julia release, so a `[compat]`
bound on one is a Julia pin in disguise.
"""
const STDLIB_DEPS = ["Dates", "Downloads", "FileWatching", "LinearAlgebra", "Logging", "Printf",
                     "Random", "SHA", "Statistics", "TOML"]

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
    "Logging" => "validation/operational/analog_common.jl",
)

function _module_root(node)
    node isa Symbol && return node === :. ? nothing : String(node)
    node isa Expr || return nothing
    # Members and aliases do not introduce additional module dependencies.
    node.head in (:(:), :as) && return _module_root(first(node.args))
    node.head === :. || throw(ArgumentError("unrecognized module import syntax"))
    isempty(node.args) && throw(ArgumentError("empty module import path"))
    first(node.args) === :. && return nothing
    return _module_root(first(node.args))
end

function _collect_loaded_modules!(found, node)
    node isa Expr || return found
    node.head in (:error, :incomplete) && throw(ArgumentError("malformed Julia dependency source"))
    node.head === :quote && return found
    if node.head in (:using, :import)
        for declaration in node.args
            name = _module_root(declaration)
            name === nothing || push!(found, name)
        end
    else
        for child in node.args
            _collect_loaded_modules!(found, child)
        end
    end
    return found
end

"Names loaded by parsed, unquoted `using`/`import` declarations under `directory`."
function _loaded_modules(directory::AbstractString)
    found = Set{String}()
    for (root, _, files) in walkdir(directory), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        _collect_loaded_modules!(found, Meta.parseall(read(path, String); filename=path))
    end
    return found
end

# The user-facing platform contract supersedes the old single-Ubuntu job.
# Validate each axis and its actual job wiring; removing a platform, runtime,
# full test/experiment entry or error propagation must make this oracle fail.
function _cross_platform_ci_contract(workflow)
    job = match(r"\n  test:\n(.*?)(?=\n  [A-Za-z0-9_-]+:|\z)"s, workflow)
    job === nothing && return false
    body = job.captures[1]
    versions = match(r"^        julia-version: (\[[^\n]+\])$"m, body)
    platforms = match(r"^        os: \[([^\n]+)\]$"m, body)
    (versions === nothing || platforms === nothing) && return false
    releases = try
        TOML.parse("versions = " * versions.captures[1])["versions"]
    catch
        return false
    end
    releases == ["1.12.6", "1.13.1"] || return false
    operating_systems = strip.(split(platforms.captures[1], ','))
    length(operating_systems) == 3 && Set(operating_systems) ==
        Set(["ubuntu-latest", "windows-latest", "macos-latest"]) || return false
    all(fragment -> occursin(fragment, body), (
        "runs-on: \${{ matrix.os }}",
        "version: \${{ matrix.julia-version }}",
        "arch: 'default'", "shell: bash", "fail-fast: false",
        "uses: julia-actions/julia-runtest@v1",
        "run: julia --startup-file=no --project=. examples/experiments.jl",
    )) || return false
    !occursin("continue-on-error:", body)
end

@testset "Declared environment contract" begin
    @testset "CI bootstraps registries before resolving a clean depot" begin
        workflow = read(joinpath(PACKAGE_ROOT, ".github", "workflows", "ci.yml"), String)
        commands = filter(line -> occursin("run: julia --project", line) &&
                                  occursin("Pkg.resolve()", line), split(workflow, '\n'))
        @test length(commands) == 2  # package and documentation environments
        for command in commands
            registry = findfirst("Pkg.Registry.add(\"General\")", command)
            resolve = findfirst("Pkg.resolve()", command)
            @test registry !== nothing
            @test registry !== nothing && first(registry) < first(resolve)
        end
        @test _cross_platform_ci_contract(workflow)
        # Independent negative mutations enforce coverage and real execution.
        @test !_cross_platform_ci_contract(replace(workflow, ", macos-latest" => ""; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "'1.12.6', '1.13.1'" => "'1.12.6'"; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "runs-on: \${{ matrix.os }}" => "runs-on: ubuntu-latest"; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "fail-fast: false" => "fail-fast: true"; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "julia-actions/julia-runtest@v1" => "omitted-package-tests"; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "examples/experiments.jl" => "omitted-experiments.jl"; count=1))
        @test !_cross_platform_ci_contract(replace(workflow, "    strategy:" => "    continue-on-error: true\n    strategy:"; count=1))
        @test !occursin("1.10", workflow)
        @test !occursin("mv Manifest.toml", workflow)
    end

    @testset "bundled socket fixtures are available to Pkg.test" begin
        @test get(PROJECT["extras"], "Sockets", nothing) ==
              "6462fe0b-24de-5631-8697-dd941f90decc"
        @test "Sockets" in PROJECT["targets"]["test"]
        @test PROJECT["extras"]["Pkg"] == "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
        @test "Pkg" in PROJECT["targets"]["test"]
    end

    @testset "deployment images carry their runtime environment" begin
        for path in ("app/Dockerfile","deploy/Dockerfile.monitor")
            source = read(joinpath(PACKAGE_ROOT,path),String)
            @test occursin("ENV JULIA_DEPOT_PATH=/opt/julia-depot",source)
            @test occursin(r"chown -R swm:swm .* /opt/julia-depot",source)
            @test occursin("USER swm",source)
        end
        monitor = read(joinpath(PACKAGE_ROOT,"deploy/Dockerfile.monitor"),String)
        @test first(findfirst("COPY src ./src",monitor)) < first(findfirst("Pkg.precompile()",monitor))
        @test _monitor_copies_point_sources_before_precompile(monitor)
        point_copy="COPY tools/actual_b_chain ./tools/actual_b_chain"
        without_point_copy=replace(monitor,point_copy=>"";count=1)
        @test !_monitor_copies_point_sources_before_precompile(without_point_copy)
        @test !_monitor_copies_point_sources_before_precompile(without_point_copy*"\n"*point_copy*"\n")
        @test occursin("COPY data ./data",monitor)
        @test occursin("COPY validation/operational/superlearner_live_claim_audit.jl",monitor)
        compose = read(joinpath(PACKAGE_ROOT,"deploy/docker-compose.yml"),String)
        @test occursin("SOLARSINDY_REVIEW_DASH_URL=http://dashboard:8723/api/health",compose)
    end

    # G12-06: `restart: always` never acts on an unhealthy HEALTHCHECK, so the compose deployment
    # ships a label-scoped `autoheal` sidecar that restarts unhealthy monitor/dashboard containers.
    @testset "compose restarts unhealthy services (autoheal sidecar)" begin
        compose = replace(read(joinpath(PACKAGE_ROOT,"deploy/docker-compose.yml"),String),"\r\n"=>"\n")
        # Split the `services:` mapping into its two-space-indented service blocks.
        services_body = split(split(compose,r"^services:\n"m)[2],r"^\S"m)[1]
        blocks = Dict{String,String}()
        for m in eachmatch(r"^  ([A-Za-z0-9_-]+):\n((?:(?:    .*|[ ]*|  #.*)\n)*)"m,services_body)
            blocks[m.captures[1]] = m.captures[2]
        end
        @test Set(keys(blocks)) == Set(["monitor","dashboard","autoheal"])
        for svc in ("monitor","dashboard")
            @test occursin(r"^    labels:\n      - com\.solarsindy\.autoheal=true$"m,blocks[svc])
        end
        sidecar = blocks["autoheal"]
        @test occursin(r"^    image: docker:29-cli$"m,sidecar)
        @test occursin(r"^    restart: always$"m,sidecar)
        @test occursin(r"^    entrypoint: \[\"/bin/sh\", \"/autoheal\.sh\"\]$"m,sidecar)
        @test occursin(r"^      - /var/run/docker\.sock:/var/run/docker\.sock:ro$"m,sidecar)
        @test occursin(r"^      - \./autoheal\.sh:/autoheal\.sh:ro$"m,sidecar)
        @test occursin(r"^      - AUTOHEAL_LABEL=com\.solarsindy\.autoheal=true$"m,sidecar)
        @test occursin("root-equivalent",compose)              # socket security note
        script = read(joinpath(PACKAGE_ROOT,"deploy/autoheal.sh"),String)
        @test startswith(script,"#!/bin/sh\n")
        @test occursin("--filter \"health=unhealthy\"",script)
        @test occursin("label=com.docker.compose.project=",script)
        @test occursin("docker restart -t 30",script)
        for path in ("app/Dockerfile","deploy/Dockerfile.monitor")
            dockerfile = read(joinpath(PACKAGE_ROOT,path),String)
            @test occursin("deploy/autoheal.sh",dockerfile)
            @test !occursin("advisory only",dockerfile)
        end
    end

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

    @testset "one Julia release across package, app, and documentation" begin
        @test haskey(COMPAT, "julia")
        declared = String(COMPAT["julia"])
        @test declared == "1.12.6"
        supported = Pkg.Types.semver_spec(declared)
        @test v"1.12.6" in supported
        @test v"1.12.7" in supported
        @test v"1.12.99" in supported
        @test v"1.13.0" in supported
        @test v"1.13.99" in supported
        for version in (v"1.10.11", v"1.11.9", v"1.12.5", v"2.0.0")
            @test !(version in supported)
        end
        for environment in ("app", "docs")
            project = TOML.parsefile(joinpath(PACKAGE_ROOT, environment, "Project.toml"))
            @test project["compat"]["julia"] == declared
        end
        readme = read(joinpath(PACKAGE_ROOT, "README.md"), String)
        @test occursin("Julia **≥ $(declared)**", readme)
        @test !occursin("Julia 1.10+", readme)
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
