module DocumentationRuntimeContractTests
using Test

function docs_contract(workflow,makefile)
    workflow=replace(workflow,"\r\n"=>"\n")
    section=match(r"(?ms)^  docs:\n(.*?)(?=^  [A-Za-z_][A-Za-z0-9_-]*:\n|\z)",workflow)
    section===nothing && return false
    body=join(filter(line->!startswith(strip(line),"#"),split(section.captures[1],'\n')),'\n')
    occursin(r"(?m)^    name: Documentation build$",body) || return false
    occursin(r"(?m)^    runs-on: ubuntu-latest$",body) || return false
    occursin(r"(?m)^    timeout-minutes: 60$",body) || return false
    occursin(r"(?m)^\s*(?:if|continue-on-error):",body) && return false
    occursin(r"\|\|\s*(?:true|exit\s+0)(?:\s|$)",body) && return false
    occursin(r"(?m)^\s*set\s+\+e",body) && return false
    expected=Set(("1.12.6","1.13.1"));active=nothing;builds=String[]
    for block in split(body,r"(?m)^      - ")[2:end]
        if startswith(block,"uses: julia-actions/setup-julia@v2\n")
            version=match(r"""(?m)^ {10}version: ['"]([0-9.]+)['"]\s*$""",block)
            version===nothing && return false
            active=version.captures[1]
            active in expected || return false
        elseif occursin("include(\"docs/make.jl\")",block)
            occursin("run: julia --startup-file=no --project=docs -e",block) || return false
            version=match(r"""VERSION\s*==\s*v"([0-9.]+)"\s*""",block)
            version===nothing && return false
            active===nothing && return false
            version.captures[1]==active || return false
            push!(builds,active)
        end
    end
    length(builds)==2 && Set(builds)==expected || return false
    makefile=join(filter(line->!startswith(strip(line),"#"),split(makefile,'\n')),'\n')
    occursin(r"checkdocs\s*=\s*:exports",makefile) &&
        occursin(r"doctest\s*=\s*true",makefile) &&
        occursin(r"warnonly\s*=\s*false",makefile)
end

function change_docs_setup(workflow,before,after)
    section=match(r"(?ms)^  docs:\n.*?(?=^  [A-Za-z_][A-Za-z0-9_-]*:\n|\z)",workflow)
    section===nothing && error("documentation section missing from mutation fixture")
    occursin(before,section.match) || error("documentation setup mutation did not match")
    changed=replace(section.match,before=>after;count=1)
    replace(workflow,section.match=>changed;count=1)
end

@testset "Both supported runtimes execute the strict manual" begin
    root=normpath(joinpath(@__DIR__,".."))
    workflow=read(joinpath(root,".github","workflows","ci.yml"),String)
    makefile=read(joinpath(root,"docs","make.jl"),String)
    @test docs_contract(workflow,makefile)
    # A source-text contract is useful only if independent execution mutations
    # remove actual coverage and are rejected, including misleading comments.
    marker="      - uses: julia-actions/setup-julia@v2\n        with:\n          version: '1.13.1'\n"
    @test occursin(marker,workflow)
    without_setup=replace(workflow,marker=>"";count=1)
    @test !docs_contract(without_setup,makefile)
    @test !docs_contract(change_docs_setup(workflow,"version: '1.13.1'","version: '1.12.6'"),makefile)
    # An unrelated earlier setup must neither satisfy nor intercept the docs mutation.
    unrelated="  unrelated:\n    steps:\n      - uses: julia-actions/setup-julia@v3\n        with:\n          version: '1.13.1'\n"
    @test docs_contract(unrelated*workflow,makefile)
    @test !docs_contract(change_docs_setup(unrelated*workflow,"version: '1.13.1'","version: '1.12.6'"),makefile)
    @test !docs_contract(replace(workflow,"VERSION == v\"1.13.1\""=>"VERSION == v\"1.12.6\"";count=1),makefile)
    @test !docs_contract(replace(workflow,"VERSION == v\"1.13.1\""=>"VERSION == v\"1.14.0\"";count=1),makefile)
    @test !docs_contract(replace(workflow,"    name: Documentation build\n"=>"    name: Documentation build\n    if: false\n";count=1),makefile)
    @test !docs_contract(replace(workflow,"      - name: Build the manual (Julia 1.13.1)\n"=>"      - name: Build the manual (Julia 1.13.1)\n        continue-on-error: true\n";count=1),makefile)
    @test !docs_contract(replace(workflow,"include(\"docs/make.jl\")'\n\n  containers:"=>"include(\"docs/make.jl\")' || true\n\n  containers:";count=1),makefile)
    @test !docs_contract(workflow,replace(makefile,"checkdocs = :exports"=>"checkdocs = :none";count=1))
    @test !docs_contract(workflow,replace(makefile,"doctest = true"=>"doctest = false";count=1))
    @test !docs_contract(workflow,replace(makefile,"warnonly = false"=>"warnonly = true";count=1))
    # Commenting out both the setup and native build cannot count as runtime13.
    first=findfirst(marker,workflow);last=findnext("\n  containers:",workflow,Base.last(first)+1)
    @test first!==nothing && last!==nothing
    tail=workflow[Base.first(first):Base.first(last)-1]
    commented=join(("# "*line for line in split(tail,'\n')),'\n')
    @test !docs_contract(replace(workflow,tail=>commented;count=1),makefile)
    @test !docs_contract(without_setup*"\n# version: '1.13.1'; include(\"docs/make.jl\")\n",makefile)
end
end
