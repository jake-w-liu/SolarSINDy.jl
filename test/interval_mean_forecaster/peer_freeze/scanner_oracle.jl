# Independent AST evidence, old-source witness and future scanner assertions.
using Test,SHA,TOML
include(joinpath(@__DIR__,"literal_cases.jl"))
const TARGET=abspath(joinpath(@__DIR__,"..","..","test_compat.jl"))
const MODE=isempty(ARGS) ? "verify" : ARGS[1]
function contains_parse_error(value)
    value isa Expr || return false
    value.head in (:error,:incomplete) && return true
    any(contains_parse_error,value.args)
end
function scanner_module(path)
    ast=Meta.parseall(read(path,String);filename=path)
    contains_parse_error(ast) && error("scanner source malformed")
    top=only(filter(x->x isa Expr && x.head==:module,ast.args))
    wrapper=Module(gensym(:PeerScanner))
    for node in top.args[end].args
        # Julia attaches docstrings through @doc macrocall wrappers.
        if node isa Expr && node.head==:macrocall && node.args[end] isa Expr && node.args[end].head==:function
            node=node.args[end]
        end
        node isa Expr && node.head==:function || continue
        Core.eval(wrapper,node)
    end
    wrapper
end
function staged_probe(f)
    directory=mktempdir(@__DIR__;prefix="active-probe-",cleanup=false)
    try
        f(directory)
    finally
        # The only recursive removal is this freshly created owned probe subtree.
        dirname(abspath(directory))==abspath(@__DIR__) || error("cleanup escaped peer subtree")
        rm(directory;recursive=true,force=true)
    end
end
source_sha=bytes2hex(sha256(read(TARGET)))
println("SCANNER_SOURCE_SHA256=",source_sha)
if MODE=="ast"
    for case in CASES
        ast=Meta.parseall(case.source;filename=case.name*".jl")
        println("CASE=",case.name," EXPECTED=",repr(case.expected)," MALFORMED=",case.malformed)
        println("NATIVE_AST=",repr(ast))
        @test contains_parse_error(ast)==case.malformed
    end
elseif MODE=="witness"
    N=scanner_module(TARGET)
    staged_probe() do directory
        write(joinpath(directory,"comma.jl"),"using Dates,SHA,TOML\n")
        actual=Base.invokelatest(Base.invokelatest(getfield,N,:_loaded_modules),directory)
        println("OLD_ACTUAL=",repr(sort!(collect(actual))))
        println("INDEPENDENT_EXPECTED=",repr(["Dates","SHA","TOML"]))
        @testset "Old parser comma false-negative witness" begin
            @test actual==Set(["Dates","SHA","TOML"])
        end
    end
elseif MODE=="verify"
    N=scanner_module(TARGET)
    @testset "Frozen literal module scanner oracles" begin
        for case in CASES
            @testset "$(case.name)" begin
                staged_probe() do directory
                    write(joinpath(directory,"fixture.jl"),case.source)
                    write(joinpath(directory,"ignored.txt"),"using IgnoredNonJulia\n")
                    if case.malformed
                        @test_throws Exception Base.invokelatest(Base.invokelatest(getfield,N,:_loaded_modules),directory)
                    else
                        @test Base.invokelatest(Base.invokelatest(getfield,N,:_loaded_modules),directory)==Set(case.expected)
                    end
                end
            end
        end
    end
else
    error("mode must be ast, witness or verify")
end
@test bytes2hex(sha256(read(TARGET)))==source_sha

