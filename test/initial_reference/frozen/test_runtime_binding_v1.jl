using Test, SHA
length(ARGS)==1 || error("COMPONENT_DIRECTORY")
component=abspath(ARGS[1])
include(joinpath(component,"DstInitialGBMReference.jl"))
include(joinpath(component,"InitialGBMReplay.jl"))
include(joinpath(component,"BoundInitialReferenceInputs.jl"))
include(joinpath(component,"AllGridInitialGBMReference.jl"))
const G=AllGridInitialGBMReference
executing=realpath(joinpath(Sys.BINDIR,Base.julia_exename()))
digest=bytes2hex(sha256(read(executing)))
function binding(path=basename(executing),sha=digest)
    (roots=Dict("julia"=>dirname(executing)),
     plan=Dict("support"=>Dict("julia"=>Dict("root"=>"julia","path"=>path,"sha256"=>sha))))
end
@testset "Native process binary binding with caller-pinned identity" begin
    record=G.execution_runtime(binding())
    @test record.path==executing
    @test record.sha256==digest
    @test record.version==string(VERSION)
    @test record.threads==Threads.nthreads()
    @test G.execution_runtime(binding())==record
    wrong=repeat(digest[1]=='a' ? "b" : "a",64)
    @test_throws ArgumentError G.execution_runtime(binding(basename(executing),wrong))
    @test_throws ArgumentError G.execution_runtime(binding(basename(executing),digest*"\n"))
    @test_throws ArgumentError G.execution_runtime(binding("../"*basename(executing),digest))
    @test_throws ArgumentError G.execution_runtime(binding("does-not-exist",digest))
    @test_throws ArgumentError G.execution_runtime((roots=Dict("julia"=>dirname(executing)),plan=Dict("support"=>Dict())))
    other=deepcopy(binding())
    delete!(other.plan["support"]["julia"],"sha256")
    @test_throws ArgumentError G.execution_runtime(other)
    # A different existing file has both a valid digest and a valid relative
    # namespace. This tests executable identity, not just digest grammar.
    foreign=basename(@__FILE__)
    root=dirname(@__FILE__)
    other=(roots=Dict("julia"=>root),plan=Dict("support"=>Dict("julia"=>Dict(
        "root"=>"julia","path"=>foreign,"sha256"=>bytes2hex(sha256(read(@__FILE__)))))))
    @test_throws ArgumentError G.execution_runtime(other)
end
