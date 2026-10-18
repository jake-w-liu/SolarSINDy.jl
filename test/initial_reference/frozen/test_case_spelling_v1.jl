using Test
length(ARGS)==1 || error("COMPONENT_DIRECTORY")
component=abspath(ARGS[1])
include(joinpath(component,"DstInitialGBMReference.jl"))
include(joinpath(component,"BoundInitialReferenceInputs.jl"))
const B=BoundInitialReferenceInputs
@testset "Canonical spelling rejects case aliases before admission" begin
    owner=realpath(mktempdir(;prefix="solarsindy-case-spelling-",cleanup=false))
    root=joinpath(owner,"owned");mkdir(root)
    file=joinpath(root,"actual.json");write(file,"{}")
    before=read(file)
    @test B.root_path(root)==root
    @test B.resolved(Dict("input"=>root),"input","actual.json")==file
    @test B.actual_spelling(file)==file
    @test_throws ArgumentError B.resolved(Dict("input"=>root),"input","ACTUAL.json")
    @test_throws ArgumentError B.root_path(joinpath(owner,"OWNED"))
    if Sys.iswindows()
        @test isfile(joinpath(root,"ACTUAL.json"))
        @test isdir(joinpath(owner,"OWNED"))
        @test B.actual_spelling(joinpath(root,"ACTUAL.json"))==file
        @test B.actual_spelling(joinpath(owner,"OWNED"))==root
    end
    @test read(file)==before
end
