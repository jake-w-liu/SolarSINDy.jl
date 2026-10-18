using Test
length(ARGS)==1 || error("COMPONENT_DIRECTORY")
component=abspath(ARGS[1])
include(joinpath(component,"DstInitialGBMReference.jl"))
include(joinpath(component,"InitialGBMReplay.jl"))
include(joinpath(component,"BoundInitialReferenceInputs.jl"))
include(joinpath(component,"AllGridInitialGBMReference.jl"))
const G=AllGridInitialGBMReference
source=repeat("a",64);matrix=repeat("b",64)
actual=Dict{String,Any}("rows"=>1000,"channels"=>["channel$i" for i in 1:17],
    "start"=>"2019-01-01T00:00:00","stop"=>"2019-02-11T15:00:00",
    "encoding"=>"little-endian Float64 hour-major; provider raw values","sha256"=>matrix)
synthetic=Dict{String,Any}("rows"=>1000,"start"=>"2019-01-01T00:00:00",
    "source_sha256"=>source,"raw_matrix_sha256"=>matrix,"synthetic_only"=>true)
@testset "Explicit original and synthetic input metadata schemas" begin
    before=deepcopy(actual)
    @test G.input_metadata_identity(actual,"exposed_development",source)==(source_sha256=source,raw_matrix_sha256=matrix)
    @test actual==before
    @test G.input_metadata_identity(synthetic,"synthetic_fixture",source)==(source_sha256=source,raw_matrix_sha256=matrix)
    @test !haskey(actual,"source_sha256") && !haskey(actual,"raw_matrix_sha256")
    @test_throws ArgumentError G.input_metadata_identity(synthetic,"exposed_development",source)
    @test_throws ArgumentError G.input_metadata_identity(actual,"synthetic_fixture",source)
    @test_throws ArgumentError G.input_metadata_identity(actual,"unknown",source)
    @test_throws ArgumentError G.input_metadata_identity(actual,"exposed_development",source*"\n")
    for (meta,mode) in ((actual,"exposed_development"),(synthetic,"synthetic_fixture"))
        for count in (true,0,-1,1_000_001,1000.)
            bad=deepcopy(meta);bad["rows"]=count
            @test_throws ArgumentError G.input_metadata_identity(bad,mode,source)
        end
        bad=deepcopy(meta);bad["extra"]=1
        @test_throws ArgumentError G.input_metadata_identity(bad,mode,source)
        bad=deepcopy(meta);delete!(bad,"rows")
        @test_throws ArgumentError G.input_metadata_identity(bad,mode,source)
        key=mode=="synthetic_fixture" ? "raw_matrix_sha256" : "sha256"
        for digest in (true,matrix*"\n",uppercase(matrix),"invalid")
            bad=deepcopy(meta);bad[key]=digest
            @test_throws ArgumentError G.input_metadata_identity(bad,mode,source)
        end
    end
    bad=deepcopy(actual);bad["encoding"]="native Float64"
    @test_throws ArgumentError G.input_metadata_identity(bad,"exposed_development",source)
    for channels in (fill("same",17),["channel$i" for i in 1:16],collect(1:17))
        bad=deepcopy(actual);bad["channels"]=channels
        @test_throws ArgumentError G.input_metadata_identity(bad,"exposed_development",source)
    end
    bad=deepcopy(synthetic);bad["synthetic_only"]=1
    @test_throws ArgumentError G.input_metadata_identity(bad,"synthetic_fixture",source)
    bad=deepcopy(synthetic);bad["source_sha256"]=matrix
    @test_throws ArgumentError G.input_metadata_identity(bad,"synthetic_fixture",source)
end
