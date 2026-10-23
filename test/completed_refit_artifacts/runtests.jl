module CompletedRefitArtifactsTests
using Test,Dates
using SolarSINDy
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const CODEC_TEST_DIRECTORY=mktempdir(;prefix="solarsindy-refit-tests-",cleanup=false)
include("protocol_fixture.jl")
include("store_fixture.jl")
include("test_codec.jl")
include("test_codec_preflight.jl")
include("test_codec_file_bound.jl")
include("test_protocol.jl")
include("test_store_cache.jl")
include("test_numerical.jl")
include("test_numerical_diagnostics.jl")
include("test_dyadic_precision.jl")
include("test_runtime_semantics.jl")
include("test_source_reconstruction.jl")
end
