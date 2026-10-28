# Fixed metadata transport controls; no network or observation access.
module CMEWindMetadataTests
using Test
python = get(ENV, "SOLARSINDY_RESEARCH_PYTHON", Sys.iswindows() ? "python" : "python3")
script = joinpath(@__DIR__, "cme_wind_metadata", "test_capture.py")
@testset "Bounded Wind metadata transport without source authority" begin
    @test success(`$python -B -X utf8 $script`)
end
end
