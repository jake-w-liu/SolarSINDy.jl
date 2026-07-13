using Test

module ExternalDstCollectorTestHarness
using Test

# In-package collector (examples/). It is a committed part of the package, so a missing file is a
# real regression, not an environment-specific skip.
const EXTERNAL_DST_COLLECTOR_SCRIPT = normpath(joinpath(@__DIR__, "..", "examples",
                                                        "external_dst_snapshot_collector.jl"))

if isfile(EXTERNAL_DST_COLLECTOR_SCRIPT)
    include(EXTERNAL_DST_COLLECTOR_SCRIPT)
end

end

@testset "Prospective external Dst snapshot collector" begin
    @test isfile(ExternalDstCollectorTestHarness.EXTERNAL_DST_COLLECTOR_SCRIPT)
    C = ExternalDstCollectorTestHarness
    @test C._parse_http_last_modified(["Last-Modified" => "Sat, 27 Jun 2026 05:10:00 GMT"]) ==
          C.DateTime(2026, 6, 27, 5, 10, 0)
    @test C._parse_temerin_model_run("Time of model run:     2026/178-05:05:44") ==
          C.DateTime(2026, 6, 27, 5, 5, 44)
    @test C._sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    @test C._selftest_external_dst_collector()
end
