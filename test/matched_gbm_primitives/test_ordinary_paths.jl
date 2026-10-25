module Matched342OrdinaryPathTests
using Test,SHA,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
@testset "Matched342 physical artifact roots and explicit alias rejection" begin
    mktempdir() do raw_root
        root=realpath(raw_root)
        physical=joinpath(root,"physical");mkdir(physical)
        alias=joinpath(root,"alias")
        # Absolute existing-directory targets use junctions on Windows and
        # ordinary directory symlinks on Unix, without privilege changes.
        symlink(physical,alias;dir_target=true)
        try
            @test realpath(alias)==physical
            @test alias!=physical
            println("MATCHED342_TEMP_ROOT raw=",raw_root," physical=",root)
            encoder=M.Encoder342(M.TrainingRows342([1,2]),ntuple(_->0,165),ntuple(_->0.0,165),
                ntuple(_->0.0,342),ntuple(_->1.0,342),repeat("a",64))
            first=M.save_encoder(joinpath(physical,"first.json"),encoder)
            loaded=M.load_encoder(first.path;expected_sha256=first.sha256,expected_training_design_sha256=repeat("a",64))
            @test loaded.training_rows==[1,2]
            @test loaded.center===encoder.center
            @test_throws ArgumentError M.save_encoder(joinpath(alias,"blocked.json"),encoder)
            @test !ispath(joinpath(physical,"blocked.json"))
            @test_throws ArgumentError M.load_encoder(joinpath(alias,"first.json");expected_sha256=first.sha256,expected_training_design_sha256=repeat("a",64))
            @test bytes2hex(sha256(read(first.path)))==first.sha256
            second=M.save_encoder(joinpath(realpath(alias),"second.json"),encoder)
            @test second.sha256==first.sha256
            @test second.bytes==first.bytes
            @test read(second.path)==read(first.path)
            @test M.load_encoder(second.path;expected_sha256=second.sha256,expected_training_design_sha256=repeat("a",64)).training_rows==[1,2]
        finally
            # Remove only the owned alias, never recursively through it.
            rm(alias)
        end
    end
end
end
