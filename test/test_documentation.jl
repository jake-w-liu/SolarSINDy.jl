using Test, SolarSINDy

@testset "Every public binding has documentation" begin
    for name in names(SolarSINDy)
        @test Base.Docs.hasdoc(SolarSINDy,name)
    end
end
