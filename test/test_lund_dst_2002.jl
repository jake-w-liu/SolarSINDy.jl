using Test, JSON3
using SolarSINDy

@testset "Published Lund 2002 reference kernel" begin
    model = SolarSINDy.LundDst2002
    oracle = JSON3.read(read(joinpath(@__DIR__, "lund_dst_2002_oracle.json")),
                        Dict{String,Any})
    @test model.initial_state() == (0.39, 0.22, -0.83, 0.14)
    previous = model.initial_state()
    for row in oracle["rows"]
        actual = model.predict_next(previous, row["bz_nt"], row["density_cm3"],
                                    row["speed_kms"])
        @test isapprox(collect(actual.state), row["state"]; rtol=0, atol=2e-13)
        @test isapprox(actual.dst_nt, row["dst_nt"]; rtol=0, atol=3e-11)
        previous = actual.state
    end
    bz = Float64[r["bz_nt"] for r in oracle["rows"]]
    density = Float64[r["density_cm3"] for r in oracle["rows"]]
    speed = Float64[r["speed_kms"] for r in oracle["rows"]]
    original = copy(bz), copy(density), copy(speed)
    complete = model.rollout(bz, density, speed)
    @test isapprox(complete.dst_nt, Float64[r["dst_nt"] for r in oracle["rows"]];
                   rtol=0, atol=3e-11)
    @test complete.state == previous
    @test (bz, density, speed) == original
    first = model.rollout(bz[1:11], density[1:11], speed[1:11])
    second = model.rollout(bz[12:end], density[12:end], speed[12:end]; state=first.state)
    @test vcat(first.dst_nt, second.dst_nt) == complete.dst_nt
    @test second.state == complete.state
    changed = copy(bz); changed[12:end] .= -80
    alternative = model.rollout(changed, density, speed)
    @test alternative.dst_nt[1:11] == complete.dst_nt[1:11]
    @test alternative.dst_nt[12:end] != complete.dst_nt[12:end]
    empty = model.rollout(Float64[], Float64[], Float64[]; state=first.state)
    @test isempty(empty.dst_nt) && empty.state == first.state
    mutable_state = collect(model.initial_state())
    @test model.predict_next(mutable_state, 0, 5, 400) ==
          model.predict_next(model.initial_state(), 0, 5, 400)
    @test mutable_state == collect(model.initial_state())
    @test_throws DimensionMismatch model.rollout([0.], [1., 2.], [400.])
    @test_throws DimensionMismatch model.predict_next((0., 0.), 0., 1., 400.)
    for state in ((NaN, 0., 0., 0.), (Inf, 0., 0., 0.), (1.01, 0., 0., 0.))
        @test_throws ArgumentError model.predict_next(state, 0., 1., 400.)
    end
    for drivers in ((NaN, 5., 400.), (Inf, 5., 400.), (0., -1., 400.),
                    (0., Inf, 400.), (0., 5., -1.), (0., 5., Inf))
        @test_throws ArgumentError model.predict_next(model.initial_state(), drivers...)
    end
    @test_throws ArgumentError model.predict_next(model.initial_state(), big"1e1000", 5., 400.)
    @test_throws ArgumentError model.rollout([0., NaN], [5., 5.], [400., 400.]; state=mutable_state)
    @test mutable_state == collect(model.initial_state())
    # Independent output-envelope consequence of bounded tanh coordinates.
    extreme = model.rollout(repeat([-80., 80., 0.], 1000),
                           repeat([0., 200., 5.], 1000),
                           repeat([0., 2000., 400.], 1000))
    @test all(isfinite, extreme.dst_nt)
    @test all(x -> -283.015 <= x <= 104.375, extreme.dst_nt)
    @test all(x -> -1 <= x <= 1, extreme.state)
end
