using Test, SolarSINDy, Dates, Random

@testset "Immutable point kernel agrees with independent physical terms" begin
    rng = MersenneTwister(20261013)
    for redundant in (false, true), clock_basis in (:full, :collapsed)
        lib = build_solar_wind_library(include_redundant_n_operational=redundant,
                                       clock_basis=clock_basis)
        buffer = zeros(length(lib))
        for _ in 1:50
            dst = -600 * rand(rng)
            speed = 300 + 900 * rand(rng)
            bz, by = 30 * randn(rng), 30 * randn(rng)
            density, pressure = 1 + 30 * rand(rng), 1 + 100 * rand(rng)
            south = bz < 0 ? -bz : 0.0
            transverse = sqrt(by * by + bz * bz)
            half_sine = sin(atan(abs(by), bz) / 2)
            oracle = Dict(
                "1"=>1.0, "V"=>speed, "Bs"=>south, "n"=>density,
                "Pdyn"=>pressure, "Dst_star"=>dst,
                "V^2"=>speed * speed, "Bs^2"=>south * south,
                "n^2"=>density * density, "V*Bs"=>speed * south,
                "n*V"=>density * speed, "n*Bs"=>density * south,
                "Pdyn*Bs"=>pressure * south, "n*V*Bs"=>density * speed * south,
                "n*V^2"=>density * speed * speed,
                "sin(θ_c/2)"=>half_sine, "sin²(θ_c/2)"=>half_sine * half_sine,
                "sin⁴(θ_c/2)"=>half_sine^4,
                "sin^(8/3)(θ_c/2)"=>half_sine^(8 / 3),
                "V*sin²(θ_c/2)"=>speed * half_sine * half_sine,
                "Newell_d_Φ"=>speed^(4 / 3) * transverse^(2 / 3) * half_sine^(8 / 3),
            )
            SolarSINDy._evaluate_point_vector!(buffer, lib, dst, speed, bz, by,
                                               density, pressure)
            expected = [oracle[name] for name in get_term_names(lib)]
            # Independently computed hypot/square and product groupings introduce
            # only rounding error; this bound covers their few arithmetic steps.
            @test buffer ≈ expected rtol=16eps(Float64) atol=0
            data = Dict("V"=>[speed], "Bz"=>[bz], "By"=>[by],
                        "Bs"=>[south], "n"=>[density], "Pdyn"=>[pressure],
                        "Dst_star"=>[dst], "theta_c"=>[atan(abs(by), bz)])
            @test vec(evaluate_library(lib, data)) == buffer
        end
        SolarSINDy._evaluate_point_vector_unchecked!(
            buffer, lib, -50.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        )
        @test (@allocated SolarSINDy._evaluate_point_vector_unchecked!(
            buffer, lib, -50.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        )) == 0
        lib.term_codes[1] = UInt8(2)
        @test_throws ArgumentError SolarSINDy._evaluate_point_vector!(
            buffer, lib, -50.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        )
    end
    # Custom functions keep their semantic contract even when their display name
    # collides with a built-in term; the immutable homogeneous code tuple is zero.
    custom = CandidateLibrary(["V", "state_square"],
        Function[data -> 2 .* data["V"], data -> data["Dst_star"].^2])
    buffer = zeros(2)
    SolarSINDy._evaluate_point_vector!(buffer, custom, -5.0, 400.0, -5.0, 0.0, 5.0, 2.0)
    @test buffer == [800.0, 25.0]
    @test simulate_sindy([0.0, -0.001], custom,
        SolarWindData([0.0, 1.0], fill(400.0, 2), fill(-5.0, 2), zeros(2),
                      fill(5.0, 2), fill(2.0, 2), fill(-5.0, 2), fill(-5.0, 2)),
        1.0; Dst0=-5.0) == [-5.0, -5.025]
end
