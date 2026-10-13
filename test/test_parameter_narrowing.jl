using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "constructors reject narrowing that invalidates their contracts" begin
        tiny = BigFloat("1e-1000")
        enormous = BigFloat("1e1000")
        near_one = BigFloat(1) - BigFloat("1e-70")
        @test_throws ArgumentError init_adaptive_conformal(gamma=tiny)
        @test_throws ArgumentError init_adaptive_conformal(gamma=enormous)
        @test_throws ArgumentError init_adaptive_conformal(target_coverage=tiny)
        @test_throws ArgumentError init_adaptive_conformal(target_coverage=near_one)
        lib = build_minimal_library()
        for parameter in (:R, :dt, :dst_var0, :coeff_var0, :q_dst, :q_coeff)
            @test_throws ArgumentError init_assimilation(
                lib, zeros(3), [2], 0.0; NamedTuple{(parameter,)}((tiny,))...,
            )
        end
        for simulate in (simulate_burton, simulate_burton_full, simulate_obrien)
            @test_throws ArgumentError simulate(fill(400.0, 2), fill(1.0, 2), tiny)
            @test_throws ArgumentError simulate(fill(400.0, 2), fill(1.0, 2), 1.0; α=tiny)
        end
        @test_throws ArgumentError burton_model([400.0], [1.0], [0.0]; τ=tiny)
        # Representable small positive values and explicit zero process noise are
        # valid; validation must distinguish zero from a narrowed nonzero input.
        ac = init_adaptive_conformal(gamma=nextfloat(0.0))
        @test ac.gamma > 0.0
        f = init_assimilation(lib, zeros(3), Int[], 0.0;
                              dst_var0=0.0, coeff_var0=0.0, q_dst=0.0, q_coeff=0.0)
        @test f.cov == zeros(1, 1)
    end

