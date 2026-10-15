using Test, SolarSINDy

@testset "Typed ACI refuses unrepresentable centers and interval bounds" begin
    base = Int64(2)^60
    for (point, observed, history) in ((base, base + 1, [1.0]),
                                      (base + 1, base, [1.0]),
                                      (base, base, [1.0]),
                                      (1.0, 1.0, [nextfloat(0.0)]),
                                      (1e308, 1e308, [1e308]),
                                      (1e308, -1e308, [3.0]))
        ac = init_adaptive_conformal(warmup=0)
        append!(ac.history, history)
        append!(ac.scratch, [3.0, 4.0])
        previous = (alpha=ac.alpha_t, history=copy(ac.history), scratch=copy(ac.scratch))
        @test_throws ArgumentError adaptive_conformal_step!(ac, point, observed)
        @test (alpha=ac.alpha_t, history=ac.history, scratch=ac.scratch) == previous
    end
    # A tiny unused candidate must not reject the actually selected large band.
    ac = init_adaptive_conformal(warmup=0)
    append!(ac.history, [nextfloat(0.0), 10.0])
    result = adaptive_conformal_step!(ac, 1.0, 1.0)
    @test result.lo == -9.0 && result.hi == 11.0 && result.covered === true
    # Integer/Float32 values exactly representable by Float64 remain supported.
    ac = init_adaptive_conformal(warmup=0)
    adaptive_conformal_step!(ac, Int32(-10), Float32(-11))
    @test ac.history == [1.0]
    gap_alpha, gap_history = ac.alpha_t, copy(ac.history)
    gap = adaptive_conformal_step!(ac, NaN, -11.0)
    @test gap.covered === missing
    @test ac.alpha_t == gap_alpha && ac.history == gap_history
    cold = adaptive_conformal_step!(init_adaptive_conformal(warmup=0), 0.0, 0.0)
    @test cold.lo == -Inf && cold.hi == Inf && cold.covered === true
end
