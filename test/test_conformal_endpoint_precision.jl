using Test, SolarSINDy

@testset "Conformal endpoints preserve a representable exact band" begin
    base = Int64(2)^60
    cal = fit_conformal([base], [base + 1], [1.0], [0.0]; min_stratum_n=1)
    lo, hi = conformal_interval(cal, base, 1.0, 0.0)
    @test lo == BigInt(base) - 1
    @test hi == BigInt(base) + 1
    @test lo < hi
    @test conformal_coverage(cal, [base], [base + 1], [1.0], [0.0]) == 1.0
    @test conformal_coverage(cal, [base], [base + 2], [1.0], [0.0]) == 0.0
    # A center that itself narrows must not translate the interval away from the
    # actual point, including the exactly zero half-width limiting case.
    exact_point = base + 1
    exact_cal = fit_conformal([exact_point], [exact_point], [1.0], [0.0])
    @test conformal_interval(exact_cal, exact_point, 1.0, 0.0) ==
          (BigFloat(exact_point), BigFloat(exact_point))
    @test conformal_coverage(exact_cal, [exact_point], [exact_point], [1.0], [0.0]) == 1.0
    ordinary = fit_conformal([-50.0], [-45.0], [1.0], [0.0])
    @test conformal_interval(ordinary, -50.0, 1.0, 0.0) === (-55.0, -45.0)
    # Precision must track the center/width exponent gap, beyond the default
    # 256-bit BigFloat context, rather than still collapsing a subnormal band.
    tiny = nextfloat(0.0)
    tiny_cal = fit_conformal([0.0], [tiny], [1.0], [0.0])
    lower, upper = conformal_interval(tiny_cal, 1.0, 1.0, 0.0)
    @test lower < 1 < upper
    @test upper - lower == 2 * BigFloat(tiny)
    huge_cal = fit_conformal([0.0], [1e308], [1.0], [0.0])
    lower, upper = conformal_interval(huge_cal, 1e308, 1.0, 0.0)
    @test isfinite(lower) && isfinite(upper)
    @test upper == 2 * BigFloat(1e308)
    # A zero-width huge integer still needs all of the center's low bits. The
    # independent integer oracle must not first narrow to a default BigFloat.
    integer_point = big(10)^300 + 1
    zero_cal = fit_conformal([integer_point], [integer_point], [1.0], [0.0])
    lower, upper = conformal_interval(zero_cal, integer_point, 1.0, 0.0)
    @test lower == integer_point && upper == integer_point
    @test conformal_coverage(zero_cal, [integer_point], [integer_point], [1.0], [0.0]) == 1.0
    @test conformal_coverage(zero_cal, [integer_point], [integer_point + 1], [1.0], [0.0]) == 0.0
    unit_cal = fit_conformal([integer_point], [integer_point + 1], [1.0], [0.0])
    lower, upper = conformal_interval(unit_cal, integer_point, 1.0, 0.0)
    @test lower == integer_point - 1 && upper == integer_point + 1
    @test conformal_coverage(unit_cal, [integer_point], [integer_point + 1], [1.0], [0.0]) == 1.0
    @test conformal_coverage(unit_cal, [integer_point], [integer_point + 2], [1.0], [0.0]) == 0.0
    # A non-dyadic rational has no exact finite BigFloat representation; retain
    # exact rational bounds rather than making an unsupported exactness claim.
    rational_point = integer_point // big(3)
    lower, upper = conformal_interval(zero_cal, rational_point, 1.0, 0.0)
    @test lower == rational_point && upper == rational_point
    @test conformal_coverage(zero_cal, [rational_point], [rational_point], [1.0], [0.0]) == 1.0
    lower, upper = conformal_interval(unit_cal, rational_point, 1.0, 0.0)
    @test lower == rational_point - 1 && upper == rational_point + 1
    @test conformal_coverage(unit_cal, [rational_point], [rational_point + 1], [1.0], [0.0]) == 1.0
    @test conformal_coverage(unit_cal, [rational_point], [rational_point + 2], [1.0], [0.0]) == 0.0
    @test conformal_interval(ordinary, 1 // 2, 1.0, 0.0) === (-4.5, 5.5)
    # The precision budget must span both operands, including the carry when
    # their sum crosses an exponent and a nonzero center below Float64's range.
    carry_point = setprecision(BigFloat, 256) do
        BigFloat(big(2)^255 + 1)
    end
    carry_cal = fit_conformal([0.0], [2.0^255], [1.0], [0.0])
    lower, upper = conformal_interval(carry_cal, carry_point, 1.0, 0.0)
    @test lower == 1 && upper == big(2)^256 + 1
    tiny_point = setprecision(BigFloat, 256) do
        BigFloat(2)^(-1200)
    end
    lower, upper = conformal_interval(unit_cal, tiny_point, 1.0, 0.0)
    expected_lower, expected_upper = setprecision(BigFloat, 2048) do
        (BigFloat(2)^(-1200) - 1, BigFloat(2)^(-1200) + 1)
    end
    @test lower != -1 && upper != 1
    @test lower == expected_lower
    @test upper == expected_upper
    @test conformal_coverage(unit_cal, [tiny_point], [tiny_point], [1.0], [0.0]) == 1.0
end
