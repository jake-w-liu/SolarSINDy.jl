# Provenance and numerical freeze for the current V2.1 discovery core.

@testset "Current discovery coefficients: revised 20/11 provenance" begin
    current = operational_core_artifacts()
    df = CSV.read(current.coefficients_csv, DataFrame)

    @test names(df) == ["term", "coefficient"]
    terms = get_term_names(build_solar_wind_library())
    @test string.(df.term) == terms
    @test length(terms) == 20
    @test !("n*V^2" in terms)

    expected = [
        0.0,
        0.0,
        -0.7311252640766392,
        0.0,
        -0.896469523144789,
        -0.05254736645358795,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0005443391551555351,
        0.058657050490999035,
        0.08762670100061405,
        -0.00023440681152932838,
        6.928450443175584,
        -38.7203669622234,
        -9.306535662532706,
        42.01021693483194,
        0.0,
        0.0,
    ]
    @test Float64.(df.coefficient) == expected
    @test count(!=(0.0), expected) == 11

    paper_coef = joinpath(@__DIR__, "..", "..", "paper", "data",
                          "real_sindy_discovery_coefficients.csv")
    isfile(paper_coef) && @test read(paper_coef) == read(current.coefficients_csv)
end
