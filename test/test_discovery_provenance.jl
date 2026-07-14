# Provenance / behavior-freeze test for the deployed discovered-ODE coefficients.
#
# Finding real_data_discovery.jl:271 — the served coefficients in
# data/real_sindy_discovery_coefficients.csv are a version-pinned historical
# best-λ STLSQ fit; no generator wrote them into version control. This test freezes
# the shipped active set and values as a regression baseline so that a future
# rerun of discovery (whose λ-selection knee is data-dependent) which shifts the
# served model trips loudly here, instead of silently diverging the operational
# equation from its archived validation evidence. real_data_discovery.jl now also
# regenerates this file + a provenance sidecar; until regeneration parity is
# demonstrated on the OMNI archive, the shipped values remain the pinned baseline.

@testset "Deployed discovery coefficients: frozen provenance baseline" begin
    d = get_data_dir()
    df = CSV.read(joinpath(d, "real_sindy_discovery_coefficients.csv"), DataFrame)

    # Column/schema contract expected by init_forecast and generate_real_figures.
    @test names(df) == ["term", "coefficient"]

    # Term column matches the standard library order exactly, so the file loads
    # index-aligned. (The library builder is the single source of the term strings,
    # so this also pins the exact unicode names without hardcoding them here.)
    lib_terms = get_term_names(build_solar_wind_library(
        include_redundant_n_v2=true,
    ))
    @test String.(df.term) == lib_terms
    @test length(df.coefficient) == length(lib_terms) == 21

    # Frozen best-λ fit: 10 active terms at their shipped values (index-aligned to
    # the library order asserted above). Zeros are pinned exactly; nonzeros to
    # full double precision.
    expected = zeros(21)
    expected[3]  = -0.6929180631210645     # Bs
    expected[6]  = -0.04790123786283618    # Dst_star
    expected[11] =  0.0005196595337899235  # n*V
    expected[12] =  0.016631273895335846   # n*Bs
    expected[14] = -6.920963980013702e-5   # n*V*Bs
    expected[15] = -1.4748104360738998e-6  # n*V^2
    expected[16] =  8.326812122307805      # sin(θ_c/2)
    expected[17] = -46.42987414561538      # sin²(θ_c/2)
    expected[18] = -12.115751533668123     # sin⁴(θ_c/2)
    expected[19] =  51.20910175620663      # sin^(8/3)(θ_c/2)

    coef = Float64.(df.coefficient)
    @test count(!=(0.0), coef) == 10       # exactly ten active terms
    for i in 1:21
        if expected[i] == 0.0
            @test coef[i] == 0.0
        else
            @test coef[i] ≈ expected[i] rtol = 1e-12
        end
    end
end
