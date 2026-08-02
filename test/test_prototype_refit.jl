# Tests pinning the eleven-term refit prototype (examples/storm_monitor.jl) and
# guarding the frozen ten-term operational identity against accidental drift.
#
# Three guarantees:
#   (a) the bundled refit coefficient artifact carries exactly the eleven paper
#       coefficients on the 20-term identifiable library (no redundant n*V^2),
#       and is byte-identical to the canonical paper artifact when present;
#   (b) the prototype loader selects the *_refit artifacts and builds a 20-term,
#       500-member forecaster with Pdyn / Pdyn*Bs active;
#   (c) the frozen operational coefficient file still decodes to the ten-term
#       support (n*V^2 active, Pdyn and Pdyn*Bs absent) — the live V2 identity.

using CSV
using DataFrames
using Dates
using Statistics

@testset "Prototype eleven-term refit" begin

    data_dir = get_data_dir()
    refit_coef_path = joinpath(data_dir, "real_sindy_discovery_coefficients_refit.csv")
    refit_incl_path = joinpath(data_dir, "real_ensemble_inclusion_refit.csv")
    refit_draws_path = joinpath(data_dir, "real_sindy_ensemble_draws_refit.csv")

    @testset "(a) refit coefficient artifact pins the eleven paper terms" begin
        @test isfile(refit_coef_path)
        df = CSV.read(refit_coef_path, DataFrame)

        # The refit uses the 20-term identifiable library in canonical order and
        # must not contain the redundant exact pressure proxy n*V^2.
        canonical_terms = get_term_names(
            build_solar_wind_library(include_redundant_n_v2 = false))
        @test length(canonical_terms) == 20
        @test !("n*V^2" in canonical_terms)
        @test string.(df.term) == canonical_terms

        # Exact coefficients pinned BY POSITION (independent of Unicode term
        # spelling), full Float64 precision, from the locked paper refit.
        expected = [
            0.0,                       # 1
            0.0,                       # V
            -0.7311252640766392,       # Bs
            0.0,                       # n
            -0.896469523144789,        # Pdyn
            -0.05254736645358795,      # Dst_star
            0.0,                       # V^2
            0.0,                       # Bs^2
            0.0,                       # n^2
            0.0,                       # V*Bs
            0.0005443391551555351,     # n*V
            0.058657050490999035,      # n*Bs
            0.08762670100061405,       # Pdyn*Bs
            -0.00023440681152932838,   # n*V*Bs
            6.928450443175584,         # sin(θ_c/2)
            -38.7203669622234,         # sin²(θ_c/2)
            -9.306535662532706,        # sin⁴(θ_c/2)
            42.01021693483194,         # sin^(8/3)(θ_c/2)
            0.0,                       # V*sin²(θ_c/2)
            0.0,                       # Newell_d_Φ
        ]
        @test Float64.(df.coefficient) == expected
        @test count(!=(0.0), Float64.(df.coefficient)) == 11

        # Byte-identity with the canonical paper artifact when the full project
        # tree is checked out (paper/ lives outside the standalone package repo).
        paper_coef = joinpath(@__DIR__, "..", "..", "paper", "data",
                              "real_sindy_discovery_coefficients.csv")
        if isfile(paper_coef)
            @test read(paper_coef) == read(refit_coef_path)       # byte-identical copy
        end

        # The committed inclusion summary uses the canonical conditional-nonzero
        # interval columns the refit ensemble path expects.
        @test isfile(refit_incl_path)
        incl = CSV.read(refit_incl_path, DataFrame)
        @test all(in(names(incl)),
                  ("conditional_nonzero_empirical_q025",
                   "conditional_nonzero_empirical_q975"))
    end

    @testset "(b) prototype loader selects the _refit artifacts" begin
        # Include the actual prototype in an isolated module. Its main() is
        # PROGRAM_FILE-guarded, so no live loop launches; we only read its paths.
        probe = Module(:StormMonitorRefitProbe)
        Base.include(probe, joinpath(@__DIR__, "..", "examples", "storm_monitor.jl"))

        @test endswith(probe.COEF_CSV, "real_sindy_discovery_coefficients_refit.csv")
        @test endswith(probe.INCL_CSV, "real_ensemble_inclusion_refit.csv")
        @test endswith(probe.DRAWS_CSV, "real_sindy_ensemble_draws_refit.csv")
        @test dirname(probe.COEF_CSV) == data_dir

        # Building the forecaster through the prototype's paths yields the
        # eleven-term / 500-member state. Passing draws_csv explicitly is what
        # makes the ensemble the refit posterior rather than the frozen draws.
        state = init_forecast(; coefficients_csv = probe.COEF_CSV,
                              ensemble_csv = probe.INCL_CSV,
                              draws_csv = probe.DRAWS_CSV,
                              t0 = DateTime(2026, 1, 1, 0), dst0 = -20.0)
        terms = get_term_names(state.lib)
        @test length(terms) == 20
        @test !("n*V^2" in terms)
        @test count(!=(0.0), state.ξ_primary) == 11
        @test size(state.ξ_ensemble, 1) == 500

        # Every regressor the refit equation needs is present and active; the two
        # pressure regressors absent from the frozen fit must both be live.
        for term in ("Pdyn", "Pdyn*Bs", "Bs", "n*V", "n*Bs", "n*V*Bs", "Dst_star")
            idx = findfirst(==(term), terms)
            @test idx !== nothing
            @test state.ξ_primary[idx] != 0.0
        end

        # When the joint refit draws are available they are actually resampled:
        # the Pdyn / Pdyn*Bs ensemble columns carry spread (not fixed-at-point).
        if isfile(refit_draws_path)
            for term in ("Pdyn", "Pdyn*Bs")
                idx = findfirst(==(term), terms)
                @test std(state.ξ_ensemble[:, idx]) > 0
            end

            # Magnitude gate separating the refit ensemble from a silent fallback
            # to the frozen draws. Both draws files carry a Pdyn column, so plain
            # std>0 cannot tell them apart; the discriminator is the spread.
            # Measured Pdyn ensemble std: refit draws ~0.112, frozen draws ~0.020.
            # A 0.05 threshold sits ~2x below the refit spread and ~2.5x above the
            # frozen spread, separating them with margin. (Pdyn*Bs cannot be used:
            # its refit std ~0.023 is below the frozen ~0.030, so only Pdyn
            # discriminates the refit posterior from the frozen fallback.)
            pdyn_idx = findfirst(==("Pdyn"), terms)
            @test std(state.ξ_ensemble[:, pdyn_idx]) > 0.05
        end
    end

    @testset "(c) frozen operational coefficients keep the ten-term identity" begin
        frozen_coef = joinpath(data_dir, "real_sindy_discovery_coefficients.csv")
        frozen_incl = joinpath(data_dir, "real_ensemble_inclusion.csv")
        @test isfile(frozen_coef)

        state = init_forecast(; coefficients_csv = frozen_coef,
                              ensemble_csv = frozen_incl,
                              t0 = DateTime(2026, 1, 1, 0), dst0 = -20.0)
        terms = get_term_names(state.lib)

        # Frozen fit lives on the 21-term library including the redundant n*V^2.
        @test length(terms) == 21
        @test "n*V^2" in terms
        @test count(!=(0.0), state.ξ_primary) == 10

        nvv_idx = findfirst(==("n*V^2"), terms)
        pdyn_idx = findfirst(==("Pdyn"), terms)
        pdyn_bs_idx = findfirst(==("Pdyn*Bs"), terms)
        bs_idx = findfirst(==("Bs"), terms)

        # n*V^2 is active in the frozen fit; the refit's pressure regressors are NOT.
        @test state.ξ_primary[nvv_idx] != 0.0
        @test state.ξ_primary[pdyn_idx] == 0.0
        @test state.ξ_primary[pdyn_bs_idx] == 0.0

        # Pin the frozen values that define the live V2 identity.
        @test state.ξ_primary[nvv_idx] == -1.4748104360738998e-6
        @test state.ξ_primary[bs_idx] == -0.6929180631210645
    end
end
