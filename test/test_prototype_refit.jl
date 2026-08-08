using CSV
using DataFrames
using Dates
using Statistics

@testset "Operational V2.1 artifact boundary" begin
    data_dir = get_data_dir()

    @testset "current and historical identities are explicit" begin
        @test canonical_operational_version() == "v2.1"
        @test canonical_operational_version(:v2) == "v2.1"
        @test canonical_operational_version("v2_1") == "v2.1"
        @test canonical_operational_version(:v2_0) == "v2.0"
        @test_throws ArgumentError canonical_operational_version(:legacy)
        @test_throws ArgumentError canonical_operational_version("v3")

        current = operational_core_artifacts()
        alias = operational_core_artifacts(:v2)
        historical = operational_core_artifacts(:v2_0)
        @test current == alias
        @test current.version == "v2.1"
        @test current.candidate_count == 20
        @test current.active_count == 11
        @test dirname(current.coefficients_csv) == normpath(data_dir)
        @test historical.version == "v2.0"
        @test historical.candidate_count == 21
        @test historical.active_count == 10
        @test dirname(historical.coefficients_csv) ==
              normpath(joinpath(data_dir, "historical", "v2_0"))

        @test !isfile(joinpath(data_dir, "real_sindy_discovery_coefficients_refit.csv"))
        @test !isfile(joinpath(data_dir, "real_ensemble_inclusion_refit.csv"))
        @test !isfile(joinpath(data_dir, "real_sindy_ensemble_draws_refit.csv"))
    end

    @testset "validated V2.1 is exactly 20 candidates and 11 active terms" begin
        core = load_operational_core()
        terms = get_term_names(core.library)
        @test core.artifacts.version == "v2.1"
        @test length(terms) == 20
        @test !("n*V^2" in terms)
        @test count(!=(0.0), core.coefficients) == 11
        for term in ("Pdyn", "Pdyn*Bs", "Bs", "n*V", "n*Bs", "n*V*Bs", "Dst_star")
            idx = findfirst(==(term), terms)
            @test idx !== nothing
            @test core.coefficients[idx] != 0.0
        end

        state = init_operational_forecast(
            t0=DateTime(2026, 1, 1), dst0=-20.0,
        )
        @test get_term_names(state.lib) == terms
        @test state.ξ_primary == core.coefficients
        @test size(state.ξ_ensemble) == (500, 20)
        for term in ("Pdyn", "Pdyn*Bs")
            idx = findfirst(==(term), terms)
            @test std(state.ξ_ensemble[:, idx]) > 0.0
        end
        pdyn_idx = findfirst(==("Pdyn"), terms)
        @test std(state.ξ_ensemble[:, pdyn_idx]) > 0.05

        stability_path = joinpath(data_dir, "real_sindy_coefficients.csv")
        stability = CSV.read(stability_path, DataFrame)
        inclusion = CSV.read(core.artifacts.ensemble_csv, DataFrame)
        @test names(stability) ==
              ["term", "coefficient", "coefficient_kind", "inclusion"]
        @test string.(stability.term) == terms
        @test Float64.(stability.coefficient) == core.coefficients
        @test string.(stability.coefficient_kind) ==
              fill("selected_full_refit_point_coefficient", length(terms))
        @test Float64.(stability.inclusion) ==
              Float64.(inclusion.inclusion_probability)
        @test nrow(stability) == 20
        @test count(!=(0.0), Float64.(stability.coefficient)) == 11
        @test !("n*V^2" in string.(stability.term))
    end

    @testset "V2.0 is preserved only through its explicit historical request" begin
        old = load_operational_core(:v2_0)
        terms = get_term_names(old.library)
        @test length(terms) == 21
        @test "n*V^2" in terms
        @test count(!=(0.0), old.coefficients) == 10
        nvv = findfirst(==("n*V^2"), terms)
        @test old.coefficients[nvv] == -1.4748104360738998e-6

        current_cal = operational_calibration_artifacts()
        old_cal = operational_calibration_artifacts(:v2_0)
        @test dirname(current_cal.point_csv) ==
              normpath(joinpath(@__DIR__, "..", "deploy"))
        @test dirname(old_cal.point_csv) ==
              normpath(joinpath(@__DIR__, "..", "deploy", "historical", "v2_0"))
        @test isfile(old_cal.point_csv)
        @test isfile(old_cal.conformal_csv)
    end

    @testset "unqualified data snapshots exclude the retired redundant term" begin
        offenders = String[]
        for (root, directories, files) in walkdir(data_dir)
            filter!(directory -> directory != "historical", directories)
            for file in files
                endswith(file, ".csv") || continue
                path = joinpath(root, file)
                occursin("n*V^2", read(path, String)) &&
                    push!(offenders, relpath(path, data_dir))
            end
        end
        @test isempty(offenders)
    end

    @testset "fast calibration rollout equals primary ensemble path" begin
        core = load_operational_core()
        t0 = DateTime(2026, 1, 1)
        dst0 = -67.5
        drivers = (V=517.0, Bz=-9.25, By=3.5, n=7.2,
                   Pdyn=dynamic_pressure(7.2, 517.0))
        fast = operational_core_forecast(core, dst0, drivers, 6)
        state = init_operational_forecast(t0=t0, dst0=dst0)
        full = forecast_ahead(
            state, drivers.V, drivers.Bz, drivers.By, drivers.n, drivers.Pdyn, 6,
        )
        @test fast == [row.dst_predicted for row in full]
        @test operational_core_forecast(core, dst0, drivers, 0) == Float64[]
        @test_throws ArgumentError operational_core_forecast(core, dst0, drivers, -1)
    end

    @testset "artifact mutations fail closed" begin
        source = operational_core_artifacts()
        mktempdir() do tmp
            cp(source.ensemble_csv, joinpath(tmp, "real_ensemble_inclusion.csv"))
            cp(source.draws_csv, joinpath(tmp, "real_sindy_ensemble_draws.csv"))
            broken = CSV.read(source.coefficients_csv, DataFrame)
            broken.coefficient[5] = 0.0
            CSV.write(joinpath(tmp, "real_sindy_discovery_coefficients.csv"), broken)
            @test_throws ArgumentError load_operational_core(; data_dir=tmp)

            broken = CSV.read(source.coefficients_csv, DataFrame)
            broken.term[1] = "unknown"
            CSV.write(joinpath(tmp, "real_sindy_discovery_coefficients.csv"), broken)
            @test_throws ArgumentError load_operational_core(; data_dir=tmp)
        end
    end

    @testset "direct monitor uses canonical artifacts" begin
        probe = Module(:StormMonitorV21Probe)
        Base.include(probe, joinpath(@__DIR__, "..", "examples", "storm_monitor.jl"))
        @test probe.CORE_VERSION == "v2.1"
        @test probe.COEF_CSV == operational_core_artifacts().coefficients_csv
        @test probe.INCL_CSV == operational_core_artifacts().ensemble_csv
        @test probe.DRAWS_CSV == operational_core_artifacts().draws_csv
    end
end
