using Test
using LinearAlgebra
using SolarSINDy

function _v22cp_test_core()
    library = build_solar_wind_library()
    coefficients = zeros(length(library))
    active = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13)
    coefficients[collect(active)] .= collect(1.0e-5:1.0e-5:11.0e-5)
    artifacts = OperationalCoreArtifacts(
        OPERATIONAL_V2_1_MODEL_VERSION,
        "synthetic-coefficients.csv",
        "synthetic-ensemble.csv",
        "synthetic-draws.csv",
        20,
        11,
    )
    return OperationalCore(artifacts, library, coefficients)
end

function _v22cp_path()
    path = Matrix{Float64}(undef, 14, 5)
    for row in 1:14
        path[row, 1] = 0.5 * row
        path[row, 2] = -0.25 * row
        path[row, 3] = -2.0 - 0.1 * row
        path[row, 4] = log(300.0 + 10.0 * row)
        path[row, 5] = log(4.0 + 0.2 * row)
    end
    return path
end

@testset verbose=true "Operational V2.2 M2-to-core path" begin
    @testset "hand-derived hourly aggregation and pressure identity" begin
        path = _v22cp_path()
        drivers = operational_v22_hourly_drivers(path)
        @test length(drivers) == 7
        first_driver = first(drivers)
        @test first_driver.Bx == (path[1, 1] + path[2, 1]) / 2
        @test first_driver.By == (path[1, 2] + path[2, 2]) / 2
        @test first_driver.Bz == (path[1, 3] + path[2, 3]) / 2
        @test first_driver.V == (exp(path[1, 4]) + exp(path[2, 4])) / 2
        @test first_driver.n == (exp(path[1, 5]) + exp(path[2, 5])) / 2
        @test first_driver.Pdyn == dynamic_pressure(first_driver.n, first_driver.V)
        @test first_driver.Pdyn !=
              (dynamic_pressure(exp(path[1, 5]), exp(path[1, 4])) +
               dynamic_pressure(exp(path[2, 5]), exp(path[2, 4]))) / 2
    end

    @testset "sequential core oracle and prefix causality" begin
        core = _v22cp_test_core()
        path = _v22cp_path()
        anchor = -25.0
        result = operational_v22_core_path_forecast(core, anchor, path)
        @test result.schema_version == OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION
        @test result.execution_scope == :low_level_research_only
        @test result.internal_step_hours == (1, 2, 3, 4, 5, 6, 7)
        @test result.supported_model_steps == (1, 2, 3, 4, 6, 7)
        state = anchor
        expected_star = Float64[]
        expected_dst = Float64[]
        for driver in result.hourly_drivers
            state = operational_core_forecast(core, state, driver, 1)[1]
            push!(expected_star, state)
            push!(expected_dst, dst_star_to_dst(state, driver.Pdyn))
        end
        @test collect(result.pred_dst_star_nt) == expected_star
        @test collect(result.pred_dst_nt) == expected_dst

        future_mutation = copy(path)
        future_mutation[3:end, :] .+= 0.2
        mutated = operational_v22_core_path_forecast(
            core, anchor, future_mutation,
        )
        @test mutated.pred_dst_nt[1] == result.pred_dst_nt[1]
        @test mutated.pred_dst_nt[2:end] != result.pred_dst_nt[2:end]

        bx_only = copy(path)
        bx_only[:, 1] .+= 100.0
        bx_result = operational_v22_core_path_forecast(core, anchor, bx_only)
        @test bx_result.pred_dst_star_nt == result.pred_dst_star_nt
        @test bx_result.pred_dst_nt == result.pred_dst_nt
        @test bx_result.hourly_drivers[1].Bx != result.hourly_drivers[1].Bx
    end

    @testset "shape, domain, and frozen-core guards" begin
        core = _v22cp_test_core()
        path = _v22cp_path()
        @test_throws DimensionMismatch operational_v22_hourly_drivers(path[1:13, :])
        @test_throws DimensionMismatch operational_v22_hourly_drivers(path[:, 1:4])
        bad = copy(path)
        bad[1, 1] = NaN
        @test_throws ArgumentError operational_v22_hourly_drivers(bad)
        overflow = copy(path)
        overflow[1, 4] = 1.0e4
        @test_throws ArgumentError operational_v22_hourly_drivers(overflow)
        @test_throws ArgumentError operational_v22_core_path_forecast(
            core, NaN, path,
        )

        wrong_version = OperationalCore(
            OperationalCoreArtifacts(
                OPERATIONAL_V2_0_MODEL_VERSION,
                "a", "b", "c", 20, 11,
            ),
            core.library,
            copy(core.coefficients),
        )
        @test_throws ArgumentError operational_v22_core_path_forecast(
            wrong_version, -25.0, path,
        )
        wrong_support = copy(core.coefficients)
        wrong_support[13] = 0.0
        invalid_core = OperationalCore(
            core.artifacts, core.library, wrong_support,
        )
        @test_throws ArgumentError operational_v22_core_path_forecast(
            invalid_core, -25.0, path,
        )
    end
end
