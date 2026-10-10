using CSV
using DataFrames
using Dates
using SolarSINDy
using Test

include(joinpath(
    @__DIR__, "..", "validation", "operational",
    "v2_2_m2_core_omni_diagnostic.jl",
))

@testset "V2.2-M2 frozen-core OMNI diagnostic" begin
    @testset "hourly Dst midpoint contract" begin
        mktemp() do path, io
            for (time, value) in (
                    ("2010-01-01T00:30:00.000Z", 5),
                    ("2010-01-01T01:30:00.000Z", 4),
                    ("2010-01-01T02:30:00.000Z", 99_999),
                    ("2010-01-01T03:30:00.000Z", -2))
                println(io, time, ',', value)
            end
            close(io)
            table = v2_2_m2_core_read_dst(path)
            @test table.issue_time_utc == [
                DateTime(2010, 1, 1, 0), DateTime(2010, 1, 1, 1),
                DateTime(2010, 1, 1, 3),
            ]
            @test table.dst_nt == [5.0, 4.0, -2.0]
        end
        mktemp() do path, io
            println(io, "2010-01-01T00:00:00.000Z,5")
            println(io, "2010-01-01T01:00:00.000Z,4")
            close(io)
            @test_throws ArgumentError v2_2_m2_core_read_dst(path)
        end
        @test occursin("id=OMNI2_H0_MRG1HR", _v22m2_core_dst_url())
        @test occursin("parameters=Time,DST1800", _v22m2_core_dst_url())
    end

    @testset "anchor pressure and baseline integration" begin
        history = zeros(25, 5)
        history[:, 4] .= log(400.0)
        history[:, 5] .= log(5.0)
        @test _v22m2_core_anchor_pressure(history) ≈
              dynamic_pressure(5.0, 400.0) rtol=1e-14
        drivers = ntuple(_ -> (
            Bx=0.0, By=0.0, Bz=-2.0, V=400.0, n=5.0,
            Pdyn=dynamic_pressure(5.0, 400.0),
        ), 7)
        paths = _v22m2_core_physics_paths(-20.0, drivers)
        @test length(paths.burton) == 7
        @test length(paths.burton_full) == 7
        @test length(paths.obrien) == 7
        @test all(isfinite, paths.burton)
        @test all(isfinite, paths.burton_full)
        @test all(isfinite, paths.obrien)
        @test_throws DimensionMismatch _v22m2_core_anchor_pressure(zeros(24, 5))
    end

    @testset "paired weekly bootstrap and frozen gate" begin
        methods = sort!(collect(union(
            Set(V22_M2_CORE_CANDIDATES),
            Set(Iterators.flatten(values(V22_M2_CORE_COMPARATORS))),
        )))
        error_level = Dict(method => 4.0 for method in methods)
        error_level["sindy_dense_driver"] = 1.0
        error_level["sindy_sparse_driver"] = 0.25
        rows = NamedTuple[]
        issues = [V22_M2_OMNI_GRID_START + Day(7 * i) for i in 0:3]
        for method in methods, lead in V22_M2_OMNI_LEADS_H, issue in issues
            push!(rows, (
                issue_time_utc=issue,
                method=method,
                lead_hours=lead,
                squared_error_nt2=error_level[method],
            ))
        end
        errors = DataFrame(rows)
        bootstrap = v2_2_m2_core_bootstrap(errors; draws=100, seed=4405)
        @test nrow(bootstrap) == 66
        @test all(bootstrap.point_improvement_nt .>= 0.5)
        @test all(bootstrap.one_sided_95_lower_nt .> 0.0)

        metric_rows = NamedTuple[]
        for method in methods, lead in V22_M2_OMNI_LEADS_H,
                regime in ("overall", "target_storm", "active_or_deepening",
                           "recovery")
            push!(metric_rows, (
                method=method,
                lead_hours=lead,
                regime=regime,
                rows=regime == "overall" ? 4 : 0,
                rmse_nt=sqrt(error_level[method]),
                mae_nt=sqrt(error_level[method]),
            ))
        end
        supported_driver = DataFrame(
            diagnostic_outcome=["PREDICTABILITY_SUPPORTED_WITHOUT_PREFIX"],
        )
        by_lead, decision = v2_2_m2_core_decision(
            DataFrame(metric_rows), bootstrap, supported_driver,
        )
        @test all(by_lead.end_to_end_gate_pass)
        @test decision.sparse_core_diagnostic_supported[1]
        @test decision.dense_core_exploratory_supported[1]
        @test !decision.promotion_eligible[1]

        unsupported_driver = DataFrame(
            diagnostic_outcome=["PREDICTABILITY_NOT_SUPPORTED"],
        )
        by_lead, decision = v2_2_m2_core_decision(
            DataFrame(metric_rows), bootstrap, unsupported_driver,
        )
        @test !any(by_lead.end_to_end_gate_pass[
            by_lead.candidate .== "sindy_sparse_driver",
        ])
        @test decision.dense_core_exploratory_supported[1]

        mismatched = copy(errors)
        row = findfirst(mismatched.method .== "dst_persistence")
        mismatched.issue_time_utc[row] += Hour(1)
        @test_throws ArgumentError v2_2_m2_core_bootstrap(
            mismatched; draws=100, seed=4405,
        )
    end
end
