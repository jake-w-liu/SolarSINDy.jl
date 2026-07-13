using Test
using SolarSINDy
using DataFrames
using Dates
using CSV
using Statistics

@testset "Operational V2 memory features are time-based" begin
    # Replay-shaped table: one row per (anchor, lead) with a tied issue_time_utc per anchor.
    # latest_dst_nt = -10·(hour), so the true k-hour Dst delta is exactly -10·k.
    function memtable(hours; leads = [1, 2, 3, 6])
        t0 = DateTime(2024, 1, 1, 0)
        rows = NamedTuple[]
        for h in hours, L in leads
            t = t0 + Hour(h)
            push!(rows, (issue_time_utc = string(t),
                         latest_dst_time_utc = string(t),
                         model_step_hours = L,
                         latest_dst_nt = -10.0 * h,
                         V_kms = 400.0, Bz_nt = -5.0, By_nt = 1.0, n_cm3 = 5.0, Pdyn_npa = 2.0))
        end
        return DataFrame(rows)
    end
    anchor(df, h) = df[df.issue_time_utc .== string(DateTime(2024, 1, 1) + Hour(h)), :]

    @testset "multi-horizon (tied issue times) resolve to true hourly deltas" begin
        df = memtable(0:6)
        SolarSINDy.add_operational_v2_features!(df)
        for h in 3:6   # anchors whose t-1h and t-3h history are both present
            sub = anchor(df, h)
            @test nrow(sub) == 4                        # 4 horizon rows share one anchor
            @test all(sub.dst_delta_1h_nt .== -10.0)    # identical across horizons, = true 1 h delta
            @test all(sub.dst_delta_3h_nt .== -30.0)    # true 3 h delta (row-offset would give 0 or 1 h)
        end
        @test all(anchor(df, 6).dst_delta_6h_nt .== -60.0)
        # No history ⇒ neutral (0), never a spurious value.
        @test all(anchor(df, 0).dst_delta_1h_nt .== 0.0)
        @test all(anchor(df, 0).dst_delta_3h_nt .== 0.0)
    end

    @testset "gap-skipped anchors: deltas span time, not rows" begin
        df = memtable([0, 1, 2, 4, 5])   # hour 3 is skipped (driver/Dst gap)
        SolarSINDy.add_operational_v2_features!(df)
        s5 = anchor(df, 5)               # t-1h=4 present, t-3h=2 present
        @test all(s5.dst_delta_1h_nt .== -10.0)          # L(5)-L(4)
        @test all(s5.dst_delta_3h_nt .== -30.0)          # L(5)-L(2), spanning the TRUE 3 hours
        s4 = anchor(df, 4)               # t-1h=3 is absent (gap) ⇒ neutral memory
        @test all(s4.dst_delta_1h_nt .== 0.0)
        @test all(s4.dst_delta_3h_nt .== 0.0)
    end

    @testset "no time axis ⇒ neutral single-row features (serving frame)" begin
        df = DataFrame(latest_dst_nt = [-100.0], V_kms = [400.0], Bz_nt = [-5.0],
                       By_nt = [1.0], n_cm3 = [5.0], Pdyn_npa = [2.0])
        SolarSINDy.add_operational_v2_features!(df)
        @test df.dst_delta_1h_nt[1] == 0.0
        @test df.dst_delta_3h_nt[1] == 0.0
    end
end

@testset "init_forecast ensemble: joint draws vs marginal fallback" begin
    lib = build_solar_wind_library()
    tn = get_term_names(lib)
    simple = filter(t -> occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", t), tn)  # avoid special-char CSV headers
    a1, a2 = simple[1], simple[2]
    i1 = findfirst(==(a1), tn); i2 = findfirst(==(a2), tn)

    mktempdir() do dir
        coef = joinpath(dir, "coef.csv")
        CSV.write(coef, DataFrame(term = [a1, a2], coefficient = [-0.5, -0.05]))
        ens = joinpath(dir, "ens.csv")
        CSV.write(ens, DataFrame(term = [a1, a2], inclusion_prob = [1.0, 1.0],
                                 median_coef = [-0.5, -0.05],
                                 ci_025 = [-1.0, -0.06], ci_975 = [-0.8, -0.04]))
        a1_draws = [-0.85, -0.95, -0.90, -0.80, -1.00]     # mean -0.90
        a2_draws = [-0.040, -0.060, -0.050, -0.055, -0.045] # mean -0.050
        draws = joinpath(dir, "draws.csv")
        CSV.write(draws, DataFrame(Symbol(a1) => a1_draws, Symbol(a2) => a2_draws))

        # --- Joint-draws path: resample rows, recenter on the deployed coefficients ---
        st = init_forecast(coefficients_csv = coef, ensemble_csv = ens, draws_csv = draws,
                           t0 = DateTime(2024, 1, 1), dst0 = -20.0)
        μ1 = mean(a1_draws); μ2 = mean(a2_draws)
        set1 = Set(round.(st.ξ_primary[i1] .+ (a1_draws .- μ1); digits = 12))
        set2 = Set(round.(st.ξ_primary[i2] .+ (a2_draws .- μ2); digits = 12))
        @test all(round(v; digits = 12) in set1 for v in st.ξ_ensemble[:, i1])
        @test all(round(v; digits = 12) in set2 for v in st.ξ_ensemble[:, i2])
        @test length(unique(st.ξ_ensemble[:, i1])) > 1                 # genuine spread
        @test isapprox(mean(st.ξ_ensemble[:, i1]), st.ξ_primary[i1]; atol = 0.1)  # centered on deployed
        for j in eachindex(tn)                                          # inactive terms carry no spread
            (j == i1 || j == i2) && continue
            @test all(st.ξ_ensemble[:, j] .== 0.0)
        end

        # --- Marginal fallback path (no draws artifact) ---
        stm = init_forecast(coefficients_csv = coef, ensemble_csv = ens,
                            draws_csv = joinpath(dir, "no_such_file.csv"),
                            t0 = DateTime(2024, 1, 1), dst0 = -20.0)
        @test std(stm.ξ_ensemble[:, i1]) > 0.0                          # active term gets marginal spread
        for j in eachindex(tn)
            (j == i1 || j == i2) && continue
            @test all(stm.ξ_ensemble[:, j] .== 0.0)
        end
        @test stm.ξ_ensemble[:, i1] != st.ξ_ensemble[:, i1]             # the two mechanisms differ
    end
end
