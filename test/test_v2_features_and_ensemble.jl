using Test
using SolarSINDy
using DataFrames
using Dates
using CSV
using Statistics

function _v2_score_fixture(n::Int; baseline::Bool=false)
    feature = collect(range(-2.0, 2.0; length=n))
    center = fill(-20.0, n)
    df = DataFrame(
        pred_dst_nt=center,
        pred_dst_ci05_nt=center .- 5.0,
        pred_dst_ci95_nt=center .+ 5.0,
        observation_dst_nt=center .+ feature,
        Bz_nt=feature,
    )
    if baseline
        df[!, :obrien_dst_nt] = center .- 1.0
        cal = OperationalV2Calibration(
            [:Bz_nt], [0.0], [1.0], [0.0, 1.0], 1.0, "allocation";
            selector_names=[:v2, :obrien], selector_rmse=zeros(2),
            selector_mae=zeros(2), selector_half_width=zeros(2),
            selected_component=:obrien,
        )
        return df, cal
    end
    cal = OperationalV2Calibration(
        [:Bz_nt], [0.0], [1.0], [0.0, 1.0], 1.0, "allocation",
    )
    return df, cal
end

function _allocated_v2_score(n::Int; baseline::Bool=false)
    df, cal = _v2_score_fixture(n; baseline=baseline)
    score_operational_v2(df, cal)
    GC.gc()
    return @allocated score_operational_v2(df, cal)
end

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

@testset "Operational V2 rejects duplicate features" begin
    @test_throws ArgumentError OperationalV2Calibration(
        [:Bz_nt, :Bz_nt], zeros(2), ones(2), zeros(3), 1.0, "duplicate",
    )
    extreme_weights = OperationalV2Calibration(
        Symbol[], Float64[], Float64[], [0.0], 1.0, "extreme weights";
        selector_names=[:a, :b], selector_rmse=[1.0, 1.0],
        selector_mae=[1.0, 1.0], selector_half_width=[0.0, 0.0],
        selector_weights=fill(floatmax(Float64), 2),
        selected_component=:ensemble,
    )
    @test extreme_weights.selector_weights == [0.5, 0.5]
    @test sum(extreme_weights.selector_weights) == 1.0

    intercept_data = DataFrame(
        pred_dst_nt=[0.0, 0.0, 0.0],
        pred_dst_ci05_nt=[-2.0, -2.0, -2.0],
        pred_dst_ci95_nt=[2.0, 2.0, 2.0],
        observation_dst_nt=[-1.0, 0.0, 1.0],
    )
    intercept_only = fit_operational_v2_calibration(
        intercept_data; feature_names=Symbol[], ridge=0.0,
    )
    @test isempty(intercept_only.feature_names)
    @test length(intercept_only.coefficients) == 1

    extreme_data = DataFrame(
        pred_dst_nt=zeros(3),
        pred_dst_ci05_nt=fill(-1.0e200, 3),
        pred_dst_ci95_nt=fill(1.0e200, 3),
        observation_dst_nt=[1.0e200, -1.0e200, 0.0],
        latest_dst_nt=zeros(3),
    )
    extreme_calibration = fit_operational_v2_calibration(
        extreme_data; feature_names=[:latest_dst_nt], ridge=1.0,
    )
    @test extreme_calibration.selector_rmse[1] ≈ sqrt(2 / 3) * 1.0e200
    @test all(isfinite, extreme_calibration.selector_rmse)

    df = DataFrame(
        pred_dst_nt=collect(-20.0:-1.0:-27.0),
        pred_dst_ci05_nt=collect(-25.0:-1.0:-32.0),
        pred_dst_ci95_nt=collect(-15.0:-1.0:-22.0),
        observation_dst_nt=collect(-21.0:-1.0:-28.0),
        Bz_nt=collect(-4.0:1.0:3.0),
    )
    @test_throws ArgumentError fit_operational_v2_calibration(
        df; feature_names=[:Bz_nt, :Bz_nt], ridge=0.0,
    )

    mktempdir() do dir
        path = joinpath(dir, "duplicate_features.csv")
        cal = OperationalV2Calibration(
            [:Bz_nt, :By_nt], zeros(2), ones(2), zeros(3), 1.0, "duplicate_read",
        )
        write_operational_v2_calibration(path, cal)
        artifact = CSV.read(path, DataFrame)
        artifact.feature[3] = artifact.feature[2]
        CSV.write(path, artifact)
        @test_throws ArgumentError read_operational_v2_calibration(path)
    end
end

@testset "Operational V2 ensemble evaluates only positive weights" begin
    features = (; Bz_nt=0.0)
    inactive_baselines = OperationalV2Calibration(
        [:Bz_nt], [0.0], [1.0], [0.0, 0.0], 1.0, "inactive_baselines";
        selector_names=[:v2, :sindy_v1, :obrien, :burton],
        selector_rmse=zeros(4), selector_mae=zeros(4),
        selector_half_width=zeros(4),
        selector_weights=[0.25, 0.75, 0.0, 0.0],
        selected_component=:ensemble,
    )
    expected = 0.25 * -20.0 + 0.75 * -20.0
    no_baselines = operational_v2_predict(
        inactive_baselines, -20.0, -25.0, -15.0, features,
    )
    nonfinite_unused = operational_v2_predict(
        inactive_baselines, -20.0, -25.0, -15.0, features;
        baselines=(; obrien=NaN, burton=Inf),
    )
    @test no_baselines.pred_dst == expected
    @test nonfinite_unused.pred_dst == expected

    active_baseline = OperationalV2Calibration(
        [:Bz_nt], [0.0], [1.0], [0.0, 0.0], 1.0, "active_baseline";
        selector_names=[:v2, :obrien], selector_rmse=zeros(2),
        selector_mae=zeros(2), selector_half_width=zeros(2),
        selector_weights=[0.5, 0.5], selected_component=:ensemble,
    )
    @test_throws ArgumentError operational_v2_predict(
        active_baseline, -20.0, -25.0, -15.0, features,
    )
    @test_throws ArgumentError operational_v2_predict(
        active_baseline, -20.0, -25.0, -15.0, features;
        baselines=(; obrien=NaN),
    )
    @test_throws ArgumentError operational_v2_predict(
        active_baseline, -20.0, -25.0, -15.0, features;
        baselines=(; obrien=missing),
    )
    active = operational_v2_predict(
        active_baseline, -20.0, -25.0, -15.0, features;
        baselines=(; obrien=-40.0),
    )
    @test active.pred_dst == -30.0
end

@testset "Operational V2 fails closed on non-finite arithmetic" begin
    cal = OperationalV2Calibration(
        [:Bz_nt], [0.0], [1.0], [0.0, 1.0], 1.0, "finite_arithmetic",
    )
    features = (; Bz_nt=0.0)
    @test_throws ArgumentError operational_v2_predict(
        cal, NaN, -25.0, -15.0, features,
    )
    @test_throws ArgumentError operational_v2_predict(
        cal, -20.0, -Inf, -15.0, features,
    )
    @test_throws ArgumentError operational_v2_predict(
        cal, -20.0, -25.0, -15.0, (; Bz_nt=Inf),
    )

    # The raw endpoint subtraction overflows, but halving each endpoint first
    # yields the correct finite half-width and finite conservative endpoints.
    wide = operational_v2_predict(
        cal, -20.0, -floatmax(Float64), floatmax(Float64), features,
    )
    @test all(isfinite, (wide.pred_dst, wide.ci05_dst, wide.ci95_dst))
    @test wide.ci05_dst == -floatmax(Float64)
    @test wide.ci95_dst == floatmax(Float64)

    correction_overflow = OperationalV2Calibration(
        [:Bz_nt], [-floatmax(Float64)], [1.0], [0.0, 1.0], 1.0,
        "correction_overflow",
    )
    @test_throws ArgumentError operational_v2_predict(
        correction_overflow, -20.0, -25.0, -15.0,
        (; Bz_nt=floatmax(Float64)),
    )
end

@testset "Operational V2 scoring avoids unused baseline allocation" begin
    df, cal = _v2_score_fixture(64)
    scored = score_operational_v2(df, cal)
    expected = clamp.(df.pred_dst_nt .+ df.Bz_nt, -2000.0, 50.0)
    @test scored.v2_pred_dst_nt == expected
    @test scored.v2_correction_dst_nt == df.Bz_nt
    @test scored.v2_pred_dst_ci05_nt == expected .- 5.0
    @test scored.v2_pred_dst_ci95_nt == expected .+ 5.0
    @test scored.v2_residual_dst_nt == df.observation_dst_nt .- expected

    alloc_200 = _allocated_v2_score(200)
    alloc_1_000 = _allocated_v2_score(1_000)
    required_baseline_alloc = _allocated_v2_score(1_000; baseline=true)
    @test alloc_1_000 <= 2_000_000
    @test alloc_1_000 <= 6alloc_200 + 131_072
    @test 2alloc_1_000 < required_baseline_alloc
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

        # If a joint-draw sidecar omits an active coefficient, that coefficient
        # remains fixed at its deployed point value; it must not be zeroed merely
        # because the uncertainty artifact is incomplete.
        partial_draws = joinpath(dir, "partial_draws.csv")
        CSV.write(partial_draws, DataFrame(Symbol(a1) => a1_draws))
        stp = init_forecast(coefficients_csv = coef, ensemble_csv = ens,
                            draws_csv = partial_draws,
                            t0 = DateTime(2024, 1, 1), dst0 = -20.0)
        @test all(stp.ξ_ensemble[:, i2] .== stp.ξ_primary[i2])
        @test all(stp.ξ_ensemble[:, i1] .!= 0.0)
    end
end


@testset "init_forecast rejects corrupt coefficient artifacts" begin
    mktempdir() do dir
        ens = joinpath(dir, "ens.csv")
        CSV.write(ens, DataFrame(term=["Bs"], inclusion_prob=[1.0],
                                 ci_025=[-1.1], ci_975=[-0.9]))
        function rejects(df)
            coef = joinpath(dir, "coef.csv")
            CSV.write(coef, df)
            @test_throws ArgumentError init_forecast(coefficients_csv=coef,
                                                      ensemble_csv=ens,
                                                      draws_csv=joinpath(dir, "missing.csv"),
                                                      t0=DateTime(2024, 1, 1), dst0=-20.0)
        end
        rejects(DataFrame(other=["Bs"], coefficient=[-1.0]))
        rejects(DataFrame(term=["Bs", "Bs"], coefficient=[-1.0, -2.0]))
        rejects(DataFrame(term=["not_a_library_term"], coefficient=[-1.0]))
        rejects(DataFrame(term=["Bs"], coefficient=[Inf]))
    end
end
