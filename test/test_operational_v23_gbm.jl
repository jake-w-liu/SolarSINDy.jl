using Test
using SolarSINDy
using DataFrames
using Dates
using EvoTrees
using Random
using Statistics

# ---------------------------------------------------------------------------
# Synthetic hourly frame with closed-form records.
#
#   row i  <->  record tag  start + Hour(i - 1)
#   V(i)   = 300 + 10 i
#   Bz(i)  = 5 - i/2                (i <= 30)   northward for i <= 9
#          = -10 + (i - 30)         (i > 30)    northward again for i > 40
#   By(i)  = -3 + i/4
#   n(i)   = 1 + i/2
#   Dst(i) = -5 - 2 i
#
# Every oracle below is written as an explicit number or an explicit record
# index, never by re-running the builder's index arithmetic.
# ---------------------------------------------------------------------------

const _V23G_START = DateTime(2015, 3, 1, 0)
const _V23G_HOURS = 72

_v23g_speed(i) = 300.0 + 10.0 * i
_v23g_bz(i) = i <= 30 ? 5.0 - 0.5 * i : -10.0 + 1.0 * (i - 30)
_v23g_by(i) = -3.0 + 0.25 * i
_v23g_density(i) = 1.0 + 0.5 * i
_v23g_dst(i) = -5.0 - 2.0 * i

function _v23g_frame(hours::Int=_V23G_HOURS)
    rows = 1:hours
    speed = [_v23g_speed(i) for i in rows]
    density = [_v23g_density(i) for i in rows]
    return DataFrame(
        time_utc=[_V23G_START + Hour(i - 1) for i in rows],
        V=speed,
        Bz=[_v23g_bz(i) for i in rows],
        By=[_v23g_by(i) for i in rows],
        n=density,
        Pdyn=[dynamic_pressure(density[i], speed[i]) for i in rows],
        Dst=[_v23g_dst(i) for i in rows],
    )
end

"Record tag of row `i` of the synthetic frame."
_v23g_tag(i::Int) = _V23G_START + Hour(i - 1)

"Issue time whose latest Dst record is row `i`."
_v23g_issue(i::Int) = _v23g_tag(i)

function _v23g_additive_problem()
    rng = MersenneTwister(20260817)
    n_train, n_test, p = 20_000, 5_000, 5
    X = rand(rng, n_train + n_test, p) .* 4.0 .- 2.0
    signal(x) = 3.0 * sin(2.0 * x[1]) + 1.5 * x[2] + 2.0 * (x[3] > 0.3) -
                0.75 * x[4]^2 + 0.5 * abs(x[5])
    y = [signal(view(X, i, :)) for i in axes(X, 1)]
    return (
        X_train=X[1:n_train, :], y_train=y[1:n_train],
        X_test=X[n_train+1:end, :], y_test=y[n_train+1:end],
    )
end

_v23g_rmse(a, b) = sqrt(mean(abs2, a .- b))

@testset verbose=true "Operational V2.3 boosted models" begin

    @testset "Direct-GBM schema constants" begin
        @test length(V23_DIRECT_EXTRA_FEATURE_NAMES) == 11
        @test V23_DIRECT_DST_LAG_HOURS == (1, 2, 3, 4, 5, 6, 12, 24)
        @test V23_DIRECT_VBS_LAG_STEPS == (1, 2, 3)
        @test V23_DIRECT_FEATURE_COUNT == 29
        @test V23_DIRECT_FEATURE_COUNT == 18 + length(V23_DIRECT_EXTRA_FEATURE_NAMES)
        @test allunique(V23_DIRECT_EXTRA_FEATURE_NAMES)
        @test V23_GBM_EVOTREES_VERSION == string(Base.pkgversion(EvoTrees))
        @test V23_GBM_DEFAULT_SEED == 22022026
    end

    @testset "Direct extra features: hand-computed oracle" begin
        frame = _v23g_frame()
        issue = _v23g_issue(31)
        X, ok = v23_direct_extra_features(frame, [issue])

        @test size(X) == (1, 11)
        @test ok isa BitVector
        @test ok[1]

        # Dst lags j in {1..6, 12, 24} read rows 30, 29, 28, 27, 26, 25, 19, 7,
        # i.e. Dst = -5 - 2i evaluated there.
        @test X[1, 1] == -65.0
        @test X[1, 2] == -63.0
        @test X[1, 3] == -61.0
        @test X[1, 4] == -59.0
        @test X[1, 5] == -57.0
        @test X[1, 6] == -55.0
        @test X[1, 7] == -43.0
        @test X[1, 8] == -19.0

        # VBs lag L reads driver record t-1-L, i.e. rows 29, 28, 27, with
        # VBs = V * max(0, -Bz) / 1000.
        @test X[1, 9] ≈ 590.0 * 9.5 / 1000.0 atol = 1e-12
        @test X[1, 10] ≈ 580.0 * 9.0 / 1000.0 atol = 1e-12
        @test X[1, 11] ≈ 570.0 * 8.5 / 1000.0 atol = 1e-12
        @test X[1, 9] ≈ 5.605 atol = 1e-12
        @test X[1, 10] ≈ 5.22 atol = 1e-12
        @test X[1, 11] ≈ 4.845 atol = 1e-12

        # Northward driver records rectify to exactly zero coupling: rows 42, 41
        # and 40 carry Bz = +2, +1 and 0.
        Xn, okn = v23_direct_extra_features(frame, [_v23g_issue(44)])
        @test okn[1]
        @test Xn[1, 9] == 0.0
        @test Xn[1, 10] == 0.0
        @test Xn[1, 11] == 0.0
        @test Xn[1, 1] == -91.0   # Dst at row 43

        # Batched evaluation must equal single-issue evaluation row by row.
        issues = [_v23g_issue(i) for i in (31, 44, 60)]
        Xb, okb = v23_direct_extra_features(frame, issues)
        @test okb == BitVector([true, true, true])
        @test Xb[1, :] == X[1, :]
        @test Xb[2, :] == Xn[1, :]
    end

    @testset "Direct extra features: incompleteness and gaps" begin
        frame = _v23g_frame()

        # Row 20 is not referenced by issue row 31 (lags hit 30..25, 19, 7), so
        # deleting it must leave the row untouched; deleting row 28 (Dst lag 3
        # and VBs lag 2) must invalidate it.
        reference, _ = v23_direct_extra_features(frame, [_v23g_issue(31)])
        without_20 = frame[Not(20), :]
        X20, ok20 = v23_direct_extra_features(without_20, [_v23g_issue(31)])
        @test ok20[1]
        @test X20[1, :] == reference[1, :]

        without_28 = frame[Not(28), :]
        X28, ok28 = v23_direct_extra_features(without_28, [_v23g_issue(31)])
        @test !ok28[1]
        @test all(isnan, X28)

        # Dst lag 24 is unavailable before row 25.
        Xearly, okearly = v23_direct_extra_features(frame, [_v23g_issue(24)])
        @test !okearly[1]
        @test all(isnan, Xearly)
        Xfirst, okfirst = v23_direct_extra_features(frame, [_v23g_issue(25)])
        @test okfirst[1]
        @test Xfirst[1, 8] == _v23g_dst(1)

        # A NaN in any required record invalidates the row.
        for (row, label) in ((19, "Dst lag 12"), (7, "Dst lag 24"))
            holed = copy(frame)
            holed.Dst[row] = NaN
            Xh, okh = v23_direct_extra_features(holed, [_v23g_issue(31)])
            @test !okh[1] || error("$label hole was ignored")
            @test all(isnan, Xh)
        end
        holed_speed = copy(frame)
        holed_speed.V[27] = NaN            # VBs lag 3
        Xs, oks = v23_direct_extra_features(holed_speed, [_v23g_issue(31)])
        @test !oks[1]
        holed_bz = copy(frame)
        holed_bz.Bz[29] = NaN              # VBs lag 1
        Xz, okz = v23_direct_extra_features(holed_bz, [_v23g_issue(31)])
        @test !okz[1]

        # Missing values expressed as `missing` behave like NaN.
        missing_frame = copy(frame)
        missing_frame.Dst = convert(Vector{Union{Missing,Float64}}, missing_frame.Dst)
        missing_frame.Dst[30] = missing
        Xm, okm = v23_direct_extra_features(missing_frame, [_v23g_issue(31)])
        @test !okm[1]
    end

    @testset "Direct extra features: post-issue mutation invariance" begin
        frame = _v23g_frame()
        issue = _v23g_issue(31)
        reference, ok_reference = v23_direct_extra_features(frame, [issue])
        @test ok_reference[1]

        # Every record tagged at or after the issue hour is future information.
        mutated = copy(frame)
        for row in 31:nrow(mutated)
            mutated.V[row] = 1.0e4 - 7.0 * row
            mutated.Bz[row] = -250.0 + row
            mutated.By[row] = 137.0 - row
            mutated.n[row] = 90.0 + row
            mutated.Pdyn[row] = 55.0
            mutated.Dst[row] = -900.0 + row
        end
        Xmut, okmut = v23_direct_extra_features(mutated, [issue])
        @test okmut[1]
        @test Xmut[1, :] == reference[1, :]

        blanked = copy(frame)
        for row in 31:nrow(blanked)
            blanked.V[row] = NaN
            blanked.Bz[row] = NaN
            blanked.By[row] = NaN
            blanked.n[row] = NaN
            blanked.Pdyn[row] = NaN
            blanked.Dst[row] = NaN
        end
        Xblank, okblank = v23_direct_extra_features(blanked, [issue])
        @test okblank[1]
        @test Xblank[1, :] == reference[1, :]

        # The invariance check is only meaningful if a pre-issue mutation does
        # move the features.
        earlier = copy(frame)
        earlier.Dst[30] += 1.0
        Xearlier, _ = v23_direct_extra_features(earlier, [issue])
        @test Xearlier[1, 1] == reference[1, 1] + 1.0
        @test Xearlier[1, 2:end] == reference[1, 2:end]

        earlier_bz = copy(frame)
        earlier_bz.Bz[28] -= 1.0           # VBs lag 2 becomes more southward
        Xbz, _ = v23_direct_extra_features(earlier_bz, [issue])
        @test Xbz[1, 10] ≈ 580.0 * 10.0 / 1000.0 atol = 1e-12
        @test Xbz[1, 10] > reference[1, 10]
    end

    @testset "Frame validation" begin
        frame = _v23g_frame()
        @test_throws ArgumentError v23_direct_extra_features(
            select(frame, Not(:Pdyn)), [_v23g_issue(31)])
        @test_throws ArgumentError v23_gdc_targets(
            select(frame, Not(:By)), [_v23g_issue(31)], 3)

        duplicated = vcat(frame, frame[31:31, :])
        @test_throws ArgumentError v23_direct_extra_features(duplicated, [_v23g_issue(31)])

        @test_throws ArgumentError v23_gdc_targets(frame, [_v23g_issue(31)], 0)
    end

    @testset "GDC targets: hand-computed oracle" begin
        frame = _v23g_frame()
        issue = _v23g_issue(31)

        # k = 1: target record is row 31, base record is row 30.
        bz, by, dlogv, dlogn = v23_gdc_targets(frame, [issue], 1)
        @test bz[1] == -9.0
        @test by[1] == 4.75
        @test dlogv[1] ≈ log(610.0) - log(600.0) atol = 1e-14
        @test dlogn[1] ≈ log(16.5) - log(16.0) atol = 1e-14

        # k = 3: target record is row 33.
        bz3, by3, dlogv3, dlogn3 = v23_gdc_targets(frame, [issue], 3)
        @test bz3[1] == -7.0
        @test by3[1] == 5.25
        @test dlogv3[1] ≈ log(630.0) - log(600.0) atol = 1e-14
        @test dlogn3[1] ≈ log(17.5) - log(16.0) atol = 1e-14
        @test dlogv3[1] > dlogv[1] > 0.0

        # k = 7 on a southward record.
        bz7, by7, dlogv7, dlogn7 = v23_gdc_targets(frame, [issue], 7)
        @test bz7[1] == -3.0               # row 37
        @test by7[1] == 6.25
        @test dlogv7[1] ≈ log(670.0) - log(600.0) atol = 1e-14
        @test dlogn7[1] ≈ log(19.5) - log(16.0) atol = 1e-14

        # Targets are reported as NaN when the target record is outside the
        # frame, and independently when a logarithm is undefined.
        bze, bye, dve, dne = v23_gdc_targets(frame, [_v23g_issue(72)], 3)
        @test all(isnan, (bze[1], bye[1], dve[1], dne[1]))

        nonpositive = copy(frame)
        nonpositive.V[33] = 0.0
        bzn, byn, dvn, dnn = v23_gdc_targets(nonpositive, [issue], 3)
        @test bzn[1] == -7.0
        @test byn[1] == 5.25
        @test isnan(dvn[1])
        @test dnn[1] ≈ log(17.5) - log(16.0) atol = 1e-14

        holed = copy(frame)
        holed.By[33] = NaN
        _, byh, dvh, _ = v23_gdc_targets(holed, [issue], 3)
        @test isnan(byh[1])
        @test dvh[1] ≈ log(630.0) - log(600.0) atol = 1e-14

        # Batched evaluation agrees with single-issue evaluation.
        issues = [_v23g_issue(i) for i in (31, 40, 55)]
        bzb, byb, dvb, dnb = v23_gdc_targets(frame, issues, 3)
        @test bzb[1] == bz3[1] && byb[1] == by3[1]
        @test dvb[1] == dlogv3[1] && dnb[1] == dlogn3[1]
        @test bzb[2] == _v23g_bz(42)
        @test dvb[3] ≈ log(_v23g_speed(57)) - log(_v23g_speed(54)) atol = 1e-14
    end

    @testset "Direct feature combiner" begin
        if isdefined(SolarSINDy, :v23_feature_matrix)
            frame = _v23g_frame()
            issues = [_v23g_issue(i) for i in (31, 44, 24)]
            X, ok = v23_direct_features(frame, issues)
            adc, adc_ok = SolarSINDy.v23_feature_matrix(frame, issues)
            extra, extra_ok = v23_direct_extra_features(frame, issues)

            @test size(X) == (3, V23_DIRECT_FEATURE_COUNT)
            @test size(adc, 2) == 18
            @test ok == BitVector(adc_ok .& extra_ok)
            @test !ok[3]                    # issue 24 has no Dst lag 24
            @test all(isnan, X[3, :])
            for row in 1:2
                @test X[row, 1:18] == adc[row, :]
                @test X[row, 19:end] == extra[row, :]
            end
            @test length(v23_direct_feature_names()) == V23_DIRECT_FEATURE_COUNT
            @test allunique(v23_direct_feature_names())

            # The combined block is causal: records strictly after the issue
            # hour cannot move any of the 29 columns.
            mutated = copy(frame)
            for row in 32:nrow(mutated)
                mutated.V[row] = 1.0e4 - 7.0 * row
                mutated.Bz[row] = -250.0 + row
                mutated.By[row] = 137.0 - row
                mutated.n[row] = 90.0 + row
                mutated.Pdyn[row] = 55.0
                mutated.Dst[row] = -900.0 + row
            end
            Xmut, okmut = v23_direct_features(mutated, [_v23g_issue(31)])
            @test okmut[1]
            @test Xmut[1, :] == X[1, :]
        else
            @info "v23_feature_matrix is not defined; skipping the direct-feature combiner test"
            @test_throws ArgumentError v23_direct_features(
                _v23g_frame(), [_v23g_issue(31)])
            @test_throws ArgumentError v23_direct_feature_names()
        end
    end

    @testset "Boosted fit recovers a known additive function" begin
        problem = _v23g_additive_problem()
        signal_sd = std(problem.y_train)
        @test signal_sd > 2.0

        model = v23_fit_gbm(
            problem.X_train, problem.y_train;
            max_depth=5, nrounds=1600, nbins=255,
        )
        train_rmse = _v23g_rmse(v23_predict(model, problem.X_train), problem.y_train)
        test_rmse = _v23g_rmse(v23_predict(model, problem.X_test), problem.y_test)

        @test train_rmse < 0.05 * signal_sd
        @test test_rmse < 0.05 * signal_sd

        # A single depth-1 stump cannot explain the function: this is what makes
        # the tolerance above informative rather than automatic.
        stump = v23_fit_gbm(
            problem.X_train, problem.y_train; max_depth=1, nrounds=1, nbins=255,
        )
        stump_rmse = _v23g_rmse(v23_predict(stump, problem.X_test), problem.y_test)
        @test stump_rmse > 0.5 * signal_sd
        @test test_rmse < 0.1 * stump_rmse
    end

    @testset "Boosted fit determinism and hyper-parameter wiring" begin
        rng = MersenneTwister(4)
        X = rand(rng, 2_000, 4) .* 2.0 .- 1.0
        y = [2.0 * X[i, 1] - 1.0 * X[i, 2]^2 + 0.5 * (X[i, 3] > 0.0) for i in axes(X, 1)]

        base = v23_fit_gbm(X, y; max_depth=4, nrounds=64)
        repeat = v23_fit_gbm(X, y; max_depth=4, nrounds=64)
        @test v23_predict(base, X) == v23_predict(repeat, X)
        @test v23_predict(base, X) isa Vector{Float64}

        deeper = v23_fit_gbm(X, y; max_depth=6, nrounds=64)
        longer = v23_fit_gbm(X, y; max_depth=4, nrounds=256)
        faster = v23_fit_gbm(X, y; max_depth=4, nrounds=64, eta=0.30)
        coarser = v23_fit_gbm(X, y; max_depth=4, nrounds=64, nbins=8)
        heavier = v23_fit_gbm(X, y; max_depth=4, nrounds=64, min_weight=512)

        base_rmse = _v23g_rmse(v23_predict(base, X), y)
        for other in (deeper, longer, faster, coarser, heavier)
            @test v23_predict(other, X) != v23_predict(base, X)
        end
        @test _v23g_rmse(v23_predict(longer, X), y) < base_rmse
        @test _v23g_rmse(v23_predict(faster, X), y) < base_rmse
        @test _v23g_rmse(v23_predict(coarser, X), y) > base_rmse
        @test _v23g_rmse(v23_predict(heavier, X), y) > base_rmse

        # Named features are carried into the fitted schema.
        named = v23_fit_gbm(X, y; max_depth=4, nrounds=8,
                            feature_names=["a", "b", "c", "d"])
        @test Symbol.(named.info[:feature_names]) == [:a, :b, :c, :d]
        @test v23_predict(base, X) ==
              v23_predict(v23_fit_gbm(X, y; max_depth=4, nrounds=64,
                                      feature_names=["x1", "x2", "x3", "x4"]), X)
    end

    @testset "Boosted model persistence round trip" begin
        rng = MersenneTwister(11)
        X = rand(rng, 800, 3)
        y = [1.5 * X[i, 1] + (X[i, 2] > 0.5) - 0.25 * X[i, 3] for i in axes(X, 1)]
        model = v23_fit_gbm(X, y; max_depth=3, nrounds=48)
        expected = v23_predict(model, X)

        mktempdir() do dir
            path = joinpath(dir, "nested", "gdc-bz-step3.bson")
            @test v23_save(model, path) == path
            @test isfile(path)
            restored = v23_load(path)
            @test restored isa EvoTrees.EvoTree
            @test v23_predict(restored, X) == expected
            @test restored.info[:feature_names] == model.info[:feature_names]

            link = joinpath(dir, "gdc-link.bson")
            symlink(path, link)
            @test_throws ArgumentError v23_load(link)
            @test_throws ArgumentError v23_save(model, link)
            @test_throws ArgumentError v23_load(joinpath(dir, "absent.bson"))
        end
    end

    @testset "Boosted fit and prediction validation" begin
        rng = MersenneTwister(19)
        X = rand(rng, 128, 3)
        y = vec(sum(X; dims=2))

        @test_throws DimensionMismatch v23_fit_gbm(X, y[1:end-1]; max_depth=3, nrounds=8)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=0, nrounds=8)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=0)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=8, nbins=1)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=8, eta=0.0)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=8, lambda=-1.0)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=8, seed=-1)
        @test_throws ArgumentError v23_fit_gbm(X, y; max_depth=3, nrounds=8, rowsample=0.0)
        @test_throws DimensionMismatch v23_fit_gbm(
            X, y; max_depth=3, nrounds=8, feature_names=["a", "b"])
        @test_throws ArgumentError v23_fit_gbm(
            X, y; max_depth=3, nrounds=8, feature_names=["a", "a", "b"])
        @test_throws ArgumentError v23_fit_gbm(X[1:1, :], y[1:1]; max_depth=3, nrounds=8)

        holed = copy(X)
        holed[5, 2] = NaN
        @test_throws ArgumentError v23_fit_gbm(holed, y; max_depth=3, nrounds=8)
        holed_target = copy(y)
        holed_target[9] = Inf
        @test_throws ArgumentError v23_fit_gbm(X, holed_target; max_depth=3, nrounds=8)

        model = v23_fit_gbm(X, y; max_depth=3, nrounds=8)
        @test_throws DimensionMismatch v23_predict(model, X[:, 1:2])
        @test_throws ArgumentError v23_predict(model, holed)
        @test_throws ArgumentError v23_predict("not a model", X)
        @test_throws ArgumentError v23_save("not a model", tempname())
    end
end
