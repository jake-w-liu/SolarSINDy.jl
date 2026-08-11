const V22D = SolarSINDy

function _v22d_hand_rollout(artifact, history)
    center = collect(artifact.center)
    scale = collect(artifact.scale)
    values = (Matrix{Float64}(history) .- transpose(center)) ./ transpose(scale)
    coefficients = operational_v22_driver_coefficients(artifact)
    for _ in 1:OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS
        next = collect(artifact.intercept)
        anchor = size(values, 1)
        for (lag_index, lag) in pairs(OPERATIONAL_V22_DRIVER_LAGS)
            next .+= coefficients[:, :, lag_index] * values[anchor - lag, :]
        end
        values = vcat(values, transpose(next))
    end
    rolled = values[(size(history, 1) + 1):end, :]
    return rolled .* transpose(scale) .+ transpose(center)
end

function _v22d_synthetic_states(; rows=1800, seed=90210)
    rng = MersenneTwister(seed)
    coefficients = zeros(5, 5, 6)
    coefficients[:, :, 1] .= Diagonal([0.42, 0.36, 0.31, 0.28, 0.24])
    coefficients[2, 1, 3] = 0.18
    coefficients[3, 4, 3] = -0.16
    coefficients[5, 2, 3] = 0.14
    states = zeros(Float64, rows, 5)
    states[1:25, :] .= randn(rng, 25, 5)
    for anchor in 25:(rows - 1)
        next = zeros(5)
        for (lag_index, lag) in pairs(OPERATIONAL_V22_DRIVER_LAGS)
            next .+= coefficients[:, :, lag_index] * states[anchor - lag, :]
        end
        states[anchor + 1, :] .= next .+ 0.03 .* randn(rng, 5)
    end
    expected_support = falses(5, 6)
    expected_support[:, 1] .= true
    expected_support[[1, 4, 2], 3] .= true
    return states, coefficients, expected_support
end

@testset "Operational V2.2-M2 sparse driver continuation" begin
    @test OPERATIONAL_V22_DRIVER_STATES == (:Bx, :By, :Bz, :logV, :logn)
    @test OPERATIONAL_V22_DRIVER_LAGS == (0, 1, 2, 6, 12, 24)
    @test OPERATIONAL_V22_DRIVER_CADENCE_MINUTES == 30
    @test OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS == 14
    @test OPERATIONAL_V22_DRIVER_STABILITY_TOLERANCE == 1.0e-8

    @testset "independent hand rollout and frozen indexing" begin
        coefficients = zeros(5, 5, 6)
        coefficients[:, :, 1] .= Diagonal([0.40, 0.35, 0.30, 0.25, 0.20])
        coefficients[1, 2, 2] = 0.08
        coefficients[3, 5, 3] = -0.06
        coefficients[4, 1, 6] = 0.04
        center = [2.0, -1.0, 4.0, 6.0, 1.5]
        scale = [2.0, 3.0, 4.0, 0.5, 0.25]
        intercept = [0.1, -0.2, 0.05, 0.0, 0.03]
        artifact = OperationalV22DriverArtifact(
            coefficients;
            center=center,
            scale=scale,
            intercept=intercept,
            fit_rows=200,
            label="hand-rollout",
        )
        history = reshape(collect(1.0:125.0), 25, 5) ./ 20
        expected = _v22d_hand_rollout(artifact, history)
        actual = operational_v22_driver_rollout(artifact, history)
        @test size(actual) == (14, 5)
        @test actual ≈ expected atol=5e-14 rtol=5e-14
        @test actual[1, 1] ≈ expected[1, 1] atol=1e-14
        @test actual[1, 4] ≈ expected[1, 4] atol=1e-14
        @test operational_v22_driver_coefficients(artifact) == coefficients
        @test operational_v22_driver_support(artifact) ==
              dropdims(any(!iszero, coefficients; dims=1); dims=1)
        @test size(operational_v22_driver_companion(artifact)) == (125, 125)
        @test operational_v22_driver_companion(artifact)[1:5, 121:125] ==
              coefficients[:, :, 6]
        @test_throws DimensionMismatch operational_v22_driver_rollout(
            artifact, history[2:end, :],
        )
    end

    @testset "known support, joint group threshold, and deterministic fit" begin
        states, true_coefficients, expected_support = _v22d_synthetic_states()
        artifact = fit_operational_v22_driver(
            states; ridge=1.0e-6, threshold=1.0e-1, label="support-recovery",
        )
        repeated = fit_operational_v22_driver(
            states; ridge=1.0e-6, threshold=1.0e-1, label="support-recovery",
        )
        @test operational_v22_driver_support(artifact) == expected_support
        @test operational_v22_driver_sha256(repeated) ==
              operational_v22_driver_sha256(artifact)
        @test operational_v22_driver_coefficients(repeated) ==
              operational_v22_driver_coefficients(artifact)
        normalization = Diagonal(collect(artifact.scale))
        fitted_standardized = operational_v22_driver_coefficients(artifact)
        fitted_original = similar(fitted_standardized)
        for lag_index in axes(fitted_standardized, 3)
            fitted_original[:, :, lag_index] .=
                normalization * fitted_standardized[:, :, lag_index] / normalization
        end
        @test maximum(abs.(fitted_original .- true_coefficients)) < 6.0e-2
        @test artifact.fit_rows == size(states, 1) - 25
        @test 1 <= artifact.threshold_iterations <= 20
        @test artifact.spectral_radius <= 1.0 + 1.0e-8
        @test all(isfinite, operational_v22_driver_rollout(
            artifact, states[(end - 24):end, :],
        ))

        # Each output coefficient is below θ, but their joint group norm exceeds θ.
        design = Matrix{Float64}(I, 40, 40)[:, 1:30]
        targets = zeros(40, 5)
        targets[:, :] .= design[:, 1] * transpose(fill(0.02, 5))
        intercept, fitted = V22D._operational_v22_driver_ridge_fit(
            design, targets, trues(30), 1.0e-6,
        )
        @test all(abs.(fitted[1, :]) .< 3.0e-2)
        @test norm(fitted[1, :]) > 3.0e-2
        @test all(isfinite, intercept)
        _, _, joint_support, _ = V22D._operational_v22_driver_threshold_fit(
            design, targets, 1.0e-6, 3.0e-2,
        )
        @test joint_support[1]
        @test count(joint_support) == 1

        zero_coefficients = zeros(5, 5, 6)
        @test_throws ArgumentError OperationalV22DriverArtifact(
            zero_coefficients;
            support_mask=trues(5, 6),
            threshold=3.0e-1,
            fit_rows=40,
        )
        zero_threshold = OperationalV22DriverArtifact(
            zero_coefficients;
            support_mask=trues(5, 6),
            threshold=0.0,
            fit_rows=40,
        )
        @test all(operational_v22_driver_support(zero_threshold))

        rng = MersenneTwister(117)
        raw_basis = randn(rng, 80, 30)
        raw_basis .-= mean(raw_basis; dims=1)
        basis = Matrix(qr(raw_basis).Q)[:, 1:30]
        cascading_design = copy(basis)
        correlation = 0.9
        cascading_design[:, 2] .=
            correlation .* basis[:, 1] .+
            sqrt(1.0 - correlation^2) .* basis[:, 2]
        cascading_targets = zeros(80, 5)
        cascading_targets[:, 1] .=
            -0.09 .* cascading_design[:, 1] .+
             0.11 .* cascading_design[:, 2]
        @test_throws ErrorException V22D._operational_v22_driver_threshold_fit(
            cascading_design,
            cascading_targets,
            1.0e-6,
            1.0e-1;
            max_iterations=1,
        )
        _, _, converged_support, converged_iterations =
            V22D._operational_v22_driver_threshold_fit(
                cascading_design,
                cascading_targets,
                1.0e-6,
                1.0e-1;
                max_iterations=2,
            )
        @test !any(converged_support)
        @test converged_iterations == 2

        constant_targets = repeat(reshape(collect(1.0:5.0), 1, 5), 40, 1)
        unpenalized_intercept, constant_coefficients =
            V22D._operational_v22_driver_ridge_fit(
                fill(3.0, 40, 30), constant_targets, trues(30), 1.0e2,
            )
        @test unpenalized_intercept == collect(1.0:5.0)
        @test all(iszero, constant_coefficients)
    end

    @testset "spectral-radius rejection without rescaling" begin
        stable_coefficients = zeros(5, 5, 6)
        stable_coefficients[:, :, 1] .= 0.5 .* Matrix{Float64}(I, 5, 5)
        stable = OperationalV22DriverArtifact(stable_coefficients; fit_rows=50)
        @test operational_v22_driver_spectral_radius(stable) ≈ 0.5 atol=2e-14
        @test operational_v22_driver_coefficients(stable) == stable_coefficients

        unstable_coefficients = copy(stable_coefficients)
        unstable_coefficients[:, :, 1] .=
            (1.0 + 2.0e-8) .* Matrix{Float64}(I, 5, 5)
        @test_throws ArgumentError OperationalV22DriverArtifact(
            unstable_coefficients; fit_rows=50,
        )
        @test unstable_coefficients[1, 1, 1] == 1.0 + 2.0e-8
        @test_throws ArgumentError fit_operational_v22_driver(
            fill(1.0, 26, 5); ridge=1.0e-6, threshold=0.0,
        )
        @test_throws ArgumentError fit_operational_v22_driver(
            randn(MersenneTwister(1), 26, 5); ridge=2.0e-6, threshold=0.0,
        )
        @test_throws ArgumentError fit_operational_v22_driver(
            randn(MersenneTwister(1), 26, 5); ridge=1.0e-6, threshold=2.0e-2,
        )

        overflowing = OperationalV22DriverArtifact(
            0.9 .* stable_coefficients;
            intercept=fill(1.0e308, 5),
            fit_rows=50,
        )
        @test_throws ArgumentError operational_v22_driver_rollout(
            overflowing, zeros(25, 5),
        )
    end

    @testset "checksummed artifact round trip and corruption" begin
        states, _, _ = _v22d_synthetic_states(rows=600)
        artifact = fit_operational_v22_driver(
            states; ridge=1.0e-5, threshold=1.0e-1, label="roundtrip",
        )
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "driver.csv")
            @test write_operational_v22_driver(path, artifact) == path
            restored = read_operational_v22_driver(path)
            @test operational_v22_driver_sha256(restored) ==
                  operational_v22_driver_sha256(artifact)
            @test operational_v22_driver_coefficients(restored) ==
                  operational_v22_driver_coefficients(artifact)
            @test operational_v22_driver_support(restored) ==
                  operational_v22_driver_support(artifact)

            invalid_support = CSV.read(path, DataFrame)
            excluded_row = findfirst(==(false), invalid_support.selected)
            @test excluded_row !== nothing
            invalid_support.selected[excluded_row] = true
            CSV.write(path, invalid_support)
            @test_throws ArgumentError read_operational_v22_driver(path)

            write_operational_v22_driver(path, artifact)
            corrupted = CSV.read(path, DataFrame)
            corrupted.coefficient_bx[1] += 0.125
            CSV.write(path, corrupted)
            @test_throws ArgumentError read_operational_v22_driver(path)

            write_operational_v22_driver(path, artifact)
            wrong_lag = CSV.read(path, DataFrame)
            wrong_lag.lag_samples[1] = 1
            CSV.write(path, wrong_lag)
            @test_throws ArgumentError read_operational_v22_driver(path)

            write_operational_v22_driver(path, artifact)
            link = joinpath(tmp, "driver-link.csv")
            symlink(path, link)
            @test_throws ArgumentError read_operational_v22_driver(link)
            @test_throws ArgumentError write_operational_v22_driver(link, artifact)
        end
    end
end
