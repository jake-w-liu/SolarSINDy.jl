using Test, SolarSINDy

@testset "Portable boost rejects nonfinite narrowing and accumulation" begin
    schema = SolarSINDy.STATIC_STACK_BOOST_FEATURES
    artifact_for(leaves) = StaticStackBoostArtifact(
        "boost-representability-oracle", 1, 0, schema, 0.4,
        10.0, 64, 7, 1, 1, 8, 0.1, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        (1, 2), (1, 2), (true, true), (0, 0), (0.0, 0.0),
        (0, 0), (0, 0), leaves, (0.0f0, 0.0f0),
    )
    artifact = artifact_for((0.0f0, 25.0f0))
    finite = NamedTuple{schema}(Tuple(fill(1.0, length(schema))))
    # A finite Real input does not justify admitting an infinite tree feature.
    for excessive in (BigFloat("1e400"), -BigFloat("1e400"))
        matrix = fill(excessive, 1, length(schema))
        features = NamedTuple{schema}(Tuple(vec(matrix)))
        @test isfinite(excessive) && isinf(Float64(excessive))
        @test_throws ArgumentError SolarSINDy._static_stack_boost_validate_matrix(matrix, schema)
        @test_throws ArgumentError SolarSINDy._static_stack_boost_raw_predict(artifact, matrix)
        @test_throws ArgumentError static_stack_boost_raw_predict(artifact, features)
        @test_throws ArgumentError static_stack_boost_predict(artifact, 1, 0, -50.0, features)
    end
    # The portable contract adds leaf contributions in Float32, exactly as the
    # original regressor does. Reject overflow rather than hide it behind a cap.
    for sign in (-1.0f0, 1.0f0)
        leaves = (sign * floatmax(Float32), sign * floatmax(Float32))
        explosive = artifact_for(leaves)
        @test all(isfinite, leaves)
        @test_throws ArgumentError static_stack_boost_raw_predict(explosive, finite)
        @test_throws ArgumentError static_stack_boost_predict(explosive, 1, 0, -50.0, finite)
        @test_throws ArgumentError SolarSINDy._static_stack_boost_raw_predict(
            explosive, ones(1, length(schema)),
        )
    end
    # Ordinary inference retains the exact existing Float32 sum and cap values.
    result = static_stack_boost_predict(artifact, 1, 0, -50.0, finite)
    @test result.raw_correction_nt == 25.0
    @test result.clipped_raw_correction_nt == 10.0
    @test result.correction_nt == 4.0
    @test result.pred_dst == -46.0
    @test static_stack_boost_raw_predict(artifact, finite) === 25.0
    @test SolarSINDy._static_stack_boost_raw_predict(artifact, ones(2, length(schema))) ==
          [25.0f0, 25.0f0]
end
