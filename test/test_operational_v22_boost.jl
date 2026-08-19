using Test
using SolarSINDy
using EvoTrees
using DataFrames
using CSV
using Random
using Statistics

const V22B = SolarSINDy
const V22B_FEATURES = OPERATIONAL_V22_BOOST_FEATURES

function _v22b_synthetic(n::Int=384)
    X = Matrix{Float64}(undef, n, length(V22B_FEATURES))
    for row in 1:n, feature in axes(X, 2)
        X[row, feature] = sin(0.017 * row * (feature + 1)) +
                          0.35 * cos(0.031 * row * (feature + 3)) +
                          0.002 * mod(row * feature, 17)
    end
    y = @. 4.5 * (X[:, 1] > 0.15) - 3.0 * (X[:, 2] < -0.2) +
           2.0 * X[:, 3] * X[:, 4] + 0.4 * X[:, 5]
    return X, y
end

const V22B_TRAINING = (
    seed=2405,
    nrounds=32,
    max_depth=3,
    nbins=32,
    eta=0.08,
    l2_regularization=2.0,
    lambda=0.0,
    gamma=0.0,
    min_weight=2.0,
    rowsample=1.0,
    colsample=1.0,
)

function _v22b_fit(X, y; rho=0.75)
    return fit_operational_v22_boost(
        X, y; model_step_hours=1, rho, feature_names=V22B_FEATURES,
        V22B_TRAINING..., label="v2.2-boost-synthetic",
    )
end

function _v22b_evotree(X, y)
    p = V22B_TRAINING
    config = EvoTreeRegressor(
        loss=:mse, metric=:rmse, nrounds=p.nrounds, bagging_size=1,
        early_stopping_rounds=typemax(Int), L2=p.l2_regularization,
        lambda=p.lambda, gamma=p.gamma, eta=p.eta, max_depth=p.max_depth,
        min_weight=p.min_weight, rowsample=p.rowsample, colsample=p.colsample,
        nbins=p.nbins, seed=p.seed, tree_type=:binary, device=:cpu,
    )
    return EvoTrees.fit(
        config; x_train=X, y_train=y,
        feature_names=collect(String.(V22B_FEATURES)), verbosity=0,
    )
end

function _v22b_extract(model, n; rho=0.75)
    p = V22B_TRAINING
    return extract_operational_v22_boost(
        model; model_step_hours=1, feature_names=V22B_FEATURES, rho,
        fit_rows=n, seed=p.seed, max_depth=p.max_depth, nbins=p.nbins,
        eta=p.eta, l2_regularization=p.l2_regularization, lambda=p.lambda,
        gamma=p.gamma, min_weight=p.min_weight, rowsample=p.rowsample,
        colsample=p.colsample, label="v2.2-boost-synthetic",
    )
end

function _v22b_named_row(row)
    return NamedTuple{V22B_FEATURES}(Tuple(Float64.(row)))
end

function _v22b_constant_artifact(raw::Float32; rho::Float64=0.4, lead::Int=1)
    # Bias plus one fitted leaf tree. Summation order and Float32 values are the
    # same as an extracted EvoTrees regressor.
    return OperationalV22BoostArtifact(
        "v2.2-boost-cap-oracle", lead, 0, V22B_FEATURES, rho,
        5.0 + 5.0 * lead, 64, 7, 1, 1, 8, 0.1, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        (1, 2), (1, 2), (true, true), (0, 0), (0.0, 0.0),
        (0, 0), (0, 0), (0.0f0, raw), (0.0f0, 0.0f0),
    )
end

@testset verbose=true "Operational V2.2 portable boosted residual" begin
    X, y = _v22b_synthetic()

    @testset "Nonlinear fit and repeat-fit determinism" begin
        artifact_a = _v22b_fit(X, y)
        artifact_b = _v22b_fit(X, y)
        fitted = V22B._operational_v22_boost_raw_predict(artifact_a, X)

        @test sqrt(mean(abs2, Float64.(fitted) .- y)) < 0.65 * std(y)
        @test V22B._operational_v22_boost_sha256(artifact_a) ==
              V22B._operational_v22_boost_sha256(artifact_b)
        @test artifact_a.tree_roots == artifact_b.tree_roots
        @test artifact_a.node_feature_index == artifact_b.node_feature_index
        @test artifact_a.node_threshold == artifact_b.node_threshold
        @test artifact_a.node_leaf_value == artifact_b.node_leaf_value
        @test !ismutabletype(OperationalV22BoostArtifact)
        @test length(artifact_a.feature_names) == 22
        @test artifact_a.feature_names == V22B_FEATURES
    end

    @testset "Extracted inference is exact at thresholds and random rows" begin
        model = _v22b_evotree(X, y)
        artifact = _v22b_extract(model, size(X, 1))
        reference = EvoTrees.predict(model, X)
        portable = V22B._operational_v22_boost_raw_predict(artifact, X)
        @test portable == reference

        rng = MersenneTwister(991)
        random_rows = randn(rng, 128, length(V22B_FEATURES))
        @test V22B._operational_v22_boost_raw_predict(artifact, random_rows) ==
              EvoTrees.predict(model, random_rows)

        split_nodes = findall(!, artifact.node_is_leaf)
        @test !isempty(split_nodes)
        for node in first(split_nodes, min(12, length(split_nodes)))
            feature = artifact.node_feature_index[node]
            threshold = artifact.node_threshold[node]
            for value in (prevfloat(threshold), threshold, nextfloat(threshold))
                edge_row = zeros(1, length(V22B_FEATURES))
                edge_row[feature] = value
                @test V22B._operational_v22_boost_raw_predict(artifact, edge_row) ==
                      EvoTrees.predict(model, edge_row)
            end
        end
    end

    @testset "Cap, shrinkage, lead, lag, and schema oracles" begin
        values = _v22b_named_row(zeros(length(V22B_FEATURES)))
        positive = _v22b_constant_artifact(25.0f0; rho=0.4, lead=1)
        high = operational_v22_boost_predict(positive, 1, 0, -30.0, values)
        @test high.raw_correction_nt == 25.0
        @test high.clipped_raw_correction_nt == 10.0
        @test high.correction_nt == 4.0
        @test high.pred_dst == -26.0
        @test high.correction_was_capped
        @test high.tree_contributions_nt == (0.0, 25.0)

        negative = _v22b_constant_artifact(-30.0f0; rho=0.5, lead=2)
        low = operational_v22_boost_predict(negative, 2, 0, -30.0, values)
        @test low.raw_correction_nt == -30.0
        @test low.clipped_raw_correction_nt == -15.0
        @test low.correction_nt == -7.5
        @test low.pred_dst == -37.5
        @test_throws ArgumentError operational_v22_boost_predict(
            positive, 2, 0, -30.0, values,
        )
        @test_throws ArgumentError operational_v22_boost_predict(
            positive, 1, 1, -30.0, values,
        )
        @test_throws ArgumentError operational_v22_boost_predict(
            positive, 1, 0, -30.0, merge(values, (unexpected=1.0,)),
        )
        reordered_names = reverse(V22B_FEATURES)
        reordered = NamedTuple{reordered_names}(reverse(Tuple(values)))
        @test_throws ArgumentError operational_v22_boost_predict(
            positive, 1, 0, -30.0, reordered,
        )
        @test_throws ArgumentError fit_operational_v22_boost(
            X[:, 1:21], y; model_step_hours=1,
            feature_names=V22B_FEATURES[1:21], V22B_TRAINING...,
        )
        @test_throws ArgumentError OperationalV22BoostArtifact(
            "bad-lag", 1, 1, V22B_FEATURES, 0.4, 10.0, 64, 7, 1, 1, 8,
            0.1, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            (1, 2), (1, 2), (true, true), (0, 0), (0.0, 0.0),
            (0, 0), (0, 0), (0.0f0, 1.0f0), (0.0f0, 0.0f0),
        )
    end

    @testset "Target and post-issue mutation cannot alter inference" begin
        artifact = _v22b_fit(X, y)
        rows = 1:12
        frame = DataFrame(
            model_step_hours=fill(1, length(rows)),
            anchor_lag_hours=zeros(Int, length(rows)),
            v2_2_pred_dst_nt=collect(range(-42.0, -18.0; length=length(rows))),
            observation_dst_nt=collect(range(-40.0, -15.0; length=length(rows))),
            post_issue_driver=collect(1.0:length(rows)),
        )
        for (feature, name) in enumerate(V22B_FEATURES)
            frame[!, name] = X[rows, feature]
        end
        mutated = copy(frame)
        mutated.observation_dst_nt .= collect(range(900.0, -900.0; length=length(rows)))
        mutated.post_issue_driver .= [isodd(i) ? Inf : -Inf for i in rows]
        scored = score_operational_v22_boost(frame, artifact)
        rescored = score_operational_v22_boost(mutated, artifact)
        forecast_columns = [
            :v2_2_boost_pred_dst_nt,
            :v2_2_boost_raw_correction_nt,
            :v2_2_boost_clipped_raw_correction_nt,
            :v2_2_boost_correction_nt,
            :v2_2_boost_correction_was_capped,
            :v2_2_boost_correction_cap_nt,
            :v2_2_boost_rho,
            :v2_2_boost_tree_count,
            :v2_2_boost_tree_contributions_nt,
            :v2_2_boost_feature_schema,
            :v2_2_boost_label,
            :v2_2_boost_artifact_sha256,
        ]
        @test scored[:, forecast_columns] == rescored[:, forecast_columns]
        @test scored.v2_2_boost_residual_dst_nt !=
              rescored.v2_2_boost_residual_dst_nt
    end

    @testset "Atomic artifact round trip and corruption rejection" begin
        artifact = _v22b_fit(X, y)
        values = _v22b_named_row(X[17, :])
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "v22-boost.csv")
            @test write_operational_v22_boost(path, artifact) == path
            restored = read_operational_v22_boost(path)
            @test V22B._operational_v22_boost_sha256(restored) ==
                  V22B._operational_v22_boost_sha256(artifact)
            @test operational_v22_boost_predict(restored, 1, 0, -20.0, values) ==
                  operational_v22_boost_predict(artifact, 1, 0, -20.0, values)

            valid = CSV.read(path, DataFrame)
            corrupted = copy(valid)
            leaf = findfirst(corrupted.is_leaf)
            corrupted.leaf_value[leaf] += 0.25
            corrupted_path = joinpath(tmp, "corrupted.csv")
            CSV.write(corrupted_path, corrupted)
            @test_throws ArgumentError read_operational_v22_boost(corrupted_path)

            bad_version = copy(valid)
            bad_version[!, :evotrees_version] = String.(bad_version.evotrees_version)
            bad_version.evotrees_version .= "0.18.6"
            bad_version_path = joinpath(tmp, "bad-version.csv")
            CSV.write(bad_version_path, bad_version)
            @test_throws ArgumentError read_operational_v22_boost(bad_version_path)

            bad_lag = copy(valid)
            bad_lag.anchor_lag_hours .= 1
            bad_lag_path = joinpath(tmp, "bad-lag.csv")
            CSV.write(bad_lag_path, bad_lag)
            @test_throws ArgumentError read_operational_v22_boost(bad_lag_path)

            bad_child = copy(valid)
            split = findfirst(!, bad_child.is_leaf)
            bad_child.left_child[split] = nrow(bad_child) + 1
            bad_child_path = joinpath(tmp, "bad-child.csv")
            CSV.write(bad_child_path, bad_child)
            @test_throws ArgumentError read_operational_v22_boost(bad_child_path)

            wrong_schema_path = joinpath(tmp, "wrong-schema.csv")
            CSV.write(wrong_schema_path, select(valid, Not(:split_gain)))
            @test_throws ArgumentError read_operational_v22_boost(wrong_schema_path)

            link = joinpath(tmp, "boost-link.csv")
            symlink(path, link)
            @test_throws ArgumentError read_operational_v22_boost(link)
            @test_throws ArgumentError write_operational_v22_boost(link, artifact)

            # S10: the digest is lowercase hex everywhere this package writes one, and every other
            # reader requires that spelling. This reader case-folded, so it accepted a digest the
            # identity never produced.
            upper = copy(valid)
            upper[!, :artifact_sha256] = uppercase.(string.(upper.artifact_sha256))
            @test occursin(r"^[0-9A-F]{64}$", upper.artifact_sha256[1])
            upper_path = joinpath(tmp, "upper-digest.csv")
            CSV.write(upper_path, upper)
            @test_throws ArgumentError read_operational_v22_boost(upper_path)
            # The same file with the digest spelled as written still loads.
            lower = copy(upper)
            lower[!, :artifact_sha256] = lowercase.(lower.artifact_sha256)
            lower_path = joinpath(tmp, "lower-digest.csv")
            CSV.write(lower_path, lower)
            @test SolarSINDy._operational_v22_boost_sha256(
                      read_operational_v22_boost(lower_path)) ==
                  SolarSINDy._operational_v22_boost_sha256(artifact)
        end
    end
end
