# Portable, bounded boosted-residual inference for Operational static stack.

import EvoTrees
import SHA

"Frozen issue-time feature schema for the static stack boosted residual."
const STATIC_STACK_BOOST_FEATURES = STATIC_STACK_RESIDUAL_FEATURES
"Anchor-relative model steps, in hours, supported by the boosted residual."
const STATIC_STACK_BOOST_SUPPORTED_MODEL_STEPS = STATIC_STACK_MODEL_STEPS
"Schema identifier for portable boosted-residual artifacts."
const STATIC_STACK_BOOST_SCHEMA_VERSION = "static_stack_boost_v1"
const STATIC_STACK_BOOST_PACKAGE_VERSION = "SolarSINDy-0.2.1"
const STATIC_STACK_BOOST_EVOTREES_VERSION = "0.18.7"

_static_stack_boost_cap(model_step_hours::Integer) =
    static_stack_correction_cap_nt(model_step_hours)

"""
Immutable, lead-specific boosted-residual artifact.

The tree ensemble is stored as flat, portable node arrays. Child indices and
tree roots are one-based global indices into those arrays. Serving therefore
does not depend on Julia object serialization or EvoTrees internals.
"""
struct StaticStackBoostArtifact
    label::String
    model_step_hours::Int
    anchor_lag_hours::Int
    feature_names::Tuple{Vararg{Symbol}}
    rho::Float64
    correction_cap_nt::Float64
    fit_rows::Int
    seed::Int
    nrounds::Int
    max_depth::Int
    nbins::Int
    eta::Float64
    l2_regularization::Float64
    lambda::Float64
    gamma::Float64
    min_weight::Float64
    rowsample::Float64
    colsample::Float64
    tree_roots::Tuple{Vararg{Int}}
    node_tree_index::Tuple{Vararg{Int}}
    node_is_leaf::Tuple{Vararg{Bool}}
    node_feature_index::Tuple{Vararg{Int}}
    node_threshold::Tuple{Vararg{Float64}}
    node_left_child::Tuple{Vararg{Int}}
    node_right_child::Tuple{Vararg{Int}}
    node_leaf_value::Tuple{Vararg{Float32}}
    node_split_gain::Tuple{Vararg{Float32}}

    function StaticStackBoostArtifact(
            label::String,
            model_step_hours::Int,
            anchor_lag_hours::Int,
            feature_names::Tuple{Vararg{Symbol}},
            rho::Float64,
            correction_cap_nt::Float64,
            fit_rows::Int,
            seed::Int,
            nrounds::Int,
            max_depth::Int,
            nbins::Int,
            eta::Float64,
            l2_regularization::Float64,
            lambda::Float64,
            gamma::Float64,
            min_weight::Float64,
            rowsample::Float64,
            colsample::Float64,
            tree_roots::Tuple{Vararg{Int}},
            node_tree_index::Tuple{Vararg{Int}},
            node_is_leaf::Tuple{Vararg{Bool}},
            node_feature_index::Tuple{Vararg{Int}},
            node_threshold::Tuple{Vararg{Float64}},
            node_left_child::Tuple{Vararg{Int}},
            node_right_child::Tuple{Vararg{Int}},
            node_leaf_value::Tuple{Vararg{Float32}},
            node_split_gain::Tuple{Vararg{Float32}})
        isempty(strip(label)) && throw(ArgumentError(
            "boost artifact label must not be empty",
        ))
        model_step_hours in STATIC_STACK_BOOST_SUPPORTED_MODEL_STEPS ||
            throw(ArgumentError("unsupported boosted-residual model step: $model_step_hours"))
        anchor_lag_hours == 0 || throw(ArgumentError(
            "only lag-zero boosted-residual artifacts are supported",
        ))
        feature_names == STATIC_STACK_BOOST_FEATURES || throw(ArgumentError(
            "boosted-residual feature schema must exactly match the frozen 22-feature schema",
        ))
        isfinite(rho) && 0.0 < rho <= 1.0 || throw(ArgumentError(
            "boosted-residual rho must be finite and in (0, 1]",
        ))
        correction_cap_nt == _static_stack_boost_cap(model_step_hours) ||
            throw(ArgumentError("boosted-residual correction cap must equal 5 + 5h nT"))
        fit_rows >= 2 || throw(ArgumentError(
            "boosted-residual artifact requires at least two fit rows",
        ))
        seed >= 0 || throw(ArgumentError("boosted-residual seed must be nonnegative"))
        1 <= nrounds <= 512 || throw(ArgumentError(
            "boosted-residual nrounds must be between 1 and 512",
        ))
        1 <= max_depth <= 3 || throw(ArgumentError(
            "boosted-residual max_depth must be between 1 and 3",
        ))
        2 <= nbins <= 255 || throw(ArgumentError(
            "boosted-residual nbins must be between 2 and 255",
        ))
        isfinite(eta) && 0.0 < eta <= 1.0 || throw(ArgumentError(
            "boosted-residual eta must be finite and in (0, 1]",
        ))
        all(x -> isfinite(x) && x >= 0.0,
            (l2_regularization, lambda, gamma)) || throw(ArgumentError(
                "boosted-residual regularization values must be finite and nonnegative",
            ))
        isfinite(min_weight) && min_weight > 0.0 || throw(ArgumentError(
            "boosted-residual min_weight must be finite and positive",
        ))
        all(x -> isfinite(x) && 0.0 < x <= 1.0,
            (rowsample, colsample)) || throw(ArgumentError(
                "boosted-residual sampling fractions must be finite and in (0, 1]",
            ))

        node_count = length(node_tree_index)
        node_count >= 2 || throw(ArgumentError(
            "boosted-residual artifact must contain a bias tree and fitted tree",
        ))
        all(length(values) == node_count for values in (
            node_is_leaf, node_feature_index, node_threshold, node_left_child,
            node_right_child, node_leaf_value, node_split_gain,
        )) || throw(DimensionMismatch(
            "boosted-residual node arrays must have identical lengths",
        ))
        length(tree_roots) == nrounds + 1 || throw(ArgumentError(
            "boosted-residual artifact must contain one bias tree plus nrounds trees",
        ))
        first(tree_roots) == 1 && collect(tree_roots) == sort!(unique(collect(tree_roots))) ||
            throw(ArgumentError(
                "boosted-residual tree roots must be unique, sorted, and start at one",
            ))
        last(tree_roots) <= node_count || throw(ArgumentError(
            "boosted-residual tree root lies outside the node arrays",
        ))
        all(tree -> 1 <= tree <= length(tree_roots), node_tree_index) || throw(ArgumentError(
            "boosted-residual node tree index is outside the tree range",
        ))

        for tree in eachindex(tree_roots)
            start = tree_roots[tree]
            stop = tree == length(tree_roots) ? node_count : tree_roots[tree + 1] - 1
            start <= stop || throw(ArgumentError(
                "boosted-residual tree $tree contains no nodes",
            ))
            all(==(tree), node_tree_index[start:stop]) || throw(ArgumentError(
                "boosted-residual node tree indices are not contiguous",
            ))
            stop - start + 1 <= 2^max_depth - 1 || throw(ArgumentError(
                "boosted-residual tree $tree exceeds max_depth metadata",
            ))
            seen = Set{Int}()
            pending = Int[start]
            while !isempty(pending)
                node = pop!(pending)
                node in seen && throw(ArgumentError(
                    "boosted-residual tree $tree contains a cycle or shared child",
                ))
                start <= node <= stop || throw(ArgumentError(
                    "boosted-residual tree $tree child leaves its node range",
                ))
                push!(seen, node)
                leaf = node_is_leaf[node]
                feature = node_feature_index[node]
                threshold = node_threshold[node]
                left = node_left_child[node]
                right = node_right_child[node]
                value = node_leaf_value[node]
                gain = node_split_gain[node]
                isfinite(value) && isfinite(gain) && gain >= 0.0f0 ||
                    throw(ArgumentError(
                        "boosted-residual node values and gains must be finite and valid",
                    ))
                if leaf
                    feature == 0 && threshold == 0.0 && left == 0 && right == 0 &&
                        gain == 0.0f0 || throw(ArgumentError(
                            "boosted-residual leaf node has split metadata",
                        ))
                else
                    1 <= feature <= length(feature_names) || throw(ArgumentError(
                        "boosted-residual split feature index is outside the schema",
                    ))
                    isfinite(threshold) || throw(ArgumentError(
                        "boosted-residual split threshold must be finite",
                    ))
                    value == 0.0f0 || throw(ArgumentError(
                        "boosted-residual internal node must have zero leaf value",
                    ))
                    push!(pending, right)
                    push!(pending, left)
                end
            end
            seen == Set(start:stop) || throw(ArgumentError(
                "boosted-residual tree $tree contains unreachable nodes",
            ))
        end
        tree_roots[1] == 1 && (length(tree_roots) == 1 || tree_roots[2] == 2) ||
            throw(ArgumentError("boosted-residual bias tree must contain one node"))
        node_is_leaf[1] || throw(ArgumentError(
            "boosted-residual first tree must be a leaf-only bias tree",
        ))

        return new(
            label, model_step_hours, anchor_lag_hours, feature_names, rho,
            correction_cap_nt, fit_rows, seed, nrounds, max_depth, nbins, eta,
            l2_regularization, lambda, gamma, min_weight, rowsample, colsample,
            tree_roots, node_tree_index, node_is_leaf, node_feature_index,
            node_threshold, node_left_child, node_right_child, node_leaf_value,
            node_split_gain,
        )
    end
end

function StaticStackBoostArtifact(
        label::AbstractString,
        model_step_hours::Integer,
        anchor_lag_hours::Integer,
        feature_names,
        rho::Real,
        correction_cap_nt::Real,
        fit_rows::Integer,
        seed::Integer,
        nrounds::Integer,
        max_depth::Integer,
        nbins::Integer,
        eta::Real,
        l2_regularization::Real,
        lambda::Real,
        gamma::Real,
        min_weight::Real,
        rowsample::Real,
        colsample::Real,
        tree_roots,
        node_tree_index,
        node_is_leaf,
        node_feature_index,
        node_threshold,
        node_left_child,
        node_right_child,
        node_leaf_value,
        node_split_gain)
    return StaticStackBoostArtifact(
        String(label), Int(model_step_hours), Int(anchor_lag_hours),
        Tuple(Symbol.(feature_names)), Float64(rho), Float64(correction_cap_nt),
        Int(fit_rows), Int(seed), Int(nrounds), Int(max_depth), Int(nbins),
        Float64(eta), Float64(l2_regularization), Float64(lambda), Float64(gamma),
        Float64(min_weight), Float64(rowsample), Float64(colsample),
        Tuple(Int.(tree_roots)), Tuple(Int.(node_tree_index)),
        Tuple(Bool.(node_is_leaf)), Tuple(Int.(node_feature_index)),
        Tuple(Float64.(node_threshold)), Tuple(Int.(node_left_child)),
        Tuple(Int.(node_right_child)), Tuple(Float32.(node_leaf_value)),
        Tuple(Float32.(node_split_gain)),
    )
end

function _static_stack_boost_current_evotrees_version()
    version = string(Base.pkgversion(EvoTrees))
    version == STATIC_STACK_BOOST_EVOTREES_VERSION || throw(ArgumentError(
        "boost extraction requires EvoTrees " *
        STATIC_STACK_BOOST_EVOTREES_VERSION * "; found $version",
    ))
    return version
end

function _static_stack_boost_validate_matrix(X::AbstractMatrix,
                                                feature_names)
    features = Tuple(Symbol.(feature_names))
    features == STATIC_STACK_BOOST_FEATURES || throw(ArgumentError(
        "boost training feature schema must exactly match the frozen 22-feature schema",
    ))
    size(X, 2) == length(features) || throw(DimensionMismatch(
        "boost training matrix must have exactly $(length(features)) columns",
    ))
    size(X, 1) >= 1 || throw(ArgumentError(
        "boost feature matrix requires at least one row",
    ))
    all(value -> value isa Real && !(value isa Bool) && isfinite(value), X) ||
        throw(ArgumentError("boost training features must be finite real values"))
    matrix = Matrix{Float64}(X)
    all(isfinite, matrix) || throw(ArgumentError(
        "boost training features must remain finite in Float64",
    ))
    return matrix, features
end

"""
    extract_static_stack_boost(model; kwargs...)

Extract an EvoTrees 0.18.7 MSE regressor into a portable immutable artifact.
Only continuous predictors with the exact frozen feature schema are accepted.
"""
function extract_static_stack_boost(
        model;
        model_step_hours::Integer,
        anchor_lag_hours::Integer=0,
        feature_names=STATIC_STACK_BOOST_FEATURES,
        rho::Real=1.0,
        fit_rows::Integer,
        seed::Integer,
        max_depth::Integer,
        nbins::Integer,
        eta::Real,
        l2_regularization::Real,
        lambda::Real,
        gamma::Real,
        min_weight::Real,
        rowsample::Real,
        colsample::Real,
        label::AbstractString="operational-static-stack-boost")
    _static_stack_boost_current_evotrees_version()
    model isa EvoTrees.EvoTree{EvoTrees.MSE,1} || throw(ArgumentError(
        "boost extraction only supports a scalar EvoTrees MSE regressor",
    ))
    features = Tuple(Symbol.(feature_names))
    features == STATIC_STACK_BOOST_FEATURES || throw(ArgumentError(
        "boost extraction feature schema must exactly match the frozen 22-feature schema",
    ))
    haskey(model.info, :feature_names) &&
        Tuple(Symbol.(model.info[:feature_names])) == features || throw(ArgumentError(
            "EvoTrees model feature schema does not match extraction schema",
        ))
    haskey(model.info, :feattypes) && all(model.info[:feattypes]) ||
        throw(ArgumentError("boost extraction only supports ordered numeric features"))
    haskey(model.info, :edges) && length(model.info[:edges]) == length(features) ||
        throw(ArgumentError("EvoTrees model edge metadata is incomplete"))
    nrounds = length(model.trees) - 1
    haskey(model.info, :nrounds) && Int(model.info[:nrounds]) == nrounds ||
        throw(ArgumentError("EvoTrees model round metadata is inconsistent"))

    tree_roots = Int[]
    node_tree_index = Int[]
    node_is_leaf = Bool[]
    node_feature_index = Int[]
    node_threshold = Float64[]
    node_left_child = Int[]
    node_right_child = Int[]
    node_leaf_value = Float32[]
    node_split_gain = Float32[]

    for (tree_index, tree) in enumerate(model.trees)
        push!(tree_roots, length(node_tree_index) + 1)
        old_nodes = Int[1]
        old_to_global = Dict{Int,Int}()
        cursor = 1
        while cursor <= length(old_nodes)
            old = old_nodes[cursor]
            old <= length(tree.split) || throw(ArgumentError(
                "EvoTrees tree contains a child outside its node arrays",
            ))
            old_to_global[old] = length(node_tree_index) + 1
            push!(node_tree_index, tree_index)
            push!(node_is_leaf, !tree.split[old])
            push!(node_feature_index, 0)
            push!(node_threshold, 0.0)
            push!(node_left_child, 0)
            push!(node_right_child, 0)
            push!(node_leaf_value, !tree.split[old] ? tree.pred[1, old] : 0.0f0)
            push!(node_split_gain, tree.split[old] ? tree.gain[old] : 0.0f0)
            if tree.split[old]
                push!(old_nodes, old << 1)
                push!(old_nodes, (old << 1) + 1)
            end
            cursor += 1
        end

        for old in old_nodes
            global_node = old_to_global[old]
            tree.split[old] || continue
            feature = tree.feat[old]
            condition_bin = Int(tree.cond_bin[old])
            1 <= feature <= length(features) || throw(ArgumentError(
                "EvoTrees split feature lies outside the frozen schema",
            ))
            edges = model.info[:edges][feature]
            1 <= condition_bin <= length(edges) || throw(ArgumentError(
                "EvoTrees split bin lies outside its edge metadata",
            ))
            threshold = Float64(edges[condition_bin])
            isfinite(threshold) || throw(ArgumentError(
                "EvoTrees split threshold is non-finite",
            ))
            node_feature_index[global_node] = feature
            node_threshold[global_node] = threshold
            node_left_child[global_node] = old_to_global[old << 1]
            node_right_child[global_node] = old_to_global[(old << 1) + 1]
        end
    end

    return StaticStackBoostArtifact(
        label, model_step_hours, anchor_lag_hours, features, rho,
        _static_stack_boost_cap(model_step_hours), fit_rows, seed, nrounds,
        max_depth, nbins, eta, l2_regularization, lambda, gamma, min_weight,
        rowsample, colsample, tree_roots, node_tree_index, node_is_leaf,
        node_feature_index, node_threshold, node_left_child, node_right_child,
        node_leaf_value, node_split_gain,
    )
end

"""
    fit_static_stack_boost(X, residual_target; kwargs...)

Fit one deterministic CPU EvoTrees model and immediately extract its portable
serving artifact. This is intentionally a low-level single-configuration fit;
model selection belongs to the leakage-controlled development workflow.
"""
function fit_static_stack_boost(
        X::AbstractMatrix,
        residual_target::AbstractVector;
        model_step_hours::Integer,
        anchor_lag_hours::Integer=0,
        feature_names=STATIC_STACK_BOOST_FEATURES,
        rho::Real=1.0,
        seed::Integer=2026045,
        nrounds::Integer=64,
        max_depth::Integer=3,
        nbins::Integer=64,
        eta::Real=0.05,
        l2_regularization::Real=5.0,
        lambda::Real=0.0,
        gamma::Real=0.0,
        min_weight::Real=8.0,
        rowsample::Real=1.0,
        colsample::Real=1.0,
        label::AbstractString="operational-static-stack-boost")
    matrix, features = _static_stack_boost_validate_matrix(X, feature_names)
    size(matrix, 1) >= 2 || throw(ArgumentError(
        "boost training matrix requires at least two rows",
    ))
    length(residual_target) == size(matrix, 1) || throw(DimensionMismatch(
        "boost residual target length must equal the training row count",
    ))
    all(value -> value isa Real && !(value isa Bool) && isfinite(value),
        residual_target) || throw(ArgumentError(
            "boost residual targets must be finite real values",
        ))
    # Validate all serving/training metadata before entering EvoTrees.
    0 <= Int(seed) || throw(ArgumentError("boosted-residual seed must be nonnegative"))
    1 <= Int(nrounds) <= 512 || throw(ArgumentError(
        "boosted-residual nrounds must be between 1 and 512",
    ))
    1 <= Int(max_depth) <= 3 || throw(ArgumentError(
        "boosted-residual max_depth must be between 1 and 3",
    ))
    2 <= Int(nbins) <= 255 || throw(ArgumentError(
        "boosted-residual nbins must be between 2 and 255",
    ))
    config = EvoTrees.EvoTreeRegressor(
        loss=:mse, metric=:rmse, nrounds=Int(nrounds), bagging_size=1,
        early_stopping_rounds=typemax(Int), L2=Float64(l2_regularization),
        lambda=Float64(lambda), gamma=Float64(gamma), eta=Float64(eta),
        max_depth=Int(max_depth), min_weight=Float64(min_weight),
        rowsample=Float64(rowsample), colsample=Float64(colsample),
        nbins=Int(nbins), seed=Int(seed), tree_type=:binary, device=:cpu,
    )
    model = EvoTrees.fit(
        config; x_train=matrix, y_train=Float64.(residual_target),
        feature_names=collect(String.(features)), verbosity=0,
    )
    return extract_static_stack_boost(
        model; model_step_hours, anchor_lag_hours, feature_names=features, rho,
        fit_rows=size(matrix, 1), seed, max_depth, nbins, eta,
        l2_regularization, lambda, gamma, min_weight, rowsample, colsample, label,
    )
end

function _static_stack_boost_feature_tuple(artifact::StaticStackBoostArtifact,
                                              features::NamedTuple)
    propertynames(features) == artifact.feature_names || throw(ArgumentError(
        "boost prediction features must exactly match the frozen ordered schema",
    ))
    values = ntuple(length(artifact.feature_names)) do index
        value = getfield(features, index)
        value isa Real && !(value isa Bool) && isfinite(value) ||
            throw(ArgumentError("boost prediction feature values must be finite real numbers"))
        converted = Float64(value)
        isfinite(converted) || throw(ArgumentError(
            "boost prediction features must remain finite in Float64",
        ))
        converted
    end
    return values
end

function _static_stack_boost_tree_contributions(
        artifact::StaticStackBoostArtifact,
        values::Tuple)
    contributions = Vector{Float32}(undef, length(artifact.tree_roots))
    for tree in eachindex(artifact.tree_roots)
        node = artifact.tree_roots[tree]
        while !artifact.node_is_leaf[node]
            feature = artifact.node_feature_index[node]
            node = values[feature] <= artifact.node_threshold[node] ?
                artifact.node_left_child[node] : artifact.node_right_child[node]
        end
        contributions[tree] = artifact.node_leaf_value[node]
    end
    return contributions
end

function _static_stack_boost_sum(contributions::AbstractVector{Float32})
    total = 0.0f0
    for contribution in contributions
        total += contribution
    end
    isfinite(total) || throw(ArgumentError(
        "boost raw correction is outside the finite Float32 inference range",
    ))
    return total
end

"Exact portable raw-correction inference for one ordered feature NamedTuple."
function static_stack_boost_raw_predict(
        artifact::StaticStackBoostArtifact,
        features::NamedTuple)
    values = _static_stack_boost_feature_tuple(artifact, features)
    return Float64(_static_stack_boost_sum(
        _static_stack_boost_tree_contributions(artifact, values),
    ))
end

function _static_stack_boost_raw_predict(
        artifact::StaticStackBoostArtifact,
        X::AbstractMatrix)
    matrix, features = _static_stack_boost_validate_matrix(X, artifact.feature_names)
    features == artifact.feature_names || throw(ArgumentError(
        "boost matrix feature schema does not match artifact",
    ))
    out = Vector{Float32}(undef, size(matrix, 1))
    for row in axes(matrix, 1)
        values = Tuple(@view(matrix[row, :]))
        out[row] = _static_stack_boost_sum(
            _static_stack_boost_tree_contributions(artifact, values),
        )
    end
    return out
end

"Apply exact raw inference followed by the frozen cap and shrinkage contract."
function static_stack_boost_predict(
        artifact::StaticStackBoostArtifact,
        model_step_hours::Integer,
        anchor_lag_hours::Integer,
        base_dst_nt::Real,
        features::NamedTuple)
    lead = Int(model_step_hours)
    lead == artifact.model_step_hours || throw(ArgumentError(
        "boost artifact supports lead $(artifact.model_step_hours) h, not $lead h",
    ))
    lag = Int(anchor_lag_hours)
    lag == artifact.anchor_lag_hours == 0 || throw(ArgumentError(
        "boost artifact supports anchor lag 0 h only",
    ))
    base_dst_nt isa Bool && throw(ArgumentError("boost base forecast must be real"))
    base = Float64(base_dst_nt)
    isfinite(base) || throw(ArgumentError("boost base forecast must be finite"))
    values = _static_stack_boost_feature_tuple(artifact, features)
    contributions = _static_stack_boost_tree_contributions(artifact, values)
    raw = Float64(_static_stack_boost_sum(contributions))
    clipped = clamp(raw, -artifact.correction_cap_nt, artifact.correction_cap_nt)
    correction = artifact.rho * clipped
    return (
        pred_dst=base + correction,
        base_dst_nt=base,
        raw_correction_nt=raw,
        clipped_raw_correction_nt=clipped,
        correction_nt=correction,
        correction_was_capped=clipped != raw,
        correction_cap_nt=artifact.correction_cap_nt,
        rho=artifact.rho,
        model_step_hours=lead,
        anchor_lag_hours=lag,
        tree_contributions_nt=Tuple(Float64.(contributions)),
        tree_leaf_count=length(contributions),
        label=artifact.label,
    )
end

"Score a frame without consulting observations or post-issue columns for inference."
function score_static_stack_boost(
        df::DataFrame,
        artifact::StaticStackBoostArtifact;
        model_step_column::Symbol=:model_step_hours,
        anchor_lag_column::Symbol=:anchor_lag_hours,
        base_column::Symbol=:static_stack_pred_dst_nt,
        observation_column::Symbol=:observation_dst_nt)
    _static_stack_require_columns(
        df, [model_step_column, anchor_lag_column, base_column,
             artifact.feature_names...],
    )
    out = copy(df)
    n = nrow(out)
    predicted = Vector{Float64}(undef, n)
    raw = Vector{Float64}(undef, n)
    clipped = Vector{Float64}(undef, n)
    correction = Vector{Float64}(undef, n)
    capped = Vector{Bool}(undef, n)
    contributions = Vector{String}(undef, n)
    for row in 1:n
        feature_values = NamedTuple{artifact.feature_names}(ntuple(
            index -> _static_stack_finite_cell(
                out, row, artifact.feature_names[index],
            ),
            length(artifact.feature_names),
        ))
        lag_value = _static_stack_finite_cell(out, row, anchor_lag_column)
        isinteger(lag_value) && typemin(Int) <= lag_value <= typemax(Int) ||
            throw(ArgumentError("boost anchor lag must be an integer"))
        result = static_stack_boost_predict(
            artifact,
            _static_stack_model_step(out, row, model_step_column),
            Int(lag_value),
            _static_stack_finite_cell(out, row, base_column),
            feature_values,
        )
        predicted[row] = result.pred_dst
        raw[row] = result.raw_correction_nt
        clipped[row] = result.clipped_raw_correction_nt
        correction[row] = result.correction_nt
        capped[row] = result.correction_was_capped
        contributions[row] = join(string.(result.tree_contributions_nt), ";")
    end
    out[!, :static_stack_boost_pred_dst_nt] = predicted
    out[!, :static_stack_boost_raw_correction_nt] = raw
    out[!, :static_stack_boost_clipped_raw_correction_nt] = clipped
    out[!, :static_stack_boost_correction_nt] = correction
    out[!, :static_stack_boost_correction_was_capped] = capped
    out[!, :static_stack_boost_correction_cap_nt] = fill(artifact.correction_cap_nt, n)
    out[!, :static_stack_boost_rho] = fill(artifact.rho, n)
    out[!, :static_stack_boost_tree_count] = fill(length(artifact.tree_roots), n)
    out[!, :static_stack_boost_tree_contributions_nt] = contributions
    out[!, :static_stack_boost_feature_schema] =
        fill(join(String.(artifact.feature_names), ";"), n)
    out[!, :static_stack_boost_label] = fill(artifact.label, n)
    out[!, :static_stack_boost_artifact_sha256] =
        fill(_static_stack_boost_sha256(artifact), n)
    if String(observation_column) in names(out)
        residual = Vector{Union{Missing,Float64}}(undef, n)
        for row in 1:n
            value = out[row, observation_column]
            if ismissing(value)
                residual[row] = missing
            else
                value isa Real && !(value isa Bool) && isfinite(value) ||
                    throw(ArgumentError(
                        "boost observation must be finite real or missing",
                    ))
                residual[row] = Float64(value) - predicted[row]
            end
        end
        out[!, :static_stack_boost_residual_dst_nt] = residual
    end
    return out
end

function _static_stack_boost_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) :
           value isa Float32 ? bitstring(value) : string(value)
    print(io, ncodeunits(text), ':', text, '|')
    return nothing
end

function _static_stack_boost_sha256(artifact::StaticStackBoostArtifact)
    io = IOBuffer()
    for value in (
            STATIC_STACK_BOOST_SCHEMA_VERSION,
            STATIC_STACK_BOOST_PACKAGE_VERSION,
            STATIC_STACK_BOOST_EVOTREES_VERSION,
            artifact.label, artifact.model_step_hours, artifact.anchor_lag_hours,
            artifact.rho, artifact.correction_cap_nt, artifact.fit_rows,
            artifact.seed, artifact.nrounds, artifact.max_depth, artifact.nbins,
            artifact.eta, artifact.l2_regularization, artifact.lambda,
            artifact.gamma, artifact.min_weight, artifact.rowsample,
            artifact.colsample,
        )
        _static_stack_boost_hash_token(io, value)
    end
    for values in (
            artifact.feature_names, artifact.tree_roots, artifact.node_tree_index,
            artifact.node_is_leaf, artifact.node_feature_index,
            artifact.node_threshold, artifact.node_left_child,
            artifact.node_right_child, artifact.node_leaf_value,
            artifact.node_split_gain,
        )
        _static_stack_boost_hash_token(io, length(values))
        for value in values
            _static_stack_boost_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _STATIC_STACK_BOOST_CSV_COLUMNS = (
    :schema_version, :package_version, :evotrees_version, :artifact_sha256,
    :label, :model_step_hours, :anchor_lag_hours, :feature_schema, :rho,
    :correction_cap_nt, :fit_rows, :seed, :nrounds, :max_depth, :nbins, :eta,
    :l2_regularization, :lambda, :gamma, :min_weight, :rowsample, :colsample,
    :tree_count, :node_count, :tree_roots, :node_index, :tree_index, :is_leaf,
    :feature_index, :threshold, :left_child, :right_child, :leaf_value,
    :split_gain,
)

"Atomically write a checksummed, strictly versioned portable boost artifact."
function write_static_stack_boost(path::AbstractString,
                                     artifact::StaticStackBoostArtifact)
    target = String(path)
    mkpath(dirname(abspath(target)))
    checksum = _static_stack_boost_sha256(artifact)
    roots = join(string.(artifact.tree_roots), ";")
    schema = join(String.(artifact.feature_names), ";")
    rows = NamedTuple[]
    for node in eachindex(artifact.node_tree_index)
        push!(rows, (
            schema_version=STATIC_STACK_BOOST_SCHEMA_VERSION,
            package_version=STATIC_STACK_BOOST_PACKAGE_VERSION,
            evotrees_version=STATIC_STACK_BOOST_EVOTREES_VERSION,
            artifact_sha256=checksum,
            label=artifact.label,
            model_step_hours=artifact.model_step_hours,
            anchor_lag_hours=artifact.anchor_lag_hours,
            feature_schema=schema,
            rho=artifact.rho,
            correction_cap_nt=artifact.correction_cap_nt,
            fit_rows=artifact.fit_rows,
            seed=artifact.seed,
            nrounds=artifact.nrounds,
            max_depth=artifact.max_depth,
            nbins=artifact.nbins,
            eta=artifact.eta,
            l2_regularization=artifact.l2_regularization,
            lambda=artifact.lambda,
            gamma=artifact.gamma,
            min_weight=artifact.min_weight,
            rowsample=artifact.rowsample,
            colsample=artifact.colsample,
            tree_count=length(artifact.tree_roots),
            node_count=length(artifact.node_tree_index),
            tree_roots=roots,
            node_index=node,
            tree_index=artifact.node_tree_index[node],
            is_leaf=artifact.node_is_leaf[node],
            feature_index=artifact.node_feature_index[node],
            threshold=artifact.node_threshold[node],
            left_child=artifact.node_left_child[node],
            right_child=artifact.node_right_child[node],
            leaf_value=artifact.node_leaf_value[node],
            split_gain=artifact.node_split_gain[node],
        ))
    end
    _write_selection_csv(target, rows)
    return target
end

function _static_stack_boost_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "boost artifact $field must be numeric",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "boost artifact $field must be finite",
    ))
    return converted
end

function _static_stack_boost_bool(value, field::AbstractString)
    value isa Bool || throw(ArgumentError("boost artifact $field must be Boolean"))
    return value
end

function _static_stack_boost_split_ints(value, field::AbstractString)
    text = string(value)
    isempty(text) && throw(ArgumentError("boost artifact $field must not be empty"))
    values = try
        parse.(Int, split(text, ";"))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("boost artifact $field is not a valid integer list"))
    end
    return values
end

"Read and validate every byte-significant field of a portable boost artifact."
function read_static_stack_boost(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "boost artifact must be a regular non-symlink file: $source",
    ))
    # Text-valued columns are read as text: a numeric-looking label ("007") would otherwise parse
    # as a number and change the hashed string. `validate = false` keeps a missing column reporting
    # through the schema check below rather than through the parser.
    df = CSV.read(IOBuffer(read(source)), DataFrame;
                  types = Dict(:schema_version => String, :package_version => String,
                               :evotrees_version => String, :artifact_sha256 => String,
                               :label => String, :feature_schema => String,
                               :tree_roots => String),
                  validate = false)
    names(df) == collect(String.(_STATIC_STACK_BOOST_CSV_COLUMNS)) ||
        throw(ArgumentError(
            "boost artifact CSV schema does not exactly match " *
            STATIC_STACK_BOOST_SCHEMA_VERSION,
        ))
    nrow(df) >= 2 || throw(ArgumentError("boost artifact CSV is incomplete"))
    for row in 1:nrow(df), column in _STATIC_STACK_BOOST_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "boost artifact CSV contains missing at row $row column $column",
        ))
    end
    schema_version = string(_static_stack_consistent_column(df, :schema_version))
    schema_version == STATIC_STACK_BOOST_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported boost artifact schema version: $schema_version",
    ))
    package_version = string(_static_stack_consistent_column(df, :package_version))
    package_version == STATIC_STACK_BOOST_PACKAGE_VERSION || throw(ArgumentError(
        "unsupported boost artifact package version: $package_version",
    ))
    evotrees_version = string(_static_stack_consistent_column(df, :evotrees_version))
    evotrees_version == STATIC_STACK_BOOST_EVOTREES_VERSION ||
        throw(ArgumentError(
            "unsupported boost artifact EvoTrees version: $evotrees_version",
        ))
    # No case folding: every writer in this package emits a lowercase hex digest, and every other
    # reader requires one. Folding here accepted a digest spelling the identity never produced.
    checksum = string(_static_stack_consistent_column(df, :artifact_sha256))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "boost artifact checksum is malformed",
    ))
    feature_names = Tuple(Symbol.(split(string(
        _static_stack_consistent_column(df, :feature_schema),
    ), ";")))
    feature_names == STATIC_STACK_BOOST_FEATURES || throw(ArgumentError(
        "boost artifact frozen feature schema is invalid",
    ))
    tree_roots = _static_stack_boost_split_ints(
        _static_stack_consistent_column(df, :tree_roots), "tree_roots",
    )
    node_count = _static_stack_csv_int(
        _static_stack_consistent_column(df, :node_count), "node_count",
    )
    tree_count = _static_stack_csv_int(
        _static_stack_consistent_column(df, :tree_count), "tree_count",
    )
    node_count == nrow(df) || throw(ArgumentError(
        "boost artifact node_count does not match its rows",
    ))
    tree_count == length(tree_roots) || throw(ArgumentError(
        "boost artifact tree_count does not match tree_roots",
    ))
    [_static_stack_csv_int(df[row, :node_index], "node_index")
     for row in 1:nrow(df)] == collect(1:nrow(df)) || throw(ArgumentError(
        "boost artifact node indices must be sequential",
    ))

    artifact = StaticStackBoostArtifact(
        string(_static_stack_consistent_column(df, :label)),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :model_step_hours),
            "model_step_hours",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :anchor_lag_hours),
            "anchor_lag_hours",
        ),
        feature_names,
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :rho), "rho",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :correction_cap_nt),
            "correction_cap_nt",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :fit_rows), "fit_rows",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :seed), "seed",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :nrounds), "nrounds",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :max_depth), "max_depth",
        ),
        _static_stack_csv_int(
            _static_stack_consistent_column(df, :nbins), "nbins",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :eta), "eta",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :l2_regularization),
            "l2_regularization",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :lambda), "lambda",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :gamma), "gamma",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :min_weight), "min_weight",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :rowsample), "rowsample",
        ),
        _static_stack_boost_float(
            _static_stack_consistent_column(df, :colsample), "colsample",
        ),
        tree_roots,
        [_static_stack_csv_int(df[row, :tree_index], "tree_index")
         for row in 1:nrow(df)],
        [_static_stack_boost_bool(df[row, :is_leaf], "is_leaf")
         for row in 1:nrow(df)],
        [_static_stack_csv_int(df[row, :feature_index], "feature_index")
         for row in 1:nrow(df)],
        [_static_stack_boost_float(df[row, :threshold], "threshold")
         for row in 1:nrow(df)],
        [_static_stack_csv_int(df[row, :left_child], "left_child")
         for row in 1:nrow(df)],
        [_static_stack_csv_int(df[row, :right_child], "right_child")
         for row in 1:nrow(df)],
        Float32.([_static_stack_boost_float(df[row, :leaf_value], "leaf_value")
                  for row in 1:nrow(df)]),
        Float32.([_static_stack_boost_float(df[row, :split_gain], "split_gain")
                  for row in 1:nrow(df)]),
    )
    _static_stack_boost_sha256(artifact) == checksum || throw(ArgumentError(
        "boost artifact checksum mismatch",
    ))
    return artifact
end
