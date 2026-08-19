using Test
using SolarSINDy
using DataFrames
using CSV

const V22R = SolarSINDy
const V22R_FEATURES = collect(OPERATIONAL_V22_RESIDUAL_FEATURES[5:10])

function _v22r_hadamard_features(n::Int, p::Int)
    ispow2(n) || error("Hadamard fixture length must be a power of two")
    p <= trailing_zeros(n) || error("Hadamard fixture needs more binary columns")
    out = Matrix{Float64}(undef, n, p)
    for row in 0:(n - 1), column in 1:p
        out[row + 1, column] = iszero((row >> (column - 1)) & 1) ? -1.0 : 1.0
    end
    return out
end

function _v22r_frame(features::Matrix{Float64}; target_shift::Float64=0.0)
    n, p = size(features)
    p == length(V22R_FEATURES) || error("fixture feature count mismatch")
    frame = DataFrame()
    for (j, name) in enumerate(V22R_FEATURES)
        frame[!, name] = features[:, j]
    end
    sample_scale = sqrt(n / (n - 1))
    residual = 2.0 .+ 3.0 .* features[:, 1] ./ sample_scale .-
               2.0 .* features[:, 2] ./ sample_scale .+ target_shift
    frame[!, :v2_2_pred_dst_nt] = fill(-20.0, n)
    frame[!, :observation_dst_nt] = frame.v2_2_pred_dst_nt .+ residual
    frame[!, :model_step_hours] = fill(1, n)
    frame[!, :latest_dst_nt] = vcat(fill(-40.0, 40), fill(-40.0, 40), fill(-10.0, n - 80))
    frame[!, :coupling_active_mvm] = zeros(n)
    frame[!, :dst_delta_1h_nt] = vcat(fill(-1.0, 40), fill(1.0, 40), zeros(n - 80))
    return frame
end

function _v22r_cell(lead::Int, candidate_features::Vector{Symbol};
                    support::Vector{Symbol}=candidate_features[1:2],
                    ranking::Vector{Symbol}=candidate_features,
                    coefficients::Vector{Float64}=[0.0, 0.0, 0.0])
    return OperationalV22ResidualCell(
        lead, support, ranking, zeros(length(support)), ones(length(support)),
        coefficients;
        ridge=1.0, top_k=length(support), fit_rows=64, validation_rows=32,
        validation_base_rmse_nt=2.0, validation_rmse_nt=1.0,
    )
end

function _v22r_core(cells, candidate_features)
    return OperationalV22ResidualCore(
        cells; label="v2.2-residual-test",
        candidate_feature_names=candidate_features,
        ridge_grid=(1.0,), top_k_grid=(2,),
    )
end

@testset verbose=true "Operational V2.2 secondary residual" begin
    @testset "Closed-form support and coefficient recovery" begin
        n = 128
        features = _v22r_hadamard_features(n, length(V22R_FEATURES))
        fit = _v22r_frame(features)
        validation = _v22r_frame(reverse(features; dims=1))
        core = fit_operational_v22_residual(
            fit, validation; feature_names=V22R_FEATURES,
        )
        cell = only(core.cells)
        shrink = (n - 1) / n # standardized X'X=n-1 and selected ridge λ=1

        @test cell.feature_names == Tuple(V22R_FEATURES[1:2])
        @test Set(cell.ranked_feature_names[1:2]) == Set(V22R_FEATURES[1:2])
        @test cell.ridge == 1.0
        @test cell.top_k == 2
        # The intercept is unpenalized; orthogonal standardized slopes have the
        # closed-form ridge multiplier (n-1)/(n-1+1).
        @test isapprox(cell.coefficients[1], 2.0; rtol=0.0, atol=2e-14)
        @test isapprox(cell.coefficients[2], 3.0 * shrink; rtol=0.0, atol=2e-13)
        @test isapprox(cell.coefficients[3], -2.0 * shrink; rtol=0.0, atol=2e-13)
        @test isapprox(collect(cell.feature_mean), zeros(2); rtol=0.0, atol=1e-15)
        @test isapprox(
            collect(cell.feature_scale), fill(sqrt(n / (n - 1)), 2);
            rtol=0.0, atol=2e-15,
        )
        @test cell.validation_rmse_nt < cell.validation_base_rmse_nt
        @test cell.validation_active_rmse_nt <= cell.validation_active_base_rmse_nt
        @test cell.validation_recovery_rmse_nt <= cell.validation_recovery_base_rmse_nt
        @test !ismutabletype(OperationalV22ResidualCell)
        @test !ismutabletype(OperationalV22ResidualCore)
    end

    @testset "Validation mutation cannot enter fit scaling or ranking" begin
        features = _v22r_hadamard_features(128, length(V22R_FEATURES))
        fit = _v22r_frame(features)
        validation_a = _v22r_frame(reverse(features; dims=1))
        validation_b = _v22r_frame(reverse(features; dims=1); target_shift=0.25)
        core_a = fit_operational_v22_residual(
            fit, validation_a; feature_names=V22R_FEATURES,
        )
        core_b = fit_operational_v22_residual(
            fit, validation_b; feature_names=V22R_FEATURES,
        )
        cell_a = only(core_a.cells)
        cell_b = only(core_b.cells)

        @test cell_a.validation_rmse_nt != cell_b.validation_rmse_nt
        @test cell_a.validation_base_rmse_nt != cell_b.validation_base_rmse_nt
        @test cell_a.feature_mean == cell_b.feature_mean
        @test cell_a.feature_scale == cell_b.feature_scale
        @test cell_a.ranked_feature_names == cell_b.ranked_feature_names
        @test cell_a.feature_names == cell_b.feature_names
        @test cell_a.coefficients == cell_b.coefficients
        @test_throws ArgumentError fit_operational_v22_residual(
            fit, validation_a; feature_names=[:observation_dst_nt, V22R_FEATURES[2:end]...],
        )
    end

    @testset "Exact cap, unsupported lead, and primary identity" begin
        candidate = V22R_FEATURES[1:2]
        positive = _v22r_cell(
            1, candidate; coefficients=[20.0, 0.0, 0.0],
        )
        positive_core = _v22r_core([positive], candidate)
        feature_values = NamedTuple{Tuple(candidate)}((1.0, -1.0))
        high = operational_v22_residual_predict(
            positive_core, 1, -30.0, feature_values,
        )
        @test high.raw_correction_nt == 20.0
        @test high.correction_nt == 10.0
        @test high.pred_dst == -20.0
        @test high.correction_was_capped
        @test high.correction_cap_nt == 5.0 + 5.0 * 1

        negative = _v22r_cell(
            2, candidate; coefficients=[-30.0, 0.0, 0.0],
        )
        negative_core = _v22r_core([negative], candidate)
        low = operational_v22_residual_predict(
            negative_core, 2, -30.0, feature_values,
        )
        @test low.raw_correction_nt == -30.0
        @test low.correction_nt == -15.0
        @test low.pred_dst == -45.0
        @test_throws ArgumentError operational_v22_residual_predict(
            positive_core, 2, -30.0, feature_values,
        )

        identity = _v22r_cell(1, candidate; coefficients=[0.0, 0.0, 0.0])
        identity_core = _v22r_core([identity], candidate)
        unchanged = operational_v22_residual_predict(
            identity_core, 1, -37.25, feature_values,
        )
        @test unchanged.raw_correction_nt == 0.0
        @test unchanged.correction_nt == 0.0
        @test unchanged.pred_dst == -37.25
        @test_throws ArgumentError OperationalV22ResidualCell(
            1, candidate, candidate, zeros(2), ones(2), [0.0, 0.0, 0.0];
            ridge=1.0, correction_cap_nt=10.1, fit_rows=64, validation_rows=32,
            validation_base_rmse_nt=2.0, validation_rmse_nt=1.0,
        )
    end

    @testset "Lead-specific logging and target/post-issue invariance" begin
        candidates = V22R_FEATURES[1:4]
        lead1 = _v22r_cell(
            1, candidates; support=candidates[1:2],
            ranking=[candidates[2], candidates[1], candidates[3], candidates[4]],
            coefficients=[1.0, 2.0, -1.0],
        )
        lead2 = _v22r_cell(
            2, candidates; support=candidates[3:4],
            ranking=[candidates[4], candidates[3], candidates[1], candidates[2]],
            coefficients=[-1.0, 0.5, 3.0],
        )
        core = _v22r_core([lead1, lead2], candidates)
        frame = DataFrame(
            model_step_hours=[1, 2], v2_2_pred_dst_nt=[-20.0, -30.0],
            observation_dst_nt=[-19.0, -28.0], post_issue_driver=[1.0, 2.0],
        )
        for (j, name) in enumerate(candidates)
            frame[!, name] = [Float64(j), Float64(j + 1)]
        end
        mutated = copy(frame)
        mutated.observation_dst_nt .= [900.0, -900.0]
        mutated.post_issue_driver .= [Inf, -Inf]
        scored = score_operational_v22_residual(frame, core)
        rescored = score_operational_v22_residual(mutated, core)
        forecast_columns = [
            :v2_2_secondary_pred_dst_nt,
            :v2_2_secondary_raw_correction_nt,
            :v2_2_secondary_correction_nt,
            :v2_2_secondary_correction_was_capped,
            :v2_2_secondary_correction_cap_nt,
            :v2_2_secondary_ridge,
            :v2_2_secondary_top_k,
            :v2_2_secondary_feature_names,
            :v2_2_secondary_coefficients,
            :v2_2_secondary_feature_contributions_nt,
            :v2_2_secondary_label,
        ]
        @test scored[:, forecast_columns] == rescored[:, forecast_columns]
        @test scored.v2_2_secondary_residual_dst_nt !=
              rescored.v2_2_secondary_residual_dst_nt
        @test scored.v2_2_secondary_feature_names == [
            join(String.(candidates[1:2]), ";"),
            join(String.(candidates[3:4]), ";"),
        ]
        @test scored.v2_2_secondary_coefficients == ["1.0;2.0;-1.0", "-1.0;0.5;3.0"]
    end

    @testset "Strict atomic artifact round trip and corruption rejection" begin
        candidates = V22R_FEATURES[1:2]
        core = _v22r_core([
            _v22r_cell(1, candidates; coefficients=[1.0, 2.0, -1.0]),
        ], candidates)
        values = NamedTuple{Tuple(candidates)}((4.0, 3.0))
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "v22-residual.csv")
            @test write_operational_v22_residual(path, core) == path
            restored = read_operational_v22_residual(path)
            @test restored.label == core.label
            @test restored.candidate_feature_names == core.candidate_feature_names
            @test restored.ridge_grid == core.ridge_grid
            @test restored.top_k_grid == core.top_k_grid
            @test restored.supported_model_steps == core.supported_model_steps
            @test operational_v22_residual_predict(restored, 1, -20.0, values) ==
                  operational_v22_residual_predict(core, 1, -20.0, values)

            valid = CSV.read(path, DataFrame)
            bad_cap = copy(valid)
            bad_cap.correction_cap_nt[1] = 11.0
            bad_cap_path = joinpath(tmp, "bad-cap.csv")
            CSV.write(bad_cap_path, bad_cap)
            @test_throws ArgumentError read_operational_v22_residual(bad_cap_path)

            bad_coefficients = copy(valid)
            bad_coefficients[!, :coefficients] = String.(bad_coefficients.coefficients)
            bad_coefficients.coefficients[1] *= ";9.0"
            bad_coefficients_path = joinpath(tmp, "bad-coefficients.csv")
            CSV.write(bad_coefficients_path, bad_coefficients)
            @test_throws DimensionMismatch read_operational_v22_residual(
                bad_coefficients_path,
            )

            duplicate_path = joinpath(tmp, "duplicate.csv")
            CSV.write(duplicate_path, vcat(valid, valid))
            @test_throws ArgumentError read_operational_v22_residual(duplicate_path)

            wrong_schema_path = joinpath(tmp, "wrong-schema.csv")
            CSV.write(wrong_schema_path, select(valid, Not(:validation_rmse_nt)))
            @test_throws ArgumentError read_operational_v22_residual(wrong_schema_path)

            link = joinpath(tmp, "residual-link.csv")
            symlink(path, link)
            @test_throws ArgumentError read_operational_v22_residual(link)
            @test_throws ArgumentError write_operational_v22_residual(link, core)

            # S3: the file's own supported-lead metadata was parsed and then discarded, so a core
            # whose metadata claimed one lead set while its cells carried another loaded silently.
            wrong_steps = copy(valid)
            wrong_steps[!, :supported_model_steps] = string.(wrong_steps.supported_model_steps)
            wrong_steps.supported_model_steps .= "1;2;3;4;6;7"
            wrong_steps_path = joinpath(tmp, "wrong-steps.csv")
            CSV.write(wrong_steps_path, wrong_steps)
            @test_throws ArgumentError read_operational_v22_residual(wrong_steps_path)

            # A metadata list that merely repeats or reorders the same leads still loads: the check
            # compares the lead SET the cells carry, not the literal text.
            reordered = copy(valid)
            reordered[!, :supported_model_steps] = string.(reordered.supported_model_steps)
            declared = sort(unique(Int.(valid.model_step_hours)))
            reordered.supported_model_steps .=
                join(vcat(reverse(declared), declared[1:1]), ";")
            reordered_path = joinpath(tmp, "reordered-steps.csv")
            CSV.write(reordered_path, reordered)
            @test read_operational_v22_residual(reordered_path).cells == restored.cells
        end
    end

    @testset "S2: a numeric-looking residual label round-trips as text" begin
        candidates = V22R_FEATURES[1:2]
        cells = [_v22r_cell(1, candidates; coefficients=[1.0, 2.0, -1.0])]
        for label in ("007", "2026")
            relabelled = OperationalV22ResidualCore(
                cells; label=label, candidate_feature_names=candidates,
                ridge_grid=(1.0,), top_k_grid=(2,),
            )
            mktempdir() do tmp
                path = joinpath(tmp, "residual.csv")
                write_operational_v22_residual(path, relabelled)
                restored = read_operational_v22_residual(path)
                @test restored.label == label
                @test restored.label isa String
                @test restored.cells == relabelled.cells
            end
        end
    end
end
