using DataFrames
using Dates
using SolarSINDy
using Test

isdefined(@__MODULE__, :run_static_stack_error_boost_omni_diagnostic) || include(joinpath(
    @__DIR__, "..", "validation", "operational",
    "static_stack_error_boost_omni_diagnostic.jl",
))

function _err_boost_test_table()
    rows = NamedTuple[]
    folds = Tuple(fold.name for fold in STATIC_STACK_DRIVER_OMNI_FOLDS)
    for (fold_index, fold) in pairs(folds), index in 1:120
        issue = DateTime(2012 + fold_index, 1, 8) + Hour(index - 1)
        values = ntuple(feature ->
            feature == length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES) ?
            0.0 : 0.01 * feature + 0.001 * index,
            length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES))
        named = NamedTuple{STATIC_STACK_ERROR_EXOGENOUS_FEATURES}(values)
        push!(rows, merge((
            fold=fold,
            issue_time_utc=issue,
            target_time_utc=issue + Hour(1),
            model_step_hours=1,
            base_center_sha256=repeat("a", 64),
            base_prediction_dst_nt=0.0,
            observation_dst_nt=2.0,
        ), named))
    end
    return DataFrame(rows)
end

@testset "static stack-M3 lag-feature tree OMNI diagnostic" begin
    @testset "frozen configuration" begin
        config = _err_boost_config(3; nrounds=64)
        @test config.max_depth == 3
        @test config.nrounds == 64
        @test config.eta == 0.03
        @test config.min_weight == 96.0
        @test config.nbins == 64
        @test config.seed == STATIC_STACK_ERROR_BOOST_SEED
        @test_throws ArgumentError _err_boost_config(2)
        @test_throws ArgumentError _err_boost_config(3; nrounds=32)
    end

    @testset "chronological selection and embargo mutation" begin
        table = _err_boost_test_table()
        result = static_stack_error_boost_select(table, 1)
        @test nrow(result.audit) ==
              length(STATIC_STACK_ERROR_BOOST_DEPTHS) *
              length(STATIC_STACK_ERROR_BOOST_ROUNDS) *
              length(STATIC_STACK_ERROR_BOOST_SHRINKAGES)
        @test result.selected.depth in STATIC_STACK_ERROR_BOOST_DEPTHS
        @test result.selected.rounds in STATIC_STACK_ERROR_BOOST_ROUNDS
        @test result.selected.shrinkage in STATIC_STACK_ERROR_BOOST_SHRINKAGES
        @test result.selected.reason == "selected_regime_safe_oof_minimum"
        @test only(result.audit.gain_nt[result.audit.selected]) > 0.0

        mutated = copy(table)
        row = findlast(mutated.fold .== "calendar_2013")
        mutated.target_time_utc[row] = DateTime(2014, 1, 1, 1)
        @test_throws ArgumentError static_stack_error_boost_select(mutated, 1)
    end

    @testset "regime safety and final-fit chronology" begin
        losses = DataFrame(
            regime=["overall", "target_storm", "active_or_deepening",
                    "recovery"],
            rows=[100, 20, 20, 20],
            gain_nt=[1.0, 0.1, 0.1, -0.1],
        )
        @test !_err_boost_regime_safe(losses)
        losses.rows[4] = 19
        @test _err_boost_regime_safe(losses)

        audit = DataFrame(
            depth=[3, 4, 3],
            rounds=[64, 64, 128],
            shrinkage=[1.0, 1.0, 0.5],
            candidate_rmse_nt=[0.5, 0.8, 0.9],
            gain_nt=[1.0, 0.8, 0.7],
            regime_safe=[false, true, true],
        )
        @test _err_boost_choose_index(audit) == 2
        audit.gain_nt[2:3] .= 0.0
        @test isnothing(_err_boost_choose_index(audit))

        oof = _err_boost_test_table()
        first_feature = first(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)
        oof.observation_dst_nt = ifelse.(
            Float64.(oof[!, first_feature]) .< 0.07, -4.0, 4.0,
        )
        ready = copy(oof[1:3, :])
        ready.issue_time_utc = DateTime(2018, 1, 1) .+ Hour.(0:2)
        ready.target_time_utc = ready.issue_time_utc .+ Hour(1)
        selected = (
            model_step_hours=1, depth=3, rounds=64, shrinkage=1.0,
            regime_safe=true, reason="test",
        )
        full = _err_boost_final_predictions(oof, ready, selected)
        half = _err_boost_final_predictions(
            oof, ready, merge(selected, (shrinkage=0.5,)),
        )
        longer = _err_boost_final_predictions(
            oof, ready, merge(selected, (rounds=128,)),
        )
        for row in eachrow(ready)
            key = (DateTime(row.issue_time_utc), 1)
            base = Float64(row.base_prediction_dst_nt)
            @test half[key] - base ≈ 0.5 * (full[key] - base)
        end
        @test any(abs(longer[key] - full[key]) > 1e-8
                  for key in keys(full))

        poison = copy(oof[end:end, :])
        poison.issue_time_utc[1] = STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP - Hour(6)
        poison.target_time_utc[1] = poison.issue_time_utc[1] + Hour(1)
        poison.observation_dst_nt[1] = 1.0e6
        poisoned_high = vcat(oof, poison; cols=:setequal)
        poison.observation_dst_nt[1] = -1.0e6
        poisoned_low = vcat(oof, poison; cols=:setequal)
        @test _err_boost_final_predictions(
            poisoned_high, ready, selected,
        ) == _err_boost_final_predictions(
            poisoned_low, ready, selected,
        )

        fallback = _err_boost_final_predictions(
            oof, ready, merge(selected, (
                depth=0, rounds=0, shrinkage=0.0,
            )),
        )
        @test all(fallback[(DateTime(row.issue_time_utc), 1)] ==
                  row.base_prediction_dst_nt for row in eachrow(ready))
    end

    @testset "loss masks and common-row fallback" begin
        loss_table = DataFrame(
            latest_dst_nt=[0.0, -40.0, -40.0],
            dst_delta_1h_nt=[0.0, -2.0, 2.0],
            observation_dst_nt=[0.0, -60.0, -20.0],
            base_prediction_dst_nt=[2.0, -55.0, -25.0],
        )
        losses = _err_boost_losses(
            loss_table, [1.0, -58.0, -22.0],
        )
        @test only(losses.rows[losses.regime .== "target_storm"]) == 1
        @test only(losses.rows[losses.regime .== "active_or_deepening"]) == 1
        @test only(losses.rows[losses.regime .== "recovery"]) == 1

        rows = NamedTuple[]
        issues = (DateTime(2018, 1, 1), DateTime(2018, 2, 1))
        methods = ("m3_sparse_error_state", STATIC_STACK_ERROR_OMNI_COMPARATORS...)
        for method in methods, lead in STATIC_STACK_DRIVER_OMNI_LEADS_H, issue in issues
            push!(rows, (
                issue_time_utc=issue,
                target_time_utc=issue + Hour(lead),
                model_step_hours=lead,
                method=method,
                comparison_role="available_comparator",
                candidate_ready=false,
                fallback_reason="missing_contiguous_24h_history",
                split_label="validation",
                predicted_dst_nt=3.0,
                observed_dst_nt=0.0,
                squared_error_nt2=9.0,
                absolute_error_nt=3.0,
                raw_candidate_correction_nt=0.0,
                applied_candidate_correction_nt=0.0,
                latest_dst_nt=0.0,
                dst_delta_1h_nt=0.0,
                target_storm=false,
                active_or_deepening=false,
                recovery=false,
            ))
        end
        predictions = Dict(
            (DateTime(2018, 1, 1), 1) => 1.0,
        )
        errors = static_stack_error_boost_score(DataFrame(rows), predictions)
        candidate = errors[errors.method .== STATIC_STACK_ERROR_BOOST_CANDIDATE, :]
        @test nrow(candidate) == length(issues) *
                                 length(STATIC_STACK_DRIVER_OMNI_LEADS_H)
        @test count(candidate.candidate_ready) == 1
        ready = candidate[(candidate.issue_time_utc .== DateTime(2018, 1, 1)) .&
                          (candidate.model_step_hours .== 1), :]
        @test only(ready.predicted_dst_nt) == 1.0
        fallback = candidate[(candidate.issue_time_utc .== DateTime(2018, 2, 1)) .&
                             (candidate.model_step_hours .== 1), :]
        @test only(fallback.predicted_dst_nt) == 3.0
        @test only(fallback.fallback_reason) ==
              "missing_contiguous_24h_history"
    end
end
