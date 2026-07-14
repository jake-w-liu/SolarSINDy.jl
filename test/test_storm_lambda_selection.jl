using CSV, DataFrames

@testset "Storm-level lambda selection" begin
    grid = storm_lambda_grid()
    @test length(grid) == 60
    @test grid[1] ≈ 1.0e-2
    @test grid[end] ≈ 1.0e4
    @test all(isapprox.(diff(log10.(grid)), 6 / 59; atol=2e-15))

    storms = [(
        storm_id = i,
        onset_time = i,
        observed = i >= 9 ? [NaN, 0.0, 0.0, 0.0] : zeros(4),
    ) for i in reverse(1:10)]
    fit_calls = Vector{Vector{Int}}()
    score_calls = Int[]
    function separation_fit(subset, lambda)
        push!(fit_calls, sort([storm.storm_id for storm in subset]))
        index = findfirst(==(lambda), grid)
        return (model=(candidate_index=index,), support=["term_$index"])
    end
    function separation_integrate(model, storm, anchor, anchor_value)
        push!(score_calls, storm.storm_id)
        @test anchor == 2
        error = abs(model.candidate_index - 15) + 1.0
        delta = error
        return [anchor_value, storm.observed[3] + delta, storm.observed[4] + delta]
    end
    separation = select_storm_lambda(storms;
        fit=separation_fit,
        integrate=separation_integrate,
        observations=storm -> storm.observed,
    )

    @test separation.selected_lambda == grid[15]
    @test count(row -> row.inner_split == "train", separation.split_records) == 8
    @test count(row -> row.inner_split == "validation", separation.split_records) == 2
    @test [row.storm_id for row in separation.split_records] == string.(1:10)
    @test all(call == collect(1:8) for call in fit_calls[1:60])
    @test fit_calls[61] == collect(1:10)
    @test Set(score_calls) == Set([9, 10])
    @test !(99 in score_calls) # an untouched outer storm cannot enter this API
    @test all(row.anchor_index == 2 for row in separation.error_records)

    weighted_storms = [(
        storm_id = i,
        onset_time = i,
        observed = i == 9 ? zeros(3) : (i == 10 ? zeros(103) : zeros(3)),
    ) for i in 1:10]
    weighted_fit(subset, lambda) = (
        model=(candidate_index=findfirst(==(lambda), grid),),
        support=["a", "b"],
    )
    function weighted_integrate(model, storm, anchor, anchor_value)
        index = model.candidate_index
        target_rmse = index == 10 ? (storm.storm_id == 9 ? 1.0 : 9.0) :
                      index == 20 ? 6.0 : 20.0
        n = length(storm.observed)
        delta = target_rmse
        return [anchor_value; fill(delta, n - 1)]
    end
    weighted = select_storm_lambda(weighted_storms;
        fit=weighted_fit,
        integrate=weighted_integrate,
        observations=storm -> storm.observed,
    )
    candidate10 = weighted.candidate_records[10]
    @test candidate10.mean_storm_rmse ≈ 5.0
    @test candidate10.standard_error ≈ 4.0
    point_weighted_rmse = sqrt((3 * 1.0^2 + 103 * 9.0^2) / 106)
    @test !isapprox(candidate10.mean_storm_rmse, point_weighted_rmse)
    @test weighted.candidate_records[20].mean_storm_rmse ≈ 6.0
    @test weighted.candidate_records[20].eligible
    @test weighted.selected_lambda == grid[20]
    @test weighted.decision_record.one_standard_error_cutoff ≈ 9.0

    function extreme_integrate(model, storm, anchor, anchor_value)
        index = model.candidate_index
        target_rmse = index == 1 ? (storm.storm_id == 9 ? 0.0 : 1.0e200) :
                      index == 60 ? 1.1e200 : 2.0e200
        return [anchor_value; fill(target_rmse, length(storm.observed) - 1)]
    end
    extreme = select_storm_lambda(weighted_storms;
        fit=weighted_fit,
        integrate=extreme_integrate,
        observations=storm -> storm.observed,
    )
    @test extreme.candidate_records[1].mean_storm_rmse == 5.0e199
    @test extreme.candidate_records[1].standard_error == 5.0e199
    @test extreme.decision_record.one_standard_error_cutoff == 1.0e200
    @test !extreme.candidate_records[60].eligible
    @test extreme.selected_lambda == grid[1]

    direct_storm = (storm_id=99, onset_time=99, observed=[0.0, 0.0])
    stable_extreme = SolarSINDy._storm_validation_error(
        nothing, direct_storm,
        (_, _, _, anchor) -> [anchor, floatmax(Float64)],
        storm -> storm.observed,
        storm -> storm.storm_id,
        storm -> storm.onset_time,
    )
    @test stable_extreme.rmse == floatmax(Float64)
    @test_throws ArgumentError SolarSINDy._storm_validation_error(
        nothing, merge(direct_storm, (observed=[false, true],)),
        (_, _, _, anchor) -> [anchor, 0.0],
        storm -> storm.observed,
        storm -> storm.storm_id,
        storm -> storm.onset_time,
    )
    @test_throws ArgumentError SolarSINDy._storm_validation_error(
        nothing, merge(direct_storm, (observed=[0.0, Inf, 1.0],)),
        (_, storm, anchor, anchor_value) -> fill(anchor_value,
                                                  length(storm.observed) - anchor + 1),
        storm -> storm.observed,
        storm -> storm.storm_id,
        storm -> storm.onset_time,
    )
    @test_throws ArgumentError SolarSINDy._storm_validation_error(
        nothing, direct_storm,
        (_, _, _, anchor) -> Any[anchor, true],
        storm -> storm.observed,
        storm -> storm.storm_id,
        storm -> storm.onset_time,
    )

    repeated = select_storm_lambda(weighted_storms;
        fit=weighted_fit,
        integrate=weighted_integrate,
        observations=storm -> storm.observed,
    )
    @test repeated.selected_lambda == weighted.selected_lambda
    @test repeated.split_records == weighted.split_records
    @test repeated.candidate_records == weighted.candidate_records
    @test repeated.error_records == weighted.error_records
    @test repeated.support_records == weighted.support_records
    @test repeated.decision_record == weighted.decision_record

    mktempdir() do first_root
        mktempdir() do second_root
            first_paths = write_storm_lambda_selection(weighted, first_root; prefix="oracle")
            second_paths = write_storm_lambda_selection(repeated, second_root; prefix="oracle")
            for field in propertynames(first_paths)
                first_path = getproperty(first_paths, field)
                second_path = getproperty(second_paths, field)
                @test isfile(first_path)
                @test read(first_path) == read(second_path)
            end
            @test :mean_storm_rmse_nt in propertynames(
                CSV.read(first_paths.candidates, DataFrame))
            @test :standard_error_nt in propertynames(
                CSV.read(first_paths.candidates, DataFrame))
            @test :rmse_nt in propertynames(CSV.read(first_paths.errors, DataFrame))
            decision_frame = CSV.read(first_paths.decision, DataFrame)
            @test :minimum_mean_storm_rmse_nt in propertynames(decision_frame)
            @test :minimum_standard_error_nt in propertynames(decision_frame)
            @test :one_standard_error_cutoff_nt in propertynames(decision_frame)
            decision = CSV.read(first_paths.decision, DataFrame)
            @test decision.selected_lambda[1] == grid[20]
            @test decision.n_inner_training_storms[1] == 8
            @test decision.n_inner_validation_storms[1] == 2
        end
    end

    @test_throws ArgumentError select_storm_lambda(weighted_storms[1:1];
        fit=weighted_fit, integrate=weighted_integrate,
        observations=storm -> storm.observed)
    duplicate_ids = copy(weighted_storms)
    duplicate_ids[2] = merge(duplicate_ids[2], (storm_id=1,))
    @test_throws ArgumentError select_storm_lambda(duplicate_ids;
        fit=weighted_fit, integrate=weighted_integrate,
        observations=storm -> storm.observed)
    @test_throws ArgumentError select_storm_lambda(weighted_storms;
        fit=(subset, lambda) -> (model=nothing,),
        integrate=weighted_integrate,
        observations=storm -> storm.observed)
    @test_throws ArgumentError select_storm_lambda(weighted_storms;
        fit=weighted_fit,
        integrate=(model, storm, anchor, anchor_value) -> ones(length(storm.observed)),
        observations=storm -> storm.observed)
    mktempdir() do root
        @test_throws ArgumentError write_storm_lambda_selection(weighted, root;
                                                                prefix="../escape")
    end

    @testset "selection persistence is all-or-nothing" begin
        mktempdir() do root
            prefix = "rollback"
            split_path = joinpath(root, "$(prefix)_inner_split.csv")
            candidate_path = joinpath(root, "$(prefix)_candidates.csv")
            collision = joinpath(root, "$(prefix)_validation_errors.csv")
            write(split_path, "old split\n")
            write(candidate_path, "old candidates\n")
            mkdir(collision)
            keep = joinpath(collision, "keep")
            write(keep, "preserve")

            @test_throws ArgumentError write_storm_lambda_selection(
                weighted, root; prefix,
            )
            @test read(split_path, String) == "old split\n"
            @test read(candidate_path, String) == "old candidates\n"
            @test isdir(collision)
            @test read(keep, String) == "preserve"
            @test !ispath(joinpath(root, "$(prefix)_support.csv"))
            @test !ispath(joinpath(root, "$(prefix)_decision.csv"))
        end

        if !Sys.iswindows()
            mktempdir() do root
                prefix = "symlink"
                referent = joinpath(root, "referent.csv")
                write(referent, "old\n")
                target = joinpath(root, "$(prefix)_inner_split.csv")
                symlink(referent, target)
                @test_throws ArgumentError write_storm_lambda_selection(
                    weighted, root; prefix,
                )
                @test islink(target)
                @test read(referent, String) == "old\n"
            end
        end
    end


    @testset "selection readers never observe mixed generations" begin
        mktempdir() do root
            paths = SolarSINDy._storm_selection_paths(root, "concurrent")
            generation_frames(generation) = NamedTuple{propertynames(paths)}(
                Tuple(DataFrame(generation=[generation])
                      for _ in propertynames(paths)),
            )
            SolarSINDy._write_selection_csv_set(paths, generation_frames(1))

            partial_ready = Channel{Nothing}(1)
            release_writer = Channel{Nothing}(1)
            writer = @async SolarSINDy._with_selection_csv_set_lock(paths) do
                fields = propertynames(paths)
                next_frames = generation_frames(2)
                CSV.write(getproperty(paths, first(fields)),
                          getproperty(next_frames, first(fields)))
                put!(partial_ready, nothing)
                take!(release_writer)
                for field in fields[2:end]
                    CSV.write(getproperty(paths, field),
                              getproperty(next_frames, field))
                end
            end
            take!(partial_ready)
            reader = @async SolarSINDy._read_selection_csv_set(paths)
            sleep(0.05)
            @test !istaskdone(reader)
            put!(release_writer, nothing)
            wait(writer)
            observed = fetch(reader)
            @test all(only(getproperty(observed, field).generation) == 2
                      for field in propertynames(observed))

            public_read = read_storm_lambda_selection(root; prefix="concurrent")
            @test all(only(getproperty(public_read, field).generation) == 2
                      for field in propertynames(public_read))
            @test !ispath(paths.split * ".set.lock")
        end
    end
end
