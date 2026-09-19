using CSV, DataFrames, Dates, JSON3, Statistics, Test, SHA

# This checker uses the previously independently reconstructed zero-delay panel,
# not the new evaluator's feature, target, quantile, or metric functions.
function check_ground_delay_output(archive, output)
    model = JSON3.read(read(joinpath(@__DIR__, "../../deploy/cmo_ground_shadow/model.json")))
    summary = CSV.read(joinpath(output,"metrics.csv"),DataFrame)
    @test nrow(summary) == 12
    @testset "Independent delayed-target and receipt reconstruction" begin
        training_path = normpath(joinpath(@__DIR__,"../../../paper_dbdt_alerts/data/dbdt_CMO.csv"))
        @test bytes2hex(sha256(read(training_path))) == model.dataset_sha256
        dataset = CSV.read(training_path,DataFrame)
        values = dataset.target_max_dbdt[(year.(dataset.datetime).==2017) .&
            (dataset.datetime .< DateTime(2017,10,1)-Minute(30))]
        @test model.climatology_native ≈ Float64(sum(BigFloat.(values))/length(values)) atol=1e-12 rtol=1e-13
        @test model.climatology_log ≈ Float64(expm1(sum(log1p.(BigFloat.(values)))/length(values))) atol=1e-12 rtol=1e-13
        for period in (2018,2021,2024,2026)
            base = CSV.read(joinpath(archive,"additional_storm_check","CMO_adjusted_$(period)_predictions.csv"),DataFrame)
            lookup = Dict(t=>i for (i,t) in enumerate(base.datetime))
            for lag in (0,5,10)
                frame = CSV.read(joinpath(output,"CMO_adjusted_$(period)_delay$(lag).csv"),DataFrame)
                @test issorted(frame.anchor) && allunique(frame.anchor)
                @test all(frame.target_start .== frame.anchor .+ Minute(lag))
                @test all(frame.target_end .== frame.anchor .+ Minute(lag+30))
                for row in eachrow(frame)
                    @test row.prediction ≈ base.prediction[lookup[row.anchor]] atol=1e-10 rtol=1e-12
                    @test row.past == base.past[lookup[row.anchor]]
                    @test row.target == base.target[lookup[row.target_start]]
                end
                scores = (frame.target .- frame.prediction)./(1 .+ frame.past)
                # Explicit receipt comparisons at every boundary; retain no more than
                # the last 1,440 values after the most recent collection gap.
                epoch = 1; previous_stop = 0; history = Float64.(model.calibration_seed)
                for k in eachindex(frame.anchor)
                    if k>1 && frame.anchor[k]-frame.anchor[k-1]>Day(1)
                        epoch=k; previous_stop=k-1; history=Float64.(model.calibration_seed)
                    end
                    while previous_stop+1 < k &&
                          frame.target_end[previous_stop+1]+Minute(lag) < frame.anchor[k]+Minute(lag)
                        previous_stop += 1
                        previous_stop >= epoch && push!(history,scores[previous_stop])
                    end
                    retained = history[max(1,length(history)-1439):end]
                    rank = min(length(retained),cld(9*(length(retained)+1),10))
                    expected = max(0,frame.prediction[k]+sort(retained)[rank]*(1+frame.past[k]))
                    @test frame.upper[k] ≈ expected atol=1e-10 rtol=1e-12
                end
                selected = summary[(summary.period.==period).&(summary.delay_minutes.==lag),:]
                @test nrow(selected) == 1
                row = only(eachrow(selected)); n=nrow(frame)
                @test row.n == n
                @test row.point_rmse ≈ Float64(sqrt(sum((BigFloat(p)-BigFloat(y))^2 for
                    (p,y) in zip(frame.prediction,frame.target))/n)) atol=1e-11 rtol=1e-13
                @test row.log_rmse ≈ sqrt(sum((log1p(p)-log1p(y))^2 for
                    (p,y) in zip(frame.prediction,frame.target))/n) atol=1e-13 rtol=1e-13
                @test row.coverage == count(frame.target .<= frame.upper)/n
                controls = (frame.past, fill(Float64(model.climatology_native),n),
                            fill(Float64(model.climatology_log),n))
                @test row.baseline_rmse ≈ minimum(sqrt(sum((p-y)^2 for (p,y) in zip(c,frame.target))/n)
                    for c in controls) atol=1e-11 rtol=1e-13
                @test row.baseline_log_rmse ≈ minimum(sqrt(sum((log1p(p)-log1p(y))^2 for
                    (p,y) in zip(c,frame.target))/n) for c in controls) atol=1e-13 rtol=1e-13
                @test row.native_pass == (row.point_rmse<=row.baseline_rmse)
                @test row.log_pass == (row.log_rmse<=row.baseline_log_rmse)
                @test row.coverage_pass == (.88<=row.coverage<=.92)
            end
        end
        decision = JSON3.read(read(joinpath(output,"decision.json")))
        @test decision.receipt_time_screen_pass == all(summary.native_pass .& summary.log_pass .& summary.coverage_pass)
        @test decision.serving_enabled == false
        @test decision.prospective_qualification == false
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("Usage: check_ground_delay_output.jl REMEDIATION_ARCHIVE DELAY_OUTPUT")
    check_ground_delay_output(ARGS...)
end
