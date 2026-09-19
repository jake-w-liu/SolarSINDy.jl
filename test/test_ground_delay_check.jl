module GroundDelayCheckTests
using Test, Dates, Statistics, JSON3
include(joinpath(@__DIR__, "../validation/operational/ground_delay_check.jl"))
using .GroundDelayCheck

@testset "Ground forecast receipt-time delay" begin
    model = (beta=[log(6.),0,0,0,0],mu=zeros(4),sigma=ones(4),
        calibration_seed=ones(1440),climatology_native=3.,climatology_log=2.)
    times = collect(DateTime(2020):Minute(1):DateTime(2020)+Minute(159))
    derivative = Float64.(1:length(times))
    @test GroundDelayCheck.model_point(model,fill(4.,30)) ≈ 5. atol=1e-14
    @test_throws ArgumentError GroundDelayCheck.model_point(model,ones(29))
    @test_throws ArgumentError GroundDelayCheck.model_point(model,fill(NaN,30))
    @test_throws ArgumentError GroundDelayCheck.model_point(model,fill(-1.,30))
    @test_throws ArgumentError GroundDelayCheck.model_point(model,fill(true,30))
    @test_throws ArgumentError GroundDelayCheck.model_point(merge(model,(;sigma=zeros(4))),ones(30))
    @test_throws ArgumentError GroundDelayCheck.evaluate_delay(times,derivative,model,3)
    @test_throws ArgumentError GroundDelayCheck.evaluate_delay(reverse(times),derivative,model,0)
    for lag in (0,5,10)
        frame = GroundDelayCheck.evaluate_delay(times,derivative,model,lag)
        @test length(frame.anchor) == 100-lag
        @test frame.anchor[1] == times[31]
        @test frame.target_start[1] == times[31]+Minute(lag)
        @test frame.target_end[1] == times[31]+Minute(lag+30)
        @test frame.target[1] == 61+lag
        @test frame.past[1] == 31
        @test frame.upper[1] ≈ 37. atol=1e-13
        changed = copy(derivative); changed[61+lag] = 10000
        second = GroundDelayCheck.evaluate_delay(times,changed,model,lag)
        @test second.prediction[1] == frame.prediction[1]
        @test second.upper[1] == frame.upper[1]
        @test second.target[1] == 10000
        missing = copy(derivative); missing[31] = NaN
        sparse = GroundDelayCheck.evaluate_delay(times,missing,model,lag)
        @test !(times[31] in sparse.anchor)
        # Independent explicit receipt chronology and finite-sample rank.
        for k in eachindex(frame.anchor)
            eligible = [j for j in 1:k-1 if
                frame.target_end[j]+Minute(lag) < frame.anchor[k]+Minute(lag)]
            history = vcat(ones(1440),[(frame.target[j]-5)/(1+frame.past[j]) for j in eligible])
            history = history[max(1,end-1439):end]
            expected = 5 + sort(history)[min(end,cld(9*(length(history)+1),10))]*(1+frame.past[k])
            @test frame.upper[k] ≈ expected atol=1e-12 rtol=1e-14
        end
        score = GroundDelayCheck.metrics(frame,model)
        @test score.n == length(frame.target)
        @test score.point_rmse ≈ sqrt(sum((x-5)^2 for x in frame.target)/score.n) atol=1e-12
        @test score.coverage == count(frame.target .<= frame.upper)/score.n
    end
    gap_times = copy(times); gap_times[90:end] .+= Minute(1)
    gapped = GroundDelayCheck.evaluate_delay(gap_times,derivative,model,0)
    @test !(times[80] in gapped.anchor)
    jittered = copy(times); jittered[40] += Second(30)
    jitter_frame = GroundDelayCheck.evaluate_delay(jittered,derivative,model,0)
    @test !(times[50] in jitter_frame.anchor)
    @test_throws ArgumentError GroundDelayCheck.evaluate_delay(times[1:40],derivative[1:40],model,10)
    frozen = JSON3.read(read(GroundDelayCheck.MODEL_PATH))
    w = Float64.(1:30)
    features = BigFloat[last(w),sum(w)/30,maximum(w),sqrt(sum((w.-sum(w)/30).^2)/29)]
    expected = max(0,Float64(expm1(BigFloat(frozen.beta[1]) + sum(BigFloat(frozen.beta[k+1]) *
        (log1p(features[k])-BigFloat(frozen.mu[k]))/BigFloat(frozen.sigma[k]) for k in 1:4))))
    @test GroundDelayCheck.model_point(frozen,w) ≈ expected atol=1e-10 rtol=1e-12
    @test frozen.serving_enabled == false
    @test frozen.operational_alerts == false
end
end
