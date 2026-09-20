module GroundAdaptiveCalibrationTests
using Test, DataFrames, Dates, Statistics
include(joinpath(@__DIR__,"../validation/operational/ground_adaptive_calibration.jl"))
const G = GroundAdaptiveCalibration

@testset "Delayed adaptive calibration has independent causal expectations" begin
    @test G.upper_estimate(5.,2.,collect(1.:20.),.1) == 43.
    @test G.upper_estimate(5.,2.,[-10.],.1) == 0.
    @test G.upper_estimate(5.,2.,[1.],0.) == Inf
    @test G.upper_estimate(5.,2.,[1.],1.) == -Inf
    @test_throws ArgumentError G.upper_estimate(5.,2.,[1.],NaN)
    @test_throws ArgumentError G.upper_estimate(5.,2.,[],.1)
    @test_throws ArgumentError G.upper_estimate(-1.,2.,[1.],.1)
    times = collect(DateTime(2020):Minute(1):DateTime(2020)+Minute(99))
    seed = collect(1.:1440.)/1440
    for delay in (0,5,10)
        frame = DataFrame(;anchor=times,target_start=times.+Minute(delay),
            target_end=times.+Minute(delay+30),prediction=fill(5.,100),
            past=fill(2.,100),target=collect(1.:100.),upper=fill(8.,100))
        out = G.recalibrate(frame,seed,delay)
        independent_upper = Float64[]
        for i in eachindex(times)
            eligible = [j for j in 1:i-1 if times[j]+Minute(30+2delay) < times[i]+Minute(delay)]
            misses = count(j -> frame.target[j]>independent_upper[j],eligible)
            a = .1 + (.1length(eligible)-misses)/6000
            scores = vcat(seed,[(frame.target[j]-5)/3 for j in eligible])
            scores = scores[max(1,length(scores)-1439):end]
            rank = min(length(scores),ceil(Int,(1-a)*(length(scores)+1)))
            bound = max(0,5+3sort(scores)[rank])
            push!(independent_upper,bound)
            @test out.received[i] == length(eligible)
            @test out.misses_received[i] == misses
            @test out.alpha[i] == a
            @test out.upper[i] == bound
        end
        @test out.received[31+delay] == 0
        @test out.received[32+delay] == 1
        @test frame.upper == fill(8.,100)
        # Flip an initially covered outcome; making an existing miss larger cannot
        # change the controller's binary feedback.
        changed = copy(frame); changed.target[1] += 1000
        mutated = G.recalibrate(changed,seed,delay)
        @test out.upper[1:31+delay] == mutated.upper[1:31+delay]
        @test out.alpha[32+delay:end] != mutated.alpha[32+delay:end]
        gapped = vcat(frame,transform(frame,:anchor=>(x->x.+Day(3))=>:anchor,
            :target_start=>(x->x.+Day(3))=>:target_start,
            :target_end=>(x->x.+Day(3))=>:target_end))
        gapout = G.recalibrate(gapped,seed,delay)
        @test gapout.upper[101:end] == out.upper
        @test gapout.alpha[101:end] == out.alpha
        @test_throws ArgumentError G.recalibrate(frame,ones(1439),delay)
        @test_throws ArgumentError G.recalibrate(frame[end:-1:1,:],seed,delay)
        wrong = copy(frame); wrong.target_end[1] += Minute(1)
        @test_throws ArgumentError G.recalibrate(wrong,seed,delay)
    end
    @test G.pinball(10.,5.) == 4.5
    @test G.pinball(5.,10.) == .5
    metrics = DataFrame([(;period,delay_minutes=delay,n=100,native_pass=true,log_pass=true,
        coverage_pass=true,finite_nonempty=true,mean_pinball=1.,original_mean_pinball=2.)
        for period in (2018,2021,2024,2026) for delay in (0,5,10)])
    @test G.decision(metrics).development_pass
    for column in (:native_pass,:log_pass,:coverage_pass,:finite_nonempty)
        bad = copy(metrics); bad[1,column] = false
        @test !G.decision(bad).development_pass
    end
    bad = copy(metrics); bad.mean_pinball .= 3
    @test !G.decision(bad).development_pass
    @test_throws ArgumentError G.decision(metrics[1:end-1,:])
    duplicate = copy(metrics); duplicate[end,:] = duplicate[1,:]
    @test_throws ArgumentError G.decision(duplicate)
    @test length(G.PANEL_HASHES) == 12
    mktempdir() do directory
        for name in keys(G.PANEL_HASHES)
            write(joinpath(directory,name), "altered panel")
        end
        output = joinpath(directory,"output")
        @test_throws ErrorException G.run_experiment(directory,output)
        @test !ispath(output)
        mkpath(output)
        @test_throws ErrorException G.run_experiment(directory,output)
    end
end
end
