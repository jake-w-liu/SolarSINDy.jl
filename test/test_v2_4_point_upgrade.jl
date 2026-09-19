module V24PointUpgradeTests

using Test, CSV, DataFrames, Dates, Random, SHA, Statistics
include(joinpath(@__DIR__, "..", "validation", "operational", "v2_4_point_upgrade.jl"))
const P = V24PointUpgrade

function fixture(; n=42, steps=(1,2,3,4,6,7))
    origin = DateTime(2026, 1, 1)
    records = NamedTuple[]
    for hour in 0:n-1, step in steps
        points = NamedTuple{P.POINT_MODELS}(Tuple(name == :v2_4e ? 0.0 : 5.0 for name in P.POINT_MODELS))
        push!(records, (issue_time_utc=origin+Hour(hour), latest_dst_time_utc=origin+Hour(hour),
            target_time_utc=origin+Hour(hour+step), model_step_hours=step, latest_dst_nt=5.0,
            observation_dst_nt=5.0, static_lo_nt=-2.0, static_hi_nt=3.0,
            model_epoch="fixed", points...))
    end
    return DataFrame(records)
end

witnesses(panel) = unique(select(panel, :issue_time_utc, :latest_dst_time_utc, :latest_dst_nt))
function at(panel, hour, step=1)
    return only(eachrow(panel[(panel.issue_time_utc .== DateTime(2026,1,1)+Hour(hour)) .&
        (panel.model_step_hours .== step), :]))
end

@testset verbose=true "Bounded point correction arithmetic" begin
    history = [fill(1.0,29); 31.0]
    for name in P.POINT_CANDIDATES
        value = P.corrected_point(name, -10, -12, -6, 1, history)
        shift = startswith(name,"Median") ? 1.0 : endswith(name,"24") ? 2.25 : 2.0
        @test value.point == -10 + shift # 23 ones plus 31, or 29 ones plus 31
        @test value.lower == -12.5 + shift
        @test value.upper == -5.0 + shift
        @test value.upper - value.lower == 7.5
        @test value.corrected
        @test value.used_n == (endswith(name,"24") ? 24 : 30)
        @test !value.cap_applied && !value.projection_applied
        sparse = P.corrected_point(name, -10, -12, -6, 1, fill(8.0,29))
        @test !sparse.corrected && sparse.used_n == 0
        @test (sparse.point,sparse.lower,sparse.upper) == (-10.0,-12.0,-6.0)
    end
    for name in P.POINT_CANDIDATES
        window = parse(Int, match(r"\d+$",name).match)
        result = P.corrected_point(name, -20, -22, -16, 1, collect(1:100)./10)
        expected = (101-window+100)/20 # independently summed arithmetic progression
        @test isapprox(result.estimate,expected; atol=1e-12,rtol=0)
        @test result.used_n == window
    end
    for step in (1,2,3,4,6,7), sign in (-1,1)
        value = P.corrected_point("Mean24", -100, -102, -96, step, fill(sign*100.0,30))
        @test value.capped == sign*(10+5step)
        @test value.point == -100 + sign*(10+5step)
        @test value.cap_applied && !value.projection_applied
    end
    upper = P.corrected_point("Median24",49,47,53,1,fill(5.0,30))
    @test (upper.point,upper.actual_shift,upper.lower,upper.upper) == (50.0,1.0,47.5,55.0)
    @test upper.projection_applied
    lower = P.corrected_point("Mean24",-1999,-2001,-1995,1,fill(-5.0,30))
    @test (lower.point,lower.actual_shift,lower.lower,lower.upper) == (-2000.0,-1.0,-2002.5,-1995.0)
    @test lower.projection_applied
    for seed in 1:5
        rng = MersenneTwister(seed)
        h = randn(rng,96)
        for name in P.POINT_CANDIDATES
            a = P.corrected_point(name,-100,-102,-96,2,h)
            b = P.corrected_point(name,-100,-102,-96,2,-h)
            @test isapprox(a.actual_shift,-b.actual_shift;atol=1e-12,rtol=0) # odd location functional
            @test a.upper-a.lower == 7.5
        end
    end
    @test_throws ArgumentError P.corrected_point("bad",0,-1,1,1,zeros(30))
    @test_throws ArgumentError P.corrected_point("Mean24",0,-1,1,5,zeros(30))
    for p in (NaN,Inf,-Inf)
        @test_throws ArgumentError P.corrected_point("Mean24",p,-2010,60,1,zeros(30))
    end
    # The range belongs to corrected outputs, not to an archived uncorrected baseline.
    for name in P.POINT_CANDIDATES, p in (60.0,-2010.0)
        sparse=P.corrected_point(name,p,p-2,p+3,1,zeros(29))
        @test (sparse.point,sparse.lower,sparse.upper)==(p,p-2,p+3)
        @test !sparse.corrected && !sparse.projection_applied && sparse.actual_shift==0
        warm=P.corrected_point(name,p,p-2,p+3,1,zeros(30))
        center=p==60 ? 50. : -2000.
        @test (warm.point,warm.lower,warm.upper)==(center,center-2.5,center+3.75)
        @test warm.corrected && warm.projection_applied && warm.capped==0
    end
    for endpoints in ((2.0,1.0),(0.0,0.0),(1.0,2.0))
        @test_throws ArgumentError P.corrected_point("Mean24",0,endpoints...,1,zeros(30))
    end
    for bad in (NaN,Inf,-Inf,missing,true)
        @test_throws ArgumentError P.corrected_point("Mean24",0,-1,1,1,Any[zeros(29)...,bad])
    end
    @test_throws ArgumentError P.corrected_point("Mean24",0,-1,1,1,fill(floatmax(Float64),30))
    @test_throws ArgumentError P.corrected_point("Mean24",0,-floatmax(Float64),floatmax(Float64),1,zeros(30))
    @test_throws ArgumentError P.corrected_point("Mean24",0,-floatmax(Float64),floatmax(Float64),1,zeros(29))
end

@testset verbose=true "Witness causality, delay and model identity" begin
    panel = fixture()
    witness = witnesses(panel)
    result = P.replay_points(panel,witness)
    for step in (1,2,3,4,6,7), name in P.POINT_CANDIDATES
        before = at(result,28+step,step)
        current = at(result,29+step,step)
        @test !before[Symbol(name,"_corrected")]
        @test before[Symbol(name,"_point")] == 0.0
        @test current[Symbol(name,"_corrected")]
        @test current[Symbol(name,"_point")] == 5.0
        @test current[Symbol(name,"_lower")] == 2.5
        @test current[Symbol(name,"_upper")] == 8.75
        @test current.history_retained_n == 30
    end
    delayed = P.replay_points(panel,witness;delay_hours=1)
    @test !at(delayed,30).Mean24_corrected && at(delayed,30).history_retained_n == 29
    @test at(delayed,31).Mean24_corrected && at(delayed,31).Mean24_point == 5.0
    @test_throws ArgumentError P.replay_points(panel,witness;delay_hours=-1)
    @test_throws ArgumentError P.replay_points(panel,witness;delay_hours=2)
    scored_future = copy(panel)
    scored_future.observation_dst_nt .= 999.0
    changed = P.replay_points(scored_future,witness)
    for name in P.POINT_CANDIDATES
        @test changed[!,Symbol(name,"_point")] == result[!,Symbol(name,"_point")]
    end
    pending = copy(panel)
    pending.observation_dst_nt .= NaN
    @test P.replay_points(pending,witness).Mean24_point == result.Mean24_point
    later = copy(witness)
    push!(later,(DateTime(2026,1,3),DateTime(2026,1,1,10),1000.0))
    @test P.replay_points(panel,later).Mean24_point == result.Mean24_point
    revised = copy(witness)
    revised.latest_dst_time_utc[26] = DateTime(2026,1,1,10)
    revised.latest_dst_nt[26] = 105.0
    revisions = P.replay_points(panel,revised)
    @test revisions.Mean24_point[1:25*6] == result.Mean24_point[1:25*6]
    @test at(revisions,39).history_retained_n == 38 # the original hour-25 witness is absent
    @test isapprox(at(revisions,39).Mean96_point,290/38;atol=1e-12,rtol=0) # 37 errors of 5 plus 105
    isolated = copy(panel)
    isolated[isolated.model_step_hours .== 7,:v2_4e] .= -100
    isolated[isolated.model_step_hours .== 7,:static_lo_nt] .= -102
    isolated[isolated.model_step_hours .== 7,:static_hi_nt] .= -97
    isolation = P.replay_points(isolated,witness)
    @test isolation.Mean24_point[isolation.model_step_hours .== 1] == result.Mean24_point[result.model_step_hours .== 1]
    epoch = copy(panel)
    epoch[epoch.issue_time_utc .>= DateTime(2026,1,2,12),:model_epoch] .= "new"
    @test at(P.replay_points(epoch,witness),40).history_retained_n == 4 # targets 37, 38, 39 and current anchor 40
    @test !at(P.replay_points(epoch,witness),40).Mean24_corrected
    @test isequal(P.replay_points(panel[end:-1:1,:],witness[end:-1:1,:]),result)
    long = fixture(n=150,steps=(1,))
    @test at(P.replay_points(long,witnesses(long)),149).history_retained_n == 96
end

@testset verbose=true "Panel guards and historical column mapping" begin
    panel = fixture(n=3)
    @test nrow(P.checked_panel(panel)) == 18
    @test_throws ArgumentError P.checked_panel(select(panel,Not(:direct_gbm)))
    @test_throws ArgumentError P.checked_panel(vcat(panel,panel[1:1,:]))
    for (column,bad) in ((:direct_gbm,Inf),(:v2_4e,51.0),(:latest_dst_nt,NaN),
        (:model_step_hours,5),(:static_lo_nt,4.0),(:static_hi_nt,-3.0))
        changed = copy(panel)
        changed[1,column] = bad
        @test_throws ArgumentError P.checked_panel(changed)
    end
    changed = copy(panel)
    changed[1,:observation_dst_nt] = Inf
    @test_throws ArgumentError P.checked_panel(changed)
    changed = copy(panel)
    changed[1,:target_time_utc] = changed[1,:issue_time_utc]
    @test_throws ArgumentError P.checked_panel(changed)
    changed = copy(panel)
    changed[1,:target_time_utc] += Hour(1)
    @test_throws ArgumentError P.checked_panel(changed)
    changed = copy(panel)
    changed[1,:model_epoch] = " "
    @test_throws ArgumentError P.checked_panel(changed)
    changed = copy(panel)
    changed[2,:observation_dst_nt] = 8.0 # target hour 2 is also present in the next issue
    @test_throws ArgumentError P.checked_panel(changed)
    bad_witness = witnesses(panel)
    bad_witness[1,:latest_dst_time_utc] += Hour(1)
    @test_throws ArgumentError P.replay_points(panel,bad_witness)
    outside=copy(panel)
    outside.v2_4e[1]=60
    outside.static_lo_nt[1]=58
    outside.static_hi_nt[1]=63
    @test P.checked_panel(outside).v2_4e[1]==60
    @test P.replay_points(outside,witnesses(outside)).Mean24_point[1]==60
    overflow=copy(panel)
    overflow.static_lo_nt[1]=-floatmax(Float64)
    overflow.static_hi_nt[1]=floatmax(Float64)
    @test_throws ArgumentError P.checked_panel(overflow)
    mktempdir() do directory
        path = joinpath(directory,"fold.csv")
        source = copy(panel)
        rename!(source,:static_lo_nt=>:v2_4e_lo_nt,:static_hi_nt=>:v2_4e_hi_nt)
        CSV.write(path,source)
        sha = bytes2hex(sha256(read(path)))
        mapped = P.historical_panel(path,sha,2026)
        @test mapped.v2_4e == panel.v2_4e
        @test mapped.frozen_v2_1 == panel.frozen_v2_1
        @test mapped.static_lo_nt == panel.static_lo_nt
        @test unique(mapped.model_epoch) == ["fold2026"]
        @test_throws ArgumentError P.historical_panel(path,"bad",2026)
        @test_throws ArgumentError P.historical_panel(path,sha,2025)
    end
end

@testset verbose=true "Storm event separation uses witnessed hours" begin
    times = collect(DateTime(2020,1,1):Hour(1):DateTime(2020,1,4,1))
    values = [-50.0;fill(-29.0,72);-60.0]
    events = P.storm_events(times,values)
    @test events == Dict(first(times)=>1,last(times)=>2)
    @test P.storm_events(times,[-50.0;fill(-30.0,72);-60.0])[last(times)] == 1
    @test P.storm_events(times[2:end],values[2:end])[last(times)] == 1 # first observed storm is event one
    keep = setdiff(eachindex(times),[30])
    @test P.storm_events(times[keep],values[keep])[last(times)] == 1
    # Seventy-two known quiet values are insufficient when they are not consecutive.
    # Unlike the shorter case above, ignoring the missing hour would create a second event.
    longer_times = collect(first(times):Hour(1):last(times)+Hour(1))
    longer_values = [-50.0;fill(-29.0,73);-60.0]
    for missing_index in (20,30,40,50,60)
        present = setdiff(eachindex(longer_times),[missing_index])
        @test P.storm_events(longer_times[present],longer_values[present])[last(longer_times)] == 1
    end
    @test P.storm_events(times[1:end-1],[-50.0;fill(-29.0,71);-60.0])[times[end-1]] == 1
    @test isempty(P.storm_events(times,fill(-49.0,length(times))))
    @test P.storm_events(reverse(times),reverse(values)) == events
    @test P.storm_events([times;first(times)],[values;-50.0]) == events
    @test_throws ArgumentError P.storm_events([times;first(times)],[values;0.0])
    @test_throws ArgumentError P.storm_events([first(times)+Minute(1)],[-50.0])
    @test_throws ArgumentError P.storm_events(times,fill(NaN,length(times)))
    @test_throws DimensionMismatch P.storm_events(times,values[2:end])
end

@testset verbose=true "Independent point metrics and paired block weights" begin
    metrics = P.point_metrics([3.0,3.0],[6.0,2.0])
    @test metrics.n == 2 && metrics.rmse_nt == sqrt(5.0)
    @test metrics.mae_nt == 2.0 && metrics.bias_nt == 1.0
    @test P.point_metrics(Float64[],Float64[]).n == 0
    @test isnan(P.point_metrics(Float64[],Float64[]).rmse_nt)
    @test_throws DimensionMismatch P.point_metrics([1.0],[1.0,2.0])
    @test_throws ArgumentError P.point_metrics([NaN],[1.0])
    @test_throws ArgumentError P.point_metrics([1.0],[Inf])
    @test_throws ArgumentError P.point_metrics([floatmax(Float64)],[-floatmax(Float64)])
    p,r,y,blocks = [1.0,3.0,3.0],[2.0,0.0,0.0],zeros(3),[1,2,2]
    bootstrap = P.paired_rmse_gain(p,r,y,blocks;reps=100,seed=17)
    @test bootstrap.gain_nt == sqrt(4/3)-sqrt(19/3)
    # Whole-block draws have only three possible values: two copies of the one-row
    # block (+1), one of each (sqrt(4/3)-sqrt(19/3)), or two two-row blocks (-3).
    rng = MersenneTwister(17)
    expected = Float64[]
    for _ in 1:100
        one,two = rand(rng,1:2),rand(rng,1:2)
        value = one == two ? (one == 1 ? 1.0 : -3.0) : sqrt(4/3)-sqrt(19/3)
        push!(expected,value)
    end
    @test bootstrap.draws == expected
    @test bootstrap.lower_nt == -3.0 && bootstrap.upper_nt == 1.0
    swapped = P.paired_rmse_gain(r,p,y,blocks;reps=100,seed=17)
    @test swapped.draws == -expected
    @test P.paired_rmse_gain(p,p,y,blocks;reps=100,seed=17).draws == zeros(100)
    @test P.paired_rmse_gain(p,r,y,fill(1,3);reps=100).draws == Float64[]
    @test isnan(P.paired_rmse_gain(p,r,y,fill(1,3);reps=100).lower_nt)
    @test_throws DimensionMismatch P.paired_rmse_gain(p,r,y,[1,2];reps=100)
    @test_throws ArgumentError P.paired_rmse_gain(p,r,y,blocks;reps=99)
    @test_throws ArgumentError P.paired_rmse_gain(p,r,y,blocks;reps=100,seed=-1)
end

end # module
