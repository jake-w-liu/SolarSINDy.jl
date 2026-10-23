module TestCompletedMinuteReplay
using Test, Dates, JSON3, SolarSINDy
using SolarSINDy.CompletedMinuteReplay
clock_oracle = JSON3.read(read(joinpath(@__DIR__,"completed_minute_clock_oracle.json")), Dict{String,Any})
@testset "Independent completed-source clock literals" begin
    names = (:latest_ground_center, :latest_ground_assumed_ready,
             :latest_omni_arrival_start, :latest_omni_assumed_ready,
             :first_target_center, :last_target_center,
             :label_assumed_ready, :first_scheduled_feedback)
    for row in clock_oracle["cases"]
        policy = ReplayPolicy(; ground_extra_ms=row["ground_extra_ms"],
            omni_extra_ms=row["omni_extra_ms"], label_extra_ms=row["label_extra_ms"])
        actual = issue_timing(DateTime(row["issue"]), policy)
        for name in names
            @test getproperty(actual,name) == DateTime(row[string(name)])
        end
        @test actual.sufficient_phase_period_minutes == row["sufficient_phase_period_minutes"]
        @test actual.legacy_30_minute_feedback_ready == row["legacy_30_minute_feedback_ready"]
        @test actual.availability_kind === :assumed_completed_source_intervals
        @test !actual.measured_receipts_available
        @test actual.label_assumed_ready == actual.last_target_center + Millisecond(45_000 + row["ground_extra_ms"] + row["label_extra_ms"])
        @test actual.latest_ground_assumed_ready <= actual.issue
        @test actual.latest_omni_assumed_ready <= actual.issue
        @test actual.latest_ground_assumed_ready + Minute(1) > actual.issue
        @test actual.latest_omni_assumed_ready + Minute(1) > actual.issue
        @test minute_phase(actual.issue,policy) == minute_phase(actual.first_scheduled_feedback,policy)
        @test actual.label_assumed_ready <= actual.first_scheduled_feedback
    end
    for row in clock_oracle["ground_boundary"]
        @test (DateTime(row["center"])+Second(45) <= DateTime(row["issue"])) == row["assumed_complete"]
    end
    for row in clock_oracle["strict_training_boundary"]
        @test training_eligible(DateTime(row["label_ready"]),DateTime(row["next_split_issue"])) == row["eligible"]
    end
    row = clock_oracle["pre_epoch_phase"]
    @test minute_phase(DateTime(row["issue"])) == row["one_based_phase"]
    @test minute_phase(DateTime(1970)) == 1
    for bad in (-1, 86_400_001, 1.5, true, Inf)
        @test_throws ArgumentError ReplayPolicy(;ground_extra_ms=bad)
        @test_throws ArgumentError ReplayPolicy(;omni_extra_ms=bad)
        @test_throws ArgumentError ReplayPolicy(;label_extra_ms=bad)
    end
    @test_throws ArgumentError issue_timing(DateTime(2020)+Millisecond(1))
    @test_throws ArgumentError minute_phase(DateTime(2020)+Millisecond(1))
    # Literal source tags exclude the incomplete current-center/current-arrival row.
    t = DateTime(2024,5,10,12)
    timing = issue_timing(t)
    @test timing.latest_ground_center == t-Minute(1)
    @test timing.latest_omni_arrival_start == t-Minute(1)
    @test timing.first_target_center > t
end

feature_oracle=JSON3.read(read(joinpath(@__DIR__,"completed_minute_feature_oracle.json")),Dict{String,Any})
issue=DateTime(feature_oracle["issue"])
mag=Dict(issue+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
omni=Dict(issue+Minute(r["offset_minutes"])=>Tuple(Float64(r[k]) for k in ("V","Bz","By","n")) for r in feature_oracle["wind_source_rows"])
@testset "Independent completed-source feature and label adapter" begin
    for expected in feature_oracle["cases"]
        policy=ReplayPolicy(;ground_extra_ms=expected["ground_extra_ms"],
            omni_extra_ms=expected["omni_extra_ms"],label_extra_ms=expected["label_extra_ms"])
        ready=DateTime(expected["label_ready"])
        features=admit_features(mag,omni,issue;policy)
        @test features.accepted
        row=features.value.row
        @test row.datetime==issue
        @test row.ground_anchor==DateTime(expected["ground_anchor"])
        @test row.omni_anchor==DateTime(expected["omni_anchor"])
        for name in (:dbdt_now,:dbdt_mean30,:dbdt_max30,:V,:Bz,:By,:n,:Bs,:VBs)
            @test getproperty(row,name)==expected[string(name)]
        end
        @test isapprox(row.dbdt_std30,expected["dbdt_std30"];rtol=0,atol=1e-13)
        @test !row.measured_receipts_available
        past_mag=Dict(k=>v for (k,v) in mag if k<=row.ground_anchor)
        past_omni=Dict(k=>v for (k,v) in omni if k<=row.omni_anchor)
        @test admit_features(past_mag,past_omni,issue;policy)==features
        @test admit_target(Dict{DateTime,Tuple{Float64,Float64}}(),issue,ready-Millisecond(1);policy).reason===:label_not_ready
        target=admit_target(mag,issue,ready;policy)
        @test target.accepted && target.value.target_max_dbdt==expected["target_max_dbdt"]
        @test target.value.label_ready==ready
        complete=replay_case(mag,omni,issue,ready;policy)
        @test complete.accepted && complete.value.row.target_max_dbdt==59.0
        @test !training_eligible(ready,ready)
        changed_mag=copy(mag);changed_mag[issue+Minute(30)]=(10_000.0,0.0)
        changed_omni=copy(omni);changed_omni[issue]=(10_000.0,-100.0,20.0,100.0)
        @test admit_features(changed_mag,changed_omni,issue;policy)==features
        @test admit_target(changed_mag,issue,ready;policy).value.target_max_dbdt>59.0
    end
    bad=copy(mag);delete!(bad,issue-Minute(15))
    @test admit_features(bad,omni,issue).reason===:missing_ground_history
    bad=copy(mag);bad[issue-Minute(15)]=(NaN,0.0)
    @test admit_features(bad,omni,issue).reason===:invalid_ground_history
    bad=copy(mag);delete!(bad,issue+Minute(15))
    @test admit_target(bad,issue,issue+Hour(1)).reason===:missing_ground_target
    bad=copy(mag);bad[issue+Minute(15)]=(Inf,0.0)
    @test admit_target(bad,issue,issue+Hour(1)).reason===:invalid_ground_target
    badwind=copy(omni);delete!(badwind,issue-Minute(1))
    @test admit_features(mag,badwind,issue).reason===:missing_omni_history
    badwind=copy(omni);badwind[issue-Minute(1)]=(400.0,-5.0,1.0,-1.0)
    @test admit_features(mag,badwind,issue).reason===:invalid_omni_history
    zero_mag=Dict(k=>(0.0,0.0) for k in keys(mag))
    zero_case=replay_case(zero_mag,omni,issue,issue+Hour(1))
    @test zero_case.accepted
    @test zero_case.value.row.dbdt_std30==zero_case.value.row.target_max_dbdt==0.0
end


struct ZeroBasedVector{N} <: AbstractVector{Float64}
    values::NTuple{N,Float64}
end
Base.size(::ZeroBasedVector{N}) where {N} = (N,)
Base.axes(::ZeroBasedVector{N}) where {N} = (0:N-1,)
Base.firstindex(::ZeroBasedVector) = 0
Base.lastindex(::ZeroBasedVector{N}) where {N} = N-1
Base.getindex(v::ZeroBasedVector{N},i::Int) where {N} =
    0<=i<N ? v.values[i+1] : throw(BoundsError(v,i))
Base.iterate(v::ZeroBasedVector{N},i::Int=0) where {N} =
    i>=N ? nothing : (v[i],i+1)

@testset "Checked extreme scheduling and ordered vector inputs" begin
    for row in clock_oracle["int64_edges"]
        issue=DateTime(Dates.UTInstant{Millisecond}(Millisecond(parse(Int64,row["issue_ms"]))))
        @test_throws OverflowError issue_timing(issue)
    end
    ground_zero=Dict(k=>ZeroBasedVector(v) for (k,v) in mag)
    wind_zero=Dict(k=>ZeroBasedVector(v) for (k,v) in omni)
    @test admit_features(ground_zero,wind_zero,issue)==admit_features(mag,omni,issue)
    @test admit_target(ground_zero,issue,issue+Hour(1))==admit_target(mag,issue,issue+Hour(1))
    @test replay_case(ground_zero,wind_zero,issue,issue+Hour(1))==replay_case(mag,omni,issue,issue+Hour(1))
end

# Boundary literals were derived with unbounded integer arithmetic before the
# history-span repair. Wrapped adversarial keys must never admit future data.
history_oracle = JSON3.read(read(joinpath(@__DIR__,"completed_minute_history_oracle.json")),Dict{String,Any})
@testset "Representable completed-source feature history" begin
    from_ms(value) = DateTime(Dates.UTInstant{Millisecond}(Millisecond(parse(Int64,value))))
    for row in history_oracle["cases"]
        edge_issue = from_ms(row["issue_ms"])
        policy = ReplayPolicy(;ground_extra_ms=row["ground_extra_ms"])
        timing = issue_timing(edge_issue,policy)
        @test Dates.value(timing.latest_ground_center) == parse(Int64,row["ground_anchor_ms"])
        exact_oldest = parse(BigInt,row["oldest_required_ground_ms"])
        edge_mag = Dict{DateTime,Tuple{Float64,Float64}}()
        for k in 0:30
            exact = parse(BigInt,row["ground_anchor_ms"])-60_000k
            wrapped = Int64(mod(exact-BigInt(typemin(Int64)),big(2)^64)+BigInt(typemin(Int64)))
            edge_mag[DateTime(Dates.UTInstant{Millisecond}(Millisecond(wrapped)))] = (Float64(k),0.0)
        end
        edge_wind = Dict(timing.latest_omni_arrival_start=>(400.0,-5.0,1.0,8.0))
        if row["expected"] == "OverflowError"
            @test exact_oldest < BigInt(typemin(Int64))
            @test_throws OverflowError admit_features(edge_mag,edge_wind,edge_issue;policy)
        else
            @test exact_oldest >= BigInt(typemin(Int64))
            admitted = admit_features(edge_mag,edge_wind,edge_issue;policy)
            @test admitted.accepted
            @test admitted.value.row.dbdt_now == admitted.value.row.dbdt_mean30 == admitted.value.row.dbdt_max30 == 1.0
            @test admitted.value.row.dbdt_std30 == 0.0
        end
    end
end

end
