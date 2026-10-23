module TestCompletedGroundAdmission
using Test, Dates, JSON3, SolarSINDy
using SolarSINDy.CompletedMinuteReplay

oracle = JSON3.read(read(joinpath(@__DIR__, "completed_ground_admission_oracle.json")), Dict{String,Any})
const GROUND_FIELDS = (:datetime, :ground_anchor, :dbdt_now, :dbdt_mean30,
    :dbdt_max30, :dbdt_std30, :availability_kind, :measured_receipts_available)

struct GuardedGround <: AbstractDict{DateTime,Tuple{Float64,Float64}}
    data::Dict{DateTime,Tuple{Float64,Float64}}
    first_allowed::DateTime
    last_allowed::DateTime
end
function permitted(source::GuardedGround, key)
    source.first_allowed <= key <= source.last_allowed ||
        error("unavailable ground source accessed")
end
Base.haskey(source::GuardedGround,key) = (permitted(source,key); haskey(source.data,key))
Base.getindex(source::GuardedGround,key) = (permitted(source,key); source.data[key])
Base.length(source::GuardedGround) = length(source.data)
Base.iterate(::GuardedGround) = error("ground admission must not scan source keys")

struct ZeroBasedGround{N} <: AbstractVector{Float64}
    values::NTuple{N,Float64}
end
Base.size(::ZeroBasedGround{N}) where {N} = (N,)
Base.axes(::ZeroBasedGround{N}) where {N} = (0:N-1,)
Base.firstindex(::ZeroBasedGround) = 0
Base.lastindex(::ZeroBasedGround{N}) where {N} = N-1
Base.getindex(value::ZeroBasedGround{N},index::Int) where {N} =
    0<=index<N ? value.values[index+1] : throw(BoundsError(value,index))
Base.iterate(value::ZeroBasedGround{N},index::Int=0) where {N} =
    index>=N ? nothing : (value[index],index+1)

@testset "Independent ground-only completed-source literals" begin
    @test length(oracle["cases"]) == 41
    for expected in oracle["cases"]
        issue = DateTime(expected["issue"])
        policy = ReplayPolicy(; ground_extra_ms=expected["ground_extra_ms"],
            omni_extra_ms=expected["omni_extra_ms"],label_extra_ms=expected["label_extra_ms"])
        anchor = DateTime(expected["ground_anchor"])
        oldest = DateTime(expected["oldest_ground_vector"])
        full = Dict(issue+Minute(k)=>(Float64(k*k),0.0) for k in -1500:31)
        original = copy(full)
        prefix = Dict(k=>v for (k,v) in full if oldest<=k<=anchor)
        admitted = admit_ground_features(GuardedGround(prefix,oldest,anchor),issue;policy)
        @test admitted.accepted
        @test admitted.reason === :complete
        row = admitted.value.row
        @test keys(row) == GROUND_FIELDS
        @test row.datetime == issue
        @test row.ground_anchor == anchor
        @test row.dbdt_now == expected["dbdt_now"]
        @test row.dbdt_mean30 == expected["dbdt_mean30"]
        @test row.dbdt_max30 == expected["dbdt_max30"]
        @test isapprox(row.dbdt_std30,expected["dbdt_std30"];rtol=0,atol=1e-13)
        @test row.availability_kind === :assumed_completed_source_intervals
        @test !row.measured_receipts_available
        @test admitted.value.timing.latest_ground_assumed_ready == DateTime(expected["latest_ground_assumed_ready"])
        @test admit_ground_features(full,issue;policy) == admitted
        changed = copy(full)
        changed[issue] = (NaN,Inf)
        changed[issue+Minute(30)] = (Inf,NaN)
        @test admit_ground_features(changed,issue;policy) == admitted
        unrelated_policy = ReplayPolicy(;ground_extra_ms=expected["ground_extra_ms"],
            omni_extra_ms=86_400_000-expected["omni_extra_ms"],
            label_extra_ms=86_400_000-expected["label_extra_ms"])
        @test admit_ground_features(full,issue;policy=unrelated_policy).value.row == row
        @test full == original
    end
end

@testset "Ground-only errors and mixed compatibility" begin
    issue = DateTime(2024,5,10,12)
    mag = Dict{DateTime,Any}(issue+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    wind = Dict(issue-Minute(1)=>(400.0,-5.0,1.0,8.0))
    absent_wind = Dict{DateTime,NTuple{4,Float64}}()
    bad_wind = Dict(issue-Minute(1)=>(400.0,-5.0,1.0,-1.0))
    @test admit_ground_features(mag,issue).accepted
    @test admit_features(mag,absent_wind,issue).reason === :missing_omni_history
    @test admit_features(mag,bad_wind,issue).reason === :invalid_omni_history
    for invalid in ((NaN,0.0),(Inf,0.0),(true,0.0),(1.0,),"invalid")
        changed = copy(mag); changed[issue-Minute(15)] = invalid
        @test admit_ground_features(changed,issue).reason === :invalid_ground_history
        @test admit_features(changed,absent_wind,issue).reason === :invalid_ground_history
    end
    missing = copy(mag); delete!(missing,issue-Minute(31))
    @test admit_ground_features(missing,issue).reason === :missing_ground_history
    @test admit_features(missing,bad_wind,issue).reason === :missing_ground_history
    zeros = Dict(k=>(0.0,0.0) for k in keys(mag))
    zero_row = admit_ground_features(zeros,issue).value.row
    @test (zero_row.dbdt_now,zero_row.dbdt_mean30,zero_row.dbdt_max30,zero_row.dbdt_std30) == (0.0,0.0,0.0,0.0)
    axis_ground = Dict(k=>ZeroBasedGround(v) for (k,v) in mag)
    @test admit_ground_features(axis_ground,issue) == admit_ground_features(mag,issue)
    @test admit_features(axis_ground,wind,issue) == admit_features(mag,wind,issue)
    mixed = admit_features(mag,wind,issue).value.row
    @test keys(mixed) == (:datetime,:ground_anchor,:omni_anchor,:dbdt_now,
        :dbdt_mean30,:dbdt_max30,:dbdt_std30,:V,:Bz,:By,:n,:Bs,:VBs,
        :availability_kind,:measured_receipts_available)
    @test (mixed.dbdt_now,mixed.dbdt_mean30,mixed.dbdt_max30,mixed.V,
        mixed.Bz,mixed.By,mixed.n,mixed.Bs,mixed.VBs) == (3.0,32.0,61.0,400.0,-5.0,1.0,8.0,5.0,2000.0)
    extreme = Dict(issue+Minute(k)=>(mod(k,4)<2 ? 0.0 : 1e308,0.0) for k in -40:0)
    @test admit_ground_features(extreme,issue).reason === :invalid_feature_summary
    @test admit_features(extreme,absent_wind,issue).reason === :missing_omni_history
    @test admit_features(extreme,bad_wind,issue).reason === :invalid_omni_history
    @test admit_features(extreme,wind,issue).reason === :invalid_feature_summary
    difference_overflow = copy(mag)
    difference_overflow[issue-Minute(15)] = (-1e308,0.0)
    difference_overflow[issue-Minute(16)] = (1e308,0.0)
    @test admit_ground_features(difference_overflow,issue).reason === :invalid_ground_history
    @test_throws ArgumentError admit_ground_features(mag,issue+Millisecond(1))
end

history = JSON3.read(read(joinpath(@__DIR__,"completed_minute_history_oracle.json")),Dict{String,Any})
@testset "Ground-only checked history boundaries" begin
    from_ms(value) = DateTime(Dates.UTInstant{Millisecond}(Millisecond(parse(Int64,value))))
    for row in history["cases"]
        issue = from_ms(row["issue_ms"])
        policy = ReplayPolicy(;ground_extra_ms=row["ground_extra_ms"])
        mag = Dict{DateTime,Tuple{Float64,Float64}}()
        for k in 0:30
            exact = parse(BigInt,row["ground_anchor_ms"])-60_000k
            wrapped = Int64(mod(exact-BigInt(typemin(Int64)),big(2)^64)+BigInt(typemin(Int64)))
            mag[DateTime(Dates.UTInstant{Millisecond}(Millisecond(wrapped)))] = (Float64(k),0.0)
        end
        if row["expected"] == "OverflowError"
            @test_throws OverflowError admit_ground_features(mag,issue;policy)
        else
            result = admit_ground_features(mag,issue;policy)
            @test result.accepted
            @test (result.value.row.dbdt_now,result.value.row.dbdt_mean30,result.value.row.dbdt_max30,result.value.row.dbdt_std30) == (1.0,1.0,1.0,0.0)
        end
    end
end
end
