module CompletedSourceDBDTTests
using Test, Dates, JSON3
using SolarSINDy: CompletedSourceDBDT
using SolarSINDy.CompletedSourceDBDT
const CMR=CompletedSourceDBDT.CMR
const ORACLE=JSON3.read(read(joinpath(@__DIR__,"integration_oracle.json")),Dict{String,Any})
const ISSUE=DateTime(ORACLE["issue"])
function calibration(;policy=CMR.ReplayPolicy(),mode=:ground11,ready=ISSUE-Millisecond(1),scores=zeros(9))
    CompletedCalibration(scores,fill(ready,length(scores));policy,mode,
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
end
struct ForbiddenDict <: AbstractDict{DateTime,Float64} end
Base.haskey(::ForbiddenDict,key)=error("forbidden future access")
Base.getindex(::ForbiddenDict,key)=error("forbidden future access")
Base.iterate(::ForbiddenDict)=error("forbidden future scan")
struct PrefixMag <: AbstractDict{DateTime,Tuple{Float64,Float64}}
    data::Dict{DateTime,Tuple{Float64,Float64}}
    last_allowed::DateTime
end
Base.haskey(source::PrefixMag,key)=key<=source.last_allowed ? haskey(source.data,key) : error("future ground read")
Base.getindex(source::PrefixMag,key)=key<=source.last_allowed ? source.data[key] : error("future ground read")
Base.iterate(::PrefixMag)=error("ground dictionary scan")
struct ZeroBasedVector{N} <: AbstractVector{Float64}
    values::NTuple{N,Float64}
end
Base.size(::ZeroBasedVector{N}) where {N}=(N,)
Base.axes(::ZeroBasedVector{N}) where {N}=(0:N-1,)
Base.firstindex(::ZeroBasedVector)=0
Base.lastindex(::ZeroBasedVector{N}) where {N}=N-1
Base.getindex(value::ZeroBasedVector{N},index::Int) where {N}=
    0<=index<N ? value.values[index+1] : throw(BoundsError(value,index))
Base.iterate(value::ZeroBasedVector{N},index::Int=0) where {N}=
    index>=N ? nothing : (value[index],index+1)
include("test_adapter.jl")
include("test_controllers.jl")
include("test_maximum.jl")
include("test_scalar_range.jl")

include("test_package_boundary.jl")
end # module CompletedSourceDBDTTests
