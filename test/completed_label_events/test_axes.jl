struct OffsetInput{T} <: AbstractVector{T}
    values::Vector{T}
end
Base.size(value::OffsetInput)=size(value.values)
Base.axes(value::OffsetInput)=(0:length(value.values)-1,)
Base.getindex(value::OffsetInput,index::Int)=0<=index<length(value.values) ? value.values[index+1] : throw(BoundsError(value,index))
struct UnopenedBatches
    accesses::Base.RefValue{Int}
end
function Base.iterate(value::UnopenedBatches)
    value.accesses[]+=1
    error("axis rejection must precede source batch access")
end

@testset verbose=true "Independent input-axis alignment rejection" begin
    path=joinpath(@__DIR__,"frozen","input_axis_oracle_v1.json")
    oracle=JSON3.read(read(path),Dict{String,Any})
    @test oracle["created_before_guard_patch"] && !oracle["original_17_fixture_groups_changed"]
    ground=Dict(I+Minute(k)=>(0.0,0.0) for k in -40:-1)
    prepared=D.prepare_schedule(ground,[I,I+Minute(1)])
    centers=Float64.(oracle["positional_centers"]);scales=Float64.(oracle["positional_scales"])
    arguments=(prepared,centers,scales)
    for role in eachindex(arguments)
        inputs=ntuple(index->index==role ? OffsetInput(arguments[index]) : arguments[index],3)
        accesses=Ref(0)
        @test_throws ArgumentError replay_shared(inputs...,UnopenedBatches(accesses),calibration();
            subscribers=specs(),stream_id=STREAM,protocol_id=PROTOCOL,asof=I+Minute(1))
        @test accesses[]==oracle["source_batch_accesses_after_rejection"]
    end
    result=replay_shared(prepared,centers,scales,[],calibration();subscribers=specs(),
        stream_id=STREAM,protocol_id=PROTOCOL,asof=I+Minute(1))
    @test result.indices==oracle["ordinary_control_indices"] &&
        only(result.issue_records).bounds[1].center==oracle["ordinary_control_bound_center"] &&
        result.source.attempted==oracle["ordinary_control_attempted"]
end
