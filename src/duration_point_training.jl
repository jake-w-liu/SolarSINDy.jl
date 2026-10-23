"""Source-bound, research-only two-stage B point designs and gains."""
module DurationPointTraining
using Dates,SHA,JSON3,LinearAlgebra
using ..IntervalMeanForecaster
using ..DurationConditionedForcing
using ..DurationCurrentStateStream
const A=IntervalMeanForecaster
const Adapter=A.DstIntervalAdapter
const Stream=A.DstIntervalStream
const D=DurationConditionedForcing
const C=DurationCurrentStateStream
const K=D.K
const CONTRACT="f0fbbbd3bd7fea2fac983a232c3cbf7265af9524cfb5ad80e45d3ac2610a148b"
const PACKAGE_ROOT=dirname(@__DIR__)
const SOURCE_FILES=("src/duration_point_training.jl","src/duration_point_values.jl","src/duration_point_contexts.jl","src/duration_point_coordinator.jl","src/duration_point_fit.jl","src/duration_point_artifacts.jl","src/duration_point_codec.jl","src/duration_point_basis.jl","src/duration_point_fit_artifacts.jl","src/duration_point_bundle.jl","src/duration_point_schema.jl","tools/actual_b_chain/ActualBChain.jl","tools/actual_b_chain/BInputs.jl","tools/actual_b_chain/BInitialPointFits.jl","tools/actual_b_chain/BSelection.jl","tools/actual_b_chain/BReferenceAdmission.jl","src/SolarSINDy.jl","Project.toml","Manifest.toml")
for name in SOURCE_FILES;Base.include_dependency(joinpath(PACKAGE_ROOT,name));end
const LOADED_FILES=Tuple(name=>bytes2hex(sha256(read(joinpath(PACKAGE_ROOT,name)))) for name in SOURCE_FILES)
const LOADED_C=C.source_identity()
const LOADED_STREAM=Stream.verify_source_identity()
export settings,fit_point_gains,predict_point_rows
_need(ok,message)=ok || throw(ArgumentError(message))
function source_identity()
    _need(C.source_identity()==LOADED_C,"loaded B point dependencies changed")
    for (name,hash) in LOADED_FILES
        _need(bytes2hex(sha256(read(joinpath(PACKAGE_ROOT,name))))==hash,"loaded B point source changed: $name")
    end
    dependencies=Tuple((name in ("duration_conditioned_forcing.jl","duration_current_state_stream.jl") ? "src/"*name : "src/interval_mean_forecaster/"*name)=>hash for (name,hash) in LOADED_C)
    (dependencies...,LOADED_FILES...)
end
include("duration_point_values.jl")
include("duration_point_contexts.jl")
include("duration_point_coordinator.jl")
include("duration_point_fit.jl")
end
