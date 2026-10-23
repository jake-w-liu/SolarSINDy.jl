"""Distinct bounded data-only B point codecs and independently replayed readers."""
module DurationPointArtifacts
using Dates,SHA,JSON3,LinearAlgebra
using ..DurationPointTraining
const T=DurationPointTraining
const D=T.D
const A=T.A
const Stream=T.Stream
const C=T.C
const FILES=("duration_point_artifacts.jl","duration_point_codec.jl","duration_point_basis.jl","duration_point_fit_artifacts.jl","duration_point_bundle.jl","duration_point_schema.jl")
const LOADED_FILES=Tuple(n=>bytes2hex(sha256(read(joinpath(@__DIR__,n)))) for n in FILES)
const LOADED_T=T.source_identity()
export export_fit_input,fit_point_gains_from_input,save_point_bundle,load_point_bundle
function source_identity()
    T._need(T.source_identity()==LOADED_T,"loaded B artifact dependencies changed")
    for (name,hash) in LOADED_FILES
        T._need(bytes2hex(sha256(read(joinpath(@__DIR__,name))))==hash,"loaded B artifact source changed")
    end
    T.source_identity()
end
include("duration_point_schema.jl")
include("duration_point_codec.jl")
include("duration_point_basis.jl")
include("duration_point_fit_artifacts.jl")
include("duration_point_bundle.jl")
end
