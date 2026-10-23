"""Separate source-bound B research executor; synthetic calls confer no actual authority."""
module ActualBChain
using SolarSINDy,Dates,SHA,JSON3
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const D=SolarSINDy.DurationConditionedForcing
include("BSelection.jl")
include("BInputs.jl")
include("BInitialPointFits.jl")
end
