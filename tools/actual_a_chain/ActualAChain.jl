module ActualAChain

using SHA
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
include(joinpath(@__DIR__,"..","initial_reference_import","InitialReferenceImport.jl"))
include("InitialAInputs.jl")
include("InitialAFits.jl")

const ReferenceImport=InitialReferenceImport
const Inputs=InitialAInputs
const InitialFits=InitialAFits
export ReferenceImport, Inputs, InitialFits, load_reference, load_prefix, recheck_prefix, run_initial_l0, verify_initial_l0

load_reference(args...;kwargs...)=ReferenceImport.load_reference(args...;kwargs...)
load_prefix(args...;kwargs...)=Inputs.load_prefix(args...;kwargs...)
recheck_prefix(args...;kwargs...)=Inputs.recheck(args...;kwargs...)
run_initial_l0(args...;kwargs...)=InitialFits.run_initial_l0(args...;kwargs...)
verify_initial_l0(args...;kwargs...)=InitialFits.verify_initial_l0(args...;kwargs...)

end # module
