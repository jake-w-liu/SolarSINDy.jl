"""
    CompletedRefitArtifacts

Research-only, source-audited synthetic ridge and chronological calibration
artifacts. Synthetic receipts cannot qualify actual-source provenance. Actual
source parsing, selected tree refits and matched-baseline execution remain
separate qualification requirements.
"""
module CompletedRefitArtifacts
using Dates, Statistics, LinearAlgebra, SHA, JSON3
import ..CompletedSourceDBDT
const D = CompletedSourceDBDT
const CMR = D.CMR
const CF = D.CF
const EPOCH = DateTime(1970)

include("fields.jl")
include("codec.jl")
include("protocol.jl")
include("types.jl")
include("store.jl")
include("runtime.jl")
include("source_input.jl")
include("ridge.jl")
include("numerical_audit.jl")
include("calibration.jl")
include("jobs.jl")
include("publication.jl")
end
