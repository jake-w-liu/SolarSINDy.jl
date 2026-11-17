#!/usr/bin/env julia
# Explicit pinned source request and fresh snapshot destination.
using SolarSINDy
function main(args)
    length(args)==2 || throw(ArgumentError("expected REQUEST DESTINATION"))
    SolarSINDy.ArchiveCatalogSourceJob.reconstruct(args[1],args[2])
    println("Archive catalog snapshot completed; independent reconstruction remains a separate check.")
end
main(ARGS)
