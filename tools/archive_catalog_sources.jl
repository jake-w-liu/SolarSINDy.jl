#!/usr/bin/env julia
# Explicit request/destination and, for actual-cache jobs, four caller path/hash
# pairs: grant, admission plan, source proof, completed software qualification.
using SolarSINDy
function main(args)
    length(args) in (2,10) || throw(ArgumentError("expected request, destination and optionally four authority path/SHA256 pairs"))
    authority = length(args)==2 ? nothing : SolarSINDy.ArchiveCatalogSourceJob.Authority(args[3:10]...)
    SolarSINDy.ArchiveCatalogSourceJob.reconstruct(args[1],args[2];authority)
    println("Archive catalog snapshot completed; independent reconstruction remains a separate check.")
end
main(ARGS)
