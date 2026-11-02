# Generated point values and untrusted metadata only; no actual selected operation.
module SelectedPointBindingSynthetic
using Dates,SHA,JSON3,SolarSINDy
const M=SolarSINDy.ArchiveSelectedPointBindings
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const L=SolarSINDy.ArchiveLockedMixed8Ridge
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const START=DateTime(2020)
digest(raw)=bytes2hex(sha256(raw))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","test","archive_source_bound_fit","case_fixture.jl"))
include(joinpath(@__DIR__,"..","test","selected_point_bindings","fixture_support.jl"))

function run_example()
    mktempdir() do temporary
        root=realpath(temporary);values=generated_selected_values(root)
        for (index,binding) in enumerate((values.fixed,values.bindings...))
            metadata=R.decode_canonical(M.encode_binding(binding);limit=M.MAX_BINDING)
            @assert !metadata["producer_evidence_verified"] && !metadata["actual_authority_granted"] && !metadata["binding_performed_new_fit"]
            path=joinpath(root,"metadata-"*string(index));pins=M.publish_metadata(binding,path)
            restored=M.read_binding(path;expected=pins,point=binding.point,rates=values.rates,point_choice=binding.point_choice)
            @assert M.encode_binding(restored)==M.encode_binding(binding)
        end
        @assert length(values.bindings)==3 && !isdefined(M,:fit_selected_archival)
    end
    println("Generated point bindings retain primary rate origins; actual selected operations unavailable; actual source fits=0.")
    nothing
end
end
