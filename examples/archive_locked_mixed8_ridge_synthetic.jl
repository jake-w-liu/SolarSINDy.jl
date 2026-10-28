# Generated-only locked ridge exercise; never grants actual fitting authority.
module ArchiveLockedMixed8RidgeSynthetic
using Dates,SHA,SolarSINDy
const F=SolarSINDy.ArchiveSourceBoundPointFit
const L=SolarSINDy.ArchiveLockedMixed8Ridge
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(bytes))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"..","test","archive_source_bound_fit","case_fixture.jl"))

function run_example()
    mktempdir() do temporary
        root=realpath(temporary)
        for role in ("primary","independent")
            directory=joinpath(root,role);mkdir(directory)
            cases=make_cases(realpath(directory),role,"mixed")
            recipe=F.LockedMixed8RidgeRecipe()
            start=START+Minute(40);stop=START+Minute(90);asof=START+Minute(211)
            expected=F.prepare_expected(cases,1,recipe;start,stop,asof)
            point=F.fit_generated(cases,1,recipe,expected;start,stop,asof).capsule
            arrays=Q.split_arrays(cases,1;start,stop,asof)
            predictions=F.predict_log(point,arrays.features)
            @assert point.model isa L.NumericModel && all(isfinite,predictions)
            output=joinpath(directory,"locked-point");pins=F.publish_fit(point,output)
            restored=F.read_published(output;expected,model_sha256=pins.model_sha256,
                receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
            @assert F.model_bytes(restored.model)==F.model_bytes(point.model)
            @assert F.predict_log(restored,arrays.features)==predictions
            receipt=R.decode_canonical(F.receipt_bytes(restored))
            @assert receipt["family"]==L.FAMILY && !receipt["actual_source_access"] && !receipt["actual_fit_performed"]
        end
    end
    println("Generated locked mixed8 ridge publication/readback agrees;actual fits=0.")
    nothing
end
end
