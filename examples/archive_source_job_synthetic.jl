using SolarSINDy, SHA

"""Reconstruct explicit generated source snapshots through both independent readers."""
function run_archive_source_job_synthetic()
    job=SolarSINDy.ArchiveCatalogSourceJob
    codec=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
    mktempdir() do temporary
        root=realpath(temporary);input_root=joinpath(root,"inputs");mkdir(input_root)
        descriptors=Any[]
        for (station,factor) in (("CMO",1),("FRD",2))
            lines=[" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
                " IAGA CODE              $station |"," Reported               XYZF |",
                " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
                "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |"]
            for minute in 0:2
                push!(lines,"2020-01-01 00:0$minute:00.000 001 $(3*minute*factor) $(4*minute*factor) 0 0")
            end
            bytes=Vector{UInt8}(codeunits(join(lines,"\n")*"\n"));name="generated-"*lowercase(station)*".txt"
            path=joinpath(input_root,name);codec._exclusive_write(path,bytes)
            push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>station,
                "stream_labels"=>[station],"product"=>"quasi-definitive","start_inclusive"=>"2020-01-01T00:00:00.000Z",
                "stop"=>"2020-01-01T00:02:00.000Z","stop_inclusive"=>true,"bytes"=>length(bytes),
                "sha256"=>bytes2hex(sha256(bytes)),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
        end
        wind=Vector{UInt8}(codeunits("2020-01-01T00:00:00.000Z,1,-2,400,5\n2020-01-01T00:01:00.000Z,1,-2,400,5\n"))
        path=joinpath(input_root,"generated-wind.csv");codec._exclusive_write(path,wind)
        push!(descriptors,Dict("logical_id"=>"generated-wind.csv","path"=>path,"kind"=>"wind","station"=>nothing,
            "stream_labels"=>["CMO","FRD"],"product"=>"OMNI_HRO_1MIN","start_inclusive"=>"2020-01-01T00:00:00.000Z",
            "stop"=>"2020-01-01T00:02:00.000Z","stop_inclusive"=>false,"bytes"=>length(wind),
            "sha256"=>bytes2hex(sha256(wind)),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
        results=Dict{String,Any}()
        for layout in ("ground-only","mixed"), role in ("primary","independent")
            inputs=layout=="ground-only" ? descriptors[1:2] : descriptors
            request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>layout,
                "role"=>role,"runtime"=>string(VERSION),"data_root"=>input_root,"inputs"=>inputs)
            tag=layout*"-"*role;request_path=joinpath(root,tag*".json")
            codec._exclusive_write(request_path,codec.canonical_bytes(request))
            manifest=job.reconstruct(request_path,joinpath(root,tag))
            @assert length(manifest["catalogs"])==(layout=="ground-only" ? 2 : 4)
            @assert length(manifest["read_trace"])==length(inputs)
            @assert !manifest["provider_receipts_available"] && !manifest["independent_source_qualification"]
            for catalog in values(manifest["catalogs"])
                @assert catalog["rows"]==(catalog["kind"]=="ground" ? 3 : 2)
                @assert catalog["missing_minutes"]==catalog["uncovered_minutes"]==0
            end
            results[tag]=manifest
        end
        for layout in ("ground-only","mixed")
            a=results[layout*"-primary"]["catalogs"];b=results[layout*"-independent"]["catalogs"]
            @assert all(a[key]["selected_sha256"]==b[key]["selected_sha256"] for key in keys(a))
        end
    end
    println("Generated archive source snapshots agree; no actual archive data or models used.")
    nothing
end
