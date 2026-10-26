using SolarSINDy,Dates,SHA

"""Reverify generated sealed cases and inspect their independent read admission."""
function run_archive_case_reader_synthetic()
    source_job=SolarSINDy.ArchiveCatalogSourceJob
    case_job=SolarSINDy.ArchiveCatalogCaseJob
    comparison=SolarSINDy.ArchiveCatalogCaseComparison
    codec=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
    clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
    digest(path)=open(io->bytes2hex(sha256(io)),path)
    start=DateTime(2020)
    mktempdir() do temporary
        root=realpath(temporary);inputs=joinpath(root,"inputs");mkdir(inputs)
        descriptors=Any[]
        for station in ("CMO","FRD")
            lines=[" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
                " IAGA CODE              $station |"," Reported               XYZF |",
                " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
                "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |"]
            for minute in 0:90
                time=Dates.format(start+Minute(minute),dateformat"yyyy-mm-dd HH:MM:SS.sss")
                push!(lines,"$time 001 $(3minute) $(4minute) 0 0")
            end
            name="generated-"*lowercase(station)*".txt";path=joinpath(inputs,name)
            bytes=Vector{UInt8}(codeunits(join(lines,"\n")*"\n"));codec._exclusive_write(path,bytes)
            push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>station,
                "stream_labels"=>[station],"product"=>"quasi-definitive","start_inclusive"=>clock(start),
                "stop"=>clock(start+Minute(90)),"stop_inclusive"=>true,"bytes"=>length(bytes),
                "sha256"=>bytes2hex(sha256(bytes)),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
        end
        values=Any[]
        for role in ("primary","independent")
            request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>"ground-only",
                "role"=>role,"runtime"=>string(VERSION),"data_root"=>inputs,"inputs"=>descriptors)
            path=joinpath(root,role*"-source-request.json");codec._exclusive_write(path,codec.canonical_bytes(request))
            source=joinpath(root,role*"-source");source_job.reconstruct(path,source)
            case_request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated","layout"=>"ground-only",
                "role"=>role,"runtime"=>string(VERSION),
                "source"=>Dict("directory"=>source,"manifest_sha256"=>digest(joinpath(source,"manifest.json")),"complete_sha256"=>digest(joinpath(source,"COMPLETE.json"))),
                "tables"=>[Dict("id"=>station*"-generated","station"=>station,"start_inclusive"=>clock(start),
                    "stop_exclusive"=>clock(start+Minute(91)),"label_asof"=>clock(start+Minute(121))) for station in ("CMO","FRD")])
            path=joinpath(root,role*"-cases-request.json");codec._exclusive_write(path,codec.canonical_bytes(case_request))
            request_pin=digest(path);destination=joinpath(root,role*"-cases")
            case_job.publish(path,destination;request_sha256=request_pin)
            verified=SolarSINDy.ArchiveCatalogCaseReader.read_reverified(path,destination;request_sha256=request_pin,
                manifest_sha256=digest(joinpath(destination,"manifest.json")),complete_sha256=digest(joinpath(destination,"COMPLETE.json")))
            value=verified.cases
            admission=codec.decode_canonical(SolarSINDy.ArchiveCatalogCaseReader.admission_bytes(verified);limit=131072)
            @assert admission["evidence_mode"]=="generated" && admission["all_artifact_bytes_recomputed"]
            @assert !admission["actual_model_fit_authorized"] && !admission["new_case_generation_authorized"]
            @assert !value.actual_source_authority_granted && !value.actual_fit_performed
            split=case_job.split_arrays(value,1;start=start+Minute(40),stop=start+Minute(80),asof=start+Minute(121))
            @assert split.issues==collect(start+Minute(40):Minute(1):start+Minute(49))
            @assert all(==(5.0),split.targets) && !split.actual_fit_performed
            push!(values,value)
        end
        for index in 1:2
            left=values[1];right=values[2]
            result=comparison.compare_case_bytes(left.features[index].payload,left.labels[index].payload,
                right.features[index].payload,right.labels[index].payload;mode=:ground11,start,stop=start+Minute(91))
            @assert result.scheduled==91 && result.features_accepted==60 && result.labels_accepted==61 && result.supervised==30
            @assert result.changed_feature_numbers==result.changed_label_numbers==0
        end
    end
    println("Generated existing-case rereads agree; no actual source access or model-fit permission.")
    nothing
end
