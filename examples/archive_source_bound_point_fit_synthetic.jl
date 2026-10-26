using SolarSINDy,Dates,SHA

"""Fit and independently reload generated source-bound point models from sealed cases."""
function run_archive_source_bound_point_fit_synthetic(;calibration_job=nothing)
    source_job=SolarSINDy.ArchiveCatalogSourceJob
    case_job=SolarSINDy.ArchiveCatalogCaseJob
    comparison=SolarSINDy.ArchiveCatalogCaseComparison
    codec=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
    clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
    digest(path)=open(io->bytes2hex(sha256(io)),path)
    start=DateTime(2020)
    last_minute=calibration_job===nothing ? 90 : 180
    label_asof=start+Minute(last_minute+31)
    mktempdir() do temporary
        root=realpath(temporary);inputs=joinpath(root,"inputs");mkdir(inputs)
        descriptors=Any[]
        for station in ("CMO","FRD")
            lines=[" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
                " IAGA CODE              $station |"," Reported               XYZF |",
                " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
                "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |"]
            for minute in 0:last_minute
                time=Dates.format(start+Minute(minute),dateformat"yyyy-mm-dd HH:MM:SS.sss")
                push!(lines,"$time 001 $(3minute) $(4minute) 0 0")
            end
            name="generated-"*lowercase(station)*".txt";path=joinpath(inputs,name)
            bytes=Vector{UInt8}(codeunits(join(lines,"\n")*"\n"));codec._exclusive_write(path,bytes)
            push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>station,
                "stream_labels"=>[station],"product"=>"quasi-definitive","start_inclusive"=>clock(start),
                "stop"=>clock(start+Minute(last_minute)),"stop_inclusive"=>true,"bytes"=>length(bytes),
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
                    "stop_exclusive"=>clock(start+Minute(last_minute+1)),"label_asof"=>clock(label_asof)) for station in ("CMO","FRD")])
            path=joinpath(root,role*"-cases-request.json");codec._exclusive_write(path,codec.canonical_bytes(case_request))
            request_pin=digest(path);destination=joinpath(root,role*"-cases")
            case_job.publish(path,destination;request_sha256=request_pin)
            value=case_job.read_verified(path,destination;request_sha256=request_pin,
                manifest_sha256=digest(joinpath(destination,"manifest.json")),complete_sha256=digest(joinpath(destination,"COMPLETE.json")))
            @assert !value.actual_source_authority_granted && !value.actual_fit_performed
            split=case_job.split_arrays(value,1;start=start+Minute(40),stop=start+Minute(80),asof=label_asof)
            @assert split.issues==collect(start+Minute(40):Minute(1):start+Minute(49))
            @assert all(==(5.0),split.targets) && !split.actual_fit_performed
            point=SolarSINDy.ArchiveSourceBoundPointFit
            classic=SolarSINDy.ArchiveClassicPointModels
            recipes=(point.BoostRecipe(depth=1,rounds=2,bins=2,seed=17,eta=.5,min_weight=1.,lambda=0.,L2=0.),
                classic.Recipe(kind=:ridge,penalty=1.,bins=2),
                classic.Recipe(kind=:boost,depth=1,trees=2,bins=2,rate=.5,minleaf=1),
                classic.Recipe(kind=:forest,depth=1,trees=2,bins=2,minleaf=1,sample_rows=10,seed=17))
            for recipe in recipes
                expected=point.prepare_expected(value,1,recipe;start=start+Minute(40),stop=start+Minute(80),asof=label_asof)
                fit=point.fit_generated(value,1,recipe,expected;start=start+Minute(40),stop=start+Minute(80),asof=label_asof)
                @assert fit.training_rows==10
                prediction=point.predict_log(fit.capsule,split.features)
                output=joinpath(root,role*"-"*point.family(recipe))
                pins=point.publish_fit(fit.capsule,output)
                restored=point.read_published(output;expected,model_sha256=pins.model_sha256,
                    receipt_sha256=pins.receipt_sha256,complete_sha256=pins.complete_sha256)
                @assert point.predict_log(restored,split.features)==prediction && all(isfinite,prediction)
                @assert !pins.actual_fit_performed
                if calibration_job!==nothing
                    k=calibration_job;cal_start=start+Minute(80);cal_stop=start+Minute(140);first_issue=cal_stop
                    outside=k.prepare_expected(value,1,restored;start=cal_start,stop=cal_stop,asof=label_asof,first_issue)
                    capsule=k.calibrate_generated(value,1,restored,outside;start=cal_start,stop=cal_stop,asof=label_asof,first_issue)
                    directory=joinpath(root,role*"-"*point.family(recipe)*"-calibration")
                    calibration_pins=k.publish_calibration(capsule,restored,outside,directory)
                    loaded=k.read_calibration(directory,restored,outside;calibration_pins...)
                    projected=k.completed_calibration(loaded,restored,outside;first_issue)
                    @assert length(projected.chronological_scores)==30 && projected.purpose===:synthetic
                    @assert SolarSINDy.CompletedMinuteReplay.issue_timing(first_issue,projected.policy).sufficient_phase_period_minutes==31
                end
            end
            push!(values,value)
        end
        for index in 1:2
            left=values[1];right=values[2]
            result=comparison.compare_case_bytes(left.features[index].payload,left.labels[index].payload,
                right.features[index].payload,right.labels[index].payload;mode=:ground11,start,stop=start+Minute(last_minute+1))
            @assert result.scheduled==last_minute+1 && result.features_accepted==last_minute-30 && result.labels_accepted==last_minute-29 && result.supervised==last_minute-60
            @assert result.changed_feature_numbers==result.changed_label_numbers==0
        end
    end
    println("Generated source-bound point fits and data-only rereads agree; actual fits=0.")
    nothing
end
