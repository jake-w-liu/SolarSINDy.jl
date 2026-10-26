function make_cases(root,role,layout;profile=:pulse)
    input=joinpath(root,"inputs");mkdir(input);descriptors=Any[]
    for station in ("CMO","FRD")
        lines=[" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
            " IAGA CODE              $station |"," Reported               XYZF |",
            " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
            "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |"]
        for i in 0:180
            # Difference89->90 changes from(3,4) to(6,8), continuously.
            if profile===:pulse
                x=i<90 ? 3i : 3*89+6*(i-89);y=i<90 ? 4i : 4*89+8*(i-89)
            elseif profile===:zero_then_10_then_20
                x=10000+(i<90 ? 0 : i<140 ? 6*(i-89) : 6*50+12*(i-139))
                y=1000+(i<90 ? 0 : i<140 ? 8*(i-89) : 8*50+16*(i-139))
            elseif profile===:five_then_zero
                x=10000+3*min(i,89);y=1000+4*min(i,89)
            else
                throw(ArgumentError("generated calibration profile"))
            end
            push!(lines,"$(Dates.format(START+Minute(i),dateformat"yyyy-mm-dd HH:MM:SS.sss")) 001 $x $y 0 0")
        end
        name="generated-"*station*".txt";path=joinpath(input,name)
        bytes=Vector{UInt8}(codeunits(join(lines,"\n")*"\n"));R._exclusive_write(path,bytes)
        push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>station,
            "stream_labels"=>[station],"product"=>"quasi-definitive","start_inclusive"=>clock(START),
            "stop"=>clock(START+Minute(180)),"stop_inclusive"=>true,"bytes"=>length(bytes),"sha256"=>digest(bytes),
            "role"=>"generated_fixture","provenance"=>"generated_fixture"))
    end
    if layout=="mixed"
        bytes=Vector{UInt8}(codeunits(join([clock(START+Minute(i))*",1,-2,400,5" for i in 0:179],"\n")*"\n"))
        name="generated-wind.csv";path=joinpath(input,name);R._exclusive_write(path,bytes)
        push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>"wind","station"=>nothing,
            "stream_labels"=>["CMO","FRD"],"product"=>"OMNI_HRO_1MIN","start_inclusive"=>clock(START),
            "stop"=>clock(START+Minute(180)),"stop_inclusive"=>false,"bytes"=>length(bytes),"sha256"=>digest(bytes),
            "role"=>"generated_fixture","provenance"=>"generated_fixture"))
    end
    req=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>layout,
        "role"=>role,"runtime"=>string(VERSION),"data_root"=>input,"inputs"=>descriptors)
    path=joinpath(root,"source-request.json");R._exclusive_write(path,R.canonical_bytes(req))
    source=joinpath(root,"source");J.reconstruct(path,source)
    request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated","layout"=>layout,
        "role"=>role,"runtime"=>string(VERSION),"source"=>Dict("directory"=>source,
            "manifest_sha256"=>digest(read(joinpath(source,"manifest.json"))),"complete_sha256"=>digest(read(joinpath(source,"COMPLETE.json")))),
        "tables"=>[Dict("id"=>station*"-generated","station"=>station,"start_inclusive"=>clock(START),
            "stop_exclusive"=>clock(START+Minute(181)),"label_asof"=>clock(START+Minute(211))) for station in ("CMO","FRD")])
    path=joinpath(root,"case-request.json");R._exclusive_write(path,R.canonical_bytes(request));rp=digest(read(path))
    output=joinpath(root,"cases");Q.publish(path,output;request_sha256=rp)
    Q.read_verified(path,output;request_sha256=rp,manifest_sha256=digest(read(joinpath(output,"manifest.json"))),complete_sha256=digest(read(joinpath(output,"COMPLETE.json"))))
end
