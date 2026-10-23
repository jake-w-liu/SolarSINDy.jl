# Fixed exclusive publication after the separately owned native job terminates.
include(joinpath(@__DIR__,"run_fit_job.jl"))
function finalize(arguments)
    R.require(length(arguments)>=4,"fixed publication action/protocol/resolver/destination required")
    action,protocolpath,resolverpath,destination=arguments[1:4]
    R.require(action in ("native","source","schedule","input","point","calibration"),"unsupported fixed publication action")
    R.require(all(isabspath,arguments[2:end]),"publication inputs must be absolute paths")
    bytes=R.bounded_read(protocolpath;limit=R.MAX_METADATA_BYTES)
    protocol=R.RefitProtocol(bytes;expected_spec_sha256=R.digest(bytes));pins=caller_pins(resolverpath)
    if action=="native"
        R.require(length(arguments)==8,"native publication arity differs")
        requestpath,work,receiptpath,logpath=arguments[5:8]
        request=R.decode_canonical(R.bounded_read(requestpath;limit=R.MAX_METADATA_BYTES))
        names=readdir(work);payloadnames=R._chunk_catalog(names,"payload";single="payload.json")
        R.require(Set(names)==Set(vcat(payloadnames,["internal-receipt.json","runtime.json"])),"native output member set differs")
        chunks=Tuple(String(R.bounded_read(joinpath(work,name))) for name in payloadnames)
        internal=R.decode_canonical(R.bounded_read(joinpath(work,"internal-receipt.json")))
        external=R.decode_canonical(R.bounded_read(receiptpath;limit=R.MAX_METADATA_BYTES))
        R.require(R.decode_canonical(R.bounded_read(joinpath(work,"runtime.json")))==external["runtime"],"native runtime output differs from external receipt")
        bundle=R._publish_native_job(destination,request,chunks,internal,external,String(R.bounded_read(logpath));protocol,pins)
    else
        R.require(length(arguments)==6,"scientific publication arity differs")
        primarypath,independentpath=arguments[5:6]
        producer=R._load_native_job(primarypath;protocol,pins);checker=R._load_native_job(independentpath;protocol,pins)
        draft=R._draft_from_job(producer,pins)
        if action=="source"
            bundle=R._publish_source_audit(destination,draft,producer,checker;pins)
        elseif action=="schedule"
            bundle=R._publish_schedule(destination,draft,producer,checker;pins)
        elseif action=="input"
            bundle=R._publish_input(destination,draft,producer,checker;pins)
        elseif action=="point"
            input=R._load_input(R._resolve(pins,R._draft_member(draft,"fit-receipt.json")["input_manifest_sha256"];directory=true);protocol,pins)
            bundle=R.publish_point(destination,draft;native_receipt=producer,independent_input_receipt=R._member_json(input,"independent-input-check.json"),independent_fit_receipt=checker,pins)
        else
            bundle=R.publish_calibration(destination,draft;native_receipt=producer,independent_cal_receipt=checker,pins)
        end
    end
    println("committed synthetic artifact ",bundle.kind," manifest=",bundle.manifest_sha256)
end
abspath(PROGRAM_FILE)==@__FILE__ && finalize(ARGS)
