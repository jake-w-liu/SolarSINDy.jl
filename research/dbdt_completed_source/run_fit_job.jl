# Fixed offline synthetic job entry. No actual parser, dynamic script/type,
# arbitrary callback, fetch, serving path or scientific outcome selection.
module SolarSINDy
include(joinpath(@__DIR__,"..","..","src","completed_minute_replay.jl"))
include(joinpath(@__DIR__,"..","..","src","completed_source_dbdt","CompletedSourceDBDT.jl"))
end
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts

function caller_pins(path)
    rows=R._array(R.decode_canonical(R.bounded_read(path)))
    mapping=Dict{String,String}()
    for row in rows
        pair=R._array(row);R.require(length(pair)==2,"invalid caller pin pair")
        pin=R.sha_string(pair[1]);R.require(!haskey(mapping,pin),"duplicate caller resolution pin")
        mapping[pin]=R.bounded_string(pair[2];nonempty=true)
    end
    R._pins(mapping)
end

function main(arguments)
    if length(arguments)==2 && arguments[1]=="inspect-runtime"
        planned=R.decode_canonical(R.bounded_read(arguments[2]))
        write(stdout,R.canonical_bytes(R._inspect_runtime(planned)))
        return
    elseif length(arguments)==1 && arguments[1]=="source-inventory"
        write(stdout,R._source_inventory_bytes());return
    end
    R.require(length(arguments)==5 && all(isabspath,arguments),"worker needs absolute protocol/request/resolver/inventory/output paths")
    protocolpath,requestpath,resolverpath,inventorypath,output=arguments
    bytes=R.bounded_read(protocolpath;limit=R.MAX_METADATA_BYTES)
    protocol=R.RefitProtocol(bytes;expected_spec_sha256=R.digest(bytes))
    requestbytes=R.bounded_read(requestpath;limit=R.MAX_METADATA_BYTES);request=R.decode_canonical(requestbytes)
    pins=caller_pins(resolverpath);runtime=R._inspect_runtime(request["required_runtime"]["argv"])
    R.require(request["required_runtime"]==runtime,"inspected native runtime differs from request")
    R.require(abspath(output)==abspath(request["work_directory"]) && isdir(output) && isempty(readdir(output)),"native output must be exact empty caller-owned workdir")
    inventory=R.bounded_read(inventorypath);inventorypin=R.digest(inventory)
    common=R._job_common(request,protocol,pins,runtime)
    R.require(common["code_hashes"]["_source_inventory"]==inventorypin,"native inventory pin differs")
    R._inventory_check(inventory,common;check_current=true)
    chunks=R._execute_request(request,protocol,pins,runtime)
    R._inventory_check(inventory,common;check_current=true)
    R.require(R.bounded_read(protocolpath;limit=R.MAX_METADATA_BYTES)==bytes && R.bounded_read(requestpath;limit=R.MAX_METADATA_BYTES)==requestbytes,"native protocol/request changed")
    for (i,chunk) in enumerate(chunks)
        name=length(chunks)==1 ? "payload.json" : R._row_name(i,"payload")
        R._exclusive_write(joinpath(output,name),R._jsonbytes(chunk))
    end
    internal=Dict{String,Any}("schema"=>"dbdt_completed_job_internal_receipt_v1","operation"=>request["operation"],
        "protocol_sha256"=>protocol.sha256,"input_manifest_sha256_or_null"=>request["input_manifest_pin_or_null"],
        "input_data_sha256"=>request["input_data_pin"],"payload_sha256"=>R._payload_digest(chunks),
        "loaded_code_hashes"=>common["code_hashes"],"runtime_id"=>common["runtime_id"],"completed"=>true)
    R._exclusive_write(joinpath(output,"internal-receipt.json"),R.canonical_bytes(internal))
    R._exclusive_write(joinpath(output,"runtime.json"),R.canonical_bytes(runtime))
    println("completed synthetic native operation ",request["operation"]," payload=",internal["payload_sha256"])
end
abspath(PROGRAM_FILE)==(@__FILE__) && main(ARGS)
