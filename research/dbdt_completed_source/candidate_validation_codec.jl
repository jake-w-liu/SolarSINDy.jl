"""Fresh bounded data-only publications; COMPLETE is always written last."""
struct _InputToken end
struct FitInput
    descriptor_json::String
    summary_json::String
    receipt_json::String
    pins::NamedTuple
    directory::String
    FitInput(d,s,r,p,path,::_InputToken)=new(String(d),String(s),String(r),p,String(path))
end
function Base.getproperty(value::FitInput,key::Symbol)
    key===:descriptor && return decode(Vector{UInt8}(codeunits(getfield(value,:descriptor_json))))
    key===:summary && return decode(Vector{UInt8}(codeunits(getfield(value,:summary_json))))
    getfield(value,key)
end
struct FitPublication
    point::F.BoundFit
    directory::String
    pins::NamedTuple
end
struct ValidationPins
    request_sha256::String
    context_sha256::String
    metric_sha256::String
    receipt_sha256::String
    manifest_sha256::String
    complete_sha256::String
    function ValidationPins(request,context,metric,receipt,manifest,complete)
        new(map(J._sha,(request,context,metric,receipt,manifest,complete))...)
    end
end
struct _ValidationToken end
struct ValidationArtifact
    context_json::String
    metric_json::String
    receipt_json::String
    records::String
    ranks::String
    pins::ValidationPins
    directory::String
    ValidationArtifact(c,m,r,records,ranks,p,d,::_ValidationToken)=new(String(c),String(m),String(r),String(copy(records)),String(copy(ranks)),p,String(d))
end
function _assert_fresh_destination(directory)
    need(directory isa AbstractString && isabspath(directory),"candidate publication path must be absolute")
    J._ordinary(dirname(directory);directory=true)
    need(!ispath(directory) && !islink(directory),"candidate publication must be fresh")
end
function _fresh_directory(directory)
    _assert_fresh_destination(directory)
    mkdir(directory);J._ordinary(directory;directory=true)
end
_entry(name,bytes)=Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>digest(bytes))
function _publish_payloads(directory,payloads,schema;limit,closecheck)
    _fresh_directory(directory);entries=Any[];total=0
    for (name,bytes) in payloads
        total+=length(bytes);need(total<=limit,"candidate publication total byte bound")
        R._exclusive_write(joinpath(directory,name),bytes);push!(entries,_entry(name,bytes))
    end
    manifest=canonical(Dict("schema"=>schema*"_manifest_v1","members"=>entries);limit=65536)
    need(total+length(manifest)+4096<=limit,"candidate publication metadata byte bound")
    R._exclusive_write(joinpath(directory,"manifest.json"),manifest)
    marker=canonical(Dict("schema"=>schema*"_complete_v1","manifest_sha256"=>digest(manifest),
        "members"=>length(entries)+2,"bytes_without_marker"=>total+length(manifest),"native_fits"=>0);limit=4096)
    closecheck()
    R._exclusive_write(joinpath(directory,"COMPLETE.json"),marker)
    (;manifest_sha256=digest(manifest),complete_sha256=digest(marker))
end
function _publish_input(d,summary,auth,directory;attempt_bytes,closecheck=()->nothing)
    request=canonical(d);body=canonical(summary)
    receipt=canonical(Dict("schema"=>"candidate_fit_input_receipt_v1","complete"=>true,
        "descriptor_sha256"=>digest(request),"summary_sha256"=>digest(body),"authorization"=>auth,
        "attempt_sha256"=>digest(attempt_bytes),"native_fits"=>0,"actual_source_access"=>d["actual_source_access"],"software_qualification"=>false);limit=65536)
    pins=_publish_payloads(directory,[("request.json",request),("summary.json",body),("receipt.json",receipt),("attempt.json",attempt_bytes)],
        "candidate_fit_input";limit=MAX_INPUT_BYTES,closecheck)
    allpins=(;descriptor_sha256=digest(request),summary_sha256=digest(body),receipt_sha256=digest(receipt),pins...)
    input=read_fit_input(directory;allpins...)
    (;input,pins=allpins)
end
function _read_payloads(directory,schema,manifest_sha256,complete_sha256;maximum,limit)
    J._ordinary(directory;directory=true);names=readdir(directory);need(2<=length(names)<=maximum,"candidate member count bound")
    identities=Dict{String,Any}()
    function read_member(name,bound)
        text,id=FS._read(joinpath(directory,name),bound);identities[name]=id;Vector{UInt8}(codeunits(text))
    end
    mb=read_member("manifest.json",65536);cb=read_member("COMPLETE.json",4096)
    need(digest(mb)==J._sha(manifest_sha256) && digest(cb)==J._sha(complete_sha256),"candidate outside manifest/COMPLETE pins")
    manifest=decode(mb;limit=65536);complete=decode(cb;limit=4096)
    J._object(manifest,Set(("schema","members")));J._object(complete,Set(("schema","manifest_sha256","members","bytes_without_marker","native_fits")))
    need(manifest["schema"]==schema*"_manifest_v1" && complete["schema"]==schema*"_complete_v1" &&
        complete["manifest_sha256"]==manifest_sha256 && F.isint(complete["members"]) && complete["members"]==length(names) &&
        F.isint(complete["native_fits"]) && complete["native_fits"]==0,"candidate publication marker")
    entries=manifest["members"];need(entries isa AbstractVector && length(entries)==length(names)-2,"candidate manifest geometry")
    seen=Set(("manifest.json","COMPLETE.json"));bodies=Dict{String,Vector{UInt8}}();total=length(mb)
    for entry in entries
        J._object(entry,Set(("name","bytes","sha256")));name=entry["name"]
        need(name isa String && occursin(r"\A[a-z][a-z0-9.-]{0,63}\z",name) && !(name in seen),"candidate member name")
        bound=name=="request.json" || name=="context.json" || name=="summary.json" ? 131072 :
            name=="receipt.json" ? 65536 : name in ("metric.json","attempt.json") ? 16384 : name=="ranks.bin" ? 4MAX_ROWS :
            occursin(r"\Ascores-[0-9]{5}\.bin\z",name) ? 8192*56 : 0
        need(F.isint(entry["bytes"]) && 0<entry["bytes"]<=bound,"candidate member byte bound")
        bytes=read_member(name,bound);need(length(bytes)==entry["bytes"] && digest(bytes)==J._sha(entry["sha256"]),"candidate member SHA/size")
        total+=length(bytes);need(total+length(cb)<=limit,"candidate total byte bound");push!(seen,name);bodies[name]=bytes
    end
    need(Set(names)==seen && F.isint(complete["bytes_without_marker"]) && complete["bytes_without_marker"]==total,"candidate fileset/total")
    function unchanged()
        need(readdir(directory)==names && all(J._identity(joinpath(directory,n))==id for (n,id) in identities),"candidate publication changed while reading")
    end
    unchanged();(;bodies,unchanged)
end
function _check_input(input::FitInput)
    d=input.descriptor;s=input.summary;receipt=decode(Vector{UInt8}(codeunits(input.receipt_json));limit=65536)
    need(digest(codeunits(input.descriptor_json))==input.pins.descriptor_sha256 && digest(codeunits(input.summary_json))==input.pins.summary_sha256 &&
        digest(codeunits(input.receipt_json))==input.pins.receipt_sha256,"candidate fit-input outside binding")
    J._object(d,Set(("schema","operation","case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index",
        "station","role","layout","candidate_id","recipe_sha256","grid_sha256","split","runtime","phase_period_minutes",
        "policy_sha256","job_sha256","point_source_sha256","actual_source_access","case_read_admission_sha256","selection_sha256","work_budget","point")))
    need(d["schema"]=="candidate_known_input_descriptor_v1" && d["operation"] in ("fixed_candidate_2017_fit_input_only","selected_R2_2017_refit_input_only") &&
        d["recipe_sha256"]==recipe_for(d["candidate_id"]).sha256 && d["grid_sha256"]==M.grid_identity(:R2) &&
        d["layout"]=="ground11" && d["phase_period_minutes"]==31 && d["policy_sha256"]==policy_identity() && d["job_sha256"]==job_identity() &&
        d["point"]===nothing && canonical(d["work_budget"])==canonical(work_budget()) &&
        d["actual_source_access"] isa Bool && d["runtime"]==string(VERSION) && d["station"] in S.STATIONS &&
        F.isint(d["table_index"]) && 1<=d["table_index"]<=16,"candidate input descriptor semantics")
    J._object(s,Set(("schema","descriptor_sha256","input_sha256","rows","first_row","last_row","cohort_sha256","counts","table","point_source_sha256","native_fits","actual_source_access")))
    need(s["schema"]=="candidate_fit_input_summary_v1" && s["descriptor_sha256"]==input.pins.descriptor_sha256 &&
        F.isint(s["rows"]) && 2<=s["rows"]<=MAX_ROWS && F.isint(s["first_row"]) && F.isint(s["last_row"]) &&
        1<=s["first_row"]<=s["last_row"]<=MAX_ROWS && s["rows"]<=s["last_row"]-s["first_row"]+1 &&
        F.isint(s["native_fits"]) && s["native_fits"]==0 && s["actual_source_access"]===d["actual_source_access"] &&
        s["point_source_sha256"]==F.source_identity(),"candidate input summary semantics")
    J._sha(s["input_sha256"]);J._sha(s["cohort_sha256"])
    _check_counts(s["counts"],s["rows"])
    J._object(receipt,Set(("schema","complete","descriptor_sha256","summary_sha256","authorization","attempt_sha256","native_fits","actual_source_access","software_qualification")))
    need(receipt["schema"]=="candidate_fit_input_receipt_v1" && receipt["complete"]===true &&
        receipt["descriptor_sha256"]==input.pins.descriptor_sha256 && receipt["summary_sha256"]==input.pins.summary_sha256 &&
        receipt["actual_source_access"]===d["actual_source_access"] && receipt["software_qualification"]===false &&
        receipt["native_fits"]===0 && (receipt["authorization"]!==nothing)===d["actual_source_access"],"candidate input receipt")
    nothing
end
function _check_counts(counts,n)
    J._object(counts,Set(("scheduled","outside_issue","not_ready_stop","not_ready_asof","feature_rejected","eligible_features","label_rejected","accepted")))
    need(all(x->F.isint(x)&&0<=x<=MAX_ROWS,values(counts)) && counts["accepted"]==n &&
        counts["eligible_features"]==counts["accepted"]+counts["label_rejected"] && counts["scheduled"]==
        sum(counts[k] for k in ("outside_issue","not_ready_stop","not_ready_asof","feature_rejected","eligible_features")),"candidate exclusion counts")
end
function read_fit_input(directory;descriptor_sha256,summary_sha256,receipt_sha256,manifest_sha256,complete_sha256)
    pins=(;descriptor_sha256=J._sha(descriptor_sha256),summary_sha256=J._sha(summary_sha256),receipt_sha256=J._sha(receipt_sha256),
        manifest_sha256=J._sha(manifest_sha256),complete_sha256=J._sha(complete_sha256))
    p=_read_payloads(directory,"candidate_fit_input",manifest_sha256,complete_sha256;maximum=6,limit=MAX_INPUT_BYTES);b=p.bodies
    need(Set(keys(b))==Set(("request.json","summary.json","receipt.json","attempt.json")),"candidate input fileset")
    need(digest(b["request.json"])==descriptor_sha256 && digest(b["summary.json"])==summary_sha256 && digest(b["receipt.json"])==receipt_sha256,"candidate input external member pins")
    input=FitInput(String(copy(b["request.json"])),String(copy(b["summary.json"])),String(copy(b["receipt.json"])),pins,directory,_InputToken())
    _check_input(input);receipt=decode(b["receipt.json"];limit=65536)
    need(digest(b["attempt.json"])==J._sha(receipt["attempt_sha256"]),"candidate input attempt binding")
    receipt["authorization"]===nothing || need(receipt["authorization"]["attempt_sha256"]==receipt["attempt_sha256"],"candidate input authority/attempt binding")
    p.unchanged();input
end
function _publish_validation(context,records,ranks,metric,auth,directory;attempt_bytes=UInt8[],closecheck=()->nothing)
    request=canonical(context["descriptor"]);cb=canonical(context);mb=canonical(metric;limit=16384)
    receipt=canonical(Dict("schema"=>"candidate_validation_receipt_v1","complete"=>true,"context_sha256"=>digest(cb),
        "authorization"=>auth,"attempt_sha256"=>digest(attempt_bytes),"actual_source_access"=>context["descriptor"]["actual_source_access"],"native_fits"=>0,"software_qualification"=>false);limit=65536)
    payloads=[("request.json",request),("context.json",cb),("metric.json",mb),("receipt.json",receipt),("ranks.bin",ranks)]
    n=context["calibration_numeric_context"]["rows"];need(2<=n<=MAX_ROWS && length(records)==56n && length(ranks)==4n,"candidate validation geometry")
    for (i,start) in enumerate(1:8192:n)
        stop=min(start+8191,n);push!(payloads,("scores-"*lpad(string(i),5,'0')*".bin",records[(start-1)*56+1:stop*56]))
    end
    p=_publish_payloads(directory,payloads,"candidate_validation";limit=MAX_VALIDATION_BYTES,closecheck)
    pins=ValidationPins(digest(request),digest(cb),digest(mb),digest(receipt),p.manifest_sha256,p.complete_sha256)
    (;artifact=read_validation(directory,pins),pins)
end
function read_validation(directory,pins::ValidationPins)
    p=_read_payloads(directory,"candidate_validation",pins.manifest_sha256,pins.complete_sha256;maximum=81,limit=MAX_VALIDATION_BYTES);b=p.bodies
    for (name,pin) in (("request.json",pins.request_sha256),("context.json",pins.context_sha256),("metric.json",pins.metric_sha256),("receipt.json",pins.receipt_sha256))
        need(haskey(b,name) && digest(b[name])==pin,"candidate validation outside metadata pin")
    end
    context=decode(b["context.json"]);metric=decode(b["metric.json"];limit=16384);receipt=decode(b["receipt.json"];limit=65536)
    J._object(context,Set(("schema","descriptor","descriptor_sha256","calibration_numeric_context","cohort_sha256","counts","records_sha256","ranks_sha256","metric_sha256")))
    need(context["schema"]=="candidate_validation_context_v1" && canonical(context["descriptor"])==b["request.json"] &&
        context["descriptor_sha256"]==pins.request_sha256 && context["metric_sha256"]==pins.metric_sha256,"candidate validation context")
    n=context["calibration_numeric_context"]["rows"];need(F.isint(n) && 2<=n<=MAX_ROWS,"candidate validation row count")
    shards=["scores-"*lpad(string(i),5,'0')*".bin" for i in 1:cld(n,8192)]
    need(Set(keys(b))==Set(vcat(["request.json","context.json","metric.json","receipt.json","ranks.bin"],shards)),"candidate validation shard inventory")
    for (i,name) in enumerate(shards);need(length(b[name])==56*min(8192,n-(i-1)*8192),"candidate validation shard length");end
    records=reduce(vcat,[b[name] for name in shards]);ranks=b["ranks.bin"]
    need(digest(records)==context["records_sha256"] && digest(ranks)==context["ranks_sha256"] && length(ranks)==4n,"candidate validation data binding")
    rows=K._decode_records(records,n);need(Int.(ltoh.(reinterpret(UInt32,ranks)))==sortperm(rows.scores;alg=MergeSort),"candidate validation stable ranks")
    _check_counts(context["counts"],n)
    need(metric["clipped"]===false && metric["scale_normalized"]===false && F.isint(metric["rows"]),"candidate metric strict types")
    need(metric==Dict("schema"=>"candidate_raw_log_mse_v1","definition"=>"Statistics.mean(abs2,log1p(raw_target)-raw_log_prediction)",
        "loss_bits"=>C.bits(log_mse(rows.targets,rows.predictions)),"rows"=>n,"clipped"=>false,"scale_normalized"=>false),"candidate raw metric reconstruction")
    J._object(receipt,Set(("schema","complete","context_sha256","authorization","attempt_sha256","actual_source_access","native_fits","software_qualification")))
    need(receipt["schema"]=="candidate_validation_receipt_v1" && receipt["complete"]===true && receipt["context_sha256"]==pins.context_sha256 &&
        receipt["actual_source_access"]===context["descriptor"]["actual_source_access"] && receipt["native_fits"]===0 &&
        receipt["software_qualification"]===false && (receipt["authorization"]!==nothing)===receipt["actual_source_access"],"candidate validation receipt")
    p.unchanged();ValidationArtifact(String(copy(b["context.json"])),String(copy(b["metric.json"])),String(copy(b["receipt.json"])),records,ranks,pins,directory,_ValidationToken())
end
