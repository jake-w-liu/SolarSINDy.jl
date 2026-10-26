const CONTEXT_FIELDS=Set(("schema","evidence_mode","case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index","role","layout","table",
    "point_context_sha256","point_recipe_sha256","point_model_sha256","point_receipt_sha256","point_source_sha256","job_sha256","runtime",
    "start_inclusive","stop_exclusive","target_asof","first_evaluation_issue","point_training_stop","rows","input_sha256","records_sha256","ranks_sha256",
    "score_definition","rank_definition","availability_kind","measured_receipts_available"))
function _decode_records(bytes,n)
    need(length(bytes)==n*ROW_BYTES,"calibration record length")
    rows=Vector{Int}(undef,n);issues=Vector{DateTime}(undef,n);ready=Vector{DateTime}(undef,n)
    targets=Vector{Float64}(undef,n);predictions=similar(targets);scales=similar(targets);scores=similar(targets)
    io=IOBuffer(bytes)
    for i in 1:n
        raw=ltoh(read(io,UInt64));need(1<=raw<=MAX_ROWS,"calibration row number");rows[i]=Int(raw)
        for times in (issues,ready)
            value=reinterpret(Int64,ltoh(read(io,UInt64)))
            need(Dates.value(DateTime(1900))<=value<=Dates.value(DateTime(2200)),"calibration record clock range")
            times[i]=DateTime(Dates.UTInstant{Millisecond}(Millisecond(value)))
        end
        for values in (targets,predictions,scales,scores);values[i]=reinterpret(Float64,ltoh(read(io,UInt64)));end
        need(isfinite(targets[i]) && targets[i]>=0 && isfinite(predictions[i]) && isfinite(scales[i]) && scales[i]>0 && isfinite(scores[i]),"calibration record domain")
        need(reinterpret(UInt64,(log1p(targets[i])-predictions[i])/scales[i])==reinterpret(UInt64,scores[i]),"calibration score formula differs")
    end
    need(issorted(rows)&&allunique(rows)&&issorted(issues)&&allunique(issues)&&issorted(ready),"calibration record chronology")
    (;rows,issues,ready,targets,predictions,scales,scores)
end
function _checked(value::CalibrationCapsule,point,expected::ExpectedCalibration)
    pins=_point_pins(point);pc=pins.checked.context
    context_bytes=Vector{UInt8}(codeunits(value.context));receipt_bytes=Vector{UInt8}(codeunits(value.receipt))
    context=R.decode_canonical(context_bytes;limit=131072);receipt=R.decode_canonical(receipt_bytes;limit=65536)
    actual=get(context,"evidence_mode",nothing)=="actual_archival_cache"
    fields=copy(CONTEXT_FIELDS);actual && push!(fields,"case_read_admission_sha256");J._object(context,fields)
    need(context["schema"]=="source_bound_calibration_context_v1" && context["evidence_mode"]==(actual ? "actual_archival_cache" : "generated") &&
        pins.checked.actual===actual && digest(context_bytes)==expected.context_sha256 && context["input_sha256"]==expected.input_sha256,"calibration external context differs")
    need(context["case_request_sha256"]==pc["case_request_sha256"] && context["case_manifest_sha256"]==pc["case_manifest_sha256"] &&
        context["case_complete_sha256"]==pc["case_complete_sha256"] && F.isint(context["table_index"]) && context["table_index"]==pc["table_index"] &&
        context["role"]==pc["role"] && context["layout"]==pc["layout"] && context["table"]==pc["table"],"calibration point/case binding differs")
    need(context["point_context_sha256"]==point.expected.context_sha256 && context["point_recipe_sha256"]==point.expected.recipe_sha256 &&
        context["point_model_sha256"]==pins.model && context["point_receipt_sha256"]==pins.receipt && context["point_source_sha256"]==point.expected.source_sha256 &&
        context["job_sha256"]==job_identity() && context["runtime"]==string(VERSION),"calibration point/code identity differs")
    need(context["score_definition"]=="(log1p(target)-point_log_prediction)/case_scale" &&
        context["rank_definition"]=="stable_MergeSort_isless_Float64_chronological_indices" &&
        context["availability_kind"]=="assumed_completed_source_intervals" && context["measured_receipts_available"]===false,"calibration convention differs")
    n=context["rows"];need(F.isint(n)&&2<=n<=MAX_ROWS,"calibration row bound")
    records=Vector{UInt8}(codeunits(value.records));ranks=Vector{UInt8}(codeunits(value.ranks))
    need(digest(records)==context["records_sha256"] && digest(ranks)==context["ranks_sha256"] && length(ranks)==4n,"calibration payload pins differ")
    arrays=_decode_records(records,n)
    order=Int.(ltoh.(reinterpret(UInt32,ranks)))
    need(order==sortperm(arrays.scores;alg=MergeSort),"calibration stable rank differs")
    start=F._fit_clock(context["start_inclusive"]);stop=F._fit_clock(context["stop_exclusive"])
    asof=F._fit_clock(context["target_asof"]);first_issue=F._fit_clock(context["first_evaluation_issue"])
    training_stop=F._fit_clock(context["point_training_stop"])
    table_start=J._clock(context["table"]["start_inclusive"]);table_stop=J._clock(context["table"]["stop_exclusive"])
    table_asof=J._clock(context["table"]["label_asof"])
    need(context["point_training_stop"]==pc["stop_exclusive"] && training_stop<=start<stop<=asof && stop<=first_issue &&
        all(t->start<=t<stop,arrays.issues) && all(t->t<stop && t<=asof,arrays.ready),"calibration split/readiness differs")
    need(table_start<=start<stop<=table_stop && asof<=table_asof &&
        all(i->D.whole(arrays.issues[i]) && Dates.value(arrays.issues[i])-Dates.value(table_start)==60000*(arrays.rows[i]-1),1:n),
        "calibration table/schedule identity differs")
    need(all(i->arrays.ready[i]==D.CMR.issue_timing(arrays.issues[i],D.CMR.ReplayPolicy()).label_assumed_ready,1:n),"calibration label clock differs")
    actual && need(DateTime(2017)<=start<stop<=DateTime(2018),"published archival calibration outside 2017")
    fields=Set(("schema","complete","context_sha256","input_sha256","records_sha256","ranks_sha256","actual_source_access","actual_model_fit_performed",
        "provider_receipts_available","software_qualification"));actual && push!(fields,"authorization");J._object(receipt,fields)
    need(receipt["schema"]=="source_bound_calibration_receipt_v1" && receipt["complete"]===true && receipt["context_sha256"]==expected.context_sha256 &&
        receipt["input_sha256"]==expected.input_sha256 && receipt["records_sha256"]==context["records_sha256"] && receipt["ranks_sha256"]==context["ranks_sha256"] &&
        receipt["actual_source_access"]===actual && receipt["actual_model_fit_performed"]===false && receipt["provider_receipts_available"]===false &&
        receipt["software_qualification"]===false,"calibration receipt differs")
    if actual
        auth=receipt["authorization"]
        J._object(auth,Set(("schema","head","grant_sha256","source_proof_sha256","qualification_sha256","case_read_admission_sha256",
            "producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256","calibration_authorized",
            "actual_model_fit_authorized","provider_receipts_available","attempt_sha256")))
        need(auth["schema"]=="archive_calibration_admission_v1" && auth["calibration_authorized"]===true && auth["actual_model_fit_authorized"]===false &&
            auth["provider_receipts_available"]===false && auth["case_read_admission_sha256"]==J._sha(context["case_read_admission_sha256"]) &&
            auth["producer_grant_sha256"]==pins.checked.receipt["authorization"]["grant_sha256"] &&
            auth["producer_source_proof_sha256"]==pins.checked.receipt["authorization"]["source_proof_sha256"] &&
            auth["producer_qualification_sha256"]==pins.checked.receipt["authorization"]["qualification_sha256"],"calibration archival admission differs")
        need(auth["head"] isa String && occursin(r"\A[0-9a-f]{40}\z",auth["head"]),"calibration source head")
        for key in ("grant_sha256","source_proof_sha256","qualification_sha256","attempt_sha256");J._sha(auth[key]);end
    end
    (;context,receipt,context_bytes,receipt_bytes,records,ranks,arrays,order,actual,first_issue)
end
"""Project checked scores into the existing completed-source replay calibration.

This projection grants no source access, new fit, empirical coverage or skill.
"""
function completed_calibration(value::CalibrationCapsule,point,expected::ExpectedCalibration;first_issue)
    v=_checked(value,point,expected)
    need(first_issue isa DateTime && v.first_issue<=first_issue,"replay starts before declared evaluation")
    cal=D.CompletedCalibration(v.arrays.scores,v.arrays.ready;model_schema=D.POINT_SCHEMA,
        point_model_sha256=v.context["point_model_sha256"],purpose=v.actual ? :refitted : :synthetic,
        mode=v.context["layout"]=="ground-only" ? :ground11 : :mixed8)
    D.check_calibration(cal,first_issue,cal.policy,cal.mode);cal
end
function _member(name,bytes)
    Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>digest(bytes))
end
"""Publish bounded binary score shards, rank indices and canonical metadata.

The completion marker is written last; partial outputs are retained on failure.
"""
function publish_calibration(value::CalibrationCapsule,point,expected::ExpectedCalibration,directory)
    v=_checked(value,point,expected);path=abspath(directory);J._ordinary(dirname(path);directory=true)
    need(!ispath(path)&&!islink(path),"calibration publication must be fresh");mkdir(path);J._ordinary(path;directory=true)
    entries=Any[]
    for (name,bytes) in (("context.json",v.context_bytes),("receipt.json",v.receipt_bytes),("ranks.bin",v.ranks))
        R._exclusive_write(joinpath(path,name),bytes);push!(entries,_member(name,bytes))
    end
    for (i,start) in enumerate(1:ROWS_PER_SHARD:v.context["rows"])
        stop=min(start+ROWS_PER_SHARD-1,v.context["rows"])
        bytes=v.records[(start-1)*ROW_BYTES+1:stop*ROW_BYTES]
        name="scores-"*lpad(i,5,'0')*".bin"
        R._exclusive_write(joinpath(path,name),bytes);push!(entries,_member(name,bytes))
    end
    manifest=R.canonical_bytes(Dict("schema"=>"source_bound_calibration_manifest_v1","context_sha256"=>expected.context_sha256,
        "rows"=>v.context["rows"],"record_bytes"=>ROW_BYTES,"rows_per_shard"=>ROWS_PER_SHARD,"members"=>entries);limit=65536)
    R._exclusive_write(joinpath(path,"manifest.json"),manifest)
    marker=R.canonical_bytes(Dict("schema"=>"source_bound_calibration_complete_v1","manifest_sha256"=>digest(manifest),
        "context_sha256"=>expected.context_sha256,"receipt_sha256"=>digest(v.receipt_bytes),"members"=>length(entries)+2,
        "actual_model_fit_performed"=>false);limit=4096)
    _checked(value,point,expected);R._exclusive_write(joinpath(path,"COMPLETE.json"),marker)
    (;manifest_sha256=digest(manifest),complete_sha256=digest(marker),receipt_sha256=digest(v.receipt_bytes))
end
"""Read only bounded data with external context, manifest, receipt and completion pins."""
function read_calibration(directory,point,expected::ExpectedCalibration;manifest_sha256,receipt_sha256,complete_sha256)
    J._ordinary(directory;directory=true);names=readdir(directory);identities=Dict{String,Any}()
    function member(name,limit)
        text,id=FS._read(joinpath(directory,name),limit);identities[name]=id;Vector{UInt8}(codeunits(text))
    end
    manifest_bytes=member("manifest.json",65536);marker_bytes=member("COMPLETE.json",4096)
    need(digest(manifest_bytes)==J._sha(manifest_sha256) && digest(marker_bytes)==J._sha(complete_sha256),"calibration external publication pin differs")
    manifest=R.decode_canonical(manifest_bytes;limit=65536);marker=R.decode_canonical(marker_bytes;limit=4096)
    J._object(manifest,Set(("schema","context_sha256","rows","record_bytes","rows_per_shard","members")))
    n=manifest["rows"];need(manifest["schema"]=="source_bound_calibration_manifest_v1" && manifest["context_sha256"]==expected.context_sha256 &&
        F.isint(n)&&2<=n<=MAX_ROWS && F.isint(manifest["record_bytes"])&&manifest["record_bytes"]==ROW_BYTES &&
        F.isint(manifest["rows_per_shard"])&&manifest["rows_per_shard"]==ROWS_PER_SHARD,"calibration manifest geometry")
    expected_names=vcat(["context.json","receipt.json","ranks.bin"],["scores-"*lpad(i,5,'0')*".bin" for i in 1:cld(n,ROWS_PER_SHARD)])
    entries=manifest["members"];need(entries isa AbstractVector && length(entries)==length(expected_names),"calibration manifest members")
    need(Set(names)==Set(vcat(expected_names,["manifest.json","COMPLETE.json"])) && length(names)==length(expected_names)+2,"calibration fileset differs")
    bodies=Dict{String,Vector{UInt8}}()
    for (i,entry) in enumerate(entries)
        J._object(entry,Set(("name","bytes","sha256")));name=expected_names[i]
        limit=i==1 ? 131072 : i==2 ? 65536 : i==3 ? 4MAX_ROWS : ROWS_PER_SHARD*ROW_BYTES
        need(entry["name"]==name && F.isint(entry["bytes"]) && 0<entry["bytes"]<=limit,"calibration member descriptor")
        bytes=member(name,limit);need(length(bytes)==entry["bytes"] && digest(bytes)==J._sha(entry["sha256"]),"calibration member content differs")
        i>=4 && need(length(bytes)==min(ROWS_PER_SHARD,n-(i-4)*ROWS_PER_SHARD)*ROW_BYTES,"calibration score shard geometry")
        bodies[name]=bytes
    end
    J._object(marker,Set(("schema","manifest_sha256","context_sha256","receipt_sha256","members","actual_model_fit_performed")))
    need(marker["schema"]=="source_bound_calibration_complete_v1" && marker["manifest_sha256"]==manifest_sha256 &&
        marker["context_sha256"]==expected.context_sha256 && marker["receipt_sha256"]==J._sha(receipt_sha256)==digest(bodies["receipt.json"]) &&
        F.isint(marker["members"]) && marker["members"]==length(names) && marker["actual_model_fit_performed"]===false,"calibration completion binding differs")
    records=reduce(vcat,[bodies[name] for name in expected_names[4:end]])
    capsule=CalibrationCapsule(String(copy(bodies["context.json"])),records,bodies["ranks.bin"],String(copy(bodies["receipt.json"])),_Token())
    checked=_checked(capsule,point,expected);need(checked.context["rows"]==n,"calibration manifest row count differs")
    need(readdir(directory)==names && all(J._identity(joinpath(directory,name))==id for (name,id) in identities),"calibration publication changed during read")
    capsule
end
