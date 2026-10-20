module InitialReferenceImport

using Dates, SHA, JSON3, CSV, DataFrames, SolarSINDy
include(joinpath(@__DIR__,"..","initial_reference","DstInitialGBMReference.jl"))
include(joinpath(@__DIR__,"..","initial_reference","BoundInitialReferenceInputs.jl"))
const B=BoundInitialReferenceInputs
const F=SolarSINDy.IntervalMeanForecaster
const T=F.DstIntervalTraining
const A=F.DstIntervalAdapter
const PROJECT=dirname(dirname(@__DIR__))
const LOADED_SHA256=bytes2hex(sha256(read(@__FILE__)))
const LIMITS=B.Limits(total_bytes=2*1024^3,files=512)
const CONTRACT="db833f075e82ad5a15e10587e538c9688958a0fac180ca5713405da43d145ff7"
const HEADERS=vcat(["origin_row","issue","source_sha256","dst_delay_hours","driver_tag","driver_receipt",
    "anchor_tag","anchor_complete","anchor_receipt","anchor"],
    ["h$(h)_$name" for h in (4,6) for name in ("target_tag","target_complete","target_receipt","target",
        "scoreable","selection_eligible","prediction","status")])

export load_reference, reference_rows, ArtifactReference

"""Immutable locations/digests, never cached mutable reference metadata.

Actual L0 consumption additionally requires the matching independently admitted
original-source token. Synthetic artifacts retain their synthetic evidence mode.
"""
struct ArtifactReference
    directory::String
    completion_sha256::String
    audit_completion_sha256::String
    admitted::Any
end
need(ok,message)=ok || throw(ArgumentError(message))
isint(x)=x isa Integer && !(x isa Bool)
function sha(path)
    B.digest_file(path,LIMITS.file_bytes).sha256
end
function absolute_file(path)
    need(path isa AbstractString && isabspath(path),"absolute proof member required")
    parent=B.root_path(dirname(path))
    B.resolved(Dict("one"=>parent),"one",basename(path))
end
function json_at(root,name,digest)
    B.json_file(B.resolved(Dict("one"=>root),"one",name),digest,LIMITS)
end
function code_identity()
    B.own_guard()
    need(sha(@__FILE__)==LOADED_SHA256,"loaded importer source changed")
    result=Dict("src/interval_mean_forecaster/"*name=>digest for (name,digest) in F.source_identity())
    for name in ("tools/initial_reference_import/export_reference.py","tools/initial_reference_import/InitialReferenceImport.jl",
            "tools/initial_reference/DstInitialGBMReference.jl","tools/initial_reference/BoundInitialReferenceInputs.jl")
        result[name]=sha(B.resolved(Dict("code"=>B.root_path(PROJECT)),"code",name))
    end
    result
end
function file_inventory(root)
    files=Pair{String,String}[];bytes=0;directories=1
    for (directory,dirs,names) in walkdir(root;follow_symlinks=false)
        for name in dirs
            path=joinpath(directory,name)
            need(!islink(path) && !B.reparse(path),"linked proof directory")
            directories+=1;need(directories<=128,"proof directory budget")
        end
        for name in names
            relative=replace(relpath(joinpath(directory,name),root),'\\'=>'/')
            path=B.resolved(Dict("proof"=>root),"proof",relative)
            size=filesize(path)
            need(length(files)<512 && size<=LIMITS.file_bytes && size<=LIMITS.total_bytes-bytes,"proof inventory budget")
            bytes+=size;push!(files,relative=>path)
        end
    end
    need(allunique(lowercase.(first.(files))),"case-aliased proof inventory")
    Dict(name=>sha(path) for (name,path) in files)
end
function datehour(value)
    need(value isa AbstractString && occursin(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00:00(?:\.0{1,3})?\z",value),"exact-hour UTC field required")
    result=DateTime(value);need(floor(result,Hour)==result,"fractional reference hour");result
end
function hash_map(value;maximum=512)
    need(value isa AbstractDict && 1<=length(value)<=maximum && all(B.hex,values(value)),"bounded digest map required")
    Dict{String,String}(value)
end
function guard_files(pinned)
    need(length(pinned)<=1200,"combined proof file budget")
    total=0
    for (path,digest) in pinned
        checked=absolute_file(path);size=filesize(checked)
        need(size<=LIMITS.file_bytes && size<=LIMITS.total_bytes-total,"combined proof byte budget")
        total+=size;need(sha(checked)==digest,"serialized reference proof changed")
    end
end

function verified_metadata(token::ArtifactReference)
    own=code_identity();root=B.root_path(token.directory)
    need(B.hex(token.completion_sha256),"caller-pinned reference completion digest required")
    need(B.hex(token.audit_completion_sha256),"caller-pinned independent audit digest required")
    meta=json_at(root,"completion.json",token.completion_sha256)
    need(meta["schema"]=="source_bound_initial_reference_selection_v1" && meta["complete"]===true,"complete serialized selection reference required")
    need(meta["contract_sha256"]==CONTRACT && isint(meta["delay_hours"]) && meta["delay_hours"]==0,"only the initial L0 reference is supported")
    need(B.hex(meta["source_sha256"]),"invalid raw source digest")
    for name in ("new_fitting","scientific_verdict","serving_promotion");need(meta[name]===false,"selection reference scope differs");end
    need(meta["implementation_sources"]==own,"selection reference/importer implementation differs")
    need(meta["audit_completion_sha256"]==token.audit_completion_sha256,"independent audit trust anchor differs")
    mode=meta["evidence_mode"];need(mode in ("synthetic_only","exposed_development_reconstruction"),"unknown reference evidence mode")
    need(isint(meta["row_count"]) && 1<=meta["row_count"]<=500_000,"selection row budget")
    need(isint(meta["prefix_rows"]) && 1<=meta["prefix_rows"]<=500_000,"raw prefix row budget")
    need(meta["cohort_counts"] isa AbstractVector && length(meta["cohort_counts"])==2 &&
         all(x->B.integer(x,1),meta["cohort_counts"]),"two positive integer cohort counts required")
    outputs=hash_map(meta["output_sha256"])
    need(Set(keys(outputs))==Set(("selection_rows.csv",)),"exact selection output map required")
    need(file_inventory(root)==merge(outputs,Dict("completion.json"=>token.completion_sha256)),"selection output inventory differs")
    roots=Dict(name=>B.root_path(meta[name*"_root"]) for name in ("audit","reference","source"))
    allroots=[root;collect(values(roots))]
    for i in eachindex(allroots),j in i+1:length(allroots)
        a=B.canonical(allroots[i]);b=B.canonical(allroots[j])
        need(a!=b && !startswith(a,b*"/") && !startswith(b,a*"/"),"overlapping proof roots")
    end
    audit=json_at(roots["audit"],"completion.json",meta["audit_completion_sha256"])
    need(audit["schema"]=="independent_initial_reference_artifact_audit_v1" && audit["complete"]===true && audit["expected_mode"]==mode,"matching completed independent artifact audit required")
    for name in ("new_fitting","author_or_ml_imported","scientific_verdict","execution_lineage_qualification","serving_promotion")
        need(audit[name]===false,"independent audit scope differs")
    end
    need(B.zero(audit["native"]["native_exit"]) && audit["native"]["timed_out"]===false,"independent numerical process did not pass")
    audited_outputs=hash_map(audit["output_sha256"])
    need(file_inventory(roots["audit"])==merge(audited_outputs,Dict("completion.json"=>meta["audit_completion_sha256"])),"independent audit inventory changed")
    plan=json_at(roots["audit"],"prepared/audit-plan.json",audited_outputs["prepared/audit-plan.json"])
    need(plan["schema"]=="independent_initial_reference_audit_inputs_v1" && plan["preflight_complete"]===true && plan["complete"]===false,"independent prepared plan differs")
    need(plan["expected_mode"]==mode,"independent plan evidence mode differs")
    numerical=audit["numerical"]
    need(numerical["complete"]===true && numerical["expected_mode"]==mode &&
         numerical["input_plan_sha256"]==audited_outputs["prepared/audit-plan.json"],"independent numerical proof differs")
    need(B.integer(numerical["checks"],1) && B.integer(numerical["model_points"],1) && B.integer(numerical["fallback_points"]),"independent numerical counts differ")
    for object in (plan,numerical),name in ("new_fitting","author_or_ml_imported","scientific_verdict")
        need(object[name]===false,"independent prepared/numerical scope differs")
    end
    for role in ("reference","source")
        need(plan[role]==roots[role],"prepared proof location differs")
        key=role*"_completion_sha256"
        need(plan[key]==audit[key]==meta[key],"upstream completion identities differ")
    end
    pinned=Dict{String,String}()
    for (name,digest) in audited_outputs
        pinned[B.resolved(roots,"audit",name)]=digest
    end
    pinned[B.resolved(roots,"audit","completion.json")]=meta["audit_completion_sha256"]
    for (path,digest) in hash_map(plan["input_files_sha256"])
        checked=absolute_file(path)
        need(any(startswith(B.canonical(checked),B.canonical(roots[role])*"/") for role in ("source","reference")),"input proof escaped its root")
        need(!haskey(pinned,checked) || pinned[checked]==digest,"conflicting proof identity")
        pinned[checked]=digest
    end
    controls=hash_map(audit["controls_sha256"];maximum=64)
    control_names=Dict{String,String}()
    for (path,digest) in controls
        checked=absolute_file(path);need(!haskey(pinned,checked) || pinned[checked]==digest,"conflicting control identity")
        name=basename(path);need(!haskey(control_names,name),"ambiguous independent control basename")
        control_names[name]=digest
        pinned[checked]=digest
    end
    reader_root=B.root_path(normpath(joinpath(@__DIR__,"..","initial_reference_audit")))
    for name in readdir(reader_root)
        splitext(name)[2] in (".py",".jl") || continue
        need(get(control_names,name,nothing)==sha(B.resolved(Dict("reader"=>reader_root),"reader",name)),
            "completed audit uses a different maintained reader implementation")
    end
    need(hash_map(meta["upstream_sha256"];maximum=1200)==pinned,"exported upstream closure differs")
    guard_files(pinned)
    reference=json_at(roots["reference"],"completion.json",meta["reference_completion_sha256"])
    original=json_at(roots["source"],"completion.json",meta["source_completion_sha256"])
    need(reference["schema"]=="all_grid_initial_validation_reference_v1" && reference["complete"]===true && original["complete"]===true && reference["evidence_mode"]==mode && reference["contract_sha256"]==CONTRACT,"completed original/reference binding differs")
    need(reference["original_completion_sha256"]==meta["source_completion_sha256"] && reference["actual_input_sha256"]==original["input_sha256"]==meta["source_sha256"],"original raw source differs")
    need(reference["extra_dst_delay_hours"]===0 && reference["candidate_a_fitting"]===false && reference["scientific_verdict"]===false,"initial reference scope differs")
    need(B.integer(reference["grid_member_count"],1) && reference["grid_member_count"]==16 &&
         B.zero(reference["native_live_objects"]),"complete clean initial sixteen-model reference required")
    need(B.integer(reference["prefix_rows"],1) && B.integer(plan["prefix_rows"],1),"integer upstream prefix rows required")
    need(reference["prefix_rows"]==plan["prefix_rows"]==meta["prefix_rows"] &&
         datehour(reference["initial_boundary"])==datehour(plan["initial_boundary"])==datehour(meta["training_boundary"]) &&
         datehour(reference["a_selection_start"])==datehour(plan["a_start"])==datehour(meta["validation_start"]) &&
         datehour(reference["a_selection_end"])==datehour(plan["a_end"])==datehour(meta["validation_end"]) &&
         datehour(meta["prefix_start"])==datehour(plan["start"]) &&
         datehour(reference["original_selection_boundary"])==datehour(plan["selection_boundary"]),"reference clock metadata differs")
    if mode=="exposed_development_reconstruction"
        need(token.admitted!==nothing,"actual reference requires the matching admitted original source")
        B.recheck(token.admitted)
        admitted=token.admitted
        need(admitted.mode=="exposed_development" && admitted.roots["artifacts"]==roots["source"],"actual original-source admission location/mode differs")
        need(original["configuration"]["mode"]=="exposed_development","actual source evidence was relabeled")
        need(admitted.plan["receipts"]["completion"]["sha256"]==meta["source_completion_sha256"] &&
             admitted.plan["expected"]["input_sha256"]==meta["source_sha256"] &&
             admitted.plan["expected"]["raw_hourly_matrix_sha256"]==reference["actual_full_raw_matrix_sha256"],"actual admitted raw/source identities differ")
    else
        need(token.admitted===nothing,"synthetic artifacts cannot consume an actual-source token")
        need(original["configuration"]["mode"]=="synthetic_fixture","synthetic evidence was relabeled")
    end
    pinned[B.resolved(Dict("export"=>root),"export","completion.json")]=token.completion_sha256
    pinned[B.resolved(Dict("export"=>root),"export","selection_rows.csv")]=outputs["selection_rows.csv"]
    (root=root,meta=meta,roots=roots,reference=reference,original=original,pinned=pinned,own=own)
end

function preflight_selection_csv(path,count)
    records=0;length_line=0;commas=0
    open(path,"r") do io
        while !eof(io)
            for byte in read(io,65536)
                if byte==0x0a
                    need(length_line>0 && commas==25,"fixed selection CSV width differs")
                    records+=1;need(records<=count+1,"selection CSV row budget")
                    length_line=0;commas=0
                else
                    need(0x20<=byte<=0x7e && byte!=0x22,"selection CSV requires unquoted ASCII scalar fields")
                    length_line+=1;need(length_line<=4096,"selection CSV physical record budget before parsing")
                    byte==0x2c && (commas+=1)
                end
            end
        end
    end
    need(length_line==0 && records==count+1,"selection CSV complete row count differs")
    nothing
end

function raw_cohorts(dst,start,initial,first,last)
    need(start<initial<=first<last,"unordered selection boundaries")
    Tuple([i for i in 1:length(dst)-h if begin
        issue=A._shift(start,i);clock=A.forecast_clock(issue,h,0)
        first<=issue<last && isfinite(dst[i]) && isfinite(dst[i+h]) &&
            A.fit_eligible(issue,clock.target_complete,clock.target_receipt,last)
    end] for h in (4,6))
end

function materialize(token::ArtifactReference)
    proof=verified_metadata(token);meta=proof.meta;root=proof.root;reference=proof.reference
    n=Int(meta["prefix_rows"]);start=datehour(meta["prefix_start"])
    initial=datehour(meta["training_boundary"]);first=datehour(meta["validation_start"]);last=datehour(meta["validation_end"])
    need(A._shift(start,n)==datehour(reference["original_selection_boundary"]) && last<=A._shift(start,n),"raw prefix clock coverage differs")
    rawpath=B.resolved(proof.roots,"reference","prefix_raw17.f64")
    need(ENDIAN_BOM==0x04030201 && filesize(rawpath)==17*n*8,"raw prefix dimensions/endianness differ")
    need(sha(rawpath)==reference["output_sha256"]["prefix_raw17.f64"],"raw prefix identity differs")
    raw=reinterpret(Float64,read(rawpath));dst=[isfinite(raw[17*(i-1)+15]) && raw[17*(i-1)+15]<99999. ? raw[17*(i-1)+15] : NaN for i in 1:n]
    cohorts=raw_cohorts(dst,start,initial,first,last)
    need(all(!isempty,cohorts) && [length(x) for x in cohorts]==meta["cohort_counts"],"complete separate lead cohorts differ")
    indices=sort!(unique(vcat(cohorts...)))
    need(length(indices)==meta["row_count"] && maximum(indices)+6<=n,"complete union or opposite-slot prefix bound differs")
    joined=Tuple(begin
        path=B.resolved(proof.roots,"reference","a_join_h$h.csv")
        need(sha(path)==reference["output_sha256"]["a_join_h$h.csv"],"per-lead join identity differs")
        frame=CSV.read(path,DataFrame;types=String,stringtype=String,strict=true)
        need(Set(names(frame))==Set(("key","row","issue","target","anchor","prediction","status")) &&
             collect(frame.row)==string.(cohorts[slot]),"complete serialized per-lead join differs")
        result=Dict{Int,NamedTuple{(:prediction,:status),Tuple{Float64,String}}}()
        for (position,row) in enumerate(eachrow(frame))
            i=cohorts[slot][position]
            need(datehour(row.issue)==A._shift(start,i) && parse(Float64,row.anchor)==dst[i] &&
                 parse(Float64,row.target)==dst[i+h],"per-lead join raw values or clock differ")
            prediction=parse(Float64,row.prediction);status=String(row.status)
            need(isfinite(prediction) && status in ("model","persistence_feature_unavailable") &&
                 (status=="model" || prediction==dst[i]),"per-lead join fallback differs")
            result[i]=(prediction=prediction,status=status)
        end
        result
    end for (slot,h) in enumerate((4,6)))
    csvpath=B.resolved(Dict("export"=>root),"export","selection_rows.csv")
    preflight_selection_csv(csvpath,length(indices))
    frame=CSV.read(csvpath,DataFrame;types=String,stringtype=String,strict=true)
    need(names(frame)==HEADERS && all(x->x isa String && length(x)<=128,Iterators.flatten(eachcol(frame))),"selection CSV schema/field size differs")
    rows=NamedTuple[]
    for (position,row) in enumerate(eachrow(frame))
        i=indices[position];issue=A._shift(start,i);clocks=Tuple(A.forecast_clock(issue,h,0) for h in (4,6))
        need(row.origin_row==string(i) && datehour(row.issue)==issue && row.source_sha256==meta["source_sha256"] && row.dst_delay_hours=="0","selection row identity differs")
        for (name,value) in ((:driver_tag,clocks[1].driver_tag),(:driver_receipt,issue),(:anchor_tag,clocks[1].anchor_tag),
                (:anchor_complete,clocks[1].anchor_complete),(:anchor_receipt,clocks[1].anchor_receipt))
            need(datehour(getproperty(row,name))==value,"selection row clock differs")
        end
        anchor=parse(Float64,row.anchor);need(isfinite(anchor) && anchor==dst[i],"raw required anchor differs")
        targets=Tuple(dst[i+h] for h in (4,6));scoreable=Tuple(isfinite(x) for x in targets)
        predictions=Float64[];statuses=String[]
        for (slot,h) in enumerate((4,6))
            field(name)=getproperty(row,Symbol("h$(h)_$name"))
            clock=clocks[slot];eligible=haskey(joined[slot],i)
            need(datehour(field("target_tag"))==clock.target_tag && datehour(field("target_complete"))==clock.target_complete &&
                 datehour(field("target_receipt"))==clock.target_receipt,"selection target clock differs")
            need(isequal(parse(Float64,field("target")),targets[slot]) && field("scoreable")==string(Int(scoreable[slot])) &&
                 field("selection_eligible")==string(Int(eligible)),"raw target, scoreable flag or selection eligibility differs")
            prediction=parse(Float64,field("prediction"));status=field("status")
            if eligible
                expected=joined[slot][i]
                need(isfinite(prediction) && prediction==expected.prediction && status==expected.status,"eligible reference prediction/status differs")
                need(status in ("model","persistence_feature_unavailable") && (status=="model" || prediction==anchor),"invalid required-anchor fallback")
            else
                need(isnan(prediction) && status=="outside_selection_cohort","ineligible opposite slot was manufactured")
            end
            push!(predictions,prediction);push!(statuses,status)
        end
        push!(rows,(issue=issue,driver_tag=clocks[1].driver_tag,driver_receipt=issue,anchor_tag=clocks[1].anchor_tag,
            anchor_complete=clocks[1].anchor_complete,anchor_receipt=clocks[1].anchor_receipt,anchor=anchor,
            target_tag=Tuple(c.target_tag for c in clocks),target_complete=Tuple(c.target_complete for c in clocks),
            target_receipt=Tuple(c.target_receipt for c in clocks),target=targets,scoreable=scoreable,
            source_sha256=String(meta["source_sha256"]),dst_delay_hours=0,prediction=Tuple(predictions),status=Tuple(statuses)))
    end
    results=reference["results"]
    need([r["horizon"] for r in results]==[4,6],"ordered reference leads required")
    ids=Tuple(begin
        selected=r["choice"]["selected_index"]
        need(isint(selected) && 1<=selected<=8,"selected initial grid member differs")
        record=r["grid"][selected]
        need(record["model_sha256"]==reference["output_sha256"][record["model"]],"selected model identity differs")
        String(record["model_sha256"])
    end for r in results)
    value=(evidence_kind=:artifact_initial_reference,training_boundary=initial,delay_hours=0,
        max_fit_target_complete=Tuple(datehour(r["choice"]["max_fit_target_complete"]) for r in results),
        max_fit_label_receipt=Tuple(datehour(r["choice"]["max_fit_label_receipt"]) for r in results),
        source_sha256=String(meta["source_sha256"]),model_ids=ids,rows=rows)
    guard_files(proof.pinned);need(code_identity()==proof.own,"import implementation changed")
    (reference=value,proof=proof)
end

"""Load a pinned serialized reference; real data additionally needs admitted source proof."""
function load_reference(directory,completion_sha256;audit_completion_sha256,admitted=nothing)
    token=ArtifactReference(B.root_path(directory),String(completion_sha256),String(audit_completion_sha256),admitted)
    materialize(token);token
end
"""Reconstruct fresh verified rows. Mutating a returned vector cannot alter the token."""
reference_rows(token::ArtifactReference)=materialize(token).reference.rows

function T.select_setting(panels::AbstractDict,token::ArtifactReference;validation_start::DateTime,
        validation_end::DateTime,delay_hours::Integer)
    value=materialize(token);reference=value.reference;meta=value.proof.meta
    need(isint(delay_hours) && delay_hours==0 && validation_start==datehour(meta["validation_start"]) &&
         validation_end==datehour(meta["validation_end"]),"imported initial reference window/latency differs")
    need(sort!(collect(keys(panels)))==collect(1:18),"all18candidatepanels required")
    issues=getproperty.(reference.rows,:issue);projected=Dict{Int,Any}()
    for index in 1:18
        candidate=panels[index]
        T._check_rows(candidate,0,reference.source_sha256;setting=A.settings()[index])
        need(all(row->row.setting_index==index && row.eta==A.settings()[index].eta,candidate),"candidate index or eta differs")
        locations=Dict(row.issue=>i for (i,row) in enumerate(candidate))
        need(all(haskey(locations,t) for t in issues),"candidate omitted a required separate-lead union row")
        projected[index]=[candidate[locations[t]] for t in issues]
    end
    result=T._select_initial_reference(projected,reference;validation_start,validation_end,delay_hours)
    guard_files(value.proof.pinned);need(code_identity()==value.proof.own,"selection importer implementation changed")
    token.admitted===nothing || B.recheck(token.admitted)
    merge(result,(evidence_mode=String(meta["evidence_mode"]),reference_completion_sha256=token.completion_sha256,
        independent_audit_completion_sha256=String(meta["audit_completion_sha256"]),scientific_verdict=false))
end

end # module
