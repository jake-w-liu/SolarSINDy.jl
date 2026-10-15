module LiveReceiptMaturity

export snapshot_report, publish_report, trusted_claim_cohort, report_payload, verify_publication

using Dates,SHA,JSON3,CSV,DataFrames,Statistics
include(joinpath(@__DIR__,"live_observation_evidence.jl"))
using .LiveObservationEvidence
const E=LiveObservationEvidence
const IMPLEMENTATION_SHA256=bytes2hex(sha256(read(@__FILE__)))
const CATEGORIES=("preliminary_actual_receipt","completed_actual_receipt","legacy_maturity_unknown","invalid_or_conflicting_evidence")
const CURRENT_MANIFEST="a2a1242cbf3688b82b35672e9188abb456b76da1e10eebaba77bef8f4f1ae9e7"
const CURRENT_MODEL="dst-superlearner+sindyfloor+conformal"
const SUPPORTED_STEPS=(1,2,3,4,6,7)
const COMPARATORS=(persistence=:persistence_dst_nt,burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,obrien=:obrien_dst_nt,sindy_core=:sindy_core_pred_dst_nt)
bytes(x)=Vector{UInt8}(codeunits(JSON3.write(JSON3.read(JSON3.write(x)))))
digest(x)=bytes2hex(sha256(bytes(x)))
hashfile(p)=bytes2hex(sha256(read(p)))
time(x::DateTime)=x
time(x)=DateTime(chopsuffix(String(x),"Z"))
function nominal_end(t)
    try
        DateTime(Dates.UTInstant{Millisecond}(Millisecond(Base.Checked.checked_add(Dates.value(time(t)),Int64(3_600_000)))))
    catch
        throw(ArgumentError("nominal target end overflows"))
    end
end
function number(x)
    (x===nothing || ismissing(x)) && return nothing
    x isa Real && !(x isa Bool) || throw(ArgumentError("numeric observation required"))
    y=Float64(x);isfinite(y) || throw(ArgumentError("finite representable observation required"));y
end
cell(r,n)=hasproperty(r,n) ? getproperty(r,n) : missing
logical(b)=(;issue_time_utc=String(b.issue_time_utc),origin_tag_utc=String(b.origin_tag_utc),
    target_tag_utc=String(b.target_tag_utc),model_step_hours=Int(b.model_step_hours),
    model_identity=String(b.model_identity),manifest_sha256=String(b.manifest_sha256))
function checked_binding(r)
    b=E.forecast_binding(r)
    mod(Dates.value(time(b.target_tag_utc)),3_600_000)==0 || throw(ArgumentError("hourly target required"))
    nominal_end(b.target_tag_utc)
    b
end
matches(b,manifest,model)=String(b.manifest_sha256)==manifest && String(b.model_identity)==model && Int(b.model_step_hours) in SUPPORTED_STEPS

function point_metrics(predicted,observed)
    n=length(observed);length(predicted)==n || throw(ArgumentError("metric vectors differ"))
    n==0 && return (;n=0,rmse=nothing,mae=nothing,bias=nothing)
    e=observed.-predicted;all(isfinite,e) || throw(ArgumentError("metric residual overflows"))
    squared=e.^2;all(isfinite,squared) || throw(ArgumentError("metric square overflows"))
    result=(;n,rmse=sqrt(mean(squared)),mae=mean(abs.(e)),bias=mean(e))
    all(isfinite,(result.rmse,result.mae,result.bias)) || throw(ArgumentError("nonfinite metric"));result
end
function served_metrics(records;label=:csv_observation)
    selected=[r for r in records if getproperty(r,label)!==nothing && r.binding!==nothing]
    y=Float64[getproperty(r,label) for r in selected]
    p=Float64[r.binding.point_dst_nt for r in selected]
    lo=Float64[r.binding.lower_dst_nt for r in selected];hi=Float64[r.binding.upper_dst_nt for r in selected]
    metrics=point_metrics(p,y)
    if isempty(y)
        return (;metrics...,coverage_90=nothing,mean_width=nothing,mean_interval_score_90=nothing,forecast_keys=String[])
    end
    width=hi.-lo
    score=width.+20 .*max.(lo.-y,0).+20 .*max.(y.-hi,0)
    all(isfinite,width) && all(isfinite,score) || throw(ArgumentError("interval score overflows"))
    (;metrics...,coverage_90=mean((lo.<=y).&(y.<=hi)),mean_width=mean(width),
        mean_interval_score_90=mean(score),forecast_keys=[r.forecast_key for r in selected])
end
function matched_metrics(table,records)
    comparisons=NamedTuple[]
    for (name,column) in pairs(COMPARATORS)
        selected=NamedTuple[];missing_keys=String[]
        for r in records
            value=try number(cell(table[r.row,:],column)) catch;nothing end
            if value===nothing
                push!(missing_keys,r.forecast_key)
            else
                push!(selected,(;record=r,value))
            end
        end
        y=Float64[x.record.primary_observation for x in selected]
        p=Float64[x.record.binding.point_dst_nt for x in selected]
        q=Float64[x.value for x in selected]
        push!(comparisons,(;method=String(name),column=String(column),matched_keys=[x.record.forecast_key for x in selected],
            missing_keys,served=point_metrics(p,y),comparator=point_metrics(q,y)))
    end
    comparisons
end

"Classify canonical verified journal state without upgrading observation evidence into issuance qualification."
function build_report(table::DataFrame,state;asof::DateTime,manifest::String=CURRENT_MANIFEST,model::String=CURRENT_MODEL,
        receipts=Dict{String,Any}(),journal_available=true)
    occursin(r"^[0-9a-f]{64}$",manifest) && !isempty(model) || throw(ArgumentError("exact target identity required"))
    bindings=Any[];errors=String[];by_key=Dict{String,Vector{Int}}();by_logical=Dict{String,Vector{Int}}()
    for (i,row) in enumerate(eachrow(table))
        binding=try checked_binding(row) catch error;push!(errors,sprint(showerror,error));nothing end
        if binding!==nothing
            push!(errors,"");push!(get!(by_key,digest(binding),Int[]),i)
            push!(get!(by_logical,digest(logical(binding)),Int[]),i)
        end
        push!(bindings,binding)
    end
    duplicate_rows=Set{Int}()
    for indices in values(by_logical)
        length(indices)>1 && union!(duplicate_rows,indices)
    end
    events_by_key=Dict{String,Vector{Any}}()
    for event in state.events
        key=String(event.forecast_key)
        haskey(by_key,key) && length(by_key[key])==1 || throw(ArgumentError("journal forecast is missing, duplicated or changed in CSV"))
        row=only(by_key[key]);row in duplicate_rows && throw(ArgumentError("conflicting journal logical forecast in CSV"))
        digest(bindings[row])==key || throw(ArgumentError("journal binding differs from CSV"))
        push!(get!(events_by_key,key,Any[]),event)
    end
    # Latest CSV value must agree with the full committed event history; asof selection is separate.
    for (key,history) in events_by_key
        row=only(by_key[key]);actual=number(cell(table[row,:],:observation_dst_nt))
        actual!==nothing && actual==Float64(last(history).observation) || throw(ArgumentError("latest CSV/journal observation differs"))
    end
    counts=Dict(c=>0 for c in CATEGORIES);records=NamedTuple[];pending=0
    for (i,row) in enumerate(eachrow(table))
        b=bindings[i];key=b===nothing ? "invalid-row-$i" : digest(b)
        raw=cell(row,:observation_dst_nt)
        observed=try number(raw) catch;nothing end
        observation_invalid=!(raw===nothing || ismissing(raw)) && observed===nothing
        invalid=b===nothing || i in duplicate_rows || observation_invalid
        status=observed===nothing && !observation_invalid ? "pending" :
            invalid ? "invalid_or_conflicting_evidence" : "legacy_maturity_unknown"
        reason=b===nothing ? errors[i] : i in duplicate_rows ? "duplicate_or_conflicting_logical_forecast" :
            observation_invalid ? "invalid_observation" : "no_eligible_actual_receipt_at_asof"
        eligible=Any[];mature=Any[]
        if !invalid
            eligible=[e for e in get(events_by_key,key,Any[]) if time(e.receipt_completed_utc)<asof &&
                time(e.verification_completed_utc)<asof && time(b.issue_time_utc)<asof]
            mature=[e for e in eligible if time(e.receipt_completed_utc)>=nominal_end(b.target_tag_utc)]
            if observed!==nothing && !isempty(eligible)
                status=isempty(mature) ? "preliminary_actual_receipt" : "completed_actual_receipt"
                reason=isempty(mature) ? "actual_capture_precedes_nominal_hour_end" : "actual_post_end_capture_and_committed_verification"
            end
        end
        status=="pending" ? (pending+=1) : (counts[status]+=1)
        first_mature=isempty(mature) ? nothing : first(mature)
        last_mature=isempty(mature) ? nothing : last(mature)
        last_eligible=isempty(eligible) ? nothing : last(eligible)
        provenance=first_mature===nothing ? nothing : get(receipts,String(first_mature.receipt.sha256),nothing)
        provenance!==nothing && !(provenance isa NamedTuple) && throw(ArgumentError("invalid capture provenance record"))
        push!(records,(;row=i,forecast_key=key,binding=b,category=status,reason,csv_observation=observed,
            model_identity_matches=b!==nothing && matches(b,manifest,model),
            primary_event_key=first_mature===nothing ? "" : String(first_mature.event_key),
            primary_observation=first_mature===nothing ? nothing : Float64(first_mature.observation),
            primary_receipt=first_mature===nothing ? nothing : String(first_mature.receipt_completed_utc),
            primary_verified=first_mature===nothing ? nothing : String(first_mature.verification_completed_utc),
            primary_capture_provenance=provenance,
            latest_eligible_event_key=last_eligible===nothing ? "" : String(last_eligible.event_key),
            latest_eligible_receipt=last_eligible===nothing ? nothing : String(last_eligible.receipt_completed_utc),
            latest_eligible_verified=last_eligible===nothing ? nothing : String(last_eligible.verification_completed_utc),
            latest_eligible_capture_provenance=last_eligible===nothing ? nothing : get(receipts,String(last_eligible.receipt.sha256),nothing),
            latest_completed_event_key=last_mature===nothing ? "" : String(last_mature.event_key),
            latest_completed_observation=last_mature===nothing ? nothing : Float64(last_mature.observation),
            latest_completed_capture_provenance=last_mature===nothing ? nothing : get(receipts,String(last_mature.receipt.sha256),nothing),
            completed_event_count=length(mature),issuance_source_qualified=false,
            issuance_qualification_reason="canonical observation journal does not establish fully qualified original issuance runtime",
            scientific_claim_eligible=false))
    end
    sum(values(counts))+pending==nrow(table) || error("category reconciliation failed")
    descriptive=[r for r in records if r.csv_observation!==nothing && r.binding!==nothing]
    current=[r for r in descriptive if r.model_identity_matches]
    current_served_label=[r for r in descriptive if String(r.binding.model_identity)==model]
    complete=[r for r in records if r.category=="completed_actual_receipt"]
    current_complete=[r for r in complete if r.model_identity_matches]
    (;schema="dst_actual_receipt_maturity_report_prototype_v1",implementation_sha256=IMPLEMENTATION_SHA256,
        as_of_utc=string(asof),manifest,model_identity=model,rows=nrow(table),pending,counts,records,
        maturity_definition="nominal [provider tag, tag+1h) completion with actual capture; not final/stable Dst or an independently established physical sampling-time convention",
        journal_available,journal_sequence=state.sequence,journal_head_sha256=state.sha256,
        scopes=(descriptive="latest snapshot CSV, preliminary-inclusive and legacy-maturity-unknown; historical current-served-label view does not require an exact manifest and is preserved separately",
            completed_primary="first eligible actual nominal-post-end receipt, label-bound descriptive evidence only; no final/stable Dst or issuance-source qualification",
            completed_revision_sensitivity="latest eligible completed revision; primary support unchanged"),
        descriptive_all=served_metrics(descriptive),descriptive_current_served_label=served_metrics(current_served_label),
        descriptive_exact_identity=served_metrics(current),
        completed_primary_all_identities=served_metrics(complete;label=:primary_observation),
        completed_primary_current_identity=served_metrics(current_complete;label=:primary_observation),
        completed_latest_revision_current_identity=served_metrics(current_complete;label=:latest_completed_observation),
        completed_primary_current_matched_comparators=matched_metrics(table,current_complete),
        qualified_scientific_support=0,claim_gate_ready=false,
        claim_gate_reason="no qualified issuance-source ledger; existing fixed057 prospective claim identity is not retargeted",
        operational_promotion=false)
end

function regular(path)
    p=abspath(path);current=p
    while true
        islink(current) && throw(ArgumentError("symbolic-link input rejected"))
        parent=dirname(current);parent==current && break;current=parent
    end
    p
end
function read_owned_snapshot(log_path,store;owned_root::AbstractString,asof::DateTime,before_recheck::Function=()->nothing,kwargs...)
    owner=regular(owned_root);log=regular(log_path);root=regular(store)
    isdir(owner) || throw(ArgumentError("explicit existing owned root required"))
    all(p->startswith(p,owner*string(Base.Filesystem.path_separator)),(log,root)) || throw(ArgumentError("only explicit owned snapshot inputs allowed"))
    csv_before=read(log);head_path=joinpath(root,"head.json")
    head_before=isfile(head_path) ? read(head_path) : nothing
    journal_available=isdir(root) && head_before!==nothing
    # A headless existing store can still contain a broken committed journal.
    state=isdir(root) ? E.verify_observation_evidence(root;require_complete=true) :
        (;sequence=0,sha256=repeat("0",64),events=Any[])
    table=CSV.read(IOBuffer(csv_before),DataFrame)
    receipts=Dict{String,Any}()
    for event in state.events
        key=String(event.receipt.sha256)
        if !haskey(receipts,key)
            r=E.load_receipt(root,event.receipt)
            receipts[key]=(;receipt_sha256=key,capture_source_sha256=String(r.capture_source_sha256),
                consumer_source_sha256=get(r,:consumer_source_sha256,nothing),raw_body_sha256=String(r.body_sha256),
                process_id=r.process_id,capture_session_id=get(r,:capture_session_id,nothing),
                receipt_completed_utc=String(r.receipt_completed_utc),source_url=String(r.source_url),
                source_qualification="actual capture provenance only; not an issuance release qualification")
        end
    end
    report=build_report(table,state;asof,receipts,journal_available,kwargs...)
    before_recheck()
    read(log)==csv_before || throw(ArgumentError("CSV snapshot changed during report"))
    (isfile(head_path) ? read(head_path) : nothing)==head_before || throw(ArgumentError("journal head changed during report"))
    (;report...,snapshot_csv_sha256=bytes2hex(sha256(csv_before)),snapshot_head_sha256=head_before===nothing ? nothing : bytes2hex(sha256(head_before)),
        canonical_verifier_source_sha256=E.CAPTURE_SOURCE_SHA256,
        canonical_verifier_does_not_requalify_historical_sources=true)
end

const LEGACY_CLAIM_MANIFEST="057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede"
struct SnapshotReport
    payload::NamedTuple
    fingerprint::String
    snapshot_root::String
end
report_payload(r::SnapshotReport)=r.payload
function _positive_limit(value,name)
    value isa Integer && !(value isa Bool) && value>0 || throw(ArgumentError("positive $name required"))
    try Int(value) catch;throw(ArgumentError("$name does not fit this runtime")) end
end
function _canonical_containment_path(path)
    current=regular(path);suffix=String[]
    while !ispath(current)
        parent=dirname(current);parent==current && throw(ArgumentError("containment path has no existing volume ancestor"))
        push!(suffix,basename(current));current=parent
    end
    # Resolve existing volume/path spelling before appending a not-yet-existing output leaf.
    resolved=realpath(regular(current))
    normpath(joinpath(resolved,reverse(suffix)...))
end
function _same_component(a,b)
    !Sys.iswindows() && return a==b
    # Native ordinal case handling avoids locale/Unicode lowercase substitutions.
    left=transcode(UInt16,String(a));right=transcode(UInt16,String(b))
    result=ccall((:CompareStringOrdinal,"kernel32"),Cint,
        (Ptr{UInt16},Cint,Ptr{UInt16},Cint,Cint),left,length(left),right,length(right),1)
    result==0 && throw(ArgumentError("native Windows path comparison failed"))
    result==2
end
function _inside(path,root)
    child=splitpath(_canonical_containment_path(path));parent=splitpath(_canonical_containment_path(root))
    length(child)>=length(parent) && all(_same_component(child[i],parent[i]) for i in eachindex(parent))
end
const READ_CHUNK_BYTES=65_536
function _bounded_read(path,limit)
    cap=_positive_limit(limit,"member budget");p=regular(path)
    isfile(p) || throw(ArgumentError("snapshot input is not a regular file"))
    result=UInt8[];sizehint!(result,min(cap,READ_CHUNK_BYTES))
    open(p,"r") do io
        while !eof(io)
            remaining=cap-length(result)
            amount=remaining>=READ_CHUNK_BYTES ? READ_CHUNK_BYTES : remaining+1
            chunk=read(io,amount)
            length(chunk)<=remaining || throw(ArgumentError("snapshot member exceeds bounded read budget"))
            append!(result,chunk)
        end
    end
    result
end
function _inventory(root;max_files,max_bytes,max_member_bytes,max_entries=max_files)
    entry_limit=_positive_limit(max_entries,"entry budget")
    path=regular(root)
    ispath(path) && !isdir(path) && throw(ArgumentError("evidence root is not a directory"))
    !isdir(path) && return (;exists=false,files=Dict{String,String}(),directories=String[],bytes=0,entries=0)
    files=Dict{String,String}();directories=String[];total=0;entries=0;seen=Set{String}()
    for (base,dirs,names) in walkdir(path;follow_symlinks=false)
        for group in (dirs,names),entry in group
            entries=try Base.Checked.checked_add(entries,1) catch;throw(ArgumentError("evidence entry count overflow")) end
            entries<=entry_limit || throw(ArgumentError("evidence entry budget exceeded"))
            p=regular(joinpath(base,entry));islink(p) && throw(ArgumentError("snapshot symlink rejected"))
            endswith(lowercase(entry),".lock") && throw(ArgumentError("evidence journal is busy"))
        end
        for entry in dirs
            push!(directories,replace(relpath(joinpath(base,entry),path),'\\'=>'/'))
        end
        for entry in names
            p=regular(joinpath(base,entry));isfile(p) || throw(ArgumentError("nonregular evidence file"))
            relative=replace(relpath(p,path),'\\'=>'/');fold=lowercase(relative)
            fold in seen && throw(ArgumentError("case-colliding evidence paths"));push!(seen,fold)
            length(files)<max_files || throw(ArgumentError("evidence file budget exceeded"))
            stat(p).size<=max_member_bytes || throw(ArgumentError("evidence member budget exceeded"))
            body=_bounded_read(p,max_member_bytes)
            total=try Base.Checked.checked_add(total,length(body)) catch;throw(ArgumentError("evidence byte budget overflow")) end
            total<=max_bytes || throw(ArgumentError("evidence byte budget exceeded"))
            files[relative]=bytes2hex(sha256(body))
        end
    end
    (;exists=true,files,directories=sort!(directories),bytes=total,entries)
end
"Read actual inputs without acquiring their journal lock; verify only a new owned stable copy."
function snapshot_report(log_path::AbstractString,store::AbstractString;asof::DateTime,
        snapshot_parent::AbstractString=tempdir(),max_files=100_000,max_bytes=1_073_741_824,
        max_member_bytes=67_108_864,max_entries=max_files,snapshot_hook::Function=stage->nothing,kwargs...)
    limits=(;max_files=_positive_limit(max_files,"file budget"),max_bytes=_positive_limit(max_bytes,"byte budget"),
        max_member_bytes=_positive_limit(max_member_bytes,"member budget"),max_entries=_positive_limit(max_entries,"entry budget"))
    log=regular(log_path);source_root=regular(store);parent=regular(snapshot_parent)
    isfile(log) || throw(ArgumentError("CSV input is missing"))
    isdir(parent) || throw(ArgumentError("snapshot parent must already exist"))
    _inside(parent,source_root) && throw(ArgumentError("owned snapshot must not be inside evidence source"))
    stat(log).size<=limits.max_member_bytes || throw(ArgumentError("CSV member budget exceeded"))
    csv_before=_bounded_read(log,limits.max_member_bytes)
    before=_inventory(source_root;limits...)
    length(before.files)+1<=limits.max_files || throw(ArgumentError("snapshot file budget includes CSV"))
    before.entries+1<=limits.max_entries || throw(ArgumentError("snapshot entry budget includes CSV"))
    total_bytes=try Base.Checked.checked_add(length(csv_before),before.bytes) catch;throw(ArgumentError("snapshot byte budget overflow")) end
    total_bytes<=limits.max_bytes || throw(ArgumentError("snapshot byte budget includes CSV"))
    snapshot_hook(:after_inventory)
    owned=mktempdir(parent;prefix="dst-receipt-snapshot-");snapshot_csv=joinpath(owned,"log.csv");snapshot_store=joinpath(owned,"store")
    E.publish_immutable(snapshot_csv,csv_before)
    if before.exists
        mkpath(snapshot_store)
        for directory in before.directories;mkpath(joinpath(snapshot_store,directory));end
        for name in sort!(collect(keys(before.files)))
            path=regular(joinpath(source_root,name));body=_bounded_read(path,limits.max_member_bytes)
            bytes2hex(sha256(body))==before.files[name] || throw(ArgumentError("evidence changed while copying"))
            E.publish_immutable(joinpath(snapshot_store,name),body)
        end
    end
    snapshot_hook(:after_copy)
    _bounded_read(log,limits.max_member_bytes)==csv_before || throw(ArgumentError("actual CSV changed during snapshot"))
    _inventory(source_root;limits...)==before || throw(ArgumentError("actual evidence changed during snapshot"))
    copied=_inventory(snapshot_store;limits...)
    copied==before || throw(ArgumentError("owned snapshot inventory differs"))
    payload=read_owned_snapshot(snapshot_csv,snapshot_store;owned_root=owned,asof,kwargs...)
    _inventory(snapshot_store;limits...)==copied || throw(ArgumentError("owned evidence changed during verification"))
    snapshot_hook(:before_source_recheck)
    _bounded_read(log,limits.max_member_bytes)==csv_before && _inventory(source_root;limits...)==before || throw(ArgumentError("source snapshot changed before report completion"))
    bound=(;payload...,schema="dst_actual_receipt_maturity_integrated_candidate_v1",
        source_csv_sha256=bytes2hex(sha256(csv_before)),source_evidence_inventory=before,
        snapshot_policy="source read only; complete inventory checked before/after copy and verification; canonical locks only in owned snapshot",
        source_paths=(;csv=log,evidence=source_root),prototype_integration_status="isolated_candidate_no_deployment")
    SnapshotReport(bound,digest(bound),owned)
end
function _checked_report(r::SnapshotReport)
    digest(r.payload)==r.fingerprint || throw(ArgumentError("report payload changed after verification"))
    r.payload.qualified_scientific_support==0 && !r.payload.claim_gate_ready && !r.payload.operational_promotion ||
        throw(ArgumentError("unsupported issuance qualification or promotion"))
    nothing
end
"The fixed057 audit gets an empty trusted cohort until genuine issuance qualification exists."
function trusted_claim_cohort(r::SnapshotReport)
    _checked_report(r)
    (;schema="legacy057_actual_receipt_claim_input_v1",manifest_sha256=LEGACY_CLAIM_MANIFEST,
        forecast_keys=(),observations=(),support=0,gate_ready=false,
        reason="no genuine qualified original-issuance ledger exists; no issuance-ledger input is accepted",
        report_fingerprint=r.fingerprint,source_csv_sha256=r.payload.source_csv_sha256,
        journal_head_sha256=r.payload.journal_head_sha256,identity_retargeted=false)
end
function _markdown(r::SnapshotReport)
    p=r.payload
    lines=["# Dst actual-receipt report", "", "As of UTC: $(p.as_of_utc)", "",
        "This is a nominal-hour receipt report. Completed does not mean final or stable Dst. Original issuance source is unqualified; no skill or deployment claim is enabled.","",
        "| Observation class | Rows |","|---|---:|"]
    for category in CATEGORIES;push!(lines,"| $category | $(p.counts[category]) |");end
    append!(lines,["| pending | $(p.pending) |","","Historical descriptive metrics remain preliminary-inclusive and may contain labels with unknown receipt maturity.","",
        "| View | n | RMSE nT |","|---|---:|---:|"])
    for (label,view) in [("Historical served label",p.descriptive_current_served_label),("Exact manifest descriptive",p.descriptive_exact_identity),
                         ("First nominal-post-end label, exact identity",p.completed_primary_current_identity),("Latest completed revision sensitivity",p.completed_latest_revision_current_identity)]
        push!(lines,"| $label | $(view.n) | $(view.rmse===nothing ? "unavailable" : string(view.rmse)) |")
    end
    append!(lines,["","Qualified scientific support:0. Legacy057 claim gate remains blocked. Primary labels are frozen at the first eligible nominal-post-end actual capture; revisions are separate sensitivity, not extra support.",""])
    join(lines,"\n")
end
"Exclusively claim a new output directory; only COMPLETED.json marks a complete publication."
function publish_report(r::SnapshotReport,directory::AbstractString;before_claim::Function=()->nothing)
    _checked_report(r);target=regular(directory)
    for source in (r.payload.source_paths.csv,r.payload.source_paths.evidence,r.snapshot_root)
        (_inside(target,source) || _inside(source,target)) && throw(ArgumentError("report output overlaps an input or owned snapshot"))
    end
    isdir(dirname(target)) || throw(ArgumentError("report output parent must exist"))
    ispath(target) && throw(ArgumentError("new absent output directory required"))
    before_claim()
    mkdir(target) # exclusive namespace claim; an already appearing destination is never reused
    files=Dict("report.json"=>bytes(r.payload),"report.md"=>Vector{UInt8}(codeunits(_markdown(r))),
        "legacy057_claim_cohort.json"=>bytes(trusted_claim_cohort(r)))
    digests=Dict{String,String}()
    for name in sort!(collect(keys(files)))
        E.publish_immutable(joinpath(target,name),files[name]);digests[name]=bytes2hex(sha256(files[name]))
    end
    completion=(;schema="dst_receipt_report_completion_v1",outputs=digests,report_fingerprint=r.fingerprint,
        implementation_sha256=IMPLEMENTATION_SHA256,canonical_verifier_source_sha256=E.CAPTURE_SOURCE_SHA256,
        claim_support=0,identity_retargeted=false)
    E.publish_immutable(joinpath(target,"COMPLETED.json"),bytes(completion))
    verify_publication(target)
    (;directory=target,completion_sha256=hashfile(joinpath(target,"COMPLETED.json")),outputs=digests)
end
function verify_publication(directory::AbstractString;expected_completion_sha256=nothing)
    root=regular(directory);isdir(root) || throw(ArgumentError("report directory missing"))
    completion_path=regular(joinpath(root,"COMPLETED.json"));isfile(completion_path) || throw(ArgumentError("report is not completed"))
    expected_completion_sha256===nothing || hashfile(completion_path)==expected_completion_sha256 || throw(ArgumentError("completion receipt hash differs"))
    completion=JSON3.read(read(completion_path))
    completion.schema=="dst_receipt_report_completion_v1" || throw(ArgumentError("unknown report completion schema"))
    Set(String.(keys(completion.outputs)))==Set(["report.json","report.md","legacy057_claim_cohort.json"]) || throw(ArgumentError("unexpected report output set"))
    Set(readdir(root))==Set(["report.json","report.md","legacy057_claim_cohort.json","COMPLETED.json"]) || throw(ArgumentError("untracked report publication member"))
    for (name,expected) in pairs(completion.outputs)
        path=regular(joinpath(root,String(name)));isfile(path) && hashfile(path)==expected || throw(ArgumentError("published report hash differs"))
    end
    payload=JSON3.read(read(joinpath(root,"report.json")))
    digest(payload)==completion.report_fingerprint || throw(ArgumentError("published report fingerprint differs"))
    completion.implementation_sha256==payload.implementation_sha256==IMPLEMENTATION_SHA256 || throw(ArgumentError("report implementation binding differs"))
    completion.canonical_verifier_source_sha256==payload.canonical_verifier_source_sha256==E.CAPTURE_SOURCE_SHA256 || throw(ArgumentError("canonical verifier binding differs"))
    payload.qualified_scientific_support==0 && !payload.claim_gate_ready && !payload.operational_promotion || throw(ArgumentError("unsupported published qualification"))
    all(r->!r.issuance_source_qualified && !r.scientific_claim_eligible,payload.records) || throw(ArgumentError("unsupported published issuance qualification"))
    claim=JSON3.read(read(joinpath(root,"legacy057_claim_cohort.json")))
    String(claim.manifest_sha256)==LEGACY_CLAIM_MANIFEST && claim.support==0 && !claim.gate_ready && !claim.identity_retargeted || throw(ArgumentError("legacy claim identity or support changed"))
    claim.forecast_keys isa AbstractVector && claim.observations isa AbstractVector &&
        isempty(claim.forecast_keys) && isempty(claim.observations) || throw(ArgumentError("trusted empty claim cohort contains unsupported entries"))
    claim.support isa Integer && !(claim.support isa Bool) && completion.claim_support isa Integer && !(completion.claim_support isa Bool) || throw(ArgumentError("claim support type differs"))
    claim.source_csv_sha256==payload.source_csv_sha256 && claim.journal_head_sha256==payload.journal_head_sha256 || throw(ArgumentError("claim snapshot binding differs"))
    claim.report_fingerprint==completion.report_fingerprint && completion.claim_support==0 && !completion.identity_retargeted || throw(ArgumentError("claim completion binding differs"))
    true
end

end
