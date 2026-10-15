module LiveObservationEvidence

using Dates, SHA, JSON3, CSV, DataFrames, Random
include(joinpath(@__DIR__,"live_log_lock.jl"))

export evidence_root, capture_response!, capture_transport_failure!, load_receipt,
       bind_observation, forecast_binding, prepare_verification!, commit_verification!,
       verify_observation_evidence, latest_forecast_evidence, publish_immutable,
       recover_observation_head!, record_verification_failure!

const SOURCE_URL="https://services.swpc.noaa.gov/products/kyoto-dst.json"
const SCHEMA="live_dst_observation_evidence_v1"
const ZERO=repeat("0",64)
const _SERIAL=Threads.Atomic{UInt64}(0)
_sha(bytes)=bytes2hex(sha256(bytes))
const CAPTURE_SOURCE_SHA256=_sha(read(@__FILE__))
const CAPTURE_SESSION_ID=bytes2hex(rand(RandomDevice(),UInt8,32))
# JSON3 preserves a JSON number's value but may read an integral Float64 as Int.
# Normalize through the reader before hashing so a persisted reference retains
# the same scientific identity after a normal JSON round trip.
_bytes(value)=Vector{UInt8}(codeunits(JSON3.write(JSON3.read(JSON3.write(value)))))
_time(value::DateTime)=value
_time(value)=DateTime(chopsuffix(String(value),"Z"))
_utc(value::DateTime)=string(value)
_obs(value)=value===nothing || ismissing(value) ? nothing : Float64(value)
_equal(a,b)=a===nothing ? b===nothing : b!==nothing && a==b
_relative(parts...)=join(String.(parts),"/")
evidence_root(log_path::AbstractString)=abspath(String(log_path))*".observation_evidence"

function _flush_file(io)
    flush(io)
    result=Sys.iswindows() ? ccall(:_commit,Cint,(Cint,),fd(io)) : ccall(:fsync,Cint,(Cint,),fd(io))
    systemerror("observation evidence file flush",result!=0)
end
function _flush_parent(path)
    Sys.iswindows() && return nothing
    open(dirname(abspath(path)),"r") do io
        result=ccall(:fsync,Cint,(Cint,),fd(io))
        systemerror("observation evidence directory flush",result!=0)
    end
end

function _regular_path(path::AbstractString)
    current=abspath(String(path))
    while true
        islink(current) && throw(ArgumentError("observation evidence path contains a symbolic link: $current"))
        parent=dirname(current);parent==current && break
        current=parent
    end
    return abspath(String(path))
end
function _root(root;create=true)
    path=_regular_path(root)
    ispath(path) && !isdir(path) && throw(ArgumentError("observation evidence root is not a directory"))
    create && mkpath(path)
    return path
end
function _path(root,relative)
    text=replace(String(relative),'\\'=>'/')
    occursin(':',text) && throw(ArgumentError("observation evidence relative path contains a drive or scheme"))
    isabspath(text) && throw(ArgumentError("observation evidence reference must be relative"))
    base=abspath(String(root));path=normpath(joinpath(base,text))
    startswith(path,base*string(Base.Filesystem.path_separator)) || throw(ArgumentError("observation evidence reference escapes root"))
    return _regular_path(path)
end

"Atomically publish bytes without replacing any existing file, even in a race."
function publish_immutable(path::AbstractString,bytes::Vector{UInt8};before_publish::Function=()->nothing)
    target=_regular_path(path);mkpath(dirname(target))
    if ispath(target)
        isfile(target) && read(target)==bytes || throw(ArgumentError("immutable evidence conflicts at $target"))
        return target
    end
    mktemp(dirname(target)) do staged,io
        write(io,bytes);_flush_file(io);close(io)
        before_publish()
        try
            hardlink(staged,target)
        catch error
            isfile(target) && !islink(target) && read(target)==bytes || throw(ArgumentError(
                "immutable evidence publication failed without replacing destination: "*sprint(showerror,error)))
        end
    end
    _flush_parent(target)
    isfile(target) && read(target)==bytes || throw(ArgumentError("immutable evidence readback differs"))
    return target
end
function _store(root,kind,value)
    bytes=_bytes(value);digest=_sha(bytes)
    relative=_relative(kind,digest*".json")
    publish_immutable(_path(root,relative),bytes)
    return (;relative_path=relative,sha256=digest)
end
function _read(root,ref)
    digest=String(ref.sha256)
    occursin(r"^[0-9a-f]{64}$",digest) || throw(ArgumentError("invalid evidence checksum"))
    path=_path(root,ref.relative_path)
    isfile(path) || throw(ArgumentError("missing observation evidence file"))
    bytes=read(path)
    _sha(bytes)==digest || throw(ArgumentError("observation evidence checksum changed"))
    try
        return JSON3.read(bytes)
    catch error
        throw(ArgumentError("invalid observation evidence JSON: "*sprint(showerror,error)))
    end
end
function _clock_fields(started,completed,started_ns,completed_ns)
    started isa DateTime && completed isa DateTime || throw(ArgumentError("observation receipt clocks are not DateTime values"))
    ns_start=tryparse(BigInt,string(started_ns));ns_end=tryparse(BigInt,string(completed_ns))
    ns_start===nothing || ns_end===nothing ? throw(ArgumentError("observation monotonic clocks are not integers")) : nothing
    range_valid=0<=ns_start<=typemax(UInt64) && 0<=ns_end<=typemax(UInt64)
    utc_valid=completed>=started
    monotonic_valid=range_valid && ns_end>=ns_start
    status=utc_valid && monotonic_valid ? "ordered" :
        !utc_valid && !monotonic_valid ? "utc_and_monotonic_regression" :
        !utc_valid ? "utc_regression" : "monotonic_regression_or_range"
    return (;request_started_utc=_utc(started),receipt_completed_utc=_utc(completed),
        monotonic_started_ns=range_valid ? UInt64(ns_start) : string(started_ns),
        monotonic_completed_ns=range_valid ? UInt64(ns_end) : string(completed_ns),
        clock_eligible=utc_valid && monotonic_valid,clock_status=status)
end
receipt_clock_eligible(record)=_clock_fields(_time(record.request_started_utc),_time(record.receipt_completed_utc),
    record.monotonic_started_ns,record.monotonic_completed_ns).clock_eligible
function capture_response!(root,response;request_started_utc::DateTime,
        receipt_completed_utc::DateTime,monotonic_started_ns::Integer,
        monotonic_completed_ns::Integer,consumer_source_sha256=nothing)
    storage=_root(root)
    status=Int(response.status);100<=status<=599 || throw(ArgumentError("invalid observation HTTP status"))
    body=response.body isa AbstractString ? Vector{UInt8}(codeunits(response.body)) : Vector{UInt8}(response.body)
    digest=_sha(body);relative=_relative("raw",digest*".body")
    publish_immutable(_path(storage,relative),body)
    clocks=_clock_fields(request_started_utc,receipt_completed_utc,monotonic_started_ns,monotonic_completed_ns)
    headers=hasproperty(response,:headers) ? [(name=String(first(p)),value=String(last(p))) for p in response.headers] : NamedTuple[]
    record=(schema_version=SCHEMA,source_url=SOURCE_URL,product_id="kyoto-dst",
        parser_version="raw_tag_exact_value_v1",clocks...,
        capture_source_sha256=CAPTURE_SOURCE_SHA256,consumer_source_sha256,
        julia_version=string(VERSION),julia_threads=Threads.nthreads(),
        process_id=getpid(),capture_session_id=CAPTURE_SESSION_ID,capture_serial=Threads.atomic_add!(_SERIAL,UInt64(1))+1,
        capture_outcome="http_response",http_status=status,response_headers=headers,
        body_sha256=digest,body_bytes=length(body),raw_relative_path=relative,
        transport_error_type="",transport_error_message="")
    return _store(storage,"fetches",record)
end
function capture_transport_failure!(root,error;request_started_utc::DateTime,
        receipt_completed_utc::DateTime,monotonic_started_ns::Integer,
        monotonic_completed_ns::Integer,consumer_source_sha256=nothing)
    storage=_root(root)
    clocks=_clock_fields(request_started_utc,receipt_completed_utc,monotonic_started_ns,monotonic_completed_ns)
    record=(schema_version=SCHEMA,source_url=SOURCE_URL,product_id="kyoto-dst",
        parser_version="raw_tag_exact_value_v1",clocks...,
        capture_source_sha256=CAPTURE_SOURCE_SHA256,consumer_source_sha256,
        julia_version=string(VERSION),julia_threads=Threads.nthreads(),
        process_id=getpid(),capture_session_id=CAPTURE_SESSION_ID,capture_serial=Threads.atomic_add!(_SERIAL,UInt64(1))+1,
        capture_outcome="transport_error",http_status=0,response_headers=NamedTuple[],
        body_sha256=nothing,body_bytes=0,raw_relative_path="",
        transport_error_type=string(typeof(error)),transport_error_message=sprint(showerror,error))
    return _store(storage,"fetches",record)
end
function _load_receipt_with_body(root,reference)
    record=_read(root,reference)
    record.schema_version==SCHEMA && record.source_url==SOURCE_URL && record.product_id=="kyoto-dst" || throw(ArgumentError("unexpected observation source or schema"))
    occursin(r"^[0-9a-f]{64}$",String(record.capture_source_sha256)) || throw(ArgumentError("observation capture source checksum is missing"))
    record.consumer_source_sha256===nothing || occursin(r"^[0-9a-f]{64}$",String(record.consumer_source_sha256)) || throw(ArgumentError("invalid observation consumer source checksum"))
    clocks=_clock_fields(_time(record.request_started_utc),_time(record.receipt_completed_utc),
        record.monotonic_started_ns,record.monotonic_completed_ns)
    get(record,:clock_eligible,clocks.clock_eligible)==clocks.clock_eligible &&
        get(record,:clock_status,clocks.clock_status)==clocks.clock_status || throw(ArgumentError("receipt clock eligibility changed"))
    body=UInt8[]
    if record.capture_outcome=="http_response"
        path=_path(root,record.raw_relative_path)
        isfile(path) || throw(ArgumentError("observation response body changed"))
        body=read(path)
        length(body)==record.body_bytes && _sha(body)==record.body_sha256 || throw(ArgumentError("observation response body changed"))
    elseif record.capture_outcome=="transport_error"
        record.body_sha256===nothing && record.body_bytes==0 && isempty(record.raw_relative_path) || throw(ArgumentError("transport failure claims received body"))
    else
        throw(ArgumentError("unknown observation capture outcome"))
    end
    return (;record,body)
end
load_receipt(root,reference)=_load_receipt_with_body(root,reference).record

# A single audit binds many forecasts to the same response. Validate the bytes
# once, then index their exact parsed rows for this operation only. No prepared
# source survives the operation: the next public call rechecks the disk bytes.
function _observation_source(root,reference)
    loaded=_load_receipt_with_body(root,reference);record=loaded.record
    receipt_clock_eligible(record) || throw(ArgumentError("observation receipt has ineligible actual clocks; raw evidence retained"))
    record.capture_outcome=="http_response" && record.http_status==200 || throw(ArgumentError("observation receipt is not HTTP200"))
    rows=try
        JSON3.read(loaded.body)
    catch
        throw(ArgumentError("observation response is not valid JSON"))
    end
    rows isa AbstractVector || throw(ArgumentError("observation response is not an array"))
    by_target=Dict{DateTime,Vector{NamedTuple{(:value,:row_sha256),Tuple{Float64,String}}}}()
    for row in rows
        row isa AbstractDict || continue
        tag=get(row,:time_tag,nothing);tag isa AbstractString || continue
        parsed=tryparse(DateTime,chopsuffix(String(tag),"Z"));parsed===nothing && continue
        raw=get(row,:dst,nothing)
        (raw===nothing || raw isa Bool || !(raw isa Real || raw isa AbstractString)) && continue
        parsed_value=tryparse(Float64,string(raw))
        parsed_value===nothing || !isfinite(parsed_value) || abs(parsed_value)>9000 ? continue : nothing
        matches=get!(()->NamedTuple{(:value,:row_sha256),Tuple{Float64,String}}[],by_target,parsed)
        push!(matches,(value=parsed_value,row_sha256=_sha(_bytes(row))))
    end
    return (;record,by_target)
end
function _cached_source!(cache,root,reference)
    key=(String(reference.relative_path),String(reference.sha256))
    return get!(() -> _observation_source(root,reference),cache,key)
end
function _bind_observation(source,target::DateTime,value::Real)
    value isa Bool && throw(ArgumentError("Boolean is not an observation"))
    isfinite(value) || throw(ArgumentError("observation must be finite"))
    mod(Dates.value(target),3_600_000)==0 || throw(ArgumentError("Kyoto provider target tag is not an hourly grid point"))
    record=source.record
    target<=_time(record.receipt_completed_utc) || throw(ArgumentError("provider target tag follows receipt"))
    matches=get(source.by_target,target,())
    isempty(matches) && throw(ArgumentError("receipt omits the finite target observation"))
    all(r->r.value==Float64(value),matches) || throw(ArgumentError("receipt value differs or provider target is ambiguous"))
    finish=try
        DateTime(Dates.UTInstant{Millisecond}(Millisecond(Base.Checked.checked_add(Dates.value(target),Int64(3_600_000)))))
    catch
        throw(ArgumentError("nominal provider interval end overflows DateTime"))
    end
    return (;target_tag_utc=_utc(target),target_nominal_end_utc=_utc(finish),
        receipt_precedes_nominal_end=_time(record.receipt_completed_utc)<finish,
        provider_value=Float64(value),provider_row_sha256=first(matches).row_sha256,
        duplicate_equal_rows=length(matches),receipt_completed_utc=String(record.receipt_completed_utc))
end
bind_observation(root,reference,target::DateTime,value::Real)=
    _bind_observation(_observation_source(root,reference),target,value)

_cell(row,name,default=nothing)=hasproperty(row,name) && !ismissing(getproperty(row,name)) ? getproperty(row,name) : default
function forecast_binding(row)
    issue=_time(_cell(row,:issue_time_utc));origin=_time(_cell(row,:latest_dst_time_utc));target=_time(_cell(row,:target_time_utc))
    origin<=issue<target || throw(ArgumentError("forecast evidence chronology is not future"))
    step=_cell(row,:model_step_hours)
    step isa Real && !(step isa Bool) && isfinite(step) && isinteger(step) && step>0 || throw(ArgumentError("invalid forecast model step"))
    step_int,step_ms,actual_ms=try
        integer=Int(step)
        (integer,Base.Checked.checked_mul(Int64(integer),Int64(3_600_000)),
            Base.Checked.checked_sub(Dates.value(target),Dates.value(origin)))
    catch
        throw(ArgumentError("forecast model-step arithmetic overflows"))
    end
    actual_ms==step_ms || throw(ArgumentError("forecast target and model step differ"))
    model=String(_cell(row,:sub_hourly_model_version,_cell(row,:model_version,"")))
    isempty(model) && throw(ArgumentError("forecast model identity is missing"))
    point=Float64(_cell(row,:served_pred_dst_nt,_cell(row,:pred_dst_nt)))
    lo=Float64(_cell(row,:served_pred_dst_ci05_nt,_cell(row,:pred_dst_ci05_nt)))
    hi=Float64(_cell(row,:served_pred_dst_ci95_nt,_cell(row,:pred_dst_ci95_nt)))
    all(isfinite,(point,lo,hi)) && lo<=point<=hi || throw(ArgumentError("invalid forecast point or interval"))
    return (;issue_time_utc=_utc(issue),origin_tag_utc=_utc(origin),target_tag_utc=_utc(target),
        model_step_hours=step_int,model_identity=model,
        manifest_sha256=String(_cell(row,:superlearner_manifest_sha256,"")),
        point_dst_nt=point,lower_dst_nt=lo,upper_dst_nt=hi)
end
_binding_key(binding)=_sha(_bytes(binding))
function _table_snapshot(log_path)
    path=_regular_path(log_path)
    isfile(path) || throw(ArgumentError("forecast log is missing"))
    bytes=read(path)
    return (;sha256=_sha(bytes),table=CSV.read(IOBuffer(bytes),DataFrame))
end
function _find_row(table,binding)
    issue=_time(binding.issue_time_utc);origin=_time(binding.origin_tag_utc);target=_time(binding.target_tag_utc)
    matches=Int[]
    for i in 1:nrow(table)
        _time(table.issue_time_utc[i])==issue && _time(table.latest_dst_time_utc[i])==origin && _time(table.target_time_utc[i])==target || continue
        _binding_key(forecast_binding(table[i,:]))==_binding_key(binding) && push!(matches,i)
    end
    length(matches)==1 || throw(ArgumentError("forecast identity is missing, changed or duplicated in log"))
    return only(matches)
end
function prepare_verification!(root,receipt,entries;log_path::AbstractString,verification_started_utc::DateTime)
    storage=_root(root);source=_observation_source(storage,receipt)
    verification_started_utc>=_time(source.record.receipt_completed_utc) || throw(ArgumentError("verification starts before receipt"))
    state=verify_observation_evidence(storage)
    state.verification_completed_utc===nothing || verification_started_utc>=_time(state.verification_completed_utc) ||
        throw(ArgumentError("verification decision predates current journal head"))
    before=_table_snapshot(log_path);prepared=NamedTuple[];keys=Set{String}()
    sources=Dict{Tuple{String,String},Any}()
    for entry in entries
        binding=entry.binding;key=_binding_key(binding)
        key in keys && throw(ArgumentError("duplicate forecast in verification intent"));push!(keys,key)
        _require_next_observation!(sources,storage,source,receipt,get(state.latest_by_forecast,key,nothing),verification_started_utc)
        index=_find_row(before.table,binding)
        prior=_obs(before.table.observation_dst_nt[index])
        _equal(prior,_obs(entry.previous_observation)) || throw(ArgumentError("previous logged observation differs"))
        entry.observation isa Real && !(entry.observation isa Bool) && isfinite(entry.observation) || throw(ArgumentError("verification observation must be finite and numeric"))
        observed=Float64(entry.observation)
        bound=_bind_observation(source,_time(binding.target_tag_utc),observed)
        push!(prepared,(;forecast_key=key,binding,previous_observation=prior,
            observation=observed,observation_binding=bound))
    end
    isempty(prepared) && throw(ArgumentError("empty verification intent"))
    payload=(;schema_version=SCHEMA,receipt,log_path=abspath(String(log_path)),
        log_before_sha256=before.sha256,verification_started_utc=_utc(verification_started_utc),
        process_id=getpid(),attempt_serial=Threads.atomic_add!(_SERIAL,UInt64(1))+1,entries=prepared)
    return _store(storage,"intents",payload)
end

function _with_journal_lock(f,root)
    _with_forecast_log_lock(f,joinpath(root,"journal");timeout_sec=30.0)
end
_journal_relative(sequence)=_relative("journal",lpad(string(sequence),20,'0')*".json")
function _head(root)
    path=_path(root,"head.json")
    isfile(path) || return (;sequence=0,sha256=ZERO)
    value=try JSON3.read(read(path)) catch; throw(ArgumentError("invalid observation journal head")) end
    Int(value.sequence)>=1 && occursin(r"^[0-9a-f]{64}$",String(value.sha256)) || throw(ArgumentError("invalid observation journal head fields"))
    return (;sequence=Int(value.sequence),sha256=String(value.sha256))
end
function _write_head(root,sequence,digest)
    target=_path(root,"head.json");bytes=_bytes((;sequence,sha256=digest))
    mktemp(dirname(target)) do path,io
        write(io,bytes);_flush_file(io);close(io)
        _regular_path(target)
        ispath(target) && !isfile(target) && throw(ArgumentError("journal head is not a regular file"))
        Base.Filesystem.rename(path,target)
    end
    _flush_parent(target)
    return nothing
end
function _event_kind(prior,previous,observed)
    prior===nothing && return previous===nothing ? "first_verification" :
        previous==observed ? "reconfirmed_legacy_observation" : "revision_of_legacy_observation"
    return prior.observation==observed ? "reconfirmed_observation" : "revised_observation"
end
function _require_next_observation!(sources,root,source,reference,prior,decision)
    prior===nothing && return nothing
    decision>=_time(prior.verification_completed_utc) || throw(ArgumentError("forecast revision decision predates prior verification"))
    previous=_cached_source!(sources,root,prior.receipt).record;next=source.record
    _time(next.request_started_utc)>=_time(previous.request_started_utc) &&
        _time(next.receipt_completed_utc)>=_time(previous.receipt_completed_utc) ||
        throw(ArgumentError("observation snapshot is older or incomparable with prior verified receipt"))
    same_clocks=next.request_started_utc==previous.request_started_utc && next.receipt_completed_utc==previous.receipt_completed_utc
    strict_monotonic_order=false
    session=get(next,:capture_session_id,nothing)
    if session!==nothing && session==get(previous,:capture_session_id,nothing)
        started=parse(BigInt,string(next.monotonic_started_ns));prior_started=parse(BigInt,string(previous.monotonic_started_ns))
        completed=parse(BigInt,string(next.monotonic_completed_ns));prior_completed=parse(BigInt,string(previous.monotonic_completed_ns))
        started>=prior_started && completed>=prior_completed ||
            throw(ArgumentError("same-session observation monotonic clocks regress"))
        strict_monotonic_order=started>prior_started || completed>prior_completed
    end
    same_clocks && !strict_monotonic_order && next.body_sha256!=previous.body_sha256 &&
        throw(ArgumentError("conflicting observation snapshots have no strict measured clock order"))
    return nothing
end
function _verify_unlocked(root;require_complete=false,head_override=nothing)
    sources=Dict{Tuple{String,String},Any}()
    head=head_override===nothing ? _head(root) : head_override
    journal_dir=_path(root,"journal")
    files=isdir(journal_dir) ? readdir(journal_dir) : String[]
    length(files)==head.sequence || throw(ArgumentError("observation journal has missing or uncommitted sequence files"))
    expected=head.sha256;records=Any[];committed=Set{String}()
    for sequence in head.sequence:-1:1
        ref=(;relative_path=_journal_relative(sequence),sha256=expected)
        record=_read(root,ref)
        record.schema_version==SCHEMA && record.sequence==sequence || throw(ArgumentError("observation journal sequence or schema changed"))
        object=_read(root,(;relative_path=_relative("commits",expected*".json"),sha256=expected))
        _bytes(record)==_bytes(object) || throw(ArgumentError("journal and immutable commit object differ"))
        push!(records,(;record,sha256=expected));expected=String(record.previous_journal_sha256)
    end
    expected==ZERO || throw(ArgumentError("observation journal origin changed"))
    reverse!(records);events=Any[];latest=Dict{String,Any}();last_completed=nothing
    for item in records
        record=item.record;intent=_read(root,record.intent)
        intent.schema_version==SCHEMA || throw(ArgumentError("verification intent schema changed"))
        String(record.intent.sha256) in committed && throw(ArgumentError("verification intent committed more than once"))
        push!(committed,String(record.intent.sha256))
        length(intent.entries)==length(record.events) || throw(ArgumentError("verification entries differ"))
        source=_cached_source!(sources,root,intent.receipt)
        _time(record.verification_completed_utc)>=_time(intent.verification_started_utc)>=_time(source.record.receipt_completed_utc) || throw(ArgumentError("verification clocks regress"))
        completed=_time(record.verification_completed_utc)
        last_completed===nothing || completed>=last_completed || throw(ArgumentError("journal verification completion regresses"))
        last_completed=completed
        for (index,event) in enumerate(record.events)
            entry=intent.entries[index];key=_binding_key(entry.binding)
            String(entry.forecast_key)==key && String(event.forecast_key)==key || throw(ArgumentError("forecast evidence key changed"))
            _bytes(event.binding)==_bytes(entry.binding) && event.observation==entry.observation || throw(ArgumentError("committed forecast or observation differs from intent"))
            _equal(_obs(event.previous_observation),_obs(entry.previous_observation)) || throw(ArgumentError("previous observation changed"))
            bound=_bind_observation(source,_time(entry.binding.target_tag_utc),entry.observation)
            _bytes(bound)==_bytes(entry.observation_binding) || throw(ArgumentError("raw observation binding changed"))
            prior=get(latest,key,nothing)
            _require_next_observation!(sources,root,source,intent.receipt,prior,_time(intent.verification_started_utc))
            expected_prior=prior===nothing ? "" : String(prior.event_key)
            String(event.prior_event_key)==expected_prior || throw(ArgumentError("forecast revision chain changed"))
            String(event.kind)==_event_kind(prior,_obs(entry.previous_observation),entry.observation) || throw(ArgumentError("forecast revision kind changed"))
            value=(;forecast_key=key,binding=event.binding,observation=Float64(event.observation),
                previous_observation=_obs(event.previous_observation),kind=String(event.kind),
                prior_event_key=String(event.prior_event_key),event_key=item.sha256*":"*string(index),
                receipt=intent.receipt,verification_completed_utc=String(record.verification_completed_utc),
                receipt_completed_utc=String(source.record.receipt_completed_utc),
                observation_binding=entry.observation_binding)
            push!(events,value);latest[key]=value
        end
    end
    intents_dir=_path(root,"intents");pending=String[];failures=Any[]
    if isdir(intents_dir)
        for name in readdir(intents_dir)
            endswith(name,".json") || throw(ArgumentError("unexpected file in intent archive"))
            digest=chopsuffix(name,".json")
            intent=_read(root,(;relative_path=_relative("intents",name),sha256=digest))
            intent.schema_version==SCHEMA || throw(ArgumentError("verification intent schema changed"))
            source=_cached_source!(sources,root,intent.receipt)
            _time(intent.verification_started_utc)>=_time(source.record.receipt_completed_utc) || throw(ArgumentError("intent precedes observation receipt"))
            for entry in intent.entries
                String(entry.forecast_key)==_binding_key(entry.binding) || throw(ArgumentError("intent forecast key changed"))
                bound=_bind_observation(source,_time(entry.binding.target_tag_utc),entry.observation)
                _bytes(bound)==_bytes(entry.observation_binding) || throw(ArgumentError("intent raw observation binding changed"))
            end
            resolution_path=_path(root,_relative("resolutions",digest*".json"))
            if isfile(resolution_path)
                digest in committed && throw(ArgumentError("intent cannot be both a committed verification and failed attempt"))
                resolution=try JSON3.read(read(resolution_path)) catch;throw(ArgumentError("invalid failed-attempt reference")) end
                failure=_read(root,resolution)
                failure.schema_version==SCHEMA && String(failure.intent.sha256)==digest &&
                    String(failure.disposition)=="failed_attempt_no_verification_claim" || throw(ArgumentError("failed-attempt intent or disposition changed"))
                failure_clock_valid=_time(failure.failed_at_utc)>=_time(intent.verification_started_utc)
                get(failure,:failure_clock_eligible,failure_clock_valid)==failure_clock_valid &&
                    get(failure,:failure_clock_status,failure_clock_valid ? "ordered" : "utc_regression")==
                        (failure_clock_valid ? "ordered" : "utc_regression") ||
                    throw(ArgumentError("failed-attempt clock eligibility changed"))
                push!(failures,failure)
            elseif !(digest in committed)
                push!(pending,digest)
            end
        end
    end
    require_complete && !isempty(pending) && throw(ArgumentError("observation evidence has uncommitted verification intents"))
    return (;sequence=head.sequence,sha256=head.sha256,events,latest_by_forecast=latest,
        verification_completed_utc=last_completed===nothing ? nothing : _utc(last_completed),
        pending_intents=pending,committed_intents=committed,failed_attempts=failures)
end
function verify_observation_evidence(root;require_complete=false)
    storage=_root(root)
    _with_journal_lock(storage) do
        _verify_unlocked(storage;require_complete)
    end
end
latest_forecast_evidence(state,binding)=get(state.latest_by_forecast,_binding_key(binding),nothing)

function commit_verification!(root,intent_ref;log_path::AbstractString,
        verification_completed_utc::DateTime,crash_hook::Function=stage->nothing)
    storage=_root(root)
    _with_journal_lock(storage) do
        state=_verify_unlocked(storage)
        String(intent_ref.sha256) in state.committed_intents && throw(ArgumentError("verification intent was already committed"))
        any(f->f.intent.sha256==intent_ref.sha256,state.failed_attempts) && throw(ArgumentError("a failed attempt cannot become a committed verification"))
        intent=_read(storage,intent_ref)
        abspath(String(log_path))==String(intent.log_path) || throw(ArgumentError("verification log path differs"))
        source=_observation_source(storage,intent.receipt)
        verification_completed_utc>=_time(intent.verification_started_utc)>=_time(source.record.receipt_completed_utc) || throw(ArgumentError("verification completion precedes receipt or decision"))
        state.verification_completed_utc===nothing || verification_completed_utc>=_time(state.verification_completed_utc) ||
            throw(ArgumentError("journal verification completion would regress"))
        actual=_table_snapshot(log_path);events=NamedTuple[]
        sources=Dict{Tuple{String,String},Any}()
        for entry in intent.entries
            index=_find_row(actual.table,entry.binding)
            _equal(_obs(actual.table.observation_dst_nt[index]),_obs(entry.observation)) || throw(ArgumentError("intended observation is not committed in live table"))
            _bind_observation(source,_time(entry.binding.target_tag_utc),entry.observation)
            key=String(entry.forecast_key);prior=get(state.latest_by_forecast,key,nothing)
            _require_next_observation!(sources,storage,source,intent.receipt,prior,_time(intent.verification_started_utc))
            push!(events,(;forecast_key=key,binding=entry.binding,observation=Float64(entry.observation),
                previous_observation=_obs(entry.previous_observation),
                kind=_event_kind(prior,_obs(entry.previous_observation),entry.observation),
                prior_event_key=prior===nothing ? "" : String(prior.event_key)))
        end
        sequence=state.sequence+1
        record=(;schema_version=SCHEMA,sequence,previous_journal_sha256=state.sha256,
            intent=intent_ref,verification_completed_utc=_utc(verification_completed_utc),
            log_after_sha256=actual.sha256,events)
        object=_store(storage,"commits",record)
        relative=_journal_relative(sequence)
        publish_immutable(_path(storage,relative),_bytes(record))
        crash_hook(:after_journal_record)
        _write_head(storage,sequence,object.sha256)
        crash_hook(:after_journal_head)
        _verify_unlocked(storage)
        return (;relative_path=relative,sha256=object.sha256)
    end
end

"Retain an actual failed write attempt without claiming a completed verification."
function record_verification_failure!(root,intent_ref,error;log_path::AbstractString,failed_at_utc::DateTime)
    storage=_root(root)
    _with_journal_lock(storage) do
        state=_verify_unlocked(storage)
        String(intent_ref.sha256) in state.committed_intents && throw(ArgumentError("a committed verification cannot be marked failed"))
        intent=_read(storage,intent_ref)
        String(intent.log_path)==abspath(String(log_path)) || throw(ArgumentError("failed-attempt log identity differs"))
        failure_clock_valid=failed_at_utc>=_time(intent.verification_started_utc)
        resolution_relative=_relative("resolutions",String(intent_ref.sha256)*".json")
        resolution_path=_path(storage,resolution_relative)
        if isfile(resolution_path)
            prior=JSON3.read(read(resolution_path));_read(storage,prior)
            return prior
        end
        regular=isfile(log_path)&&!islink(log_path)
        digest=regular ? _sha(read(log_path)) : nothing
        failure=(;schema_version=SCHEMA,intent=intent_ref,
            disposition="failed_attempt_no_verification_claim",failed_at_utc=_utc(failed_at_utc),
            failure_clock_eligible=failure_clock_valid,
            failure_clock_status=failure_clock_valid ? "ordered" : "utc_regression",
            error_type=string(typeof(error)),error_message=sprint(showerror,error),
            log_state_at_failure=regular ? "captured_hash_only_commit_status_unknown" : "missing_or_nonregular",
            log_sha256_at_failure=digest,original_verification_completed_utc=nothing)
        reference=_store(storage,"failures",failure)
        publish_immutable(resolution_path,_bytes(reference))
        _verify_unlocked(storage)
        return reference
    end
end

"Recover only an already written, content-bound commit whose head install was interrupted."
function recover_observation_head!(root)
    storage=_root(root)
    _with_journal_lock(storage) do
        head=_head(storage);relative=_journal_relative(head.sequence+1);path=_path(storage,relative)
        isfile(path) || return _verify_unlocked(storage)
        digest=_sha(read(path))
        _read(storage,(;relative_path=_relative("commits",digest*".json"),sha256=digest))
        candidate=(;sequence=head.sequence+1,sha256=digest)
        _verify_unlocked(storage;head_override=candidate)
        _write_head(storage,candidate.sequence,candidate.sha256)
        return _verify_unlocked(storage)
    end
end

end
