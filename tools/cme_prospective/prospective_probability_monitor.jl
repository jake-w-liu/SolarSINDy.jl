module CMEProspectiveMonitor

using Dates, JSON3, SHA, HTTP, Sockets
export verify_export, verify_collector_source, collect!, write_status, recoverable_acquisition

digest(path)=open(io->bytes2hex(sha256(io)),path)

function verify_collector_source(path,expected_sha)
    occursin(r"^[a-f0-9]{64}$",expected_sha) || error("pinned launcher collector SHA256 is required")
    isfile(path) && !islink(path) && digest(path)==expected_sha ||
        error("collector source differs from pinned launcher")
    return expected_sha
end

function transport_failure(exception)
    # HTTP's timeout boundary preserves an underlying error in CapturedException.
    # Inspect it only after recoverable_acquisition has admitted an HTTP boundary;
    # a captured local file error must remain fatal. Unknown/deep chains fail closed.
    for depth in 0:32
        exception isa InterruptException && return false
        exception isa HTTP.TimeoutError && return true
        exception isa Union{Base.IOError,EOFError,Sockets.DNSError} && return true
        if exception isa Union{HTTP.ConnectError,HTTP.RequestError}
            depth<32 || return false
            exception=exception.error
        elseif exception isa CapturedException
            depth<32 || return false
            exception=exception.ex
        else
            return false
        end
    end
    false
end

"Only known transport failures or exact producer transient-status errors may retry."
function recoverable_acquisition(error)
    error isa InterruptException && return false
    error isa Union{HTTP.TimeoutError,HTTP.ConnectError,HTTP.RequestError} && return transport_failure(error)
    return error isa ErrorException && occursin(r"^DONKI HTTP status (408|425|429|5[0-9]{2})$",error.msg)
end

"Verify an export against an independently freeze-bound receipt, before loading it."
function verify_export(snapshot,expected_receipt_sha;expected_members=401)
    receipt_path=joinpath(snapshot,"SOURCE_EXPORT_RECEIPT.json")
    digest(receipt_path)==expected_receipt_sha || error("production export receipt differs")
    receipt=JSON3.read(read(receipt_path))
    files=merge(Dict(String(k)=>String(v) for (k,v) in pairs(receipt.package_members)),
                Dict(String(k)=>String(v) for (k,v) in pairs(receipt.producer_members)))
    length(files)==expected_members || error("production export membership differs")
    for (name,sha) in files
        isabspath(name) && error("absolute export member")
        any(==(".."),split(replace(name,'\\'=>'/'),'/')) && error("unsafe export member")
        path=joinpath(snapshot,split(name,'/')...)
        isfile(path) && !islink(path) && digest(path)==sha || error("changed production source: $name")
    end
    return receipt
end

"Atomic replaceable operational status; scientific records are owned by the producer."
function write_status(path,value)
    mkpath(dirname(path))
    temporary,io=mktemp(dirname(path))
    try
        write(io,JSON3.write(value),"\n");close(io)
        mv(temporary,path;force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
end

"One collector, fixed hourly cadence, identity checks before every real acquisition."
function collect!(store,actions;deadline,interval_seconds=3600.0,max_cycles=0,
                  clock=()->now(UTC),sleep_fn=sleep,status_fn=x->nothing,
                  retryable=recoverable_acquisition)
    isfinite(interval_seconds) && interval_seconds>0 || throw(ArgumentError("invalid collection interval"))
    max_cycles isa Int && max_cycles>=0 || throw(ArgumentError("invalid cycle limit"))
    isdir(store) && !islink(store) || error("reviewed store must exist")
    claim=joinpath(store,".prospective_collector.claim")
    mkdir(claim) # exclusive; never clear an existing collector's claim
    cycles=0
    previous=nothing
    try
        while true
            actions.verify() # corruption is fatal, outside acquisition error recovery
            started=clock()
            previous===nothing || started>=previous || error("collector UTC clock moved backwards")
            started<=deadline || break
            result="ok"; detail=""; capture_key=""; batch_key=""; capture=nothing
            try
                capture=actions.capture()
            catch error
                error isa InterruptException && rethrow()
                retryable(error) || rethrow()
                result="acquisition_failed"
                detail=sprint(showerror,error)
            end
            if capture!==nothing
                capture_key=basename(capture)
                # A batch/publication failure may follow partially issued rows.
                # Stop for inspection; never silently continue with an unaudited batch.
                batch=actions.issue_all(capture)
                batch_key=basename(batch)
            end
            finished=clock()
            finished>=started || error("collector UTC clock moved backwards")
            cycles+=1
            status_fn((;schema_version=1,pid=getpid(),cycle=cycles,started_at_utc=string(started),
                completed_at_utc=string(finished),status=result,error=detail,capture_key,batch_key,
                collection_end_utc=string(deadline),outcomes_retrieved=false,scores_written=false))
            previous=finished
            (max_cycles>0 && cycles>=max_cycles || finished>=deadline) && break
            remaining=interval_seconds-Dates.value(finished-started)/1000
            # A slow cycle is followed by a full interval; no catch-up burst.
            delay=remaining>0 ? remaining : interval_seconds
            sleep_fn(min(delay,max(0.0,Dates.value(deadline-finished)/1000)))
        end
        return cycles
    finally
        rm(claim) # only the empty directory successfully created by this process
    end
end
end # module

if abspath(PROGRAM_FILE)==@__FILE__
    using Dates, JSON3
    length(ARGS) in (3,4) || error("usage: prospective_probability_monitor.jl SNAPSHOT STORE FREEZE_KEY [MAX_CYCLES]")
    snapshot,store,key=abspath(ARGS[1]),abspath(ARGS[2]),ARGS[3]
    occursin(r"^[a-f0-9]{64}$",key) || error("invalid freeze key")
    freeze=joinpath(store,"freezes",key)
    review=JSON3.read(read(joinpath(freeze,"review_receipt.json")))
    VERSION==v"1.12.6" && Base.JLOptions().startupfile==2 && Base.JLOptions().use_pkgimages==0 ||
        error("use reviewed Julia1.12.6 --startup-file=no --pkgimages=no")
    Base.active_project()==joinpath(snapshot,"SolarSINDy.jl","Project.toml") || error("production project differs")
    expected_members=length(review.source_members)
    CMEProspectiveMonitor.verify_export(snapshot,String(review.source_export_receipt_sha256);expected_members)
    collector_sha256=get(ENV,"CME_COLLECTOR_SHA256","")
    CMEProspectiveMonitor.verify_collector_source(@__FILE__,collector_sha256)
    include(joinpath(snapshot,"lead_time","prospective_probability.jl"))
    P=CMEProspectiveForecast
    function verify_production()
        CMEProspectiveMonitor.verify_collector_source(@__FILE__,collector_sha256)
        isfile(joinpath(store,"RETIREMENT.json")) && error("collection store is retired")
        CMEProspectiveMonitor.verify_export(snapshot,String(review.source_export_receipt_sha256);expected_members)
        meta,_=P.load_freeze(freeze)
        P.same_store(store,meta)
        return nothing
    end
    verify_production()
    output=joinpath(store,"monitor")
    mkpath(output)
    actions=(;verify=verify_production,capture=()->P.capture!(store,freeze),
              issue_all=capture->P.issue_capture!(store,freeze,capture))
    status_fn=value->begin
        value=(;value...,collector_sha256,freeze_key=key)
        CMEProspectiveMonitor.write_status(joinpath(output,"status.json"),value)
        open(joinpath(output,"cycles.jsonl"),"a") do io
            write(io,JSON3.write(value),"\n")
        end
        println(JSON3.write(value));flush(stdout)
    end
    CMEProspectiveMonitor.collect!(store,actions;deadline=P.COLLECTION_END,
        max_cycles=length(ARGS)==4 ? parse(Int,ARGS[4]) : 0,status_fn)
end
