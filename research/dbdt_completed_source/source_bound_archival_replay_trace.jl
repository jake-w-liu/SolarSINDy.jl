const TRACE_DOMAIN="DBDT_ARCHIVAL_REPLAY_TRACE_V1\0"
# At most16 subscribers, with at most30 partial-rate transitions per issue,
# plus issuance/terminal bookkeeping, on a single600000-row case table.
const MAX_TRACE_SECTION_RECORDS=32*16*600000
function _trace_value(value,depth=0)
    need(depth<=16,"replay trace nesting bound")
    if value isa UInt64
        return Dict("type"=>"UInt64","bits"=>string(value;base=16,pad=16))
    elseif value===nothing || value isa Bool || value isa Integer || value isa AbstractString
        return value
    elseif value isa Float64
        return Dict("type"=>"Float64","bits"=>F.bits(value))
    elseif value isa DateTime
        return Dict("type"=>"DateTime","utc"=>F.clock(value))
    elseif value isa Symbol
        return Dict("type"=>"Symbol","value"=>String(value))
    elseif value isa NamedTuple
        return Dict(String(key)=>_trace_value(item,depth+1) for (key,item) in pairs(value))
    elseif value isa AbstractDict
        need(all(key->key isa AbstractString || key isa Symbol,keys(value)) && allunique(String.(collect(keys(value)))),
            "replay trace dictionary keys")
        return Dict(String(key)=>_trace_value(item,depth+1) for (key,item) in value)
    elseif value isa Tuple || value isa AbstractVector
        value isa AbstractVector && Base.require_one_based_indexing(value)
        need(length(value)<=8192,"replay trace nested array bound")
        return [_trace_value(item,depth+1) for item in value]
    elseif value isa D.CMR.ReplayPolicy || value isa E.SubscriberSpec || value isa E.WindowState
        return Dict("type"=>String(nameof(typeof(value))),"fields"=>Dict(String(key)=>_trace_value(getfield(value,key),depth+1) for key in fieldnames(typeof(value))))
    end
    throw(ArgumentError("unsupported replay trace data type"))
end
function _trace_frame!(context,bytes)
    need(bytes isa Vector{UInt8} && 0<length(bytes)<=262144,"replay trace frame bound")
    count=UInt64(length(bytes));prefix=UInt8[UInt8((count>>shift)&0xff) for shift in 0:8:56]
    SHA.update!(context,prefix);SHA.update!(context,bytes);nothing
end
function _trace_start(request_sha256)
    context=SHA.SHA2_256_CTX();SHA.update!(context,codeunits(TRACE_DOMAIN))
    _trace_frame!(context,R.canonical_bytes(Dict("schema"=>"archival_shared_replay_numeric_trace_v1",
        "request_sha256"=>J._sha(request_sha256));limit=4096));context
end
function _trace_section!(context,name,records)
    need(name isa String && 0<ncodeunits(name)<=128,"replay trace section name")
    count=length(records);need(0<=count<=MAX_TRACE_SECTION_RECORDS,"replay trace section count")
    _trace_frame!(context,R.canonical_bytes(Dict("section"=>name,"records"=>count);limit=4096))
    seen=0
    for row in records
        seen+=1;need(seen<=count,"replay trace section grew during iteration")
        _trace_frame!(context,R.canonical_bytes(_trace_value(row);limit=262144))
    end
    need(seen==count,"replay trace section changed during iteration");nothing
end
function _trace_fields!(context,name,value,excluded=())
    selected=[key for key in keys(value) if key ∉ excluded]
    _trace_section!(context,name,((;name=String(key),value=getproperty(value,key)) for key in selected))
end
function _trace_owner!(context,owner)
    snapshot=owner.snapshot
    selected=[key for key in fieldnames(typeof(snapshot)) if key ∉ (:calibration,:subscribers,:pending,:seen)]
    _trace_section!(context,"owner.snapshot",((;name=String(key),value=getfield(snapshot,key)) for key in selected))
    _trace_fields!(context,"owner.calibration",snapshot.calibration,(:chronological_scores,:sorted_scores))
    for key in (:chronological_scores,:sorted_scores)
        _trace_section!(context,"owner.calibration."*String(key),getproperty(snapshot.calibration,key))
    end
    _trace_section!(context,"owner.pending",snapshot.pending);_trace_section!(context,"owner.seen",snapshot.seen)
    _trace_section!(context,"owner.subscriber_count",(length(snapshot.subscribers),))
    for (index,state) in enumerate(snapshot.subscribers)
        keys_without_scores=[key for key in fieldnames(typeof(state)) if key!==:scores]
        _trace_section!(context,"owner.subscriber."*string(index),((;name=String(key),value=getfield(state,key)) for key in keys_without_scores))
        _trace_section!(context,"owner.subscriber."*string(index)*".scores",state.scores)
    end
end
function _trace_publisher!(context,name,portfolio)
    _trace_fields!(context,name*".summary",portfolio,(:issued,:updates,:warmup_ids,:states))
    _trace_section!(context,name*".issued",portfolio.issued);_trace_section!(context,name*".updates",portfolio.updates)
    _trace_section!(context,name*".warmup",portfolio.warmup_ids)
    _trace_section!(context,name*".states",((;nominal=state.nominal,resolved=state.resolved,
        misses=state.misses,log_wealth=state.log_wealth) for state in portfolio.states))
end
"""Digest typed numeric replay data incrementally; this pure function grants no authority.

Explicit floating words preserve Inf, NaN and signed zero. The encoding never
deserializes Julia objects or follows a data-supplied path. Full arrays are framed
row by row; the bounded live owner and final publisher states are also included.
"""
function result_digest(result::NamedTuple,feature_rows;request_sha256)
    need((feature_rows isa AbstractVector || feature_rows isa Tuple) && length(feature_rows)==result.audit.scheduled &&
        0<=length(feature_rows)<=600000,"replay trace scheduled feature rows")
    context=_trace_start(request_sha256)
    _trace_fields!(context,"result.summary",result,(:shared,:publisher_upper,:publisher_absolute,:static_bounds,
        :row_ids,:accepted_rows,:centers,:scales,:receipt_rejections,:receipt_pending))
    _trace_section!(context,"features",feature_rows)
    for key in (:row_ids,:accepted_rows,:centers,:scales,:receipt_rejections,:receipt_pending)
        haskey(result,key) && _trace_section!(context,"result."*String(key),getproperty(result,key))
    end
    if result.shared!==nothing
        shared=result.shared
        _trace_fields!(context,"shared.summary",shared,(:owner,:events,:subscriber_events,:issue_records,:subscribers,:indices))
        for key in (:events,:subscriber_events,:issue_records,:subscribers,:indices)
            _trace_section!(context,"shared."*String(key),getproperty(shared,key))
        end
        _trace_owner!(context,shared.owner)
        _trace_publisher!(context,"publisher.upper",result.publisher_upper)
        _trace_publisher!(context,"publisher.absolute",result.publisher_absolute)
    end
    if result.static_bounds!==nothing
        fixed=result.static_bounds
        _trace_fields!(context,"static.summary",fixed,(:bounds,))
        count=length(result.accepted_rows)
        need(all(bounds->length(bounds)==count,values(fixed.bounds)),"replay trace static denominator")
        _trace_section!(context,"static.bounds",((;row_id=result.accepted_rows[i],
            het_gaussian=fixed.bounds.het_gaussian[i],raw_split=fixed.bounds.raw_split[i],
            normalized_split=fixed.bounds.normalized_split[i],regime_split=fixed.bounds.regime_split[i]) for i in 1:count))
    end
    bytes2hex(SHA.digest!(context))
end
