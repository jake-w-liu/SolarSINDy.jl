"""One-pass current-state bridge for explicitly admitted archival receipt order."""
module DurationCurrentStateStream

using Dates, SHA
using ..DurationConditionedForcing
const D=DurationConditionedForcing
const LOADED_SELF=bytes2hex(sha256(read(@__FILE__)))
const LOADED_D=D.source_identity()
const CHECKPOINT_SCHEMA="b_archival_current_state_checkpoint_v1"
const Hashes=Tuple{Vararg{Pair{String,String}}}
export CurrentStateCheckpoint, with_current_state_tracker

_need(ok,message)=ok || throw(ArgumentError(message))
function source_identity()
    _need(bytes2hex(sha256(read(@__FILE__)))==LOADED_SELF,"loaded current-state bridge source changed")
    _need(D.source_identity()==LOADED_D,"loaded duration/A dependencies changed")
    (LOADED_D...,"duration_current_state_stream.jl"=>LOADED_SELF)
end

"""Immutable, data-only in-memory restart point; resume validates it by one replay.

This is a distinct archival bridge checkpoint, not an A checkpoint or fitted
duration-model artifact. It carries no executable or serialized SHA context.
"""
struct CurrentStateCheckpoint
    schema::String
    source_sha256::String
    wind_sequence_sha256::String
    source_row_count::Int
    implementation_hashes::Hashes
    last_issue::Union{Nothing,DateTime}
    consumed_count::Int
    input_prefix_sha256::String
    previous_tag::Union{Nothing,DateTime}
    previous_state::Symbol
    age::Int
    run::Int
    checkpoint_id::String
end

function _checkpoint_data(cp::CurrentStateCheckpoint)
    (;schema=cp.schema,source_sha256=cp.source_sha256,wind_sequence_sha256=cp.wind_sequence_sha256,
      source_row_count=cp.source_row_count,
      implementation_hashes=[(;name=String(first(p)),sha256=String(last(p))) for p in cp.implementation_hashes],
      last_issue=cp.last_issue,consumed_count=cp.consumed_count,input_prefix_sha256=cp.input_prefix_sha256,
      previous_tag=cp.previous_tag,previous_state=cp.previous_state,age=cp.age,run=cp.run)
end

# All storage has a fixed concrete type. In particular source length never
# becomes a Tuple type parameter, and no mutable context/source escapes in data.
mutable struct _Tracker
    rows::Vector{D.WindHour}
    source::String
    sequence::String
    hashes::Hashes
    owner::Task
    active::Bool
    busy::Bool
    last_issue::Union{Nothing,DateTime}
    count::Int
    context::SHA.SHA2_256_CTX
    previous::Union{Nothing,DateTime}
    state::Symbol
    age::Int
    run::Int
    admission_hash_bytes::Int
    eligible_hash_bytes::Int
    advance_calls::Int
    digest_snapshots::Int
    source_checks::Int
end
function _line(row::D.WindHour)
    bits(x)=isnan(x) ? "missing" : string(reinterpret(UInt64,x))
    join((Dates.value(row.tag),Dates.value(row.completed),Dates.value(row.received),bits(row.bz),bits(row.q)),"|")*"\n"
end
function _admit(rows,source,hashes)
    _need(rows isa AbstractVector && axes(rows)==(Base.OneTo(length(rows)),),"ordinary one-based WindHour vector required")
    _need(length(rows)<=D.MAX_HOURS && all(x->x isa D.WindHour,rows),"bounded WindHour sequence required")
    _need(D._hex(source),"lowercase source digest required")
    copied=Vector{D.WindHour}(rows)
    context=SHA.SHA2_256_CTX();bytes=0
    previous=nothing
    for row in copied
        _need(mod(Dates.value(row.tag),D.HOUR_MS)==0 && row.completed==D._shift(row.tag,1),"exact hourly completion required")
        _need(row.received>=row.completed,"receipt predates completion")
        _need((isfinite(row.bz)||isnan(row.bz))&&(isfinite(row.q)||isnan(row.q)) && (isnan(row.q)||row.q>=0),"invalid wind value")
        if previous!==nothing
            _need(previous.tag<row.tag,"ordered unique source hours required")
            _need(previous.received<=row.received,"archival monotone receipts required; late insertion unsupported")
        end
        D._known(row) # Also checks finite North q==0; missing rows remain missing.
        line=_line(row);SHA.update!(context,codeunits(line));bytes+=ncodeunits(line)
        previous=row
    end
    sequence=bytes2hex(SHA.digest!(context))
    _Tracker(copied,String(source),sequence,hashes,current_task(),true,false,nothing,0,
        SHA.SHA2_256_CTX(),nothing,:missing,0,0,bytes,0,0,0,1)
end
function _scope(tracker)
    _need(tracker.active,"current-state closure is outside its scope")
    _need(current_task()===tracker.owner && !tracker.busy,"current-state closure requires its serial owner task")
end
function _trace_step(row,previous,state,age,run)
    D._known(row) || return (nothing,:missing,0,0)
    next=D._state(row)
    contiguous=previous!==nothing && row.tag==D._shift(previous,1)
    if contiguous && next==state
        run+=1;age=age>0 ? min(age+1,49) : run>=49 ? 49 : 0
    elseif contiguous
        run=1;age=1
    else
        run=1;age=0
    end
    (row.tag,next,age,run)
end
function _advance!(tracker,issue)
    _scope(tracker)
    _need(issue isa DateTime,"DateTime issue required")
    latest_tag=D._shift(issue,-1) # Overflow is checked before any mutable state.
    _need(tracker.last_issue===nothing || issue>=tracker.last_issue,"issue cannot move backwards")
    tracker.busy=true
    try
        count=tracker.count;context=copy(tracker.context)
        previous,state,age,run=tracker.previous,tracker.state,tracker.age,tracker.run
        bytes=0
        while count<length(tracker.rows)
            row=tracker.rows[count+1]
            row.completed<=issue && row.received<=issue || break
            previous,state,age,run=_trace_step(row,previous,state,age,run)
            line=_line(row);SHA.update!(context,codeunits(line));bytes+=ncodeunits(line)
            count+=1
        end
        reason=count==0 || tracker.rows[count].tag!=latest_tag ? :latest_driver_unavailable :
            state==:missing ? :latest_driver_missing : age==0 ? :unknown_initial_age : :available
        q=reason==:available ? tracker.rows[count].q : nothing
        prefix=bytes2hex(SHA.digest!(copy(context)))
        data=(;status=reason==:available ? :available : :unavailable,reason,issue,state,age,q,
            input_prefix_sha256=prefix,availability_kind=:assumed_completed_hour)
        result=merge(data,(state_id=D._hash(data),))
        # Only a complete result commits the candidate context/trace/cursor.
        tracker.count=count;tracker.context=context;tracker.previous=previous
        tracker.state=state;tracker.age=age;tracker.run=run;tracker.last_issue=issue
        tracker.eligible_hash_bytes+=bytes;tracker.advance_calls+=1;tracker.digest_snapshots+=1
        result
    finally
        tracker.busy=false
    end
end
function _snapshot(tracker)
    _scope(tracker)
    prefix=bytes2hex(SHA.digest!(copy(tracker.context)))
    fields=(CHECKPOINT_SCHEMA,tracker.source,tracker.sequence,length(tracker.rows),tracker.hashes,
        tracker.last_issue,tracker.count,prefix,tracker.previous,tracker.state,tracker.age,tracker.run)
    cp=CurrentStateCheckpoint(fields...,repeat("0",64))
    CurrentStateCheckpoint(fields...,D._hash(_checkpoint_data(cp)))
end
function _resume!(tracker,checkpoint)
    _need(checkpoint isa CurrentStateCheckpoint,"unsupported current-state checkpoint type")
    cp=checkpoint
    _need(cp.schema==CHECKPOINT_SCHEMA && cp.source_sha256==tracker.source && cp.wind_sequence_sha256==tracker.sequence &&
        cp.source_row_count==length(tracker.rows) && cp.implementation_hashes==tracker.hashes,"checkpoint source/configuration mismatch")
    _need(D._hex(cp.checkpoint_id) && D._hash(_checkpoint_data(cp))==cp.checkpoint_id,"checkpoint identity differs")
    _need(0<=cp.consumed_count<=length(tracker.rows),"checkpoint cursor outside source")
    cp.last_issue===nothing || _advance!(tracker,cp.last_issue)
    _need(isequal(_checkpoint_data(_snapshot(tracker)),_checkpoint_data(cp)),"checkpoint replay/trace/prefix differs")
    tracker
end

# Private diagnostics for the generated resource audit; no observations/models
# or guard bypass are exposed through the public API.
_statistics(tracker::_Tracker)=(;source_rows=length(tracker.rows),consumed_rows=tracker.count,
    admission_hash_bytes=tracker.admission_hash_bytes,eligible_hash_bytes=tracker.eligible_hash_bytes,
    advance_calls=tracker.advance_calls,digest_snapshots=tracker.digest_snapshots,source_checks=tracker.source_checks)

"""Call `f((advance,snapshot))` in a serial, source-verified archival tracker scope.

Rows must be unique/tag-ordered, ordinary one-based WindHour vectors with
nondecreasing receipts. Equal receipts enter together in tag order. On this
explicit admitted domain, `advance(issue)` exactly matches public current_state,
including its prefix/state IDs; issues are nondecreasing and equality is valid.
The private source copy isolates later caller-container mutation. `snapshot()`
returns an immutable in-memory checkpoint; resume reconstructs its prefix once.
Callback outputs are provisional until the final dependency check. No live
late-revision support, fitting boundary bypass, model artifact or serving action
is supplied. Both closures expire when f returns, including on exceptions.
"""
function with_current_state_tracker(f::Function,rows;source_sha256::AbstractString,checkpoint=nothing)
    hashes=source_identity()
    tracker=_admit(rows,source_sha256,hashes)
    try
        checkpoint===nothing || _resume!(tracker,checkpoint)
        advance=issue->_advance!(tracker,issue)
        snapshot=()->_snapshot(tracker)
        value=f((advance,snapshot))
        cp=_snapshot(tracker)
        (;value,checkpoint=cp,source_sha256=tracker.source,wind_sequence_sha256=tracker.sequence,implementation_hashes=hashes)
    finally
        tracker.active=false
        _need(source_identity()==hashes,"current-state bridge sources changed during scope")
        tracker.source_checks+=1
    end
end

end # module
