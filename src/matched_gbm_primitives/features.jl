"An explicitly assumed receipt grid. Fields contain a private censored copy."
struct HourlyInputs342
    time::Vector{DateTime}
    values::Matrix{Float64}
    received::Matrix{DateTime}
    delay_hours::Int
    source_sha256::String
    input_id::String
end
function _input_id(times,values,received,delay,source)
    io=IOBuffer();write(io,"matched342-input-v1|$source|$delay|$(length(times))\n")
    for i in eachindex(times)
        write(io,htol(Dates.value(times[i])))
        for j in 1:17
            write(io,htol(reinterpret(UInt64,values[j,i])));write(io,htol(Dates.value(received[j,i])))
        end
    end
    bytes2hex(sha256(take!(io)))
end
function HourlyInputs342(times,values;delay_hours=0,received=nothing,source_sha256)
    delay=_int(delay_hours,0,1);n=length(times)
    _need(1<=n<=MAX_ROWS&&size(values)==(17,n)&&_hex(source_sha256),"bounded17-channel source required")
    Base.require_one_based_indexing(times,values)
    _need(all(t->t isa DateTime,times),"DateTime source tags required")
    tags=collect(DateTime,times)
    _need(all(t->floor(t,Hour)==t,tags)&&all(diff(tags).==Hour(1)),"contiguous exact-hour source required")
    _need(all(x->x isa Real&&!(x isa Bool)&&(isfinite(x)||isnan(x)),values),"raw Inf/Bool/source type rejected")
    matrix=Matrix{Float64}(values);_need(all(x->isfinite(x)||isnan(x),matrix),"raw conversion overflow")
    matrix=Features.censor_fills(matrix)
    receipts=if received===nothing
        [_shift(tags[i],1+(j==15 ? delay : 0)) for j in 1:17,i in 1:n]
    else
        _need(size(received)==(17,n)&&all(x->x isa DateTime,received),"17-channel receipt shape/types differ")
        Base.require_one_based_indexing(received);Matrix{DateTime}(received)
    end
    for i in 1:n,j in 1:17
        _need(receipts[j,i]>=_shift(tags[i],1+(j==15 ? delay : 0)),"receipt precedes declared completion/delay")
    end
    wind_receipts=[max(receipts[6,i],receipts[8,i]) for i in 1:n]
    _need(issorted(wind_receipts),"wind requires admitted nondecreasing archival receipts")
    id=_input_id(tags,matrix,receipts,delay,String(source_sha256))
    HourlyInputs342(tags,matrix,receipts,delay,String(source_sha256),id)
end
function _snapshot(input::HourlyInputs342)
    snapshot=HourlyInputs342(input.time,input.values;delay_hours=input.delay_hours,
        received=input.received,source_sha256=input.source_sha256)
    _need(snapshot.input_id==input.input_id,"input data changed")
    snapshot
end
function _index(input,t::DateTime)
    delta=try Base.Checked.checked_sub(Dates.value(t),Dates.value(first(input.time))) catch;throw(ArgumentError("source index clock overflow"));end
    mod(delta,HOUR_MS)==0 || return 0
    k=fld(delta,HOUR_MS)+1
    1<=k<=length(input.time) ? Int(k) : 0
end
function _cell(input,channel,t,issue)
    k=_index(input,t)
    k>0&&_shift(t,1)<=issue&&input.received[channel,k]<=issue ? input.values[channel,k] : NaN
end
function _wind(input)
    rows=Duration.WindHour[];sizehint!(rows,length(input.time))
    for i in eachindex(input.time)
        bz=input.values[6,i];v=input.values[8,i]
        q=isfinite(bz)&&isfinite(v)&&v>=0 ? max(v*max(-bz,0.)/1000-.49,0.) : NaN
        _need(isfinite(q)||isnan(q),"derived injection overflow")
        push!(rows,Duration.WindHour(input.time[i],_shift(input.time[i],1),max(input.received[6,i],input.received[8,i]),bz,q))
    end
    rows
end
mutable struct _Trace
    previous::Union{Nothing,DateTime}
    state::Symbol
    age::Int
    run::Int
    censored::Bool
    cursor::Int
end
_Trace()=_Trace(nothing,:missing,0,0,false,0)
function _trace_step!(trace,row)
    if !Duration._known(row)
        trace.previous=nothing;trace.state=:missing;trace.age=0;trace.run=0;trace.censored=false;return
    end
    state=Duration._state(row);contiguous=trace.previous!==nothing&&row.tag==_shift(trace.previous,1)
    if !contiguous
        trace.run=1;trace.age=0;trace.censored=true
    elseif state!=trace.state
        trace.run=1;trace.age=1;trace.censored=false
    else
        trace.run+=1;trace.age=trace.age>0 ? min(trace.age+1,49) : trace.run>=49 ? 49 : 0
    end
    trace.previous=row.tag;trace.state=state
end
function _context!(trace,rows,issue,delay)
    while trace.cursor<length(rows)
        row=rows[trace.cursor+1]
        row.completed<=issue&&row.received<=issue || break
        _trace_step!(trace,row);trace.cursor+=1
    end
    latest=trace.cursor==0 ? nothing : rows[trace.cursor]
    unavailable=latest===nothing||latest.tag!=_shift(issue,-1)
    missing=latest!==nothing&&!Duration._known(latest)
    elapsed=latest===nothing ? Int64(0) : try Base.Checked.checked_sub(Dates.value(issue),Dates.value(latest.completed)) catch;throw(ArgumentError("wind age clock overflow"));end
    age=Float64(elapsed)/HOUR_MS
    known=trace.state in (:N,:S)
    Float64[delay,age,trace.state==:N,trace.state==:S,known&&trace.age>0,trace.age,
        min(trace.run,49),trace.censored,trace.age==49,known&&trace.age==0,unavailable,missing]
end
"Return12 finite receipt/trace fields from ordered archival wind rows at an exact-hour issue."
function trace_context(rows,issue::DateTime;delay_hours=0)
    delay=_int(delay_hours,0,1);_need(floor(issue,Hour)==issue,"exact-hour issue required")
    Duration._ordered(rows);_need(issorted(getproperty.(rows,:received)),"admitted receipt order required")
    _context!(_Trace(),rows,issue,delay)
end
"One generated feature row, its exact available anchor and admitted raw-input identity."
struct FeatureRow342
    issue::DateTime
    delay_hours::Int
    values::Vector{Float64}
    anchor::Union{Nothing,Float64}
    input_id::String
end
function _row(input,issue,context)
    # Virtual Dst frame covers exactly the24lag ladder; raw access extends
    # only to issue-26 under L1. Driver tags are never shifted or forward filled.
    tags=[_shift(issue,j) for j in -24:0]
    column(channel)=[_cell(input,channel,t,issue) for t in tags]
    frame=(time_utc=tags,By=column(5),Bz=column(6),V=column(8),n=column(9),Pdyn=column(12),
        Dst=[_cell(input,15,_shift(t,-1-input.delay_hours),issue) for t in tags])
    block,_=Parent.direct_gbm_features(frame,DateTime[issue]);base=collect(@view block[1,:])
    for channel in 1:17,lag in RAW_LAGS
        push!(base,_cell(input,channel,_shift(issue,-1-lag),issue))
    end
    _need(length(base)==BASE_WIDTH&&all(x->isfinite(x)||isnan(x),base),"nonfinite derived design")
    values=vcat(base,Float64.(isnan.(base)),context)
    anchor=_cell(input,15,_shift(issue,-1-input.delay_hours),issue)
    FeatureRow342(issue,input.delay_hours,values,isfinite(anchor) ? anchor : nothing,input.input_id)
end
"Deliver ordered FeatureRow342 chunks of at most chunk_rows (1:4096), checking source/input identity before and after."
function foreach_feature(sink::Function,input::HourlyInputs342,issues;chunk_rows=128)
    source_identity();chunk=_int(chunk_rows,1,4096)
    _need(1<=length(issues)<=MAX_ROWS&&all(t->t isa DateTime&&floor(t,Hour)==t,issues)&&issorted(issues),"ordered bounded exact-hour issues required")
    times=collect(DateTime,issues);snapshot=_snapshot(input);rows=_wind(snapshot);trace=_Trace();buffer=FeatureRow342[]
    for issue in times
        push!(buffer,_row(snapshot,issue,_context!(trace,rows,issue,input.delay_hours)))
        if length(buffer)==chunk;sink(buffer);buffer=FeatureRow342[];end
    end
    isempty(buffer)||sink(buffer)
    _need(_input_id(input.time,input.values,input.received,input.delay_hours,input.source_sha256)==input.input_id,"input changed during feature scope")
    source_identity();nothing
end
"Collect receipt-aware rows, the raw342-column matrix (NaN denotes missing), and ordered names."
function build_features(input::HourlyInputs342,issues)
    rows=FeatureRow342[]
    foreach_feature(input,issues) do chunk;append!(rows,chunk);end
    matrix=Matrix{Float64}(undef,length(rows),WIDTH)
    for i in eachindex(rows),j in 1:WIDTH;matrix[i,j]=rows[i].values[j];end
    (;rows,matrix,names=collect(FEATURE_NAMES))
end
"Return separate h4/h6 train/evaluation row indices with exact anchors and168-hour completion-and-receipt embargoes."
function cohorts(input::HourlyInputs342,horizon;training_boundary::DateTime,validation_start::DateTime,validation_end::DateTime)
    h=_int(horizon,4,6);_need(h in (4,6)&&training_boundary<=validation_start<validation_end,"cohort clock/split differs")
    source_identity();raw=_snapshot(input);train=Int[];validation=Int[];issues=[_shift(t,1) for t in raw.time]
    traincut=_shift(training_boundary,-168);valcut=_shift(validation_end,-168)
    for (i,issue) in enumerate(issues)
        ai=_index(raw,_shift(issue,-1-raw.delay_hours));ti=_index(raw,_shift(issue,h-1))
        ai>0&&ti>0&&isfinite(raw.values[15,ai])&&isfinite(raw.values[15,ti])&&raw.received[15,ai]<=issue || continue
        complete=_shift(issue,h);received=raw.received[15,ti]
        issue<training_boundary&&complete<=traincut&&received<=traincut&&push!(train,i)
        validation_start<=issue<validation_end&&complete<=valcut&&received<=valcut&&push!(validation,i)
    end
    (;horizon=h,delay_hours=input.delay_hours,issues,train,validation)
end
