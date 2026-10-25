"""
Pure, bounded IAGA-2002 and OMNI HAPI CSV readers with immutable selected and
raw-row provenance. Provider support clocks and archive admission are separate.
"""
module ArchiveFormatParsers
using Dates, SHA

const MAX_BYTES=8*1024*1024
const MAX_RAW_ROWS=50_000
const RECORD_BYTES=48
const EPOCH=DateTime(1970)
const _NUMBER=r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
need(test,message)=test ? nothing : throw(ArgumentError(message))
struct _Checked end
const _CHECKED=_Checked()
struct _TextLines
    text::String
end
Base.IteratorSize(::Type{_TextLines})=Base.SizeUnknown()
Base.eltype(::Type{_TextLines})=SubString{String}
@inline function Base.iterate(lines::_TextLines,start::Int=1)
    raw=lines.text
    start>ncodeunits(raw)&&return nothing
    newline=findnext('\n',raw,start)
    if newline===nothing
        stop=lastindex(raw);following=ncodeunits(raw)+1
    else
        stop=prevind(raw,newline);following=nextind(raw,newline)
        stop>=start&&raw[stop]=='\r'&&(stop=prevind(raw,stop))
    end
    (SubString(raw,start,stop),following)
end
struct _Row
    time::Int64
    line::UInt32
    mask::UInt8
    reasons::UInt8
    values::NTuple{4,Float64}
end
struct ParsedBody
    kind::Symbol
    station::String
    product::String
    start::DateTime
    stop::DateTime
    body_sha256::String
    rows_sha256::String
    storage::String
    rows::Int
    raw_rows::Int
    duplicates::Int
    complete_vectors::Int
    missing_minutes::Int
    reason_counts::NTuple{4,NTuple{4,Int}}
    raw_storage::String
    raw_rows_sha256::String
    function ParsedBody(::_Checked,args...)
        new(args...)
    end
end
Base.length(body::ParsedBody)=body.rows
_ms(t::DateTime)=Base.Checked.checked_sub(Dates.value(t),Dates.value(EPOCH))
function _period(start,stop)
    need(start isa DateTime&&stop isa DateTime,"explicit DateTime period required")
    need(floor(start,Minute)==start&&floor(stop,Minute)==stop&&
        0<Dates.value(stop-start)<=32*86_400_000,"bounded positive whole-minute period required")
end
function _body(text::AbstractString)
    need(ncodeunits(text)<=MAX_BYTES,"raw body exceeds8MiB")
    raw=String(text)
    need(isvalid(raw)&&!occursin('\0',raw),"invalid UTF8/NUL body")
    raw
end
function _scalar(text)
    token=strip(text)
    need(0<ncodeunits(token)<=128,"numeric token bound differs")
    need(occursin(_NUMBER,token)||token in ("NaN","Inf","+Inf","-Inf"),"malformed numeric token")
    value=tryparse(Float64,token)
    need(value!==nothing,"unparseable numeric token")
    value::Float64
end
function _ground_scalar(text)
    v=_scalar(text)
    v==99999.0&&return (0.0,1)
    v==88888.0&&return (0.0,2)
    isfinite(v)&&abs(v)<90000 ? (v,0) : (0.0,3)
end
function _wind_scalar(text,column)
    v=_scalar(text)
    v==(column in (1,2) ? 9999.99 : column==3 ? 99999.9 : 999.99)&&return (0.0,1)
    good=isfinite(v)&&(column in (1,2) ? abs(v)<9000 : column==3 ? 100<v<90000 : 0<v<900)
    good ? (v,0) : (0.0,3)
end
function _date_time(date_text,time_text,doy_text=nothing)
    need(occursin(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",date_text)&&
        occursin(r"^[0-9]{2}:[0-9]{2}:00\.000$",time_text),"invalid exact-minute timestamp syntax")
    date=tryparse(Date,date_text,dateformat"yyyy-mm-dd")
    need(date!==nothing,"invalid calendar date")
    if doy_text!==nothing
        need(occursin(r"^[0-9]{3}$",doy_text)&&parse(Int,doy_text)==dayofyear(date),"calendar/DOY mismatch")
    end
    hour=parse(Int,time_text[1:2]);minute=parse(Int,time_text[4:5])
    need((0<=hour<24&&0<=minute<60)||(hour==24&&minute==0),"invalid minute clock")
    DateTime(date)+Hour(hour)+Minute(minute)
end
function _hapi_time(token)
    need(occursin(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:00\.000Z$",token),"invalid HAPI exact-minute UTC tag")
    _date_time(token[1:10],token[12:23])
end
function _record(time,line,values,reasons)
    mask=UInt8(0);packed=UInt8(0)
    for j in eachindex(values)
        reasons[j]==0&&(mask|=UInt8(1<<(j-1)))
        packed|=UInt8(reasons[j]<<(2*(j-1)))
    end
    vals=ntuple(j->j<=length(values) ? Float64(values[j]) : 0.0,4)
    _Row(_ms(time),UInt32(line),mask,packed,vals)
end
function _admit!(rows,locations,witnesses,row,fullmask)
    if haskey(witnesses,row.time)
        mask,values=witnesses[row.time]
        for j in 1:4
            bit=UInt8(1<<(j-1))
            if (mask&row.mask&bit)!=0
                need(values[j]==row.values[j],"contradictory valid duplicate component")
            end
        end
        values=ntuple(j->(mask&UInt8(1<<(j-1)))!=0 ? values[j] : row.values[j],4)
        witnesses[row.time]=(mask|row.mask,values)
    else
        witnesses[row.time]=(row.mask,row.values)
    end
    # The union above checks all raw witnesses; it never supplies a projection.
    # Retain one actual original row so complementary partial rows stay partial.
    if !haskey(locations,row.time)
        push!(rows,row);locations[row.time]=length(rows);return
    end
    index=locations[row.time];prior=rows[index]
    if row.mask==fullmask&&prior.mask!=fullmask;rows[index]=row;end
    nothing
end
# Streaming SHA uses its fixed byte buffer, avoiding CodeUnits alias checks.
_sha_string(raw::String)=bytes2hex(sha256(IOBuffer(raw)))
function _write_row!(io::IO,row::_Row)
    write(io,htol(row.time),htol(row.line),row.mask,row.reasons,htol(UInt16(0)))
    for value in row.values;write(io,htol(reinterpret(UInt64,value)));end
    nothing
end
function _finish(kind,station,product,start,stop,raw,rows,rawcount,counts,ledger)
    sort!(rows;by=r->r.time)
    io=IOBuffer()
    for r in rows
        _write_row!(io,r)
    end
    storage=String(take!(io));need(ncodeunits(storage)==RECORD_BYTES*length(rows),"internal packed row width differs")
    raw_storage=String(take!(ledger))
    need(ncodeunits(raw_storage)==RECORD_BYTES*rawcount,"internal raw-row ledger width differs")
    expected=div(Dates.value(stop-start),60_000)+(kind==:ground ? 1 : 0)
    fullmask=kind==:ground ? UInt8(3) : UInt8(15)
    ParsedBody(_CHECKED,kind,String(station),String(product),start,stop,
        _sha_string(raw),_sha_string(storage),storage,
        length(rows),rawcount,rawcount-length(rows),count(r->r.mask==fullmask,rows),expected-length(rows),
        ntuple(j->ntuple(k->counts[j,k],4),4),raw_storage,_sha_string(raw_storage))
end
function _count!(counts,reasons)
    for j in eachindex(reasons);counts[j,reasons[j]+1]+=1;end
end
function parse_ground(text::AbstractString;station::AbstractString,product::AbstractString,start::DateTime,stop::DateTime)
    _period(start,stop);raw=_body(text)
    need(occursin(r"^[A-Z]{3}$",station)&&product=="quasi-definitive","unsupported ground station/product declaration")
    headers=Dict{String,String}();active=false;last=nothing;first_identity=nothing
    rows=_Row[];locations=Dict{Int64,Int}();counts=zeros(Int,4,4);rawcount=0
    ledger=IOBuffer()
    witnesses=Dict{Int64,Tuple{UInt8,NTuple{4,Float64}}}()
    for (lineno,line) in enumerate(_TextLines(raw))
        need(ncodeunits(line)<=4096,"source line exceeds4096bytes")
        isempty(line)&&continue
        s=strip(line);isempty(s)&&continue;startswith(s,"#")&&continue
        if startswith(s,"DATE")
            need(endswith(s,"|"),"IAGA data header lacks terminator")
            names=split(strip(chop(s;tail=1)))
            need(length(names)==7&&names[1:3]==["DATE","TIME","DOY"]&&names[4]==station*"X"&&names[5]==station*"Y","unsupported IAGA component columns")
            need(all(k->haskey(headers,k),("Format","Source of Data","IAGA CODE","Reported","Data Interval Type","Data Type")),"missing IAGA identity header")
            need(headers["Format"]=="IAGA-2002"&&headers["IAGA CODE"]==station&&headers["Data Type"]==product&&
                headers["Reported"] in ("XYNULNUL","XYZF")&&occursin(r"\b1-minute\b",headers["Data Interval Type"]),"IAGA identity/product/interval mismatch")
            suffix=headers["Reported"]=="XYZF" ? [station*"Z",station*"F"] : [station*"NUL",station*"NUL"]
            need(names[6:7]==suffix,"IAGA unreported/component mapping differs")
            identity=Tuple(headers[k] for k in ("Format","Source of Data","IAGA CODE","Reported","Data Interval Type","Data Type"))
            first_identity===nothing ? (first_identity=identity) : need(identity==first_identity,"mixed IAGA segment identity")
            active=true;last=nothing;continue
        elseif '0'<=first(s)<='9'
            need(active,"IAGA data precedes identity header")
            fields=split(s);need(length(fields)==7,"IAGA data field count differs")
            tag=_date_time(fields[1],fields[2],fields[3])
            need(start<=tag<=stop,"IAGA tag outside declared interval")
            need(last===nothing||tag>=last,"backward IAGA tag within segment");last=tag
            rawcount+=1;need(rawcount<=MAX_RAW_ROWS,"raw record count exceeds50000")
            x,rx=_ground_scalar(fields[4]);y,ry=_ground_scalar(fields[5])
            _scalar(fields[6]);_scalar(fields[7])
            row=_record(tag,lineno,(x,y),(rx,ry))
            _write_row!(ledger,row)
            _count!(counts,(rx,ry));_admit!(rows,locations,witnesses,row,UInt8(3));continue
        end
        need(endswith(s,"|"),"unknown IAGA record")
        content=strip(chop(s;tail=1))
        m=match(r"^(Format|Source of Data|IAGA CODE|Reported|Data Interval Type|Data Type)\s+(.+)$",content)
        if m!==nothing
            key=String(m[1]);value=String(strip(m[2]))
            if key=="Format";empty!(headers);active=false;end
            need(!haskey(headers,key)||headers[key]==value,"contradictory IAGA header")
            headers[key]=value
        else
            need(occursin(r"^(Station Name|Geodetic Latitude|Geodetic Longitude|Elevation|Sensor Orientation|Digital Sampling|Publication date)\s+.+$",content),"unsupported IAGA header record")
        end
    end
    need(active&&first_identity!==nothing,"no complete IAGA identity/data header")
    _finish(:ground,station,product,start,stop,raw,rows,rawcount,counts,ledger)
end
function _csv_token(text)
    t=strip(text)
    if startswith(t,'"')||endswith(t,'"')
        need(ncodeunits(t)>=2&&startswith(t,'"')&&endswith(t,'"'),"malformed CSV quote")
        t=chop(t;head=1,tail=1)
    end
    need(!occursin('"',t),"embedded CSV quote")
    t
end
function parse_wind(text::AbstractString;station::AbstractString,product::AbstractString,start::DateTime,stop::DateTime)
    _period(start,stop);raw=_body(text)
    need(occursin(r"^[A-Z]{3}$",station)&&product=="OMNI_HRO_1MIN","unsupported wind declaration")
    rows=_Row[];locations=Dict{Int64,Int}();counts=zeros(Int,4,4);rawcount=0;last=nothing
    ledger=IOBuffer()
    witnesses=Dict{Int64,Tuple{UInt8,NTuple{4,Float64}}}()
    for (lineno,line) in enumerate(_TextLines(raw))
        need(ncodeunits(line)<=4096,"source line exceeds4096bytes")
        isempty(line)&&continue
        isempty(strip(line))&&continue
        fields=_csv_token.(split(line,',';keepempty=true));need(length(fields)==5,"HAPI CSV field count differs")
        tag=_hapi_time(fields[1]);need(start<=tag<stop,"HAPI tag outside half-open interval")
        need(last===nothing||tag>=last,"backward HAPI tag");last=tag
        rawcount+=1;need(rawcount<=MAX_RAW_ROWS,"raw record count exceeds50000")
        rawvalues=ntuple(j->_wind_scalar(fields[j+1],j),4)
        order=(3,2,1,4);values=ntuple(j->rawvalues[order[j]][1],4);reasons=ntuple(j->rawvalues[order[j]][2],4)
        row=_record(tag,lineno,values,reasons)
        _write_row!(ledger,row)
        _count!(counts,reasons);_admit!(rows,locations,witnesses,row,UInt8(15))
    end
    _finish(:wind,station,product,start,stop,raw,rows,rawcount,counts,ledger)
end
function _word(storage,at,n)
    value=UInt64(0)
    for j in 0:n-1;value|=UInt64(codeunit(storage,at+j))<<(8j);end
    value
end
function _packed_row(storage::String,index::Integer,count::Int)
    need(!(index isa Bool)&&1<=index<=count,"row index outside immutable source")
    at=(Int(index)-1)*RECORD_BYTES+1
    stamp=reinterpret(Int64,_word(storage,at,8));line=UInt32(_word(storage,at+8,4))
    mask=codeunit(storage,at+12);reasons=codeunit(storage,at+13)
    values=ntuple(j->reinterpret(Float64,_word(storage,at+16+8(j-1),8)),4)
    (;tag=EPOCH+Millisecond(stamp),source_line=line,valid_mask=mask,reason_bits=reasons,values)
end
row(body::ParsedBody,index::Integer)=_packed_row(body.storage,index,length(body))
raw_row(body::ParsedBody,index::Integer)=_packed_row(body.raw_storage,index,body.raw_rows)
function complete_values(body::ParsedBody,index::Integer)
    r=row(body,index)
    if body.kind==:ground;return r.valid_mask==3 ? (r.values[1],r.values[2]) : nothing;end
    r.valid_mask==15 ? r.values : nothing
end
end
