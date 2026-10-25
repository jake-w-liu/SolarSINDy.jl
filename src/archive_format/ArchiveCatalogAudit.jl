"""
    ArchiveCatalogAudit

Independent batch reconstruction of the per-stream catalog wire contract from
ArchiveFormatAudit bodies. A separate identity encoder, grouped raw-witness
comparison and byte serializer reconstruct every selected original record.
This namespace does not call the primary catalog and grants no actual-source,
support-clock, archive-fit or forecasting qualification.
"""
module ArchiveCatalogAudit
using Dates,SHA
using ..ArchiveFormatAudit: AuditBody,raw_record

require(condition,message)=condition ? nothing : throw(ArgumentError(message))
const ORIGIN=DateTime(1970)
const BODY_LIMIT=32
const ROW_LIMIT=1_600_000
struct CatalogAudit
    kind::Symbol
    station::String
    product::String
    bodies::Tuple{Vararg{AuditBody}}
    body_ids::Tuple{Vararg{String}}
    selected_storage::String
    selected_sha256::String
    raw_catalog_sha256::String
    raw_rows::Int
    rows::Int
    duplicates::Int
    complete_vectors::Int
    declared_minutes::Int
    missing_minutes::Int
    uncovered_minutes::Int
end
Base.length(c::CatalogAudit)=c.rows
struct _Entry
    stamp::Int64
    line::UInt32
    mask::UInt8
    reasons::UInt8
    values::NTuple{4,Float64}
    body::Int32
end
function _stamp(t::DateTime)
    difference=big(Dates.value(t))-big(Dates.value(ORIGIN))
    require(typemin(Int64)<=difference<=typemax(Int64),"audit timestamp outside binary range")
    Int64(difference)
end
function _bytes(number::UInt64,width)
    UInt8[(number>>(8*i))&UInt64(255) for i in 0:width-1]
end
_signed_bytes(n::Int64)=_bytes(reinterpret(UInt64,n),8)
function audit_identity(body::AuditBody)
    require(length(body.body_digest)==64&&all(c->c in "0123456789abcdef",body.body_digest),"audit body digest syntax")
    fields=[collect(codeunits(String(body.kind))),collect(codeunits(body.station)),collect(codeunits(body.product)),_signed_bytes(_stamp(body.start)),_signed_bytes(_stamp(body.stop)),hex2bytes(body.body_digest)]
    encoded=collect(codeunits("DBDTCATBODY1\0"))
    for field in fields;append!(encoded,_bytes(UInt64(length(field)),8));append!(encoded,field);end
    bytes2hex(sha256(encoded))
end
function _validate(body::AuditBody)
    require(body.kind in (:ground,:wind)&&length(body.station)==3&&all(c->'A'<=c<='Z',body.station),"audit catalog kind/station")
    require(body.product==(body.kind===:ground ? "quasi-definitive" : "OMNI_HRO_1MIN"),"audit catalog product")
    span=big(Dates.value(body.stop))-big(Dates.value(body.start))
    require(floor(body.start,Minute)==body.start&&floor(body.stop,Minute)==body.stop&&0<span<=32*86_400_000,"audit catalog period")
    require(0<=body.unique_count<=body.record_count<=50_000,"audit catalog count bound")
    require(ncodeunits(body.raw_records)==48*body.record_count&&ncodeunits(body.selected_records)==48*body.unique_count,"audit packed width")
    require(bytes2hex(sha256(IOBuffer(body.raw_records)))==body.raw_digest,"audit raw digest differs")
    require(bytes2hex(sha256(IOBuffer(body.selected_records)))==body.selected_digest,"audit selected digest differs")
end
function _entry(body,number,bi,prior)
    row=raw_record(body,number)
    require(row.source_line>prior,"audit raw source line order")
    require(floor(row.tag,Minute)==row.tag&&body.start<=row.tag&&(body.kind===:ground ? row.tag<=body.stop : row.tag<body.stop),"audit raw row outside period")
    require(row.valid_mask<=(body.kind===:ground ? 3 : 15),"audit raw validity mask")
    for channel in 1:4
        bit=UInt8(1<<(channel-1));valid=(row.valid_mask&bit)!=0;reason=(row.reason_bits>>(2*(channel-1)))&UInt8(3);value=row.values[channel]
        if body.kind===:ground&&channel>2
            require(!valid&&reason==0&&reinterpret(UInt64,value)==0,"audit unused channel")
        elseif valid
            require(reason==0&&isfinite(value),"audit valid component")
        else
            require(reason!=0&&reinterpret(UInt64,value)==0,"audit unavailable component")
        end
    end
    _Entry(_stamp(row.tag),row.source_line,row.valid_mask,row.reason_bits,row.values,Int32(bi))
end
function _coverage(bodies)
    events=Dict{Int64,Int}()
    for body in bodies
        first=_stamp(body.start)÷60_000
        beyond=_stamp(body.stop)÷60_000+(body.kind===:ground ? 1 : 0)
        events[first]=get(events,first,0)+1;events[beyond]=get(events,beyond,0)-1
    end
    tags=sort!(collect(keys(events)));active=0;covered=0
    for i in 1:length(tags)-1
        active+=events[tags[i]]
        active>0&&(covered+=tags[i+1]-tags[i])
    end
    covered,last(tags)-first(tags)-covered
end
function _append_selected!(bytes,entry,id)
    append!(bytes,_signed_bytes(entry.stamp));append!(bytes,_bytes(UInt64(entry.line),4))
    push!(bytes,entry.mask,entry.reasons,0x00,0x00)
    for value in entry.values;append!(bytes,_bytes(reinterpret(UInt64,value),8));end
    append!(bytes,hex2bytes(id))
end
function audit_catalog(inputs)
    require(inputs isa AbstractVector||inputs isa Tuple,"audit explicit body collection")
    require(1<=length(inputs)<=BODY_LIMIT&&all(x->x isa AuditBody,inputs),"audit catalog body bound/type")
    for body in inputs;_validate(body);end
    key=(first(inputs).kind,first(inputs).station,first(inputs).product)
    require(all(x->(x.kind,x.station,x.product)==key,inputs),"audit mixed stream")
    pairs=sort([(audit_identity(body),body) for body in inputs];by=first)
    ids=Tuple(pair[1] for pair in pairs);bodies=Tuple(pair[2] for pair in pairs)
    require(length(Set(ids))==length(ids),"audit duplicate body identity")
    total=sum(x.record_count for x in bodies);require(total<=ROW_LIMIT,"audit catalog raw row bound")
    entries=_Entry[];sizehint!(entries,total)
    rawencoding=collect(codeunits("DBDTCATRAW1\0"));sizehint!(rawencoding,length(rawencoding)+40*length(bodies)+48*total)
    for (bi,body) in enumerate(bodies)
        append!(rawencoding,hex2bytes(ids[bi]));append!(rawencoding,_bytes(UInt64(ncodeunits(body.raw_records)),8));append!(rawencoding,codeunits(body.raw_records))
        prior=UInt32(0)
        for number in 1:body.record_count
            entry=_entry(body,number,bi,prior);prior=entry.line;push!(entries,entry)
        end
    end
    sort!(entries;by=e->(e.stamp,e.body,e.line))
    selected=UInt8[];full=key[1]===:ground ? UInt8(3) : UInt8(15);i=1;unique=0;complete=0
    while i<=length(entries)
        firstentry=entries[i];chosen=firstentry;found_complete=firstentry.mask==full
        observed=falses(4);values=zeros(4);j=i
        while j<=length(entries)&&entries[j].stamp==firstentry.stamp
            entry=entries[j]
            for channel in 1:4
                entry.mask&UInt8(1<<(channel-1))==0&&continue
                require(!observed[channel]||values[channel]==entry.values[channel],"audit contradictory cross-body valid component")
                values[channel]=entry.values[channel];observed[channel]=true
            end
            if !found_complete&&entry.mask==full;chosen=entry;found_complete=true;end
            j+=1
        end
        _append_selected!(selected,chosen,ids[chosen.body]);complete+=found_complete;unique+=1;i=j
    end
    require(length(selected)==80*unique,"audit selected record width")
    declared,uncovered=_coverage(bodies);require(unique<=declared,"audit coverage denominator")
    selectedhash=bytes2hex(sha256(selected));rawhash=bytes2hex(sha256(rawencoding))
    CatalogAudit(key...,bodies,ids,String(selected),selectedhash,rawhash,total,unique,total-unique,complete,declared,declared-unique,uncovered)
end
function _decode_unsigned(text,start,width)
    bits=UInt64(0)
    for i in reverse(start:start+width-1);bits=(bits<<8)|UInt64(codeunit(text,i));end
    bits
end
function audit_witness(c::CatalogAudit,index::Integer)
    require(!(index isa Bool)&&1<=index<=length(c),"audit selected index")
    offset=(Int(index)-1)*80+1;text=c.selected_storage
    stamp=reinterpret(Int64,_decode_unsigned(text,offset,8))
    id=bytes2hex(UInt8[codeunit(text,i) for i in offset+48:offset+79])
    (;tag=ORIGIN+Millisecond(stamp),source_line=UInt32(_decode_unsigned(text,offset+8,4)),valid_mask=codeunit(text,offset+12),reason_bits=codeunit(text,offset+13),values=ntuple(j->reinterpret(Float64,_decode_unsigned(text,offset+16+8*(j-1),8)),4),body_id=id)
end
function audit_locate(c::CatalogAudit,tag::DateTime)
    require(floor(tag,Minute)==tag,"audit lookup minute precision")
    target=_stamp(tag);a=1;b=length(c)
    while a<=b
        center=(a+b)÷2;row=audit_witness(c,center)
        row.tag==tag&&return row
        if _stamp(row.tag)<target;a=center+1;else;b=center-1;end
    end
    nothing
end
end
