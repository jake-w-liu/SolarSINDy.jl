"""
    ArchiveSourceCatalog

Pure per-stream catalogs over immutable parsed body ledgers. Canonical body
identities select original witnesses independently of input enumeration order.
Every raw row participates in cross-body consistency checks; complementary
partial measurements are never stitched. One catalog admits at most32 bodies
and1600000 raw rows. Selected records retain the exact48-byte original row plus
a32-byte body identity. Declared missing minutes and uncovered gaps are distinct.
Wind station identifiers are caller stream labels, not physical CSV stations.
This namespace grants no file access, provider-clock or actual-source authority.
"""
module ArchiveSourceCatalog
using Dates, SHA
using ..ArchiveFormatParsers: ParsedBody, raw_row

const MAX_BODIES=32
const MAX_RAW_ROWS=1_600_000
const WITNESS_BYTES=80
const EPOCH=DateTime(1970)
need(ok,message)=ok ? nothing : throw(ArgumentError(message))
struct _Checked end
struct SourceCatalog
    kind::Symbol
    station::String
    product::String
    bodies::Tuple{Vararg{ParsedBody}}
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
    function SourceCatalog(::_Checked,args...);new(args...);end
end
Base.length(c::SourceCatalog)=c.rows
_hash(s::String)=bytes2hex(sha256(IOBuffer(s)))
function _milliseconds(t::DateTime)
    n=Int128(Dates.value(t))-Int128(Dates.value(EPOCH))
    need(typemin(Int64)<=n<=typemax(Int64),"catalog timestamp overflow")
    Int64(n)
end
function _frame!(io,bytes)
    write(io,htol(UInt64(length(bytes))));write(io,bytes)
end
function body_identity(body::ParsedBody)
    need(occursin(r"^[0-9a-f]{64}$",body.body_sha256),"body digest syntax differs")
    io=IOBuffer();write(io,"DBDTCATBODY1\0")
    for s in (String(body.kind),body.station,body.product);_frame!(io,codeunits(s));end
    for t in (body.start,body.stop)
        piece=IOBuffer();write(piece,htol(_milliseconds(t)));_frame!(io,take!(piece))
    end
    _frame!(io,hex2bytes(body.body_sha256));bytes2hex(sha256(IOBuffer(take!(io))))
end
function _coverage(bodies)
    intervals=sort([(_milliseconds(b.start)÷60_000,_milliseconds(b.stop)÷60_000-(b.kind==:wind ? 1 : 0)) for b in bodies])
    lo,hi=first(intervals);firstlo=lo;declared=0
    for (a,b) in Iterators.drop(intervals,1)
        if a<=hi+1;hi=max(hi,b)
        else;declared+=hi-lo+1;lo=a;hi=b;end
    end
    declared+=hi-lo+1
    declared,hi-firstlo+1-declared
end
function _check_body(body)
    need(body.kind in (:ground,:wind)&&occursin(r"^[A-Z]{3}$",body.station),"catalog stream identity differs")
    need(body.product==(body.kind==:ground ? "quasi-definitive" : "OMNI_HRO_1MIN"),"catalog product differs")
    span=Int128(Dates.value(body.stop))-Int128(Dates.value(body.start))
    need(floor(body.start,Minute)==body.start&&floor(body.stop,Minute)==body.stop&&0<span<=32*86_400_000,"catalog body interval differs")
    need(0<=body.rows<=body.raw_rows<=50_000,"catalog per-body count differs")
    need(ncodeunits(body.raw_storage)==48*body.raw_rows&&ncodeunits(body.storage)==48*body.rows,"catalog packed body width differs")
    need(_hash(body.raw_storage)==body.raw_rows_sha256&&_hash(body.storage)==body.rows_sha256,"catalog packed body digest differs")
    nothing
end
function _check_row(row,body,priorline)
    need(row.source_line>priorline,"raw source line order differs")
    need(floor(row.tag,Minute)==row.tag&&body.start<=row.tag&&(body.kind==:ground ? row.tag<=body.stop : row.tag<body.stop),"catalog raw tag outside declaration")
    full=body.kind==:ground ? UInt8(3) : UInt8(15)
    need(row.valid_mask&~full==0,"catalog mask outside stream")
    for j in 1:4
        valid=row.valid_mask&UInt8(1<<(j-1))!=0;reason=(row.reason_bits>>(2*(j-1)))&UInt8(3);value=row.values[j]
        if body.kind==:ground&&j>2
            need(!valid&&reason==0&&reinterpret(UInt64,value)==0,"ground unused channel differs")
        elseif valid
            need(reason==0&&isfinite(value),"valid row reason or value differs")
        else
            need(reason in (1,2,3)&&reinterpret(UInt64,value)==0,"unavailable row reason or payload differs")
        end
    end
    nothing
end
function _write_witness!(io,row,id)
    write(io,htol(_milliseconds(row.tag)),htol(row.source_line),row.valid_mask,row.reason_bits,htol(UInt16(0)))
    for value in row.values;write(io,htol(reinterpret(UInt64,value)));end
    write(io,hex2bytes(id))
end
function build_catalog(input)
    need(input isa AbstractVector||input isa Tuple,"explicit body collection required")
    need(1<=length(input)<=MAX_BODIES&&all(b->b isa ParsedBody,input),"catalog body count or type differs")
    for b in input;_check_body(b);end
    firstbody=first(input);key=(firstbody.kind,firstbody.station,firstbody.product)
    need(all(b->(b.kind,b.station,b.product)==key,input),"mixed catalog stream declarations")
    ids=body_identity.(input);need(length(unique(ids))==length(ids),"duplicate catalog body identity")
    order=sortperm(collect(ids));bodies=Tuple(input[i] for i in order);bodyids=Tuple(ids[i] for i in order)
    rawcount=sum(b.raw_rows for b in bodies);need(rawcount<=MAX_RAW_ROWS,"catalog raw-row bound exceeded")
    witnesses=Dict{Int64,Tuple{UInt8,NTuple{4,Float64}}}()
    chosen=Dict{Int64,Tuple{Int,Int,UInt8}}();full=key[1]==:ground ? UInt8(3) : UInt8(15)
    rawframe=IOBuffer();write(rawframe,"DBDTCATRAW1\0")
    for (bi,body) in enumerate(bodies)
        write(rawframe,hex2bytes(bodyids[bi]),htol(UInt64(ncodeunits(body.raw_storage))),codeunits(body.raw_storage))
        priorline=UInt32(0)
        for ri in 1:body.raw_rows
            row=raw_row(body,ri);_check_row(row,body,priorline);priorline=row.source_line;tag=_milliseconds(row.tag)
            if haskey(witnesses,tag)
                mask,values=witnesses[tag]
                for j in 1:4
                    bit=UInt8(1<<(j-1))
                    need(mask&row.valid_mask&bit==0||values[j]==row.values[j],"contradictory valid cross-body component")
                end
                unionvalues=ntuple(j->mask&UInt8(1<<(j-1))!=0 ? values[j] : row.values[j],4)
                witnesses[tag]=(mask|row.valid_mask,unionvalues)
            else;witnesses[tag]=(row.valid_mask,row.values);end
            if !haskey(chosen,tag)||(row.valid_mask==full&&chosen[tag][3]!=full);chosen[tag]=(bi,ri,row.valid_mask);end
        end
    end
    packed=IOBuffer();complete=0
    for tag in sort!(collect(keys(chosen)))
        bi,ri,mask=chosen[tag];_write_witness!(packed,raw_row(bodies[bi],ri),bodyids[bi]);complete+=mask==full
    end
    storage=String(take!(packed));need(ncodeunits(storage)==WITNESS_BYTES*length(chosen),"catalog selected width differs")
    declared,uncovered=_coverage(bodies);need(length(chosen)<=declared,"catalog selection exceeds declared coverage")
    SourceCatalog(_Checked(),key...,bodies,bodyids,storage,_hash(storage),bytes2hex(sha256(IOBuffer(take!(rawframe)))),rawcount,length(chosen),rawcount-length(chosen),complete,declared,declared-length(chosen),uncovered)
end
function _word(storage,at,width)
    bits=UInt64(0)
    for j in 0:width-1;bits|=UInt64(codeunit(storage,at+j))<<(8*j);end
    bits
end
function witness(c::SourceCatalog,index::Integer)
    need(!(index isa Bool)&&1<=index<=length(c),"catalog witness index differs")
    at=WITNESS_BYTES*(Int(index)-1)+1;tag=reinterpret(Int64,_word(c.selected_storage,at,8))
    values=ntuple(j->reinterpret(Float64,_word(c.selected_storage,at+16+8*(j-1),8)),4)
    id=bytes2hex([codeunit(c.selected_storage,j) for j in at+48:at+79])
    (;tag=EPOCH+Millisecond(tag),source_line=UInt32(_word(c.selected_storage,at+8,4)),valid_mask=codeunit(c.selected_storage,at+12),reason_bits=codeunit(c.selected_storage,at+13),values,body_id=id)
end
function locate(c::SourceCatalog,tag::DateTime)
    need(floor(tag,Minute)==tag,"catalog lookup needs exact minute")
    target=_milliseconds(tag);lo=1;hi=length(c)
    while lo<=hi
        mid=lo+(hi-lo)÷2;stamp=reinterpret(Int64,_word(c.selected_storage,WITNESS_BYTES*(mid-1)+1,8))
        stamp==target&&return witness(c,mid)
        if stamp<target;lo=mid+1;else;hi=mid-1;end
    end
    nothing
end
end
