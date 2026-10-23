_need(ok,msg)=T._need(ok,msg)
_sha(x)=T._sha(x)
_id(x)=T._id(x)
_bytes(x)=T._bytes(x)
_fields(d,f)=T._fields(d,f)
function _ordinary(path)
    full=abspath(path);current=full
    while true
        _need(!islink(current),"linked B artifact path rejected")
        isfile(current)&&_need(stat(current).nlink==1,"hard-linked B artifact member rejected")
        if Sys.iswindows()&&ispath(current)
            attributes=ccall((:GetFileAttributesW,"kernel32"),UInt32,(Cwstring,),current)
            _need(attributes!=typemax(UInt32)&&(attributes&UInt32(0x400))==0,"reparse B artifact path rejected")
        end
        parent=dirname(current);parent==current&&break;current=parent
    end
    full
end
function _root_new(path)
    full=_ordinary(path);_need(isdir(dirname(full))&&!ispath(full),"new ordinary B output required")
    mkdir(full);full
end
function _member(root,name)
    _need(name isa AbstractString&&isascii(name)&&!isempty(name)&&!isabspath(name)&&!occursin('\\',name)&&!occursin(':',name)&&all(x->!isempty(x)&&x!="."&&x!="..",split(name,'/')),"B relative member differs")
    _ordinary(joinpath(root,split(name,'/')...))
end
function _exclusive(path,body)
    _ordinary(dirname(path));_need(!ispath(path),"refusing B member overwrite")
    flags=Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL
    file=Base.Filesystem.open(path,flags,0o600)
    try
        _need(write(file,body)==length(body),"short B member write")
    finally close(file)
    end
end
function _bound(path,limit=T.MAX_META)
    _ordinary(path);_need(isfile(path)&&filesize(path)<=limit,"bounded regular B member required")
    declared=filesize(path)
    body=open(path,"r") do io;read(io,min(limit+1,declared+1));end
    _need(length(body)==declared&&filesize(path)==declared,"B member length changed during bounded read");body
end
function _json(path;fields=nothing)
    body=_bound(path);_need(!(length(body)>=3&&body[1:3]==UInt8[0xef,0xbb,0xbf]),"B UTF8 BOM rejected")
    object=try JSON3.read(body) catch;throw(ArgumentError("invalid B JSON"));end
    _need(_bytes(object)==body,"noncanonical or duplicate-key B JSON rejected")
    fields===nothing||_fields(object,fields)
    object,body
end
# Opaque software qualification receipts retain their native pretty formatting.
# Validate the consumed counter's original JSON token, since JSON3 deliberately
# normalizes integral Float64 literals (e.g. 0.0) to Int64 by default.
function _qualification_integer(body,counter_path)
    _need(length(body)<=T.MAX_META,"qualification metadata bound exceeded")
    n=length(body);position=Ref(1);path=String[];target=String.(collect(counter_path));found=Ref(false);result=Ref(Int64(0))
    whitespace()=begin
        while position[]<=n&&body[position[]] in (0x20,0x09,0x0a,0x0d);position[]+=1;end
    end
    function quoted()
        first=position[];_need(first<=n&&body[first]==0x22,"qualification JSON string required");position[]+=1
        while position[]<=n
            byte=body[position[]];position[]+=1
            if byte==0x22
                return String(JSON3.read(body[first:position[]-1]))
            elseif byte==0x5c
                _need(position[]<=n,"unterminated qualification escape");position[]+=1
            else
                _need(byte>=0x20,"qualification string control character")
            end
        end
        throw(ArgumentError("unterminated qualification string"))
    end
    function value(depth)
        _need(depth<=12,"qualification JSON nesting exceeds limit");whitespace();_need(position[]<=n,"incomplete qualification JSON")
        byte=body[position[]]
        if byte==0x7b
            position[]+=1;whitespace();names=Set{String}()
            if position[]<=n&&body[position[]]==0x7d;position[]+=1;return;end
            while true
                whitespace();name=quoted();_need(!(name in names),"duplicate qualification field");push!(names,name)
                whitespace();_need(position[]<=n&&body[position[]]==0x3a,"qualification colon required");position[]+=1
                push!(path,name);value(depth+1);pop!(path);whitespace()
                _need(position[]<=n,"incomplete qualification object");separator=body[position[]];position[]+=1
                separator==0x7d&&break;_need(separator==0x2c,"qualification object separator differs")
            end
        elseif byte==0x5b
            position[]+=1;whitespace()
            if position[]<=n&&body[position[]]==0x5d;position[]+=1;return;end
            while true
                push!(path,"#");value(depth+1);pop!(path);whitespace()
                _need(position[]<=n,"incomplete qualification array");separator=body[position[]];position[]+=1
                separator==0x5d&&break;_need(separator==0x2c,"qualification array separator differs")
            end
        elseif byte==0x22
            quoted();_need(path!=target,"qualification counter must have an unsigned integer token")
        else
            first=position[]
            while position[]<=n&&!(body[position[]] in (0x20,0x09,0x0a,0x0d,0x2c,0x5d,0x7d));position[]+=1;end
            _need(position[]>first,"qualification scalar missing")
            if path==target
                count=position[]-first
                _need(!found[]&&1<=count<=19&&all(x->0x30<=x<=0x39,@view(body[first:position[]-1]))&&(count==1||body[first]!=0x30),"qualification counter must have a canonical bounded unsigned integer token")
                result[]=try parse(Int64,String(body[first:position[]-1])) catch;throw(ArgumentError("qualification counter exceeds Int64"));end
                found[]=true
            end
        end
    end
    try
        value(0);whitespace();_need(position[]==n+1&&found[],"qualification counter token absent or trailing data")
    catch e
        e isa ArgumentError&&rethrow()
        throw(ArgumentError("invalid qualification JSON token grammar"))
    end
    result[]
end
_qualification_counter(body)=(_need(_qualification_integer(body,("harness","counts","FAIL"))==0,"qualification FAIL must be integer token zero");true)
const SOFTWARE_QUALIFICATION_FIELDS=(:schema,:complete,:evidence_mode,:head,:source_sha256,:harness,:native_jobs,:docs_and_containers,:root_qualification_sha256,:source_proof_sha256,:canonical_harness_sha256,:software_qualification_only,:actual_panel_fit_or_outcome_qualified,:serving_promotion)
function _software_qualification(body;schema,evidence_mode,head,sources)
    # Token grammar is checked before JSON3's integer normalization.
    counts=ntuple(i->_qualification_integer(body,("harness","counts",("PASS","WARN","FAIL")[i])),3)
    _need(counts[3]==0,"component software qualification has failures")
    q=try JSON3.read(body) catch;throw(ArgumentError("invalid component software qualification"));end
    _fields(q,SOFTWARE_QUALIFICATION_FIELDS)
    _need(q.schema==schema&&q.complete===true&&q.evidence_mode==evidence_mode&&q.head==head&&q.software_qualification_only===true&&q.actual_panel_fit_or_outcome_qualified===false&&q.serving_promotion===false,"component software qualification schema/head/scope differs")
    _fields(q.harness,(:counts,:all17_independent_workers_complete,:unfiltered_pkg_test_complete,:experiments_entry_complete));_fields(q.harness.counts,(:PASS,:WARN,:FAIL))
    _need(q.harness.all17_independent_workers_complete===true&&q.harness.unfiltered_pkg_test_complete===true&&q.harness.experiments_entry_complete===true,"component full package/experiments/harness qualification required")
    for (field,value) in zip((:PASS,:WARN,:FAIL),counts);_need(T._int(getproperty(q.harness.counts,field),0,typemax(Int64))==value,"component qualification counter differs");end
    _need(q.source_sha256 isa AbstractDict&&T._hashmap(pairs(q.source_sha256))==T._hashmap(sources),"component qualification complete expected source map differs")
    for field in (:root_qualification_sha256,:source_proof_sha256,:canonical_harness_sha256);hash=getproperty(q,field);_need(T._hex(hash)&&hash!=repeat("0",64),"component raw qualification/source/harness identity link absent");end
    native=collect(q.native_jobs);other=collect(q.docs_and_containers)
    _need(length(native)==6&&length(other)==2,"component exact eight qualification cells required")
    for cell in vcat(native,other)
        _fields(cell,(:name,:conclusion,:receipt_sha256,:log_sha256))
        _need(cell.conclusion=="success"&&all(hash->T._hex(hash)&&hash!=repeat("0",64),(cell.receipt_sha256,cell.log_sha256)),"component qualification cell result/raw identity differs")
    end
    _need(Set(String(x.name) for x in native)==Set("Julia $v / $os" for v in ("1.12.6","1.13.1") for os in ("ubuntu-latest","windows-latest","macos-latest"))&&Set(String(x.name) for x in other)==Set(("Documentation build","Julia 1.12 deployment images")),"component exact eight qualification cell names required")
    q
end
function _file_sha(path)
    _ordinary(path);_need(isfile(path)&&filesize(path)<=T.MAX_FILE,"bounded B data file required")
    context=SHA.SHA2_256_CTX();total=0
    open(path,"r") do io
        while !eof(io)
            data=read(io,65536);total+=length(data);_need(total<=T.MAX_FILE,"B data file grew beyond bound");SHA.update!(context,data)
        end
    end
    _need(total==filesize(path),"B file size changed while hashing")
    bytes2hex(SHA.digest!(context)),total
end
function _source(expected)
    observed=T._hashmap(source_identity())
    _need(T._hashmap(expected)==observed,"B expected artifact source map differs")
    observed
end
_sources_data(x)=T._hashdata(T._hashmap(x))
function _sources(d)
    _need(d isa AbstractVector&&length(d)<=2048,"B source-map vector required")
    T._hashmap([begin _fields(x,(:name,:sha256));String(x.name)=>String(x.sha256) end for x in d])
end
function _invalidate(root,name,body)
    path=_member(root,name)
    if body!==nothing&&isfile(path)&&_bound(path)==body
        preserved=path*".incomplete"
        _exclusive(preserved,body);rm(path) # Only the exact completion owned by this attempt.
    end
end
function _header(io,magic,record)
    write(io,magic);write(io,htol(UInt32(1)));write(io,htol(UInt32(record)))
end
function _check_header(io,magic,record)
    _need(read(io,8)==collect(codeunits(magic))&&ltoh(read(io,UInt32))==1&&ltoh(read(io,UInt32))==record,"B binary header differs")
end
_put_float(io,x)=write(io,htol(isfinite(x) ? reinterpret(UInt64,x) : UInt64(0x7ff8000000000000)))
function _get_float(io;missing=false)
    bits=ltoh(read(io,UInt64));value=reinterpret(Float64,bits)
    _need(isfinite(value)||(missing&&bits==UInt64(0x7ff8000000000000)),"B finite/canonical missing scalar differs")
    value
end
_put_hash(io,x)=(_need(T._hex(x),"B row digest differs");write(io,hex2bytes(x)))
_get_hash(io)=bytes2hex(read(io,32))
const A_REASONS=(:model,:anchor_missing,:known_driver_missing,:warmup)
const B_REASONS=(:available,:latest_driver_unavailable,:latest_driver_missing,:unknown_initial_age,:global_state_support_missing,:first_S_mark_support_missing)
function _basis_bytes(row::T.JoinedRow)
    d=row.design.values;io=IOBuffer();write(io,htol(Int64(Dates.value(d.issue))))
    for x in (d.anchor,row.target...,d.offsets...,d.coefficients[1]...,d.coefficients[2]...);_put_float(io,x);end
    flags=UInt8(Int(isfinite(d.anchor))+2Int(isfinite(row.target[1]))+4Int(isfinite(row.target[2]))+8Int(d.model_available))
    ar=findfirst(==(d.a_reason),A_REASONS);br=findfirst(==(d.b_reason),B_REASONS)
    _need(ar!==nothing&&br!==nothing,"B basis reason differs")
    write(io,flags,UInt8(ar-1),UInt8(br-1));write(io,zeros(UInt8,5))
    for x in (d.a_prefix,d.wind_prefix,d.state_id,d.law_id,d.a_checkpoint_id);_put_hash(io,x);end
    body=take!(io);_need(length(body)==248,"B basis record length differs");body
end
function _read_basis_row(io,binding,law,physical,role,index)
    ms=ltoh(read(io,Int64));issue=DateTime(Dates.UTInstant{Millisecond}(Millisecond(ms)))
    values=ntuple(_->_get_float(io;missing=true),9)
    anchor=values[1];targets=(values[2],values[3]);offsets=(values[4],values[5]);coeff=((values[6],values[7]),(values[8],values[9]))
    flags=read(io,UInt8);ar=read(io,UInt8);br=read(io,UInt8);reserved=read(io,5)
    _need(flags&0xf0==0&&ar<length(A_REASONS)&&br<length(B_REASONS)&&reserved==zeros(UInt8,5),"B basis flags/reason/reserved differ")
    _need((flags&1!=0)==isfinite(anchor)&&(flags&2!=0)==isfinite(targets[1])&&(flags&4!=0)==isfinite(targets[2]),"B basis finite flags differ")
    available=flags&8!=0;areason=A_REASONS[Int(ar)+1];breason=B_REASONS[Int(br)+1]
    _need(available==(areason==:model&&breason==:available),"B basis availability/reason differs")
    ids=ntuple(_->_get_hash(io),5)
    _need(ids[4]==law.model_id&&(!available||ids[5]!=repeat("0",64)),"B basis law/checkpoint identity differs")
    _need(all(isfinite,Iterators.flatten(coeff))&&all(x->isfinite(x)||isnan(anchor),offsets),"B basis numeric domain differs")
    if !available
        _need(isequal(offsets,(anchor,anchor))&&all(x->isequal(x,0.),Iterators.flatten(coeff)),"B fallback is not exact common persistence")
    end
    v=T.DesignValues(index,issue,anchor,offsets,coeff,areason==:model,available,areason,breason,ids...,binding.context_id)
    role==:training ? T.BTrainingRow(T.BTrainingDesignRow(v),targets) : T.BForecastRow(T.BForecastDesignRow(v),targets)
end
