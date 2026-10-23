# New B JSON has one canonical representation. Source-length values are Vectors.
const MAX_ROWS=500_000
const MAX_CHUNK=4096
const MAX_META=2_097_152
const MAX_FILE=536_870_912
const HashMap=Vector{Pair{String,String}}
const PairFloat=NTuple{2,Float64}
const Coefficients=NTuple{2,PairFloat}
_sha(x)=bytes2hex(sha256(x))
_hex(x)=x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)
_bits(x::Float64)=(_need(isfinite(x),"finite B Float64 required");string(reinterpret(UInt64,x);base=16,pad=16))
function _float(x)
    _need(x isa AbstractString&&occursin(r"\A[0-9a-f]{16}\z",x),"canonical B float bits required")
    value=reinterpret(Float64,parse(UInt64,x;base=16));_need(isfinite(value),"finite B float required");value
end
_int(x,lo,hi)=(_need(x isa Integer&&!(x isa Bool)&&lo<=x<=hi,"bounded non-Boolean integer required");Int(x))
function _clock(x)
    _need(x isa AbstractString&&occursin(r"\A(?:0|-?[1-9][0-9]*)\z",x),"canonical clock required")
    value=try parse(Int64,x) catch;throw(ArgumentError("clock outside Int64"));end
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(value)))
end
function _canonical(io,x,depth=0)
    _need(depth<=12,"B JSON nesting exceeds limit")
    if x isa NamedTuple || x isa AbstractDict
        names=String.(collect(keys(x)));_need(all(n->isascii(n)&&!isempty(n),names)&&allunique(names),"unique ASCII JSON keys required")
        sort!(names);write(io,'{')
        for (i,name) in enumerate(names)
            i>1&&write(io,',');JSON3.write(io,name);write(io,':')
            value=x isa NamedTuple ? getproperty(x,Symbol(name)) : haskey(x,name) ? x[name] : x[Symbol(name)]
            _canonical(io,value,depth+1)
        end
        write(io,'}')
    elseif x isa Tuple || x isa AbstractVector
        write(io,'[');for (i,v) in enumerate(x);i>1&&write(io,',');_canonical(io,v,depth+1);end;write(io,']')
    elseif x isa Float64
        JSON3.write(io,_bits(x))
    elseif x isa DateTime
        JSON3.write(io,string(Dates.value(x)))
    elseif x isa Symbol
        JSON3.write(io,String(x))
    elseif x isa AbstractString || x isa Bool || x===nothing || (x isa Integer&&!(x isa Bool))
        JSON3.write(io,x)
    else
        throw(ArgumentError("unsupported B JSON value $(typeof(x))"))
    end
end
_bytes(x)=begin io=IOBuffer();_canonical(io,x);write(io,'\n');take!(io) end
_id(x)=_sha(_bytes(x))
function _fields(d,fields)
    _need(Set(String.(keys(d)))==Set(String.(fields)),"B schema field set differs");d
end
function _hashmap(x)
    pairs=Pair{String,String}[String(first(p))=>String(last(p)) for p in x]
    sort!(pairs;by=first)
    _need(length(pairs)<=2048&&allunique(first.(pairs)),"bounded unique B source map required")
    for (name,hash) in pairs
        _need(ncodeunits(name)<=256&&isascii(name)&&!startswith(name,"/")&&!occursin('\\',name)&&all(x->x!=".."&&x!="."&&!isempty(x),split(name,'/'))&&_hex(hash),"B source map entry differs")
    end
    pairs
end
_hashdata(x::HashMap)=[(;name=first(p),sha256=last(p)) for p in x]
function _input_digest(input)
    times=SHA.SHA2_256_CTX();raw=SHA.SHA2_256_CTX()
    buffer=IOBuffer()
    for t in input.time
        write(buffer,htol(Int64(Dates.value(t))))
        position(buffer)>=65536&&SHA.update!(times,take!(buffer))
    end
    position(buffer)>0&&SHA.update!(times,take!(buffer))
    for x in input.values
        word=isnan(x) ? UInt64(0x7ff8000000000000) : reinterpret(UInt64,x)
        write(buffer,htol(word));position(buffer)>=65536&&SHA.update!(raw,take!(buffer))
    end
    position(buffer)>0&&SHA.update!(raw,take!(buffer))
    _id((;schema="b_point_input_prefix_v1",source_sha256=input.source_sha256,time_sha256=bytes2hex(SHA.digest!(times)),raw_sha256=bytes2hex(SHA.digest!(raw)),source_rows=length(input.time)))
end
struct PointInputView
    schema::String
    input::A.HourlyInputs
    wind::Vector{D.WindHour}
    evidence_mode::Symbol
    source_sha256::String
    input_prefix_sha256::String
    wind_sequence_sha256::String
    input_admission_sha256::String
    source_hours::Int
    source_hashes::HashMap
    contract_sha256::String
    view_id::String
end
_view_data(v::PointInputView)=(;schema=v.schema,evidence_mode=v.evidence_mode,source_sha256=v.source_sha256,input_prefix_sha256=v.input_prefix_sha256,wind_sequence_sha256=v.wind_sequence_sha256,input_admission_sha256=v.input_admission_sha256,source_hours=v.source_hours,source_hashes=_hashdata(v.source_hashes),contract_sha256=v.contract_sha256)
function _point_input_view(input::A.HourlyInputs,wind::AbstractVector{D.WindHour},binding)
    hashes=_hashmap(source_identity());n=length(input.time)
    _need(1<=n<=MAX_ROWS&&axes(input.values)==(Base.OneTo(17),Base.OneTo(n))&&axes(wind)==(Base.OneTo(n),),"bounded matching ordinary source axes required")
    _need(_hex(input.source_sha256)&&input.availability_kind==:assumed_completed_hour,"assumed-hour hash-bound input required")
    mode=Symbol(binding.evidence_mode);_need(mode in (:synthetic_only,:archival_assumed_completion_research),"B evidence mode differs")
    _need(_hex(binding.input_admission_sha256)&&binding.contract_sha256==CONTRACT,"B input binding differs")
    owned=A.HourlyInputs(copy(input.time),copy(input.values);source_sha256=input.source_sha256,availability_kind=input.availability_kind)
    rows=copy(wind);D._ordered(rows)
    clean(channel,i)=begin x=owned.values[channel,i];(!isfinite(x)||x>=Adapter.Features.FILL_LIMITS[channel]) ? NaN : x end
    for i in 1:n
        row=rows[i];bz=clean(6,i);speed=clean(8,i)
        q=isfinite(bz)&&isfinite(speed)&&speed>=0 ? max(speed*max(-bz,0.)/1000-.49,0.) : NaN
        _need(row.tag==owned.time[i]&&row.completed==D._shift(row.tag,1)&&row.received==row.completed,"archival completion clock differs")
        _need(isequal(row.bz,bz)&&isequal(row.q,q),"wind/raw source values differ")
    end
    fields=("b_point_input_view_v1",owned,rows,mode,owned.source_sha256,_input_digest(owned),D._prefix_digest(rows),String(binding.input_admission_sha256),n,hashes,CONTRACT)
    v=PointInputView(fields...,repeat("0",64));PointInputView(fields...,_id(_view_data(v)))
end
function _recheck(v::PointInputView)
    _need(v.schema=="b_point_input_view_v1"&&v.contract_sha256==CONTRACT&&v.source_hashes==_hashmap(source_identity()),"B input source binding differs")
    _need(v.source_hours==length(v.input.time)==length(v.wind)&&v.input_prefix_sha256==_input_digest(v.input)&&v.wind_sequence_sha256==D._prefix_digest(v.wind)&&v.view_id==_id(_view_data(v)),"B input view changed")
    true
end
struct BSetting
    index::Int
    a_index::Int
    physical_schedule::Int
    taus::PairFloat
    sigma::Float64
    eta::Float64
    beta::Int
end
settings()=[BSetting(2*(a.index-1)+bi,a.index,(a.index-1)÷2+1,a.taus,a.sigma,a.eta,beta) for a in Adapter.settings() for (bi,beta) in enumerate((5,20))]
function _setting(s::BSetting)
    _need(1<=s.index<=36&&isequal(s,settings()[s.index]),"B fixed setting differs");s
end
struct PointTrainingBinding
    schema::String
    role::Symbol
    fold_id::String
    boundary::DateTime
    cutoff::DateTime
    delay_hours::Int
    source_sha256::String
    input_admission_sha256::String
    input_prefix_sha256::String
    input_view_id::String
    training_mask_sha256::String
    wind_sequence_sha256::String
    hazard_model_id::String
    hazard_training_prefix_sha256::String
    source_hashes::HashMap
    contract_sha256::String
    context_id::String
end
function _binding_data(b::PointTrainingBinding)
    (;schema=b.schema,role=b.role,fold_id=b.fold_id,boundary=b.boundary,cutoff=b.cutoff,delay_hours=b.delay_hours,source_sha256=b.source_sha256,input_admission_sha256=b.input_admission_sha256,input_prefix_sha256=b.input_prefix_sha256,input_view_id=b.input_view_id,training_mask_sha256=b.training_mask_sha256,wind_sequence_sha256=b.wind_sequence_sha256,hazard_model_id=b.hazard_model_id,hazard_training_prefix_sha256=b.hazard_training_prefix_sha256,source_hashes=_hashdata(b.source_hashes),contract_sha256=b.contract_sha256)
end
struct DesignValues
    index::Int
    issue::DateTime
    anchor::Float64
    offsets::PairFloat
    coefficients::Coefficients
    a_model_available::Bool
    model_available::Bool
    a_reason::Symbol
    b_reason::Symbol
    a_prefix::String
    wind_prefix::String
    state_id::String
    law_id::String
    a_checkpoint_id::String
    context_id::String
end
struct BTrainingDesignRow
    values::DesignValues
end
struct BForecastDesignRow
    values::DesignValues
end
struct BTrainingRow
    design::BTrainingDesignRow
    target::PairFloat
end
struct BForecastRow
    design::BForecastDesignRow
    target::PairFloat
end
struct BTrainingPanel
    rows::Vector{BTrainingRow}
    binding::PointTrainingBinding
    law::D.DurationModel
    physical_schedule::Int
    basis_sha256::String
    evidence_mode::Symbol
end
struct BForecastPanel
    rows::Vector{BForecastRow}
    binding::PointTrainingBinding
    law::D.DurationModel
    physical_schedule::Int
end
struct BPointPredictionRow
    issue::DateTime
    means::PairFloat
    anchor::Float64
    target::PairFloat
    model_available::Bool
    reason::Symbol
    source_sha256::String
    delay_hours::Int
end
struct BPointModel
    schema::String
    evidence_mode::Symbol
    training_role::Symbol
    contract_sha256::String
    fold_id::String
    setting_index::Int
    a_setting_index::Int
    beta::Int
    taus::PairFloat
    sigma::Float64
    eta::Float64
    delay_hours::Int
    boundary::DateTime
    cutoff::DateTime
    gains::PairFloat
    hazard_model_id::String
    hazard_training_prefix_sha256::String
    source_sha256::String
    input_prefix_sha256::String
    input_view_id::String
    input_admission_sha256::String
    context_id::String
    source_hashes::HashMap
    fit_count::Int
    informative_count::Int
    storm_scalar_count::Int
    fit_mask_sha256::String
    fit_data_sha256::String
    fit_input_manifest_sha256::String
    max_fit_issue::DateTime
    max_target_complete::DateTime
    max_label_receipt::DateTime
    solver::NamedTuple
    model_id::String
    law::D.DurationModel # Private ownership; persisted through the unchanged D codec.
end
