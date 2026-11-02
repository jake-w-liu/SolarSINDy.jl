module CMEGSMCDFMinuteReader

using SHA, Dates, JSON3
include(joinpath(@__DIR__, "gsm_minute_labels.jl"))
const Labels = CMEGSMMinuteLabels
const SOURCE_SHA = bytes2hex(sha256(read(@__FILE__)))
const READER_UUID = "a9737db6-c05c-4e48-868b-6bc41491d9d9"
const READER_VERSION = v"0.5.1"
const MASTER_SHA = "1cd0bfaf378745ae1d3a685eb8b3bbf4878b905e2d01ce169b5c6ae90d77e347"
const MAX_RAW_BYTES = 64*1024^2
const BUFFER_BYTES = 1024^2
const PHYSICAL_MINIMUM = 8*1024^3
const RSS_LIMIT = 8*1024^3
const API_SOURCE_PINS = (
    "Project.toml" => "1861598b276a1910ce35162c65ca9cd47cb1f34ddba898d029ed116d1685efc5",
    "src/CommonDataFormat.jl" => "f5918d48fec0b9ce302a03d6989723bf10373c227b85c0eb2d35257b0bcc751a",
    "src/dataset.jl" => "7b9da9bcdef245f2fb978feeefb78e23f9f1a66f250d06195ee958b291b3abfb",
    "src/variable.jl" => "f46d0165de048ff6599806f0bac8edaefa7ccd9eeab2ec9fd1f71aa59cd43f1c",
    "src/epochs.jl" => "c2b90f6c42e84d887260c0a34e0abb4b19008624df437d72e5e7f0a095e4486a",
    "src/decompress.jl" => "fd09e9a0d044e7f7c78cc2e8c3da5158669de26d364ddef84a4a3ccc5c631567",
    "src/loading/variable.jl" => "7fe6f9bd91bf60c6e66626591972fbb120e5481ebef47249cb97fe6d383c8f28",
    "src/records/vdr.jl" => "3f5c90091636e0e5847442f285bb1477a70a9b440a808405b1f9df3608962b14",
)

need(value,message) = value || throw(ArgumentError(message))
hex64(value) = value isa AbstractString && occursin(r"^[0-9a-f]{64}$",value)

function source_guard()
    need(bytes2hex(sha256(read(@__FILE__))) == SOURCE_SHA, "loaded/disk CDF adapter differs")
    Labels.source_guard()
end

"This checks API identity, not native package/source/scientific qualification."
function validate_reader_module(reader::Module)
    id = Base.PkgId(reader)
    need(id.name == "CommonDataFormat" && string(id.uuid) == READER_UUID, "only the registered CommonDataFormat module is supported")
    need(Base.pkgversion(reader) == READER_VERSION, "registered CommonDataFormat0.5.1 required")
    entry = pathof(reader); need(entry isa String, "installed reader source path missing")
    root = dirname(dirname(entry))
    for (relative,expected) in API_SOURCE_PINS
        bytes = Labels.Bootstrap.bounded_bytes(joinpath(root,split(relative,'/')...),65536)
        need(bytes2hex(sha256(bytes)) == expected, "registered reader API source differs: "*relative)
    end
    reader
end

"No reader import/Pkg action occurs here. The outer admitted environment supplies the real module."
function _real_api(reader::Module)
    validate_reader_module(reader)
    (; open_dataset=path -> reader.CDFDataset(path), close_dataset=close,
        variable=(ds,name) -> ds[name], global_attribute=(ds,key) -> get(ds.attrib,key,nothing),
        attribute=(var,key) -> get(var.attrib,key,nothing), dimensions=size, storage_type=eltype,
        record_varying=reader.is_record_varying,
        sparse_mode=var -> Int(var.vdr.s_records), elements=var -> Int(var.vdr.num_elems),
        read_variable=Array, epoch_type=reader.Epoch,
        epoch_instant=value -> getfield(value,:instant),
        pad_bytes=(var,ds,n) -> collect(reader.pad_bytes(var.vdr,ds,n)),
        big_endian=var -> reader.is_big_endian_encoding(var))
end

function _physical_and_peak_guard(phase)
    phase === :admit && need(Sys.free_memory() >= PHYSICAL_MINIMUM, "actual CDF operation requires8GiB available physical memory")
    need(Sys.maxrss() <= RSS_LIMIT, "native Julia peak RSS exceeded8GiB")
end

function _file_state(info)
    (; device=info.device,inode=info.inode,size=info.size,mtime=info.mtime,ctime=info.ctime)
end

"Bounded streaming raw hash. The adapter never downloads or mutates a source file."
function _file_witness(path,expected_sha256)
    need(path isa AbstractString && isabspath(path) && normpath(path) == path, "canonical absolute source path required")
    need(isfile(path) && !islink(path) && realpath(path) == path && hex64(expected_sha256), "ordinary pinned source file required")
    before = stat(path); need(0 < before.size <= MAX_RAW_BYTES, "raw source byte bound")
    context = SHA.SHA2_256_CTX(); count = 0
    open(path,"r") do io
        need(_file_state(stat(io)) == _file_state(before), "source changed before raw open")
        while !eof(io)
            bytes = read(io,min(BUFFER_BYTES,MAX_RAW_BYTES+1-count))
            count += length(bytes); need(count <= MAX_RAW_BYTES, "raw source grew beyond bound")
            SHA.update!(context,bytes)
        end
        need(_file_state(stat(io)) == _file_state(before), "opened raw source changed")
    end
    after = stat(path); need(_file_state(before) == _file_state(after) && count == before.size, "raw source changed while hashing")
    actual = bytes2hex(SHA.digest!(context)); need(actual == expected_sha256, "raw source SHA differs")
    (; path=String(path),sha256=actual,state=_file_state(after))
end

function _with_checked_file(action,api,path,expected_sha256,memory_guard)
    memory_guard(:admit); before = _file_witness(path,expected_sha256)
    ds = nothing; result = nothing; errors = Exception[]
    try
        ds = api.open_dataset(path)
        memory_guard(:opened)
        result = action(ds,before)
    catch error
        push!(errors,error)
    finally
        if ds !== nothing
            try
                api.close_dataset(ds)
            catch error
                push!(errors,error)
            end
        end
        try
            need(_file_witness(path,expected_sha256) == before, "source identity changed across CDF operation")
        catch error
            push!(errors,error)
        end
        try
            memory_guard(:finished)
        catch error
            push!(errors,error)
        end
    end
    isempty(errors) || throw(length(errors) == 1 ? only(errors) : CompositeException(errors))
    result
end

function _scalar(value,label)
    if value isa AbstractVector
        need(length(value) == 1,"scalar metadata required: "*label)
        return only(value)
    end
    value
end
function _text(value,label)
    value = _scalar(value,label)
    need(value isa AbstractString,"text metadata required: "*label)
    String(value)
end
function _numeric_attribute(api,var,key,::Type{T},expected) where T
    value = api.attribute(var,key)
    values = value isa AbstractVector ? collect(value) : [value]
    need(length(values) == length(expected) && all(x -> x isa T,values) && values == expected, "native typed metadata differs: "*key)
    T[value for value in values]
end
function _expected_bytes(value,big_endian)
    bits = value isa Float32 ? reinterpret(UInt32,value) : value isa Float64 ? reinterpret(UInt64,value) :
        value isa Int32 ? reinterpret(UInt32,value) : throw(ArgumentError("unsupported metadata byte type"))
    width = sizeof(bits)
    UInt8[(bits >> (8*(big_endian ? width-1-index : index))) & 0xff for index in 0:width-1]
end

function _variable_schema(api,ds,name,storage,dimensions,record_varying,pad)
    var = api.variable(ds,name)
    need(api.storage_type(var) === storage && api.dimensions(var) == dimensions, "native type/dimensions differ: "*name)
    need(api.record_varying(var) === record_varying && api.elements(var) == 1, "record variance/elements differ: "*name)
    # The reader repeats previous physical records in this mode, forbidden by
    # the locked protocol. Reject before any variable Array/read call.
    sparse = api.sparse_mode(var)
    need(sparse in (0,1), "previous-record carry-forward or unknown sparse mode forbidden: "*name)
    need(api.pad_bytes(var,ds,sizeof(pad)) == _expected_bytes(pad,api.big_endian(var)), "native effective pad differs: "*name)
    var
end

function _static_label_variable(api,ds,name,width)
    var = api.variable(ds,name)
    need(api.storage_type(var) === String && api.dimensions(var) == (3,1), "static coordinate-label shape/type differs")
    need(api.record_varying(var) === false && api.elements(var) == width && api.sparse_mode(var) == 0, "static coordinate-label descriptor differs")
    need(_text(api.attribute(var,"VAR_TYPE"),name) == "metadata", "coordinate labels must be metadata")
    var
end
function _static_labels(api,ds,name,width,expected)
    var = _static_label_variable(api,ds,name,width)
    values = vec(api.read_variable(var))
    need(values == expected,"GSM component order or coordinate labels differ")
    String.(values)
end

"Selected native descriptors/attributes only; no variable Array/read calls."
function _metadata(api,ds)
    need(_text(api.global_attribute(ds,"Logical_source"),"Logical_source") == "wi_h0_mfi", "wrong original CDF logical source")
    epoch = api.variable(ds,"Epoch")
    dims = api.dimensions(epoch)
    need(length(dims) == 1 && 0 <= only(dims) <= Labels.MAX_CHUNK_RECORDS,"one-dimensional bounded Epoch required")
    n = Int(only(dims))
    epoch = _variable_schema(api,ds,"Epoch",api.epoch_type,(n,),true,0.0)
    field = _variable_schema(api,ds,"BGSM",Float32,(3,n),true,Labels.NATIVE_PAD32)
    points = _variable_schema(api,ds,"NUM_PTS",Int32,(n,),true,Int32(-2147483647))
    for (var,name,units,var_type) in ((epoch,"Epoch","ms","support_data"),
            (field,"BGSM","nT (1min)","data"),(points,"NUM_PTS"," ","support_data"))
        need(_text(api.attribute(var,"UNITS"),name*" units") == units,"units differ: "*name)
        need(_text(api.attribute(var,"TIME_RES"),name*" cadence") == "1 min","cadence differs: "*name)
        need(_text(api.attribute(var,"VAR_TYPE"),name*" type") == var_type,"variable role differs: "*name)
    end
    need(_text(api.attribute(epoch,"CATDESC"),"Epoch description") == "Time, Centered, Number of milliseconds since the epoch (1 min)", "centered Time support metadata differs")
    need(_text(api.attribute(epoch,"MONOTON"),"Epoch monotonicity") == "INCREASE", "Time monotonic metadata differs")
    for var in (field,points)
        need(_text(api.attribute(var,"DEPEND_0"),"DEPEND_0") == "Epoch","quality/vector Time dependency differs")
    end
    need(_text(api.attribute(field,"LABL_PTR_1"),"LABL_PTR_1") == "label_bgsm" &&
        _text(api.attribute(field,"DEPEND_1"),"DEPEND_1") == "cartesian","GSM coordinate dependency differs")
    need(_text(api.attribute(field,"CATDESC"),"BGSM description") == "Magnetic field vector in GSM cartesian coordinates (1 min)", "GSM frame description differs")
    field_fill = _numeric_attribute(api,field,"FILLVAL",Float32,[Labels.NATIVE_FILL32])
    minimum = _numeric_attribute(api,field,"VALIDMIN",Float32,fill(Float32(-65534),3))
    maximum = _numeric_attribute(api,field,"VALIDMAX",Float32,fill(Float32(65534),3))
    point_fill = _numeric_attribute(api,points,"FILLVAL",Int32,[Int32(-1)])
    point_minimum = _numeric_attribute(api,points,"VALIDMIN",Int32,[Int32(0)])
    point_maximum = _numeric_attribute(api,points,"VALIDMAX",Int32,[Int32(1500)])
    epoch_fill = _numeric_attribute(api,epoch,"FILLVAL",Float64,[-1e31])
    bounds = [_scalar(api.attribute(epoch,key),key) for key in ("VALIDMIN","VALIDMAX")]
    need(all(value -> value isa Float64 && isfinite(value),bounds) && bounds[1] < bounds[2],"Time validity bounds differ")
    _static_label_variable(api,ds,"label_bgsm",8)
    _static_label_variable(api,ds,"cartesian",11)
    # Only copied primitives leave this function; no dataset/lazy variable escapes.
    signature = (; logical_source="wi_h0_mfi",records=n,epoch_type="CDF_EPOCH",field_type="CDF_REAL4",point_type="CDF_INT4",
        epoch_units="ms",field_units="nT (1min)",point_units=" ",cadence_ms=60000,centred=true,
        field_fill_bits=[string(reinterpret(UInt32,value);base=16,pad=8) for value in field_fill],field_minimum=minimum,field_maximum=maximum,
        point_fill,point_minimum,point_maximum,epoch_fill,epoch_bounds=bounds,
        label_descriptor=(name="label_bgsm",width=8,shape=(3,1)),cartesian_descriptor=(name="cartesian",width=11,shape=(3,1)),
        sparse_modes=(api.sparse_mode(epoch),api.sparse_mode(field),api.sparse_mode(points)),master_sha256=MASTER_SHA)
    (; epoch,field,points,n,signature,sha256=bytes2hex(sha256(JSON3.write(signature))))
end

function _epoch_milliseconds(api,values,lower,upper,bounds)
    need(length(values) <= Labels.MAX_CHUNK_RECORDS,"Time record budget")
    times = Int64[]
    for value in values
        need(value isa api.epoch_type,"native CDF_EPOCH value required")
        instant = api.epoch_instant(value)
        need(instant isa Float64 && isfinite(instant) && instant != -1e31 && instant != 0 &&
            isinteger(instant) && abs(instant) <= 9007199254740992.0,"noninteger/fill/pad CDF_EPOCH forbidden")
        need(bounds[1] <= instant <= bounds[2],"Time outside native validity metadata")
        # DateTime(Epoch) in0.5.1 rounds. Validate first, then use exact integer
        # subtraction instead of that rounding conversion.
        stamp = Int128(instant)-Int128(62167219200000)
        need(typemin(Int64) <= stamp <= typemax(Int64) && lower <= stamp < upper,"Time outside exact frozen UTC domain")
        push!(times,Int64(stamp))
    end
    need(issorted(times) && allunique(times),"duplicate/nonincreasing decoded Time records")
    times
end

struct TimePass
    file::NamedTuple
    metadata::NamedTuple
    metadata_sha256::String
    axis::Labels.TimeAxis
end

function _time_pass(api,path;raw_id,expected_raw_sha256,start_ms,stop_ms,memory_guard)
    first = Labels.millis(start_ms); stop = Labels.millis(stop_ms); need(first < stop,"UTC source domain")
    _with_checked_file(api,path,expected_raw_sha256,memory_guard) do ds,witness
        metadata = _metadata(api,ds)
        values = api.read_variable(metadata.epoch)
        need(size(values) == (metadata.n,),"decoded Epoch dimensions differ")
        times = _epoch_milliseconds(api,values,first,stop,metadata.signature.epoch_bounds)
        axis = Labels.time_phase_runs(times,first,stop;raw_id,raw_sha256=expected_raw_sha256,
            source_group="original-CDF:wi_h0_mfi:Epoch",metadata_sha256=metadata.sha256)
        TimePass(witness,metadata.signature,metadata.sha256,axis)
    end
end

"Requires an already admitted, owned native worker. This adapter grants no source or native admission."
function metadata_time_pass(reader::Module,path;raw_id,expected_raw_sha256,start_ms,stop_ms)
    source_guard(); api = _real_api(reader)
    _time_pass(api,path;raw_id,expected_raw_sha256,start_ms,stop_ms,memory_guard=_physical_and_peak_guard)
end

function _same_axis(first,second)
    first.times == second.times && first.start_ms == second.start_ms && first.stop_ms == second.stop_ms &&
        first.raw_id == second.raw_id && first.raw_sha256 == second.raw_sha256 &&
        first.time_sha256 == second.time_sha256 && first.parameters == second.parameters
end

function _qc_pass(api,path,time_pass::TimePass,schedule::Labels.PhaseSchedule;memory_guard)
    Labels.schedule_guard(schedule); Labels.axis_guard(time_pass.axis)
    need(time_pass.axis.raw_sha256 == time_pass.file.sha256 &&
        time_pass.axis.parameters.metadata_sha256 == time_pass.metadata_sha256 &&
        bytes2hex(sha256(JSON3.write(time_pass.metadata))) == time_pass.metadata_sha256,
        "Time-pass raw/metadata/axis bindings differ")
    need(schedule.resolved && _same_axis(schedule.axis,time_pass.axis),"explicit phase freeze must bind the exact Time pass")
    need(path == time_pass.file.path && _file_witness(path,time_pass.file.sha256) == time_pass.file,"same frozen CDF required for second pass")
    _with_checked_file(api,path,time_pass.file.sha256,memory_guard) do ds,witness
        need(witness == time_pass.file,"CDF identity differs from Time pass")
        metadata = _metadata(api,ds)
        need(metadata.signature == time_pass.metadata && metadata.sha256 == time_pass.metadata_sha256,"CDF metadata changed between passes")
        values = api.read_variable(metadata.epoch)
        need(size(values) == (metadata.n,),"decoded Epoch dimensions differ")
        times = _epoch_milliseconds(api,values,time_pass.axis.start_ms,time_pass.axis.stop_ms,metadata.signature.epoch_bounds)
        need(times == time_pass.axis.times,"Time changed between phase freeze and value pass")
        _static_labels(api,ds,"label_bgsm",8,["Bx (GSM)","By (GSM)","Bz (GSM)"])
        _static_labels(api,ds,"cartesian",11,["x-component","y-component","z-component"])
        # These are the only magnetic/count payload reads in the adapter. All
        # schema, file, native Time and explicit phase checks precede them.
        field = api.read_variable(metadata.field)
        points = api.read_variable(metadata.points)
        need(eltype(field) === Float32 && size(field) == (3,metadata.n) &&
            eltype(points) === Int32 && size(points) == (metadata.n,),"decoded native payload type/geometry differs")
        vectors = [(field[1,index],field[2,index],field[3,index]) for index in 1:metadata.n]
        Labels.minute_qc(schedule,times,vectors,times,points)
    end
end

function read_qc_pass(reader::Module,path,time_pass::TimePass,schedule::Labels.PhaseSchedule)
    source_guard(); api = _real_api(reader)
    _qc_pass(api,path,time_pass,schedule;memory_guard=_physical_and_peak_guard)
end

end
