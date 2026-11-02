module CMEGSMCDFReaderTests

using Test, SHA
include(joinpath(@__DIR__,"cdf_minute_reader.jl"))
const R = CMEGSMCDFMinuteReader
const G = R.Labels

struct FixtureEpoch
    instant::Float64
end
mutable struct FixtureVariable
    name::String
    storage::DataType
    dims::Tuple
    attributes::Dict{String,Any}
    record_varying::Bool
    sparse::Int
    elements::Int
    pad::Vector{UInt8}
    big_endian::Bool
    values::Any
end
mutable struct FixtureDataset
    variables::Dict{String,FixtureVariable}
    reads::Vector{String}
    opens::Int
    closes::Int
    fail_open::Bool
    fail_close::Bool
    fail_read::String
end
Base.show(::IO,::FixtureDataset) = error("dataset display is forbidden")
Base.show(::IO,::FixtureVariable) = error("variable display is forbidden")

function fixture(;big_endian=false)
    offset = 62167219200000.0
    epoch_attrs = Dict{String,Any}("UNITS"=>"ms","TIME_RES"=>"1 min","VAR_TYPE"=>"support_data",
        "CATDESC"=>"Time, Centered, Number of milliseconds since the epoch (1 min)","MONOTON"=>"INCREASE",
        "FILLVAL"=>[-1e31],"VALIDMIN"=>[offset-60000],"VALIDMAX"=>[offset+240000])
    field_attrs = Dict{String,Any}("UNITS"=>"nT (1min)","TIME_RES"=>"1 min","VAR_TYPE"=>"data",
        "CATDESC"=>"Magnetic field vector in GSM cartesian coordinates (1 min)","DEPEND_0"=>"Epoch",
        "DEPEND_1"=>"cartesian","LABL_PTR_1"=>"label_bgsm",
        "FILLVAL"=>[reinterpret(Float32,UInt32(0xf2fc6f7c))],
        "VALIDMIN"=>Float32[-65534,-65534,-65534],"VALIDMAX"=>Float32[65534,65534,65534])
    point_attrs = Dict{String,Any}("UNITS"=>" ","TIME_RES"=>"1 min","VAR_TYPE"=>"support_data",
        "DEPEND_0"=>"Epoch","FILLVAL"=>Int32[-1],"VALIDMIN"=>Int32[0],"VALIDMAX"=>Int32[1500])
    # Literal bytes are independent of the adapter's metadata byte encoder.
    field_pad = big_endian ? UInt8[0xf1,0x49,0xf2,0xca] : UInt8[0xca,0xf2,0x49,0xf1]
    point_pad = big_endian ? UInt8[0x80,0x00,0x00,0x01] : UInt8[0x01,0x00,0x00,0x80]
    variables = Dict(
        "Epoch"=>FixtureVariable("Epoch",FixtureEpoch,(3,),epoch_attrs,true,0,1,zeros(UInt8,8),big_endian,
            FixtureEpoch[FixtureEpoch(offset+30000),FixtureEpoch(offset+90000),FixtureEpoch(offset+150000)]),
        "BGSM"=>FixtureVariable("BGSM",Float32,(3,3),field_attrs,true,0,1,field_pad,big_endian,
            Float32[1 2 3;4 5 6;-2 0 -7]),
        "NUM_PTS"=>FixtureVariable("NUM_PTS",Int32,(3,),point_attrs,true,0,1,point_pad,big_endian,Int32[1,1,1]),
        "label_bgsm"=>FixtureVariable("label_bgsm",String,(3,1),Dict{String,Any}("VAR_TYPE"=>"metadata"),false,0,8,UInt8[],big_endian,
            reshape(["Bx (GSM)","By (GSM)","Bz (GSM)"],3,1)),
        "cartesian"=>FixtureVariable("cartesian",String,(3,1),Dict{String,Any}("VAR_TYPE"=>"metadata"),false,0,11,UInt8[],big_endian,
            reshape(["x-component","y-component","z-component"],3,1)))
    ds = FixtureDataset(variables,String[],0,0,false,false,"")
    api = (; open_dataset=path -> begin
            ds.opens += 1
            ds.fail_open && error("generated constructor failure")
            ds
        end,
        close_dataset=value -> begin
            @assert value === ds
            ds.closes += 1
            ds.fail_close && error("generated close failure")
            nothing
        end,
        variable=(value,name) -> value.variables[name],
        global_attribute=(value,key) -> key == "Logical_source" ? ["wi_h0_mfi"] : nothing,
        attribute=(var,key) -> get(var.attributes,key,nothing),dimensions=var -> var.dims,
        storage_type=var -> var.storage,record_varying=var -> var.record_varying,
        sparse_mode=var -> var.sparse,elements=var -> var.elements,
        read_variable=var -> begin
            push!(ds.reads,var.name)
            ds.fail_read == var.name && error("generated variable read failure")
            copy(var.values)
        end,
        epoch_type=FixtureEpoch,epoch_instant=value -> value.instant,
        pad_bytes=(var,value,n) -> copy(var.pad),big_endian=var -> var.big_endian)
    (; ds,api)
end

function with_raw(action)
    mktempdir() do temporary
        root = realpath(temporary); path = joinpath(root,"generated-not-a-CDF.bin")
        write(path,"GENERATED API-SPY BYTES; NOT A CDF; NO OBSERVATIONS\n")
        digest = bytes2hex(sha256(read(path)))
        action(path,digest)
    end
end
function pass(api,path,digest;memory_guard=phase -> nothing)
    R._time_pass(api,path;raw_id="generated-original-file",expected_raw_sha256=digest,
        start_ms=0,stop_ms=180000,memory_guard)
end

@testset "Exact integer Epoch and metadata byte literals" begin
    f = fixture()
    bounds = [62167219100000.0,62167220000000.0]
    actual = R._epoch_milliseconds(f.api,FixtureEpoch[FixtureEpoch(62167219199999.0),FixtureEpoch(62167219200000.0),FixtureEpoch(62167219200001.0)],-1000,1000,bounds)
    @test actual == [-1,0,1]
    for value in (62167219200000.5,NaN,Inf,-1e31,0.0,9007199254740994.0)
        @test_throws ArgumentError R._epoch_milliseconds(f.api,[FixtureEpoch(value)],-1000,1000,bounds)
    end
    @test_throws ArgumentError R._epoch_milliseconds(f.api,[FixtureEpoch(62167219200001.0),FixtureEpoch(62167219200000.0)],-1000,1000,bounds)
    @test_throws ArgumentError R._epoch_milliseconds(f.api,[FixtureEpoch(62167219200000.0),FixtureEpoch(62167219200000.0)],-1000,1000,bounds)
    @test R._expected_bytes(reinterpret(Float32,UInt32(0xf149f2ca)),false) == UInt8[0xca,0xf2,0x49,0xf1]
    @test R._expected_bytes(reinterpret(Float32,UInt32(0xf149f2ca)),true) == UInt8[0xf1,0x49,0xf2,0xca]
    @test R._expected_bytes(Int32(-2147483647),false) == UInt8[1,0,0,0x80]
    @test R._expected_bytes(Int32(-2147483647),true) == UInt8[0x80,0,0,1]
    # Pure spies never pass the public real-module identity check.
    @test_throws ArgumentError R.validate_reader_module(@__MODULE__)
end

@testset "Only Epoch is decoded in the Time pass; same-file value pass calls real QC" begin
    for big in (false,true)
        with_raw() do path,digest
            f = fixture(;big_endian=big); phases = Symbol[]
            result = pass(f.api,path,digest;memory_guard=phase -> push!(phases,phase))
            @test f.ds.reads == ["Epoch"]
            @test f.ds.opens == f.ds.closes == 1
            @test phases == [:admit,:opened,:finished]
            @test result.axis.times == [30000,90000,150000]
            @test result.axis.parameters.field == "BGSM" && result.axis.parameters.quality_variable == "NUM_PTS"
            @test result.axis.parameters.metadata_sha256 == result.metadata_sha256
            @test result.file.sha256 == digest
            schedule = G.freeze_phase_domains(result.axis)
            qc = R._qc_pass(f.api,path,result,schedule;memory_guard=phase -> nothing)
            @test f.ds.reads == ["Epoch","Epoch","label_bgsm","cartesian","BGSM","NUM_PTS"]
            @test f.ds.opens == f.ds.closes == 2
            @test qc.bz == [-2.0,0.0,-7.0]
            label = G.label_event(G.mo_event("generated",0,180000,1,2),[schedule],[qc])
            @test label.peak_bs_nt == 7 && label.valid_count == 3 && label.expected_count == 3
            @test label.extrema_supports == [(centre_ms=150000,support_start_ms=120000,support_stop_ms=180000)]
        end
    end
end

@testset "Every selected schema rejection precedes any variable data read" begin
    mutations = [
        f -> (f.ds.variables["BGSM"].attributes["UNITS"]="nT (3sec)"),
        f -> (f.ds.variables["BGSM"].attributes["DEPEND_0"]="Epoch3"),
        f -> (f.ds.variables["NUM_PTS"].attributes["DEPEND_0"]="Epoch3"),
        f -> (f.ds.variables["BGSM"].attributes["DEPEND_1"]="spherical"),
        f -> (f.ds.variables["BGSM"].attributes["LABL_PTR_1"]="label_bgse"),
        f -> (f.ds.variables["BGSM"].attributes["FILLVAL"]=[-1e31]),
        f -> (f.ds.variables["BGSM"].attributes["VALIDMIN"]=Float32[-65535,-65534,-65534]),
        f -> (f.ds.variables["NUM_PTS"].attributes["VALIDMAX"]=Int32[1501]),
        f -> (f.ds.variables["Epoch"].attributes["CATDESC"]="Time, Beginning, one minute"),
        f -> (f.ds.variables["Epoch"].attributes["MONOTON"]="FALSE"),
        f -> (f.ds.variables["BGSM"].storage=Float64),
        f -> (f.ds.variables["NUM_PTS"].storage=Float32),
        f -> (f.ds.variables["BGSM"].dims=(3,4)),
        f -> (f.ds.variables["NUM_PTS"].dims=(4,)),
        f -> (f.ds.variables["Epoch"].dims=(1,3)),
        f -> (f.ds.variables["BGSM"].record_varying=false),
        f -> (f.ds.variables["BGSM"].elements=2),
        f -> (f.ds.variables["BGSM"].pad=UInt8[0,0,0,0]),
        f -> (f.ds.variables["NUM_PTS"].pad=UInt8[0,0,0,0]),
        f -> (f.ds.variables["label_bgsm"].elements=8000000),
        f -> (f.ds.variables["cartesian"].record_varying=true),
    ]
    for mutation in mutations
        with_raw() do path,digest
            f = fixture(); mutation(f)
            @test_throws ArgumentError pass(f.api,path,digest)
            @test isempty(f.ds.reads) && f.ds.opens == f.ds.closes == 1
        end
    end
    for name in ("Epoch","BGSM","NUM_PTS")
        with_raw() do path,digest
            f = fixture(); f.ds.variables[name].sparse=2
            @test_throws ArgumentError pass(f.api,path,digest)
            @test isempty(f.ds.reads) && f.ds.closes==1
        end
    end
end

@testset "Phase, file, Time and component identity fail before BGSM" begin
    with_raw() do path,digest
        f = fixture(); p = pass(f.api,path,digest)
        schedule = G.freeze_phase_domains(p.axis)
        write(path,"changed source")
        @test_throws ArgumentError R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
        @test f.ds.reads == ["Epoch"] && f.ds.opens==1
    end
    with_raw() do path,digest
        f = fixture(); p = pass(f.api,path,digest); schedule = G.freeze_phase_domains(p.axis)
        f.ds.variables["Epoch"].values[2]=FixtureEpoch(62167219290001.0)
        @test_throws ArgumentError R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
        @test f.ds.reads == ["Epoch","Epoch"] && f.ds.closes==2
    end
    with_raw() do path,digest
        f = fixture(); p = pass(f.api,path,digest); schedule = G.freeze_phase_domains(p.axis)
        f.ds.variables["Epoch"].attributes["VALIDMAX"][1] += 60000
        @test_throws ArgumentError R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
        @test f.ds.reads == ["Epoch"] && f.ds.closes==2
    end
    with_raw() do path,digest
        f = fixture(); p = pass(f.api,path,digest)
        axis = G.time_phase_runs(p.axis.times,0,180000;raw_id=p.axis.raw_id,raw_sha256=digest,
            source_group=p.axis.parameters.source_group,metadata_sha256=repeat("b",64))
        forged = R.TimePass(p.file,p.metadata,p.metadata_sha256,axis)
        @test_throws ArgumentError R._qc_pass(f.api,path,forged,G.freeze_phase_domains(axis);memory_guard=phase -> nothing)
        @test f.ds.opens==1 && f.ds.reads==["Epoch"]
    end
    for name in ("label_bgsm","cartesian")
        with_raw() do path,digest
            f = fixture(); p = pass(f.api,path,digest); schedule = G.freeze_phase_domains(p.axis)
            reverse!(f.ds.variables[name].values;dims=1)
            @test_throws ArgumentError R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
            @test !("BGSM" in f.ds.reads) && !("NUM_PTS" in f.ds.reads) && f.ds.closes==2
        end
    end
    with_raw() do path,digest
        f = fixture(); f.ds.variables["Epoch"].values=FixtureEpoch[FixtureEpoch(62167219230000.0),FixtureEpoch(62167219290000.0),FixtureEpoch(62167219320000.0)]
        p = pass(f.api,path,digest); unresolved = G.freeze_phase_domains(p.axis)
        @test !unresolved.resolved
        @test_throws ArgumentError R._qc_pass(f.api,path,p,unresolved;memory_guard=phase -> nothing)
        @test f.ds.opens==1 && f.ds.reads==["Epoch"]
        domains = [G.PhaseDomain(0,120000,30000),G.PhaseDomain(120000,180000,0)]
        resolved = G.freeze_phase_domains(p.axis;domains)
        @test R._qc_pass(f.api,path,p,resolved;memory_guard=phase -> nothing).bz == [-2,0,-7]
    end
end

@testset "Constructor/read/close/resource failures retain honest cleanup" begin
    with_raw() do path,digest
        f = fixture(); f.ds.fail_open=true
        @test_throws ErrorException pass(f.api,path,digest)
        @test f.ds.opens==1 && f.ds.closes==0 && isempty(f.ds.reads)
    end
    with_raw() do path,digest
        f = fixture(); f.ds.fail_read="Epoch"
        @test_throws ErrorException pass(f.api,path,digest)
        @test f.ds.closes==1 && f.ds.reads==["Epoch"]
    end
    for name in ("BGSM","NUM_PTS")
        with_raw() do path,digest
            f = fixture(); p = pass(f.api,path,digest); schedule=G.freeze_phase_domains(p.axis)
            f.ds.fail_read=name
            @test_throws ErrorException R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
            @test f.ds.closes==2
        end
    end
    with_raw() do path,digest
        f = fixture(); f.ds.fail_close=true
        @test_throws ErrorException pass(f.api,path,digest)
        @test f.ds.closes==1
    end
    with_raw() do path,digest
        f = fixture(); f.ds.fail_read="Epoch"; f.ds.fail_close=true
        @test_throws CompositeException pass(f.api,path,digest)
        @test f.ds.closes==1
    end
    with_raw() do path,digest
        f = fixture()
        @test_throws ArgumentError pass(f.api,path,digest;memory_guard=phase -> throw(ArgumentError("generated unavailable RAM")))
        @test f.ds.opens==0 && isempty(f.ds.reads)
    end
    with_raw() do path,digest
        f = fixture()
        guard = phase -> phase === :opened && throw(ArgumentError("generated peak RSS violation"))
        @test_throws ArgumentError pass(f.api,path,digest;memory_guard=guard)
        @test f.ds.opens==f.ds.closes==1 && isempty(f.ds.reads)
    end
    with_raw() do path,digest
        f = fixture(); p=pass(f.api,path,digest); schedule=G.freeze_phase_domains(p.axis)
        f.ds.variables["NUM_PTS"].values=Int32[1,1]
        @test_throws ArgumentError R._qc_pass(f.api,path,p,schedule;memory_guard=phase -> nothing)
        @test f.ds.closes==2
    end
end

println("CME_CDF_GENERATED_API_SPIES_ONLY;REAL_PACKAGE_NOT_IMPORTED;GOLDEN_CDF_AND_NATIVE_GATES_PENDING")
end
