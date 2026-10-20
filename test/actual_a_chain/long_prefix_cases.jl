using Test, Dates, SHA, TOML, JSON3, LinearAlgebra, SolarSINDy
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","..","tools","actual_a_chain","ActualAChain.jl"))
const C=ActualAChain
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
const S=F.DstIntervalStream
const T=F.DstIntervalTraining
const X=C.InitialFits
length(ARGS)==1 || error("NEW_OWNED_OUTPUT")
const OUT=abspath(only(ARGS))
isdir(dirname(OUT)) && !ispath(OUT) || error("new output under an existing parent required")
mkdir(OUT)
const ORACLE=joinpath(@__DIR__,"frozen","long_prefix_oracle_v1.json")
@test bytes2hex(sha256(read(ORACLE)))=="1c10d7c1dcd6229cc4127a5a16314aeb3b7fac8f9ee73403d9f5017e298fe956"
const O=JSON3.read(read(ORACLE),Dict{String,Any})
const N=30000
const START=DateTime(2019)
const BOUNDARY=START+Hour(28000)
const VALUES=zeros(17,N)
for i in 1:N
    VALUES[6,i]=-Float64(1+i%7)
    VALUES[8,i]=Float64(380+i%17)
    VALUES[12,i]=Float64(1+i%5)
    VALUES[15,i]=i in (5000,17000,25000) ? NaN : -Float64(20+i%83)
    i in (10000,20000) && (VALUES[8,i]=NaN)
end
const RAW_SHA=bytes2hex(sha256(reinterpret(UInt8,vec(VALUES))))
@test RAW_SHA==O["raw_sha256"]
const INPUT=S.HourlyInputs([START+Hour(i-1) for i in 1:N],VALUES;source_sha256=RAW_SHA)
indices_digest(rows)=bytes2hex(sha256(reinterpret(UInt8,htol.(Int64.(rows)))))
function native_handles()
    if Sys.iswindows()
        process=ccall((:GetCurrentProcess,"kernel32"),Ptr{Cvoid},())
        value=Ref{UInt32}(0)
        ccall((:GetProcessHandleCount,"kernel32"),Int32,(Ptr{Cvoid},Ref{UInt32}),process,value)!=0 || error("native handle query failed")
        return Int(value[])
    end
    for root in ("/proc/self/fd","/dev/fd")
        isdir(root) && return length(readdir(root))
    end
    nothing
end
function run_chunk(chunk)
    directory=joinpath(OUT,"chunk-"*string(chunk));mkdir(directory)
    first_setting,second_setting=A.settings()[1:2]
    joint=Int[];available=Int[];storms=0
    second_export=Ref{Any}();streamed=Ref{Any}()
    first_export=T.export_fit_input(joinpath(directory,"fit-1"),BOUNDARY,first_setting;
            delay_hours=0,source_sha256=RAW_SHA,max_source_rows=N,max_chunk_rows=chunk) do first_sink
        second_export[]=T.export_fit_input(joinpath(directory,"fit-2"),BOUNDARY,second_setting;
                delay_hours=0,source_sha256=RAW_SHA,max_source_rows=N,max_chunk_rows=chunk) do second_sink
            streamed[]=S.stream_basis(INPUT;delay_hours=0,taus=first_setting.taus,sigma=first_setting.sigma,
                chunk_hours=chunk,sink=rows->begin
                    first_sink(rows);second_sink(rows)
                    for row in rows
                        all(row.scoreable) && all(A.fit_eligible(row.issue,row.target_complete[j],row.target_receipt[j],BOUNDARY) for j in 1:2) || continue
                        index=Int(Dates.value(row.issue-START)÷3_600_000)
                        push!(joint,index)
                        row.model_available && push!(available,index)
                        storms+=count(<=( -50),row.target)
                    end
                end)
        end
    end
    @test length(joint)==O["joint_count"]
    @test length(available)==O["available_joint_count"]
    @test length(joint)-length(available)==O["fallback_joint_count"]
    @test storms==O["storm_target_count"]
    @test indices_digest(joint)==O["joint_indices_int64le_sha256"]
    @test indices_digest(available)==O["available_joint_indices_int64le_sha256"]
    @test streamed[].row_count==N
    @test streamed[].max_buffered_rows==min(chunk,N)
    models=Any[]
    for exported in (first_export,second_export[])
        model=T.fit_gains_from_input(exported.directory;expected_manifest_sha256=exported.manifest_sha256)
        manifest=TOML.parsefile(joinpath(exported.directory,"manifest.toml"))
        @test model.fit_count==O["joint_count"]
        @test model.informative_count==O["available_joint_count"]
        @test model.storm_count==O["storm_target_count"]
        @test manifest["data_bytes"]==64O["joint_count"]
        @test X.verify_fit_certificate(model,manifest,joinpath(exported.directory,"joint-fit.f64"))
        push!(models,model)
    end
    (models=Tuple(models),checkpoint=C.InitialFits.B._checkpoint_data(streamed[].checkpoint))
end
records=Any[];baseline=nothing
@testset "Long generated stream preserves full joint records across extreme chunk sizes" begin
    for chunk in (1,4096)
        GC.gc();before=native_handles()
        measured=@timed run_chunk(chunk)
        result=measured.value
        if baseline===nothing
            global baseline=result
        else
            @test isequal(result,baseline)
        end
        GC.gc();after=native_handles()
        record=(chunk=chunk,rows=N,joint_origins=O["joint_count"],fallback_joint_origins=O["fallback_joint_count"],
            model_count=2,elapsed_seconds=measured.time,allocated_bytes=measured.bytes,gc_seconds=measured.gctime,
            maximum_process_rss_bytes=Sys.maxrss(),native_handles_before=before,native_handles_after=after,
            output_bytes=C.InitialFits.stage_inventory(joinpath(OUT,"chunk-"*string(chunk))).bytes,
            model_ids=[model.model_id for model in result.models])
        push!(records,record);println(JSON3.write(record));flush(stdout)
    end
end
report=(complete=true,generated_only=true,actual_data=false,new_actual_fitting=false,scientific_verdict=false,
    component_scope="First physical setting, two eta fits, raw stream and disk certificate closure; complete audited-reference 18-model stage is checked separately.",
    independent_oracle_sha256=bytes2hex(sha256(read(ORACLE))),measurements=records)
C.InitialFits.B._write_new(joinpath(OUT,"long-prefix-summary.json"),Vector{UInt8}(codeunits(JSON3.write(report))))
