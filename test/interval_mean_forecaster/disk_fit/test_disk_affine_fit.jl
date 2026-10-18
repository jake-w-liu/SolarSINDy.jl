using Test, Dates, SHA, TOML
include(joinpath(@__DIR__,"closed_file_oracle_v2.jl"))
include(ARGS[1])
include(joinpath(@__DIR__,"DiskAffineGainFit.jl"))
const F=IntervalMeanForecaster
const T=F.DstIntervalTraining
const A=T.A
const D=DiskAffineGainFit
const SOURCE=repeat("a",64)
const START=DateTime(2019,1,1)
const BOUNDARY=START+Hour(1000)
const ROOT=ARGS[2]
mkdir(ROOT)
owned_file_handle_oracle_controls(joinpath(ROOT,"handle-oracle-controls"))
const counter=Ref(0)
function owned()
    counter[]+=1
    joinpath(ROOT,"case-"*string(counter[]))
end

function row(issue,setting,delay,z,residual;available=true,missing_first=false)
    clocks=Tuple(A.forecast_clock(issue,h,delay) for h in (4,6))
    labels=missing_first ? (NaN,-40.) : (-60.,-40.)
    anchor=-55.
    offsets=available ? Tuple(labels[j]-residual[j] for j in 1:2) : (anchor,anchor)
    !available && (labels=(-50.,-50.))
    (;issue,driver_tag=clocks[1].driver_tag,driver_receipt=issue,
      anchor_tag=clocks[1].anchor_tag,anchor_complete=clocks[1].anchor_complete,
      anchor_receipt=clocks[1].anchor_receipt,anchor,
      target_tag=Tuple(c.target_tag for c in clocks),
      target_complete=Tuple(c.target_complete for c in clocks),
      target_receipt=Tuple(c.target_receipt for c in clocks),target=labels,
      scoreable=(isfinite(labels[1]),true),model_available=available,
      offset=offsets,coefficients=z,taus=setting.taus,sigma=setting.sigma,
      source_sha256=SOURCE,dst_delay_hours=delay)
end

function fixture(setting,delay;kind=:interior,fallbacks=10,missing=false,maturity=false)
    rows=NamedTuple[]
    for i in 1:180
        if kind==:interior
            designs=((1.,0.),(0.,1.),(1.,1.))
            z=(designs[mod1(i,3)],designs[mod1(i+1,3)])
            residual=Tuple(2v[1]+3v[2] for v in z)
        elseif kind==:rank_one
            z=((1.,2.),(1.,2.));residual=(5.,5.)
        elseif kind==:axis
            z=((1.,0.),(0.,1.));residual=(-2.,3.)
        else
            z=((0.,0.),(0.,0.));residual=(0.,0.)
        end
        push!(rows,row(START+Hour(i-1),setting,delay,z,residual))
    end
    for i in 1:fallbacks
        push!(rows,row(START+Hour(179+i),setting,delay,((0.,0.),(0.,0.)),(5.,5.);available=false))
    end
    if missing
        # Unavailable driver/target origin is excluded, rather than imputed.
        push!(rows,row(START+Hour(220),setting,delay,((0.,0.),(0.,0.)),(0.,0.);available=false,missing_first=true))
        rows[end]=merge(rows[end],(target=(NaN,-50.),scoreable=(false,true)))
    end
    if maturity
        equality=BOUNDARY-Hour(168+6+delay)
        push!(rows,row(equality,setting,delay,((1.,0.),(0.,1.)),(2.,3.)))
        push!(rows,row(equality+Hour(1),setting,delay,((1.,0.),(0.,1.)),(2000.,3000.)))
    end
    rows
end

function export_rows(rows,setting,delay;chunk=24,boundary=BOUNDARY,kwargs...)
    destination=owned()
    receipt=D.export_fit_input(destination,boundary,setting;delay_hours=delay,source_sha256=SOURCE,kwargs...) do sink
        for i in 1:chunk:length(rows)
            sink(rows[i:min(i+chunk-1,end)])
        end
    end
    receipt
end
solve(receipt)=D.solve_fit_input(receipt.directory;expected_manifest_sha256=receipt.manifest_sha256)
manifest(receipt)=TOML.parsefile(joinpath(receipt.directory,"manifest.toml"))
function change_manifest(receipt,change)
    m=manifest(receipt);change(m)
    path=joinpath(receipt.directory,"manifest.toml")
    open(io->TOML.print(io,m;sorted=true),path,"w")
    merge(receipt,(manifest_sha256=bytes2hex(sha256(read(path))),))
end
function copied(receipt)
    destination=owned();cp(receipt.directory,destination)
    merge(receipt,(directory=destination,))
end
function rejected_producer(produce;setting=A.settings()[1],delay=0,kwargs...)
    destination=owned()
    @test_throws Exception D.export_fit_input(produce,destination,BOUNDARY,setting;
        delay_hours=delay,source_sha256=SOURCE,kwargs...)
    @test !isfile(joinpath(destination,"manifest.toml"))
    if isfile(joinpath(destination,"joint-fit.f64"))
        # The independent controls above distinguish held and closed handles.
        @test owned_file_handles_closed(joinpath(destination,"joint-fit.f64"))
    end
    destination
end

@testset "bounded disk fit independent contract" begin
    @testset "literal gains, exact disk order, original solver regression" begin
        for delay in (0,1), index in (1,2)
            setting=A.settings()[index]
            rows=fixture(setting,delay;missing=true,maturity=true)
            reference=T.fit_gains(rows,BOUNDARY,setting;delay_hours=delay,source_sha256=SOURCE)
            before=solve(export_rows(rows,setting,delay;boundary=BOUNDARY-Millisecond(1)))
            @test before.fit_count==190
            @test before.informative_count==180
            @test all(isapprox.(before.gains,(2.,3.);atol=1e-12,rtol=0))
            equality=BOUNDARY-Hour(168+6+delay)
            @test A.fit_eligible(equality,equality+Hour(6),equality+Hour(6+delay),BOUNDARY)
            @test !A.fit_eligible(equality,equality+Hour(6),equality+Hour(6+delay)+Millisecond(1),BOUNDARY)
            previous=nothing
            for chunk in (1,7,24,4096)
                receipt=export_rows(rows,setting,delay;chunk)
                result=solve(receipt);m=manifest(receipt)
                @test all(isapprox.(result.gains,(2.,3.);atol=1e-12,rtol=0))
                @test result.gains==reference.gains
                @test result.solver==reference.solver
                for field in (:fit_count,:informative_count,:storm_count,:fit_mask_sha256,
                              :fit_data_sha256,:max_target_complete,:max_label_receipt,:max_fit_issue)
                    @test getproperty(result,field)==getproperty(reference,field)
                end
                @test result.fit_count==191
                @test result.informative_count==181
                @test m["total_source_rows"]==193
                @test result.max_label_receipt==BOUNDARY-Hour(168)
                @test m["data_bytes"]==64*191
                # The first origin has design(1,0)/residual2/label-60 then
                # design(0,1)/residual3/label-40. Read directly, no exporter helper.
                first_eight=open(joinpath(receipt.directory,"joint-fit.f64"),"r") do io
                    [reinterpret(Float64,ltoh(read(io,UInt64))) for _ in 1:8]
                end
                @test first_eight==[1.,0.,2.,-60.,0.,1.,3.,-40.]
                if previous!==nothing
                    @test read(joinpath(receipt.directory,"joint-fit.f64"))==read(joinpath(previous.directory,"joint-fit.f64"))
                    @test m["fit_mask_sha256"]==manifest(previous)["fit_mask_sha256"]
                    @test m["fit_data_sha256"]==manifest(previous)["fit_data_sha256"]
                end
                previous=receipt
            end
        end
    end
    @testset "rank and nonnegative boundary literal oracles" begin
        setting=A.settings()[1]
        for (kind,expected) in ((:rank_one,(1.,2.)),(:axis,(0.,3.)))
            result=solve(export_rows(fixture(setting,0;kind,fallbacks=0),setting,0))
            @test all(isapprox.(result.gains,expected;atol=1e-12,rtol=0))
            kind==:rank_one && @test result.solver.rank==1
        end
        rows=fixture(setting,0;kind=:zero,fallbacks=0)
        rejected_producer(sink->sink(rows))
        rejected_producer(sink->sink(fixture(setting,0;fallbacks=0)[1:99]))
    end
    @testset "producer, order, clock, identity and bounded input failures" begin
        setting=A.settings()[1];rows=fixture(setting,0)
        rejected_producer(sink->sink(NamedTuple[]))
        rejected_producer(sink->sink(reverse(rows)))
        rejected_producer(sink->begin sink(rows[1:100]);sink(rows[100:110]) end)
        rejected_producer(sink->begin sink(rows[1:100]);sink(rows[99:110]) end)
        rejected_producer(sink->sink([merge(rows[1],(source_sha256=repeat("b",64),))]))
        rejected_producer(sink->sink([merge(rows[1],(taus=(1.,2.),))]))
        rejected_producer(sink->sink([merge(rows[1],(target_receipt=(rows[1].target_receipt[1]+Millisecond(1),rows[1].target_receipt[2]),))]))
        rejected_producer(sink->sink([merge(rows[1],(coefficients=((Inf,0.),(0.,1.)),))]))
        rejected_producer(sink->sink(rows);max_chunk_rows=2)
        rejected_producer(sink->begin sink(rows[1:100]);sink(rows[101:end]) end;max_source_rows=150)
        rejected_producer(sink->begin sink(rows[1:100]);error("injected producer failure") end)
        destination=owned();mkdir(destination)
        sentinel=joinpath(destination,"foreign.txt");write(sentinel,"preserve")
        @test_throws Exception D.export_fit_input(sink->sink(rows),destination,BOUNDARY,setting;delay_hours=0,source_sha256=SOURCE)
        @test read(sentinel,String)=="preserve"
        @test !isfile(joinpath(destination,"manifest.toml"))
        @test_throws ArgumentError D.export_fit_input(sink->sink(rows),owned(),BOUNDARY,setting;delay_hours=false,source_sha256=SOURCE)
        @test_throws ArgumentError D.export_fit_input(sink->sink(rows),owned(),BOUNDARY,setting;delay_hours=0,source_sha256=SOURCE*"\n")
    end
    @testset "pinned bytes and rebound semantic corruption" begin
        setting=A.settings()[2]
        good=export_rows(fixture(setting,0),setting,0)
        @test_throws ArgumentError D.solve_fit_input(good.directory;expected_manifest_sha256=repeat("0",64))
        @test_throws ArgumentError D.solve_fit_input(good.directory;expected_manifest_sha256=good.manifest_sha256*"\n")
        for transform in (bytes->bytes[1:end-1],bytes->vcat(bytes,0x00),bytes->begin bytes[1]⊻=0x01;bytes end)
            bad=copied(good);path=joinpath(bad.directory,"joint-fit.f64")
            write(path,transform(read(path)))
            @test_throws ArgumentError solve(bad)
        end
        for change in (m->(m["fit_rows"]-=2),m->(m["storm_rows"]+=1),
                       m->(m["informative_origins"]-=1),m->(m["max_label_receipt_ms"]=string(parse(Int64,m["max_label_receipt_ms"])+1)),
                       m->(m["source_hashes"]["DiskAffineGainFit.jl"]=repeat("0",64)),
                       m->(m["data_sha256"]*="\n"),m->(m["unexpected"]=true),
                       m->(m["setting_index"]=true),m->(m["joint_origins"]=true))
            bad=change_manifest(copied(good),change)
            @test_throws ArgumentError solve(bad)
        end
        bad=copied(good);path=joinpath(bad.directory,"joint-fit.f64")
        bytes=read(path);bytes[1:8]=reinterpret(UInt8,[htol(reinterpret(UInt64,Inf))]);write(path,bytes)
        bad=change_manifest(bad,m->(m["data_sha256"]=bytes2hex(sha256(bytes))))
        @test_throws ArgumentError solve(bad)
    end
end
