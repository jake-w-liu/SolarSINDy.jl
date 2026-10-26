module ArchiveCatalogMaximumTests
using Dates,Printf,SHA,JSON3,Test,SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers;const A=SolarSINDy.ArchiveFormatAudit
const C=SolarSINDy.ArchiveSourceCatalog;const I=SolarSINDy.ArchiveCatalogAudit
include(joinpath(@__DIR__,"windows","maximum_support.jl"))
include(joinpath(@__DIR__,"lookup_resource_support.jl"))
function maximum_inputs()
    firstday=DateTime(2020);primary=P.ParsedBody[];independent=A.AuditBody[]
    frozen=JSON3.read(read(joinpath(@__DIR__,"fixtures","catalog_v1.json"),String),Any)
    header=join(split(first(frozen["bodies"])["text"],'\n')[1:7],'\n')*"\n"
    for piece in 0:31
        start=firstday+Day(31*piece);stop=start+Day(31);io=IOBuffer();write(io,header)
        for row in 0:49_999
            tag=start+Minute(min(row,44_640))
            @printf(io,"%s %s %03d 1 2 0 0\n",Dates.format(Date(tag),dateformat"yyyy-mm-dd"),Dates.format(Time(tag),dateformat"HH:MM:SS.sss"),dayofyear(tag))
        end
        body=String(take!(io));@assert ncodeunits(body)<=8*1024*1024
        kw=(;station="CMO",product="quasi-definitive",start,stop)
        push!(primary,P.parse_ground(body;kw...));push!(independent,A.audit_ground(body;kw...))
        @assert primary[end].raw_rows==independent[end].record_count==50_000
        @assert primary[end].raw_storage==independent[end].raw_records
        println("GENERATED_MAXIMUM_BODY_VERIFIED ",piece+1);flush(stdout)
    end
    primary,independent
end
function measured_catalog(f,inputs)
    warm=f(inputs);@assert length(warm)==1_428_481
    warm=nothing;GC.gc();value=nothing
    allocated=@allocated value=f(inputs)
    allocated,value
end
function run_maximum_test()
primary,independent=maximum_inputs()
allocated_primary,p=measured_catalog(C.build_catalog,primary)
allocated_independent,a=measured_catalog(I.audit_catalog,independent)
@testset "Frozen full 32-body 1.6-million-raw-row catalog geometry" begin
    for c in (p,a)
        @test length(c.bodies)==32
        @test c.raw_rows==1_600_000
        @test length(c)==32*31*1440+1==1_428_481
        @test c.duplicates==171_519
        @test c.complete_vectors==1_428_481
        @test c.declared_minutes==1_428_481
        @test c.missing_minutes==0
        @test c.uncovered_minutes==0
        @test ncodeunits(c.selected_storage)==114_278_480
    end
    @test allocated_primary<=4*1024^3
    @test allocated_independent<=4*1024^3
    @test p.selected_storage==a.selected_storage
    @test p.selected_sha256==a.selected_sha256
    @test p.raw_catalog_sha256==a.raw_catalog_sha256
    @test sum(ncodeunits(b.raw_storage) for b in primary)==76_800_000
    @test sum(ncodeunits(b.raw_records) for b in independent)==76_800_000
    for index in unique([1,2,44_641,44_642,714_241,1_428_480,1_428_481,collect(1:14_284:1_428_481)...])
        row=C.witness(p,index);audit=I.audit_witness(a,index)
        @test row.tag==DateTime(2020)+Minute(index-1)
        @test row.valid_mask==3&&row.reason_bits==0
        @test isequal(row.values,(1.0,2.0,0.0,0.0))
        @test isequal(row,audit)
        bi=findfirst(==(row.body_id),p.body_ids);@test bi!==nothing
        raw=P.raw_row(p.bodies[bi],Int(row.source_line)-7)
        @test raw.tag==row.tag&&raw.source_line==row.source_line&&isequal(raw.values,row.values)
    end
    @test C.witness(p,length(p)).source_line==44_648
    @test_throws ArgumentError C.build_catalog([primary;first(primary)])
    @test_throws ArgumentError I.audit_catalog([independent;first(independent)])
end
check_completed_window_maximum(p,a)
check_catalog_lookup_resource(a)
record=(;complete=true,runtime=string(VERSION),body_count=32,raw_rows=1_600_000,selected_rows=length(p),selected_bytes=ncodeunits(p.selected_storage),primary_cumulative_allocated_bytes=allocated_primary,independent_cumulative_allocated_bytes=allocated_independent,unchanged_allocation_ceiling_bytes=4*1024^3,selected_digest=p.selected_sha256,raw_catalog_digest=p.raw_catalog_sha256,actual_data=false,actual_fits=0)
return record
end
println(JSON3.write(run_maximum_test()))
end
