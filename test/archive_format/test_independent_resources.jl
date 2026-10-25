module IndependentArchiveResourceTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const A=SolarSINDy.ArchiveFormatAudit
const P=SolarSINDy.ArchiveFormatParsers
const O=JSON3.read(read(joinpath(@__DIR__,"fixtures/independent_resources_v1.json"),String),Any)
const M=O["metadata"]
const KW=(;station=M["station"],product=M["product"],start=DateTime(M["start"]),stop=DateTime(M["stop"]))
A.audit_wind(O["body_line"];KW...)
records=Any[]
@testset "Independent reader intermediate ordinary resources" begin
    for oracle in O["records"]
        n=oracle["rows"];n>4096&&continue
        measured=@timed A.audit_wind(repeat(O["body_line"],n);KW...)
        body=measured.value
        @test body.record_count==n&&body.unique_count==1&&body.duplicate_count==n-1
        @test body.body_digest==oracle["body_sha256"]
        @test body.raw_digest==oracle["raw_digest"]
        @test body.selected_digest==oracle["selected_digest"]
        @test ncodeunits(body.raw_records)==48*n&&ncodeunits(body.selected_records)==48
        @test A.raw_record(body,n).source_line==oracle["last_source_line"]
        @test sizeof(A._AuditRow)==48
        @test measured.bytes<=8*1024*1024+4096*n
        push!(records,(;n,seconds=measured.time,allocated_bytes=measured.bytes,raw_bytes=ncodeunits(body.raw_records),selected_bytes=ncodeunits(body.selected_records)))
    end
end

records=Any[]
@testset "Independent reader frozen maximum per-body bounds" begin
    oracle=only(filter(c->c["rows"]==50_000,O["records"]))
    source=repeat(O["body_line"],50_000)
    measured=@timed A.audit_wind(source;KW...);body=measured.value
    @test body.record_count==50_000&&body.unique_count==1&&body.duplicate_count==49_999
    @test body.body_digest==oracle["body_sha256"]
    @test body.raw_digest==oracle["raw_digest"]
    @test body.selected_digest==oracle["selected_digest"]
    @test ncodeunits(body.raw_records)==2_400_000&&ncodeunits(body.selected_records)==48
    @test A.raw_record(body,50_000).source_line==50_000
    @test measured.bytes<=512*1024*1024
    push!(records,(;label="maximum_raw_rows",seconds=measured.time,allocated_bytes=measured.bytes,raw_rows=body.record_count))
    @test_throws ArgumentError A.audit_wind(source*O["body_line"];KW...)
    blank="\n"^(8*1024*1024)
    measured=@timed A.audit_wind(blank;KW...);body=measured.value
    @test body.record_count==0&&body.unique_count==0
    @test body.body_digest==O["maximum_blank_sha256"]
    @test body.raw_digest==body.selected_digest==O["empty_sha256"]
    @test measured.bytes<=512*1024*1024
    push!(records,(;label="maximum_blank_body",seconds=measured.time,allocated_bytes=measured.bytes,raw_rows=body.record_count))
    @test_throws ArgumentError A.audit_wind(blank*"\n";KW...)
    @test A.audit_wind(" "^4096*"\n"*O["body_line"];KW...).record_count==1
    @test_throws ArgumentError A.audit_wind(" "^4097*"\n"*O["body_line"];KW...)
    @test A.audit_wind(replace(O["body_line"],",400,"=>","*("0."*"0"^126)*",");KW...).record_count==1
    @test_throws ArgumentError A.audit_wind(replace(O["body_line"],",400,"=>","*("0."*"0"^127)*",");KW...)
    @test_throws ArgumentError A.audit_wind(O["body_line"];KW...,start=KW.start+Millisecond(1))
    @test_throws ArgumentError A.audit_wind(O["body_line"];KW...,stop=KW.start+Day(33))
end

end
