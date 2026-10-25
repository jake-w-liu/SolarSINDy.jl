module ArchiveFormatRawRowTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers
const O=JSON3.read(read(joinpath(@__DIR__,"fixtures/raw_rows_v1.json"),String),Any)
@testset "All raw source rows retain independent packed provenance" begin
    for case in O["cases"]
        parser=case["kind"]=="ground" ? P.parse_ground : P.parse_wind
        product=case["kind"]=="ground" ? "quasi-definitive" : "OMNI_HRO_1MIN"
        body=parser(case["body"];station=O["station"],product,start=DateTime(O["start"]),stop=DateTime(O["stop"]))
        @test body.raw_rows==case["raw_rows"]
        @test length(body)==case["unique_rows"]
        @test body.raw_storage isa String
        @test !ismutabletype(typeof(body))
        @test ncodeunits(body.raw_storage)==48*body.raw_rows
        @test bytes2hex(codeunits(body.raw_storage))==case["raw_ledger_hex"]
        @test body.raw_rows_sha256==case["raw_ledger_sha256"]
        @test bytes2hex(codeunits(body.storage))==case["selected_hex"]
        @test body.rows_sha256==case["selected_sha256"]
        @test body.body_sha256==bytes2hex(sha256(IOBuffer(case["body"])))
        for (index,expected) in enumerate(case["rows"])
            row=P.raw_row(body,index)
            @test row isa NamedTuple
            @test !ismutabletype(typeof(row))
            @test row.tag==P.EPOCH+Millisecond(expected["utc_ms"])
            @test row.source_line==expected["line"]
            @test row.valid_mask==expected["mask"]
            @test row.reason_bits==expected["reasons"]
            for field in 1:4
                actual=IOBuffer();write(actual,htol(reinterpret(UInt64,row.values[field])))
                @test bytes2hex(take!(actual))==expected["value_words"][field]
            end
        end
        for invalid in (true,false,0,-1,body.raw_rows+1,typemax(Int128))
            @test_throws ArgumentError P.raw_row(body,invalid)
        end
        @test_throws MethodError P.raw_row(body,1.0)
        if body.raw_rows>0
            @test P.raw_row(body,big(1))==P.raw_row(body,1)
        end
    end
    # Exercise the ordinary maximum raw-row count without multiplying a tuple
    # type by sample size. All duplicates retain their original line positions.
    line="2020-01-01T00:00:00.000Z,1,2,400,5\n"
    maximum_body=P.parse_wind(repeat(line,50_000);station="CMO",product="OMNI_HRO_1MIN",
        start=DateTime(O["start"]),stop=DateTime(O["stop"]))
    @test maximum_body.raw_rows==50_000
    @test length(maximum_body)==1
    @test ncodeunits(maximum_body.raw_storage)==2_400_000
    @test P.raw_row(maximum_body,50_000).source_line==50_000
    @test P.raw_row(maximum_body,50_000).values==(400.,2.,1.,5.)
    @test maximum_body.raw_rows_sha256==bytes2hex(sha256(IOBuffer(maximum_body.raw_storage)))
end

end
