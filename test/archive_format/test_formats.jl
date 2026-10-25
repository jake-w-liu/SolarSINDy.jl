module ArchiveFormatLiteralTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers
const INPUT=JSON3.read(read(joinpath(@__DIR__,"fixtures/format_v1.json"),String),Any)
hex(v)=string(reinterpret(UInt64,v);base=16,pad=16)
function parse_case(c;body=c["body"])
    m=c["metadata"];f=c["kind"]=="ground" ? P.parse_ground : P.parse_wind
    f(body;station=m["station"],product=m["product"],start=DateTime(m["start"]),stop=DateTime(m["stop"]))
end
@testset "Frozen independent generated raw-format cases" begin
    for c in INPUT["cases"]
        @testset "$(c["name"])" begin
            if !c["accept"]
                @test_throws ArgumentError parse_case(c)
                continue
            end
            b=parse_case(c);expected=c["expected_rows"]
            @test b.body_sha256==c["body_sha256"]
            @test length(b)==length(expected)
            @test ncodeunits(b.storage)==48length(b)
            @test b.rows_sha256==bytes2hex(sha256(codeunits(b.storage)))
            @test b.raw_rows>=length(b)&&b.duplicates==b.raw_rows-length(b)
            minutes=div(Dates.value(b.stop-b.start),60_000)+(b.kind==:ground ? 1 : 0)
            @test b.missing_minutes==minutes-length(b)
            for (i,e) in enumerate(expected)
                r=P.row(b,i)
                @test Dates.value(r.tag-DateTime(1970))==e["time"]
                @test hex.(r.values)==Tuple(e["values_f64_hex"])
                @test r.valid_mask==e["valid_mask"]
                @test r.reason_bits==e["reason_bits"]
                @test r.source_line>0
                complete=e["valid_mask"]==(b.kind==:ground ? 3 : 15)
                @test (P.complete_values(b,i)!==nothing)==complete
            end
            @test_throws ArgumentError P.row(b,false)
            @test_throws ArgumentError P.row(b,0)
            @test_throws ArgumentError P.row(b,length(b)+1)
        end
    end
end
@testset "Independent bounds, raw identity and immutable projection" begin
    ground=first(INPUT["cases"]);wind=only(filter(c->c["name"]=="wind_normal",INPUT["cases"]))
    for c in (ground,wind)
        a=parse_case(c);b=parse_case(c;body=replace(c["body"],"\n"=>"\r\n"))
        @test a.rows_sha256==b.rows_sha256
        @test a.body_sha256!=b.body_sha256
        @test_throws ErrorException setfield!(a,:rows,2)
        @test P.complete_values(a,1) isa Tuple
    end
    m=wind["metadata"];kwargs=(;station=m["station"],product=m["product"],start=DateTime(m["start"]),stop=DateTime(m["stop"]))
    w=wind["body"]
    @test P.parse_wind(" "^4096*"\n"*w;kwargs...).raw_rows==1
    @test_throws ArgumentError P.parse_wind(" "^4097*"\n"*w;kwargs...)
    @test P.parse_wind(replace(w,",400,"=>","*("0."*"0"^126)*",");kwargs...).raw_rows==1
    @test_throws ArgumentError P.parse_wind(replace(w,",400,"=>","*("0."*"0"^127)*",");kwargs...)
    @test_throws ArgumentError P.parse_wind(" "^(8*1024*1024+1);kwargs...)
    @test P.parse_wind("\n"^(8*1024*1024);kwargs...).raw_rows==0
    a=P.parse_wind(w^50_000;kwargs...)
    @test a.raw_rows==50_000&&a.duplicates==49_999&&length(a)==1
    @test_throws ArgumentError P.parse_wind(w^50_001;kwargs...)
    @test_throws ArgumentError P.parse_wind(w;kwargs...,start=DateTime(2017)+Millisecond(1))
    @test_throws ArgumentError P.parse_wind(w;kwargs...,stop=DateTime(2017)+Day(33))
end

end
