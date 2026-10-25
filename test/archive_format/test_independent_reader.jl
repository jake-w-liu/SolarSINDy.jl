module IndependentArchiveReaderTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const A=SolarSINDy.ArchiveFormatAudit
const P=SolarSINDy.ArchiveFormatParsers
const FIXTURES=joinpath(@__DIR__,"fixtures")
word(value)=string(reinterpret(UInt64,value);base=16,pad=16)
function parse_case(c, independent=true)
    m=c["metadata"];which=c["kind"]=="ground"
    f=independent ? (which ? A.audit_ground : A.audit_wind) : (which ? P.parse_ground : P.parse_wind)
    f(c["body"];station=m["station"],product=m["product"],start=DateTime(m["start"]),stop=DateTime(m["stop"]))
end
function compare_all(a,p)
    @test a.kind==p.kind && a.station==p.station && a.product==p.product
    @test a.start==p.start && a.stop==p.stop
    @test a.body_digest==p.body_sha256
    @test a.selected_digest==p.rows_sha256
    @test a.raw_digest==p.raw_rows_sha256
    @test a.selected_records==p.storage
    @test a.raw_records==p.raw_storage
    @test a.record_count==p.raw_rows
    @test a.unique_count==p.rows
    @test a.duplicate_count==p.duplicates
    @test a.complete_count==p.complete_vectors
    @test a.absent_minutes==p.missing_minutes
    @test a.classification_counts==p.reason_counts
    for i in 1:a.unique_count
        @test isequal(A.record(a,i),P.row(p,i))
        @test isequal(A.complete_components(a,i),P.complete_values(p,i))
    end
    for i in 1:a.record_count;@test isequal(A.raw_record(a,i),P.raw_row(p,i));end
end
const FORMAT=JSON3.read(read(joinpath(FIXTURES,"format_v1.json"),String),Any)
@testset "Independent reader: frozen format expectations" begin
    for c in FORMAT["cases"]
        @testset "$(c["name"])" begin
            accepted=true;result=nothing
            try;result=parse_case(c);catch error;error isa ArgumentError||rethrow();accepted=false;end
            @test accepted==c["accept"]
            if c["accept"] && accepted
                @test result.body_digest==c["body_sha256"]
                @test result.unique_count==length(c["expected_rows"])
                for (index,expected) in enumerate(c["expected_rows"])
                    row=A.record(result,index)
                    @test Dates.value(row.tag-DateTime(1970))==expected["time"]
                    @test word.(row.values)==Tuple(expected["values_f64_hex"])
                    @test row.valid_mask==expected["valid_mask"]
                    @test row.reason_bits==expected["reason_bits"]
                    @test (A.complete_components(result,index)!==nothing)==(expected["valid_mask"]==(c["kind"]=="ground" ? 3 : 15))
                end
                @test_throws ArgumentError A.record(result,true)
                @test_throws ArgumentError A.record(result,0)
                @test_throws ArgumentError A.record(result,result.unique_count+1)
                compare_all(result,parse_case(c,false))
            end
        end
    end
end
const RAW=JSON3.read(read(joinpath(FIXTURES,"raw_rows_v1.json"),String),Any)
@testset "Independent reader: exact raw and selected binary literals" begin
    for c in RAW["cases"]
        f=c["kind"]=="ground" ? A.audit_ground : A.audit_wind
        product=c["kind"]=="ground" ? "quasi-definitive" : "OMNI_HRO_1MIN"
        kwargs=(;station=RAW["station"],product,start=DateTime(RAW["start"]),stop=DateTime(RAW["stop"]))
        body=f(c["body"];kwargs...)
        @test bytes2hex(codeunits(body.raw_records))==c["raw_ledger_hex"]
        @test bytes2hex(codeunits(body.selected_records))==c["selected_hex"]
        @test body.raw_digest==c["raw_ledger_sha256"]
        @test body.selected_digest==c["selected_sha256"]
        @test body.record_count==c["raw_rows"] && body.unique_count==c["unique_rows"]
        @test !ismutabletype(typeof(body))
        for (index,expected) in enumerate(c["rows"])
            row=A.raw_record(body,index)
            @test row.source_line==expected["line"]
            @test row.valid_mask==expected["mask"] && row.reason_bits==expected["reasons"]
            @test row.tag==DateTime(1970)+Millisecond(expected["utc_ms"])
            for j in 1:4
                io=IOBuffer();write(io,htol(reinterpret(UInt64,row.values[j])))
                @test bytes2hex(take!(io))==expected["value_words"][j]
            end
        end
        for invalid in (true,false,0,-1,body.record_count+1,typemax(Int128));@test_throws ArgumentError A.raw_record(body,invalid);end
        @test_throws MethodError A.raw_record(body,1.0)
        prod=(c["kind"]=="ground" ? P.parse_ground : P.parse_wind)(c["body"];kwargs...)
        compare_all(body,prod)
    end
end
const HIDDEN=JSON3.read(read(joinpath(FIXTURES,"hidden_duplicates_v1.json"),String),Any)
@testset "Independent reader: all raw duplicate witnesses" begin
    for c in HIDDEN["cases"]
        if occursin("positive",c["name"])
            body=parse_case(c);@test body.unique_count==1 && A.complete_components(body,1)===nothing
            compare_all(body,parse_case(c,false))
        else
            @test_throws ArgumentError parse_case(c)
        end
    end
end

end
