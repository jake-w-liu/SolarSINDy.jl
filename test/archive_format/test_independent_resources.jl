module IndependentArchiveResourceTests
using Test, Dates, JSON3, SolarSINDy
const A=SolarSINDy.ArchiveFormatAudit
const O=JSON3.read(read(joinpath(@__DIR__,"fixtures/independent_resources_v1.json"),String),Any)
const M=O["metadata"]
const KW=(;station=M["station"],product=M["product"],start=DateTime(M["start"]),stop=DateTime(M["stop"]))
@testset "Independent reader grammar and time boundaries" begin
    @test A.audit_wind(" "^4096*"\n"*O["body_line"];KW...).record_count==1
    @test_throws ArgumentError A.audit_wind(" "^4097*"\n"*O["body_line"];KW...)
    @test A.audit_wind(replace(O["body_line"],",400,"=>","*("0."*"0"^126)*",");KW...).record_count==1
    @test_throws ArgumentError A.audit_wind(replace(O["body_line"],",400,"=>","*("0."*"0"^127)*",");KW...)
    @test_throws ArgumentError A.audit_wind(O["body_line"];KW...,start=KW.start+Millisecond(1))
    @test_throws ArgumentError A.audit_wind(O["body_line"];KW...,stop=KW.start+Day(33))
end

end
