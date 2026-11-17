module IndependentArchiveLineTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const A=SolarSINDy.ArchiveFormatAudit
const P=SolarSINDy.ArchiveFormatParsers
@testset "Independent literal line boundaries" begin
    for (text,want) in (("",String[]),("\n",[""]),("a\n",["a"]),
            ("\r",["\r"]),("a\rb",["a\rb"]),("a\r\n",["a"]),
            ("\r\n\n",["",""]),("α\r\n雪\n\n終",["α","雪","","終"]),
            ("a\n\r",["a","\r"]),("a\r\r\n",["a\r"]))
        @test collect(A._AuditLines(text))==want
    end
end
@testset "Independent Base line compatibility" begin
    rng=MersenneTwister(947421)
    alphabet=["a","α","雪","\r","\n"," ","#"]
    for _ in 1:2000
        text=join(rand(rng,alphabet,rand(rng,0:200)))
        @test collect(A._AuditLines(text))==collect(eachline(IOBuffer(text)))
    end
end
println("LINE_ITERATOR_LITERAL_AND_BASE_ORACLES_PASS")


end
