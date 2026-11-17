module ArchiveFormatHashTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers
const INPUT=JSON3.read(read(joinpath(@__DIR__,"fixtures/hash_vectors_v1.json"),String))
stream_hash(text::String)=P._sha_string(text)
@testset "Independent frozen SHA256 byte vectors" begin
    for case in INPUT.cases
        n=Int(case.bytes);pattern=hex2bytes(String(case.pattern_hex))
        data=repeat(pattern,cld(n,length(pattern)))[1:n]
        text=String(copy(data))
        @test bytes2hex(sha256(data))==case.expected_sha256
        @test stream_hash(text)==case.expected_sha256
        @test ncodeunits(text)==n&&collect(codeunits(text))==data
    end
end

end
