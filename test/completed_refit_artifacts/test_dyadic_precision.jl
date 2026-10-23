@testset "operand precision is preserved by the diagnostic codec" begin
    path=joinpath(@__DIR__,"fixtures","exact_dyadic_literals_v2.json")
    @test R.digest(read(path))=="db5ed71a9afc0df651c30b74d5222b6059c68fd52c64b10f0aa2a25f4362260a"
    literal=R.JSON3.read(read(path,String),Any)
    positive,negative=setprecision(BigFloat,512) do
        value=BigFloat(1)+ldexp(BigFloat(1),-500)
        value,-value
    end
    @test precision(positive)==512
    @test precision(negative)==512
    for ambient in (128,256,768)
        setprecision(BigFloat,ambient) do
            previous_rounding=rounding(BigFloat)
            for direction in (RoundNearest,RoundDown,RoundUp)
                setrounding(BigFloat,direction) do
                    @test R._dyadic(positive)==literal["positive_exact"]
                    @test R._dyadic(negative)==literal["negative_exact"]
                    @test R._read_dyadic(literal["positive_exact"],512)==positive
                    @test R._read_dyadic(literal["negative_exact"],512)==negative
                    @test precision(BigFloat)==ambient
                    @test rounding(BigFloat)==direction
                end
            end
            @test rounding(BigFloat)==previous_rounding
            oversized=setprecision(BigFloat,1024) do
                BigFloat(1)+ldexp(BigFloat(1),-900)
            end
            @test_throws ArgumentError R._dyadic(oversized)
            @test precision(BigFloat)==ambient
        end
    end
end
