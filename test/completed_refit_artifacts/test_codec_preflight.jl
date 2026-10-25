struct OversizeUncopiedString <: AbstractString end
Base.ncodeunits(::OversizeUncopiedString) = 4097
Base.String(::OversizeUncopiedString) = error("oversized string copied before its byte preflight")
@testset "completed refit reject oversized strings before copying" begin
    # A SubString can otherwise allocate its entire backing view during String
    # conversion. This pure probe throws if conversion is attempted at all.
    @test_throws ArgumentError R.bounded_string(OversizeUncopiedString())
end
