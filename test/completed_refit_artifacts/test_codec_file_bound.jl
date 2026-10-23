@testset "completed refit member read retains its hard byte cap" begin
    # Synthetic resource-only fixture: the helper must not read an oversized
    # member merely because a caller supplies a larger keyword limit.
    directory = joinpath(CODEC_TEST_DIRECTORY,"member-bound")
    @test !ispath(directory)
    mkpath(directory)
    path = joinpath(directory,"oversize-member.bin")
    open(path,"w") do io
        write(io,zeros(UInt8,R.MAX_MEMBER_BYTES+1))
    end
    @test filesize(path) == R.MAX_MEMBER_BYTES+1
    @test_throws ArgumentError R.bounded_read(path;limit=R.MAX_MEMBER_BYTES+1)
    @test_throws ArgumentError R.bounded_read(path)
end
