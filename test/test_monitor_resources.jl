module MonitorResourceTests
using Test
include(joinpath(@__DIR__, "..", "validation", "monitor_soak.jl"))

@testset "platform resource counters detect owned file handles" begin
    rss = _rss_bytes()
    @test rss === missing || rss > 0
    @test !Sys.iswindows() || rss isa Int
    mktempdir() do dir
        # Compile opening/closing and the counter before measuring growth.
        open(io -> nothing, joinpath(dir, "warmup"), "w")
        _fd_count()
        baseline = _fd_count()
        files = IOStream[]
        try
            for i in 1:32
                push!(files, open(joinpath(dir, string(i)), "w"))
            end
            @test _fd_count() >= baseline + 32
        finally
            foreach(close, files)
        end
        @test _fd_count() <= baseline
    end
end
end
