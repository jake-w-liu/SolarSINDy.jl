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
            if Sys.iswindows()
                @test _fd_count() == baseline + 32
            else
                @test _fd_count() >= baseline + 32
            end
        finally
            foreach(close, files)
        end
        @test _fd_count() <= baseline
    end
end

@testset "File metric excludes native events and releases snapshot/query resources" begin
    @static if Sys.iswindows()
        native_count() = begin
            result = Ref{UInt32}(0)
            process = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
            ok = ccall((:GetProcessHandleCount, "kernel32"), Cint,
                (Ptr{Cvoid}, Ref{UInt32}), process, result)
            ok != 0 || error("independent native handle oracle failed")
            Int(result[])
        end
        new_event() = ccall((:CreateEventW, "kernel32"), Ptr{Cvoid},
            (Ptr{Cvoid}, Cint, Cint, Ptr{UInt16}), C_NULL, 0, 0, C_NULL)
        close_event(event) = ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), event)
        # Compile every sampling and native event operation before comparing.
        event = new_event()
        @test event != C_NULL
        @test close_event(event) != 0
        _windows_handle_snapshot_counts()
        _fd_count()
        _kernel_handle_count()
        native_count()
        baseline = _windows_handle_snapshot_counts()
        @test _fd_count() == baseline.files
        @test _kernel_handle_count() == native_count()
        before = native_count()
        event = new_event()
        event != C_NULL || error("CreateEventW failed in independent oracle")
        try
            @test native_count() == before + 1
            @test _kernel_handle_count() == before + 1
            snapshot = _windows_handle_snapshot_counts()
            @test snapshot.files == baseline.files
            @test get(snapshot.type_counts, "Event", 0) == get(baseline.type_counts, "Event", 0) + 1
            @test _fd_count() == baseline.files
        finally
            close_event(event) != 0 || error("CloseHandle failed in independent oracle")
        end
        @test native_count() == before
        @test _fd_count() == baseline.files
        before = native_count()
        counts = [_fd_count() for _ in 1:32]
        @test all(==(baseline.files), counts)
        @test native_count() == before
        @test sizeof(_WindowsPssHandleEntry) == 136
        @test fieldoffset(_WindowsPssHandleEntry, 13) == 64
    else
        @test _kernel_handle_count() === missing
    end
end
end
