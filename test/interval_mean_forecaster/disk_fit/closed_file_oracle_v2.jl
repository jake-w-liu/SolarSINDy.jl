"""Observe an ordinary owned file's open-handle state in this process."""
function owned_file_handles_closed(path::AbstractString)
    isfile(path) && !islink(path) || throw(ArgumentError("ordinary owned file required"))
    if Sys.iswindows()
        native=replace(abspath(path),'/'=>'\\')
        extended=startswith(native,"\\\\") ? "\\\\?\\UNC\\"*native[3:end] : "\\\\?\\"*native
        wide=transcode(UInt16,extended*"\0")
        handle=GC.@preserve wide ccall((:CreateFileW,"kernel32"),Ptr{Cvoid},
            (Ptr{UInt16},UInt32,UInt32,Ptr{Cvoid},UInt32,UInt32,Ptr{Cvoid}),
            pointer(wide),0x80000000,0,C_NULL,3,0x80,C_NULL)
        if handle==Ptr{Cvoid}(typemax(UInt))
            code=ccall((:GetLastError,"kernel32"),UInt32,())
            code in (32,33) && return false
            error("exclusive owned-file query failed with Windows code $code")
        end
        ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),handle)!=0 ||
            error("exclusive owned-file probe handle could not be closed")
        return true
    elseif Sys.isunix()
        directory=isdir("/proc/self/fd") ? "/proc/self/fd" : "/dev/fd"
        isdir(directory) || error("own descriptor directory unavailable")
        target=stat(path)
        target.ioerrno==0 && target.inode!=0 || error("owned-file identity unavailable")
        for label in readdir(directory)
            number=tryparse(Int,label)
            number===nothing && continue
            number>=0 || continue
            observed=try
                stat(RawFD(number))
            catch error
                # readdir's own descriptor may close before this fstat.
                error isa Base.IOError && error.code==Base.UV_EBADF && continue
                rethrow()
            end
            observed.ioerrno==Base.UV_EBADF && continue
            observed.ioerrno==0 || error("own descriptor fstat failed: $(observed.ioerrno)")
            isfile(observed) && (observed.device,observed.inode)==(target.device,target.inode) && return false
        end
        return true
    end
    error("owned-file handle oracle unsupported on this platform")
end

function owned_file_handle_oracle_controls(directory::AbstractString)
    mkdir(directory)
    target=joinpath(directory,"held-file.bin")
    other=joinpath(directory,"unrelated-file.bin")
    bytes=UInt8[0x13,0x57,0x9b,0xdf]
    write(target,bytes);write(other,reverse(bytes))
    @testset "owned file handle oracle detects held and closed controls" begin
        @test owned_file_handles_closed(target)
        held=open(target,"r")
        try
            @test !owned_file_handles_closed(target)
        finally
            close(held)
        end
        @test owned_file_handles_closed(target)
        unrelated=open(other,"r")
        try
            @test owned_file_handles_closed(target)
            @test !owned_file_handles_closed(other)
        finally
            close(unrelated)
        end
        @test read(target)==bytes
        @test_throws ArgumentError owned_file_handles_closed(joinpath(directory,"absent.bin"))
        @test_throws ArgumentError owned_file_handles_closed(directory)
    end
    nothing
end
