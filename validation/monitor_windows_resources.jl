# Supported Win32 process snapshots distinguish File objects from all kernel handles.
# https://learn.microsoft.com/windows/win32/api/processsnapshot/ns-processsnapshot-pss_handle_entry
@static if Sys.iswindows()
Sys.WORD_SIZE == 64 || error("the Windows File-handle metric requires the verified Win64 snapshot layout")
# Layout transcribed from Microsoft's public processsnapshot.h PSS_HANDLE_ENTRY.
# The largest type-specific union member is the 48-byte Thread record on Win64.
struct _WindowsPssHandleEntry
    handle::Ptr{Cvoid}
    flags::UInt32
    object_type::UInt32
    capture_time::UInt64
    attributes::UInt32
    granted_access::UInt32
    handle_count::UInt32
    pointer_count::UInt32
    paged_pool_charge::UInt32
    nonpaged_pool_charge::UInt32
    creation_time::UInt64
    type_name_length::UInt16
    type_name::Ptr{UInt16}
    object_name_length::UInt16
    object_name::Ptr{UInt16}
    type_specific_information::NTuple{6,UInt64}
end

_windows_resource_process_handle() = ccall((:GetCurrentProcess, "kernel32"), Ptr{Cvoid}, ())
function _kernel_handle_count()
    result = Ref{UInt32}(0)
    ccall((:GetProcessHandleCount, "kernel32"), Cint,
        (Ptr{Cvoid}, Ref{UInt32}), _windows_resource_process_handle(), result) != 0 || error("GetProcessHandleCount failed")
    return Int(result[])
end

function _windows_classify_unnamed_handle(handle)
    duplicate = Ref{Ptr{Cvoid}}(C_NULL)
    ok = ccall((:DuplicateHandle, "kernel32"), Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}, UInt32, Cint, UInt32),
        _windows_resource_process_handle(), handle, _windows_resource_process_handle(), duplicate, 0, 0, 2)
    if ok == 0
        code = ccall((:GetLastError, "kernel32"), UInt32, ())
        code == 50 || error("DuplicateHandle failed while classifying a snapshot handle: Win32 $code")
        flags = Ref{UInt32}(0)
        valid = ccall((:GetHandleInformation, "kernel32"), Cint, (Ptr{Cvoid}, Ref{UInt32}), handle, flags)
        ccall((:SetLastError, "kernel32"), Cvoid, (UInt32,), 0)
        file_type = ccall((:GetFileType, "kernel32"), UInt32, (Ptr{Cvoid},), handle)
        file_error = ccall((:GetLastError, "kernel32"), UInt32, ())
        valid_after = ccall((:GetHandleInformation, "kernel32"), Cint, (Ptr{Cvoid}, Ref{UInt32}), handle, flags)
        if valid != 0 && valid_after != 0 && file_type == 0 && file_error == 6
            return "ValidNonFileNotDuplicable_$code"
        end
        error("cannot stabilize/classify unnamed snapshot handle: DuplicateHandle=$code valid=$valid/$valid_after GetFileType=$file_type/$file_error")
    end
    try
        ccall((:SetLastError, "kernel32"), Cvoid, (UInt32,), 0)
        file_type = ccall((:GetFileType, "kernel32"), UInt32, (Ptr{Cvoid},), duplicate[])
        last_error = ccall((:GetLastError, "kernel32"), UInt32, ())
        file_type in (1, 2, 3) && return "File"
        file_type == 0 && last_error == 0 && return "File"
        file_type == 0 && last_error == 6 && return "NonFileByGetFileType"
        error("cannot classify stabilized unnamed handle: GetFileType=$file_type Win32=$last_error")
    finally
        ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), duplicate[]) != 0 || error("cannot close query duplicate")
    end
end

function _windows_handle_snapshot_counts()
    snapshot, marker = Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)
    flags = UInt32(0x04 | 0x08 | 0x10)
    code = ccall((:PssCaptureSnapshot, "kernel32"), UInt32,
        (Ptr{Cvoid}, UInt32, UInt32, Ref{Ptr{Cvoid}}), _windows_resource_process_handle(), flags, 0, snapshot)
    code == 0 || error("PssCaptureSnapshot failed with Win32 code $code")
    type_counts = Dict{String,Int}()
    entry = Ref{_WindowsPssHandleEntry}()
    try
        code = ccall((:PssWalkMarkerCreate, "kernel32"), UInt32,
            (Ptr{Cvoid}, Ref{Ptr{Cvoid}}), C_NULL, marker)
        code == 0 || error("PssWalkMarkerCreate failed with Win32 code $code")
        while true
            code = ccall((:PssWalkSnapshot, "kernel32"), UInt32,
                (Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Ref{_WindowsPssHandleEntry}, UInt32),
                snapshot[], 2, marker[], entry, sizeof(_WindowsPssHandleEntry))
            code == 259 && break # ERROR_NO_MORE_ITEMS
            code == 0 || error("PssWalkSnapshot failed with Win32 code $code")
            value = entry[]
            classified = value.type_name_length > 0 && iseven(value.type_name_length) && value.type_name_length <= 1024 && value.type_name != C_NULL
            if !classified
                name = _windows_classify_unnamed_handle(value.handle)
                type_counts[name] = get(type_counts, name, 0) + 1
                continue
            end
            words = copy(unsafe_wrap(Vector{UInt16}, value.type_name, Int(value.type_name_length) ÷ 2))
            terminator = findfirst(iszero, words)
            if terminator !== nothing
                all(iszero, words[terminator:end]) || error("unexpected embedded NUL in type name")
                resize!(words, terminator - 1)
            end
            name = transcode(String, words)
            type_counts[name] = get(type_counts, name, 0) + 1
        end
    finally
        try
            if marker[] != C_NULL
                code = ccall((:PssWalkMarkerFree, "kernel32"), UInt32, (Ptr{Cvoid},), marker[])
                code == 0 || error("PssWalkMarkerFree failed with Win32 code $code")
            end
        finally
            code = ccall((:PssFreeSnapshot, "kernel32"), UInt32,
                (Ptr{Cvoid}, Ptr{Cvoid}), _windows_resource_process_handle(), snapshot[])
            code == 0 || error("PssFreeSnapshot failed with Win32 code $code")
        end
    end
    return (;files=get(type_counts, "File", 0), type_counts)
end


_windows_file_handle_count() = _windows_handle_snapshot_counts().files
else
_kernel_handle_count() = missing
end
