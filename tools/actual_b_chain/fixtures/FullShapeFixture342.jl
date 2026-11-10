"""Deterministic generated source17 resource fixture; contains no archive data."""
module FullShapeFixture342
using SHA
export SOURCE_ROWS,CHANNELS,value_at,write_raw
const SOURCE_ROWS=500000
const CHANNELS=17
const CHUNK_ROWS=256

function _index(value,low,high)
    value isa Integer&&!(value isa Bool)&&low<=value<=high||throw(ArgumentError("generated fixture index bound"))
    Int(value)
end
"A raw channel at a zero-based hour; all arithmetic is exact in Float64."
function value_at(channel,hour)
    j=_index(channel,1,CHANNELS);t=_index(hour,0,SOURCE_ROWS-1)
    if j==6
        Float64(mod(div(t,16),9)-4)
    elseif j==8
        Float64(350+4*mod(div(t,8),31))
    elseif j==12
        1.0+Float64(mod(div(t,4),16))/8.0
    elseif j==15
        -80.0+Float64(mod(t,256))/4.0
    else
        Float64(mod(13*t+7*j,16))/4.0
    end
end
function write_raw(path;rows=SOURCE_ROWS)
    n=_index(rows,1,SOURCE_ROWS)
    path isa AbstractString&&isabspath(path)||throw(ArgumentError("absolute generated output required"))
    full=normpath(String(path));parent=dirname(full)
    full==String(path)&&isdir(parent)&&realpath(parent)==parent&&!ispath(full)&&!islink(full)||
        throw(ArgumentError("fresh canonical generated output required"))
    flags=Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL
    file=Base.Filesystem.open(full,flags,0o600)
    context=SHA.SHA2_256_CTX();buffer=IOBuffer();total=0
    try
        for first in 0:CHUNK_ROWS:n-1
            last=min(first+CHUNK_ROWS-1,n-1)
            # Matrix{Float64}(17,n) column-major order: channel changes fastest.
            for hour in first:last,channel in 1:CHANNELS
                write(buffer,htol(reinterpret(UInt64,value_at(channel,hour))))
            end
            body=take!(buffer)
            write(file,body)==length(body)||throw(ArgumentError("short generated fixture write"))
            SHA.update!(context,body);total+=length(body)
        end
    finally
        close(file)
    end
    total==136*n&&filesize(full)==total||throw(ArgumentError("generated fixture byte count differs"))
    (;path=full,sha256=bytes2hex(SHA.digest!(context)),bytes=total,source_rows=n)
end
end
