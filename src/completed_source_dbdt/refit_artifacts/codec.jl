# The data-only codec does not infer float fields from strings. Fixed typed
# schemas call float_bits/read_float_bits at their explicitly declared fields.
const CODEC = "dbdt_canonical_json_bits_v1"
const MAX_METADATA_BYTES = 1_048_576
const MAX_MEMBER_BYTES = 8_388_608
const MAX_BUNDLE_BYTES = 1_073_741_824
const MAX_GRAPH_BYTES = Int64(4_294_967_296)
const MAX_GRAPH_FILES = 4096
const MAX_FILES = 256
const MAX_ROWS = 600_000
const CHUNK_ROWS = 8192
const MAX_DEPTH = 16
const MAX_FIELDS = 32
const MAX_ARRAY = 8192
const MAX_STRING_BYTES = 4096
const MAX_ID_BYTES = 512

fail(message) = throw(ArgumentError(message))
require(condition::Bool, message) = condition || fail(message)
digest(bytes::Union{AbstractVector{UInt8},AbstractString}) = bytes2hex(sha256(bytes))

function bounded_string(value; limit=MAX_STRING_BYTES, nonempty=false)
    require(value isa AbstractString, "expected a string")
    require(ncodeunits(value) <= limit, "string byte bound exceeded before copy")
    text = String(value)
    require(isvalid(text), "invalid UTF-8 string")
    require(ncodeunits(text) <= limit, "string byte bound exceeded")
    require(!nonempty || !isempty(text), "empty identifier")
    text
end

function identifier(value)
    text = bounded_string(value; limit=MAX_ID_BYTES, nonempty=true)
    require(all(c -> !iscntrl(c), text), "control character in identifier")
    text
end

function sha_string(value)
    text = bounded_string(value; limit=64)
    require(occursin(r"^[0-9a-f]{64}$", text), "invalid SHA256")
    text
end

function checked_integer(value, low::Integer, high::Integer)
    require(value isa Integer && !(value isa Bool), "expected an integer, not Bool")
    require(low <= value <= high, "integer outside declared bounds")
    Int64(value)
end

function float_value(value)
    require(value isa Real && !(value isa Bool), "expected a real scalar, not Bool")
    require(isfinite(value), "nonfinite scalar")
    converted = try
        Float64(value)
    catch
        fail("scalar cannot be converted to Float64")
    end
    require(isfinite(converted), "Float64 scalar overflow")
    require(value == 0 || converted != 0, "nonzero scalar underflow to Float64 zero")
    converted
end

float_bits(value) = string(reinterpret(UInt64, float_value(value)); base=16, pad=16)
function read_float_bits(value)
    text = bounded_string(value; limit=16)
    require(occursin(r"^[0-9a-f]{16}$", text), "invalid canonical Float64 bits")
    result = reinterpret(Float64, parse(UInt64, text; base=16))
    require(isfinite(result), "nonfinite encoded Float64")
    result
end

function time_text(value::DateTime)
    string(Base.Checked.checked_sub(Dates.value(value), Dates.value(EPOCH)))
end
function read_time_text(value)
    text = bounded_string(value; limit=20)
    require(occursin(r"^(0|-[1-9][0-9]*|[1-9][0-9]*)$", text), "noncanonical UTC millisecond text")
    delta = tryparse(Int64, text)
    require(delta !== nothing, "UTC millisecond overflow")
    total = Base.Checked.checked_add(Dates.value(EPOCH), delta)
    result = DateTime(Dates.UTInstant{Millisecond}(Millisecond(total)))
    require(time_text(result) == text, "UTC millisecond conversion differs")
    result
end

function schema_fields(value, schema::AbstractString)
    require(value isa AbstractDict, "expected a schema object")
    require(haskey(FIELD_KEYS, schema), "unknown fixed schema")
    require(length(value) == length(FIELD_KEYS[schema]), "missing or extra schema field")
    require(all(k -> k isa String, keys(value)), "schema keys must be strings")
    require(Set(keys(value)) == Set(FIELD_KEYS[schema]), "unknown or missing schema field")
    value
end

mutable struct ByteBudget
    buffer::IOBuffer
    used::Int
    limit::Int
end
function emit!(out::ByteBudget, value::Union{AbstractString,AbstractVector{UInt8}})
    amount = value isa AbstractString ? ncodeunits(value) : length(value)
    require(amount <= out.limit - out.used, "encoded member byte bound exceeded")
    write(out.buffer, value)
    out.used += amount
    nothing
end

function encode_value!(out::ByteBudget, value, depth::Int)
    if value === nothing
        emit!(out, "null")
    elseif value isa Bool
        emit!(out, value ? "true" : "false")
    elseif value isa Integer
        checked_integer(value, typemin(Int64), typemax(Int64))
        emit!(out, string(value))
    elseif value isa AbstractString
        text = bounded_string(value)
        emit!(out, JSON3.write(text))
    elseif value isa AbstractDict
        require(depth < MAX_DEPTH, "JSON nesting depth exceeded")
        require(length(value) <= MAX_FIELDS, "JSON object field bound exceeded")
        require(all(k -> k isa String, keys(value)), "JSON object keys must be strings")
        names = sort!(collect(keys(value)))
        require(all(k -> isascii(k) && ncodeunits(k) <= MAX_STRING_BYTES, names), "JSON keys must be bounded ASCII")
        emit!(out, "{")
        for (index, key) in enumerate(names)
            index > 1 && emit!(out, ",")
            encode_value!(out, key, depth + 1)
            emit!(out, ":")
            encode_value!(out, value[key], depth + 1)
        end
        emit!(out, "}")
    elseif value isa AbstractVector || value isa Tuple
        require(depth < MAX_DEPTH, "JSON nesting depth exceeded")
        value isa AbstractVector && Base.require_one_based_indexing(value)
        require(length(value) <= MAX_ARRAY, "JSON array item bound exceeded")
        emit!(out, "[")
        for (index, element) in enumerate(value)
            index > 1 && emit!(out, ",")
            encode_value!(out, element, depth + 1)
        end
        emit!(out, "]")
    else
        fail("unsupported data-only JSON value; floats require declared bit fields")
    end
    nothing
end

function canonical_bytes(value; limit=MAX_MEMBER_BYTES)
    require(0 < limit <= MAX_MEMBER_BYTES, "invalid member byte limit")
    out = ByteBudget(IOBuffer(), 0, Int(limit))
    encode_value!(out, value, 0)
    emit!(out, "\n")
    take!(out.buffer)
end

# This is a resource preflight, not a second JSON parser. JSON3 checks syntax;
# fixed schemas and exact re-encoding then reject duplicate/noncanonical forms.
function lexical_preflight(bytes::Vector{UInt8}, limit::Int)
    require(0 < length(bytes) <= limit, "invalid JSON member byte length")
    text = String(copy(bytes))
    require(isvalid(text), "invalid UTF-8 JSON")
    kinds = UInt8[]
    counts = Int[]
    in_string = false
    escaped = false
    raw_string_bytes = 0
    for byte in bytes
        if in_string
            if escaped
                escaped = false
            elseif byte == 0x5c
                escaped = true
            elseif byte == 0x22
                in_string = false
            end
            raw_string_bytes += 1
            require(raw_string_bytes <= 6MAX_STRING_BYTES + 2, "JSON lexical string byte bound exceeded")
            continue
        end
        if byte == 0x22
            in_string = true
            raw_string_bytes = 0
        elseif byte == 0x7b || byte == 0x5b
            push!(kinds, byte)
            push!(counts, 0)
            require(length(kinds) <= MAX_DEPTH, "JSON lexical nesting bound exceeded")
        elseif byte == 0x7d || byte == 0x5d
            require(!isempty(kinds), "unmatched JSON container")
            require((byte == 0x7d && last(kinds) == 0x7b) || (byte == 0x5d && last(kinds) == 0x5b), "mismatched JSON container")
            pop!(kinds)
            pop!(counts)
        elseif byte == 0x3a
            require(!isempty(kinds) && last(kinds) == 0x7b, "JSON colon outside object")
            counts[end] += 1
            require(last(counts) <= MAX_FIELDS, "JSON lexical object field bound exceeded")
        elseif byte == 0x2c && !isempty(kinds) && last(kinds) == 0x5b
            counts[end] += 1
            require(last(counts) < MAX_ARRAY, "JSON lexical array item bound exceeded")
        elseif byte == 0x2e
            fail("JSON float atoms are forbidden; use typed hexadecimal bits")
        end
    end
    require(!in_string && isempty(kinds), "unterminated JSON value")
    text
end

function decode_canonical(input::Union{AbstractString,AbstractVector{UInt8}}; limit=MAX_MEMBER_BYTES)
    require(0 < limit <= MAX_MEMBER_BYTES, "invalid member byte limit")
    input isa AbstractVector && Base.require_one_based_indexing(input)
    amount = input isa AbstractString ? ncodeunits(input) : length(input)
    require(0 < amount <= limit, "JSON member byte bound exceeded")
    bytes = input isa AbstractString ? Vector{UInt8}(codeunits(input)) : Vector{UInt8}(input)
    text = lexical_preflight(bytes, Int(limit))
    value = try
        JSON3.read(text, Any)
    catch error
        error isa InterruptException && rethrow()
        fail("invalid JSON syntax")
    end
    # JSON3's Any mapping supplies ordinary Dict{String,Any}/Vector{Any}; all
    # encoded values are revalidated under the same allocation/shape bounds.
    require(canonical_bytes(value; limit) == bytes, "duplicate key or noncanonical JSON bytes")
    value
end

function bounded_read(path::AbstractString; limit=MAX_MEMBER_BYTES)
    capacity = checked_integer(limit, 1, MAX_MEMBER_BYTES)
    require(!islink(path) && isfile(path), "member must be an ordinary file")
    size = filesize(path)
    require(0 <= size <= capacity, "member size bound exceeded before read")
    data = open(path, "r") do io
        read(io, Int(size))
    end
    require(length(data) == size && filesize(path) == size, "member changed during read")
    data
end
