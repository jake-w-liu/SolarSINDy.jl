"""
Independent archival-format reconstruction with a separate decimal lexer,
grouped duplicate verification and explicit binary-record encoding.
"""
module ArchiveFormatAudit
using Dates, SHA

const LIMIT_BYTES = 8 * 1024 * 1024
const LIMIT_ROWS = 50_000
const WIDTH = 48
const ZERO_DATE = DateTime(1970)
reject(message) = throw(ArgumentError(message))
check(condition, message) = condition ? nothing : reject(message)

struct _AuditLines{T}
    parts::T
    bytes::Int
end
_AuditLines(raw::String) = _AuditLines(eachsplit(raw, '\n'; keepempty=true), ncodeunits(raw))
Base.IteratorSize(::Type{<:_AuditLines}) = Base.SizeUnknown()
Base.eltype(::Type{<:_AuditLines}) = SubString{String}
@inline function _line_item(lines::_AuditLines, item, consumed::Int)
    (item === nothing || consumed >= lines.bytes) && return nothing
    fragment, following = item
    width = ncodeunits(fragment)
    terminated = consumed + width < lines.bytes
    record = terminated && endswith(fragment, '\r') ? chop(fragment; tail=1) : fragment
    (record, (following, consumed + width + 1))
end
@inline Base.iterate(lines::_AuditLines) = _line_item(lines, iterate(lines.parts), 0)
@inline Base.iterate(lines::_AuditLines, state) = _line_item(lines, iterate(lines.parts, state[1]), state[2])

struct AuditBody
    kind::Symbol
    station::String
    product::String
    start::DateTime
    stop::DateTime
    body_digest::String
    selected_digest::String
    raw_digest::String
    selected_records::String
    raw_records::String
    record_count::Int
    unique_count::Int
    duplicate_count::Int
    complete_count::Int
    absent_minutes::Int
    classification_counts::NTuple{4,NTuple{4,Int}}
end

struct _AuditRow
    stamp::Int64
    line::UInt32
    mask::UInt8
    reason::UInt8
    values::NTuple{4,Float64}
end

function _envelope(text, start, stop)
    check(start isa DateTime && stop isa DateTime, "explicit minute interval required")
    span = Int128(Dates.value(stop)) - Int128(Dates.value(start))
    check(floor(start, Minute) == start && floor(stop, Minute) == stop &&
          0 < span <= 32 * 86_400_000, "invalid bounded minute interval")
    check(ncodeunits(text) <= LIMIT_BYTES, "body byte limit")
    result = String(text)
    check(isvalid(result) && !occursin('\0', result), "invalid body encoding")
    result
end

digit(c) = '0' <= c <= '9'
function _decimal(value)
    token = strip(String(value)); n = ncodeunits(token)
    check(0 < n <= 128, "numeric token limit")
    token in ("NaN", "Inf", "+Inf", "-Inf") && return parse(Float64, token)
    # Independent finite-state lexical check: no production regular expression.
    i = 1
    codeunit(token, i) in (0x2b, 0x2d) && (i += 1)
    before = 0
    while i <= n && 0x30 <= codeunit(token, i) <= 0x39; i += 1; before += 1; end
    after = 0
    if i <= n && codeunit(token, i) == 0x2e
        i += 1
        while i <= n && 0x30 <= codeunit(token, i) <= 0x39; i += 1; after += 1; end
    end
    check(before + after > 0, "numeric mantissa required")
    if i <= n && codeunit(token, i) in (0x65, 0x45)
        i += 1
        i <= n && codeunit(token, i) in (0x2b, 0x2d) && (i += 1)
        first_digit = i
        while i <= n && 0x30 <= codeunit(token, i) <= 0x39; i += 1; end
        check(i > first_digit, "numeric exponent required")
    end
    check(i == n + 1, "unexpected numeric suffix")
    parse(Float64, token)
end

function _utc_minute(text, doy=nothing)
    fields = match(r"\A([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):00\.000Z\z", text)
    check(fields !== nothing, "exact UTC minute syntax required")
    year, month, day, hour, minute = parse.(Int, fields.captures)
    date = Date(year, month, day)
    if doy !== nothing
        check(ncodeunits(doy) == 3 && all(digit, doy) && parse(Int, doy) == dayofyear(date), "day-of-year mismatch")
    end
    check((hour < 24 && minute < 60) || (hour == 24 && minute == 0), "invalid UTC minute")
    DateTime(date) + Hour(hour) + Minute(minute)
end

function _ground_value(token)
    number = _decimal(token)
    number == 99999.0 && return (0.0, 1)
    number == 88888.0 && return (0.0, 2)
    isfinite(number) && abs(number) < 90000 ? (number, 0) : (0.0, 3)
end

function _wind_value(token, position)
    number = _decimal(token)
    fill = position <= 2 ? 9999.99 : position == 3 ? 99999.9 : 999.99
    number == fill && return (0.0, 1)
    valid = isfinite(number) && (position <= 2 ? abs(number) < 9000 : position == 3 ?
            100 < number < 90000 : 0 < number < 900)
    valid ? (number, 0) : (0.0, 3)
end

function _entry(tag, line, pairs)
    stamp = Int128(Dates.value(tag)) - Int128(Dates.value(ZERO_DATE))
    check(typemin(Int64) <= stamp <= typemax(Int64), "timestamp storage overflow")
    mask = UInt8(0); reason = UInt8(0)
    for j in eachindex(pairs)
        pairs[j][2] == 0 && (mask |= UInt8(1 << (j - 1)))
        reason |= UInt8(pairs[j][2] << (2 * (j - 1)))
    end
    values = ntuple(j -> j <= length(pairs) ? Float64(pairs[j][1]) : 0.0, 4)
    _AuditRow(Int64(stamp), UInt32(line), mask, reason, values)
end

function _put_word!(bytes, index, word::UInt64, width)
    for k in 0:(width - 1); bytes[index + k] = UInt8((word >> (8 * k)) & 0xff); end
end

function _encode(rows)
    bytes = Vector{UInt8}(undef, WIDTH * length(rows))
    for (index, row) in enumerate(rows)
        base = WIDTH * (index - 1) + 1
        _put_word!(bytes, base, reinterpret(UInt64, row.stamp), 8)
        _put_word!(bytes, base + 8, UInt64(row.line), 4)
        bytes[base + 12] = row.mask; bytes[base + 13] = row.reason
        bytes[base + 14] = 0; bytes[base + 15] = 0
        for j in 1:4; _put_word!(bytes, base + 16 + 8 * (j - 1), reinterpret(UInt64, row.values[j]), 8); end
    end
    String(bytes)
end

_digest(bytes::String) = bytes2hex(sha256(IOBuffer(bytes)))
function _summarize(kind, station, product, start, stop, text, rows)
    # Sorting and grouped reconstruction are deliberately separate from the
    # production reader's online duplicate-witness map.
    ordered = sort(rows; by = row -> (row.stamp, row.line))
    selected = _AuditRow[]; first = 1; full = UInt8(kind == :ground ? 3 : 15)
    while first <= length(ordered)
        last = first
        while last < length(ordered) && ordered[last + 1].stamp == ordered[first].stamp; last += 1; end
        for component in 1:4
            bit = UInt8(1 << (component - 1)); witness = nothing
            for index in first:last
                row = ordered[index]
                if row.mask & bit != 0
                    witness === nothing ? (witness = row.values[component]) :
                        check(witness == row.values[component], "conflicting duplicate component")
                end
            end
        end
        chosen = first
        for index in first:last
            if ordered[index].mask == full; chosen = index; break; end
        end
        push!(selected, ordered[chosen]); first = last + 1
    end
    counts = zeros(Int, 4, 4)
    for row in rows, component in 1:(kind == :ground ? 2 : 4)
        reason = Int((row.reason >> (2 * (component - 1))) & 0x03)
        counts[component, reason + 1] += 1
    end
    raw = _encode(rows); normalized = _encode(selected)
    span = Int128(Dates.value(stop)) - Int128(Dates.value(start))
    expected = Int(span ÷ 60_000) + (kind == :ground ? 1 : 0)
    AuditBody(kind, String(station), String(product), start, stop, _digest(text),
        _digest(normalized), _digest(raw), normalized, raw, length(rows), length(selected),
        length(rows) - length(selected), count(row -> row.mask == full, selected), expected - length(selected),
        ntuple(j -> ntuple(k -> counts[j, k], 4), 4))
end

const IDENTITY_FIELDS = ("Format", "Source of Data", "IAGA CODE", "Reported", "Data Interval Type", "Data Type")
const OTHER_HEADERS = ("Station Name", "Geodetic Latitude", "Geodetic Longitude", "Elevation", "Sensor Orientation", "Digital Sampling", "Publication date")
function _header_value(text, label)
    startswith(text, label) || return nothing
    next = nextind(text, lastindex(label))
    next <= lastindex(text) && isspace(text[next]) || return nothing
    value = strip(SubString(text, next))
    isempty(value) ? nothing : String(value)
end

function audit_ground(text::AbstractString; station::AbstractString, product::AbstractString, start::DateTime, stop::DateTime)
    raw = _envelope(text, start, stop)
    check(ncodeunits(station) == 3 && all(c -> 'A' <= c <= 'Z', station) && product == "quasi-definitive", "unsupported ground declaration")
    headers = Dict{String,String}(); identity = nothing; active = false; previous = nothing; rows = _AuditRow[]
    for (line, record) in enumerate(_AuditLines(raw))
        check(ncodeunits(record) <= 4096, "line byte limit")
        content = strip(record)
        (isempty(content) || startswith(content, '#')) && continue
        if startswith(content, "DATE")
            check(endswith(content, '|'), "ground data-column terminator required")
            names = split(strip(chop(content; tail=1)))
            check(length(names) == 7 && names[1:3] == ["DATE", "TIME", "DOY"], "ground timestamp columns differ")
            check(all(name -> haskey(headers, name), IDENTITY_FIELDS), "incomplete ground identity")
            check(headers["Format"] == "IAGA-2002" && headers["IAGA CODE"] == station && headers["Data Type"] == product &&
                  headers["Reported"] in ("XYNULNUL", "XYZF") && occursin(r"\b1-minute\b", headers["Data Interval Type"]), "ground identity mismatch")
            tail = headers["Reported"] == "XYZF" ? [station * "Z", station * "F"] : [station * "NUL", station * "NUL"]
            check(names[4:7] == [station * "X", station * "Y", tail...], "ground component columns differ")
            current = Tuple(headers[name] for name in IDENTITY_FIELDS)
            identity === nothing ? (identity = current) : check(identity == current, "mixed ground segments")
            active = true; previous = nothing
        elseif digit(first(content))
            check(active, "data before complete ground header")
            fields = split(content); check(length(fields) == 7, "ground record field count")
            tag = _utc_minute(string(fields[1], 'T', fields[2], 'Z'), fields[3])
            check(start <= tag <= stop && (previous === nothing || tag >= previous), "ground timestamp outside ordered interval")
            previous = tag
            check(length(rows) < LIMIT_ROWS, "raw row limit")
            x = _ground_value(fields[4]); y = _ground_value(fields[5]); _decimal(fields[6]); _decimal(fields[7])
            push!(rows, _entry(tag, line, (x, y)))
        else
            check(endswith(content, '|'), "unknown ground record")
            header = strip(chop(content; tail=1)); recognized = false
            for label in IDENTITY_FIELDS
                value = _header_value(header, label); value === nothing && continue
                if label == "Format"; empty!(headers); active = false; end
                check(!haskey(headers, label) || headers[label] == value, "conflicting ground header")
                headers[label] = value; recognized = true; break
            end
            recognized || check(any(label -> _header_value(header, label) !== nothing, OTHER_HEADERS), "unsupported ground header")
        end
    end
    check(active && identity !== nothing, "complete ground segment absent")
    _summarize(:ground, station, product, start, stop, raw, rows)
end

function _csv_field(field)
    text = strip(String(field)); quotes = count(==('"'), text)
    if quotes == 0; return text; end
    check(quotes == 2 && startswith(text, '"') && endswith(text, '"'), "invalid CSV quoting")
    chop(text; head=1, tail=1)
end

function audit_wind(text::AbstractString; station::AbstractString, product::AbstractString, start::DateTime, stop::DateTime)
    raw = _envelope(text, start, stop)
    check(ncodeunits(station) == 3 && all(c -> 'A' <= c <= 'Z', station) && product == "OMNI_HRO_1MIN", "unsupported wind declaration")
    rows = _AuditRow[]; previous = nothing
    for (line, record) in enumerate(_AuditLines(raw))
        check(ncodeunits(record) <= 4096, "line byte limit")
        isempty(strip(record)) && continue
        fields = _csv_field.(split(record, ','; keepempty=true))
        check(length(fields) == 5, "wind record field count")
        tag = _utc_minute(fields[1])
        check(start <= tag < stop && (previous === nothing || tag >= previous), "wind timestamp outside ordered interval")
        previous = tag; check(length(rows) < LIMIT_ROWS, "raw row limit")
        values = [_wind_value(fields[j + 1], j) for j in 1:4]
        push!(rows, _entry(tag, line, (values[3], values[2], values[1], values[4])))
    end
    _summarize(:wind, station, product, start, stop, raw, rows)
end

function _unsigned(storage, first, bytes)
    value = UInt64(0)
    for index in (first + bytes - 1):-1:first; value = (value << 8) | UInt64(codeunit(storage, index)); end
    value
end
function _decode(storage, count, index::Integer)
    check(!(index isa Bool) && 1 <= index <= count, "record index outside audit body")
    first = WIDTH * (Int(index) - 1) + 1
    stamp = reinterpret(Int64, _unsigned(storage, first, 8))
    values = ntuple(j -> reinterpret(Float64, _unsigned(storage, first + 16 + 8 * (j - 1), 8)), 4)
    (; tag=ZERO_DATE + Millisecond(stamp), source_line=UInt32(_unsigned(storage, first + 8, 4)),
       valid_mask=codeunit(storage, first + 12), reason_bits=codeunit(storage, first + 13), values)
end
record(body::AuditBody, index::Integer) = _decode(body.selected_records, body.unique_count, index)
raw_record(body::AuditBody, index::Integer) = _decode(body.raw_records, body.record_count, index)
function complete_components(body::AuditBody, index::Integer)
    row = record(body, index)
    body.kind == :ground ? (row.valid_mask == 3 ? (row.values[1], row.values[2]) : nothing) :
        (row.valid_mask == 15 ? row.values : nothing)
end
end
