"""Utilities for byte-stable PlotlySupply PDF exports."""

const PLOTLY_PDF_FIXED_DATE = "D:20000101000000+00'00'"

function _pdf_find_bytes(data::Vector{UInt8}, needle::Vector{UInt8}, start::Int=1)
    isempty(needle) && return start
    last_start = length(data) - length(needle) + 1
    start > last_start && return nothing
    for first_index in max(start, 1):last_start
        matched = true
        for offset in eachindex(needle)
            if data[first_index + offset - 1] != needle[offset]
                matched = false
                break
            end
        end
        matched && return first_index
    end
    return nothing
end

function plotlysupply_pdf_has_volatile_metadata(path::AbstractString)
    isfile(path) || return false
    data = read(path)
    markers = (
        "/Creator (Chromium)", "/Title (jl_", "/CreationDate (D:", "/ModDate (D:",
    )
    all(marker -> _pdf_find_bytes(data, Vector{UInt8}(codeunits(marker))) !== nothing,
        markers) || return false
    title = try
        String(copy(data[_pdf_parenthesized_value(data, "Title")]))
    catch e
        e isa InterruptException && rethrow()
        return true
    end
    fixed_dates = all(field -> _pdf_find_bytes(
            data, Vector{UInt8}(codeunits("/$field ($PLOTLY_PDF_FIXED_DATE)")),
        ) !== nothing, ("CreationDate", "ModDate"))
    return !(occursin(r"^jl_0+\.html$", title) && fixed_dates)
end

function _pdf_parenthesized_value(data::Vector{UInt8}, field::AbstractString)
    prefix = Vector{UInt8}(codeunits("/$field ("))
    prefix_start = _pdf_find_bytes(data, prefix)
    prefix_start === nothing && error("PlotlySupply PDF is missing /$field metadata")
    value_start = prefix_start + length(prefix)
    value_end_marker = _pdf_find_bytes(data, UInt8[')'], value_start)
    value_end_marker === nothing && error("PlotlySupply PDF has unterminated /$field metadata")
    return value_start:(value_end_marker - 1)
end

function _pdf_replace_same_length!(data::Vector{UInt8}, range::UnitRange{Int},
                                   replacement::AbstractString, field::AbstractString)
    bytes = Vector{UInt8}(codeunits(replacement))
    length(bytes) == length(range) || error(
        "PlotlySupply PDF /$field metadata length changed; refusing to invalidate xref offsets",
    )
    copyto!(data, first(range), bytes, 1, length(bytes))
    return data
end

"""
    normalize_plotlysupply_pdf!(path)

Replace Chromium's volatile creation/modification timestamps and temporary HTML title with
fixed, equal-length values. Equal-length substitution preserves every PDF object and xref offset,
so plotted content is untouched while repeated PlotlySupply exports become byte-identical.
"""
function normalize_plotlysupply_pdf!(path::AbstractString)
    lowercase(splitext(path)[2]) == ".pdf" || throw(ArgumentError(
        "PlotlySupply metadata normalization requires a PDF path",
    ))
    isfile(path) && !islink(path) || throw(ArgumentError(
        "PlotlySupply PDF must be a regular non-symlink file: $path",
    ))
    data = read(path)
    length(data) >= 5 && data[1:5] == Vector{UInt8}(codeunits("%PDF-")) || error(
        "PlotlySupply output is not a PDF: $path",
    )

    title_range = _pdf_parenthesized_value(data, "Title")
    title = String(copy(data[title_range]))
    startswith(title, "jl_") && endswith(title, ".html") || error(
        "unexpected PlotlySupply PDF title: $title",
    )
    token_length = length(title) - length("jl_") - length(".html")
    token_length >= 1 || error("unexpected PlotlySupply PDF temporary title: $title")
    _pdf_replace_same_length!(
        data, title_range, "jl_" * repeat("0", token_length) * ".html", "Title",
    )

    for field in ("CreationDate", "ModDate")
        range = _pdf_parenthesized_value(data, field)
        _pdf_replace_same_length!(data, range, PLOTLY_PDF_FIXED_DATE, field)
    end

    parent = dirname(abspath(path))
    temp_path, io = mktemp(parent)
    try
        write(io, data)
        close(io)
        chmod(temp_path, filemode(path) & 0o777)
        mv(temp_path, path; force=true)
    finally
        isopen(io) && close(io)
        ispath(temp_path) && rm(temp_path; force=true)
    end
    return path
end
