"""Pure bounded lookups from catalog witnesses under assumed completion clocks."""
module ArchiveCompletedWindows
using Dates
import ..ArchiveSourceCatalog
import ..ArchiveCatalogAudit
import ..CompletedSourceDBDT
import ..CompletedMinuteReplay
const C = ArchiveSourceCatalog
const D = CompletedSourceDBDT
const M = CompletedMinuteReplay
const MAX_QUERIES = 256
need(ok, message) = ok ? nothing : throw(ArgumentError(message))

mutable struct _Window{N} <: AbstractDict{DateTime,NTuple{N,Float64}}
    catalog::C.SourceCatalog
    first::DateTime
    last::DateTime
    trace::Vector{NamedTuple}
end
function _read!(w::_Window, tag, operation)
    need(tag isa DateTime && floor(tag, Minute) == tag, "exact-minute source query required")
    need(w.first <= tag <= w.last, "source query outside declared window")
    need(length(w.trace) < MAX_QUERIES, "bounded source query count exceeded")
    row = C.locate(w.catalog, tag)
    full = w.catalog.kind === :ground ? UInt8(3) : UInt8(15)
    present = row !== nothing && row.valid_mask == full
    push!(w.trace, (; operation, tag, present,
        body_id=row === nothing ? nothing : row.body_id,
        source_line=row === nothing ? UInt32(0) : row.source_line,
        valid_mask=row === nothing ? UInt8(0) : row.valid_mask))
    row, present
end
Base.haskey(w::_Window, tag) = last(_read!(w, tag, :haskey))
function Base.getindex(w::_Window{N}, tag) where N
    row, present = _read!(w, tag, :getindex)
    present || throw(KeyError(tag))
    ntuple(i -> row.values[i], N)
end
Base.iterate(::_Window) = throw(ArgumentError("source window iteration is forbidden"))
Base.length(::_Window) = throw(ArgumentError("source window scans are forbidden"))
function _ground(catalog)
    need(catalog.kind === :ground, "ground catalog required")
end
function _pins(ground, wind=nothing)
    (; ground_selected_sha256=ground.selected_sha256, ground_raw_sha256=ground.raw_catalog_sha256,
       wind_selected_sha256=wind === nothing ? nothing : wind.selected_sha256,
       wind_raw_sha256=wind === nothing ? nothing : wind.raw_catalog_sha256,
       availability_kind=:assumed_completed_source_intervals,
       measured_receipts_available=false, source_bound_provenance_verified=false)
end

"""Read only feature support. A ground-only call never inspects its wind argument."""
function prepare_catalog_issue(ground::C.SourceCatalog, issue::DateTime;
        policy=M.ReplayPolicy(), mode=:ground11, wind=nothing)
    _ground(ground)
    need(mode in (:ground11, :mixed8), "supported feature layout required")
    timing = M.issue_timing(issue, policy)
    first = M.shift(timing.latest_ground_center, Int64(-30 * 60_000))
    if mode === :mixed8
        need(wind isa C.SourceCatalog && wind.kind === :wind && wind.station == ground.station,
             "matching wind catalog required")
    end
    mag = _Window{2}(ground, first, timing.latest_ground_center, NamedTuple[])
    if mode === :mixed8
        omni = _Window{4}(wind, timing.latest_omni_arrival_start, timing.latest_omni_arrival_start, NamedTuple[])
        result = D.prepare_issue(mag, issue; policy, mode, omni)
        wind_trace = Tuple(omni.trace)
    else
        result = D.prepare_issue(mag, issue; policy, mode)
        wind_trace = ()
    end
    (; accepted=result.accepted, reason=result.reason, issue,
       features=result.accepted ? Tuple(result.features) : nothing,
       ground11=result.accepted ? Tuple(result.ground11) : nothing,
       timing, ground_trace=Tuple(mag.trace), wind_trace,
       source_pins=_pins(ground, mode === :mixed8 ? wind : nothing))
end

"""Admit the target after strict split and within-stream maturity gates, before any read."""
function evaluate_catalog_label(ground::C.SourceCatalog, issue::DateTime, asof::DateTime;
        policy=M.ReplayPolicy(), label_ready_before=nothing)
    _ground(ground)
    timing = M.issue_timing(issue, policy)
    need(label_ready_before === nothing || label_ready_before isa DateTime, "invalid split boundary")
    reason = label_ready_before !== nothing && !M.training_eligible(timing.label_assumed_ready, label_ready_before) ?
        :split_not_strictly_ready : timing.label_assumed_ready > asof ? :label_not_ready : :complete
    if reason !== :complete
        return (; accepted=false, reason, target=nothing, timing, ground_trace=(), source_pins=_pins(ground))
    end
    mag = _Window{2}(ground, issue, timing.last_target_center, NamedTuple[])
    result = M.admit_target(mag, issue, asof; policy)
    (; accepted=result.accepted, reason=result.reason,
       target=result.accepted ? result.value.target_max_dbdt : nothing,
       timing, ground_trace=Tuple(mag.trace), source_pins=_pins(ground))
end
end
