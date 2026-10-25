"""Independent catalog read-window route over independently reconstructed bodies."""
module ArchiveCompletedWindowAudit
using Dates
import ..ArchiveSourceCatalog
import ..ArchiveCatalogAudit
import ..CompletedSourceDBDT
import ..CompletedMinuteReplay
const A = ArchiveCatalogAudit
const Replay = CompletedMinuteReplay
const Features = CompletedSourceDBDT
require(ok, message) = ok ? nothing : throw(ArgumentError(message))

mutable struct _AuditWindow{N} <: AbstractDict{DateTime,NTuple{N,Float64}}
    body::A.CatalogAudit
    lower::DateTime
    upper::DateTime
    reads::Vector{NamedTuple}
end
function _lookup(w::_AuditWindow, time, operation)
    require(time isa DateTime && floor(time, Minute) == time, "audit exact minute needed")
    require(w.lower <= time <= w.upper, "audit query outside support")
    require(length(w.reads) < 256, "audit query bound")
    # This path calls only the independent catalog lookup, never the primary
    # window, primary parser, primary catalog or its selected witness decoder.
    record = A.audit_locate(w.body, time)
    mask = w.body.kind === :ground ? UInt8(3) : UInt8(15)
    found = record !== nothing && record.valid_mask == mask
    item = record === nothing ?
        (; operation, tag=time, present=false, body_id=nothing, source_line=UInt32(0), valid_mask=UInt8(0)) :
        (; operation, tag=time, present=found, body_id=record.body_id, source_line=record.source_line, valid_mask=record.valid_mask)
    push!(w.reads, item)
    record, found
end
Base.haskey(w::_AuditWindow, time) = _lookup(w, time, :haskey)[2]
function Base.getindex(w::_AuditWindow{N}, time) where N
    record, found = _lookup(w, time, :getindex)
    found || throw(KeyError(time))
    Tuple(record.values[i] for i in 1:N)::NTuple{N,Float64}
end
Base.iterate(::_AuditWindow) = throw(ArgumentError("audit window cannot iterate"))
Base.length(::_AuditWindow) = throw(ArgumentError("audit window cannot scan"))
function _identities(mag, solar=nothing)
    (; ground_selected_sha256=mag.selected_sha256, ground_raw_sha256=mag.raw_catalog_sha256,
       wind_selected_sha256=solar === nothing ? nothing : solar.selected_sha256,
       wind_raw_sha256=solar === nothing ? nothing : solar.raw_catalog_sha256,
       availability_kind=:assumed_completed_source_intervals,
       measured_receipts_available=false, source_bound_provenance_verified=false)
end
"""Audit feature support through independent catalog windows and the shared, separately verified feature primitive."""
function audit_catalog_issue(mag::A.CatalogAudit, issue::DateTime;
        policy=Replay.ReplayPolicy(), mode=:ground11, wind=nothing)
    require(mag.kind === :ground, "audit ground stream needed")
    require(mode === :ground11 || mode === :mixed8, "audit supported layout needed")
    timing = Replay.issue_timing(issue, policy)
    low = Replay.shift(timing.latest_ground_center, Int64(-1_800_000))
    if mode === :mixed8
        require(wind isa A.CatalogAudit && wind.kind === :wind && wind.station == mag.station,
                "audit matching wind stream needed")
    end
    ground_view = _AuditWindow{2}(mag, low, timing.latest_ground_center, NamedTuple[])
    if mode === :ground11
        prepared = Features.prepare_issue(ground_view, issue; policy, mode)
        wind_reads = ()
    else
        wind_view = _AuditWindow{4}(wind, timing.latest_omni_arrival_start, timing.latest_omni_arrival_start, NamedTuple[])
        prepared = Features.prepare_issue(ground_view, issue; policy, mode, omni=wind_view)
        wind_reads = Tuple(wind_view.reads)
    end
    (; accepted=prepared.accepted, reason=prepared.reason, issue,
       features=prepared.accepted ? Tuple(prepared.features) : nothing,
       ground11=prepared.accepted ? Tuple(prepared.ground11) : nothing,
       timing, ground_trace=Tuple(ground_view.reads), wind_trace=wind_reads,
       source_pins=_identities(mag, mode === :mixed8 ? wind : nothing))
end
"""Audit target support after strict split and maturity gates, using the independent catalog lookup."""
function audit_catalog_label(mag::A.CatalogAudit, issue::DateTime, asof::DateTime;
        policy=Replay.ReplayPolicy(), label_ready_before=nothing)
    require(mag.kind === :ground, "audit label ground stream needed")
    timing = Replay.issue_timing(issue, policy)
    require(label_ready_before === nothing || label_ready_before isa DateTime, "audit split boundary type")
    if label_ready_before !== nothing && timing.label_assumed_ready >= label_ready_before
        return (; accepted=false, reason=:split_not_strictly_ready, target=nothing, timing,
                ground_trace=(), source_pins=_identities(mag))
    elseif asof < timing.label_assumed_ready
        return (; accepted=false, reason=:label_not_ready, target=nothing, timing,
                ground_trace=(), source_pins=_identities(mag))
    end
    source = _AuditWindow{2}(mag, issue, timing.last_target_center, NamedTuple[])
    result = Replay.admit_target(source, issue, asof; policy)
    (; accepted=result.accepted, reason=result.reason,
       target=result.accepted ? result.value.target_max_dbdt : nothing,
       timing, ground_trace=Tuple(source.reads), source_pins=_identities(mag))
end
end
