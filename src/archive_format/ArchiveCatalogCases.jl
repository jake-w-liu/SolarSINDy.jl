"""Pure immutable feature schedules and separately matured labels over agreed catalog values.

This namespace performs no file access, grants no actual-source authority and
does not qualify a model fit, historical receipt or scientific result.
"""
module ArchiveCatalogCases
using Dates, SHA
import ..ArchiveSourceCatalog
import ..ArchiveCatalogAudit
import ..ArchiveCompletedWindows
import ..ArchiveCompletedWindowAudit
import ..CompletedMinuteReplay
const C = ArchiveSourceCatalog
const A = ArchiveCatalogAudit
const W = ArchiveCompletedWindows
const V = ArchiveCompletedWindowAudit
const M = CompletedMinuteReplay
export CatalogFeatureSchedule, CatalogLabelAudit, seal_catalog_features,
       audit_catalog_labels, feature_row, label_row, case_arrays, case_counts

const MAX_ROWS = 600_000
const FEATURE_BYTES = 160
const LABEL_BYTES = 64
const FEATURE_REASONS = (:complete, :missing_ground_history, :invalid_ground_history,
    :missing_omni_history, :invalid_omni_history, :invalid_feature_summary)
const LABEL_REASONS = (:complete, :split_not_strictly_ready, :label_not_ready,
    :missing_ground_target, :invalid_ground_target)
const EPOCH = DateTime(1970)
const Catalog = Union{C.SourceCatalog,A.CatalogAudit}
need(ok, message) = ok ? nothing : throw(ArgumentError(message))
struct _Checked end

"""Immutable160-byte feature records with complete scheduled-issue membership and source pins."""
struct CatalogFeatureSchedule
    station::String
    mode::Symbol
    policy::M.ReplayPolicy
    source_pins::NamedTuple
    rows::Int
    payload::String
    payload_sha256::String
    schedule_sha256::String
    function CatalogFeatureSchedule(::_Checked, args...); new(args...); end
end
"""Immutable64-byte label records bound to a sealed feature schedule and explicit readiness cutoffs."""
struct CatalogLabelAudit
    feature_schedule_sha256::String
    asof::DateTime
    label_ready_before::Union{Nothing,DateTime}
    rows::Int
    payload::String
    payload_sha256::String
    audit_sha256::String
    function CatalogLabelAudit(::_Checked, args...); new(args...); end
end
Base.length(value::Union{CatalogFeatureSchedule,CatalogLabelAudit}) = value.rows
_hash(value::String) = bytes2hex(sha256(IOBuffer(value)))
function _stamp(time::DateTime)
    value = Int128(Dates.value(time)) - Int128(Dates.value(EPOCH))
    need(typemin(Int64) <= value <= typemax(Int64), "case timestamp outside signed64 Unix milliseconds")
    Int64(value)
end
_int(io, value::Int64) = write(io, htol(value))
_float(io, value::Float64) = write(io, htol(reinterpret(UInt64,value)))
function _frame(io, value::AbstractString)
    write(io, htol(UInt64(ncodeunits(value))), codeunits(value))
end
function _code(reason, choices)
    at = findfirst(==(reason), choices)
    need(at !== nothing, "unknown case admission reason")
    UInt8(at-1)
end
function _support(ground, wind=())
    io = IOBuffer(); write(io, "DBDTCASUPPORT1\0")
    for (group, reads) in ((UInt8(1),ground),(UInt8(2),wind))
        need(length(reads) <= 256, "case support query limit")
        write(io, group, htol(UInt32(length(reads))))
        for row in reads
            need(row.operation in (:haskey,:getindex), "unknown support query operation")
            write(io, row.operation === :haskey ? UInt8(0) : UInt8(1))
            _int(io,_stamp(row.tag)); write(io,UInt8(row.present),UInt8(row.body_id !== nothing))
            if row.body_id === nothing
                write(io,zeros(UInt8,32))
            else
                need(occursin(r"^[0-9a-f]{64}$",row.body_id), "support body identity syntax")
                write(io,hex2bytes(row.body_id))
            end
            write(io,htol(row.source_line),row.valid_mask)
        end
    end
    sha256(take!(io))
end
_prepare(g::C.SourceCatalog,t;kw...) = W.prepare_catalog_issue(g,t;kw...)
_prepare(g::A.CatalogAudit,t;kw...) = V.audit_catalog_issue(g,t;kw...)
_label(g::C.SourceCatalog,t,asof;kw...) = W.evaluate_catalog_label(g,t,asof;kw...)
_label(g::A.CatalogAudit,t,asof;kw...) = V.audit_catalog_label(g,t,asof;kw...)
_locate(g::C.SourceCatalog,t) = C.locate(g,t)
_locate(g::A.CatalogAudit,t) = A.audit_locate(g,t)
function _pins(g,wind,mode)
    need(g.kind === :ground, "case ground catalog required")
    if mode === :mixed8
        need(wind isa typeof(g) && wind.kind === :wind && wind.station == g.station,
             "case mixed layout needs matching wind catalog role/station")
    end
    (; ground_selected_sha256=g.selected_sha256,ground_raw_sha256=g.raw_catalog_sha256,
       wind_selected_sha256=mode === :mixed8 ? wind.selected_sha256 : nothing,
       wind_raw_sha256=mode === :mixed8 ? wind.raw_catalog_sha256 : nothing,
       availability_kind=:assumed_completed_source_intervals,
       measured_receipts_available=false,source_bound_provenance_verified=false)
end
function _context(io,station,mode,policy,pins)
    _frame(io,station); _frame(io,String(mode))
    for value in (policy.ground_extra_ms,policy.omni_extra_ms,policy.label_extra_ms)
        _int(io,value)
    end
    for value in (pins.ground_selected_sha256,pins.ground_raw_sha256,
                  pins.wind_selected_sha256,pins.wind_raw_sha256)
        write(io,UInt8(value !== nothing))
        value === nothing || write(io,hex2bytes(value))
    end
end
function _latest_rate(g,issue,policy)
    anchor = M.issue_timing(issue,policy).latest_ground_center
    need(anchor < issue, "case history must precede issue")
    before = _locate(g,anchor-Minute(1)); after = _locate(g,anchor)
    need(before !== nothing && after !== nothing && before.valid_mask == after.valid_mask == 3,
         "accepted case latest-rate support missing")
    rate = hypot(after.values[1]-before.values[1],after.values[2]-before.values[2])
    need(isfinite(rate) && rate >= 0, "accepted case latest rate invalid")
    rate
end

"""Seal every explicitly scheduled issue using only completed feature support."""
function seal_catalog_features(g::Catalog,issues;policy=M.ReplayPolicy(),mode=:ground11,wind=nothing)
    need(mode in (:ground11,:mixed8), "case feature layout differs")
    need(policy isa M.ReplayPolicy, "case replay policy required")
    need(issues isa AbstractVector || issues isa Tuple, "explicit bounded issue collection required")
    issues isa AbstractVector && Base.require_one_based_indexing(issues)
    count = length(issues)
    need(0 < count <= MAX_ROWS, "case schedule count outside bounds")
    # Own the schedule before validation and before source access. Subsequent
    # caller changes must not replace validated issues during feature work.
    owned = Vector{DateTime}(undef,count)
    for index in 1:count
        time = issues[index]
        need(time isa DateTime && floor(time,Minute)==time, "whole-minute issue schedule required")
        owned[index] = time
    end
    need(all(owned[i] < owned[i+1] for i in 1:count-1), "case issues must increase strictly")
    pins = _pins(g,wind,mode)
    io = IOBuffer(sizehint=FEATURE_BYTES*count)
    for issue in owned
        row = _prepare(g,issue;policy,mode,wind)
        _int(io,_stamp(issue)); _int(io,_stamp(row.timing.label_assumed_ready))
        write(io,UInt8(row.accepted),_code(row.reason,FEATURE_REASONS),zeros(UInt8,6))
        if row.accepted
            now = _latest_rate(g,issue,policy)
            scale = 1+row.ground11[3]
            need(isfinite(scale) && scale >= 1 && all(isfinite,row.features), "accepted case feature values invalid")
            _float(io,now); _float(io,scale)
            for index in 1:11
                _float(io,index <= length(row.features) ? row.features[index] : 0.0)
            end
        else
            write(io,zeros(UInt8,104))
        end
        write(io,_support(row.ground_trace,row.wind_trace))
    end
    payload = String(take!(io))
    need(ncodeunits(payload) == FEATURE_BYTES*count, "case feature record width differs")
    identity = IOBuffer(); write(identity,"DBDTCASEFEATURES1\0")
    _context(identity,g.station,mode,policy,pins); _frame(identity,payload)
    CatalogFeatureSchedule(_Checked(),g.station,mode,policy,pins,count,payload,_hash(payload),
                           bytes2hex(sha256(take!(identity))))
end
function _word(value,at,width)
    result = UInt64(0)
    for offset in 0:width-1
        result |= UInt64(codeunit(value,at+offset)) << (8*offset)
    end
    result
end
_time(value,at) = EPOCH+Millisecond(reinterpret(Int64,_word(value,at,8)))
_number(value,at) = reinterpret(Float64,_word(value,at,8))
function _index(value,index,width)
    need(index isa Integer && !(index isa Bool) && 1 <= index <= length(value), "case row index outside bounds")
    width*(Int(index)-1)+1
end
function _digest(value,at)
    bytes2hex(UInt8[codeunit(value,index) for index in at:at+31])
end
"""Decode one immutable feature record. Rejections have no numerical payload."""
function feature_row(value::CatalogFeatureSchedule,index)
    at = _index(value,index,FEATURE_BYTES); bytes = value.payload
    accepted = codeunit(bytes,at+16) == 1
    reason = FEATURE_REASONS[Int(codeunit(bytes,at+17))+1]
    width = value.mode === :ground11 ? 11 : 8
    (; issue=_time(bytes,at),label_ready=_time(bytes,at+8),accepted,reason,
       raw_now=accepted ? _number(bytes,at+24) : nothing,
       scale=accepted ? _number(bytes,at+32) : nothing,
       features=accepted ? ntuple(index->_number(bytes,at+40+8*(index-1)),width) : nothing,
       support_sha256=_digest(bytes,at+128))
end

"""Audit all scheduled labels only after the schedule is immutable and readiness gates pass.

Label status is recorded even for a feature-rejected row; only the accepted
intersection is eligible for fitting. This preserves separate exclusion counts.
"""
function audit_catalog_labels(schedule::CatalogFeatureSchedule,g::Catalog;asof,label_ready_before=nothing)
    need(asof isa DateTime, "case label asof must be DateTime")
    need(label_ready_before === nothing || label_ready_before isa DateTime, "case split boundary must be DateTime")
    need(g.kind === :ground && g.station == schedule.station &&
         g.selected_sha256 == schedule.source_pins.ground_selected_sha256 &&
         g.raw_catalog_sha256 == schedule.source_pins.ground_raw_sha256, "case label source differs from sealed features")
    io = IOBuffer(sizehint=LABEL_BYTES*length(schedule))
    for index in 1:length(schedule)
        issue = _time(schedule.payload,FEATURE_BYTES*(index-1)+1)
        row = _label(g,issue,asof;policy=schedule.policy,label_ready_before)
        _int(io,_stamp(issue)); _int(io,_stamp(row.timing.label_assumed_ready))
        write(io,UInt8(row.accepted),_code(row.reason,LABEL_REASONS),zeros(UInt8,6))
        _float(io,row.accepted ? row.target : 0.0)
        write(io,_support(row.ground_trace))
    end
    payload = String(take!(io)); need(ncodeunits(payload) == LABEL_BYTES*length(schedule), "case label record width differs")
    identity = IOBuffer(); write(identity,"DBDTCASELABELS1\0",hex2bytes(schedule.schedule_sha256))
    _int(identity,_stamp(asof)); write(identity,UInt8(label_ready_before !== nothing))
    label_ready_before === nothing || _int(identity,_stamp(label_ready_before))
    _frame(identity,payload)
    CatalogLabelAudit(_Checked(),schedule.schedule_sha256,asof,label_ready_before,length(schedule),payload,
                      _hash(payload),bytes2hex(sha256(take!(identity))))
end
"""Decode one matured or explicitly excluded label record."""
function label_row(value::CatalogLabelAudit,index)
    at = _index(value,index,LABEL_BYTES); bytes = value.payload
    accepted = codeunit(bytes,at+16) == 1
    reason = LABEL_REASONS[Int(codeunit(bytes,at+17))+1]
    (;issue=_time(bytes,at),label_ready=_time(bytes,at+8),accepted,reason,
      target=accepted ? _number(bytes,at+24) : nothing,support_sha256=_digest(bytes,at+32))
end
function _matching(schedule,labels)
    need(schedule.schedule_sha256 == labels.feature_schedule_sha256 && length(schedule) == length(labels),
         "case labels do not match sealed feature schedule")
end
"""Count all scheduled cases and independent feature/label exclusions."""
function case_counts(schedule::CatalogFeatureSchedule,labels::CatalogLabelAudit)
    _matching(schedule,labels); feature=0; label=0; supervised=0
    for index in 1:length(schedule)
        f = codeunit(schedule.payload,FEATURE_BYTES*(index-1)+17)==1
        l = codeunit(labels.payload,LABEL_BYTES*(index-1)+17)==1
        feature += f; label += l; supervised += f && l
    end
    (;scheduled=length(schedule),features_accepted=feature,labels_accepted=label,supervised,
      features_rejected=length(schedule)-feature,labels_rejected=length(schedule)-label)
end
"""Return caller-owned arrays for the exactly matched accepted intersection; no fit is performed."""
function case_arrays(schedule::CatalogFeatureSchedule,labels::CatalogLabelAudit)
    counts = case_counts(schedule,labels); width = schedule.mode === :ground11 ? 11 : 8
    X = Matrix{Float64}(undef,counts.supervised,width); y = Vector{Float64}(undef,counts.supervised)
    now = similar(y); scales = similar(y); issues = Vector{DateTime}(undef,counts.supervised)
    rows = Vector{Int}(undef,counts.supervised); at=0
    for index in 1:length(schedule)
        f=feature_row(schedule,index); l=label_row(labels,index)
        need(f.issue==l.issue && f.label_ready==l.label_ready, "case joined row identity differs")
        if f.accepted && l.accepted
            at+=1; issues[at]=f.issue; rows[at]=index; y[at]=l.target; now[at]=f.raw_now; scales[at]=f.scale
            for column in 1:width; X[at,column]=f.features[column]; end
        end
    end
    (;issues,rows,features=X,targets=y,raw_now=now,scales,counts,
      availability_kind=:assumed_completed_source_intervals,measured_receipts_available=false,
      source_bound_provenance_verified=false,actual_fit_performed=false)
end
end
