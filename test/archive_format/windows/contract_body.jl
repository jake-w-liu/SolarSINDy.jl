using Dates, JSON3, Test, SHA, SolarSINDy
const ROOT = @__DIR__
const FIX = JSON3.read(read(joinpath(ROOT,"fixtures","FROZEN_INPUTS_AND_ORACLES_V1.json"), String), Any)
const RULES = JSON3.read(read(joinpath(ROOT,"fixtures","NUMERICAL_ORACLE_RULES_V1.json"), String), Any)
ROLE in ("primary", "audit") || error("role")
const P = SolarSINDy.ArchiveFormatParsers
const A = SolarSINDy.ArchiveFormatAudit
const C = SolarSINDy.ArchiveSourceCatalog
const I = SolarSINDy.ArchiveCatalogAudit
const M = SolarSINDy.CompletedMinuteReplay
const EPOCH_DAY = DateTime(2020)
const ISSUE = EPOCH_DAY + Minute(FIX["issue_minute"])
const API = ROLE == "primary" ? SolarSINDy.ArchiveCompletedWindows : SolarSINDy.ArchiveCompletedWindowAudit
const prepare = ROLE == "primary" ? API.prepare_catalog_issue : API.audit_catalog_issue
const label = ROLE == "primary" ? API.evaluate_catalog_label : API.audit_catalog_label

function catalog(text, kind; station="CMO")
    kw = (;station, product=kind === :ground ? "quasi-definitive" : "OMNI_HRO_1MIN",
           start=EPOCH_DAY, stop=EPOCH_DAY + Minute(90))
    if ROLE == "primary"
        body = kind === :ground ? P.parse_ground(text;kw...) : P.parse_wind(text;kw...)
        return C.build_catalog([body])
    else
        body = kind === :ground ? A.audit_ground(text;kw...) : A.audit_wind(text;kw...)
        return I.audit_catalog([body])
    end
end
floatbits(value) = string(reinterpret(UInt64, Float64(value));base=16,pad=16)
function encode(value)
    value isa DateTime && return Dates.format(value,dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    value isa Float64 && return floatbits(value)
    value isa NamedTuple && return Dict(String(key)=>encode(getproperty(value,key)) for key in keys(value))
    value isa Tuple && return [encode(item) for item in value]
    value isa Symbol && return String(value)
    value
end
results = Dict{String,Any}()
g = catalog(FIX["ground"], :ground)
w = catalog(FIX["wind"], :wind)
source_hashes = (g.selected_sha256,g.raw_catalog_sha256,w.selected_sha256,w.raw_catalog_sha256)

@testset "Frozen clocks and feature-only exact lookup support" begin
    for spec in FIX["policies"]
        policy = M.ReplayPolicy(;ground_extra_ms=spec["ground_extra_ms"],
            omni_extra_ms=spec["omni_extra_ms"],label_extra_ms=spec["label_extra_ms"])
        for mode in (:ground11,:mixed8)
            row = prepare(g, ISSUE;policy,mode,wind=mode === :ground11 ? :forbidden : w)
            @test row.accepted && row.reason === :complete
            @test row.timing.latest_ground_center == EPOCH_DAY+Minute(spec["ground_last_minute"])
            @test row.timing.latest_omni_arrival_start == EPOCH_DAY+Minute(spec["wind_minute"])
            @test row.timing.label_assumed_ready == EPOCH_DAY+Millisecond(spec["label_ready_ms_from_epoch_day"])
            @test row.timing.first_scheduled_feedback == EPOCH_DAY+Minute(spec["feedback_minute"])
            @test row.timing.sufficient_phase_period_minutes == spec["phase_count"]
            @test length(row.ground_trace) == FIX["ground_feature_native_queries"]
            @test minimum(read.tag for read in row.ground_trace) == EPOCH_DAY+Minute(spec["ground_first_minute"])
            @test maximum(read.tag for read in row.ground_trace) == EPOCH_DAY+Minute(spec["ground_last_minute"])
            @test length(unique(read.tag for read in row.ground_trace)) == FIX["maximum_ground_unique_tags"]
            @test isimmutable(row.features) && isimmutable(row.ground_trace)
            @test all(read.present for read in row.ground_trace)
            @test !row.source_pins.measured_receipts_available && !row.source_pins.source_bound_provenance_verified
            @test row.source_pins.availability_kind === :assumed_completed_source_intervals
            @test row.source_pins.ground_selected_sha256 == g.selected_sha256
            @test row.source_pins.ground_raw_sha256 == g.raw_catalog_sha256
            for (actual,expected) in zip(row.ground11,FIX["ground11_expected"])
                @test isapprox(actual,expected;atol=RULES["ground11_absolute_tolerance"],rtol=0)
            end
            if mode === :mixed8
                @test floatbits.(row.features) == Tuple(FIX["mixed8_expected_bits"])
                @test length(row.wind_trace) == FIX["wind_native_queries"]
                @test all(read.tag == EPOCH_DAY+Minute(spec["wind_minute"]) for read in row.wind_trace)
                @test row.source_pins.wind_selected_sha256 == w.selected_sha256
            else
                @test isempty(row.wind_trace)
                @test row.source_pins.wind_selected_sha256 === nothing
                @test row.features == row.ground11
            end
            results[spec["name"]*"_"*String(mode)] = encode(row)
        end
        ready = EPOCH_DAY+Millisecond(spec["label_ready_ms_from_epoch_day"])
        for asof in (ready-Millisecond(1),ready)
            result = label(g,ISSUE,asof;policy)
            if asof < ready
                @test !result.accepted && result.reason === :label_not_ready
                @test isempty(result.ground_trace) && result.target === nothing
            else
                @test result.accepted && result.reason === :complete
                @test floatbits(result.target) == FIX["target_expected_bits"]
                @test length(result.ground_trace) == FIX["target_native_queries"]
                @test minimum(read.tag for read in result.ground_trace) == ISSUE
                @test maximum(read.tag for read in result.ground_trace) == ISSUE+Minute(30)
            end
            results[spec["name"]*"_label_"*string(asof==ready)] = encode(result)
        end
        for boundary in (ready,ready+Millisecond(1))
            result = label(g,ISSUE,ready;policy,label_ready_before=boundary)
            if boundary == ready
                @test !result.accepted && result.reason === :split_not_strictly_ready
                @test isempty(result.ground_trace)
            else
                @test result.accepted
            end
            results[spec["name"]*"_split_"*string(boundary==ready)] = encode(result)
        end
    end
end
@testset "Future labels cannot determine feature eligibility or values" begin
    plain = catalog(FIX["ground_no_future_spike"],:ground)
    original = prepare(g,ISSUE;mode=:mixed8,wind=w)
    changed = prepare(plain,ISSUE;mode=:mixed8,wind=w)
    @test original.accepted && changed.accepted
    @test reinterpret(UInt64,collect(original.features)) == reinterpret(UInt64,collect(changed.features))
    @test reinterpret(UInt64,collect(original.ground11)) == reinterpret(UInt64,collect(changed.ground11))
    @test all(read.tag < ISSUE for read in original.ground_trace)
    target = label(plain,ISSUE,ISSUE+Minute(31))
    @test target.accepted && floatbits(target.target) == FIX["target_without_spike_bits"]
    results["future_changed_features"] = encode(changed)
    results["future_changed_label"] = encode(target)
end
@testset "Unavailable original vectors and exact exclusion traces" begin
    for key in ("ground_missing_history","ground_partial_history")
        current = catalog(FIX[key],:ground)
        row = prepare(current,ISSUE;mode=:mixed8,wind=w)
        @test !row.accepted && row.reason === :missing_ground_history
        @test row.features === nothing && isempty(row.wind_trace)
        @test length(row.ground_trace) == FIX["missing_history_queries_before_reject"]
        @test !last(row.ground_trace).present && last(row.ground_trace).tag == EPOCH_DAY+Minute(17)
        results[key] = encode(row)
    end
    badwind = catalog(FIX["wind_partial_anchor"],:wind)
    row = prepare(g,ISSUE;mode=:mixed8,wind=badwind)
    @test !row.accepted && row.reason === :missing_omni_history
    @test length(row.wind_trace)==1 && !only(row.wind_trace).present
    @test prepare(g,ISSUE;mode=:ground11,wind=badwind).accepted
    results["wind_partial"] = encode(row)
    missingtarget = catalog(FIX["ground_missing_target"],:ground)
    @test prepare(missingtarget,ISSUE).accepted
    target = label(missingtarget,ISSUE,ISSUE+Minute(31))
    @test !target.accepted && target.reason === :missing_ground_target
    @test length(target.ground_trace) == FIX["missing_target_queries_before_reject"]
    results["missing_target"] = encode(target)
end
@testset "Window bounds, query caps and forbidden scan behavior" begin
    fresh_view() = ROLE == "primary" ? API._Window{2}(g,EPOCH_DAY+Minute(1),EPOCH_DAY+Minute(31),NamedTuple[]) :
                                      API._AuditWindow{2}(g,EPOCH_DAY+Minute(1),EPOCH_DAY+Minute(31),NamedTuple[])
    view_trace(v) = ROLE == "primary" ? v.trace : v.reads
    for time in (EPOCH_DAY,EPOCH_DAY+Minute(32),EPOCH_DAY+Minute(1)+Millisecond(1),true)
        view=fresh_view();trace=view_trace(view)
        @test_throws ArgumentError haskey(view,time)
        @test isempty(trace)
        view=fresh_view();trace=view_trace(view)
        @test_throws ArgumentError view[time]
        @test isempty(trace)
    end
    view=fresh_view();trace=view_trace(view)
    @test_throws ArgumentError iterate(view)
    @test_throws ArgumentError length(view)
    for _ in 1:FIX["per_window_maximum_query_count"]
        @test haskey(view,EPOCH_DAY+Minute(2))
    end
    @test_throws ArgumentError haskey(view,EPOCH_DAY+Minute(2))
    @test length(trace)==FIX["per_window_maximum_query_count"]
    @test_throws ArgumentError prepare(g,ISSUE+Millisecond(1))
    @test_throws ArgumentError prepare(g,ISSUE;mode=:unknown)
    @test_throws ArgumentError prepare(g,ISSUE;mode=:mixed8,wind=:forbidden)
    @test_throws ArgumentError prepare(w,ISSUE)
    @test_throws ArgumentError label(g,ISSUE,ISSUE+Minute(31);label_ready_before=true)
    @test source_hashes == (g.selected_sha256,g.raw_catalog_sha256,w.selected_sha256,w.raw_catalog_sha256)
end
result_path=joinpath(mktempdir(;prefix="ss-archive-window-contract-"),"result.json")
bytes = Vector{UInt8}(codeunits(JSON3.write(results)))
file = Base.Filesystem.open(result_path,Base.JL_O_WRONLY|Base.JL_O_CREAT|Base.JL_O_EXCL,0o600)
try
    write(file,bytes)==length(bytes) || error("partial result write")
finally
    close(file)
end
read(result_path)==bytes || error("exclusive result readback differs")
println("catalog-window-contract-complete role=",ROLE," cases=",length(results))
