module CMEGSMMinuteLabels

using Dates, SHA, Statistics, CSV, JSON3
include(joinpath(@__DIR__, "bootstrap_indices.jl"))
const Bootstrap = CMEGSMBootstrap
const PROTOCOL_SHA = "df44b33c835f1df7479d11e672676054db9c0925a829fee7c2059e481cd58276"
const POPULATION_SHA = "b81228b4b0954dfec881d5a57ae9f7a76bb7e9a73aeee84bf17a0dcfdaf99fb9"
const PROVIDER_SHA = "4ab51b795b549e7a26c0a8f17206abcad0c493ddd7255f62549335d0300d0d24"
const COVERAGES = ((9, 10), (3, 4), (19, 20))
const MINUTE = Int64(60000)
const HALF_SUPPORT = Int64(30000)
const MAX_CHUNK_RECORDS = 100000
const MAX_CHUNKS = 20000
const MAX_EVENT_SLOTS = 100000
const MAX_TOTAL_SLOTS = 2000000
const EPOCH = DateTime(1970, 1, 1)
const NATIVE_FILL32 = reinterpret(Float32, UInt32(0xf2fc6f7c))
const NATIVE_PAD32 = reinterpret(Float32, UInt32(0xf149f2ca))
const SOURCE_SHA = bytes2hex(sha256(read(@__FILE__)))

need(value, message) = value || throw(ArgumentError(message))
hex64(x) = x isa AbstractString && occursin(r"^[0-9a-f]{64}$", x)
function source_guard()
    need(bytes2hex(sha256(read(@__FILE__))) == SOURCE_SHA, "loaded/disk GSM core differs")
    Bootstrap.source_guard()
end
function millis(value)
    need(value isa Integer && !(value isa Bool) && typemin(Int64) <= value <= typemax(Int64), "integer UTC milliseconds required")
    Int64(value)
end
function utc_millis(text::AbstractString)
    need(occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d(?::\d\d(?:\.\d{1,3})?)?(?:Z|\+00:00)$", text), "strict UTC clock required")
    body = endswith(text, "Z") ? chop(text) : chop(text; tail=6)
    Dates.value(DateTime(body) - EPOCH)
end
function time_digest(values)
    io = IOBuffer()
    for value in values
        write(io, string(value), '\n')
    end
    bytes2hex(sha256(take!(io)))
end

"Time-only observations. A change bracket does not choose a phase boundary."
struct TimeAxis
    times::Vector{Int64}
    start_ms::Int64
    stop_ms::Int64
    raw_id::String
    raw_sha256::String
    time_sha256::String
    runs::Vector{NamedTuple}
    changes::Vector{NamedTuple}
    parameters::NamedTuple
end

function time_phase_runs(times_ms, chunk_start_ms, chunk_stop_ms; raw_id, raw_sha256,
        source_group, metadata_sha256, dataset="WI_H0_MFI", field="BGSM",
        time_variable="Epoch", quality_variable="NUM_PTS", frame="GSM Cartesian", units="nT")
    first = millis(chunk_start_ms); stop = millis(chunk_stop_ms)
    need(first < stop, "empty Time-only chunk domain")
    need(raw_id isa AbstractString && 1 <= ncodeunits(raw_id) <= 256 && hex64(raw_sha256), "raw chunk identity required")
    need(source_group isa AbstractString && 1 <= ncodeunits(source_group) <= 256 && hex64(metadata_sha256), "source group and metadata byte pin required")
    need(dataset == "WI_H0_MFI" && field == "BGSM" && time_variable == "Epoch" &&
        quality_variable == "NUM_PTS" && frame == "GSM Cartesian" && units == "nT", "source dataset/frame/units/parameters differ")
    need(length(times_ms) <= MAX_CHUNK_RECORDS, "Time-only record bound")
    times = Int64[millis(value) for value in times_ms]
    need(all(first <= value < stop for value in times), "Time outside frozen chunk domain")
    need(issorted(times) && allunique(times), "duplicate or nonincreasing Time records")
    runs = NamedTuple[]; changes = NamedTuple[]
    if !isempty(times)
        begin_index = 1; phase = mod(times[1], MINUTE)
        for index in 2:length(times)
            next_phase = mod(times[index], MINUTE)
            if next_phase != phase
                push!(runs, (; first_index=begin_index, last_index=index-1, phase_ms=phase,
                    first_observed_ms=times[begin_index], last_observed_ms=times[index-1]))
                push!(changes, (; old_phase_ms=phase, new_phase_ms=next_phase,
                    last_old_ms=times[index-1], first_new_ms=times[index]))
                begin_index = index; phase = next_phase
            end
        end
        push!(runs, (; first_index=begin_index, last_index=length(times), phase_ms=phase,
            first_observed_ms=times[begin_index], last_observed_ms=times[end]))
    end
    parameters = (; source_group=String(source_group), metadata_sha256=String(metadata_sha256),
        dataset=String(dataset), field=String(field), time_variable=String(time_variable),
        quality_variable=String(quality_variable), frame=String(frame), units=String(units))
    TimeAxis(times, first, stop, String(raw_id), String(raw_sha256), time_digest(times), runs, changes, parameters)
end

function axis_guard(axis::TimeAxis)
    need(time_digest(axis.times) == axis.time_sha256, "Time-only axis changed")
    rebuilt = time_phase_runs(axis.times, axis.start_ms, axis.stop_ms; raw_id=axis.raw_id, raw_sha256=axis.raw_sha256, axis.parameters...)
    need(rebuilt.runs == axis.runs && rebuilt.changes == axis.changes, "Time-only regime metadata changed")
end

struct PhaseDomain
    start_ms::Int64
    stop_ms::Int64
    phase_ms::Int64
    function PhaseDomain(start_ms, stop_ms, phase_ms)
        a = millis(start_ms); b = millis(stop_ms); p = millis(phase_ms)
        need(a < b && 0 <= p < MINUTE, "phase domain or phase outside bounds")
        new(a, b, p)
    end
end

struct PhaseSchedule
    axis::TimeAxis
    domains::Vector{PhaseDomain}
    resolved::Bool
    reason::Symbol
    freeze_reference::String
end

"Freeze domains using Time only. Explicit domains are required after an observed phase change."
function freeze_phase_domains(axis::TimeAxis; domains=nothing, freeze_reference=axis.time_sha256)
    axis_guard(axis); need(hex64(freeze_reference), "Time-only freeze reference required")
    if domains === nothing
        if isempty(axis.runs)
            return PhaseSchedule(axis, PhaseDomain[], false, :empty_time_axis, String(freeze_reference))
        elseif length(axis.runs) > 1
            return PhaseSchedule(axis, PhaseDomain[], false, :phase_change_boundary_unresolved, String(freeze_reference))
        end
        domains = [PhaseDomain(axis.start_ms, axis.stop_ms, only(axis.runs).phase_ms)]
    end
    need(domains isa AbstractVector && 1 <= length(domains) <= MAX_CHUNK_RECORDS && all(d -> d isa PhaseDomain, domains), "phase domain records required")
    values = PhaseDomain[domain for domain in domains]
    need(values[1].start_ms == axis.start_ms && values[end].stop_ms == axis.stop_ms, "phase domains must cover frozen chunk domain")
    need(all(values[index].stop_ms == values[index+1].start_ms for index in 1:length(values)-1), "phase domains overlap or leave a gap")
    cursor = 1
    for stamp in axis.times
        while stamp >= values[cursor].stop_ms
            cursor += 1
        end
        domain = values[cursor]
        need(domain.start_ms <= stamp < domain.stop_ms && mod(stamp, MINUTE) == domain.phase_ms, "Time-only phase differs from frozen domain")
    end
    # This is a numerical schedule, not a source-admission authority. The outer
    # Time-only reader must retain the pre-value boundary decision and receipt.
    PhaseSchedule(axis, values, true, :resolved, String(freeze_reference))
end

function schedule_guard(schedule::PhaseSchedule)
    axis_guard(schedule.axis)
    if schedule.resolved
        checked = freeze_phase_domains(schedule.axis; domains=schedule.domains, freeze_reference=schedule.freeze_reference)
        need(checked.resolved && schedule.reason == :resolved, "resolved phase schedule differs")
    else
        need(isempty(schedule.domains) && schedule.reason in (:empty_time_axis, :phase_change_boundary_unresolved), "unresolved phase schedule differs")
    end
end

"Full centred supports only; stop is exclusive but a support may end exactly at stop."
function expected_centres(start_ms, stop_ms, schedules::AbstractVector{PhaseSchedule})
    a = millis(start_ms); b = millis(stop_ms); need(a < b, "empty MO window")
    need(length(schedules) <= MAX_CHUNKS, "source chunk bound")
    low = Int128(a) + HALF_SUPPORT; high = Int128(b) - HALF_SUPPORT
    low > high && return (; resolved=true, centres=Int64[], reasons=Symbol[])
    ordered = sort(collect(schedules); by=s -> s.axis.start_ms)
    need(all(ordered[i].axis.stop_ms <= ordered[i+1].axis.start_ms for i in 1:length(ordered)-1), "overlapping source chunk domains")
    cursor = low; centres = Int64[]; reasons = Symbol[]
    for schedule in ordered
        axis = schedule.axis
        left = max(low, Int128(axis.start_ms)); right = min(high + 1, Int128(axis.stop_ms))
        left >= right && continue
        schedule_guard(schedule)
        left > cursor && push!(reasons, :missing_time_domain)
        cursor = max(cursor, right)
        if !schedule.resolved
            push!(reasons, schedule.reason)
            continue
        end
        for domain in schedule.domains
            lo = max(low, Int128(domain.start_ms)); hi = min(high, Int128(domain.stop_ms)-1)
            lo > hi && continue
            first = lo + mod(Int128(domain.phase_ms) - lo, MINUTE)
            first > hi && continue
            count = Int(div(hi-first, MINUTE)) + 1
            need(length(centres)+count <= MAX_EVENT_SLOTS, "event slot bound")
            append!(centres, (Int64(first + Int128(index)*MINUTE) for index in 0:count-1))
        end
    end
    cursor <= high && push!(reasons, :missing_time_domain)
    sort!(centres); need(allunique(centres), "duplicate expected centre")
    (; resolved=isempty(reasons), centres, reasons=unique(reasons))
end

struct QCChunk
    schedule::PhaseSchedule
    bz::Vector{Union{Nothing,Float64}}
    reasons::Vector{Vector{Symbol}}
    numerical_sha256::String
end

function qc_digest(schedule, bz, reasons)
    io = IOBuffer()
    write(io, repr((schedule.axis.raw_id,schedule.axis.raw_sha256,schedule.axis.time_sha256,
        schedule.axis.parameters,schedule.freeze_reference,
        Tuple((d.start_ms,d.stop_ms,d.phase_ms) for d in schedule.domains))), '\n')
    for (value,why) in zip(bz,reasons)
        write(io, value === nothing ? "missing" : repr(value), ':', join(string.(why),','), '\n')
    end
    bytes2hex(sha256(take!(io)))
end

"Exactly aligned Epoch arrays; invalid vectors remain explicit records. No nearest-time join."
function minute_qc(schedule::PhaseSchedule, bgsm_epochs, vectors, point_epochs, counts)
    schedule_guard(schedule)
    need(schedule.resolved, "freeze Time-only phase domains before vector access")
    times = schedule.axis.times
    need(length(bgsm_epochs) == length(point_epochs) == length(times), "Epoch geometry differs")
    need(all(millis(bgsm_epochs[i]) == times[i] == millis(point_epochs[i]) for i in eachindex(times)), "BGSM/NUM_PTS must use the exact same Epoch")
    need(length(vectors) == length(counts) == length(times), "vector/count record geometry differs")
    bz = Union{Nothing,Float64}[]; reasons = Vector{Symbol}[]
    for index in eachindex(times)
        why = Symbol[]; vector = vectors[index]; count = counts[index]
        if !(vector isa AbstractVector || vector isa Tuple) || length(vector) != 3
            push!(why, :vector_dimension)
        else
            for component in 1:3
                value = vector[component]
                tag = if !(value isa Real) || value isa Bool
                    :not_real
                # WI_H0_MFI BGSM is REAL4. Lossless widening does not recover
                # the decimal Float64 literal. Accept each exact representation
                # without rounding a neighbouring input into a marker.
                elseif value == -1e31 || value == Float64(NATIVE_FILL32)
                    :fill
                elseif value == -1e30 || value == Float64(NATIVE_PAD32)
                    :pad
                elseif !isfinite(value)
                    :nonfinite
                elseif !(-65534 <= value <= 65534)
                    :range
                else
                    nothing
                end
                tag === nothing || push!(why, Symbol("component$(component)_$(tag)"))
            end
        end
        if !(count isa Integer) || count isa Bool
            push!(why, :point_count_not_integer)
        elseif !(1 <= count <= 1500)
            push!(why, count == -1 ? :point_count_fill : count == 0 ? :point_count_zero : :point_count_range)
        end
        push!(bz, isempty(why) ? Float64(vector[3]) : nothing)
        push!(reasons, why)
    end
    QCChunk(schedule, bz, reasons, qc_digest(schedule,bz,reasons))
end

struct MOEvent
    id::String
    start_ms::Int64
    stop_ms::Int64
    field_nt::Union{Nothing,Float64}
    speed_km_s::Union{Nothing,Float64}
end
function mo_event(id, start_ms, stop_ms, field, speed)
    need(id isa AbstractString && 1 <= ncodeunits(id) <= 128, "event ID")
    a = millis(start_ms); b = millis(stop_ms); need(a < b, "MO interval")
    covariate(x) = x === nothing || ismissing(x) ? nothing : begin
        need(x isa Real && !(x isa Bool), "numeric catalogue covariate required")
        value = Float64(x); isfinite(value) ? value : nothing
    end
    MOEvent(String(id), a, b, covariate(field), covariate(speed))
end

struct Population
    events::Tuple{Vararg{MOEvent}}
    generated::Bool
    population_sha256::String
    provider_sha256::String
end

function population_guard(population::Population)
    reference = Bootstrap.load_population()
    events = population.events
    need(length(events) == 551 && Tuple(e.id for e in events) == reference.ids, "all551 original IDs and order required")
    need(Tuple(year(EPOCH+Millisecond(e.start_ms)) for e in events) == reference.event_years, "frozen MO-start year universe differs")
    need(population.population_sha256 == POPULATION_SHA && population.provider_sha256 == PROVIDER_SHA, "population provenance differs")
    reference
end

"Generated inputs exercise numerics only; this constructor grants no source admission."
function generated_population(events::AbstractVector{MOEvent})
    value = Population(Tuple(events), true, POPULATION_SHA, PROVIDER_SHA)
    population_guard(value); value
end

function finite_catalogue(text)
    ismissing(text) && return nothing
    value = tryparse(Float64, String(text))
    value === nothing || !isfinite(value) ? nothing : value
end

"Load only the frozen catalogue covariates/windows, never any GSM data."
function load_frozen_events(population_path, provider_csv_path)
    popbytes = Bootstrap.bounded_bytes(population_path, 262144)
    csvbytes = Bootstrap.bounded_bytes(provider_csv_path, 16*1024^2)
    need(bytes2hex(sha256(popbytes)) == POPULATION_SHA && bytes2hex(sha256(csvbytes)) == PROVIDER_SHA, "exact frozen population/provider bytes required")
    data = JSON3.read(popbytes)
    need(String(data.schema) == "frozen_cme_Wind_MO_population_before_GSM_reanalysis_v1" && data.complete === true && length(data.rows) == 551, "frozen population schema")
    selected = Dict{String,Any}()
    wanted = Set(String(row.id) for row in data.rows)
    for row in CSV.File(IOBuffer(csvbytes); types=String, stringtype=String)
        id = String(row.icmecat_id)
        if id in wanted
            need(!haskey(selected, id), "duplicate frozen catalogue ID")
            selected[id] = row
        end
    end
    need(Set(keys(selected)) == wanted, "missing frozen catalogue ID")
    events = MOEvent[]
    for frozen in data.rows
        id = String(frozen.id); row = selected[id]
        a = utc_millis(String(frozen.start_inclusive)); b = utc_millis(String(frozen.stop_exclusive))
        need(String(row.sc_insitu) == "Wind" && utc_millis(String(row.mo_start_time)) == a && utc_millis(String(row.mo_end_time)) == b, "catalogue ID/window join differs")
        speed = finite_catalogue(row.mo_speed_mean)
        frozen_speed = frozen.catalogue_mean_speed_km_s
        need((frozen_speed === nothing && speed === nothing) || (frozen_speed isa Real && speed !== nothing && Float64(frozen_speed) == speed), "frozen mean-speed join differs")
        push!(events, mo_event(id, a, b, finite_catalogue(row.mo_bmean), speed))
    end
    value = Population(Tuple(events), false, POPULATION_SHA, PROVIDER_SHA)
    population_guard(value); value
end

function coverage_admitted(valid::Integer, expected::Integer, ratio)
    need(!(valid isa Bool) && !(expected isa Bool) && 0 <= valid <= expected, "coverage counts")
    need(ratio in COVERAGES, "only prespecified coverage ratios allowed")
    valid >= 2 && expected >= 2 && Int128(valid)*ratio[2] >= Int128(expected)*ratio[1]
end

struct EventLabel
    id::String
    expected_count::Union{Nothing,Int}
    valid_count::Int
    peak_bs_nt::Union{Nothing,Float64}
    admitted::NTuple{3,Bool}
    reasons::Vector{Symbol}
    slots::Vector{NamedTuple}
    extrema_supports::Vector{NamedTuple}
    source_chunks::Vector{NamedTuple}
end

function label_event(event::MOEvent, schedules::AbstractVector{PhaseSchedule}, chunks::AbstractVector{QCChunk})
    slots = expected_centres(event.start_ms, event.stop_ms, schedules)
    sources = [(; raw_id=s.axis.raw_id, raw_sha256=s.axis.raw_sha256, time_sha256=s.axis.time_sha256,
        time_freeze_reference=s.freeze_reference, declared_source_parameters=s.axis.parameters, resolved=s.resolved,
        domains=[(; start_ms=d.start_ms, stop_ms=d.stop_ms, phase_ms=d.phase_ms) for d in s.domains],
        phase_changes=copy(s.axis.changes)) for s in schedules if s.axis.start_ms < event.stop_ms && event.start_ms < s.axis.stop_ms]
    if !slots.resolved
        known = [(; centre_ms=stamp, valid=false, bz_nt=nothing,
            reasons=[:not_evaluated_unresolved_time_domain], raw_id=nothing, raw_sha256=nothing) for stamp in slots.centres]
        return EventLabel(event.id, nothing, 0, nothing, (false,false,false), slots.reasons, known, NamedTuple[], sources)
    end
    lookup = Dict{Int64,Tuple{QCChunk,Int}}()
    for chunk in chunks
        axis = chunk.schedule.axis
        (axis.stop_ms <= event.start_ms || axis.start_ms >= event.stop_ms) && continue
        schedule_guard(chunk.schedule)
        need(chunk.schedule.resolved && length(chunk.bz) == length(chunk.reasons) == length(chunk.schedule.axis.times), "QC chunk geometry or unresolved schedule")
        need(qc_digest(chunk.schedule,chunk.bz,chunk.reasons) == chunk.numerical_sha256, "QC numerical records changed")
        for (index, stamp) in enumerate(chunk.schedule.axis.times)
            if event.start_ms <= stamp < event.stop_ms
                need(!haskey(lookup, stamp), "duplicate provider records across chunks")
                lookup[stamp] = (chunk, index)
            end
        end
    end
    records = NamedTuple[]; values = Float64[]; valid_stamps = Int64[]
    for stamp in slots.centres
        if !haskey(lookup, stamp)
            owner = only(s for s in schedules if s.axis.start_ms <= stamp < s.axis.stop_ms)
            push!(records, (; centre_ms=stamp, valid=false, bz_nt=nothing, reasons=[:missing], raw_id=owner.axis.raw_id, raw_sha256=owner.axis.raw_sha256))
        else
            chunk, index = lookup[stamp]; bz = chunk.bz[index]; why = chunk.reasons[index]
            need((bz === nothing) == !isempty(why) && (bz === nothing || (isfinite(bz) && -65534 <= bz <= 65534)), "QC record state changed")
            axis = chunk.schedule.axis
            need(any(s -> s.axis.raw_id == axis.raw_id && s.axis.raw_sha256 == axis.raw_sha256 &&
                s.axis.time_sha256 == axis.time_sha256 && s.axis.parameters == axis.parameters &&
                s.freeze_reference == chunk.schedule.freeze_reference && s.domains == chunk.schedule.domains, schedules), "QC chunk differs from expected Time-only schedule")
            push!(records, (; centre_ms=stamp, valid=bz !== nothing, bz_nt=bz, reasons=copy(why), raw_id=axis.raw_id, raw_sha256=axis.raw_sha256))
            if bz !== nothing
                push!(values, bz); push!(valid_stamps, stamp)
            end
        end
    end
    n = length(values); expected = length(records)
    admitted = ntuple(i -> coverage_admitted(n, expected, COVERAGES[i]), 3)
    why = Symbol[]
    expected < 2 && push!(why, :fewer_than_two_expected_slots)
    n < 2 && push!(why, :fewer_than_two_valid_vectors)
    for (index, ratio) in enumerate(COVERAGES)
        admitted[index] || push!(why, Symbol("coverage_$(ratio[1])_$(ratio[2])_not_admitted"))
    end
    minimum_bz = isempty(values) ? nothing : minimum(values)
    peak = minimum_bz === nothing ? nothing : max(0.0, -minimum_bz)
    supports = minimum_bz === nothing ? NamedTuple[] : [(; centre_ms=stamp,
        support_start_ms=stamp-HALF_SUPPORT, support_stop_ms=stamp+HALF_SUPPORT)
        for (stamp,bz) in zip(valid_stamps,values) if bz == minimum_bz]
    EventLabel(event.id, expected, n, peak, admitted, why, records, supports, sources)
end

function label_population(population::Population, schedules::AbstractVector{PhaseSchedule}, chunks::AbstractVector{QCChunk})
    source_guard(); population_guard(population)
    need(length(schedules) <= MAX_CHUNKS && length(chunks) <= MAX_CHUNKS, "source chunk bound")
    need(allunique(s.axis.raw_id for s in schedules) && allunique(c.schedule.axis.raw_id for c in chunks), "duplicate chunk identity")
    # Count all expected support slots before materializing population records.
    total = 0
    for event in population.events
        count = expected_centres(event.start_ms, event.stop_ms, schedules)
        total += length(count.centres); need(total <= MAX_TOTAL_SLOTS, "population slot allocation bound")
    end
    labels = [label_event(event, schedules, chunks) for event in population.events]
    need(length(labels) == 551 && all(labels[i].id == population.events[i].id for i in 1:551), "population disposition order differs")
    (; labels, generated_inputs_only=population.generated, source_authority_granted=false,
        scientific_qualification=false, protocol_sha256=PROTOCOL_SHA, total_expected_slots=total)
end

function checked_weights(values, weights)
    need(length(values) == length(weights) <= 551, "statistic geometry")
    need(all(value -> value isa Real && !(value isa Bool) && isfinite(value), values), "finite statistic values required")
    need(all(w -> w isa Integer && !(w isa Bool) && 0 <= w <= 31, weights), "whole-year multiplicity required")
    Float64.(values), Int.(weights)
end

function weighted_pearson(x, y, weights)
    values, w = checked_weights(x, weights); other, _ = checked_weights(y, weights)
    total = sum(w)
    total < 2 && return (; valid=false, value=nothing, reason=:insufficient_support)
    selected = findall(>(0), w)
    sx = maximum(abs(values[i]) for i in selected); sy = maximum(abs(other[i]) for i in selected)
    (sx == 0 || sy == 0) && return (; valid=false, value=nothing, reason=:zero_variance)
    mx = sum(w[i]*(values[i]/sx) for i in selected)/total
    my = sum(w[i]*(other[i]/sy) for i in selected)/total
    vx = sum(w[i]*(values[i]/sx-mx)^2 for i in selected)
    vy = sum(w[i]*(other[i]/sy-my)^2 for i in selected)
    (vx == 0 || vy == 0) && return (; valid=false, value=nothing, reason=:zero_variance)
    covariance = sum(w[i]*(values[i]/sx-mx)*(other[i]/sy-my) for i in selected)
    value = covariance/sqrt(vx*vy)
    need(isfinite(value) && abs(value) <= 1+16eps(Float64), "nonfinite or invalid correlation")
    (; valid=true, value=clamp(value,-1.0,1.0), reason=:valid)
end

"Midranks of the replicated draw, including ties introduced by year multiplicity."
function weighted_mid_ranks(values, weights)
    x, w = checked_weights(values, weights)
    order = sort(findall(>(0), w); by=i -> x[i], alg=MergeSort)
    ranks = zeros(Float64, length(x)); occupied = 0; cursor = 1
    while cursor <= length(order)
        stop = cursor
        while stop < length(order) && x[order[stop+1]] == x[order[cursor]]
            stop += 1
        end
        width = sum(w[order[i]] for i in cursor:stop)
        rank = occupied + (width+1)/2
        for index in cursor:stop
            ranks[order[index]] = rank
        end
        occupied += width; cursor = stop+1
    end
    ranks
end

function paired_correlations(field, speed, target, weights=ones(Int,length(target)))
    need(length(field) == length(speed) == length(target) == length(weights), "paired statistic cohort differs")
    pf = weighted_pearson(field,target,weights); ps = weighted_pearson(speed,target,weights)
    rf = weighted_mid_ranks(field,weights); rs = weighted_mid_ranks(speed,weights); rt = weighted_mid_ranks(target,weights)
    sf = weighted_pearson(rf,rt,weights); ss = weighted_pearson(rs,rt,weights)
    pc = pf.valid && ps.valid ? pf.value-abs(ps.value) : nothing
    sc = sf.valid && ss.valid ? sf.value-abs(ss.value) : nothing
    (; pearson_field=pf, pearson_speed=ps, spearman_field=sf, spearman_speed=ss,
        pearson_contrast=pc, spearman_contrast=sc)
end

function matched_inputs(population::Population, labels::AbstractVector{EventLabel}, coverage)
    reference = population_guard(population)
    need(length(labels) == 551 && all(labels[i].id == population.events[i].id for i in 1:551), "all551 label dispositions required in original order")
    position = findfirst(==(coverage), COVERAGES); need(position !== nothing, "undeclared coverage")
    indices = Int[]; field = Float64[]; speed = Float64[]; target = Float64[]; years = Int[]
    for (index,event) in enumerate(population.events)
        label = labels[index]
        admitted = label.expected_count !== nothing && coverage_admitted(label.valid_count,label.expected_count,coverage)
        need(label.admitted[position] == admitted, "label coverage flags differ from counts")
        if admitted && event.field_nt !== nothing && event.speed_km_s !== nothing && event.field_nt > 0 && event.speed_km_s > 0
            need(label.peak_bs_nt !== nothing && isfinite(label.peak_bs_nt) && label.peak_bs_nt >= 0, "admitted label value")
            push!(indices,index); push!(field,event.field_nt); push!(speed,event.speed_km_s)
            push!(target,label.peak_bs_nt); push!(years,reference.event_years[index])
        end
    end
    (; indices,field,speed,target,years,year_universe=collect(reference.years),
        admitted_labels=count(label -> label.admitted[position], labels))
end

function paired_statistics(population::Population, labels::AbstractVector{EventLabel}; coverage=(9,10))
    input = matched_inputs(population,labels,coverage)
    stats = paired_correlations(input.field,input.speed,input.target)
    (; input..., statistics=stats, full_population_gate=input.admitted_labels >= 496,
        matched_population_gate=length(input.indices) >= 492, coverage)
end

function year_weights(event_years, year_universe, selected_indices)
    need(issorted(year_universe) && allunique(year_universe) && 1 <= length(year_universe) <= 31, "fixed sorted year universe required")
    need(length(selected_indices) == length(year_universe), "whole fixed year-slot universe required")
    frequency = zeros(Int,length(year_universe))
    for index in selected_indices
        need(index isa Integer && !(index isa Bool) && 1 <= index <= length(year_universe), "frozen year index outside universe")
        frequency[index] += 1
    end
    lookup = Dict(value=>index for (index,value) in enumerate(year_universe))
    need(all(haskey(lookup,value) for value in event_years), "event year outside frozen universe")
    [frequency[lookup[value]] for value in event_years]
end

function percentile_type7(values, probability::Rational)
    need(1 <= length(values) <= 10000 && all(isfinite,values) && 0 <= probability <= 1, "finite bounded percentile values/probability required")
    x = sort(Float64.(values)); length(x) == 1 && return only(x)
    whole, remainder = divrem((length(x)-1)*numerator(probability),denominator(probability))
    j = Int(whole)+1; fraction = remainder/denominator(probability)
    (1-fraction)*x[j]+fraction*x[min(j+1,length(x))]
end

function contrast_summary(draws)
    need(length(draws) == 10000, "retain exactly10000 declared draws")
    pearson = Float64[]; spearman = Float64[]; invalid = 0
    for (index,draw) in enumerate(draws)
        need(draw.draw == index && draw.valid isa Bool, "draw order or validity flag differs")
        p = draw.pearson_contrast; s = draw.spearman_contrast
        need((p === nothing || (p isa Real && isfinite(p))) && (s === nothing || (s isa Real && isfinite(s))), "invalid contrast encoding")
        if draw.valid
            need(p !== nothing && s !== nothing && isempty(draw.reasons), "valid draw statistics differ")
            push!(pearson,p); push!(spearman,s)
        else
            need(!isempty(draw.reasons) && (p === nothing || s === nothing), "invalid draw must retain its actual missing statistic and reason")
            invalid += 1
        end
    end
    interval(values) = isempty(values) ? nothing : (percentile_type7(values,1//40),percentile_type7(values,39//40))
    (; valid_count=10000-invalid, invalid_count=invalid, uncertainty_gate=invalid <= 100,
        pearson_interval=interval(pearson), spearman_interval=interval(spearman))
end

"Consume frozen indices only; this function contains no RNG or redraw path."
function bootstrap_contrasts(population::Population, labels::AbstractVector{EventLabel}, indices::AbstractMatrix;
        coverage=(9,10), index_sha256)
    need(hex64(index_sha256), "outside-pinned index bytes required")
    reference = population_guard(population)
    need(size(indices) == (10000,reference.K) && reference.K == 31, "exact10000-by31 frozen index matrix required")
    need(all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= reference.K, indices), "frozen index domain differs")
    input = matched_inputs(population,labels,coverage)
    draws = NamedTuple[]
    for draw in 1:10000
        weights = year_weights(input.years,input.year_universe,view(indices,draw,:))
        stats = paired_correlations(input.field,input.speed,input.target,weights)
        valid = stats.pearson_contrast !== nothing && stats.spearman_contrast !== nothing
        push!(draws,(; draw, valid, pearson_contrast=stats.pearson_contrast,
            spearman_contrast=stats.spearman_contrast,
            reasons=[stat.reason for stat in (stats.pearson_field,stats.pearson_speed,stats.spearman_field,stats.spearman_speed) if !stat.valid]))
    end
    summary = contrast_summary(draws)
    (; draws, summary...,
        index_sha256=String(index_sha256), coverage, source_authority_granted=false, scientific_qualification=false)
end

function comparison_report(population::Population, labels::AbstractVector{EventLabel}, indices::AbstractMatrix; index_sha256)
    summaries = [paired_statistics(population,labels;coverage) for coverage in COVERAGES]
    uncertainties = [bootstrap_contrasts(population,labels,indices;coverage,index_sha256) for coverage in COVERAGES]
    primary = summaries[1]; uncertainty = uncertainties[1]; stats = primary.statistics
    direction(s) = s.statistics.pearson_contrast !== nothing && s.statistics.spearman_contrast !== nothing &&
        s.statistics.pearson_contrast > 0 && s.statistics.spearman_contrast > 0
    positive = stats.pearson_field.valid && stats.spearman_field.valid && stats.pearson_field.value > 0 && stats.spearman_field.value > 0
    intervals = uncertainty.pearson_interval !== nothing && uncertainty.spearman_interval !== nothing &&
        uncertainty.pearson_interval[1] > 0 && uncertainty.spearman_interval[1] > 0
    gate = primary.full_population_gate && primary.matched_population_gate && positive && uncertainty.uncertainty_gate &&
        intervals && all(direction,summaries)
    (; summaries, uncertainties, numerical_support_gate=gate, generated_inputs_only=population.generated,
        protocol_sha256=PROTOCOL_SHA, source_authority_granted=false, scientific_qualification=false)
end

function comparison_from_frozen_artifact(population::Population, labels::AbstractVector{EventLabel}, directory;
        expected_source_head, expected_matrix_sha256)
    artifact = Bootstrap.validate_artifact(directory; expected_source_head,expected_matrix_sha256)
    need(artifact.manifest["role"] == "primary" && artifact.manifest["runtime"] == "1.12.6", "frozen primary bootstrap artifact required")
    # Native three-control equality and source admission remain external gates.
    comparison_report(population,labels,artifact.values;index_sha256=artifact.matrix_sha256)
end

end
