"""Unprivileged pure decisions for the completed-source 2017 selection protocol.

These functions read no files, targets, observations or model objects. They do
not fit, replay, publish a decision, verify producer evidence or grant authority.
The actual-source coordinator and its qualification graph remain separate work.
"""
module CompletedSelectionPrimitives
using Dates, Statistics, SHA

export GammaExpectation, GammaMetrics, PointExpectation, PointMetrics,
       NoEligibleGamma, choose_gamma, choose_R2, split_clock_eligible

const STATIONS = ("CMO", "FRD")
const GAMMAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
const MAX_ROWS = 600000
const MINUTE_MS = Int64(60000)
const YEAR_START = DateTime(2017)
const YEAR_STOP = DateTime(2018)
const DEFAULT_LABEL_DELAY_MS = Int64(1845000)

need(condition, message) = condition || throw(ArgumentError(message))
integer(value) = value isa Integer && !(value isa Bool)
function bounded_int(value, lower, upper, message)
    need(integer(value) && lower <= value <= upper, message)
    Int(value)
end
function text(value, maximum, message)
    need(value isa AbstractString && 0 < ncodeunits(value) <= maximum, message)
    String(value)
end
function finite64(value, message)
    need(value isa Real && !(value isa Bool) && isfinite(value), message)
    converted = try
        Float64(value)
    catch
        throw(ArgumentError(message))
    end
    need(isfinite(converted) && (iszero(value) || !iszero(converted)), message)
    converted
end
function period31(value)
    need(integer(value) && value == 31, "primary completed-source policy period must be31")
    31
end
function grid_names(schedule)
    schedule === :phase && return ("phase_declared",)
    schedule === :fifo && return ("phase_declared", "rate_matched_1_over_30")
    schedule === :fifo_clock31_sensitivity && return ("clock_matched_1_over_31_sensitivity",)
    throw(ArgumentError("unsupported fixed selection schedule"))
end
function grid_values(name)
    name == "phase_declared" && return GAMMAS
    name == "rate_matched_1_over_30" && return map(value -> value / 30, GAMMAS)
    name == "clock_matched_1_over_31_sensitivity" && return map(value -> value / 31, GAMMAS)
    throw(ArgumentError("unknown fixed gamma grid"))
end

"""Exact station-specific cohort expectations; this value grants no authority.

Counts and normalizers are independently supplied expectations, not inferred
from the candidate table. Natural station cohorts may differ; their losses get
equal station weight as in the historical rule. Primary FIFO retains division30.
The explicitly named division31 sensitivity cannot enter a primary decision.
"""
struct GammaExpectation
    schedule::Symbol
    cohorts::NTuple{2,String}
    counts::NTuple{2,Int}
    normalizations::NTuple{2,Float64}
    phase_period_minutes::Int
    function GammaExpectation(schedule; cohorts, counts, normalizations, phase_period_minutes)
        grid_names(schedule)
        need(cohorts isa Tuple && counts isa Tuple && normalizations isa Tuple &&
             length(cohorts) == length(counts) == length(normalizations) == 2,
             "exact CMO/FRD expectation tuples required")
        names = map(value -> text(value,512,"bounded cohort identity required"), cohorts)
        sizes = map(value -> bounded_int(value,1,MAX_ROWS,"invalid expected cohort count"), counts)
        scales = map(value -> finite64(value,"invalid expected station normalizer"), normalizations)
        need(all(>(0),scales), "station normalizers must be positive")
        new(schedule,names,sizes,scales,period31(phase_period_minutes))
    end
end

"""One station/candidate report. Numeric validation supplies no provenance proof.

All count fields are explicit. A finite negative bound is support-empty. An
ineligible report may retain a genuinely finite loss or use nothing; a missing
or nonfinite loss can never become an eligible objective.
"""
struct GammaMetrics
    station::String
    schedule::Symbol
    grid::String
    gamma::Float64
    cohort::String
    n::Int
    normalization::Float64
    normalized_pinball::Union{Nothing,Float64}
    full_count::Int
    empty_count::Int
    positive_inf_count::Int
    negative_inf_count::Int
    finite_negative_count::Int
    nan_count::Int
    phase_period_minutes::Int
    function GammaMetrics(;station,schedule,grid,gamma,cohort,n,normalization,normalized_pinball,
            full_count,empty_count,positive_inf_count,negative_inf_count,finite_negative_count,nan_count,
            phase_period_minutes)
        station = text(station,16,"invalid station label")
        need(station in STATIONS,"unexpected station")
        grid_names(schedule)
        grid = text(grid,128,"invalid grid label")
        value = finite64(gamma,"invalid gamma")
        need(0 < value <= 1,"gamma outside zero/one domain")
        cohort = text(cohort,512,"invalid cohort identity")
        count = bounded_int(n,1,MAX_ROWS,"invalid candidate cohort count")
        normalizer = finite64(normalization,"invalid station normalizer")
        need(normalizer > 0,"station normalizer must be positive")
        counts = map(value -> bounded_int(value,0,count,"invalid bound count"),
            (full_count,empty_count,positive_inf_count,negative_inf_count,finite_negative_count,nan_count))
        full,empty,positive,negative,finite_negative,nan = counts
        need(full == positive && empty == negative + finite_negative,
             "full/empty count definitions disagree")
        need(full + empty + nan <= count,"bound categories exceed attempted rows")
        loss = normalized_pinball === nothing ? nothing :
            finite64(normalized_pinball,"nonfinite candidate objective")
        need(loss === nothing || loss >= 0,"proper loss cannot be negative")
        eligible = full == empty == nan == 0
        need(!eligible || loss !== nothing,"eligible candidate needs a finite objective")
        new(station,schedule,grid,value,cohort,count,normalizer,loss,full,empty,
            positive,negative,finite_negative,nan,period31(phase_period_minutes))
    end
end

function put_text(io, value)
    body = codeunits(value)
    need(length(body) <= 1024,"decision text exceeds frame bound")
    write(io,htol(UInt32(length(body))),body)
end
put_int(io, value) = write(io,htol(UInt64(value)))
put_float(io, value::Float64) = write(io,htol(reinterpret(UInt64,value)))
function put_optional(io, value)
    write(io,UInt8(value !== nothing))
    value === nothing || put_float(io,value)
end
function gamma_input_digest(expected, rows)
    io = IOBuffer()
    put_text(io,"COMPLETED_SELECTION_GAMMA_INPUT_V1")
    put_text(io,String(expected.schedule))
    put_int(io,expected.phase_period_minutes)
    for index in 1:2
        put_text(io,STATIONS[index]);put_text(io,expected.cohorts[index])
        put_int(io,expected.counts[index]);put_float(io,expected.normalizations[index])
    end
    for row in rows
        put_text(io,row.station);put_text(io,row.grid);put_float(io,row.gamma)
        put_text(io,row.cohort);put_int(io,row.n);put_float(io,row.normalization)
        put_optional(io,row.normalized_pinball)
        for value in (row.full_count,row.empty_count,row.positive_inf_count,
                      row.negative_inf_count,row.finite_negative_count,row.nan_count)
            put_int(io,value)
        end
    end
    need(position(io) <= 65536,"decision input frame exceeds bound")
    bytes2hex(sha256(take!(io)))
end
function station_report(row)
    (;station=row.station,cohort=row.cohort,n=row.n,normalization=row.normalization,
      normalized_pinball=row.normalized_pinball,full_count=row.full_count,empty_count=row.empty_count,
      positive_inf_count=row.positive_inf_count,negative_inf_count=row.negative_inf_count,
      finite_negative_count=row.finite_negative_count,nan_count=row.nan_count)
end

"""Fail closed while retaining every ineligible candidate in the error value."""
struct NoEligibleGamma <: Exception
    schedule::Symbol
    candidate_table::Tuple
    input_sha256::String
end
Base.showerror(io::IO, error::NoEligibleGamma) = print(io,"no declared gamma candidate avoided forbidden bounds")

"""Choose using first eligible grid, equal station mean loss, then smaller gamma.

Every station/candidate must occur once and match its supplied expectation.
The returned digest binds numeric input and the decision; it is not producer,
source, execution, publication or scientific evidence. No tolerance defines ties.
"""
function choose_gamma(rows::AbstractVector, expected::GammaExpectation)
    Base.require_one_based_indexing(rows)
    names = grid_names(expected.schedule)
    need(length(rows) == 14 * length(names) && all(row -> row isa GammaMetrics,rows),
         "complete fixed station/gamma table required")
    admitted = Dict{Tuple{String,String,Float64},GammaMetrics}()
    for row in rows
        station = findfirst(==(row.station),STATIONS)
        need(station !== nothing && row.schedule === expected.schedule &&
             row.phase_period_minutes == expected.phase_period_minutes,
             "candidate schedule/station/policy differs")
        need(row.grid in names && row.gamma in grid_values(row.grid),"candidate outside declared gamma grid")
        need(row.cohort == expected.cohorts[station] && row.n == expected.counts[station] &&
             row.normalization == expected.normalizations[station],"candidate cohort or normalizer differs")
        key = (row.station,row.grid,row.gamma)
        need(!haskey(admitted,key),"duplicate station/gamma row")
        admitted[key] = row
    end
    ordered = GammaMetrics[]
    candidates = NamedTuple[]
    for name in names, gamma in grid_values(name)
        pair = Tuple(begin
            key = (station,name,gamma)
            need(haskey(admitted,key),"missing station/gamma row")
            admitted[key]
        end for station in STATIONS)
        append!(ordered,pair)
        reasons = Symbol[]
        any(row -> row.full_count > 0,pair) && push!(reasons,:full_bound)
        any(row -> row.empty_count > 0,pair) && push!(reasons,:support_empty_bound)
        any(row -> row.nan_count > 0,pair) && push!(reasons,:nonfinite_bound)
        eligible = isempty(reasons)
        objective = eligible ? mean((pair[1].normalized_pinball::Float64,
                                    pair[2].normalized_pinball::Float64)) : nothing
        need(objective === nothing || isfinite(objective),"station-mean objective overflow")
        push!(candidates,(;grid=name,gamma,eligible,reasons=Tuple(reasons),objective,
                          station_reports=map(station_report,pair)))
    end
    input_sha = gamma_input_digest(expected,ordered)
    selected = nothing
    for name in names
        best = nothing
        for candidate in candidates
            candidate.grid == name && candidate.eligible || continue
            if best === nothing || candidate.objective < best.objective ||
                    (candidate.objective == best.objective && candidate.gamma < best.gamma)
                best = candidate
            end
        end
        if best !== nothing
            selected = best
            break
        end
    end
    selected === nothing && throw(NoEligibleGamma(expected.schedule,Tuple(candidates),input_sha))
    io = IOBuffer();put_text(io,"COMPLETED_SELECTION_GAMMA_DECISION_V1");put_text(io,input_sha)
    put_text(io,selected.grid);put_float(io,selected.gamma);put_float(io,selected.objective)
    sensitivity = expected.schedule === :fifo_clock31_sensitivity
    (;schema="completed_selection_pure_gamma_decision_v1",schedule=expected.schedule,
      selected_grid=selected.grid,gamma=selected.gamma,objective=selected.objective,
      candidate_table=Tuple(candidates),input_sha256=input_sha,decision_sha256=bytes2hex(sha256(take!(io))),
      phase_period_minutes=expected.phase_period_minutes,primary_selection=!sensitivity,sensitivity_only=sensitivity,
      actual_source_access=false,actual_fits=0,source_bound_provenance_verified=false,
      software_qualification=false,scientific_qualification=false,actual_coordinator_implemented=false)
end

const POINT_GRID = (
    (id="ridge-0.01",family=:ridge,depth=0,trees=0,penalty=0.01),
    (id="ridge-0.1",family=:ridge,depth=0,trees=0,penalty=0.1),
    (id="ridge-1",family=:ridge,depth=0,trees=0,penalty=1.0),
    (id="ridge-10",family=:ridge,depth=0,trees=0,penalty=10.0),
    (id="ridge-100",family=:ridge,depth=0,trees=0,penalty=100.0),
    (id="boost-2-50",family=:boost,depth=2,trees=50,penalty=0.0),
    (id="boost-2-100",family=:boost,depth=2,trees=100,penalty=0.0),
    (id="boost-2-200",family=:boost,depth=2,trees=200,penalty=0.0),
    (id="boost-3-50",family=:boost,depth=3,trees=50,penalty=0.0),
    (id="boost-3-100",family=:boost,depth=3,trees=100,penalty=0.0),
    (id="boost-3-200",family=:boost,depth=3,trees=200,penalty=0.0),
    (id="boost-4-50",family=:boost,depth=4,trees=50,penalty=0.0),
    (id="boost-4-100",family=:boost,depth=4,trees=100,penalty=0.0),
    (id="boost-4-200",family=:boost,depth=4,trees=200,penalty=0.0),
    (id="forest-6-64",family=:forest,depth=6,trees=64,penalty=0.0),
    (id="forest-9-64",family=:forest,depth=9,trees=64,penalty=0.0),
)
const FAMILIES = (:ridge,:boost,:forest)

struct PointExpectation
    station::String
    cohort::String
    n::Int
    phase_period_minutes::Int
    function PointExpectation(;station,cohort,n,phase_period_minutes)
        station = text(station,16,"invalid point station")
        need(station in STATIONS,"unexpected point station")
        new(station,text(cohort,512,"invalid point cohort"),
            bounded_int(n,1,MAX_ROWS,"invalid point count"),period31(phase_period_minutes))
    end
end
struct PointMetrics
    station::String
    candidate_id::String
    family::Symbol
    cohort::String
    n::Int
    selection_log_mse::Float64
    phase_period_minutes::Int
    function PointMetrics(;station,candidate_id,family,cohort,n,selection_log_mse,phase_period_minutes)
        station = text(station,16,"invalid point station")
        need(station in STATIONS && family in FAMILIES,"unexpected point station/family")
        loss = finite64(selection_log_mse,"nonfinite point selection loss")
        need(loss >= 0,"mean squared error cannot be negative")
        new(station,text(candidate_id,128,"invalid point candidate"),family,
            text(cohort,512,"invalid point cohort"),bounded_int(n,1,MAX_ROWS,"invalid point count"),
            loss,period31(phase_period_minutes))
    end
end

"""Retain all16 candidates, select within each family, then flag the primary.

Exact ties retain the first candidate in POINT_GRID, independently of input row
order. Family ties retain ridge, boost, forest order. All family winners remain
present even when they are not primary; no selected-refit authority is granted.
"""
function choose_R2(rows::AbstractVector, expected::PointExpectation)
    Base.require_one_based_indexing(rows)
    need(length(rows) == 16 && all(row -> row isa PointMetrics,rows),"all16 fixed point candidates required")
    admitted = Dict{String,PointMetrics}()
    for row in rows
        index = findfirst(spec -> spec.id == row.candidate_id,POINT_GRID)
        need(index !== nothing && POINT_GRID[index].family === row.family,"point candidate is outside fixed family/grid")
        need(row.station == expected.station && row.cohort == expected.cohort && row.n == expected.n &&
             row.phase_period_minutes == expected.phase_period_minutes,"point candidate cohort/policy differs")
        need(!haskey(admitted,row.candidate_id),"duplicate point candidate")
        admitted[row.candidate_id] = row
    end
    ordered = Tuple(begin
        need(haskey(admitted,spec.id),"missing point candidate")
        admitted[spec.id]
    end for spec in POINT_GRID)
    selected = NamedTuple[]
    for family in FAMILIES
        best = nothing
        for (index,spec) in enumerate(POINT_GRID)
            spec.family === family || continue
            row = ordered[index]
            if best === nothing || row.selection_log_mse < best.selection_log_mse
                best = row
            end
        end
        push!(selected,(;family,candidate_id=best.candidate_id,selection_log_mse=best.selection_log_mse))
    end
    primary = selected[1]
    for candidate in selected[2:end]
        candidate.selection_log_mse < primary.selection_log_mse && (primary=candidate)
    end
    io = IOBuffer();put_text(io,"COMPLETED_SELECTION_R2_INPUT_V1")
    put_text(io,expected.station);put_text(io,expected.cohort);put_int(io,expected.n)
    put_int(io,expected.phase_period_minutes)
    for (spec,row) in zip(POINT_GRID,ordered)
        put_text(io,spec.id);put_text(io,String(spec.family));put_int(io,spec.depth)
        put_int(io,spec.trees);put_float(io,spec.penalty);put_float(io,row.selection_log_mse)
    end
    input_sha = bytes2hex(sha256(take!(io)))
    put_text(io,"COMPLETED_SELECTION_R2_DECISION_V1");put_text(io,input_sha)
    for candidate in selected;put_text(io,candidate.candidate_id);end
    put_text(io,String(primary.family));put_text(io,primary.candidate_id)
    table = Tuple((;candidate_id=row.candidate_id,family=row.family,cohort=row.cohort,n=row.n,
                    selection_log_mse=row.selection_log_mse) for row in ordered)
    (;schema="completed_selection_pure_R2_decision_v1",station=expected.station,
      candidate_table=table,family_winners=Tuple(selected),primary_family=primary.family,
      primary_candidate_id=primary.candidate_id,input_sha256=input_sha,
      decision_sha256=bytes2hex(sha256(take!(io))),phase_period_minutes=expected.phase_period_minutes,
      selected_family_refits_required=3,actual_source_access=false,actual_fits=0,
      source_bound_provenance_verified=false,software_qualification=false,
      scientific_qualification=false,actual_coordinator_implemented=false)
end

"""Pure2017 issue/readiness membership; this function has no target argument.

The issue must be a whole UTC minute in the half-open start/stop interval.
Both the interval and readiness boundary stay inside2017. Readiness equals
I+30min45s plus declared nonnegative ground/label delays and must be STRICTLY
before boundary. Callers must apply this predicate before opening target rows.
This predicate cannot verify whether an external coordinator obeyed that rule.
"""
function split_clock_eligible(issue::DateTime,boundary::DateTime;
        start::DateTime=YEAR_START,stop::DateTime=boundary,ground_extra_ms=0,label_extra_ms=0)
    ground = bounded_int(ground_extra_ms,0,86400000,"invalid declared ground delay")
    label = bounded_int(label_extra_ms,0,86400000,"invalid declared label delay")
    need(mod(Dates.value(issue),MINUTE_MS) == 0,"issue must be a whole UTC minute")
    need(start < stop <= boundary,"invalid half-open split/boundary order")
    YEAR_START <= start < stop <= boundary <= YEAR_STOP || return false
    start <= issue < stop || return false
    delay = Base.Checked.checked_add(DEFAULT_LABEL_DELAY_MS,
        Base.Checked.checked_add(Int64(ground),Int64(label)))
    expected = Base.Checked.checked_add(Dates.value(issue),delay)
    ready = issue + Millisecond(delay)
    need(Dates.value(ready) == expected,"selection readiness clock overflow")
    ready < boundary
end
end
