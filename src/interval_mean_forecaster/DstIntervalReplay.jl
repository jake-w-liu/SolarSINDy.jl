module DstIntervalReplay

using Dates, LinearAlgebra, SHA
include("DstIntervalObserver.jl")
using .DstIntervalObserver

export Checkpoint, ForcingInterval, MeanObservation, ScenarioInterval,
       ReplayCache, replay_asof, replay!, rollout_targets, record_forecast!,
       affine_gain_replay, evaluate_affine

const N = DstIntervalObserver
const MAX_BINS = 4096
const MAX_RECORDS = 32768

_values(x,label) = Tuple(N._vector(collect(x),label))
_matrix(x) = [x[i][j] for i in eachindex(x),j in eachindex(x)]
_rows(x) = Tuple(Tuple(x[i,j] for j in axes(x,2)) for i in axes(x,1))
_hours(a,b) = Float64(Int128(Dates.value(b))-Int128(Dates.value(a)))/3_600_000
function _id(x::AbstractString,label)
    isempty(x) && throw(ArgumentError("$label must be nonempty"))
    ncodeunits(x)<=1024 || throw(ArgumentError("$label exceeds1024 bytes"))
    String(x)
end

struct Checkpoint
    boundary::DateTime
    mean::Tuple{Vararg{Float64}}
    covariance::Tuple
    function Checkpoint(boundary::DateTime,mean::AbstractVector{<:Real},covariance::AbstractMatrix{<:Real})
        m = _values(mean,"checkpoint mean")
        p = N._covariance(covariance,length(m),"checkpoint covariance")
        new(boundary,m,_rows(p))
    end
end

struct ForcingInterval
    id::String
    start::DateTime
    stop::DateTime
    forcing::Tuple{Vararg{Float64}}
    received::DateTime
    function ForcingInterval(id::AbstractString,start::DateTime,stop::DateTime,forcing::AbstractVector{<:Real},received::DateTime)
        stop>start || throw(ArgumentError("forcing interval must have positive duration"))
        new(_id(id,"forcing ID"),start,stop,_values(forcing,"forcing"),received)
    end
end

struct MeanObservation
    id::String
    bin_id::String
    start::DateTime
    stop::DateTime
    value::Float64
    variance::Float64
    received::DateTime
    verified::DateTime
    function MeanObservation(id::AbstractString,bin_id::AbstractString,start::DateTime,stop::DateTime,
            value::Real,variance::Real,received::DateTime,verified::DateTime)
        stop>start || throw(ArgumentError("observation interval must have positive duration"))
        y,r=N._number(value,"observed mean"),N._number(variance,"observation variance")
        r>0 || throw(ArgumentError("observation variance must be positive"))
        new(_id(id,"observation ID"),_id(bin_id,"observation bin ID"),start,stop,y,r,received,verified)
    end
end

struct ScenarioInterval
    start::DateTime
    stop::DateTime
    forcing::Tuple{Vararg{Float64}}
    function ScenarioInterval(start::DateTime,stop::DateTime,forcing::AbstractVector{<:Real})
        stop>start || throw(ArgumentError("scenario interval must have positive duration"))
        new(start,stop,_values(forcing,"scenario forcing"))
    end
end

# Receipt IDs are caller-supplied synthetic identities in this prototype. This
# representation is not a cryptographic adapter for the production journal.
_signature(x::ForcingInterval)=(x.id,x.start,x.stop,x.forcing,x.received)
_signature(x::MeanObservation)=(x.id,x.bin_id,x.start,x.stop,x.value,x.variance,x.received,x.verified)
_signature(::Nothing)=nothing
_equivalent(a::MeanObservation,b::MeanObservation)=
    (a.bin_id,a.start,a.stop,a.value,a.variance,a.received)==
    (b.bin_id,b.start,b.stop,b.value,b.variance,b.received)

function _fingerprint(parts...)
    # Length-framed canonical component representations. Record tuples contain
    # finite Float64s, explicit DateTimes and strings; no mutable storage.
    io=IOBuffer()
    for part in parts
        s=repr(part)
        print(io,ncodeunits(s),':',s)
    end
    bytes2hex(sha256(take!(io)))
end

function _config(initial,lambdas,sigmas)
    l,s=_values(lambdas,"lambda"),_values(sigmas,"sigma")
    length(l)==length(s)==length(initial.mean) || throw(DimensionMismatch("mode count differs"))
    all(>=(0),l) && all(>=(0),s) || throw(ArgumentError("lambda/sigma must be nonnegative"))
    l,s
end

function _admit(initial,intervals,observations,boundary,asof,receipt_boundary,max_bins,max_records)
    boundary>=initial.boundary || throw(ArgumentError("model boundary precedes checkpoint"))
    asof>=boundary || throw(ArgumentError("decision clock precedes model boundary"))
    receipt_boundary in (:strict,:hypothetical_inclusive) || throw(ArgumentError("unknown receipt boundary policy"))
    0<max_bins<=MAX_BINS && 0<max_records<=MAX_RECORDS || throw(ArgumentError("unsupported resource bound"))
    length(intervals)<=max_bins && length(observations)<=max_records || throw(ArgumentError("input exceeds explicit resource bounds"))
    eligible(t)=receipt_boundary==:strict ? t<asof : t<=asof
    known=Dict{String,ForcingInterval}()
    for x in intervals
        x isa ForcingInterval || throw(ArgumentError("forcing input has an unsupported record type"))
        x.stop<=boundary && x.start>=initial.boundary && eligible(x.received) || continue
        x.received>=x.stop || throw(ArgumentError("forcing receipt predates nominal completion"))
        length(x.forcing)==length(initial.mean) || throw(DimensionMismatch("forcing mode count differs"))
        if haskey(known,x.id)
            _signature(known[x.id])==_signature(x) || throw(ArgumentError("conflicting forcing ID"))
        else
            known[x.id]=x
        end
    end
    bins=sort!(collect(values(known));by=x->(x.start,x.stop,x.id))
    cursor=initial.boundary
    for x in bins
        x.start==cursor || throw(ArgumentError("known forcing must exactly tile checkpoint to model boundary"))
        cursor=x.stop
    end
    cursor==boundary || throw(ArgumentError("known forcing does not reach model boundary"))
    byid=Dict(x.id=>x for x in bins)
    identities=Dict{String,MeanObservation}()
    capture_clocks=Dict{Tuple{String,DateTime},MeanObservation}()
    selected=Dict{String,MeanObservation}()
    for x in observations
        x isa MeanObservation || throw(ArgumentError("observation input has an unsupported record type"))
        x.stop<=boundary && x.start>=initial.boundary && eligible(x.received) && eligible(x.verified) || continue
        x.received>=x.stop && x.verified>=x.received || throw(ArgumentError("observation clocks violate completion/receipt order"))
        haskey(byid,x.bin_id) || throw(ArgumentError("eligible observation has no known forcing interval"))
        b=byid[x.bin_id]
        (x.start,x.stop)==(b.start,b.stop) || throw(ArgumentError("observation interval does not match forcing interval"))
        if haskey(identities,x.id)
            _signature(identities[x.id])==_signature(x) || throw(ArgumentError("observation identity reused with changed content"))
        else
            identities[x.id]=x
        end
        capture_key=(x.bin_id,x.received)
        if haskey(capture_clocks,capture_key)
            _equivalent(capture_clocks[capture_key],x) || throw(ArgumentError("equal-clock observation conflict"))
        else
            capture_clocks[capture_key]=x
        end
        if !haskey(selected,x.bin_id)
            selected[x.bin_id]=x
        else
            previous=selected[x.bin_id]
            if x.received>previous.received
                selected[x.bin_id]=x
            elseif x.received==previous.received
                _equivalent(x,previous) || throw(ArgumentError("equal-clock observation conflict"))
                # Equivalent likelihoods have one deterministic representative;
                # verification time only gates eligibility, never orders captures.
                (x.verified,x.id)<(previous.verified,previous.id) && (selected[x.bin_id]=x)
            end
        end
    end
    Tuple(bins),Tuple(get(selected,x.id,nothing) for x in bins)
end

function _advance(state,l,s,b,observation)
    j=interval_moments(collect(state.mean),_matrix(state.covariance),collect(l),collect(s),collect(b.forcing),_hours(b.start,b.stop))
    if observation===nothing
        return Checkpoint(b.stop,j.endpoint_mean,j.endpoint_covariance),zeros(length(l))
    end
    posterior=condition_interval(j,observation.value,observation.variance)
    Checkpoint(b.stop,posterior.mean,posterior.covariance),posterior.gain
end

function _replay(initial,l,s,bins,selected,asof,policy;previous=nothing)
    common=0
    if previous!==nothing
        for i in 1:min(length(bins),length(previous.intervals))
            _signature(bins[i])==_signature(previous.intervals[i]) &&
            _signature(selected[i])==_signature(previous.selected[i]) || break
            common=i
        end
    end
    history=previous===nothing ? Checkpoint[initial] : collect(previous.history[1:common+1])
    gains=previous===nothing ? Tuple[] : Tuple[previous.gains[i] for i in 1:common]
    state=history[end]
    for i in common+1:length(bins)
        state,gain=_advance(state,l,s,bins[i],selected[i])
        push!(history,state)
        push!(gains,Tuple(gain))
    end
    fingerprint=_fingerprint(initial.boundary,initial.mean,initial.covariance,l,s,
        Tuple(_signature(x) for x in bins),Tuple(_signature(x) for x in selected),asof,policy)
    (;state,history=Tuple(history),gains=Tuple(gains),intervals=bins,selected,asof,
       receipt_boundary=policy,fingerprint,reused_steps=common,replayed_steps=length(bins)-common)
end

"""Exact-prefix retrospective replay. The clocks and supplied IDs are explicit
synthetic evidence inputs; this is not a production observation-journal adapter.
Full arrays are allowed. Later-ending/not-yet-eligible records are excluded.
One latest capture likelihood per bin is retained, and the chain is recomputed
from a checkpoint before any changed likelihood. Issued forecasts are separate.
"""
function replay_asof(initial::Checkpoint,lambdas,sigmas,intervals,observations,boundary::DateTime;
        asof::DateTime=boundary,receipt_boundary::Symbol=:strict,
        max_bins::Int=MAX_BINS,max_records::Int=MAX_RECORDS)
    l,s=_config(initial,lambdas,sigmas)
    bins,selected=_admit(initial,intervals,observations,boundary,asof,receipt_boundary,max_bins,max_records)
    _replay(initial,l,s,bins,selected,asof,receipt_boundary)
end

mutable struct ReplayCache
    const initial::Checkpoint
    const lambdas::Tuple{Vararg{Float64}}
    const sigmas::Tuple{Vararg{Float64}}
    snapshot::Any
    const issued::Dict{String,Any}
    function ReplayCache(initial::Checkpoint,lambdas,sigmas)
        l,s=_config(initial,lambdas,sigmas)
        new(initial,l,s,nothing,Dict{String,Any}())
    end
end

"""Single-owner cache. No concurrency or external field mutation is supported.
An error leaves the last complete snapshot and issued records unchanged.
"""
function replay!(cache::ReplayCache,intervals,observations,boundary::DateTime;
        asof::DateTime=boundary,receipt_boundary::Symbol=:strict,
        max_bins::Int=MAX_BINS,max_records::Int=MAX_RECORDS)
    bins,selected=_admit(cache.initial,intervals,observations,boundary,asof,receipt_boundary,max_bins,max_records)
    result=_replay(cache.initial,cache.lambdas,cache.sigmas,bins,selected,asof,receipt_boundary;previous=cache.snapshot)
    cache.snapshot=result
    result
end

"""Propagate fixed, issue-time scenario forcing without future observations.
Each returned mean/variance describes that interval's average sum(x). These are
scenario-conditional Gaussian variances; future-driver uncertainty is absent.
"""
function rollout_targets(state::Checkpoint,lambdas,sigmas,scenarios)
    l,s=_config(state,lambdas,sigmas)
    length(scenarios)<=MAX_BINS || throw(ArgumentError("too many scenario intervals"))
    out=NamedTuple[]
    cursor=state.boundary
    current=state
    for step in scenarios
        step isa ScenarioInterval || throw(ArgumentError("unsupported scenario type"))
        step.start==cursor || throw(ArgumentError("future scenario must exactly tile from state boundary"))
        length(step.forcing)==length(l) || throw(DimensionMismatch("scenario mode count differs"))
        j=interval_moments(collect(current.mean),_matrix(current.covariance),collect(l),collect(s),collect(step.forcing),_hours(step.start,step.stop))
        push!(out,(start=step.start,stop=step.stop,mean=j.interval_mean,variance=j.interval_variance))
        current=Checkpoint(step.stop,j.endpoint_mean,j.endpoint_covariance)
        cursor=step.stop
    end
    Tuple(out)
end

function record_forecast!(cache::ReplayCache,id::AbstractString,result,scenarios)
    key=_id(id,"forecast ID")
    cache.snapshot!==nothing && result===cache.snapshot ||
        throw(ArgumentError("forecast must bind the current complete cache snapshot"))
    targets=rollout_targets(result.state,cache.lambdas,cache.sigmas,scenarios)
    record=(id=key,issued_at=result.asof,model_boundary=result.state.boundary,
        replay_fingerprint=result.fingerprint,scenarios=Tuple((x.start,x.stop,x.forcing) for x in scenarios),targets)
    if haskey(cache.issued,key)
        cache.issued[key]==record || throw(ArgumentError("issued forecast identity is immutable"))
        return cache.issued[key]
    end
    length(cache.issued)<MAX_RECORDS || throw(ArgumentError("issued-record bound exceeded"))
    cache.issued[key]=record
    record
end

"""Affine gain-basis replay, with no parameter estimation. Forcing in each bin
is baseline + forcing_bases[bin_id]*theta. The initial mean can likewise have
initial_basis*theta. Observation values/R and all coefficients/eligibility are
fixed. Gains and covariance do not depend on theta. Missing basis keys mean0.
"""
function affine_gain_replay(initial::Checkpoint,lambdas,sigmas,intervals,observations,boundary::DateTime,
        forcing_bases::AbstractDict,initial_basis::AbstractMatrix{<:Real};kwargs...)
    n=length(initial.mean)
    size(initial_basis,1)==n && 1<=size(initial_basis,2)<=16 || throw(DimensionMismatch("initial gain basis shape unsupported"))
    coefficients=map(x->N._number(x,"initial gain basis"),initial_basis)
    p=size(coefficients,2)
    l,s=_config(initial,lambdas,sigmas)
    result=replay_asof(initial,l,s,intervals,observations,boundary;kwargs...)
    for (i,bin) in enumerate(result.intervals)
        basis=get(forcing_bases,bin.id,zeros(n,p))
        size(basis)==(n,p) || throw(DimensionMismatch("forcing gain basis shape differs"))
        basis=map(x->N._number(x,"forcing gain basis"),basis)
        kernels=[interval_kernel(l[j],s[j],_hours(bin.start,bin.stop)) for j in 1:n]
        phi=[x.phi for x in kernels]
        c=[x.mean_coefficient for x in kernels]
        u=[x.forcing_endpoint for x in kernels]
        v=[x.forcing_mean for x in kernels]
        bar_coefficients=coefficients'*c+basis'*v
        coefficients=Diagonal(phi)*coefficients+Diagonal(u)*basis-
            collect(result.gains[i])*bar_coefficients'
        all(isfinite,coefficients) || throw(ArgumentError("affine coefficient exceeds finite result range"))
    end
    (;base=result.state,coefficients=_rows(coefficients),replay=result,parameter_count=p)
end

function evaluate_affine(affine,parameters::AbstractVector{<:Real})
    theta=N._vector(parameters,"gain-basis parameters")
    length(theta)==affine.parameter_count || throw(DimensionMismatch("gain-basis parameter count differs"))
    coefficients=[affine.coefficients[i][j] for i in eachindex(affine.base.mean),j in 1:affine.parameter_count]
    mean=collect(affine.base.mean)+coefficients*theta
    Checkpoint(affine.base.boundary,mean,_matrix(affine.base.covariance))
end

end # module
