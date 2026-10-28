"""Owned numeric values for the publisher's uncorrected UP-OCP Algorithm 1.

These values carry no archival-source, fitting or coverage authority. Sequential
label availability, score adaptation and interleaving belong to the caller.
"""
module PublisherUPValues
using SHA,SolarSINDy
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const REFERENCE=joinpath(@__DIR__,"retained_publisher_up.jl")
const REFERENCE_SHA256="4cca17d9850a6dbd2ae76220fba4fbcd049a1e2b0a40b15c00dd820b588bce6e"
bytes2hex(sha256(read(REFERENCE)))==REFERENCE_SHA256 || error("retained publisher Algorithm1 source differs")
include("retained_publisher_up.jl")
const U=DBDTUniversalPortfolio
const MAX_UPDATES=1_200_000
export State,radius,advance,encode_state,decode_state
need(ok,message)=ok || throw(ArgumentError(message))
bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
function unbits(value)
    need(value isa AbstractString && occursin(r"\A[0-9a-f]{16}\z",value),"publisher floating bit syntax")
    result=reinterpret(Float64,parse(UInt64,value;base=16))
    need(isfinite(result),"publisher state must be finite");result
end
struct _Token end
"""Immutable owned numerical state; it asserts no observation provenance."""
struct State
    nominal::Float64
    resolved::Int
    misses::Int
    log_wealth::Float64
    function State(nominal,resolved,misses,log_wealth,::_Token)
        need(nominal isa Real && !(nominal isa Bool) && isfinite(nominal),"publisher nominal domain")
        alpha=Float64(nominal);need(0<alpha<1,"publisher nominal Float64 domain")
        need(resolved isa Integer && !(resolved isa Bool) && misses isa Integer && !(misses isa Bool) &&
            0<=misses<=resolved<=MAX_UPDATES,"publisher count geometry")
        need(log_wealth isa Real && !(log_wealth isa Bool) && isfinite(log_wealth),"publisher wealth domain")
        wealth=Float64(log_wealth);need(isfinite(wealth) && (resolved!=0 || (misses==0 && wealth==0)),"publisher wealth/initial state")
        new(alpha,Int(resolved),Int(misses),wealth)
    end
end
"""Initialize the publisher state with wealth1 and zero resolved/missed labels."""
State(;nominal=.1)=State(nominal,0,0,0.,_Token())
function _native(state::State)
    result=U.UniversalPortfolio(;nominal=state.nominal)
    result.resolved=state.resolved;result.misses=state.misses;result.log_wealth=state.log_wealth
    result
end
"""Return the uncorrected publisher radius; no score is observed or state changed."""
radius(state::State)=U.radius(_native(state))
"""Apply exactly one finite nonnegative score and return a new owned state.

Equality with the current radius is a hit. The input state stays unchanged.
This function does not authorize reading a label before its completion time.
"""
function advance(state::State,score)
    need(state.resolved<MAX_UPDATES,"publisher update capacity reached")
    need(score isa Real && !(score isa Bool) && isfinite(score) && score>=0,"publisher score domain")
    value=Float64(score);need(isfinite(value),"publisher score Float64 overflow")
    native=_native(state);miss=U.update!(native,value)
    next=State(native.nominal,native.resolved,native.misses,native.log_wealth,_Token())
    (;state=next,miss)
end
"""Encode bounded numeric fields with explicit floating bits, without provenance."""
function encode_state(state::State)
    R.canonical_bytes(Dict("schema"=>"publisher_up_numeric_state_v1","algorithm"=>"publisher_uncorrected_algorithm1",
        "publisher_reference_sha256"=>REFERENCE_SHA256,"nominal"=>bits(state.nominal),"resolved"=>state.resolved,
        "misses"=>state.misses,"log_wealth"=>bits(state.log_wealth),"numeric_values_only"=>true,
        "actual_source_authority_granted"=>false,"actual_fits"=>0);limit=4096)
end
"""Decode externally pinned canonical numeric fields; this grants no source authority."""
function decode_state(bytes;expected_sha256)
    need(bytes isa AbstractVector{UInt8} && length(bytes)<=4096,"publisher numeric state byte bound")
    need(expected_sha256 isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",expected_sha256) &&
        bytes2hex(sha256(bytes))==expected_sha256,"publisher numeric external pin")
    value=R.decode_canonical(copy(bytes);limit=4096)
    fields=Set(("schema","algorithm","publisher_reference_sha256","nominal","resolved","misses","log_wealth",
        "numeric_values_only","actual_source_authority_granted","actual_fits"))
    need(value isa AbstractDict && Set(keys(value))==fields,"publisher numeric state fields")
    need(value["schema"]=="publisher_up_numeric_state_v1" && value["algorithm"]=="publisher_uncorrected_algorithm1" &&
        value["publisher_reference_sha256"]==REFERENCE_SHA256 && value["numeric_values_only"]===true &&
        value["actual_source_authority_granted"]===false && value["actual_fits"] isa Integer && !(value["actual_fits"] isa Bool) && value["actual_fits"]==0,
        "publisher numeric state declarations")
    result=State(unbits(value["nominal"]),value["resolved"],value["misses"],unbits(value["log_wealth"]),_Token())
    need(encode_state(result)==bytes,"publisher numeric state reconstruction differs");result
end
end
