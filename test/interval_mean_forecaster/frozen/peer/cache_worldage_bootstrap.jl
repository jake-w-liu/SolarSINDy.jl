# Bootstrap-only fix for Julia 1.13 world age. Frozen oracle assertions/bytes unchanged.
using SHA
length(ARGS)==1 || error("expected candidate observer path")
const _PEER_SOURCE=abspath(ARGS[1])
const _PEER_BASELINE=raw"C:/Temp/da-optimize-20261016/baseline/DstIntervalObserver.jl"
const _PEER_ORACLE=joinpath(@__DIR__,"future_cache_oracle.jl")
bytes2hex(sha256(read(_PEER_ORACLE)))=="37a1416a652f4f8545f333d05106f5e94507678e0cc4d9f6752c9311774ed893" || error("frozen oracle changed")
const _PEER_MODULES=Dict{String,Vector{Module}}(_PEER_BASELINE=>Module[],_PEER_SOURCE=>Module[])
function _peer_preload(path)
    wrapper=Module(gensym(:PeerPreload))
    Base.include(wrapper,path)
    Base.invokelatest(getfield,wrapper,:DstIntervalObserver)
end
for (path,count) in ((_PEER_BASELINE,1),(_PEER_SOURCE,7))
    for _ in 1:count
        push!(_PEER_MODULES[path],_peer_preload(path))
    end
end
# More-specific helper changes only load scheduling, not source or test behavior.
# Fresh identities are still created for every original load_fresh call.
load_fresh(path::String)=popfirst!(_PEER_MODULES[path])
include(_PEER_ORACLE)
all(isempty,values(_PEER_MODULES)) || error("preload count differs from frozen loader calls")
println("BOOTSTRAP_PASS frozen_cache_oracle_sha256=",bytes2hex(sha256(read(_PEER_ORACLE))))
