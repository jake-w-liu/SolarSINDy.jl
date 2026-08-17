# serving_depth_safe.jl — the depth-safe alerting center of the served product.
#
# This file holds one function and nothing else, because two processes need the same definition:
# the package (through `operational_v22_serving.jl`) and the dashboard application, which runs in
# its own environment and cannot load the package. The application includes this file directly, so
# the published severity and the packaged serving contract can never drift apart.
#
# Keep it dependency-free: no `using`, no package types, no constants from elsewhere.

"""
    v22_serving_depth_safe_center(served_v2_2, served_v2_1) -> Float64

Alerting center of the served product: the deeper of the stacked center and the V2.1 center it
replaced. The stack can blend toward persistence and therefore report a shallower storm than V2.1
would have; taking the minimum keeps the published severity from regressing when the stack is
conservative, while still letting a deeper stacked center escalate.

A non-finite input is dropped rather than propagated, so one unusable component cannot erase a
severity that the other component still supports.
"""
function v22_serving_depth_safe_center(served_v2_2::Real, served_v2_1::Real)
    stacked = Float64(served_v2_2)
    previous = Float64(served_v2_1)
    isfinite(stacked) || return previous
    isfinite(previous) || return stacked
    return min(stacked, previous)
end
