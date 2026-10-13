# serving_depth_safe.jl — the depth-safe alerting center of the served product.
#
# This file holds one function and nothing else, because two processes need the same definition:
# the package (through `static_stack_serving.jl`) and the dashboard application, which runs in
# its own environment and cannot load the package. The application includes this file directly, so
# the published severity and the packaged serving contract can never drift apart.
#
# Keep it dependency-free: no `using`, no package types, no constants from elsewhere.

"""
    static_stack_serving_depth_safe_center(served_static_stack, served_inertial_sindy) -> Float64

Alerting center of the served product: the deeper of the stacked center and the inertial-SINDy center it
replaced. The stack can blend toward persistence and therefore report a shallower storm than inertial-SINDy
would have; taking the minimum keeps the published severity from regressing when the stack is
conservative, while still letting a deeper stacked center escalate.

A non-finite input is dropped rather than propagated, so one unusable component cannot erase a
severity that the other component still supports.
"""
function static_stack_serving_depth_safe_center(served_static_stack::Real, served_inertial_sindy::Real)
    stacked = Float64(served_static_stack)
    previous = Float64(served_inertial_sindy)
    isfinite(stacked) || return previous
    isfinite(previous) || return stacked
    return min(stacked, previous)
end

"""
    superlearner_serving_depth_safe_center(served, continuity...) -> Float64

Alerting center of a served product with more than one continuity partner: the deepest of the served
center and every earlier center it replaced. The super-learner is served over the static stack
stack, which was itself served over the inertial-SINDy operator, so the published severity is taken against
both — a candidate that blends toward a shallower state cannot lower a warning either predecessor
would have raised, while a deeper candidate still escalates one. This is where depth safety lives for
the deployed product: the served point forecast is the stack center itself, and the static stack is one
of the experts that center is fitted over rather than a hard minimum on it.

Non-finite inputs are dropped rather than propagated, so one unusable stage cannot erase a severity
the remaining stages still support. With every input non-finite the served value is returned
unchanged, which is what the caller already publishes as its center.
"""
function superlearner_serving_depth_safe_center(served::Real, continuity::Real...)
    deepest = Float64(served)
    for value in continuity
        candidate = Float64(value)
        isfinite(candidate) || continue
        deepest = isfinite(deepest) ? min(deepest, candidate) : candidate
    end
    return deepest
end
