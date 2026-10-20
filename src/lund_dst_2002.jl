"""
Published four-state Lund Dst reference kernel (Lundstedt, Gleisner and Wintoft,
2002, Appendix A; doi:10.1029/2002GL016151). No fitting, source acquisition,
cadence repair, missing-value imputation, or future-driver policy is performed.
"""
module LundDst2002

export initial_state, predict_next, rollout

const INPUT_WEIGHTS = ((-0.5946, 0.0904, 0.0428),
                       (0.6215, 0.4485, 0.2516),
                       (-1.4217, 0.4988, 0.1409),
                       (0.3372, -0.5065, 0.7190))
const CONTEXT_WEIGHTS = ((0.6099, 0.2223, -0.4379, 0.0053),
                         (0.7935, 0.5000, -1.2654, -0.0428),
                         (-0.3496, 0.1024, 0.3605, 0.1790),
                         (-0.2067, 0.2270, 0.5446, 0.1019))
const HIDDEN_BIAS = (-0.1081, -0.7252, -0.2307, 0.5406)
const OUTPUT_WEIGHTS = (0.2535, 0.3172, -0.4860, -0.2346)
const OUTPUT_BIAS = 0.0712

"Published initial latent mean, independent of any observed Dst."
initial_state() = (0.39, 0.22, -0.83, 0.14)

function checked_state(state::Union{Tuple,AbstractVector})
    length(state) == 4 || throw(DimensionMismatch("Lund Dst state requires four coordinates"))
    all(x -> x isa Real && isfinite(x) && -1 <= x <= 1, state) ||
        throw(ArgumentError("Lund Dst state must be finite and lie in [-1,1]"))
    return ntuple(i -> Float64(state[i]), 4)
end

function checked_driver(value::Real, name, nonnegative)
    isfinite(value) && (!nonnegative || value >= 0) ||
        throw(ArgumentError("Lund Dst $name must be finite" *
                            (nonnegative ? " and nonnegative" : "")))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("Lund Dst $name exceeds Float64 range"))
    return converted
end

function advance(state::NTuple{4,Float64}, bz::Float64, density::Float64, speed::Float64)
    input = (bz / 30, density / 80 - 1, speed / 400 - 1.5)
    next_state = ntuple(4) do i
        driving = sum(INPUT_WEIGHTS[i][j] * input[j] for j in 1:3)
        memory = sum(CONTEXT_WEIGHTS[i][j] * state[j] for j in 1:4)
        tanh(driving + memory + HIDDEN_BIAS[i])
    end
    output = sum(OUTPUT_WEIGHTS[j] * next_state[j] for j in 1:4) + OUTPUT_BIAS
    return (state=next_state, dst_nt=150output - 100)
end

"""
    predict_next(state, bz_nt, density_cm3, speed_kms)

Advance the published one-hour reference once using Bz (nT), proton density
(cm^-3), and speed (km/s). All four hidden coordinates use the previous state;
the Dst output uses the newly updated state. Return `(state, dst_nt)` without
mutating the input. Coefficients retain the precision displayed in Appendix A.

The caller must provide a contiguous hourly source sequence and verify its
interval/receipt convention. This arithmetic kernel does not turn a raw hour
tag into an operational availability timestamp. It consumes no observed Dst.
Further steps require explicitly supplied drivers, not hidden realized wind.
"""
function predict_next(state::Union{Tuple,AbstractVector}, bz_nt::Real,
                      density_cm3::Real, speed_kms::Real)
    previous = checked_state(state)
    bz = checked_driver(bz_nt, "Bz", false)
    density = checked_driver(density_cm3, "density", true)
    speed = checked_driver(speed_kms, "speed", true)
    return advance(previous, bz, density, speed)
end

"""
    rollout(bz_nt, density_cm3, speed_kms; state=initial_state())

Apply the published recurrence to aligned contiguous hourly input vectors.
Return one next-label prediction per input and the final four-coordinate state.
An empty input returns empty predictions and the validated unchanged state.
All supplied rows are consumed as drivers; a multi-hour forecast must separately
declare which future-driver values are known or assumed. No gap filling, pressure
correction, clipping, or assimilation of observed Dst is added.
"""
function rollout(bz_nt::AbstractVector{<:Real}, density_cm3::AbstractVector{<:Real},
                 speed_kms::AbstractVector{<:Real}; state=initial_state())
    length(bz_nt) == length(density_cm3) == length(speed_kms) ||
        throw(DimensionMismatch("Lund Dst driver vectors must have equal lengths"))
    previous = checked_state(state)
    result = Vector{Float64}(undef, length(bz_nt))
    for (slot, values) in enumerate(zip(bz_nt, density_cm3, speed_kms))
        bz, density, speed = values
        point = advance(previous, checked_driver(bz, "Bz", false),
                        checked_driver(density, "density", true),
                        checked_driver(speed, "speed", true))
        previous = point.state
        result[slot] = point.dst_nt
    end
    return (state=previous, dst_nt=result)
end

end
