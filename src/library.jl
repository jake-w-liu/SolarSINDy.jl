# Candidate library construction for SINDy

"""
    CandidateLibrary

Stores candidate function library for SINDy discovery.
- `names`: human-readable names for each library term
- `evaluate`: function (data_dict) -> Matrix where each column is a library term
"""
struct CandidateLibrary
    names::Vector{String}
    funcs::Vector{Function}  # Each func: Dict -> Vector{Float64}
    term_codes::Vector{UInt8}
end

CandidateLibrary(names::Vector{String}, funcs::Vector{Function}) =
    CandidateLibrary(names, funcs, _term_codes(names))

const TERM_FALLBACK = UInt8(0)
const TERM_ONE = UInt8(1)
const TERM_V = UInt8(2)
const TERM_BS = UInt8(3)
const TERM_N = UInt8(4)
const TERM_PDYN = UInt8(5)
const TERM_DST_STAR = UInt8(6)
const TERM_V2 = UInt8(7)
const TERM_BS2 = UInt8(8)
const TERM_N2 = UInt8(9)
const TERM_V_BS = UInt8(10)
const TERM_N_V = UInt8(11)
const TERM_N_BS = UInt8(12)
const TERM_PDYN_BS = UInt8(13)
const TERM_N_V_BS = UInt8(14)
const TERM_N_V2 = UInt8(15)
const TERM_SIN_HALF = UInt8(16)
const TERM_SIN_HALF2 = UInt8(17)
const TERM_SIN_HALF4 = UInt8(18)
const TERM_SIN_HALF_8_3 = UInt8(19)
const TERM_V_SIN_HALF2 = UInt8(20)
const TERM_NEWELL = UInt8(21)

function _term_code(name::String)
    return name == "1" ? TERM_ONE :
           name == "V" ? TERM_V :
           name == "Bs" ? TERM_BS :
           name == "n" ? TERM_N :
           name == "Pdyn" ? TERM_PDYN :
           name == "Dst_star" ? TERM_DST_STAR :
           name == "V^2" ? TERM_V2 :
           name == "Bs^2" ? TERM_BS2 :
           name == "n^2" ? TERM_N2 :
           name == "V*Bs" ? TERM_V_BS :
           name == "n*V" ? TERM_N_V :
           name == "n*Bs" ? TERM_N_BS :
           name == "Pdyn*Bs" ? TERM_PDYN_BS :
           name == "n*V*Bs" ? TERM_N_V_BS :
           name == "n*V^2" ? TERM_N_V2 :
           name == "sin(θ_c/2)" ? TERM_SIN_HALF :
           name == "sin²(θ_c/2)" ? TERM_SIN_HALF2 :
           name == "sin⁴(θ_c/2)" ? TERM_SIN_HALF4 :
           name == "sin^(8/3)(θ_c/2)" ? TERM_SIN_HALF_8_3 :
           name == "V*sin²(θ_c/2)" ? TERM_V_SIN_HALF2 :
           name == "Newell_d_Φ" ? TERM_NEWELL :
           TERM_FALLBACK
end

_term_codes(names::Vector{String}) = [_term_code(name) for name in names]

function Base.length(lib::CandidateLibrary)
    return length(lib.names)
end

"""
    get_term_names(lib::CandidateLibrary)

Return vector of term names.
"""
get_term_names(lib::CandidateLibrary) = lib.names

"""
    evaluate_library(lib::CandidateLibrary, data::Dict{String,Vector{Float64}})

Evaluate all library functions on data, returning matrix Θ (n_samples × n_terms).
"""
function evaluate_library(lib::CandidateLibrary, data::Dict{String,Vector{Float64}})
    n = length(first(values(data)))
    p = length(lib)
    Θ = zeros(n, p)
    for (j, f) in enumerate(lib.funcs)
        Θ[:, j] = f(data)
    end
    return Θ
end

"""
    build_solar_wind_library(; max_poly_order=2, include_trig=true,
                              include_cross=true, include_known=true)

Build candidate library for solar wind-magnetosphere coupling.

Variables: V (velocity), Bz (IMF z-component), By (IMF y-component),
           n (density), Pdyn (dynamic pressure), Dst (current state)

Terms include:
- Constant (1)
- Linear: V, Bs, n, Pdyn, Dst, Dst*
- Polynomial: V², Bs², n², VBs, nV, nBs, PdynBs, ...
- Trigonometric: sin(θ_c/2), sin²(θ_c/2), sin⁴(θ_c/2), sin^(8/3)(θ_c/2)
- Known coupling: VBs (Burton), V^(4/3)B_T^(2/3)sin^(8/3)(θ_c/2) (Newell)
- Decay: Dst/τ
"""
function build_solar_wind_library(; max_poly_order::Int=2,
                                    include_trig::Bool=true,
                                    include_cross::Bool=true,
                                    include_known::Bool=true)
    names = String[]
    funcs = Function[]

    # Constant term
    push!(names, "1")
    push!(funcs, d -> ones(length(d["V"])))

    # Linear terms
    for var in ["V", "Bs", "n", "Pdyn", "Dst_star"]
        push!(names, var)
        push!(funcs, let v=var; d -> d[v]; end)
    end

    # Quadratic terms
    if max_poly_order >= 2
        for var in ["V", "Bs", "n"]
            push!(names, "$(var)^2")
            push!(funcs, let v=var; d -> d[v].^2; end)
        end
    end

    # Cross terms
    if include_cross
        cross_pairs = [("V", "Bs"), ("n", "V"), ("n", "Bs"),
                       ("Pdyn", "Bs")]
        for (a, b) in cross_pairs
            push!(names, "$(a)*$(b)")
            push!(funcs, let aa=a, bb=b; d -> d[aa] .* d[bb]; end)
        end

        if max_poly_order >= 2
            push!(names, "n*V*Bs")
            push!(funcs, d -> d["n"] .* d["V"] .* d["Bs"])

            push!(names, "n*V^2")
            push!(funcs, d -> d["n"] .* d["V"].^2)
        end
    end

    # Trigonometric terms (IMF clock angle)
    if include_trig
        push!(names, "sin(θ_c/2)")
        push!(funcs, d -> sin.(d["theta_c"] ./ 2))

        push!(names, "sin²(θ_c/2)")
        push!(funcs, d -> sin.(d["theta_c"] ./ 2).^2)

        push!(names, "sin⁴(θ_c/2)")
        push!(funcs, d -> sin.(d["theta_c"] ./ 2).^4)

        push!(names, "sin^(8/3)(θ_c/2)")
        push!(funcs, d -> sin.(d["theta_c"] ./ 2).^(8/3))

        # V * sin terms
        push!(names, "V*sin²(θ_c/2)")
        push!(funcs, d -> d["V"] .* sin.(d["theta_c"] ./ 2).^2)
    end

    # Known coupling functions
    if include_known
        # Burton: V*Bs (already in cross terms, but explicit)
        # Newell universal coupling: V^(4/3) * B_T^(2/3) * sin^(8/3)(θ_c/2)
        push!(names, "Newell_d_Φ")
        push!(funcs, d -> begin
            BT = sqrt.(d["By"].^2 .+ d["Bz"].^2)
            d["V"].^(4/3) .* max.(BT, 1e-10).^(2/3) .* sin.(d["theta_c"] ./ 2).^(8/3)
        end)

        # Note: Dst* decay term is already included as linear "Dst_star"
        # The SINDy coefficient will absorb 1/τ
    end

    return CandidateLibrary(names, funcs)
end

"""
    build_minimal_library()

Minimal library for testing: constant, Dst*, V*Bs only.
"""
function build_minimal_library()
    names = ["1", "Dst_star", "V*Bs"]
    funcs = Function[
        d -> ones(length(d["V"])),
        d -> d["Dst_star"],
        d -> d["V"] .* d["Bs"]
    ]
    return CandidateLibrary(names, funcs)
end
