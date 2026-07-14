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
    # The public vectors predate the optimized point evaluator and remain
    # mutable for compatibility.  Preserve their construction-time semantics
    # in immutable tuples so coordinated vector mutation cannot silently make
    # batch and point evaluation implement different equations.
    _contract_names::Tuple
    _contract_funcs::Tuple
    _contract_term_codes::Tuple
    function CandidateLibrary(names::Vector{String}, funcs::Vector{Function},
                              term_codes::Vector{UInt8}, ::Val{:trusted_codes})
        length(names) == length(funcs) == length(term_codes) ||
            throw(DimensionMismatch(
                "candidate-library lengths differ: names=$(length(names)), " *
                "funcs=$(length(funcs)), term_codes=$(length(term_codes))"
            ))
        isempty(names) && throw(ArgumentError("candidate library must contain at least one term"))
        all(name -> !isempty(strip(name)), names) ||
            throw(ArgumentError("candidate-library names must not be empty or whitespace"))
        length(unique(names)) == length(names) ||
            throw(ArgumentError("candidate-library names must be unique"))
        return new(copy(names), copy(funcs), copy(term_codes),
                   Tuple(names), Tuple(funcs), Tuple(term_codes))
    end
end

CandidateLibrary(names::Vector{String}, funcs::Vector{Function}) =
    CandidateLibrary(names, funcs, fill(UInt8(0), length(names)), Val(:trusted_codes))

# Term codes bypass user-supplied functions in the point evaluator, so only the
# package's canonical builders may opt into them. Inferring a code from a custom
# term's name can make `evaluate_library` and `simulate_sindy` implement different
# mathematics (e.g. a user function named "V" that intentionally returns 2V).
_fast_candidate_library(names::Vector{String}, funcs::Vector{Function}) =
    CandidateLibrary(names, funcs, _term_codes(names), Val(:trusted_codes))

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

# A non-fallback code is the term's mathematical contract.  Evaluate coded
# terms from that contract in batch mode as well as in the optimized point
# kernel.  This also makes direct use of the low-level trusted-code constructor
# safe: a caller-supplied function cannot redefine a coded built-in term.
function _evaluate_coded_library_term(
    code::UInt8,
    data::Dict{String,Vector{Float64}},
    n_samples::Int,
)
    code == TERM_ONE && return ones(n_samples)
    code == TERM_V && return data["V"]
    code == TERM_BS && return data["Bs"]
    code == TERM_N && return data["n"]
    code == TERM_PDYN && return data["Pdyn"]
    code == TERM_DST_STAR && return data["Dst_star"]
    code == TERM_V2 && return data["V"].^2
    code == TERM_BS2 && return data["Bs"].^2
    code == TERM_N2 && return data["n"].^2
    code == TERM_V_BS && return data["V"] .* data["Bs"]
    code == TERM_N_V && return data["n"] .* data["V"]
    code == TERM_N_BS && return data["n"] .* data["Bs"]
    code == TERM_PDYN_BS && return data["Pdyn"] .* data["Bs"]
    code == TERM_N_V_BS && return data["n"] .* data["V"] .* data["Bs"]
    code == TERM_N_V2 && return data["n"] .* data["V"].^2

    sin_half = sin.(data["theta_c"] ./ 2)
    code == TERM_SIN_HALF && return sin_half
    code == TERM_SIN_HALF2 && return sin_half.^2
    code == TERM_SIN_HALF4 && return sin_half.^4
    code == TERM_SIN_HALF_8_3 && return sin_half.^(8 / 3)
    code == TERM_V_SIN_HALF2 && return data["V"] .* sin_half.^2
    if code == TERM_NEWELL
        bt = hypot.(data["By"], data["Bz"])
        return map(_newell_coupling_value, data["V"], bt, sin_half)
    end
    throw(ArgumentError("unknown optimized candidate-library term code $code"))
end

function Base.length(lib::CandidateLibrary)
    return length(lib.names)
end

# CandidateLibrary is immutable only at the outer struct level; its three
# vectors remain mutable for backwards compatibility.  Keep the constant-time
# structural guard separate so point-evaluation hot loops can fail closed before
# indexing parallel vectors without paying for a full semantic audit per sample.
function _validate_candidate_library_structure(lib::CandidateLibrary)
    n_terms = length(lib.names)
    n_terms >= 1 || throw(ArgumentError(
        "candidate library must contain at least one term",
    ))
    length(lib.funcs) == n_terms && length(lib.term_codes) == n_terms &&
        length(lib._contract_names) == n_terms &&
        length(lib._contract_funcs) == n_terms &&
        length(lib._contract_term_codes) == n_terms ||
        throw(DimensionMismatch(
            "candidate-library vectors and semantic contract must have equal lengths",
        ))
    return n_terms
end

function _validate_candidate_library(lib::CandidateLibrary)
    n_terms = _validate_candidate_library_structure(lib)
    all(name -> !isempty(strip(name)), lib.names) || throw(ArgumentError(
        "candidate-library names must not be empty or whitespace",
    ))
    length(unique(lib.names)) == n_terms || throw(ArgumentError(
        "candidate-library names must be unique",
    ))
    for index in 1:n_terms
        name = lib._contract_names[index]
        code = lib._contract_term_codes[index]
        lib.names[index] == name || throw(ArgumentError(
            "candidate-library term name was mutated after construction",
        ))
        lib.funcs[index] === lib._contract_funcs[index] || throw(ArgumentError(
            "candidate-library term function was mutated after construction",
        ))
        lib.term_codes[index] == code || throw(ArgumentError(
            "candidate-library term code was mutated after construction",
        ))
        code == TERM_FALLBACK || code == _term_code(name) ||
            throw(ArgumentError(
                "candidate-library term code is inconsistent with term $name",
            ))
    end
    return n_terms
end

"""
    get_term_names(lib::CandidateLibrary)

Return a copy of the term-name vector. Mutating the result cannot desynchronise
the library's names from its optimized point-evaluation term codes.
"""
function get_term_names(lib::CandidateLibrary)
    _validate_candidate_library(lib)
    return collect(String, lib._contract_names)
end

"""
    evaluate_library(lib::CandidateLibrary, data::Dict{String,Vector{Float64}})

Evaluate all library functions on data, returning matrix Θ (n_samples × n_terms).
"""
function evaluate_library(lib::CandidateLibrary, data::Dict{String,Vector{Float64}})
    _validate_candidate_library(lib)
    isempty(data) && throw(ArgumentError("cannot evaluate a candidate library on empty data"))
    n = length(first(values(data)))
    n >= 1 || throw(ArgumentError("candidate-library data must contain at least one sample"))
    all(length(v) == n for v in values(data)) ||
        throw(DimensionMismatch("all candidate-library data vectors must have equal length"))
    all(v -> all(isfinite, v), values(data)) ||
        throw(ArgumentError("candidate-library data must contain only finite values"))
    p = length(lib)
    Θ = zeros(n, p)
    for (j, f) in enumerate(lib._contract_funcs)
        code = lib._contract_term_codes[j]
        values_j = code == TERM_FALLBACK ?
                   f(data) : _evaluate_coded_library_term(code, data, n)
        values_j isa AbstractVector || throw(ArgumentError(
            "candidate-library term $(lib._contract_names[j]) must return a vector"
        ))
        length(values_j) == n || throw(DimensionMismatch(
            "candidate-library term $(lib._contract_names[j]) returned $(length(values_j)) values; expected $n"
        ))
        all(value -> value isa Real && !(value isa Bool) && isfinite(value), values_j) ||
            throw(ArgumentError(
                "candidate-library term $(lib._contract_names[j]) returned a non-finite or non-real value"
            ))
        converted = Float64.(values_j)
        all(isfinite, converted) || throw(ArgumentError(
            "candidate-library term $(lib._contract_names[j]) exceeds the supported Float64 range",
        ))
        Θ[:, j] = converted
    end
    return Θ
end

"""
    build_solar_wind_library(; max_poly_order=2, include_trig=true,
                              include_cross=true, include_known=true,
                              include_redundant_n_v2=false,
                              clock_basis=:full)

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

The canonical library omits `n*V^2` because the package's proton-only pressure
convention makes it exactly proportional to `Pdyn`; including both makes their
individual coefficients non-identifiable. Set `include_redundant_n_v2=true`
only to reproduce or load a legacy 21-term artifact.

`clock_basis=:full` retains the five correlated empirical clock-angle terms.
`clock_basis=:collapsed` omits those proxy columns and, when
`include_known=true`, represents clock-angle dependence only through the Newell
coupling term. The collapsed option is intended as an identifiability control.
"""
function build_solar_wind_library(; max_poly_order::Int=2,
                                    include_trig::Bool=true,
                                    include_cross::Bool=true,
                                    include_known::Bool=true,
                                    include_redundant_n_v2::Bool=false,
                                    clock_basis::Symbol=:full)
    max_poly_order in (1, 2) ||
        throw(ArgumentError("max_poly_order must be 1 or 2, got $max_poly_order"))
    clock_basis in (:full, :collapsed) || throw(ArgumentError(
        "clock_basis must be :full or :collapsed, got $clock_basis"
    ))
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

            if include_redundant_n_v2
                push!(names, "n*V^2")
                push!(funcs, d -> d["n"] .* d["V"].^2)
            end
        end
    end

    # Trigonometric terms (IMF clock angle)
    if include_trig && clock_basis == :full
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
            BT = hypot.(d["By"], d["Bz"])
            map(_newell_coupling_value, d["V"], BT,
                sin.(d["theta_c"] ./ 2))
        end)

        # Note: Dst* decay term is already included as linear "Dst_star"
        # The SINDy coefficient will absorb 1/τ
    end

    return _fast_candidate_library(names, funcs)
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
    return _fast_candidate_library(names, funcs)
end
