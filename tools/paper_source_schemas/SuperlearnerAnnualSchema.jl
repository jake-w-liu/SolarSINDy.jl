module SuperlearnerAnnualSchema

using CSV
using DataFrames
using SHA

export HeaderPlan, AnnualInputPlan, header_plan, read_header_plan,
       prepare_annual_inputs, adapt_annual_columns!, read_annual_frame

const YEARS = Tuple(2014:2025)
const ANNUAL_NAMES = Tuple("learn_year_$(year).csv" for year in YEARS)
const HEADER_ALIASES = (
    "v2_4e" => "superlearner_e",
    "v2_4e_lo_nt" => "superlearner_e_lo_nt",
    "v2_4e_hi_nt" => "superlearner_e_hi_nt",
    "static_v2_2" => "static_stack",
    "v2_3_shadow" => "analog_shadow",
    "served_v2_1" => "served_inertial_sindy",
)
const REQUIRED_COLUMNS = (
    "model_step_hours", "observation_dst_nt", "depth_bin",
    "superlearner_e", "superlearner_e_lo_nt", "superlearner_e_hi_nt",
    "static_stack", "analog_shadow", "direct_gbm", "served_inertial_sindy",
)
const _ALIASES = Dict(HEADER_ALIASES)

struct HeaderPlan
    raw_names::Tuple{Vararg{String}}
    canonical_names::Tuple{Vararg{String}}
    selected_source_names::Tuple{Vararg{String}}
    selected_canonical_names::Tuple{Vararg{String}}
end

struct AnnualInputPlan
    year::Int
    path::String
    raw_bytes::Int
    raw_sha256::String
    header::HeaderPlan
end

_require(ok, message) = ok ? nothing : throw(ArgumentError(message))

"Validate literal column names without inspecting or transforming any values."
function header_plan(raw_names::AbstractVector{<:AbstractString})::HeaderPlan
    raw = Tuple(String.(raw_names))
    _require(!isempty(raw) && all(name -> !isempty(name), raw), "annual CSV header contains an empty name")
    _require(length(Set(raw)) == length(raw), "annual CSV contains duplicate raw headers")
    for (legacy, canonical) in HEADER_ALIASES
        _require(!(legacy in raw && canonical in raw),
                 "annual CSV contains both $legacy and $canonical")
    end
    canonical = Tuple(get(_ALIASES, name, name) for name in raw)
    _require(length(Set(canonical)) == length(canonical), "annual CSV canonical headers collide")
    _require(all(name -> name in canonical, REQUIRED_COLUMNS),
             "annual CSV is missing a required center, bound or comparator header")
    selected = Tuple(i for i in eachindex(raw) if canonical[i] in REQUIRED_COLUMNS)
    HeaderPlan(raw, canonical, Tuple(raw[i] for i in selected),
               Tuple(canonical[i] for i in selected))
end

function _ordinary_file(path)
    _require(isfile(path) && !islink(path), "ordinary annual CSV file required: $path")
    path
end

"Read the first record as string data so CSV cannot uniquify raw header names."
function read_header_plan(path::AbstractString)::HeaderPlan
    _ordinary_file(path)
    first_record = CSV.read(path, DataFrame; header=false, limit=1, ntasks=1,
                           types=String, delim=',', normalizenames=false,
                           missingstring=nothing, ignoreemptyrows=false, strict=true)
    _require(nrow(first_record) == 1, "annual CSV must have a first header record")
    raw = String[]
    for column in 1:ncol(first_record)
        value = first_record[1, column]
        _require(value isa AbstractString && !isempty(value), "annual CSV header name is empty")
        push!(raw, String(value))
    end
    header_plan(raw)
end

function _verify_raw(path, expected_bytes, expected_sha)
    _ordinary_file(path)
    _require(filesize(path) == expected_bytes, "annual CSV raw size differs: $(basename(path))")
    actual = open(io -> bytes2hex(sha256(io)), path, "r")
    _require(actual == expected_sha && filesize(path) == expected_bytes,
             "annual CSV raw digest or size differs: $(basename(path))")
    nothing
end

"Admit all twelve raw byte references first, then all twelve complete headers."
function prepare_annual_inputs(directory::AbstractString, expected::AbstractDict)
    _require(length(expected) == length(ANNUAL_NAMES) &&
             all(name -> name isa AbstractString, keys(expected)) &&
             Set(String.(keys(expected))) == Set(ANNUAL_NAMES),
             "annual manifest must contain exactly the twelve final 2014:2025 files")
    _require(isdir(directory), "annual CSV directory is missing")
    paths = Vector{String}(undef, length(YEARS))
    pins = Vector{Tuple{Int,String}}(undef, length(YEARS))
    # No header parser runs until every original raw size and SHA has passed.
    for (index, name) in enumerate(ANNUAL_NAMES)
        reference = expected[name]
        _require(reference isa Tuple && length(reference) == 2,
                 "annual manifest reference must be a (raw size, SHA256) tuple")
        raw_bytes, raw_sha256 = reference
        _require(raw_bytes isa Integer && !(raw_bytes isa Bool) &&
                 0 < raw_bytes <= typemax(Int), "annual raw size must be a positive integer")
        _require(raw_sha256 isa AbstractString && occursin(r"^[0-9a-f]{64}$", raw_sha256),
                 "annual raw SHA256 must have its exact lowercase hexadecimal form")
        paths[index] = abspath(joinpath(directory, name))
        pins[index] = (Int(raw_bytes), String(raw_sha256))
        _verify_raw(paths[index], pins[index][1], pins[index][2])
    end
    ntuple(length(YEARS)) do index
        AnnualInputPlan(YEARS[index], paths[index], pins[index][1], pins[index][2],
                        read_header_plan(paths[index]))
    end
end

"Rename only column labels; keep the same frame, buffers, values and order."
function adapt_annual_columns!(frame::DataFrame)
    plan = header_plan(names(frame))
    rename!(frame, collect(plan.canonical_names); makeunique=false)
    frame
end

function _same_header(left::HeaderPlan, right::HeaderPlan)
    left.raw_names == right.raw_names && left.canonical_names == right.canonical_names &&
        left.selected_source_names == right.selected_source_names &&
        left.selected_canonical_names == right.selected_canonical_names
end

"Use the existing payload parser, then apply the checked label-only mapping."
function read_annual_frame(plan::AnnualInputPlan)::DataFrame
    _require(plan.year in YEARS && basename(plan.path) == "learn_year_$(plan.year).csv" &&
             plan.raw_bytes > 0 && occursin(r"^[0-9a-f]{64}$", plan.raw_sha256),
             "annual input plan has an invalid year, filename or raw reference")
    _verify_raw(plan.path, plan.raw_bytes, plan.raw_sha256)
    _require(_same_header(read_header_plan(plan.path), plan.header), "annual CSV header plan changed")
    # These are the builder's original payload read options. Header-probe types,
    # missing handling and threading options do not affect payload columns.
    frame = CSV.read(plan.path, DataFrame; select=Symbol.(collect(plan.header.selected_source_names)))
    _require(Tuple(names(frame)) == plan.header.selected_source_names,
             "annual selected source column order differs")
    adapt_annual_columns!(frame)
    _require(Tuple(names(frame)) == plan.header.selected_canonical_names,
             "annual selected canonical column order differs")
    _verify_raw(plan.path, plan.raw_bytes, plan.raw_sha256)
    frame
end

end
