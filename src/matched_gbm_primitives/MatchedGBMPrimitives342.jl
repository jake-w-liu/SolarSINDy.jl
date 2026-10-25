"""Receipt-aware342-column numerical primitives; no serving or fit permission."""
module MatchedGBMPrimitives342

using Dates, SHA, Statistics, JSON3
const Parent=parentmodule(@__MODULE__)
const Features=Parent.IntervalMeanForecaster.DstIntervalAdapter.Features
const Duration=Parent.DurationConditionedForcing
const WIDTH=342
const BASE_WIDTH=165
const MAX_ROWS=500_000
const MAX_BYTES=4*1024^2
const HOUR_MS=Int64(3_600_000)
"SHA256 of the frozen ordered342-coordinate feature schema."
const FEATURE_SCHEMA_SHA256="109464af1630d16ac766b4f8791ad9f065a528d0aabac201a11fb68a4156249d"
const FEATURE_CONTRACT_SHA256="2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683"
const RAW_LAGS=(0,1,2,3,4,8,12,24)
const _RAW_NAMES=Tuple(Features.RAW_NAMES)
const _FILL_LIMITS=Tuple(Features.FILL_LIMITS)
const CONTEXT_NAMES=("latest_observation_age_hours","latest_eligible_wind_age_hours",
    "trace_state_n","trace_state_s","trace_age_known","trace_known_age_capped49",
    "trace_observed_run_capped49","trace_left_censored","trace_age49plus",
    "trace_age_unknown","latest_driver_unavailable","latest_driver_missing")
"Ordered342 names: maintained29, raw17-by8lags,165 missing flags,12 receipt/trace fields."
const FEATURE_NAMES=let
    base=vcat("gbm29/".*String.(Parent.direct_gbm_feature_names()),
        ["raw17/$(name)_raw_lag_$(lag)" for name in Features.RAW_NAMES for lag in RAW_LAGS])
    Tuple(vcat(base,"missing/".*base,"context/".*collect(CONTEXT_NAMES)))
end
_need(ok,message)=ok || throw(ArgumentError(message))
_int(x,lo,hi)=(_need(x isa Integer&&!(x isa Bool)&&lo<=x<=hi,"integer outside supported bounds");Int(x))
_hex(x)=x isa AbstractString&&occursin(r"\A[0-9a-f]{64}\z",x)
_shift(t::DateTime,h::Integer)=Duration._shift(t,h)
_bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
function _float(s)
    _need(s isa AbstractString&&occursin(r"\A[0-9a-f]{16}\z",s),"Float64 bit string differs")
    x=reinterpret(Float64,parse(UInt64,s;base=16));_need(isfinite(x),"nonfinite Float64 parameter");x
end

include("features.jl")
include("encoder.jl")
include("MatchedScalarTreeReplay342.jl")
include("codec.jl")
include("fitting.jl")
include("model_codec.jl")

const _SOURCE_PATHS=(@__FILE__,joinpath(@__DIR__,"features.jl"),joinpath(@__DIR__,"encoder.jl"),
    joinpath(@__DIR__,"MatchedScalarTreeReplay342.jl"),
    joinpath(@__DIR__,"codec.jl"),
    joinpath(@__DIR__,"fitting.jl"),joinpath(@__DIR__,"model_codec.jl"),
    joinpath(@__DIR__,"..","analog_features.jl"),joinpath(@__DIR__,"..","analog_gbm.jl"),
    joinpath(@__DIR__,"..","interval_mean_forecaster","sandu_fullchannel_features.jl"))
const _SOURCE_HASHES=Tuple(bytes2hex(sha256(read(path))) for path in _SOURCE_PATHS)
"Verify loaded component/upstream source bytes and return ordered basename-to-SHA pairs."
function source_identity()
    _need(all(bytes2hex(sha256(read(p)))==h for (p,h) in zip(_SOURCE_PATHS,_SOURCE_HASHES)),"loaded matched342 source changed")
    _need(Tuple(Features.RAW_NAMES)==_RAW_NAMES&&Tuple(Features.FILL_LIMITS)==_FILL_LIMITS,"maintained feature constants changed")
    Duration.source_identity();Parent.IntervalMeanForecaster.DstIntervalAdapter.verify_source_identity()
    Tuple(basename(p)=>h for (p,h) in zip(_SOURCE_PATHS,_SOURCE_HASHES))
end

export HourlyInputs342, FeatureRow342, FEATURE_NAMES, FEATURE_SCHEMA_SHA256,
    foreach_feature, build_features, trace_context, cohorts, Encoder342,
    fit_encoder, transform, source_identity, MatchedScalarTreeReplay342,
    save_encoder,load_encoder,preprocessing_sha256,
    SyntheticAdmission342,ExpectedModelBindings342,FittedCell342,
    prepare_synthetic_bindings,fit_synthetic_cell,save_model,load_model,GRID342
end
