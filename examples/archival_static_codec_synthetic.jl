module ArchivalStaticCodecSynthetic
using SHA,JSON3,SolarSINDy
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_shared_replay.jl"))
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_archival_static.jl"))
const AS=SourceBoundArchivalStatic
"""Roundtrip independent opaque fixture bytes, without archival access or fitting."""
function run_example()
    oracle=JSON3.read(read(joinpath(@__DIR__,"..","test","publisher_up_values","archival_static_codec_oracle.json"),String),Dict{String,Any})
    context=hex2bytes(oracle["context_hex"]);numeric=hex2bytes(oracle["numeric_hex"]);receipt=hex2bytes(oracle["receipt_hex"])
    expected=AS.ExpectedStatic(oracle["context_sha256"],oracle["numeric_sha256"])
    @assert oracle["fixture_only"]===true && oracle["actual_source_authority_granted"]===false
    @assert length(numeric)==186 && bytes2hex(sha256(numeric))=="1449b3e36a0d75a728654e0b5229a351ba2c88ae280b3c5a49bbaa10ed0e65aa"
    mktempdir() do root
        path=joinpath(root,"opaque-static-fixture")
        prepared=AS._prepare_publication(path,context,numeric,receipt,expected)
        @assert !ispath(joinpath(path,"COMPLETE.json"))
        pins=AS._complete_publication(prepared)
        restored=AS._read_publication(path,expected;pins...)
        @assert restored.context==context && restored.numeric==numeric && restored.receipt==receipt && restored.codec_only
        values=SourceBoundSharedReplay.B._decode(restored.numeric)
        @assert values.weights==[2.,0.,0.] && values.raw==5. && values.normalized==2.5 && values.alpha==.5
        (;codec_only=true,fixture_only=true,actual_source_access=false,actual_point_model_fits=0,uncertainty_parameter_fits=0)
    end
end
end
