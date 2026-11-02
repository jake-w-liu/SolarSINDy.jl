include(joinpath(@__DIR__, "bootstrap_indices.jl"))
using .CMEGSMBootstrap, TOML, SHA
include(joinpath(@__DIR__, "test_bootstrap_indices.jl"))
include(joinpath(@__DIR__, "test_native_generated_artifact.jl"))

function arguments(values)
    result = Dict{String,String}()
    length(values) % 2 == 0 || error("paired CLI arguments required")
    for index in 1:2:length(values)
        key = values[index]
        key in ("--output", "--role", "--source-head", "--runtime", "--primary") || error("unknown CLI argument")
        haskey(result, key) && error("duplicate CLI argument")
        result[key] = values[index+1]
    end
    all(haskey(result, key) for key in ("--output", "--role", "--source-head", "--runtime")) || error("missing CLI argument")
    result
end

function main(values)
    options = arguments(values)
    role = options["--role"]
    (role == "control") == haskey(options, "--primary") || error("only controls require the primary reference")
    result = CMEGSMBootstrap.generate(options["--output"]; role, source_head=options["--source-head"],
        expected_runtime=VersionNumber(options["--runtime"]))
    test_native_generated_artifact(result; role, source_head=options["--source-head"])
    if role == "control"
        primary = options["--primary"]
        manifest = TOML.parsefile(joinpath(primary, "INDEX_MANIFEST.toml"))
        manifest["role"] == "primary" && manifest["runtime"] == "1.12.6" || error("frozen primary identity differs")
        reference = CMEGSMBootstrap.validate_artifact(primary; expected_source_head=options["--source-head"],
            expected_matrix_sha256=manifest["matrix_sha256"])
        same = result.matrix_sha256 == reference.matrix_sha256 &&
            read(joinpath(result.directory, "BOOTSTRAP_INDICES.csv")) == read(joinpath(primary, "BOOTSTRAP_INDICES.csv"))
        CMEGSMBootstrap.exclusive_toml(result.directory*".comparison.toml", Dict(
            "schema"=>"cme_gsm_bootstrap_control_comparison_v1", "primary_matrix_sha256"=>reference.matrix_sha256,
            "control_matrix_sha256"=>result.matrix_sha256, "runtime"=>string(VERSION), "same_bytes"=>same,
            "primary_replaced"=>false, "source_authority_granted"=>false))
        same || error("measured control RNG bytes differ; no outcome access is permitted")
    end
    rss = Sys.maxrss()
    CMEGSMBootstrap.exclusive_toml(result.directory*".resource.toml", Dict("schema"=>"bootstrap_native_resource_v1",
        "runtime"=>string(VERSION), "peak_Julia_process_RSS_bytes"=>rss, "RSS_limit_bytes"=>8*1024^3,
        "source_authority_granted"=>false))
    rss <= 8*1024^3 || error("native Julia peak RSS exceeded8GiB")
    println("BOOTSTRAP_NATIVE_GENERATION_COMPLETE;ROLE=", role, ";RUNTIME=", VERSION,
        ";DRAWS=", result.draws, ";K=", result.K, ";MATRIX_SHA256=", result.matrix_sha256,
        ";SOURCE_AUTHORITY=false;SCIENCE_QUALIFICATION=false")
    nothing
end

main(ARGS)
