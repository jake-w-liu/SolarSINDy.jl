# The caller has already loaded and source-verified SolarSINDy's actual namespace.
# API cases use that namespace. A separately pinned historical codec is used
# only for valid-byte comparison; no observational input or selection occurs.
const BOUNDARY_RESOURCE_PINS = Dict(
    "digest_api_regressions.jl" => "232327624eda8804904c1aeb1a3695587858cc36b0d3c66ca7a280e5f12ae169",
    "package_grammar_adapters_v1.jl" => "6d993aec4f7a27a3e185e3651ef6f0a604d6ba534bdad8fbdd5499356048e295",
    "producer_cases_frozen_v2.jl" => "6a6b637f1da002110d7e241293a7487c4592f1dcd70634849435bc399580e10c",
    "training_provenance_binding.jl" => "b9b8e214e79bdf43ba4657c3c4e42157849474399bddce7046c09cfab77405f4",
)
for (name, digest) in BOUNDARY_RESOURCE_PINS
    @test hashfile(joinpath(@__DIR__, name)) == digest
end
include("digest_api_regressions.jl")
include("producer_cases_frozen_v2.jl")
include("package_grammar_adapters_v1.jl")
const HISTORICAL_CODEC = baseline_namespace().DstIntervalArtifacts

@testset "Package-loaded strict grammar and producer admission" begin
    test_training_provenance_binding()
    run_digest_api_regressions(IMF)

    # Bounded generated fixture follows the maintained synthetic example's
    # explicit recipe and fitting boundary. It does not read an archive.
    origin = DateTime(2018)
    n = 900
    values = ones(17, n)
    values[8, :] .= 400.
    values[6, :] = [-6 + 2sin(i/31) for i in 1:n]
    values[12, :] = [2 + .2cos(i/41) for i in 1:n]
    values[15, :] = [-70 + 10sin(i/23) for i in 1:n]
    rawpath = joinpath(OUT, "generated-hourly.lst")
    ispath(rawpath) && error("fresh synthetic output path required")
    open(rawpath, "w") do io
        for i in 1:n
            tag = origin + Hour(i-1)
            fields = fill("0", 57)
            fields[1:3] .= string.((year(tag), dayofyear(tag), hour(tag)))
            for (channel, column) in enumerate(IMF.DstIntervalAdapter.Features.RAW_COLUMNS)
                fields[column] = repr(values[channel, i])
            end
            println(io, join(fields, " "))
        end
    end
    digest = hashfile(rawpath)
    input = IMF.read_author_hourly(rawpath; expected_sha256=digest)
    @test input.values == values
    rows = NamedTuple[]
    streamed = IMF.stream_basis(input; delay_hours=0, taus=(4.,24.), sigma=3.,
        chunk_hours=24, sink=chunk -> append!(rows, chunk))
    @test length(rows) == streamed.row_count == n
    @test streamed.max_buffered_rows <= 24
    model = IMF.fit_gains(rows, origin+Hour(700), IMF.settings()[9];
        delay_hours=0, source_sha256=digest)
    @test length(IMF.predict_rows(model, rows)) == n

    # Historical serializer comparison is distinct from the independent byte
    # grammar oracle and from codec self-roundtrip validation. Both encoders
    # receive the identical public payload, including truthful current pins.
    payloads = (model, IMF.DstIntervalArtifacts._checkpoint_data(streamed.checkpoint),
        (case.value for case in producer_cases_frozen_v2() if case.valid)...)
    for value in payloads
        historical = HISTORICAL_CODEC._toml_bytes(Dict("value"=>HISTORICAL_CODEC._encode(value)))
        @test IMF.DstIntervalArtifacts._typed_payload_bytes(value) == historical
    end
    typed = run_frozen_typed_codec_cases(IMF, model, streamed.checkpoint, OUT)
    producer = run_frozen_producer_cases(IMF, model, streamed.checkpoint, OUT)
    @test (typed.direct_cases, typed.public_cases, producer.producer_cases) == (22,3,16)
end
