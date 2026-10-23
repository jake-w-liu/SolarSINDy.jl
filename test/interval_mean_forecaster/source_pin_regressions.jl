module PackageSourcePinRegressions

using Test, Dates, SHA

const PARENT = parentmodule(@__MODULE__)
const IMF = getfield(PARENT, :IMF)
const ROOT = getfield(PARENT, :PACKAGE_ROOT)
const SOURCE = getfield(PARENT, :SOURCE_ROOT)
const OUT = getfield(PARENT, :OUT)
const OLD_STREAM = "7a6319dd3b572f9337c3573a50620cf666ada77c4b4c13bb995564807c1f037d"
const CURRENT_STREAM = "20a63654c429aa8be219f857b1e83a2529f334499e38198d8f332a07f22fcd44"
const EXPECTED = Dict(
    "DstIntervalObserver.jl" => "48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816",
    "DstIntervalReplay.jl" => "f2962c092bc6bfd6be82a5b8ad95e2c494dddbc4a2addda4f57140e31d0c6fa5",
    "DstIntervalAdapter.jl" => "b432a8fc65d3e425a67ff2b225a6c0e5323ae7fdb8822dd228677b7998a5fe5a",
    "DstIntervalStream.jl" => CURRENT_STREAM,
    "DstIntervalTraining.jl" => "2e13631e04cd48b36b033721935cee7dc47da76bd692f84ef0d3daae5c142b89",
    "DstIntervalArtifacts.jl" => "fbeb40ecfe1c9ff32a17ba6c97b9eadccdd4c88e924aeddeb7ce00616fc3c29e",
    "DstIntervalExperiment.jl" => "50e9bd1876af69a927f1bf6211925285f532e2f6a040cff5460b12c308d0fedd",
    "sandu_fullchannel_features.jl" => "2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c",
)
digest(path) = bytes2hex(sha256(read(path)))

@testset "Reviewed current stream source pin remains strict" begin
    @test getfield(PARENT, :SOURCE_PINS) == EXPECTED
    @test Dict(IMF.source_identity()) == EXPECTED
    @test digest(joinpath(SOURCE, "DstIntervalStream.jl")) == CURRENT_STREAM
    @test digest(joinpath(ROOT, "test", "b_stream_bridge", "frozen_a187", "DstIntervalStream.jl")) == OLD_STREAM
    @test CURRENT_STREAM != OLD_STREAM
    private = joinpath(OUT, "reviewed-stream-source-pin")
    mkdir(private)
    for (name, hash) in EXPECTED
        cp(joinpath(SOURCE, name), joinpath(private, name); force=false)
        @test digest(joinpath(private, name)) == hash
    end
    target = Module(gensym(:ReviewedCurrentStreamSource))
    for name in ("DstIntervalAdapter.jl", "DstIntervalStream.jl", "DstIntervalTraining.jl",
                 "DstIntervalArtifacts.jl", "DstIntervalExperiment.jl")
        Base.include(target, joinpath(private, name))
    end
    stream = getfield(target, :DstIntervalStream)
    experiment = getfield(target, :DstIntervalExperiment)
    @test Dict(experiment.verify_source_identity()) == EXPECTED
    values = ones(17, 180)
    values[8, :] .= 400.; values[6, :] .= -5.
    values[12, :] .= 2.; values[15, :] .= -30.
    input = stream.HourlyInputs([DateTime(2018)+Hour(i-1) for i in 1:180], values;
        source_sha256=bytes2hex(sha256("reviewed-current-stream-source-pin-synthetic-v1")))
    @test stream.stream_basis(input; delay_hours=0, taus=(2.,12.), sigma=1.).row_count == 180
    path = joinpath(private, "DstIntervalStream.jl")
    original = read(path)
    calls = Ref(0)
    try
        write(path, vcat(original, codeunits("\n# owned source-pin rejection probe\n")))
        laundered = copy(EXPECTED)
        laundered["DstIntervalStream.jl"] = digest(path)
        @test laundered != EXPECTED
        @test laundered["DstIntervalStream.jl"] != CURRENT_STREAM
        @test_throws ArgumentError stream.verify_source_identity()
        @test_throws ArgumentError experiment.verify_source_identity()
        @test_throws ArgumentError stream.stream_basis(input; delay_hours=0,
            taus=(2.,12.), sigma=1., issue_sink=payload -> (calls[]+=1))
        @test calls[] == 0
    finally
        write(path, original)
    end
    @test Dict(experiment.verify_source_identity()) == EXPECTED
    @test stream.stream_basis(input; delay_hours=0, taus=(2.,12.), sigma=1.).row_count == 180
    for (name, hash) in EXPECTED
        @test digest(joinpath(private, name)) == hash
    end
end

function check_stale_bundle(directory, manifest_sha256)
    stale = copy(EXPECTED)
    stale["DstIntervalStream.jl"] = OLD_STREAM
    @test_throws ArgumentError IMF.load_bundle(directory;
        expected_manifest_sha256=manifest_sha256, expected_sources=stale)
end

end
