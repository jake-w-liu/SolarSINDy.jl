using Test, Dates, SHA, TOML, SolarSINDy

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SOURCE_ROOT = joinpath(PACKAGE_ROOT, "src", "interval_mean_forecaster")
const FROZEN = joinpath(@__DIR__, "frozen")
const IMF = SolarSINDy.IntervalMeanForecaster
const SOURCE_PINS = Dict(
    "DstIntervalObserver.jl" => "48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816",
    "DstIntervalReplay.jl" => "08f8088e55b3b10bb189d4f89360dd72e30e2c5f0f162f7c687e92a7f4ce1165",
    "DstIntervalAdapter.jl" => "20f68a96ea7edaa88433bec70f1ffddb74128221862e8476f8d2e1704ddda22f",
    "DstIntervalStream.jl" => "7a6319dd3b572f9337c3573a50620cf666ada77c4b4c13bb995564807c1f037d",
    "DstIntervalTraining.jl" => "85c7a141bac5eefbba321df36140c2fd68cddcac3b04e79fa5fe5806261c89b5",
    "DstIntervalArtifacts.jl" => "fbeb40ecfe1c9ff32a17ba6c97b9eadccdd4c88e924aeddeb7ce00616fc3c29e",
    "DstIntervalExperiment.jl" => "50e9bd1876af69a927f1bf6211925285f532e2f6a040cff5460b12c308d0fedd",
    "sandu_fullchannel_features.jl" => "2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c",
)
const BASELINE_PINS = Dict(
    "DstIntervalObserver.jl" => "657cf432ffdcc09f3058a57e79e327c159292fe6e98318d4f0dc6bc917b4a04c",
    "DstIntervalReplay.jl" => "08f8088e55b3b10bb189d4f89360dd72e30e2c5f0f162f7c687e92a7f4ce1165",
    "DstIntervalAdapter.jl" => "9ec8c88d430be03937709f787dd85bd251c628f2d3d462a4991f677a7fdda3c2",
    "DstIntervalStream.jl" => "65ffc4c8568c00e563c5a58662b634a5da361c913253b5c5dca3721edb5b7dd8",
    "DstIntervalTraining.jl" => "d3860019346736857216c0ac83c24432bfc2f887ada46585c1a7fa4f8efd30a2",
    "DstIntervalArtifacts.jl" => "ca549ba82247e68b7312edb3573700caa12a035ebe24541a5dc9ca8a72a5f2ac",
    "DstIntervalExperiment.jl" => "50e9bd1876af69a927f1bf6211925285f532e2f6a040cff5460b12c308d0fedd",
    "sandu_fullchannel_features.jl" => "2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c",
)
hashfile(path) = bytes2hex(sha256(read(path)))

function verify_bindings()
    @test realpath(pathof(SolarSINDy)) == realpath(joinpath(PACKAGE_ROOT, "src", "SolarSINDy.jl"))
    @test Dict(IMF.source_identity()) == SOURCE_PINS
    for (name, digest) in SOURCE_PINS
        @test hashfile(joinpath(SOURCE_ROOT, name)) == digest
        @test hashfile(joinpath(FROZEN, "baseline", name)) == BASELINE_PINS[name]
    end
    # These are actual generic functions/types imported by the maintained package.
    for (f, file) in ((IMF.interval_kernel, "DstIntervalObserver.jl"),
                      (IMF.DstIntervalAdapter.Replay.replay_asof, "DstIntervalReplay.jl"),
                      (IMF.read_author_hourly, "sandu_fullchannel_features.jl"),
                      (IMF.nnls2, "DstIntervalAdapter.jl"),
                      (IMF.stream_basis, "DstIntervalStream.jl"),
                      (IMF.fit_gains, "DstIntervalTraining.jl"),
                      (IMF.save_bundle, "DstIntervalArtifacts.jl"),
                      (IMF.run_synthetic_experiment, "DstIntervalExperiment.jl"))
        @test all(m -> realpath(String(m.file)) == realpath(joinpath(SOURCE_ROOT, file)), methods(f))
    end
    pinfile = joinpath(FROZEN, "pins.toml")
    @test hashfile(pinfile) == "b16190516e455d4f3178d55b4d6bdf8e36b0524013f99bfbcd44a7677bef2017"
    frozen_pins = TOML.parsefile(pinfile)["files"]
    @test length(frozen_pins) == 40
    for (relative, digest) in frozen_pins
        @test hashfile(joinpath(FROZEN, split(relative, '/')...)) == digest
    end
    @test hashfile(joinpath(@__DIR__, "peer_freeze", "literal_cases.jl")) ==
        "e0c2ca7b2c1a11e691283f2172b73c6989e887858264c8470595a1f144c09885"
    @test hashfile(joinpath(@__DIR__, "peer_freeze", "scanner_oracle.jl")) ==
        "f9b551b4f40ed315d9cdd930a2b7537debc1a4f9521367d8b224864952016c62"
    true
end

function bind_modules!(target, namespace=IMF)
    for (name, value) in (
        :DstIntervalAdapter => namespace.DstIntervalAdapter,
        :DstIntervalReplay => namespace.DstIntervalAdapter.Replay,
        :DstIntervalObserver => namespace.DstIntervalAdapter.Kernel,
        :DstIntervalStream => namespace.DstIntervalStream,
        :DstIntervalTraining => namespace.DstIntervalTraining,
        :DstIntervalArtifacts => namespace.DstIntervalArtifacts,
        :DstIntervalExperiment => namespace.DstIntervalExperiment)
        Core.eval(target, :(const $name = $value))
    end
    target
end

function baseline_namespace()
    target = Module(gensym(:FrozenUncachedBaseline))
    for name in ("DstIntervalAdapter.jl", "DstIntervalStream.jl", "DstIntervalTraining.jl",
                 "DstIntervalArtifacts.jl", "DstIntervalExperiment.jl")
        Base.include(target, joinpath(FROZEN, "baseline", name))
    end
    target
end

# Frozen assertions are kept as bytes. This expression mapper changes only the
# original source-loading statements and source-path metadata, never assertions.
function bound_expression(node)
    node isa Expr || return node
    if node.head === :call && node.args[1] === :include && length(node.args) == 2 && node.args[2] isa String
        file = node.args[2]
        if haskey(SOURCE_PINS, file)
            return :(nothing) # the aliases above are the package-loaded modules
        elseif file == "test_training.jl"
            return :(Main.bound_include(@__MODULE__, $(joinpath(FROZEN, "original", file))))
        end
    elseif node.head === :macrocall && node.args[1] === Symbol("@__DIR__")
        return SOURCE_ROOT
    end
    Expr(node.head, (bound_expression(child) for child in node.args)...)
end
bound_include(target, path) = Base.include(bound_expression, target, path)

function evaluate_tail(target, path, marker, prefix)
    original = read(path, String)
    parts = split(original, marker; limit=2)
    length(parts) == 2 || error("frozen binding marker absent: $path")
    # The marker and everything following it retain their original bytes.
    Base.include_string(target, prefix * marker * parts[2], path)
end

function set_arguments(values)
    empty!(ARGS)
    append!(ARGS, values)
end
