# Native functional worker. No benchmarks or optional skip paths are present.
length(ARGS) == 2 || error("worker requires job and owned evidence directory")
const JOB = ARGS[1]
const OUT = abspath(ARGS[2])
isdir(OUT) || error("parent must create exclusive evidence directory")
include("bindings.jl")
using LinearAlgebra
const CONTEXT_BEFORE = (precision(BigFloat), rounding(BigFloat))
const BEFORE = Dict(IMF.source_identity())
verify_bindings()

if JOB in ("training362", "grammar-producer")
    @test hashfile(joinpath(@__DIR__, "boundary_regressions", "training_provenance_binding.jl")) == "b9b8e214e79bdf43ba4657c3c4e42157849474399bddce7046c09cfab77405f4"
    include(joinpath(@__DIR__, "boundary_regressions", "training_provenance_binding.jl"))
end

if JOB == "disk-fit"
    include(joinpath(@__DIR__, "disk_fit", "run.jl"))
elseif JOB == "grammar-producer"
    @test hashfile(joinpath(@__DIR__, "boundary_regressions", "run.jl")) == "e1f6512dcf65549343796d5feecaf31be95e4be6fe69c29814258b24f81ce075"
    include(joinpath(@__DIR__, "boundary_regressions", "run.jl"))
elseif JOB == "scanner"
    set_arguments(["verify"])
    scanner_target = Module(gensym(:DependencyScannerOracle))
    Core.eval(scanner_target, :(include(path) = Base.include($scanner_target, path)))
    Base.include(scanner_target, joinpath(@__DIR__, "peer_freeze", "scanner_oracle.jl"))
elseif JOB == "smoke"
    @testset "Actual maintained namespace smoke" begin
        @test IMF.HourlyInputs === IMF.DstIntervalAdapter.Features.HourlyInputs
        @test IMF.interval_kernel(0., 2., 3.).phi === 1.
        n = 900
        values = ones(17, n); values[8, :] .= 400.; values[6, :] .= -5.
        values[12, :] .= 2.; values[15, :] .= -30.
        digest = bytes2hex(sha256("maintained namespace synthetic smoke"))
        input = IMF.HourlyInputs([DateTime(2018) + Hour(i-1) for i in 1:n], values; source_sha256=digest)
        for delay in (0, 1)
            rows = NamedTuple[]
            streamed = IMF.stream_basis(input; delay_hours=delay, taus=(4., 24.), sigma=3.,
                chunk_hours=7, sink=part -> append!(rows, part))
            @test streamed.row_count == length(rows) == n
            @test rows[168].model_available
            model = IMF.fit_gains(rows, DateTime(2018)+Hour(700), IMF.settings()[9];
                delay_hours=delay, source_sha256=digest)
            @test length(IMF.predict_rows(model, rows)) == n
            saved = IMF.save_bundle(joinpath(OUT, "smoke-L$delay"), model, streamed.checkpoint;
                provenance=(synthetic_fixture=true, input_sha256=digest))
            loaded = IMF.load_bundle(saved.directory; expected_manifest_sha256=saved.manifest_sha256,
                expected_sources=IMF.DstIntervalArtifacts.verify_source_identity())
            @test isequal(loaded.model, model)
            @test isequal(loaded.checkpoint, streamed.checkpoint)
        end
    end
elseif JOB == "independent2272"
    const DstIntervalObserver = IMF.DstIntervalAdapter.Kernel
    const DstIntervalReplay = IMF.DstIntervalAdapter.Replay
    for file in ("test_oracle_self_20261015.jl", "test_fixed_temporal_self_20261015.jl",
                 "test_interval_moments_peer_20261015.jl", "test_condition_peer_20261015.jl",
                 "test_replay_peer_20261015.jl", "test_cache_targets_affine_peer_20261015.jl")
        include(joinpath(FROZEN, "independent", file))
    end
elseif JOB == "original"
    for file in ("test_kernels.jl", "test_replay.jl", "test_replay_boundaries.jl",
                 "test_primitives.jl", "test_stream.jl", "test_training.jl",
                 "test_reference_clocks.jl", "test_artifacts.jl", "test_experiment.jl")
        original_target = bind_modules!(Module(gensym(:PackageBoundOriginal)))
        bound_include(original_target, joinpath(FROZEN, "original", file))
    end
    # The portable copy has the identical original assertion body and only
    # removes the original Windows-only temporary-root assumption.
    bound_include(Module(gensym(:PrivateIdentity)), joinpath(FROZEN, "original", "test_loaded_identity_portable.jl"))
elseif JOB in ("public-baseline", "public-candidate")
    const SOURCE = joinpath(SOURCE_ROOT, "DstIntervalObserver.jl")
    const MODE = JOB == "public-baseline" ? "record" : "verify"
    const OUTDIR = dirname(OUT)
    if JOB == "public-baseline"
        Base.include(Main, joinpath(FROZEN, "baseline", "DstIntervalObserver.jl"))
        const N = DstIntervalObserver
        @test hashfile(joinpath(FROZEN, "baseline", "DstIntervalObserver.jl")) == BASELINE_PINS["DstIntervalObserver.jl"]
    else
        const N = IMF.DstIntervalAdapter.Kernel
    end
    evaluate_tail(Main, joinpath(FROZEN, "peer", "public_oracle.jl"), "bits(x::Float64)", "using Test,SHA,LinearAlgebra\n")
elseif JOB == "cache514"
    const SOURCE = joinpath(SOURCE_ROOT, "DstIntervalObserver.jl")
    const BASELINE = joinpath(FROZEN, "baseline", "DstIntervalObserver.jl")
    const PRELOAD = Dict(BASELINE => Module[], SOURCE => Module[])
    for (path, count) in ((BASELINE, 1), (SOURCE, 7))
        for i in 1:count
            if path == SOURCE && i == 1
                push!(PRELOAD[path], IMF.DstIntervalAdapter.Kernel)
            else
                preloaded_target = Module(gensym(:FreshCacheOracle))
                Base.include(preloaded_target, path)
                push!(PRELOAD[path], getfield(preloaded_target, :DstIntervalObserver))
            end
        end
    end
    load_fresh(path::String) = popfirst!(PRELOAD[path])
    evaluate_tail(Main, joinpath(FROZEN, "peer", "future_cache_oracle.jl"), "function load_fresh(path)", "using Test,SHA,LinearAlgebra\n")
    @test all(isempty, values(PRELOAD))
elseif JOB in ("stream-baseline", "stream-candidate")
    target = JOB == "stream-baseline" ? baseline_namespace() : IMF
    const DstIntervalAdapter = target.DstIntervalAdapter
    const DstIntervalStream = target.DstIntervalStream
    set_arguments([SOURCE_ROOT, OUT, JOB == "stream-baseline" ? "record" : joinpath(dirname(OUT), "stream-baseline")])
    prefix = "using Dates,SHA,TOML,Test\nusing .DstIntervalAdapter,.DstIntervalStream\n" *
        "const SOURCE=abspath(ARGS[1]);const RECORD=ARGS[3]==\"record\";const REFERENCE=RECORD ? OUT : abspath(ARGS[3])\n" *
        "const BEFORE=Dict(DstIntervalStream.verify_source_identity())\n"
    # BEFORE is already defined by the worker; compare the module source guard
    # using an isolated module to preserve the frozen driver's own names.
    target_worker = Module(gensym(:StreamBits))
    Core.eval(target_worker, :(const DstIntervalAdapter = $DstIntervalAdapter))
    Core.eval(target_worker, :(const DstIntervalStream = $DstIntervalStream))
    Core.eval(target_worker, :(const OUT = $OUT))
    evaluate_tail(target_worker, joinpath(FROZEN, "qualification", "stream_bit_oracle.jl"), "function token(io,x)", prefix)
elseif JOB == "training362"
    const BASELINE = joinpath(FROZEN, "baseline")
    const CANDIDATE = SOURCE_ROOT
    # Its output directory is deliberately absent on entry to the frozen body.
    const TRAIN_OUT = joinpath(OUT, "bundles")
    set_arguments([BASELINE, CANDIDATE, TRAIN_OUT])
    target = Module(gensym(:TrainingBits))
    baseline = baseline_namespace()
    Core.eval(target, :(const MODULES = ($baseline, $IMF)))
    prefix = "using Dates,SHA,Test,TOML\nconst BASELINE=abspath(ARGS[1]);const CANDIDATE=abspath(ARGS[2]);const OUT=abspath(ARGS[3]);mkdir(OUT)\n"
    evaluate_training_provenance(target, joinpath(FROZEN, "qualification", "compare_training_bits.jl"),
        "const B=MODULES[1]", prefix, SOURCE_PINS, BASELINE_PINS,
        keys(Dict(IMF.DstIntervalTraining.verify_source_identity())))
elseif JOB == "warm44"
    set_arguments([SOURCE_ROOT, OUT])
    include(joinpath(FROZEN, "qualification", "warm_source_guard.jl"))
elseif JOB in ("tenk-baseline", "tenk-candidate")
    target = Module(gensym(:TenThousandBits))
    namespace = JOB == "tenk-baseline" ? baseline_namespace() : IMF
    Core.eval(target, :(const DstIntervalAdapter = $(namespace.DstIntervalAdapter)))
    Core.eval(target, :(const DstIntervalStream = $(namespace.DstIntervalStream)))
    set_arguments([SOURCE_ROOT, OUT, joinpath(dirname(OUT), "tenk-baseline")])
    path = joinpath(FROZEN, "qualification", "compare_frozen_10k_v2.jl")
    original = read(path, String)
    start = "using .DstIntervalAdapter,.DstIntervalStream"
    tail = split(original, start; limit=2)[2]
    prefix = "using Dates,SHA,TOML,Test,InteractiveUtils\nconst BASE=abspath(ARGS[1]);const OUT=abspath(ARGS[2])\n" * start
    if JOB == "tenk-baseline"
        # Reuse the frozen token encoder and exact10k fixture, then record only
        # uncached-baseline output. No candidate result can supply expectations.
        setup = split(tail, "@testset"; limit=2)[1]
        Base.include_string(target, prefix * setup, path)
        Base.include_string(target, "for delay in (0,1)\nrows=NamedTuple[]\n" *
            "stream_basis(INPUT;delay_hours=delay,taus=(4.,24.),sigma=3.,chunk_hours=24,sink=x->append!(rows,x))\n" *
            "write(joinpath(OUT,\"unmodified-L\$delay-row-bits.bin\"),bit_payload(rows))\nend\n", path)
    else
        Base.include_string(target, prefix * tail, path)
    end
elseif JOB == "large79"
    numerical = joinpath(@__DIR__, "numerical_regressions")
    numerical_pins = Dict(
        "package_namespace.jl" => "fb44b27edf30a7868265bc01b9565a4ed1b58a14194b0d0c962f8fdc8a9d8ed7",
        "test_large_nnls.jl" => "c79f47320fcb88498b51adc2ec5ac5629ed8e320e156c6ac3541b17e69e10515",
    )
    for (name,digest) in numerical_pins
        @test hashfile(joinpath(numerical,name)) == digest
    end
    observations = joinpath(OUT,"large-nnls-observations.toml")
    set_arguments([joinpath(numerical,"package_namespace.jl"),observations])
    # Only the two-line alias loads a namespace; the original79 oracle bytes
    # are included unmodified in a fresh test namespace over the actual package.
    target = Module(gensym(:PackageBoundLargeNNLS))
    Core.eval(target, :(include(path) = Base.include($target,path)))
    Base.include(target,joinpath(numerical,"test_large_nnls.jl"))
    result = TOML.parsefile(observations)
    @test result["assertions_passed"] === true
    @test result["synthetic_only"] === true
    @test result["actual_archive_read"] === false
    @test length(result["observations"]) == 12
    @test result["source_hashes"] == Dict(IMF.DstIntervalAdapter.verify_source_identity())
    for (name,digest) in numerical_pins
        @test hashfile(joinpath(numerical,name)) == digest
    end
elseif JOB == "stage4"
    target = bind_modules!(Module(gensym(:PackageStageFour)))
    set_arguments([SOURCE_ROOT, OUT, joinpath(FROZEN, "independent", "synthetic_stage4_fallback.tsv")])
    path = joinpath(FROZEN, "independent", "stage4_peer.jl")
    prefix = "using Test,Dates,SHA,TOML\nconst SOURCEDIR=ARGS[1];const OUTDIR=ARGS[2];const FIXTURE=ARGS[3]\n"
    evaluate_tail(target, path, "const A=DstIntervalAdapter", prefix)
else
    error("unknown mandatory functional job: $JOB")
end

@test (precision(BigFloat), rounding(BigFloat)) == CONTEXT_BEFORE
@test Dict(IMF.source_identity()) == BEFORE
verify_bindings()
open(joinpath(OUT, "worker.toml"), "w") do io
    TOML.print(io, Dict("complete" => true, "job" => JOB, "julia_version" => string(VERSION),
        "kernel" => string(Sys.KERNEL), "architecture" => string(Sys.ARCH),
        "package_source" => pathof(SolarSINDy), "source_sha256" => BEFORE,
        "worker_sha256" => hashfile(@__FILE__), "bindings_sha256" => hashfile(joinpath(@__DIR__, "bindings.jl")),
        "blas_threads" => BLAS.get_num_threads(), "julia_threads" => Threads.nthreads(),
        "scope" => "bounded synthetic functional qualification; no empirical skill or deployment claim"), sorted=true)
end
println("PACKAGE_BOUND_WORKER_COMPLETE=", JOB)
