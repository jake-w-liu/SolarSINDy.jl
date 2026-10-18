using Test, Dates, SHA, SolarSINDy
const DF = SolarSINDy.IntervalMeanForecaster
const DT = DF.DstIntervalTraining
const DISK_OUTPUT = abspath(ARGS[2])
isdir(DISK_OUTPUT) || error("caller must create owned evidence directory")
const DISK_SOURCE_BEFORE = DF.source_identity()

function frozen_disk_oracle(name, destination, expected)
    path = joinpath(@__DIR__, name)
    @test bytes2hex(sha256(read(path))) == expected
    target = Module(gensym(:FrozenDiskOracle))
    Core.eval(target, :(include(path) = Base.include($target, path)))
    original = copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS, [joinpath(@__DIR__, "bind_package.jl"), destination])
        Base.include(target, path)
    finally
        empty!(ARGS)
        append!(ARGS, original)
    end
    target
end

const DISK_ORACLE = frozen_disk_oracle("test_disk_affine_fit.jl",
    joinpath(DISK_OUTPUT, "frozen-390"),
    "cbfbbb49ff72c2580a777bed2bf4ca85b9bcd27c4773b91f78f8196f09eb2c4e")
frozen_disk_oracle("shape_admission_witness.jl",
    joinpath(DISK_OUTPUT, "frozen-shape-6"),
    "36d9d430644541bcfe6e7ce7ef2af254c62374e4e717058a4b6a45ebb3fa45ad")

@testset "canonical disk-fit model and prediction integration" begin
    @test DF.export_fit_input === DT.export_fit_input
    @test DF.fit_gains_from_input === DT.fit_gains_from_input
    for f in (DF.export_fit_input, DF.fit_gains_from_input)
        @test all(m -> realpath(String(m.file)) ==
            realpath(joinpath(@__DIR__, "..", "..", "..", "src",
                "interval_mean_forecaster", "DstIntervalTraining.jl")), methods(f))
    end
    field_names = (:schema, :evidence_mode, :contract_sha256, :gains, :setting_index,
        :taus, :sigma, :eta, :delay_hours, :boundary, :source_sha256, :source_hashes,
        :fit_count, :informative_count, :storm_count, :fit_data_sha256, :fit_mask_sha256,
        :max_target_complete, :max_label_receipt, :max_fit_issue, :solver, :model_id)
    for delay in (0, 1), index in (1, 2), chunk in (1, 7, 24, 4096)
        setting = DF.settings()[index]
        rows = DISK_ORACLE.fixture(setting, delay; missing=true, maturity=true)
        input_before = bytes2hex(sha256(repr(rows)))
        expected = DF.fit_gains(rows, DISK_ORACLE.BOUNDARY, setting;
            delay_hours=delay, source_sha256=DISK_ORACLE.SOURCE)
        input = DISK_ORACLE.export_rows(rows, setting, delay; chunk)
        model = DF.fit_gains_from_input(input.directory;
            expected_manifest_sha256=input.manifest_sha256)
        @test keys(model) == field_names
        @test isequal(model, expected)
        @test model.schema == "candidate_a_intervalmean_research_v1"
        @test model.evidence_mode == "synthetic_or_unqualified_research"
        @test all(isapprox.(model.gains, (2., 3.); atol=1e-12, rtol=0))
        @test model.fit_count == 191 && model.informative_count == 181
        @test model.max_label_receipt == DISK_ORACLE.BOUNDARY-Hour(168)
        @test isequal(DF.predict_rows(model, rows), DF.predict_rows(expected, rows))
        @test bytes2hex(sha256(repr(rows))) == input_before
        @test_throws ArgumentError DF.fit_gains_from_input(input.directory;
            expected_manifest_sha256=repeat("0", 64))
        @test_throws ArgumentError DF.fit_gains_from_input(input.directory;
            expected_manifest_sha256=input.manifest_sha256*"\n")
    end
end

@testset "stream, canonical model, prediction and bundle path" begin
    for delay in (0, 1)
        origin = DateTime(2018)
        values = ones(17, 900)
        values[8, :] .= 400.
        values[6, :] .= -5.
        values[12, :] .= 2.
        values[15, :] .= -30.
        digest = bytes2hex(sha256("canonical disk-fit generated stream fixture"))
        input = DF.HourlyInputs([origin+Hour(i-1) for i in 1:900], values;
            source_sha256=digest)
        rows = NamedTuple[]
        streamed = DF.stream_basis(input; delay_hours=delay, taus=(4.,24.), sigma=3.,
            chunk_hours=7, sink=part -> append!(rows, part))
        setting = DF.settings()[9]
        exported = DF.export_fit_input(joinpath(DISK_OUTPUT, "fit-L$delay"),
            origin+Hour(700), setting; delay_hours=delay, source_sha256=digest) do sink
            for i in 1:13:length(rows)
                sink(rows[i:min(i+12, end)])
            end
        end
        fitted = DF.fit_gains_from_input(exported.directory;
            expected_manifest_sha256=exported.manifest_sha256)
        memory = DF.fit_gains(rows, origin+Hour(700), setting;
            delay_hours=delay, source_sha256=digest)
        @test isequal(fitted, memory)
        saved = DF.save_bundle(joinpath(DISK_OUTPUT, "bundle-L$delay"), fitted,
            streamed.checkpoint; provenance=(synthetic_fixture=true,
                input_sha256=digest, fit_input_manifest_sha256=exported.manifest_sha256))
        loaded = DF.load_bundle(saved.directory;
            expected_manifest_sha256=saved.manifest_sha256,
            expected_sources=DF.DstIntervalArtifacts.verify_source_identity())
        @test isequal(loaded.model, fitted)
        @test isequal(loaded.checkpoint, streamed.checkpoint)
        @test loaded.provenance.fit_input_manifest_sha256 == exported.manifest_sha256
        @test isequal(DF.predict_rows(loaded.model, rows), DF.predict_rows(memory, rows))
    end
end

@testset "ordinary pinned fit file and source-drift guards" begin
    setting = DF.settings()[1]
    input = DISK_ORACLE.export_rows(DISK_ORACLE.fixture(setting, 0), setting, 0)
    data = joinpath(input.directory, "joint-fit.f64")
    preserved = data*".preserved"
    before = bytes2hex(sha256(read(data)))
    mv(data, preserved)
    mkdir(data)
    @test_throws ArgumentError DF.fit_gains_from_input(input.directory;
        expected_manifest_sha256=input.manifest_sha256)
    @test bytes2hex(sha256(read(preserved))) == before
    if Sys.isunix()
        rm(data)
        symlink(preserved, data)
        @test_throws ArgumentError DF.fit_gains_from_input(input.directory;
            expected_manifest_sha256=input.manifest_sha256)
        @test islink(data) && bytes2hex(sha256(read(preserved))) == before
    end
    copy_root = joinpath(DISK_OUTPUT, "owned-source-drift-copy")
    mkdir(copy_root)
    source_root = joinpath(@__DIR__, "..", "..", "..", "src", "interval_mean_forecaster")
    for name in (first.(DISK_SOURCE_BEFORE)..., "IntervalMeanForecaster.jl")
        cp(joinpath(source_root, name), joinpath(copy_root, name))
    end
    target = Module(gensym(:OwnedDiskSourceDrift))
    Base.include(target, joinpath(copy_root, "IntervalMeanForecaster.jl"))
    copied = getfield(target, :IntervalMeanForecaster)
    open(io -> println(io, "# injected drift in owned copy"),
        joinpath(copy_root, "DstIntervalTraining.jl"), "a")
    destination = joinpath(DISK_OUTPUT, "must-not-exist-after-source-drift")
    @test_throws ArgumentError copied.export_fit_input(sink -> nothing,
        destination, DateTime(2019), copied.settings()[1];
        delay_hours=0, source_sha256=repeat("a",64))
    @test !ispath(destination)
end
@test DF.source_identity() == DISK_SOURCE_BEFORE

const DISK_METADATA = let
    path = joinpath(@__DIR__, "model_metadata_oracle_frozen_v1.jl")
    @test bytes2hex(sha256(read(path))) ==
        "cf47443235df968be97ae6119edabbc1c53584deae03e19870c0f20abb0f0403"
    target = Module(gensym(:IndependentCanonicalModelMetadata))
    Core.eval(target, :(include(path) = Base.include($target,path)))
    Core.eval(target, :(const DISK_FIT_NAMESPACE = $DF))
    original = copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS,["unused",joinpath(DISK_OUTPUT,"canonical-metadata")])
        Base.include(target,path)
    finally
        empty!(ARGS);append!(ARGS,original)
    end
    target
end
include("canonical_boundary_regressions.jl")
@test DF.source_identity() == DISK_SOURCE_BEFORE

const DISK_FIT_NAMESPACE=DF
let path=joinpath(@__DIR__,"recipe_metadata_regressions_frozen_v1.jl")
    @test bytes2hex(sha256(read(path)))=="f3ab8e8416c5b107a68579d51c905f7e5c2d460b0422c62d8169fcbb9cbd26f9"
    include(path)
end
@test DF.source_identity()==DISK_SOURCE_BEFORE
