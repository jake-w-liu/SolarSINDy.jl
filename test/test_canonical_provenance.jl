include(joinpath(@__DIR__, "..", "validation", "canonical_provenance.jl"))
include(joinpath(@__DIR__, "..", "validation", "artifact_closure.jl"))

@testset "Canonical provenance fails closed" begin
    @test CANONICAL_OMNI_SHA256 ==
        "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"

    mktempdir() do tmp
        wrong_omni = joinpath(tmp, "omni.csv")
        write(wrong_omni, "not the frozen archive\n")
        @test_throws ErrorException verify_omni_input(wrong_omni; mode=:canonical)
        test_record = verify_omni_input(wrong_omni; mode=:test)
        @test test_record.sha256 == provenance_sha256(wrong_omni)
        @test !test_record.canonical
        @test_throws ArgumentError verify_omni_input(wrong_omni; mode=:unknown)
    end
end

@testset "Manifested CSV writes are fail-closed and restore prior pairs" begin
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        mkpath(joinpath(root, "validation"))
        write(joinpath(root, "Project.toml"), "name = \"Fixture\"\n")
        write(joinpath(root, "Manifest.toml"), "julia_version = \"$(VERSION)\"\n")
        write(joinpath(root, "src", "Fixture.jl"), "module Fixture\nend\n")
        producer = joinpath(root, "validation", "produce.jl")
        write(producer, "# fixture producer\n")
        input = joinpath(root, "input.csv")
        write(input, "x\n1\n")
        output = joinpath(root, "result.csv")

        write_manifested_csv(output, DataFrame(value=[1]);
            producer_script=producer,
            input_paths=(source=input,),
            selection_record=(kind="fixture",),
            deterministic=true,
            package_root=root,
            mode=:test,
        )
        original_output = read(output)
        original_manifest = read(output * ".manifest.json")
        @test_throws ArgumentError write_manifested_csv(
            output, DataFrame(value=[2]);
            producer_script=joinpath(root, "missing-producer.jl"),
            input_paths=(source=input,),
            selection_record=(kind="rollback_fixture",),
            deterministic=true,
            package_root=root,
            mode=:test,
        )
        @test read(output) == original_output
        @test read(output * ".manifest.json") == original_manifest
        @test verify_output_manifest(output; package_root=root) !== nothing

        if !Sys.iswindows()
            output_bytes = read(output)
            output_referent = joinpath(root, "result_referent.csv")
            write(output_referent, output_bytes)
            rm(output)
            symlink(output_referent, output)
            @test_throws ErrorException verify_output_manifest(
                output; package_root=root, verify_source=false,
            )
            rm(output)
            write(output, output_bytes)

            input_bytes = read(input)
            input_referent = joinpath(root, "input_referent.csv")
            write(input_referent, input_bytes)
            rm(input)
            symlink(input_referent, input)
            @test_throws ErrorException verify_output_manifest(
                output; package_root=root, verify_source=false,
            )
            rm(input)
            write(input, input_bytes)

            manifest = output * ".manifest.json"
            manifest_bytes = read(manifest)
            manifest_referent = joinpath(root, "manifest_referent.json")
            write(manifest_referent, manifest_bytes)
            rm(manifest)
            symlink(manifest_referent, manifest)
            @test_throws ErrorException verify_output_manifest(
                output; package_root=root, verify_source=false,
            )
            rm(manifest)
            write(manifest, manifest_bytes)
            @test verify_output_manifest(output; package_root=root) !== nothing
        end

        collision = joinpath(root, "collision.csv")
        mkdir(collision)
        keep = joinpath(collision, "keep")
        write(keep, "preserve")
        @test_throws ArgumentError write_manifested_csv(
            collision, DataFrame(value=[3]);
            producer_script=producer,
            input_paths=(source=input,),
            selection_record=(kind="collision_fixture",),
            deterministic=true,
            package_root=root,
            mode=:test,
        )
        @test isdir(collision)
        @test read(keep, String) == "preserve"

        if !Sys.iswindows()
            referent = joinpath(root, "referent.csv")
            write(referent, "old\n")
            link = joinpath(root, "link.csv")
            symlink(referent, link)
            @test_throws ArgumentError write_manifested_csv(
                link, DataFrame(value=[4]);
                producer_script=producer,
                input_paths=(source=input,),
                selection_record=(kind="symlink_fixture",),
                deterministic=true,
                package_root=root,
                mode=:test,
            )
            @test islink(link)
            @test read(referent, String) == "old\n"
        end
    end
end

@testset "Manifested figures and explicit artifact closure fail closed" begin
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        mkpath(joinpath(root, "validation"))
        write(joinpath(root, "Project.toml"), "name = \"Fixture\"\n")
        write(joinpath(root, "Manifest.toml"), "julia_version = \"$(VERSION)\"\n")
        write(joinpath(root, "src", "Fixture.jl"), "module Fixture\nend\n")
        producer = joinpath(root, "validation", "produce.jl")
        other_producer = joinpath(root, "validation", "other.jl")
        write(producer, "# fixture producer\n")
        write(other_producer, "# wrong producer\n")
        data_dir = joinpath(root, "data")
        figs_dir = joinpath(root, "figs")
        mkpath(data_dir)
        mkpath(figs_dir)
        first_csv = joinpath(data_dir, "first.csv")
        second_csv = joinpath(data_dir, "second.csv")
        write_manifested_csv(first_csv, DataFrame(value=[1]);
            producer_script=producer, input_paths=(;),
            selection_record=(kind="fixture",), deterministic=true,
            package_root=root, mode=:test)
        write_manifested_csv(second_csv, DataFrame(value=[2]);
            producer_script=producer, input_paths=(source=first_csv,),
            selection_record=(kind="fixture",), deterministic=true,
            package_root=root, mode=:test)

        expected_data = Dict("first.csv" => producer, "second.csv" => producer)
        closed = verify_artifact_closure(data_dir;
            expected_artifacts=expected_data, package_root=root)
        @test closed.count == 2
        @test_throws ErrorException verify_artifact_closure(data_dir;
            expected_artifacts=merge(expected_data,
                Dict("missing.csv" => producer)), package_root=root)
        @test_throws ErrorException verify_artifact_closure(data_dir;
            expected_artifacts=Dict("first.csv" => other_producer),
            package_root=root)
        extra_csv = joinpath(data_dir, "diagnostic.csv")
        write_manifested_csv(extra_csv, DataFrame(value=[3]);
            producer_script=producer, input_paths=(source=first_csv,),
            selection_record=(kind="diagnostic",), deterministic=true,
            package_root=root, mode=:test)
        @test_throws ErrorException verify_artifact_closure(data_dir;
            expected_artifacts=expected_data, package_root=root)
        rm(extra_csv)
        rm(extra_csv * ".manifest.json")

        figure = joinpath(figs_dir, "figure.pdf")
        write_manifested_figure(figure, path -> write(path, "%PDF-1.4\nfixture\n");
            producer_script=producer, input_paths=(source=first_csv,),
            selection_record=(kind="fixture_figure",),
            backend="PlotlySupply.jl", deterministic=true,
            package_root=root, mode=:test)
        record = verify_output_manifest(figure; package_root=root)
        @test record["metadata"]["figure_backend"] == "PlotlySupply.jl"
        @test verify_artifact_closure(figs_dir;
            extensions=(".pdf",),
            expected_artifacts=Dict("figure.pdf" => producer),
            package_root=root).count == 1
        prior_figure = read(figure)
        prior_manifest = read(figure * ".manifest.json")
        @test_throws ArgumentError write_manifested_figure(
            figure, path -> write(path, "%PDF-1.4\nreplacement\n");
            producer_script=producer, input_paths=(source=first_csv,),
            selection_record=(kind="bad_backend",), backend="Plots.jl",
            deterministic=true, package_root=root, mode=:test)
        @test read(figure) == prior_figure
        @test read(figure * ".manifest.json") == prior_manifest
        @test_throws ArgumentError write_manifested_figure(
            figure, path -> write(path, "%PDF-1.4\nreplacement\n");
            producer_script=joinpath(root, "missing-producer.jl"),
            input_paths=(source=first_csv,),
            selection_record=(kind="rollback",), backend="PlotlySupply.jl",
            deterministic=true, package_root=root, mode=:test)
        @test read(figure) == prior_figure
        @test read(figure * ".manifest.json") == prior_manifest
        @test_throws ErrorException write_manifested_figure(
            joinpath(figs_dir, "unmanifested.pdf"),
            path -> write(path, "%PDF-1.4\nfixture\n");
            producer_script=producer,
            input_paths=(source=joinpath(root, "missing.csv"),),
            selection_record=(kind="missing_input",), backend="PlotlySupply.jl",
            deterministic=true, package_root=root, mode=:test)

        paths = (root=root, data=data_dir, figs=figs_dir, explicit=true, mode=:test)
        receipt = run_artifact_closure(paths;
            data_inventory=expected_data,
            figure_inventory=Dict("figure.pdf" => producer),
            package_root=root).receipt
        @test isfile(receipt) && isfile(receipt * ".manifest.json")
        receipt_record = verify_output_manifest(receipt; package_root=root)
        @test receipt_record["selection_record"]["required_data_artifacts"] == 2
        @test receipt_record["selection_record"]["required_figure_artifacts"] == 1
        first_manifest = first_csv * ".manifest.json"
        first_manifest_bytes = read(first_manifest)
        write(first_manifest, "not json\n")
        @test_throws ErrorException verify_output_manifest(
            receipt; package_root=root, verify_source=false,
        )
        write(first_manifest, first_manifest_bytes)
        @test verify_output_manifest(receipt; package_root=root) !== nothing
    end
end

@testset "Per-output manifests bind output, inputs, code, and environment" begin
    mktempdir() do root
        mkpath(joinpath(root, "src"))
        mkpath(joinpath(root, "validation"))
        write(joinpath(root, "Project.toml"), "name = \"Fixture\"\n")
        write(joinpath(root, "Manifest.toml"), "julia_version = \"$(VERSION)\"\n")
        write(joinpath(root, "src", "Fixture.jl"), "module Fixture\nend\n")
        producer = joinpath(root, "validation", "produce.jl")
        write(producer, "# fixture producer\n")
        input = joinpath(root, "input.csv")
        output = joinpath(root, "result.csv")
        write(input, "x\n1\n")
        write(output, "y\n2\n")

        manifest = write_output_manifest(output;
            producer_script=producer,
            input_paths=(source=input,),
            selection_record=(lambda=1.0, rule="fixture"),
            seed=42,
            package_root=root,
            mode=:test,
        )
        @test isfile(manifest)
        record = verify_output_manifest(output; package_root=root)
        @test record["randomness"]["kind"] == "seeded"
        @test record["randomness"]["seed"] == 42
        @test endswith(String(record["timestamp_utc"]), "Z")
        @test String(record["source_identity"]["project"]["sha256"]) ==
            provenance_sha256(joinpath(root, "Project.toml"))
        @test String(record["source_identity"]["manifest"]["sha256"]) ==
            provenance_sha256(joinpath(root, "Manifest.toml"))
        @test verified_output_path(output; package_root=root) == abspath(output)
        @test_throws ErrorException verify_output_manifest(
            output; package_root=root, require_canonical=true,
        )

        write(output, "tampered\n")
        @test_throws ErrorException verify_output_manifest(output; package_root=root)
        write(output, "y\n2\n")
        @test verify_output_manifest(output; package_root=root) !== nothing

        write(input, "tampered input\n")
        @test_throws ErrorException verify_output_manifest(output; package_root=root)
        write(input, "x\n1\n")

        write(joinpath(root, "Project.toml"), "name = \"Changed\"\n")
        @test_throws ErrorException verify_output_manifest(output; package_root=root)
        @test verify_output_manifest(output; package_root=root,
                                     verify_source=false) !== nothing

        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=(;), package_root=root, mode=:test,
        )
        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=nothing, deterministic=true,
            package_root=root, mode=:test,
        )
        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=(;), seed=1, deterministic=true,
            package_root=root, mode=:test,
        )
        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=(;), seed=-1, package_root=root, mode=:test,
        )
        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=(;), seed=true, package_root=root, mode=:test,
        )
        @test_throws ArgumentError write_output_manifest(output;
            producer_script=producer, input_paths=(source=input,),
            selection_record=(;), seed=big(typemax(Int)) + 1,
            package_root=root, mode=:test,
        )
        @test_throws ErrorException write_output_manifest(output;
            producer_script=producer, input_paths=(omni=input,),
            selection_record=(;), deterministic=true,
            package_root=root, mode=:canonical,
        )
    end
end

function _fixture_catalog()
    return [
        StormCatalogEntry(1, DateTime(2019, 11, 30), -80.0,
            DateTime(2019, 11, 30, 2), DateTime(2019, 11, 30, 12),
            36.0, 24, "val", 1, 37),
        StormCatalogEntry(2, DateTime(2019, 12, 1), -90.0,
            DateTime(2019, 12, 1, 2), DateTime(2019, 12, 1, 12),
            36.0, 25, "test", 38, 74),
    ]
end

@testset "Catalog manifests reject stale catalogs and parameters" begin
    package_root = normpath(joinpath(@__DIR__, ".."))
    mktempdir() do tmp
        omni = joinpath(tmp, "omni_extracted.csv")
        write(omni, "fixture frozen input\n")
        catalog_path = joinpath(tmp, "storm_catalog.csv")
        parameters = storm_catalog_parameters(year_start=2019, year_end=2019)
        write_verified_storm_catalog(_fixture_catalog(), catalog_path;
            omni_path=omni,
            producer_script=@__FILE__,
            parameters=parameters,
            package_root=package_root,
            mode=:test,
        )
        loaded = load_verified_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        )
        @test length(loaded) == 2
        @test names(CSV.read(catalog_path, DataFrame))[3:4] ==
            ["min_dst_star", "min_dst_star_time"]
        record = verify_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        )
        @test record["metadata"]["catalog"]["n_storms"] == 2
        @test record["metadata"]["catalog"]["split_counts"][3]["count"] == 1
        @test record["metadata"]["catalog"]["cycle_counts"][2]["solar_cycle"] == 25

        prior_catalog = read(catalog_path)
        prior_manifest = read(catalog_path * ".manifest.json")
        replacement = copy(_fixture_catalog())
        replacement[1] = StormCatalogEntry(
            1, DateTime(2019, 11, 30), -123.0,
            DateTime(2019, 11, 30, 2), DateTime(2019, 11, 30, 12),
            36.0, 24, "val", 1, 37,
        )
        @test_throws ArgumentError write_verified_storm_catalog(
            replacement, catalog_path;
            omni_path=omni,
            producer_script=joinpath(tmp, "missing-producer.jl"),
            parameters=parameters,
            package_root=package_root,
            mode=:test,
        )
        @test read(catalog_path) == prior_catalog
        @test read(catalog_path * ".manifest.json") == prior_manifest
        @test verify_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        ) !== nothing
        @test_throws ErrorException verify_storm_catalog(catalog_path;
            omni_path=omni,
            parameters=storm_catalog_parameters(year_start=2019, year_end=2020),
            package_root=package_root, mode=:test, verify_source=false,
        )

        # Even if an attacker refreshes the plain hashes, semantic cycle/split
        # validation still rejects a stale pre-month-boundary assignment.
        frame = CSV.read(catalog_path, DataFrame)
        frame.split[2] = "val"
        CSV.write(catalog_path, frame)
        manifest_path = catalog_path * ".manifest.json"
        forged = JSON3.read(read(manifest_path, String), Dict{String,Any})
        digest = provenance_sha256(catalog_path)
        forged["output"]["sha256"] = digest
        forged["output"]["bytes"] = filesize(catalog_path)
        forged["metadata"]["catalog"]["catalog_sha256"] = digest
        _provenance_atomic_json(manifest_path, forged)
        @test_throws ErrorException verify_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        )
    end
end

@testset "Canonical consumers never fall back to an unmanifested catalog" begin
    mktempdir() do tmp
        catalog_path = joinpath(tmp, "legacy_catalog.csv")
        save_storm_catalog(_fixture_catalog(), catalog_path)
        omni = joinpath(tmp, "omni.csv")
        write(omni, "fixture\n")
        @test length(load_validation_storm_catalog(catalog_path;
            omni_path=omni, mode=:test, verify_source=false)) == 2
        @test_throws ErrorException load_validation_storm_catalog(catalog_path;
            omni_path=omni, mode=:canonical, verify_source=false)
    end
end

@testset "Catalog regeneration is deterministic at month-level boundaries" begin
    package_root = normpath(joinpath(@__DIR__, ".."))
    mktempdir() do tmp
        start = DateTime(2019, 11, 29)
        times = [start + Hour(i - 1) for i in 1:96]
        dst = fill(0.0, length(times))
        dst[50:55] .= -90.0
        omni = joinpath(tmp, "omni_extracted.csv")
        CSV.write(omni, DataFrame(
            year=year.(times), doy=dayofyear.(times), hour=hour.(times),
            By=fill(0.0, length(times)), Bz=fill(-5.0, length(times)),
            T=fill(1.0e5, length(times)), n=fill(5.0, length(times)),
            V=fill(500.0, length(times)), Pdyn=fill(2.0, length(times)),
            Dst=dst, AE=fill(100.0, length(times)),
            AL=fill(-50.0, length(times)), AU=fill(50.0, length(times)),
        ))
        catalog_path = joinpath(tmp, "storm_catalog.csv")
        parameters = storm_catalog_parameters(
            year_start=2019, year_end=2019, dst_thresh=-50.0,
            window_pre=2, window_post=10, min_separation=5,
        )
        first_catalog = regenerate_verified_storm_catalog(omni, catalog_path;
            producer_script=@__FILE__, parameters=parameters,
            package_root=package_root, mode=:test,
        )
        first_hash = provenance_sha256(catalog_path)
        second_catalog = regenerate_verified_storm_catalog(omni, catalog_path;
            producer_script=@__FILE__, parameters=parameters,
            package_root=package_root, mode=:test,
        )
        @test provenance_sha256(catalog_path) == first_hash
        @test first_catalog == second_catalog
        @test only(second_catalog).solar_cycle == 25
        @test only(second_catalog).split == "test"
        @test verify_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        ) !== nothing

        write(omni, "tampered\n")
        @test_throws ErrorException verify_storm_catalog(catalog_path;
            omni_path=omni, parameters=parameters, package_root=package_root,
            mode=:test, verify_source=false,
        )
    end
end
