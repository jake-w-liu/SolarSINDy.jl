include(joinpath(@__DIR__, "..", "validation", "output_paths.jl"))

@testset "Validation output paths are explicit and fail-closed" begin
    package_root = normpath(joinpath(@__DIR__, ".."))
    withenv("SOLARSINDY_OUTPUT_ROOT" => nothing,
            "SOLARSINDY_OMNI_EXTRACTED" => nothing,
            "SOLARSINDY_RUN_MODE" => nothing) do
        paths = validation_output_paths()
        @test paths.root == package_root
        @test paths.data == joinpath(package_root, "data")
        @test paths.figs == joinpath(package_root, "figs")
        @test paths.omni == joinpath(package_root, "data", "omni_extracted.csv")
        @test !paths.explicit
        @test paths.mode == :noncanonical
        @test !paths.canonical
    end

    mktempdir() do root
        override = joinpath(root, "frozen.csv")
        withenv("SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_OMNI_EXTRACTED" => nothing,
                "SOLARSINDY_RUN_MODE" => nothing) do
            paths = validation_output_paths()
            @test paths.root == abspath(root)
            @test paths.data == joinpath(abspath(root), "data")
            @test paths.figs == joinpath(abspath(root), "figs")
            @test paths.omni == joinpath(abspath(root), "data", "source",
                                         "omni_extracted.csv")
            @test paths.explicit
            @test paths.mode == :canonical
            @test paths.canonical
            @test isdir(paths.data) && isdir(paths.figs)
        end
        withenv("SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_OMNI_EXTRACTED" => override,
                "SOLARSINDY_RUN_MODE" => nothing) do
            @test_throws ArgumentError validation_output_paths()
        end
        withenv("SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_OMNI_EXTRACTED" => override,
                "SOLARSINDY_RUN_MODE" => "test") do
            @test validation_output_paths().omni == abspath(override)
            @test validation_output_paths().mode == :test
        end
        withenv("SOLARSINDY_RUN_MODE" => "invalid") do
            @test_throws ArgumentError validation_output_paths()
        end
    end
end

@testset "Operational evidence path prefers preserved paper evidence" begin
    probe = Module(:OperationalPathProbe)
    withenv("SOLARSINDY_OPERATIONAL_OUTPUT_DIR" => nothing,
            "SOLARSINDY_OPERATIONAL_EVIDENCE_DIR" => nothing) do
        Base.include(probe, joinpath(@__DIR__, "..", "validation", "operational", "paths.jl"))
    end

    package_root = normpath(joinpath(@__DIR__, ".."))
    paper_evidence = normpath(joinpath(package_root, "..", "paper_v2_monitor",
                                       "data", "source", "operational"))
    package_evidence = joinpath(package_root, "data", "operational_validation")
    expected = isdir(paper_evidence) ? abspath(paper_evidence) :
               isdir(package_evidence) ? abspath(package_evidence) :
               abspath(joinpath(package_root, "validation", "output", "operational"))
    @test probe.OPERATIONAL_EVIDENCE_DIR == expected

    empty_override_probe = Module(:OperationalEmptyOutputOverrideProbe)
    withenv("SOLARSINDY_OPERATIONAL_OUTPUT_DIR" => "",
            "SOLARSINDY_OPERATIONAL_EVIDENCE_DIR" => nothing) do
        Base.include(empty_override_probe,
                     joinpath(@__DIR__, "..", "validation", "operational", "paths.jl"))
    end
    @test empty_override_probe.OPERATIONAL_OUTPUT_DIR ==
          joinpath(package_root, "validation", "output", "operational")

    mktempdir() do root
        first_dir = joinpath(root, "first")
        second_dir = joinpath(root, "second")
        mkpath(first_dir)
        mkpath(second_dir)
        withenv("SOLARSINDY_TEST_EVIDENCE" => nothing) do
            @test probe._operational_path(
                "SOLARSINDY_TEST_EVIDENCE", joinpath(root, "fallback"),
                first_dir, second_dir; directory = true,
            ) == abspath(first_dir)
        end
        withenv("SOLARSINDY_TEST_EVIDENCE" => second_dir) do
            @test probe._operational_path(
                "SOLARSINDY_TEST_EVIDENCE", joinpath(root, "fallback"),
                first_dir; directory = true,
            ) == abspath(second_dir)
        end

        output_dir = joinpath(root, "output")
        paper_dir = joinpath(root, "paper")
        package_dir = joinpath(root, "package")
        mkpath.((output_dir, paper_dir, package_dir))
        artifact = "storm_replay_report.md"
        write(joinpath(package_dir, artifact), "package")
        write(joinpath(paper_dir, artifact), "paper")
        withenv("SOLARSINDY_OPERATIONAL_EVIDENCE_DIR" => nothing) do
            @test probe.operational_evidence_dir(
                artifact; output_dir=output_dir, paper_dir=paper_dir, package_dir=package_dir,
            ) == abspath(paper_dir)
            write(joinpath(output_dir, artifact), "generated")
            @test probe.operational_evidence_dir(
                artifact; output_dir=output_dir, paper_dir=paper_dir, package_dir=package_dir,
            ) == abspath(output_dir)
            @test probe.operational_evidence_dir(
                "missing.csv"; output_dir=output_dir, paper_dir=paper_dir,
                package_dir=package_dir,
            ) == abspath(output_dir)
            @test_throws ArgumentError probe.operational_evidence_dir(
                ; output_dir=output_dir, paper_dir=paper_dir, package_dir=package_dir,
            )
        end
        explicit = joinpath(root, "explicit-but-incomplete")
        withenv("SOLARSINDY_OPERATIONAL_EVIDENCE_DIR" => explicit) do
            @test probe.operational_evidence_dir(
                artifact; output_dir=output_dir, paper_dir=paper_dir, package_dir=package_dir,
            ) == abspath(explicit)
        end
    end
end
