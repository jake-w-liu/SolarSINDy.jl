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
