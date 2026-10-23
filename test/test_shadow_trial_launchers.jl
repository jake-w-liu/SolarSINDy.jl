module TestShadowTrialLaunchers
using Test,JSON3
include(joinpath(@__DIR__,"bash_runtime.jl"))

@testset "Launchd trial pins reach only the two consumers" begin
    bash=solarsindy_test_bash()
    python=nothing
    for candidate in unique(filter(!isnothing,[Sys.which("python3"),Sys.which("python")]))
        try
            if strip(_solarsindy_bash_output(`$candidate -c "import plistlib; print('PLISTLIB_OK')"`))=="PLISTLIB_OK"
                python=candidate
                break
            end
        catch
            # A Windows app alias is not a working Python interpreter.
        end
    end
    python===nothing && error("Python stdlib plistlib is required for independent launchd XML parsing")
    mktempdir() do directory
        output=joinpath(directory,"actual-render-output")
        script=joinpath(@__DIR__,"shadow_trial_launcher_render.py")
        repo=normpath(joinpath(@__DIR__,".."))
        command=`$python $script --repo $repo --bash $bash --output $output`
        process=run(pipeline(ignorestatus(command);stdout=devnull,stderr=devnull))
        record=JSON3.read(read(joinpath(output,"receipt.json")),Dict{String,Any})
        @test success(process)
        @test record["complete"]
        @test get(record,"error",nothing)===nothing
        @test length(record["checks"])==123
        for check in record["checks"]
            @testset "$(check["name"])" begin
                @test check["passed"]
            end
        end
        @test !record["installer_executed"]
        @test !record["launchctl_executed"]
        @test !record["home_modified"]
    end
end
end
