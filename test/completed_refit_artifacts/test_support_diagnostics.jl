@testset "Support command failures retain raw bytes and preserve the original exit" begin
    driver=CompletedRefitArtifactsExample.CompletedRefitNativeDriver
    mktempdir() do temporary
        root=realpath(temporary);parent=joinpath(root,"raw");mkdir(parent)
        script=joinpath(root,"controlled_failure.py")
        write(script,"import os,sys\nos.write(1,b'a'*70000)\nos.write(2,b'CONTROLLED_SUPPORT_FAILURE\\n')\nsys.exit(7)\n")
        capture=joinpath(root,"diagnostic.log");failure=nothing
        open(capture,"w") do stream
            redirect_stderr(stream) do
                try;driver._run_support_command(Cmd([driver.choose_python(),"-B",script]),parent)
                catch error;failure=error
                end
            end
        end
        @test failure isa ProcessFailedException
        @test length(failure.procs)==1 && only(failure.procs).exitcode==7
        full=read(joinpath(parent,"native.log"));visible=read(capture,String)
        @test length(full)>70000 && occursin("CONTROLLED_SUPPORT_FAILURE",String(copy(full)))
        @test occursin("SUPPORT_TEST_FAILURE_RAW=",visible) && occursin("CONTROLLED_SUPPORT_FAILURE",visible)
        @test ncodeunits(visible)<=65536+ncodeunits(joinpath(parent,"native.log"))+128
    end
end
