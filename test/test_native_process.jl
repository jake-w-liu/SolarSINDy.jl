using Test

@testset "Native process success, failure and timeout" begin
    names = Sys.iswindows() ? ("python", "python3") : ("python3", "python")
    python = get(ENV, "SOLARSINDY_RESEARCH_PYTHON", something(Sys.which(names[1]), Sys.which(names[2]), ""))
    isfile(python) || error("Python 3.11+ is required for the process cleanup checks")
    script = joinpath(@__DIR__, "test_native_process.py")
    process = run(pipeline(Cmd([python, "-I", "-B", "-X", "utf8", script]); stdout=stdout, stderr=stderr))
    @test success(process)
end
