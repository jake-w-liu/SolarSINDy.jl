module HashContextTests
using Test, SHA
const PROJECT=dirname(@__DIR__)
const WORKER=joinpath(@__DIR__,"hash_context","worker.jl")

@testset "Research identifiers and saved models are independent of Main imports" begin
    root=realpath(mktempdir(;prefix="solarsindy-hash-context-",cleanup=false))
    passed=false
    function worker(mode,args...)
        command=`$(Base.julia_cmd()) --startup-file=no --pkgimages=no --threads=1 --depwarn=error --project=$PROJECT $WORKER $mode $root $args`
        process=run(ignorestatus(command))
        @test process.exitcode==0
        @test process.termsignal==0
        process.exitcode==0 && process.termsignal==0 || error("retained hash-context failure evidence: "*root)
    end
    try
        worker("oracles")
        worker("produce")
        digest=bytes2hex(sha256(read(joinpath(root,"cases.toml"))))
        for mode in ("load","load-shadow","load-using-dates")
            worker(mode,digest)
        end
        @test bytes2hex(sha256(read(joinpath(root,"cases.toml"))))==digest
        passed=true
    finally
        if passed
            @test realpath(dirname(root))==realpath(tempdir()) && startswith(basename(root),"solarsindy-hash-context-")
            @test !islink(root)
            for (directory,dirs,files) in walkdir(root;follow_symlinks=false),name in vcat(dirs,files)
                @test !islink(joinpath(directory,name))
            end
            rm(root;recursive=true)
        else
            println(stderr,"FAILED_HASH_CONTEXT_EVIDENCE_ROOT=",root)
        end
    end
end
end
