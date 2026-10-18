using Dates,SHA,Test
const SOURCE=abspath(ARGS[1]);const OUT=abspath(ARGS[2]);mkpath(OUT)
const FILES=("DstIntervalObserver.jl","DstIntervalReplay.jl","sandu_fullchannel_features.jl",
    "DstIntervalAdapter.jl","DstIntervalStream.jl","DstIntervalTraining.jl","DstIntervalArtifacts.jl","DstIntervalExperiment.jl")
const PRIVATE=mktempdir(OUT;prefix="loaded-warm-source-",cleanup=false)
for name in FILES;cp(joinpath(SOURCE,name),joinpath(PRIVATE,name);force=false);end
const SEPARATE=Module(gensym(:WarmSourceIdentity))
for name in ("DstIntervalAdapter.jl","DstIntervalStream.jl","DstIntervalTraining.jl","DstIntervalArtifacts.jl","DstIntervalExperiment.jl")
    Base.include(SEPARATE,joinpath(PRIVATE,name))
end
const A=getfield(SEPARATE,:DstIntervalAdapter);const S=getfield(SEPARATE,:DstIntervalStream)
const E=getfield(SEPARATE,:DstIntervalExperiment);const K=A.Kernel
const VALUES=ones(17,300)
VALUES[8,:].=400.;VALUES[6,:].=-5.;VALUES[12,:].=2.;VALUES[15,:].=-20.
const INPUT=S.HourlyInputs([DateTime(2018)+Hour(i-1) for i in 1:300],VALUES;
    source_sha256=bytes2hex(sha256("cache-warm-identity-fixture-v1")))
@testset "warm-cache loaded source mismatch rejects before stream work" begin
    S.stream_basis(INPUT;delay_hours=0,taus=(4.,24.),sigma=3.)
    @test length(K._KERNEL_CACHE)==2
    expected=E.verify_source_identity()
    @test Dict(expected)==Dict(name=>bytes2hex(sha256(read(joinpath(PRIVATE,name)))) for name in FILES)
    cached=[(key,Tuple(repr(v) for v in value)) for (key,value) in K._KERNEL_CACHE]
    for name in FILES
        path=joinpath(PRIVATE,name);original=read(path);calls=Ref(0)
        try
            write(path,vcat(original,codeunits("\n# disposable post-load identity probe\n")))
            @test_throws ArgumentError E.verify_source_identity()
            if !(name in ("DstIntervalTraining.jl","DstIntervalArtifacts.jl","DstIntervalExperiment.jl"))
                @test_throws ArgumentError S.stream_basis(INPUT;delay_hours=0,taus=(4.,24.),sigma=3.,sink=rows->(calls[]+=1))
                @test calls[]==0
            end
            @test [(key,Tuple(repr(v) for v in value)) for (key,value) in K._KERNEL_CACHE]==cached
        finally
            write(path,original)
        end
        @test E.verify_source_identity()==expected
    end
    for name in FILES
        @test read(joinpath(PRIVATE,name))==read(joinpath(SOURCE,name))
    end
end
println("WARM_SOURCE_GUARD_COMPLETE retained=",PRIVATE)
