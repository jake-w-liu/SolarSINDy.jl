using Test, Dates, SHA, LinearAlgebra
const source_root=@__DIR__
const private_root=mktempdir("C:/Temp";prefix="candidate-a-loaded-identity-")
const files=("DstIntervalObserver.jl","DstIntervalReplay.jl","sandu_fullchannel_features.jl",
    "DstIntervalAdapter.jl","DstIntervalStream.jl","DstIntervalTraining.jl")
for name in files
    cp(joinpath(source_root,name),joinpath(private_root,name);force=false)
end
const separate=Module(gensym(:PrivateAdapterIdentity))
Base.include(separate,joinpath(private_root,"DstIntervalAdapter.jl"))
Base.include(separate,joinpath(private_root,"DstIntervalStream.jl"))
Base.include(separate,joinpath(private_root,"DstIntervalTraining.jl"))
const A=getfield(separate,:DstIntervalAdapter)
const S=getfield(separate,:DstIntervalStream)
const T=getfield(separate,:DstIntervalTraining)
@testset "loadedcode cannot be rebound to later diskbytes" begin
    expected=T.verify_source_identity()
    @test length(expected)==6
    for name in files
        path=joinpath(private_root,name)
        original=read(path)
        try
            write(path,vcat(original,codeunits("\n# synthetic post-load mutation\n")))
            @test_throws ArgumentError T.verify_source_identity()
            if name!="DstIntervalTraining.jl"
                @test_throws ArgumentError S.verify_source_identity()
            end
            if !(name in ("DstIntervalTraining.jl","DstIntervalStream.jl"))
                @test_throws ArgumentError A.nnls2(Matrix{Float64}(I,2,2),[1.,2.])
            end
            @test_throws ArgumentError T.fit_gains(NamedTuple[],DateTime(2020),A.settings()[1];delay_hours=0,source_sha256=repeat("0",64))
        finally
            write(path,original)
        end
        @test T.verify_source_identity()==expected
    end
    values=ones(17,3);values[8,:].=400.;values[6,:].=-5.;values[12,:].=2.;values[15,:].=-20.
    input=S.HourlyInputs([DateTime(2018)+Hour(i-1) for i in 1:3],values;source_sha256=bytes2hex(sha256("identityfixture")))
    stream_path=joinpath(private_root,"DstIntervalStream.jl")
    original=read(stream_path)
    calls=Ref(0)
    try
        sink=rows->begin
            calls[]+=1
            calls[]==1 && write(stream_path,vcat(original,codeunits("\n# mutate during synthetic execution\n")))
        end
        @test_throws ArgumentError S.stream_basis(input;delay_hours=0,taus=(2.,12.),sigma=1.,chunk_hours=1,sink)
        @test calls[]>0
    finally
        write(stream_path,original)
    end
    @test T.verify_source_identity()==expected
    @test_throws ArgumentError T.predict_rows((source_hashes=(),),NamedTuple[])
    for name in files
        @test bytes2hex(sha256(read(joinpath(private_root,name))))==bytes2hex(sha256(read(joinpath(source_root,name))))
    end
end
println("PRIVATE_FIXTURE=",private_root)
for name in files
    println(name,"=",bytes2hex(sha256(read(joinpath(source_root,name)))))
end
