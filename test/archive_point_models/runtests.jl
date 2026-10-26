module ArchivePointNumericPackageTests
using Test,Dates,SHA,SolarSINDy
const SELECTED_POINT_TREE_SOURCE=realpath(joinpath(@__DIR__,"..","..","src","archive_point_models","ArchivePointNumericTrees.jl"))
const P=SolarSINDy.ArchivePointNumericTrees
const C=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const LIB_SOURCE_SHA="f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f"
@assert realpath(String(only(methods(P.predict)).file))==SELECTED_POINT_TREE_SOURCE
println("POINT_TREE_MODULE_LOADED")
println("POINT_TREE_MODULE_SOURCE_SHA256=",bytes2hex(sha256(read(SELECTED_POINT_TREE_SOURCE))))
constant_tree(value,n)=P.NumericTree(zeros(Int,n),zeros(Int,n),vcat(Float32(value),zeros(Float32,n-1)),falses(n))
function threshold_model(mode;edge=.5,left=5.0f0,right=7.0f0)
    p=mode===:ground11 ? 11 : 8
    tree=P.NumericTree([1,0,0],[1,0,0],Float32[0,left,right],Bool[true,false,false])
    P.NumericModel(mode,[[edge] for _ in 1:p],[constant_tree(0,1),tree];depth=2,rounds=1,bins=2)
end
@testset "Literal independent threshold and Float32 oracles" begin
    for mode in (:ground11,:mixed8)
        p=mode===:ground11 ? 11 : 8
        model=threshold_model(mode);X=zeros(6,p);X[:,1]=[-1.,prevfloat(.5),.5,nextfloat(.5),1.,1e300]
        @test P.predict(model,X)==[5.,5.,5.,7.,7.,7.]
        @test P.predict(model,zeros(0,p))==Float64[]
        signed=threshold_model(mode;edge=-0.0);X=zeros(4,p);X[:,1]=[-1.,-0.,+0.,1.]
        @test reinterpret(UInt64,signed.edges[1][1])==0x8000000000000000
        @test P.predict(signed,X)==[5.,5.,7.,7.]
        high=Float32(2^24)
        rounded=P.NumericModel(mode,[[.5] for _ in 1:p],[constant_tree(high,1),constant_tree(1,3),constant_tree(-high,3)];depth=2,rounds=2,bins=2)
        @test P.predict(rounded,zeros(1,p))==[0.]
        @test P.predict(P.decode_model(P.encode_model(model)),zeros(1,p))==[5.]
    end
end
@testset "Numeric ownership,format and invalid geometry" begin
    edges=[[.5] for _ in 1:8];features=[1,0,0];conditions=[1,0,0];leaves=Float32[0,5,7];flags=Bool[true,false,false]
    tree=P.NumericTree(features,conditions,leaves,flags);model=P.NumericModel(:mixed8,edges,[constant_tree(0,1),tree];depth=2,rounds=1,bins=2)
    bytes=P.encode_model(model);baseline=copy(bytes)
    edges[1][1]=99;features[1]=8;conditions[1]=0;leaves[2]=-90;flags[1]=false
    @test P.encode_model(model)==baseline
    restored=P.decode_model(bytes);fill!(bytes,0x00)
    @test P.encode_model(restored)==baseline
    @test P.predict(restored,zeros(1,8))==[5.]
    @test_throws ArgumentError P.NumericTree(true,[0],[0.],[false])
    @test_throws ArgumentError P.NumericTree([true],[0],[0.],[false])
    @test_throws ArgumentError P.NumericTree([0],[0],[NaN],[false])
    @test_throws ArgumentError P.NumericTree([0],[0],[1e-60],[false])
    @test_throws ArgumentError P.NumericModel(:mixed8,[[.5] for _ in 1:8],[constant_tree(0,1)];depth=true,rounds=0,bins=2)
    @test_throws ArgumentError P.NumericModel(:mixed8,[[.5] for _ in 1:8],[constant_tree(0,1)];depth=1,rounds=true,bins=2)
    @test_throws ArgumentError P.NumericModel(:ground11,[[.5] for _ in 1:8],[constant_tree(0,1)];depth=1,rounds=0,bins=2)
    @test_throws ArgumentError P.NumericModel(:mixed8,[[.5,.4] for _ in 1:8],[constant_tree(0,1)];depth=1,rounds=0,bins=3)
    @test_throws ArgumentError P.NumericModel(:mixed8,[[.5] for _ in 1:8],[P.NumericTree([1],[1],[0.],[true])];depth=1,rounds=0,bins=2)
    @test_throws ArgumentError P.predict(model,fill(Inf,1,8))
    @test_throws ArgumentError P.predict(model,zeros(1,11))
    for alter in (
        x->(x["depth"]=true),x->(x["rounds"]=401),x->(x["bins"]=256),x->(x["unknown"]=1),
        x->(x["trees"][1]["prediction_bits"][1]="7fc00000"),
        x->(x["edges"][1][1]="7ff0000000000000"),
        x->(x["trees"][2]["feature"][1]=9),
        x->(x["trees"][2]["condition"][1]=2),
        x->(x["trees"][2]["split"][2]=true))
        value=P.object(model);alter(value)
        @test_throws ArgumentError P.decode_model(C.canonical_bytes(value;limit=P.MAX_BYTES))
    end
    changed=P.object(model);changed["trees"][2]["prediction_bits"][2]=P.bits(6.0f0)
    # A valid changed numeric model is not a provenance violation of this pure
    # codec. Source-bound outer pins and fit checks remain a separate obligation.
    @test P.predict(P.decode_model(C.canonical_bytes(changed;limit=P.MAX_BYTES)),zeros(1,8))==[6.]
end
@testset "Pinned author fit and independent replay on both layouts" begin
    @test SolarSINDy.MatchedGBMPrimitives342._trusted_evo_sha()==LIB_SOURCE_SHA
    for mode in (:ground11,:mixed8)
        p=mode===:ground11 ? 11 : 8;X=zeros(128,p);X[1:64,1].=.25;X[65:128,1].=.75;y=vcat(zeros(64),fill(8.,64))
        native=SolarSINDy.gbm_fit(X,y;max_depth=2,nrounds=1,eta=1.,min_weight=1,seed=27102026,nbins=2,lambda=0.,L2=0.)
        numeric=P.from_native(native;mode,depth=2,rounds=1,bins=2)
        @test P.predict(numeric,X)==y
        @test reinterpret(UInt64,P.predict(numeric,X))==reinterpret(UInt64,SolarSINDy.gbm_predict(native,X))
        @test numeric.edges[1]==(.25,)
        probes=zeros(6,p);probes[:,1]=[-1e300,prevfloat(.25),.25,nextfloat(.25),.75,1e300]
        @test P.predict(numeric,probes)==[0.,0.,0.,8.,8.,8.]
        @test P.predict(numeric,probes)==SolarSINDy.gbm_predict(native,probes)
        encoded=P.encode_model(numeric);loaded=P.decode_model(encoded)
        @test P.encode_model(loaded)==encoded
        @test P.predict(loaded,probes)==[0.,0.,0.,8.,8.,8.]
        constant=SolarSINDy.gbm_fit(X,fill(2.,128);max_depth=2,nrounds=3,eta=.5,min_weight=1,seed=27102026,nbins=2,lambda=0.,L2=0.)
        @test P.predict(P.from_native(constant;mode,depth=2,rounds=3,bins=2),probes)==fill(2.,6)
        @test_throws ArgumentError P.from_native(native;mode,depth=2,rounds=2,bins=2)
    end
end
struct BeyondRows <: AbstractMatrix{Float64} end
Base.size(::BeyondRows)=(600001,11)
Base.getindex(::BeyondRows,indices...)=error("row cap must reject before reading values")
@testset "Joint maximum numeric model and matrix geometry" begin
    split=Bool[index<=31 for index in 1:63]
    features=Int[flag ? 1 : 0 for flag in split]
    conditions=Int[flag ? 127 : 0 for flag in split]
    tree=P.NumericTree(features,conditions,zeros(Float32,63),split)
    model=P.NumericModel(:ground11,[collect(1.:254.) for _ in 1:11],vcat([constant_tree(2,1)],fill(tree,400));depth=6,rounds=400,bins=255)
    bytes=P.encode_model(model);@test length(bytes)<P.MAX_BYTES
    @test P.encode_model(P.decode_model(bytes))==bytes
    @test_throws ArgumentError P.predict(model,BeyondRows())
    X=zeros(600000,11);prediction=P.predict(model,X)
    @test length(prediction)==600000
    @test all(==(2.),prediction)
    @test Sys.maxrss()<=8*1024^3
    println("POINT_TREE_MAXIMUM_ROWS=600000;ROUNDS=400;DEPTH=6;PEAK_RSS_BYTES=",Sys.maxrss())
end
println("GENERATED_POINT_TREE_PRIMITIVE_PASSED;ACTUAL_FITS=0")

end
