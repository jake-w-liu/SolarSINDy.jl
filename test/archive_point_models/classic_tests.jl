module ArchiveClassicPointModelTests
using Test,Statistics,Random,SolarSINDy
const C=SolarSINDy.ArchiveClassicPointModels
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
@testset "Retained baseline literal numerical and ownership oracles" begin
    for p in (8,11)
        X=ones(4,p);X[:,1]=[-3.,-1.,1.,3.];sigma=sqrt(20/3);y=5 .+ 2 .* X[:,1] ./ sigma
        before=copy(X);target=copy(y);r=C.Recipe(kind=:ridge,penalty=1.)
        fitted=C.fit_numeric(X,y,r);model=fitted.model
        @test X==before && y==target
        @test isapprox(collect(model.coefficients),vcat(5.,1.5,zeros(p-1));rtol=0,atol=8e-15)
        @test isapprox(C.predict(model,X),5 .+ 1.5 .* X[:,1] ./ sigma;rtol=0,atol=8e-15)
        @test model.center[1]==0 && isapprox(model.scale[1],sigma;rtol=0,atol=1e-15)
        @test C.predict(C.decode_model(C.encode_model(model)),X)==C.predict(model,X)
        @test isempty(C.predict(model,zeros(0,p)))
        prediction=C.predict(model,X);fitted.native.coefficients[1]+=999.;fitted.cuts[1][1]=999.
        @test C.predict(model,X)==prediction
        X=ones(8,p);X[:,1]=vcat(fill(-1.,4),fill(1.,4));y=vcat(zeros(4),fill(2.,4))
        boost=C.fit_numeric(X,y,C.Recipe(kind=:boost,depth=1,trees=2,bins=2,rate=.5,minleaf=1))
        @test C.predict(boost.model,X)==vcat(fill(.25,4),fill(1.75,4))
        @test boost.model.trees[1].feature==(1,0,0) && boost.model.trees[1].threshold==(0x01,0x00,0x00)
        @test boost.model.trees[1].value==(0.,-1.,1.)
        @test C.predict(C.decode_model(C.encode_model(boost.model)),X)==C.predict(boost.model,X)
        saved=C.predict(boost.model,X);boost.native.trees[1].value[2]=88.
        @test C.predict(boost.model,X)==saved
        # At depth0, each tree is exactly its bootstrap sample mean. No tree
        # splitting or model prediction code enters this independent oracle.
        seed=7102017;count=7;draws=8;rng=MersenneTwister(seed)
        means=[sum(y[i] for i in rand(rng,1:8,draws))/draws for _ in 1:count]
        forest=C.fit_numeric(X,y,C.Recipe(kind=:forest,depth=0,trees=count,bins=2,minleaf=1,sample_rows=draws,seed=seed))
        @test [only(t.value) for t in forest.model.trees]==means
        @test C.predict(forest.model,X)==fill(sum(means)/count,8)
        @test C.predict(C.decode_model(C.encode_model(forest.model)),X)==C.predict(forest.model,X)
        @test forest.model.recipe.settings.seed==seed
    end
end
@testset "Retained recipe and numeric codec boundaries" begin
    for kwargs in ((;kind=:wrong),(;kind=:ridge,depth=1),(;kind=:ridge,penalty=NaN),
                   (;kind=:ridge,penalty=true),(;kind=:boost,depth=5),(;kind=:boost,trees=0),
                   (;kind=:boost,seed=1),(;kind=:forest,depth=10),(;kind=:forest,trees=129),
                   (;kind=:forest,sample_rows=0),(;kind=:forest,seed=true),(;kind=:forest,bins=256))
        @test_throws ArgumentError C.Recipe(;kwargs...)
    end
    @test_throws ArgumentError C.NumericTree([0],[0],[0],[0],[big(10)^400])
    @test_throws ArgumentError C.NumericTree([true],[0],[0],[0],[1.])
    @test_throws ArgumentError C.NumericTree([0],[256],[0],[0],[1.])
    @test_throws ArgumentError C.NumericTree([0],[0],[0],[0],[Inf])
    r=C.Recipe(kind=:boost,depth=1,trees=1,bins=2,minleaf=1)
    cuts=[Float64[0.] for _ in 1:8]
    leaf=C.NumericTree([0],[0],[0],[0],[1.])
    model=C.NumericModel(r,:mixed8,cuts,[leaf];intercept=1.)
    @test C.predict(model,zeros(2,8))==fill(1.05,2)
    @test_throws ArgumentError C.NumericModel(r,:mixed8,cuts,[C.NumericTree([1],[1],[1],[1],[0.])])
    @test_throws ArgumentError C.NumericModel(r,:mixed8,cuts,[C.NumericTree([0,0],[0,0],[0,0],[0,0],[1.,2.])])
    @test_throws ArgumentError C.NumericModel(r,:mixed8,[Float64[NaN] for _ in 1:8],[leaf])
    @test_throws ArgumentError C.NumericModel(r,:mixed8,[BigFloat[big(10)^400] for _ in 1:8],[leaf])
    @test_throws ArgumentError C.predict(model,fill(NaN,1,8))
    @test_throws ArgumentError C.predict(model,zeros(1,11))
    bytes=C.encode_model(model);object=R.decode_canonical(bytes;limit=8_388_608)
    for (field,bad) in (("schema","wrong"),("mode","wrong"),("trees",[]),("intercept","7ff0000000000000"))
        changed=deepcopy(object);changed[field]=bad
        @test_throws ArgumentError C.decode_model(R.canonical_bytes(changed;limit=8_388_608))
    end
    changed=deepcopy(object);changed["foreign"]=false
    @test_throws ArgumentError C.decode_model(R.canonical_bytes(changed;limit=8_388_608))
    changed=deepcopy(object);changed["trees"][1]["left"][1]=true
    @test_throws ArgumentError C.decode_model(R.canonical_bytes(changed;limit=8_388_608))
    @test_throws ArgumentError C.decode_model(vcat(bytes,UInt8('\n')))
end
println("RETAINED_CLASSIC_POINT_ORACLES_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0")
end
