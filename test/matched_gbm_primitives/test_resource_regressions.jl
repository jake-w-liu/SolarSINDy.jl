module Matched342ResourceRegressionTests
using Test,Dates,JSON3,SHA,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
const R=M.MatchedScalarTreeReplay342
function _input(;missing=false)
    path=joinpath(dirname(pathof(SolarSINDy)),"..","test","matched_gbm_primitives","fixtures","generated_fit_oracles_v1.json")
    fixture=JSON3.read(read(path,String),Any);values=Matrix{Float64}(undef,17,1024)
    for i in 1:1024,j in 1:17;values[j,i]=fixture["raw_rows_f64"][i][j];end
    missing&&fill!(values,NaN)
    M.HourlyInputs342([DateTime(2000)+Hour(i-1) for i in 1:1024],values;delay_hours=0,source_sha256=fixture["source_sha256"])
end
function _max_model()
    bias=(;feature=[0],condition=[0],prediction=Float32[0],split=[false]);trees=NamedTuple[bias]
    for tree in 1:400
        features=zeros(Int,63);conditions=zeros(Int,63);predictions=zeros(Float32,63);splits=falses(63)
        features[1]=(tree-1)%342+1;conditions[1]=(tree-1)%254+1;splits[1]=true
        sign=isodd(tree) ? 1.0f0 : -1.0f0
        predictions[2]=sign*Float32(tree%7+1)/4.0f0;predictions[3]=sign*Float32(tree%11+3)/4.0f0
        push!(trees,(;feature=features,condition=conditions,prediction=predictions,split=collect(splits)))
    end
    R.Model([collect(1.0:254.0) for _ in 1:342],trees;max_depth=6,nrounds=400)
end
_feature_allocated(input,issues)=@allocated M.build_features(input,issues)
_replay_allocated(model,X)=@allocated R.predict_deltas(model,X)
function run_tests(resource=JSON3.read(read(joinpath(@__DIR__,"fixtures/resource_oracles_v1.json"),String),Any))
    @testset "Matched342 direct public collector and replay resource regressions" begin
        @test resource["schema"]=="matched342_resource_independent_oracles_v1"
        input=_input();issues=[t+Hour(1) for t in input.time]
        @test_throws ArgumentError M.build_features(input,DateTime[])
        single=M.build_features(input,issues[100:100]);repeated=M.build_features(input,[issues[100],issues[100]])
        @test size(single.matrix)==(1,342)
        @test size(repeated.matrix)==(2,342)
        @test reinterpret(UInt64,vec(repeated.matrix[1:1,:]))==reinterpret(UInt64,vec(single.matrix))
        @test reinterpret(UInt64,vec(repeated.matrix[2:2,:]))==reinterpret(UInt64,vec(single.matrix))
        allmissing=M.build_features(_input(;missing=true),[issues[100],issues[100]])
        for rows in ([1],[1,2])
            encoder=M.fit_encoder(allmissing.matrix,rows)
            @test collect(encoder.finite_counts)==zeros(Int,165)
            @test reinterpret(UInt64,collect(encoder.imputation))==zeros(UInt64,165)
            @test collect(encoder.scale)==ones(342)
            @test all(isfinite,M.transform(encoder,allmissing.matrix))
        end
        for _ in 1:3;M.build_features(input,issues);end
        _feature_allocated(input,issues)
        feature_allocated=_feature_allocated(input,issues)
        model=_max_model();edges=model.edges;trees=model.trees
        X=Float64[i==1 ? 0.5 : i==2 ? nextfloat(0.5) : i==3 ? floatmax(Float64) : i==4 ? -0.0 : ((17i+31j)%511)/2 for i in 1:128,j in 1:342]
        predicted=R.predict_deltas(model,X)
        for i in 1:128
            @test string(reinterpret(UInt64,predicted[i]);base=16,pad=16)==resource["max_geometry_prediction_f64_bits"][i]
        end
        empty_result=R.predict_deltas(model,zeros(0,342))
        @test empty_result isa Vector{Float64}
        @test isempty(empty_result)
        for _ in 1:3;R.predict_deltas(model,X);end
        _replay_allocated(model,X);replay_allocated=_replay_allocated(model,X)
        @test model.edges===edges&&model.trees===trees
        println("MATCHED342_PUBLIC_RESOURCE_BYTES ",JSON3.write((;feature_rows1024=feature_allocated,max_geometry_replay_rows128=replay_allocated,
            feature_limit=128*1024^2,replay_limit=8*1024^2,latency_threshold=false)));flush(stdout)
        @test feature_allocated<=128*1024^2
        @test replay_allocated<=8*1024^2
    end
end
end
