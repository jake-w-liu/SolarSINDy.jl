module Matched342TreeReplayTests
using Test,JSON3,SHA,SolarSINDy
const R=SolarSINDy.MatchedGBMPrimitives342.MatchedScalarTreeReplay342
const FIX=joinpath(@__DIR__,"fixtures","tree_literals_v1.json")
f64(s)=reinterpret(Float64,parse(UInt64,s;base=16))
f32(s)=reinterpret(Float32,parse(UInt32,s;base=16))
bits(x)=string(reinterpret(UInt64,x);base=16,pad=16)
tree(f,c,p,s)=(;feature=f,condition=c,prediction=p,split=s)
constant_tree(x,n=3)=tree(zeros(Int,n),zeros(Int,n),vcat([x],zeros(n-1)),falses(n))
function hand_model(index;edges=[fill(.5,1) for _ in 1:342],bias=2.0^24,left=1.,right=3.,tail=-2.0^24)
    split=tree([index,0,0],[1,0,0],[0.,left,right],[true,false,false])
    R.Model(edges,[constant_tree(bias,1),split,constant_tree(tail)];max_depth=2,nrounds=2)
end
function run_tests()
    oracle=JSON3.read(read(FIX,String),Any)
    @testset "Independent342 scalar tree full index literals" begin
        for row in oracle["index_cases"]
            model=hand_model(Int(row["feature_index"]));values=zeros(342);values[row["nonzero_input_coordinate"]]=f64(row["input_f64_bits"])
            @test bits(R.predict_delta(model,values))==row["expected_delta_f64_bits"]
        end
        model=hand_model(342);X=zeros(4,342);X[:,342].=[prevfloat(.5),.5,nextfloat(.5),floatmax(Float64)]
        @test R.predict_deltas(model,X)==[0.,0.,4.,4.]
        @test_throws ArgumentError R.predict_delta(model,zeros(341))
        @test_throws MethodError R.predict_delta(model,zeros(Float32,342))
        for x in (NaN,Inf,-Inf)
            bad=zeros(342);bad[342]=x;@test_throws ArgumentError R.predict_delta(model,bad)
        end
        source=read(joinpath(@__DIR__,"..","..","tools","initial_reference_audit","IndependentScalarTreeReplay.jl"),String)
        @test occursin("length(raw_edges) == 29",source)
        @test occursin("length(values) == 29",source)
    end
    @testset "Independent342 bin boundaries and total order" begin
        edge=f64.(oracle["edges_254_f64_bits"]);model=hand_model(1;edges=[copy(edge) for _ in 1:342])
        @test model.nbins==255
        for row in oracle["edge_boundary_cases"]
            @test R._bin(model.edges[1],f64(row["input_f64_bits"]))==row["expected_bin"]
        end
        highsplit=tree([342,0,0],[254,0,0],[0.,1.,3.],[true,false,false])
        high=R.Model([copy(edge) for _ in 1:342],[constant_tree(2.0^24,1),highsplit,constant_tree(-2.0^24)];max_depth=2,nrounds=2)
        values=zeros(342);values[342]=edge[end];@test R.predict_delta(high,values)==0.
        values[342]=nextfloat(edge[end]);@test R.predict_delta(high,values)==4.
        for key in ("duplicate_edges","signed_zero_edges")
            data=oracle[key];edges=f64.(data["edges_f64_bits"]);m=hand_model(1;edges=[copy(edges) for _ in 1:342])
            for row in data["cases"]
                @test R._bin(m.edges[1],f64(row["input_f64_bits"]))==row["expected_bin"]
            end
        end
    end
    @testset "Independent342 Float32 order and model bounds" begin
        for row in oracle["ordered_float32_cases"]
            haskey(row,"leaf_f32_bits")||continue
            leaves=f32.(row["leaf_f32_bits"]);rounds=length(leaves)-1
            trees=[constant_tree(Float64(leaves[1]),1)];append!(trees,[constant_tree(Float64(x)) for x in leaves[2:end]])
            model=R.Model([Float64[.5] for _ in 1:342],trees;max_depth=2,nrounds=rounds)
            if get(row,"expected","")=="reject at second addition; later cancellation does not rescue"
                @test_throws ArgumentError R.predict_delta(model,zeros(342))
            else
                result=R.predict_delta(model,zeros(342));expected=get(row,"expected_delta_f64_bits",nothing)
                if expected===nothing
                    expected=bits(Float64(f32(get(row,"expected_f32_bits",get(row,"mutant_result_f32_bits",nothing)))))
                end
                @test bits(result)==expected
            end
        end
        base=[Float64[.5] for _ in 1:342];bias=constant_tree(0.,1)
        for bad in (base[1:29],base[1:341],vcat(base,[Float64[.5]]))
            @test_throws ArgumentError R.Model(bad,[bias];max_depth=1,nrounds=0)
        end
        @test R.predict_delta(R.Model(base,[bias];max_depth=1,nrounds=0),zeros(342))==0.
        for depth in (0,7,true)
            @test_throws ArgumentError R.Model(base,[bias];max_depth=depth,nrounds=0)
        end
        for rounds in (-1,401,false)
            @test_throws ArgumentError R.Model(base,[bias];max_depth=1,nrounds=rounds)
        end
        @test_throws ArgumentError R.Model(base,[bias,bias];max_depth=1,nrounds=0)
        for feature in (0,343,true),condition in (0,1)
            split=tree(Any[feature,0,0],Any[condition,0,0],[0.,1.,3.],[true,false,false])
            @test_throws ArgumentError R.Model(base,[bias,split];max_depth=2,nrounds=1)
        end
        for condition in (0,2,true)
            split=tree([342,0,0],Any[condition,0,0],[0.,1.,3.],[true,false,false])
            @test_throws ArgumentError R.Model(base,[bias,split];max_depth=2,nrounds=1)
        end
        @test_throws ArgumentError R.Model(base,[tree([1],[1],[0.],[true])];max_depth=1,nrounds=0)
        @test_throws ArgumentError R.Model(base,[bias,tree([1],[1],[1.],[true])];max_depth=1,nrounds=1)
        @test_throws ArgumentError R.Model(base,[bias,tree([0,1,0],[0,1,0],[0.,1.,1.],[false,true,false])];max_depth=2,nrounds=1)
        @test_throws ArgumentError R.Model(base,[bias,(;feature=[0],condition=[0],prediction=[0.],split=[false],extra=1)];max_depth=1,nrounds=1)
        for prediction in (Inf,NaN,1.0 + 2.0^-30)
            @test_throws ArgumentError R.Model(base,[constant_tree(prediction,1)];max_depth=1,nrounds=0)
        end
        bad=copy(base);bad[1]=Float64[];@test_throws ArgumentError R.Model(bad,[bias];max_depth=1,nrounds=0)
        bad=copy(base);bad[1]=[.5,.4];@test_throws ArgumentError R.Model(bad,[bias];max_depth=1,nrounds=0)
        bad=copy(base);bad[1]=[.5,.6];@test_throws ArgumentError R.Model(bad,[bias];max_depth=1,nrounds=0)
        rawedges=[Float64[.5] for _ in 1:342];model=hand_model(342;edges=rawedges)
        rawedges[342][1]=2.;values=zeros(342);values[342]=1.
        @test R.predict_delta(model,values)==4. # no mutable caller-array alias
        @test_throws MethodError setindex!(model.edges[342],2.,1)
        maximum_model=R.Model([collect(1.0:254.0) for _ in 1:342],vcat([bias],[constant_tree(0.,63) for _ in 1:400]);max_depth=6,nrounds=400)
        @test length(maximum_model.trees)==401
        @test R.predict_delta(maximum_model,zeros(342))==0.
    end
end
run_tests()
end
