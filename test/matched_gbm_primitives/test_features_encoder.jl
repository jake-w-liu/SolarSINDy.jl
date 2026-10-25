module Matched342FeatureEncoderTests
using Test,Dates,JSON3,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
const D=SolarSINDy.DurationConditionedForcing
const FIX=joinpath(@__DIR__,"fixtures")
const T0=DateTime(2000)
read_fixture(name)=JSON3.read(read(joinpath(FIX,name),String),Any)
bits(x)=string(reinterpret(UInt64,x);base=16,pad=16)
function raw_constant(n)
    constants=Float64[2,-90,0,0,0,-2,2,400,1,100000,1,1,4/5,5,0,10,1/25]
    values=repeat(constants,1,n)
    for i in 1:n;values[15,i]=-101+i;end
    DateTime[T0+Hour(i-1) for i in 1:n],values
end
function receipts(tags,delay)
    [tags[i]+Hour(1+(j==15 ? delay : 0)) for j in 1:17,i in eachindex(tags)]
end
function matrix_of(rows)
    X=Matrix{Float64}(undef,length(rows),342)
    for i in eachindex(rows),j in 1:342;X[i,j]=rows[i][j]===nothing ? NaN : Float64(rows[i][j]);end
    X
end
function check_bits(values,expected)
    @test length(values)==length(expected)
    for (x,b) in zip(values,expected)
        b===nothing ? (@test isnan(x)) : (@test bits(x)==b)
    end
end
function run_tests()
    @testset "Matched342 ordered features and receipts" begin
        six=read_fixture("six_vectors_v1.json")
        @test collect(M.FEATURE_NAMES)==six["feature_names"]
        for expected in six["vectors"]
            delay=Int(expected["delay"]);variant=expected["variant"];tags,values=raw_constant(100);received=receipts(tags,delay)
            variant=="missing_latest_bz"&&(values[6,end]=NaN)
            if variant=="delayed_latest_driver"
                for j in 1:17;j==15|| (received[j,end]=T0+Hour(101));end
            end
            input=M.HourlyInputs342(tags,values;delay_hours=delay,received,source_sha256=repeat("a",64))
            result=M.build_features(input,DateTime[T0+Hour(100)])
            check_bits(vec(result.matrix),[x===nothing ? nothing : string(reinterpret(UInt64,Float64(x));base=16,pad=16) for x in expected["values"]])
            @test result.rows[1].anchor==-1-delay
        end
        boundary=read_fixture("boundary_inputs_v1.json");expected=read_fixture("boundary_expected_v1.json")
        for fixture in boundary["trace_cases"],delay in boundary["trace_delays"]
            rows=D.WindHour[D.WindHour(T0+Hour(r["tag_hour"]),T0+Hour(r["complete_hour"]),T0+Hour(r["receipt_hour"]),r["bz"]===nothing ? NaN : r["bz"],r["q"]===nothing ? NaN : r["q"]) for r in fixture["wind_rows"]]
            oracle=only(filter(x->x["name"]==fixture["name"]&&x["delay"]==delay,expected["trace_vectors"]))
            @test M.trace_context(rows,T0+Hour(fixture["issue_hour"]);delay_hours=delay)==oracle["context"]
        end
        for oracle in expected["perlead_cohorts"]
            tags,values=raw_constant(1024);oracle["variant"]=="missing600"&&(values[15,600]=NaN)
            input=M.HourlyInputs342(tags,values;delay_hours=oracle["delay"],source_sha256=repeat("b",64))
            c=M.cohorts(input,oracle["horizon"];training_boundary=T0+Hour(512),validation_start=T0+Hour(512),validation_end=T0+Hour(1024))
            @test c.train==oracle["training_issues"]
            @test c.validation==oracle["scoreable_evaluation_issues"]
        end
        binary=read_fixture("binary_clock_v1.json")
        # The nontrivial raw array is17channels, unlike standalone342designs.
        raw=Matrix{Float64}(undef,17,100)
        for i in 1:100,j in 1:17;raw[j,i]=Float64(binary["raw_nontrivial_values"][i][j]);end
        for oracle in binary["clock_vectors"]
            delay=Int(oracle["delay"]);tags=[T0+Hour(i-1) for i in 1:100];received=receipts(tags,delay)
            oracle["variant"]=="latest_bz_one_ms_late"&&(received[6,end]+=Millisecond(1))
            input=M.HourlyInputs342(tags,raw;delay_hours=delay,received,source_sha256=repeat("c",64))
            result=M.build_features(input,DateTime[T0+Hour(100)]);check_bits(vec(result.matrix),oracle["values_f64_bits"])
            @test bits(result.rows[1].anchor)==oracle["anchor_f64_bits"]
        end
        tags,values=raw_constant(100);input=M.HourlyInputs342(tags,values;source_sha256=repeat("d",64))
        original=M.build_features(input,DateTime[T0+Hour(50)]).matrix
        future=copy(values);future[:,60:end].*=2;future[15,60:end].=-9000
        future_input=M.HourlyInputs342(tags,future;source_sha256=repeat("e",64))
        @test isequal(original,M.build_features(future_input,DateTime[T0+Hour(50)]).matrix)
        values[6,99]=NaN # constructor copied caller data
        @test isequal(original,M.build_features(input,DateTime[T0+Hour(50)]).matrix)
        input.values[6,20]=NaN
        @test_throws ArgumentError M.build_features(input,DateTime[T0+Hour(50)])
        tags,values=raw_constant(100);values[9,end]=0
        input=M.HourlyInputs342(tags,values;source_sha256=repeat("f",64));row=only(M.build_features(input,DateTime[T0+Hour(100)]).rows)
        @test all(isnan,@view row.values[1:29])
        @test row.values[30+8*8]==0. # density rawlag0 retained
        @test all(==(1.),@view row.values[166:194])
        @test row.anchor!==nothing
        for value in (Inf,-Inf)
            bad=copy(values);bad[6,1]=value
            @test_throws ArgumentError M.HourlyInputs342(tags,bad;source_sha256=repeat("f",64))
        end
        @test_throws ArgumentError M.HourlyInputs342(tags,trues(17,100);source_sha256=repeat("f",64))
        bad=copy(values);bad[6,10]=-floatmax(Float64)
        input=M.HourlyInputs342(tags,bad;source_sha256=repeat("f",64))
        @test_throws ArgumentError M.build_features(input,DateTime[T0+Hour(100)])
    end
    @testset "Matched342 training-only encoder binary characterization" begin
        data=read_fixture("boundary_inputs_v1.json");binary=read_fixture("binary_clock_v1.json")["binary_scaler"]
        X=matrix_of(data["scaler_primitive"]["matrix"]);encoder=M.fit_encoder(X,[1,2,3])
        check_bits(encoder.imputation,binary["imputation_f64_bits"])
        check_bits(encoder.center,binary["center_f64_bits"])
        graphs=read_fixture("binary_scaler_graphs_v2.json")["graphs"]
        matching=[name for (name,graph) in graphs if bits.(collect(encoder.scale))==graph["scale_f64_bits"]]
        @test length(matching)==1
        length(matching)==1||error("unrecognized maintained Statistics reduction graph")
        selected=graphs[only(matching)]
        check_bits(encoder.scale,selected["scale_f64_bits"])
        Z=M.transform(encoder,X)
        for i in 1:4;check_bits(vec(Z[i,:]),selected["normalized_f64_bits"][i]);end
        poison=copy(X);poison[4,1]=1e6;poison[4,3]=2e6;poison_encoder=M.fit_encoder(poison,[1,2,3])
        @test encoder.center==poison_encoder.center
        @test encoder.scale==poison_encoder.scale
        @test encoder.imputation==poison_encoder.imputation
        @test encoder.training_design_sha256==poison_encoder.training_design_sha256
        one=M.fit_encoder(X,[1]);@test all(==(1.),one.scale)
        @test_throws ArgumentError M.fit_encoder(X,Int[])
        @test_throws ArgumentError M.fit_encoder(X,[1,1])
        @test_throws ArgumentError M.fit_encoder(X,[2,1])
        @test_throws ArgumentError M.fit_encoder(X,[true])
        bad=copy(X);bad[1,166]=1;@test_throws ArgumentError M.fit_encoder(bad,[1,2,3])
        bad=copy(X);bad[4,1]=Inf;@test_throws ArgumentError M.fit_encoder(bad,[1,2,3])
        @test_throws ArgumentError M.transform(encoder,X[:,1:341])
    end
end
run_tests()
end
