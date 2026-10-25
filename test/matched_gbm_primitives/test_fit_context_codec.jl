module Matched342FitContextCodecTests
using Test,Dates,JSON3,SHA,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
const R=M.MatchedScalarTreeReplay342
const FIX=joinpath(@__DIR__,"fixtures/generated_fit_oracles_v1.json")
const F=JSON3.read(read(FIX,String),Any)
const T0=DateTime(2000)
const SPLIT=(;training_boundary=T0+Hour(512),validation_start=T0+Hour(512),validation_end=T0+Hour(1024))
const AUDIT="0087c702a6ef3f131f257d5f54adf2235ed2a2810c4235a8d6038dfe266773ef"
function input_for(delay)
    values=Matrix{Float64}(undef,17,1024)
    for i in 1:1024,j in 1:17;values[j,i]=F["raw_rows_f64"][i][j];end
    M.HourlyInputs342([T0+Hour(i-1) for i in 1:1024],values;delay_hours=delay,source_sha256=F["source_sha256"])
end
function admission_for(input)
    M.SyntheticAdmission342(input;fixture_sha256=bytes2hex(sha256(read(FIX))),external_audit_witness_sha256=AUDIT)
end
bits(x)=string(reinterpret(UInt64,x);base=16,pad=16)
function rehash!(object)
    without=Dict(key=>value for (key,value) in object if key!="model_id")
    object["model_id"]=bytes2hex(sha256(M._canonical(without)));object
end
function boolean_array_mutant!(object,key;tree=2)
    values=Any[object["trees"][tree][key]...];values[1]=true
    @test values[1]===true
    object["trees"][tree][key]=values
end
function hand_grid_model()
    edges=[vcat(0.5,collect(1.5:1.0:62.5)) for _ in 1:342]
    bias=(;feature=[0],condition=[0],prediction=Float32[2^24],split=[false])
    trees=NamedTuple[bias]
    for i in 1:200
        feature=zeros(Int,15);condition=zeros(Int,15);prediction=zeros(Float32,15);split=falses(15)
        if i==1
            feature[1]=30;condition[1]=1;split[1]=true;prediction[2]=1;prediction[3]=3
        elseif i==2
            prediction[1]=-Float32(2^24)
        end
        push!(trees,(;feature,condition,prediction,split=collect(split)))
    end
    R.Model(edges,trees;max_depth=4,nrounds=200)
end
function run_tests()
    selected=nothing
    @testset "Matched342 independent generated fit context clocks/targets" begin
        @test Dates.value(T0)==F["origin_julia_milliseconds"]
        @test M._trusted_evo_sha()==M.EVOTREES_SOURCE_SHA256
        for delay in (0,1)
            input=input_for(delay);@test input.input_id==F["input_ids"][string(delay)]
            admission=admission_for(input)
            for h in (4,6)
                oracle=only(filter(x->x["delay"]==delay&&x["horizon"]==h,F["cells"]))
                prepared=M._prepare_synthetic(input,admission,h;SPLIT...)
                @test prepared.cohort.train==oracle["train_indices"]
                @test prepared.cohort.validation==oracle["validation_indices"]
                @test prepared.mask==oracle["training_mask_sha256"]
                @test bits.(prepared.targets)==oracle["target_f64_bits"]
                @test minimum(prepared.targets)==oracle["minimum_target"]
                @test maximum(prepared.targets)==oracle["maximum_target"]
                expected=M.ExpectedModelBindings342(admission,delay,h,1;
                    training_mask_sha256=prepared.mask,preprocessing_sha256=M.preprocessing_sha256(prepared.encoder),
                    evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
                @test M._check_bindings(expected)===nothing
                if delay==0&&h==4
                    @test count(==(16.),prepared.targets)==174
                    @test count(==(32.),prepared.targets)==166
                    @test prepared.design[prepared.cohort.train,30]==prepared.targets
                    selected=expected
                end
            end
        end
    end
    @testset "Matched342 exact21-field externally pinned scalar model codec" begin
        expected=selected;model=hand_grid_model();object=M._model_object(model,expected)
        @test length(object)==21
        @test length(object["trees"][1])==4
        @test M._decode_model(object,expected) isa R.Model
        @test_throws MethodError M.save_model("unused",model;expected_bindings=expected)
        x=zeros(342);x[30]=0.5;@test R.predict_delta(model,x)==0.
        x[30]=nextfloat(0.5);@test R.predict_delta(model,x)==4.
        mktempdir() do dir
            dir=realpath(dir)
            written=M._write_json(joinpath(dir,"model.json"),object)
            loaded=M.load_model(written.path;expected_sha256=written.sha256,expected_bindings=expected)
            @test R.predict_delta(loaded,x)==4.
            @test M._model_object(loaded,expected)==object
            @test_throws ArgumentError M.load_model(written.path;expected_sha256=repeat("0",64),expected_bindings=expected)
            changed=deepcopy(object);changed["trees"][1]["prediction_f32_bits"][1]="4b800001"
            written=M._write_json(joinpath(dir,"changed.json"),changed)
            @test_throws ArgumentError M.load_model(written.path;expected_sha256=bytes2hex(sha256(M._canonical(object))),expected_bindings=expected)
        end
        for key in keys(object)
            bad=deepcopy(object);delete!(bad,key)
            @test_throws ArgumentError M._decode_model(bad,expected)
        end
        for (key,value) in (("feature_count",29),("feature_count",341),("feature_count",343),
                ("feature_count",true),("cell_delay",1),("cell_horizon",6),("setting_index",2),
                ("max_depth",6),("nrounds",400),("nbins",255),("seed",22022027),
                ("schema","wrong"),("feature_schema_sha256",repeat("0",64)),
                ("feature_contract_sha256",repeat("0",64)),("loss_contract_sha256",repeat("0",64)),
                ("input_admission_sha256",repeat("0",64)),("training_mask_sha256",repeat("0",64)),
                ("preprocessing_sha256",repeat("0",64)),("evotrees_source_sha256",repeat("0",64)))
            bad=deepcopy(object);bad[key]=value;rehash!(bad)
            @test_throws ArgumentError M._decode_model(bad,expected)
        end
        for mutate in (
                o->(o["extra"]=1),o->(o["trees"][2]["extra"]=1),
                o->(o["trees"][2]["feature"][1]=343),o->(o["trees"][2]["feature"][1]=0),
                o->boolean_array_mutant!(o,"feature"),o->(o["trees"][2]["condition"][1]=0),
                o->boolean_array_mutant!(o,"condition"),o->(o["trees"][1]["split"][1]=true),
                o->(o["numeric_feature_types"][342]=false),
                o->(o["trees"][2]["prediction_f32_bits"][2]="7f800000"),
                o->(o["trees"][2]["prediction_f32_bits"][2]="7fc00000"),
                o->(o["edges_f64_bits"][342][1]="7ff0000000000000"),
                o->(o["edges_f64_bits"][342][1]="3FE0000000000000"),
                o->pop!(o["trees"]),o->pop!(o["edges_f64_bits"][1]),
                o->(o["trees"][1]["split"][1]=1))
            bad=deepcopy(object);mutate(bad);rehash!(bad)
            @test_throws ArgumentError M._decode_model(bad,expected)
        end
    end
end
end
Matched342FitContextCodecTests.run_tests()
