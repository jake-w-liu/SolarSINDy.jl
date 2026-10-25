module Matched342GeneratedFitTests
using Test,Dates,SHA,SolarSINDy
const M=SolarSINDy.MatchedGBMPrimitives342
const R=M.MatchedScalarTreeReplay342
const C=getfield(parentmodule(@__MODULE__),:Matched342FitContextCodecTests)
bits(x)=reinterpret(UInt64,x)
function run_tests()
    @testset "Matched342 genuine generated fixed8 fit/save-load replay" begin
        input=C.input_for(0);admission=C.admission_for(input)
        prepared=M._prepare_synthetic(input,admission,4;C.SPLIT...)
        design=prepared.design;standardized=M.transform(prepared.encoder,design)
        @test minimum(prepared.targets)==16.
        @test maximum(prepared.targets)==32.
        @test minimum(design[prepared.cohort.train,30])==16.
        @test maximum(design[prepared.cohort.train,30])==32.
        public_expected=M.prepare_synthetic_bindings(input,admission,4,1;C.SPLIT...,loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
        @test public_expected.training_mask_sha256==prepared.mask
        for setting in 1:8
            expected=M.ExpectedModelBindings342(admission,0,4,setting;training_mask_sha256=prepared.mask,
                preprocessing_sha256=M.preprocessing_sha256(prepared.encoder),evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,
                loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
            println("SYNTHETIC_NATIVE_FIT_START setting=",setting," ",time());flush(stdout)
            fitted=M.fit_synthetic_cell(input,expected;C.SPLIT...)
            println("SYNTHETIC_NATIVE_FIT_DONE setting=",setting," ",time());flush(stdout)
            @test fitted.training_rows==340
            @test fitted.validation_rows==341
            @test fitted.capsule.recipe==M._recipe(setting)
            @test M.preprocessing_sha256(fitted.capsule.encoder)==expected.preprocessing_sha256
            native=SolarSINDy.gbm_predict(fitted.native,standardized)
            @test length(unique(native))>1
            @test any(any(tree.split[i]&&tree.feature[i]>29 for i in eachindex(tree.feature)) for tree in fitted.capsule.model.trees)
            replay=R.predict_deltas(fitted.capsule.model,standardized)
            @test length(replay)==1024
            for i in eachindex(native);@test bits(native[i])==bits(replay[i]);end
            mktempdir() do dir
                saved_encoder=M.save_encoder(joinpath(dir,"encoder.json"),fitted.capsule.encoder)
                @test saved_encoder.sha256==expected.preprocessing_sha256
                loaded_encoder=M.load_encoder(saved_encoder.path;expected_sha256=expected.preprocessing_sha256,
                    expected_training_design_sha256=fitted.capsule.encoder.training_design_sha256)
                reloaded_design=M.transform(loaded_encoder,design)
                @test reinterpret(UInt64,vec(reloaded_design))==reinterpret(UInt64,vec(standardized))
                saved=M.save_model(joinpath(dir,"model.json"),fitted.capsule;expected_bindings=expected)
                loaded=M.load_model(saved.path;expected_sha256=saved.sha256,expected_bindings=expected)
                loaded_predictions=R.predict_deltas(loaded,reloaded_design)
                for i in eachindex(native);@test bits(native[i])==bits(loaded_predictions[i]);end
                @test saved.bytes<=M.MAX_BYTES
                @test_throws ArgumentError M.save_model(saved.path,fitted.capsule;expected_bindings=expected)
                @test_throws MethodError M.save_model(joinpath(dir,"bare.json"),fitted.native;expected_bindings=expected)
                if setting==1
                    previous=saved.sha256
                    fitted.native.trees[1].pred[1,1]+=2.0f0
                    @test any(bits(a)!=bits(b) for (a,b) in zip(native,SolarSINDy.gbm_predict(fitted.native,standardized)))
                    fitted.native.info[:feature_names][1]=:changed_after_fit
                    fitted.native.info[:edges][30][1]=nextfloat(fitted.native.info[:edges][30][1])
                    after=M.save_model(joinpath(dir,"after-native-mutation.json"),fitted.capsule;expected_bindings=expected)
                    @test after.sha256==previous
                    @test read(after.path)==read(saved.path)
                    @test_throws MethodError setindex!(fitted.capsule.model.trees[1].prediction,0.0f0,1)
                    @test_throws ArgumentError (fitted.capsule.encoder.training_rows[1]=2)
                    wrong=M.ExpectedModelBindings342(admission,0,4,2;training_mask_sha256=prepared.mask,
                        preprocessing_sha256=expected.preprocessing_sha256,evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,
                        loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
                    @test_throws ArgumentError M.save_model(joinpath(dir,"wrong-context.json"),fitted.capsule;expected_bindings=wrong)
                    @test !ispath(joinpath(dir,"wrong-context.json"))
                end
            end
        end
        wrong=M.ExpectedModelBindings342(admission,0,4,1;training_mask_sha256=repeat("0",64),
            preprocessing_sha256=M.preprocessing_sha256(prepared.encoder),evotrees_source_sha256=M.EVOTREES_SOURCE_SHA256,
            loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
        @test_throws ArgumentError M.fit_synthetic_cell(input,wrong;C.SPLIT...)
        input.values[1,1]+=1.
        @test_throws ArgumentError M.fit_synthetic_cell(input,public_expected;C.SPLIT...)
        println("SYNTHETIC_NATIVE_FIXED_GRID_ALL8_DONE");flush(stdout)
    end
end
end
Matched342GeneratedFitTests.run_tests()
