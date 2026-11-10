function saved_state_cell_checks(root)
    f=source_pair_fixture(root)
    for (delay,h) in ((0,4),(0,6),(1,4),(1,6))
        cell=FC.prepare_cell(f.pair,delay,h);expected=FC.expected_bindings(cell,1)
        independent=independent_fit_binding(f.admissions[delay+1],f.fixtures[delay+1],h,1)
        @test minimum(cell.targets)<maximum(cell.targets)
        println("MATCHED342_BOUND_NATIVE_START L=",delay," h=",h);flush(stdout)
        fitted=FC.fit_generated_cell(cell,expected)
        previous=M.fit_synthetic_cell(independent.input,independent.expected;independent.split...)
        observed=FC.predict_cell(cell,fitted.capsule,expected;native=fitted.native)
        # This older numerical entry remains unchanged by the new source-bound
        # composition. Compare its native fit on the same complete training set.
        old_design=M.build_features(independent.input,FC.cell_metadata(cell).issues).matrix
        old_X=M.transform(previous.capsule.encoder,old_design)
        old_prediction=SolarSINDy.gbm_predict(previous.native,old_X[cell.union,:])
        @test f64words(observed)==f64words(old_prediction)
        @test fitted.training_rows==previous.training_rows==length(cell.cohort.train)
        @test fitted.validation_rows==previous.validation_rows==length(cell.cohort.validation)
        @test fitted.capsule.recipe==previous.capsule.recipe
        @test fitted.capsule.model.edges==previous.capsule.model.edges
        @test M.preprocessing_sha256(fitted.capsule.encoder)==M.preprocessing_sha256(previous.capsule.encoder)
        out=mkdir(joinpath(root,"L$delay-h$h"))
        saved_encoder=FC.save_encoder(joinpath(out,"encoder.json"),cell)
        encoder=M.load_encoder(saved_encoder.path;expected_sha256=saved_encoder.sha256,
            expected_training_design_sha256=cell.encoder.training_design_sha256)
        saved=FC.save_model(joinpath(out,"model.json"),fitted.capsule;expected_bindings=expected)
        @test length(keys(b2_json(saved.path)))==21
        @test all(length(keys(tree))==4 for tree in b2_json(saved.path)["trees"])
        model=FC.load_model(saved.path;expected_sha256=saved.sha256,expected_bindings=expected)
        reloaded=FC.replay_cell(cell,model,encoder,expected;expected_model_sha256=saved.sha256,
            expected_encoder_sha256=saved_encoder.sha256)
        @test f64words(reloaded)==f64words(observed)
        @test_throws ArgumentError FC.load_model(saved.path;expected_sha256=repeat("0",64),expected_bindings=expected)
        wrong=FC.expected_bindings(cell,2)
        @test_throws ArgumentError FC.load_model(saved.path;expected_sha256=saved.sha256,expected_bindings=wrong)
        @test_throws ArgumentError FC.save_model(joinpath(out,"wrong-setting.json"),fitted.capsule;expected_bindings=wrong)
        @test !ispath(joinpath(out,"wrong-setting.json"))
        @test_throws TypeError M.load_model(saved.path;expected_sha256=saved.sha256,expected_bindings=expected)
        @test_throws TypeError FC.load_model(saved.path;expected_sha256=saved.sha256,expected_bindings=independent.expected)
        @test_throws MethodError FC.save_model(joinpath(out,"bare-native.json"),fitted.native;expected_bindings=expected)
        # Changing the native training object cannot alter the immutable export.
        fitted.native.trees[1].pred[1,1]+=2.0f0
        fitted.native.info[:feature_names][1]=:mutated_native_after_snapshot
        resaved=FC.save_model(joinpath(out,"immutable.json"),fitted.capsule;expected_bindings=expected)
        @test read(resaved.path)==read(saved.path)
        @test resaved.sha256==saved.sha256
        @test_throws ArgumentError FC.predict_cell(cell,fitted.capsule,expected;native=fitted.native)
        # Even a coherently rehashed model cannot change its bound delay.
        mutant=b2_json(saved.path);mutant["cell_delay"]=1-delay
        delete!(mutant,"model_id");mutant["model_id"]=b2_id(mutant)
        mutant_path=joinpath(out,"rehash-wrong-delay.json");mutant_sha=b2_write(mutant_path,mutant)
        @test_throws ArgumentError FC.load_model(mutant_path;expected_sha256=mutant_sha,expected_bindings=expected)
        println("MATCHED342_BOUND_NATIVE_DONE L=",delay," h=",h);flush(stdout)
    end
end
