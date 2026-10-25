using SolarSINDy,Dates,JSON3,SHA

"Generated raw-channel software oracle with one controlled fit and exact saved-data replay."
function run_matched_gbm_primitives_synthetic()
    M=SolarSINDy.MatchedGBMPrimitives342;R=M.MatchedScalarTreeReplay342
    fixture_path=joinpath(@__DIR__,"..","test","matched_gbm_primitives","fixtures","generated_fit_oracles_v1.json")
    fixture=JSON3.read(read(fixture_path,String),Any)
    values=Matrix{Float64}(undef,17,1024)
    for i in 1:1024,j in 1:17;values[j,i]=fixture["raw_rows_f64"][i][j];end
    t0=DateTime(2000);tags=[t0+Hour(i-1) for i in 1:1024]
    input=M.HourlyInputs342(tags,values;delay_hours=0,source_sha256=fixture["source_sha256"])
    @assert input.input_id==fixture["input_ids"]["0"]
    # Packaged independent generated-data literals are the external witness;
    # they grant no actual-source fit, release or forecast-skill qualification.
    admission=M.SyntheticAdmission342(input;fixture_sha256=bytes2hex(sha256(read(@__FILE__))),
        external_audit_witness_sha256=bytes2hex(sha256(read(fixture_path))))
    split=(;training_boundary=t0+Hour(512),validation_start=t0+Hour(512),validation_end=t0+Hour(1024))
    expected=M.prepare_synthetic_bindings(input,admission,4,1;split...,loss_contract_sha256=M.FEATURE_CONTRACT_SHA256)
    oracle=only(filter(x->x["delay"]==0&&x["horizon"]==4,fixture["cells"]))
    @assert expected.training_mask_sha256==oracle["training_mask_sha256"]
    fitted=M.fit_synthetic_cell(input,expected;split...)
    design=M.build_features(input,[t+Hour(1) for t in tags]).matrix
    normalized=M.transform(fitted.capsule.encoder,design)
    native=SolarSINDy.gbm_predict(fitted.native,normalized)
    @assert length(unique(native))>1
    @assert any(any(tree.split[i]&&tree.feature[i]>29 for i in eachindex(tree.feature)) for tree in fitted.capsule.model.trees)
    mktempdir() do root
        encoder_member=M.save_encoder(joinpath(root,"encoder.json"),fitted.capsule.encoder)
        model_member=M.save_model(joinpath(root,"model.json"),fitted.capsule;expected_bindings=expected)
        encoder=M.load_encoder(encoder_member.path;expected_sha256=expected.preprocessing_sha256,
            expected_training_design_sha256=fitted.capsule.encoder.training_design_sha256)
        model=M.load_model(model_member.path;expected_sha256=model_member.sha256,expected_bindings=expected)
        replay=R.predict_deltas(model,M.transform(encoder,design))
        @assert reinterpret(UInt64,native)==reinterpret(UInt64,replay)
    end
    println("Matched342 generated fit/save-load replay PASS:1024 rows and extended-feature split")
    true
end
