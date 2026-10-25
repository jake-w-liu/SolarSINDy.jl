@testset "pinned synthetic protocol identity and declared conventions" begin
    ground=make_protocol(protocol_fixture())
    @test R._spec(ground)["layout"]=="ground11"
    @test R._recipe(R.RidgeRecipe(ground;role=:candidate,penalty=1))["intercept_penalty"]===false
    mixed=make_protocol(protocol_fixture(;mode="mixed8"))
    @test R._recipe(R.RidgeRecipe(mixed;role=:fixed_mixed8,penalty=1))["intercept_penalty"]===true
    @test_throws ArgumentError R.RidgeRecipe(ground;role=:candidate,penalty=1,decision_pin=repeat("1",64))
    for key in ("evidence_mode","availability_kind","runtime_contract","numeric_contract")
        bad=protocol_fixture();bad[key]="unapproved"
        @test_throws ArgumentError make_protocol(bad)
    end
    for key in ("protocol_id","station","ground_product")
        bad=protocol_fixture();bad[key]=""
        @test_throws ArgumentError make_protocol(bad)
    end
    bad=protocol_fixture();reverse!(bad["feature_names"])
    @test_throws ArgumentError make_protocol(bad)
    bad=protocol_fixture();bad["model_recipes"][1]["intercept_penalty"]=true
    @test_throws ArgumentError make_protocol(bad)
    bad=protocol_fixture();bad["policy"]["ground_extra_ms"]=true
    @test bad["policy"]["ground_extra_ms"]===true
    @test_throws ArgumentError make_protocol(bad)
    bad=protocol_fixture();bad["measured_receipts_available"]=true
    @test_throws ArgumentError make_protocol(bad)
    bytes=R.canonical_bytes(protocol_fixture())
    @test_throws ArgumentError R.RefitProtocol(bytes;expected_spec_sha256=repeat("0",64))
    @test_throws ArgumentError R.RefitProtocol(zeros(UInt8,R.MAX_METADATA_BYTES+1);expected_spec_sha256=repeat("0",64))
end
@testset "exactly one declared recipe is addressable" begin
    @test R.RidgeRecipe(make_protocol(protocol_fixture());role=:candidate,penalty=1).sha256 isa String
    @test_throws ArgumentError R.RidgeRecipe(make_protocol(protocol_fixture());role=:candidate,penalty=2)
    duplicate=protocol_fixture();push!(duplicate["model_recipes"],deepcopy(first(duplicate["model_recipes"])))
    @test_throws ArgumentError R.RidgeRecipe(make_protocol(duplicate);role=:candidate,penalty=1)
    distinct=protocol_fixture()
    split=deepcopy(first(distinct["splits"]));split["name"]="fit-early"
    split["issue_stop_exclusive"]=R.time_text(DateTime(2017,7,1));split["label_ready_before"]=R.time_text(DateTime(2017,7,1));split["target_admit_asof"]=R.time_text(DateTime(2017,7,1))
    push!(distinct["splits"],split)
    recipe=deepcopy(first(distinct["model_recipes"]));recipe["split"]="fit-early";push!(distinct["model_recipes"],recipe)
    @test_throws ArgumentError R.RidgeRecipe(make_protocol(distinct);role=:candidate,penalty=1)
end
@testset "ground-only absent wind descriptor retains the ground route" begin
    # Frozen bootstrap oracle ground_wind_forbidden: absent wind is described
    # but never opened or used to alter an otherwise valid ground-only route.
    protocol=make_protocol(protocol_fixture(;absent_wind=true))
    @test R._spec(protocol)["wind_product_or_null"]===nothing
    @test length(R._spec(protocol)["source_permissions"])==2
end
