function run_authority_tests(results)
    @testset "Read admission, generated fitting and archival fitting remain separate" begin
        entry=results[("primary","ground-only")];value=entry.value;root=entry.root
        request_path=joinpath(root,"case-request.json");directory=joinpath(root,"cases")
        reread=B.read_reverified(request_path,directory;request_sha256=value.request_sha256,
            manifest_sha256=value.manifest_sha256,complete_sha256=value.complete_sha256)
        absent=joinpath(root,"never-created");pin="0"^64
        fit_authority=F.FitAuthority(absent,pin,absent,pin,absent,pin)
        recipe=F.BoostRecipe(depth=2,rounds=2,bins=2,seed=27102026,eta=1.,min_weight=1.,lambda=0.,L2=0.)
        expected=entry.fit.capsule.expected;start=START+Minute(40);stop=START+Minute(120);asof=START+Minute(211)
        for authority in (nothing,Dict(),true,fit_authority)
            @test_throws ArgumentError F.fit_archival(reread,1,recipe,expected;start,stop,asof,authority)
        end
        @test_throws MethodError F.fit_archival(value,1,recipe,expected;start,stop,asof,authority=fit_authority)
        @test !ispath(absent) && !ispath(joinpath(root,"POINT_FIT_ATTEMPT.json"))
        # A deliberate invalid capsule flips a generated source flag, as an
        # admission-negative fixture. No archival qualification or grant exists.
        invalid_cases=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,value.labels,
            value.request_sha256,value.manifest_sha256,value.complete_sha256,true,false)
        invalid=B.ReverifiedCases(invalid_cases,reread.admission_json,B._ReadToken())
        for (a,b) in ((DateTime(2016),DateTime(2017)),(DateTime(2018),DateTime(2019)),
                      (DateTime(2017),DateTime(2018)+Millisecond(1)))
            @test_throws ArgumentError F._archival_request(invalid,1,a,b,b+Minute(31))
        end
        @test_throws ArgumentError F._archival_request(invalid,1,DateTime(2017),DateTime(2018),DateTime(2018)+Minute(31))
        admission=Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"actual_archival_cache",
            "request_sha256"=>value.request_sha256,"artifact_manifest_sha256"=>value.manifest_sha256,
            "artifact_complete_sha256"=>value.complete_sha256,"producer_head"=>"0"^40,"producer_grant_sha256"=>pin,
            "producer_source_proof_sha256"=>pin,"producer_qualification_sha256"=>pin,"reader_head"=>"0"^40,
            "reader_grant_sha256"=>pin,"reader_source_proof_sha256"=>pin,"reader_qualification_sha256"=>pin,
            "all_artifact_bytes_recomputed"=>true,"new_case_generation_authorized"=>false,
            "actual_model_fit_authorized"=>false,"provider_receipts_available"=>false,"actual_fits"=>0)
        declaration=B.ReverifiedCases(invalid_cases,String(R.canonical_bytes(admission;limit=131072)),B._ReadToken())
        first=DateTime(2017);last=DateTime(2017,10);known=DateTime(2018)
        base=Dict("schema"=>"archive_source_bound_point_fit_grant_v1","operation"=>"fixed_recipe_2017_point_fit_only",
            "family"=>F.family(recipe),"case_read_admission_sha256"=>digest(B.admission_bytes(declaration)),
            "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,
            "case_complete_sha256"=>value.complete_sha256,"context_sha256"=>expected.context_sha256,
            "recipe_sha256"=>expected.recipe_sha256,"input_sha256"=>pin,"source_sha256"=>expected.source_sha256,
            "source_proof_sha256"=>pin,"qualification_sha256"=>pin,"library_sha256"=>F.LIBRARY_SHA,
            "table_index"=>1,"start_inclusive"=>clock(first),"stop_exclusive"=>clock(last),"target_asof"=>clock(known),
            "head"=>"0"^40,"actual_source_access_authorized"=>true,"actual_model_fit_authorized"=>true,
            "new_case_generation_authorized"=>false,"dataset_discovery"=>false,"provider_receipts_available"=>false,
            "expected_model_fits"=>1,"execution_directory"=>root)
        # Only deliberately invalid permits are written. Qualification paths
        # stay absent, and no positive archival admission or fit is simulated.
        variants=[("operation","read_only"),("family","wrong"),("case_read_admission_sha256","1"^64),
            ("case_request_sha256","1"^64),("context_sha256","1"^64),("recipe_sha256","1"^64),
            ("source_sha256","1"^64),("source_proof_sha256","1"^64),("qualification_sha256","1"^64),
            ("library_sha256","1"^64),("table_index",true),("stop_exclusive",clock(last+Minute(1))),
            ("actual_source_access_authorized",false),("actual_model_fit_authorized",false),
            ("new_case_generation_authorized",true),("dataset_discovery",true),("provider_receipts_available",true),
            ("expected_model_fits",2)]
        for (index,(field,bad)) in enumerate(variants)
            grant=copy(base);grant[field]=bad;bytes=R.canonical_bytes(grant;limit=131072)
            path=joinpath(root,"invalid-fit-permit-$index.json");R._exclusive_write(path,bytes)
            supplied=F.FitAuthority(path,digest(bytes),absent,pin,absent,pin)
            message=nothing
            try;F._admit_archival(declaration,1,recipe,expected,first,last,known,supplied)
            catch error;error isa ArgumentError || rethrow();message=error.msg
            end
            @test message=="archival point fit grant differs"
        end
        @test !ispath(absent) && !ispath(joinpath(root,"POINT_FIT_ATTEMPT.json"))
        @test_throws ArgumentError F._recipe_bytes(R.canonical_bytes(["foreign"];limit=16384))
        @test_throws ArgumentError F._fit_clock("2020-02-30T00:00:00.000Z")
        @test_throws ArgumentError F._fit_clock("2020-01-01T00:00:00Z")
        # Valid fractional cutoffs remain supported for strict availability.
        @test F._fit_clock("2020-01-01T01:02:45.001Z")==START+Minute(62)+Second(45)+Millisecond(1)
    end
end
