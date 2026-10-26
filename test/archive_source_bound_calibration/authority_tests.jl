function run_invalid_calibration_authority_tests(value,rereverified,point,expected,root)
    @testset "Calibration permits reject invalid fields before absent qualification" begin
        pin="0"^64;absent=joinpath(root,"absent-qualification")
        invalid_cases=Q.VerifiedCases(Q._Checked(),value.request_json,value.source,value.features,value.labels,
            value.request_sha256,value.manifest_sha256,value.complete_sha256,true,false)
        admission=Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"actual_archival_cache",
            "request_sha256"=>value.request_sha256,"artifact_manifest_sha256"=>value.manifest_sha256,"artifact_complete_sha256"=>value.complete_sha256,
            "producer_head"=>"0"^40,"producer_grant_sha256"=>pin,"producer_source_proof_sha256"=>pin,"producer_qualification_sha256"=>pin,
            "reader_head"=>"0"^40,"reader_grant_sha256"=>pin,"reader_source_proof_sha256"=>pin,"reader_qualification_sha256"=>pin,
            "all_artifact_bytes_recomputed"=>true,"new_case_generation_authorized"=>false,"actual_model_fit_authorized"=>false,
            "provider_receipts_available"=>false,"actual_fits"=>0)
        declaration=B.ReverifiedCases(invalid_cases,String(R.canonical_bytes(admission;limit=131072)),B._ReadToken())
        first=DateTime(2017);last=DateTime(2017,10);known=DateTime(2018);evaluation=known
        pins=K._point_pins(point)
        base=Dict("schema"=>"archive_source_bound_calibration_grant_v1","operation"=>"fixed_model_2017_scores_only",
            "case_read_admission_sha256"=>digest(B.admission_bytes(declaration)),"case_request_sha256"=>value.request_sha256,
            "case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
            "point_context_sha256"=>point.expected.context_sha256,"point_recipe_sha256"=>point.expected.recipe_sha256,
            "point_model_sha256"=>pins.model,"point_receipt_sha256"=>pins.receipt,"context_sha256"=>expected.context_sha256,
            "input_sha256"=>expected.input_sha256,"job_sha256"=>K.job_identity(),"source_proof_sha256"=>pin,"qualification_sha256"=>pin,
            "head"=>"0"^40,"table_index"=>1,"start_inclusive"=>clock(first),"stop_exclusive"=>clock(last),"target_asof"=>clock(known),
            "first_evaluation_issue"=>clock(evaluation),"actual_source_access_authorized"=>true,"actual_model_fit_authorized"=>false,
            "new_case_generation_authorized"=>false,"calibration_authorized"=>true,"dataset_discovery"=>false,
            "provider_receipts_available"=>false,"execution_directory"=>root)
        producer=F.FitAuthority(absent,pin,absent,pin,absent,pin)
        variants=[("schema","wrong"),("operation","fit"),("case_request_sha256","1"^64),("case_manifest_sha256","1"^64),
            ("case_read_admission_sha256","1"^64),("point_context_sha256","1"^64),("point_recipe_sha256","1"^64),
            ("point_model_sha256","1"^64),("point_receipt_sha256","1"^64),("context_sha256","1"^64),("input_sha256","1"^64),
            ("job_sha256","1"^64),("source_proof_sha256","1"^64),("qualification_sha256","1"^64),("table_index",true),
            ("first_evaluation_issue",clock(evaluation+Minute(1))),("actual_source_access_authorized",false),("actual_model_fit_authorized",true),
            ("new_case_generation_authorized",true),("calibration_authorized",false),("dataset_discovery",true),("provider_receipts_available",true)]
        for (i,(field,bad)) in enumerate(variants)
            # No baseline permit is written, no qualification file is created,
            # and each deliberately invalid permit must fail before source use.
            grant=copy(base);grant[field]=bad;bytes=R.canonical_bytes(grant;limit=131072)
            path=joinpath(root,"invalid-calibration-permit-$i.json");R._exclusive_write(path,bytes)
            supplied=K.CalibrationAuthority(path,digest(bytes),absent,pin,absent,pin)
            message=nothing
            try;K._admit(declaration,1,point,expected,first,last,known,evaluation,supplied,producer)
            catch error;error isa ArgumentError || rethrow();message=error.msg
            end
            @test message=="archival calibration grant differs"
        end
        @test !ispath(absent) && !ispath(joinpath(root,"CALIBRATION_ATTEMPT.json"))
    end
end
