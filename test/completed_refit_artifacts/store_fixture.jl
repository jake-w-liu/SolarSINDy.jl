function structural_native_context(protocol)
    spec=R._spec(protocol)
    common=Dict{String,Any}(k=>deepcopy(spec[k]) for k in ("protocol_id","station","ground_product","wind_product_or_null","units","layout","feature_names","cohort_kind","policy","availability_kind","measured_receipts_available"))
    merge!(common,Dict{String,Any}("protocol_sha256"=>protocol.sha256,"source_manifest_sha256_or_null"=>nothing,
        "schedule_manifest_sha256_or_null"=>nothing,"package_commit"=>repeat("1",40),"package_tree"=>repeat("2",40),
        "code_hashes"=>Dict("structural-test"=>repeat("3",64)),"julia_project_sha256"=>repeat("4",64),
        "julia_manifest_sha256"=>repeat("5",64),"runtime_id"=>repeat("6",64),
        "kind_context"=>Dict{String,Any}("operation"=>"source_audit","payload_type"=>"source_audit",
            "input_manifest_sha256_or_null"=>nothing,"input_data_sha256"=>repeat("7",64),"payload_sha256"=>repeat("8",64),
            "internal_receipt_sha256"=>repeat("9",64),"native_receipt_sha256"=>repeat("a",64))))
    common
end
function structural_native_members(protocol)
    request=Dict{String,Any}("schema"=>"dbdt_completed_job_request_v1","job_id"=>"structural-test",
        "operation"=>"source_audit","evidence_mode"=>"synthetic_only","protocol_pin"=>protocol.sha256,
        "input_manifest_pin_or_null"=>nothing,"input_data_pin"=>repeat("7",64),"recipe_pin_or_null"=>nothing,
        "source_permission_pin_or_null"=>repeat("8",64),"required_runtime"=>Dict{String,Any}(),
        "dependency_pins"=>Any[],"work_directory"=>CODEC_TEST_DIRECTORY)
    Pair{String,String}["job.json"=>R._jsontext(request),"payload.json"=>"{}\n","internal-receipt.json"=>"{}\n",
        "native-receipt.json"=>"{}\n","native.log"=>""]
end
function structural_native_bundle(directory)
    protocol=make_protocol(protocol_fixture())
    bundle=R._publish_bundle(directory,protocol,"native_job",structural_native_context(protocol),structural_native_members(protocol),0;pins=Dict{String,String}())
    pins=Dict(bundle.manifest_sha256=>bundle.directory,bundle.commit_sha256=>bundle.directory)
    protocol,bundle,pins
end
