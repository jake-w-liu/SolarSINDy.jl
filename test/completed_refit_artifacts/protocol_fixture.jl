function protocol_fixture(;mode="ground11",absent_wind=false)
    wind=mode=="mixed8" ? "synthetic_arrival_wind" : nothing
    permission=Dict{String,Any}("body_id"=>"ground","path_or_null"=>abspath(joinpath(@__DIR__,"never-opened-ground.json")),
        "bytes_or_null"=>1,"sha256_or_null"=>repeat("1",64),"format"=>"dbdt_synthetic_ground_vectors_v1",
        "period_start"=>R.time_text(DateTime(2017)),"period_stop"=>R.time_text(DateTime(2018)),
        "station"=>"SYN","product"=>"synthetic_ground","neighbor_role"=>"primary","disposition"=>"approved","approval_sha256"=>repeat("2",64))
    permissions=Any[permission]
    if wind!==nothing || absent_wind
        push!(permissions,Dict{String,Any}("body_id"=>"wind","path_or_null"=>nothing,"bytes_or_null"=>nothing,
            "sha256_or_null"=>nothing,"format"=>"dbdt_synthetic_arrival_wind_v1",
            "period_start"=>R.time_text(DateTime(2017)),"period_stop"=>R.time_text(DateTime(2018)),"station"=>"SYN",
            "product"=>"synthetic_arrival_wind","neighbor_role"=>"primary","disposition"=>"absent","approval_sha256"=>repeat("2",64)))
    end
    split=Dict{String,Any}("name"=>"fit","issue_start_inclusive"=>R.time_text(DateTime(2017)),
        "issue_stop_exclusive"=>R.time_text(DateTime(2017,10,1)),"label_ready_before"=>R.time_text(DateTime(2017,10,1)),
        "target_admit_asof"=>R.time_text(DateTime(2017,10,1)),"min_supervised_rows"=>3)
    recipe=Dict{String,Any}("family"=>"ridge","role"=>(mode=="mixed8" ? "fixed_mixed8" : "candidate"),
        "penalty"=>"3ff0000000000000","intercept_penalty"=>mode=="mixed8",
        "std_rule"=>(mode=="mixed8" ? "sample_std_plus_1e_minus9" : "sample_std_zero_nonfinite_to_one"),
        "split"=>"fit","selection_decision_pin"=>nothing)
    Dict{String,Any}("schema"=>"dbdt_completed_refit_protocol_v1","evidence_mode"=>"synthetic_only",
        "protocol_id"=>"protocol-synthetic-"*mode,"station"=>"SYN","ground_product"=>"synthetic_ground",
        "wind_product_or_null"=>wind,"units"=>(mode=="mixed8" ? "ground_nT_rate_nT_per_min_wind_km_s_nT_cm3" : "ground_nT_rate_nT_per_min"),
        "layout"=>mode,"feature_names"=>collect(mode=="mixed8" ? R.MIXED_FEATURES : R.GROUND_FEATURES),"cohort_kind"=>"natural",
        "policy"=>Dict{String,Any}("ground_extra_ms"=>0,"omni_extra_ms"=>0,"label_extra_ms"=>0),"source_permissions"=>permissions,
        "neighbor_policy"=>"explicit_approved_only","splits"=>Any[split],"minima_mode"=>"synthetic_oracle",
        "model_recipes"=>Any[recipe],"selection_rule"=>"frozen_2017_mse_first_exact_tie_ridge_boost_forest_v1",
        "runtime_contract"=>"owned_julia_1_12_6_or_1_13_1_single_threaded_v1","numeric_contract"=>"directed_ridge_fit_v1_portable_prediction_v2",
        "availability_kind"=>"assumed_completed_source_intervals","measured_receipts_available"=>false)
end
function make_protocol(value)
    bytes=R.canonical_bytes(value;limit=R.MAX_METADATA_BYTES)
    R.RefitProtocol(bytes;expected_spec_sha256=R.digest(bytes))
end
