# Shared generated-only exercise for package tests and experiments.
function selected_full_fixtures()
    path=joinpath(pkgdir(SolarSINDy),"test","selected_point_bindings","full_codec_oracles.json")
    JSON3.read(read(path),Dict{String,Any})
end
function generated_gamma_decision(cases,points)
    d=deepcopy(selected_full_fixtures()["decisions"]["gamma_primary"]);request=d["request"]
    descriptors=[M._point_descriptor(point) for point in points]
    for (index,station) in enumerate(("CMO","FRD"))
        p=descriptors[index];case=request["source_case_pins"][index]
        case["request_sha256"]=cases.request_sha256;case["manifest_sha256"]=cases.manifest_sha256
        case["complete_sha256"]=cases.complete_sha256;case["table_index"]=index
        c=d["origin_calibrations"][index];c["point_context_sha256"]=p["context_sha256"];c["point_model_sha256"]=p["model_sha256"]
        for e in request["producer_expectations"]
            e["origin_station"]==station || continue
            e["case_request_sha256"]=cases.request_sha256;e["case_manifest_sha256"]=cases.manifest_sha256;e["case_complete_sha256"]=cases.complete_sha256
            for field in M.SHAS;e["point_"*field]=p[field];end
        end
    end
    request_value=M.SelectionRequest(request)
    M.ParsedSelectionDecision(request_value,d["candidate_reports"];origin_points=descriptors,origin_calibrations=d["origin_calibrations"])
end
function generated_point_decision(cases)
    d=deepcopy(selected_full_fixtures()["decisions"]["R2"]);request=d["request"]
    c=only(request["source_case_pins"]);c["request_sha256"]=cases.request_sha256;c["manifest_sha256"]=cases.manifest_sha256;c["complete_sha256"]=cases.complete_sha256
    for e in request["producer_expectations"]
        e["case_request_sha256"]=cases.request_sha256;e["case_manifest_sha256"]=cases.manifest_sha256;e["case_complete_sha256"]=cases.complete_sha256
    end
    M.ParsedSelectionDecision(M.SelectionRequest(request),d["candidate_reports"])
end
function generated_selected_values(root)
    mixed=joinpath(root,"mixed");mkdir(mixed);mixed_cases=make_cases(realpath(mixed),"primary","mixed")
    start=START+Minute(40);stop=START+Minute(90);asof=START+Minute(211)
    points=F.BoundFit[]
    for index in 1:2
        recipe=L.Recipe();expected=F.prepare_expected(mixed_cases,index,recipe;start,stop,asof)
        push!(points,F.fit_generated(mixed_cases,index,recipe,expected;start,stop,asof).capsule)
    end
    gamma=generated_gamma_decision(mixed_cases,points);rates=M.PrimaryRateChoice(gamma)
    fixed=M.bind_fixed_point(points[1],rates)
    ground=joinpath(root,"ground");mkdir(ground);ground_cases=make_cases(realpath(ground),"primary","ground-only")
    decision=generated_point_decision(ground_cases);bindings=M.SelectedPointBinding[]
    for candidate in ("ridge-0.01","boost-2-50","forest-6-64")
        recipe=M.recipe_for_candidate(candidate);expected=F.prepare_expected(ground_cases,1,recipe;start,stop,asof)
        point=F.fit_generated(ground_cases,1,recipe,expected;start,stop,asof).capsule
        push!(bindings,M.bind_generated_selected_point(point,rates,decision))
    end
    (;fixed,bindings,rates,gamma,decision,points,ground_cases)
end
