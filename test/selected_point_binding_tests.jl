# Written controls; native Julia execution remains pending resource admission.
module SelectedPointBindingTests
using Test,Dates,SHA,JSON3,SolarSINDy
const M=SolarSINDy.ArchiveSelectedPointBindings
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const L=SolarSINDy.ArchiveLockedMixed8Ridge
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const START=DateTime(2020)
digest(raw)=bytes2hex(sha256(raw))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
include(joinpath(@__DIR__,"archive_source_bound_fit","case_fixture.jl"))
include(joinpath(@__DIR__,"selected_point_bindings","fixture_support.jl"))
const FULL=selected_full_fixtures()
const LITERAL=JSON3.read(read(joinpath(@__DIR__,"selected_point_bindings","literal_metadata_oracles.json")),Dict{String,Any})

@testset "Independent full canonical and primitive-frame oracles" begin
    @test FULL["fixture_only"]===true && FULL["actual_sources_or_producers"]===false
    @test L.Recipe().sha256==FULL["L_recipe_sha256"]
    for kind in ("gamma_primary","fifo_clock31_sensitivity","R2")
        d=FULL["decisions"][kind];request=M.SelectionRequest(d["request"]);decision=M.ParsedSelectionDecision(d)
        @test digest(M.encode_request(request))==FULL["canonical_request_sha256"][kind]
        @test digest(M.encode_decision(decision))==FULL["canonical_decision_sha256"][kind]
        @test length(M.encode_request(request))==FULL["canonical_request_bytes"][kind]
        @test length(M.encode_decision(decision))==FULL["canonical_decision_bytes"][kind]
        @test M.encode_request(M.decode_request(M.encode_request(request);expected_sha256=request.sha256))==M.encode_request(request)
        @test M.encode_decision(M.decode_decision(M.encode_decision(decision);expected_sha256=decision.sha256))==M.encode_decision(decision)
        @test M._pure(d["request"],d["candidate_reports"])==d["pure_decisions"]
        need_snapshot=M.encode_request(request);source=deepcopy(d["request"]);owned=M.SelectionRequest(source)
        source["policy_sha256"]="0"^64
        @test M.encode_request(owned)==need_snapshot
    end
    primary=M.ParsedSelectionDecision(FULL["decisions"]["gamma_primary"])
    sensitivity=M.ParsedSelectionDecision(FULL["decisions"]["fifo_clock31_sensitivity"])
    @test M.PrimaryRateChoice(primary).decision===primary
    @test M.SensitivityRateChoice(sensitivity).decision===sensitivity
    @test_throws ArgumentError M.PrimaryRateChoice(sensitivity)
    @test_throws ArgumentError M.SensitivityRateChoice(primary)
end

@testset "Exact schema, evidence, grid, split and parsed-status rejection" begin
    original=FULL["decisions"]["gamma_primary"]
    for (field,bad) in (("phase_period_minutes",30),("phase_period_minutes",true),("nominal_bits",C.bits(.2)),
                       ("grid_sha256","0"^64),("layout","ground11"))
        request=deepcopy(original["request"]);request[field]=bad
        @test_throws ArgumentError M.SelectionRequest(request)
    end
    request=deepcopy(original["request"]);request["splits"]["validation"]["stop_exclusive"]="2025-01-01T00:00:00.000Z"
    @test_throws ArgumentError M.SelectionRequest(request)
    request=deepcopy(original["request"]);reverse!(request["producer_expectations"])
    @test_throws ArgumentError M.SelectionRequest(request)
    request=deepcopy(original["request"]);pop!(request["producer_expectations"])
    @test_throws ArgumentError M.SelectionRequest(request)
    request=deepcopy(original["request"]);request["cohort_expectations"][1]["normalization_bits"]=C.bits(0.)
    @test_throws ArgumentError M.SelectionRequest(request)
    request=deepcopy(original["request"]);request["producer_expectations"][1]["unexpected"]="pin"
    @test_throws ArgumentError M.SelectionRequest(request)
    for field in ("producer_evidence_verified","actual_authority_granted")
        d=deepcopy(original);d[field]=true
        @test_throws ArgumentError M.ParsedSelectionDecision(d)
    end
    for mutate in (d->pop!(d["candidate_reports"]),d->(d["candidate_reports"][2]=deepcopy(d["candidate_reports"][1])),
        d->(d["candidate_evidence"][1]["validation_payload_sha256"]="0"^64),
        d->(d["pure_decisions"]["phase"]["gamma_bits"]=C.bits(.1)),
        d->(d["origin_points"][1]["model_sha256"]="0"^64))
        d=deepcopy(original);mutate(d)
        @test_throws ArgumentError M.ParsedSelectionDecision(d)
    end
    raw=M.encode_decision(M.ParsedSelectionDecision(original))
    @test_throws ArgumentError M.decode_decision(raw;expected_sha256="0"^64)
    @test_throws ArgumentError M.decode_decision(raw[1:end-1];expected_sha256=digest(raw[1:end-1]))
    @test_throws ArgumentError M.decode_request(fill(UInt8('x'),M.MAX_REQUEST+1);expected_sha256="0"^64)
    for case in LITERAL["binding_cases"]
        if case["expected_metadata_binding"]
            @test M._validate_binding(case["value"])==case["value"]
        else
            @test_throws ArgumentError M._validate_binding(case["value"])
        end
    end
    # Actual selected support must not be advertised by a throwing placeholder.
    for name in (:fit_selected_archival,:calibrate_selected_archival,:replay_selected_archival,:ProducerCheckedSelection)
        @test !isdefined(M,name)
    end
end

@testset "Bounded publication and semantic rejection after outer rehash" begin
    mktempdir() do temporary
        root=realpath(temporary);decision=M.ParsedSelectionDecision(FULL["decisions"]["gamma_primary"])
        request=M.SelectionRequest(FULL["decisions"]["R2"]["request"])
        request_path=joinpath(root,"request");request_pins=M.publish_metadata(request,request_path)
        @test M.encode_request(M.read_request(request_path;expected=request_pins))==M.encode_request(request)
        path=joinpath(root,"decision");pins=M.publish_metadata(decision,path)
        @test M.encode_decision(M.read_decision(path;expected=pins))==M.encode_decision(decision)
        @test_throws ArgumentError M.publish_metadata(decision,path)
        @test_throws ArgumentError M.read_decision(path;expected=M.MetadataPins(pins.request_sha256,"0"^64,pins.receipt_sha256,pins.complete_sha256))
        changed=joinpath(root,"changed");cp(path,changed)
        payload=R.decode_canonical(read(joinpath(changed,"payload.json"));limit=M.MAX_DECISION)
        payload["pure_decisions"]["phase"]["gamma_bits"]=C.bits(.1)
        raw=R.canonical_bytes(payload;limit=M.MAX_DECISION);write(joinpath(changed,"payload.json"),raw)
        receipt=R.decode_canonical(read(joinpath(changed,"receipt.json"));limit=65536);receipt["payload_sha256"]=digest(raw)
        receipt_raw=R.canonical_bytes(receipt;limit=65536);write(joinpath(changed,"receipt.json"),receipt_raw)
        marker=R.decode_canonical(read(joinpath(changed,"COMPLETE.json"));limit=4096)
        marker["payload_sha256"]=digest(raw);marker["receipt_sha256"]=digest(receipt_raw)
        marker["bytes_without_marker"]=filesize(joinpath(changed,"request.json"))+length(raw)+length(receipt_raw)
        complete=R.canonical_bytes(marker;limit=4096);write(joinpath(changed,"COMPLETE.json"),complete)
        rehashed=M.MetadataPins(pins.request_sha256,digest(raw),digest(receipt_raw),digest(complete))
        @test_throws ArgumentError M.read_decision(changed;expected=rehashed)
    end
end

@testset "Generated zero-fit identity and explicit transfer bind served models" begin
    mktempdir() do temporary
        root=realpath(temporary);values=generated_selected_values(root)
        for (index,binding) in enumerate((values.fixed,values.bindings...))
            raw=M.encode_binding(binding);v=R.decode_canonical(raw;limit=M.MAX_BINDING)
            @test v["actual_authority_granted"]===false && v["producer_evidence_verified"]===false && v["binding_performed_new_fit"]===false
            @test_throws MethodError convert(F.BoundFit,binding)
            if index==1
                @test v["transfer_policy"]=="locked_mixed8_identity_v1"
            else
                @test v["transfer_policy"]=="primary_mixed8_to_ground11_v1"
                @test v["deployment_point"]["layout"]=="ground11"
                @test v["deployment_point"]["model_sha256"]!=v["gamma_origin"][1]["point"]["model_sha256"]
                @test v["primary_rate_choice"]["decision_sha256"]==values.gamma.sha256
            end
            path=joinpath(root,"binding-"*string(index));pins=M.publish_metadata(binding,path)
            restored=M.read_binding(path;expected=pins,point=binding.point,rates=values.rates,point_choice=binding.point_choice)
            @test M.encode_binding(restored)==raw
            tampered=deepcopy(v);tampered["actual_authority_granted"]=true;bad=R.canonical_bytes(tampered;limit=M.MAX_BINDING)
            @test_throws ArgumentError M.decode_binding(bad;expected_sha256=digest(bad),point=binding.point,rates=values.rates,point_choice=binding.point_choice)
        end
        @test_throws MethodError M.bind_fixed_point(values.points[1],M.SensitivityRateChoice(M.ParsedSelectionDecision(FULL["decisions"]["fifo_clock31_sensitivity"])))
        @test_throws MethodError M.bind_fixed_point(values.points[1],values.rates,()->error("no callbacks"))
        # Self-consistent untrusted metadata remains unable to bind a different
        # origin model to an outside checked F capsule.
        d=R.decode_canonical(M.encode_decision(values.gamma);limit=M.MAX_DECISION)
        d["origin_points"][1]["model_sha256"]="0"^64;d["origin_calibrations"][1]["point_model_sha256"]="0"^64
        for e in d["request"]["producer_expectations"];e["origin_station"]=="CMO" && (e["point_model_sha256"]="0"^64);end
        d["candidate_evidence"]=d["request"]["producer_expectations"]
        d["request_sha256"]=digest(R.canonical_bytes(d["request"];limit=M.MAX_REQUEST))
        forged=M.PrimaryRateChoice(M.ParsedSelectionDecision(d))
        @test_throws ArgumentError M.bind_fixed_point(values.points[1],forged)
        r2=R.decode_canonical(M.encode_decision(values.decision);limit=M.MAX_DECISION)
        r2["request"]["policy_sha256"]="0"^64;r2["request_sha256"]=digest(R.canonical_bytes(r2["request"];limit=M.MAX_REQUEST))
        @test_throws ArgumentError M.bind_generated_selected_point(values.bindings[1].point,values.rates,M.ParsedSelectionDecision(r2))
    end
end
end
