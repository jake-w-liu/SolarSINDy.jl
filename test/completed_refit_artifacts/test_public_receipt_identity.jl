@testset "public native and independent receipt arguments keep exact manifests" begin
    results=CompletedRefitArtifactsExample.run_example()
    for layout in (:ground11,:mixed8)
        result=results[layout];pins=result.pins;protocol=result.protocol
        primary_pin=R._member_json(result.point,"native-fit-receipt.json")["native_job_manifest_sha256"]
        check_pin=R._member_json(result.point,"independent-fit-check.json")["native_job_manifest_sha256"]
        primary=R._load_native_job(pins[primary_pin];protocol,pins,expected_operation="fit_ridge")
        checker=R._load_native_job(pins[check_pin];protocol,pins,expected_operation="fit_independent_check")
        draft=R._draft_from_job(primary,pins)
        reference=R._draft_member(draft,"fit-receipt.json")
        input=R._load_input(pins[reference["input_manifest_sha256"]];protocol,pins)
        input_receipt=R._member_json(input,"independent-input-check.json")
        parent=mktempdir(;prefix="ss-r1-public-pin-",cleanup=false)
        for (label,native,independent) in (("string",primary_pin,check_pin),("bundle",primary,checker))
            published=R.publish_point(joinpath(parent,"point-"*label),draft;native_receipt=native,
                independent_input_receipt=input_receipt,independent_fit_receipt=independent,pins)
            @test published.manifest_sha256==result.index["point_pin"]
        end
        wrong=repeat("f",64)
        @test wrong!=primary_pin && wrong!=check_pin
        for argument in (:native,:independent)
            badpins=copy(pins);badpins[wrong]=pins[argument===:native ? primary_pin : check_pin]
            destination=joinpath(parent,"point-bad-"*String(argument))
            failure=try
                R.publish_point(destination,draft;native_receipt=argument===:native ? wrong : primary_pin,
                    independent_input_receipt=input_receipt,independent_fit_receipt=argument===:independent ? wrong : check_pin,pins=badpins)
                nothing
            catch error
                error
            end
            @test failure isa ArgumentError
            @test failure!==nothing && occursin("native receipt manifest argument differs",sprint(showerror,failure))
            @test !ispath(destination)
        end
        # The actual-fit repeat is another genuine job with the same operation
        # and payload. It discriminates the Bundle branch from role/schema errors.
        native_directory=dirname(dirname(pins[primary_pin]))
        terminal=R.decode_canonical(R.bounded_read(joinpath(native_directory,"TERMINAL.json")))
        repeated=only(s for s in terminal["stages"] if s["kind"]=="native" && endswith(dirname(s["directory"]),"024-publish-ridge-repeat"))
        repeated_pin=repeated["manifest_sha256"]
        @test repeated_pin!=primary_pin
        badpins=copy(pins);badpins[primary_pin]=pins[repeated_pin]
        failure=try
            R.publish_point(joinpath(parent,"point-bad-bundle"),draft;native_receipt=primary,
                independent_input_receipt=input_receipt,independent_fit_receipt=checker,pins=badpins)
            nothing
        catch error
            error
        end
        @test failure isa ArgumentError
        @test failure!==nothing && occursin("native receipt manifest argument differs",sprint(showerror,failure))
        @test !ispath(joinpath(parent,"point-bad-bundle"))

        cal_pin=R._member_json(result.cal,"native-cal-receipt.json")["native_job_manifest_sha256"]
        cal_check_pin=R._member_json(result.cal,"independent-cal-check.json")["native_job_manifest_sha256"]
        cal_primary=R._load_native_job(pins[cal_pin];protocol,pins,expected_operation="calibrate")
        cal_checker=R._load_native_job(pins[cal_check_pin];protocol,pins,expected_operation="cal_independent_check")
        cal_draft=R._draft_from_job(cal_primary,pins)
        for (label,native,independent) in (("string",cal_pin,cal_check_pin),("bundle",cal_primary,cal_checker))
            published=R.publish_calibration(joinpath(parent,"cal-"*label),cal_draft;native_receipt=native,independent_cal_receipt=independent,pins)
            @test published.manifest_sha256==result.index["calibration_pin"]
        end
        for argument in (:native,:independent)
            badpins=copy(pins);badpins[wrong]=pins[argument===:native ? cal_pin : cal_check_pin]
            destination=joinpath(parent,"cal-bad-"*String(argument))
            failure=try
                R.publish_calibration(destination,cal_draft;native_receipt=argument===:native ? wrong : cal_pin,
                    independent_cal_receipt=argument===:independent ? wrong : cal_check_pin,pins=badpins)
                nothing
            catch error
                error
            end
            @test failure isa ArgumentError
            @test failure!==nothing && occursin("native receipt manifest argument differs",sprint(showerror,failure))
            @test !ispath(destination)
        end
    end
end
