include(joinpath(@__DIR__,"..","..","examples","completed_refit_artifacts.jl"))
@testset "genuine native synthetic point and complete chronology fixtures" begin
    @test CompletedRefitArtifactsExample.CompletedRefitNativeDriver.run_support_tests()===nothing
    fixture=joinpath(@__DIR__,"fixtures")
    oracle_path=joinpath(fixture,"nonzero_end_to_end_expectations_v2.json")
    source_path=joinpath(fixture,"nonzero_source_cohorts_v2.json")
    @test R.digest(read(oracle_path))=="233134896ec285a61c400e4befae590dcce2c8ba1c04276378dbcf60a5ba0271"
    @test R.digest(read(source_path))=="f4817c8f94cf318d08f3088ba0965f6f5439ec52b52acc7ce751ccb8ef1956f7"
    oracle=R.JSON3.read(read(oracle_path,String),Any)
    sources=R.JSON3.read(read(source_path,String),Any)
    windows=sort(sources["source_rows"];by=w->DateTime(w["issue"]))
    bits_float(text)=reinterpret(Float64,parse(UInt64,text;base=16))
    results=CompletedRefitArtifactsExample.run_example()
    @test length(windows)==20
    for layout in (:ground11,:mixed8)
        result=results[layout];expected=oracle["layouts"][String(layout)]
        @test length(result.predictions)==20
        for (i,window) in enumerate(windows)
            key=window["fixture_row"]
            category,ordinal=split(key,'-');position=parse(Int,ordinal)
            name=category=="fit" ? "fit_predictions_decimal90" : category=="cal" ? "cal_predictions_decimal90" : "new_prediction_centers_decimal90"
            @test result.predictions[i].accepted
            @test isapprox(result.predictions[i].prediction,parse(Float64,expected[name][position]);atol=1e-13,rtol=0)
            @test isapprox(result.predictions[i].scale,bits_float(oracle["feature_expectations"][key]["scale_bits"]);atol=1e-13,rtol=0)
            @test result.predictions[i].point_manifest_sha256==result.index["point_pin"]
        end
        scores=result.bridge.calibration.chronological_scores
        sorted=result.bridge.calibration.sorted_scores
        @test length(scores)==length(sorted)==9
        for i in 1:9
            @test isapprox(scores[i],parse(Float64,expected["cal_scores_chronological_decimal90"][i]);atol=1e-13,rtol=0)
            @test isapprox(sorted[i],parse(Float64,expected["cal_scores_sorted_decimal90"][i]);atol=1e-13,rtol=0)
        end
        for bound in expected["finite_rank_bounds"]
            alpha=bound["alpha"]=="1/10" ? 0.1 : 0.2
            rank=ceil(Int,(length(scores)+1)*(1-alpha))
            @test rank==bound["rank"]
            @test isapprox(sorted[rank],parse(Float64,bound["score_decimal90"]);atol=1e-13,rtol=0)
            position=only(findall(w->w["fixture_row"]==bound["prediction_row"],windows))
            predicted=result.predictions[position]
            upper=SolarSINDy.CompletedSourceDBDT.upper_bound(alpha,predicted.prediction,predicted.scale,sorted)
            @test isapprox(upper,parse(Float64,bound["upper_decimal90"]);atol=1e-13,rtol=0)
        end
        first_issue=DateTime(2017,11,1)
        @test result.bridge.calibration.purpose===:synthetic
        @test !result.bridge.source_bound_provenance_verified
        @test result.bridge.artifact_pin==result.index["calibration_pin"]
        @test_throws ArgumentError R.completed_calibration(result.cal;first_issue=first_issue-Millisecond(1))
        @test_throws ArgumentError R.load_calibration(result.pins[result.index["calibration_pin"]];point=result.point,pins=result.pins,first_issue=first_issue-Millisecond(1))
        wrong_role=layout===:ground11 ? :fixed_mixed8 : :candidate
        @test_throws ArgumentError R.load_point(result.pins[result.index["point_pin"]];pins=result.pins,expected_role=wrong_role)
        @test_throws ArgumentError R.load_point(result.pins[result.index["point_pin"]];pins=Dict{String,String}(),expected_role=layout===:ground11 ? :candidate : :fixed_mixed8)
        independent=R.completed_calibration(result.cal;first_issue)
        @test independent.calibration.chronological_scores!==scores
        original=first(scores);scores[1]=original+1
        @test first(R.completed_calibration(result.cal;first_issue).calibration.chronological_scores)==original
        scores[1]=original
    end
end
