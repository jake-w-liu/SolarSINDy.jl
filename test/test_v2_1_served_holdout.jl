module V21ServedHoldoutTests

using Test
using CSV
using DataFrames
using Dates

const SERVED_HOLDOUT_SCRIPT = normpath(joinpath(
    @__DIR__, "..", "validation", "operational", "v2_1_served_holdout.jl",
))
include(SERVED_HOLDOUT_SCRIPT)

@testset "Complete-hour served-stack V2.1 chronological holdout" begin
    @test _selftest_v21_served_holdout()

    snapshot_dir = normpath(joinpath(@__DIR__, "..", "data", "operational_validation"))
    summary_path = joinpath(snapshot_dir, "v2_1_served_holdout_summary.csv")
    audit_path = joinpath(snapshot_dir, "v2_1_served_holdout_audit.csv")
    @test isfile(summary_path)
    @test isfile(audit_path)

    summary = CSV.read(summary_path, DataFrame)
    overall = summary[summary.cohort .== "overall", :]
    @test nrow(overall) == 1
    @test overall.n_rows[1] == 135_817
    @test overall.served_hits[1] == 117_575
    @test overall.served_coverage[1] == 117_575 / 135_817
    @test isapprox(overall.served_rmse_nt[1], 7.533793960278298; atol=1e-12)
    @test overall.frozen_tail_hits[1] == 116_890
    @test overall.frozen_tail_coverage[1] == 116_890 / 135_817
    @test overall.pooled_gate_applies[1]
    @test overall.pooled_gate_pass[1]

    lead_rows = summary[(summary.activity_regime .== "all") .& (summary.lead_h .> 0), :]
    by_lead = Dict(Int(r.lead_h) => r for r in eachrow(lead_rows))
    @test sort(collect(keys(by_lead))) == [1, 2, 3, 4, 6, 7]
    @test by_lead[1].n_rows == 22_639
    @test by_lead[2].n_rows == 22_638
    @test by_lead[3].n_rows == 22_637
    @test by_lead[4].n_rows == 22_636
    @test by_lead[6].n_rows == 22_634
    @test by_lead[7].n_rows == 22_633
    @test isapprox(by_lead[7].served_coverage, 0.837582291344497; atol=1e-15)

    storm = summary[summary.cohort .== "storm", :][1, :]
    @test storm.n_rows == 1_326
    @test storm.served_hits == 551
    @test storm.served_coverage == 551 / 1_326
    @test isapprox(storm.served_rmse_nt, 27.541780070063663; atol=1e-12)

    audit = CSV.read(audit_path, DataFrame)
    @test nrow(audit) == 1
    @test audit.model_version[1] == "v2.1"
    @test audit.candidate_count[1] == 20
    @test audit.active_count[1] == 11
    @test audit.strict_forecast_origin_separation[1]
    @test audit.interval_policy[1] == "static_conformal_shifted_to_complete_hour_served_center"
    @test audit.holdout_residual_updates[1] == 0
    @test audit.heldout_promotion_evidence[1]
    @test audit.served_pooled_gate_pass[1]
    @test audit.supported_model_steps[1] == "1;2;3;4;6;7"
    @test audit.supported_model_step_count[1] == 6
    @test audit.support_validation_complete[1]
    @test isapprox(
        audit.minimum_supported_step_coverage[1],
        by_lead[7].served_coverage;
        atol=1e-15,
    )
    @test audit.point_calibration_sha256[1] ==
          "8062496d5b1a37d4b567c56432edb3d829a89a16af26f555c883071270cfb6bd"
    @test audit.conformal_calibration_sha256[1] ==
          "dbccf99d6455b391ebb59f884c51b573971f7e17539ec2fdb24faca59b1ba692"
    @test audit.calibration_scored_sha256[1] ==
          "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a"
    @test audit.omni_sha256[1] ==
          "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"

    source = read(SERVED_HOLDOUT_SCRIPT, String)
    @test occursin("force_frozen=true", source)
    @test occursin("holdout_residual_updates=[0]", source)
    @test occursin("static_conformal_shifted_to_complete_hour_served_center", source)
end

end
