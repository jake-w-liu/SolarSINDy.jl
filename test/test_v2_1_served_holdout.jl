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
    @test overall.n_rows[1] == 90_400
    @test overall.served_hits[1] == 78_766
    @test overall.served_coverage[1] == 78_766 / 90_400
    @test isapprox(overall.served_rmse_nt[1], 6.6380611116770964; atol=1e-12)
    @test overall.frozen_tail_hits[1] == 78_250
    @test overall.frozen_tail_coverage[1] == 78_250 / 90_400
    @test overall.pooled_gate_applies[1]
    @test overall.pooled_gate_pass[1]

    by_lead = Dict(Int(r.lead_h) => r for r in eachrow(summary[summary.activity_regime .== "all", :]))
    @test by_lead[1].n_rows == 22_602
    @test by_lead[2].n_rows == 22_601
    @test by_lead[3].n_rows == 22_600
    @test by_lead[6].n_rows == 22_597
    @test isapprox(by_lead[6].served_coverage, 0.8486967296543789; atol=1e-15)

    storm = summary[summary.cohort .== "storm", :][1, :]
    @test storm.n_rows == 884
    @test storm.served_hits == 440
    @test storm.served_coverage == 440 / 884
    @test isapprox(storm.served_rmse_nt, 23.293227846077198; atol=1e-12)

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
    @test audit.point_calibration_sha256[1] ==
          "2bae4a153428be3fbdc77fbd27424986ea0dcf04ce5735581063b01a9bd3213e"
    @test audit.conformal_calibration_sha256[1] ==
          "b5d40ebdce9f42555a5c8872b247d9e64bcef4b35ab9f7b9bc27710dd812d2bb"
    @test audit.calibration_scored_sha256[1] ==
          "917263e818cfad8faaedd4eef461f035027fa8b8015b50af381c3b0eace48090"
    @test audit.omni_sha256[1] ==
          "5b9f068431fe3d5f4406360cd8176f6631d03d28417c99e0117e1058400fdb97"

    source = read(SERVED_HOLDOUT_SCRIPT, String)
    @test occursin("force_frozen=true", source)
    @test occursin("holdout_residual_updates=[0]", source)
    @test occursin("static_conformal_shifted_to_complete_hour_served_center", source)
end

end
