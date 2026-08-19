module V2ReadinessSelfTestTests

# The readiness audit carries its own self-test: fixture dashboard payloads, fixture live-log stage
# windows, regime guards, split/holdout contracts and verdict arithmetic. It went stale once — the
# fixture payload kept an older driver-assumption sentence while the audit had begun requiring the
# served stage to be disclosed, so `--self-test` exited 1 while a green package suite reported nothing.
# Running it here is what keeps the audit's own guard from regressing unnoticed.
#
# The script is loaded into a throwaway module: it defines two hundred top-level names and resolves
# evidence paths at include time, neither of which belongs in the test namespace.

using Test
using Dates

const AUDIT_PATH = normpath(joinpath(@__DIR__, "..", "validation", "operational",
                                     "v2_readiness_audit.jl"))

@testset "Readiness audit self-test" begin
    @test isfile(AUDIT_PATH)
    audit = Module(:ReadinessAuditSelfTestProbe)
    # A bare module has no single-argument `include`, and the audit includes its path helper that way.
    Base.eval(audit, :(include(path) = Base.include($audit, path)))
    Base.include(audit, AUDIT_PATH)

    # The self-test runs every fixture assertion and throws on the first failure, so a true return is
    # the whole contract. It must also stay non-trivial: a self-test that silently stopped exercising
    # its fixtures would still return true.
    passed = redirect_stdout(devnull) do
        audit.selftest_readiness_audit()
    end
    @test passed === true

    # A self-test that stopped exercising its fixtures would still return true, and part of its
    # fixture set is guarded on local artifacts that a fresh clone does not carry. Assert the count
    # against the artifact-independent floor so the difference is stated rather than silent.
    @test audit.SELFTEST_CHECK_COUNT[] >= audit.SELFTEST_MIN_CHECKS
    @test audit.SELFTEST_MIN_CHECKS >= 42

    # The audit requires the served stage to be disclosed in the dashboard payload. A fixture payload
    # that omits the stack clause is exactly the regression this file exists to catch, so assert the
    # requirement in both directions rather than trusting the fixture.
    served_state = audit.AuditState()
    served_state.live_metrics[:served_n] = 0
    served_state.live_metrics[:newest_cycle_served_label] = audit.EXPECTED_SUBHOURLY
    payload = Dict{String,Any}(
        "available" => true,
        "model_version" => audit.EXPECTED_MODEL_VERSION,
        "served_model_version" => audit.EXPECTED_SUBHOURLY,
        "generated_utc" => "2026-06-26T07:14:30Z",
        "forecast_issue_utc" => "2026-06-26T06:30:00Z",
        "latest_solar_wind_utc" => "2026-06-26T06:28:00Z",
        "lead_time" => Dict{String,Any}("driver_assumption" => "no pipeline description at all"),
        "calibration" => Dict{String,Any}("v2_n_verified" => 0, "v2_rmse_nt" => nothing,
                                          "audit_baseline_rmse_nt" => nothing),
    )
    audit.audit_dashboard_payload!(served_state, payload, "test://no-pipeline";
                                   now_utc = DateTime(2026, 6, 26, 7, 15, 0))
    @test any(c -> c.level == :fail && c.name == "dashboard API V2-tail assumption",
              served_state.checks)
end

end # module
