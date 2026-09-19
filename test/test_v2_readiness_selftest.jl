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
using DataFrames

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

    @testset "external scores require completed receipt before target" begin
        # Hand-set errors 100, 3, and 4: only 3 and 4 have valid prospective receipts.
        external = DataFrame(
            source=fill("fixture", 4), issue_utc=fill("2026-09-07T09:00:00Z", 4),
            fetched_utc=fill("2026-09-07T09:59:00Z", 4),
            target_utc=["2026-09-07T10:00:00Z", "2026-09-07T11:00:00Z",
                        "2026-09-07T12:00:00Z", "2026-09-07T13:00:00Z"],
            receipt_completed_utc=Union{Missing,String}[
                "2026-09-07T10:30:00Z", "2026-09-07T10:30:00Z",
                "2026-09-07T10:30:00Z", missing],
            lead_h=[1.0, 2.0, 3.0, 4.0], forecast_dst_nt=[100.0, 3.0, -4.0, 1000.0],
            observed_dst_nt=[0.0, 0.0, 0.0, 0.0], abs_error_nt=[100.0, 3.0, 4.0, 1000.0],
            forecast_cadence_min=fill(60.0, 4), issue_basis=fill("http_last_modified", 4),
            source_url=fill("fixture", 4), raw_sha256=fill(repeat("a", 64), 4),
            raw_path=fill("fixture.raw", 4), source_max_target_utc=fill("2026-09-07T13:00:00Z", 4),
            row_role=fill("future_forecast", 4),
            observed_time_utc=["2026-09-07T10:00:00Z", "2026-09-07T11:00:00Z",
                              "2026-09-07T12:00:00Z", "2026-09-07T13:00:00Z"],
            observed_gap_min=zeros(4), scored_utc=fill("2026-09-07T14:00:00Z", 4),
        )
        summary = only(eachrow(audit._external_dst_summary_from_log(external)))
        @test (summary.n_rows, summary.n_scored, summary.n_eligible,
               summary.n_late, summary.n_legacy, summary.n_invalid) == (4, 2, 2, 1, 1, 0)
        @test summary.rmse_nt ≈ sqrt(25 / 2) rtol=2eps(Float64)
        @test summary.mae_nt == 3.5
        @test summary.max_receipt_lead_h == 1.5
        legacy = select(external, Not(:receipt_completed_utc))
        legacy_summary = only(eachrow(audit._external_dst_summary_from_log(legacy)))
        @test legacy_summary.n_scored == 0
        @test legacy_summary.n_legacy == 4
        @test ismissing(legacy_summary.rmse_nt)
        mktempdir() do dir
            log_path, report_path = joinpath(dir, "external.csv"), joinpath(dir, "audit.md")
            audit.CSV.write(log_path, external)
            state = audit.AuditState()
            audit.audit_external_dst_snapshots!(state; path=log_path, report=report_path)
            @test any(c -> c.name == "external Dst timing exclusions" &&
                           c.level == :warn, state.checks)
            @test any(c -> c.name == "external Dst prospective score provenance" &&
                           c.level == :fail, state.checks)
            @test only(state.external_dst_metrics.n_scored) == 2
            audit.write_report(state, report_path)
            report = read(report_path, String)
            @test occursin("Unknown completion", report)
            @test occursin("strictly before target", report)
            @test occursin("| fixture | 4 | 2 | 2 | 1.500 | 3.54 | 3.50 |", report)
            external.receipt_completed_utc[2] = "2026-09-07T09:58:00Z"
            audit.CSV.write(log_path, external)
            invalid = audit.AuditState()
            audit.audit_external_dst_snapshots!(invalid; path=log_path, report=report_path)
            @test any(c -> c.name == "external Dst receipt chronology" &&
                           c.level == :fail, invalid.checks)
            @test only(invalid.external_dst_metrics.n_scored) == 1
            @test only(invalid.external_dst_metrics.rmse_nt) == 4.0
        end
    end

    # Kyoto can remain on the same Dst anchor across consecutive issue hours while new L1
    # measurements arrive. Those are distinct forecasts. A repeated row inside one issue hour is
    # still a duplicate, matching the live append key exactly.
    pending = DataFrame(
        model_version=fill("v2.1", 2),
        issue_time_utc=[DateTime(2026, 8, 24, 12, 48), DateTime(2026, 8, 24, 13, 19)],
        latest_dst_time_utc=fill(DateTime(2026, 8, 24, 12), 2),
        target_time_utc=fill(DateTime(2026, 8, 24, 14), 2),
    )
    @test nrow(audit.pending_duplicate_groups(pending)) == 0
    pending.issue_time_utc[2] = DateTime(2026, 8, 24, 12, 59)
    @test nrow(audit.pending_duplicate_groups(pending)) == 1

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

    # Reader-facing readiness output must identify the effective product. V2.1 remains valid
    # predecessor evidence, but it must not be presented as the current operational method.
    source = read(AUDIT_PATH, String)
    stale_component_count = occursin("one of its six components", source)
    scope_boundary = occursin("not held-out evidence for a later served ensemble", source)
    @test !stale_component_count
    @test scope_boundary
    mktempdir() do dir
        report = joinpath(dir, "readiness.md")
        audit.write_report(audit.AuditState(), report)
        text = read(report, String)
        @test occursin("# Operational V2.4e Readiness Audit", text)
        @test occursin("exact Operational V2.4e bundle", text)
        @test !occursin("# Operational V2.1 Readiness Audit", text)
        @test !occursin("recomputes Operational V2.1 readiness", text)
    end
end

end # module
