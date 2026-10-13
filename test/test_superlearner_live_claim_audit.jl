using Test
using CSV
using DataFrames
using Dates
using JSON3
using SHA

isdefined(Main, :SuperlearnerLiveClaimAudit) ||
    include(joinpath(@__DIR__, "..", "validation", "operational",
                     "superlearner_live_claim_audit.jl"))

function _claim_audit_fixture(; digest::String=SuperlearnerLiveClaimAudit.SHADOW_CONFIG_SHA256,
                              statuses=["ok", "ok", "warmup:16/30", "warmup:13/30"])
    issue = DateTime(2026, 8, 25, 0, 30)
    public_horizons = [1, 2, 3, 6]
    targets = floor(issue, Hour) .+ Hour.(public_horizons)
    steps = [1, 2, 3, 6]
    point = [-20.0, -21.0, -22.0, -23.0]
    observations = Union{Missing,Float64}[-20.0, -22.0, missing, missing]
    static_lo = point .- 4.0
    static_hi = point .+ 4.0
    shadow_lo = Union{Missing,Float64}[]
    shadow_hi = Union{Missing,Float64}[]
    shadow_history = Union{Missing,Int}[]
    shadow_location = Union{Missing,Float64}[]
    for (index, status) in enumerate(statuses)
        if status == "ok"
            scale = steps[index] == 1 ? 1.50 : 1.20
            push!(shadow_lo, point[index] - scale * (point[index] - static_lo[index]))
            push!(shadow_hi, point[index] + scale * (static_hi[index] - point[index]))
            push!(shadow_history, 30)
            push!(shadow_location, 0.0)
        else
            matched = match(r"^warmup:(\d+)/30$", status)
            push!(shadow_lo, missing)
            push!(shadow_hi, missing)
            push!(shadow_history, matched === nothing ? missing : parse(Int, matched.captures[1]))
            push!(shadow_location, missing)
        end
    end
    return DataFrame(
        issue_time_utc=fill(issue, 4),
        latest_dst_time_utc=fill(floor(issue, Hour), 4),
        target_time_utc=targets,
        horizon_hours=[(target - issue) / Hour(1) for target in targets],
        model_step_hours=steps,
        observation_dst_nt=observations,
        served_pred_dst_nt=point,
        served_pred_dst_ci05_nt=static_lo,
        served_pred_dst_ci95_nt=static_hi,
        sub_hourly_pred_dst_nt=point,
        sub_hourly_pred_dst_ci05_nt=static_lo,
        sub_hourly_pred_dst_ci95_nt=static_hi,
        sub_hourly_model_version=fill(SuperlearnerLiveClaimAudit.SERVED_IDENTITY, 4),
        superlearner_manifest_sha256=fill(SuperlearnerLiveClaimAudit.SERVED_MANIFEST_SHA256, 4),
        superlearner_status=fill("ok", 4),
        superlearner_pred_dst_nt=point,
        superlearner_ci05_nt=static_lo,
        superlearner_ci95_nt=static_hi,
        superlearner_cal_shadow_model_version=fill(SuperlearnerLiveClaimAudit.SHADOW_IDENTITY, 4),
        superlearner_cal_shadow_config_sha256=fill(digest, 4),
        superlearner_cal_shadow_status=statuses,
        superlearner_cal_shadow_ci05_nt=shadow_lo,
        superlearner_cal_shadow_ci95_nt=shadow_hi,
        superlearner_cal_shadow_history_n=shadow_history,
        superlearner_cal_shadow_location_shift_nt=shadow_location,
        static_stack_pred_dst_nt=point .+ 2.0,
        analog_shadow_pred_dst_nt=point .+ 1.5,
        direct_gbm_pred_dst_nt=point .+ 1.0,
        inertial_sindy_served_pred_dst_nt=point .+ 2.5,
        persistence_dst_nt=point .+ 3.0,
        burton_dst_nt=point .+ 3.5,
        burton_full_dst_nt=point .+ 4.0,
        obrien_dst_nt=point .+ 4.5,
    )
end

function _claim_ready_fixture()
    base = DateTime(2026, 8, 25)
    supported_steps = collect(SuperlearnerLiveClaimAudit.SUPPORTED_MODEL_STEPS)
    step_rows = Dict(step => 0 for step in supported_steps)
    event_starts = [24, 124, 224, 324, 424]
    records = NamedTuple[]
    row_index = 0
    for day_offset in 0:39, issue_hour in 0:19
        issue = base + Day(day_offset) + Hour(issue_hour) + Minute(30)
        anchor = floor(issue, Hour) - Hour(isodd(issue_hour))
        for public_horizon in (1, 2, 3, 6)
            row_index += 1
            target = floor(issue, Hour) + Hour(public_horizon)
            step = Int((target - anchor) / Hour(1))
            step_rows[step] += 1
            miss = step_rows[step] % 10 == 0
            target_hour = round(Int, (target - base) / Hour(1))
            storm = any(start <= target_hour < start + 21 for start in event_starts)
            observation = storm ? -60.0 : -10.0
            point = observation + (miss ? 30.1 : 0.0)
            static_half_width = miss ? 20.0 : 1.0
            scale = step == 1 ? 1.50 : 1.20
            push!(records, (
                issue_time_utc=issue,
                latest_dst_time_utc=anchor,
                target_time_utc=target,
                horizon_hours=(target - issue) / Hour(1),
                model_step_hours=step,
                observation_dst_nt=observation,
                served_pred_dst_nt=point,
                served_pred_dst_ci05_nt=point - static_half_width,
                served_pred_dst_ci95_nt=point + static_half_width,
                sub_hourly_pred_dst_nt=point,
                sub_hourly_pred_dst_ci05_nt=point - static_half_width,
                sub_hourly_pred_dst_ci95_nt=point + static_half_width,
                sub_hourly_model_version=SuperlearnerLiveClaimAudit.SERVED_IDENTITY,
                superlearner_manifest_sha256=SuperlearnerLiveClaimAudit.SERVED_MANIFEST_SHA256,
                superlearner_status="ok",
                superlearner_pred_dst_nt=point,
                superlearner_ci05_nt=point - static_half_width,
                superlearner_ci95_nt=point + static_half_width,
                superlearner_cal_shadow_model_version=SuperlearnerLiveClaimAudit.SHADOW_IDENTITY,
                superlearner_cal_shadow_config_sha256=SuperlearnerLiveClaimAudit.SHADOW_CONFIG_SHA256,
                superlearner_cal_shadow_status="ok",
                superlearner_cal_shadow_ci05_nt=point - static_half_width * scale,
                superlearner_cal_shadow_ci95_nt=point + static_half_width * scale,
                superlearner_cal_shadow_history_n=100,
                superlearner_cal_shadow_location_shift_nt=0.0,
                static_stack_pred_dst_nt=observation + 20.0,
                analog_shadow_pred_dst_nt=observation + 12.0,
                direct_gbm_pred_dst_nt=observation + 14.0,
                inertial_sindy_served_pred_dst_nt=observation + 14.0,
                persistence_dst_nt=observation + 16.0,
                burton_dst_nt=observation + 18.0,
                burton_full_dst_nt=observation + 19.0,
                obrien_dst_nt=observation + 17.0,
            ))
        end
    end
    return DataFrame(records)
end

@testset "super-learner prospective claim audit canonicalization" begin
    fixture = _claim_audit_fixture()
    canonical = SuperlearnerLiveClaimAudit.canonical_shadow_rows(fixture)
    @test nrow(canonical.rows) == 4
    @test canonical.deployment_rows == 4
    @test canonical.warmup_rows == 2
    @test canonical.unexpected_status_rows == 0
    @test isempty(canonical.violations)
    @test canonical.rows.horizon_hours == [1, 2, 3, 6]

    status = SuperlearnerLiveClaimAudit.prospective_status(fixture; bootstrap_reps=100)
    @test status.integrity.gate_pass
    @test status.marginal.n_rows == 2
    @test status.marginal.complete_issue_cycles == 0
    @test !status.marginal.claim_ready
    @test !status.storm.claim_ready

    wrong_digest = SuperlearnerLiveClaimAudit.canonical_shadow_rows(
        _claim_audit_fixture(; digest="0"^64),
    )
    @test nrow(wrong_digest.rows) == 0
    @test length(wrong_digest.violations) == 4

    wrong_manifest = _claim_audit_fixture()
    wrong_manifest.superlearner_manifest_sha256[1] = "0"^64
    manifest = SuperlearnerLiveClaimAudit.canonical_shadow_rows(wrong_manifest)
    @test any(value -> occursin("served-manifest", value), manifest.violations)

    changed_served_value = _claim_audit_fixture()
    changed_served_value.superlearner_pred_dst_nt[1] += 1.0
    served_value = SuperlearnerLiveClaimAudit.canonical_shadow_rows(changed_served_value)
    @test any(value -> occursin("preserve the served", value), served_value.violations)

    changed_shadow_rule = _claim_audit_fixture()
    changed_shadow_rule.superlearner_cal_shadow_ci05_nt[1] -= 1.0
    shadow_rule = SuperlearnerLiveClaimAudit.canonical_shadow_rows(changed_shadow_rule)
    @test any(value -> occursin("frozen A3", value), shadow_rule.violations)

    nonfuture = _claim_audit_fixture()
    nonfuture.target_time_utc[1] = nonfuture.latest_dst_time_utc[1]
    chronology = SuperlearnerLiveClaimAudit.canonical_shadow_rows(nonfuture)
    @test any(value -> occursin("chronology", value), chronology.violations)

    leaked_anchor = _claim_audit_fixture()
    leaked_anchor.latest_dst_time_utc[1] = leaked_anchor.issue_time_utc[1] + Minute(1)
    leaked = SuperlearnerLiveClaimAudit.canonical_shadow_rows(leaked_anchor)
    @test any(value -> occursin("chronology", value), leaked.violations)

    wrong_wall_lead = _claim_audit_fixture()
    wrong_wall_lead.horizon_hours[1] += 0.25
    bad_horizon = SuperlearnerLiveClaimAudit.canonical_shadow_rows(wrong_wall_lead)
    @test any(value -> occursin("public horizon", value), bad_horizon.violations)

    boolean_step = _claim_audit_fixture()
    boolean_step[!, :model_step_hours] = Any[true, 2, 3, 6]
    malformed = SuperlearnerLiveClaimAudit.canonical_shadow_rows(boolean_step)
    @test any(value -> occursin("malformed", value), malformed.violations)

    wrong_step = _claim_audit_fixture()
    wrong_step.model_step_hours[2] = 3
    @test any(value -> occursin("Dst-anchor geometry", value),
              SuperlearnerLiveClaimAudit.canonical_shadow_rows(wrong_step).violations)

    for step in (2.5, 1.0e100)
        unsupported = _claim_audit_fixture()
        unsupported[!, :model_step_hours] = Float64.(unsupported.model_step_hours)
        unsupported.model_step_hours[2] = step
        @test any(value -> occursin("unsupported internal", value),
                  SuperlearnerLiveClaimAudit.canonical_shadow_rows(unsupported).violations)
    end

    old_anchor = _claim_audit_fixture()
    old_anchor.latest_dst_time_utc[2] -= Hour(2)
    old_anchor.model_step_hours[2] = 4
    @test any(value -> occursin("Dst-anchor geometry", value),
              SuperlearnerLiveClaimAudit.canonical_shadow_rows(old_anchor).violations)

    before_deployment = _claim_audit_fixture()
    for column in (:issue_time_utc, :latest_dst_time_utc, :target_time_utc)
        before_deployment[!, column] .-= Day(1)
    end
    @test length(SuperlearnerLiveClaimAudit.canonical_shadow_rows(before_deployment).violations) == 4

    for observation in (NaN, Inf, -Inf, true, "invalid")
        malformed_observation = _claim_audit_fixture()
        malformed_observation[!, :observation_dst_nt] = Any[-20.0, observation, missing, missing]
        checked = SuperlearnerLiveClaimAudit.prospective_status(malformed_observation; bootstrap_reps=100)
        @test !checked.integrity.gate_pass
        @test any(value -> occursin("malformed observation", value), checked.integrity.violations)
    end

    warmup_endpoints = _claim_audit_fixture()
    warmup_endpoints.superlearner_cal_shadow_ci05_nt[3] = -30.0
    warmup_endpoints.superlearner_cal_shadow_ci95_nt[3] = -10.0
    invalid_warmup = SuperlearnerLiveClaimAudit.canonical_shadow_rows(warmup_endpoints)
    @test any(value -> occursin("warm-up", value), invalid_warmup.violations)

    oversized_warmup = _claim_audit_fixture()
    oversized_warmup.superlearner_cal_shadow_status[3] = "warmup:" * "9"^100 * "/30"
    @test any(value -> occursin("warm-up", value),
              SuperlearnerLiveClaimAudit.canonical_shadow_rows(oversized_warmup).violations)

    exact_duplicate = vcat(fixture, fixture[1:1, :])
    deduplicated = SuperlearnerLiveClaimAudit.canonical_shadow_rows(exact_duplicate)
    @test nrow(deduplicated.rows) == 4
    @test isempty(deduplicated.violations)

    # Missing outcomes and warm-up endpoints are equal to themselves when retries repeat them.
    for index in (3, 4)
        repeated_pending = SuperlearnerLiveClaimAudit.canonical_shadow_rows(vcat(fixture, fixture[index:index, :]))
        @test nrow(repeated_pending.rows) == 4
        @test isempty(repeated_pending.violations)
    end

    conflicting_row = copy(fixture[1:1, :])
    conflicting_row.static_stack_pred_dst_nt[1] -= 1.0
    conflict = SuperlearnerLiveClaimAudit.canonical_shadow_rows(vcat(fixture, conflicting_row))
    @test any(value -> occursin("conflicting exact-time", value), conflict.violations)

    for (column, value) in ((:issue_time_utc, 123), (:superlearner_cal_shadow_model_version, 123),
                            (:superlearner_cal_shadow_history_n, 1.0e100))
        malformed_field = _claim_audit_fixture()
        malformed_field[!, column] = Any[malformed_field[index, column] for index in 1:4]
        malformed_field[1, column] = value
        checked = SuperlearnerLiveClaimAudit.prospective_status(malformed_field; bootstrap_reps=100)
        @test !checked.integrity.gate_pass
        @test !isempty(checked.integrity.violations)
    end

    overlapping = _claim_audit_fixture(; statuses=fill("ok", 4))
    next_issue = copy(overlapping)
    for column in (:issue_time_utc, :latest_dst_time_utc, :target_time_utc)
        next_issue[!, column] .+= Hour(1)
    end
    next_issue.observation_dst_nt[1] = -30.0
    contradictory = vcat(overlapping, next_issue)
    checked = SuperlearnerLiveClaimAudit.prospective_status(contradictory; bootstrap_reps=100)
    @test !checked.integrity.gate_pass
    @test any(value -> occursin("conflicting observations", value), checked.integrity.violations)
    @test !(overlapping.target_time_utc[2] in
            SuperlearnerLiveClaimAudit.canonical_shadow_rows(contradictory).rows.target_time_utc)
end

@testset "super-learner storm events require an observed 72-hour quiet separator" begin
    base = DateTime(2026, 1, 1)
    targets = collect(base:Hour(1):(base + Hour(73)))
    observations = fill(-10.0, length(targets))
    observations[1] = -60.0
    observations[end] = -70.0
    grouped = SuperlearnerLiveClaimAudit.group_storm_events(targets, observations)
    @test grouped[targets[1]] == 1
    @test grouped[targets[end]] == 2

    gapped_targets = vcat(targets[1:36], targets[38:end])
    gapped_observations = vcat(observations[1:36], observations[38:end])
    gapped = SuperlearnerLiveClaimAudit.group_storm_events(gapped_targets, gapped_observations)
    @test gapped[gapped_targets[1]] == 1
    @test gapped[gapped_targets[end]] == 1
end

@testset "super-learner claim inference is deterministic and persisted" begin
    values = [1.0, 0.0, 1.0, 1.0]
    days = [Date(2026, 1, 1), Date(2026, 1, 1),
            Date(2026, 1, 2), Date(2026, 1, 2)]
    first_ci = SuperlearnerLiveClaimAudit._bootstrap_mean(values, days, 200, 42)
    second_ci = SuperlearnerLiveClaimAudit._bootstrap_mean(values, days, 200, 42)
    @test first_ci == second_ci

    mktempdir() do dir
        log_path = joinpath(dir, "live.csv")
        json_path = joinpath(dir, "status.json")
        report_path = joinpath(dir, "status.md")
        CSV.write(log_path, _claim_audit_fixture())
        status = SuperlearnerLiveClaimAudit.run_claim_audit(
            log_path, json_path, report_path; bootstrap_reps=100,
        )
        @test isfile(json_path)
        @test isfile(report_path)
        parsed = JSON3.read(read(json_path, String))
        @test parsed.shadow_config_sha256 == SuperlearnerLiveClaimAudit.SHADOW_CONFIG_SHA256
        @test endswith(String(parsed.generated_utc), "Z")
        @test !parsed.marginal.claim_ready
        @test occursin("NOT YET JUSTIFIED", read(report_path, String))
        @test !status.storm.claim_ready
    end
end

@testset "super-learner claims require every frozen gate" begin
    status = SuperlearnerLiveClaimAudit.prospective_status(
        _claim_ready_fixture(); bootstrap_reps=200,
    )
    @test status.integrity.gate_pass
    # Forty days × twenty cycles × four horizons; the two anchor lags have equal support.
    @test status.marginal.n_rows == 3_200
    @test status.marginal.complete_issue_cycles == 800
    @test status.marginal.consecutive_days == 40
    @test status.marginal.coverage_90 == 0.90
    @test status.marginal.claim_ready
    @test all(row -> row.n == (row.model_step_hours in (2, 3) ? 800 : 400) && row.gate_pass,
              status.marginal.by_step)
    @test status.storm.n_matched_rows >= 200
    @test status.storm.n_independent_events == 5
    @test all(row -> row.n >= 30 && row.n_events == 5 && row.gate_pass,
              status.storm.by_step)
    @test status.storm.claim_ready

    # Thirty storm rows at a step cannot substitute for the five independent events.
    sparse_step_events = _claim_ready_fixture()
    canonical = SuperlearnerLiveClaimAudit.canonical_shadow_rows(sparse_step_events).rows
    event_map = SuperlearnerLiveClaimAudit.group_storm_events(
        collect(canonical.target_time_utc[.!ismissing.(canonical.observation_dst_nt)]),
        Float64.(canonical.observation_dst_nt[.!ismissing.(canonical.observation_dst_nt)]),
    )
    allowmissing!(sparse_step_events, :static_stack_pred_dst_nt)
    for row in eachrow(sparse_step_events)
        row.model_step_hours == 1 || continue
        event = get(event_map, row.target_time_utc, 0)
        event >= 5 && (row.static_stack_pred_dst_nt = missing)
    end
    sparse_status = SuperlearnerLiveClaimAudit.prospective_status(
        sparse_step_events; bootstrap_reps=200,
    )
    step_one = only(filter(row -> row.model_step_hours == 1,
                           sparse_status.storm.by_step))
    @test step_one.n >= 30
    @test step_one.n_events == 4
    @test !step_one.gate_pass
    @test !sparse_status.storm.claim_ready
end

function _write_claim_archive(dir, index, frame)
    archive_dir = joinpath(dir, "archive")
    mkpath(archive_dir)
    name = index == 0 ? "live_forecast_log_archive.csv" :
                       "live_forecast_log_archive.$index.csv"
    path = joinpath(archive_dir, name)
    CSV.write(path, frame)
    manifest = Dict("archived_rows" => nrow(frame), "archive_bytes" => filesize(path),
                    "segment_index" => index, "last_segment_rows" => nrow(frame),
                    "last_segment_sha256" => bytes2hex(sha256(read(path))))
    write(path * ".manifest.json", JSON3.write(manifest))
    return path
end

@testset "super-learner prospective claims retain archived evidence" begin
    fixture = _claim_ready_fixture()
    # Missing deployment metadata cannot hide an otherwise observable post-freeze row.
    missing_identity = copy(fixture[1:1, :])
    missing_identity.superlearner_cal_shadow_model_version[1] = ""
    status = SuperlearnerLiveClaimAudit.prospective_status(vcat(fixture, missing_identity); bootstrap_reps=100)
    @test !status.integrity.gate_pass
    @test !status.marginal.claim_ready
    @test !status.storm.claim_ready
    allowmissing!(missing_identity, :superlearner_cal_shadow_model_version)
    missing_identity.superlearner_cal_shadow_model_version[1] = missing
    @test !SuperlearnerLiveClaimAudit.prospective_status(
        vcat(fixture, missing_identity); bootstrap_reps=100,
    ).integrity.gate_pass
    absent_column = select(missing_identity, Not(:superlearner_cal_shadow_model_version))
    @test !SuperlearnerLiveClaimAudit.prospective_status(absent_column; bootstrap_reps=100).integrity.gate_pass
    historical = copy(absent_column)
    historical.issue_time_utc[1] = SuperlearnerLiveClaimAudit.SHADOW_START_UTC - Hour(1)
    @test SuperlearnerLiveClaimAudit.prospective_status(historical; bootstrap_reps=100).integrity.gate_pass

    mktempdir() do dir
        path = joinpath(dir, "live.csv")
        good = _claim_ready_fixture()
        bad = copy(good[1:1, :])
        bad.superlearner_manifest_sha256[1] = "0"^64
        CSV.write(path, good)
        _write_claim_archive(dir, 0, bad)
        status = SuperlearnerLiveClaimAudit.prospective_status(path; bootstrap_reps=100)
        @test !status.integrity.gate_pass
        @test !status.marginal.claim_ready
        @test !status.storm.claim_ready
        @test status.integrity.deployment_rows == 3_201
        @test any(value -> occursin("served-manifest", value), status.integrity.violations)
    end

    # Schema changes create numeric segments; retaining rows cannot change any statistic.
    mktempdir() do dir
        path = joinpath(dir, "live.csv")
        fixture = _claim_ready_fixture()
        for index in 0:10
            frame = copy(fixture[(100index + 1):(100index + 100), :])
            frame[!, Symbol("extra_$index")] = fill("quoted,\nvalue", 100)
            _write_claim_archive(dir, index, frame)
        end
        CSV.write(path, fixture[1101:end, :])
        actual = SuperlearnerLiveClaimAudit.prospective_status(path; bootstrap_reps=100)
        expected = SuperlearnerLiveClaimAudit.prospective_status(fixture; bootstrap_reps=100)
        @test actual.integrity == expected.integrity
        @test actual.marginal == expected.marginal
        @test actual.storm == expected.storm
    end
end

@testset "super-learner incomplete source snapshots persist a failed claim gate" begin
    # Windows requires Administrator/Developer Mode to create symlinks; probe once so the
    # :archive_symlink fault can be skipped when the OS denies creation.
    can_symlink = mktempdir() do dir
        target = joinpath(dir, "target")
        write(target, "probe")
        try
            symlink(target, joinpath(dir, "link"))
            true
        catch e
            e isa Base.IOError || rethrow()
            false
        end
    end
    for fault in (:missing_hot, :pending_append, :pending_retention, :busy_lock,
                  :missing_manifest, :corrupt_manifest, :wrong_bytes, :wrong_rows,
                  :wrong_index, :missing_segment, :orphan_manifest, :archive_symlink)
        fault == :archive_symlink && !can_symlink && continue
        mktempdir() do dir
            path = joinpath(dir, "live.csv")
            CSV.write(path, _claim_audit_fixture())
            archive = _write_claim_archive(dir, 0, _claim_audit_fixture())
            receipt = archive * ".manifest.json"
            if fault == :missing_hot
                rm(path)
            elseif fault in (:pending_append, :pending_retention)
                suffix = fault == :pending_append ? ".append.json" : ".retention.json"
                write(path * suffix, "{}")
            elseif fault == :busy_lock
                mkdir(path * ".lock")
            elseif fault == :missing_manifest
                rm(receipt)
            elseif fault == :corrupt_manifest
                write(receipt, "{broken")
            elseif fault in (:wrong_bytes, :wrong_rows, :wrong_index)
                manifest = JSON3.read(read(receipt, String), Dict{String,Any})
                key = fault == :wrong_bytes ? "archive_bytes" :
                      fault == :wrong_rows ? "archived_rows" : "segment_index"
                manifest[key] += 1
                write(receipt, JSON3.write(manifest))
            elseif fault == :missing_segment
                _write_claim_archive(dir, 2, _claim_audit_fixture())
            elseif fault == :orphan_manifest
                rm(archive)
            elseif fault == :archive_symlink
                mv(archive, archive * ".saved")
                symlink(archive * ".saved", archive)
            end
            json_path = joinpath(dir, "status.json")
            write(json_path, "{\"marginal\":{\"claim_ready\":true}}")
            before = isfile(path) ? read(path) : nothing
            status = SuperlearnerLiveClaimAudit.run_claim_audit(
                path, json_path, joinpath(dir, "status.md");
                bootstrap_reps=100, lock_timeout_sec=0.02,
            )
            @test !status.integrity.gate_pass
            @test !status.marginal.claim_ready
            @test !status.storm.claim_ready
            @test !isempty(status.integrity.violations)
            @test !JSON3.read(read(json_path, String)).marginal.claim_ready
            @test (isfile(path) ? read(path) : nothing) == before
        end
    end
end
