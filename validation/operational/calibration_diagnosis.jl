module CalibrationDiagnosis

using CSV, DataFrames, Dates, JSON3, SHA, Statistics
include(joinpath(@__DIR__, "v2_4_live_claim_audit.jl"))
const Audit = V24LiveClaimAudit

function interval_metrics(observed, point, lo, hi, static_lo, static_hi)
    n = length(observed)
    n > 0 && all(length(v) == n for v in (point, lo, hi, static_lo, static_hi)) ||
        throw(ArgumentError("interval vectors must be nonempty and equally sized"))
    all(v -> all(x -> x isa Real && !(x isa Bool) && isfinite(x), v),
        (observed, point, lo, hi, static_lo, static_hi)) ||
        throw(ArgumentError("interval values must be finite numbers"))
    all(lo .< hi) && all(static_lo .< static_hi) ||
        throw(ArgumentError("interval widths must be positive"))
    covered = count(lo .<= observed .<= hi)
    below = count(observed .< lo); above = count(observed .> hi)
    score(l, u, y) = u-l + 20max(l-y, 0) + 20max(y-u, 0)
    return (; n, covered, below, above, coverage=covered/n,
        static_coverage=count(static_lo .<= observed .<= static_hi)/n,
        width_ratio=sum(hi .- lo)/sum(static_hi .- static_lo),
        mean_width_nt=mean(hi .- lo),
        mean_point_residual_nt=mean(observed .- point),
        mean_interval_center_residual_nt=mean(observed .- (lo .+ hi)./2),
        point_rmse_nt=sqrt(mean(abs2, observed .- point)),
        mean_score_difference_nt=mean(score.(lo, hi, observed) .-
                                      score.(static_lo, static_hi, observed)))
end

function summarize_intervals(rows, columns)
    records = NamedTuple[]
    groups = isempty(columns) ? [rows] : groupby(rows, columns; sort=true)
    for group in groups
        isempty(group) && continue
        key = (; (column => first(group[!, column]) for column in columns)...)
        values = interval_metrics(group.observation_dst_nt, group.point_dst_nt,
            group.shadow_lo_dst_nt, group.shadow_hi_dst_nt,
            group.static_lo_dst_nt, group.static_hi_dst_nt)
        push!(records, merge(key, values))
    end
    return DataFrame(records)
end

function a3_rows(raw)
    canonical = Audit.canonical_shadow_rows(raw)
    rows = filter(row -> row.status == "ok" && !ismissing(row.observation_dst_nt) &&
        !ismissing(row.shadow_lo_dst_nt) && !ismissing(row.shadow_hi_dst_nt), canonical.rows)
    rows.issue_day = Date.(rows.issue_time_utc)
    rows.anchor_lag_hours = Int.((rows.issue_hour_utc .- rows.latest_dst_time_utc)./Hour(1))
    rows.anchor_age_minutes = (rows.issue_time_utc .- rows.latest_dst_time_utc)./Minute(1)
    return rows, canonical.violations
end

function ground_metrics(frame)
    nrow(frame) > 0 || throw(ArgumentError("ground panel must be nonempty"))
    for column in (:prediction, :target, :past, :upper)
        all(x -> x isa Real && !(x isa Bool) && isfinite(x) && x >= 0, frame[!, column]) ||
            throw(ArgumentError("ground panel must contain finite nonnegative values"))
    end
    n = nrow(frame); covered = count(frame.target .<= frame.upper)
    return (; n, covered, missed=n-covered, coverage=covered/n,
        mean_upper_nt_min=mean(frame.upper),
        mean_target_nt_min=mean(frame.target),
        point_rmse_nt_min=sqrt(mean(abs2, frame.prediction .- frame.target)),
        past_rmse_nt_min=sqrt(mean(abs2, frame.past .- frame.target)),
        mean_exceedance_nt_min=mean(max.(frame.target .- frame.upper, 0)))
end

function calibration_phases(anchor, target_end, delay)
    delay in (0, 5, 10) || throw(ArgumentError("unsupported receipt delay"))
    length(anchor) == length(target_end) && issorted(anchor) && allunique(anchor) ||
        throw(ArgumentError("anchors must be ordered and unique with matching targets"))
    target_end == anchor .+ Minute(delay+30) || throw(ArgumentError("wrong target ends"))
    phase = String[]; epoch = 1; received = 0
    for k in eachindex(anchor)
        if k > 1 && anchor[k]-anchor[k-1] > Day(1)
            epoch = k
            received = k-1
        end
        while received+1 < k && target_end[received+1]+Minute(delay) < anchor[k]+Minute(delay)
            received += 1
        end
        push!(phase, received-epoch+1 < 1440 ? "seed_present" : "live_residuals_only")
    end
    return phase
end

function run_diagnosis(snapshot, delay_directory, output)
    ispath(output) && error("Refusing to overwrite diagnosis: $output")
    raw = CSV.read(snapshot, DataFrame; strict=true)
    rows, violations = a3_rows(raw)
    isempty(rows) && error("No eligible A3 rows")
    mkpath(output)
    CSV.write(joinpath(output, "a3_rows.csv"), rows)
    for (name, columns) in (("overall", Symbol[]), ("step", [:model_step_hours]),
            ("anchor_lag", [:anchor_lag_hours]),
            ("step_lag", [:model_step_hours, :anchor_lag_hours]), ("day", [:issue_day]))
        CSV.write(joinpath(output, "a3_$(name).csv"), summarize_intervals(rows, columns))
    end
    sources = Dict(basename(snapshot) => bytes2hex(sha256(read(snapshot))))
    ground_daily = NamedTuple[]
    ground_phase = NamedTuple[]
    for period in (2018, 2021, 2024, 2026), delay in (0, 5, 10)
        name = "CMO_adjusted_$(period)_delay$(delay).csv"
        path = joinpath(delay_directory, name)
        sources[name] = bytes2hex(sha256(read(path)))
        frame = CSV.read(path, DataFrame; strict=true)
        frame.anchor_day = Date.(frame.anchor)
        # Seed replacement is based on the number of received residuals, not elapsed time.
        frame.calibration_phase = calibration_phases(frame.anchor, frame.target_end, delay)
        for group in groupby(frame, :anchor_day; sort=true)
            push!(ground_daily, merge((;period, delay_minutes=delay,
                anchor_day=first(group.anchor_day)), ground_metrics(group)))
        end
        for group in groupby(frame, :calibration_phase; sort=true)
            push!(ground_phase, merge((;period, delay_minutes=delay,
                calibration_phase=first(group.calibration_phase)), ground_metrics(group)))
        end
    end
    CSV.write(joinpath(output, "ground_day.csv"), DataFrame(ground_daily))
    CSV.write(joinpath(output, "ground_phase.csv"), DataFrame(ground_phase))
    receipt = (;generated_utc=string(now(UTC))*"Z", source_sha256=sources,
        source_code_sha256=bytes2hex(sha256(read(@__FILE__))),
        cohort_code_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"v2_4_live_claim_audit.jl")))),
        a3_integrity_findings=violations, diagnosis_only=true,
        candidate_selected=false, serving_changed=false)
    write(joinpath(output, "receipt.json"), JSON3.write(receipt))
    return receipt
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: calibration_diagnosis.jl LIVE_SNAPSHOT DELAY_PANELS NEW_OUTPUT")
    run_diagnosis(ARGS...)
end

end
