module ExternalDstTiming

using DataFrames
using Dates
using Printf
using Statistics

function _parse_external_time(x)
    (x === missing || x === nothing) && return missing
    s = strip(String(string(x)))
    isempty(s) && return missing
    s = replace(s, " " => "T")
    m = match(r"^(.{19})(?:\.(\d+))?Z?$", s)
    m === nothing && return missing
    base = m.captures[1]
    parsed = nothing
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS", dateformat"yyyy/mm/dd-HH:MM:SS")
        parsed = tryparse(DateTime, base, fmt)
        parsed !== nothing && break
    end
    parsed === nothing && return missing
    fraction = m.captures[2]
    fraction === nothing && return parsed
    digits = first(fraction, min(length(fraction), 3))
    return parsed + Millisecond(parse(Int, rpad(digits, 3, '0')))
end

"""Classify receipt evidence without reconstructing missing historical completion times."""
function external_dst_timing(row)
    issue = _parse_external_time(row.issue_utc)
    fetched = _parse_external_time(row.fetched_utc)
    target = _parse_external_time(row.target_utc)
    any(ismissing, (issue, fetched, target)) && return :invalid
    target > issue || return :invalid
    receipt_value = hasproperty(row, :receipt_completed_utc) ? row.receipt_completed_utc : missing
    if ismissing(receipt_value)
        return target <= fetched ? :late : :legacy
    end
    receipt = _parse_external_time(receipt_value)
    ismissing(receipt) && return :invalid
    receipt >= fetched && receipt >= issue || return :invalid
    return target > receipt ? :eligible : :late
end

function _external_stable_rmse(values::AbstractVector{<:Real})
    isempty(values) && return missing
    scale = maximum(abs, values)
    isfinite(scale) || throw(ArgumentError("external Dst residuals must be finite"))
    scale == 0 && return 0.0
    normalized = Float64.(values) ./ Float64(scale)
    result = Float64(scale) * sqrt(mean(abs2, normalized))
    isfinite(result) || throw(ArgumentError(
        "external Dst RMSE exceeds the supported Float64 range",
    ))
    return result
end

function external_dst_summary(df::DataFrame)
    out = DataFrame(source=String[], n_rows=Int[], n_scored=Int[],
        n_issues=Int[], max_lead_h=Float64[],
        rmse_nt=Union{Missing, Float64}[], mae_nt=Union{Missing, Float64}[],
        n_eligible=Int[], n_late=Int[], n_legacy=Int[], n_invalid=Int[],
        max_receipt_lead_h=Union{Missing, Float64}[])
    isempty(df) && return out
    for sub in groupby(df, :source)
        timing = external_dst_timing.(eachrow(sub))
        eligible = timing .== :eligible
        scored = eligible .& .!ismissing.(sub.observed_dst_nt)
        errors = Float64.(sub.forecast_dst_nt[scored]) .- Float64.(sub.observed_dst_nt[scored])
        all(isfinite, errors) || throw(ArgumentError(
            "external Dst residual exceeds the supported Float64 range",
        ))
        rmse = _external_stable_rmse(errors)
        mae = isempty(errors) ? missing : mean(abs.(errors))
        receipt_leads = [Dates.value(_parse_external_time(row.target_utc) -
            _parse_external_time(row.receipt_completed_utc)) / 3_600_000
            for row in eachrow(sub[eligible, :])]
        push!(out, (String(first(sub.source)), nrow(sub), count(scored),
            length(unique(String.(sub.issue_utc))), maximum(Float64.(sub.lead_h)),
            rmse, mae, count(eligible), count(==(:late), timing),
            count(==(:legacy), timing), count(==(:invalid), timing),
            isempty(receipt_leads) ? missing : maximum(receipt_leads)))
    end
    return sort!(out, :source)
end

function write_external_dst_metrics(io::IO, summary::DataFrame)
    println(io, "Scores require recorded response completion strictly before target, with fetch-start and source issue at/before completion. Legacy rows without completion evidence and late rows are excluded; historical error fields are retained. Maximum receipt lead is target minus completion.\n")
    println(io, "| Source | Archived rows | Receipt-future | Scored | Max receipt lead [h] | RMSE [nT] | MAE [nT] |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|")
    for row in eachrow(summary)
        lead = ismissing(row.max_receipt_lead_h) ? "unavailable" : @sprintf("%.3f", row.max_receipt_lead_h)
        rmse = ismissing(row.rmse_nt) ? "pending" : @sprintf("%.2f", row.rmse_nt)
        mae = ismissing(row.mae_nt) ? "pending" : @sprintf("%.2f", row.mae_nt)
        println(io, "| $(row.source) | $(row.n_rows) | $(row.n_eligible) | $(row.n_scored) | $lead | $rmse | $mae |")
    end
    println(io, "\n| Source | Known late | Unknown completion | Invalid chronology |")
    println(io, "|---|---:|---:|---:|")
    for row in eachrow(summary)
        println(io, "| $(row.source) | $(row.n_late) | $(row.n_legacy) | $(row.n_invalid) |")
    end
    println(io, "\nThe four timing categories partition archived rows. A legacy target at/before fetch-start is known late; other legacy rows have unknown completion. Repeated retrieval does not replace the first record's timestamps.")
    return nothing
end

end
