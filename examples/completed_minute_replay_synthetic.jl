using Dates, SolarSINDy
using SolarSINDy.CompletedMinuteReplay

"Exercise source admission and feedback clocks without an observational skill claim."
function run_completed_minute_replay_synthetic()
    issue=DateTime(2024,12,31,23,59)
    ground=Dict(issue+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    wind=Dict(issue-Minute(1)=>(400.0,-5.0,1.0,8.0))
    prefix=Dict(k=>v for (k,v) in ground if k<issue)
    features=admit_features(prefix,wind,issue)
    @assert features.accepted && features.value.row.dbdt_now==3.0
    @assert admit_target(ground,issue,issue+Minute(30)).reason===:label_not_ready
    result=replay_case(ground,wind,issue,issue+Minute(31))
    @assert result.accepted && result.value.row.target_max_dbdt==59.0
    @assert result.value.row.label_ready==DateTime(2025,1,1,0,29,45)
    @assert result.value.timing.sufficient_phase_period_minutes==31
    @assert !training_eligible(result.value.row.label_ready,result.value.row.label_ready)
    @assert !result.value.row.measured_receipts_available
    println("Completed-source minute replay: prefix features, explicit label readiness and year boundary PASS; no skill verdict")
    true
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_completed_minute_replay_synthetic()
end
