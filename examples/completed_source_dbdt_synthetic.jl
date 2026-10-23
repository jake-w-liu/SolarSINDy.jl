using Dates
using SolarSINDy.CompletedSourceDBDT

"""Exercise the actual adapter/controller chain on a hand-derived polynomial only."""
function run_synthetic_completed_source_replay()
    issue=DateTime(2024,1,1)
    ground=Dict(issue+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    prepared=prepare_schedule(ground,[issue])
    prediction=bind_predictions(prepared,X->zeros(size(X,1)))
    @assert prepared[1].source_row.dbdt_now==3.0
    @assert prepared[1].source_row.dbdt_mean30==32.0
    @assert prepared[1].source_row.dbdt_max30==61.0
    @assert prepared[1].ground11[8:11]==[0.0,1.0,0.0,1.0]
    audit=case_audit(prepared,ground;asof=issue+Minute(31))
    @assert only(audit).target==59.0
    cal=CompletedCalibration(zeros(9),fill(issue-Millisecond(1),9);
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    labels=Dict(issue=>59.0)
    phase=replay_phase(prepared,prediction.centers,prediction.scales,labels,cal;
        gamma=0.1,asof=issue+Minute(31))
    fifo=replay_fifo(prepared,prediction.centers,prediction.scales,labels,cal;
        gamma=0.1/31,asof=issue+Minute(31))
    receipts=materialize_receipts(ground,issue+Minute(1):Minute(1):issue+Minute(30);
        asof=issue+Minute(31))
    early=replay_receipts(prepared,prediction.centers,prediction.scales,receipts.batches,cal;
        gamma=0.1/31,asof=issue+Minute(31))
    @assert phase.indices==fifo.indices==early.indices==[1]
    @assert phase.phase_period_minutes==31
    @assert phase.certificate.misses==fifo.certificate.misses==early.certificate.misses==1
    @assert early.certificate.completed_scores==1
    @assert only(filter(event->event.kind===:score,early.events)).target==59.0
    @assert !early.measured_receipts_available
    (;target=59.0,phase_period=31,issued=1,completed_scores=1,
      actual_receipts_proved=false,observational_skill_evaluated=false)
end

if abspath(PROGRAM_FILE)==@__FILE__
    println(run_synthetic_completed_source_replay())
end
