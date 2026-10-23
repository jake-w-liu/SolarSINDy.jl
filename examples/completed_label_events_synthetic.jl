using Dates, SolarSINDy
using SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents

"""Exercise all five fixed consumers on the same hand-derived synthetic targets."""
function run_synthetic_completed_label_events()
    D=SolarSINDy.CompletedSourceDBDT
    issue=DateTime(2024,1,1)
    ground=Dict(issue+Minute(k)=>(k>=1 ? 1.0 : 0.0,0.0) for k in -40:74)
    prepared=D.prepare_schedule(ground,issue:Minute(1):issue+Minute(43))
    calibration=D.CompletedCalibration(zeros(9),fill(issue-Millisecond(1),9);
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    consumers=(SubscriberSpec(:static,:static),SubscriberSpec(:phase,:phase),
        SubscriberSpec(:fifo,:fifo;gamma=.1/31),SubscriberSpec(:early,:early),SubscriberSpec(:maturity,:maturity))
    receipts=D.materialize_receipts(ground,issue+Minute(1):Minute(1):issue+Minute(73);asof=issue+Minute(74))
    result=replay_shared(prepared,zeros(44),ones(44),receipts.batches,calibration;
        subscribers=consumers,stream_id="synthetic-ground-step",protocol_id="synthetic-shared44-v1",asof=issue+Minute(74))
    @assert result.source.attempted==result.source.completed_targets==44
    @assert result.source.target_unknown==0 && result.study_pass
    @assert all(row.binary_resolved==row.completed_scores==44 for row in result.subscribers)
    targets=[row.target for row in result.events if row.kind===:completed_target]
    @assert targets==vcat([1.0],zeros(43))
    @assert !result.context.source_bound_provenance_verified && !result.measured_receipts_available
    (;attempted=44,completed=44,subscribers=5,actual_refits_proved=false,observational_skill_evaluated=false)
end

"""Check policy-bounded live state while preserving all1000 attempted windows."""
function run_synthetic_completed_label_events_bounded()
    D=SolarSINDy.CompletedSourceDBDT
    issue=DateTime(2024,1,1)
    ground=Dict(issue+Minute(k)=>(0.0,0.0) for k in -40:1000)
    prepared=D.prepare_schedule(ground,issue:Minute(1):issue+Minute(999))
    calibration=D.CompletedCalibration(zeros(9),fill(issue-Millisecond(1),9);
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    owner=EventReplay(calibration;start=issue,subscribers=(SubscriberSpec(:phase,:phase),
        SubscriberSpec(:early,:early;rolling_capacity=5),SubscriberSpec(:maturity,:maturity;rolling_capacity=5)),
        stream_id="synthetic-ground-zero",protocol_id="synthetic-shared1000-v1")
    for (index,row) in enumerate(prepared)
        time=row.issue
        plan=prepare_time(owner,time;observations=((;center=time-Minute(1),rate=0.0),),
            issues=(IssueRequest(row,0.0,1.0),),transaction_id="synthetic/"*string(index))
        commit_time!(owner,plan)
    end
    proof=source_certificate(owner)
    @assert proof.attempted==1000 && proof.completed_targets==969 && proof.pending_targets==31
    @assert proof.max_pending==31 && proof.max_retained_source<=33
    @assert subscriber_certificate(owner,:early).rolling_score_count==5
    (;attempted=1000,completed=969,pending=31,max_pending=31,source_bound_provenance_verified=false)
end

if abspath(PROGRAM_FILE)==@__FILE__
    println(run_synthetic_completed_label_events())
    println(run_synthetic_completed_label_events_bounded())
end
