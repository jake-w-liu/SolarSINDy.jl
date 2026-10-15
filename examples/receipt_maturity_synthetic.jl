module ReceiptMaturitySyntheticExperiment
using Dates,SHA,CSV,DataFrames,JSON3
include(joinpath(@__DIR__,"live_receipt_maturity.jl"))
const R=LiveReceiptMaturity
const E=R.LiveObservationEvidence
function run()
    root=mktempdir();log=joinpath(root,"synthetic.csv");store=joinpath(root,"evidence")
    target=DateTime(2020,1,1,10);origin=target-Hour(1)
    table=DataFrame(issue_time_utc=[string(origin+Minute(30))],latest_dst_time_utc=[string(origin)],
        target_time_utc=[string(target)],model_step_hours=[1],sub_hourly_model_version=[R.CURRENT_MODEL],
        superlearner_manifest_sha256=[R.CURRENT_MANIFEST],served_pred_dst_nt=[10.],
        served_pred_dst_ci05_nt=[0.],served_pred_dst_ci95_nt=[20.],observation_dst_nt=Union{Missing,Float64}[missing])
    CSV.write(log,table)
    function commit(value,received,verified;change_csv)
        response=(;status=200,headers=["x-synthetic-fixture"=>"no network or historical evidence"],
            body=JSON3.write([Dict("time_tag"=>string(target)*"Z","dst"=>value)]))
        reference=E.capture_response!(store,response;request_started_utc=received,receipt_completed_utc=received,
            monotonic_started_ns=Dates.value(received),monotonic_completed_ns=Dates.value(received),
            consumer_source_sha256=bytes2hex(sha256(read(@__FILE__))))
        prior=ismissing(table.observation_dst_nt[1]) ? nothing : table.observation_dst_nt[1]
        intent=E.prepare_verification!(store,reference,[(;binding=E.forecast_binding(table[1,:]),previous_observation=prior,observation=value)];
            log_path=log,verification_started_utc=verified-Millisecond(1))
        change_csv && (table.observation_dst_nt[1]=value;CSV.write(log,table))
        E.commit_verification!(store,intent;log_path=log,verification_completed_utc=verified)
    end
    commit(15.,target+Hour(1)-Millisecond(1),target+Hour(1)+Millisecond(1);change_csv=true)
    partial=R.snapshot_report(log,store;asof=target+Hour(1)+Millisecond(2))
    @assert partial.payload.counts["preliminary_actual_receipt"]==1
    @assert partial.payload.completed_primary_current_identity.n==0
    before=read(log)
    commit(15.,target+Hour(1),target+Hour(1)+Millisecond(4);change_csv=false)
    @assert read(log)==before
    complete=R.snapshot_report(log,store;asof=target+Hour(1)+Millisecond(5))
    @assert complete.payload.completed_primary_current_identity.n==1
    @assert complete.payload.completed_primary_current_identity.rmse==5.
    @assert complete.payload.completed_primary_current_identity.mean_interval_score_90==20.
    @assert R.trusted_claim_cohort(complete).support==0
    output=joinpath(root,"new_report");publication=R.publish_report(complete,output)
    @assert R.verify_publication(output;expected_completion_sha256=publication.completion_sha256)
    result=(;schema="receipt_maturity_synthetic_experiment_v1",scope="synthetic mock journal only; no real observations/fit/network",
        completed_support=1,qualified_claim_support=0,claim_ready=false,publication_completion_sha256=publication.completion_sha256,
        unchanged_csv_on_reconfirmation=true,nominal_not_final=true)
    write(joinpath(root,"synthetic_experiment_receipt.json"),JSON3.write(result))
    println("Receipt maturity synthetic experiment PASS: ",JSON3.write(result))
    true
end
end
run_receipt_maturity_synthetic()=ReceiptMaturitySyntheticExperiment.run()
