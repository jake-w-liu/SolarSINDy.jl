@testset verbose=true "All-subscriber rollback and transaction ownership" begin
    owner=state(;subscribers=specs(;rolling=9))
    first=transact!(owner,I;issues=(request(;center=-1e308,scale=1e-308),))
    @test all(bound.upper==-1.0 for bound in only(first.issues).bounds)
    before=owner.snapshot
    @test_throws ArgumentError prepare_time(owner,I+Minute(31);observations=rate_rows(1,30),transaction_id="overflow")
    @test owner.snapshot===before
    @test source_certificate(owner).completed_targets==0 && isempty(owner.snapshot.seen)
    @test all(subscriber_certificate(owner,n).binary_resolved==0 for n in (:static,:phase,:fifo,:early,:maturity))
    @test all(isnan,only(owner.snapshot.pending).rates)
    owner=state();a=prepare_time(owner,I;issues=(request(),),transaction_id="a")
    b=prepare_time(owner,I;issues=(request(),),transaction_id="b")
    @test source_certificate(owner).attempted==0 # Preparation alone emits no owned mutation.
    committed=commit_time!(owner,b);before=owner.snapshot
    @test_throws ArgumentError commit_time!(owner,a)
    @test owner.snapshot===before
    @test commit_time!(owner,b).repeated && owner.snapshot===before
    repeated=prepare_time(owner,I;issues=(request(),),transaction_id="b")
    @test commit_time!(owner,repeated).repeated && owner.snapshot===before
    @test_throws ArgumentError prepare_time(owner,I;issues=(request(;center=1.0),),transaction_id="b")
    @test_throws ArgumentError prepare_time(owner,I;transaction_id="other")
    @test_throws ArgumentError prepare_time(owner,I+Minute(1);transaction_id="b")
    @test_throws ArgumentError commit_time!(state(),b)
    @test owner.snapshot===before
    # Calibrations cannot be mutated around a new owner, or after its snapshot.
    cal=calibration();owner=EventReplay(cal;start=I,subscribers=specs(),stream_id=STREAM,protocol_id=PROTOCOL)
    cal.chronological_scores[1]=1.0
    @test owner.snapshot.calibration.chronological_scores==ntuple(_->0.0,9)
    @test_throws ArgumentError EventReplay(cal;start=I,subscribers=specs(),stream_id=STREAM,protocol_id=PROTOCOL)
end

@testset verbose=true "Invalid source batches and guarded subscriber grammar" begin
    bads=[[(;center=I+Minute(1),rate=1.0),(;center=I+Minute(1),rate=2.0)],
          [(;center=I+Minute(1)+Millisecond(1),rate=0.0)],[(;center=I+Minute(1),rate=true)],
          [(;center=I+Minute(1),rate=NaN)],[(;center=I+Minute(1),rate=Inf)],
          [(;center=I+Minute(1),rate=BigFloat(2)^(-1076))],[(;center=I+Minute(1),rate=-1.0)],
          [(;center=I+Minute(1),rate=0.0,extra=1)],[(;center=I+Minute(1),rate=0.0,source_id="")]]
    for bad in bads,reverse_order in (false,true)
        owner=state();transact!(owner,I;issues=(request(),));before=owner.snapshot
        source=vcat([(center=I+Minute(2),rate=0.0)],bad);reverse_order && reverse!(source)
        @test_throws ArgumentError prepare_time(owner,I+Minute(3);observations=source,transaction_id="bad")
        @test owner.snapshot===before
    end
    owner=state();transact!(owner,I;issues=(request(),));before=owner.snapshot
    @test_throws ArgumentError prepare_time(owner,offset(104999);observations=rate_rows(1,1),transaction_id="early")
    @test owner.snapshot===before
    transact!(owner,I+Minute(2);rows=rate_rows(1,1;positive_at=1))
    before=owner.snapshot
    @test_throws ArgumentError prepare_time(owner,I+Minute(3);observations=rate_rows(1,1;positive_at=1,value=3),transaction_id="conflict")
    @test_throws ArgumentError prepare_time(owner,I+Minute(1);transaction_id="backward")
    @test owner.snapshot===before
    duplicate=transact!(owner,I+Minute(3);rows=rate_rows(1,1;positive_at=1))
    @test only(duplicate.events).kind===:duplicate_source && isempty(duplicate.subscriber_events)
    @test_throws ArgumentError SubscriberSpec(:bad,:publisher_up)
    @test_throws ArgumentError SubscriberSpec(:bad,()->error("must not call"))
    @test_throws ArgumentError EventReplay(calibration();start=I,subscribers=(()->error("must not call"),),stream_id=STREAM,protocol_id=PROTOCOL)
    @test_throws ArgumentError EventReplay(calibration();start=I,subscribers=(SubscriberSpec(:x,:fifo),SubscriberSpec(:x,:early)),stream_id=STREAM,protocol_id=PROTOCOL)
    for kind in (:static,:phase,:fifo)
        @test_throws ArgumentError SubscriberSpec(:fixed,kind;rolling_capacity=1)
    end
    @test_throws ArgumentError SubscriberSpec(:x,:early;rolling_capacity=true)
    @test_throws ArgumentError SubscriberSpec(:x,:early;mandatory=1)
    @test_throws ArgumentError state(;grace=true)
    @test_throws ArgumentError state(;grace=86400001)
    tiny=BigFloat(2)^(-1076)
    @test_throws ArgumentError SubscriberSpec(:x,:early;gamma=tiny)
    @test_throws ArgumentError request(;scale=tiny)
    delta=nextfloat(0.0);owner=state();transact!(owner,I;issues=(request(),))
    transact!(owner,offset(105000);rows=[(;center=I+Minute(1),rate=BigFloat(delta))])
    @test only(owner.snapshot.pending).rates[1]===delta && subscriber_certificate(owner,:early).binary_misses==1
end

@testset verbose=true "Checked clock/id/capacity extremes and strict warmup" begin
    for (grace,capacity,safe) in zip((0,1,60000,60001,86400000),(32,33,33,34,1472),(32,32,33,33,1472))
        owner=state(;grace,subscribers=(SubscriberSpec(:p,:phase;phase_period=:safe),))
        @test owner.snapshot.capacity==capacity && only(owner.snapshot.subscribers).period==safe
        @test safe_terminal_period(I;grace_ms=grace)==safe
    end
    owner=state();owner.snapshot=E.replace_fields(owner.snapshot;last_id=typemax(Int));before=owner.snapshot
    @test_throws OverflowError prepare_time(owner,I;issues=(request(),),transaction_id="overflow_id")
    @test owner.snapshot===before
    raw_date(value)=DateTime(Dates.UTInstant{Millisecond}(Millisecond(value)))
    maximum_raw=typemax(Int64);due=maximum_raw-mod(maximum_raw,60000)
    start=raw_date(due-31*60000);grace=maximum_raw-due
    owner=state(;start,grace);before=owner.snapshot
    @test_throws OverflowError prepare_time(owner,start;issues=(request(start),),transaction_id="overflow_expiry")
    @test owner.snapshot===before
    owner=state();owner.snapshot=E.replace_fields(owner.snapshot;clock=raw_date(typemin(Int64)));before=owner.snapshot
    @test_throws OverflowError prepare_time(owner,owner.snapshot.clock;transaction_id="overflow_cutoff")
    @test owner.snapshot===before
    owner=state();owner.snapshot=E.replace_fields(owner.snapshot;revision=typemax(Int));before=owner.snapshot
    @test_throws OverflowError prepare_time(owner,I;transaction_id="overflow_revision")
    @test owner.snapshot===before

    fixture=FIXTURES["strict_warmup_membership_and_period"];policy=CMR.ReplayPolicy(ground_extra_ms=15000);cal=calibration(;policy)
    records=[(;id=r["id"],issue=offset(r["issue_offset_ms"]),label_ready=offset(r["label_ready_offset_ms"]),
              completed_at=offset(r["completed_at_offset_ms"]),phase_period_minutes=r["period"],
              point_model_sha256=cal.point_model_sha256,policy,mode=:ground11) for r in fixture["input"]["candidates"]]
    @test warmup_membership(records[1:3];start=I,calibration=cal,phase_period_minutes=31)==(1,)
    @test_throws ArgumentError warmup_membership(records;start=I,calibration=cal,phase_period_minutes=31)
    @test_throws ArgumentError warmup_membership(vcat(records[1:1],records[1:1]);start=I,calibration=cal,phase_period_minutes=31)
    wrong=merge(records[1],(;point_model_sha256=repeat("2",64)))
    @test_throws ArgumentError warmup_membership([wrong];start=I,calibration=cal,phase_period_minutes=31)
end
