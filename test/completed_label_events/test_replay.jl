struct FutureBatch
    received::DateTime
end

@testset verbose=true "Independent signed-score queue values preserve fixed calibration" begin
    oracle=JSON3.read(read(joinpath(@__DIR__,"frozen","score_queue_oracle_v1.json")),Dict{String,Any})
    owner=state(;subscribers=specs(;rolling=oracle["rolling_capacity"]))
    for (index,center) in enumerate(oracle["completed_centers"])
        issue=I+Minute((index-1)*32)
        transact!(owner,issue;issues=(request(issue;center),))
        result=transact!(owner,issue+Minute(31);rows=[(;center=issue+Minute(k),rate=0.0) for k in 1:30])
        @test all(row.score==oracle["complete_scores"][index] for row in result.subscriber_events if row.kind===:score)
        for name in (:early,:maturity)
            selected=subscriber(owner,name)
            @test collect(selected.scores)==oracle["rolling_storage_after_each_completion"][index]
            @test selected.score_head==oracle["rolling_head_after_each_completion"][index]
        end
    end
    for name in (:static,:phase,:fifo)
        @test collect(subscriber(owner,name).scores)==oracle["fixed_storage_after_all_completions"]
        @test subscriber_certificate(owner,name).completed_scores==3
    end
end
Base.getproperty(batch::FutureBatch,name::Symbol)=name===:received ? getfield(batch,:received) : error("future receipt payload opened")

@testset verbose=true "Shared44-issue independent trace and flush invariance" begin
    mag=Dict(I+Minute(k)=>(k>=1 ? 1.0 : 0.0,0.0) for k in -40:80)
    prepared=D.prepare_schedule(mag,I:Minute(1):I+Minute(43))
    source=[(;received=I+Minute(k+1),observations=rate_rows(k,k;positive_at=1,value=1.0)) for k in 1:73]
    subscribers=(SubscriberSpec(:phase,:phase),SubscriberSpec(:fifo,:fifo;gamma=.1/31),
        SubscriberSpec(:static,:static),SubscriberSpec(:early,:early),SubscriberSpec(:maturity,:maturity))
    initial=replay_shared(prepared,zeros(44),ones(44),source,calibration();subscribers,stream_id=STREAM,protocol_id=PROTOCOL,asof=I+Minute(43),flush=false)
    flushed=replay_shared(prepared,zeros(44),ones(44),source,calibration();subscribers,stream_id=STREAM,protocol_id=PROTOCOL,asof=I+Minute(74))
    oracle=FIXTURES["default_44_issue_trace_and_flush"]["expected"]
    @test initial.indices==flushed.indices==collect(1:44)
    @test isequal(initial.issue_records,flushed.issue_records)
    @test initial.source.completed_targets==13 && initial.source.target_unknown==31
    @test flushed.source.completed_targets==44 && flushed.source.target_unknown==0
    @test initial.study_pass && flushed.study_pass
    for (slot,name) in ((1,"phase_trace"),(2,"fifo_trace"))
        for (actual,expected) in zip(initial.issue_records,oracle[name])
            bound=actual.bounds[slot]
            @test isapprox(bound.alpha,literal_alpha(expected["alpha"]);atol=1e-14,rtol=0) # Frozen exact Fraction trace.
            @test bound.upper==(expected["upper"]=="Inf" ? Inf : expected["upper"])
        end
    end
    @test all(c.binary_resolved==44 && abs(something(c.accounting_residual,0.0))<1e-12 for c in flushed.subscribers) # Original short accounting tolerance.
    # Metadata gates access before touching a forbidden future payload.
    future=replay_shared(prepared[1:1],[0.0],[1.0],[FutureBatch(I+Hour(1))],calibration();
        subscribers,stream_id=STREAM,protocol_id=PROTOCOL,asof=I)
    @test future.skipped_future==1 && future.source.pending_targets==1
end

@testset verbose=true "Timely and missing1000-issue bounded source state" begin
    owner=state(;subscribers=specs(;rolling=5))
    for minute in 0:999
        time=I+Minute(minute)
        transact!(owner,time;rows=[(;center=time-Minute(1),rate=0.0)],issues=(request(time),))
    end
    proof=source_certificate(owner)
    @test proof.attempted==1000 && proof.completed_targets==969 && proof.pending_targets==31
    @test proof.max_pending==31 && proof.max_retained_source<=33 && proof.retained_source<=33
    @test subscriber_certificate(owner,:early).rolling_score_count==subscriber_certificate(owner,:maturity).rolling_score_count==5
    @test all(abs(something(subscriber_certificate(owner,name).accounting_residual,0.0))<1e-10 for name in (:static,:phase,:fifo,:early,:maturity)) # Frozen long accounting tolerance.
    # No receipt can reopen closed tickets after retention expires.
    transact!(owner,I+Minute(1100));before=source_certificate(owner)
    stale=transact!(owner,I+Minute(1101);rows=rate_rows(1,1;positive_at=1))
    @test only(stale.events).kind===:ignored_stale && source_certificate(owner).completed_targets==before.completed_targets
    @test isempty(owner.snapshot.seen) && isempty(owner.snapshot.pending)

    owner=state()
    for minute in 0:70
        time=I+Minute(minute)
        # Advance strict-expiry timers independently of receipt presence.
        minute>31 && transact!(owner,time-Minute(1)+Millisecond(1))
        transact!(owner,time;issues=(request(time),))
    end
    @test !source_certificate(owner).study_pass
    @test source_certificate(owner).max_pending<=32 && length(owner.snapshot.pending)<=32
    phase=subscriber_certificate(owner,:phase)
    @test phase.attempted==71 && phase.unavailable>0 && phase.issued_bounds+phase.unavailable==71
    @test phase.binary_resolved==0 && phase.all_attempted_enclosure.unknown==71
end
