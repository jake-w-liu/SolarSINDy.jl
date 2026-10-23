function state_snapshot(state)
    (;clock=state.clock,alpha=state.alpha,issued=state.issued,resolved=state.resolved,misses=state.misses,
      censored=state.censored_labels,scores=copy(state.scores),head=state.score_head,
      mature=state.mature_scores,seen=copy(state.seen),
      pending=Dict(id=>(copy(ticket.rates),ticket.feedback_applied) for (id,ticket) in state.pending))
end

@testset "Independent receipt clock and exactly-once early feedback" begin
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:1)
    prepared=prepare_issue(mag,ISSUE)
    for early in (false,true)
        state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.1,early_feedback=early)
        issued=issue!(state,prepared,0.0,1.0)
        @test issued.upper==0.0 && state.pending[1].target_end==ISSUE+Minute(30)
        @test state.pending[1].label_ready==ISSUE+Minute(30)+Second(45)
        @test state.pending[1].feedback_due==ISSUE+Minute(31)
        first_rate=[(;center=ISSUE+Minute(1),rate=2.0)]
        before=state_snapshot(state)
        @test_throws ArgumentError receive!(state,first_rate;received=ISSUE+Minute(1)+Second(44)+Millisecond(999))
        @test isequal(state_snapshot(state),before)
        events=receive!(state,first_rate;received=ISSUE+Minute(1)+Second(45))
        @test count(event->event.kind===:feedback,events)==Int(early)
        @test state.resolved==Int(early) && state.misses==Int(early)
        @test isapprox(state.alpha,early ? 0.01 : 0.1;rtol=0,atol=1e-14)
        duplicate=receive!(state,first_rate;received=ISSUE+Minute(1)+Second(45))
        @test isempty(duplicate) && state.mature_scores==0
        before=state_snapshot(state)
        @test_throws ArgumentError receive!(state,[(;center=ISSUE+Minute(1),rate=3.0)];received=ISSUE+Minute(1)+Second(45))
        @test isequal(state_snapshot(state),before)
        remaining=[(;center=ISSUE+Minute(k),rate=0.0) for k in 2:30]
        events=receive!(state,remaining;received=ISSUE+Minute(30)+Second(45))
        @test count(event->event.kind===:feedback,events)==Int(!early)
        @test only(filter(event->event.kind===:score,events)).target==2.0
        @test state.resolved==state.misses==state.mature_scores==1
        @test isempty(state.pending) && state.censored_labels==0
        @test abs(certificate(state).accounting_residual)<1e-12
    end
    state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.1)
    issue!(state,prepared,0.0,1.0)
    all_rates=[(;center=ISSUE+Minute(k),rate=k==1 ? 2.0 : 0.0) for k in 1:30]
    events=receive!(state,all_rates;received=ISSUE+Minute(31))
    @test state.mature_scores==1 && isempty(state.pending)
    @test state.censored_labels==0 && only(filter(event->event.kind===:score,events)).target==2.0
end

@testset "Atomic batches, ordering, partial targets and stale source retention" begin
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:2)
    rows=prepare_schedule(mag,[ISSUE,ISSUE+Minute(1)])
    state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.1,grace_ms=60_000)
    issue!(state,rows[1],0.0,1.0);issue!(state,rows[2],0.0,1.0)
    before=state_snapshot(state)
    conflict=[(;center=ISSUE+Minute(1),rate=2.0),(;center=ISSUE+Minute(1),rate=3.0)]
    @test_throws ArgumentError receive!(state,conflict;received=ISSUE+Minute(2))
    @test isequal(state_snapshot(state),before)
    observations=[(;center=ISSUE+Minute(k),rate=k==1 ? 2.0 : 0.0) for k in 1:31]
    push!(observations,(;center=ISSUE+Minute(2),rate=0.0))
    events=receive!(state,reverse(observations);received=ISSUE+Minute(32))
    @test [event.id for event in events if event.kind===:feedback]==[1,2]
    @test [event.target for event in events if event.kind===:score]==[2.0,0.0]
    @test state.resolved==2 && state.misses==1 && state.mature_scores==2
    @test isapprox(state.alpha,0.02;rtol=0,atol=1e-14)
    @test isempty(state.pending)
    before=state_snapshot(state)
    @test_throws ArgumentError receive!(state,[(;center=ISSUE+Minute(1),rate=3.0)];received=ISSUE+Minute(33))
    @test isequal(state_snapshot(state),before)
    advance!(state,ISSUE+Minute(120))
    @test isempty(state.seen)
    alpha=state.alpha
    old=receive!(state,[(;center=ISSUE+Minute(1),rate=3.0)];received=ISSUE+Minute(121))
    @test only(old).kind===:ignored_stale && isempty(state.seen) && state.alpha==alpha
    for early in (false,true)
        state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.1,early_feedback=early)
        issue!(state,rows[1],0.0,1.0)
        partial=[(;center=ISSUE+Minute(k),rate=k==1 ? 2.0 : 0.0) for k in 1:30 if k!=15]
        receive!(state,partial;received=ISSUE+Minute(31))
        @test state.resolved==Int(early) && state.mature_scores==0
        advance!(state,ISSUE+Minute(31)+Millisecond(1))
        proof=certificate(state)
        @test proof.censored_labels==1 && isempty(state.pending)
        @test proof.unknown==Int(!early)
        @test proof.lower_gap<=-0.1+Int(early)<=proof.upper_gap
        @test abs(proof.accounting_residual)<1e-12
    end
    state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.1)
    issue!(state,rows[1],-1e308,1e-308)
    before=state_snapshot(state)
    @test_throws ArgumentError receive!(state,[(;center=ISSUE+Minute(k),rate=0.0) for k in 1:30];received=ISSUE+Minute(31))
    @test isequal(state_snapshot(state),before)
end

@testset "Whole synthetic source-to-receipt replay and flush invariance" begin
    mag=Dict(ISSUE+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    rows=prepare_schedule(mag,[ISSUE])
    bound=bind_predictions(rows,X->zeros(size(X,1)))
    @test isapprox(only(bound.scales),1+ORACLE["feature_cases"][1]["ground11"][3];rtol=0,atol=1e-13)
    pending=materialize_receipts(ForbiddenDict(),[ISSUE+Minute(1)];asof=ISSUE+Minute(1)+Second(44)+Millisecond(999),scheduled=false)
    @test pending.pending==[ISSUE+Minute(1)] && isempty(pending.batches)
    for scheduled in (false,true)
        receipts=materialize_receipts(mag,ISSUE+Minute(1):Minute(1):ISSUE+Minute(30);asof=ISSUE+Minute(31),scheduled)
        @test length(receipts.batches)==30 && isempty(receipts.pending) && isempty(receipts.rejections)
        @test maximum(only(batch.observations).rate for batch in receipts.batches)==59.0
        @test !receipts.measured_receipts_available
        run=replay_receipts(rows,[0.0],[1.0],receipts.batches,calibration();gamma=0.1,asof=ISSUE+Minute(31))
        noflush=replay_receipts(rows,[0.0],[1.0],receipts.batches,calibration();gamma=0.1,asof=ISSUE+Minute(31),flush=false)
        @test run.upper==noflush.upper==[0.0]
        @test run.certificate.issued==run.certificate.resolved==run.certificate.misses==run.certificate.completed_scores==1
        @test noflush.certificate.unknown==1 && run.certificate.unknown==0
        @test only(filter(event->event.kind===:score,run.events)).target==59.0
        @test run.state.censored_labels==0
        @test !run.measured_receipts_available
    end
    source=copy(mag);delete!(source,ISSUE+Minute(10))
    receipts=materialize_receipts(source,[ISSUE+Minute(10),ISSUE+Minute(11)];asof=ISSUE+Minute(31))
    @test length(receipts.rejections)==2 && isempty(receipts.batches)
    bad=copy(mag);bad[ISSUE+Minute(10)]=(NaN,0.0)
    @test only(materialize_receipts(bad,[ISSUE+Minute(10)];asof=ISSUE+Minute(31)).rejections).reason===:invalid_ground_rate
end

@testset "Bounded synthetic rolling and source state" begin
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:1000)
    state=CompletedMaximumController(calibration();start=ISSUE,gamma=0.005/31,rolling_capacity=5)
    for offset in 0:999
        time=ISSUE+Minute(offset)
        receive!(state,[(;center=time-Minute(1),rate=0.0)];received=time)
        issue!(state,prepare_issue(mag,time),0.0,1.0)
    end
    @test state.issued==1000 && state.resolved==state.mature_scores==969
    @test length(state.pending)==31 && state.max_pending==31
    @test state.max_seen<=33 && length(state.seen)<=33
    @test length(state.scores)==5
    @test certificate(state).unknown==31
    @test abs(certificate(state).accounting_residual)<1e-10
end

@testset "Shared source and label-processing delay in receipt paths" begin
    policy=CMR.ReplayPolicy(;ground_extra_ms=120_000,omni_extra_ms=180_000,label_extra_ms=240_000)
    mag=Dict(ISSUE+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    row=prepare_issue(mag,ISSUE;policy)
    state=CompletedMaximumController(calibration(;policy);start=ISSUE,gamma=0.1)
    issue!(state,row,0.0,1.0)
    @test state.pending[1].label_ready==ISSUE+Minute(36)+Second(45)
    @test state.pending[1].feedback_due==ISSUE+Minute(37)
    @test_throws ArgumentError receive!(state,[(;center=ISSUE+Minute(1),rate=2.0)];received=ISSUE+Minute(7)+Second(44)+Millisecond(999))
    first=receive!(state,[(;center=ISSUE+Minute(1),rate=2.0)];received=ISSUE+Minute(7)+Second(45))
    @test only(first).kind===:feedback && state.resolved==1
    last=receive!(state,[(;center=ISSUE+Minute(k),rate=0.0) for k in 2:30];received=ISSUE+Minute(37))
    @test only(last).kind===:score && state.censored_labels==0 && isempty(state.pending)
    receipts=materialize_receipts(mag,[ISSUE+Minute(30)];policy,asof=ISSUE+Minute(36)+Second(44)+Millisecond(999),scheduled=false)
    @test receipts.pending==[ISSUE+Minute(30)]
    receipts=materialize_receipts(mag,[ISSUE+Minute(30)];policy,asof=ISSUE+Minute(37))
    @test only(receipts.batches).received==ISSUE+Minute(37)
    axis=Dict(k=>ZeroBasedVector(v) for (k,v) in mag)
    @test materialize_receipts(axis,[ISSUE+Minute(30)];policy,asof=ISSUE+Minute(37))==receipts
end
