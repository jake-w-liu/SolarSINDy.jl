@testset verbose=true "Frozen explicit context and plan identity isolation" begin
    path=joinpath(@__DIR__,"frozen","context_identity_oracles_v1.json")
    @test bytes2hex(sha256(read(path)))=="c379796efa83f140196b5ece0376552bb0cfb949c22d26c178aaa249f13e2cf9"
    oracle=JSON3.read(read(path),Dict{String,Any})
    @test STREAM==oracle["stream_id"] && PROTOCOL==oracle["protocol_id"]
    for label in (""," \t\n",1,repeat("a",513))
        @test_throws ArgumentError EventReplay(calibration();start=I,subscribers=specs(),stream_id=label,protocol_id=PROTOCOL)
        @test_throws ArgumentError EventReplay(calibration();start=I,subscribers=specs(),stream_id=STREAM,protocol_id=label)
    end
    @test_throws UndefKeywordError EventReplay(calibration();start=I,subscribers=specs(),protocol_id=PROTOCOL)
    @test_throws UndefKeywordError EventReplay(calibration();start=I,subscribers=specs(),stream_id=STREAM)
    owner=state();context=owner.snapshot.context
    @test context.stream_id==STREAM && context.protocol_id==PROTOCOL
    for (key,value) in oracle["required_context"]
        actual=getproperty(context,Symbol(key))
        @test (actual isa Symbol ? String(actual) : actual)==value
    end
    @test D.policy_equal(context.policy,CMR.ReplayPolicy())
    plan=prepare_time(owner,I;issues=(request(),),transaction_id="context")
    other=EventReplay(calibration();start=I,subscribers=specs(),stream_id=STREAM,protocol_id=oracle["alternate_protocol_id"])
    before=other.snapshot
    @test_throws ArgumentError commit_time!(other,plan)
    @test other.snapshot===before && source_certificate(other).attempted==0
    # Exercise the context guard independently of the owner-token guard.
    owner.snapshot=E.replace_fields(owner.snapshot;context=merge(context,(;protocol_id=oracle["alternate_protocol_id"])))
    before=owner.snapshot
    @test_throws ArgumentError commit_time!(owner,plan)
    @test owner.snapshot===before

    owner=state(;grace=60000);all_records=Any[]
    capture(result)=append!(all_records,vcat(collect(result.events),collect(result.subscriber_events),collect(result.issues)))
    capture(transact!(owner,I;issues=(request(),)))
    capture(transact!(owner,I+Minute(2);rows=rate_rows(1,1;positive_at=1)))
    capture(transact!(owner,I+Minute(3);rows=rate_rows(1,1;positive_at=1)))
    capture(transact!(owner,I+Minute(31);issues=(request(I+Minute(31)),)))
    capture(transact!(owner,I+Minute(32);rows=rate_rows(2,30)))
    capture(transact!(owner,I+Minute(64)+Millisecond(1)))
    capture(transact!(owner,I+Minute(100);rows=rate_rows(1,1;positive_at=1)))
    @test Set(String(row.kind) for row in all_records)==Set(oracle["metadata_event_kinds"])
    @test all(isequal(row.context,context) for row in all_records)
    @test isequal(source_certificate(owner).context,context)
    @test all(isequal(subscriber_certificate(owner,name).context,context) for name in (:static,:phase,:fifo,:early,:maturity))
    before=owner.snapshot
    repeated=transact!(owner,I+Minute(100);rows=rate_rows(1,1;positive_at=1))
    @test repeated.repeated && isempty(repeated.events) && isempty(repeated.subscriber_events) && owner.snapshot===before
    mag=Dict(I+Minute(k)=>(0.0,0.0) for k in -40:0)
    prepared=D.prepare_schedule(mag,[I])
    output=replay_shared(prepared,[0.0],[1.0],[],calibration();subscribers=specs(),stream_id=STREAM,protocol_id=PROTOCOL,asof=I)
    @test isequal(output.context,context) && isequal(output.source.context,context)
    @test all(isequal(row.context,context) for row in output.issue_records)
    @test all(isequal(row.context,context) for row in output.subscribers)
    @test output.context.source_bound_provenance_verified===false && output.requires_source_bound_refits
    cal=calibration();cal.label_ready[1]=I-Millisecond(2)
    cal.label_ready[2]=I-Millisecond(3)
    @test_throws ArgumentError EventReplay(cal;start=I,subscribers=specs(),stream_id=STREAM,protocol_id=PROTOCOL)
end
